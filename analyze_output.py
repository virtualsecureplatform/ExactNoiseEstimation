#!/usr/bin/env python3
"""Analyze Dice output for exact noise estimation of TFHE External Product.

Runs Dice on single-term programs, parses outputs, convolves distributions
in Python to build full component distributions, and compares exact variance
with the Gaussian approximation from extpnoisecalc.

Usage:
    python3 analyze_output.py              # runs Dice on each single-term program
    python3 analyze_output.py --from-files # reads pre-saved output files
"""

import argparse
import os
import subprocess
import sys
from fractions import Fraction
from math import comb

import numpy as np


# ---------------------------------------------------------------------------
# Dice output parsing
# ---------------------------------------------------------------------------

def run_dice(dice_file: str, dice_bin: str = None) -> str:
    """Run Dice on a .dice file and return stdout."""
    if dice_bin is None:
        base = os.path.dirname(os.path.abspath(__file__))
        dice_bin_path = os.path.join(base, "dice", "_build", "default", "bin", "dice.exe")
    else:
        dice_bin_path = dice_bin

    print(f"  Running Dice on {os.path.basename(dice_file)}...", end=" ", flush=True)
    result = subprocess.run(
        [dice_bin_path, dice_file],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    print("done.", flush=True)
    return result.stdout


def parse_dice_output(output: str) -> dict[int, Fraction]:
    """Parse Dice's Joint Distribution table into {unsigned_value: probability}."""
    dist = {}
    in_table = False
    for line in output.splitlines():
        line = line.strip()
        if "Joint Distribution" in line:
            in_table = True
            continue
        if in_table and line.startswith("Value"):
            continue
        if in_table and line.startswith("="):
            break
        if in_table and line:
            parts = line.split("\t")
            if len(parts) >= 2:
                val = int(parts[0].strip())
                prob_str = parts[1].strip()
                if prob_str and prob_str != "0":
                    dist[val] = Fraction(prob_str)
    return dist


# ---------------------------------------------------------------------------
# Distribution manipulation
# ---------------------------------------------------------------------------

def to_signed(unsigned_dist: dict[int, Fraction], qbit: int) -> dict[int, Fraction]:
    """Convert unsigned int(qbit) distribution to signed (two's complement)."""
    q = 1 << qbit
    half = q >> 1
    signed = {}
    for uval, prob in unsigned_dist.items():
        if prob == 0:
            continue
        sval = uval if uval < half else uval - q
        signed[sval] = signed.get(sval, Fraction(0)) + prob
    return signed


def dist_to_array(signed_dist: dict[int, Fraction]) -> tuple[np.ndarray, int]:
    """Convert signed distribution to (pmf_array, min_val)."""
    if not signed_dist:
        return np.array([1.0]), 0
    min_val = min(signed_dist.keys())
    max_val = max(signed_dist.keys())
    arr = np.zeros(max_val - min_val + 1, dtype=np.float64)
    for v, p in signed_dist.items():
        arr[v - min_val] = float(p)
    return arr, min_val


def negate_dist(arr: np.ndarray, min_val: int) -> tuple[np.ndarray, int]:
    """Negate a distribution: if X has pmf arr starting at min_val,
    return the pmf of -X."""
    max_val = min_val + len(arr) - 1
    new_min = -max_val
    return arr[::-1], new_min


def convolve(dists: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
    """Convolve multiple PMF arrays (with offsets)."""
    result_arr, result_off = dists[0]
    for arr, off in dists[1:]:
        result_arr = np.convolve(result_arr, arr)
        result_off += off
    return result_arr, result_off


def convolve_n(arr: np.ndarray, min_val: int, n: int) -> tuple[np.ndarray, int]:
    """Convolve a distribution with itself n times."""
    if n == 0:
        return np.array([1.0]), 0
    if n == 1:
        return arr.copy(), min_val
    result = arr.copy()
    result_off = min_val
    for _ in range(n - 1):
        result = np.convolve(result, arr)
        result_off += min_val
    return result, result_off


def compute_stats(pmf: np.ndarray, min_val: int) -> dict:
    """Compute statistics from a PMF array."""
    values = np.arange(len(pmf)) + min_val
    total = np.sum(pmf)
    mean = np.sum(values * pmf) / total
    variance = np.sum((values - mean) ** 2 * pmf) / total
    std = np.sqrt(variance)

    abs_centered = np.abs(values - mean)
    tails = {}
    for mult in [1, 2, 3, 4, 5]:
        threshold = mult * std
        tail_prob = np.sum(pmf[abs_centered > threshold]) / total
        tails[f"|X-mu|>{mult}s"] = tail_prob

    return {"mean": mean, "variance": variance, "std": std,
            "total_prob": total, "tails": tails}


def cbd_signed_dist(eta: int) -> dict[int, Fraction]:
    """CBD(eta) as signed distribution."""
    total = 1 << (2 * eta)
    return {v: Fraction(comb(2 * eta, eta + v), total) for v in range(-eta, eta + 1)}


# ---------------------------------------------------------------------------
# Gaussian-approximation variance (from keyvariation.py)
# ---------------------------------------------------------------------------

def decomp_round_variance_pow2(qbit: int, basebit: int, levels: int) -> float:
    kept_bits = levels * basebit
    remaining_bits = qbit - kept_bits
    if remaining_bits <= 0:
        return 0.0
    roundwidth = float(2 ** remaining_bits)
    return roundwidth * roundwidth / 12.0 - 1.0 / 12.0


def extpnoisecalc_components(N, k, l, la, Bgbit, eta, qbit,
                              exp_key=0.5, var_key=0.25):
    """Compute expected variances matching keyvariation.py formula."""
    Bg = 1 << Bgbit
    cbd_var = eta / 2.0
    Ed2 = (Bg**2 + 2) / 12.0  # E[d^2] = Var(d) + E[d]^2

    # Step 1: digit × TRGSW row noise
    nonce_digit_cbd_var = la * k * N * Ed2 * cbd_var
    main_digit_cbd_var = l * N * Ed2 * cbd_var

    # Step 2: rounding
    nonce_roundvar = decomp_round_variance_pow2(qbit, Bgbit, la)
    nonnonce_roundvar = decomp_round_variance_pow2(qbit, Bgbit, l)

    key_round_var = nonce_roundvar * (k * N * (var_key + exp_key**2))
    key_extra_var = k * N / 4.0 * var_key
    direct_round_var = nonnonce_roundvar
    input_var = cbd_var

    comp3_var = key_round_var + key_extra_var + direct_round_var + input_var

    return {
        "nonce_digit_cbd": nonce_digit_cbd_var,
        "main_digit_cbd": main_digit_cbd_var,
        "key_round": key_round_var,
        "key_extra": key_extra_var,
        "direct_round": direct_round_var,
        "input_noise": input_var,
        "comp3_total": comp3_var,
        "total": nonce_digit_cbd_var + main_digit_cbd_var + comp3_var,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-files", action="store_true",
                        help="Read pre-saved output files")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(base_dir, "generated")

    # Parameters
    N, k, l, la = 4, 1, 2, 2
    Bgbit, Bg, eta, qbit = 2, 4, 1, 8
    q = 1 << qbit

    # Single-term programs
    programs = {
        "single_digit_cbd": os.path.join(gen_dir, "single_digit_cbd.dice"),
        "single_key_round": os.path.join(gen_dir, "single_key_round.dice"),
        "single_direct_round": os.path.join(gen_dir, "single_direct_round.dice"),
    }
    output_files = {name: os.path.join(gen_dir, f"output_{name}.txt")
                    for name in programs}

    # Run Dice or load pre-saved outputs
    print("=" * 60)
    print("Step 1: Compute single-term distributions via Dice")
    print("=" * 60)

    raw_dists = {}
    for name, dfile in programs.items():
        ofile = output_files[name]
        if args.from_files:
            if not os.path.exists(ofile):
                print(f"ERROR: {ofile} not found.")
                sys.exit(1)
            with open(ofile) as f:
                output = f.read()
            print(f"  Loaded {os.path.basename(ofile)}")
        else:
            if not os.path.exists(dfile):
                print(f"ERROR: {dfile} not found. Run gen_extprod_dice.py first.")
                sys.exit(1)
            output = run_dice(dfile)
            with open(ofile, "w") as f:
                f.write(output)

        unsigned = parse_dice_output(output)
        signed = to_signed(unsigned, qbit)
        arr, off = dist_to_array(signed)
        raw_dists[name] = (arr, off)

        stats = compute_stats(arr, off)
        print(f"    {name}: support [{off},{off+len(arr)-1}], "
              f"mean={stats['mean']:.4f}, var={stats['variance']:.4f}")

    # Also compute CBD distribution analytically
    cbd_dist = cbd_signed_dist(eta)
    cbd_arr, cbd_off = dist_to_array(cbd_dist)
    raw_dists["input_cbd"] = (cbd_arr, cbd_off)
    cbd_stats = compute_stats(cbd_arr, cbd_off)
    print(f"    input_cbd: support [{cbd_off},{cbd_off+len(cbd_arr)-1}], "
          f"mean={cbd_stats['mean']:.4f}, var={cbd_stats['variance']:.4f}")

    # Build component distributions via convolution
    print("\n" + "=" * 60)
    print("Step 2: Build component distributions via convolution")
    print("=" * 60)

    digit_arr, digit_off = raw_dists["single_digit_cbd"]
    key_arr, key_off = raw_dists["single_key_round"]
    direct_arr, direct_off = raw_dists["single_direct_round"]

    # Component 1: Nonce digit×CBD = convolve k*N copies
    nonce_count = k * N
    nonce_arr, nonce_off = convolve_n(digit_arr, digit_off, nonce_count)
    nonce_stats = compute_stats(nonce_arr, nonce_off)
    print(f"\n  Nonce digit*CBD ({nonce_count} terms convolved):")
    print(f"    Support: [{nonce_off}, {nonce_off + len(nonce_arr) - 1}]")
    print(f"    Mean:     {nonce_stats['mean']:.6f}")
    print(f"    Variance: {nonce_stats['variance']:.6f}")

    # Component 2: Main digit×CBD = convolve N copies
    main_count = N
    main_arr, main_off = convolve_n(digit_arr, digit_off, main_count)
    main_stats = compute_stats(main_arr, main_off)
    print(f"\n  Main digit*CBD ({main_count} terms convolved):")
    print(f"    Support: [{main_off}, {main_off + len(main_arr) - 1}]")
    print(f"    Mean:     {main_stats['mean']:.6f}")
    print(f"    Variance: {main_stats['variance']:.6f}")

    # Component 3: key×rounding + direct rounding + input noise
    # key_noise = key_pos - key_neg where key_pos has 1 term, key_neg has N-1 terms
    n_pos = 1
    n_neg = N - 1
    key_pos_arr, key_pos_off = convolve_n(key_arr, key_off, n_pos)
    key_neg_arr, key_neg_off = convolve_n(key_arr, key_off, n_neg)
    # Negate the neg distribution
    neg_key_neg_arr, neg_key_neg_off = negate_dist(key_neg_arr, key_neg_off)
    # key_noise = key_pos + (-key_neg)
    key_noise_arr, key_noise_off = convolve(
        [(key_pos_arr, key_pos_off), (neg_key_neg_arr, neg_key_neg_off)]
    )
    key_noise_stats = compute_stats(key_noise_arr, key_noise_off)
    print(f"\n  Key*rounding ({n_pos} pos - {n_neg} neg):")
    print(f"    Support: [{key_noise_off}, {key_noise_off + len(key_noise_arr) - 1}]")
    print(f"    Mean:     {key_noise_stats['mean']:.6f}")
    print(f"    Variance: {key_noise_stats['variance']:.6f}")

    # Full component 3 = key_noise + direct + input_cbd
    comp3_arr, comp3_off = convolve([
        (key_noise_arr, key_noise_off),
        (direct_arr, direct_off),
        (cbd_arr, cbd_off),
    ])
    comp3_stats = compute_stats(comp3_arr, comp3_off)
    print(f"\n  Component 3 total (key + direct + input):")
    print(f"    Support: [{comp3_off}, {comp3_off + len(comp3_arr) - 1}]")
    print(f"    Mean:     {comp3_stats['mean']:.6f}")
    print(f"    Variance: {comp3_stats['variance']:.6f}")

    # Total noise = nonce + main + comp3
    print("\n" + "=" * 60)
    print("Step 3: Convolve all components for total noise distribution")
    print("=" * 60)

    total_arr, total_off = convolve([
        (nonce_arr, nonce_off),
        (main_arr, main_off),
        (comp3_arr, comp3_off),
    ])
    total_stats = compute_stats(total_arr, total_off)

    print(f"\n  Total noise distribution:")
    print(f"    Support: [{total_off}, {total_off + len(total_arr) - 1}]")
    print(f"    Mean:     {total_stats['mean']:.6f}")
    print(f"    Variance: {total_stats['variance']:.6f}")
    print(f"    Std:      {total_stats['std']:.6f}")
    print(f"    Sum(P):   {total_stats['total_prob']:.10f}")
    print(f"\n    Tail probabilities:")
    for desc, prob in total_stats["tails"].items():
        print(f"      P({desc}) = {prob:.6e}")

    # Compare with Gaussian approximation
    print("\n" + "=" * 60)
    print("Step 4: Comparison with Gaussian approximation (extpnoisecalc)")
    print("=" * 60)

    expected = extpnoisecalc_components(N, k, l, la, Bgbit, eta, qbit)

    print(f"\n  Gaussian formula breakdown:")
    for key, val in expected.items():
        print(f"    {key:25s} = {val:.6f}")

    print(f"\n  {'Component':<35} {'Exact Var':>12} {'Gauss Var':>12} {'Ratio':>8}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")

    comparisons = [
        ("Nonce digit*CBD", nonce_stats["variance"], "nonce_digit_cbd"),
        ("Main digit*CBD", main_stats["variance"], "main_digit_cbd"),
        ("Comp3 (key+direct+input)", comp3_stats["variance"], "comp3_total"),
    ]
    for label, exact_v, ekey in comparisons:
        gauss_v = expected[ekey]
        ratio = exact_v / gauss_v if gauss_v > 0 else float("inf")
        print(f"  {label:<35} {exact_v:>12.4f} {gauss_v:>12.4f} {ratio:>8.4f}")

    exact_total = total_stats["variance"]
    gauss_total = expected["total"]
    ratio = exact_total / gauss_total if gauss_total > 0 else float("inf")
    print(f"  {'TOTAL':<35} {exact_total:>12.4f} {gauss_total:>12.4f} {ratio:>8.4f}")

    # Gaussian tail comparison
    try:
        from scipy.special import erfc
        print(f"\n  Gaussian tail comparison:")
        gauss_std = np.sqrt(gauss_total)
        print(f"    Exact  std={total_stats['std']:.4f}, mean={total_stats['mean']:.4f}")
        print(f"    Gauss  std={gauss_std:.4f}")
        print(f"\n    {'Thresh':<10} {'Exact tail':>15} {'Gauss tail':>15} {'Ratio':>10}")
        for mult in [1, 2, 3, 4, 5]:
            exact_tail = total_stats["tails"][f"|X-mu|>{mult}s"]
            gauss_tail = float(erfc(mult / np.sqrt(2)))
            ratio = exact_tail / gauss_tail if gauss_tail > 0 else float("inf")
            print(f"    {mult} sigma    {exact_tail:>15.6e} {gauss_tail:>15.6e} {ratio:>10.4f}")
    except ImportError:
        print("\n  (scipy not available, skipping Gaussian tail comparison)")

    # Print full PMF for non-zero entries
    print("\n" + "=" * 60)
    print("Full PMF (non-zero entries)")
    print("=" * 60)
    nonzero = [(total_off + i, total_arr[i])
               for i in range(len(total_arr)) if total_arr[i] > 1e-15]
    print(f"\n  {'Value':>6}  {'Probability':>15}")
    for val, prob in nonzero:
        print(f"  {val:>6d}  {prob:>15.10f}")

    print(f"\n  Total non-zero entries: {len(nonzero)}")
    print()


if __name__ == "__main__":
    main()
