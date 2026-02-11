#!/usr/bin/env python3
"""Generate Dice programs for exact noise estimation of TFHE External Product.

Generates single-term .dice programs for each independent noise component.
All iteration/accumulation is done via Python convolution (not BDD iterate),
keeping each Dice program small (~12 BDD variables).

Parameters: N=4, k=1, l=la=2, Bgbit=2, Bg=4, eta=4, qbit=15, q=32768.
"""

import os
from fractions import Fraction
from math import comb
from dataclasses import dataclass


@dataclass
class TFHEParams:
    N: int = 4        # polynomial degree
    k: int = 1        # TRLWE dimension
    l: int = 2        # main decomposition levels
    la: int = 2       # nonce decomposition levels
    Bgbit: int = 2    # base bit width
    Bg: int = 4       # decomposition base
    eta: int = 4      # CBD parameter
    qbit: int = 15    # log2(q)
    q: int = 32768    # ciphertext modulus

    @property
    def remaining_bits(self) -> int:
        return self.qbit - self.l * self.Bgbit

    @property
    def halfBg(self) -> int:
        return self.Bg // 2

    @property
    def offset(self) -> int:
        off = 0
        for i in range(1, self.l + 1):
            off += (self.Bg // 2) * (1 << (self.qbit - i * self.Bgbit))
        return off

    @property
    def roundoffset(self) -> int:
        rb = self.remaining_bits
        return (1 << (rb - 1)) if rb > 0 else 0

    @property
    def total_offset(self) -> int:
        return self.offset + self.roundoffset


def format_uniform(qbit: int) -> str:
    q = 1 << qbit
    return "discrete(" + ", ".join([f"1/{q}"] * q) + ")"


def cbd_distribution(eta: int, qbit: int) -> list[Fraction]:
    q = 1 << qbit
    total = 1 << (2 * eta)
    dist = [Fraction(0)] * q
    for v in range(-eta, eta + 1):
        dist[v % q] = Fraction(comb(2 * eta, eta + v), total)
    return dist


def format_discrete(dist: list[Fraction]) -> str:
    return "discrete(" + ", ".join(str(p) for p in dist) + ")"


def gen_uniform(var_name: str, qbit: int, indent: str = "") -> str:
    """Generate Dice code to sample var_name ~ Uniform[0, 2^qbit).

    Uses discrete(1/q, ..., 1/q) for qbit <= 13.
    For larger qbit, splits into independent low/high parts to work around
    a Dice BDD bug where discrete with > 2^13 non-zero entries drops
    lower-order BDD variables.
    """
    MAX_SAFE_BITS = 13
    q = 1 << qbit

    if qbit <= MAX_SAFE_BITS:
        expr = "discrete(" + ", ".join([f"1/{q}"] * q) + ")"
        return f"{indent}let {var_name} = {expr} in"

    # Split into low and high parts, each within safe range
    low_bits = qbit // 2
    high_bits = qbit - low_bits
    n_low = 1 << low_bits
    n_high = 1 << high_bits
    step = n_low  # high values are multiples of 2^low_bits

    # c_lo: uniform over {0, ..., n_low-1} as int(qbit)
    low_probs = [f"1/{n_low}"] * n_low + ["0"] * (q - n_low)
    # c_hi: uniform over {0, step, 2*step, ...} as int(qbit)
    high_probs = ["0"] * q
    for i in range(n_high):
        high_probs[i * step] = f"1/{n_high}"

    lines = [
        f"{indent}let {var_name}_lo = discrete(" + ", ".join(low_probs) + ") in",
        f"{indent}let {var_name}_hi = discrete(" + ", ".join(high_probs) + ") in",
        f"{indent}let {var_name} = {var_name}_hi + {var_name}_lo in",
    ]
    return "\n".join(lines)


def gen_decomp_digits_only(var_name: str, params: TFHEParams, indent: str = "    ") -> tuple[str, list[str]]:
    """Generate decomposition code that only extracts digits (no eps)."""
    P = params
    qbit = P.qbit
    lines = []

    off_var = f"{var_name}_off"
    lines.append(f"{indent}let {off_var} = {var_name} + int({qbit}, {P.total_offset}) in")

    digit_vars = []
    prev_raw = None
    for level in range(P.l):
        raw_var = f"{var_name}_d{level}_raw"
        digit_var = f"{var_name}_d{level}"
        shift_top = qbit - (level + 1) * P.Bgbit

        if level == 0:
            lines.append(f"{indent}let {raw_var} = {off_var} >> {shift_top} in")
        else:
            lines.append(f"{indent}let {raw_var} = ({off_var} >> {shift_top}) - ({prev_raw} << {P.Bgbit}) in")

        lines.append(f"{indent}let {digit_var} = {raw_var} - int({qbit}, {P.halfBg}) in")
        digit_vars.append(digit_var)
        prev_raw = raw_var

    return "\n".join(lines), digit_vars


def gen_decomp_eps_only(var_name: str, params: TFHEParams, indent: str = "    ") -> tuple[str, str]:
    """Generate decomposition code that only extracts rounding error eps."""
    P = params
    qbit = P.qbit
    lines = []

    off_var = f"{var_name}_off"
    lines.append(f"{indent}let {off_var} = {var_name} + int({qbit}, {P.total_offset}) in")

    digit_vars = []
    prev_raw = None
    for level in range(P.l):
        raw_var = f"{var_name}_d{level}_raw"
        digit_var = f"{var_name}_d{level}"
        shift_top = qbit - (level + 1) * P.Bgbit

        if level == 0:
            lines.append(f"{indent}let {raw_var} = {off_var} >> {shift_top} in")
        else:
            lines.append(f"{indent}let {raw_var} = ({off_var} >> {shift_top}) - ({prev_raw} << {P.Bgbit}) in")

        lines.append(f"{indent}let {digit_var} = {raw_var} - int({qbit}, {P.halfBg}) in")
        digit_vars.append(digit_var)
        prev_raw = raw_var

    # eps = c - sum_i (d_i << shift_i), using shifts for power-of-2 h values
    eps_var = f"{var_name}_eps"
    eps_expr = var_name
    for level in range(P.l):
        shift_val = (P.l - 1 - level) * P.Bgbit + P.remaining_bits
        eps_expr += f" - ({digit_vars[level]} << {shift_val})"
    lines.append(f"{indent}let {eps_var} = {eps_expr} in")

    return "\n".join(lines), eps_var


def cbd_magnitude_probs(eta: int) -> list[Fraction]:
    """Return P(|CBD(eta)| = m) for m = 0, 1, ..., eta, padded to next power of 2."""
    total = 1 << (2 * eta)
    probs = [Fraction(comb(2 * eta, eta), total)]  # P(|e|=0)
    for m in range(1, eta + 1):
        probs.append(Fraction(2 * comb(2 * eta, eta + m), total))
    # Pad to next power of 2
    nbits = (len(probs) - 1).bit_length()
    target = 1 << nbits
    while len(probs) < target:
        probs.append(Fraction(0))
    return probs


def gen_d_times_m(d_var: str, m: int, qbit: int) -> str:
    """Generate Dice expression for d * m (constant multiply) using shifts and adds.

    Returns an expression (not a let-binding) computing d_var * m mod 2^qbit.
    """
    if m == 0:
        return f"int({qbit}, 0)"
    if m == 1:
        return d_var
    # Decompose m into binary: use shifts and adds
    # e.g., m=3: (d << 1) + d, m=5: (d << 2) + d, m=6: (d << 2) + (d << 1)
    bits = []
    v = m
    pos = 0
    while v > 0:
        if v & 1:
            bits.append(pos)
        v >>= 1
        pos += 1
    terms = [f"({d_var} << {b})" if b > 0 else d_var for b in bits]
    return " + ".join(terms)


def gen_digit_cbd_product(digit_var: str, level: int, eta: int, qbit: int, indent: str = "") -> list[str]:
    """Generate optimized digit*CBD(eta) code using magnitude+sign encoding.

    Encodes CBD(eta) compactly:
      1. Sample magnitude |e| from {0,...,eta} as a small discrete (ceil(log2(eta+1)) BDD vars)
      2. Sample sign from flip 0.5 (1 BDD var)
      3. Compute d * |e| using conditionals with shifts/adds (no BDD multiply)
      4. Apply sign

    Total: ceil(log2(eta+1)) + 1 BDD vars per CBD sample (vs qbit for full encoding).
    """
    lines = []
    pv = f"prod{level}"
    mag_var = f"e{level}_mag"
    sign_var = f"e{level}_sign"
    abs_var = f"abs_prod{level}"

    mag_probs = cbd_magnitude_probs(eta)
    nbits = (len(mag_probs) - 1).bit_length()

    # Sample magnitude
    mag_discrete = "discrete(" + ", ".join(str(p) for p in mag_probs) + ")"
    lines.append(f"{indent}let {mag_var} = {mag_discrete} in")

    # Build nested if-then-else for d * |e|
    # Start from highest magnitude, work down
    expr = f"int({qbit}, 0)"  # default for magnitude 0
    for m in range(eta, 0, -1):
        d_times_m = gen_d_times_m(digit_var, m, qbit)
        expr = f"if {mag_var} == int({nbits}, {m}) then {d_times_m} else ({expr})"
    lines.append(f"{indent}let {abs_var} = {expr} in")

    # Sample sign and apply
    lines.append(f"{indent}let {sign_var} = flip 0.5 in")
    lines.append(
        f"{indent}let {pv} = if {mag_var} == int({nbits}, 0) then int({qbit}, 0) "
        f"else (if {sign_var} then {abs_var} else int({qbit}, 0) - {abs_var}) in"
    )

    return lines


def gen_single_digit_cbd(params: TFHEParams, output_dir: str) -> str:
    """Generate single-term: sum_i d_i * ebar_i for one polynomial coefficient.

    This is one term of the nonce or main digit×CBD sum. Since la=l and
    Bga=Bg for our toy params, nonce and main share the same single-term
    distribution.

    Uses magnitude+sign encoding for CBD to avoid BDD multiply.
    WARNING: May OOM for large qbit (>= 15) due to BDD size.
    """
    P = params
    qbit = P.qbit

    lines = []
    lines.append(f"// Single term: d0*e0 + d1*e1 where d_i from decomposition, e_i ~ CBD({P.eta})")
    lines.append(f"// d_i in {{-{P.halfBg},...,{P.halfBg - 1}}}, e_i in {{-{P.eta},...,{P.eta}}}")
    lines.append(gen_uniform("c", qbit))

    decomp_code, digit_vars = gen_decomp_digits_only("c", P, indent="")
    lines.append(decomp_code)

    # Generate optimized digit*CBD products using magnitude+sign encoding
    for level in range(P.l):
        prod_lines = gen_digit_cbd_product(digit_vars[level], level, P.eta, qbit)
        lines.extend(prod_lines)

    prod_vars = [f"prod{i}" for i in range(P.l)]
    lines.append(" + ".join(prod_vars))

    program = "\n".join(lines)
    path = os.path.join(output_dir, "single_digit_cbd.dice")
    with open(path, "w") as f:
        f.write(program + "\n")
    print(f"Generated {path}")
    return path


def gen_single_digit(params: TFHEParams, output_dir: str) -> str:
    """Generate single-term: extract one decomposition digit from uniform c.

    Outputs d0 ∈ {-Bg/2, ..., Bg/2-1} as int(qbit).
    The digit*CBD product is computed analytically in Python.
    This is much lighter on BDD than gen_single_digit_cbd.
    """
    P = params
    qbit = P.qbit

    lines = []
    lines.append(f"// Single decomposition digit d0 from uniform c")
    lines.append(f"// d0 in {{-{P.halfBg},...,{P.halfBg - 1}}}")
    lines.append(gen_uniform("c", qbit))

    # Decomposition: extract just d0 (top digit)
    off_var = "c_off"
    lines.append(f"let {off_var} = c + int({qbit}, {P.total_offset}) in")
    shift_top = qbit - P.Bgbit
    raw_var = "c_d0_raw"
    lines.append(f"let {raw_var} = {off_var} >> {shift_top} in")
    lines.append(f"{raw_var} - int({qbit}, {P.halfBg})")

    program = "\n".join(lines)
    path = os.path.join(output_dir, "single_digit.dice")
    with open(path, "w") as f:
        f.write(program + "\n")
    print(f"Generated {path}")
    return path


def gen_single_key_round(params: TFHEParams, output_dir: str) -> str:
    """Generate single-term: s * eps where s ~ Bernoulli(1/2), eps from decomposition.

    This is one term of the key × rounding noise.
    """
    P = params
    qbit = P.qbit

    lines = []
    lines.append(f"// Single term: s * eps where s ~ Bernoulli(1/2)")
    lines.append(f"// eps from decomposition rounding, range {{-{1 << (P.remaining_bits - 1)},...,{(1 << (P.remaining_bits - 1)) - 1}}}")
    lines.append(gen_uniform("c", qbit))

    decomp_code, eps_var = gen_decomp_eps_only("c", P, indent="")
    lines.append(decomp_code)

    lines.append(f"let s = flip 0.5 in")
    lines.append(f"if s then {eps_var} else int({qbit}, 0)")

    program = "\n".join(lines)
    path = os.path.join(output_dir, "single_key_round.dice")
    with open(path, "w") as f:
        f.write(program + "\n")
    print(f"Generated {path}")
    return path


def gen_single_direct_round(params: TFHEParams, output_dir: str) -> str:
    """Generate single-term: -eps (direct rounding from b polynomial)."""
    P = params
    qbit = P.qbit

    lines = []
    lines.append(f"// Single term: -eps (direct rounding error)")
    lines.append(gen_uniform("c", qbit))

    decomp_code, eps_var = gen_decomp_eps_only("c", P, indent="")
    lines.append(decomp_code)

    lines.append(f"int({qbit}, 0) - {eps_var}")

    program = "\n".join(lines)
    path = os.path.join(output_dir, "single_direct_round.dice")
    with open(path, "w") as f:
        f.write(program + "\n")
    print(f"Generated {path}")
    return path


def main():
    params = TFHEParams()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parameters: N={params.N}, k={params.k}, l={params.l}, la={params.la}")
    print(f"  Bgbit={params.Bgbit}, Bg={params.Bg}, eta={params.eta}")
    print(f"  qbit={params.qbit}, q={params.q}")
    print(f"  remaining_bits={params.remaining_bits}")
    print(f"  offset={params.offset}, roundoffset={params.roundoffset}")
    print(f"  total_offset={params.total_offset}")
    print()

    gen_single_digit(params, output_dir)
    gen_single_key_round(params, output_dir)
    gen_single_direct_round(params, output_dir)

    print(f"\nAll single-term programs generated in {output_dir}/")
    print(f"\nConvolution plan (done in analyze_output.py):")
    print(f"  Digit*CBD: Dice validates digit dist, then digit×CBD product computed analytically")
    print(f"  Nonce digit*CBD: convolve {params.l * params.k * params.N} copies of digit×CBD")
    print(f"  Main digit*CBD:  convolve {params.l * params.N} copies of digit×CBD")
    print(f"  Key*rounding:    convolve 1 pos + {params.N - 1} neg copies of single_key_round")
    print(f"  Direct rounding: 1 copy of single_direct_round")
    print(f"  Input noise:     CBD({params.eta}) distribution (computed analytically)")


if __name__ == "__main__":
    main()
