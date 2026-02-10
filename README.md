# ExactNoiseEstimation

Exact probability distributions for noise in the TFHE External Product
operation, computed using the [Dice](https://github.com/SHoltzen/dice)
probabilistic programming language (BDD-based exact discrete inference).

The standard noise analysis (in `../Parameter-Selection/`) tracks only
**variances** under a Gaussian approximation. This tool computes the
**full probability mass function** of each noise term, providing exact
tail probabilities that are critical for setting security parameters.

## Directory Structure

```
ExactNoiseEstimation/
  README.md                    # this file
  gen_extprod_dice.py          # generator — produces .dice programs
  analyze_output.py            # analyzer  — runs Dice, convolves, validates
  generated/                   # generated programs and cached outputs
    single_digit_cbd.dice      #   digit * CBD(eta) single-term program
    single_key_round.dice      #   key * rounding   single-term program
    single_direct_round.dice   #   -eps (direct rounding) single-term program
    output_single_*.txt        #   cached Dice outputs (for --from-files)
  dice/                        # Dice compiler (OCaml+Rust, git submodule)
```

## Quick Start

### Prerequisites

- Python 3.10+ with `numpy` (and optionally `scipy` for tail comparison)
- OCaml toolchain: `opam`, `dune` (for building Dice)
- Rust toolchain: `cargo` (for the `rsdd` BDD library used by Dice)

### 1. Build Dice

```bash
cd dice
git submodule update --init --recursive
eval $(opam env)
dune build
```

### 2. Generate Dice Programs

```bash
python3 gen_extprod_dice.py
```

This creates three `.dice` files in `generated/`, one per independent
single-term noise component.

### 3. Run the Analysis

```bash
python3 analyze_output.py            # runs Dice, then analyzes
python3 analyze_output.py --from-files  # reuse cached Dice outputs
```

The analysis script:

1. Runs each `.dice` program through the Dice compiler (~0.3 s each).
2. Parses the exact probability tables output by Dice.
3. Convolves single-term distributions in Python (`numpy.convolve`) to
   build the full noise distribution for each component.
4. Compares exact variances against the Gaussian formula from
   `../Parameter-Selection/python/noiseestimation/keyvariation.py`.
5. Reports tail probabilities and a full PMF table.

## Noise Model

We model the External Product `TRGSW(p=1) * TRLWE` for a single output
coefficient (index j=0) under the following toy parameters:

| Parameter | Value | Description                      |
|-----------|-------|----------------------------------|
| N         | 4     | Polynomial degree                |
| k         | 1     | TRLWE dimension                  |
| l = la    | 2     | Decomposition levels             |
| Bgbit     | 2     | Base bit width (Bg = 4)          |
| eta       | 1     | CBD parameter                    |
| qbit      | 8     | Torus precision (q = 256)        |

The output noise decomposes into three independent components:

### Component 1 — Nonce digit * CBD  (k*N = 4 terms)

Each term samples a ciphertext coefficient `c ~ Uniform[0, q)`,
deterministically decomposes it into digits `d_0, d_1`, and multiplies
each digit by an independent `CBD(eta)` sample. The signs from negacyclic
convolution do not affect the distribution (digit*CBD products are
symmetric).

### Component 2 — Main digit * CBD  (N = 4 terms)

Same structure as Component 1 but for the `b` polynomial row of the
TRLWE.

### Component 3 — Key * rounding + direct rounding + input noise

- **Key * rounding**: `s[m] * eps_a[m]` where `s ~ Bernoulli(1/2)` and
  `eps` is the deterministic rounding error from decomposition.  For
  j=0 negacyclic convolution: 1 positive term, N-1 = 3 negative terms.
  Signs matter here because `eps` is not symmetric.
- **Direct rounding**: `-eps_b[0]` from the `b`-polynomial decomposition.
- **Input noise**: `e[0] ~ CBD(eta)` from the fresh ciphertext.

### Key Design Choice

Rather than modeling the rounding error as an independent uniform random
variable, we **sample `c ~ Uniform[0, q)` and decompose it
deterministically** inside the Dice program using shifts and arithmetic.
The digits and rounding error are derived quantities, not independently
sampled.  For a power-of-2 torus, the decomposition is bijective, so
digits and eps happen to be independent anyway — but the deterministic
approach is more principled and generalizes to non-power-of-2 moduli.

## BDD Performance Optimizations

Dice compiles probabilistic programs into Binary Decision Diagrams
(BDDs).  Three optimizations were essential to keep execution fast:

| Technique | Speedup | Rationale |
|-----------|---------|-----------|
| Conditionals instead of multiply | ~450x | `d * e` for two random 8-bit integers takes ~107 s in BDD. For CBD(1) with values {-1, 0, 1}, rewriting as `if e_nz then (if e_pos then d else -d) else 0` takes ~0.24 s. |
| Left-shift instead of multiply by constant | ~10x | `d << 6` instead of `d * int(8, 64)` avoids the BDD multiplier for power-of-2 constants. |
| Single-term Dice + Python convolution | unbounded | Even 4 iterations of a 12-variable function can exceed 18 GB of BDD memory.  Computing one term in Dice and convolving N copies in Python is instant and uses negligible memory. |

## Sample Output

```
Component                              Exact Var    Gauss Var    Ratio
----------------------------------- ------------ ------------ --------
Nonce digit*CBD                           6.0000       6.0000   1.0000
Main digit*CBD                            6.0000       6.0000   1.0000
Comp3 (key+direct+input)                 64.5000      64.5000   1.0000
TOTAL                                    76.5000      76.5000   1.0000
```

For the toy parameters, the exact variance matches the Gaussian formula
perfectly. The exact tail probabilities diverge at higher sigmas:

```
Thresh          Exact tail      Gauss tail      Ratio
1 sigma       3.330253e-01    3.173105e-01     1.0495
2 sigma       4.467112e-02    4.550026e-02     0.9818
3 sigma       2.176023e-03    2.699796e-03     0.8060
5 sigma       1.354998e-07    5.733031e-07     0.2363
```

At 5 sigma the exact tail probability is 4.2x smaller than the Gaussian
prediction, meaning the Gaussian approximation is conservative.

## Related Files

- `../Parameter-Selection/python/noiseestimation/keyvariation.py` —
  `extpnoisecalc` (Gaussian variance formula, lines 72-96)
- `../TFHEpp/include/trgsw.hpp` — Decomposition implementation
  (lines 75-170) and External Product (lines 308-443)
