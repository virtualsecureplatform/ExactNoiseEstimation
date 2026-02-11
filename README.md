# ExactNoiseEstimation

Exact probability distributions for noise in the TFHE External Product
operation, computed using the [Alea.jl](https://github.com/Tractables/Alea.jl)
probabilistic programming system (BDD-based exact discrete inference).

The standard noise analysis (in `../Parameter-Selection/`) tracks only
**variances** under a Gaussian approximation. This tool computes the
**full probability mass function** of each noise term, providing exact
tail probabilities that are critical for setting security parameters.

## Directory Structure

```
ExactNoiseEstimation/
  README.md
  extprod_alea.jl            # exact External Product noise via Alea.jl
  alea/                      # Alea.jl (git submodule)
```

## Quick Start

### Prerequisites

- Julia 1.8.5+
- Python 3 (for Alea's SymPy dependency)

Install SymPy (required by Alea):

```bash
pip3 install sympy
```

### 1. Initialize the Alea Submodule

```bash
git submodule update --init --recursive
```

### 2. Instantiate Julia Dependencies

```bash
julia --project=alea -e "import Pkg; Pkg.instantiate()"
```

### 3. Run the Exact External Product Analysis

```bash
julia --project=alea extprod_alea.jl
```

Convolution backend options:

- FFT (default, fast but approximate): `julia --project=alea extprod_alea.jl --conv=fft`
- Exact (slow, O(n^2) per convolution): `julia --project=alea extprod_alea.jl --conv=exact`

You can also set `EXTPROD_CONV=fft` or `EXTPROD_CONV=exact` in the environment.
From Julia, you can use `Alea.set_conv_mode!(:fft)` or `Alea.set_conv_mode!(:exact)` and query via `Alea.conv_mode()`.

TRGSW plaintext mode options:

- Fixed plaintext `p=1` (default): `--trgsw-pt=fixed`
- Binary plaintext `p~Bernoulli(0.5)`: `--trgsw-pt=binary`

You can also set `TRGSW_PT=fixed|binary` in the environment.

The script:

1. Uses Alea to compute exact **single-term** distributions for:
   - a decomposition digit
   - key × rounding noise
   - direct rounding noise
2. Computes digit×CBD analytically and convolves multiple terms to build
   full component distributions.
3. Reports exact means, variances, and tail probabilities.
4. Compares exact variances to the Gaussian formula from
   `../Parameter-Selection/python/noiseestimation/keyvariation.py`.

## Noise Model (Toy Parameters)

We model the External Product `TRGSW(p=1) * TRLWE` for a single output
coefficient (index j=0) under the following toy parameters:

| Parameter | Value | Description                      |
|-----------|-------|----------------------------------|
| N         | 512   | Polynomial degree                |
| k         | 2     | TRLWE dimension                  |
| l = la    | 2     | Decomposition levels             |
| Bgbit     | 9     | Base bit width (Bg = 512)        |
| eta       | 4     | CBD parameter                    |
| qbit      | 25    | Torus precision (q = 33554432)   |

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
  `eps` is the deterministic rounding error from decomposition. For
  j=0 negacyclic convolution: 1 positive term, N-1 = 3 negative terms.
  Signs matter here because `eps` is not symmetric.
- **Direct rounding**: `-eps_b[0]` from the `b`-polynomial decomposition.
- **Input noise**: `e[0] ~ CBD(eta)` from the fresh ciphertext.

### Key Design Choice

Rather than modeling the rounding error as an independent uniform random
variable, we **sample `c ~ Uniform[0, q)` and decompose it
deterministically** inside the Alea program using shifts and arithmetic.
The digits and rounding error are derived quantities, not independently
sampled. For a power-of-2 torus, the decomposition is bijective, so
digits and eps happen to be independent anyway — but the deterministic
approach is more principled and generalizes to non-power-of-2 moduli.

### Scalability Choice: Single-Term + Convolution

Even with BDD-based exact inference, a direct model of all `N` terms can
blow up in size. Instead, we compute **one term** exactly in Alea and
convolve `N` independent copies in Julia. This keeps the Alea programs
small while still producing exact PMFs.

## Related Files

- `../Parameter-Selection/python/noiseestimation/keyvariation.py` —
  `extpnoisecalc` (Gaussian variance formula)
- `../TFHEpp/include/trgsw.hpp` — Decomposition and External Product
  implementations
