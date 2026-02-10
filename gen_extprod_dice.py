#!/usr/bin/env python3
"""Generate Dice programs for exact noise estimation of TFHE External Product.

Generates single-term .dice programs for each independent noise component.
All iteration/accumulation is done via Python convolution (not BDD iterate),
keeping each Dice program small (~12 BDD variables).

Toy parameters: N=4, k=1, l=la=2, Bgbit=2, Bg=4, eta=1, qbit=8, q=256.
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
    eta: int = 1      # CBD parameter
    qbit: int = 8     # log2(q)
    q: int = 256      # ciphertext modulus

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


def gen_single_digit_cbd(params: TFHEParams, output_dir: str) -> str:
    """Generate single-term: sum_i d_i * ebar_i for one polynomial coefficient.

    This is one term of the nonce or main digit×CBD sum. Since la=l and
    Bga=Bg for our toy params, nonce and main share the same single-term
    distribution.

    Uses conditional branching for CBD(1) to avoid BDD multiply.
    """
    P = params
    qbit = P.qbit

    lines = []
    lines.append(f"// Single term: d0*e0 + d1*e1 where d_i from decomposition, e_i ~ CBD({P.eta})")
    lines.append(f"// d_i in {{-{P.halfBg},...,{P.halfBg - 1}}}, e_i in {{-{P.eta},...,{P.eta}}}")
    lines.append(f"let c = {format_uniform(qbit)} in")

    decomp_code, digit_vars = gen_decomp_digits_only("c", P, indent="")
    lines.append(decomp_code)

    # Generate optimized digit*CBD(1) products using conditionals
    for level in range(P.l):
        dv = digit_vars[level]
        nz = f"e{level}_nz"
        pos = f"e{level}_pos"
        pv = f"prod{level}"
        lines.append(f"let {nz} = flip 0.5 in")
        lines.append(f"let {pos} = flip 0.5 in")
        lines.append(
            f"let {pv} = if {nz} then "
            f"(if {pos} then {dv} else int({qbit}, 0) - {dv}) "
            f"else int({qbit}, 0) in"
        )

    prod_vars = [f"prod{i}" for i in range(P.l)]
    lines.append(" + ".join(prod_vars))

    program = "\n".join(lines)
    path = os.path.join(output_dir, "single_digit_cbd.dice")
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
    lines.append(f"let c = {format_uniform(qbit)} in")

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
    lines.append(f"let c = {format_uniform(qbit)} in")

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

    gen_single_digit_cbd(params, output_dir)
    gen_single_key_round(params, output_dir)
    gen_single_direct_round(params, output_dir)

    print(f"\nAll single-term programs generated in {output_dir}/")
    print(f"\nConvolution plan (done in analyze_output.py):")
    print(f"  Nonce digit*CBD: convolve {params.k * params.N} copies of single_digit_cbd")
    print(f"  Main digit*CBD:  convolve {params.N} copies of single_digit_cbd")
    print(f"  Key*rounding:    convolve 1 pos + {params.N - 1} neg copies of single_key_round")
    print(f"  Direct rounding: 1 copy of single_direct_round")
    print(f"  Input noise:     CBD({params.eta}) distribution (computed analytically)")


if __name__ == "__main__":
    main()
