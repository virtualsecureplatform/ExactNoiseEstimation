#!/usr/bin/env julia
"""Exact noise estimation for TFHE Blind Rotate via CMUXPMBX loops.

Models Blind Rotate as n sequential CMUXPMBX operations.
Default loop composition is stateful:
  E_{t+1} = E_t (+) ext_base
where E_t is the carried ciphertext noise marginal and ext_base is the
fresh external-product base noise for one loop.
"""

import Pkg
const ALEA_PROJECT = joinpath(@__DIR__, "alea")
Pkg.activate(ALEA_PROJECT)

using Alea
using JLD2
using Printf

struct TFHEParams
    N::Int
    k::Int
    l::Int
    la::Int
    Bgbit::Int
    Bg::Int
    eta::Int
    qbit::Int
    q::Int
end

function TFHEParams()
    N = 512
    k = 2
    l = 2
    la = 2
    Bgbit = 9
    Bg = 1 << Bgbit
    eta = 4
    qbit = 25
    q = 1 << qbit
    TFHEParams(N, k, l, la, Bgbit, Bg, eta, qbit, q)
end

remaining_bits(p::TFHEParams) = p.qbit - p.l * p.Bgbit
halfBg(p::TFHEParams) = p.Bg ÷ 2

function offset(p::TFHEParams)
    off = 0
    for i in 1:p.l
        off += (p.Bg ÷ 2) * (1 << (p.qbit - i * p.Bgbit))
    end
    off
end

function roundoffset(p::TFHEParams)
    rb = remaining_bits(p)
    rb > 0 ? (1 << (rb - 1)) : 0
end

total_offset(p::TFHEParams) = offset(p) + roundoffset(p)

function rshift(x::DistUInt{W}, n::Int) where W
    if n <= 0
        return x
    elseif n >= W
        return DistUInt{W}(0)
    else
        return DistUInt{W}(vcat(fill(false, n), x.bits[1:W-n]))
    end
end

reinterpret_signed(x::DistUInt{W}) where W = DistInt{W}(x.bits)

function decompose_digits(c::DistUInt{W}, p::TFHEParams) where W
    c_off = c + DistUInt{W}(total_offset(p))
    digits = DistInt{W}[]
    prev_raw = nothing
    for level in 0:(p.l - 1)
        shift_top = p.qbit - (level + 1) * p.Bgbit
        raw = rshift(c_off, shift_top)
        if level > 0
            raw = raw - (prev_raw << p.Bgbit)
        end
        digit = reinterpret_signed(raw - DistUInt{W}(halfBg(p)))
        push!(digits, digit)
        prev_raw = raw
    end
    digits
end

function decompose_eps(c::DistUInt{W}, p::TFHEParams) where W
    digits = decompose_digits(c, p)
    eps = reinterpret_signed(c)
    for (level, digit) in enumerate(digits)
        shift_val = (p.l - level) * p.Bgbit + remaining_bits(p)
        eps = eps - (digit << shift_val)
    end
    eps
end

function dist_to_dict(d)::Dict{Int, Float64}
    out = Dict{Int, Float64}()
    for (k, v) in d
        out[Int(k)] = float(v)
    end
    out
end

function dist_to_array(dist::Dict{Int, Float64})
    minv = minimum(keys(dist))
    maxv = maximum(keys(dist))
    arr = zeros(Float64, maxv - minv + 1)
    for (k, v) in dist
        arr[k - minv + 1] = v
    end
    return arr, minv
end

function cbd_dist(eta::Int)::Dict{Int, Float64}
    total = 2.0^(2 * eta)
    dist = Dict{Int, Float64}()
    for v in -eta:eta
        dist[v] = binomial(2 * eta, eta + v) / total
    end
    dist
end

function product_dist(a::Dict{Int, Float64}, b::Dict{Int, Float64})
    out = Dict{Int, Float64}()
    for (va, pa) in a
        for (vb, pb) in b
            out[va * vb] = get(out, va * vb, 0.0) + pa * pb
        end
    end
    out
end

function parse_conv_mode()
    for arg in ARGS
        if arg == "--exact" || arg == "--conv=exact"
            return :exact
        elseif arg == "--fft" || arg == "--conv=fft"
            return :fft
        elseif startswith(arg, "--conv=")
            mode = split(arg, "=", limit=2)[2]
            return mode == "exact" ? :exact : :fft
        end
    end
    env_mode = lowercase(get(ENV, "EXTPROD_CONV", "fft"))
    return env_mode == "exact" ? :exact : :fft
end

function parse_loops()
    for arg in ARGS
        if startswith(arg, "--loops=")
            return parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    env_loops = get(ENV, "BR_LOOPS", "")
    return env_loops == "" ? 636 : parse(Int, env_loops)
end

function parse_loop_model()
    for arg in ARGS
        if startswith(arg, "--loop-model=")
            model = lowercase(split(arg, "=", limit=2)[2])
            if model == "stateful" || model == "iid"
                return Symbol(model)
            else
                error("Unknown --loop-model=$(model). Use stateful or iid.")
            end
        elseif arg == "--stateful"
            return :stateful
        elseif arg == "--iid"
            return :iid
        end
    end
    env_model = lowercase(get(ENV, "BR_LOOP_MODEL", "stateful"))
    if env_model == "stateful" || env_model == "iid"
        return Symbol(env_model)
    end
    error("Unknown BR_LOOP_MODEL=$(env_model). Use stateful or iid.")
end

function cmux_pmbx_loop_terms(p::TFHEParams)
    c = uniform(DistUInt{p.qbit}, 0, p.q)
    digits = decompose_digits(c, p)
    d0 = digits[1]
    eps = decompose_eps(c, p)

    s = flip(0.5)
    key_round = ifelse(s, eps, DistInt{p.qbit}(0))
    direct_round = -eps

    digit_dist = dist_to_dict(pr(d0))
    key_round_dist = dist_to_dict(pr(key_round))
    direct_round_dist = dist_to_dict(pr(direct_round))

    digit_arr, digit_off = dist_to_array(digit_dist)
    key_arr, key_off = dist_to_array(key_round_dist)
    direct_arr, direct_off = dist_to_array(direct_round_dist)

    cbd = cbd_dist(p.eta)
    digit_cbd = product_dist(digit_dist, cbd)
    digit_cbd_arr, digit_cbd_off = dist_to_array(digit_cbd)
    cbd_arr, cbd_off = dist_to_array(cbd)

    nonce_count = p.la * p.k * p.N
    nonce_arr, nonce_off = convolve_n(digit_cbd_arr, digit_cbd_off, nonce_count)

    main_count = p.l * p.N
    main_arr, main_off = convolve_n(digit_cbd_arr, digit_cbd_off, main_count)

    n_pos = 1
    n_neg = p.N - 1
    key_pos_arr, key_pos_off = convolve_n(key_arr, key_off, n_pos)
    key_neg_arr, key_neg_off = convolve_n(key_arr, key_off, n_neg)
    neg_key_neg_arr, neg_key_neg_off = negate_dist(key_neg_arr, key_neg_off)
    key_noise_arr, key_noise_off = convolve_pair(
        key_pos_arr, key_pos_off, neg_key_neg_arr, neg_key_neg_off
    )

    # External product base (without input noise)
    ext_base_arr, ext_base_off = convolve_pair(nonce_arr, nonce_off, main_arr, main_off)
    ext_base_arr, ext_base_off = convolve_pair(ext_base_arr, ext_base_off, key_noise_arr, key_noise_off)
    ext_base_arr, ext_base_off = convolve_pair(ext_base_arr, ext_base_off, direct_arr, direct_off)

    # p=1 path (exact): c0 + ext_base + (X^a*e0 - e0) => ext_base + e_a
    exact_p1_arr, exact_p1_off = convolve_pair(ext_base_arr, ext_base_off, cbd_arr, cbd_off)
    # p=0 path: c0 + extprod(TRGSW(0), diff) => ext_base + e0
    p0_arr, p0_off = convolve_pair(ext_base_arr, ext_base_off, cbd_arr, cbd_off)
    # Under this model both branches have the same marginal PMF.
    cmux_exact_arr, cmux_exact_off = mix_two(exact_p1_arr, exact_p1_off, p0_arr, p0_off, 0.5)

    return (
        ext_base_arr=ext_base_arr,
        ext_base_off=ext_base_off,
        input_arr=cbd_arr,
        input_off=cbd_off,
        cmux_arr=cmux_exact_arr,
        cmux_off=cmux_exact_off,
    )
end

function main()
    Alea.set_conv_mode!(parse_conv_mode())
    println("Convolution mode: $(Alea.conv_mode())")

    p = TFHEParams()

    println("="^60)
    println("Step 1: Single-loop CMUXPMBX exact distribution")
    println("="^60)

    terms = cmux_pmbx_loop_terms(p)
    cmux_arr = terms.cmux_arr
    cmux_off = terms.cmux_off
    cmux_stats = dist_stats(cmux_arr, cmux_off)

    @printf("\n  CMUXPMBX exact (single loop):\n")
    @printf("    Support: [%d, %d]\n", cmux_stats.min, cmux_stats.max)
    @printf("    Mean:     %.6f\n", cmux_stats.mean)
    @printf("    Variance: %.6f\n", cmux_stats.variance)

    println("\n" * "="^60)
    println("Step 2: Blind Rotate loops (convolution)")
    println("="^60)

    loops = parse_loops()
    loop_model = parse_loop_model()
    println("  Loop model: $(loop_model)")

    if loop_model == :stateful
        # Stateful composition: one carried noise term plus independent ext_base per loop.
        ext_n_arr, ext_n_off = convolve_n(terms.ext_base_arr, terms.ext_base_off, loops)
        br_arr, br_off = convolve_pair(ext_n_arr, ext_n_off, terms.input_arr, terms.input_off)
    else
        # Legacy approximation: i.i.d. convolution of single-loop CMUX output PMF.
        br_arr, br_off = convolve_n(cmux_arr, cmux_off, loops)
    end
    br_stats = dist_stats(br_arr, br_off)

    @printf("\n  Blind Rotate total (n=%d):\n", loops)
    @printf("    Support: [%d, %d]\n", br_stats.min, br_stats.max)
    @printf("    Mean:     %.6f\n", br_stats.mean)
    @printf("    Variance: %.6f\n", br_stats.variance)
    @printf("    Std:      %.6f\n", br_stats.std)
    @printf("    Sum(P):   %.10f\n", br_stats.total)

    println("\n    Tail probabilities:")
    for mult in 1:5
        @printf("      P(|X-mu|>%ds) = %.6e\n", mult, br_stats.tails[mult])
    end

    # Save full PMF to JLD2
    outdir = joinpath(@__DIR__, "generated")
    mkpath(outdir)
    outfile = joinpath(outdir, "br_n$(loops)_$(loop_model).jld2")
    jldsave(outfile; br_arr, br_off, loops, loop_model=String(loop_model),
            cmux_arr, cmux_off, params=p)
    println("\n  PMF saved to $(outfile)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
