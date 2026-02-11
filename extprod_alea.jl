#!/usr/bin/env julia
"""Exact noise estimation for TFHE External Product using Alea.jl.

Computes exact single-term distributions with Alea (BDD-based inference),
then builds full component distributions via convolution and compares
against the Gaussian variance formula from Parameter-Selection.
"""

import Pkg
const ALEA_PROJECT = joinpath(@__DIR__, "alea")
Pkg.activate(ALEA_PROJECT)

using Alea
using FFTW
using Printf

const PT_MODE = Ref(:fixed) # :fixed (p=1) or :binary (p~Bernoulli(0.5))

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

# Right shift for DistUInt (not provided by Alea)
function rshift(x::DistUInt{W}, n::Int) where W
    if n <= 0
        return x
    elseif n >= W
        return DistUInt{W}(0)
    else
        return DistUInt{W}(vcat(fill(false, n), x.bits[1:W-n]))
    end
end

# Interpret an unsigned bit pattern as a signed two's-complement integer
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

function normalize_pmf!(arr::Vector{Float64})
    for i in eachindex(arr)
        if arr[i] < 0 && arr[i] > -1e-12
            arr[i] = 0.0
        end
    end
    s = sum(arr)
    s == 0.0 || (arr ./= s)
    return arr
end

function convolve_fft(a::Vector{Float64}, b::Vector{Float64})
    n = length(a) + length(b) - 1
    nfft = nextpow(2, n)
    apad = zeros(Float64, nfft)
    bpad = zeros(Float64, nfft)
    apad[1:length(a)] = a
    bpad[1:length(b)] = b
    fa = rfft(apad)
    fb = rfft(bpad)
    c = irfft(fa .* fb, nfft)
    c = c[1:n]
    normalize_pmf!(c)
    return c
end

function convolve_exact(a::Vector{Float64}, b::Vector{Float64})
    n = length(a) + length(b) - 1
    c = zeros(Float64, n)
    @inbounds for i in eachindex(a)
        ai = a[i]
        if ai != 0.0
            for j in eachindex(b)
                c[i + j - 1] += ai * b[j]
            end
        end
    end
    normalize_pmf!(c)
    return c
end

function convolve_impl()
    Alea.conv_mode() == :fft ? convolve_fft : convolve_exact
end

function convolve_pair(a::Vector{Float64}, off_a::Int, b::Vector{Float64}, off_b::Int)
    conv = convolve_impl()
    return conv(a, b), off_a + off_b
end

function convolve_n(arr::Vector{Float64}, off::Int, n::Int)
    if n == 0
        return [1.0], 0
    end
    res = [1.0]
    base = arr
    m = n
    conv = convolve_impl()
    while m > 0
        if (m & 1) == 1
            res = conv(res, base)
        end
        m >>= 1
        if m > 0
            base = conv(base, base)
        end
    end
    return res, off * n
end

function negate_dist(arr::Vector{Float64}, off::Int)
    new_off = -(off + length(arr) - 1)
    return reverse(arr), new_off
end

function dist_stats(arr::Vector{Float64}, off::Int)
    total = sum(arr)
    if total == 0.0
        return (mean=0.0, variance=0.0, std=0.0, total=0.0,
                tails=Dict{Int, Float64}(), min=0, max=0)
    end
    mean = 0.0
    for i in eachindex(arr)
        mean += (off + (i - 1)) * arr[i]
    end
    mean /= total
    var = 0.0
    for i in eachindex(arr)
        v = off + (i - 1) - mean
        var += v * v * arr[i]
    end
    var /= total
    std = sqrt(var)
    tails = Dict{Int, Float64}()
    for mult in 1:5
        thresh = mult * std
        tail = 0.0
        for i in eachindex(arr)
            v = off + (i - 1)
            if abs(v - mean) > thresh
                tail += arr[i]
            end
        end
        tails[mult] = tail / total
    end
    minv = off
    maxv = off + length(arr) - 1
    return (mean=mean, variance=var, std=std, total=total, tails=tails, min=minv, max=maxv)
end

function decomp_round_variance_pow2(qbit::Int, basebit::Int, levels::Int)
    kept_bits = levels * basebit
    remaining = qbit - kept_bits
    if remaining <= 0
        return 0.0
    end
    roundwidth = 2.0^remaining
    return roundwidth * roundwidth / 12.0 - 1.0 / 12.0
end

function extpnoisecalc_components(p::TFHEParams; exp_key=0.5, var_key=0.25)
    cbd_var = p.eta / 2.0
    Ed2 = (p.Bg^2 + 2) / 12.0

    nonce_digit_cbd_var = p.la * p.k * p.N * Ed2 * cbd_var
    main_digit_cbd_var = p.l * p.N * Ed2 * cbd_var

    nonce_roundvar = decomp_round_variance_pow2(p.qbit, p.Bgbit, p.la)
    nonnonce_roundvar = decomp_round_variance_pow2(p.qbit, p.Bgbit, p.l)

    key_round_var = nonce_roundvar * (p.k * p.N * (var_key + exp_key^2))
    key_extra_var = p.k * p.N / 4.0 * var_key
    direct_round_var = nonnonce_roundvar
    input_var = cbd_var

    comp3_var = key_round_var + key_extra_var + direct_round_var + input_var

    return Dict(
        "nonce_digit_cbd" => nonce_digit_cbd_var,
        "main_digit_cbd" => main_digit_cbd_var,
        "key_round" => key_round_var,
        "key_extra" => key_extra_var,
        "direct_round" => direct_round_var,
        "input_noise" => input_var,
        "comp3_total" => comp3_var,
        "total" => nonce_digit_cbd_var + main_digit_cbd_var + comp3_var,
    )
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

function parse_pt_mode()
    for arg in ARGS
        if arg == "--trgsw-pt=binary" || arg == "--pt=binary"
            return :binary
        elseif arg == "--trgsw-pt=fixed" || arg == "--pt=fixed"
            return :fixed
        elseif startswith(arg, "--trgsw-pt=") || startswith(arg, "--pt=")
            mode = split(arg, "=", limit=2)[2]
            return mode == "binary" ? :binary : :fixed
        end
    end
    env_mode = lowercase(get(ENV, "TRGSW_PT", "fixed"))
    return env_mode == "binary" ? :binary : :fixed
end

function mix_with_delta0(arr::Vector{Float64}, off::Int, p_one::Float64)
    # Mix p_one * arr + (1-p_one) * delta0
    if p_one <= 0.0
        return [1.0], 0
    elseif p_one >= 1.0
        return arr, off
    end
    minv = min(off, 0)
    maxv = max(off + length(arr) - 1, 0)
    out = zeros(Float64, maxv - minv + 1)
    for i in eachindex(arr)
        out[(off - minv) + i] += p_one * arr[i]
    end
    out[(0 - minv) + 1] += (1.0 - p_one)
    normalize_pmf!(out)
    return out, minv
end

function main()
    Alea.set_conv_mode!(parse_conv_mode())
    println("Convolution mode: $(Alea.conv_mode())")
    PT_MODE[] = parse_pt_mode()
    println("TRGSW plaintext mode: $(PT_MODE[])")

    p = TFHEParams()

    println("="^60)
    println("Step 1: Single-term distributions via Alea")
    println("="^60)

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

    digit_stats = dist_stats(digit_arr, digit_off)
    key_stats = dist_stats(key_arr, key_off)
    direct_stats = dist_stats(direct_arr, direct_off)

    @printf("  single_digit: support [%d,%d], mean=%.4f, var=%.4f\n",
        digit_stats.min, digit_stats.max, digit_stats.mean, digit_stats.variance)
    @printf("  single_key_round: support [%d,%d], mean=%.4f, var=%.4f\n",
        key_stats.min, key_stats.max, key_stats.mean, key_stats.variance)
    @printf("  single_direct_round: support [%d,%d], mean=%.4f, var=%.4f\n",
        direct_stats.min, direct_stats.max, direct_stats.mean, direct_stats.variance)

    println("\n  Computing digit×CBD(eta) product analytically...")
    cbd = cbd_dist(p.eta)
    digit_cbd = product_dist(digit_dist, cbd)
    digit_cbd_arr, digit_cbd_off = dist_to_array(digit_cbd)
    digit_cbd_stats = dist_stats(digit_cbd_arr, digit_cbd_off)
    @printf("    digit*CBD: support [%d,%d], mean=%.4f, var=%.4f\n",
        digit_cbd_stats.min, digit_cbd_stats.max, digit_cbd_stats.mean, digit_cbd_stats.variance)

    cbd_arr, cbd_off = dist_to_array(cbd)
    cbd_stats = dist_stats(cbd_arr, cbd_off)
    @printf("    input_cbd: support [%d,%d], mean=%.4f, var=%.4f\n",
        cbd_stats.min, cbd_stats.max, cbd_stats.mean, cbd_stats.variance)

    println("\n" * "="^60)
    println("Step 2: Build component distributions via convolution")
    println("="^60)

    nonce_count = p.la * p.k * p.N
    nonce_arr, nonce_off = convolve_n(digit_cbd_arr, digit_cbd_off, nonce_count)
    nonce_stats = dist_stats(nonce_arr, nonce_off)
    @printf("\n  Nonce digit*CBD (%d terms):\n", nonce_count)
    @printf("    Support: [%d, %d]\n", nonce_stats.min, nonce_stats.max)
    @printf("    Mean:     %.6f\n", nonce_stats.mean)
    @printf("    Variance: %.6f\n", nonce_stats.variance)

    main_count = p.l * p.N
    main_arr, main_off = convolve_n(digit_cbd_arr, digit_cbd_off, main_count)
    main_stats = dist_stats(main_arr, main_off)
    @printf("\n  Main digit*CBD (%d terms):\n", main_count)
    @printf("    Support: [%d, %d]\n", main_stats.min, main_stats.max)
    @printf("    Mean:     %.6f\n", main_stats.mean)
    @printf("    Variance: %.6f\n", main_stats.variance)

    n_pos = 1
    n_neg = p.N - 1
    key_pos_arr, key_pos_off = convolve_n(key_arr, key_off, n_pos)
    key_neg_arr, key_neg_off = convolve_n(key_arr, key_off, n_neg)
    neg_key_neg_arr, neg_key_neg_off = negate_dist(key_neg_arr, key_neg_off)
    key_noise_arr, key_noise_off = convolve_pair(
        key_pos_arr, key_pos_off, neg_key_neg_arr, neg_key_neg_off
    )
    key_noise_stats = dist_stats(key_noise_arr, key_noise_off)
    @printf("\n  Key*rounding (%d pos - %d neg):\n", n_pos, n_neg)
    @printf("    Support: [%d, %d]\n", key_noise_stats.min, key_noise_stats.max)
    @printf("    Mean:     %.6f\n", key_noise_stats.mean)
    @printf("    Variance: %.6f\n", key_noise_stats.variance)

    comp3_arr, comp3_off = convolve_pair(key_noise_arr, key_noise_off, direct_arr, direct_off)
    comp3_arr, comp3_off = convolve_pair(comp3_arr, comp3_off, cbd_arr, cbd_off)
    comp3_stats = dist_stats(comp3_arr, comp3_off)
    @printf("\n  Component 3 total (key + direct + input):\n")
    @printf("    Support: [%d, %d]\n", comp3_stats.min, comp3_stats.max)
    @printf("    Mean:     %.6f\n", comp3_stats.mean)
    @printf("    Variance: %.6f\n", comp3_stats.variance)

    println("\n" * "="^60)
    println("Step 3: Convolve all components for total noise distribution")
    println("="^60)

    total_arr, total_off = convolve_pair(nonce_arr, nonce_off, main_arr, main_off)
    total_arr, total_off = convolve_pair(total_arr, total_off, comp3_arr, comp3_off)
    total_stats = dist_stats(total_arr, total_off)

    @printf("\n  Total noise distribution:\n")
    @printf("    Support: [%d, %d]\n", total_stats.min, total_stats.max)
    @printf("    Mean:     %.6f\n", total_stats.mean)
    @printf("    Variance: %.6f\n", total_stats.variance)
    @printf("    Std:      %.6f\n", total_stats.std)
    @printf("    Sum(P):   %.10f\n", total_stats.total)

    println("\n    Tail probabilities:")
    for mult in 1:5
        @printf("      P(|X-mu|>%ds) = %.6e\n", mult, total_stats.tails[mult])
    end

    if PT_MODE[] == :binary
        pt_arr, pt_off = mix_with_delta0(total_arr, total_off, 0.5)
        pt_stats = dist_stats(pt_arr, pt_off)
        @printf("\n  Total distribution with binary TRGSW plaintext (p~Bernoulli(0.5)):\n")
        @printf("    Support: [%d, %d]\n", pt_stats.min, pt_stats.max)
        @printf("    Mean:     %.6f\n", pt_stats.mean)
        @printf("    Variance: %.6f\n", pt_stats.variance)
        @printf("    Std:      %.6f\n", pt_stats.std)
        @printf("    Sum(P):   %.10f\n", pt_stats.total)
        println("\n    Tail probabilities:")
        for mult in 1:5
            @printf("      P(|X-mu|>%ds) = %.6e\n", mult, pt_stats.tails[mult])
        end
    end

    println("\n" * "="^60)
    println("Step 4: Comparison with Gaussian approximation (extpnoisecalc)")
    println("="^60)

    expected = extpnoisecalc_components(p)

    println("\n  Gaussian formula breakdown:")
    for key in ["nonce_digit_cbd", "main_digit_cbd", "key_round", "key_extra",
                "direct_round", "input_noise", "comp3_total", "total"]
        @printf("    %-25s = %.6f\n", key, expected[key])
    end

    println("\n  Component                           Exact Var    Gauss Var     Ratio")
    println("  " * "-"^35 * " " * "-"^12 * " " * "-"^12 * " " * "-"^8)
    comparisons = [
        ("Nonce digit*CBD", nonce_stats.variance, "nonce_digit_cbd"),
        ("Main digit*CBD", main_stats.variance, "main_digit_cbd"),
        ("Comp3 (key+direct+input)", comp3_stats.variance, "comp3_total"),
    ]
    for (label, exact_v, key) in comparisons
        gauss_v = expected[key]
        ratio = gauss_v > 0 ? exact_v / gauss_v : Inf
        @printf("  %-35s %12.4f %12.4f %8.4f\n", label, exact_v, gauss_v, ratio)
    end
    exact_total = total_stats.variance
    gauss_total = expected["total"]
    ratio = gauss_total > 0 ? exact_total / gauss_total : Inf
    @printf("  %-35s %12.4f %12.4f %8.4f\n", "TOTAL", exact_total, gauss_total, ratio)

    println("\n" * "="^60)
    println("Full PMF (non-zero entries)")
    println("="^60)

    println("\n  Value    Probability")
    printed = 0
    for i in eachindex(total_arr)
        v = total_arr[i]
        if v > 1e-15
            printed += 1
            @printf("  %6d  %15.10e\n", total_off + (i - 1), v)
        end
    end
    println("\n  Total entries > 1e-15: $printed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
