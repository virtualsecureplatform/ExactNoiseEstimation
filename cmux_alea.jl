#!/usr/bin/env julia
"""Exact noise estimation for TFHE CMUX using Alea.jl.

Models CMUX via External Product + addition and highlights dependence between
input-difference noise and the added ciphertext.
"""

import Pkg
const ALEA_PROJECT = joinpath(@__DIR__, "alea")
Pkg.activate(ALEA_PROJECT)

using Alea
using FFTW
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

function main()
    Alea.set_conv_mode!(parse_conv_mode())
    println("Convolution mode: $(Alea.conv_mode())")

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
    dcbd_stats = dist_stats(digit_cbd_arr, digit_cbd_off)
    @printf("    digit*CBD: support [%d,%d], mean=%.4f, var=%.4f\n",
        dcbd_stats.min, dcbd_stats.max, dcbd_stats.mean, dcbd_stats.variance)

    cbd_arr, cbd_off = dist_to_array(cbd)
    cbd_stats = dist_stats(cbd_arr, cbd_off)
    @printf("    input_cbd: support [%d,%d], mean=%.4f, var=%.4f\n",
        cbd_stats.min, cbd_stats.max, cbd_stats.mean, cbd_stats.variance)

    println("\n" * "="^60)
    println("Step 2: Build External Product components")
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

    # External product base (without input noise)
    ext_base_arr, ext_base_off = convolve_pair(nonce_arr, nonce_off, main_arr, main_off)
    ext_base_arr, ext_base_off = convolve_pair(ext_base_arr, ext_base_off, key_noise_arr, key_noise_off)
    ext_base_arr, ext_base_off = convolve_pair(ext_base_arr, ext_base_off, direct_arr, direct_off)
    ext_base_stats = dist_stats(ext_base_arr, ext_base_off)
    @printf("\n  External Product base (no input noise):\n")
    @printf("    Support: [%d, %d]\n", ext_base_stats.min, ext_base_stats.max)
    @printf("    Mean:     %.6f\n", ext_base_stats.mean)
    @printf("    Variance: %.6f\n", ext_base_stats.variance)

    # Input difference noise: e1 - e0
    neg_cbd_arr, neg_cbd_off = negate_dist(cbd_arr, cbd_off)
    input_diff_arr, input_diff_off = convolve_pair(cbd_arr, cbd_off, neg_cbd_arr, neg_cbd_off)
    input_diff_stats = dist_stats(input_diff_arr, input_diff_off)
    @printf("\n  Input diff noise (e1 - e0):\n")
    @printf("    Support: [%d, %d]\n", input_diff_stats.min, input_diff_stats.max)
    @printf("    Mean:     %.6f\n", input_diff_stats.mean)
    @printf("    Variance: %.6f\n", input_diff_stats.variance)

    # External product total using input diff
    ext_total_arr, ext_total_off = convolve_pair(ext_base_arr, ext_base_off, input_diff_arr, input_diff_off)
    ext_total_stats = dist_stats(ext_total_arr, ext_total_off)
    @printf("\n  External Product total (with input diff):\n")
    @printf("    Support: [%d, %d]\n", ext_total_stats.min, ext_total_stats.max)
    @printf("    Mean:     %.6f\n", ext_total_stats.mean)
    @printf("    Variance: %.6f\n", ext_total_stats.variance)

    println("\n" * "="^60)
    println("Step 3: CMUX noise distributions")
    println("="^60)

    # Naive CMUX: external product (with input diff) + c0 noise (assumed independent)
    cmux_naive_arr, cmux_naive_off = convolve_pair(ext_total_arr, ext_total_off, cbd_arr, cbd_off)
    cmux_naive_stats = dist_stats(cmux_naive_arr, cmux_naive_off)
    @printf("\n  CMUX naive (extprod + c0 noise, independent):\n")
    @printf("    Support: [%d, %d]\n", cmux_naive_stats.min, cmux_naive_stats.max)
    @printf("    Mean:     %.6f\n", cmux_naive_stats.mean)
    @printf("    Variance: %.6f\n", cmux_naive_stats.variance)

    # Exact CMUX (dependency accounted): ext_base + e1 (since -e0 + e0 cancels)
    cmux_exact_arr, cmux_exact_off = convolve_pair(ext_base_arr, ext_base_off, cbd_arr, cbd_off)
    cmux_exact_stats = dist_stats(cmux_exact_arr, cmux_exact_off)
    @printf("\n  CMUX exact (dependency accounted):\n")
    @printf("    Support: [%d, %d]\n", cmux_exact_stats.min, cmux_exact_stats.max)
    @printf("    Mean:     %.6f\n", cmux_exact_stats.mean)
    @printf("    Variance: %.6f\n", cmux_exact_stats.variance)

    println("\n  Tail probabilities (exact CMUX):")
    for mult in 1:5
        @printf("    P(|X-mu|>%ds) = %.6e\n", mult, cmux_exact_stats.tails[mult])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
