using MKL
using Random
using CSV
using DataFrames
using MAT
using LinearAlgebra
using Statistics
using Printf
using JuMP
using KNITRO
using Hypatia
import MathOptInterface as MOI

include("../misc/util.jl")
include("../misc/dual_glinx.jl")
include("../misc/heuristics.jl")
include("../solvers/solver_knitro.jl")
include("../gscaling/gscaling_glinx.jl")

function haar_orthogonal(n::Int; rng::AbstractRNG = Random.default_rng())
    F = qr(randn(rng, n, n))
    return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
end

ipr(u::AbstractVector) = sum(x -> x^4, u) / sum(abs2, u)^2

function eig_desc(C0::AbstractMatrix)
    F = eigen(Symmetric(Matrix(C0)))
    return reverse(F.values), F.vectors[:, end:-1:1]
end

function make_instance(; n::Int, t::Int, δ::Int = 1, ψ0::Real = 1.0, σ::Real = 10.0,
                       ε::Real = 0.5, m::Int = 20, jit_top::Real = 0.05, jit_bot::Real = 1e-3,
                       Q = nothing, base_eigs = nothing,
                       rng::AbstractRNG = Random.default_rng())

    r = t + δ
    nmid = n - r - m

    nmid >= 0 || error("need t + δ + m ≤ n")
    ε + jit_bot < σ || @warn "ε + jit_bot ≥ σ: the 'top' block may not be the top of the spectrum"

    top = ψ0 .+ σ .* (1 .+ jit_top .* rand(rng, r))

    if base_eigs === nothing
        v = rand(rng, nmid)
    else
        b = sort(collect(base_eigs); rev = true)
        mid_orig = b[r+1:r+nmid]
        lo, hi = minimum(b), (nmid > 0 ? maximum(mid_orig) : 1.0)
        v = hi > lo ? (mid_orig .- lo) ./ (hi - lo) : zeros(nmid)
    end

    mid = ψ0 .+ jit_bot .+ ε .* v
    bot = ψ0 .+ jit_bot .* rand(rng, m)

    m > 0 && (bot[argmin(bot)] = ψ0)

    λ = vcat(sort(top; rev = true), sort(mid; rev = true), sort(bot; rev = true))

    Qm = Q === nothing ? haar_orthogonal(n; rng) : Matrix(Q)
    size(Qm) == (n, n) || error("Q must be $n×$n")

    C = Qm * Diagonal(λ) * Qm'

    return Symmetric((C + C') / 2), λ, Qm
end

function generate_dataset(; n::Int, t::Int, configs, instances_per_config::Int = 1,
                          Q = nothing, base_eigs = nothing,
                          output_dir::AbstractString = "data/generated", seed::Int = 2026)

    mkpath(output_dir)

    instance_id = 0

    for config in configs
        for rep in 1:instances_per_config
            instance_id += 1
            rng = MersenneTwister(seed + instance_id)

            C, λ, _ = make_instance(; n, t, δ = config.δ, ψ0 = config.ψ0,
                                    σ = config.σ, ε = config.ε, m = config.m,
                                    Q, base_eigs, rng)

            filename = "data$(n)_inst$(lpad(instance_id, 3, '0')).mat"
            filepath = joinpath(output_dir, filename)

            file = matopen(filepath, "w")
            write(file, "C", Matrix(C))
            write(file, "lambda", λ)
            write(file, "n", n)
            write(file, "t", t)
            write(file, "delta", config.δ)
            write(file, "psi0", config.ψ0)
            write(file, "sigma", config.σ)
            write(file, "epsilon", config.ε)
            write(file, "m", config.m)
            write(file, "seed", seed + instance_id)
            close(file)

            println("Saved $filename  (δ=$(config.δ), σ=$(config.σ), " *
                    "ε=$(config.ε), m=$(config.m))")
        end
    end

    println("\nGenerated $instance_id instances in $output_dir")
end

function node_check(C::AbstractMatrix, ψ::Real, t::Int, s::Int,
                    J::AbstractVector{Int}, Z::AbstractVector{Int}; tol::Real = 1e-9)

    n = size(C, 1)

    fixed = falses(n)
    fixed[J] .= true
    fixed[Z] .= true

    free = findall(!, fixed)
    k = s - length(J)

    (k >= 0 && k <= length(free)) ||
        error("infeasible node: |J| = $(length(J)), |Z| = $(length(Z))")

    Sbar = if k == 0
        Int[]
    else
        d = [C[i, i] for i in free]
        free[partialsortperm(d, 1:k; rev = true)]
    end

    T = vcat(J, Sbar)

    B = Matrix(C[T, T]) - ψ * I

    trB = tr(B)
    λmax = eigmax(Symmetric(B))

    holds = trB - t * λmax >= -tol * max(1.0, abs(trB))

    return (
        holds = holds,
        ρ = trB / λmax,
        UB = t * log(trB / t + ψ),
        T = T
    )
end

function gmesp_obj(C::AbstractMatrix, S::AbstractVector{Int}, t::Int)
    λ = eigvals(Symmetric(Matrix(C[S, S])))
    q = min(t, length(S))
    return sum(log, λ[end-q+1:end])
end

function local_search_lb(C::AbstractMatrix, s::Int, t::Int; maxpasses::Int = 20)

    n = size(C, 1)

    S = Int[]
    inS = falses(n)

    for _ in 1:s
        best, bi = -Inf, 0

        for i in 1:n
            inS[i] && continue

            v = gmesp_obj(C, vcat(S, i), t)
            v > best && ((best, bi) = (v, i))
        end

        push!(S, bi)
        inS[bi] = true
    end

    val = gmesp_obj(C, S, t)

    for _ in 1:maxpasses
        improved = false

        for a in eachindex(S), j in 1:n
            inS[j] && continue

            S2 = copy(S)
            S2[a] = j

            v = gmesp_obj(C, S2, t)

            if v > val + 1e-12
                inS[S[a]] = false
                inS[j] = true
                S, val, improved = S2, v, true
                break
            end
        end

        improved || break
    end

    return val, sort(S)
end

function weighted_sample(rng::AbstractRNG, idx::AbstractVector{Int},
                         w::AbstractVector, k::Int)

    k == 0 && return Int[]

    keys = log.(rand(rng, length(idx))) ./ w

    return idx[partialsortperm(keys, 1:k; rev = true)]
end

function score_instance(C::AbstractMatrix; t::Int, s_list, nsamples::Int = 2000,
                        β::Real = 1.0, weights = nothing, max_zero_frac::Real = 0.5,
                        tol::Real = 1e-9, compute_lb::Bool = true,
                        rng::AbstractRNG = Random.default_rng())

    n = size(C, 1)
    ψ = eigmin(Symmetric(Matrix(C)))

    w = weights === nothing ?
        [max(C[i, i] - ψ, 1e-12)^β for i in 1:n] :
        max.(collect(float.(weights)), 1e-12)

    all_idx = collect(1:n)
    results = NamedTuple[]

    for s in s_list

        root = node_check(C, ψ, t, s, Int[], Int[]; tol)

        LB = compute_lb ? first(local_search_lb(C, s, t)) : -Inf

        nholds = 0
        npruned = 0
        nshallow = 0
        hshallow = 0

        ρs = Float64[]

        maxZ = floor(Int, max_zero_frac * (n - s))

        for _ in 1:nsamples

            nJ = rand(rng, 0:s-1)
            J = weighted_sample(rng, all_idx, w, nJ)

            rest = setdiff(all_idx, J)

            nZ = rand(rng, 0:maxZ)
            Z = weighted_sample(rng, rest, 1 ./ w[rest], nZ)

            res = node_check(C, ψ, t, s, J, Z; tol)

            push!(ρs, res.ρ)

            nholds += res.holds
            npruned += (res.UB <= LB)

            if nJ <= s ÷ 2
                nshallow += 1
                hshallow += res.holds
            end
        end

        push!(results, (
            s = s,
            ψ = ψ,
            root_holds = root.holds,
            root_ρ = root.ρ,
            root_UB = root.UB,
            LB = LB,
            root_gap_tr = root.UB - LB,
            frac_holds = nholds / nsamples,
            frac_holds_shallow = nshallow > 0 ? hshallow / nshallow : NaN,
            frac_holds_deep = (nsamples - nshallow) > 0 ?
                              (nholds - hshallow) / (nsamples - nshallow) : NaN,
            frac_trace_prunes = npruned / nsamples,
            ρ_q10 = quantile(ρs, 0.10),
            ρ_med = median(ρs),
            ρ_q90 = quantile(ρs, 0.90)
        ))
    end

    return results
end

function print_scores(results; t)

    @printf(
        "%4s %6s %7s %9s %8s %8s %8s %8s %8s  %s\n",
        "s", "root", "root_ρ", "root_gap", "holds",
        "shallow", "deep", "trPrune", "ρ_med", "ρ(q10,q90)"
    )

    for r in results
        @printf(
            "%4d %6s %7.3f %9.4f %8.3f %8.3f %8.3f %8.3f %8.3f  (%.2f, %.2f)\n",
            r.s,
            r.root_holds ? "yes" : "no",
            r.root_ρ,
            r.root_gap_tr,
            r.frac_holds,
            r.frac_holds_shallow,
            r.frac_holds_deep,
            r.frac_trace_prunes,
            r.ρ_med,
            r.ρ_q10,
            r.ρ_q90
        )
    end

    println(
        "  (condition holds ⇔ ρ ≥ t = $t; root_gap uses the trace bound, which equals",
        " the DDGFact⁺ root bound when root = yes)"
    )
end

function grid_search(; n::Int, t::Int, s_list,
                     δs = 0:3,
                     σ_over_ψ0 = [5.0, 20.0, 100.0],
                     εs = [0.1, 0.5, 2.0],
                     ms = [20, 30, 40],
                     ψ0::Real = 1.0,
                     Q = nothing,
                     base_eigs = nothing,
                     nsamples::Int = 500,
                     compute_lb::Bool = false,
                     seed::Int = 1)

    rows = []

    for δ in δs, ratio in σ_over_ψ0, ε in εs, m in ms

        t + δ + m <= n || continue

        rng = MersenneTwister(seed)

        C, _, _ = make_instance(
            ;
            n,
            t,
            δ,
            ψ0,
            σ = ratio * ψ0,
            ε = ε * ψ0,
            m,
            Q,
            base_eigs,
            rng
        )

        res = score_instance(
            C;
            t,
            s_list,
            nsamples,
            compute_lb,
            rng
        )

        push!(rows, (
            δ = δ,
            σ_over_ψ0 = ratio,
            ε = ε,
            m = m,
            mean_holds = mean(r.frac_holds for r in res),
            root_all = all(r.root_holds for r in res),
            min_holds = minimum(r.frac_holds for r in res)
        ))
    end

    sort!(rows; by = r -> -r.mean_holds)

    @printf(
        "%3s %9s %6s %4s %10s %10s %9s\n",
        "δ", "σ/ψ0", "ε/ψ0", "m",
        "mean_hold", "min_hold", "root_all"
    )

    for r in rows
        @printf(
            "%3d %9.1f %6.2f %4d %10.3f %10.3f %9s\n",
            r.δ,
            r.σ_over_ψ0,
            r.ε,
            r.m,
            r.mean_holds,
            r.min_holds,
            r.root_all ? "yes" : "no"
        )
    end

    return rows
end

function generate_candidates(; n::Int, t::Int, δs = 0:3, σ_over_ψ0 = [5.0, 20.0, 100.0], εs = [0.1, 0.5, 2.0], ms = [20, 30, 40], ψ0::Real = 1.0, jit_top::Real = 0.05, jit_bot::Real = 1e-3, Q = nothing, base_eigs = nothing, source_file::AbstractString, seed::Int = 1)

    candidates = NamedTuple[]
    candidate_id = 0

    println()
    println("============================================================")
    println("GENERATING INSTANCES")
    println("============================================================")

    for δ in δs
        for ratio in σ_over_ψ0
            for εratio in εs
                for mval in ms
                    candidate_id += 1

                    generation_seed = seed + candidate_id
                    rng = MersenneTwister(generation_seed)

                    C, λ, _ = make_instance(; n, t, δ, ψ0, σ = ratio * ψ0, ε = εratio * ψ0, m = mval, jit_top, jit_bot, Q, base_eigs, rng)

                    push!(candidates, (
                        id = candidate_id,
                        C = Matrix(C),
                        λ = λ,
                        n = n,
                        t = t,
                        delta = δ,
                        psi0 = Float64(ψ0),
                        sigma = Float64(ratio * ψ0),
                        epsilon = Float64(εratio * ψ0),
                        m = mval,
                        jit_top = Float64(jit_top),
                        jit_bot = Float64(jit_bot),
                        generation_seed = generation_seed,
                        source_file = String(source_file)
                    ))
                end
            end
        end
    end

    println("Generated $(length(candidates)) instances.")

    return candidates
end


function evaluate_candidates(candidates; s_list, nsamples::Int = 1000, glinx_tol::Real = 1e-2, atol::Real = 1e-10, sampling_seed::Int = 10000)

    rows = NamedTuple[]

    println()
    println("============================================================")
    println("EVALUATING $(length(candidates)) INSTANCES")
    println("============================================================")

    for (j, candidate) in enumerate(candidates)
        rng = MersenneTwister(sampling_seed + candidate.id)

        res = score_instance(candidate.C; t = candidate.t, s_list, nsamples, compute_lb = false, rng)

        Csym = Symmetric(candidate.C)
        psi = eigmin(Csym) - atol

        for r in res
            s = r.s

            _, z_ddgfact_plus = aug_ddfact_gmesp(Csym, s, candidate.t, psi; atol = atol)

            gamma_opt = compute_gamma_linx_armijo(Csym, s, candidate.t; tol = glinx_tol)
            _, _, z_glinx, _ = linx_gmesp(Csym, s, candidate.t; gamma = gamma_opt, return_model = true)

            _, z_ls = run_all_LS(Csym, s, candidate.t)

            ddgfact_plus_gap = z_ddgfact_plus - z_ls
            glinx_gap = z_glinx - z_ls
            gap_difference = glinx_gap - ddgfact_plus_gap

            push!(rows, (
                id = candidate.id,
                n = candidate.n,
                t = candidate.t,
                s = s,

                delta = candidate.delta,
                psi0 = candidate.psi0,
                sigma = candidate.sigma,
                epsilon = candidate.epsilon,
                m = candidate.m,
                jit_top = candidate.jit_top,
                jit_bot = candidate.jit_bot,
                generation_seed = candidate.generation_seed,
                source_file = candidate.source_file,

                sampling_seed = sampling_seed + candidate.id,
                nsamples = nsamples,

                frac_holds = r.frac_holds,
                frac_holds_shallow = r.frac_holds_shallow,
                frac_holds_deep = r.frac_holds_deep,

                root_holds = r.root_holds,
                root_rho = r.root_ρ,

                rho_q10 = r.ρ_q10,
                rho_median = r.ρ_med,
                rho_q90 = r.ρ_q90,

                ddgfact_plus_bound = z_ddgfact_plus,
                glinx_bound = z_glinx,
                local_search_lb = z_ls,

                ddgfact_plus_gap = ddgfact_plus_gap,
                glinx_gap = glinx_gap,
                gap_difference = gap_difference,

                gamma_glinx = gamma_opt
            ))
        end

        @printf("Finished instance %3d / %3d\n", j, length(candidates))
    end

    return DataFrame(rows)
end


function save_statistics_table(df::DataFrame; output_path::AbstractString = "data/generated/instance_statistics.csv")

    mkpath(dirname(output_path))
    CSV.write(output_path, df)

    println()
    println("============================================================")
    println("STATISTICS SAVED")
    println("============================================================")
    println("Instances: $(length(unique(df.id)))")
    println("Rows: $(nrow(df))")
    println("File: $output_path")
end


function reconstruct_instance(row)

    n = Int(row.n)
    t = Int(row.t)

    file = matopen(String(row.source_file))
    C0 = n == 63 ? read(file, "A") : read(file, "C")
    close(file)

    C0 = Symmetric((C0 + C0') / 2)
    base_eigs, Q = eig_desc(C0)

    rng = MersenneTwister(Int(row.generation_seed))

    C, λ, _ = make_instance(; n, t, δ = Int(row.delta), ψ0 = row.psi0, σ = row.sigma, ε = row.epsilon, m = Int(row.m), jit_top = row.jit_top, jit_bot = row.jit_bot, Q, base_eigs, rng)

    return C, λ
end


n = 63
t = 2

s_list = [10, 15, 20, 25, 30]

ψ0 = 1.0
jit_top = 0.05
jit_bot = 1e-3

generation_seed = 1
sampling_seed = 10000

nsamples = 1000
glinx_tol = 1e-2
atol = 1e-10

blas_threads = 16
MKL.set_num_threads(blas_threads)

input_filepath = "data/data$(n).mat"
output_dir = "data/generated"
statistics_path = joinpath(output_dir, "instance_statistics.csv")

file = matopen(input_filepath)
C0 = n == 63 ? read(file, "A") : read(file, "C")
close(file)

C0 = Symmetric((C0 + C0') / 2)
base, Q = eig_desc(C0)

candidates = generate_candidates(; n, t, δs = 0:3, σ_over_ψ0 = [5.0, 20.0, 100.0], εs = [0.1, 0.5, 2.0], ms = [20, 30, 40], ψ0, jit_top, jit_bot, Q, base_eigs = base, source_file = input_filepath, seed = generation_seed)

println()
println("Warming up solvers...")

n_warmup = 9
s_warmup = 7
t_warmup = 5

C_warmup = Symmetric(Matrix(C0[1:n_warmup, 1:n_warmup]))
psi_warmup = eigmin(C_warmup) - atol

aug_ddfact_gmesp(C_warmup, s_warmup, t_warmup, psi_warmup; atol = atol)

gamma_warmup = compute_gamma_linx_armijo(C_warmup, s_warmup, t_warmup; tol = glinx_tol)
linx_gmesp(C_warmup, s_warmup, t_warmup; gamma = gamma_warmup, return_model = true)

GC.gc()

println("Solver warm-up complete.")

statistics = evaluate_candidates(candidates; s_list, nsamples, glinx_tol, atol, sampling_seed)

save_statistics_table(statistics; output_path = statistics_path)