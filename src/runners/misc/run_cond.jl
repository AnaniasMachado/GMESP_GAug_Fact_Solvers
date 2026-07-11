using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Printf

include("../misc/util.jl")
include("../misc/heuristics.jl")

include("../misc/dual.jl")
include("../misc/var_fixing.jl")

include("../bnb/bnb_t1_plus.jl")


# ============================================================
# Choose instance
# ============================================================
data_n = 63
k = 32

t = 1

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
C = Symmetric(C)

n = size(C, 1)


# ============================================================
# Prefix-deviation DDGFact+ certificate for t = 1 GMESP
#
# Checks the theorem:
#
# Let P_s = {π_1, ..., π_s}.
# For each r = 1,...,s, define
#
#   A_r = {π_1, ..., π_{r-1}},
#   banned = {π_r}.
#
# The branch is:
#
#   A_r ⊆ S, π_r ∉ S, |S| = s.
#
# Every competitor S != P_s belongs to exactly one such branch,
# where r is the first missing element of P_s.
#
# For t = 1, DDGFact+ gives the closed-form branch upper bound:
#
#   UB_r(ψ) =
#       ψ
#       + sum_{i in A_r} (C_ii - ψ)
#       + sum_{i in T_r} (C_ii - ψ),
#
# where T_r contains the largest s-r+1 values of C_ii - ψ
# among indices not in A_r and not equal to π_r.
#
# If UB_r(ψ) <= λmax(C[P_s, P_s]) for all r,
# then P_s is globally optimal for GMESP at t = 1.
#
# Single CSV output:
#   data63_k32_t1_prefix_deviation_ddgfact_certificate.csv
#
# Paste after loading C.
# ============================================================

using LinearAlgebra
using DataFrames
using CSV
using Printf


# ============================================================
# Parameters
# ============================================================

output_csv = "data63_k32_t1_prefix_deviation_ddgfact_certificate.csv"

tol = 1e-10

# By default, use the strongest feasible psi for C - psi I >= 0.
# You can replace this with a value from your solver output if desired.
psi_mode = :lambda_min

# Optional: also test looser psi values.
# For example: extra_psi_values = [0.0]
extra_psi_values = Float64[]


# ============================================================
# Known proved-optimal supports
# ============================================================

known_opt = Dict{Int, Vector{Int}}(
    2  => [18, 31],
    3  => [15, 18, 31],
    4  => [15, 18, 24, 31],
    5  => [15, 18, 24, 26, 31],
    6  => [7, 15, 18, 24, 26, 31],
    7  => [7, 15, 18, 20, 24, 26, 31],
    8  => [1, 7, 15, 18, 20, 24, 26, 31],
    9  => [1, 7, 15, 18, 20, 22, 24, 26, 31],

    24 => [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15,
           16, 18, 20, 21, 22, 23, 24, 26, 29, 30, 31, 32],

    25 => [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15,
           16, 18, 20, 21, 22, 23, 24, 26, 29, 30, 31, 32],

    26 => [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15,
           16, 17, 18, 20, 21, 22, 23, 24, 26, 29, 30, 31, 32],

    27 => [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 29, 30, 31, 32],

    28 => [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32],

    29 => [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32],

    30 => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32],

    31 => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32],
)


# ============================================================
# Utilities
# ============================================================

set_string(S::Vector{Int}) = join(sort(S), " ")
order_string(π::Vector{Int}) = join(π, " ")

function symdiff_size(A::Vector{Int}, B::Vector{Int})
    SA = Set(A)
    SB = Set(B)
    return length(union(setdiff(SA, SB), setdiff(SB, SA)))
end

function lmax(A::AbstractMatrix)
    if size(A, 1) == 0
        return 0.0
    end
    return maximum(eigvals(Symmetric(Matrix(A))))
end

function support_lmax(C::AbstractMatrix, S::Vector{Int})
    S = sort(S)
    return lmax(Matrix(C)[S, S])
end

function prefix_support(π::Vector{Int}, s::Int)
    return sort(π[1:s])
end

function build_known_nested_order(n::Int)
    strong_prefix = [18, 31, 15, 24, 26, 7, 20, 1, 22]

    # From large-s complements:
    # weakest to stronger: 25, 28, 9, 19, 27, 14, 17, 3
    # therefore stronger to weakest:
    tail_strong_to_weak = [3, 17, 14, 27, 19, 9, 28, 25]

    used = Set(vcat(strong_prefix, tail_strong_to_weak))
    middle = [i for i in 1:n if !(i in used)]

    return vcat(strong_prefix, middle, tail_strong_to_weak)
end

function choose_psi_values(C::AbstractMatrix; psi_mode::Symbol, extra_psi_values::Vector{Float64})
    λmin = minimum(eigvals(Symmetric(Matrix(C))))

    values = Float64[]

    if psi_mode == :lambda_min
        push!(values, λmin)
    elseif psi_mode == :zero
        push!(values, 0.0)
    elseif psi_mode == :both
        push!(values, λmin)
        push!(values, 0.0)
    else
        error("Unknown psi_mode: $psi_mode")
    end

    append!(values, extra_psi_values)

    # Keep only feasible values ψ <= λmin(C), up to a tiny tolerance.
    feasible = Float64[]
    for ψ in values
        if ψ <= λmin + 1e-9
            push!(feasible, ψ)
        else
            @warn "Skipping infeasible psi = $ψ because psi > lambda_min(C) = $λmin"
        end
    end

    return unique(feasible), λmin
end

function top_k_by_score(
    candidates::Vector{Int},
    scores::Vector{Float64},
    k::Int,
)
    if k <= 0
        return Int[]
    end

    if k > length(candidates)
        error("Asked for k = $k elements, but only $(length(candidates)) candidates are available.")
    end

    order = sort(candidates, by = i -> (-scores[i], i))
    return order[1:k]
end


# ============================================================
# Branch bound
# ============================================================

function prefix_deviation_branch_bound_t1(
    Cdiag::Vector{Float64},
    π::Vector{Int},
    s::Int,
    r::Int,
    ψ::Float64,
)
    n = length(Cdiag)

    A = r == 1 ? Int[] : π[1:(r - 1)]
    banned = π[r]

    fixed_set = Set(A)
    allowed = [
        i for i in 1:n
        if !(i in fixed_set) && i != banned
    ]

    remaining_k = s - length(A)

    d = Cdiag .- ψ

    T = top_k_by_score(allowed, d, remaining_k)

    selected_for_bound = vcat(A, T)

    ub_linear =
        ψ +
        sum(d[i] for i in selected_for_bound)

    ub_log = ub_linear > 0 ? log(ub_linear) : -Inf

    return (
        A = collect(A),
        banned = banned,
        T = collect(T),
        selected_for_bound = collect(selected_for_bound),
        ub_linear = ub_linear,
        ub_log = ub_log,
        remaining_k = remaining_k,
    )
end


# ============================================================
# Full certificate check
# ============================================================

function run_prefix_deviation_ddgfact_certificate(
    C::AbstractMatrix,
    known_opt::Dict{Int, Vector{Int}};
    output_csv::String = "data63_k32_t1_prefix_deviation_ddgfact_certificate.csv",
    psi_mode::Symbol = :lambda_min,
    extra_psi_values::Vector{Float64} = Float64[],
    tol::Float64 = 1e-10,
)
    Cmat = Matrix(Symmetric(C))
    n = size(Cmat, 1)

    Cdiag = diag(Cmat)

    π = build_known_nested_order(n)

    psi_values, lambda_min_C = choose_psi_values(
        Cmat;
        psi_mode = psi_mode,
        extra_psi_values = extra_psi_values,
    )

    rows = DataFrame(
        s = Int[],
        r = Int[],
        psi = Float64[],
        lambda_min_C = Float64[],

        prefix_support = String[],
        known_support = String[],
        prefix_matches_known = Bool[],
        prefix_hamming_known = Int[],

        lambda_prefix = Float64[],
        log_lambda_prefix = Float64[],

        fixed_one_A = String[],
        fixed_zero_banned = Int[],
        remaining_k = Int[],
        top_completion_T = String[],
        selected_for_bound = String[],

        ddgfact_branch_ub_linear = Float64[],
        ddgfact_branch_ub_log = Float64[],
        branch_slack_linear = Float64[],
        branch_slack_log = Float64[],
        branch_certificate_holds = Bool[],

        worst_branch_slack_for_s = Float64[],
        worst_branch_r_for_s = Int[],
        certificate_holds_for_s = Bool[],

        ordering = String[],
    )

    # Store temporary branch data first, then fill per-s summaries.
    temp_rows = Vector{NamedTuple}()

    for ψ in psi_values
        for s in sort(collect(keys(known_opt)))
            P = prefix_support(π, s)
            S_known = sort(known_opt[s])

            h = symdiff_size(P, S_known)

            λP = support_lmax(Cmat, P)
            log_λP = λP > 0 ? log(λP) : -Inf

            branch_slacks = Float64[]
            branch_rs = Int[]

            branch_data = []

            for r in 1:s
                bd = prefix_deviation_branch_bound_t1(
                    collect(Cdiag),
                    π,
                    s,
                    r,
                    ψ,
                )

                slack_linear = λP - bd.ub_linear
                slack_log = log_λP - bd.ub_log
                holds = slack_linear >= -tol

                push!(branch_slacks, slack_linear)
                push!(branch_rs, r)

                push!(
                    branch_data,
                    (
                        r = r,
                        bd = bd,
                        slack_linear = slack_linear,
                        slack_log = slack_log,
                        holds = holds,
                    ),
                )
            end

            worst_idx = argmin(branch_slacks)
            worst_slack = branch_slacks[worst_idx]
            worst_r = branch_rs[worst_idx]
            cert_s = all(branch_slacks .>= -tol)

            for item in branch_data
                bd = item.bd

                push!(
                    rows,
                    (
                        s,
                        item.r,
                        ψ,
                        lambda_min_C,

                        set_string(P),
                        set_string(S_known),
                        h == 0,
                        h,

                        λP,
                        log_λP,

                        set_string(bd.A),
                        bd.banned,
                        bd.remaining_k,
                        set_string(bd.T),
                        set_string(bd.selected_for_bound),

                        bd.ub_linear,
                        bd.ub_log,
                        item.slack_linear,
                        item.slack_log,
                        item.holds,

                        worst_slack,
                        worst_r,
                        cert_s,

                        order_string(π),
                    ),
                )
            end
        end
    end

    CSV.write(output_csv, rows)

    println()
    println("Prefix-deviation DDGFact+ certificate written to:")
    println("  ", output_csv)
    println()

    println("Summary by psi and s:")
    summary = combine(
        groupby(rows, [:psi, :s]),
        :certificate_holds_for_s => first => :certificate_holds,
        :worst_branch_slack_for_s => first => :worst_slack,
        :worst_branch_r_for_s => first => :worst_r,
        :prefix_matches_known => first => :prefix_matches_known,
        :prefix_hamming_known => first => :prefix_hamming_known,
        :lambda_prefix => first => :lambda_prefix,
    )

    sort!(summary, [:psi, :s])

    show(summary, allrows = true, allcols = true)
    println()

    println()
    println("Failed worst branches:")
    failed = summary[summary.certificate_holds .== false, :]
    if nrow(failed) == 0
        println("  None. Certificate holds for all tested s.")
    else
        show(failed, allrows = true, allcols = true)
        println()
    end

    return rows, summary
end


# ============================================================
# Run
# ============================================================

prefix_dev_rows, prefix_dev_summary =
    run_prefix_deviation_ddgfact_certificate(
        C,
        known_opt;
        output_csv = output_csv,
        psi_mode = psi_mode,
        extra_psi_values = extra_psi_values,
        tol = tol,
    )