using LinearAlgebra, Printf

# =============================================================================
# Branch-and-bound for GMESP
#
# Bound at a node = the DDGFact relaxation (`ddfact_gmesp_fix`) together with its
# closed-form TRUE dual (`compute_ddfact_dual_gap`).  The dual gives
#   • a certified upper bound  (dual_obj ≥ relaxation opt ≥ node optimum), and
#   • dual solutions (nu, ups) used for variable fixing
#
# Variable handling:
#   • variables fixed to 0 are dropped → we REDUCE C to C[keep, keep];
#   • variables fixed to 1 stay in the problem with x = 1 (no Schur complement).
#
# Branching: most-fractional free variable.  Open list: best-first by upper bound.
# Stop when the best UB can no longer beat the incumbent, or the time limit fires —
# so the incumbent returned is the true optimum (no optimality tolerance).
# =============================================================================

struct GMESPNode
    F1::Vector{Int}            # variables fixed to 1 (kept in the problem)
    F0::Vector{Int}            # variables fixed to 0 (matrix reduced)
    ub::Float64                # certified upper bound on this subtree
    x::Vector{Float64}         # relaxation solution over `keep`
    keep::Vector{Int}          # original indices still present (= 1:n \ F0)
end


# Exact GMESP objective at an integer subset S (|S| = s ≥ t).
function _gmesp_obj(C::AbstractMatrix, S::Vector{Int}, t::Int)
    λ = reverse(eigvals(Symmetric(Matrix(C[S, S]))))
    return sum(log, max.(λ[1:t], 1e-30))
end


# Node relaxation.  Returns the certified upper bound `ub`, the relaxation point
# `x` over `keep`, the dual solutions (nu, ups) over `keep`, `keep` itself, and a
# `determined` flag marking a single-point leaf.
#   keep = 1:n \ F0   (fix-to-0 reduces the matrix to C[keep, keep])
#   the indices in F1 are forced to x = 1 inside the relaxation.
function _gmesp_node(C::Symmetric, s::Int, t::Int, F1::Vector{Int}, F0::Vector{Int})
    n = size(C, 1)
    keep = setdiff(1:n, F0)
    n_red = length(keep)
    # Infeasible subproblems (s, t are invariant with t ≤ s; only F1/F0 vary).
    (length(F1) > s || s > n_red) &&
        return (ub = -Inf, x = Float64[], nu = Float64[], ups = Float64[],
                keep = keep, determined = false)

    # Determined subset → a single forced integer point (a leaf): no solver.  S = F1
    # (all of S fixed to 1) or S = keep (every kept variable must be 1).  `ub` is
    # the DDFact relaxation value — a genuine UPPER bound (≥ Γ_t(C[S,S])); the true
    # objective Γ_t(C[S,S]) is a feasible value and is scored into the incumbent (a
    # LOWER bound) by `_register!`.  `determined = true` tells `_child` it's a leaf.
    if length(F1) == s || n_red == s
        S = length(F1) == s ? sort(F1) : copy(keep)
        Sset = Set(S)
        x = [keep[k] in Sset ? 1.0 : 0.0 for k in eachindex(keep)]
        return (ub = _ddfact_value_at(C, S, t), x = x, nu = zeros(n_red), ups = zeros(n_red),
                keep = keep, determined = true)
    end

    Ck   = Symmetric(Matrix(C[keep, keep]))
    posn = Dict(v => k for (k, v) in enumerate(keep))   # original → keep-local
    fix1 = sort([posn[i] for i in F1])

    x, _ = ddfact_gmesp_fix(Ck, s, t, fix1)

    xL = zeros(n_red); xL[fix1] .= 1.0                   # forced vars: lower = 1
    xU = ones(n_red)
    r = compute_ddfact_dual_gap(Ck, x, s, t; xL = xL, xU = xU)
    return (ub = r.dual_obj, x = x, nu = r.nu, ups = r.ups,
            keep = keep, determined = false)
end


# Subset implied by a relaxation point: the s coordinates with the largest x
# (exactly the support when x is binary).  Returns (true z_gmesp(C[S,S]), S).
function _subset_value(C::Symmetric, s::Int, t::Int,
                       keep::Vector{Int}, x::Vector{Float64})
    (isempty(x) || length(keep) < s) && return (-Inf, Int[])
    order = sortperm(x, rev = true)
    S = sort(keep[order[1:s]])
    v = try _gmesp_obj(C, S, t) catch; -Inf end
    return (v, S)
end

# True when every coordinate of x is within 1e-6 of 0 or 1, i.e. x is an integer
# (0/1) point whose support is a genuine subset rather than a fractional point.
function _is_integer_point(x::AbstractVector)
    isempty(x) && return false
    max_dist_to_integer = maximum(abs.(x .- round.(x)))
    return max_dist_to_integer ≤ 1e-6
end

# DDFact relaxation objective at the integer point 1_S — the GMESP objective value 
# the B&B sees at an integer node.  
# Used to report the integrality gap  Γ_t(C[S,S]) − z_gmesp(C[S,S]).
function _ddfact_value_at(C::AbstractMatrix, S::Vector{Int}, t::Int)
    λ = reverse(eigvals(Symmetric(Matrix(C[S, S]))))
    iota, mid = find_iota(λ, t)
    fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
    return fval + (t - iota) * log(mid)
end


# Branching variable (original index), or 0 if there is no free variable left.
# Prefer the most-fractional free variable; if every free variable is already
# integer in the relaxation, fall back to ANY free variable.  This fallback is
# essential: the DDFact bound is NOT tight at integer points for GMESP (t < s),
# so an integral relaxation does not resolve a node — we must keep partitioning
# until the subset is fully determined.
function _branch_var(keep::Vector{Int}, x::Vector{Float64}, F1::Vector{Int})
    F1set = Set(F1)
    best_k, best_f = 0, -1.0
    for k in eachindex(keep)
        keep[k] in F1set && continue
        f = min(x[k], 1 - x[k])
        if f > best_f; best_f = f; best_k = k; end
    end
    return best_k == 0 ? 0 : keep[best_k]
end


# Variable fixing from the dual variables.  margin = UB − LB.  For a free keep[k]:
#   nu[k]  > margin ⟹ x = 1 (add to F1);   ups[k] > margin ⟹ x = 0 (add to F0).
function variable_fixing_dual_soln(F1::Vector{Int}, F0::Vector{Int}, keep::Vector{Int},
                                   nu::Vector{Float64}, ups::Vector{Float64},
                                   ub::Float64, lb::Float64; tol::Real = 1e-9)
    margin = ub - lb
    F1n, F0n = copy(F1), copy(F0)
    F1set = Set(F1)
    for k in eachindex(keep)
        i = keep[k]
        i in F1set && continue
        if nu[k] > margin + tol
            push!(F1n, i)
        elseif ups[k] > margin + tol
            push!(F0n, i)
        end
    end
    return sort(F1n), sort(F0n)
end


"""
    solve_bnb_ddfact(C, s, t; time_limit=3600.0, verbose=true)
        -> (S_best, stats)

Branch-and-bound for `max_{|S|=s} Γ_t(C[S,S])` using the DDFact relaxation and
its closed-form dual for bounding and variable fixing from the dual variables.

`stats` is a NamedTuple: `nodes, lb, ub, gap, root_ub, nfix0, nfix1,
n_int_sols, int_gap_max, int_gap_avg, int_gap_opt, wall_time, time_limit_hit,
tree_exhausted`.  `tree_exhausted = true` means the incumbent is proved optimal
(open list emptied); otherwise `ub` is the best remaining DDFact bound.

Integrality-gap metric (DDFact is not tight at integer points when t < s):
for every integer relaxation solution found (‖x − round(x)‖∞ ≤ 1e-6) we evaluate
the true objective Γ_t(C[support]) and record `ub(integer point) − true`.
`n_int_sols` counts them, `int_gap_max` / `int_gap_avg` are the largest and the
average such gap, and `int_gap_opt` is the gap at the returned incumbent.
"""
function solve_bnb_ddfact(C::Symmetric, s::Int, t::Int;
                          time_limit::Real = 3600.0, verbose::Bool = true)
    n = size(C, 1)
    @assert 1 <= t <= s < n "need 1 ≤ t ≤ s < n"

    # Incumbent (lower bound) from local search.
    x_ls, lb = run_all_LS(C, s, t)
    S_inc = sort(findall(x_ls .> 0.5))

    # Integrality-gap metric over the integer relaxation solutions found.
    n_int_sols  = 0
    int_gap_max = 0.0
    int_gap_sum = 0.0       # running sum of the gaps, for the average

    # ONLY when a node's relaxation point is integer: score its support with the
    # TRUE GMESP objective, improve the incumbent, and record the integrality gap.
    function _register!(r)
        _is_integer_point(r.x) || return        # skip infeasible / fractional points
        true_obj, S = _subset_value(C, s, t, r.keep, r.x)
        isfinite(true_obj) || return
        int_gap = r.ub - true_obj               # integrality gap: node bound − true objective
        n_int_sols  += 1
        int_gap_sum += int_gap
        int_gap_max  = max(int_gap_max, int_gap)
        if true_obj > lb
            lb = true_obj
            S_inc = S
        end
    end

    # Root node.
    r = _gmesp_node(C, s, t, Int[], Int[])
    ub_root = r.ub
    _register!(r)
    verbose && @printf("root:  lb = %.4f   ub = %.4f   gap = %.4f\n",
                       lb, ub_root, ub_root - lb)

    open = [GMESPNode(Int[], Int[], ub_root, r.x, r.keep)]
    nodes, nfix0, nfix1 = 1, 0, 0
    t0 = time()
    time_limit_hit = false

    # Bound a child, fix variables, prune; push to `open` if it survives.
    function _child(F1::Vector{Int}, F0::Vector{Int})
        r = _gmesp_node(C, s, t, F1, F0)
        nodes += 1
        _register!(r)
        r.determined && return                              # leaf: single point, scored
        r.ub ≤ lb && return                                 # prune: can't beat incumbent
        F1f, F0f = variable_fixing_dual_soln(F1, F0, r.keep, r.nu, r.ups, r.ub, lb)
        if length(F1f) > length(F1) || length(F0f) > length(F0)
            nfix1 += length(F1f) - length(F1)
            nfix0 += length(F0f) - length(F0)
            r = _gmesp_node(C, s, t, F1f, F0f)             # re-bound after fixing
            _register!(r)
            r.determined && return                          # fixing determined it → leaf
            r.ub ≤ lb && return
            F1, F0 = F1f, F0f
        end
        push!(open, GMESPNode(F1, F0, r.ub, r.x, r.keep))
    end

    while !isempty(open)
        if time() - t0 ≥ time_limit
            time_limit_hit = true
            break
        end
        # Best-first: sort so the largest-UB node is first.
        sort!(open; by = nd -> nd.ub, rev = true)

        # If the largest remaining UB can't beat the incumbent, nothing can imply optimal.
        first(open).ub ≤ lb && break

        node = popfirst!(open)

        i = _branch_var(node.keep, node.x, node.F1)
        i == 0 && continue                                  # no free variable left

        _child(sort(vcat(node.F1, i)), node.F0)             # x_i = 1
        _child(node.F1, sort(vcat(node.F0, i)))             # x_i = 0

        if verbose && nodes % 1000 == 0
            best_ub = isempty(open) ? lb : maximum(nd -> nd.ub, open)
            @printf("nodes = %4d  open = %4d  fix0 = %3d  fix1 = %3d  lb = %.4f  ub = %.4f  gap = %.4f\n",
                    nodes, length(open), nfix0, nfix1, lb, best_ub, best_ub - lb)
        end
    end

    # When the open list empties, every subtree was either pruned (UB ≤ LB) or
    # branched down to a fully-determined subset, so the incumbent is optimal by exhaustion
    # When stopped early, the best open UB is a valid (loose) bound.
    tree_exhausted = isempty(open)
    final_ub = tree_exhausted ? lb : maximum(nd -> nd.ub, open)
    status = time_limit_hit ? "TIME LIMIT" : (tree_exhausted ? "OPTIMAL (exhausted)" : "OPTIMAL (gap)")

    # Integrality gap at the returned incumbent: DDFact value at 1_{S_inc} − Γ_t(C[S_inc]).
    int_gap_opt = isempty(S_inc) ? NaN :
                  _ddfact_value_at(C, S_inc, t) - _gmesp_obj(C, S_inc, t)
    int_gap_avg = n_int_sols > 0 ? int_gap_sum / n_int_sols : NaN

    verbose && @printf("done [%s].  nodes = %d   fix0 = %d  fix1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
                       status, nodes, nfix0, nfix1, lb, final_ub, final_ub - lb)
    verbose && @printf("           integer solns = %d   max int-gap = %.4f   avg int-gap = %.4f   int-gap@opt = %.4f\n",
                       n_int_sols, int_gap_max, int_gap_avg, int_gap_opt)

    stats = (nodes = nodes, lb = lb, ub = final_ub, gap = final_ub - lb,
             root_ub = ub_root, nfix0 = nfix0, nfix1 = nfix1,
             n_int_sols = n_int_sols, int_gap_max = int_gap_max, int_gap_avg = int_gap_avg,
             int_gap_opt = int_gap_opt,
             wall_time = time() - t0, time_limit_hit = time_limit_hit,
             tree_exhausted = tree_exhausted)
    return S_inc, stats
end
