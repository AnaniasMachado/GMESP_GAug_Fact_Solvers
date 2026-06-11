using LinearAlgebra, Printf

# =============================================================================
# Specialized branch-and-bound for GMESP with t = 1 using DDGFact^+
#
# Bound at a node = the closed-form DDGFact^+ relaxation for t = 1.
#
# Since t = 1, DDGFact^+ has the closed-form relaxation:
#
#   max log(sum_i x_i ||F_i||^2 + psi)
#
# over the fixed-variable cardinality set.
#
# Therefore each node bound is obtained by:
#   • keeping all variables fixed to 1;
#   • excluding all variables fixed to 0;
#   • filling the remaining cardinality with the largest row norms of F.
#
# Variable handling:
#   • variables fixed to 0 are recorded in F0;
#   • variables fixed to 1 are recorded in F1;
#   • the matrix is NOT reduced, because the t = 1 bound depends only on
#     the row norms ||F_i||^2 = C_ii - psi.
#
# Branching:
#   • first try to branch on a variable selected by the relaxation but not
#     already fixed to 1;
#   • among those, choose the selected variable with smallest row norm;
#   • if every selected variable is already fixed, branch on the best
#     nonselected free variable.
#
# Stop when the best UB can no longer beat the incumbent, or the time limit fires.
# =============================================================================

struct GMESPNodeT1DDGFactplus
    F1::Vector{Int}            # variables fixed to 1
    F0::Vector{Int}            # variables fixed to 0
    ub::Float64                # DDGFact^+ t = 1 upper bound on this subtree
    S_relax::Vector{Int}       # subset selected by the closed-form relaxation
end


# Exact GMESP objective at an integer subset S for t = 1.
function _gmesp_obj_t1(C::AbstractMatrix, S::Vector{Int})
    λmax = maximum(eigvals(Symmetric(Matrix(C[S, S]))))
    return log(max(λmax, 1e-30))
end


# Highest feasible positive psi for DDGFact^+.
function _default_psi_t1_ddgfactplus(
    C::Symmetric{<:Real,<:AbstractMatrix};
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    λmin = minimum(eigvals(Symmetric(Matrix(C))))
    psi = max(psi_floor, λmin - psi_margin)

    if psi <= 0.0
        error(
            "DDGFactplus requires psi > 0. Pass psi explicitly or use a " *
            "positive definite matrix with λmin(C) > psi_margin.",
        )
    end

    return psi
end


# Closed-form DDGFact^+ t = 1 node bound.
#
# F1 and F0 are original indices.
# The selected set is:
#   S_relax = F1 ∪ {largest remaining row norms among free variables}.
function _ddgfactplus_t1_node_bound(
    row_norms::Vector{Float64},
    psi::Float64,
    s::Int,
    F1::Vector{Int},
    F0::Vector{Int};
    atol::Float64 = 1e-8,
)
    n = length(row_norms)

    if !isempty(intersect(F1, F0))
        return (
            feasible = false,
            determined = false,
            ub = -Inf,
            S_relax = Int[],
            x = zeros(Float64, n),
        )
    end

    if length(F1) > s
        return (
            feasible = false,
            determined = false,
            ub = -Inf,
            S_relax = Int[],
            x = zeros(Float64, n),
        )
    end

    F1set = Set(F1)
    F0set = Set(F0)
    free = [i for i in 1:n if !(i in F1set) && !(i in F0set)]
    m = s - length(F1)

    if m < 0 || m > length(free)
        return (
            feasible = false,
            determined = false,
            ub = -Inf,
            S_relax = Int[],
            x = zeros(Float64, n),
        )
    end

    S_free =
        m == 0 ? Int[] :
        free[partialsortperm(row_norms[free], 1:m; rev = true)]

    S_relax = sort(union(F1, S_free))

    x = zeros(Float64, n)
    x[S_relax] .= 1.0

    denom = dot(row_norms, x) + psi

    if denom <= atol
        return (
            feasible = false,
            determined = false,
            ub = -Inf,
            S_relax = Int[],
            x = zeros(Float64, n),
        )
    end

    determined = length(F1) == s || n - length(F0) == s

    return (
        feasible = true,
        determined = determined,
        ub = log(denom),
        S_relax = S_relax,
        x = x,
    )
end


# Branching variable for DDGFact^+ with t = 1.
#
# Prefer a selected free variable, because branching on x_j = 0 removes the
# current relaxation-selected subset from that child.
#
# Among selected free variables, choose the one with smallest row norm.  This
# usually creates a strong x_j = 0 child while keeping the x_j = 1 child natural.
#
# If no selected variable is free, branch on the best nonselected free variable.
function _branch_var_t1_ddgfactplus(
    row_norms::Vector{Float64},
    S_relax::Vector{Int},
    F1::Vector{Int},
    F0::Vector{Int},
)
    n = length(row_norms)

    F1set = Set(F1)
    F0set = Set(F0)
    Sset = Set(S_relax)

    selected_free = [
        j for j in S_relax
        if !(j in F1set) && !(j in F0set)
    ]

    if !isempty(selected_free)
        k = argmin(row_norms[selected_free])
        return selected_free[k]
    end

    nonselected_free = [
        j for j in 1:n
        if !(j in F1set) && !(j in F0set) && !(j in Sset)
    ]

    if !isempty(nonselected_free)
        k = argmax(row_norms[nonselected_free])
        return nonselected_free[k]
    end

    return 0
end


"""
    solve_bnb_ddgfactplus_t1(C, s; psi=nothing, time_limit=3600.0, verbose=true)
        -> (S_best, stats)

Specialized branch-and-bound for GMESP with `t = 1` using the closed-form
DDGFact^+ relaxation.

This routine does not call KNITRO.  At each node, the DDGFact^+ t = 1
relaxation is solved by selecting the largest remaining row norms of `F`, while
respecting variables fixed to one and zero.

`stats` is a NamedTuple with:
`nodes, lb, ub, gap, root_ub, nfix0, nfix1, wall_time, time_limit_hit,
tree_exhausted, psi`.
"""
function solve_bnb_ddgfactplus_t1(
    C::Symmetric,
    s::Int;
    psi::Union{Nothing,Float64} = nothing,
    time_limit::Real = 3600.0,
    verbose::Bool = true,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)
    @assert 1 <= s < n "need 1 ≤ s < n"

    psi_node =
        psi === nothing ?
        _default_psi_t1_ddgfactplus(
            C;
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        ) :
        psi

    # For t = 1:
    #   ||F_i||^2 = C_ii - psi
    # where F F' = C - psi I.
    row_norms = collect(diag(C)) .- psi_node

    if minimum(row_norms) < -atol
        error("Invalid psi: some row norms C_ii - psi are negative.")
    end

    row_norms[abs.(row_norms) .<= atol] .= 0.0

    # Incumbent (lower bound) from local search.
    x_ls, lb = run_all_LS(C, s, 1)
    S_inc = sort(findall(x_ls .> 0.5))

    # Register a feasible subset by evaluating the TRUE GMESP objective.
    # This updates the lower bound only.
    function _register_subset!(S::Vector{Int})
        length(S) == s || return

        true_obj = try
            _gmesp_obj_t1(C, S)
        catch
            -Inf
        end

        if true_obj > lb
            lb = true_obj
            S_inc = sort(S)
        end

        return nothing
    end

    # Root node.
    root = _ddgfactplus_t1_node_bound(
        row_norms,
        psi_node,
        s,
        Int[],
        Int[];
        atol = atol,
    )

    root.feasible || error("Root node is infeasible.")

    ub_root = root.ub
    _register_subset!(root.S_relax)

    verbose && @printf(
        "root:  DDGFactplus t=1   lb = %.4f   ub = %.4f   gap = %.4f\n",
        lb,
        ub_root,
        ub_root - lb,
    )

    # Faithful to the original DDGFact code:
    # put the root in the open list and let the main loop decide whether
    # the largest remaining UB can still beat the incumbent.
    open = [GMESPNodeT1DDGFactplus(Int[], Int[], ub_root, root.S_relax)]

    nodes = 1
    nfix0 = 0
    nfix1 = 0

    report_every = 1000
    next_report = report_every

    t0 = time()
    time_limit_hit = false

    # Bound a child, score its relaxation-selected subset, prune;
    # push to `open` if it survives.
    function _child(F1::Vector{Int}, F0::Vector{Int})
        r = _ddgfactplus_t1_node_bound(
            row_norms,
            psi_node,
            s,
            F1,
            F0;
            atol = atol,
        )

        nodes += 1

        r.feasible || return

        # The relaxation-selected set is always a feasible GMESP subset.
        # Its TRUE objective updates the incumbent, while r.ub remains the
        # DDGFact^+ relaxation upper bound for this node.
        _register_subset!(r.S_relax)

        r.determined && return                              # leaf: single point, scored
        r.ub <= lb && return                                # prune: can't beat incumbent

        push!(
            open,
            GMESPNodeT1DDGFactplus(
                sort(F1),
                sort(F0),
                r.ub,
                r.S_relax,
            ),
        )

        return nothing
    end

    while !isempty(open)
        if time() - t0 >= time_limit
            time_limit_hit = true
            break
        end

        # Best-first: sort so the largest-UB node is first.
        sort!(open; by = nd -> nd.ub, rev = true)

        # If the largest remaining UB can't beat the incumbent, nothing can imply optimal.
        first(open).ub <= lb && break

        node = popfirst!(open)

        j = _branch_var_t1_ddgfactplus(
            row_norms,
            node.S_relax,
            node.F1,
            node.F0,
        )

        j == 0 && continue

        # Branch x_j = 1.
        F1_child = sort(vcat(node.F1, j))
        F0_child = node.F0

        if length(F1_child) <= s
            nfix1 += 1
            _child(F1_child, F0_child)
        end

        # Branch x_j = 0.
        F1_child = node.F1
        F0_child = sort(vcat(node.F0, j))

        if n - length(F0_child) >= s
            nfix0 += 1
            _child(F1_child, F0_child)
        end

        if verbose && nodes >= next_report
            best_ub = isempty(open) ? lb : maximum(nd -> nd.ub, open)

            @printf(
                "nodes = %4d  open = %4d  branch0 = %3d  branch1 = %3d  lb = %.4f  ub = %.4f  gap = %.4f\n",
                nodes,
                length(open),
                nfix0,
                nfix1,
                lb,
                best_ub,
                best_ub - lb,
            )
            flush(stdout)

            next_report += 1000
        end
    end

    # Faithful to the original DDGFact code:
    # when the open list empties, every subtree was either pruned or branched down
    # to a fully determined subset, so the incumbent is optimal by exhaustion.
    # When stopped early, the best open UB is the valid remaining upper bound.
    tree_exhausted = isempty(open)
    final_ub = tree_exhausted ? lb : maximum(nd -> nd.ub, open)
    status =
        time_limit_hit ? "TIME LIMIT" :
        tree_exhausted ? "OPTIMAL (exhausted)" :
        "OPTIMAL (gap)"

    verbose && @printf(
        "done [%s].  nodes = %d   branch0 = %d  branch1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
        status,
        nodes,
        nfix0,
        nfix1,
        lb,
        final_ub,
        final_ub - lb,
    )

    stats = (
        nodes = nodes,
        lb = lb,
        ub = final_ub,
        gap = final_ub - lb,
        root_ub = ub_root,
        nfix0 = nfix0,
        nfix1 = nfix1,
        wall_time = time() - t0,
        time_limit_hit = time_limit_hit,
        tree_exhausted = tree_exhausted,
        psi = psi_node,
        relaxation = :DDGFactplus,
        t = 1,
    )

    return S_inc, stats
end