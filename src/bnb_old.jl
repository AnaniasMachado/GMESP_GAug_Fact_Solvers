using LinearAlgebra, Printf

# =============================================================================
# Branch-and-bound for GMESP
#
# Bound at a node = the selected relaxation (`DDGFact`, `DDGFactplus`, or
# `DDGFactplusUpsilon`) together with its associated dual.  The dual gives
#   • a certified upper bound  (dual_obj ≥ relaxation opt ≥ node optimum), and
#   • dual solutions used for dual variable fixing
#
# Variable handling:
#   • variables fixed to 0 are dropped → we REDUCE C to C[keep, keep];
#   • variables fixed to 1 stay in the problem with x = 1 (no Schur complement).
#
# Branching: most-fractional free variable.  Open list: best-first by upper bound.
# Stop when the best UB can no longer beat the incumbent, or the time limit fires —
# so the incumbent returned is the true optimum (no optimality tolerance).
# =============================================================================

const DDGFact = :DDGFact
const DDGFactplus = :DDGFactplus
const DDGFactplusUpsilon = :DDGFactplusUpsilon

struct GMESPNode
    F1::Vector{Int}            # variables fixed to 1 (kept in the problem)
    F0::Vector{Int}            # variables fixed to 0 (matrix reduced)
    ub::Float64                # certified upper bound on this subtree
    x::Vector{Float64}         # relaxation solution over `keep`
    keep::Vector{Int}          # original indices still present (= 1:n \ F0)
    gamma::Union{Nothing,Vector{Float64}} # only for Upsilon calibration
end


# Exact GMESP objective at an integer subset S (|S| = s ≥ t).
function _gmesp_obj(C::AbstractMatrix, S::Vector{Int}, t::Int)
    λ = reverse(eigvals(Symmetric(Matrix(C[S, S]))))
    return sum(log, max.(λ[1:t], 1e-30))
end


# Normalize the relaxation flag.
function _normalize_relaxation(relaxation)
    r = relaxation isa Symbol ? relaxation : Symbol(relaxation)

    if r ∉ (:DDGFact, :DDGFactplus, :DDGFactplusUpsilon)
        error("relaxation must be one of DDGFact, DDGFactplus, or DDGFactplusUpsilon.")
    end

    return r
end


# Highest feasible positive psi for DDGFact^+ at the current reduced node.
function _default_psi(
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


# Convert a BFGS parameter-set dictionary into keyword arguments for
# `calibrate_upsilon_bfgs_ddfactplus`.
function _calibrate_upsilon_bfgs_from_param_set(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set::Dict;
    J1::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    atol::Float64 = 1e-10,
)
    return calibrate_upsilon_bfgs_ddfactplus(
        C,
        s,
        t;
        J1 = J1,
        theta0 = theta0,
        atol = atol,
        max_iter = get(param_set, :max_iter, get(param_set, :max_bfgs_iter, 20)),
        grad_tol = get(param_set, :grad_tol, 1e-6),
        step_tol = get(param_set, :step_tol, 1e-10),
        psi_margin = get(param_set, :psi_margin, 1e-8),
        psi_floor = get(param_set, :psi_floor, 0.0),
        alpha0 = get(param_set, :alpha0, 1.0),
        alpha_min = get(param_set, :alpha_min, 1e-8),
        alpha_decay = get(param_set, :alpha_decay, 0.5),
        armijo_c1 = get(param_set, :armijo_c1, 1e-4),
        curvature_tol = get(param_set, :curvature_tol, 1e-10),
        max_backtracks = get(param_set, :max_backtracks, 20),
        max_theta_norm = get(param_set, :max_theta_norm, 50.0),
        psi_derivative = get(param_set, :psi_derivative, true),
        use_steepest_descent_fallback =
            get(param_set, :use_steepest_descent_fallback, true),
        verbose = get(param_set, :verbose_bfgs, false),
    )
end


# Node relaxation.  Returns the certified upper bound `ub`, the relaxation point
# `x` over `keep`, the auxiliary solution `y` over `keep` when present, the dual
# solution, `keep` itself, and a `determined` flag marking a single-point leaf.
#   keep = 1:n \ F0   (fix-to-0 reduces the matrix to C[keep, keep])
#   the indices in F1 are forced to x = 1 inside the relaxation.
function _gmesp_node(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int};
    relaxation = DDGFact,
    psi::Union{Nothing,Float64} = nothing,
    bfgs_param_set::Symbol = :default,
    bfgs_param_sets::Dict = bfgs_param_sets,
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
    upsilon_fixing::Symbol = :simple,
)
    relaxation = _normalize_relaxation(relaxation)

    n = size(C, 1)
    keep = setdiff(1:n, F0)
    n_red = length(keep)

    if !isempty(intersect(F1, F0))
        return (ub = -Inf, x = Float64[], y = Float64[], keep = keep,
                determined = false, relaxation = relaxation,
                dual_solution = nothing, l = Float64[], c = Float64[],
                gamma = nothing, psi = nothing, F = nothing,
                upsilon_fixing = upsilon_fixing)
    end

    # Infeasible subproblems (s, t are invariant with t ≤ s; only F1/F0 vary).
    (length(F1) > s || s > n_red) &&
        return (ub = -Inf, x = Float64[], y = Float64[], keep = keep,
                determined = false, relaxation = relaxation,
                dual_solution = nothing, l = Float64[], c = Float64[],
                gamma = nothing, psi = nothing, F = nothing,
                upsilon_fixing = upsilon_fixing)

    # Determined subset → a single forced integer point (a leaf): no solver.  S = F1
    # (all of S fixed to 1) or S = keep (every kept variable must be 1).  `ub` is
    # the selected relaxation value at that integer point — a genuine UPPER bound
    # for the singleton subtree.
    # The true objective Γ_t(C[S,S]) is a feasible value and is scored into the
    # incumbent (a LOWER bound) by `_register!`.  `determined = true` tells
    # `_child` it's a leaf.
    if length(F1) == s || n_red == s
        S = length(F1) == s ? sort(F1) : copy(keep)
        Sset = Set(S)
        x = [keep[k] in Sset ? 1.0 : 0.0 for k in eachindex(keep)]

        Ck = Symmetric(Matrix(C[keep, keep]))

        if relaxation == :DDGFact
            ub_node = DDGFact_value_at_x(
                x,
                Ck,
                t;
                atol = atol,
            )

            return (ub = ub_node, x = x, y = Float64[], keep = keep,
                    determined = true, relaxation = relaxation,
                    dual_solution = nothing, l = Float64[], c = Float64[],
                    gamma = nothing, psi = 0.0, F = nothing,
                    upsilon_fixing = upsilon_fixing)
        elseif relaxation == :DDGFactplus
            psi_node =
                psi === nothing ?
                _default_psi(Ck; psi_margin = psi_margin, psi_floor = psi_floor) :
                psi

            ub_node = DDGFactplus_value_at_x(
                x,
                Ck,
                t,
                psi_node;
                atol = atol,
            )

            return (ub = ub_node, x = x, y = Float64[], keep = keep,
                    determined = true, relaxation = relaxation,
                    dual_solution = nothing, l = Float64[], c = Float64[],
                    gamma = nothing, psi = psi_node, F = nothing,
                    upsilon_fixing = upsilon_fixing)
        else
            haskey(bfgs_param_sets, bfgs_param_set) ||
                error("Unknown BFGS parameter set: $bfgs_param_set.")

            posn = Dict(v => k for (k, v) in enumerate(keep))
            fix1 = Int[posn[i] for i in F1]
            sort!(fix1)

            calib = _calibrate_upsilon_bfgs_from_param_set(
                Ck,
                s,
                t,
                bfgs_param_sets[bfgs_param_set];
                J1 = fix1,
                theta0 = theta0,
                atol = atol,
            )

            gamma = calib.gamma
            psi_node = calib.psi

            ub_node, y, F = DDGFactplusUpsilon_value_at_x(
                x,
                Ck,
                gamma,
                t,
                psi_node;
                atol = atol,
            )

            return (ub = ub_node, x = x, y = y, keep = keep,
                    determined = true, relaxation = relaxation,
                    dual_solution = nothing, l = Float64[], c = Float64[],
                    gamma = gamma, psi = psi_node, F = F,
                    primal_obj = ub_node, bfgs = calib,
                    upsilon_fixing = upsilon_fixing)
        end
    end

    Ck   = Symmetric(Matrix(C[keep, keep]))
    posn = Dict(v => k for (k, v) in enumerate(keep))   # original → keep-local
    fix1 = Int[posn[i] for i in F1]
    sort!(fix1)

    l = zeros(n_red)
    c = ones(n_red)
    l[fix1] .= 1.0                                      # forced vars: lower = 1

    if relaxation == :DDGFact
        x, primal_obj = ddfact_gmesp(
            Ck,
            s,
            t;
            J1 = fix1,
            atol = atol,
        )

        F = factorize_matrix(Ck; atol = atol)

        dual_sol = DGFact_dual_solution_from_DDGFact_x(
            x,
            F,
            s,
            t;
            l = l,
            c = c,
            atol = atol,
        )

        return (ub = dual_sol.objective_value, x = x, y = Float64[],
                keep = keep, determined = false, relaxation = relaxation,
                dual_solution = dual_sol, l = l, c = c,
                gamma = nothing, psi = 0.0, F = F,
                primal_obj = primal_obj, upsilon_fixing = upsilon_fixing)
    elseif relaxation == :DDGFactplus
        psi_node =
            psi === nothing ?
            _default_psi(Ck; psi_margin = psi_margin, psi_floor = psi_floor) :
            psi

        x, primal_obj = aug_ddfact_gmesp(
            Ck,
            s,
            t,
            psi_node;
            J1 = fix1,
            atol = atol,
        )

        F = factorize_matrix(Ck; psi = psi_node, atol = atol)

        dual_sol = DGFactplus_dual_solution_from_DDGFactplus_x(
            x,
            F,
            s,
            t,
            psi_node;
            l = l,
            c = c,
            atol = atol,
        )

        return (ub = dual_sol.objective_value, x = x, y = Float64[],
                keep = keep, determined = false, relaxation = relaxation,
                dual_solution = dual_sol, l = l, c = c,
                gamma = nothing, psi = psi_node, F = F,
                primal_obj = primal_obj, upsilon_fixing = upsilon_fixing)
    else
        haskey(bfgs_param_sets, bfgs_param_set) ||
            error("Unknown BFGS parameter set: $bfgs_param_set.")

        calib = _calibrate_upsilon_bfgs_from_param_set(
            Ck,
            s,
            t,
            bfgs_param_sets[bfgs_param_set];
            J1 = fix1,
            theta0 = theta0,
            atol = atol,
        )

        gamma = calib.gamma
        psi_node = calib.psi
        x = copy(calib.x)
        y = copy(calib.y)
        primal_obj = calib.obj

        F = scaled_factorize_matrix(
            Ck,
            gamma,
            psi_node;
            atol = atol,
        )

        dual_sol = DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
            x,
            gamma,
            F,
            s,
            t,
            psi_node;
            yhat = y,
            l = l,
            c = c,
            atol = atol,
            silent = true,
        )

        return (ub = dual_sol.objective_value, x = x, y = y,
                keep = keep, determined = false, relaxation = relaxation,
                dual_solution = dual_sol, l = l, c = c,
                gamma = gamma, psi = psi_node, F = F,
                primal_obj = primal_obj, bfgs = calib,
                upsilon_fixing = upsilon_fixing)
    end
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


# Branching variable (original index), or 0 if there is no free variable left.
# Prefer the most-fractional free variable; if every free variable is already
# integer in the relaxation, fall back to ANY free variable.  This fallback is
# essential: the relaxation bound need not be tight at integer points for GMESP
# (t < s), so an integral relaxation does not resolve a node — we must keep
# partitioning until the subset is fully determined.
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


# Variable fixing from the selected relaxation solution.  The fixing routines
# return keep-local indices; this method also passes the node-cardinality
# parameters needed by strong DDGFact^+_Upsilon fixing.
#
# fixing_rule:
#   :dual   -> use only dual variable fixing
#   :primal -> use only primal variable fixing
#   :both   -> use the union of dual and primal variable fixing
#
# For DDGFact, only :dual is supported.
function variable_fixing_relaxation_soln(
    F1::Vector{Int},
    F0::Vector{Int},
    r,
    s::Int,
    t::Int,
    lb::Float64;
    fixing_rule::Symbol = :dual,
    tol::Real = 1e-9,
)
    r.dual_solution === nothing && return sort(F1), sort(F0)

    if fixing_rule ∉ (:dual, :primal, :both)
        error("fixing_rule must be one of :dual, :primal, or :both.")
    end

    if r.relaxation == :DDGFact && fixing_rule != :dual
        error("DDGFact currently supports only fixing_rule = :dual.")
    end

    fix_zero = Int[]
    fix_one = Int[]

    # ------------------------------------------------------------
    # Dual variable fixing
    # ------------------------------------------------------------
    if fixing_rule in (:dual, :both)
        if r.relaxation == :DDGFact
            fixing_dual = var_fixing_from_DGFact(
                r.dual_solution.upsilon,
                r.dual_solution.nu,
                r.ub,
                lb;
                l = r.l,
                c = r.c,
                atol = Float64(tol),
            )
        elseif r.relaxation == :DDGFactplus
            fixing_dual = var_fixing_from_DGFactplus(
                r.dual_solution.upsilon,
                r.dual_solution.nu,
                r.ub,
                lb;
                l = r.l,
                c = r.c,
                atol = Float64(tol),
            )
        else
            if r.upsilon_fixing == :strong
                fixing_dual = var_fixing_from_DGFactplusUpsilon_strong(
                    r.dual_solution.upsilon,
                    r.dual_solution.nu,
                    r.dual_solution.eta,
                    r.dual_solution.rho,
                    r.ub,
                    lb,
                    s,
                    t;
                    l = r.l,
                    c = r.c,
                    atol = Float64(tol),
                    silent = true,
                )
            else
                fixing_dual = var_fixing_from_DGFactplusUpsilon_simple(
                    r.dual_solution.upsilon,
                    r.dual_solution.nu,
                    r.dual_solution.eta,
                    r.dual_solution.rho,
                    r.ub,
                    lb;
                    l = r.l,
                    c = r.c,
                    atol = Float64(tol),
                )
            end
        end

        append!(fix_zero, fixing_dual.fix_zero)
        append!(fix_one, fixing_dual.fix_one)
    end

    # ------------------------------------------------------------
    # Primal variable fixing
    # ------------------------------------------------------------
    if fixing_rule in (:primal, :both)
        if r.relaxation == :DDGFactplus
            fixing_primal = var_fixing_DDGFactplus_primal(
                r.x,
                r.F,
                s,
                t,
                r.psi,
                lb;
                l = r.l,
                c = r.c,
                atol = Float64(tol),
            )

            append!(fix_zero, fixing_primal.fix_zero)
            append!(fix_one, fixing_primal.fix_one)
        elseif r.relaxation == :DDGFactplusUpsilon
            fixing_primal = var_fixing_DDGFactplusUpsilon_primal(
                r.x,
                r.y,
                r.gamma,
                r.F,
                s,
                t,
                r.psi,
                lb;
                l = r.l,
                c = r.c,
                atol = Float64(tol),
                silent = true,
            )

            append!(fix_zero, fixing_primal.fix_zero)
            append!(fix_one, fixing_primal.fix_one)
        end
    end

    fix_zero = sort(unique(fix_zero))
    fix_one = sort(unique(fix_one))

    both_fixed = intersect(fix_zero, fix_one)

    if !isempty(both_fixed)
        # If the selected rules imply contradictory fixings, pass the contradiction
        # through to the child.  The next node call detects F1 ∩ F0 ≠ ∅ and prunes it.
        F1n = sort(union(F1, r.keep[fix_one]))
        F0n = sort(union(F0, r.keep[fix_zero]))

        return F1n, F0n
    end

    F1n = sort(union(F1, r.keep[fix_one]))
    F0n = sort(union(F0, r.keep[fix_zero]))

    return F1n, F0n
end


"""
    solve_bnb_ddfact(C, s, t; relaxation=DDGFact, fixing_rule=:dual,
                     time_limit=3600.0, verbose=true)
        -> (S_best, stats)

Branch-and-bound for `max_{|S|=s} Γ_t(C[S,S])` using one of the relaxations
`DDGFact`, `DDGFactplus`, or `DDGFactplusUpsilon` and its associated dual for
bounding.  Variable fixing can be selected with `fixing_rule`.

For `relaxation = DDGFactplus`, pass `psi = ...` or let the code choose
`psi = max(psi_floor, λmin(C_node) - psi_margin)` at each node.

For `relaxation = DDGFactplusUpsilon`, the code calibrates Upsilon with
`calibrate_upsilon_bfgs_ddfactplus` using the parameter set selected by
`bfgs_param_set`.

Variable fixing:
  - `fixing_rule = :dual` uses only dual variable fixing.
  - `fixing_rule = :primal` uses only primal variable fixing.
  - `fixing_rule = :both` uses the union of dual and primal variable fixing.
  - For `DDGFact`, only `fixing_rule = :dual` is supported.

For `DDGFactplusUpsilon`, `upsilon_fixing = :simple` or `:strong` chooses the
dual Upsilon fixing rule when `fixing_rule` includes dual fixing.

`stats` is a NamedTuple: `nodes, lb, ub, gap, root_ub, nfix0, nfix1,
n_int_sols, int_gap_max, int_gap_avg, int_gap_opt, wall_time, time_limit_hit,
tree_exhausted, relaxation, fixing_rule`.  `tree_exhausted = true` means the
incumbent is proved optimal (open list emptied); otherwise `ub` is the best
remaining bound.

Integrality-gap metric:
for every integer relaxation solution found (‖x − round(x)‖∞ ≤ 1e-6) we evaluate
the true objective Γ_t(C[support]) and record `ub(integer point) − true`.
`n_int_sols` counts them, `int_gap_max` / `int_gap_avg` are the largest and the
average such gap, and `int_gap_opt` is the gap at the returned incumbent.
"""
function solve_bnb_ddfact(
    C::Symmetric,
    s::Int,
    t::Int;
    relaxation = DDGFact,
    fixing_rule::Symbol = :dual,
    psi::Union{Nothing,Float64} = nothing,
    time_limit::Real = 3600.0,
    verbose::Bool = true,
    bfgs_param_set::Symbol = :default,
    bfgs_param_sets::Dict = bfgs_param_sets,
    upsilon_fixing::Symbol = :simple,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)
    @assert 1 <= t <= s < n "need 1 ≤ t ≤ s < n"

    relaxation = _normalize_relaxation(relaxation)

    if fixing_rule ∉ (:dual, :primal, :both)
        error("fixing_rule must be one of :dual, :primal, or :both.")
    end

    if relaxation == :DDGFact && fixing_rule != :dual
        error("DDGFact currently supports only fixing_rule = :dual.")
    end

    if upsilon_fixing ∉ (:simple, :strong)
        error("upsilon_fixing must be either :simple or :strong.")
    end

    # Incumbent (lower bound) from local search.
    x_ls, lb = run_all_LS(C, s, t)
    S_inc = sort(findall(x_ls .> 0.5))

    # Integrality-gap metric over the integer relaxation solutions found.
    n_int_sols  = 0
    int_gap_max = 0.0
    int_gap_sum = 0.0       # running sum of the gaps, for the average
    int_gap_inc = NaN       # gap at the incumbent when it came from an integer relaxation

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
            int_gap_inc = int_gap
        end
    end

    node_kwargs = (
        relaxation = relaxation,
        psi = psi,
        bfgs_param_set = bfgs_param_set,
        bfgs_param_sets = bfgs_param_sets,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
    )

    # Root node.
    r = _gmesp_node(C, s, t, Int[], Int[]; node_kwargs...)
    ub_root = r.ub
    _register!(r)
    verbose && @printf("root:  relaxation = %s   fixing = %s   lb = %.4f   ub = %.4f   gap = %.4f\n",
                       String(relaxation), String(fixing_rule), lb, ub_root, ub_root - lb)

    open = r.determined || r.ub ≤ lb ? GMESPNode[] :
       [GMESPNode(Int[], Int[], ub_root, r.x, r.keep, r.gamma)]
    nodes, nfix0, nfix1 = 1, 0, 0
    report_every = 1000
    next_report = report_every
    t0 = time()
    time_limit_hit = false

    # Bound a child, fix variables, prune; push to `open` if it survives.
    function _child(
        F1::Vector{Int},
        F0::Vector{Int},
        parent_keep::Vector{Int},
        parent_gamma::Union{Nothing,Vector{Float64}},
    )
        theta0 = nothing

        if relaxation == :DDGFactplusUpsilon && parent_gamma !== nothing
            # child_keep = 1:n \ F0
            child_keep = setdiff(1:size(C, 1), F0)

            parent_pos = Dict(v => k for (k, v) in enumerate(parent_keep))

            gamma0 = Float64[]
            for idx in child_keep
                if !haskey(parent_pos, idx)
                    error("Child keep is not a subset of parent keep.")
                end

                push!(gamma0, parent_gamma[parent_pos[idx]])
            end

            theta0 = log.(gamma0)
        end

        r = _gmesp_node(C, s, t, F1, F0; node_kwargs..., theta0 = theta0)
        nodes += 1
        _register!(r)
        r.determined && return                              # leaf: single point, scored
        r.ub ≤ lb && return                                 # prune: can't beat incumbent

        F1f, F0f = variable_fixing_relaxation_soln(
            F1,
            F0,
            r,
            s,
            t,
            lb;
            fixing_rule = fixing_rule,
            tol = atol,
        )

        if length(F1f) > length(F1) || length(F0f) > length(F0)
            nfix1 += length(F1f) - length(F1)
            nfix0 += length(F0f) - length(F0)

            theta0_rebound = nothing

            if relaxation == :DDGFactplusUpsilon && r.gamma !== nothing
                keep_rebound = setdiff(1:n, F0f)
                pos = Dict(v => k for (k, v) in enumerate(r.keep))
                theta0_rebound = log.([r.gamma[pos[i]] for i in keep_rebound])
            end

            r = _gmesp_node(C, s, t, F1f, F0f; node_kwargs..., theta0 = theta0_rebound) # re-bound after fixing
            _register!(r)
            r.determined && return                          # fixing determined it → leaf
            r.ub ≤ lb && return
            F1, F0 = F1f, F0f
        end
        push!(open, GMESPNode(F1, F0, r.ub, r.x, r.keep, r.gamma))
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
        #  no free variable left
        i == 0 && continue

        # Branch setting variable to 1
        F1_child = copy(node.F1)
        push!(F1_child, i)
        sort!(F1_child)

        _child(F1_child, node.F0, node.keep, node.gamma)

        # Branch setting variable to 0
        F0_child = copy(node.F0)
        push!(F0_child, i)
        sort!(F0_child)

        _child(node.F1, F0_child, node.keep, node.gamma)

        if verbose && nodes >= next_report
            best_ub = isempty(open) ? lb : maximum(nd -> nd.ub, open)

            @printf(
                "nodes = %4d  open = %4d  fix0 = %3d  fix1 = %3d  lb = %.4f  ub = %.4f  gap = %.4f\n",
                nodes,
                length(open),
                nfix0,
                nfix1,
                lb,
                best_ub,
                best_ub - lb,
            )
            flush(stdout)

            next_report += report_every
        end
    end

    # When the open list empties, every subtree was either pruned (UB ≤ LB) or
    # branched down to a fully-determined subset, so the incumbent is optimal by exhaustion
    # When stopped early, the best open UB is a valid (loose) bound.
    tree_exhausted = isempty(open)
    final_ub = tree_exhausted ? lb : maximum(nd -> nd.ub, open)
    status = time_limit_hit ? "TIME LIMIT" : (tree_exhausted ? "OPTIMAL (exhausted)" : "OPTIMAL (gap)")

    # Integrality gap at the returned incumbent when it was found as an integer
    # relaxation solution.  If the incumbent came only from local search, this is NaN.
    int_gap_opt = int_gap_inc
    int_gap_avg = n_int_sols > 0 ? int_gap_sum / n_int_sols : NaN

    verbose && @printf("done [%s].  relaxation = %s   fixing = %s   nodes = %d   fix0 = %d  fix1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
                       status, String(relaxation), String(fixing_rule), nodes, nfix0, nfix1, lb, final_ub, final_ub - lb)
    verbose && @printf("           integer solns = %d   max int-gap = %.4f   avg int-gap = %.4f   int-gap@opt = %.4f\n",
                       n_int_sols, int_gap_max, int_gap_avg, int_gap_opt)

    stats = (nodes = nodes, lb = lb, ub = final_ub, gap = final_ub - lb,
             root_ub = ub_root, nfix0 = nfix0, nfix1 = nfix1,
             n_int_sols = n_int_sols, int_gap_max = int_gap_max, int_gap_avg = int_gap_avg,
             int_gap_opt = int_gap_opt,
             wall_time = time() - t0, time_limit_hit = time_limit_hit,
             tree_exhausted = tree_exhausted,
             relaxation = relaxation,
             fixing_rule = fixing_rule,
             bfgs_param_set = bfgs_param_set,
             upsilon_fixing = upsilon_fixing)
    return S_inc, stats
end