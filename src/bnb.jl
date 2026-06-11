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
# Stop when the best UB can no longer beat the incumbent, or the time limit fires.
# =============================================================================

const DDGFact = :DDGFact
const DDGFactplus = :DDGFactplus
const DDGFactplusUpsilon = :DDGFactplusUpsilon

struct GMESPNode
    F1::Vector{Int}
    F0::Vector{Int}
    ub::Float64
    x::Vector{Float64}
    keep::Vector{Int}
    gamma::Union{Nothing,Vector{Float64}}
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


# Convert a parameter-set dictionary into keyword arguments for either
# your custom BFGS implementation or the Optim.jl BFGS wrapper.
#
# upsilon_calibration:
#   :custom_bfgs -> calibrate_upsilon_bfgs_ddfactplus
#   :optim_bfgs  -> calibrate_upsilon_optim_bfgs_ddfactplus
function _calibrate_upsilon_from_param_set(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set::Dict;
    J1::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    atol::Float64 = 1e-10,
    upsilon_calibration::Symbol = :custom_bfgs,
)
    if upsilon_calibration == :custom_bfgs
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
    elseif upsilon_calibration == :optim_bfgs
        return calibrate_upsilon_optim_bfgs_ddfactplus(
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
            t1_reformulation = get(param_set, :t1_reformulation, true),
            theta_perturbation = get(param_set, :theta_perturbation, 1e-4),
            verbose = get(param_set, :verbose_bfgs, false),
        )
    else
        error("upsilon_calibration must be either :custom_bfgs or :optim_bfgs.")
    end
end


# Node relaxation.
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
    upsilon_calibration::Symbol = :custom_bfgs,
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

    if length(F1) > s || s > n_red
        return (ub = -Inf, x = Float64[], y = Float64[], keep = keep,
                determined = false, relaxation = relaxation,
                dual_solution = nothing, l = Float64[], c = Float64[],
                gamma = nothing, psi = nothing, F = nothing,
                upsilon_fixing = upsilon_fixing)
    end

    # Determined subset → single integer point.
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

            calib = _calibrate_upsilon_from_param_set(
                Ck,
                s,
                t,
                bfgs_param_sets[bfgs_param_set];
                J1 = fix1,
                theta0 = theta0,
                atol = atol,
                upsilon_calibration = upsilon_calibration,
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

    Ck = Symmetric(Matrix(C[keep, keep]))
    posn = Dict(v => k for (k, v) in enumerate(keep))
    fix1 = Int[posn[i] for i in F1]
    sort!(fix1)

    l = zeros(n_red)
    c = ones(n_red)
    l[fix1] .= 1.0

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

        calib = _calibrate_upsilon_from_param_set(
            Ck,
            s,
            t,
            bfgs_param_sets[bfgs_param_set];
            J1 = fix1,
            theta0 = theta0,
            atol = atol,
            upsilon_calibration = upsilon_calibration,
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


# Subset implied by a relaxation point.
function _subset_value(
    C::Symmetric,
    s::Int,
    t::Int,
    keep::Vector{Int},
    x::Vector{Float64},
)
    (isempty(x) || length(keep) < s) && return (-Inf, Int[])
    order = sortperm(x, rev = true)
    S = sort(keep[order[1:s]])
    v = try
        _gmesp_obj(C, S, t)
    catch
        -Inf
    end
    return (v, S)
end


function _is_integer_point(x::AbstractVector)
    isempty(x) && return false
    max_dist_to_integer = maximum(abs.(x .- round.(x)))
    return max_dist_to_integer ≤ 1e-6
end


function _branch_var(keep::Vector{Int}, x::Vector{Float64}, F1::Vector{Int})
    F1set = Set(F1)
    best_k, best_f = 0, -1.0

    for k in eachindex(keep)
        keep[k] in F1set && continue
        f = min(x[k], 1 - x[k])

        if f > best_f
            best_f = f
            best_k = k
        end
    end

    return best_k == 0 ? 0 : keep[best_k]
end


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

    F1n = sort(union(F1, r.keep[fix_one]))
    F0n = sort(union(F0, r.keep[fix_zero]))

    return F1n, F0n
end


"""
    solve_bnb_ddfact(C, s, t; relaxation=DDGFact, fixing_rule=:dual,
                     upsilon_calibration=:custom_bfgs,
                     warm_start_parent_upsilon=true, ...)

Branch-and-bound for `max_{|S|=s} Γ_t(C[S,S])`.

For `relaxation = DDGFactplusUpsilon`, choose the Upsilon calibration with:

  - `upsilon_calibration = :custom_bfgs`
  - `upsilon_calibration = :optim_bfgs`

The parameter set selected by `bfgs_param_set` is passed to either method.

If `warm_start_parent_upsilon = true`, child nodes initialize theta0 from the
parent node's gamma restricted to the child's reduced keep set. Rebounds after
variable fixing are also warm-started from the previous node solution.

If `warm_start_parent_upsilon = false`, all child/rebound calibrations use their
default initialization.
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
    upsilon_calibration::Symbol = :custom_bfgs,
    warm_start_parent_upsilon::Bool = true,
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

    if upsilon_calibration ∉ (:custom_bfgs, :optim_bfgs)
        error("upsilon_calibration must be either :custom_bfgs or :optim_bfgs.")
    end

    x_ls, lb = run_all_LS(C, s, t)
    S_inc = sort(findall(x_ls .> 0.5))

    n_int_sols = 0
    int_gap_max = 0.0
    int_gap_sum = 0.0
    int_gap_inc = NaN

    function _register!(r)
        _is_integer_point(r.x) || return
        true_obj, S = _subset_value(C, s, t, r.keep, r.x)
        isfinite(true_obj) || return

        int_gap = r.ub - true_obj
        n_int_sols += 1
        int_gap_sum += int_gap
        int_gap_max = max(int_gap_max, int_gap)

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
        upsilon_calibration = upsilon_calibration,
    )

    r = _gmesp_node(C, s, t, Int[], Int[]; node_kwargs...)
    ub_root = r.ub
    _register!(r)

    verbose && @printf(
        "root:  relaxation = %s   fixing = %s   calibration = %s   warm_start = %s   lb = %.4f   ub = %.4f   gap = %.4f\n",
        String(relaxation),
        String(fixing_rule),
        String(upsilon_calibration),
        string(warm_start_parent_upsilon),
        lb,
        ub_root,
        ub_root - lb,
    )

    open =
        r.determined || r.ub ≤ lb ?
        GMESPNode[] :
        [GMESPNode(Int[], Int[], ub_root, r.x, r.keep, r.gamma)]

    nodes = 1
    nfix0 = 0
    nfix1 = 0

    report_every = 1000
    next_report = report_every

    t0 = time()
    time_limit_hit = false

    function _child(
        F1::Vector{Int},
        F0::Vector{Int},
        parent_keep::Vector{Int},
        parent_gamma::Union{Nothing,Vector{Float64}},
    )
        theta0 = nothing

        if warm_start_parent_upsilon &&
           relaxation == :DDGFactplusUpsilon &&
           parent_gamma !== nothing

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

        r.determined && return
        r.ub ≤ lb && return

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

            if warm_start_parent_upsilon &&
               relaxation == :DDGFactplusUpsilon &&
               r.gamma !== nothing

                keep_rebound = setdiff(1:n, F0f)
                pos = Dict(v => k for (k, v) in enumerate(r.keep))

                gamma0_rebound = Float64[]

                for idx in keep_rebound
                    if !haskey(pos, idx)
                        error("Rebound keep is not a subset of current keep.")
                    end

                    push!(gamma0_rebound, r.gamma[pos[idx]])
                end

                theta0_rebound = log.(gamma0_rebound)
            end

            r = _gmesp_node(
                C,
                s,
                t,
                F1f,
                F0f;
                node_kwargs...,
                theta0 = theta0_rebound,
            )

            _register!(r)

            r.determined && return
            r.ub ≤ lb && return

            F1 = F1f
            F0 = F0f
        end

        push!(open, GMESPNode(F1, F0, r.ub, r.x, r.keep, r.gamma))
    end

    while !isempty(open)
        if time() - t0 ≥ time_limit
            time_limit_hit = true
            break
        end

        sort!(open; by = nd -> nd.ub, rev = true)

        first(open).ub ≤ lb && break

        node = popfirst!(open)

        i = _branch_var(node.keep, node.x, node.F1)
        i == 0 && continue

        F1_child = copy(node.F1)
        push!(F1_child, i)
        sort!(F1_child)

        _child(F1_child, node.F0, node.keep, node.gamma)

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

    tree_exhausted = isempty(open)
    final_ub = tree_exhausted ? lb : maximum(nd -> nd.ub, open)

    status =
        time_limit_hit ?
        "TIME LIMIT" :
        (tree_exhausted ? "OPTIMAL (exhausted)" : "OPTIMAL (gap)")

    int_gap_opt = int_gap_inc
    int_gap_avg = n_int_sols > 0 ? int_gap_sum / n_int_sols : NaN

    verbose && @printf(
        "done [%s].  relaxation = %s   fixing = %s   calibration = %s   warm_start = %s   nodes = %d   fix0 = %d  fix1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
        status,
        String(relaxation),
        String(fixing_rule),
        String(upsilon_calibration),
        string(warm_start_parent_upsilon),
        nodes,
        nfix0,
        nfix1,
        lb,
        final_ub,
        final_ub - lb,
    )

    verbose && @printf(
        "           integer solns = %d   max int-gap = %.4f   avg int-gap = %.4f   int-gap@opt = %.4f\n",
        n_int_sols,
        int_gap_max,
        int_gap_avg,
        int_gap_opt,
    )

    stats = (
        nodes = nodes,
        lb = lb,
        ub = final_ub,
        gap = final_ub - lb,
        root_ub = ub_root,
        nfix0 = nfix0,
        nfix1 = nfix1,
        n_int_sols = n_int_sols,
        int_gap_max = int_gap_max,
        int_gap_avg = int_gap_avg,
        int_gap_opt = int_gap_opt,
        wall_time = time() - t0,
        time_limit_hit = time_limit_hit,
        tree_exhausted = tree_exhausted,
        relaxation = relaxation,
        fixing_rule = fixing_rule,
        bfgs_param_set = bfgs_param_set,
        upsilon_fixing = upsilon_fixing,
        upsilon_calibration = upsilon_calibration,
        warm_start_parent_upsilon = warm_start_parent_upsilon,
    )

    return S_inc, stats
end