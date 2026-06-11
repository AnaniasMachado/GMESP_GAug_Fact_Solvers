using LinearAlgebra
using Printf

# =============================================================================
# Utilities for GMESP branch-and-bound
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


# =============================================================================
# Basic utilities
# =============================================================================

function _normalize_relaxation(relaxation)
    r = relaxation isa Symbol ? relaxation : Symbol(relaxation)

    if r ∉ (:DDGFact, :DDGFactplus, :DDGFactplusUpsilon)
        error("relaxation must be one of DDGFact, DDGFactplus, or DDGFactplusUpsilon.")
    end

    return r
end


function _normalize_fixing_rule(fixing_rule::Symbol)
    if fixing_rule ∉ (:none, :dual, :primal, :both)
        error("fixing_rule must be one of :none, :dual, :primal, or :both.")
    end

    return fixing_rule
end


function _validate_bnb_options(
    relaxation::Symbol,
    fixing_rule::Symbol,
    upsilon_fixing::Symbol,
)
    if fixing_rule ∉ (:none, :dual, :primal, :both)
        error("fixing_rule must be one of :none, :dual, :primal, or :both.")
    end

    if relaxation == :DDGFact && fixing_rule ∉ (:none, :dual)
        error("DDGFact currently supports only fixing_rule = :none or :dual.")
    end

    if upsilon_fixing ∉ (:simple, :strong)
        error("upsilon_fixing must be either :simple or :strong.")
    end

    return nothing
end


# Exact GMESP objective at an integer subset S.
function _gmesp_obj(C::AbstractMatrix, S::Vector{Int}, t::Int)
    λ = reverse(eigvals(Symmetric(Matrix(C[S, S]))))
    return sum(log, max.(λ[1:t], 1e-30))
end


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

    return v, S
end


function _is_integer_point(x::AbstractVector)
    isempty(x) && return false
    max_dist_to_integer = maximum(abs.(x .- round.(x)))
    return max_dist_to_integer ≤ 1e-6
end


function _branch_var(
    keep::Vector{Int},
    x::Vector{Float64},
    F1::Vector{Int},
)
    F1set = Set(F1)
    best_k = 0
    best_f = -1.0

    for k in eachindex(keep)
        keep[k] in F1set && continue

        f = min(x[k], 1.0 - x[k])

        if f > best_f
            best_f = f
            best_k = k
        end
    end

    return best_k == 0 ? 0 : keep[best_k]
end


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


# =============================================================================
# Upsilon calibration
# =============================================================================

function _calibrate_upsilon_from_param_set(
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
        t1_reformulation = get(param_set, :t1_reformulation, true),
        t1_fallback = get(param_set, :t1_fallback, true),
        t1_fallback_limit = get(param_set, :t1_fallback_limit, 1),
        theta_perturbation = get(param_set, :theta_perturbation, 1e-4),
        use_steepest_descent_fallback =
            get(param_set, :use_steepest_descent_fallback, true),
        verbose = get(param_set, :verbose_bfgs, false),
    )
end


# =============================================================================
# Optional root psi = 0 Upsilon warm start
# =============================================================================

function _calibrate_root_upsilon_psi0_from_param_set(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set::Dict;
    atol::Float64 = 1e-10,
)
    return calibrate_upsilon_bfgs_ddfact(
        C,
        s,
        t;
        J1 = Int[],
        theta0 = nothing,
        atol = atol,
        max_iter = get(param_set, :max_iter, get(param_set, :max_bfgs_iter, 20)),
        grad_tol = get(param_set, :grad_tol, 1e-6),
        step_tol = get(param_set, :step_tol, 1e-10),
        alpha0 = get(param_set, :alpha0, 1.0),
        alpha_min = get(param_set, :alpha_min, 1e-8),
        alpha_decay = get(param_set, :alpha_decay, 0.5),
        armijo_c1 = get(param_set, :armijo_c1, 1e-4),
        curvature_tol = get(param_set, :curvature_tol, 1e-10),
        max_backtracks = get(param_set, :max_backtracks, 20),
        max_theta_norm = get(param_set, :max_theta_norm, 50.0),
        t1_reformulation = get(param_set, :t1_reformulation, true),
        t1_fallback = get(param_set, :t1_fallback, true),
        t1_fallback_limit = get(param_set, :t1_fallback_limit, 1),
        theta_perturbation = get(param_set, :theta_perturbation, 1e-4),
        use_steepest_descent_fallback =
            get(param_set, :use_steepest_descent_fallback, true),
        verbose = get(param_set, :verbose_bfgs, false),
    )
end


function _root_theta0_from_psi0_warm_start(
    C::Symmetric,
    s::Int,
    t::Int;
    relaxation::Symbol,
    root_psi0_warm_start::Bool,
    root_psi0_bfgs_param_set::Symbol,
    bfgs_param_sets::Dict,
    atol::Float64,
    verbose::Bool,
)
    if !root_psi0_warm_start || relaxation != :DDGFactplusUpsilon
        return nothing, nothing
    end

    haskey(bfgs_param_sets, root_psi0_bfgs_param_set) ||
        error("Unknown root psi=0 BFGS parameter set: $root_psi0_bfgs_param_set.")

    root_psi0_calib = _calibrate_root_upsilon_psi0_from_param_set(
        C,
        s,
        t,
        bfgs_param_sets[root_psi0_bfgs_param_set];
        atol = atol,
    )

    root_theta0 = copy(root_psi0_calib.theta)

    if verbose
        @printf(
            "root psi=0 Upsilon warm-start: param_set = %s   obj = %.4f   psi = %.4f   improved = %s\n",
            String(root_psi0_bfgs_param_set),
            root_psi0_calib.obj,
            root_psi0_calib.psi,
            string(root_psi0_calib.improved),
        )
        flush(stdout)
    end

    return root_theta0, root_psi0_calib
end


function _local_fix1_indices(
    keep::Vector{Int},
    F1::Vector{Int},
)
    posn = Dict(v => k for (k, v) in enumerate(keep))
    fix1 = Int[posn[i] for i in F1]
    sort!(fix1)
    return fix1
end


# =============================================================================
# Node relaxation
# =============================================================================

function _infeasible_node_return(
    keep::Vector{Int},
    relaxation::Symbol,
    upsilon_fixing::Symbol,
)
    return (
        ub = -Inf,
        x = Float64[],
        y = Float64[],
        keep = keep,
        determined = false,
        relaxation = relaxation,
        dual_solution = nothing,
        l = Float64[],
        c = Float64[],
        gamma = nothing,
        psi = nothing,
        F = nothing,
        upsilon_fixing = upsilon_fixing,
        reused_parent_upsilon = false,
    )
end


function _determined_node(
    C::Symmetric,
    Ck::Symmetric,
    s::Int,
    t::Int,
    keep::Vector{Int},
    F1::Vector{Int},
    x::Vector{Float64};
    relaxation::Symbol,
    psi::Union{Nothing,Float64},
    bfgs_param_set::Symbol,
    bfgs_param_sets::Dict,
    theta0::Union{Nothing,Vector{Float64}},
    fixed_upsilon_gamma::Union{Nothing,Vector{Float64}},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    upsilon_fixing::Symbol,
)
    if relaxation == :DDGFact
        ub_node = DDGFact_value_at_x(
            x,
            Ck,
            t;
            atol = atol,
        )

        return (
            ub = ub_node,
            x = x,
            y = Float64[],
            keep = keep,
            determined = true,
            relaxation = relaxation,
            dual_solution = nothing,
            l = Float64[],
            c = Float64[],
            gamma = nothing,
            psi = 0.0,
            F = nothing,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
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

        return (
            ub = ub_node,
            x = x,
            y = Float64[],
            keep = keep,
            determined = true,
            relaxation = relaxation,
            dual_solution = nothing,
            l = Float64[],
            c = Float64[],
            gamma = nothing,
            psi = psi_node,
            F = nothing,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
    else
        fix1 = _local_fix1_indices(keep, F1)

        if fixed_upsilon_gamma !== nothing
            gamma = copy(fixed_upsilon_gamma)

            if length(gamma) != size(Ck, 1)
                error("fixed Upsilon gamma must have length equal to size(Ck, 1).")
            end

            if any(gamma .<= 0.0)
                error("fixed Upsilon gamma must be strictly positive.")
            end

            psi_node, λmin_node = max_feasible_psi(
                Ck,
                gamma;
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )

            ub_node, y, F = DDGFactplusUpsilon_value_at_x(
                x,
                Ck,
                gamma,
                t,
                psi_node;
                atol = atol,
            )

            return (
                ub = ub_node,
                x = x,
                y = y,
                keep = keep,
                determined = true,
                relaxation = relaxation,
                dual_solution = nothing,
                l = Float64[],
                c = Float64[],
                gamma = gamma,
                psi = psi_node,
                F = F,
                primal_obj = ub_node,
                bfgs = nothing,
                upsilon_fixing = upsilon_fixing,
                reused_parent_upsilon = true,
            )
        end

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

        return (
            ub = ub_node,
            x = x,
            y = y,
            keep = keep,
            determined = true,
            relaxation = relaxation,
            dual_solution = nothing,
            l = Float64[],
            c = Float64[],
            gamma = gamma,
            psi = psi_node,
            F = F,
            primal_obj = ub_node,
            bfgs = calib,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
    end
end


function _bound_ddgfact_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    atol::Float64,
)
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

    return x, primal_obj, F, dual_sol
end


function _bound_ddgfactplus_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    psi::Union{Nothing,Float64},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
)
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

    return x, primal_obj, F, dual_sol, psi_node
end


function _bound_ddgfactplus_upsilon_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    bfgs_param_set::Symbol,
    bfgs_param_sets::Dict,
    theta0::Union{Nothing,Vector{Float64}},
    atol::Float64,
)
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

    return x, y, primal_obj, gamma, psi_node, F, dual_sol, calib
end


function _bound_ddgfactplus_upsilon_node_fixed_gamma(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64},
    gamma::Vector{Float64};
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
)
    if length(gamma) != size(Ck, 1)
        error("fixed Upsilon gamma must have length equal to size(Ck, 1).")
    end

    if any(gamma .<= 0.0)
        error("fixed Upsilon gamma must be strictly positive.")
    end

    psi_node, λmin_node = max_feasible_psi(
        Ck,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    x, y, primal_obj = aug_ddfact_upsilon_gmesp(
        Ck,
        gamma,
        s,
        t,
        psi_node;
        J1 = fix1,
        atol = atol,
    )

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

    return x, y, primal_obj, gamma, psi_node, F, dual_sol, λmin_node
end


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
    fixed_upsilon_gamma::Union{Nothing,Vector{Float64}} = nothing,
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
        return _infeasible_node_return(keep, relaxation, upsilon_fixing)
    end

    if length(F1) > s || s > n_red
        return _infeasible_node_return(keep, relaxation, upsilon_fixing)
    end

    Ck = Symmetric(Matrix(C[keep, keep]))

    if length(F1) == s || n_red == s
        S = length(F1) == s ? sort(F1) : copy(keep)
        Sset = Set(S)
        x = [keep[k] in Sset ? 1.0 : 0.0 for k in eachindex(keep)]

        return _determined_node(
            C,
            Ck,
            s,
            t,
            keep,
            F1,
            x;
            relaxation = relaxation,
            psi = psi,
            bfgs_param_set = bfgs_param_set,
            bfgs_param_sets = bfgs_param_sets,
            theta0 = theta0,
            fixed_upsilon_gamma = fixed_upsilon_gamma,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            upsilon_fixing = upsilon_fixing,
        )
    end

    fix1 = _local_fix1_indices(keep, F1)

    l = zeros(n_red)
    c = ones(n_red)
    l[fix1] .= 1.0

    if relaxation == :DDGFact
        x, primal_obj, F, dual_sol =
            _bound_ddgfact_node(
                Ck,
                s,
                t,
                fix1,
                l,
                c;
                atol = atol,
            )

        return (
            ub = dual_sol.objective_value,
            x = x,
            y = Float64[],
            keep = keep,
            determined = false,
            relaxation = relaxation,
            dual_solution = dual_sol,
            l = l,
            c = c,
            gamma = nothing,
            psi = 0.0,
            F = F,
            primal_obj = primal_obj,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
    elseif relaxation == :DDGFactplus
        x, primal_obj, F, dual_sol, psi_node =
            _bound_ddgfactplus_node(
                Ck,
                s,
                t,
                fix1,
                l,
                c;
                psi = psi,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )

        return (
            ub = dual_sol.objective_value,
            x = x,
            y = Float64[],
            keep = keep,
            determined = false,
            relaxation = relaxation,
            dual_solution = dual_sol,
            l = l,
            c = c,
            gamma = nothing,
            psi = psi_node,
            F = F,
            primal_obj = primal_obj,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
    else
        if fixed_upsilon_gamma !== nothing
            gamma_fixed = copy(fixed_upsilon_gamma)

            x, y, primal_obj, gamma, psi_node, F, dual_sol, λmin_node =
                _bound_ddgfactplus_upsilon_node_fixed_gamma(
                    Ck,
                    s,
                    t,
                    fix1,
                    l,
                    c,
                    gamma_fixed;
                    atol = atol,
                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                )

            return (
                ub = dual_sol.objective_value,
                x = x,
                y = y,
                keep = keep,
                determined = false,
                relaxation = relaxation,
                dual_solution = dual_sol,
                l = l,
                c = c,
                gamma = gamma,
                psi = psi_node,
                F = F,
                primal_obj = primal_obj,
                bfgs = nothing,
                upsilon_fixing = upsilon_fixing,
                reused_parent_upsilon = true,
            )
        end

        x, y, primal_obj, gamma, psi_node, F, dual_sol, calib =
            _bound_ddgfactplus_upsilon_node(
                Ck,
                s,
                t,
                fix1,
                l,
                c;
                bfgs_param_set = bfgs_param_set,
                bfgs_param_sets = bfgs_param_sets,
                theta0 = theta0,
                atol = atol,
            )

        return (
            ub = dual_sol.objective_value,
            x = x,
            y = y,
            keep = keep,
            determined = false,
            relaxation = relaxation,
            dual_solution = dual_sol,
            l = l,
            c = c,
            gamma = gamma,
            psi = psi_node,
            F = F,
            primal_obj = primal_obj,
            bfgs = calib,
            upsilon_fixing = upsilon_fixing,
            reused_parent_upsilon = false,
        )
    end
end


# =============================================================================
# Variable fixing
# =============================================================================

function _dual_variable_fixing(
    r,
    s::Int,
    t::Int,
    lb::Float64;
    tol::Real = 1e-9,
)
    if r.relaxation == :DDGFact
        return var_fixing_from_DGFact(
            r.dual_solution.upsilon,
            r.dual_solution.nu,
            r.ub,
            lb;
            l = r.l,
            c = r.c,
            atol = Float64(tol),
        )
    elseif r.relaxation == :DDGFactplus
        return var_fixing_from_DGFactplus(
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
            return var_fixing_from_DGFactplusUpsilon_strong(
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
            return var_fixing_from_DGFactplusUpsilon_simple(
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
end


function _primal_variable_fixing(
    r,
    s::Int,
    t::Int,
    lb::Float64;
    tol::Real = 1e-9,
)
    if r.relaxation == :DDGFactplus
        return var_fixing_DDGFactplus_primal(
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
    elseif r.relaxation == :DDGFactplusUpsilon
        return var_fixing_DDGFactplusUpsilon_primal(
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
    else
        return (fix_zero = Int[], fix_one = Int[])
    end
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
    fixing_rule = _normalize_fixing_rule(fixing_rule)

    if fixing_rule == :none
        return sort(F1), sort(F0)
    end

    r.dual_solution === nothing && return sort(F1), sort(F0)

    if r.relaxation == :DDGFact && fixing_rule ∉ (:none, :dual)
        error("DDGFact currently supports only fixing_rule = :none or :dual.")
    end

    fix_zero = Int[]
    fix_one = Int[]

    if fixing_rule in (:dual, :both)
        fixing_dual = _dual_variable_fixing(
            r,
            s,
            t,
            lb;
            tol = tol,
        )

        append!(fix_zero, fixing_dual.fix_zero)
        append!(fix_one, fixing_dual.fix_one)
    end

    if fixing_rule in (:primal, :both)
        fixing_primal = _primal_variable_fixing(
            r,
            s,
            t,
            lb;
            tol = tol,
        )

        append!(fix_zero, fixing_primal.fix_zero)
        append!(fix_one, fixing_primal.fix_one)
    end

    fix_zero = sort(unique(fix_zero))
    fix_one = sort(unique(fix_one))

    F1n = sort(union(F1, r.keep[fix_one]))
    F0n = sort(union(F0, r.keep[fix_zero]))

    return F1n, F0n
end


# =============================================================================
# Upsilon warm start / reuse utilities
# =============================================================================

function _theta0_from_parent_gamma(
    n::Int,
    F0::Vector{Int},
    parent_keep::Vector{Int},
    parent_gamma::Union{Nothing,Vector{Float64}},
)
    parent_gamma === nothing && return nothing

    child_keep = setdiff(1:n, F0)
    parent_pos = Dict(v => k for (k, v) in enumerate(parent_keep))

    gamma0 = Float64[]

    for idx in child_keep
        if !haskey(parent_pos, idx)
            error("Child keep is not a subset of parent keep.")
        end

        push!(gamma0, parent_gamma[parent_pos[idx]])
    end

    return log.(gamma0)
end


function _theta0_from_current_gamma_after_fixing(
    n::Int,
    F0f::Vector{Int},
    current_keep::Vector{Int},
    current_gamma::Union{Nothing,Vector{Float64}},
)
    current_gamma === nothing && return nothing

    keep_rebound = setdiff(1:n, F0f)
    pos = Dict(v => k for (k, v) in enumerate(current_keep))

    gamma0 = Float64[]

    for idx in keep_rebound
        if !haskey(pos, idx)
            error("Rebound keep is not a subset of current keep.")
        end

        push!(gamma0, current_gamma[pos[idx]])
    end

    return log.(gamma0)
end


function _gamma_from_parent_gamma(
    n::Int,
    F0::Vector{Int},
    parent_keep::Vector{Int},
    parent_gamma::Union{Nothing,Vector{Float64}},
)
    parent_gamma === nothing && return nothing

    child_keep = setdiff(1:n, F0)
    parent_pos = Dict(v => k for (k, v) in enumerate(parent_keep))

    gamma_child = Float64[]

    for idx in child_keep
        if !haskey(parent_pos, idx)
            error("Child keep is not a subset of parent keep.")
        end

        push!(gamma_child, parent_gamma[parent_pos[idx]])
    end

    return gamma_child
end


function _gamma_from_current_gamma_after_fixing(
    n::Int,
    F0f::Vector{Int},
    current_keep::Vector{Int},
    current_gamma::Union{Nothing,Vector{Float64}},
)
    current_gamma === nothing && return nothing

    keep_rebound = setdiff(1:n, F0f)
    pos = Dict(v => k for (k, v) in enumerate(current_keep))

    gamma_rebound = Float64[]

    for idx in keep_rebound
        if !haskey(pos, idx)
            error("Rebound keep is not a subset of current keep.")
        end

        push!(gamma_rebound, current_gamma[pos[idx]])
    end

    return gamma_rebound
end