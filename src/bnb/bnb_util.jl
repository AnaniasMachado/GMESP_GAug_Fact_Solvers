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


function _normalize_calibration_method(calibration_method)
    m = calibration_method isa Symbol ? calibration_method : Symbol(calibration_method)

    if m ∉ (:bfgs, :rbfgs, :prox_step)
        error("calibration_method must be one of :bfgs, :rbfgs, or :prox_step.")
    end

    return m
end


function _validate_bnb_options(
    relaxation::Symbol,
    fixing_rule::Symbol,
    upsilon_fixing::Symbol,
    calibration_method::Symbol,
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

    _normalize_calibration_method(calibration_method)

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
# B&B timing accumulator
# =============================================================================

const BNB_TIMING = Dict{Symbol,Float64}()

function _reset_bnb_timing!()
    empty!(BNB_TIMING)

    BNB_TIMING[:knitro_time] = 0.0
    BNB_TIMING[:relaxation_solve_time] = 0.0
    BNB_TIMING[:upsilon_calibration_time] = 0.0
    BNB_TIMING[:factorization_time] = 0.0
    BNB_TIMING[:dual_solution_time] = 0.0
    BNB_TIMING[:variable_fixing_time] = 0.0
    BNB_TIMING[:variable_fixing_calls] = 0.0
    BNB_TIMING[:open_list_time] = 0.0
    BNB_TIMING[:node_setup_time] = 0.0
    BNB_TIMING[:bound_computation_time] = 0.0

    return nothing
end


function _add_bnb_timing!(key::Symbol, value::Real)
    BNB_TIMING[key] = get(BNB_TIMING, key, 0.0) + Float64(value)
    return nothing
end


function _copy_bnb_timing()
    return copy(BNB_TIMING)
end


# =============================================================================
# Upsilon calibration dispatch
# =============================================================================

function _calibrate_upsilon_bfgs_from_params(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    return calibrate_upsilon_bfgs_ddfactplus(
        C,
        s,
        t;
        J1 = J1,
        theta0 = nothing,
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
        # DDGFact+_Upsilon relaxation tolerances.
        knitro_outlev = get(param_set, :knitro_outlev, nothing),
        knitro_opttol = get(param_set, :knitro_opttol, nothing),
        knitro_feastol = get(param_set, :knitro_feastol, nothing),
        verbose = get(param_set, :verbose, get(param_set, :verbose_bfgs, false)),
    )
end


function _calibrate_upsilon_rbfgs_from_params(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    return solve_regularized_bfgs_upsilon_calibration(
        C,
        s,
        t;
        J1 = J1,
        q0 = nothing,

        atol = atol,
        psi_margin = get(param_set, :psi_margin, 1e-7),
        psi_floor = get(param_set, :psi_floor, 0.0),
        psi_derivative = get(param_set, :psi_derivative, true),
        t1_reformulation = get(param_set, :t1_reformulation, false),

        q_bound = get(param_set, :q_bound, Inf),
        max_q_norm_inf = get(param_set, :max_q_norm_inf, 20.0),

        max_iter = get(param_set, :max_iter, get(param_set, :max_regbfgs_iter, 50)),

        B0_scale = get(param_set, :B0_scale, 1.0),

        mu0 = get(param_set, :mu0, 1e-2),
        mu_min = get(param_set, :mu_min, 1e-10),
        mu_max = get(param_set, :mu_max, 1e8),
        mu_decrease = get(param_set, :mu_decrease, 0.2),
        mu_increase = get(param_set, :mu_increase, 5.0),
        eta1 = get(param_set, :eta1, 0.05),
        eta2 = get(param_set, :eta2, 0.75),
        max_inner_regularization = get(param_set, :max_inner_regularization, 20),

        normalize_direction = get(param_set, :normalize_direction, false),
        max_direction_norm = get(param_set, :max_direction_norm, 10.0),

        armijo_c1 = get(param_set, :armijo_c1, 1e-4),
        accept_tol = get(param_set, :accept_tol, 1e-12),
        alpha0 = get(param_set, :alpha0, 1.0),
        alpha_min = get(param_set, :alpha_min, 1e-12),
        alpha_decay = get(param_set, :alpha_decay, 0.5),
        max_backtracks = get(param_set, :max_backtracks, 30),

        nonmonotone = get(param_set, :nonmonotone, true),
        nonmonotone_window = get(param_set, :nonmonotone_window, 10),

        curvature_tol = get(param_set, :curvature_tol, 1e-12),
        damping_delta = get(param_set, :damping_delta, 0.2),
        reset_B_on_failed_update = get(param_set, :reset_B_on_failed_update, false),

        project_spd = get(param_set, :project_spd, true),
        min_B_eig = get(param_set, :min_B_eig, 1e-8),
        max_B_eig = get(param_set, :max_B_eig, 1e8),
        reset_B_on_bad = get(param_set, :reset_B_on_bad, true),
        max_B_norm = get(param_set, :max_B_norm, 1e8),

        grad_tol = get(param_set, :grad_tol, 1e-8),
        step_tol = get(param_set, :step_tol, 1e-10),

        cache_digits = get(param_set, :cache_digits, 12),
        diagnostics = get(param_set, :diagnostics, false),
        verbose = get(param_set, :verbose, get(param_set, :verbose_regbfgs, false)),
    )
end


function _calibrate_upsilon_prox_step_from_params(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    param_set;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    return solve_one_step_proximal_knitro_upsilon_calibration(
        C,
        s,
        t;
        J1 = J1,
        J0 = Int[],

        theta0 = get(param_set, :theta0, nothing),

        rho = get(param_set, :rho, 1e-3),

        theta_perturbation = get(param_set, :theta_perturbation, 1e-2),
        center_initial_theta = get(param_set, :center_initial_theta, false),

        # Prefer new name, but allow old q_bound dictionaries to keep working.
        theta_bound = get(param_set, :theta_bound, get(param_set, :q_bound, 20.0)),

        psi_margin = get(param_set, :psi_margin, 1e-7),
        psi_floor = get(param_set, :psi_floor, 0.0),
        psi_derivative = get(param_set, :psi_derivative, true),
        t1_reformulation = get(param_set, :t1_reformulation, false),

        atol = atol,

        # DDGFact+_Upsilon relaxation tolerances.
        relax_knitro_outlev = get(param_set, :relax_knitro_outlev, nothing),
        relax_knitro_opttol = get(param_set, :relax_knitro_opttol, nothing),
        relax_knitro_feastol = get(param_set, :relax_knitro_feastol, nothing),

        # Proximal subproblem tolerances.
        knitro_feastol = get(param_set, :knitro_feastol, 1e-6),
        knitro_opttol = get(param_set, :knitro_opttol, 1e-2),
        knitro_xtol = get(param_set, :knitro_xtol, 1e-4),
        knitro_ftol = get(param_set, :knitro_ftol, 1e-5),

        knitro_maxtime_real = get(param_set, :knitro_maxtime_real, Inf),
        knitro_algorithm = get(param_set, :knitro_algorithm, nothing),
        knitro_bar_murule = get(param_set, :knitro_bar_murule, nothing),
        knitro_honorbnds = get(param_set, :knitro_honorbnds, 1),
        knitro_outlev = get(param_set, :knitro_outlev, 0),

        cache_digits = get(param_set, :cache_digits, 12),
        diagnostics = get(param_set, :diagnostics, false),
        verbose = get(param_set, :verbose, false),
    )
end


function _calibrate_upsilon_from_params(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    calibration_method::Symbol,
    calibration_params;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    method = _normalize_calibration_method(calibration_method)

    if method == :bfgs
        return _calibrate_upsilon_bfgs_from_params(
            C,
            s,
            t,
            calibration_params;
            J1 = J1,
            atol = atol,
        )
    elseif method == :rbfgs
        return _calibrate_upsilon_rbfgs_from_params(
            C,
            s,
            t,
            calibration_params;
            J1 = J1,
            atol = atol,
        )
    elseif method == :prox_step
        return _calibrate_upsilon_prox_step_from_params(
            C,
            s,
            t,
            calibration_params;
            J1 = J1,
            atol = atol,
        )
    else
        error("Unknown calibration method: $(method).")
    end
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
    calibration_method::Symbol,
    calibration_params,
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    upsilon_fixing::Symbol,
)
    if relaxation == :DDGFact
        local ub_node

        bound_time = @elapsed begin
            ub_node = DDGFact_value_at_x(
                x,
                Ck,
                t;
                atol = atol,
            )
        end

        _add_bnb_timing!(:relaxation_solve_time, bound_time)
        _add_bnb_timing!(:knitro_time, bound_time)
        _add_bnb_timing!(:bound_computation_time, bound_time)

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
        )

    elseif relaxation == :DDGFactplus
        local psi_node

        psi_time = @elapsed begin
            psi_node =
                psi === nothing ?
                _default_psi(Ck; psi_margin = psi_margin, psi_floor = psi_floor) :
                psi
        end

        _add_bnb_timing!(:node_setup_time, psi_time)

        local ub_node

        bound_time = @elapsed begin
            ub_node = DDGFactplus_value_at_x(
                x,
                Ck,
                t,
                psi_node;
                atol = atol,
            )
        end

        _add_bnb_timing!(:relaxation_solve_time, bound_time)
        _add_bnb_timing!(:knitro_time, bound_time)
        _add_bnb_timing!(:bound_computation_time, bound_time)

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
        )

    else
        local fix1

        setup_time = @elapsed begin
            fix1 = _local_fix1_indices(keep, F1)
        end

        _add_bnb_timing!(:node_setup_time, setup_time)

        local calib

        calibration_time = @elapsed begin
            calib = _calibrate_upsilon_from_params(
                Ck,
                s,
                t,
                calibration_method,
                calibration_params;
                J1 = fix1,
                atol = atol,
            )
        end

        _add_bnb_timing!(:upsilon_calibration_time, calibration_time)
        _add_bnb_timing!(:knitro_time, calibration_time)

        gamma = calib.gamma
        psi_node = calib.psi

        local ub_node
        local y
        local F

        value_time = @elapsed begin
            ub_node, y, F = DDGFactplusUpsilon_value_at_x(
                x,
                Ck,
                gamma,
                t,
                psi_node;
                atol = atol,
            )
        end

        _add_bnb_timing!(:factorization_time, value_time)

        _add_bnb_timing!(
            :bound_computation_time,
            calibration_time + value_time,
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
            calibration = calib,
            upsilon_fixing = upsilon_fixing,
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
    local x
    local primal_obj

    relaxation_time = @elapsed begin
        x, primal_obj = ddfact_gmesp(
            Ck,
            s,
            t;
            J1 = fix1,
            atol = atol,
        )
    end

    _add_bnb_timing!(:relaxation_solve_time, relaxation_time)
    _add_bnb_timing!(:knitro_time, relaxation_time)

    local F

    factorization_time = @elapsed begin
        F = factorize_matrix(Ck; atol = atol)
    end

    _add_bnb_timing!(:factorization_time, factorization_time)

    local dual_sol

    dual_time = @elapsed begin
        dual_sol = DGFact_dual_solution_from_DDGFact_x(
            x,
            F,
            s,
            t;
            l = l,
            c = c,
            atol = atol,
        )
    end

    _add_bnb_timing!(:dual_solution_time, dual_time)

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

    local x
    local primal_obj

    relaxation_time = @elapsed begin
        x, primal_obj = aug_ddfact_gmesp(
            Ck,
            s,
            t,
            psi_node;
            J1 = fix1,
            atol = atol,
        )
    end

    _add_bnb_timing!(:relaxation_solve_time, relaxation_time)
    _add_bnb_timing!(:knitro_time, relaxation_time)

    local F

    factorization_time = @elapsed begin
        F = factorize_matrix(Ck; psi = psi_node, atol = atol)
    end

    _add_bnb_timing!(:factorization_time, factorization_time)

    local dual_sol

    dual_time = @elapsed begin
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
    end

    _add_bnb_timing!(:dual_solution_time, dual_time)

    return x, primal_obj, F, dual_sol, psi_node
end


function _bound_ddgfactplus_upsilon_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    calibration_method::Symbol,
    calibration_params,
    atol::Float64,
)
    local calib

    calibration_time = @elapsed begin
        calib = _calibrate_upsilon_from_params(
            Ck,
            s,
            t,
            calibration_method,
            calibration_params;
            J1 = fix1,
            atol = atol,
        )
    end

    _add_bnb_timing!(:upsilon_calibration_time, calibration_time)
    _add_bnb_timing!(:knitro_time, calibration_time)

    gamma = calib.gamma
    psi_node = calib.psi
    x = copy(calib.x)
    y = copy(calib.y)
    primal_obj = calib.obj

    local F

    factorization_time = @elapsed begin
        F = scaled_factorize_matrix(
            Ck,
            gamma,
            psi_node;
            atol = atol,
        )
    end

    _add_bnb_timing!(:factorization_time, factorization_time)

    local dual_sol

    dual_time = @elapsed begin
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
    end

    _add_bnb_timing!(:dual_solution_time, dual_time)

    return x, y, primal_obj, gamma, psi_node, F, dual_sol, calib
end


function _gmesp_node(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int};
    relaxation = DDGFact,
    psi::Union{Nothing,Float64} = nothing,
    calibration_method::Symbol = :bfgs,
    calibration_params = Dict{Symbol,Any}(),
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
    upsilon_fixing::Symbol = :simple,
)
    relaxation = _normalize_relaxation(relaxation)
    calibration_method = _normalize_calibration_method(calibration_method)

    local n
    local keep
    local n_red
    local infeasible_intersection
    local infeasible_size
    local Ck

    setup_time = @elapsed begin
        n = size(C, 1)
        keep = setdiff(1:n, F0)
        n_red = length(keep)

        infeasible_intersection = !isempty(intersect(F1, F0))
        infeasible_size = length(F1) > s || s > n_red

        if !(infeasible_intersection || infeasible_size)
            Ck = Symmetric(Matrix(C[keep, keep]))
        end
    end

    _add_bnb_timing!(:node_setup_time, setup_time)

    if infeasible_intersection || infeasible_size
        return _infeasible_node_return(keep, relaxation, upsilon_fixing)
    end

    if length(F1) == s || n_red == s
        local S
        local Sset
        local x

        setup_time = @elapsed begin
            S = length(F1) == s ? sort(F1) : copy(keep)
            Sset = Set(S)
            x = [keep[k] in Sset ? 1.0 : 0.0 for k in eachindex(keep)]
        end

        _add_bnb_timing!(:node_setup_time, setup_time)

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
            calibration_method = calibration_method,
            calibration_params = calibration_params,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            upsilon_fixing = upsilon_fixing,
        )
    end

    local fix1
    local l
    local c

    setup_time = @elapsed begin
        fix1 = _local_fix1_indices(keep, F1)

        l = zeros(n_red)
        c = ones(n_red)
        l[fix1] .= 1.0
    end

    _add_bnb_timing!(:node_setup_time, setup_time)

    if relaxation == :DDGFact
        local x
        local primal_obj
        local F
        local dual_sol

        bound_time = @elapsed begin
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
        end

        _add_bnb_timing!(:bound_computation_time, bound_time)

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
        )
    elseif relaxation == :DDGFactplus
        local x
        local primal_obj
        local F
        local dual_sol
        local psi_node

        bound_time = @elapsed begin
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
        end

        _add_bnb_timing!(:bound_computation_time, bound_time)

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
        )
    else
        local x
        local y
        local primal_obj
        local gamma
        local psi_node
        local F
        local dual_sol
        local calib

        bound_time = @elapsed begin
            x, y, primal_obj, gamma, psi_node, F, dual_sol, calib =
                _bound_ddgfactplus_upsilon_node(
                    Ck,
                    s,
                    t,
                    fix1,
                    l,
                    c;
                    calibration_method = calibration_method,
                    calibration_params = calibration_params,
                    atol = atol,
                )
        end

        _add_bnb_timing!(:bound_computation_time, bound_time)

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
            calibration = calib,
            upsilon_fixing = upsilon_fixing,
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
