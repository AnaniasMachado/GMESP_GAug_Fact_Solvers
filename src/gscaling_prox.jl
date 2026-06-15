# =============================================================================
# gscaling_prox.jl
#
# Proximal-point calibration for the original spectral DDGFact+_Upsilon
# formulation.
#
# This solves a sequence of proximal subproblems:
#
#     minimize_theta  V(theta) + (rho / 2) * ||theta - theta_center||^2
#
# where V(theta) is the original DDGFact+_Upsilon calibration value:
#
#     V(theta) = DDGFact+_Upsilon(C, s, t; gamma = exp(theta), psi(theta))
#
# Each evaluation of V(theta) and grad V(theta) calls the original spectral
# oracle:
#
#     eval_ddfactplus_upsilon_calibration(...)
#
# The proximal subproblem is solved only with Knitro.
#
# Required includes before this file:
#
#     include("gscaling_util.jl")
#
# and all files defining:
#
#     add_knitro_options!
#     aug_ddfact_upsilon_gmesp
#     ddfact_upsilon_t1_knitro
#
# =============================================================================

using LinearAlgebra
using JuMP
using KNITRO
import MathOptInterface as MOI
using Printf


# =============================================================================
# Helpers
# =============================================================================

function _prox_sym(C)
    return Symmetric(Matrix{Float64}(Matrix(C)))
end


function _prox_project_q(q::Vector{Float64}, q_bound::Float64)
    if isfinite(q_bound)
        return clamp.(q, -q_bound, q_bound)
    else
        return copy(q)
    end
end


function _prox_cache_key(q::Vector{Float64}; digits::Int = 12)
    return join(string.(round.(q; digits = digits)), ",")
end


function _prox_eval_original_oracle(
    C::Symmetric{<:Real,<:AbstractMatrix},
    q::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
)
    obj, g, gamma, psi, lambda_min, x, y =
        eval_ddfactplus_upsilon_calibration(
            C,
            q,
            s,
            t;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
        )

    return (
        obj = Float64(obj),
        g = Vector{Float64}(g),
        gamma = Vector{Float64}(gamma),
        psi = Float64(psi),
        lambda_min = Float64(lambda_min),
        x = Vector{Float64}(x),
        y = Vector{Float64}(y),
        q = copy(q),
    )
end


function _prox_eval_original_oracle_cached!(
    cache::Dict{String,Any},
    C::Symmetric{<:Real,<:AbstractMatrix},
    q::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
    cache_digits::Int,
)
    q_eval = copy(q)
    key = _prox_cache_key(q_eval; digits = cache_digits)

    if haskey(cache, key)
        return cache[key]
    end

    val = _prox_eval_original_oracle(
        C,
        q_eval,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
    )

    cache[key] = val
    return val
end


function _prox_set_knitro_attribute!(
    model::Model,
    name::String,
    value;
    verbose::Bool,
)
    try
        set_optimizer_attribute(model, name, value)
    catch err
        verbose && println("  warning: could not set Knitro option $name = $value: ", err)
    end

    return nothing
end


function _prox_knitro_status_is_acceptable(term_stat, primal_stat)
    if primal_stat == MOI.FEASIBLE_POINT
        return true
    end

    if primal_stat == MOI.NEARLY_FEASIBLE_POINT
        return true
    end

    if term_stat == MOI.LOCALLY_SOLVED
        return true
    end

    if term_stat == MOI.ALMOST_LOCALLY_SOLVED
        return true
    end

    return false
end


function _prox_projected_gradient_residual(
    theta::Vector{Float64},
    grad::Vector{Float64},
    q_bound::Float64,
)
    if isfinite(q_bound)
        projected = clamp.(theta .- grad, -q_bound, q_bound)
        residual = theta .- projected
        return norm(residual), norm(residual, Inf), residual
    else
        return norm(grad), norm(grad, Inf), copy(grad)
    end
end


function _prox_eval_prox_point!(
    cache::Dict{String,Any},
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    rho::Float64,
    q_bound::Float64,
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
    cache_digits::Int,
)
    theta_eval = _prox_project_q(theta, q_bound)

    val = _prox_eval_original_oracle_cached!(
        cache,
        C,
        theta_eval,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    prox_grad = val.g .+ rho .* (theta_eval .- theta_center)

    prox_obj =
        val.obj +
        0.5 * rho * sum((theta_eval[i] - theta_center[i])^2 for i in eachindex(theta_eval))

    res_norm, res_norm_inf, res_vec =
        _prox_projected_gradient_residual(theta_eval, prox_grad, q_bound)

    return (
        theta = theta_eval,
        val = val,
        prox_grad = prox_grad,
        prox_obj = prox_obj,
        residual = res_vec,
        residual_norm = res_norm,
        residual_norm_inf = res_norm_inf,
    )
end


# =============================================================================
# Knitro proximal subproblem
# =============================================================================

function _solve_original_upsilon_prox_subproblem_knitro!(
    cache::Dict{String,Any},
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    rho::Float64,
    q_bound::Float64,
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
    cache_digits::Int,

    # Knitro stopping tolerances.
    knitro_feastol::Float64,
    knitro_opttol::Float64,
    knitro_xtol::Float64,
    knitro_ftol::Float64,
    knitro_maxtime_real::Float64,
    knitro_algorithm::Union{Nothing,Int},
    knitro_bar_murule::Union{Nothing,Int},
    knitro_honorbnds::Union{Nothing,Int},
    knitro_outlev::Int,

    verbose::Bool,
)
    n = length(theta_center)

    model = Model(KNITRO.Optimizer)
    add_knitro_options!(model)

    if knitro_outlev <= 0
        set_silent(model)
    end

    _prox_set_knitro_attribute!(model, "feastol", knitro_feastol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "opttol", knitro_opttol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "xtol", knitro_xtol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "ftol", knitro_ftol; verbose = verbose)

    if isfinite(knitro_maxtime_real)
        _prox_set_knitro_attribute!(
            model,
            "maxtime_real",
            knitro_maxtime_real;
            verbose = verbose,
        )
    end

    if knitro_algorithm !== nothing
        _prox_set_knitro_attribute!(model, "algorithm", knitro_algorithm; verbose = verbose)
    end

    if knitro_bar_murule !== nothing
        _prox_set_knitro_attribute!(model, "bar_murule", knitro_bar_murule; verbose = verbose)
    end

    if knitro_honorbnds !== nothing
        _prox_set_knitro_attribute!(model, "honorbnds", knitro_honorbnds; verbose = verbose)
    end

    if knitro_outlev > 0
        _prox_set_knitro_attribute!(model, "outlev", knitro_outlev; verbose = verbose)
    end

    if isfinite(q_bound)
        @variable(model, -q_bound <= q[1:n] <= q_bound)
    else
        @variable(model, q[1:n])
    end

    for i in 1:n
        set_start_value(q[i], theta_center[i])
    end

    function prox_value_f(qvals...)
        qvec = collect(qvals)

        if isfinite(q_bound)
            qvec .= clamp.(qvec, -q_bound, q_bound)
        end

        val = _prox_eval_original_oracle_cached!(
            cache,
            C,
            qvec,
            s,
            t;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
            cache_digits = cache_digits,
        )

        prox_term =
            0.5 * rho * sum((qvec[i] - theta_center[i])^2 for i in 1:n)

        return val.obj + prox_term
    end

    function prox_value_grad!(gout, qvals...)
        qvec = collect(qvals)

        if isfinite(q_bound)
            qvec .= clamp.(qvec, -q_bound, q_bound)
        end

        val = _prox_eval_original_oracle_cached!(
            cache,
            C,
            qvec,
            s,
            t;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
            cache_digits = cache_digits,
        )

        for i in 1:n
            gout[i] = val.g[i] + rho * (qvec[i] - theta_center[i])
        end

        return nothing
    end

    register(
        model,
        :original_upsilon_prox_value,
        n,
        prox_value_f,
        prox_value_grad!,
    )

    @NLobjective(model, Min, original_upsilon_prox_value(q...))

    optimize!(model)

    term_stat = termination_status(model)
    primal_stat = primal_status(model)

    if primal_stat == MOI.NO_SOLUTION
        error(
            "Knitro returned no primal solution for the proximal subproblem. " *
            "termination_status = $term_stat, primal_status = $primal_stat",
        )
    end

    theta_new = value.(q)

    if isfinite(q_bound)
        theta_new .= clamp.(theta_new, -q_bound, q_bound)
    end

    eval_new = _prox_eval_prox_point!(
        cache,
        C,
        theta_new,
        theta_center,
        s,
        t;
        J1 = J1,
        J0 = J0,
        rho = rho,
        q_bound = q_bound,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    return (
        theta = eval_new.theta,
        val = eval_new.val,
        prox_obj = eval_new.prox_obj,
        status = term_stat,
        primal_status = primal_stat,
        acceptable_status = _prox_knitro_status_is_acceptable(term_stat, primal_stat),
        inner_iters = 0,
        residual_norm = eval_new.residual_norm,
        residual_norm_inf = eval_new.residual_norm_inf,
        last_inner_step_norm = NaN,
        last_inner_step_norm_inf = NaN,
    )
end


# =============================================================================
# One-step proximal calibration
# =============================================================================

function solve_one_step_proximal_knitro_upsilon_calibration(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    q0::Union{Nothing,Vector{Float64}} = nothing,

    # Proximal parameter.
    rho::Float64 = 1e-2,

    # Initialization.
    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    # Bounds.
    q_bound::Float64 = 20.0,

    # Original DDGFact+_Upsilon oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Knitro subproblem tolerances.
    knitro_feastol::Float64 = 1e-8,
    knitro_opttol::Float64 = 1e-6,
    knitro_xtol::Float64 = 1e-10,
    knitro_ftol::Float64 = 1e-12,
    knitro_maxtime_real::Float64 = Inf,
    knitro_algorithm::Union{Nothing,Int} = nothing,
    knitro_bar_murule::Union{Nothing,Int} = nothing,
    knitro_honorbnds::Union{Nothing,Int} = 1,
    knitro_outlev::Int = 0,

    # Cache / output.
    cache_digits::Int = 12,
    diagnostics::Bool = false,
    verbose::Bool = true,
)
    Csym = _prox_sym(C)
    n = size(Csym, 1)

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert all(i -> 1 <= i <= n, J0)
    @assert isempty(intersect(J1, J0))
    @assert length(J1) <= s
    @assert s <= n - length(J0)
    @assert 1 <= t <= s <= n
    @assert rho > 0.0

    theta_center = if theta0 !== nothing
        copy(theta0)
    elseif q0 !== nothing
        copy(q0)
    elseif theta_perturbation == 0.0
        zeros(n)
    else
        theta_perturbation .* randn(n)
    end

    if length(theta_center) != n
        error("theta0/q0 must have length equal to size(C, 1).")
    end

    if center_initial_theta
        theta_center .-= mean(theta_center)
    end

    theta_center = _prox_project_q(theta_center, q_bound)

    cache = Dict{String,Any}()

    unscaled_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        zeros(n),
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    initial_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_center,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    label = "OneStep-Prox-Knitro-Upsilon"

    if verbose
        @printf(
            "%s init | obj = %.12e | unscaled = %.12e | psi = %.6e | ||grad V|| = %.3e | gamma [%.4e, %.4e]\n",
            label,
            initial_val.obj,
            unscaled_val.obj,
            initial_val.psi,
            norm(initial_val.g),
            minimum(initial_val.gamma),
            maximum(initial_val.gamma),
        )
        flush(stdout)
    end

    sub = _solve_original_upsilon_prox_subproblem_knitro!(
        cache,
        Csym,
        theta_center,
        s,
        t;
        J1 = J1,
        J0 = J0,
        rho = rho,
        q_bound = q_bound,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,

        knitro_feastol = knitro_feastol,
        knitro_opttol = knitro_opttol,
        knitro_xtol = knitro_xtol,
        knitro_ftol = knitro_ftol,
        knitro_maxtime_real = knitro_maxtime_real,
        knitro_algorithm = knitro_algorithm,
        knitro_bar_murule = knitro_bar_murule,
        knitro_honorbnds = knitro_honorbnds,
        knitro_outlev = knitro_outlev,

        verbose = verbose,
    )

    final_theta = copy(sub.theta)
    final_val = sub.val

    step_vec = final_theta .- theta_center
    step_norm = norm(step_vec)
    step_norm_inf = norm(step_vec, Inf)

    final_original_grad_norm = norm(final_val.g)
    final_original_grad_norm_inf = norm(final_val.g, Inf)

    hist = diagnostics ? Any[] : nothing

    if diagnostics
        push!(
            hist,
            (
                prox_iter = 1,
                obj = final_val.obj,
                prox_obj = sub.prox_obj,
                psi = final_val.psi,
                lambda_min = final_val.lambda_min,
                original_grad_norm = final_original_grad_norm,
                original_grad_norm_inf = final_original_grad_norm_inf,
                step_norm = step_norm,
                step_norm_inf = step_norm_inf,
                gamma_min = minimum(final_val.gamma),
                gamma_max = maximum(final_val.gamma),
                theta_norm = norm(final_theta),
                theta_norm_inf = norm(final_theta, Inf),
                status = sub.status,
                primal_status = sub.primal_status,
                acceptable_status = sub.acceptable_status,
                subproblem_residual_norm = sub.residual_norm,
                subproblem_residual_norm_inf = sub.residual_norm_inf,
                cache_size = length(cache),
            ),
        )
    end

    status_history = Any[
        (
            prox_iter = 1,
            status = sub.status,
            primal_status = sub.primal_status,
            acceptable_status = sub.acceptable_status,
            prox_obj = sub.prox_obj,
            step_norm = step_norm,
            step_norm_inf = step_norm_inf,
            subproblem_residual_norm = sub.residual_norm,
            subproblem_residual_norm_inf = sub.residual_norm_inf,
        ),
    ]

    if verbose
        @printf(
            "%s final | obj = %.12e | unscaled = %.12e | improved = %s | prox_obj = %.12e | psi = %.6e | ||grad V|| = %.3e | step = %.3e | step_inf = %.3e | subres_inf = %.3e | sub_ok = %s | status = %s | gamma [%.4e, %.4e] | cache = %d\n",
            label,
            final_val.obj,
            unscaled_val.obj,
            string(final_val.obj < unscaled_val.obj),
            sub.prox_obj,
            final_val.psi,
            final_original_grad_norm,
            step_norm,
            step_norm_inf,
            sub.residual_norm_inf,
            string(sub.acceptable_status),
            string(sub.status),
            minimum(final_val.gamma),
            maximum(final_val.gamma),
            length(cache),
        )
        flush(stdout)
    end

    return (
        gamma = final_val.gamma,
        theta = final_theta,
        q = final_theta,
        psi = final_val.psi,
        lambda_min = final_val.lambda_min,
        x = final_val.x,
        y = final_val.y,
        z = final_val.obj,
        obj = final_val.obj,

        best_ub = final_val.obj,
        best_q = final_theta,
        best_gamma = final_val.gamma,
        best_psi = final_val.psi,
        best_lambda_min_S = final_val.lambda_min,
        best_min_gamma = minimum(final_val.gamma),
        best_max_gamma = maximum(final_val.gamma),

        best_eval = final_val,
        final_q = final_theta,
        final_eval = final_val,
        history = hist,
        status_history = status_history,

        initial_obj = initial_val.obj,
        initial_gamma = initial_val.gamma,
        initial_theta = theta_center,
        initial_psi = initial_val.psi,

        unscaled_obj = unscaled_val.obj,
        unscaled_gamma = unscaled_val.gamma,
        unscaled_psi = unscaled_val.psi,
        improved = final_val.obj < unscaled_val.obj,

        cache_size = length(cache),
        num_evals = length(cache),
        prox_iters = 1,
        rho = rho,
        prox_subproblem_solver = :knitro_one_step,
        stop_reason = "one_proximal_step",

        final_original_grad_norm = final_original_grad_norm,
        final_original_grad_norm_inf = final_original_grad_norm_inf,

        last_prox_obj = sub.prox_obj,
        last_prox_obj_change = missing,

        last_step_norm = step_norm,
        last_step_norm_inf = step_norm_inf,

        last_inner_iters = sub.inner_iters,
        last_subproblem_residual_norm = sub.residual_norm,
        last_subproblem_residual_norm_inf = sub.residual_norm_inf,
        last_subproblem_acceptable_status = sub.acceptable_status,

        knitro_status = sub.status,
        knitro_primal_status = sub.primal_status,
    )
end


# =============================================================================
# Public solver
# =============================================================================

function solve_proximal_knitro_upsilon_calibration(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    q0::Union{Nothing,Vector{Float64}} = nothing,

    # Proximal parameter.
    rho::Float64 = 1e-2,

    # Outer stopping tolerances.
    prox_obj_abs_tol::Float64 = 1e-10,
    prox_step_tol::Float64 = 1e-16,

    # Optional wall-time safeguard. This is not an iteration limit.
    max_wall_time::Float64 = Inf,

    # Initialization.
    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    # Bounds.
    q_bound::Float64 = 20.0,

    # Original DDGFact+_Upsilon oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Knitro subproblem tolerances.
    knitro_feastol::Float64 = 1e-8,
    knitro_opttol::Float64 = 1e-6,
    knitro_xtol::Float64 = 1e-10,
    knitro_ftol::Float64 = 1e-12,
    knitro_maxtime_real::Float64 = Inf,
    knitro_algorithm::Union{Nothing,Int} = nothing,
    knitro_bar_murule::Union{Nothing,Int} = nothing,
    knitro_honorbnds::Union{Nothing,Int} = 1,
    knitro_outlev::Int = 0,

    # Cache / output.
    cache_digits::Int = 12,
    diagnostics::Bool = false,
    verbose::Bool = true,
)
    Csym = _prox_sym(C)
    n = size(Csym, 1)

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert all(i -> 1 <= i <= n, J0)
    @assert isempty(intersect(J1, J0))
    @assert length(J1) <= s
    @assert s <= n - length(J0)
    @assert 1 <= t <= s <= n
    @assert rho > 0.0
    @assert prox_obj_abs_tol >= 0.0
    @assert prox_step_tol >= 0.0

    theta = if theta0 !== nothing
        copy(theta0)
    elseif q0 !== nothing
        copy(q0)
    elseif theta_perturbation == 0.0
        zeros(n)
    else
        theta_perturbation .* randn(n)
    end

    if length(theta) != n
        error("theta0/q0 must have length equal to size(C, 1).")
    end

    if center_initial_theta
        theta .-= mean(theta)
    end

    theta = _prox_project_q(theta, q_bound)

    cache = Dict{String,Any}()

    unscaled_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        zeros(n),
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    current_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        theta,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    best_val = current_val
    best_theta = copy(theta)

    hist = diagnostics ? Any[] : nothing
    status_history = Any[]

    t_start = time()
    prox_iter = 0
    stop_reason = ""

    previous_prox_obj = Inf

    last_prox_obj = Inf
    last_prox_obj_change = Inf
    last_step_norm = Inf
    last_step_norm_inf = Inf
    last_original_grad_norm = norm(current_val.g)
    last_original_grad_norm_inf = norm(current_val.g, Inf)
    last_inner_iters = 0
    last_subproblem_residual_norm = Inf
    last_subproblem_residual_norm_inf = Inf
    last_subproblem_acceptable_status = false

    label = "Prox-Knitro-Upsilon"

    if verbose
        @printf(
            "%s init | obj = %.12e | unscaled = %.12e | psi = %.6e | ||grad V|| = %.3e | gamma [%.4e, %.4e]\n",
            label,
            current_val.obj,
            unscaled_val.obj,
            current_val.psi,
            last_original_grad_norm,
            minimum(current_val.gamma),
            maximum(current_val.gamma),
        )
        flush(stdout)
    end

    while stop_reason == ""
        if time() - t_start >= max_wall_time
            stop_reason = "max_wall_time reached"
            break
        end

        prox_iter += 1

        theta_center = copy(theta)

        sub = _solve_original_upsilon_prox_subproblem_knitro!(
            cache,
            Csym,
            theta_center,
            s,
            t;
            J1 = J1,
            J0 = J0,
            rho = rho,
            q_bound = q_bound,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
            cache_digits = cache_digits,

            knitro_feastol = knitro_feastol,
            knitro_opttol = knitro_opttol,
            knitro_xtol = knitro_xtol,
            knitro_ftol = knitro_ftol,
            knitro_maxtime_real = knitro_maxtime_real,
            knitro_algorithm = knitro_algorithm,
            knitro_bar_murule = knitro_bar_murule,
            knitro_honorbnds = knitro_honorbnds,
            knitro_outlev = knitro_outlev,

            verbose = verbose,
        )

        theta = copy(sub.theta)
        current_val = sub.val

        if current_val.obj < best_val.obj
            best_val = current_val
            best_theta = copy(theta)
        end

        step_vec = theta .- theta_center
        step_norm = norm(step_vec)
        step_norm_inf = norm(step_vec, Inf)

        prox_obj = sub.prox_obj
        prox_obj_change =
            isfinite(previous_prox_obj) ?
            abs(prox_obj - previous_prox_obj) :
            Inf

        original_grad_norm = norm(current_val.g)
        original_grad_norm_inf = norm(current_val.g, Inf)

        last_prox_obj = prox_obj
        last_prox_obj_change = prox_obj_change
        last_step_norm = step_norm
        last_step_norm_inf = step_norm_inf
        last_original_grad_norm = original_grad_norm
        last_original_grad_norm_inf = original_grad_norm_inf
        last_inner_iters = sub.inner_iters
        last_subproblem_residual_norm = sub.residual_norm
        last_subproblem_residual_norm_inf = sub.residual_norm_inf
        last_subproblem_acceptable_status = sub.acceptable_status

        push!(
            status_history,
            (
                prox_iter = prox_iter,
                status = sub.status,
                primal_status = sub.primal_status,
                acceptable_status = sub.acceptable_status,
                prox_obj = prox_obj,
                prox_obj_change = prox_obj_change,
                step_norm = step_norm,
                step_norm_inf = step_norm_inf,
                inner_iters = last_inner_iters,
                subproblem_residual_norm = last_subproblem_residual_norm,
                subproblem_residual_norm_inf = last_subproblem_residual_norm_inf,
            ),
        )

        if diagnostics
            push!(
                hist,
                (
                    prox_iter = prox_iter,
                    obj = current_val.obj,
                    best_obj = best_val.obj,
                    prox_obj = prox_obj,
                    prox_obj_change = prox_obj_change,
                    psi = current_val.psi,
                    lambda_min = current_val.lambda_min,

                    original_grad_norm = original_grad_norm,
                    original_grad_norm_inf = original_grad_norm_inf,

                    step_norm = step_norm,
                    step_norm_inf = step_norm_inf,

                    gamma_min = minimum(current_val.gamma),
                    gamma_max = maximum(current_val.gamma),
                    theta_norm = norm(theta),
                    theta_norm_inf = norm(theta, Inf),

                    status = sub.status,
                    primal_status = sub.primal_status,
                    acceptable_status = sub.acceptable_status,
                    inner_iters = last_inner_iters,
                    subproblem_residual_norm = last_subproblem_residual_norm,
                    subproblem_residual_norm_inf = last_subproblem_residual_norm_inf,
                    cache_size = length(cache),
                ),
            )
        end

        if verbose
            @printf(
                "%s iter %3d | obj = %.12e | best = %.12e | prox_obj = %.12e | Δprox_obj = %.3e | psi = %.6e | step = %.3e | step_inf = %.3e | ||grad V|| = %.3e | subres_inf = %.3e | sub_ok = %s | status = %s | gamma [%.4e, %.4e]\n",
                label,
                prox_iter,
                current_val.obj,
                best_val.obj,
                prox_obj,
                prox_obj_change,
                current_val.psi,
                step_norm,
                step_norm_inf,
                original_grad_norm,
                last_subproblem_residual_norm_inf,
                string(sub.acceptable_status),
                string(sub.status),
                minimum(current_val.gamma),
                maximum(current_val.gamma),
            )
            flush(stdout)
        end

        if prox_iter >= 2 &&
           prox_obj_change <= prox_obj_abs_tol &&
           step_norm <= prox_step_tol
            stop_reason = "prox_obj_and_step_tol"
        end

        previous_prox_obj = prox_obj
    end

    final_val = best_val
    final_theta = best_theta

    final_original_grad_norm = norm(final_val.g)
    final_original_grad_norm_inf = norm(final_val.g, Inf)

    if verbose
        @printf(
            "%s final | obj = %.12e | unscaled = %.12e | improved = %s | psi = %.6e | ||grad V|| = %.3e | last_prox_obj = %.12e | last_Δprox_obj = %.3e | last_step = %.3e | gamma [%.4e, %.4e] | prox_iters = %d | stop = %s | cache = %d\n",
            label,
            final_val.obj,
            unscaled_val.obj,
            string(final_val.obj < unscaled_val.obj),
            final_val.psi,
            final_original_grad_norm,
            last_prox_obj,
            last_prox_obj_change,
            last_step_norm,
            minimum(final_val.gamma),
            maximum(final_val.gamma),
            prox_iter,
            stop_reason,
            length(cache),
        )
        flush(stdout)
    end

    return (
        gamma = final_val.gamma,
        theta = final_theta,
        q = final_theta,
        psi = final_val.psi,
        lambda_min = final_val.lambda_min,
        x = final_val.x,
        y = final_val.y,
        z = final_val.obj,
        obj = final_val.obj,

        best_ub = final_val.obj,
        best_q = final_theta,
        best_gamma = final_val.gamma,
        best_psi = final_val.psi,
        best_lambda_min_S = final_val.lambda_min,
        best_min_gamma = minimum(final_val.gamma),
        best_max_gamma = maximum(final_val.gamma),

        best_eval = final_val,
        final_q = theta,
        final_eval = current_val,
        history = hist,
        status_history = status_history,

        unscaled_obj = unscaled_val.obj,
        unscaled_gamma = unscaled_val.gamma,
        unscaled_psi = unscaled_val.psi,
        improved = final_val.obj < unscaled_val.obj,

        cache_size = length(cache),
        num_evals = length(cache),
        prox_iters = prox_iter,
        rho = rho,
        prox_subproblem_solver = :knitro,
        stop_reason = stop_reason,

        final_original_grad_norm = final_original_grad_norm,
        final_original_grad_norm_inf = final_original_grad_norm_inf,

        last_prox_obj = last_prox_obj,
        last_prox_obj_change = last_prox_obj_change,

        last_step_norm = last_step_norm,
        last_step_norm_inf = last_step_norm_inf,

        last_inner_iters = last_inner_iters,
        last_subproblem_residual_norm = last_subproblem_residual_norm,
        last_subproblem_residual_norm_inf = last_subproblem_residual_norm_inf,
        last_subproblem_acceptable_status = last_subproblem_acceptable_status,
    )
end