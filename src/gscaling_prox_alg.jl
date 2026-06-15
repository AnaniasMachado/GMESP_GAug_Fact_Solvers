# ============================================================
# gscaling_prox_alg.jl
#
# Simple full-BFGS method for the proximal calibration subproblem:
#
#     min_theta  f(theta) + (rho / 2) ||theta - theta_bar||^2
#
# Uses:
#   - full inverse-Hessian BFGS approximation
#   - monotone Armijo backtracking
#   - no box constraints
#   - theta safety bound to avoid exp(theta) overflow
#
# Stopping rule:
#   stop if any of:
#     ||prox_grad|| <= grad_tol
#     ||step||      <= step_tol
#     relative proximal-objective decrease <= obj_tol
# ============================================================


# ============================================================
# Result type
# ============================================================

struct ProxCalibrationAlgResult
    theta::Vector{Float64}
    gamma::Vector{Float64}
    psi::Float64
    lambda_min::Float64

    x::Vector{Float64}
    y::Vector{Float64}

    obj::Float64
    prox_obj::Float64
    prox_grad_norm::Float64

    iterations::Int
    num_evals::Int
    best_iteration::Int
    status::Symbol
    method::Symbol

    history_obj::Vector{Float64}
    history_prox_obj::Vector{Float64}
    history_prox_grad_norm::Vector{Float64}
    history_step_norm::Vector{Float64}
    history_alpha::Vector{Float64}
end


# ============================================================
# Evaluation of proximal objective
# ============================================================

function _prox_alg_eval(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    theta_bar::Vector{Float64},
    s::Int,
    t::Int,
    rho::Float64;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
)
    obj, g, gamma, psi, lambda_min, x, y =
        eval_ddfactplus_upsilon_calibration(
            C,
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
        )

    prox_diff =
        theta .- theta_bar

    prox_obj =
        obj +
        0.5 * rho * dot(prox_diff, prox_diff)

    prox_grad =
        g .+
        rho .* prox_diff

    prox_grad_norm =
        norm(prox_grad)

    return (
        obj = obj,
        g = g,
        prox_obj = prox_obj,
        prox_grad = prox_grad,
        prox_grad_norm = prox_grad_norm,
        gamma = gamma,
        psi = psi,
        lambda_min = lambda_min,
        x = x,
        y = y,
    )
end


function _prox_alg_eval_isfinite(ev)
    return (
        isfinite(ev.obj) &&
        isfinite(ev.prox_obj) &&
        isfinite(ev.prox_grad_norm) &&
        all(isfinite, ev.prox_grad) &&
        all(isfinite, ev.gamma) &&
        all(isfinite, ev.x) &&
        all(isfinite, ev.y)
    )
end


function _prox_alg_theta_is_safe(
    theta::Vector{Float64};
    max_abs_theta::Float64 = 50.0,
)
    return all(isfinite, theta) && maximum(abs.(theta)) <= max_abs_theta
end


function _prox_alg_try_eval(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    theta_bar::Vector{Float64},
    s::Int,
    t::Int,
    rho::Float64;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    max_abs_theta::Float64 = 50.0,
)
    if !_prox_alg_theta_is_safe(theta; max_abs_theta = max_abs_theta)
        return (
            success = false,
            eval = nothing,
            status = :invalid_trial_theta,
        )
    end

    try
        ev =
            _prox_alg_eval(
                C,
                theta,
                theta_bar,
                s,
                t,
                rho;
                J1 = J1,
                J0 = J0,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = psi_derivative,
                t1_reformulation = t1_reformulation,
            )

        if !_prox_alg_eval_isfinite(ev)
            return (
                success = false,
                eval = nothing,
                status = :invalid_trial_eval,
            )
        end

        return (
            success = true,
            eval = ev,
            status = :ok,
        )
    catch
        return (
            success = false,
            eval = nothing,
            status = :eval_error,
        )
    end
end


# ============================================================
# Result constructor
# ============================================================

function _prox_alg_make_result(
    method::Symbol,
    best_theta::Vector{Float64},
    best_eval,
    best_iteration::Int,
    iterations_done::Int,
    num_evals::Int,
    status::Symbol,
    history_obj::Vector{Float64},
    history_prox_obj::Vector{Float64},
    history_prox_grad_norm::Vector{Float64},
    history_step_norm::Vector{Float64},
    history_alpha::Vector{Float64},
)
    return ProxCalibrationAlgResult(
        copy(best_theta),
        copy(best_eval.gamma),
        best_eval.psi,
        best_eval.lambda_min,
        copy(best_eval.x),
        copy(best_eval.y),
        best_eval.obj,
        best_eval.prox_obj,
        best_eval.prox_grad_norm,
        iterations_done,
        num_evals,
        best_iteration,
        status,
        method,
        history_obj,
        history_prox_obj,
        history_prox_grad_norm,
        history_step_norm,
        history_alpha,
    )
end


# ============================================================
# Armijo backtracking for proximal objective
# ============================================================

function _prox_bfgs_armijo_line_search(
    eval_trial,
    current_eval,
    theta::Vector{Float64},
    direction::Vector{Float64};
    alpha0::Float64,
    alpha_min::Float64,
    alpha_decay::Float64,
    c1::Float64,
    max_backtracks::Int,
)
    phi0 =
        current_eval.prox_obj

    dphi0 =
        dot(current_eval.prox_grad, direction)

    if !(dphi0 < 0.0)
        return (
            accepted = false,
            theta = copy(theta),
            eval = nothing,
            alpha = 0.0,
            num_evals = 0,
            status = :not_descent,
        )
    end

    alpha =
        alpha0

    num_evals =
        0

    for _ in 0:max_backtracks
        if alpha < alpha_min
            return (
                accepted = false,
                theta = copy(theta),
                eval = nothing,
                alpha = alpha,
                num_evals = num_evals,
                status = :alpha_too_small,
            )
        end

        theta_trial =
            theta .+
            alpha .* direction

        trial_result =
            eval_trial(theta_trial)

        num_evals +=
            1

        if trial_result.success
            trial_eval =
                trial_result.eval

            armijo_rhs =
                phi0 +
                c1 * alpha * dphi0

            if trial_eval.prox_obj <= armijo_rhs
                return (
                    accepted = true,
                    theta = theta_trial,
                    eval = trial_eval,
                    alpha = alpha,
                    num_evals = num_evals,
                    status = :armijo,
                )
            end
        end

        alpha *=
            alpha_decay
    end

    return (
        accepted = false,
        theta = copy(theta),
        eval = nothing,
        alpha = alpha,
        num_evals = num_evals,
        status = :max_backtracks,
    )
end


# ============================================================
# Simple full-BFGS solver
# ============================================================

function solve_proximal_calibration_alg(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    theta_bar::Vector{Float64};
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    rho::Float64 = 1e-3,

    max_iter::Int = 200,
    grad_tol::Float64 = 1e-4,
    step_tol::Float64 = 1e-4,
    obj_tol::Float64 = 1e-5,

    bfgs_alpha0::Float64 = 1.0,
    bfgs_alpha_min::Float64 = 1e-10,
    bfgs_alpha_decay::Float64 = 0.5,
    bfgs_armijo_c1::Float64 = 1e-4,
    bfgs_max_backtracks::Int = 40,
    bfgs_curvature_tol::Float64 = 1e-12,
    bfgs_reset_on_bad_curvature::Bool = false,

    max_abs_theta::Float64 = 50.0,

    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,

    verbose::Bool = true,

    ignored_kwargs...,
)
    @assert rho > 0.0
    @assert max_iter >= 0
    @assert grad_tol >= 0.0
    @assert step_tol >= 0.0
    @assert obj_tol >= 0.0
    @assert bfgs_alpha0 > 0.0
    @assert bfgs_alpha_min > 0.0
    @assert 0.0 < bfgs_alpha_decay < 1.0
    @assert 0.0 < bfgs_armijo_c1 < 1.0
    @assert bfgs_max_backtracks >= 0
    @assert bfgs_curvature_tol >= 0.0
    @assert max_abs_theta > 0.0
    @assert length(theta_bar) == size(C, 1)

    theta =
        theta0 === nothing ?
        copy(theta_bar) :
        copy(theta0)

    @assert length(theta) == length(theta_bar)

    n =
        length(theta)

    function eval_trial(theta_trial::Vector{Float64})
        return _prox_alg_try_eval(
            C,
            theta_trial,
            theta_bar,
            s,
            t,
            rho;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
            max_abs_theta = max_abs_theta,
        )
    end

    initial_result =
        eval_trial(theta)

    if !initial_result.success
        error("Initial theta is invalid: $(initial_result.status)")
    end

    current =
        initial_result.eval

    num_evals =
        1

    best_theta =
        copy(theta)

    best_eval =
        current

    best_iteration =
        0

    history_obj =
        Float64[]

    history_prox_obj =
        Float64[]

    history_prox_grad_norm =
        Float64[]

    history_step_norm =
        Float64[]

    history_alpha =
        Float64[]

    push!(history_obj, current.obj)
    push!(history_prox_obj, current.prox_obj)
    push!(history_prox_grad_norm, current.prox_grad_norm)
    push!(history_step_norm, 0.0)
    push!(history_alpha, 0.0)

    status =
        :max_iter

    iterations_done =
        0

    Hinv =
        Matrix{Float64}(I, n, n)

    alpha_start =
        bfgs_alpha0

    if verbose
        println(
            "BFGS iter = 0",
            " | obj = ", current.obj,
            " | prox_obj = ", current.prox_obj,
            " | prox_grad_norm = ", current.prox_grad_norm,
        )
        flush(stdout)
    end

    if current.prox_grad_norm <= grad_tol
        status =
            :first_order

        return _prox_alg_make_result(
            :bfgs,
            best_theta,
            best_eval,
            best_iteration,
            iterations_done,
            num_evals,
            status,
            history_obj,
            history_prox_obj,
            history_prox_grad_norm,
            history_step_norm,
            history_alpha,
        )
    end

    for k in 1:max_iter
        iterations_done =
            k

        grad_old =
            copy(current.prox_grad)

        prox_obj_old =
            current.prox_obj

        direction =
            -(Hinv * grad_old)

        if dot(direction, grad_old) >= 0.0 || !all(isfinite, direction)
            Hinv =
                Matrix{Float64}(I, n, n)

            direction =
                -grad_old
        end

        ls_result =
            _prox_bfgs_armijo_line_search(
                eval_trial,
                current,
                theta,
                direction;
                alpha0 = alpha_start,
                alpha_min = bfgs_alpha_min,
                alpha_decay = bfgs_alpha_decay,
                c1 = bfgs_armijo_c1,
                max_backtracks = bfgs_max_backtracks,
            )

        num_evals +=
            ls_result.num_evals

        if !ls_result.accepted
            status =
                ls_result.status

            if verbose
                println(
                    "BFGS stopped at iter = ", k,
                    " | status = ", status,
                    " | prox_obj = ", current.prox_obj,
                    " | prox_grad_norm = ", current.prox_grad_norm,
                )
                flush(stdout)
            end

            break
        end

        theta_old =
            copy(theta)

        theta =
            ls_result.theta

        current =
            ls_result.eval

        svec =
            theta .- theta_old

        yvec =
            current.prox_grad .- grad_old

        step_norm =
            norm(svec)

        prox_obj_change =
            abs(prox_obj_old - current.prox_obj)

        prox_obj_tol_threshold =
            obj_tol * max(1.0, abs(prox_obj_old))

        push!(history_obj, current.obj)
        push!(history_prox_obj, current.prox_obj)
        push!(history_prox_grad_norm, current.prox_grad_norm)
        push!(history_step_norm, step_norm)
        push!(history_alpha, ls_result.alpha)

        if current.prox_obj < best_eval.prox_obj
            best_theta =
                copy(theta)

            best_eval =
                current

            best_iteration =
                k
        end

        sy =
            dot(svec, yvec)

        if sy > bfgs_curvature_tol * max(1.0, norm(svec) * norm(yvec))
            rho_bfgs =
                1.0 / sy

            V =
                Matrix{Float64}(I, n, n) .-
                rho_bfgs .* (svec * yvec')

            Hinv =
                V * Hinv * V' .+
                rho_bfgs .* (svec * svec')

            Hinv =
                0.5 .* (Hinv .+ Hinv')
        else
            if bfgs_reset_on_bad_curvature
                Hinv =
                    Matrix{Float64}(I, n, n)
            end
        end

        if verbose
            println(
                "BFGS iter = ", k,
                " | obj = ", current.obj,
                " | prox_obj = ", current.prox_obj,
                " | best_prox_obj = ", best_eval.prox_obj,
                " | prox_grad_norm = ", current.prox_grad_norm,
                " | prox_obj_change = ", prox_obj_change,
                " | alpha = ", ls_result.alpha,
                " | step_norm = ", step_norm,
                " | sy = ", sy,
            )
            flush(stdout)
        end

        if current.prox_grad_norm <= grad_tol
            status =
                :first_order

            break
        end

        if step_norm <= step_tol
            status =
                :small_step

            break
        end

        if prox_obj_change <= prox_obj_tol_threshold
            status =
                :small_obj_change

            break
        end

        alpha_start =
            min(bfgs_alpha0, ls_result.alpha / bfgs_alpha_decay)
    end

    return _prox_alg_make_result(
        :bfgs,
        best_theta,
        best_eval,
        best_iteration,
        iterations_done,
        num_evals,
        status,
        history_obj,
        history_prox_obj,
        history_prox_grad_norm,
        history_step_norm,
        history_alpha,
    )
end