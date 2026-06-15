# =============================================================================
# gscaling_rbfgs.jl
#
# Dense regularized BFGS method for DDGFact+_Upsilon calibration.
#
# The method optimizes over q = log.(gamma), so gamma = exp.(q) is always
# positive. At each iteration it:
#
#   1. Evaluates the calibration oracle at q.
#   2. Maintains a dense Hessian approximation B.
#   3. Computes a regularized quasi-Newton direction:
#
#          (B + mu I) d = -g.
#
#   4. Accepts the step using Armijo backtracking, optionally nonmonotone.
#   5. Updates B with Powell-damped BFGS.
#   6. Updates mu using the ratio between actual and model reduction.
#
# Public entry point:
#
#     solve_regularized_bfgs_upsilon_calibration(C, s, t; ...)
#
# =============================================================================

using LinearAlgebra
using Printf


# =============================================================================
# Basic helpers
# =============================================================================

function _rbfgs_sym(C)
    return Symmetric(Matrix{Float64}(Matrix(C)))
end


function _rbfgs_project_q(q::Vector{Float64}, q_bound::Float64)
    return isfinite(q_bound) ? clamp.(q, -q_bound, q_bound) : q
end


function _rbfgs_cache_key(q::Vector{Float64}; digits::Int = 12)
    return join(string.(round.(q; digits = digits)), ",")
end


function _rbfgs_eval_oracle(
    C::Symmetric{<:Real,<:AbstractMatrix},
    q::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
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
            J0 = Int[],
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
        psi = psi,
        lambda_min = lambda_min,
        x = x,
        y = y,
        q = copy(q),
    )
end


function _rbfgs_eval_oracle_cached!(
    cache::Dict{String,Any},
    C::Symmetric{<:Real,<:AbstractMatrix},
    q::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
    cache_digits::Int,
)
    key = _rbfgs_cache_key(q; digits = cache_digits)

    if haskey(cache, key)
        return cache[key]
    end

    val = _rbfgs_eval_oracle(
        C,
        q,
        s,
        t;
        J1 = J1,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
    )

    cache[key] = val
    return val
end


# =============================================================================
# Matrix utilities
# =============================================================================

function _rbfgs_initial_B(
    n::Int;
    B0_scale::Float64,
    min_B_eig::Float64,
    max_B_eig::Float64,
)
    scale = isfinite(B0_scale) && B0_scale > 0.0 ? B0_scale : 1.0
    scale = clamp(scale, min_B_eig, max_B_eig)

    return scale .* Matrix{Float64}(I, n, n)
end


function _rbfgs_project_spd(
    B::Matrix{Float64};
    min_B_eig::Float64,
    max_B_eig::Float64,
)
    Bsym = Symmetric(0.5 .* (B .+ B'))
    F = eigen(Bsym)

    vals = clamp.(F.values, min_B_eig, max_B_eig)

    Bproj = F.vectors * Diagonal(vals) * F.vectors'
    return Matrix{Float64}(Symmetric(0.5 .* (Bproj .+ Bproj')))
end


function _rbfgs_regularized_direction(
    B::Matrix{Float64},
    g::Vector{Float64},
    mu::Float64,
)
    n = length(g)
    Breg = Symmetric(0.5 .* (B .+ B') .+ mu .* Matrix{Float64}(I, n, n))

    d = try
        -(Breg \ g)
    catch
        fill(NaN, n)
    end

    return d, Breg
end


function _rbfgs_model_reduction(
    Breg::Symmetric{Float64,Matrix{Float64}},
    g::Vector{Float64},
    step::Vector{Float64},
)
    return -dot(g, step) - 0.5 * dot(step, Breg * step)
end


function _rbfgs_clip_direction!(
    d::Vector{Float64};
    max_direction_norm::Float64,
)
    if isfinite(max_direction_norm)
        nd = norm(d)

        if nd > max_direction_norm && nd > 0.0
            d .*= max_direction_norm / nd
        end
    end

    return d
end


function _rbfgs_normalize_direction!(
    d::Vector{Float64};
    normalize_direction::Bool,
)
    if normalize_direction
        nd = norm(d)

        if nd > 0.0
            d ./= max(1.0, nd)
        end
    end

    return d
end


function _rbfgs_repair_B(
    B::Matrix{Float64},
    n::Int;
    project_spd::Bool,
    min_B_eig::Float64,
    max_B_eig::Float64,
    reset_B_on_bad::Bool,
    B0_scale::Float64,
    max_B_norm::Float64,
)
    Bnorm = try
        opnorm(B)
    catch
        Inf
    end

    bad = !isfinite(Bnorm) || Bnorm > max_B_norm || any(!isfinite, B)

    if bad && reset_B_on_bad
        return _rbfgs_initial_B(
            n;
            B0_scale = B0_scale,
            min_B_eig = min_B_eig,
            max_B_eig = max_B_eig,
        ), true
    end

    if project_spd || bad
        return _rbfgs_project_spd(
            B;
            min_B_eig = min_B_eig,
            max_B_eig = max_B_eig,
        ), bad
    end

    return Matrix{Float64}(Symmetric(0.5 .* (B .+ B'))), bad
end


function _rbfgs_damped_bfgs_update(
    B::Matrix{Float64},
    svec::Vector{Float64},
    yvec::Vector{Float64};
    damping_delta::Float64,
    curvature_tol::Float64,
)
    Bs = B * svec
    sBs = dot(svec, Bs)
    sy = dot(svec, yvec)

    if !isfinite(sBs) || sBs <= curvature_tol * max(1.0, norm(svec)^2)
        return B, false, sy, sBs, 0.0
    end

    # Powell damping: enforce s' ybar >= damping_delta * s' B s.
    theta = 1.0

    if sy < damping_delta * sBs
        denom = sBs - sy

        if abs(denom) <= eps(Float64) * max(1.0, abs(sBs), abs(sy))
            theta = 0.0
        else
            theta = (1.0 - damping_delta) * sBs / denom
            theta = clamp(theta, 0.0, 1.0)
        end
    end

    ybar = theta .* yvec .+ (1.0 - theta) .* Bs
    sybar = dot(svec, ybar)

    if !isfinite(sybar) || sybar <= curvature_tol * norm(svec) * max(1.0, norm(ybar))
        return B, false, sybar, sBs, theta
    end

    Bnew =
        B .-
        (Bs * Bs') ./ sBs .+
        (ybar * ybar') ./ sybar

    Bnew = Matrix{Float64}(Symmetric(0.5 .* (Bnew .+ Bnew')))

    return Bnew, true, sybar, sBs, theta
end


function _rbfgs_spectral_diagnostics(B::Matrix{Float64}, mu::Float64)
    Bvals = eigvals(Symmetric(0.5 .* (B .+ B')))
    Hvals = 1.0 ./ max.(Bvals .+ mu, eps(Float64))

    return (
        B_norm = opnorm(B),
        B_min_eig = minimum(Bvals),
        B_max_eig = maximum(Bvals),
        H_norm = maximum(Hvals),
        H_min_eig = minimum(Hvals),
        H_max_eig = maximum(Hvals),
    )
end


function _rbfgs_no_spectral_diagnostics(B::Matrix{Float64}, mu::Float64)
    return (
        B_norm = opnorm(B),
        B_min_eig = NaN,
        B_max_eig = NaN,
        H_norm = NaN,
        H_min_eig = NaN,
        H_max_eig = NaN,
    )
end


# =============================================================================
# Main solver
# =============================================================================

function solve_regularized_bfgs_upsilon_calibration(
    C,
    s::Int,
    t::Int;
    q0::Union{Nothing,Vector{Float64}} = nothing,

    # Node information.
    J1::AbstractVector{<:Integer} = Int[],

    # Oracle options.
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = false,

    # Iterate bounds.
    q_bound::Float64 = Inf,
    max_q_norm_inf::Float64 = 20.0,

    # Iteration limit.
    max_iter::Int = 50,

    # Initial Hessian approximation.
    B0_scale::Float64 = 1.0,

    # Regularization parameter in (B + mu I)d = -g.
    mu0::Float64 = 1e-2,
    mu_min::Float64 = 1e-10,
    mu_max::Float64 = 1e8,
    mu_decrease::Float64 = 0.2,
    mu_increase::Float64 = 5.0,
    eta1::Float64 = 0.05,
    eta2::Float64 = 0.75,
    max_inner_regularization::Int = 20,

    # Direction safeguards.
    normalize_direction::Bool = false,
    max_direction_norm::Float64 = 10.0,

    # Armijo line search.
    armijo_c1::Float64 = 1e-4,
    accept_tol::Float64 = 1e-12,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-12,
    alpha_decay::Float64 = 0.5,
    max_backtracks::Int = 30,

    # Nonmonotone Armijo reference.
    nonmonotone::Bool = true,
    nonmonotone_window::Int = 10,

    # BFGS update.
    curvature_tol::Float64 = 1e-12,
    damping_delta::Float64 = 0.2,
    reset_B_on_failed_update::Bool = false,

    # Hessian safeguards.
    project_spd::Bool = true,
    min_B_eig::Float64 = 1e-8,
    max_B_eig::Float64 = 1e8,
    reset_B_on_bad::Bool = true,
    max_B_norm::Float64 = 1e8,

    # Stopping.
    grad_tol::Float64 = 1e-8,
    step_tol::Float64 = 1e-10,

    # Cache / output.
    cache_digits::Int = 12,
    diagnostics::Bool = false,
    verbose::Bool = true,
)
    Csym = _rbfgs_sym(C)
    n = size(Csym, 1)

    q = q0 === nothing ? zeros(n) : copy(q0)
    q = _rbfgs_project_q(q, q_bound)

    cache = Dict{String,Any}()

    current = _rbfgs_eval_oracle_cached!(
        cache,
        Csym,
        q,
        s,
        t;
        J1 = J1,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    eval_count = 1

    best = current
    best_q = copy(q)

    mu = clamp(mu0, mu_min, mu_max)

    B = _rbfgs_initial_B(
        n;
        B0_scale = B0_scale,
        min_B_eig = min_B_eig,
        max_B_eig = max_B_eig,
    )

    B, _ = _rbfgs_repair_B(
        B,
        n;
        project_spd = project_spd,
        min_B_eig = min_B_eig,
        max_B_eig = max_B_eig,
        reset_B_on_bad = reset_B_on_bad,
        B0_scale = B0_scale,
        max_B_norm = max_B_norm,
    )

    obj_history = Float64[current.obj]
    history = diagnostics ? Any[] : nothing

    if diagnostics
        sd = _rbfgs_spectral_diagnostics(B, mu)

        push!(
            history,
            (
                iter = 0,
                obj = current.obj,
                best_obj = best.obj,
                psi = current.psi,
                lambda_min_scaled = current.lambda_min,
                grad_norm = norm(current.g),
                mu = mu,
                ratio = NaN,
                actual_reduction = 0.0,
                model_reduction = 0.0,
                alpha = 0.0,
                step_norm = 0.0,
                direction_norm = 0.0,
                directional_derivative = 0.0,
                accepted = true,
                inner_regularization_steps = 0,
                backtracks = 0,
                eval_count = eval_count,
                gamma_min = minimum(current.gamma),
                gamma_max = maximum(current.gamma),
                q_norm = norm(q),
                q_norm_inf = norm(q, Inf),
                B_updated = false,
                B_bad = false,
                damping_theta = NaN,
                sybar = NaN,
                sBs = NaN,
                B_norm = sd.B_norm,
                B_min_eig = sd.B_min_eig,
                B_max_eig = sd.B_max_eig,
                H_norm = sd.H_norm,
                H_min_eig = sd.H_min_eig,
                H_max_eig = sd.H_max_eig,
            ),
        )
    end

    if verbose
        @printf(
            "R-BFGS init | obj = %.12e | psi = %.6e | ||g|| = %.3e | gamma [%.4e, %.4e] | mu = %.2e | ||q||_inf = %.3e\n",
            current.obj,
            current.psi,
            norm(current.g),
            minimum(current.gamma),
            maximum(current.gamma),
            mu,
            norm(q, Inf),
        )
    end

    for k in 1:max_iter
        g = current.g
        gnorm = norm(g)

        if gnorm <= grad_tol
            verbose && println("  stopping: gradient norm below tolerance")
            break
        end

        ref_obj = if nonmonotone
            window_start = max(1, length(obj_history) - nonmonotone_window + 1)
            maximum(obj_history[window_start:end])
        else
            current.obj
        end

        accepted = false
        mu_trial = mu

        q_new = copy(q)
        val_new = current
        step_new = zeros(n)
        d_new = zeros(n)

        alpha_new = 0.0
        ratio_new = -Inf
        actual_reduction_new = -Inf
        model_reduction_new = -Inf
        inner_used = 0
        bt_used = 0

        for inner in 0:max_inner_regularization
            inner_used = inner

            d, Breg = _rbfgs_regularized_direction(B, g, mu_trial)

            if any(!isfinite, d) || dot(g, d) >= 0.0
                mu_trial = min(mu_max, mu_increase * mu_trial)

                if mu_trial >= mu_max
                    break
                end

                continue
            end

            _rbfgs_clip_direction!(d; max_direction_norm = max_direction_norm)
            _rbfgs_normalize_direction!(d; normalize_direction = normalize_direction)

            gtd = dot(g, d)

            if !isfinite(gtd) || gtd >= 0.0
                mu_trial = min(mu_max, mu_increase * mu_trial)

                if mu_trial >= mu_max
                    break
                end

                continue
            end

            alpha = alpha0

            for bt in 0:max_backtracks
                bt_used = bt

                q_trial = q .+ alpha .* d
                q_trial = _rbfgs_project_q(q_trial, q_bound)

                if isfinite(max_q_norm_inf) && norm(q_trial, Inf) > max_q_norm_inf
                    alpha *= alpha_decay

                    if alpha < alpha_min
                        break
                    end

                    continue
                end

                step = q_trial .- q
                step_norm = norm(step)

                if step_norm <= step_tol
                    alpha *= alpha_decay

                    if alpha < alpha_min
                        break
                    end

                    continue
                end

                model_reduction = _rbfgs_model_reduction(Breg, g, step)

                if !isfinite(model_reduction) || model_reduction <= 0.0
                    alpha *= alpha_decay

                    if alpha < alpha_min
                        break
                    end

                    continue
                end

                val_trial = _rbfgs_eval_oracle_cached!(
                    cache,
                    Csym,
                    q_trial,
                    s,
                    t;
                    J1 = J1,
                    atol = atol,
                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                    psi_derivative = psi_derivative,
                    t1_reformulation = t1_reformulation,
                    cache_digits = cache_digits,
                )

                eval_count += 1

                actual_reduction = ref_obj - val_trial.obj
                ratio = actual_reduction / model_reduction

                armijo_rhs = ref_obj + armijo_c1 * alpha * gtd + accept_tol
                armijo_ok = val_trial.obj <= armijo_rhs

                if armijo_ok
                    accepted = true

                    q_new = q_trial
                    val_new = val_trial
                    step_new = step
                    d_new = d

                    alpha_new = alpha
                    ratio_new = ratio
                    actual_reduction_new = actual_reduction
                    model_reduction_new = model_reduction

                    break
                end

                alpha *= alpha_decay

                if alpha < alpha_min
                    break
                end
            end

            if accepted
                break
            end

            mu_trial = min(mu_max, mu_increase * mu_trial)

            if mu_trial >= mu_max
                break
            end
        end

        if !accepted
            mu = min(mu_max, mu_increase * mu)

            if verbose
                @printf(
                    "  iter %3d | rejected | obj = %.12e | best = %.12e | mu -> %.2e | evals = %d\n",
                    k,
                    current.obj,
                    best.obj,
                    mu,
                    eval_count,
                )
            end

            if mu >= mu_max
                verbose && println("  stopping: mu reached mu_max")
                break
            end

            continue
        end

        q_old = copy(q)
        current_old = current

        q = q_new
        current = val_new
        push!(obj_history, current.obj)

        svec = q .- q_old
        yvec = current.g .- current_old.g

        B_update, B_updated, sybar, sBs, theta_damp =
            _rbfgs_damped_bfgs_update(
                B,
                svec,
                yvec;
                damping_delta = damping_delta,
                curvature_tol = curvature_tol,
            )

        if B_updated
            B = B_update
        elseif reset_B_on_failed_update
            B = _rbfgs_initial_B(
                n;
                B0_scale = B0_scale,
                min_B_eig = min_B_eig,
                max_B_eig = max_B_eig,
            )
        end

        B, B_bad = _rbfgs_repair_B(
            B,
            n;
            project_spd = project_spd,
            min_B_eig = min_B_eig,
            max_B_eig = max_B_eig,
            reset_B_on_bad = reset_B_on_bad,
            B0_scale = B0_scale,
            max_B_norm = max_B_norm,
        )

        if ratio_new >= eta2
            mu = max(mu_min, mu_decrease * mu_trial)
        elseif ratio_new >= eta1
            mu = mu_trial
        else
            mu = min(mu_max, mu_increase * mu_trial)
        end

        if current.obj < best.obj - accept_tol
            best = current
            best_q = copy(q)
        end

        step_norm = norm(svec)
        gtd_step = dot(current_old.g, step_new)

        if diagnostics
            sd = _rbfgs_spectral_diagnostics(B, mu)

            push!(
                history,
                (
                    iter = k,
                    obj = current.obj,
                    best_obj = best.obj,
                    psi = current.psi,
                    lambda_min_scaled = current.lambda_min,
                    grad_norm = norm(current.g),
                    mu = mu,
                    mu_used = mu_trial,
                    ratio = ratio_new,
                    actual_reduction = actual_reduction_new,
                    model_reduction = model_reduction_new,
                    alpha = alpha_new,
                    step_norm = step_norm,
                    direction_norm = norm(d_new),
                    directional_derivative = gtd_step,
                    accepted = true,
                    inner_regularization_steps = inner_used,
                    backtracks = bt_used,
                    eval_count = eval_count,
                    gamma_min = minimum(current.gamma),
                    gamma_max = maximum(current.gamma),
                    q_norm = norm(q),
                    q_norm_inf = norm(q, Inf),
                    B_updated = B_updated,
                    B_bad = B_bad,
                    damping_theta = theta_damp,
                    sybar = sybar,
                    sBs = sBs,
                    B_norm = sd.B_norm,
                    B_min_eig = sd.B_min_eig,
                    B_max_eig = sd.B_max_eig,
                    H_norm = sd.H_norm,
                    H_min_eig = sd.H_min_eig,
                    H_max_eig = sd.H_max_eig,
                ),
            )
        end

        if verbose
            @printf(
                "  iter %3d | obj = %.12e | best = %.12e | alpha = %.2e | ratio = %.3e | mu_used = %.2e | mu_next = %.2e | step = %.3e | ||g|| = %.3e | gamma [%.4e, %.4e] | Bupd = %s | evals = %d\n",
                k,
                current.obj,
                best.obj,
                alpha_new,
                ratio_new,
                mu_trial,
                mu,
                step_norm,
                norm(current.g),
                minimum(current.gamma),
                maximum(current.gamma),
                string(B_updated),
                eval_count,
            )
        end

        if step_norm <= step_tol
            verbose && println("  stopping: accepted step norm below tolerance")
            break
        end
    end

    final_q = copy(best_q)

    final = _rbfgs_eval_oracle_cached!(
        cache,
        Csym,
        final_q,
        s,
        t;
        J1 = J1,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
    )

    sd_final =
        diagnostics ?
        _rbfgs_spectral_diagnostics(B, mu) :
        _rbfgs_no_spectral_diagnostics(B, mu)

    return (
        gamma = final.gamma,
        theta = final_q,
        q = final_q,
        psi = final.psi,
        lambda_min = final.lambda_min,
        x = final.x,
        y = final.y,
        obj = final.obj,

        best_ub = final.obj,
        best_q = final_q,
        best_gamma = final.gamma,
        best_psi = final.psi,
        best_lambda_min_S = final.lambda_min,
        best_min_gamma = minimum(final.gamma),
        best_max_gamma = maximum(final.gamma),

        best_eval = (
            obj = final.obj,
            g = final.g,
            gamma = final.gamma,
            psi = final.psi,
            lambda_min = final.lambda_min,
            x = final.x,
            y = final.y,
        ),

        final_q = final_q,
        history = history,
        num_evals = eval_count,
        cache_size = length(cache),

        max_iter = max_iter,
        final_mu = mu,

        final_B_norm = sd_final.B_norm,
        final_B_min_eig = sd_final.B_min_eig,
        final_B_max_eig = sd_final.B_max_eig,

        final_H_norm = sd_final.H_norm,
        final_H_min_eig = sd_final.H_min_eig,
        final_H_max_eig = sd_final.H_max_eig,
    )
end