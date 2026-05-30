using LinearAlgebra
using Printf


# ------------------------------------------------------------
# BFGS calibration of Upsilon with psi = highest feasible value
# ------------------------------------------------------------
function calibrate_upsilon_bfgs_ddfactplus(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,
    max_iter::Int = 20,
    grad_tol::Float64 = 1e-6,
    step_tol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-8,
    armijo_c1::Float64 = 1e-4,
    curvature_tol::Float64 = 1e-10,
    max_backtracks::Int = 20,
    max_theta_norm::Float64 = 50.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    verbose::Bool = true,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n

    # Initial values:
    # Upsilon = ones(n), psi = lambda_min(C), implemented with a margin.
    # theta = 1e-4 .* randn(n)
    theta = t1_reformulation ? 1e-4 .* randn(n) : zeros(n)

    obj, g, gamma, psi, λmin, x, y =
        eval_ddfactplus_upsilon_calibration(
            C,
            theta,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
        )

    unscaled_obj = obj
    unscaled_gamma = copy(gamma)
    unscaled_psi = psi
    unscaled_x = copy(x)
    unscaled_y = copy(y)

    best_obj = obj
    best_theta = copy(theta)
    best_gamma = copy(gamma)
    best_psi = psi
    best_x = copy(x)
    best_y = copy(y)

    # Inverse Hessian approximation.
    H = Matrix{Float64}(I, n, n)

    if verbose
        @printf(
            "BFGS iter %3d | obj = %.12e | psi = %.12e | lambda_min = %.12e | ||g|| = %.3e\n",
            0,
            obj,
            psi,
            λmin,
            norm(g),
        )
    end

    for k in 1:max_iter
        ng = norm(g)

        if ng <= grad_tol
            verbose && println("Stopping: gradient norm below tolerance.")
            break
        end

        # BFGS descent direction for minimization.
        p = -H * g

        # If H became bad, reset to steepest descent.
        if dot(p, g) >= 0.0 || !all(isfinite, p)
            H .= Matrix{Float64}(I, n, n)
            p = -g
        end

        # Avoid huge moves in log-scaling space.
        np = norm(p)
        if np > 0.0
            p = p / max(1.0, np)
        end

        accepted = false
        alpha = alpha0

        obj_trial = NaN
        g_trial = similar(g)
        gamma_trial = similar(gamma)
        theta_trial = similar(theta)
        psi_trial = NaN
        λmin_trial = NaN
        x_trial = similar(x)
        y_trial = similar(y)

        for bt in 1:max_backtracks
            theta_candidate = theta .+ alpha .* p

            # Prevent extreme exp(theta) overflow / underflow.
            if norm(theta_candidate, Inf) > max_theta_norm
                alpha *= 0.5
                if alpha < alpha_min
                    break
                end
                continue
            end

            try
                obj_cand, g_cand, gamma_cand, psi_cand, λmin_cand, x_cand, y_cand =
                    eval_ddfactplus_upsilon_calibration(
                        C,
                        theta_candidate,
                        s,
                        t;
                        atol = atol,
                        psi_margin = psi_margin,
                        psi_floor = psi_floor,
                        psi_derivative = psi_derivative,
                        t1_reformulation = t1_reformulation,
                    )

                # Armijo condition for minimizing the upper bound value.
                if obj_cand <= obj + armijo_c1 * alpha * dot(g, p)
                    theta_trial .= theta_candidate
                    obj_trial = obj_cand
                    g_trial .= g_cand
                    gamma_trial .= gamma_cand
                    psi_trial = psi_cand
                    λmin_trial = λmin_cand
                    x_trial .= x_cand
                    y_trial .= y_cand
                    accepted = true
                    break
                end
            catch err
                # Treat failed inner solves as rejected trial points.
                verbose && println("  rejected trial step due to error: ", err)
            end

            alpha *= 0.5
            if alpha < alpha_min
                break
            end
        end

        if !accepted
            verbose && println("Stopping: line search failed.")
            break
        end

        s_bfgs = theta_trial .- theta
        y_bfgs = g_trial .- g

        sy = dot(s_bfgs, y_bfgs)

        if sy > curvature_tol * norm(s_bfgs) * max(norm(y_bfgs), 1.0)
            ρ = 1.0 / sy
            V = Matrix{Float64}(I, n, n) .- ρ .* (s_bfgs * y_bfgs')
            H = V * H * V' .+ ρ .* (s_bfgs * s_bfgs')
            H = Symmetric(0.5 .* (H .+ H')) |> Matrix
        else
            # Nonconvex/nonsmooth safeguard: skip or reset.
            H .= Matrix{Float64}(I, n, n)
        end

        theta .= theta_trial
        gamma .= gamma_trial
        psi = psi_trial
        λmin = λmin_trial
        x .= x_trial
        y .= y_trial
        obj = obj_trial
        g .= g_trial

        if obj < best_obj
            best_obj = obj
            best_theta = copy(theta)
            best_gamma = copy(gamma)
            best_psi = psi
            best_x = copy(x)
            best_y = copy(y)
        end

        if verbose
            @printf(
                "BFGS iter %3d | obj = %.12e | best = %.12e | psi = %.12e | lambda_min = %.12e | alpha = %.2e | ||g|| = %.3e\n",
                k,
                obj,
                best_obj,
                psi,
                λmin,
                alpha,
                norm(g),
            )
        end

        if norm(s_bfgs) <= step_tol
            verbose && println("Stopping: step norm below tolerance.")
            break
        end
    end

    return (
        gamma = best_gamma,
        theta = best_theta,
        psi = best_psi,
        x = best_x,
        y = best_y,
        obj = best_obj,
        improved = best_obj < unscaled_obj,
        unscaled_obj = unscaled_obj,
    )
end

