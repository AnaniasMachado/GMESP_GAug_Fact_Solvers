using LinearAlgebra
using Printf


# ------------------------------------------------------------
# Evaluate either the t=1 reformulation oracle or the original oracle
# ------------------------------------------------------------
function eval_ddfactplus_upsilon_calibration_oracle(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    use_t1_oracle::Bool = true,
)
    return eval_ddfactplus_upsilon_calibration(
        C,
        theta,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = use_t1_oracle,
    )
end


# ------------------------------------------------------------
# BFGS calibration of Upsilon with psi = highest feasible value
# Debug trajectory + t1 fallback, without extra debug logs.
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
    t1_fallback::Bool = true,
    theta_perturbation::Float64 = 1e-4,
    use_steepest_descent_fallback::Bool = true,
    verbose::Bool = true,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n

    # Same solution-trajectory initialization as the debug code.
    # Set theta_perturbation = 0.0 if you want theta = zeros(n).
    theta = theta_perturbation == 0.0 ? zeros(n) : theta_perturbation .* randn(n)

    # Start with the requested oracle.
    # If true: t=1 reformulation oracle.
    # If false: original formulation oracle.
    current_use_t1_oracle = (t == 1) && t1_reformulation

    # Fallback is only relevant when we start from the t=1 oracle.
    fallback_available = (t == 1) && t1_reformulation && t1_fallback
    fallback_used = false
    final_oracle = current_use_t1_oracle ? "t1_reformulation" : "original"

    obj, g, gamma, psi, λmin, x, y =
        eval_ddfactplus_upsilon_calibration_oracle(
            C,
            theta,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            use_t1_oracle = current_use_t1_oracle,
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

    H = Matrix{Float64}(I, n, n)

    if verbose
        @printf(
            "BFGS iter %3d | oracle = %s | obj = %.12e | psi = %.12e | lambda_min = %.12e | ||g|| = %.3e\n",
            0,
            current_use_t1_oracle ? "t1" : "orig",
            obj,
            psi,
            λmin,
            norm(g),
        )
    end

    for k in 1:max_iter
        ng = norm(g)

        # ------------------------------------------------------------
        # Termination by gradient norm.
        # If the t1 oracle terminates, try original oracle once.
        # ------------------------------------------------------------
        if ng <= grad_tol
            if fallback_available && current_use_t1_oracle
                verbose && println(
                    "t=1 oracle reached grad_tol. Falling back to original formulation oracle.",
                )

                current_use_t1_oracle = false
                fallback_available = false
                fallback_used = true
                final_oracle = "original"

                # Reset BFGS memory after switching oracle.
                H .= Matrix{Float64}(I, n, n)

                obj, g, gamma, psi, λmin, x, y =
                    eval_ddfactplus_upsilon_calibration_oracle(
                        C,
                        theta,
                        s,
                        t;
                        atol = atol,
                        psi_margin = psi_margin,
                        psi_floor = psi_floor,
                        psi_derivative = psi_derivative,
                        use_t1_oracle = false,
                    )

                if obj < best_obj
                    best_obj = obj
                    best_theta = copy(theta)
                    best_gamma = copy(gamma)
                    best_psi = psi
                    best_x = copy(x)
                    best_y = copy(y)
                end

                if norm(g) <= grad_tol
                    verbose && println(
                        "Stopping: original formulation oracle also has gradient norm below tolerance.",
                    )
                    break
                end
            else
                verbose && println("Stopping: gradient norm below tolerance.")
                break
            end
        end

        # ------------------------------------------------------------
        # BFGS direction
        # ------------------------------------------------------------
        p = -H * g

        if dot(p, g) >= 0.0 || !all(isfinite, p)
            verbose && println("  H produced non-descent or nonfinite direction. Resetting H.")
            H .= Matrix{Float64}(I, n, n)
            p = -g
        end

        np = norm(p)
        if np > 0.0
            p = p / max(1.0, np)
        end

        descent_pred = dot(g, p)

        accepted = false
        accepted_by_sd_fallback = false
        alpha = alpha0

        obj_trial = NaN
        g_trial = similar(g)
        gamma_trial = similar(gamma)
        theta_trial = similar(theta)
        psi_trial = NaN
        λmin_trial = NaN
        x_trial = similar(x)
        y_trial = similar(y)

        # ------------------------------------------------------------
        # First try Armijo line search along BFGS direction.
        # ------------------------------------------------------------
        for bt in 1:max_backtracks
            theta_candidate = theta .+ alpha .* p

            if norm(theta_candidate, Inf) > max_theta_norm
                alpha *= 0.5
                if alpha < alpha_min
                    break
                end
                continue
            end

            try
                obj_cand, g_cand, gamma_cand, psi_cand, λmin_cand, x_cand, y_cand =
                    eval_ddfactplus_upsilon_calibration_oracle(
                        C,
                        theta_candidate,
                        s,
                        t;
                        atol = atol,
                        psi_margin = psi_margin,
                        psi_floor = psi_floor,
                        psi_derivative = psi_derivative,
                        use_t1_oracle = current_use_t1_oracle,
                    )

                if obj_cand <= obj + armijo_c1 * alpha * descent_pred
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
                verbose && println("  rejected BFGS trial step due to error: ", err)
            end

            alpha *= 0.5
            if alpha < alpha_min
                break
            end
        end

        # ------------------------------------------------------------
        # Debug-code behavior:
        # If BFGS line search fails, try steepest descent and accept any
        # strict improvement.
        # ------------------------------------------------------------
        if !accepted && use_steepest_descent_fallback
            verbose && println("  BFGS line search failed. Trying steepest-descent fallback.")

            p_sd = -copy(g)
            np_sd = norm(p_sd)

            if np_sd > 0.0 && all(isfinite, p_sd)
                p_sd ./= max(1.0, np_sd)

                alpha = alpha0

                for bt in 1:max_backtracks
                    theta_candidate = theta .+ alpha .* p_sd

                    if norm(theta_candidate, Inf) > max_theta_norm
                        alpha *= 0.5
                        if alpha < alpha_min
                            break
                        end
                        continue
                    end

                    try
                        obj_cand, g_cand, gamma_cand, psi_cand, λmin_cand, x_cand, y_cand =
                            eval_ddfactplus_upsilon_calibration_oracle(
                                C,
                                theta_candidate,
                                s,
                                t;
                                atol = atol,
                                psi_margin = psi_margin,
                                psi_floor = psi_floor,
                                psi_derivative = psi_derivative,
                                use_t1_oracle = current_use_t1_oracle,
                            )

                        if obj_cand < obj
                            theta_trial .= theta_candidate
                            obj_trial = obj_cand
                            g_trial .= g_cand
                            gamma_trial .= gamma_cand
                            psi_trial = psi_cand
                            λmin_trial = λmin_cand
                            x_trial .= x_cand
                            y_trial .= y_cand
                            accepted = true
                            accepted_by_sd_fallback = true

                            # Debug-code behavior: reset H after SD fallback.
                            H .= Matrix{Float64}(I, n, n)
                            break
                        end
                    catch err
                        verbose && println("  rejected SD trial step due to error: ", err)
                    end

                    alpha *= 0.5
                    if alpha < alpha_min
                        break
                    end
                end
            end
        end

        # ------------------------------------------------------------
        # t1 fallback:
        # If t1 oracle failed to improve, switch once to original oracle.
        # Then repeat the same BFGS/SD logic using original oracle.
        # ------------------------------------------------------------
        if !accepted && fallback_available && current_use_t1_oracle
            verbose && println(
                "t=1 oracle failed to improve. Falling back to original formulation oracle.",
            )

            current_use_t1_oracle = false
            fallback_available = false
            fallback_used = true
            final_oracle = "original"

            # Reset BFGS memory after switching oracle.
            H .= Matrix{Float64}(I, n, n)

            obj, g, gamma, psi, λmin, x, y =
                eval_ddfactplus_upsilon_calibration_oracle(
                    C,
                    theta,
                    s,
                    t;
                    atol = atol,
                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                    psi_derivative = psi_derivative,
                    use_t1_oracle = false,
                )

            if obj < best_obj
                best_obj = obj
                best_theta = copy(theta)
                best_gamma = copy(gamma)
                best_psi = psi
                best_x = copy(x)
                best_y = copy(y)
            end

            if norm(g) <= grad_tol
                verbose && println(
                    "Stopping: original formulation oracle has gradient norm below tolerance.",
                )
                break
            end

            # Recompute BFGS direction with the original-oracle gradient.
            p = -H * g

            if dot(p, g) >= 0.0 || !all(isfinite, p)
                verbose && println("  H produced non-descent or nonfinite direction. Resetting H.")
                H .= Matrix{Float64}(I, n, n)
                p = -g
            end

            np = norm(p)
            if np > 0.0
                p = p / max(1.0, np)
            end

            descent_pred = dot(g, p)

            accepted = false
            accepted_by_sd_fallback = false
            alpha = alpha0

            # BFGS Armijo with original oracle.
            for bt in 1:max_backtracks
                theta_candidate = theta .+ alpha .* p

                if norm(theta_candidate, Inf) > max_theta_norm
                    alpha *= 0.5
                    if alpha < alpha_min
                        break
                    end
                    continue
                end

                try
                    obj_cand, g_cand, gamma_cand, psi_cand, λmin_cand, x_cand, y_cand =
                        eval_ddfactplus_upsilon_calibration_oracle(
                            C,
                            theta_candidate,
                            s,
                            t;
                            atol = atol,
                            psi_margin = psi_margin,
                            psi_floor = psi_floor,
                            psi_derivative = psi_derivative,
                            use_t1_oracle = false,
                        )

                    if obj_cand <= obj + armijo_c1 * alpha * descent_pred
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
                    verbose && println("  rejected original-BFGS trial step due to error: ", err)
                end

                alpha *= 0.5
                if alpha < alpha_min
                    break
                end
            end

            # SD fallback with original oracle.
            if !accepted && use_steepest_descent_fallback
                verbose && println(
                    "  Original formulation BFGS line search failed. Trying steepest-descent fallback.",
                )

                p_sd = -copy(g)
                np_sd = norm(p_sd)

                if np_sd > 0.0 && all(isfinite, p_sd)
                    p_sd ./= max(1.0, np_sd)

                    alpha = alpha0

                    for bt in 1:max_backtracks
                        theta_candidate = theta .+ alpha .* p_sd

                        if norm(theta_candidate, Inf) > max_theta_norm
                            alpha *= 0.5
                            if alpha < alpha_min
                                break
                            end
                            continue
                        end

                        try
                            obj_cand, g_cand, gamma_cand, psi_cand, λmin_cand, x_cand, y_cand =
                                eval_ddfactplus_upsilon_calibration_oracle(
                                    C,
                                    theta_candidate,
                                    s,
                                    t;
                                    atol = atol,
                                    psi_margin = psi_margin,
                                    psi_floor = psi_floor,
                                    psi_derivative = psi_derivative,
                                    use_t1_oracle = false,
                                )

                            if obj_cand < obj
                                theta_trial .= theta_candidate
                                obj_trial = obj_cand
                                g_trial .= g_cand
                                gamma_trial .= gamma_cand
                                psi_trial = psi_cand
                                λmin_trial = λmin_cand
                                x_trial .= x_cand
                                y_trial .= y_cand
                                accepted = true
                                accepted_by_sd_fallback = true

                                H .= Matrix{Float64}(I, n, n)
                                break
                            end
                        catch err
                            verbose && println("  rejected original-SD trial step due to error: ", err)
                        end

                        alpha *= 0.5
                        if alpha < alpha_min
                            break
                        end
                    end
                end
            end
        end

        if !accepted
            verbose && println("Stopping: line search failed.")
            break
        end

        # ------------------------------------------------------------
        # BFGS update.
        # Debug-code H trajectory:
        # if SD fallback was accepted, skip curvature update and reset H.
        # ------------------------------------------------------------
        s_bfgs = theta_trial .- theta
        y_bfgs = g_trial .- g

        sy = dot(s_bfgs, y_bfgs)

        if !accepted_by_sd_fallback &&
           sy > curvature_tol * norm(s_bfgs) * max(norm(y_bfgs), 1.0)

            ρ = 1.0 / sy
            V = Matrix{Float64}(I, n, n) .- ρ .* (s_bfgs * y_bfgs')
            H = V * H * V' .+ ρ .* (s_bfgs * s_bfgs')
            H = Symmetric(0.5 .* (H .+ H')) |> Matrix
        else
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
            accepted_type = accepted_by_sd_fallback ? "SD" : "BFGS"
            @printf(
                "BFGS iter %3d | oracle = %s | accepted = %s | obj = %.12e | best = %.12e | psi = %.12e | lambda_min = %.12e | alpha = %.2e | ||g|| = %.3e\n",
                k,
                current_use_t1_oracle ? "t1" : "orig",
                accepted_type,
                obj,
                best_obj,
                psi,
                λmin,
                alpha,
                norm(g),
            )
        end

        # ------------------------------------------------------------
        # Step norm termination.
        # If t1 oracle terminates, try original oracle once.
        # ------------------------------------------------------------
        if norm(s_bfgs) <= step_tol
            if fallback_available && current_use_t1_oracle
                verbose && println(
                    "t=1 oracle reached step_tol. Falling back to original formulation oracle.",
                )

                current_use_t1_oracle = false
                fallback_available = false
                fallback_used = true
                final_oracle = "original"

                # Reset BFGS memory after switching oracle.
                H .= Matrix{Float64}(I, n, n)

                obj, g, gamma, psi, λmin, x, y =
                    eval_ddfactplus_upsilon_calibration_oracle(
                        C,
                        theta,
                        s,
                        t;
                        atol = atol,
                        psi_margin = psi_margin,
                        psi_floor = psi_floor,
                        psi_derivative = psi_derivative,
                        use_t1_oracle = false,
                    )

                if obj < best_obj
                    best_obj = obj
                    best_theta = copy(theta)
                    best_gamma = copy(gamma)
                    best_psi = psi
                    best_x = copy(x)
                    best_y = copy(y)
                end

                if norm(g) <= grad_tol
                    verbose && println(
                        "Stopping: original formulation oracle has gradient norm below tolerance.",
                    )
                    break
                end

                continue
            else
                verbose && println("Stopping: step norm below tolerance.")
                break
            end
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
        fallback_used = fallback_used,
        final_oracle = final_oracle,
    )
end