using LinearAlgebra
using Printf

# ------------------------------------------------------------
# Directional derivative diagnostic for the calibration oracle
# ------------------------------------------------------------
function check_calibration_directional_derivative(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    g::Vector{Float64},
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
)
    obj0, _, gamma0, psi0, λmin0, x0, y0 =
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

    p = -copy(g)
    np = norm(p)

    if np == 0.0 || !all(isfinite, p)
        println("Directional derivative check skipped: zero or nonfinite gradient.")
        return
    end

    p ./= np

    predicted = dot(g, p)

    println("------------------------------------------------------------")
    println("Directional derivative check")
    println("------------------------------------------------------------")
    println("t1_reformulation = ", t1_reformulation)
    println("psi_derivative   = ", psi_derivative)
    println("obj0             = ", obj0)
    println("psi0             = ", psi0)
    println("lambda_min0      = ", λmin0)
    println("||g||            = ", norm(g))
    println("dot(g, -g/||g||) = ", predicted)

    for eps in eps_list
        obj_eps, _, _, psi_eps, λmin_eps, _, _ =
            eval_ddfactplus_upsilon_calibration(
                C,
                theta .+ eps .* p,
                s,
                t;
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = psi_derivative,
                t1_reformulation = t1_reformulation,
            )

        fd = (obj_eps - obj0) / eps

        @printf(
            "eps = %.1e | obj_eps - obj0 = %.6e | fd = %.6e | predicted = %.6e | psi_eps = %.6e | lambda_min_eps = %.6e\n",
            eps,
            obj_eps - obj0,
            fd,
            predicted,
            psi_eps,
            λmin_eps,
        )
    end

    println("------------------------------------------------------------")
end


# ------------------------------------------------------------
# Debug version of BFGS calibration of Upsilon
# ------------------------------------------------------------
function calibrate_upsilon_bfgs_ddfactplus_debug(
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
    finite_diff_check::Bool = true,
    print_trial_steps::Bool = true,
    use_steepest_descent_fallback::Bool = true,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n

    theta = 1e-4 .* randn(n)

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

    H = Matrix{Float64}(I, n, n)

    accepted_bfgs_steps = 0
    accepted_sd_fallback_steps = 0
    line_search_failures = 0
    curvature_resets = 0

    if verbose
        println("============================================================")
        println("Debug BFGS calibration")
        println("============================================================")
        println("t1_reformulation = ", t1_reformulation)
        println("psi_derivative   = ", psi_derivative)
        @printf(
            "BFGS iter %3d | obj = %.12e | psi = %.12e | lambda_min = %.12e | ||g|| = %.3e\n",
            0,
            obj,
            psi,
            λmin,
            norm(g),
        )
    end

    if finite_diff_check
        check_calibration_directional_derivative(
            C,
            theta,
            g,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
        )
    end

    for k in 1:max_iter
        ng = norm(g)

        if ng <= grad_tol
            verbose && println("Stopping: gradient norm below tolerance.")
            break
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

        verbose && @printf(
            "Iter %3d direction | dot(g,p) = %.6e | ||p|| = %.6e | ||g|| = %.6e\n",
            k,
            descent_pred,
            norm(p),
            ng,
        )

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
        # Armijo line search along BFGS direction
        # ------------------------------------------------------------
        for bt in 1:max_backtracks
            theta_candidate = theta .+ alpha .* p

            if norm(theta_candidate, Inf) > max_theta_norm
                print_trial_steps && @printf(
                    "  BFGS bt %2d | alpha = %.3e rejected: ||theta_candidate||_Inf too large\n",
                    bt,
                    alpha,
                )
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

                armijo_rhs = obj + armijo_c1 * alpha * descent_pred
                decrease = obj_cand - obj

                print_trial_steps && @printf(
                    "  BFGS bt %2d | alpha = %.3e | obj_cand = %.12e | decrease = %.6e | armijo_rhs - obj = %.6e | accepted = %s\n",
                    bt,
                    alpha,
                    obj_cand,
                    decrease,
                    armijo_rhs - obj,
                    string(obj_cand <= armijo_rhs),
                )

                if obj_cand <= armijo_rhs
                    theta_trial .= theta_candidate
                    obj_trial = obj_cand
                    g_trial .= g_cand
                    gamma_trial .= gamma_cand
                    psi_trial = psi_cand
                    λmin_trial = λmin_cand
                    x_trial .= x_cand
                    y_trial .= y_cand
                    accepted = true
                    accepted_bfgs_steps += 1
                    break
                end
            catch err
                print_trial_steps && println("  BFGS bt $bt rejected due to error: ", err)
            end

            alpha *= 0.5
            if alpha < alpha_min
                break
            end
        end

        # ------------------------------------------------------------
        # Steepest descent fallback: accept any strict improvement
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
                        print_trial_steps && @printf(
                            "  SD bt %2d | alpha = %.3e rejected: ||theta_candidate||_Inf too large\n",
                            bt,
                            alpha,
                        )
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

                        decrease = obj_cand - obj

                        print_trial_steps && @printf(
                            "  SD bt %2d | alpha = %.3e | obj_cand = %.12e | decrease = %.6e | accepted = %s\n",
                            bt,
                            alpha,
                            obj_cand,
                            decrease,
                            string(obj_cand < obj),
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
                            accepted_sd_fallback_steps += 1
                            H .= Matrix{Float64}(I, n, n)
                            break
                        end
                    catch err
                        print_trial_steps && println("  SD bt $bt rejected due to error: ", err)
                    end

                    alpha *= 0.5
                    if alpha < alpha_min
                        break
                    end
                end
            end
        end

        if !accepted
            line_search_failures += 1
            verbose && println("Stopping: line search failed.")
            break
        end

        # ------------------------------------------------------------
        # BFGS update
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
            curvature_resets += 1
            H .= Matrix{Float64}(I, n, n)

            verbose && @printf(
                "  Curvature update skipped/reset | sy = %.6e | ||s|| = %.6e | ||y|| = %.6e\n",
                sy,
                norm(s_bfgs),
                norm(y_bfgs),
            )
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
                "BFGS iter %3d | accepted = %s | obj = %.12e | best = %.12e | psi = %.12e | lambda_min = %.12e | alpha = %.2e | ||g|| = %.3e\n",
                k,
                accepted_type,
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
        accepted_bfgs_steps = accepted_bfgs_steps,
        accepted_sd_fallback_steps = accepted_sd_fallback_steps,
        line_search_failures = line_search_failures,
        curvature_resets = curvature_resets,
    )
end