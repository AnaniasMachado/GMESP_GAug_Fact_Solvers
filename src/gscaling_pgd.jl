# ============================================================
# Penalty for violating C_gamma >= psi I
# ============================================================
function gamma_feasibility_penalty_gradient(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    psi::Float64;
    rho::Float64 = 1e4,
    margin::Float64 = 1e-8,
)
    n = size(C, 1)

    Cgamma = scaled_matrix(C, gamma)

    eig = eigen(Cgamma)
    λmin = eig.values[1]
    vmin = eig.vectors[:, 1]

    violation = psi + margin - λmin

    if violation <= 0
        return 0.0, zeros(n), λmin
    end

    penalty = rho * violation^2

    # d lambda_min / d theta_i = lambda_min * v_i^2
    grad_lambda_theta = λmin .* (vmin .^ 2)

    grad_penalty = -2.0 * rho * violation .* grad_lambda_theta

    return penalty, grad_penalty, λmin
end


# ============================================================
# Calibration of gamma using penalized unconstrained steps
# ============================================================
function calibrate_gamma_ddfactplus_upsilon_penalty(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    psi::Float64;
    atol = 1e-10,
    max_calib_iter::Int = 5,
    alpha0::Float64 = 1e-2,
    rho::Float64 = 1e4,
    margin::Float64 = 1e-8,
    tau::Float64 = 1e-8,
    verbose::Bool = true,
)
    n = size(C, 1)

    # theta = log(gamma), start from gamma = e
    theta = zeros(n)
    gamma = exp.(theta)

    x, y, obj = aug_ddfact_upsilon_gmesp(
        C,
        gamma,
        s,
        t,
        psi;
        atol = atol,
    )

    penalty, _, λmin = gamma_feasibility_penalty_gradient(
        C,
        gamma,
        psi;
        rho = rho,
        margin = margin,
    )

    penalized_obj = obj + penalty

    best_gamma = copy(gamma)
    best_x = copy(x)
    best_y = copy(y)
    best_obj = obj
    best_penalized_obj = penalized_obj

    if verbose
        println(
            "calib iter = 0, obj = $obj, penalty = $penalty, ",
            "penalized = $penalized_obj, lambda_min = $λmin"
        )
    end

    for k in 1:max_calib_iter
        # -------------------------
        # Calibration subgradient
        # -------------------------
        g_theta = theta_calibration_subgradient(
            C,
            gamma,
            x,
            y,
            psi,
            t;
            atol = atol,
        )

        penalty, g_penalty, λmin = gamma_feasibility_penalty_gradient(
            C,
            gamma,
            psi;
            rho = rho,
            margin = margin,
        )

        g_total = g_theta .+ g_penalty
        ng = norm(g_total)

        if ng <= 1e-12
            verbose && println("Stopping: small theta gradient.")
            break
        end

        # -------------------------
        # Standard decreasing step size
        # -------------------------
        alpha = alpha0 / sqrt(k)
        d = g_total / ng

        theta_trial = theta .- alpha .* d
        gamma_trial = exp.(theta_trial)

        x_trial, y_trial, obj_trial = aug_ddfact_upsilon_gmesp(
            C,
            gamma_trial,
            s,
            t,
            psi;
            atol = atol,
        )

        penalty_trial, _, λmin_trial = gamma_feasibility_penalty_gradient(
            C,
            gamma_trial,
            psi;
            rho = rho,
            margin = margin,
        )

        penalized_trial = obj_trial + penalty_trial

        if verbose
            println(
                "calib iter = $k, obj = $obj_trial, penalty = $penalty_trial, ",
                "penalized = $penalized_trial, lambda_min = $λmin_trial, alpha = $alpha"
            )
        end

        # Move to trial point
        theta = theta_trial
        gamma = gamma_trial
        x = x_trial
        y = y_trial
        obj = obj_trial
        penalized_obj = penalized_trial

        # Keep best feasible or nearly feasible bound
        if λmin_trial >= psi - tau && obj_trial < best_obj
            best_gamma = copy(gamma_trial)
            best_x = copy(x_trial)
            best_y = copy(y_trial)
            best_obj = obj_trial
            best_penalized_obj = penalized_trial
        end
    end

    return best_gamma, best_x, best_y, best_obj
end