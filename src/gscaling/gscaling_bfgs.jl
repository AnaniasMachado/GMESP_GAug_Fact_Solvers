using LinearAlgebra
using Printf


# ------------------------------------------------------------
# BFGS objective
# ------------------------------------------------------------
function _bfgs_eval_upsilon_objective(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    x0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    y0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    use_t1_oracle::Bool = true,
    knitro_outlev::Union{Nothing,Int} = nothing,
    knitro_opttol::Union{Nothing,Float64} = nothing,
    knitro_feastol::Union{Nothing,Float64} = nothing,
)
    return eval_ddfactplus_upsilon_calibration_objective(
        C,
        theta,
        s,
        t;
        J1 = J1,
        J0 = Int[],
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = use_t1_oracle,
        x0 = x0,
        y0 = y0,
        knitro_outlev = knitro_outlev,
        knitro_opttol = knitro_opttol,
        knitro_feastol = knitro_feastol,
    )
end


# ------------------------------------------------------------
# BFGS subgradient
# ------------------------------------------------------------
function _bfgs_eval_upsilon_subgradient(
    C::Symmetric{<:Real,<:AbstractMatrix},
    val,
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
)
    return eval_ddfactplus_upsilon_calibration_subgradient(
        C,
        val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
    )
end


# ------------------------------------------------------------
# Armijo line search along a supplied direction p
# ------------------------------------------------------------
function _bfgs_armijo_line_search(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    obj::Float64,
    g::Vector{Float64},
    p::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    x0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    y0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-8,
    alpha_decay::Float64 = 0.5,
    armijo_c1::Float64 = 1e-4,
    max_backtracks::Int = 20,
    max_theta_norm::Float64 = 50.0,
    use_t1_oracle::Bool = true,
    knitro_outlev::Union{Nothing,Int} = nothing,
    knitro_opttol::Union{Nothing,Float64} = nothing,
    knitro_feastol::Union{Nothing,Float64} = nothing,
    verbose::Bool = false,
)
    descent_pred = dot(g, p)
    alpha = alpha0

    for bt in 1:max_backtracks
        theta_candidate = theta .+ alpha .* p

        if norm(theta_candidate, Inf) > max_theta_norm
            alpha *= alpha_decay
            alpha < alpha_min && break
            continue
        end

        try
            val_cand = _bfgs_eval_upsilon_objective(
                C,
                theta_candidate,
                s,
                t;
                J1 = J1,
                atol = atol,
                x0 = x0,
                y0 = y0,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                use_t1_oracle = use_t1_oracle,
                knitro_outlev = knitro_outlev,
                knitro_opttol = knitro_opttol,
                knitro_feastol = knitro_feastol,
            )

            if val_cand.obj <= obj + armijo_c1 * alpha * descent_pred
                g_cand = _bfgs_eval_upsilon_subgradient(
                    C,
                    val_cand,
                    s,
                    t;
                    atol = atol,
                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                    psi_derivative = psi_derivative,
                )

                return (
                    accepted = true,
                    alpha = alpha,
                    theta = theta_candidate,
                    obj = val_cand.obj,
                    g = g_cand,
                    gamma = val_cand.gamma,
                    psi = val_cand.psi,
                    λmin = val_cand.λmin,
                    x = val_cand.x,
                    y = val_cand.y,
                )
            end
        catch err
            verbose && println("  rejected BFGS trial step due to error: ", err)
        end

        alpha *= alpha_decay
        alpha < alpha_min && break
    end

    return (
        accepted = false,
        alpha = alpha,
        theta = theta,
        obj = obj,
        g = g,
        gamma = Float64[],
        psi = NaN,
        λmin = NaN,
        x = Float64[],
        y = Float64[],
    )
end


# ------------------------------------------------------------
# Steepest-descent fallback line search
#
# Accepts any strict improvement.
# ------------------------------------------------------------
function _bfgs_steepest_descent_line_search(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    obj::Float64,
    g::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
    x0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    y0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-8,
    alpha_decay::Float64 = 0.5,
    max_backtracks::Int = 20,
    max_theta_norm::Float64 = 50.0,
    use_t1_oracle::Bool = true,
    knitro_outlev::Union{Nothing,Int} = nothing,
    knitro_opttol::Union{Nothing,Float64} = nothing,
    knitro_feastol::Union{Nothing,Float64} = nothing,
    verbose::Bool = false,
)
    p_sd = -copy(g)
    np_sd = norm(p_sd)

    if np_sd == 0.0 || !all(isfinite, p_sd)
        return (
            accepted = false,
            alpha = alpha0,
            theta = theta,
            obj = obj,
            g = g,
            gamma = Float64[],
            psi = NaN,
            λmin = NaN,
            x = Float64[],
            y = Float64[],
        )
    end

    p_sd ./= max(1.0, np_sd)

    alpha = alpha0

    for bt in 1:max_backtracks
        theta_candidate = theta .+ alpha .* p_sd

        if norm(theta_candidate, Inf) > max_theta_norm
            alpha *= alpha_decay
            alpha < alpha_min && break
            continue
        end

        try
            val_cand = _bfgs_eval_upsilon_objective(
                C,
                theta_candidate,
                s,
                t;
                J1 = J1,
                atol = atol,
                x0 = x0,
                y0 = y0,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                use_t1_oracle = use_t1_oracle,
                knitro_outlev = knitro_outlev,
                knitro_opttol = knitro_opttol,
                knitro_feastol = knitro_feastol,
            )

            if val_cand.obj < obj
                g_cand = _bfgs_eval_upsilon_subgradient(
                    C,
                    val_cand,
                    s,
                    t;
                    atol = atol,
                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                    psi_derivative = psi_derivative,
                )

                return (
                    accepted = true,
                    alpha = alpha,
                    theta = theta_candidate,
                    obj = val_cand.obj,
                    g = g_cand,
                    gamma = val_cand.gamma,
                    psi = val_cand.psi,
                    λmin = val_cand.λmin,
                    x = val_cand.x,
                    y = val_cand.y,
                )
            end
        catch err
            verbose && println("  rejected SD trial step due to error: ", err)
        end

        alpha *= alpha_decay
        alpha < alpha_min && break
    end

    return (
        accepted = false,
        alpha = alpha,
        theta = theta,
        obj = obj,
        g = g,
        gamma = Float64[],
        psi = NaN,
        λmin = NaN,
        x = Float64[],
        y = Float64[],
    )
end


# ------------------------------------------------------------
# Shared BFGS core
# ------------------------------------------------------------
function _calibrate_upsilon_bfgs_core(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    method_name::String,
    J1::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    atol::Float64 = 1e-10,
    max_iter::Int = 20,
    grad_tol::Float64 = 1e-6,
    step_tol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-8,
    alpha_decay::Float64 = 0.5,
    armijo_c1::Float64 = 1e-4,
    curvature_tol::Float64 = 1e-10,
    max_backtracks::Int = 20,
    max_theta_norm::Float64 = 50.0,
    t1_reformulation::Bool = true,
    t1_fallback::Bool = true,
    t1_fallback_limit::Int = 1,
    theta_perturbation::Float64 = 1e-4,
    use_steepest_descent_fallback::Bool = true,
    knitro_outlev::Union{Nothing,Int} = nothing,
    knitro_opttol::Union{Nothing,Float64} = nothing,
    knitro_feastol::Union{Nothing,Float64} = nothing,
    verbose::Bool = true,
)
    n = size(C, 1)

    @assert 1 <= t <= s <= n
    @assert t1_fallback_limit >= 0

    J1 = sort(unique(collect(J1)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert length(J1) <= s

    if theta0 === nothing
        theta = theta_perturbation == 0.0 ? zeros(n) : theta_perturbation .* randn(n)
    else
        if length(theta0) != n
            error("theta0 must have length equal to size(C, 1).")
        end

        theta = copy(theta0)
    end

    x_start::Union{Nothing,Vector{Float64}} = nothing
    y_start::Union{Nothing,Vector{Float64}} = nothing

    current_use_t1_oracle = (t == 1) && t1_reformulation

    fallback_used = false
    fallback_count = 0
    final_oracle = current_use_t1_oracle ? "t1_reformulation" : "original"

    can_fallback() =
        (t == 1) &&
        t1_reformulation &&
        t1_fallback &&
        current_use_t1_oracle &&
        fallback_count < t1_fallback_limit

    val = _bfgs_eval_upsilon_objective(
        C,
        theta,
        s,
        t;
        J1 = J1,
        atol = atol,
        x0 = x_start,
        y0 = y_start,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        use_t1_oracle = current_use_t1_oracle,
        knitro_outlev = knitro_outlev,
        knitro_opttol = knitro_opttol,
        knitro_feastol = knitro_feastol,
    )

    g = _bfgs_eval_upsilon_subgradient(
        C,
        val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
    )

    obj = val.obj
    gamma = copy(val.gamma)
    psi = val.psi
    λmin = val.λmin
    x = copy(val.x)
    y = copy(val.y)

    x_start = copy(x)
    y_start = copy(y)  

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

    function update_best!()
        if obj < best_obj
            best_obj = obj
            best_theta = copy(theta)
            best_gamma = copy(gamma)
            best_psi = psi
            best_x = copy(x)
            best_y = copy(y)
        end

        return nothing
    end

    function try_one_step!(use_t1_oracle::Bool)
        p = -H * g

        if dot(p, g) >= 0.0 || !all(isfinite, p)
            verbose && println("  H produced non-descent or nonfinite direction. Resetting H.")
            H .= Matrix{Float64}(I, n, n)
            p = -copy(g)
        end

        np = norm(p)

        if np > 0.0
            p ./= max(1.0, np)
        end

        trial = _bfgs_armijo_line_search(
            C,
            theta,
            obj,
            g,
            p,
            s,
            t;
            J1 = J1,
            atol = atol,
            x0 = x_start,
            y0 = y_start,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            alpha0 = alpha0,
            alpha_min = alpha_min,
            alpha_decay = alpha_decay,
            armijo_c1 = armijo_c1,
            max_backtracks = max_backtracks,
            max_theta_norm = max_theta_norm,
            use_t1_oracle = use_t1_oracle,
            knitro_outlev = knitro_outlev,
            knitro_opttol = knitro_opttol,
            knitro_feastol = knitro_feastol,
            verbose = verbose,
        )

        accepted_by_sd_fallback = false

        if !trial.accepted && use_steepest_descent_fallback
            verbose && println("  BFGS line search failed. Trying steepest-descent fallback.")

            trial = _bfgs_steepest_descent_line_search(
                C,
                theta,
                obj,
                g,
                s,
                t;
                J1 = J1,
                atol = atol,
                x0 = x_start,
                y0 = y_start,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = psi_derivative,
                alpha0 = alpha0,
                alpha_min = alpha_min,
                alpha_decay = alpha_decay,
                max_backtracks = max_backtracks,
                max_theta_norm = max_theta_norm,
                use_t1_oracle = use_t1_oracle,
                knitro_outlev = knitro_outlev,
                knitro_opttol = knitro_opttol,
                knitro_feastol = knitro_feastol,
                verbose = verbose,
            )

            accepted_by_sd_fallback = trial.accepted

            if accepted_by_sd_fallback
                H .= Matrix{Float64}(I, n, n)
            end
        end

        if !trial.accepted
            return (
                accepted = false,
                accepted_by_sd_fallback = false,
                alpha = trial.alpha,
                step_norm = Inf,
            )
        end

        theta_trial = trial.theta
        obj_trial = trial.obj
        g_trial = trial.g
        gamma_trial = trial.gamma
        psi_trial = trial.psi
        λmin_trial = trial.λmin
        x_trial = trial.x
        y_trial = trial.y

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

        x_start = copy(x_trial)
        y_start = copy(y_trial)

        update_best!()

        return (
            accepted = true,
            accepted_by_sd_fallback = accepted_by_sd_fallback,
            alpha = trial.alpha,
            step_norm = norm(s_bfgs),
        )
    end

    function fallback_to_original!(reason::String)
        verbose && println(
            "t=1 oracle $reason. Falling back to original formulation oracle " *
            "($(fallback_count + 1)/$t1_fallback_limit).",
        )

        fallback_count += 1
        fallback_used = true

        current_use_t1_oracle = false
        final_oracle = "original"

        H .= Matrix{Float64}(I, n, n)

        val = _bfgs_eval_upsilon_objective(
            C,
            theta,
            s,
            t;
            J1 = J1,
            atol = atol,
            x0 = x_start,
            y0 = y_start,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            use_t1_oracle = false,
            knitro_outlev = knitro_outlev,
            knitro_opttol = knitro_opttol,
            knitro_feastol = knitro_feastol,
        )

        g = _bfgs_eval_upsilon_subgradient(
            C,
            val,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
        )

        obj = val.obj
        gamma = copy(val.gamma)
        psi = val.psi
        λmin = val.λmin
        x = copy(val.x)
        y = copy(val.y)

        x_start = copy(x)
        y_start = copy(y)

        update_best!()

        if norm(g) <= grad_tol
            verbose && println(
                "Stopping: original formulation oracle has gradient norm below tolerance.",
            )
            return false
        end

        result = try_one_step!(false)

        if !result.accepted
            verbose && println("Stopping: original formulation fallback failed to improve.")
            return false
        end

        current_use_t1_oracle = true
        final_oracle = "t1_reformulation"

        return true
    end

    if verbose
        @printf(
            "%s iter %3d | oracle = %s | obj = %.12e | psi = %.12e | lambda_min = %.12e | ||g|| = %.3e\n",
            method_name,
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

        if ng <= grad_tol
            if can_fallback()
                ok = fallback_to_original!("reached grad_tol")
                ok || break
                continue
            else
                verbose && println("Stopping: gradient norm below tolerance.")
                break
            end
        end

        result = try_one_step!(current_use_t1_oracle)

        if !result.accepted
            if can_fallback()
                ok = fallback_to_original!("failed to improve")
                ok || break
                continue
            else
                verbose && println("Stopping: line search failed.")
                break
            end
        end

        if verbose
            accepted_type = result.accepted_by_sd_fallback ? "SD" : "BFGS"

            @printf(
                "%s iter %3d | oracle = %s | accepted = %s | obj = %.12e | best = %.12e | psi = %.12e | lambda_min = %.12e | alpha = %.2e | ||g|| = %.3e | fallbacks = %d/%d\n",
                method_name,
                k,
                current_use_t1_oracle ? "t1" : "orig",
                accepted_type,
                obj,
                best_obj,
                psi,
                λmin,
                result.alpha,
                norm(g),
                fallback_count,
                t1_fallback_limit,
            )
        end

        if result.step_norm <= step_tol
            if can_fallback()
                ok = fallback_to_original!("reached step_tol")
                ok || break
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
        fallback_count = fallback_count,
    )
end


# ------------------------------------------------------------
# DDGFactplus_Upsilon BFGS calibration with psi = psi(gamma)
# ------------------------------------------------------------
function calibrate_upsilon_bfgs_ddfactplus(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,
    atol::Float64 = 1e-10,
    max_iter::Int = 20,
    grad_tol::Float64 = 1e-6,
    step_tol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    alpha0::Float64 = 1.0,
    alpha_min::Float64 = 1e-8,
    alpha_decay::Float64 = 0.5,
    armijo_c1::Float64 = 1e-4,
    curvature_tol::Float64 = 1e-10,
    max_backtracks::Int = 20,
    max_theta_norm::Float64 = 50.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    t1_fallback::Bool = true,
    t1_fallback_limit::Int = 1,
    theta_perturbation::Float64 = 1e-4,
    use_steepest_descent_fallback::Bool = true,
    knitro_outlev::Union{Nothing,Int} = nothing,
    knitro_opttol::Union{Nothing,Float64} = nothing,
    knitro_feastol::Union{Nothing,Float64} = nothing,
    verbose::Bool = true,
)
    return _calibrate_upsilon_bfgs_core(
        C,
        s,
        t;
        method_name = "BFGS-DDGFactplus-Upsilon",
        J1 = J1,
        theta0 = theta0,
        atol = atol,
        max_iter = max_iter,
        grad_tol = grad_tol,
        step_tol = step_tol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        alpha0 = alpha0,
        alpha_min = alpha_min,
        alpha_decay = alpha_decay,
        armijo_c1 = armijo_c1,
        curvature_tol = curvature_tol,
        max_backtracks = max_backtracks,
        max_theta_norm = max_theta_norm,
        t1_reformulation = t1_reformulation,
        t1_fallback = t1_fallback,
        t1_fallback_limit = t1_fallback_limit,
        theta_perturbation = theta_perturbation,
        use_steepest_descent_fallback = use_steepest_descent_fallback,
        knitro_outlev = knitro_outlev,
        knitro_opttol = knitro_opttol,
        knitro_feastol = knitro_feastol,
        verbose = verbose,
    )
end