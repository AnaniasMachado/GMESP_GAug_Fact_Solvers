using LinearAlgebra
using Printf
using LBFGSB

eps_lbfgsb = 1e-6


# ------------------------------------------------------------
# L-BFGS-B calibration of Upsilon
# ------------------------------------------------------------
function calibrate_upsilon_lbfgsb_ddfactplus(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,

    # Bounds on Upsilon = gamma
    gamma_lower::Float64 = 1e-6,
    gamma_upper::Float64 = 1e6,

    # psi(gamma) = lambda_min(C_gamma) - psi_margin
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,

    # L-BFGS-B parameters
    lbfgsb_m::Int = 10,
    factr::Float64 = 1e7,
    pgtol::Float64 = 1e-2,
    iprint::Int = -1,
    maxfun::Int = 15_000,
    maxiter::Int = 200,

    # Evaluation parameters
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,

    # Output
    verbose::Bool = true,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n
    @assert gamma_lower > 0.0
    @assert gamma_upper > gamma_lower

    theta_lower = log(gamma_lower)
    theta_upper = log(gamma_upper)

    lb = fill(theta_lower, n)
    ub = fill(theta_upper, n)

    theta_init = 1e-4 .* randn(n)

    # Project the initial point into the box, just in case.
    theta_init .= clamp.(theta_init, theta_lower, theta_upper)

    # Store best point seen by objective/gradient evaluations.
    # This is useful because LBFGSB may stop with an abnormal line-search flag
    # or return a point that is not better than an earlier evaluated point.
    best_obj = Ref(Inf)
    best_theta = Ref(copy(theta_init))
    best_gamma = Ref(exp.(theta_init))
    best_psi = Ref(NaN)
    best_λmin = Ref(NaN)
    best_x = Ref(zeros(n))
    best_y = Ref(zeros(n))
    eval_count = Ref(0)

    last_obj = Ref(Inf)
    last_g = Ref(zeros(n))
    last_theta = Ref(copy(theta_init))

    function f(theta::Vector{Float64})
        eval_count[] += 1

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

        last_obj[] = obj
        last_g[] .= g
        last_theta[] .= theta

        if obj < best_obj[] - eps_lbfgsb
            best_obj[] = obj
            best_theta[] = copy(theta)
            best_gamma[] = copy(gamma)
            best_psi[] = psi
            best_λmin[] = λmin
            best_x[] = copy(x)
            best_y[] = copy(y)
        end

        if verbose
            @printf(
                "LBFGSB eval %5d | obj = %.12e | best = %.12e | psi = %.12e | lambda_min = %.12e | ||g|| = %.3e\n",
                eval_count[],
                obj,
                best_obj[],
                psi,
                λmin,
                norm(g),
            )
        end

        return obj
    end

    function g!(z::Vector{Float64}, theta::Vector{Float64})
        # We recompute the objective/gradient here instead of trying to reuse
        # f(theta), because LBFGSB may call f and g! independently.
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

        z .= g

        last_obj[] = obj
        last_g[] .= g
        last_theta[] .= theta

        if obj < best_obj[] - eps_lbfgsb
            best_obj[] = obj
            best_theta[] = copy(theta)
            best_gamma[] = copy(gamma)
            best_psi[] = psi
            best_λmin[] = λmin
            best_x[] = copy(x)
            best_y[] = copy(y)
        end

        return nothing
    end

    if verbose
        println("Starting L-BFGS-B calibration")
        println("theta bounds: [$theta_lower, $theta_upper]")
        println("gamma bounds: [$gamma_lower, $gamma_upper]")
        flush(stdout)
    end

    fout, theta_out = lbfgsb(
        f,
        g!,
        theta_init;
        lb = lb,
        ub = ub,
        m = lbfgsb_m,
        factr = factr,
        pgtol = pgtol,
        iprint = iprint,
        maxfun = maxfun,
        maxiter = maxiter,
    )

    # Evaluate returned point once more, mainly to make sure the returned
    # theta_out is represented in the stored best candidate if appropriate.
    obj_out, g_out, gamma_out, psi_out, λmin_out, x_out, y_out =
        eval_ddfactplus_upsilon_calibration(
            C,
            theta_out,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            t1_reformulation = t1_reformulation,
        )

    if obj_out < best_obj[] - eps_lbfgsb
        best_obj[] = obj_out
        best_theta[] = copy(theta_out)
        best_gamma[] = copy(gamma_out)
        best_psi[] = psi_out
        best_λmin[] = λmin_out
        best_x[] = copy(x_out)
        best_y[] = copy(y_out)
    end

    return (
        gamma = best_gamma[],
        theta = best_theta[],
        psi = best_psi[],
        lambda_min = best_λmin[],
        x = best_x[],
        y = best_y[],
        obj = best_obj[],
        theta_out = theta_out,
        obj_out = obj_out,
        fout = fout,
        eval_count = eval_count[],
        gamma_lower = gamma_lower,
        gamma_upper = gamma_upper,
        theta_lower = theta_lower,
        theta_upper = theta_upper,
    )
end