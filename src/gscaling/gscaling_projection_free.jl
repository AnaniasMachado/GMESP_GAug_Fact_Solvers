using LinearAlgebra
using JuMP
using Gurobi


# ============================================================
# Projection-free minimax calibration for DDGFact^+_Upsilon
#
# Implements the three algorithms from the paper:
#
#   :lmo_lmo  -> Algorithm 1
#   :lmo_po   -> Algorithm 2
#   :po_lmo   -> Algorithm 3
#
# Problem:
#
#   min_{theta in [-B,B]^n} max_{(x,y) in P}
#
#       Gamma_t(M_{theta,psi(theta)}(x); psi(theta))
#       - theta' y
#
# Dynamic smoothing on q=(x,y):
#
#   L_beta(theta,q)
#       = L(theta,q) - beta/2 ||q - q_ref||^2.
# ============================================================


# ============================================================
# Gurobi options
# ============================================================

function _pf_set_gurobi_options!(
    model::JuMP.Model;
    gurobi_output_flag::Int = 0,
    gurobi_opttol::Union{Nothing,Float64} = 1e-8,
    gurobi_feastol::Union{Nothing,Float64} = 1e-8,
)
    set_optimizer_attribute(model, "OutputFlag", gurobi_output_flag)

    if gurobi_opttol !== nothing
        set_optimizer_attribute(model, "OptimalityTol", gurobi_opttol)
        set_optimizer_attribute(model, "BarConvTol", gurobi_opttol)
    end

    if gurobi_feastol !== nothing
        set_optimizer_attribute(model, "FeasibilityTol", gurobi_feastol)
    end

    return nothing
end


# ============================================================
# Feasible initialization
# ============================================================

function _pf_initial_xy(
    n::Int,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
)
    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    x = zeros(Float64, n)
    x[J1v] .= 1.0

    free = setdiff(1:n, union(J1v, J0v))
    rem = s - length(J1v)

    x[free] .= rem / length(free)

    y = (t / s) .* x

    return x, y
end


# ============================================================
# LMO over theta box
#
# Solves:
#
#   min_{v in [-B,B]^n} g'v.
# ============================================================

function _pf_lmo_theta_box(
    gtheta::Vector{Float64},
    theta_bound::Float64,
)
    return -theta_bound .* sign.(gtheta)
end


# ============================================================
# Projection over theta box
# ============================================================

function _pf_project_theta_box(
    theta::Vector{Float64},
    theta_bound::Float64,
)
    return clamp.(theta, -theta_bound, theta_bound)
end


# ============================================================
# Reusable LMO model over q=(x,y)
#
# Solves:
#
#   min dx'x + dy'y
#
# s.t.
#   sum x = s
#   sum y = t
#   0 <= y <= x <= 1
#   x[J1] = 1
#   x[J0] = 0
#   y[J0] = 0
# ============================================================

mutable struct _PFQLinearOracle
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    y::Vector{JuMP.VariableRef}
    n::Int
end


function _pf_build_q_lmo_model(
    n::Int,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    gurobi_output_flag::Int = 0,
    gurobi_opttol::Union{Nothing,Float64} = 1e-8,
    gurobi_feastol::Union{Nothing,Float64} = 1e-8,
)
    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    model = Model(Gurobi.Optimizer)

    _pf_set_gurobi_options!(
        model;
        gurobi_output_flag = gurobi_output_flag,
        gurobi_opttol = gurobi_opttol,
        gurobi_feastol = gurobi_feastol,
    )

    @variable(model, 0.0 <= x[1:n] <= 1.0)
    @variable(model, 0.0 <= y[1:n] <= 1.0)

    @constraint(model, sum(x[i] for i in 1:n) == s)
    @constraint(model, sum(y[i] for i in 1:n) == t)
    @constraint(model, [i in 1:n], y[i] <= x[i])

    for i in J1v
        @constraint(model, x[i] == 1.0)
    end

    for i in J0v
        @constraint(model, x[i] == 0.0)
        @constraint(model, y[i] == 0.0)
    end

    @objective(model, Min, 0.0)

    return _PFQLinearOracle(
        model,
        [x[i] for i in 1:n],
        [y[i] for i in 1:n],
        n,
    )
end


function _pf_lmo_q!(
    oracle::_PFQLinearOracle,
    dx::Vector{Float64},
    dy::Vector{Float64},
)
    n = oracle.n

    @objective(
        oracle.model,
        Min,
        sum(dx[i] * oracle.x[i] + dy[i] * oracle.y[i] for i in 1:n)
    )

    optimize!(oracle.model)

    return Vector{Float64}(value.(oracle.x)),
           Vector{Float64}(value.(oracle.y))
end


# ============================================================
# Reusable projection model over q=(x,y)
#
# Solves:
#
#   min 0.5||x-qx||^2 + 0.5||y-qy||^2
#
# s.t. (x,y) in P.
# ============================================================

mutable struct _PFQProjectionOracle
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    y::Vector{JuMP.VariableRef}
    n::Int
end


function _pf_build_q_projection_model(
    n::Int,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    gurobi_output_flag::Int = 0,
    gurobi_opttol::Union{Nothing,Float64} = 1e-8,
    gurobi_feastol::Union{Nothing,Float64} = 1e-8,
)
    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    model = Model(Gurobi.Optimizer)

    _pf_set_gurobi_options!(
        model;
        gurobi_output_flag = gurobi_output_flag,
        gurobi_opttol = gurobi_opttol,
        gurobi_feastol = gurobi_feastol,
    )

    @variable(model, 0.0 <= x[1:n] <= 1.0)
    @variable(model, 0.0 <= y[1:n] <= 1.0)

    @constraint(model, sum(x[i] for i in 1:n) == s)
    @constraint(model, sum(y[i] for i in 1:n) == t)
    @constraint(model, [i in 1:n], y[i] <= x[i])

    for i in J1v
        @constraint(model, x[i] == 1.0)
    end

    for i in J0v
        @constraint(model, x[i] == 0.0)
        @constraint(model, y[i] == 0.0)
    end

    @objective(model, Min, 0.0)

    return _PFQProjectionOracle(
        model,
        [x[i] for i in 1:n],
        [y[i] for i in 1:n],
        n,
    )
end


function _pf_project_q!(
    oracle::_PFQProjectionOracle,
    qx::Vector{Float64},
    qy::Vector{Float64},
)
    n = oracle.n

    @objective(
        oracle.model,
        Min,
        0.5 * sum((oracle.x[i] - qx[i])^2 for i in 1:n)
        +
        0.5 * sum((oracle.y[i] - qy[i])^2 for i in 1:n)
    )

    optimize!(oracle.model)

    return Vector{Float64}(value.(oracle.x)),
           Vector{Float64}(value.(oracle.y))
end


# ============================================================
# Gamma_t spectral subgradient with respect to M
# ============================================================

function _pf_Gamma_t_matrix_subgradient(
    M::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    n = size(M, 1)

    eig = eigen(Symmetric(0.5 .* (M .+ M')))
    lambda = eig.values
    Q = eig.vectors

    perm = sortperm(lambda; rev = true)
    lambdas = Vector{Float64}(lambda[perm])
    Qs = Q[:, perm]

    shifted = copy(lambdas)
    shifted[1:t] .+= psi

    iota, mid = find_iota(shifted, t; atol = atol)

    g_eigs = fill(1.0 / mid, n)

    if iota > 0
        for j in 1:iota
            g_eigs[j] = 1.0 / shifted[j]
        end
    end

    G = Qs * Diagonal(g_eigs) * Qs'
    G = Symmetric(0.5 .* (G .+ G'))

    return Matrix(G), iota, mid
end


# ============================================================
# Subgradient with respect to q=(x,y)
# ============================================================

function _pf_q_gradient(
    x::Vector{Float64},
    theta::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    M = F' * Diagonal(x) * F
    M = Matrix(Symmetric(0.5 .* (M .+ M')))

    G_M, iota, mid = _pf_Gamma_t_matrix_subgradient(
        M,
        t,
        psi;
        atol = atol,
    )

    gx = Vector{Float64}(diag(F * G_M * F'))
    gy = -theta

    return gx, gy, iota, mid
end


# ============================================================
# Full state gradients
# ============================================================

function _pf_state_gradients(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    s::Int,
    t::Int;
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    atol::Float64,
)
    gamma = exp.(theta)

    psi, lambda_min_val = max_feasible_psi(
        C,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    F = scaled_factorize_matrix(
        C,
        gamma,
        psi;
        atol = atol,
    )

    if psi_derivative
        gtheta = theta_calibration_subgradient_with_psi_chain(
            C,
            gamma,
            x,
            y,
            psi,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        )
    else
        gtheta = theta_calibration_subgradient(
            C,
            gamma,
            x,
            y,
            psi,
            s,
            t;
            atol = atol,
        )
    end

    gx, gy, iota, mid = _pf_q_gradient(
        x,
        theta,
        F,
        t,
        psi;
        atol = atol,
    )

    return (
        gtheta = gtheta,
        gx = gx,
        gy = gy,
        gamma = gamma,
        psi = psi,
        lambda_min = lambda_min_val,
        iota = iota,
        mid = mid,
    )
end


# ============================================================
# State objective
# ============================================================

function _pf_state_objective(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    t::Int;
    psi_margin::Float64,
    psi_floor::Float64,
    atol::Float64,
)
    gamma = exp.(theta)

    psi, lambda_min_val = max_feasible_psi(
        C,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    F = scaled_factorize_matrix(
        C,
        gamma,
        psi;
        atol = atol,
    )

    obj = Gamma_t_upsilon_from_F(
        x,
        y,
        gamma,
        F,
        t,
        psi,
    )

    return (
        obj = obj,
        gamma = gamma,
        psi = psi,
        lambda_min = lambda_min_val,
    )
end


# ============================================================
# Main projection-free calibration routine
# ============================================================

function calibrate_upsilon_projection_free_ddfactplus(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    algorithm::Symbol = :po_lmo,

    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    theta0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    x0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    y0::Union{Nothing,AbstractVector{<:Real}} = nothing,

    max_iter::Int = 1000,
    min_iter::Int = 100,
    iteration_power::Float64 = 1.5,

    theta_bound::Float64 = 20.0,

    tau0::Float64 = 0.02,
    tau_power::Float64 = 2.0 / 3.0,

    beta0::Float64 = 1.0,
    beta_power::Float64 = 1.0 / 6.0,

    Lqq_hat::Float64 = 1.0,

    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    atol::Float64 = 1e-10,

    gurobi_output_flag::Int = 0,
    gurobi_opttol::Union{Nothing,Float64} = 1e-8,
    gurobi_feastol::Union{Nothing,Float64} = 1e-8,

    diagnostic_knitro_outlev = nothing,
    diagnostic_knitro_opttol::Union{Nothing,Float64} = 1e-8,
    diagnostic_knitro_feastol::Union{Nothing,Float64} = 1e-5,

    diagnostics::Bool = false,
    verbose::Bool = false,
)
    n = size(C, 1)

    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    fixed_n = length(union(J1v, J0v))
    free_n = n - fixed_n

    adaptive_max_iter =
        min(
            max_iter,
            max(
                min_iter,
                ceil(Int, free_n^iteration_power),
            ),
        )

    theta =
        theta0 === nothing ?
        zeros(Float64, n) :
        Vector{Float64}(theta0)

    theta = clamp.(theta, -theta_bound, theta_bound)

    if x0 === nothing || y0 === nothing
        x, y = _pf_initial_xy(
            n,
            s,
            t;
            J1 = J1v,
            J0 = J0v,
        )
    else
        x = Vector{Float64}(x0)
        y = Vector{Float64}(y0)
    end

    x_ref = copy(x)
    y_ref = copy(y)

    q_lmo = _pf_build_q_lmo_model(
        n,
        s,
        t;
        J1 = J1v,
        J0 = J0v,
        gurobi_output_flag = gurobi_output_flag,
        gurobi_opttol = gurobi_opttol,
        gurobi_feastol = gurobi_feastol,
    )

    q_proj = _pf_build_q_projection_model(
        n,
        s,
        t;
        J1 = J1v,
        J0 = J0v,
        gurobi_output_flag = gurobi_output_flag,
        gurobi_opttol = gurobi_opttol,
        gurobi_feastol = gurobi_feastol,
    )

    history = NamedTuple[]

    iterations_done = 0
    stop_reason = :adaptive_max_iter

    for k in 0:(adaptive_max_iter - 1)
        tau = tau0 / (k + 1)^tau_power
        beta = beta0 / (k + 1)^beta_power

        do_verbose =
            verbose &&
            (k == 0 || (k + 1) % 10 == 0 || k + 1 == adaptive_max_iter)

        grads = _pf_state_gradients(
            C,
            theta,
            x,
            y,
            s,
            t;
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            atol = atol,
        )

        gx_beta = grads.gx .- beta .* (x .- x_ref)
        gy_beta = grads.gy .- beta .* (y .- y_ref)

        if diagnostics || do_verbose
            state_val = _pf_state_objective(
                C,
                theta,
                x,
                y,
                t;
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                atol = atol,
            )
        end

        if diagnostics
            gamma_diag = exp.(theta)

            actual_x = similar(x)
            actual_y = similar(y)
            actual_obj = NaN

            actual_obj_runtime = @elapsed begin
                actual_x, actual_y, actual_obj =
                    aug_ddfact_upsilon_gmesp(
                        C,
                        gamma_diag,
                        s,
                        t,
                        state_val.psi;
                        J1 = J1v,
                        x0 = x,
                        y0 = y,
                        atol = atol,
                        knitro_outlev = diagnostic_knitro_outlev,
                        knitro_opttol = diagnostic_knitro_opttol,
                        knitro_feastol = diagnostic_knitro_feastol,
                    )
            end
        end

        if algorithm == :lmo_lmo
            vtheta = _pf_lmo_theta_box(grads.gtheta, theta_bound)

            theta_new =
                theta .+ tau .* (vtheta .- theta)

            ux, uy = _pf_lmo_q!(
                q_lmo,
                -gx_beta,
                -gy_beta,
            )

            qdiff_norm_sq =
                norm(ux .- x)^2 + norm(uy .- y)^2

            numerator =
                dot(gx_beta, ux .- x) +
                dot(gy_beta, uy .- y)

            raw_gamma_step =
                numerator / ((Lqq_hat + beta) * qdiff_norm_sq)

            gamma_step =
                !isfinite(raw_gamma_step) ?
                0.0 :
                clamp(raw_gamma_step, 0.0, 1.0)

            x_new = x .+ gamma_step .* (ux .- x)
            y_new = y .+ gamma_step .* (uy .- y)

        elseif algorithm == :lmo_po
            vtheta = _pf_lmo_theta_box(grads.gtheta, theta_bound)

            theta_new =
                theta .+ tau .* (vtheta .- theta)

            gamma_step = 1.0 / (Lqq_hat + beta)

            x_new, y_new = _pf_project_q!(
                q_proj,
                x .+ gamma_step .* gx_beta,
                y .+ gamma_step .* gy_beta,
            )

        elseif algorithm == :po_lmo
            theta_new =
                _pf_project_theta_box(
                    theta .- tau .* grads.gtheta,
                    theta_bound,
                )

            ux, uy = _pf_lmo_q!(
                q_lmo,
                -gx_beta,
                -gy_beta,
            )

            qdiff_norm_sq =
                norm(ux .- x)^2 + norm(uy .- y)^2

            numerator =
                dot(gx_beta, ux .- x) +
                dot(gy_beta, uy .- y)

            raw_gamma_step =
                numerator / ((Lqq_hat + beta) * qdiff_norm_sq)

            gamma_step =
                !isfinite(raw_gamma_step) ?
                0.0 :
                clamp(raw_gamma_step, 0.0, 1.0)

            x_new = x .+ gamma_step .* (ux .- x)
            y_new = y .+ gamma_step .* (uy .- y)

        else
            error("Unknown projection-free algorithm: $(algorithm).")
        end

        iterations_done = k + 1

        if diagnostics
            push!(history, (
                iter = k + 1,
                algorithm = algorithm,
                tau = tau,
                beta = beta,
                gamma_step = gamma_step,

                psi = state_val.psi,
                lambda_min = state_val.lambda_min,

                state_obj = state_val.obj,
                actual_obj = actual_obj,
                actual_obj_runtime = actual_obj_runtime,
                surrogate_diff = actual_obj - state_val.obj,

                gtheta_norm = norm(grads.gtheta),
                gx_norm = norm(grads.gx),
                gy_norm = norm(grads.gy),
                gx_beta_norm = norm(gx_beta),
                gy_beta_norm = norm(gy_beta),

                iota = grads.iota,
                mid = grads.mid,
            ))
        end

        if do_verbose
            println(
                "PF iter=", k + 1,
                " algorithm=", algorithm,
                " state_obj=", state_val.obj,
                " tau=", tau,
                " beta=", beta,
                " gamma=", gamma_step,
            )
        end

        theta = theta_new
        x = x_new
        y = y_new
    end

    return (
        algorithm = algorithm,

        theta = copy(theta),
        x = copy(x),
        y = copy(y),

        iterations = iterations_done,
        max_iter = max_iter,
        adaptive_max_iter = adaptive_max_iter,
        min_iter = min_iter,
        iteration_power = iteration_power,
        free_n = free_n,
        fixed_n = fixed_n,
        stop_reason = stop_reason,

        theta_bound = theta_bound,
        tau0 = tau0,
        tau_power = tau_power,
        beta0 = beta0,
        beta_power = beta_power,
        Lqq_hat = Lqq_hat,

        history = history,
    )
end