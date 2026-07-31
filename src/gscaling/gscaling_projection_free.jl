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
# Dynamic-programming LMO over q=(x,y)
# ============================================================

function _pf_lmo_q_dp!(
    dx::AbstractVector{<:Real},
    dy::AbstractVector{<:Real},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
)
    n = length(dx)

    length(dy) == n ||
        throw(DimensionMismatch(
            "dx and dy must have the same length."
        ))

    0 <= t <= s <= n ||
        throw(ArgumentError(
            "The cardinalities must satisfy 0 <= t <= s <= n."
        ))

    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    all(i -> 1 <= i <= n, J1v) ||
        throw(ArgumentError("Every index in J1 must belong to 1:n."))

    all(i -> 1 <= i <= n, J0v) ||
        throw(ArgumentError("Every index in J0 must belong to 1:n."))

    isempty(intersect(J1v, J0v)) ||
        throw(ArgumentError("J1 and J0 must be disjoint."))

    length(J1v) <= s ||
        throw(ArgumentError(
            "The constraint x[J1] = 1 is infeasible because length(J1) > s."
        ))

    n - length(J0v) >= s ||
        throw(ArgumentError(
            "The constraint x[J0] = 0 leaves fewer than s selectable indices."
        ))

    # State classification:
    #
    #   0 -> free index:
    #        (0,0), (1,0), or (1,1)
    #
    #   1 -> J1 index:
    #        (1,0) or (1,1)
    #
    #   2 -> J0 index:
    #        only (0,0)
    index_type = zeros(UInt8, n)

    @inbounds for i in J1v
        index_type[i] = 0x01
    end

    @inbounds for i in J0v
        index_type[i] = 0x02
    end

    # Two rolling DP layers.
    #
    # Matrix entry [a+1,b+1] corresponds to cardinalities (a,b).
    previous = fill(Inf, s + 1, t + 1)
    current  = fill(Inf, s + 1, t + 1)

    previous[1, 1] = 0.0

    # Backtracking flags.
    #
    # For a reachable state after processing index i:
    #
    #   chose_x[i,a+1,b+1] = true
    #       means x_i = 1
    #
    #   chose_y[i,a+1,b+1] = true
    #       means y_i = 1
    #
    # Therefore:
    #
    #   false, false -> (0,0)
    #   true,  false -> (1,0)
    #   true,  true  -> (1,1)
    #
    # BitArray stores one bit per entry, substantially reducing
    # backtracking memory relative to Array{UInt8,3}.
    chose_x = falses(n, s + 1, t + 1)
    chose_y = falses(n, s + 1, t + 1)

    # Prefix counts used to restrict the set of states visited.
    number_j1_processed = 0
    number_j0_processed = 0

    @inbounds for i in 1:n
        fill!(current, Inf)

        type_i = index_type[i]

        if type_i == 0x01
            number_j1_processed += 1
        elseif type_i == 0x02
            number_j0_processed += 1
        end

        # After processing i indices:
        #
        # - at least all processed J1 indices must have x=1;
        # - at most all processed non-J0 indices can have x=1;
        # - y cardinality cannot exceed x cardinality.
        maximum_x_after =
            min(s, i - number_j0_processed)

        minimum_x_after =
            min(s, number_j1_processed)

        # Bounds for states before processing index i.
        maximum_x_before =
            min(s, (i - 1) - (number_j0_processed -
                              (type_i == 0x02 ? 1 : 0)))

        minimum_x_before =
            min(s, number_j1_processed -
                   (type_i == 0x01 ? 1 : 0))

        for a in minimum_x_before:maximum_x_before
            maximum_y_before = min(t, a)

            for b in 0:maximum_y_before
                old_value = previous[a + 1, b + 1]

                isfinite(old_value) || continue

                # ------------------------------------------------
                # State (x_i,y_i) = (0,0)
                #
                # Allowed for free and J0 indices, but not J1.
                # ------------------------------------------------
                if type_i != 0x01
                    if minimum_x_after <= a <= maximum_x_after &&
                       old_value < current[a + 1, b + 1]

                        current[a + 1, b + 1] = old_value

                        chose_x[i, a + 1, b + 1] = false
                        chose_y[i, a + 1, b + 1] = false
                    end
                end

                # J0 permits no state with x_i=1.
                type_i == 0x02 && continue

                # ------------------------------------------------
                # State (x_i,y_i) = (1,0)
                # ------------------------------------------------
                if a < s
                    new_a = a + 1
                    candidate = old_value + Float64(dx[i])

                    if minimum_x_after <= new_a <= maximum_x_after &&
                       candidate < current[new_a + 1, b + 1]

                        current[new_a + 1, b + 1] = candidate

                        chose_x[i, new_a + 1, b + 1] = true
                        chose_y[i, new_a + 1, b + 1] = false
                    end
                end

                # ------------------------------------------------
                # State (x_i,y_i) = (1,1)
                # ------------------------------------------------
                if a < s && b < t
                    new_a = a + 1
                    new_b = b + 1

                    candidate =
                        old_value +
                        Float64(dx[i]) +
                        Float64(dy[i])

                    if minimum_x_after <= new_a <= maximum_x_after &&
                       candidate < current[new_a + 1, new_b + 1]

                        current[new_a + 1, new_b + 1] = candidate

                        chose_x[i, new_a + 1, new_b + 1] = true
                        chose_y[i, new_a + 1, new_b + 1] = true
                    end
                end
            end
        end

        previous, current = current, previous
    end

    optimal_value = previous[s + 1, t + 1]

    isfinite(optimal_value) ||
        error(
            "The dynamic-programming q-LMO found no feasible solution."
        )

    # ------------------------------------------------------------
    # Backtrack from cardinality state (s,t).
    # ------------------------------------------------------------
    x = zeros(Float64, n)
    y = zeros(Float64, n)

    a = s
    b = t

    @inbounds for i in n:-1:1
        xi = chose_x[i, a + 1, b + 1]
        yi = chose_y[i, a + 1, b + 1]

        if xi
            x[i] = 1.0
            a -= 1
        end

        if yi
            y[i] = 1.0
            b -= 1
        end
    end

    a == 0 && b == 0 ||
        error(
            "Internal error while backtracking the dynamic-programming q-LMO."
        )

    return x, y
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
    lmo_q_solver::Bool = true,

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

    q_lmo = algorithm == :lmo_lmo || (algorithm == :po_lmo && lmo_q_solver) ?
        _pf_build_q_lmo_model(
            n,
            s,
            t;
            J1 = J1v,
            J0 = J0v,
            gurobi_output_flag = gurobi_output_flag,
            gurobi_opttol = gurobi_opttol,
            gurobi_feastol = gurobi_feastol,
        ) :
        nothing

    q_proj = algorithm == :lmo_po ?
        _pf_build_q_projection_model(
            n,
            s,
            t;
            J1 = J1v,
            J0 = J0v,
            gurobi_output_flag = gurobi_output_flag,
            gurobi_opttol = gurobi_opttol,
            gurobi_feastol = gurobi_feastol,
        ) :
        nothing

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

            if lmo_q_solver
                ux, uy = _pf_lmo_q!(
                    q_lmo,
                    -gx_beta,
                    -gy_beta,
                )
            else
                ux, uy = _pf_lmo_q_dp!(
                    -gx_beta,
                    -gy_beta,
                    s,
                    t;
                    J1 = J1v,
                    J0 = J0v,
                )
            end

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

            if lmo_q_solver
                ux, uy = _pf_lmo_q!(
                    q_lmo,
                    -gx_beta,
                    -gy_beta,
                )
            else
                ux, uy = _pf_lmo_q_dp!(
                    -gx_beta,
                    -gy_beta,
                    s,
                    t;
                    J1 = J1v,
                    J0 = J0v,
                )
            end

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