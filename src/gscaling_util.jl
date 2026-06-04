# ============================================================
# Scaled matrix
# ============================================================
function scaled_matrix(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
)
    Dsqrt = Diagonal(sqrt.(gamma))
    return Symmetric(Dsqrt * Matrix(C) * Dsqrt)
end


# ============================================================
# Scaled matrix factor
# ============================================================
function scaled_factorize_matrix(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    psi::Float64;
    atol = 1e-8,
)
    n = size(C, 1)
    @assert length(gamma) == n
    @assert all(gamma .> 0)
    Cgamma = scaled_matrix(C, gamma)
    λ, Q = eigen(Cgamma - psi * I)
    perm = length(λ):-1:1
    λ = λ[perm]
    Q = Q[:, perm]
    # clamp small near-zero values before sqrt
    λ = map(x -> x < atol ? 0.0 : x, λ)
    F = Q * Diagonal(sqrt.(λ))
    return F
end


# ============================================================
# Feasibility of gamma
# ============================================================
function is_gamma_feasible(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    psi::Float64;
    atol = 1e-8,
)
    if any(gamma .<= 0)
        return false
    end
    Cgamma = scaled_matrix(C, gamma)
    return eigmin(Cgamma) >= psi - atol
end


# ============================================================
# Auxiliary t = 1 denominator
# ============================================================
function gamma_t1_denominator(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    psi::Float64,
    s::Int;
    atol::Float64 = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert all(gamma .> 0)

    cdiag = diag(C)

    denom = dot(x, gamma .* cdiag) - psi * (s - 1)

    if denom <= atol
        error("Invalid t=1 logarithm denominator: denom = $denom")
    end

    return denom
end


# ============================================================
# t = 1 gradient with respect to gamma
# ============================================================
function gamma_calibration_gradient_t1(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int;
    atol::Float64 = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert all(gamma .> 0)

    cdiag = diag(C)
    denom = gamma_t1_denominator(C, gamma, x, psi, s; atol = atol)

    return x .* cdiag ./ denom .- y ./ gamma
end


# ============================================================
# t = 1 gradient with respect to theta = log(gamma)
# ============================================================
function theta_calibration_gradient_t1(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int;
    atol::Float64 = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert all(gamma .> 0)

    cdiag = diag(C)
    denom = gamma_t1_denominator(C, gamma, x, psi, s; atol = atol)

    return x .* gamma .* cdiag ./ denom .- y
end


# ============================================================
# t = 1 derivative with respect to psi
# ============================================================
function gamma_t1_dGamma_dpsi(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    psi::Float64,
    s::Int;
    atol::Float64 = 1e-8,
)
    denom = gamma_t1_denominator(C, gamma, x, psi, s; atol = atol)

    return (1.0 - s) / denom
end


# ============================================================
# t = 1 gradient with respect to theta = log(gamma),
# including psi(theta) chain rule
# ============================================================
function theta_calibration_gradient_t1_with_psi_chain(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int,
    dpsi_dtheta::Vector{Float64};
    atol::Float64 = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert length(dpsi_dtheta) == n
    @assert all(gamma .> 0)

    g_fixed = theta_calibration_gradient_t1(
        C,
        gamma,
        x,
        y,
        psi,
        s;
        atol = atol,
    )

    dGamma_dpsi = gamma_t1_dGamma_dpsi(
        C,
        gamma,
        x,
        psi,
        s;
        atol = atol,
    )

    return g_fixed .+ dGamma_dpsi .* dpsi_dtheta
end


# ============================================================
# Subgradient with respect to gamma
# ============================================================
function gamma_calibration_subgradient(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int,
    t::Int;
    atol = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert all(gamma .> 0)
    @assert 1 <= t <= s <= n

    if t == 1
        return gamma_calibration_gradient_t1(
            C,
            gamma,
            x,
            y,
            psi,
            s;
            atol = atol,
        )
    end

    Cgamma = scaled_matrix(C, gamma)

    sqrtx = sqrt.(max.(x, 0.0))
    Xsqrt = Diagonal(sqrtx)

    Mtilde = Xsqrt * (Cgamma - psi * I) * Xsqrt
    Mtilde = Symmetric(0.5 * (Mtilde + Mtilde'))

    λ, Q = eigen(Mtilde)

    perm = length(λ):-1:1
    λ = λ[perm]
    Q = Q[:, perm]

    I_t = zeros(n)
    I_t[1:t] .= 1.0

    λshift = λ + psi * I_t

    iota, mid_val = find_iota(λshift, t)

    eigDual = zeros(n)

    if iota > 0
        eigDual[1:iota] .= 1.0 ./ λshift[1:iota]
    end

    eigDual[iota+1:end] .= 1.0 / mid_val

    G = Q * Diagonal(eigDual) * Q'
    W = Xsqrt * G * Xsqrt

    g_gamma = (diag(Cgamma * W) .- y) ./ gamma

    return g_gamma
end


# ============================================================
# Subgradient with respect to theta = log(gamma)
# ============================================================
function theta_calibration_subgradient(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int,
    t::Int;
    atol = 1e-8,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert all(gamma .> 0)
    @assert 1 <= t <= s <= n

    if t == 1
        return theta_calibration_gradient_t1(
            C,
            gamma,
            x,
            y,
            psi,
            s;
            atol = atol,
        )
    end

    Cgamma = scaled_matrix(C, gamma)

    sqrtx = sqrt.(max.(x, 0.0))
    Xsqrt = Diagonal(sqrtx)

    Mtilde = Xsqrt * (Cgamma - psi * I) * Xsqrt
    Mtilde = Symmetric(0.5 * (Mtilde + Mtilde'))

    λ, Q = eigen(Mtilde)

    perm = length(λ):-1:1
    λ = λ[perm]
    Q = Q[:, perm]

    I_t = zeros(n)
    I_t[1:t] .= 1.0

    λshift = λ + psi * I_t

    iota, mid_val = find_iota(λshift, t)

    eigDual = zeros(n)

    if iota > 0
        eigDual[1:iota] .= 1.0 ./ λshift[1:iota]
    end

    eigDual[iota+1:end] .= 1.0 / mid_val

    G = Q * Diagonal(eigDual) * Q'
    W = Xsqrt * G * Xsqrt

    # Gradient wrt theta = gamma .* gradient wrt gamma
    # = diag(Cgamma * W) - y
    g_theta = diag(Cgamma * W) .- y

    return g_theta
end


# ============================================================
# Subgradient with respect to theta = log(gamma) and psi(gamma)
# ============================================================
function theta_calibration_subgradient_with_psi_chain(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    x::Vector{Float64},
    y::Vector{Float64},
    psi::Float64,
    s::Int,
    t::Int;
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert length(x) == n
    @assert length(y) == n
    @assert all(gamma .> 0)
    @assert 1 <= t <= s <= n

    # ------------------------------------------------------------
    # C_gamma and eigenpair for lambda_min(C_gamma)
    # ------------------------------------------------------------
    Cgamma = scaled_matrix(C, gamma)

    eig_Cgamma = eigen(Cgamma)
    λmin_Cgamma = eig_Cgamma.values[1]
    vmin = eig_Cgamma.vectors[:, 1]

    # If psi = max(psi_floor, lambda_min - psi_margin), then the
    # derivative of psi wrt theta is zero when the floor is active.
    raw_psi = λmin_Cgamma - psi_margin

    if raw_psi > psi_floor + atol
        dpsi_dtheta = λmin_Cgamma .* (vmin .^ 2)
    else
        dpsi_dtheta = zeros(n)
    end

    if t == 1
        return theta_calibration_gradient_t1_with_psi_chain(
            C,
            gamma,
            x,
            y,
            psi,
            s,
            dpsi_dtheta;
            atol = atol,
        )
    end

    # ------------------------------------------------------------
    # General t > 1 spectral subgradient
    # ------------------------------------------------------------
    sqrtx = sqrt.(max.(x, 0.0))
    Xsqrt = Diagonal(sqrtx)

    Mtilde = Xsqrt * (Cgamma - psi * I) * Xsqrt
    Mtilde = Symmetric(0.5 * (Mtilde + Mtilde'))

    λ, Q = eigen(Mtilde)

    # Sort eigenvalues decreasingly, as in the rest of your code.
    perm = length(λ):-1:1
    λ = λ[perm]
    Q = Q[:, perm]

    I_t = zeros(n)
    I_t[1:t] .= 1.0

    λshift = λ + psi .* I_t

    iota, mid_val = find_iota(λshift, t)

    eigDual = zeros(n)

    if iota > 0
        eigDual[1:iota] .= 1.0 ./ λshift[1:iota]
    end

    eigDual[iota+1:end] .= 1.0 / mid_val

    # ------------------------------------------------------------
    # Spectral subgradient matrices
    # ------------------------------------------------------------
    G = Q * Diagonal(eigDual) * Q'
    W = Xsqrt * G * Xsqrt

    # ------------------------------------------------------------
    # Fixed-psi gradient wrt theta
    #
    # g_fixed_i = (Cgamma * W)_ii - y_i
    # ------------------------------------------------------------
    g_fixed = diag(Cgamma * W) .- y

    # ------------------------------------------------------------
    # Chain-rule term from psi(theta)
    #
    # dGamma/dpsi = sum_{j=1}^t eigDual_j - tr(W)
    # ------------------------------------------------------------
    dGamma_dpsi = sum(@view eigDual[1:t]) - tr(W)

    g_chain = dGamma_dpsi .* dpsi_dtheta

    g_total = g_fixed .+ g_chain

    return g_total
end


# ------------------------------------------------------------
# Largest feasible psi for a given gamma
# ------------------------------------------------------------
function max_feasible_psi(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64};
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
)
    Cgamma = scaled_matrix(C, gamma)
    λmin = eigmin(Cgamma)

    # "Highest feasible" psi, with a small margin to avoid numerical
    # issues in the factorization Cgamma - psi*I.
    psi = max(psi_floor, λmin - psi_margin)

    return psi, λmin
end


# ------------------------------------------------------------
# Objective and gradient evaluation for BFGS
# ------------------------------------------------------------
function eval_ddfactplus_upsilon_calibration(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    s::Int,
    t::Int;
    atol::Float64 = 1e-10,
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
)
    gamma = exp.(theta)

    psi, λmin = max_feasible_psi(
        C,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    if (t == 1) && t1_reformulation
        # Reformulation for t = 1
        # result_t1_reform =
        #     ddfact_upsilon_t1_ipopt(
        #         C,
        #         gamma,
        #         s,
        #         psi;
        #         atol = atol,
        #     )
        
        result_t1_reform =
            ddfact_upsilon_t1_knitro(
                C,
                gamma,
                s,
                psi;
                atol = atol,
            )

        x = result_t1_reform.x
        y = result_t1_reform.y
        obj = result_t1_reform.primal_obj

        reform_gap = abs(result_t1_reform.obj_val - result_t1_reform.primal_obj)
        if reform_gap > 1e-6
            @warn "t=1 reformulation objective and recovered primal objective differ" reform_gap
        end
    else
        # General formulation
        x, y, obj = aug_ddfact_upsilon_gmesp(
            C,
            gamma,
            s,
            t,
            psi;
            atol = atol,
        )
    end

    if psi_derivative
        # Corrected subgradient:
        # This is the theta-gradient for z(C, s, t; psi(gamma), gamma).
        # We consider d psi(gamma) / d theta,
        g = theta_calibration_subgradient_with_psi_chain(
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
        # Heuristic subgradient:
        # This is the theta-gradient for fixed psi, evaluated at the current
        # highest feasible psi. We intentionally ignore d psi(gamma) / d theta,
        # since the calibration is heuristic.
        g = theta_calibration_subgradient(
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

    return obj, g, gamma, psi, λmin, x, y
end