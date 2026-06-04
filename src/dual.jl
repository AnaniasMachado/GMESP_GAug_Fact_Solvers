using LinearAlgebra
using JuMP
using Gurobi
import MathOptInterface as MOI

# ============================================================
# Construct Theta_hat from a feasible DDGFact solution xhat
#
# Paper notation:
#   M(xhat) = F' * Diagonal(xhat) * F
#   M(xhat) = sum lambda_i u_i u_i'
#
# Then:
#   beta_i = 1/lambda_i,          i <= iota
#   beta_i = 1/delta,             iota < i <= rank(M)
#   beta_i = (1 + epsilon)/delta, rank(M) < i <= k
#
# where delta = sum_{i=iota+1}^k lambda_i / (t - iota)
# ============================================================
function construct_Theta_from_x_DDGFact(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int;
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    M = M_psi(xhat, F)

    eig = eigen(Symmetric(M))
    lambda = eig.values
    U = eig.vectors

    # Sort eigenvalues in nonincreasing order
    perm = sortperm(lambda, rev = true)
    lambda = lambda[perm]
    U = U[:, perm]

    k = length(lambda)

    if t < 1 || t > k
        error("t must satisfy 1 <= t <= size(M, 1).")
    end

    rank_hat = count(v -> v > atol, lambda)

    if rank_hat < t
        error("M_psi(xhat) must have rank at least t.")
    end

    iota, delta = find_iota(lambda, t; atol = atol)

    beta = zeros(Float64, k)

    if iota > 0
        beta[1:iota] .= 1.0 ./ lambda[1:iota]
    end

    if iota < rank_hat
        beta[(iota + 1):rank_hat] .= 1.0 / delta
    end

    if rank_hat < k
        beta[(rank_hat + 1):k] .= (1.0 + epsilon) / delta
    end

    Theta = Symmetric(U * Diagonal(beta) * U')

    return Theta, beta, lambda, iota, delta, rank_hat
end

# ============================================================
# Solve G(Theta_hat) in closed form for the no-side-constraint case
#
# Given d = diag(F * Theta_hat * F'), solve
#
#   min  -upsilon'l + nu'c + tau*s
#   s.t. d + upsilon - nu - tau*e = 0
#        upsilon >= 0, nu >= 0
#
# ============================================================
function solve_GTheta_no_side_constraints(
    d::Vector{Float64},
    s::Int;
    l::Vector{Float64} = zeros(length(d)),
    c::Vector{Float64} = ones(length(d)),
    atol::Float64 = 1e-8,
)
    n = length(d)

    if length(l) != n || length(c) != n
        error("l, c, and d must have the same length.")
    end

    if any(l .> c .+ atol)
        error("Bounds must satisfy l <= c.")
    end

    if sum(l) > s + atol || sum(c) < s - atol
        error("The fixed-variable polytope is empty: need sum(l) <= s <= sum(c).")
    end

    # Sort d in nonincreasing order
    sigma = sortperm(d, rev = true)

    # Find phi as in the paper
    phi = 0

    for j in 1:n
        lhs = sum(c[sigma[1:j]]) + sum(l[sigma[(j + 1):n]])
        if lhs <= s + atol
            phi = j
        else
            break
        end
    end

    P = sigma[1:phi]

    central = nothing
    Q = Int[]

    if phi < n
        central = sigma[phi + 1]
        if phi + 2 <= n
            Q = sigma[(phi + 2):n]
        end
    end

    xstar = zeros(Float64, n)

    if !isempty(P)
        xstar[P] .= c[P]
    end

    if !isempty(Q)
        xstar[Q] .= l[Q]
    end

    if central !== nothing
        xstar[central] =
            s -
            (isempty(P) ? 0.0 : sum(c[P])) -
            (isempty(Q) ? 0.0 : sum(l[Q]))
    end

    tau = central === nothing ? 0.0 : d[central]

    upsilon = zeros(Float64, n)
    nu = zeros(Float64, n)

    if !isempty(P)
        nu[P] .= d[P] .- tau
    end

    if !isempty(Q)
        upsilon[Q] .= tau .- d[Q]
    end

    # Clean tiny numerical negatives
    upsilon[abs.(upsilon) .<= atol] .= 0.0
    nu[abs.(nu) .<= atol] .= 0.0

    if minimum(upsilon) < -atol
        error("Constructed upsilon is not nonnegative.")
    end

    if minimum(nu) < -atol
        error("Constructed nu is not nonnegative.")
    end

    residual = d + upsilon - nu .- tau

    if maximum(abs.(residual)) > 1e-6
        error("Dual equality constraint is not satisfied. Residual = $(maximum(abs.(residual))).")
    end

    if abs(sum(xstar) - s) > 1e-6
        error("Constructed primal xstar does not satisfy e'x = s.")
    end

    linear_value = -dot(upsilon, l) + dot(nu, c) + tau * s

    return (
        upsilon = upsilon,
        nu = nu,
        tau = tau,
        xstar = xstar,
        sigma = sigma,
        P = P,
        Q = Q,
        central = central,
        linear_value = linear_value,
        residual = residual,
    )
end

# ============================================================
# Objective value of a feasible DGFact solution
#
# Fixed-variable DGFact objective:
#   - sum_{ell=k-t+1}^k log(lambda_ell(Theta))
#   - upsilon'l + nu'c + tau*s - t
# ============================================================
function DGFact_objective_value(
    Theta::AbstractMatrix{Float64},
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    tau::Float64,
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
)
    theta_eigs = sort(eigvals(Symmetric(Theta)), rev = true)
    k = length(theta_eigs)

    if t < 1 || t > k
        error("t must satisfy 1 <= t <= size(Theta, 1).")
    end

    smallest_t = theta_eigs[(k - t + 1):k]

    if any(smallest_t .<= 0.0)
        return (
            objective_value = Inf,
            spectral_part = Inf,
            linear_part = -dot(upsilon, l) + dot(nu, c) + tau * s,
        )
    end

    spectral_part = -sum(log.(smallest_t)) - t
    linear_part = -dot(upsilon, l) + dot(nu, c) + tau * s

    return (
        objective_value = spectral_part + linear_part,
        spectral_part = spectral_part,
        linear_part = linear_part,
    )
end

# ============================================================
# Full construction:
#
# Input:
#   xhat feasible for DDGFact
#   F factorization matrix
#   t
#
# Output:
#   Theta, upsilon, nu, tau feasible for DGFact
#   for the no-side-constraint case.
# ============================================================
function DGFact_dual_solution_from_DDGFact_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    Theta, beta, lambda, iota, delta, rank_hat =
        construct_Theta_from_x_DDGFact(
            xhat,
            F,
            t;
            epsilon = epsilon,
            atol = atol,
        )

    # d = diag(F * Theta * F')
    d = vec(sum((F * Theta) .* F; dims = 2))

    sol_G = solve_GTheta_no_side_constraints(
        d,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    obj = DGFact_objective_value(
        Theta,
        sol_G.upsilon,
        sol_G.nu,
        sol_G.tau,
        s,
        t;
        l = l,
        c = c,
    )

    return (
        Theta = Theta,
        upsilon = sol_G.upsilon,
        nu = sol_G.nu,
        tau = sol_G.tau,
        objective_value = obj.objective_value,
        spectral_part = obj.spectral_part,
        linear_part = obj.linear_part,
        d = d,
        xstar_GTheta = sol_G.xstar,
        beta = beta,
        lambda = lambda,
        iota = iota,
        delta = delta,
        rank_hat = rank_hat,
        sigma = sol_G.sigma,
        P = sol_G.P,
        Q = sol_G.Q,
        central = sol_G.central,
        residual = sol_G.residual,
    )
end

# ============================================================
# h_psi(w)
#
# Manuscript definition:
#   h_psi(w) = -log(w) + psi*w - 1,  0 < w < 1/psi
#            = log(psi),             w >= 1/psi
#
# This implementation assumes psi > 0.
# ============================================================
function h_psi_value(
    w::Float64,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    if psi <= 0.0
        error("This h_psi implementation for DGFact^+ assumes psi > 0.")
    end

    if w <= 0.0
        return Inf
    end

    threshold = 1.0 / psi

    if w < threshold - atol
        return -log(w) + psi * w - 1.0
    else
        return log(psi)
    end
end

# ============================================================
# Construct Theta_hat from a feasible DDGFact^+ solution xhat
#
# Manuscript notation:
#   Mhat = M_psi(xhat) = F' * Diagonal(xhat) * F
#   Mhat = Q * Diagonal(lambda) * Q'
#
# Define:
#   y_l = lambda_l + psi,  l = 1,...,t
#   y_l = lambda_l,        l = t+1,...,k
#
# Let iota be the index associated with y.
#
# Then:
#   beta_l = 1/y_l,                       l = 1,...,iota
#   beta_l = (t-iota) / sum_{j=iota+1}^k y_j,
#                                           l = iota+1,...,k
#
# Theta_hat = Q * Diagonal(beta) * Q'
# ============================================================
function construct_Theta_from_x_DDGFactplus(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-8,
)
    if psi <= 0.0
        error("The DGFact^+ associated-solution construction assumes psi > 0.")
    end

    Mhat = M_psi(xhat, F)

    eig = eigen(Symmetric(Mhat))
    lambda = eig.values
    Q = eig.vectors

    # Sort eigenvalues in nonincreasing order
    perm = sortperm(lambda, rev = true)
    lambda = lambda[perm]
    Q = Q[:, perm]

    k = length(lambda)

    if t < 1 || t > k
        error("t must satisfy 1 <= t <= size(Mhat, 1).")
    end

    # Clean tiny numerical negatives from PSD eigenvalues
    lambda[abs.(lambda) .<= atol] .= 0.0

    # y = lambda(Mhat) + psi * I_t
    y = copy(lambda)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t; atol = atol)

    beta = zeros(Float64, k)

    if iota > 0
        beta[1:iota] .= 1.0 ./ y[1:iota]
    end

    if iota < k
        beta[(iota + 1):k] .= 1.0 / mid
    end

    Theta = Symmetric(Q * Diagonal(beta) * Q')

    return (
        Theta = Theta,
        beta = beta,
        lambda = lambda,
        y = y,
        iota = iota,
        mid = mid,
    )
end

# ============================================================
# Objective value of a feasible DGFact^+ solution
#
# Fixed-variable DGFact^+ objective:
#   sum_{ell=k-t+1}^k h_psi(lambda_ell(Theta))
#   - upsilon'l + nu'c + tau*s
# ============================================================
function DGFactplus_objective_value(
    Theta::AbstractMatrix{Float64},
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    tau::Float64,
    s::Int,
    t::Int,
    psi::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-10,
)
    theta_eigs = sort(eigvals(Symmetric(Theta)), rev = true)
    k = length(theta_eigs)

    if t < 1 || t > k
        error("t must satisfy 1 <= t <= size(Theta, 1).")
    end

    smallest_t = theta_eigs[(k - t + 1):k]

    spectral_part = sum(h_psi_value(w, psi; atol = atol) for w in smallest_t)
    linear_part = -dot(upsilon, l) + dot(nu, c) + tau * s

    return (
        objective_value = spectral_part + linear_part,
        spectral_part = spectral_part,
        linear_part = linear_part,
    )
end

# ============================================================
# Full construction of a feasible DGFact^+ solution
# from a feasible DDGFact^+ solution xhat.
#
# Input:
#   xhat feasible for DDGFact^+
#   F such that C - psi*I = F*F'
#   t, psi
#
# Output:
#   Theta, upsilon, nu, tau feasible for DGFact^+
#
# Assumes:
#   - psi > 0
#   - no side constraints
#   - fixed-variable bounds l <= x <= c
#   - l,c are binary vectors
# ============================================================
function DGFactplus_dual_solution_from_DDGFactplus_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
)
    if psi <= 0.0
        error("DGFact^+ construction assumes psi > 0.")
    end

    n = length(xhat)

    if length(l) != n || length(c) != n
        error("xhat, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    theta_data = construct_Theta_from_x_DDGFactplus(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    Theta = theta_data.Theta

    # d = diag(F * Theta * F')
    d = vec(sum((F * Theta) .* F; dims = 2))

    sol_G = solve_GTheta_no_side_constraints(
        d,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    obj = DGFactplus_objective_value(
        Theta,
        sol_G.upsilon,
        sol_G.nu,
        sol_G.tau,
        s,
        t,
        psi;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        Theta = Theta,
        upsilon = sol_G.upsilon,
        nu = sol_G.nu,
        tau = sol_G.tau,
        objective_value = obj.objective_value,
        spectral_part = obj.spectral_part,
        linear_part = obj.linear_part,
        d = d,
        xstar_GTheta = sol_G.xstar,
        beta = theta_data.beta,
        lambda = theta_data.lambda,
        y = theta_data.y,
        iota = theta_data.iota,
        mid = theta_data.mid,
        sigma = sol_G.sigma,
        P = sol_G.P,
        Q = sol_G.Q,
        central = sol_G.central,
        residual = sol_G.residual,
    )
end

# ============================================================
# Solve G_Upsilon(Theta_hat) by LP using Gurobi
#
# Given:
#   d = diag(F * Theta_hat * F')
#   q = log.(gamma)
#
# Solve:
#   min  -upsilon'l + nu'c + tau*s + alpha*t
#   s.t. d + upsilon - nu + rho - tau*e = 0
#        -q + eta - rho - alpha*e = 0
#        upsilon >= 0, nu >= 0, eta >= 0, rho >= 0
#
# This is the auxiliary LP from DGFact^+_Upsilon.
# ============================================================
function solve_GTheta_upsilon_with_gurobi(
    d::Vector{Float64},
    gamma::Vector{Float64},
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(d)),
    c::Vector{Float64} = ones(length(d)),
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    n = length(d)

    if length(gamma) != n || length(l) != n || length(c) != n
        error("d, gamma, l, and c must have the same length.")
    end

    if any(gamma .<= 0.0)
        error("All entries of gamma must be strictly positive.")
    end

    if any(l .> c .+ atol)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    if sum(l) > s + atol || sum(c) < s - atol
        error("The fixed-variable polytope is empty: need sum(l) <= s <= sum(c).")
    end

    q = log.(gamma)

    model = Model(Gurobi.Optimizer)

    if silent
        set_optimizer_attribute(model, "OutputFlag", 0)
    end

    @variable(model, upsilon[1:n] >= 0.0)
    @variable(model, nu[1:n] >= 0.0)
    @variable(model, eta[1:n] >= 0.0)
    @variable(model, rho[1:n] >= 0.0)

    # Free variables
    @variable(model, tau)
    @variable(model, alpha)

    @constraint(
        model,
        [i in 1:n],
        d[i] + upsilon[i] - nu[i] + rho[i] - tau == 0.0
    )

    @constraint(
        model,
        [i in 1:n],
        -q[i] + eta[i] - rho[i] - alpha == 0.0
    )

    @objective(
        model,
        Min,
        -sum(upsilon[i] * l[i] for i in 1:n)
        + sum(nu[i] * c[i] for i in 1:n)
        + tau * s
        + alpha * t
    )

    optimize!(model)

    status = termination_status(model)

    if status != MOI.OPTIMAL
        error("GTheta_upsilon LP was not solved to optimality. Status = $status.")
    end

    upsilon_val = value.(upsilon)
    nu_val = value.(nu)
    eta_val = value.(eta)
    rho_val = value.(rho)
    tau_val = value(tau)
    alpha_val = value(alpha)

    # Clean tiny numerical values
    upsilon_val[abs.(upsilon_val) .<= atol] .= 0.0
    nu_val[abs.(nu_val) .<= atol] .= 0.0
    eta_val[abs.(eta_val) .<= atol] .= 0.0
    rho_val[abs.(rho_val) .<= atol] .= 0.0

    residual_x = d + upsilon_val - nu_val + rho_val .- tau_val
    residual_y = -q + eta_val - rho_val .- alpha_val

    # if maximum(abs.(residual_x)) > 1e-6
    #     error("x-dual equality residual too large: $(maximum(abs.(residual_x))).")
    # end

    # if maximum(abs.(residual_y)) > 1e-6
    #     error("y-dual equality residual too large: $(maximum(abs.(residual_y))).")
    # end

    # if minimum(upsilon_val) < -atol
    #     error("Constructed upsilon is not nonnegative: $(minimum(upsilon_val))")
    # end

    # if minimum(nu_val) < -atol
    #     error("Constructed nu is not nonnegative: $(minimum(nu_val))")
    # end

    # if minimum(eta_val) < -atol
    #     error("Constructed eta is not nonnegative: $(minimum(eta_val))")
    # end

    # if minimum(rho_val) < -atol
    #     error("Constructed rho is not nonnegative: $(minimum(rho_val))")
    # end

    linear_value =
        -dot(upsilon_val, l) +
        dot(nu_val, c) +
        tau_val * s +
        alpha_val * t

    return (
        upsilon = upsilon_val,
        nu = nu_val,
        eta = eta_val,
        rho = rho_val,
        tau = tau_val,
        alpha = alpha_val,
        linear_value = linear_value,
        objective_value = objective_value(model),
        residual_x = residual_x,
        residual_y = residual_y,
        status = status,
    )
end

# ============================================================
# Objective value of a feasible DGFact^+_Upsilon solution
#
# Fixed-variable DGFact^+_Upsilon objective:
#   sum_{ell=k-t+1}^k h_psi(lambda_ell(Theta))
#   - upsilon'l + nu'c + tau*s + alpha*t
# ============================================================
function DGFactplusUpsilon_objective_value(
    Theta::AbstractMatrix{Float64},
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    tau::Float64,
    alpha::Float64,
    s::Int,
    t::Int,
    psi::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-10,
)
    theta_eigs = sort(eigvals(Symmetric(Theta)), rev = true)
    k = length(theta_eigs)

    if t < 1 || t > k
        error("t must satisfy 1 <= t <= size(Theta, 1).")
    end

    smallest_t = theta_eigs[(k - t + 1):k]

    spectral_part = sum(h_psi_value(w, psi; atol = atol) for w in smallest_t)

    linear_part =
        -dot(upsilon, l) +
        dot(nu, c) +
        tau * s +
        alpha * t

    return (
        objective_value = spectral_part + linear_part,
        spectral_part = spectral_part,
        linear_part = linear_part,
    )
end

# ============================================================
# Full construction of a feasible DGFact^+_Upsilon solution
# from a feasible DDGFact^+_Upsilon solution.
#
# Input:
#   xhat feasible for DDGFact^+_Upsilon
#   gamma = Upsilon vector
#   F such that D_gamma^(1/2) C D_gamma^(1/2) - psi*I = F*F'
#   t, psi
#
# Optional:
#   yhat can be passed only for diagnostics.
# ============================================================
function DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
    xhat::Vector{Float64},
    gamma::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64;
    yhat = nothing,
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    if psi <= 0.0
        error("DGFact^+_Upsilon construction assumes psi > 0.")
    end

    n = length(xhat)

    if length(gamma) != n || length(l) != n || length(c) != n
        error("xhat, gamma, l, and c must have the same length.")
    end

    if any(gamma .<= 0.0)
        error("All entries of gamma must be strictly positive.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    if abs(sum(xhat) - s) > 1e-5
        @warn "xhat does not satisfy e'x = s within tolerance." sum_xhat = sum(xhat) s = s
    end

    theta_data = construct_Theta_from_x_DDGFactplus(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    Theta = theta_data.Theta

    # d = diag(F * Theta * F')
    d = vec(sum((F * Theta) .* F; dims = 2))

    sol_G = solve_GTheta_upsilon_with_gurobi(
        d,
        gamma,
        s,
        t;
        l = l,
        c = c,
        atol = atol,
        silent = silent,
    )

    obj = DGFactplusUpsilon_objective_value(
        Theta,
        sol_G.upsilon,
        sol_G.nu,
        sol_G.tau,
        sol_G.alpha,
        s,
        t,
        psi;
        l = l,
        c = c,
        atol = atol,
    )

    q = log.(gamma)

    primal_linear_value =
        yhat === nothing ? nothing : dot(d, xhat) - dot(q, yhat)

    return (
        Theta = Theta,
        upsilon = sol_G.upsilon,
        nu = sol_G.nu,
        eta = sol_G.eta,
        rho = sol_G.rho,
        tau = sol_G.tau,
        alpha = sol_G.alpha,
        objective_value = obj.objective_value,
        spectral_part = obj.spectral_part,
        linear_part = obj.linear_part,
        d = d,
        q = q,
        beta = theta_data.beta,
        lambda = theta_data.lambda,
        y = theta_data.y,
        iota = theta_data.iota,
        mid = theta_data.mid,
        residual_x = sol_G.residual_x,
        residual_y = sol_G.residual_y,
        primal_linear_value = primal_linear_value,
        Gtheta_status = sol_G.status,
    )
end