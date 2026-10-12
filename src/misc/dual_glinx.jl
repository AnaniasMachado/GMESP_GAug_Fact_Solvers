using LinearAlgebra, JuMP, MosekTools
import MathOptInterface as MOI


function build_exact_dual_model(C, s::Integer, t::Real;
                                optimizer=Mosek.Optimizer, use_soc=true)
    H = factorize_OP(C, compact=false)
    n, p = size(C)

    model = Model(optimizer)
    set_silent(model)

    # Dual variables
    @variable(model, Θ[1:p, 1:p], Symmetric)
    @variable(model, T[1:n, 1:n], PSD)
    @variable(model, tau)
    @variable(model, nu)
    @variable(model, beta[1:n] >= 0)
    @variable(model, tlog)

    @constraint(model, [tlog; 1.0; triangle_vec(Θ)] in MOI.LogDetConeTriangle(n))

    @variable(model, M[1:n, 1:n], PSD)

    HΘH = H * Θ * transpose(H)

    if use_soc
        @variable(model, W[1:n, 1:n])
        @constraint(model, M .== 0.5 * (W + W') + T + HΘH + tau * Matrix(1.0I, n, n))
        for i in 1:n
            @constraint(model, [beta[i] + nu; W[i, 1:n]] in MOI.SecondOrderCone(n + 1))
        end
    else
        @constraint(model, M .== T + HΘH + tau * Matrix(1.0I, n, n))
    end

    @objective(model, Min, tr(Θ) - tlog - n + tau * t + tr(T) + nu * s + sum(beta))

    optimize!(model)
    return objective_value(model) + t * log(eigmax(C))
end


# ─────────────────────────────────────────────────────────────────────────────
# Alternating projections: repair solver (x, X) to true feasibility
#
#   C1: X ≽ 0              (PSD)
#   C2: tr(X) = t           (trace)
#   C3: sum(x) = s, 0 ≤ x ≤ 1  (box + sum)
#   C4: diag(x) - X ≽ 0    (coupling)
# ─────────────────────────────────────────────────────────────────────────────

function _project_psd!(X::Matrix{Float64})
    X .= 0.5 .* (X .+ X')
    F = eigen(Symmetric(X))
    X .= F.vectors * Diagonal(max.(F.values, 0.0)) * F.vectors'
    X .= 0.5 .* (X .+ X')
end

function _project_trace!(X::Matrix{Float64}, t_val::Real)
    n = size(X, 1)
    shift = (t_val - tr(X)) / n
    for i in 1:n
        X[i, i] += shift
    end
end

function _project_box_sum(x::Vector{Float64}, s::Real,
                          lbs::Vector{Float64}, ubs::Vector{Float64})
    n = length(x)
    λ_lo = minimum(x .- ubs) - 1.0
    λ_hi = maximum(x .- lbs) + 1.0
    for _ in 1:200
        λ_mid = 0.5 * (λ_lo + λ_hi)
        s_mid = sum(clamp(x[i] - λ_mid, lbs[i], ubs[i]) for i in 1:n)
        if s_mid > s + 1e-14
            λ_lo = λ_mid
        elseif s_mid < s - 1e-14
            λ_hi = λ_mid
        else
            break
        end
        if λ_hi - λ_lo < 1e-15
            break
        end
    end
    λ_star = 0.5 * (λ_lo + λ_hi)
    return [clamp(x[i] - λ_star, lbs[i], ubs[i]) for i in 1:n]
end

function _project_coupling!(x::Vector{Float64}, X::Matrix{Float64})
    W = diagm(x) - X
    W .= 0.5 .* (W .+ W')
    F = eigen(Symmetric(W))
    W_proj = F.vectors * Diagonal(max.(F.values, 0.0)) * F.vectors'
    X .= diagm(x) .- W_proj
    X .= 0.5 .* (X .+ X')
end

function project_relaxation_feasible(x_raw::Vector{Float64},
                                     X_raw::AbstractMatrix{Float64},
                                     s::Int, t_val::Int;
                                     lbs::Vector{Float64}=zeros(length(x_raw)),
                                     ubs::Vector{Float64}=ones(length(x_raw)),
                                     max_iter::Int=100,
                                     tol::Float64=1e-6)
    x = copy(x_raw)
    X = Matrix(0.5 .* (X_raw .+ X_raw'))

    for iter in 1:max_iter
        viol_psd  = max(-eigmin(Symmetric(X)), 0.0)
        viol_tr   = abs(tr(X) - t_val)
        viol_sum  = abs(sum(x) - s)
        viol_box  = max(maximum(max.(lbs .- x, 0.0)),
                        maximum(max.(x .- ubs, 0.0)))
        viol_coup = max(-eigmin(Symmetric(diagm(x) - X)), 0.0)
        max_viol  = max(viol_psd, viol_tr, viol_sum, viol_box, viol_coup)

        if max_viol <= tol
            return x, Symmetric(X), true, max_viol
        end

        _project_psd!(X)
        _project_trace!(X, t_val)
        x = _project_box_sum(x, s, lbs, ubs)
        _project_coupling!(x, X)
    end

    viol_psd  = max(-eigmin(Symmetric(X)), 0.0)
    viol_tr   = abs(tr(X) - t_val)
    viol_sum  = abs(sum(x) - s)
    viol_box  = max(maximum(max.(lbs .- x, 0.0)),
                    maximum(max.(x .- ubs, 0.0)))
    viol_coup = max(-eigmin(Symmetric(diagm(x) - X)), 0.0)
    max_viol  = max(viol_psd, viol_tr, viol_sum, viol_box, viol_coup)

    @warn "Alternating projections did not converge" max_iter max_viol
    return x, Symmetric(X), false, max_viol
end


# ─────────────────────────────────────────────────────────────────────────────
# Envelope construction: optimal (τ, ν, υ) from effective diagonal
# ─────────────────────────────────────────────────────────────────────────────

function _envelope_tau_nu_ups(dY_eff::Vector{Float64},
                              xL::Vector{Float64}, xU::Vector{Float64},
                              s::Int, n::Int)
    sigma = sortperm(dY_eff, rev=true)
    sum_lbs = sum(xL)
    cumsum_diff = 0.0
    phi = 0
    for j in 1:n
        cumsum_diff += xU[sigma[j]] - xL[sigma[j]]
        if sum_lbs + cumsum_diff <= s + 1e-10
            phi = j
        else
            break
        end
    end
    tau = (phi < n) ? dY_eff[sigma[phi + 1]] : 0.0
    nu  = zeros(n)
    ups = zeros(n)
    for i in 1:phi
        nu[sigma[i]] = dY_eff[sigma[i]] - tau
    end
    for i in (phi + 2):n
        ups[sigma[i]] = tau - dY_eff[sigma[i]]
    end
    return tau, nu, ups
end


# ─────────────────────────────────────────────────────────────────────────────
# Extract Y from solver dual of diag(x) - X ≽ 0
# scale=2.0 because the glinx bound uses 0.5*logdet
# ─────────────────────────────────────────────────────────────────────────────

function _extract_solver_Y(model::Model, n::Int; scale::Float64=2.0)
    Y_raw = dual(model[:con_psd_Y])
    if Y_raw isa AbstractVector
        Ymat = zeros(n, n)
        k = 1
        for col in 1:n
            for row in col:n
                Ymat[row, col] = Y_raw[k]
                Ymat[col, row] = Y_raw[k]
                k += 1
            end
        end
        return Symmetric(Ymat / scale)
    else
        return Symmetric((1.0 / (2.0 * scale)) * (Y_raw + Y_raw'))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Dual bound from truly feasible primal + solver Y
#
#   Θ̂ = ½L⁻¹     where  L = γCX_feasC + I - X_feas
#   ξ = -λ_min(Y - γCΘ̂C + Θ̂)
#   τ, ν, υ  from  diag(Y)
#
#   dual_obj = -½logdet(2Θ̂) + tr(Θ̂) + ν'u - υ'l + τs + ξt - ½n - ½t·log(γ)
# ─────────────────────────────────────────────────────────────────────────────

function _compute_dual_from_feasible_primal(X_feas::Symmetric,
                                            Y_solver::Symmetric,
                                            C::AbstractMatrix,
                                            gamma::Float64,
                                            n::Int, s::Int, t::Int,
                                            xL::Vector{Float64},
                                            xU::Vector{Float64})
    L = gamma * C * X_feas * C + I - X_feas
    L = Symmetric(0.5 * (L + L'))

    λ_min_L = eigmin(L)
    if λ_min_L < 1e-10
        δ = 1e-8 - λ_min_L
        L = Symmetric(L + δ * I)
    end

    Linv = inv(L)
    Linv = Symmetric(0.5 * (Linv + Linv'))
    Theta = Symmetric(0.5 * Linv)

    Y = copy(Matrix(Y_solver))
    if eigmin(Symmetric(Y)) < -1e-10
        FY = eigen(Symmetric(Y))
        Y = FY.vectors * Diagonal(max.(FY.values, 0.0)) * FY.vectors'
    end
    Y = Symmetric(0.5 * (Y + Y'))

    M = Symmetric(Matrix(Y) - gamma * C * Theta * C + Matrix(Theta))
    xi = -eigmin(M)

    tau, nu, ups = _envelope_tau_nu_ups(diag(Y), xL, xU, s, n)

    logdet_2T = logdet(2.0 * Theta)
    dual_obj = -0.5 * logdet_2T + tr(Theta) +
               dot(nu, xU) - dot(ups, xL) +
               tau * s + xi * t -
               0.5 * n - 0.5 * t * log(gamma)

    primal_obj = 0.5 * logdet(L) - 0.5 * t * log(gamma)

    return dual_obj, primal_obj
end


# ─────────────────────────────────────────────────────────────────────────────
# Compute glinx duality gap (using feasible primal from alternating projections)
#
#   x, X, z_rel, model = linx_gmesp(C, s, t; gamma=γ, return_model=true)
#   gap = compute_glinx_dual_gap(model, x, X, C, gamma, s, t)
#
# Returns dual_gap = dual_obj - primal_obj_feas ≥ 0 (should be small).
# primal_obj_feas is the objective at the projected feasible point.
# ─────────────────────────────────────────────────────────────────────────────

function compute_glinx_dual_gap(model::Model, x::Vector{Float64},
                                X::AbstractMatrix{Float64},
                                C::AbstractMatrix, gamma::Float64,
                                s::Int, t::Int)
    n = size(C, 1)
    xL = zeros(n)
    xU = ones(n)

    Y = _extract_solver_Y(model, n)

    x_feas, X_feas, converged, max_viol = project_relaxation_feasible(
        x, Matrix(X), s, t)
    if !converged
        @warn "Feasibility projection did not converge" max_viol
    end

    dual_obj, primal_obj = _compute_dual_from_feasible_primal(
        X_feas, Y, C, gamma, n, s, t, xL, xU)

    dual_gap = dual_obj - primal_obj
    if dual_gap < -1e-6
        @warn "Negative duality gap (numerical issue)" dual_gap dual_obj primal_obj
    end

    return dual_gap
end


# ─────────────────────────────────────────────────────────────────────────────
# Extract Ω from solver dual of X ≽ 0 (named constraint con_psd_X).
# scale=2.0 because glinx uses 0.5*logdet.
# ─────────────────────────────────────────────────────────────────────────────

function _extract_solver_Omega(model::Model, n::Int; scale::Float64=2.0)
    Ω_raw = dual(model[:con_psd_X])
    if Ω_raw isa AbstractVector
        Ωmat = zeros(n, n)
        k = 1
        for col in 1:n
            for row in col:n
                Ωmat[row, col] = Ω_raw[k]
                Ωmat[col, row] = Ω_raw[k]
                k += 1
            end
        end
        return Symmetric(Ωmat / scale)
    else
        return Symmetric((1.0 / (2.0 * scale)) * (Ω_raw + Ω_raw'))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# GLINX dual construction (paper procedure).
#
#   1. Use the raw solver primal (x̂, X̂).
#   2. Θ̂ := ½ (γ C X̂ C + I − X̂)⁻¹.
#   3. Extract Ẑ (dual of diag(x)−X ≽ 0) and project it onto the PSD cone.
#   4. π̂ = 0, Ŵ = 0, η̂ = 0 (no Ax≤b, no SOC).
#   5. (τ̂, ν̂, υ̂) from the diag(Ẑ) envelope so that
#         υ̂ − ν̂ − τ̂·e + diag(Ẑ) = 0.
#   6. M := Ẑ − γ C Θ̂ C + Θ̂, ξ̂ := −λ_min(M), Ω̂ := M + ξ̂ I.
#      Small positive diagonal margins guard against eigenvalue roundoff.
#      Keeping the solver's Ω and shifting only ξ does NOT enforce equality.
#
# Returns a NamedTuple with the dual bound, the primal value at (x̂, X̂),
# the raw-primal gap, and numerical feasibility diagnostics. The caller must
# check/repair primal feasibility before interpreting the gap as a duality gap.
# ─────────────────────────────────────────────────────────────────────────────

function compute_glinx_dual_gap_v2(model::Model, x::Vector{Float64},
                                   X::AbstractMatrix{Float64},
                                   C::AbstractMatrix, gamma::Float64,
                                   s::Int, t::Int;
                                   xL::Vector{Float64}=zeros(length(x)),
                                   xU::Vector{Float64}=ones(length(x)),
                                   feasibility_tol::Float64=1e-8,
                                   return_certificate::Bool=false)
    n = size(C, 1)
    isfinite(gamma) && gamma > 0 || throw(ArgumentError("gamma must be finite and positive"))
    1 <= t <= s <= n || throw(ArgumentError("require 1 <= t <= s <= n"))
    all(xL .<= xU) && sum(xL) <= s <= sum(xU) ||
        throw(ArgumentError("infeasible box/sum bounds"))

    # 1. Raw primal
    X_raw = Symmetric(0.5 * (Matrix(X) + Matrix(X)'))

    # 2. Θ̂ from raw primal
    L = gamma * C * X_raw * C + I - X_raw
    L = Symmetric(0.5 * (Matrix(L) + Matrix(L)'))
    # Fail if L is not positive definite; do not silently change the primal
    # objective by regularizing L and report that value as the primal bound.
    Lfactor = cholesky(L)
    Linv = inv(Lfactor)
    Linv = Symmetric(0.5 * (Matrix(Linv) + Matrix(Linv)'))
    Theta = Symmetric(0.5 * Matrix(Linv))

    # 3. Extract Ẑ, project to PSD using the existing projection helper.
    Z_solver = _extract_solver_Y(model, n)
    Zmat = Matrix(Z_solver)
    _project_psd!(Zmat)
    z_margin = 100 * eps(Float64) * max(1.0, opnorm(Zmat, Inf))
    Z = Symmetric(Zmat + z_margin * I)

    # 5. Envelope (τ̂, ν̂, υ̂) from diag(Ẑ)
    tau, nu, ups = _envelope_tau_nu_ups(diag(Z), xL, xU, s, n)

    # 6. Construct Ω to satisfy stationarity, rather than reuse the solver Ω.
    Mmat = Matrix(Z) - gamma * Matrix(C) * Matrix(Theta) * Matrix(C) + Matrix(Theta)
    M = Symmetric(0.5 * (Mmat + Mmat'))
    omega_margin = 100 * eps(Float64) * max(1.0, opnorm(M, Inf))
    xi = -eigmin(M) + omega_margin
    Ω = Symmetric(Matrix(M) + xi * I)

    # Dual objective: π̂=0, Ŵ=0, η̂=0, and PSD-cone duals multiply zero RHS.
    logdet_2T = logdet(cholesky(Symmetric(2.0 * Matrix(Theta))))
    dual_obj = -0.5 * logdet_2T + tr(Theta) +
               dot(nu, xU) - dot(ups, xL) +
               tau * s + xi * t -
               0.5 * n - 0.5 * t * log(gamma)

    primal_obj = 0.5 * logdet(Lfactor) - 0.5 * t * log(gamma)

    # Residuals for dual feasibility (should be ≈0 up to rounding)
    res_x  = norm(ups .- nu .- tau .* ones(n) .+ diag(Z), Inf)
    res_mat = norm(-Matrix(Ω) + Mmat + xi * I, Inf)
    min_eig_Z = eigmin(Z)
    min_eig_Omega = eigmin(Ω)
    min_eig_Theta = eigmin(Theta)
    dual_feasible = all(isfinite, (dual_obj, res_x, res_mat, min_eig_Z,
                                  min_eig_Omega, min_eig_Theta)) &&
                    max(res_x, res_mat) <= feasibility_tol &&
                    min(min_eig_Z, min_eig_Omega, minimum(nu), minimum(ups)) >= 0 &&
                    min_eig_Theta > 0

    gap = dual_obj - primal_obj
    if gap < -1e-6
        @warn "Negative duality gap (numerical issue)" gap dual_obj primal_obj
    end

    result = (dual_obj=dual_obj, primal_obj=primal_obj, gap=gap,
              xi=xi, tau=tau, res_x=res_x, res_mat=res_mat,
              min_eig_Z=min_eig_Z, min_eig_Omega=min_eig_Omega,
              min_eig_Theta=min_eig_Theta, dual_feasible=dual_feasible,
              z_margin=z_margin, omega_margin=omega_margin)
    return return_certificate ? merge(result, (Theta=Theta, Z=Z, Omega=Ω,
                                               nu=nu, ups=ups)) : result
end


# ─────────────────────────────────────────────────────────────────────────────
# GNLP-ID / GNLP-ID-comp dual construction (paper procedure).
#
#   GNLP-ID  (comp=false):  primal  max  logdet(I − HᵀXH) + t log λ₁(C)
#     I − HHᵀ = (1/λ₁) C.  Θ̂ := (I − HᵀX̂H)⁻¹  (size p×p).
#     stationarity:  −Ω + Z + H Θ Hᵀ + ξ I = 0
#     ξ̂ := −λ_min(−Ω̂ + Ẑ + H Θ̂ Hᵀ)
#     dual_obj = −logdet(Θ̂) + tr(Θ̂) − p + νᵀu − υᵀl + τs + ξt + t log λ₁(C)
#
#   GNLP-ID-comp (comp=true):  primal  max  logdet(I + GᵀXG) + t log λₙ(C)
#     I + GGᵀ = (1/λₙ) C.  Θ̂ := (I + GᵀX̂G)⁻¹  (size q×q).
#     stationarity:  −Ω + Z − G Θ Gᵀ + ξ I = 0
#     ξ̂ := −λ_min(−Ω̂ + Ẑ − G Θ̂ Gᵀ)
#     dual_obj = −logdet(Θ̂) + tr(Θ̂) − q + νᵀu − υᵀl + τs + ξt + t log λₙ(C)
#
# Procedure (analogous to GLINX):
#   1. Raw primal (x̂, X̂) from the solver.
#   2. Θ̂ = (I ∓ FᵀX̂F)⁻¹ with F = H (non-comp) or F = G (comp).
#   3. Extract Ẑ, Ω̂ (no ½-rescaling: primal has no ½ factor), project to PSD.
#   4. π̂ = 0, Ŵ = 0, η̂ = 0 (no Ax≤b, no SOC).
#   5. (τ̂, ν̂, υ̂) envelope from diag(Ẑ).
#   6. ξ̂ from λ_min of the matrix-stationarity residual.
# ─────────────────────────────────────────────────────────────────────────────

function compute_nlpid_dual_gap(model::Model, x::Vector{Float64},
                                X::AbstractMatrix{Float64},
                                C::AbstractMatrix, s::Int, t::Int;
                                comp::Bool=false,
                                xL::Vector{Float64}=zeros(length(x)),
                                xU::Vector{Float64}=ones(length(x)))
    n = size(C, 1)
    Csym = C isa Symmetric ? C : Symmetric(Matrix(C))

    # Factorize: F = H for comp=false, F = G for comp=true
    if comp
        F = factorize_OP_comp(Csym)
        λ_extr = eigmin(Csym)
    else
        F = factorize_OP(Csym)
        λ_extr = eigmax(Csym)
    end
    q = size(F, 2)

    # 1. Raw primal
    X_raw = Symmetric(0.5 * (Matrix(X) + Matrix(X)'))

    # 2. Θ̂ from raw primal (q×q)
    FtXF = Matrix(F)' * Matrix(X_raw) * Matrix(F)
    if comp
        E = I + FtXF
    else
        E = I - FtXF
    end
    E = Symmetric(0.5 * (Matrix(E) + Matrix(E)'))
    λ_min_E = eigmin(E)
    if λ_min_E < 1e-12
        δ = 1e-10 - λ_min_E
        E = Symmetric(Matrix(E) + δ * I)
    end
    Einv = inv(E)
    Theta = Symmetric(0.5 * (Matrix(Einv) + Matrix(Einv)'))

    # 3. Extract Ẑ, Ω̂ with scale=1 (primal has no ½ factor) and project to PSD
    Z_solver = _extract_solver_Y(model, n; scale=1.0)
    FZ = eigen(Symmetric(Matrix(Z_solver)))
    Z = Symmetric(FZ.vectors * Diagonal(max.(FZ.values, 0.0)) * FZ.vectors')

    Ω_solver = _extract_solver_Omega(model, n; scale=1.0)
    FO = eigen(Symmetric(Matrix(Ω_solver)))
    Ω = Symmetric(FO.vectors * Diagonal(max.(FO.values, 0.0)) * FO.vectors')

    # 5. Envelope (τ̂, ν̂, υ̂) from diag(Ẑ)
    tau, nu, ups = _envelope_tau_nu_ups(diag(Z), xL, xU, s, n)

    # 6. ξ̂ from matrix stationarity residual
    FΘFt = Matrix(F) * Matrix(Theta) * Matrix(F)'
    if comp
        M = Symmetric(-Matrix(Ω) + Matrix(Z) - FΘFt)
    else
        M = Symmetric(-Matrix(Ω) + Matrix(Z) + FΘFt)
    end
    xi = -eigmin(M)

    # Dual and primal objectives
    logdet_Θ = logdet(Matrix(Theta))
    dual_obj = -logdet_Θ + tr(Theta) - q +
               dot(nu, xU) - dot(ups, xL) +
               tau * s + xi * t +
               t * log(λ_extr)

    primal_obj = logdet(Matrix(E)) + t * log(λ_extr)

    # Residuals
    res_x  = norm(ups .- nu .- tau .* ones(n) .+ diag(Z), Inf)
    res_mat = norm(Matrix(M) + xi * I, Inf)

    gap = dual_obj - primal_obj
    if gap < -1e-6
        @warn "Negative duality gap (numerical issue)" gap dual_obj primal_obj comp
    end

    return (dual_obj=dual_obj, primal_obj=primal_obj, gap=gap,
            xi=xi, tau=tau, res_x=res_x, res_mat=res_mat)
end


# ─────────────────────────────────────────────────────────────────────────────
# DDGFact dual construction (closed form; Nikolov / Weijun adaptation).
#
# Primal (GMESP, no Ax≤b):  max  Γ_t(F(x)),  s.t. eᵀx = s,  l ≤ x ≤ c.
#   F from factorize_matrix(C),  F(x) = Fᵀ diag(x) F,  size k×k.
#
# Given primal x̂ with spectral decomposition F(x̂) = Σ λ̂_ℓ û_ℓ û_ℓᵀ,
#   compute ι̂ via find_iota, and set δ̂ = (Σ_{ℓ>ι̂} λ̂_ℓ) / (t − ι̂).
# Define β̂_ℓ = 1/λ̂_ℓ for 1 ≤ ℓ ≤ ι̂, and β̂_ℓ = 1/δ̂ for ℓ > ι̂
#   (this ε = 0 limit is fine: the (1+ε)/δ̂ correction for ℓ > r̂ in the
#    paper only matters for strict Θ̂ ≻ 0; it doesn't change the dual value).
# Set Θ̂ = Σ β̂_ℓ û_ℓ û_ℓᵀ.  Then −Σ_{bottom t eig}log(λ(Θ̂)) = Γ_t(F(x̂)).
#
# (τ̂, ν̂, υ̂) from the diag(F Θ̂ Fᵀ) envelope (exact closed form for l≤x≤c).
#
#   dual_obj = Γ_t(F(x̂)) + νᵀc − υᵀl + τs − t
#   primal_obj = Γ_t(F(x̂))
#   gap = νᵀc − υᵀl + τs − t ≥ 0.
# ─────────────────────────────────────────────────────────────────────────────

function compute_ddfact_dual_gap(C::AbstractMatrix, x::Vector{Float64},
                                 s::Integer, t::Integer;
                                 xL::Vector{Float64}=zeros(length(x)),
                                 xU::Vector{Float64}=ones(length(x)))
    n = size(C, 1)
    Csym = C isa Symmetric ? C : Symmetric(Matrix(C))

    F = factorize_matrix(Csym)                # n × k
    FtxF = Symmetric(F' * Diagonal(x) * F)    # k × k
    eig = eigen(Symmetric(Matrix(FtxF)))
    perm = length(eig.values):-1:1
    λ = eig.values[perm]
    U = eig.vectors[:, perm]

    iota, mid = find_iota(λ, Int(t))

    fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
    fval += (t - iota) * log(mid)

    # β̂_ℓ  (= eigenvalues of Θ̂ in the û-basis, inverted appropriately)
    β = similar(λ)
    for ℓ in 1:iota
        β[ℓ] = 1.0 / λ[ℓ]
    end
    for ℓ in (iota + 1):length(λ)
        β[ℓ] = 1.0 / mid
    end

    # diag(F Θ̂ Fᵀ) = Σ_ℓ β̂_ℓ (F û_ℓ)ᵢ²
    FU = F * U
    diag_FThetaFt = vec(sum((FU .^ 2) .* β', dims=2))

    # envelope (τ̂, ν̂, υ̂) — exact closed form of the gap LP
    tau, nu, ups = _envelope_tau_nu_ups(diag_FThetaFt, xL, xU, Int(s), n)

    dual_obj = fval + dot(nu, xU) - dot(ups, xL) + tau * s - t
    primal_obj = fval
    gap = dual_obj - primal_obj

    # residual of diag(FΘFᵀ) + υ − ν − τe = 0
    res_x = norm(ups .- nu .- tau .* ones(n) .+ diag_FThetaFt, Inf)

    if gap < -1e-6
        @warn "Negative DDGFact dual gap (numerical issue)" gap dual_obj primal_obj
    end

    return (dual_obj=dual_obj, primal_obj=primal_obj, gap=gap,
            tau=tau, nu=nu, ups=ups, res_x=res_x, iota=iota, delta=mid)
end
