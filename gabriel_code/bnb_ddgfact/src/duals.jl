using LinearAlgebra

# =============================================================================
# DDFact closed-form dual for GMESP - certified upper bound and dual solutions
# =============================================================================

# Closed-form (τ, ν, υ): a true dual feasible solution for the Γ_t problem.
function compute_ddfact_dual_feasible_soln(dx_eff::Vector{Float64},
                                           xL::Vector{Float64}, xU::Vector{Float64},
                                           s::Int, n::Int)
    sigma = sortperm(dx_eff, rev=true)
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
    tau = (phi < n) ? dx_eff[sigma[phi + 1]] : 0.0
    nu  = zeros(n)
    ups = zeros(n)
    for i in 1:phi
        nu[sigma[i]] = dx_eff[sigma[i]] - tau
    end
    for i in (phi + 2):n
        ups[sigma[i]] = tau - dx_eff[sigma[i]]
    end
    return tau, nu, ups
end

# DDGFact dual construction 
#
# Primal (GMESP, no Ax≤b):  max Γ_t(F(x)),  s.t. eᵀx = s,  l ≤ x ≤ c.
#   dual_obj = Γ_t(F(x̂)) + νᵀc − υᵀl + τs − t      (certified upper bound)
#   gap      = νᵀc − υᵀl + τs − t ≥ 0.
# Returns nu, ups too — the dual variables used for variable fixing.
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

    # (τ̂, ν̂, υ̂): true dual feasible solution for the Γ_t problem
    tau, nu, ups = compute_ddfact_dual_feasible_soln(diag_FThetaFt, xL, xU, Int(s), n)

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
