using LinearAlgebra
using Arpack

# ============================================================
# Compute F such that C - psi*I = F*F'
# ============================================================
function compute_F(
    C::AbstractMatrix{Float64};
    psi::Float64 = 0.0,
    atol::Float64 = 1e-8,
)
    psi = max(0.0, psi)

    Cpsi = Symmetric(Matrix(C) - psi * I)
    eig = eigen(Cpsi)

    λ = eig.values
    Q = eig.vectors

    perm = sortperm(λ, rev = true)
    λ = λ[perm]
    Q = Q[:, perm]

    if minimum(λ) < -atol
        error("C - psi*I is not positive semidefinite. Minimum eigenvalue = $(minimum(λ)).")
    end

    λ = map(v -> abs(v) <= atol ? 0.0 : v, λ)

    F = Q * Diagonal(sqrt.(λ))

    return F
end

# ============================================================
# M_psi(x) = F' * Diagonal(x) * F
# ============================================================
function M_psi(x::Vector{Float64}, F::AbstractMatrix{Float64})
    if length(x) != size(F, 1)
        error("Dimension mismatch: length(x) must equal the number of rows of F.")
    end

    return Symmetric(F' * (F .* reshape(x, :, 1)))
end

# ============================================================
# Iota index for phi_t
# Input y must be sorted in nonincreasing order
# ============================================================
function find_iota(
    y::Vector{Float64},
    t::Int;
    atol::Float64 = 1e-10,
)
    if t < 1 || t > length(y)
        error("t must satisfy 1 <= t <= length(y).")
    end

    tail_sum = sum(y)
    mid = tail_sum / t

    if mid >= y[1] - atol
        return 0, mid
    end

    for iota in 1:(t - 1)
        tail_sum -= y[iota]
        mid = tail_sum / (t - iota)

        if y[iota] > mid + atol && mid >= y[iota + 1] - atol
            return iota, mid
        end
    end

    error("No valid iota found. Check that y is sorted and nonnegative.")
end

# ============================================================
# f_t(M_psi(x); psi)
# Original spectral objective, not the concave envelope.
# ============================================================
function f_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    M = M_psi(x, F)

    λ = eigvals(M)
    λs = sort(λ, rev = true)

    return sum(log.(λs[1:t] .+ psi))
end

function f_t(
    x::Vector{Float64},
    C::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)
    return f_t_from_F(x, F, t, psi)
end

# ============================================================
# Gamma_t(M_psi(x); psi)
# Concave envelope objective used in DDGFact^+
# ============================================================
function Gamma_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    M = M_psi(x, F)

    λ = eigvals(M)
    λs = sort(λ, rev = true)

    y = copy(λs)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t)

    val = 0.0

    if iota > 0
        val += sum(log.(y[1:iota]))
    end

    val += (t - iota) * log(mid)

    return val
end

function Gamma_t(
    x::Vector{Float64},
    C::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)
    return Gamma_t_from_F(x, F, t, psi)
end

# ============================================================
# Spectral Bound: sum_{ell=1}^t log(lambda_ell(C))
# ============================================================
function spectral_bound(
    C::AbstractMatrix{Float64},
    t::Int,
)
    λ = eigvals(Symmetric(C))
    λs = sort(λ, rev = true)

    return sum(log.(λs[1:t]))
end

# ============================================================
# Closed-form solution of DDGFact^+ for t = 1
# Selects the s largest row norms of F
# ============================================================
function closed_form_t1_from_F(
    F::AbstractMatrix{Float64},
    s::Int,
)
    n = size(F, 1)

    if s < 1 || s > n
        error("s must satisfy 1 <= s <= n.")
    end

    row_norms = vec(sum(abs2, F; dims = 2))
    S_star = partialsortperm(row_norms, 1:s; rev = true)

    x = zeros(Float64, n)
    x[S_star] .= 1.0

    return x, S_star, row_norms
end

function closed_form_t1(
    C::AbstractMatrix{Float64},
    s::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)
    return closed_form_t1_from_F(F, s)
end

# ============================================================
# Spectral subgradient of Gamma_t at M
# ============================================================
function spectral_subgradient_Gamma_t(
    M::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    eig = eigen(Symmetric(M))
    λ = eig.values
    Q = eig.vectors

    perm = sortperm(λ, rev = true)
    λs = λ[perm]
    Qs = Q[:, perm]

    y = copy(λs)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t; atol = atol)

    k = length(λs)
    g = zeros(Float64, k)

    if iota > 0
        g[1:iota] .= 1.0 ./ y[1:iota]
    end

    if iota < k
        g[(iota + 1):k] .= 1.0 / mid
    end

    G = Qs * Diagonal(g) * Qs'

    return Symmetric(G)
end

# ============================================================
# Subgradient of x -> Gamma_t(M_psi(x); psi)
# ============================================================
function x_subgradient_Gamma_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    M = M_psi(x, F)
    G = spectral_supergradient_Gamma_t(M, t, psi; atol = atol)

    grad = vec(sum((F * G) .* F; dims = 2))

    return grad
end

function x_supergradient_Gamma_t(
    x::Vector{Float64},
    C::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    F = compute_F(C; psi = psi)
    return x_supergradient_Gamma_t_from_F(x, F, t, psi; atol = atol)
end