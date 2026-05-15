using LinearAlgebra, Arpack

# ============================================================
# A(t_a) factorization
# ============================================================
function compute_At(C::Matrix{Float64}, t_a::Float64)
    F = cholesky(Symmetric(C - t_a * I))
    return F.U
end

function compute_At(C::Symmetric{Float64, <:AbstractMatrix}, t_a::Float64)
    F = cholesky(Symmetric(Matrix(C) - t_a * I))
    return F.U
end

# ============================================================
# M_{\psi}(x) = A(\psi) * Diagonal(x) * A(\psi)'
# ============================================================
function M_psi(x::Vector{Float64}, At::AbstractMatrix{Float64})
    return At * (At' .* x)
end

# ============================================================
# Iota index for phi_t
# ============================================================
function find_iota(
    λ::Vector{Float64},
    t::Int;
    atol::Float64 = 1e-10,
)
    tail_sum = sum(λ)
    mid = tail_sum / t

    if mid >= λ[1] - atol
        return 0, mid
    end

    for k in 1:t-1
        tail_sum -= λ[k]
        mid = tail_sum / (t - k)

        if λ[k] > mid + atol && mid >= λ[k+1] - atol
            return k, mid
        end
    end

    error("Something is wrong. No k satisfies the condition.")
end

# ============================================================
# f_t Function
# ============================================================
function f_t(
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64,
)
    # Build M_{\psi}(x) = A(\psi) * Diag(x) * A(\psi)'
    M = M_psi(x, At)

    # Compute the t largest eigenvalues using Arpack
    λ, _ = eigs(Symmetric(M), nev = t, which = :LM)
    λ = real.(λ)

    return sum(log.(λ .+ t_a))
end

# ============================================================
# GAug-Fact Objective Function
# ============================================================
function Gamma_t_from_At(
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64,
)
    M = M_psi(x, At)
    λ = eigvals(Symmetric(M))
    λs = sort(λ, rev = true)

    y = copy(λs)
    y[1:t] .+= t_a

    iota, mid = find_iota(y, t)

    if iota > 0
        val = sum(log.(y[1:iota]))
    else
        val = 0.0
    end

    val += (t - iota) * log(mid)

    return val
end

function Gamma_t(
    x::Vector{Float64},
    C::Matrix{Float64},
    t::Int,
    t_a::Float64,
)
    At = compute_At(C, t_a)
    return Gamma_t_from_At(x, At, t, t_a)
end

# ============================================================
# Spectral Bound
# ============================================================
function spectral_bound_util(
    C::AbstractMatrix{Float64},
    t::Int64,
)
    # Compute the t largest eigenvalues using Arpack
    λ, _ = eigs(Symmetric(C), nev = t, which = :LM)
    λ = real.(λ)

    # Sum of log of eigenvalues
    return sum(log.(λ))
end

# ============================================================
# Simplex Solution (closed-form for t=1)
# ============================================================
function simplex_sol(At::AbstractMatrix{Float64}, s::Int)
    n = size(At, 2)

    # Compute squared norm of each column
    col_norms = vec(sum(abs2, At; dims = 1))

    # Get indices of the s largest norms
    sorted_indices = partialsortperm(col_norms, 1:s; rev = true)
    S_star = sorted_indices[1:s]

    # Build solution vector
    x = zeros(n)
    x[S_star] .= 1.0

    return x
end

# ============================================================
# Spectral subgradient of Gamma_t
# ============================================================
function spectral_subgradient_Gamma_t(M::Matrix{Float64}, t::Int, t_a::Float64)
    # --- Eigen-decomposition ---
    eig = eigen(Symmetric(M))
    λ = eig.values
    U = eig.vectors
    n = length(λ)

    # --- Sort eigenvalues descending ---
    perm = sortperm(λ, rev = true)
    λs = Vector{Float64}(λ[perm])
    Us = U[:, perm]

    # --- Add t_a ---
    λs[1:t] .+= t_a

    # --- Determine k ---
    iota, mid = find_iota(λs, t)

    # --- Construct subgradient matrix ---
    Y = zeros(n, n)

    # Top iota eigenvectors
    if iota > 0
        Y .+= Us[:, 1:iota] * Diagonal(1 ./ λs[1:iota]) * Us[:, 1:iota]'
    end

    # Fractional weight for remaining eigenvectors
    if iota < n
        weight = 1.0 / mid
        U_tail = Us[:, iota+1:end]
        Y .+= U_tail * (weight * I(size(U_tail, 2))) * U_tail'
    end

    return Y
end

# ============================================================
# Spectral value and gradient of Gamma_t
# ============================================================
# function spectral_value_gradient_Gamma_t(
#     M::Matrix{Float64},
#     At::AbstractMatrix{Float64},
#     t::Int,
#     t_a::Float64,
# )
#     # --- Eigen-decomposition ---
#     eig = eigen(Symmetric(M))
#     λ = eig.values
#     U = eig.vectors
#     n = length(λ)

#     # --- Sort eigenvalues descending ---
#     perm = sortperm(λ, rev = true)
#     λs = Vector{Float64}(λ[perm])
#     Us = U[:, perm]

#     # --- Add t_a ---
#     λs[1:t] .+= t_a

#     # --- Determine iota ---
#     iota, mid = find_iota(λs, t)

#     # --- Compute objective value ---
#     if iota > 0
#         val = sum(log.(λs[1:iota]))
#     else
#         val = 0.0
#     end

#     val += (t - iota) * log(mid)

#     # --- Construct eigenvalue weights ---
#     weights = fill(1.0 / mid, n)

#     if iota > 0
#         weights[1:iota] .= 1.0 ./ λs[1:iota]
#     end

#     # --- Compute gradient without forming the full subgradient matrix ---
#     B = At' * Us
#     grad = vec(sum((B .^ 2) .* weights'; dims = 2))

#     return val, grad
# end

# ============================================================
# Spectral value and gradient of Gamma_t
# ============================================================
function spectral_value_gradient_Gamma_t(
    M::Matrix{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64,
)
    n = size(M, 1)

    # --- Top t eigenpairs only ---
    λ, U = eigs(Symmetric(M), nev = t, which = :LM)

    λ = real.(λ)
    U = real.(U)

    # --- Sort eigenvalues descending ---
    perm = sortperm(λ, rev = true)
    λtop = Vector{Float64}(λ[perm])
    Utop = U[:, perm]

    # --- Add t_a to the top t eigenvalues ---
    ytop = copy(λtop)
    ytop .+= t_a

    # --- Total sum of y ---
    # y_j = λ_j + t_a for j <= t
    # y_j = λ_j       for j > t
    total_sum_y = tr(M) + t * t_a

    # --- Determine iota ---
    iota = 0
    mid = total_sum_y / t

    if !(mid >= ytop[1] - 1e-10)
        prefix_sum = 0.0

        found = false

        for j in 1:t-1
            prefix_sum += ytop[j]
            mid = (total_sum_y - prefix_sum) / (t - j)

            if ytop[j] > mid + 1e-10 && mid >= ytop[j+1] - 1e-10
                iota = j
                found = true
                break
            end
        end

        if !found
            error("Something is wrong. No iota satisfies the condition.")
        end
    end

    # --- Compute objective value ---
    if iota > 0
        val = sum(log.(ytop[1:iota]))
    else
        val = 0.0
    end

    val += (t - iota) * log(mid)

    # --- Compute gradient without full eigendecomposition ---
    col_norms = vec(sum(abs2, At; dims = 1))
    grad = (1.0 / mid) .* col_norms

    if iota > 0
        U_iota = Utop[:, 1:iota]
        B = At' * U_iota

        correction_weights = (1.0 ./ ytop[1:iota]) .- (1.0 / mid)
        grad .+= vec(sum((B .^ 2) .* correction_weights'; dims = 2))
    end

    return val, grad
end
