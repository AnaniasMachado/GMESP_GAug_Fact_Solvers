using LinearAlgebra
using Arpack

# ============================================================
# Compute F such that C - psi*I = F*F'
# ============================================================
function factorize_matrix(
    C::Symmetric{<:Real,<:AbstractMatrix};
    psi::Float64 = 0.0,
    atol::Float64 = 1e-8,
)
    psi = max(0.0, psi)

    Cpsi = Symmetric(Matrix(C) - psi * I)
    λ, Q = eigen(Cpsi)

    # Permutation for descending order
    perm = sortperm(λ, rev = true)
    λ = λ[perm]
    Q = Q[:, perm]

    # Clamp small near-zeros values before sqrt
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
    C::Symmetric{<:Real,<:AbstractMatrix},
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
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)
    return Gamma_t_from_F(x, F, t, psi)
end

# ============================================================
# DDGFact objective value at a fixed x
#
# This is Gamma_t(M_0(x); 0), i.e., the DDGFact relaxation
# objective evaluated at x.
# ============================================================
function DDGFact_value_at_x(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int;
    atol::Float64 = 1e-8,
)
    F = factorize_matrix(C; psi = 0.0, atol = atol)

    return Gamma_t_from_F(
        x,
        F,
        t,
        0.0,
    )
end


# ============================================================
# DDGFact^+ objective value at a fixed x
#
# This is Gamma_t(M_psi(x); psi), i.e., the DDGFact^+
# relaxation objective evaluated at x.
# ============================================================
function DDGFactplus_value_at_x(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-8,
)
    F = factorize_matrix(C; psi = psi, atol = atol)

    return Gamma_t_from_F(
        x,
        F,
        t,
        psi,
    )
end


# ============================================================
# DDGFact^+_Upsilon objective value from F
#
# Objective:
#
#   Gamma_t(M_{Upsilon,psi}(x); psi)
#       - sum_i log(gamma_i) y_i
#
# Here F satisfies
#
#   Diagonal(sqrt(gamma)) * C * Diagonal(sqrt(gamma)) - psi I = F F'
# ============================================================
function Gamma_t_upsilon_from_F(
    x::Vector{Float64},
    y::Vector{Float64},
    gamma::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    n = length(x)

    if length(y) != n || length(gamma) != n || size(F, 1) != n
        error("x, y, gamma, and F must have compatible dimensions.")
    end

    if any(gamma .<= 0.0)
        error("gamma must be strictly positive.")
    end

    log_gamma = log.(gamma)

    return Gamma_t_from_F(
        x,
        F,
        t,
        psi,
    ) - dot(log_gamma, y)
end


# ============================================================
# Best y for DDGFact^+_Upsilon at a binary x
#
# Given binary x and sum(y) = t, 0 <= y <= x, the term involving y is
#
#   - sum_i log(gamma_i) y_i.
#
# Thus, for fixed x, the best y selects the t indices in support(x)
# with the smallest log(gamma_i).
# ============================================================
function best_y_upsilon_at_binary_x(
    x::Vector{Float64},
    gamma::Vector{Float64},
    t::Int;
    atol::Float64 = 1e-8,
)
    n = length(x)

    if length(gamma) != n
        error("x and gamma must have the same length.")
    end

    if any(gamma .<= 0.0)
        error("gamma must be strictly positive.")
    end

    support = findall(i -> x[i] >= 1.0 - atol, 1:n)

    if t < 1 || t > length(support)
        error("Need 1 <= t <= number of selected variables in x.")
    end

    log_gamma = log.(gamma)

    selected_y =
        support[partialsortperm(log_gamma[support], 1:t; rev = false)]

    y = zeros(Float64, n)
    y[selected_y] .= 1.0

    return y
end


# ============================================================
# DDGFact^+_Upsilon objective value at a fixed binary x
#
# This evaluates
#
#   Gamma_t(M_{Upsilon,psi}(x); psi)
#       - sum_i log(gamma_i) y_i
#
# using the best y for the given binary x.
# ============================================================
function DDGFactplusUpsilon_value_at_x(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-8,
)
    n = length(x)

    if length(gamma) != n || size(C, 1) != n
        error("x, gamma, and C must have compatible dimensions.")
    end

    F = scaled_factorize_matrix(
        C,
        gamma,
        psi;
        atol = atol,
    )

    y = best_y_upsilon_at_binary_x(
        x,
        gamma,
        t;
        atol = atol,
    )

    val = Gamma_t_upsilon_from_F(
        x,
        y,
        gamma,
        F,
        t,
        psi,
    )

    return val, y, F
end

# ============================================================
# Spectral Bound: sum_{ell=1}^t log(lambda_ell(C))
# ============================================================
function spectral_bound(
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
)
    λ = reverse(eigvals(C))
    return sum(log, @view λ[1:t])
end

# ============================================================
# Closed-form solution of DDGFact^+ for t = 1
# Selects the s largest row norms of F
# ============================================================
function closed_form_t1_from_F(
    F::AbstractMatrix{Float64},
    s::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
)
    n = size(F, 1)

    if s < 1 || s > n
        error("s must satisfy 1 <= s <= n.")
    end

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    if any(i -> i < 1 || i > n, J1)
        error("All indices in J1 must satisfy 1 <= i <= n.")
    end

    if any(i -> i < 1 || i > n, J0)
        error("All indices in J0 must satisfy 1 <= i <= n.")
    end

    if !isempty(intersect(J1, J0))
        error("J1 and J0 must be disjoint.")
    end

    if length(J1) > s
        error("The fixed-one set J1 cannot have cardinality larger than s.")
    end

    Jfree = setdiff(1:n, union(J1, J0))
    m = s - length(J1)

    if m > length(Jfree)
        error("Not enough free variables to satisfy cardinality s.")
    end

    row_norms = vec(sum(abs2, F; dims = 2))

    S_free =
        m == 0 ? Int[] :
        Jfree[partialsortperm(row_norms[Jfree], 1:m; rev = true)]

    S_star = sort(union(J1, S_free))

    x = zeros(Float64, n)
    x[S_star] .= 1.0

    return x, S_star, row_norms
end

function closed_form_t1(
    C::Symmetric{<:Real,<:AbstractMatrix},
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
    G = spectral_subgradient_Gamma_t(M, t, psi; atol = atol)

    grad = vec(sum((F * G) .* F; dims = 2))

    return grad
end

function x_subgradient_Gamma_t(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    F = compute_F(C; psi = psi)
    return x_subgradient_Gamma_t_from_F(x, F, t, psi; atol = atol)
end