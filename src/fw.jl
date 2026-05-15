using LinearAlgebra

# ============================================================
# Exact line search
# ============================================================
function fw_exact_line_search(
    x::Vector{Float64},
    d::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64;
    γmax::Float64,
    tol::Float64 = 1e-5,
    maxiter::Int = 50,
)
    γlo, γhi = 0.0, γmax

    for _ in 1:maxiter
        γ = 0.5 * (γlo + γhi)

        xγ = x .+ γ .* d
        Mγ = M_psi(xγ, At)
        _, gradγ = spectral_value_gradient_Gamma_t(Mγ, At, t, t_a)

        deriv = dot(gradγ, d)

        if abs(deriv) ≤ tol
            return γ
        elseif deriv > 0
            γlo = γ
        else
            γhi = γ
        end
    end

    return 0.5 * (γlo + γhi)
end

# ============================================================
# Frank-Wolfe from precomputed A(psi)
# ============================================================
function fw_gaug_fact_from_At(
    At::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    t_a::Float64;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
    maxiter::Int = 10_000,
)
    n = size(At, 2)

    # Initial feasible point
    x = fill(s / n, n)
    v = zeros(n)
    gap = Inf

    for k in 1:maxiter
        # --- Build M(x) ---
        M = M_psi(x, At)

        # --- Paper subgradient ---
        _, grad = spectral_value_gradient_Gamma_t(M, At, t, t_a)

        # --- Linear minimization oracle ---
        idx = partialsortperm(grad, 1:s; rev = true)
        fill!(v, 0.0)
        v[idx] .= 1.0

        # --- FW gap ---
        d = v .- x
        gap = dot(grad, d)

        if gap ≤ tol
            return x, gap, k
        end

        # --- Step size (standard) ---
        if !line_search
            γ = 2.0 / (k + 2)
        else
            γ = fw_exact_line_search(x, d, At, t, t_a; γmax = 1.0)
        end

        # --- Update ---
        x .+= γ .* d
    end

    @warn "FW reached maxiter" gap maxiter t

    return x, gap, maxiter
end

# ============================================================
# Frank-Wolfe
# ============================================================
function fw_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
    maxiter::Int = 10_000,
)
    At = cholesky(Symmetric(C - t_a * I)).U

    return fw_gaug_fact_from_At(
        At,
        s,
        t,
        t_a;
        tol = tol,
        line_search = line_search,
        maxiter = maxiter,
    )
end