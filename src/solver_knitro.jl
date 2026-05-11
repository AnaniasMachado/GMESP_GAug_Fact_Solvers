using LinearAlgebra
using JuMP
using KNITRO

function spectral_bound(C::Symmetric{<:Real,<:AbstractMatrix}, t::Integer)
    λ = reverse(eigvals(C))
    return sum(log, @view λ[1:t])
end

function factorize_matrix(
    C::Symmetric{<:Real,<:AbstractMatrix};
    atol = 1e-8,
    ta = 0,
)
    ta = max(0, ta)
    λ, Q = eigen(C - ta * I)

    perm = length(λ):-1:1
    λ = λ[perm]
    Q = Q[:, perm]

    λ = map(x -> x < atol ? 0.0 : x, λ)
    F = Q * Diagonal(sqrt.(λ))
    return F
end

function find_iota(
    λ::Vector{Float64},
    t::Int64;
    atol = 1e-8,
    check::Bool = false,
)
    k = length(λ)

    if check
        @assert 1 <= t <= k
        @assert all(x -> x >= 0, λ)
        @assert issorted(λ; rev = true)
    end

    tail_sum = sum(λ)
    mid = tail_sum / t

    if mid >= λ[1] - atol
        return 0, mid
    end

    for iota in 1:t-1
        tail_sum -= λ[iota]
        mid = tail_sum / (t - iota)

        if (mid >= λ[iota+1] - atol) && (λ[iota] > mid + atol)
            return iota, mid
        end
    end

    error("Something is wrong. No ι satisfies the condition.")
end

function add_knitro_options!(model; silent::Bool = true)
    set_optimizer_attribute(model, "opttol", 1e-8)
    set_optimizer_attribute(model, "opttol_abs", 1e-8)
    set_optimizer_attribute(model, "feastol", 1e-8)
    set_optimizer_attribute(model, "feastol_abs", 1e-8)
end

function ddfact_gmesp(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Integer,
    t::Integer;
    atol = 1e-10,
    silent::Bool = true,
)
    n = size(C, 1)
    Fg = factorize_matrix(C)

    model = Model(KNITRO.Optimizer)
    add_knitro_options!(model; silent = silent)

    @variable(model, atol <= x[i = 1:n] <= 1 - atol)

    function gamma_value_and_gradient(xvec::Vector{Float64})
        X = Fg' * Diagonal(xvec) * Fg
        X = Symmetric(0.5 * (X + X'))

        λ, U_eig = eigen(X)

        perm = length(λ):-1:1
        λ = Vector{Float64}(λ[perm])
        U_eig = U_eig[:, perm]

        iota, mid_val = find_iota(λ, t)

        fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
        fval += (t - iota) * log(mid_val)

        eigDual = zeros(Float64, n)
        if iota > 0
            eigDual[1:iota] .= 1.0 ./ λ[1:iota]
        end
        eigDual[iota+1:end] .= 1.0 / mid_val

        K1 = Fg * U_eig
        grad = vec(sum(K1 .^ 2 .* eigDual', dims = 2))

        return fval, grad
    end

    function gamma_bound_gmesp_f(xargs...)
        xvec = collect(Float64, xargs)
        fval, _ = gamma_value_and_gradient(xvec)
        return fval
    end

    function gamma_bound_gmesp_∇f(g, xargs...)
        xvec = collect(Float64, xargs)
        _, grad = gamma_value_and_gradient(xvec)

        for i in 1:n
            g[i] = grad[i]
        end

        return nothing
    end

    register(
        model,
        :gamma_bound,
        n,
        gamma_bound_gmesp_f,
        gamma_bound_gmesp_∇f,
    )

    @NLobjective(model, Max, gamma_bound(x...))
    @constraint(model, sum(x) == s)

    optimize!(model)

    obj_val = objective_value(model)
    x_val = value.(x)

    return x_val, obj_val
end

function aug_ddfact_gmesp(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Integer,
    t::Integer;
    atol = 1e-10,
    silent::Bool = true,
)
    n = size(C, 1)

    I_t = zeros(Float64, n)
    I_t[1:t] .= 1.0

    ta = eigmin(C) - atol
    Fg = factorize_matrix(C; ta = ta)

    model = Model(KNITRO.Optimizer)
    add_knitro_options!(model; silent = silent)

    @variable(model, atol <= x[i = 1:n] <= 1 - atol)

    function aug_gamma_value_and_gradient(xvec::Vector{Float64})
        X = Fg' * Diagonal(xvec) * Fg
        X = Symmetric(0.5 * (X + X'))

        λ, U_eig = eigen(X)

        perm = length(λ):-1:1
        λ = Vector{Float64}(λ[perm] + ta * I_t)
        U_eig = U_eig[:, perm]

        iota, mid_val = find_iota(λ, t)

        fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
        fval += (t - iota) * log(mid_val)

        eigDual = zeros(Float64, n)
        if iota > 0
            eigDual[1:iota] .= 1.0 ./ λ[1:iota]
        end
        eigDual[iota+1:end] .= 1.0 / mid_val

        K1 = Fg * U_eig
        grad = vec(sum(K1 .^ 2 .* eigDual', dims = 2))

        return fval, grad
    end

    function aug_gamma_bound_gmesp_f(xargs...)
        xvec = collect(Float64, xargs)
        fval, _ = aug_gamma_value_and_gradient(xvec)
        return fval
    end

    function aug_gamma_bound_gmesp_∇f(g, xargs...)
        xvec = collect(Float64, xargs)
        _, grad = aug_gamma_value_and_gradient(xvec)

        for i in 1:n
            g[i] = grad[i]
        end

        return nothing
    end

    register(
        model,
        :aug_gamma_bound,
        n,
        aug_gamma_bound_gmesp_f,
        aug_gamma_bound_gmesp_∇f,
    )

    @NLobjective(model, Max, aug_gamma_bound(x...))
    @constraint(model, sum(x) == s)

    optimize!(model)

    obj_val = objective_value(model)
    x_val = value.(x)

    return x_val, obj_val
end