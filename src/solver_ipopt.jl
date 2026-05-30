function spectral_bound_solver(
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Integer
)
    λ = reverse(eigvals(C))
    return sum(log, @view λ[1:t])
end

function factorize_matrix(
    C::Symmetric{<:Real,<:AbstractMatrix};
    atol=1e-8,
    psi = 0
)
    psi = max(0,psi);          
    λ,Q = eigen(C - psi*I);
    perm = length(λ):-1:1           # permutation for descending order
    λ = λ[perm];
    Q = Q[:, perm];
    # clamp small near-zeros values before sqrt
    λ = map(x -> x < atol ? 0.0 : x, λ);
    F = Q * Diagonal(sqrt.(λ));
    return F
end

function find_iota(
    λ::Vector{Float64},
    t::Int64;
    atol=1e-5,
    check::Bool = false
)
    # check if λ satisfies conditions
    k = length(λ)
    if check
        @assert(1 <= t <= k)
        @assert(all(x -> x >= 0, λ))
        @assert(issorted(λ;rev=true))
    end
    # case iota = 0
    tail_sum = sum(λ);         # Σ_{ℓ=1}^k λ_ℓ
    mid      = tail_sum/t    # (1/(t-ι)) Σ_{ℓ=ι+1}^k λ_ℓ
    if mid >= (λ[1] - atol); return 0, mid; end; 
    # case iota > 0
    for iota in 1:t-1
        tail_sum -= λ[iota]         # Σ_{ℓ=ι+1}^k λ_ℓ
        mid       = tail_sum / (t - iota)   # (1/(t-ι)) Σ_{ℓ=ι+1}^k λ_ℓ  
        # check condition           
        if (mid >= λ[iota+1]- atol) && (λ[iota] > mid + atol)
            return iota, mid                         # iota and middle value
        end
    end
    error("Something is wrong. No ι satisfies the condition.")
end

function add_ipopt_options!(model)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-8)
    set_optimizer_attribute(model, "dual_inf_tol", 1e-8)
    set_optimizer_attribute(model, "compl_inf_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_constr_viol_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_iter", 0)
end


function ddfact_gmesp(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Integer,
    t::Integer;
    atol = 1e-10,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n

    model = Model(Ipopt.Optimizer)
    add_ipopt_options!(model)
    set_silent(model)

    @variable(model, atol <= x[i = 1:n] <= 1 - atol)
    @constraint(model, sum(x) == s)

    if t == 1
        # ------------------------------------------------------------
        # Exact differentiable t = 1 case:
        #
        # DDGFact has gamma = e and psi = 0.
        #
        # Gamma_1 = log(tr(X))
        #         = log(sum_i x_i C_ii)
        #
        # grad_i = C_ii / sum_j x_j C_jj
        # ------------------------------------------------------------
        cdiag = collect(diag(C))

        function gamma_bound_t1_f(x...)
            xvec = collect(x)
            denom = dot(xvec, cdiag)

            if denom <= atol
                error("Invalid t=1 DDGFact denominator: denom = $denom")
            end

            return log(denom)
        end

        function gamma_bound_t1_∇f(g, x...)
            xvec = collect(x)
            denom = dot(xvec, cdiag)

            if denom <= atol
                error("Invalid t=1 DDGFact denominator in gradient: denom = $denom")
            end

            for i in 1:n
                g[i] = cdiag[i] / denom
            end
        end

        register(
            model,
            :gamma_bound_t1,
            n,
            gamma_bound_t1_f,
            gamma_bound_t1_∇f,
        )

        @NLobjective(model, Max, gamma_bound_t1(x...))
    else
        # ------------------------------------------------------------
        # General t > 1 spectral objective/subgradient
        # ------------------------------------------------------------
        Fg = factorize_matrix(C)
        dx_cache = Ref(zeros(n))

        function gamma_bound_gmesp_f(x...)
            xvec = collect(x)

            X = Fg' * diagm(xvec) * Fg
            X = Symmetric(0.5 * (X + X'))

            λ, U_eig = eigen(X)

            perm = length(λ):-1:1
            λ = λ[perm]
            U_eig = U_eig[:, perm]

            iota, mid_val = find_iota(λ, t)

            fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
            fval += (t - iota) * log(mid_val)

            eigDual = zeros(n)

            if iota > 0
                eigDual[1:iota] .= 1.0 ./ λ[1:iota]
            end

            eigDual[iota+1:end] .= 1.0 / mid_val

            K1 = Fg * U_eig
            dx_cache[] = vec(sum(K1 .^ 2 .* eigDual', dims = 2))

            return fval
        end

        function gamma_bound_gmesp_∇f(g, x...)
            dx = dx_cache[]
            for i in 1:n
                g[i] = dx[i]
            end
        end

        register(
            model,
            :gamma_bound,
            n,
            gamma_bound_gmesp_f,
            gamma_bound_gmesp_∇f,
        )

        @NLobjective(model, Max, gamma_bound(x...))
    end

    optimize!(model)

    obj_val = objective_value(model)
    x_val = value.(x)

    return x_val, obj_val
end


function aug_ddfact_gmesp(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Integer,
    t::Integer,
    psi::Float64;
    atol = 1e-10,
)
    n = size(C, 1)
    @assert 1 <= t <= s <= n

    model = Model(Ipopt.Optimizer)
    add_ipopt_options!(model)
    set_silent(model)

    @variable(model, atol <= x[i = 1:n] <= 1 - atol)
    @constraint(model, sum(x) == s)

    if t == 1
        # ------------------------------------------------------------
        # Exact differentiable t = 1 case:
        #
        # Aug_DDGFact has gamma = e and fixed psi.
        #
        # F F' = C - psi I
        #
        # Gamma_1 = log(tr(F' X F) + psi)
        #         = log(sum_i x_i (C_ii - psi) + psi)
        #
        # Since e'x = s:
        #         = log(sum_i x_i C_ii - psi * (s - 1))
        #
        # grad_i = (C_ii - psi) / denom
        # ------------------------------------------------------------
        cdiag = collect(diag(C))
        d = cdiag .- psi

        function aug_gamma_bound_t1_f(x...)
            xvec = collect(x)
            denom = dot(xvec, d) + psi

            if denom <= atol
                error("Invalid t=1 Aug_DDGFact denominator: denom = $denom")
            end

            return log(denom)
        end

        function aug_gamma_bound_t1_∇f(g, x...)
            xvec = collect(x)
            denom = dot(xvec, d) + psi

            if denom <= atol
                error("Invalid t=1 Aug_DDGFact denominator in gradient: denom = $denom")
            end

            for i in 1:n
                g[i] = d[i] / denom
            end
        end

        register(
            model,
            :aug_gamma_bound_t1,
            n,
            aug_gamma_bound_t1_f,
            aug_gamma_bound_t1_∇f,
        )

        @NLobjective(model, Max, aug_gamma_bound_t1(x...))
    else
        # ------------------------------------------------------------
        # General t > 1 spectral objective/subgradient
        # ------------------------------------------------------------
        I_t = zeros(n)
        I_t[1:t] .= 1.0

        Fg = factorize_matrix(C; psi = psi)
        dx_cache = Ref(zeros(n))

        function aug_gamma_bound_gmesp_f(x...)
            xvec = collect(x)

            X = Fg' * diagm(xvec) * Fg
            X = Symmetric(0.5 * (X + X'))

            λ, U_eig = eigen(X)

            perm = length(λ):-1:1
            λ = λ[perm] + psi * I_t
            U_eig = U_eig[:, perm]

            iota, mid_val = find_iota(λ, t)

            fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
            fval += (t - iota) * log(mid_val)

            eigDual = zeros(n)

            if iota > 0
                eigDual[1:iota] .= 1.0 ./ λ[1:iota]
            end

            eigDual[iota+1:end] .= 1.0 / mid_val

            K1 = Fg * U_eig
            dx_cache[] = vec(sum(K1 .^ 2 .* eigDual', dims = 2))

            return fval
        end

        function aug_gamma_bound_gmesp_∇f(g, x...)
            dx = dx_cache[]
            for i in 1:n
                g[i] = dx[i]
            end
        end

        register(
            model,
            :aug_gamma_bound,
            n,
            aug_gamma_bound_gmesp_f,
            aug_gamma_bound_gmesp_∇f,
        )

        @NLobjective(model, Max, aug_gamma_bound(x...))
    end

    optimize!(model)

    obj_val = objective_value(model)
    x_val = value.(x)

    return x_val, obj_val
end


function aug_ddfact_upsilon_gmesp(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    s::Integer,
    t::Integer,
    psi::Float64;
    atol = 1e-10,
)
    n = size(C, 1)

    @assert length(gamma) == n
    @assert all(gamma .> 0)
    @assert 1 <= t <= s <= n

    log_gamma = log.(gamma)

    model = Model(Ipopt.Optimizer)
    add_ipopt_options!(model)
    set_silent(model)

    @variable(model, atol <= x[i = 1:n] <= 1 - atol)
    @variable(model, atol <= y[i = 1:n] <= 1 - atol)

    @constraint(model, sum(x) == s)
    @constraint(model, sum(y) == t)
    @constraint(model, [i = 1:n], y[i] <= x[i])

    if t == 1
        # ------------------------------------------------------------
        # Exact differentiable t = 1 case:
        #
        # Upsilon case:
        #
        # F F' = D_gamma^{1/2} C D_gamma^{1/2} - psi I
        #
        # d_i = gamma_i C_ii - psi
        #
        # Gamma_1 = log(sum_i x_i d_i + psi)
        #         = log(sum_i x_i gamma_i C_ii - psi * (s - 1))
        #
        # grad_x_i = d_i / denom
        #
        # The y-part is linear:
        #     - sum_i log(gamma_i) y_i
        # and JuMP handles that part directly.
        # ------------------------------------------------------------
        cdiag = collect(diag(C))
        d = gamma .* cdiag .- psi

        function aug_gamma_upsilon_bound_t1_f(x...)
            xvec = collect(x)
            denom = dot(xvec, d) + psi

            if denom <= atol
                error("Invalid t=1 Upsilon denominator: denom = $denom")
            end

            return log(denom)
        end

        function aug_gamma_upsilon_bound_t1_∇f(g, x...)
            xvec = collect(x)
            denom = dot(xvec, d) + psi

            if denom <= atol
                error("Invalid t=1 Upsilon denominator in gradient: denom = $denom")
            end

            for i in 1:n
                g[i] = d[i] / denom
            end
        end

        register(
            model,
            :aug_gamma_upsilon_bound_t1,
            n,
            aug_gamma_upsilon_bound_t1_f,
            aug_gamma_upsilon_bound_t1_∇f,
        )

        @NLobjective(
            model,
            Max,
            aug_gamma_upsilon_bound_t1(x...) -
            sum(log_gamma[i] * y[i] for i in 1:n)
        )
    else
        # ------------------------------------------------------------
        # General t > 1 spectral objective/subgradient
        # ------------------------------------------------------------
        Fg = scaled_factorize_matrix(
            C,
            gamma,
            psi;
            atol = atol,
        )

        I_t = zeros(n)
        I_t[1:t] .= 1.0

        dx_cache = Ref(zeros(n))

        function aug_gamma_upsilon_bound_gmesp_f(x...)
            xvec = collect(x)

            X = Fg' * diagm(xvec) * Fg
            X = Symmetric(0.5 * (X + X'))

            λ, U_eig = eigen(X)

            perm = length(λ):-1:1
            λ = λ[perm] + psi * I_t
            U_eig = U_eig[:, perm]

            iota, mid_val = find_iota(λ, t)

            fval = iota == 0 ? 0.0 : sum(log, @view λ[1:iota])
            fval += (t - iota) * log(mid_val)

            eigDual = zeros(n)

            if iota > 0
                eigDual[1:iota] .= 1.0 ./ λ[1:iota]
            end

            eigDual[iota+1:end] .= 1.0 / mid_val

            K1 = Fg * U_eig
            dx_cache[] = vec(sum(K1 .^ 2 .* eigDual', dims = 2))

            return fval
        end

        function aug_gamma_upsilon_bound_gmesp_∇f(g, x...)
            dx = dx_cache[]
            for i in 1:n
                g[i] = dx[i]
            end
        end

        register(
            model,
            :aug_gamma_upsilon_bound,
            n,
            aug_gamma_upsilon_bound_gmesp_f,
            aug_gamma_upsilon_bound_gmesp_∇f,
        )

        @NLobjective(
            model,
            Max,
            aug_gamma_upsilon_bound(x...) -
            sum(log_gamma[i] * y[i] for i in 1:n)
        )
    end

    optimize!(model)

    obj_val = objective_value(model)
    x_val = value.(x)
    y_val = value.(y)

    return x_val, y_val, obj_val
end
