function spectral_bound(C::Symmetric{<:Real,<:AbstractMatrix},t::Integer)
    λ = reverse(eigvals(C))
    return sum(log, @view λ[1:t])
end

function factorize_matrix(C::Symmetric{<:Real,<:AbstractMatrix}; atol=1e-8, ta = 0)
    ta = max(0,ta);          
    λ,Q = eigen(C - ta*I);
    perm = length(λ):-1:1           # permutation for descending order
    λ = λ[perm];
    Q = Q[:, perm];
    # clamp small near-zeros values before sqrt
    λ = map(x -> x < atol ? 0.0 : x, λ);
    F = Q * Diagonal(sqrt.(λ));
    return F
end

function find_iota(λ::Vector{Float64}, t::Int64; atol=1e-8,check::Bool = false)
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


function ddfact_gmesp(C::Symmetric{<:Real,<:AbstractMatrix},s::Integer,t::Integer; atol = 1e-10)
    n = size(C,1);
    Fg = factorize_matrix(C);
    model = Model(Ipopt.Optimizer);
    add_ipopt_options!(model)
    set_silent(model)   
    @variable(model, atol <= x[i=1:n]<= 1-atol);
    function gamma_bound_gmesp_f(x...)
        x = collect(x);
        X = Fg'*diagm(x)*Fg;
        X = Symmetric(0.5*(X+X'));
        # eigendecomposition of X
        λ, U_eig = eigen(X);
        perm = length(λ):-1:1           # permutation for descending order
        λ = λ[perm];
        U_eig = U_eig[:, perm];
        # compute iota
        iota, mid_val = find_iota(λ,t);
        # objective function value
        iota == 0 ? fval = 0 : fval = sum(log, @view λ[1:iota]);
        fval += (t - iota) * log(mid_val)
        # auxiliary variables for supgradient 
        eigDual = zeros(n);
        eigDual[1:iota] = 1.0./λ[1:iota];
        eigDual[iota+1:end] .= 1.0/mid_val;
        # compute supgradient dx
        K1 = Fg*U_eig;
        global dx = vec(sum(K1.^2 .* eigDual', dims=2));
        return fval
    end
    function gamma_bound_gmesp_∇f(g,x...)
        for i = (1:n)
            g[i] = dx[i];
        end    
    end
    register(model, :gamma_bound, n, gamma_bound_gmesp_f, gamma_bound_gmesp_∇f)
    @NLobjective(model, Max, gamma_bound(x...))
    @constraint(model, sum(x) == s);
    optimize!(model)
    obj_val = objective_value(model)
    x = value.(x);
    return x,obj_val
end

function aug_ddfact_gmesp(C::Symmetric{<:Real,<:AbstractMatrix},s::Integer,t::Integer; atol = 1e-10)
    n = size(C,1);
    I_t = zeros(n); I_t[1:t] .= 1;
    ta = eigmin(C)-atol;
    Fg = factorize_matrix(C;ta=ta);
    model = Model(Ipopt.Optimizer)
    add_ipopt_options!(model)
    set_silent(model)
    @variable(model, atol <= x[i=1:n]<= 1 - atol);
    function aug_gamma_bound_gmesp_f(x...)
        x = collect(x);
        X = Fg'*diagm(x)*Fg;
        X = Symmetric(0.5*(X+X'));
        # eigendecomposition of X
        λ, U_eig = eigen(X);
        perm = length(λ):-1:1           # permutation for descending order
        λ = λ[perm] + ta*I_t;
        U_eig = U_eig[:, perm];
        # compute iota
        iota, mid_val = find_iota(λ,t);
        # objective function value
        iota == 0 ? fval = 0 : fval = sum(log, @view λ[1:iota]);
        fval += (t - iota) * log(mid_val)
        # auxiliary variable for supgradient 
        eigDual = zeros(n);
        eigDual[1:iota] = 1.0./λ[1:iota];
        eigDual[iota+1:end] .= 1.0/mid_val;
        # compute supgradient dx
        K1 = Fg*U_eig;
        global dx = vec(sum(K1.^2 .* eigDual', dims=2));
        return fval
    end
    function aug_gamma_bound_gmesp_∇f(g,x...)
        for i = (1:n)
            g[i] = dx[i];
        end    
    end
    register(model, :aug_gamma_bound, n, aug_gamma_bound_gmesp_f, aug_gamma_bound_gmesp_∇f)
    @NLobjective(model, Max, aug_gamma_bound(x...))
    @constraint(model, sum(x) == s);
    optimize!(model)
    obj_val = objective_value(model)
    x = value.(x);
    return x,obj_val
end
