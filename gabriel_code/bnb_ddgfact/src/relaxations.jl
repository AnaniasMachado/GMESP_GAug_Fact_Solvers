using LinearAlgebra, JuMP, Ipopt

# =============================================================================
# DDFact relaxation for GMESP.
# Functions brought verbatim from ~/Documents/GitHub/ananias/relaxations.jl
# (the augmented bound `aug_ddfact_gmesp` was intentionally NOT brought over).
#
# `ddfact_gmesp_fix` is the only addition: ananias's `ddfact_gmesp` with the
# `fix1` variables forced to x = 1, which the branch-and-bound needs (fix-to-1
# keeps the variable in the problem; fix-to-0 is handled by the caller, which
# reduces C to its principal submatrix on the kept indices).
# =============================================================================

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

# ddfact_gmesp with the variables in `fix1` forced to x = 1 (needed by the B&B).
# Identical to `ddfact_gmesp` above except for the `fix(...)` loop.
function ddfact_gmesp_fix(C::Symmetric{<:Real,<:AbstractMatrix},s::Integer,t::Integer,
                          fix1::Vector{Int}; atol = 1e-10)
    n = size(C,1);
    Fg = factorize_matrix(C);
    model = Model(Ipopt.Optimizer);
    add_ipopt_options!(model)
    set_silent(model)
    @variable(model, atol <= x[i=1:n]<= 1-atol);
    for i in fix1
        fix(x[i], 1.0; force=true)
    end
    function gamma_bound_fix_f(x...)
        x = collect(x);
        X = Fg'*diagm(x)*Fg;
        X = Symmetric(0.5*(X+X'));
        λ, U_eig = eigen(X);
        perm = length(λ):-1:1
        λ = λ[perm];
        U_eig = U_eig[:, perm];
        iota, mid_val = find_iota(λ,t);
        iota == 0 ? fval = 0 : fval = sum(log, @view λ[1:iota]);
        fval += (t - iota) * log(mid_val)
        eigDual = zeros(n);
        eigDual[1:iota] = 1.0./λ[1:iota];
        eigDual[iota+1:end] .= 1.0/mid_val;
        K1 = Fg*U_eig;
        global dx_fix = vec(sum(K1.^2 .* eigDual', dims=2));
        return fval
    end
    function gamma_bound_fix_∇f(g,x...)
        for i = (1:n)
            g[i] = dx_fix[i];
        end
    end
    register(model, :gamma_bound_fix, n, gamma_bound_fix_f, gamma_bound_fix_∇f)
    @NLobjective(model, Max, gamma_bound_fix(x...))
    @constraint(model, sum(x) == s);
    optimize!(model)
    obj_val = objective_value(model)
    x = value.(x);
    return x,obj_val
end
