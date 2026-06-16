using LinearAlgebra
using JuMP
using Gurobi
using Ipopt
import MathOptInterface as MOI

function add_ipopt_options!(model)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-8)
    set_optimizer_attribute(model, "dual_inf_tol", 1e-8)
    set_optimizer_attribute(model, "compl_inf_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_constr_viol_tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_iter", 0)
end


function add_knitro_options!(model)
    set_optimizer_attribute(model, "outlev", 0)
    set_optimizer_attribute(model, "opttol", 1e-8)
    set_optimizer_attribute(model, "feastol", 1e-8)
end

# ============================================================
# Compute the auxiliary variables for the t = 1 reformulation
# ============================================================
function compute_Tk_Lk(
    d::AbstractVector{<:Real},
    s::Integer;
    J1::Vector{Int} = Int[],
)
    n = length(d)
    @assert 1 <= s <= n
    @assert all(1 .<= J1 .<= n)
    @assert length(unique(J1)) == length(J1)
    @assert length(J1) <= s

    order = sortperm(collect(d); rev = true)

    K = Int[]
    T_list = Vector{Vector{Int}}()
    L = Float64[]

    for k in 1:n
        required = unique(vcat(J1, k))
        if length(required) > s
            continue
        end

        T = copy(required)
        required_set = Set(required)

        for i in order
            if length(T) == s
                break
            end
            if !(i in required_set)
                push!(T, i)
            end
        end

        sort!(T)
        push!(K, k)
        push!(T_list, T)
        push!(L, sum(d[i] for i in T))
    end

    return K, T_list, L
end


# ============================================================
# Recover the weights from active set
# LP-based version, more stable for calibration
# ============================================================
function recover_weights_from_active_set(
    alpha::Float64,
    eta::Float64,
    R::Vector{Float64},
    rho::Vector{Float64},
    K::Vector{Int},
    T_list::Vector{Vector{Int}};
    active_tol::Float64 = 1e-7,
)
    vals = alpha .* R .- rho
    maxval = maximum(vals)
    tol = active_tol * max(1.0, abs(maxval), abs(eta))

    active_pos = findall(abs.(vals .- maxval) .<= tol)

    if isempty(active_pos)
        active_pos = [argmax(vals)]
    end

    target = 1.0 / alpha

    # If only one active cut is detected, return it.
    if length(active_pos) == 1
        p = active_pos[1]
        return [K[p]], [1.0], [T_list[p]]
    end

    Ract = R[active_pos]

    # If target is outside the convex hull due to numerical tolerance,
    # enlarge the active set slightly before giving up.
    Rmin = minimum(Ract)
    Rmax = maximum(Ract)

    if target < Rmin - tol || target > Rmax + tol
        enlarged_tol = 100.0 * tol
        active_pos2 = findall(abs.(vals .- maxval) .<= enlarged_tol)

        if !isempty(active_pos2)
            active_pos = active_pos2
            Ract = R[active_pos]
            Rmin = minimum(Ract)
            Rmax = maximum(Ract)
        end
    end

    # If still outside the convex hull, use the closest active point.
    # This is a numerical fallback.
    if target < Rmin - tol || target > Rmax + tol
        closest_local = argmin(abs.(Ract .- target))
        p = active_pos[closest_local]
        return [K[p]], [1.0], [T_list[p]]
    end

    # Solve a small LP to find convex weights over all active cuts:
    #
    #   sum omega = 1
    #   sum omega_p R_p = target
    #   omega >= 0
    #
    # Objective is zero; this is only a feasibility problem.
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    m = length(active_pos)

    @variable(model, omega[1:m] >= 0.0)
    @constraint(model, sum(omega[q] for q in 1:m) == 1.0)
    @constraint(model, sum(omega[q] * R[active_pos[q]] for q in 1:m) == target)
    @objective(model, Min, 0.0)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        omega_val = value.(omega)

        keep = findall(omega_val .> 1e-9)

        if isempty(keep)
            closest_local = argmin(abs.(Ract .- target))
            p = active_pos[closest_local]
            return [K[p]], [1.0], [T_list[p]]
        end

        omega_keep = omega_val[keep]
        omega_keep ./= sum(omega_keep)

        selected_pos = active_pos[keep]

        return K[selected_pos], collect(omega_keep), T_list[selected_pos]
    else
        # Fallback to two-point interpolation.
        low_candidates = [p for p in active_pos if R[p] <= target + tol]
        high_candidates = [p for p in active_pos if R[p] >= target - tol]

        if isempty(low_candidates) || isempty(high_candidates)
            closest_local = argmin(abs.(Ract .- target))
            p = active_pos[closest_local]
            return [K[p]], [1.0], [T_list[p]]
        end

        p_low = low_candidates[argmax(R[low_candidates])]
        p_high = high_candidates[argmin(R[high_candidates])]

        if p_low == p_high || abs(R[p_high] - R[p_low]) <= tol
            return [K[p_low]], [1.0], [T_list[p_low]]
        end

        w_low = (R[p_high] - target) / (R[p_high] - R[p_low])
        w_high = 1.0 - w_low

        return [K[p_low], K[p_high]], [w_low, w_high], [T_list[p_low], T_list[p_high]]
    end
end


# ============================================================
# Solves the t=1 generalized-scaled DDGFact+_Upsilon relaxation using the
# one-dimensional convex reformulation.
# ============================================================
function ddfact_upsilon_t1_ipopt(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    s::Integer,
    psi::Float64;
    J1::Vector{Int} = Int[],
    d = nothing,
    atol::Float64 = 1e-10,
    active_tol::Float64 = 1e-7,
    silent::Bool = true,
)
    n = size(C, 1)

    @assert size(C, 2) == n
    @assert length(gamma) == n
    @assert all(gamma .> 0)
    @assert 1 <= s <= n
    @assert length(J1) <= s
    @assert all(1 .<= J1 .<= n)

    rho = log.(gamma)

    d_vec =
        d === nothing ?
        collect(gamma .* diag(C) .- psi) :
        collect(Float64.(d))

    @assert length(d_vec) == n

    K, T_list, L = compute_Tk_Lk(d_vec, s; J1 = J1)
    R = psi .+ L

    if any(R .<= 0.0)
        bad = K[findfirst(R .<= 0.0)]
        error(
            "Encountered R_k <= 0 for k = $bad. " *
            "The logarithm domain assumption is violated for at least one support."
        )
    end

    model = Model(Ipopt.Optimizer)
    add_ipopt_options!(model)
    if silent
        set_silent(model)
    end

    alpha_start = length(R) / sum(R)

    @variable(model, alpha >= atol, start = alpha_start)
    @variable(model, eta)

    @constraint(model, envelope[p = 1:length(K)], eta >= R[p] * alpha - rho[K[p]])

    @NLobjective(model, Min, eta - log(alpha) - 1.0)

    optimize!(model)

    alpha_val = value(alpha)
    eta_val = value(eta)
    obj_val = objective_value(model)

    active_k, omega, active_T = recover_weights_from_active_set(
        alpha_val,
        eta_val,
        R,
        rho[K],
        K,
        T_list;
        active_tol = active_tol,
    )

    x_val = zeros(n)
    y_val = zeros(n)

    for (w, k, T) in zip(omega, active_k, active_T)
        for i in T
            x_val[i] += w
        end
        y_val[k] += w
    end

    # Objective recomputed from recovered primal solution.
    primal_obj = log(dot(d_vec, x_val) + psi) - dot(rho, y_val)

    return (
        x = x_val,
        y = y_val,
        obj_val = obj_val,
        primal_obj = primal_obj,
        alpha = alpha_val,
        eta = eta_val,
        active_k = active_k,
        omega = omega,
        active_T = active_T,
        R = R,
        L = L,
        K = K,
        d = d_vec,
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        dual_status = dual_status(model),
    )
end

function ddfact_upsilon_t1_knitro(
    C::Symmetric{<:Real,<:AbstractMatrix},
    gamma::Vector{Float64},
    s::Integer,
    psi::Float64;
    J1::Vector{Int} = Int[],
    d = nothing,
    atol::Float64 = 1e-10,
    active_tol::Float64 = 1e-7,
    silent::Bool = true,
)
    n = size(C, 1)

    @assert size(C, 2) == n
    @assert length(gamma) == n
    @assert all(gamma .> 0)
    @assert 1 <= s <= n
    @assert length(J1) <= s
    @assert all(1 .<= J1 .<= n)

    rho = log.(gamma)

    d_vec =
        d === nothing ?
        collect(gamma .* diag(C) .- psi) :
        collect(Float64.(d))

    @assert length(d_vec) == n

    K, T_list, L = compute_Tk_Lk(d_vec, s; J1 = J1)
    R = psi .+ L

    if any(R .<= 0.0)
        bad = K[findfirst(R .<= 0.0)]
        error(
            "Encountered R_k <= 0 for k = $bad. " *
            "The logarithm domain assumption is violated for at least one support."
        )
    end

    model = Model(KNITRO.Optimizer)
    add_knitro_options!(model)

    if silent
        set_silent(model)
    end

    alpha_start = length(R) / sum(R)

    @variable(model, alpha >= atol, start = alpha_start)
    @variable(model, eta)

    @constraint(model, envelope[p = 1:length(K)], eta >= R[p] * alpha - rho[K[p]])

    @NLobjective(model, Min, eta - log(alpha) - 1.0)

    optimize!(model)

    alpha_val = value(alpha)
    eta_val = value(eta)
    obj_val = objective_value(model)

    active_k, omega, active_T = recover_weights_from_active_set(
        alpha_val,
        eta_val,
        R,
        rho[K],
        K,
        T_list;
        active_tol = active_tol,
    )

    x_val = zeros(n)
    y_val = zeros(n)

    for (w, k, T) in zip(omega, active_k, active_T)
        for i in T
            x_val[i] += w
        end
        y_val[k] += w
    end

    primal_obj = log(dot(d_vec, x_val) + psi) - dot(rho, y_val)

    return (
        x = x_val,
        y = y_val,
        obj_val = obj_val,
        primal_obj = primal_obj,
        alpha = alpha_val,
        eta = eta_val,
        active_k = active_k,
        omega = omega,
        active_T = active_T,
        R = R,
        L = L,
        K = K,
        d = d_vec,
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        dual_status = dual_status(model),
    )
end