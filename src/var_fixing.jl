using JuMP
using Gurobi
import MathOptInterface as MOI

# ============================================================
# Variable fixing from a feasible DGFact dual solution
#
# Theorem 11 rule:
#   x_j^* = 0 if zeta - LB < upsilon_j
#   x_j^* = 1 if zeta - LB < nu_j
#
# Here:
#   LB    = objective value of a feasible primal integer solution
#   zeta  = objective value of a feasible DGFact dual solution
# ============================================================
function variable_fixing_from_dual_solution_DGFact(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    zeta::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
)
    n = length(upsilon)

    if length(nu) != n || length(l) != n || length(c) != n
        error("upsilon, nu, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    gap = zeta - LB

    # Theorem 11 uses strict inequalities:
    #   zeta - LB < upsilon_j
    #   zeta - LB < nu_j
    # Numerically, we use a tolerance.
    fix_zero = findall(j -> upsilon[j] > gap + atol, 1:n)
    fix_one  = findall(j -> nu[j]      > gap + atol, 1:n)

    # Conflicts with current binary bounds
    conflict_zero = [j for j in fix_zero if l[j] == 1.0]
    conflict_one  = [j for j in fix_one  if c[j] == 0.0]
    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one]  .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        gap = gap,
        both_fixed = both_fixed,
        conflict_zero = conflict_zero,
        conflict_one = conflict_one,
        infeasible_bounds = infeasible_bounds,
    )
end

# ============================================================
# Full variable-fixing routine from a DDGFact solution xhat
#
# This first constructs a feasible DGFact dual solution from xhat,
# then applies the variable-fixing test.
#
# Assumes the no-side-constraint case, as in the closed-form G(Theta)
# construction from the paper.
# ============================================================
function variable_fixing_from_DDGFact_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    dual_sol = DGFact_dual_solution_from_DDGFact_x(
        xhat,
        F,
        s,
        t;
        l = l,
        c = c,
        epsilon = epsilon,
        atol = atol,
    )

    fixing = variable_fixing_from_dual_solution_DGFact(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.objective_value,
        LB;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end

# ============================================================
# Variable fixing from a feasible DGFact^+ dual solution
#
# Theorem rule:
#   x_j^* = 0 if zeta - LB < upsilon_j
#   x_j^* = 1 if zeta - LB < nu_j
#
# Here:
#   LB   = objective value of a feasible GMESP solution
#   zeta = objective value of a feasible DGFact^+ solution
# ============================================================
function variable_fixing_from_dual_solution_DGFactplus(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    zeta::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
)
    n = length(upsilon)

    if length(nu) != n || length(l) != n || length(c) != n
        error("upsilon, nu, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    gap = zeta - LB

    # The theorem uses strict inequalities:
    #   zeta - LB < upsilon_j
    #   zeta - LB < nu_j
    # Numerically, we use a tolerance.
    fix_zero = findall(j -> upsilon[j] > gap + atol, 1:n)
    fix_one  = findall(j -> nu[j]      > gap + atol, 1:n)

    # These are safety checks against contradictory fixings
    conflict_zero = [j for j in fix_zero if l[j] == 1.0]
    conflict_one  = [j for j in fix_one  if c[j] == 0.0]
    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one]  .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        gap = gap,
        both_fixed = both_fixed,
        conflict_zero = conflict_zero,
        conflict_one = conflict_one,
        infeasible_bounds = infeasible_bounds,
    )
end

# ============================================================
# Full variable-fixing routine from a DDGFact^+ solution xhat
#
# First constructs a feasible DGFact^+ solution from xhat,
# then applies the variable-fixing test.
#
# Assumes:
#   - no side constraints
#   - psi > 0
#   - l,c binary
# ============================================================
function variable_fixing_from_DDGFactplus_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
)
    dual_sol = DGFactplus_dual_solution_from_DDGFactplus_x(
        xhat,
        F,
        s,
        t,
        psi;
        l = l,
        c = c,
        atol = atol,
    )

    fixing = variable_fixing_from_dual_solution_DGFactplus(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.objective_value,
        LB;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end

# ============================================================
# Variable fixing from a feasible DGFact^+_Upsilon dual solution
#
# Rule:
#   x_j^* = 0 if zeta - LB < upsilon_j
#   x_j^* = 1 if zeta - LB < nu_j
#
# Here:
#   LB   = objective value of a feasible GMESP solution
#   zeta = objective value of a feasible DGFact^+_Upsilon solution
# ============================================================
function variable_fixing_from_dual_solution_DGFactplusUpsilon(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    eta::Vector{Float64},
    rho::Vector{Float64},
    zeta::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
)
    n = length(upsilon)

    if length(nu) != n || length(eta) != n || length(rho) != n ||
       length(l) != n || length(c) != n
        error("upsilon, nu, eta, rho, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    gap = zeta - LB

    # Only apply fixing tests to variables that are free at the current node.
    free = findall(j -> l[j] == 0.0 && c[j] == 1.0, 1:n)

    # If x_j = 1, then 0 <= y_j <= 1 and the forced penalty is at least
    # upsilon_j + min(eta_j, rho_j).
    zero_penalty = upsilon .+ min.(eta, rho)

    # If x_j = 0, then y_j = 0 and the forced penalty is nu_j.
    one_penalty = nu

    fix_zero = [j for j in free if zero_penalty[j] > gap + atol]
    fix_one  = [j for j in free if one_penalty[j]  > gap + atol]

    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one]  .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        gap = gap,
        zero_penalty = zero_penalty,
        one_penalty = one_penalty,
        both_fixed = both_fixed,
        infeasible_bounds = infeasible_bounds,
    )
end

# ============================================================
# Full variable-fixing routine from a DDGFact^+_Upsilon solution
#
# First constructs a feasible DGFact^+_Upsilon dual solution from
# (xhat, yhat), then applies the variable-fixing test.
#
# Assumes:
#   - no side constraints
#   - psi > 0
#   - l,c binary
#   - F satisfies D_gamma^(1/2) C D_gamma^(1/2) - psi*I = F*F'
# ============================================================
function variable_fixing_from_DDGFactplusUpsilon_xy(
    xhat::Vector{Float64},
    yhat::Vector{Float64},
    gamma::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    dual_sol = DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
        xhat,
        gamma,
        F,
        s,
        t,
        psi;
        yhat = yhat,
        l = l,
        c = c,
        atol = atol,
        silent = silent,
    )

    fixing = variable_fixing_from_dual_solution_DGFactplusUpsilon(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.eta,
        dual_sol.rho,
        dual_sol.objective_value,
        LB;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end

# ============================================================
# Strong variable fixing from a feasible DGFact+_Upsilon dual solution
#
# For each free variable j, solve two LPs:
#
#   p_if_one[j]  = min P(x,y) subject to x_j = 1
#   p_if_zero[j] = min P(x,y) subject to x_j = 0
#
# where
#
#   P(x,y) =
#       upsilon'(x-l)
#     + nu'(c-x)
#     + eta'y
#     + rho'(x-y)
#
# Then:
#   x_j^* = 0 if zeta - LB < p_if_one[j]
#   x_j^* = 1 if zeta - LB < p_if_zero[j]
# ============================================================
function variable_fixing_from_dual_solution_DGFactplusUpsilon_strong(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    eta::Vector{Float64},
    rho::Vector{Float64},
    zeta::Float64,
    LB::Float64,
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    n = length(upsilon)

    if length(nu) != n || length(eta) != n || length(rho) != n ||
       length(l) != n || length(c) != n
        error("upsilon, nu, eta, rho, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    if sum(l) > s + atol || sum(c) < s - atol
        error("The fixed-variable polytope is empty: need sum(l) <= s <= sum(c).")
    end

    if t < 0 || t > s
        error("Need 0 <= t <= s.")
    end

    gap = zeta - LB

    free = findall(j -> l[j] == 0.0 && c[j] == 1.0, 1:n)

    # Build one LP and reuse it by changing the bound of x[j].
    model = Model(Gurobi.Optimizer)

    if silent
        set_silent(model)
    end

    @variable(model, x[i = 1:n])
    @variable(model, y[i = 1:n] >= 0.0)

    for i in 1:n
        set_lower_bound(x[i], l[i])
        set_upper_bound(x[i], c[i])
    end

    @constraint(model, sum(x[i] for i in 1:n) == s)
    @constraint(model, sum(y[i] for i in 1:n) == t)
    @constraint(model, [i in 1:n], y[i] <= x[i])

    @objective(
        model,
        Min,
        sum(
            upsilon[i] * (x[i] - l[i]) +
            nu[i]      * (c[i] - x[i]) +
            eta[i]     * y[i] +
            rho[i]     * (x[i] - y[i])
            for i in 1:n
        )
    )

    p_if_one = fill(Inf, n)
    p_if_zero = fill(Inf, n)

    status_if_one = Vector{Any}(fill(nothing, n))
    status_if_zero = Vector{Any}(fill(nothing, n))

    function solve_with_fixed_value!(j::Int, value::Float64)
        old_lb = lower_bound(x[j])
        old_ub = upper_bound(x[j])

        set_lower_bound(x[j], value)
        set_upper_bound(x[j], value)

        optimize!(model)

        status = termination_status(model)

        penalty =
            status == MOI.OPTIMAL ? objective_value(model) : Inf

        # Restore original node bounds
        set_lower_bound(x[j], old_lb)
        set_upper_bound(x[j], old_ub)

        return penalty, status
    end

    for j in free
        p_if_one[j], status_if_one[j] = solve_with_fixed_value!(j, 1.0)
        p_if_zero[j], status_if_zero[j] = solve_with_fixed_value!(j, 0.0)
    end

    # If x_j = 1 leads to penalty larger than the available gap,
    # then x_j = 1 is impossible, so fix x_j = 0.
    fix_zero = [j for j in free if p_if_one[j] > gap + atol]

    # If x_j = 0 leads to penalty larger than the available gap,
    # then x_j = 0 is impossible, so fix x_j = 1.
    fix_one = [j for j in free if p_if_zero[j] > gap + atol]

    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one] .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        gap = gap,
        p_if_one = p_if_one,
        p_if_zero = p_if_zero,
        status_if_one = status_if_one,
        status_if_zero = status_if_zero,
        both_fixed = both_fixed,
        infeasible_bounds = infeasible_bounds,
    )
end

function variable_fixing_from_DDGFactplusUpsilon_xy_strong(
    xhat::Vector{Float64},
    yhat::Vector{Float64},
    gamma::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    dual_sol = DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
        xhat,
        gamma,
        F,
        s,
        t,
        psi;
        yhat = yhat,
        l = l,
        c = c,
        atol = atol,
        silent = silent,
    )

    fixing = variable_fixing_from_dual_solution_DGFactplusUpsilon_strong(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.eta,
        dual_sol.rho,
        dual_sol.objective_value,
        LB,
        s,
        t;
        l = l,
        c = c,
        atol = atol,
        silent = silent,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end