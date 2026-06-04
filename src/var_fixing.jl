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
function var_fixing_from_DGFact(
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
function var_fixing_DDGFact_dual(
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

    fixing = var_fixing_from_DGFact(
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
function var_fixing_from_DGFactplus(
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
function var_fixing_DDGFactplus_dual(
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

    fixing = var_fixing_from_DGFactplus(
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
# Linear maximization over the fixed-variable cardinality polytope
#
# Computes:
#   max { g'x : l <= x <= c, e'x = s }
#
# where l,c are binary vectors.
# ============================================================
function max_linear_cardinality_binary_bounds(
    g::Vector{Float64},
    s::Int;
    l::Vector{Float64} = zeros(length(g)),
    c::Vector{Float64} = ones(length(g)),
    atol::Float64 = 1e-8,
)
    n = length(g)

    if length(l) != n || length(c) != n
        error("g, l, and c must have the same length.")
    end

    if any(l .> c)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            feasible = false,
            selected_free = Int[],
        )
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    N1 = findall(i -> l[i] == 1.0 && c[i] == 1.0, 1:n)
    Nf = findall(i -> l[i] == 0.0 && c[i] == 1.0, 1:n)

    m = s - length(N1)

    if m < 0 || m > length(Nf)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            feasible = false,
            selected_free = Int[],
        )
    end

    selected_free =
        m == 0 ? Int[] : Nf[partialsortperm(g[Nf], 1:m; rev = true)]

    xstar = copy(l)
    xstar[selected_free] .= 1.0

    value = dot(g, xstar)

    return (
        value = value,
        xstar = xstar,
        feasible = true,
        selected_free = selected_free,
    )
end

# ============================================================
# Primal variable fixing for DDGFact^+
#
# Given a feasible relaxation point xhat, compute a supergradient g of
# x -> Gamma_t(M_psi(x); psi), build the affine upper estimator
#
#   Gamma_t(M_psi(xhat); psi) + g'(x - xhat),
#
# and use it to prove that the restricted problems x_j = 0 or x_j = 1
# cannot contain an optimal GMESP solution.
#
# Rule:
#   x_j^* = 1 if UB(x_j = 0) < LB
#   x_j^* = 0 if UB(x_j = 1) < LB
# ============================================================
function var_fixing_DDGFactplus_primal(
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
    n = length(xhat)

    if length(l) != n || length(c) != n
        error("xhat, l, and c must have the same length.")
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

    if abs(sum(xhat) - s) > 1e-5
        @warn "xhat does not satisfy e'x = s within tolerance." sum_xhat = sum(xhat) s = s
    end

    # Subgradient at xhat
    g = x_subgradient_Gamma_t_from_F(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    gamma_value = Gamma_t_from_F(
        xhat,
        F,
        t,
        psi,
    )

    # Root/node affine upper bound:
    #   Gamma(xhat) - g'xhat + max{g'x : l <= x <= c, e'x = s}
    lin_root = max_linear_cardinality_binary_bounds(
        g,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    if !lin_root.feasible
        error("The root/node fixed-variable polytope is infeasible.")
    end

    UB =
        gamma_value -
        dot(g, xhat) +
        lin_root.value

    gap = UB - LB

    free = findall(j -> l[j] == 0.0 && c[j] == 1.0, 1:n)

    UB_if_zero = fill(-Inf, n)
    UB_if_one  = fill(-Inf, n)

    loss_if_zero = fill(Inf, n)
    loss_if_one  = fill(Inf, n)

    status_if_zero = fill(false, n)
    status_if_one  = fill(false, n)

    fix_zero = Int[]
    fix_one = Int[]

    for j in free
        # ----------------------------------------------------
        # Test x_j = 0.
        # If the restricted upper bound is below LB, then
        # no optimal solution can satisfy x_j = 0, so fix x_j = 1.
        # ----------------------------------------------------
        l_zero = copy(l)
        c_zero = copy(c)
        l_zero[j] = 0.0
        c_zero[j] = 0.0

        lin_zero = max_linear_cardinality_binary_bounds(
            g,
            s;
            l = l_zero,
            c = c_zero,
            atol = atol,
        )

        if lin_zero.feasible
            UB_if_zero[j] =
                gamma_value -
                dot(g, xhat) +
                lin_zero.value

            loss_if_zero[j] = UB - UB_if_zero[j]
            status_if_zero[j] = true

            if UB_if_zero[j] < LB - atol
                push!(fix_one, j)
            end
        else
            # If imposing x_j = 0 makes the node infeasible, then x_j must be 1.
            UB_if_zero[j] = -Inf
            loss_if_zero[j] = Inf
            push!(fix_one, j)
        end

        # ----------------------------------------------------
        # Test x_j = 1.
        # If the restricted upper bound is below LB, then
        # no optimal solution can satisfy x_j = 1, so fix x_j = 0.
        # ----------------------------------------------------
        l_one = copy(l)
        c_one = copy(c)
        l_one[j] = 1.0
        c_one[j] = 1.0

        lin_one = max_linear_cardinality_binary_bounds(
            g,
            s;
            l = l_one,
            c = c_one,
            atol = atol,
        )

        if lin_one.feasible
            UB_if_one[j] =
                gamma_value -
                dot(g, xhat) +
                lin_one.value

            loss_if_one[j] = UB - UB_if_one[j]
            status_if_one[j] = true

            if UB_if_one[j] < LB - atol
                push!(fix_zero, j)
            end
        else
            # If imposing x_j = 1 makes the node infeasible, then x_j must be 0.
            UB_if_one[j] = -Inf
            loss_if_one[j] = Inf
            push!(fix_zero, j)
        end
    end

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

        UB = UB,
        LB = LB,
        gap = gap,

        gamma_value = gamma_value,
        g = g,
        linear_root_value = lin_root.value,
        root_linear_solution = lin_root.xstar,

        UB_if_zero = UB_if_zero,
        UB_if_one = UB_if_one,
        loss_if_zero = loss_if_zero,
        loss_if_one = loss_if_one,

        status_if_zero = status_if_zero,
        status_if_one = status_if_one,

        both_fixed = both_fixed,
        infeasible_bounds = infeasible_bounds,
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
function var_fixing_from_DGFactplusUpsilon_simple(
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
function var_fixing_DDGFactplusUpsilon_dual_simple(
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

    fixing = var_fixing_from_DGFactplusUpsilon_simple(
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
function var_fixing_from_DGFactplusUpsilon_strong(
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

function var_fixing_DDGFactplusUpsilon_dual_strong(
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

    fixing = var_fixing_from_DGFactplusUpsilon_strong(
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

# ============================================================
# Compare simple and strong variable fixing for DDGFact+_Upsilon
#
# This constructs the DGFact+_Upsilon dual solution only once,
# so both fixing rules use the same certificate.
# ============================================================
function var_fixing_DDGFactplusUpsilon_dual_compare(
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

    fixing_simple = var_fixing_from_DGFactplusUpsilon_simple(
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

    fixing_strong = var_fixing_from_DGFactplusUpsilon_strong(
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

    simple_zero = Set(fixing_simple.fix_zero)
    simple_one  = Set(fixing_simple.fix_one)

    strong_zero = Set(fixing_strong.fix_zero)
    strong_one  = Set(fixing_strong.fix_one)

    simple_fixed = union(simple_zero, simple_one)
    strong_fixed = union(strong_zero, strong_one)

    simple_not_strong_zero = setdiff(simple_zero, strong_zero)
    simple_not_strong_one  = setdiff(simple_one, strong_one)
    simple_not_strong      = setdiff(simple_fixed, strong_fixed)

    dominance_holds =
        isempty(simple_not_strong_zero) &&
        isempty(simple_not_strong_one)

    return (
        dual_solution = dual_sol,

        fixing_simple = fixing_simple,
        fixing_strong = fixing_strong,

        n_fixed_simple = length(simple_fixed),
        n_fixed_strong = length(strong_fixed),

        n_fix_zero_simple = length(simple_zero),
        n_fix_one_simple = length(simple_one),

        n_fix_zero_strong = length(strong_zero),
        n_fix_one_strong = length(strong_one),

        dominance_holds = dominance_holds,
        simple_not_strong_zero = collect(simple_not_strong_zero),
        simple_not_strong_one = collect(simple_not_strong_one),
        simple_not_strong = collect(simple_not_strong),
    )
end

# ============================================================
# Linear maximization for the DDGFact+_Upsilon primal fixing rule
#
# Computes:
#
#   max { g'x - q'y :
#         l <= x <= c,
#         e'x = s,
#         e'y = t,
#         0 <= y <= x }
#
# using Gurobi.
# ============================================================
function max_linear_upsilon_xy_bounds(
    g::Vector{Float64},
    q::Vector{Float64},
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(g)),
    c::Vector{Float64} = ones(length(g)),
    fixed_index::Union{Nothing,Int} = nothing,
    fixed_value::Union{Nothing,Float64} = nothing,
    atol::Float64 = 1e-8,
    silent::Bool = true,
)
    n = length(g)

    if length(q) != n || length(l) != n || length(c) != n
        error("g, q, l, and c must have the same length.")
    end

    if any(l .> c)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            ystar = zeros(Float64, n),
            feasible = false,
            status = nothing,
        )
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) || any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    if sum(l) > s + atol || sum(c) < s - atol
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            ystar = zeros(Float64, n),
            feasible = false,
            status = nothing,
        )
    end

    if t < 0 || t > s
        error("Need 0 <= t <= s.")
    end

    l_model = copy(l)
    c_model = copy(c)

    if fixed_index !== nothing
        if fixed_value === nothing
            error("If fixed_index is given, fixed_value must also be given.")
        end

        j = fixed_index

        if j < 1 || j > n
            error("fixed_index must be between 1 and n.")
        end

        l_model[j] = fixed_value
        c_model[j] = fixed_value

        if l_model[j] < l[j] - atol || c_model[j] > c[j] + atol
            return (
                value = -Inf,
                xstar = zeros(Float64, n),
                ystar = zeros(Float64, n),
                feasible = false,
                status = nothing,
            )
        end
    end

    if any(l_model .> c_model .+ atol)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            ystar = zeros(Float64, n),
            feasible = false,
            status = nothing,
        )
    end

    if sum(l_model) > s + atol || sum(c_model) < s - atol
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            ystar = zeros(Float64, n),
            feasible = false,
            status = nothing,
        )
    end

    model = Model(Gurobi.Optimizer)

    if silent
        set_silent(model)
    end

    @variable(model, x[1:n])
    @variable(model, y[1:n] >= 0.0)

    for i in 1:n
        set_lower_bound(x[i], l_model[i])
        set_upper_bound(x[i], c_model[i])
    end

    @constraint(model, sum(x[i] for i in 1:n) == s)
    @constraint(model, sum(y[i] for i in 1:n) == t)
    @constraint(model, [i in 1:n], y[i] <= x[i])

    @objective(
        model,
        Max,
        sum(g[i] * x[i] - q[i] * y[i] for i in 1:n)
    )

    optimize!(model)

    status = termination_status(model)

    if status != MOI.OPTIMAL
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            ystar = zeros(Float64, n),
            feasible = false,
            status = status,
        )
    end

    return (
        value = objective_value(model),
        xstar = value.(x),
        ystar = value.(y),
        feasible = true,
        status = status,
    )
end

# ============================================================
# Primal variable fixing for DDGFact+_Upsilon
#
# Given a feasible relaxation point (xhat, yhat), compute a supergradient
# g of
#
#   x -> Gamma_t(M_{Upsilon,psi}(x); psi),
#
# build the affine upper estimator
#
#   Gamma_t(M_{Upsilon,psi}(xhat); psi)
#   - g'xhat
#   + max { g'x - q'y : feasible (x,y) },
#
# and use restricted versions with x_j = 0 or x_j = 1.
#
# Rule:
#   x_j^* = 1 if UB(x_j = 0) < LB
#   x_j^* = 0 if UB(x_j = 1) < LB
#
# Here:
#   q = log.(gamma)
#   F satisfies D_gamma^(1/2) C D_gamma^(1/2) - psi*I = F*F'
# ============================================================
function var_fixing_DDGFactplusUpsilon_primal(
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
    n = length(xhat)

    if length(yhat) != n || length(gamma) != n || length(l) != n || length(c) != n
        error("xhat, yhat, gamma, l, and c must have the same length.")
    end

    if any(gamma .<= 0.0)
        error("All entries of gamma must be strictly positive.")
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

    if abs(sum(xhat) - s) > 1e-5
        @warn "xhat does not satisfy e'x = s within tolerance." sum_xhat = sum(xhat) s = s
    end

    if abs(sum(yhat) - t) > 1e-5
        @warn "yhat does not satisfy e'y = t within tolerance." sum_yhat = sum(yhat) t = t
    end

    if maximum(yhat .- xhat) > 1e-5 || minimum(yhat) < -1e-5
        @warn "yhat may violate 0 <= y <= x."
    end

    q = log.(gamma)

    # Supergradient at xhat
    g = x_subgradient_Gamma_t_from_F(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    gamma_value = Gamma_t_from_F(
        xhat,
        F,
        t,
        psi,
    )

    # Root/node affine upper bound:
    #
    #   Gamma(xhat) - g'xhat
    #   + max { g'x - q'y : l <= x <= c, e'x=s, e'y=t, 0 <= y <= x }
    lin_root = max_linear_upsilon_xy_bounds(
        g,
        q,
        s,
        t;
        l = l,
        c = c,
        atol = atol,
        silent = silent,
    )

    if !lin_root.feasible
        error("The root/node DDGFact+_Upsilon linear upper-bound problem is infeasible.")
    end

    UB =
        gamma_value -
        dot(g, xhat) +
        lin_root.value

    gap = UB - LB

    free = findall(j -> l[j] == 0.0 && c[j] == 1.0, 1:n)

    UB_if_zero = fill(-Inf, n)
    UB_if_one  = fill(-Inf, n)

    loss_if_zero = fill(Inf, n)
    loss_if_one  = fill(Inf, n)

    status_if_zero = Vector{Any}(fill(nothing, n))
    status_if_one  = Vector{Any}(fill(nothing, n))

    fix_zero = Int[]
    fix_one = Int[]

    for j in free
        # ----------------------------------------------------
        # Test x_j = 0.
        # If the restricted upper bound is below LB, then
        # no optimal solution can satisfy x_j = 0, so fix x_j = 1.
        # ----------------------------------------------------
        lin_zero = max_linear_upsilon_xy_bounds(
            g,
            q,
            s,
            t;
            l = l,
            c = c,
            fixed_index = j,
            fixed_value = 0.0,
            atol = atol,
            silent = silent,
        )

        status_if_zero[j] = lin_zero.status

        if lin_zero.feasible
            UB_if_zero[j] =
                gamma_value -
                dot(g, xhat) +
                lin_zero.value

            loss_if_zero[j] = UB - UB_if_zero[j]

            if UB_if_zero[j] < LB - atol
                push!(fix_one, j)
            end
        else
            # If imposing x_j = 0 makes the node infeasible, then x_j must be 1.
            UB_if_zero[j] = -Inf
            loss_if_zero[j] = Inf
            push!(fix_one, j)
        end

        # ----------------------------------------------------
        # Test x_j = 1.
        # If the restricted upper bound is below LB, then
        # no optimal solution can satisfy x_j = 1, so fix x_j = 0.
        # ----------------------------------------------------
        lin_one = max_linear_upsilon_xy_bounds(
            g,
            q,
            s,
            t;
            l = l,
            c = c,
            fixed_index = j,
            fixed_value = 1.0,
            atol = atol,
            silent = silent,
        )

        status_if_one[j] = lin_one.status

        if lin_one.feasible
            UB_if_one[j] =
                gamma_value -
                dot(g, xhat) +
                lin_one.value

            loss_if_one[j] = UB - UB_if_one[j]

            if UB_if_one[j] < LB - atol
                push!(fix_zero, j)
            end
        else
            # If imposing x_j = 1 makes the node infeasible, then x_j must be 0.
            UB_if_one[j] = -Inf
            loss_if_one[j] = Inf
            push!(fix_zero, j)
        end
    end

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

        UB = UB,
        LB = LB,
        gap = gap,

        gamma_value = gamma_value,
        g = g,
        q = q,

        linear_root_value = lin_root.value,
        root_linear_x = lin_root.xstar,
        root_linear_y = lin_root.ystar,

        UB_if_zero = UB_if_zero,
        UB_if_one = UB_if_one,
        loss_if_zero = loss_if_zero,
        loss_if_one = loss_if_one,

        status_if_zero = status_if_zero,
        status_if_one = status_if_one,

        both_fixed = both_fixed,
        infeasible_bounds = infeasible_bounds,
    )
end