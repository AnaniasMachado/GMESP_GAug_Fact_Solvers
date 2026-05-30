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