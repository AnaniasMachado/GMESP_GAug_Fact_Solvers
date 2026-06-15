using LinearAlgebra
using Printf

# =============================================================================
# General branch-and-bound for GMESP
#
# Requires:
#   include("bnb_util.jl")
#
# Main options:
#   - fixing_rule = :none
#   - root_bfgs_param_set separate from bfgs_param_set
#   - root_after_psi0_bfgs_param_set for the second root optimization when
#     root_psi0_warm_start = true
#   - warm_start_parent_upsilon option
#   - reuse_parent_upsilon option
#   - optional root psi=0 Upsilon warm start before root DDGFactplusUpsilon
# =============================================================================


# =============================================================================
# Initial incumbent
# =============================================================================

function _initial_incumbent(
    C::Symmetric,
    s::Int,
    t::Int,
)
    x_ls, lb = run_all_LS(C, s, t)
    S_inc = sort(findall(x_ls .> 0.5))

    return x_ls, lb, S_inc
end


# =============================================================================
# Integer relaxation solution registration
# =============================================================================

function _register_integer_solution!(
    C::Symmetric,
    s::Int,
    t::Int,
    r,
    state::Base.RefValue,
)
    _is_integer_point(r.x) || return nothing

    true_obj, S = _subset_value(C, s, t, r.keep, r.x)
    isfinite(true_obj) || return nothing

    st = state[]

    int_gap = r.ub - true_obj

    n_int_sols = st.n_int_sols + 1
    int_gap_sum = st.int_gap_sum + int_gap
    int_gap_max = max(st.int_gap_max, int_gap)

    lb = st.lb
    S_inc = st.S_inc
    int_gap_inc = st.int_gap_inc

    if true_obj > lb
        lb = true_obj
        S_inc = S
        int_gap_inc = int_gap
    end

    state[] = (
        lb = lb,
        S_inc = S_inc,
        n_int_sols = n_int_sols,
        int_gap_sum = int_gap_sum,
        int_gap_max = int_gap_max,
        int_gap_inc = int_gap_inc,
    )

    return nothing
end


# =============================================================================
# Node kwargs
# =============================================================================

function _make_node_kwargs(;
    relaxation,
    psi,
    bfgs_param_set,
    bfgs_param_sets,
    atol,
    psi_margin,
    psi_floor,
    upsilon_fixing,
)
    return (
        relaxation = relaxation,
        psi = psi,
        bfgs_param_set = bfgs_param_set,
        bfgs_param_sets = bfgs_param_sets,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
    )
end


# =============================================================================
# Open-list utilities
# =============================================================================

function _push_if_survives!(
    open::Vector{GMESPNode},
    F1::Vector{Int},
    F0::Vector{Int},
    r,
    lb::Float64,
)
    r.determined && return false
    r.ub ≤ lb && return false

    push!(open, GMESPNode(F1, F0, r.ub, r.x, r.keep, r.gamma))

    return true
end


# =============================================================================
# Child bounding
# =============================================================================

function _child_theta0(
    n::Int,
    F0::Vector{Int},
    parent_keep::Vector{Int},
    parent_gamma::Union{Nothing,Vector{Float64}};
    relaxation::Symbol,
    warm_start_parent_upsilon::Bool,
)
    if !warm_start_parent_upsilon || relaxation != :DDGFactplusUpsilon
        return nothing
    end

    return _theta0_from_parent_gamma(
        n,
        F0,
        parent_keep,
        parent_gamma,
    )
end


function _rebound_theta0(
    n::Int,
    F0f::Vector{Int},
    r;
    relaxation::Symbol,
    warm_start_parent_upsilon::Bool,
)
    if !warm_start_parent_upsilon || relaxation != :DDGFactplusUpsilon
        return nothing
    end

    return _theta0_from_current_gamma_after_fixing(
        n,
        F0f,
        r.keep,
        r.gamma,
    )
end


function _child_fixed_gamma(
    n::Int,
    F0::Vector{Int},
    parent_keep::Vector{Int},
    parent_gamma::Union{Nothing,Vector{Float64}};
    relaxation::Symbol,
    reuse_parent_upsilon::Bool,
)
    if !reuse_parent_upsilon || relaxation != :DDGFactplusUpsilon
        return nothing
    end

    return _gamma_from_parent_gamma(
        n,
        F0,
        parent_keep,
        parent_gamma,
    )
end


function _rebound_fixed_gamma(
    n::Int,
    F0f::Vector{Int},
    r;
    relaxation::Symbol,
    reuse_parent_upsilon::Bool,
)
    if !reuse_parent_upsilon || relaxation != :DDGFactplusUpsilon
        return nothing
    end

    return _gamma_from_current_gamma_after_fixing(
        n,
        F0f,
        r.keep,
        r.gamma,
    )
end


function _bound_child!(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int},
    parent_keep::Vector{Int},
    parent_gamma::Union{Nothing,Vector{Float64}},
    open::Vector{GMESPNode},
    state::Base.RefValue,
    counters::Base.RefValue,
    node_kwargs;
    relaxation::Symbol,
    fixing_rule::Symbol,
    warm_start_parent_upsilon::Bool,
    reuse_parent_upsilon::Bool,
    atol::Float64,
)
    n = size(C, 1)

    fixed_upsilon_gamma = _child_fixed_gamma(
        n,
        F0,
        parent_keep,
        parent_gamma;
        relaxation = relaxation,
        reuse_parent_upsilon = reuse_parent_upsilon,
    )

    theta0 =
        fixed_upsilon_gamma === nothing ?
        _child_theta0(
            n,
            F0,
            parent_keep,
            parent_gamma;
            relaxation = relaxation,
            warm_start_parent_upsilon = warm_start_parent_upsilon,
        ) :
        nothing

    r = _gmesp_node(
        C,
        s,
        t,
        F1,
        F0;
        node_kwargs...,
        theta0 = theta0,
        fixed_upsilon_gamma = fixed_upsilon_gamma,
    )

    cnt = counters[]
    counters[] = (
        nodes = cnt.nodes + 1,
        nfix0 = cnt.nfix0,
        nfix1 = cnt.nfix1,
    )

    _register_integer_solution!(C, s, t, r, state)

    st = state[]

    r.determined && return nothing
    r.ub ≤ st.lb && return nothing

    F1f, F0f = variable_fixing_relaxation_soln(
        F1,
        F0,
        r,
        s,
        t,
        st.lb;
        fixing_rule = fixing_rule,
        tol = atol,
    )

    if length(F1f) > length(F1) || length(F0f) > length(F0)
        cnt = counters[]

        counters[] = (
            nodes = cnt.nodes,
            nfix0 = cnt.nfix0 + length(F0f) - length(F0),
            nfix1 = cnt.nfix1 + length(F1f) - length(F1),
        )

        fixed_upsilon_gamma_rebound = _rebound_fixed_gamma(
            n,
            F0f,
            r;
            relaxation = relaxation,
            reuse_parent_upsilon = reuse_parent_upsilon,
        )

        theta0_rebound =
            fixed_upsilon_gamma_rebound === nothing ?
            _rebound_theta0(
                n,
                F0f,
                r;
                relaxation = relaxation,
                warm_start_parent_upsilon = warm_start_parent_upsilon,
            ) :
            nothing

        r = _gmesp_node(
            C,
            s,
            t,
            F1f,
            F0f;
            node_kwargs...,
            theta0 = theta0_rebound,
            fixed_upsilon_gamma = fixed_upsilon_gamma_rebound,
        )

        _register_integer_solution!(C, s, t, r, state)

        st = state[]

        r.determined && return nothing
        r.ub ≤ st.lb && return nothing

        F1 = F1f
        F0 = F0f
    end

    st = state[]

    _push_if_survives!(
        open,
        F1,
        F0,
        r,
        st.lb,
    )

    return nothing
end


# =============================================================================
# Logging
# =============================================================================

function _print_root_log(
    relaxation::Symbol,
    fixing_rule::Symbol,
    root_bfgs_param_set::Symbol,
    effective_root_bfgs_param_set::Symbol,
    bfgs_param_set::Symbol,
    root_psi0_warm_start::Bool,
    root_psi0_bfgs_param_set::Symbol,
    root_after_psi0_bfgs_param_set::Symbol,
    upsilon_fixing::Symbol,
    warm_start_parent_upsilon::Bool,
    reuse_parent_upsilon::Bool,
    lb::Float64,
    ub_root::Float64,
)
    @printf(
        "root:  relaxation = %s   fixing = %s   root_bfgs = %s   effective_root_bfgs = %s   node_bfgs = %s   root_psi0 = %s/%s   root_after_psi0 = %s   upsilon_fixing = %s   warm_start = %s   reuse_parent = %s   lb = %.4f   ub = %.4f   gap = %.4f\n",
        String(relaxation),
        String(fixing_rule),
        String(root_bfgs_param_set),
        String(effective_root_bfgs_param_set),
        String(bfgs_param_set),
        string(root_psi0_warm_start),
        String(root_psi0_bfgs_param_set),
        String(root_after_psi0_bfgs_param_set),
        String(upsilon_fixing),
        string(warm_start_parent_upsilon),
        string(reuse_parent_upsilon),
        lb,
        ub_root,
        ub_root - lb,
    )

    flush(stdout)

    return nothing
end


function _print_progress(
    nodes::Int,
    open::Vector{GMESPNode},
    nfix0::Int,
    nfix1::Int,
    lb::Float64,
)
    best_ub = isempty(open) ? lb : maximum(nd -> nd.ub, open)

    @printf(
        "nodes = %4d  open = %4d  fix0 = %3d  fix1 = %3d  lb = %.4f  ub = %.4f  gap = %.4f\n",
        nodes,
        length(open),
        nfix0,
        nfix1,
        lb,
        best_ub,
        best_ub - lb,
    )

    flush(stdout)

    return nothing
end


function _print_final_log(
    status::String,
    relaxation::Symbol,
    fixing_rule::Symbol,
    root_bfgs_param_set::Symbol,
    effective_root_bfgs_param_set::Symbol,
    bfgs_param_set::Symbol,
    root_psi0_warm_start::Bool,
    root_psi0_bfgs_param_set::Symbol,
    root_after_psi0_bfgs_param_set::Symbol,
    upsilon_fixing::Symbol,
    warm_start_parent_upsilon::Bool,
    reuse_parent_upsilon::Bool,
    nodes::Int,
    nfix0::Int,
    nfix1::Int,
    lb::Float64,
    ub::Float64,
    n_int_sols::Int,
    int_gap_max::Float64,
    int_gap_avg::Float64,
    int_gap_opt::Float64,
)
    @printf(
        "done [%s].  relaxation = %s   fixing = %s   root_bfgs = %s   effective_root_bfgs = %s   node_bfgs = %s   root_psi0 = %s/%s   root_after_psi0 = %s   upsilon_fixing = %s   warm_start = %s   reuse_parent = %s   nodes = %d   fix0 = %d  fix1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
        status,
        String(relaxation),
        String(fixing_rule),
        String(root_bfgs_param_set),
        String(effective_root_bfgs_param_set),
        String(bfgs_param_set),
        string(root_psi0_warm_start),
        String(root_psi0_bfgs_param_set),
        String(root_after_psi0_bfgs_param_set),
        String(upsilon_fixing),
        string(warm_start_parent_upsilon),
        string(reuse_parent_upsilon),
        nodes,
        nfix0,
        nfix1,
        lb,
        ub,
        ub - lb,
    )

    @printf(
        "           integer solns = %d   max int-gap = %.4f   avg int-gap = %.4f   int-gap@opt = %.4f\n",
        n_int_sols,
        int_gap_max,
        int_gap_avg,
        int_gap_opt,
    )

    flush(stdout)

    return nothing
end


# =============================================================================
# Main solver
# =============================================================================

"""
    solve_bnb_ddfact(
        C, s, t;
        relaxation = DDGFact,
        fixing_rule = :dual,
        root_bfgs_param_set = :default,
        root_after_psi0_bfgs_param_set = nothing,
        bfgs_param_set = :default,
        root_psi0_warm_start = false,
        root_psi0_bfgs_param_set = nothing,
        warm_start_parent_upsilon = true,
        reuse_parent_upsilon = false,
        ...
    )

General branch-and-bound for GMESP.

Main options:

  - `relaxation = DDGFact`, `DDGFactplus`, or `DDGFactplusUpsilon`.
  - `fixing_rule = :none`, `:dual`, `:primal`, or `:both`.

BFGS effort options:

  - `root_bfgs_param_set` controls root Upsilon calibration when
    `root_psi0_warm_start = false`.
  - `root_psi0_bfgs_param_set` controls the first root calibration with
    fixed `psi = 0` when `root_psi0_warm_start = true`.
  - `root_after_psi0_bfgs_param_set` controls the second root calibration with
    `psi = psi(gamma)` initialized from the `psi = 0` root solution when
    `root_psi0_warm_start = true`.
  - `bfgs_param_set` controls all non-root node calibrations.

Warm start / reuse options:

  - `root_psi0_warm_start = true` first calibrates root Upsilon with fixed
    `psi = 0`, then uses the resulting `theta` as the start for the usual
    root DDGFactplusUpsilon calibration with `psi = psi(gamma)`.
  - `warm_start_parent_upsilon = true` initializes child-node Upsilon BFGS
    from the parent.
  - `reuse_parent_upsilon = true` skips child/rebound Upsilon BFGS and directly
    reuses the parent/current gamma restricted to the child/rebound keep set.
    In this case, `warm_start_parent_upsilon` is ignored for those nodes.

For `DDGFact`, only `fixing_rule = :none` and `fixing_rule = :dual` are supported.
"""
function solve_bnb_ddfact(
    C::Symmetric,
    s::Int,
    t::Int;
    relaxation = DDGFact,
    fixing_rule::Symbol = :dual,
    psi::Union{Nothing,Float64} = nothing,
    time_limit::Real = 3600.0,
    verbose::Bool = true,
    root_bfgs_param_set::Union{Nothing,Symbol} = nothing,
    root_after_psi0_bfgs_param_set::Union{Nothing,Symbol} = nothing,
    bfgs_param_set::Symbol = :default,
    bfgs_param_sets::Dict = bfgs_param_sets,
    root_psi0_warm_start::Bool = false,
    root_psi0_bfgs_param_set::Union{Nothing,Symbol} = nothing,
    upsilon_fixing::Symbol = :simple,
    warm_start_parent_upsilon::Bool = true,
    reuse_parent_upsilon::Bool = false,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)

    @assert 1 <= t <= s < n "need 1 ≤ t ≤ s < n"

    relaxation = _normalize_relaxation(relaxation)
    fixing_rule = _normalize_fixing_rule(fixing_rule)

    root_bfgs_param_set =
        root_bfgs_param_set === nothing ?
        bfgs_param_set :
        root_bfgs_param_set

    root_psi0_bfgs_param_set =
        root_psi0_bfgs_param_set === nothing ?
        root_bfgs_param_set :
        root_psi0_bfgs_param_set

    root_after_psi0_bfgs_param_set =
        root_after_psi0_bfgs_param_set === nothing ?
        root_bfgs_param_set :
        root_after_psi0_bfgs_param_set

    effective_root_bfgs_param_set =
        root_psi0_warm_start && relaxation == :DDGFactplusUpsilon ?
        root_after_psi0_bfgs_param_set :
        root_bfgs_param_set

    _validate_bnb_options(
        relaxation,
        fixing_rule,
        upsilon_fixing,
    )

    if reuse_parent_upsilon &&
       warm_start_parent_upsilon &&
       relaxation == :DDGFactplusUpsilon &&
       verbose

        println(
            "reuse_parent_upsilon = true: child/rebound nodes reuse gamma directly; " *
            "warm_start_parent_upsilon is ignored for those nodes.",
        )
        flush(stdout)
    end

    _, lb0, S_inc0 = _initial_incumbent(C, s, t)

    state = Ref((
        lb = lb0,
        S_inc = S_inc0,
        n_int_sols = 0,
        int_gap_sum = 0.0,
        int_gap_max = 0.0,
        int_gap_inc = NaN,
    ))

    counters = Ref((
        nodes = 1,
        nfix0 = 0,
        nfix1 = 0,
    ))

    root_node_kwargs = _make_node_kwargs(
        relaxation = relaxation,
        psi = psi,
        bfgs_param_set = effective_root_bfgs_param_set,
        bfgs_param_sets = bfgs_param_sets,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
    )

    node_kwargs = _make_node_kwargs(
        relaxation = relaxation,
        psi = psi,
        bfgs_param_set = bfgs_param_set,
        bfgs_param_sets = bfgs_param_sets,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
    )

    # ------------------------------------------------------------
    # Optional root psi = 0 Upsilon warm start.
    #
    # This first solves the fixed-psi Upsilon calibration at the root.
    # Then its theta is used as the initial point for the usual root
    # DDGFactplusUpsilon calibration with psi = psi(gamma), using
    # effective_root_bfgs_param_set.
    # ------------------------------------------------------------
    root_theta0, root_psi0_calib =
        _root_theta0_from_psi0_warm_start(
            C,
            s,
            t;
            relaxation = relaxation,
            root_psi0_warm_start = root_psi0_warm_start,
            root_psi0_bfgs_param_set = root_psi0_bfgs_param_set,
            bfgs_param_sets = bfgs_param_sets,
            atol = atol,
            verbose = verbose,
        )

    # Root node always solves/calibrates normally.
    # reuse_parent_upsilon only applies to child/rebound nodes.
    r_root = _gmesp_node(
        C,
        s,
        t,
        Int[],
        Int[];
        root_node_kwargs...,
        theta0 = root_theta0,
        fixed_upsilon_gamma = nothing,
    )

    ub_root = r_root.ub

    _register_integer_solution!(
        C,
        s,
        t,
        r_root,
        state,
    )

    st = state[]

    verbose && _print_root_log(
        relaxation,
        fixing_rule,
        root_bfgs_param_set,
        effective_root_bfgs_param_set,
        bfgs_param_set,
        root_psi0_warm_start,
        root_psi0_bfgs_param_set,
        root_after_psi0_bfgs_param_set,
        upsilon_fixing,
        warm_start_parent_upsilon,
        reuse_parent_upsilon,
        st.lb,
        ub_root,
    )

    open =
        r_root.determined || r_root.ub ≤ st.lb ?
        GMESPNode[] :
        [GMESPNode(Int[], Int[], ub_root, r_root.x, r_root.keep, r_root.gamma)]

    report_every = 1000
    next_report = report_every

    t0 = time()
    time_limit_hit = false

    while !isempty(open)
        if time() - t0 ≥ time_limit
            time_limit_hit = true
            break
        end

        sort!(open; by = nd -> nd.ub, rev = true)

        st = state[]

        first(open).ub ≤ st.lb && break

        node = popfirst!(open)

        i = _branch_var(node.keep, node.x, node.F1)
        i == 0 && continue

        F1_child = copy(node.F1)
        push!(F1_child, i)
        sort!(F1_child)

        _bound_child!(
            C,
            s,
            t,
            F1_child,
            node.F0,
            node.keep,
            node.gamma,
            open,
            state,
            counters,
            node_kwargs;
            relaxation = relaxation,
            fixing_rule = fixing_rule,
            warm_start_parent_upsilon = warm_start_parent_upsilon,
            reuse_parent_upsilon = reuse_parent_upsilon,
            atol = atol,
        )

        F0_child = copy(node.F0)
        push!(F0_child, i)
        sort!(F0_child)

        _bound_child!(
            C,
            s,
            t,
            node.F1,
            F0_child,
            node.keep,
            node.gamma,
            open,
            state,
            counters,
            node_kwargs;
            relaxation = relaxation,
            fixing_rule = fixing_rule,
            warm_start_parent_upsilon = warm_start_parent_upsilon,
            reuse_parent_upsilon = reuse_parent_upsilon,
            atol = atol,
        )

        cnt = counters[]
        st = state[]

        if verbose && cnt.nodes >= next_report
            _print_progress(
                cnt.nodes,
                open,
                cnt.nfix0,
                cnt.nfix1,
                st.lb,
            )

            next_report += report_every
        end
    end

    st = state[]
    cnt = counters[]

    tree_exhausted = isempty(open)
    final_ub = tree_exhausted ? st.lb : maximum(nd -> nd.ub, open)

    status =
        time_limit_hit ?
        "TIME LIMIT" :
        (tree_exhausted ? "OPTIMAL (exhausted)" : "OPTIMAL (gap)")

    int_gap_avg =
        st.n_int_sols > 0 ?
        st.int_gap_sum / st.n_int_sols :
        NaN

    int_gap_opt = st.int_gap_inc

    verbose && _print_final_log(
        status,
        relaxation,
        fixing_rule,
        root_bfgs_param_set,
        effective_root_bfgs_param_set,
        bfgs_param_set,
        root_psi0_warm_start,
        root_psi0_bfgs_param_set,
        root_after_psi0_bfgs_param_set,
        upsilon_fixing,
        warm_start_parent_upsilon,
        reuse_parent_upsilon,
        cnt.nodes,
        cnt.nfix0,
        cnt.nfix1,
        st.lb,
        final_ub,
        st.n_int_sols,
        st.int_gap_max,
        int_gap_avg,
        int_gap_opt,
    )

    stats = (
        nodes = cnt.nodes,
        lb = st.lb,
        ub = final_ub,
        gap = final_ub - st.lb,
        root_ub = ub_root,
        nfix0 = cnt.nfix0,
        nfix1 = cnt.nfix1,
        n_int_sols = st.n_int_sols,
        int_gap_max = st.int_gap_max,
        int_gap_avg = int_gap_avg,
        int_gap_opt = int_gap_opt,
        wall_time = time() - t0,
        time_limit_hit = time_limit_hit,
        tree_exhausted = tree_exhausted,
        relaxation = relaxation,
        fixing_rule = fixing_rule,
        root_bfgs_param_set = root_bfgs_param_set,
        effective_root_bfgs_param_set = effective_root_bfgs_param_set,
        bfgs_param_set = bfgs_param_set,
        root_psi0_warm_start = root_psi0_warm_start,
        root_psi0_bfgs_param_set = root_psi0_bfgs_param_set,
        root_after_psi0_bfgs_param_set = root_after_psi0_bfgs_param_set,
        root_psi0_obj =
            root_psi0_calib === nothing ? NaN : root_psi0_calib.obj,
        root_psi0_improved =
            root_psi0_calib === nothing ? missing : root_psi0_calib.improved,
        upsilon_fixing = upsilon_fixing,
        warm_start_parent_upsilon = warm_start_parent_upsilon,
        reuse_parent_upsilon = reuse_parent_upsilon,
    )

    return st.S_inc, stats
end