using LinearAlgebra
using Printf
using DataStructures: BinaryMaxHeap

# =============================================================================
# General branch-and-bound for GMESP
#
# Requires:
#   include("bnb_util.jl")
#
# Main options:
#   - relaxation = DDGFact, DDGFactplus, or DDGFactplusUpsilon
#   - fixing_rule = :none, :dual, :primal, or :both
#   - calibration_method = :bfgs, :ppa_one, or :ppa_full for DDGFactplusUpsilon
#   - root_calibration_params controls root-node Upsilon calibration
#   - node_calibration_params controls child/rebound-node Upsilon calibration
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
    calibration_method,
    calibration_params,
    atol,
    psi_margin,
    psi_floor,
    upsilon_fixing,
    need_dual,
)
    return (
        relaxation = relaxation,
        psi = psi,
        calibration_method = calibration_method,
        calibration_params = calibration_params,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
        need_dual = need_dual,
    )
end


# =============================================================================
# Open-list utilities
# =============================================================================


function _push_if_survives!(
    open,
    F1::Vector{Int},
    F0::Vector{Int},
    r,
    lb::Float64,
    depth::Int,
)
    r.determined && return false
    r.ub ≤ lb && return false

    push!(open, GMESPNode(F1, F0, r.ub, r.x, r.keep, depth, r.gamma))

    return true
end


# =============================================================================
# Child bounding
# =============================================================================


function _bound_child!(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int},
    parent::GMESPNode,
    open,
    state::Base.RefValue,
    counters::Base.RefValue,
    node_kwargs;
    fixing_rule::Symbol,
    atol::Float64,
    recalibrate_k::Int,
)
    child_depth = parent.depth + 1

    child_keep = setdiff(1:size(C, 1), F0)

    recalibrate_child =
        node_kwargs.relaxation != :DDGFactplusUpsilon ||
        parent.gamma === nothing ||
        child_depth % recalibrate_k == 0

    fixed_gamma_child =
        recalibrate_child || node_kwargs.relaxation != :DDGFactplusUpsilon ?
        nothing :
        _restrict_gamma_to_child_keep(parent.keep, parent.gamma, child_keep)

    child_kwargs = merge(
        node_kwargs,
        (
            calibrate_upsilon = recalibrate_child,
            fixed_gamma = fixed_gamma_child,
        ),
    )

    # First bound of the child.
    r = _gmesp_node(
        C,
        s,
        t,
        F1,
        F0;
        child_kwargs...,
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

    local F1f
    local F0f

    fixing_time = @elapsed begin
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
    end

    _add_bnb_timing!(:variable_fixing_time, fixing_time)
    _add_bnb_timing!(:variable_fixing_calls, 1.0)

    if length(F1f) > length(F1) || length(F0f) > length(F0)
        cnt = counters[]

        counters[] = (
            nodes = cnt.nodes,
            nfix0 = cnt.nfix0 + length(F0f) - length(F0),
            nfix1 = cnt.nfix1 + length(F1f) - length(F1),
        )

        # Refit after variable fixing, but do not recalibrate again.
        refit_keep = setdiff(1:size(C, 1), F0f)

        refit_gamma =
            node_kwargs.relaxation == :DDGFactplusUpsilon && r.gamma !== nothing ?
            _restrict_gamma_to_child_keep(r.keep, r.gamma, refit_keep) :
            nothing

        refit_kwargs = merge(
            node_kwargs,
            (
                calibrate_upsilon = node_kwargs.relaxation != :DDGFactplusUpsilon,
                fixed_gamma = refit_gamma,
            ),
        )

        r = _gmesp_node(
            C,
            s,
            t,
            F1f,
            F0f;
            refit_kwargs...,
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
        child_depth,
    )

    return nothing
end


# =============================================================================
# Logging
# =============================================================================

function _print_root_log(
    relaxation::Symbol,
    fixing_rule::Symbol,
    calibration_method::Symbol,
    upsilon_fixing::Symbol,
    lb::Float64,
    ub_root::Float64,
)
    @printf(
        "root:  relaxation = %s   fixing = %s   calibration = %s   upsilon_fixing = %s   lb = %.4f   ub = %.4f   gap = %.4f\n",
        String(relaxation),
        String(fixing_rule),
        String(calibration_method),
        String(upsilon_fixing),
        lb,
        ub_root,
        ub_root - lb,
    )

    flush(stdout)

    return nothing
end


function _print_progress(
    nodes::Int,
    open,
    nfix0::Int,
    nfix1::Int,
    lb::Float64,
)
    best_ub = isempty(open) ? lb : first(open).ub

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
    calibration_method::Symbol,
    upsilon_fixing::Symbol,
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
        "done [%s].  relaxation = %s   fixing = %s   calibration = %s   upsilon_fixing = %s   nodes = %d   fix0 = %d  fix1 = %d   lb = %.4f   ub = %.4f   gap = %.4f\n",
        status,
        String(relaxation),
        String(fixing_rule),
        String(calibration_method),
        String(upsilon_fixing),
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
        calibration_method = :bfgs,
        root_calibration_params = Dict{Symbol,Any}(),
        node_calibration_params = Dict{Symbol,Any}(),
        ...
    )

General branch-and-bound for GMESP.

Main options:

  - `relaxation = DDGFact`, `DDGFactplus`, or `DDGFactplusUpsilon`.
  - `fixing_rule = :none`, `:dual`, `:primal`, or `:both`.
  - `calibration_method = :bfgs`, `:ppa_one`, or `:ppa_full` for DDGFactplusUpsilon.
  - `root_calibration_params` controls root-node Upsilon calibration.
  - `node_calibration_params` controls all child and rebound node calibrations.

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
    calibration_method::Symbol = :bfgs,
    root_calibration_params = Dict{Symbol,Any}(),
    node_calibration_params = Dict{Symbol,Any}(),
    upsilon_fixing::Symbol = :simple,
    recalibrate_k::Int = 1,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)

    @assert 1 <= t <= s < n "need 1 ≤ t ≤ s < n"
    @assert recalibrate_k >= 1 "recalibrate_k must be at least 1"

    relaxation = _normalize_relaxation(relaxation)
    fixing_rule = _normalize_fixing_rule(fixing_rule)
    calibration_method = _normalize_calibration_method(calibration_method)

    _validate_bnb_options(
        relaxation,
        fixing_rule,
        upsilon_fixing,
        calibration_method,
    )

    _, lb0, S_inc0 = _initial_incumbent(C, s, t)
    _reset_bnb_timing!()

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

    need_dual = fixing_rule in (:dual, :both)

    root_node_kwargs = _make_node_kwargs(
        relaxation = relaxation,
        psi = psi,
        calibration_method = calibration_method,
        calibration_params = root_calibration_params,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
        need_dual = need_dual,
    )

    child_node_kwargs = _make_node_kwargs(
        relaxation = relaxation,
        psi = psi,
        calibration_method = calibration_method,
        calibration_params = node_calibration_params,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        upsilon_fixing = upsilon_fixing,
        need_dual = need_dual,
    )

    r_root = _gmesp_node(
        C,
        s,
        t,
        Int[],
        Int[];
        root_node_kwargs...,
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
        calibration_method,
        upsilon_fixing,
        st.lb,
        ub_root,
    )

    open = BinaryMaxHeap{GMESPNode}()

    if !(r_root.determined || r_root.ub ≤ st.lb)
        push!(
            open,
            GMESPNode(
                Int[],
                Int[],
                ub_root,
                r_root.x,
                r_root.keep,
                0,
                r_root.gamma,
            ),
        )
    end

    report_every = 1000
    next_report = report_every

    t0 = time()
    time_limit_hit = false

    while !isempty(open)
        if time() - t0 ≥ time_limit
            time_limit_hit = true
            break
        end

        local node

        open_list_time = @elapsed begin
            st = state[]

            if first(open).ub ≤ st.lb
                node = nothing
            else
                node = pop!(open)
            end
        end

        _add_bnb_timing!(:open_list_time, open_list_time)

        node === nothing && break

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
            node,
            open,
            state,
            counters,
            child_node_kwargs;
            fixing_rule = fixing_rule,
            atol = atol,
            recalibrate_k = recalibrate_k,
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
            node,
            open,
            state,
            counters,
            child_node_kwargs;
            fixing_rule = fixing_rule,
            atol = atol,
            recalibrate_k = recalibrate_k,
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
    final_ub = tree_exhausted ? st.lb : first(open).ub

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
        calibration_method,
        upsilon_fixing,
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

    bnb_timing = _copy_bnb_timing()

    dual_solution_time = get(bnb_timing, :dual_solution_time, 0.0)
    variable_fixing_direct_time = get(bnb_timing, :variable_fixing_time, 0.0)

    variable_fixing_total_time =
        fixing_rule in (:dual, :both) ?
        variable_fixing_direct_time + dual_solution_time :
        variable_fixing_direct_time

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
        calibration_method = calibration_method,
        upsilon_fixing = upsilon_fixing,
        root_calibration_params = root_calibration_params,
        node_calibration_params = node_calibration_params,
        recalibrate_k = recalibrate_k,

        knitro_time = get(bnb_timing, :knitro_time, 0.0),
        relaxation_solve_time = get(bnb_timing, :relaxation_solve_time, 0.0),
        upsilon_calibration_time = get(bnb_timing, :upsilon_calibration_time, 0.0),
        factorization_time = get(bnb_timing, :factorization_time, 0.0),
        bound_computation_time = get(bnb_timing, :bound_computation_time, 0.0),
        open_list_time = get(bnb_timing, :open_list_time, 0.0),
        node_setup_time = get(bnb_timing, :node_setup_time, 0.0),

        dual_solution_time = dual_solution_time,
        variable_fixing_direct_time = variable_fixing_direct_time,
        variable_fixing_time = variable_fixing_total_time,
        variable_fixing_calls = Int(round(get(bnb_timing, :variable_fixing_calls, 0.0))),
    )

    return st.S_inc, stats
end
