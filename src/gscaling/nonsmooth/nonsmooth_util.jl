using LinearAlgebra
using Printf
using Statistics

import NonSmoothProblems as NSP
import NonSmoothSolvers as NSS


mutable struct GScalingNonSmoothProblem <: NSP.NonSmoothPb
    C::Symmetric{Float64,Matrix{Float64}}
    s::Int
    t::Int
    J1::Vector{Int}
    J0::Vector{Int}

    atol::Float64
    psi_margin::Float64
    psi_floor::Float64
    psi_derivative::Bool
    t1_reformulation::Bool

    cache_digits::Int
    cache::Dict{String,GScalingOptimOracleValue}
    counters::GScalingOptimEvalCounters
    warm_start_state::GScalingOptimWarmStartState

    theta_center::Union{Nothing,Vector{Float64}}
    rho::Float64

    theta_bound::Float64
    project_evals::Bool

    relax_knitro_outlev::Union{Nothing,Int}
    relax_knitro_opttol::Union{Nothing,Float64}
    relax_knitro_feastol::Union{Nothing,Float64}
end


function _nss_effective_theta(
    pb::GScalingNonSmoothProblem,
    theta_raw::AbstractVector{<:Real},
)
    theta = Vector{Float64}(theta_raw)

    if pb.project_evals
        theta .= _optim_project_theta(theta, pb.theta_bound)
    end

    return theta
end


function _nss_eval_original_oracle_cached!(
    pb::GScalingNonSmoothProblem,
    theta_raw::AbstractVector{<:Real},
)
    theta = _nss_effective_theta(pb, theta_raw)

    return _optim_eval_original_oracle_cached!(
        pb.cache,
        pb.C,
        theta,
        pb.s,
        pb.t;
        J1 = pb.J1,
        J0 = pb.J0,
        atol = pb.atol,
        psi_margin = pb.psi_margin,
        psi_floor = pb.psi_floor,
        t1_reformulation = pb.t1_reformulation,
        cache_digits = pb.cache_digits,
        counters = pb.counters,
        warm_start_state = pb.warm_start_state,
        relax_knitro_outlev = pb.relax_knitro_outlev,
        relax_knitro_opttol = pb.relax_knitro_opttol,
        relax_knitro_feastol = pb.relax_knitro_feastol,
    )
end


function NSP.F(
    pb::GScalingNonSmoothProblem,
    theta_raw::AbstractVector{<:Real},
)
    theta = _nss_effective_theta(pb, theta_raw)
    val = _nss_eval_original_oracle_cached!(pb, theta)

    obj = val.obj

    if pb.theta_center !== nothing
        obj += (0.5 / pb.rho) * sum(
            (theta[i] - pb.theta_center[i])^2 for i in eachindex(theta)
        )
    end

    return obj
end


function NSP.∂F_elt(
    pb::GScalingNonSmoothProblem,
    theta_raw::AbstractVector{<:Real},
)
    theta = _nss_effective_theta(pb, theta_raw)
    val = _nss_eval_original_oracle_cached!(pb, theta)

    g = Vector{Float64}(
        _optim_get_subgradient!(
            pb.C,
            val,
            pb.s,
            pb.t;
            atol = pb.atol,
            psi_margin = pb.psi_margin,
            psi_floor = pb.psi_floor,
            psi_derivative = pb.psi_derivative,
            counters = pb.counters,
        ),
    )

    if pb.theta_center !== nothing
        g .+= (theta .- pb.theta_center) ./ pb.rho
    end

    return g
end


function NSP.is_differentiable(
    pb::GScalingNonSmoothProblem,
    theta_raw::AbstractVector{<:Real},
)
    return true
end


function _nss_make_problem(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,

    cache_digits::Int,

    theta_center::Union{Nothing,Vector{Float64}} = nothing,
    rho::Float64 = 1.0,

    theta_bound::Float64 = 20.0,
    project_evals::Bool = false,

    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    J1_clean = sort(unique(collect(Int, J1)))
    J0_clean = sort(unique(collect(Int, J0)))

    @assert all(i -> 1 <= i <= n, J1_clean)
    @assert all(i -> 1 <= i <= n, J0_clean)
    @assert isempty(intersect(J1_clean, J0_clean))
    @assert length(J1_clean) <= s
    @assert s <= n - length(J0_clean)
    @assert 1 <= t <= s <= n

    if theta_center !== nothing
        @assert rho > 0.0

        if length(theta_center) != n
            error("theta_center must have length equal to size(C, 1).")
        end

        theta_center = _optim_project_theta(copy(theta_center), theta_bound)
    end

    return GScalingNonSmoothProblem(
        Csym,
        s,
        t,
        J1_clean,
        J0_clean,

        atol,
        psi_margin,
        psi_floor,
        psi_derivative,
        t1_reformulation,

        cache_digits,
        Dict{String,GScalingOptimOracleValue}(),
        GScalingOptimEvalCounters(),
        GScalingOptimWarmStartState(),

        theta_center,
        rho,

        theta_bound,
        project_evals,

        relax_knitro_outlev,
        relax_knitro_opttol,
        relax_knitro_feastol,
    )
end


function _nss_initial_theta(
    n::Int;
    theta0::Union{Nothing,Vector{Float64}},
    theta_perturbation::Float64,
    center_initial_theta::Bool,
    theta_bound::Float64,
)
    theta_start = if theta0 !== nothing
        copy(theta0)
    elseif theta_perturbation == 0.0
        zeros(Float64, n)
    else
        theta_perturbation .* randn(n)
    end

    if length(theta_start) != n
        error("theta0 must have length equal to size(C, 1).")
    end

    if center_initial_theta
        theta_start .-= mean(theta_start)
    end

    theta_start = _optim_project_theta(theta_start, theta_bound)

    return theta_start
end


function _nss_run!(
    pb::GScalingNonSmoothProblem,
    theta_start::Vector{Float64},
    method;
    maxiters::Int,
    time_limit::Float64,
    show_trace::Bool,
    trace_length::Int,
)
    state = NSS.initial_state(method, copy(theta_start), pb)

    trace = Any[]

    theta_best = copy(theta_start)
    fx_best = Float64(NSP.F(pb, theta_best))

    push!(trace, (
        x = copy(theta_best),
        Fx = fx_best,
        F_x = fx_best,
        it = 0,
        time = 0.0,
        status = "initial",
    ))

    start_time = time()

    for it in 1:maxiters
        elapsed = time() - start_time

        if elapsed > time_limit
            push!(trace, (
                x = copy(theta_best),
                Fx = fx_best,
                F_x = fx_best,
                it = it,
                time = elapsed,
                status = "time_limit",
            ))
            break
        end

        iterationstatus = "unknown"

        try
            updateinformation, iterationstatus =
                NSS.update_iterate!(state, method, pb)
        catch err
            theta_candidate = copy(theta_best)

            try
                theta_candidate =
                    Vector{Float64}(NSS.get_minimizer_candidate(state))
            catch
                theta_candidate = copy(theta_best)
            end

            fx_candidate = Float64(NSP.F(pb, theta_candidate))

            if fx_candidate < fx_best
                theta_best .= theta_candidate
                fx_best = fx_candidate
            end

            elapsed = time() - start_time

            push!(trace, (
                x = copy(theta_best),
                Fx = fx_best,
                F_x = fx_best,
                it = it,
                time = elapsed,
                status = "solver_error",
                error = sprint(showerror, err),
            ))

            @warn "NonSmoothSolvers iteration failed; returning best candidate found so far." method=typeof(method) iteration=it error=sprint(showerror, err)

            break
        end

        theta_current = Vector{Float64}(NSS.get_minimizer_candidate(state))
        fx_current = Float64(NSP.F(pb, theta_current))
        elapsed = time() - start_time

        if fx_current < fx_best
            theta_best .= theta_current
            fx_best = fx_current
        end

        push!(trace, (
            x = copy(theta_current),
            Fx = fx_current,
            F_x = fx_current,
            it = it,
            time = elapsed,
            status = string(iterationstatus),
        ))

        if show_trace && (
            it == 1 ||
            it == maxiters ||
            mod(it, max(1, ceil(Int, maxiters / trace_length))) == 0
        )
            @printf(
                "  NSS it = %4d    F = %.12e    best = %.12e    status = %s    time = %.2fs\n",
                it,
                fx_current,
                fx_best,
                string(iterationstatus),
                elapsed,
            )
        end

        if string(iterationstatus) == "problem_solved"
            break
        end

        if string(iterationstatus) == "iteration_failed"
            break
        end
    end

    return theta_best, trace
end


function _nss_trace_vector(trace)
    return collect(trace)
end


function _nss_iterations(trace)
    tr = _nss_trace_vector(trace)
    return max(length(tr) - 1, 0)
end


function _nss_trace_obj(trace_item)
    if hasproperty(trace_item, :Fx)
        return Float64(getproperty(trace_item, :Fx))
    elseif hasproperty(trace_item, :F_x)
        return Float64(getproperty(trace_item, :F_x))
    else
        return NaN
    end
end


function _nss_last_trace_value(trace)
    tr = _nss_trace_vector(trace)
    return isempty(tr) ? NaN : _nss_trace_obj(tr[end])
end


function run_upsilon_nss_ddfactplus(
    C,
    s::Int,
    t::Int,
    method;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    theta_bound::Float64 = 20.0,
    project_evals::Bool = false,

    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    maxiters::Int = 50,
    time_limit::Float64 = Inf,
    show_trace::Bool = true,
    trace_length::Int = 50,

    cache_digits::Int = 12,
    verbose::Bool = true,
    method_name::String = string(typeof(method)),
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    @assert maxiters >= 1

    theta_start = _nss_initial_theta(
        n;
        theta0 = theta0,
        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,
        theta_bound = theta_bound,
    )

    pb = _nss_make_problem(
        Csym,
        s,
        t;
        J1 = J1,
        J0 = J0,

        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,

        cache_digits = cache_digits,

        theta_center = nothing,
        rho = 1.0,

        theta_bound = theta_bound,
        project_evals = project_evals,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    unscaled_val = _nss_eval_original_oracle_cached!(pb, zeros(Float64, n))
    initial_val = _nss_eval_original_oracle_cached!(pb, theta_start)

    theta_nss, trace = _nss_run!(
        pb,
        theta_start,
        method;
        maxiters = maxiters,
        time_limit = time_limit,
        show_trace = show_trace,
        trace_length = trace_length,
    )

    theta_final = Vector{Float64}(theta_nss)
    theta_final = _optim_project_theta(theta_final, theta_bound)

    final_val = _nss_eval_original_oracle_cached!(pb, theta_final)

    g_final = _optim_get_subgradient!(
        pb.C,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = pb.counters,
    )

    g_norm = _optim_norm2(g_final)

    if verbose
        @printf("\nNonSmoothSolvers original calibration finished: %s\n", method_name)
        @printf("  unscaled obj      = %.12e\n", unscaled_val.obj)
        @printf("  initial obj       = %.12e\n", initial_val.obj)
        @printf("  final obj         = %.12e\n", final_val.obj)
        @printf("  ||g_final||       = %.12e\n", g_norm)
        @printf("  iterations        = %d\n", _nss_iterations(trace))
        @printf("  objective solves  = %d\n", pb.counters.num_objective_solves)
        @printf("  obj cache hits    = %d / %d\n",
            pb.counters.num_objective_cache_hits,
            pb.counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits   = %d / %d\n",
            pb.counters.num_subgradient_cache_hits,
            pb.counters.num_subgradient_requests,
        )
    end

    return (
        method_name = method_name,

        theta = theta_final,
        gamma = final_val.gamma,
        psi = final_val.psi,
        lambda_min = final_val.lambda_min,
        obj = final_val.obj,
        g = g_final,
        g_norm = g_norm,
        x = final_val.x,
        y = final_val.y,

        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        unscaled_obj = unscaled_val.obj,

        nss_problem = pb,
        nss_method = method,
        nss_trace = trace,
        trace_final_obj = _nss_last_trace_value(trace),
        iterations = _nss_iterations(trace),

        cache = pb.cache,
        counters = pb.counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            pb.counters.num_objective_cache_hits,
            pb.counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            pb.counters.num_subgradient_cache_hits,
            pb.counters.num_subgradient_requests,
        ),
    )
end


function run_upsilon_prox_nss_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int,
    method;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    rho::Float64 = 1e-2,
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_bound::Float64 = 20.0,
    project_evals::Bool = false,

    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    maxiters::Int = 50,
    time_limit::Float64 = Inf,
    show_trace::Bool = true,
    trace_length::Int = 50,

    cache_digits::Int = 12,
    verbose::Bool = true,
    method_name::String = string(typeof(method)),
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    @assert rho > 0.0
    @assert maxiters >= 1

    if length(theta_center) != n
        error("theta_center must have length equal to size(C, 1).")
    end

    theta_center_projected = _optim_project_theta(copy(theta_center), theta_bound)

    theta_start = if theta0 !== nothing
        copy(theta0)
    else
        copy(theta_center_projected)
    end

    if length(theta_start) != n
        error("theta0 must have length equal to size(C, 1).")
    end

    theta_start = _optim_project_theta(theta_start, theta_bound)

    pb = _nss_make_problem(
        Csym,
        s,
        t;
        J1 = J1,
        J0 = J0,

        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,

        cache_digits = cache_digits,

        theta_center = theta_center_projected,
        rho = rho,

        theta_bound = theta_bound,
        project_evals = project_evals,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    center_val = _nss_eval_original_oracle_cached!(pb, theta_center_projected)
    initial_val = _nss_eval_original_oracle_cached!(pb, theta_start)

    initial_prox_obj =
        initial_val.obj +
        (0.5 / rho) * sum(
            (theta_start[i] - theta_center_projected[i])^2 for i in 1:n
        )

    theta_nss, trace = _nss_run!(
        pb,
        theta_start,
        method;
        maxiters = maxiters,
        time_limit = time_limit,
        show_trace = show_trace,
        trace_length = trace_length,
    )

    theta_final = Vector{Float64}(theta_nss)
    theta_final = _optim_project_theta(theta_final, theta_bound)

    final_val = _nss_eval_original_oracle_cached!(pb, theta_final)

    g_original_final = _optim_get_subgradient!(
        pb.C,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = pb.counters,
    )

    prox_gradient =
        g_original_final .+ (theta_final .- theta_center_projected) ./ rho

    prox_obj =
        final_val.obj +
        (0.5 / rho) * sum(
            (theta_final[i] - theta_center_projected[i])^2 for i in 1:n
        )

    if verbose
        @printf("\nNonSmoothSolvers proximal subproblem finished: %s\n", method_name)
        @printf("  center V(theta)     = %.12e\n", center_val.obj)
        @printf("  initial prox obj    = %.12e\n", initial_prox_obj)
        @printf("  final prox obj      = %.12e\n", prox_obj)
        @printf("  final V(theta)      = %.12e\n", final_val.obj)
        @printf("  prox step norm      = %.12e\n", _optim_norm2(theta_final .- theta_center_projected))
        @printf("  ||prox grad||       = %.12e\n", _optim_norm2(prox_gradient))
        @printf("  iterations          = %d\n", _nss_iterations(trace))
        @printf("  objective solves    = %d\n", pb.counters.num_objective_solves)
        @printf("  obj cache hits      = %d / %d\n",
            pb.counters.num_objective_cache_hits,
            pb.counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits     = %d / %d\n",
            pb.counters.num_subgradient_cache_hits,
            pb.counters.num_subgradient_requests,
        )
    end

    return (
        method_name = method_name,

        theta = theta_final,
        gamma = final_val.gamma,
        psi = final_val.psi,
        lambda_min = final_val.lambda_min,

        obj = final_val.obj,
        prox_obj = prox_obj,
        g_original = g_original_final,
        g_prox = prox_gradient,
        g_prox_norm = _optim_norm2(prox_gradient),
        prox_step_norm = _optim_norm2(theta_final .- theta_center_projected),

        x = final_val.x,
        y = final_val.y,

        theta_center = theta_center_projected,
        center_obj = center_val.obj,
        center_prox_obj = center_val.obj,
        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        initial_prox_obj = initial_prox_obj,

        nss_problem = pb,
        nss_method = method,
        nss_trace = trace,
        trace_final_obj = _nss_last_trace_value(trace),
        iterations = _nss_iterations(trace),

        cache = pb.cache,
        counters = pb.counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            pb.counters.num_objective_cache_hits,
            pb.counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            pb.counters.num_subgradient_cache_hits,
            pb.counters.num_subgradient_requests,
        ),
    )
end