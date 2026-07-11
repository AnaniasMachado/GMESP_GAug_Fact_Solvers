using LinearAlgebra
using Printf
using Statistics

import ProximalAlgorithms as PA
import ProximalCore
import DifferentiationInterface as DI


mutable struct GScalingProxAlgSmoothObjective
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

    theta_bound::Float64
    project_f_evals::Bool

    relax_knitro_outlev::Union{Nothing,Int}
    relax_knitro_opttol::Union{Nothing,Float64}
    relax_knitro_feastol::Union{Nothing,Float64}
end


struct GScalingProxAlgBoxIndicator
    theta_bound::Float64
end


function (g::GScalingProxAlgBoxIndicator)(theta::AbstractVector{<:Real})
    if !isfinite(g.theta_bound)
        return zero(Float64)
    end

    @inbounds for v in theta
        if v < -g.theta_bound || v > g.theta_bound
            return Inf
        end
    end

    return zero(Float64)
end


function ProximalCore.prox!(
    y::AbstractVector,
    g::GScalingProxAlgBoxIndicator,
    x::AbstractVector,
    gamma,
)
    if isfinite(g.theta_bound)
        @inbounds for i in eachindex(x)
            y[i] = clamp(Float64(x[i]), -g.theta_bound, g.theta_bound)
        end
    else
        y .= x
    end

    return zero(Float64)
end


function _proxalg_effective_theta(
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = Vector{Float64}(theta_raw)

    if f.project_f_evals
        theta .= _optim_project_theta(theta, f.theta_bound)
    end

    return theta
end


function _proxalg_eval_original_oracle_cached!(
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = _proxalg_effective_theta(f, theta_raw)

    return _optim_eval_original_oracle_cached!(
        f.cache,
        f.C,
        theta,
        f.s,
        f.t;
        J1 = f.J1,
        J0 = f.J0,
        atol = f.atol,
        psi_margin = f.psi_margin,
        psi_floor = f.psi_floor,
        t1_reformulation = f.t1_reformulation,
        cache_digits = f.cache_digits,
        counters = f.counters,
        warm_start_state = f.warm_start_state,
        relax_knitro_outlev = f.relax_knitro_outlev,
        relax_knitro_opttol = f.relax_knitro_opttol,
        relax_knitro_feastol = f.relax_knitro_feastol,
    )
end


function (f::GScalingProxAlgSmoothObjective)(
    theta_raw::AbstractVector{<:Real},
)
    val = _proxalg_eval_original_oracle_cached!(f, theta_raw)
    return Float64(val.obj)
end


function _proxalg_value_and_gradient_raw!(
    grad_out::AbstractVector,
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = _proxalg_effective_theta(f, theta_raw)
    val = _proxalg_eval_original_oracle_cached!(f, theta)

    g = _optim_get_subgradient!(
        f.C,
        val,
        f.s,
        f.t;
        atol = f.atol,
        psi_margin = f.psi_margin,
        psi_floor = f.psi_floor,
        psi_derivative = f.psi_derivative,
        counters = f.counters,
    )

    grad_out .= g

    return Float64(val.obj)
end


function ProximalCore.gradient!(
    grad_out::AbstractVector,
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    return _proxalg_value_and_gradient_raw!(grad_out, f, theta_raw)
end


function DI.value_and_gradient(
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    grad_out = similar(theta_raw, Float64)
    val = _proxalg_value_and_gradient_raw!(grad_out, f, theta_raw)
    return val, grad_out
end


function DI.gradient(
    f::GScalingProxAlgSmoothObjective,
    theta_raw::AbstractVector{<:Real},
)
    grad_out = similar(theta_raw, Float64)
    _proxalg_value_and_gradient_raw!(grad_out, f, theta_raw)
    return grad_out
end


function _proxalg_make_smooth_objective(
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

    theta_bound::Float64 = 20.0,
    project_f_evals::Bool = false,

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

    return GScalingProxAlgSmoothObjective(
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

        theta_bound,
        project_f_evals,

        relax_knitro_outlev,
        relax_knitro_opttol,
        relax_knitro_feastol,
    )
end


function _proxalg_initial_theta(
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


function _proxalg_run_forward_backward(
    alg,
    theta_start::Vector{Float64},
    f::GScalingProxAlgSmoothObjective,
    g::GScalingProxAlgBoxIndicator;
    gamma::Float64,
    adaptive::Bool,
    minimum_gamma::Float64,
    reduce_gamma::Float64,
    increase_gamma::Float64,
)
    Lf = 1.0 / gamma

    return alg(
        ;
        x0 = theta_start,
        f = f,
        g = g,
        Lf = Lf,
        gamma = gamma,
        adaptive = adaptive,
        minimum_gamma = minimum_gamma,
    )
end


function run_upsilon_proxalg_pg_ddfactplus(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    theta_bound::Float64 = 20.0,
    project_f_evals::Bool = false,

    gamma::Float64 = 1e-3,
    adaptive::Bool = true,
    minimum_gamma::Float64 = 1e-8,
    reduce_gamma::Float64 = 0.5,
    increase_gamma::Float64 = 1.0,

    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    maxiters::Int = 50,
    tol::Float64 = 1e-4,
    cache_digits::Int = 12,

    verbose::Bool = true,
    method_name::String = "ProximalAlgorithms.ForwardBackward",
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    @assert maxiters >= 1
    @assert gamma > 0.0
    @assert minimum_gamma > 0.0
    @assert 0.0 < reduce_gamma < 1.0
    @assert increase_gamma > 0.0

    theta_start = _proxalg_initial_theta(
        n;
        theta0 = theta0,
        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,
        theta_bound = theta_bound,
    )

    f = _proxalg_make_smooth_objective(
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

        theta_bound = theta_bound,
        project_f_evals = project_f_evals,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    g = GScalingProxAlgBoxIndicator(theta_bound)

    unscaled_val = _proxalg_eval_original_oracle_cached!(f, zeros(Float64, n))
    initial_val = _proxalg_eval_original_oracle_cached!(f, theta_start)

    alg = PA.ForwardBackward(
        maxit = maxiters,
        tol = tol,
        verbose = verbose,
        freq = 1,
    )

    theta_pg, iterations = _proxalg_run_forward_backward(
        alg,
        theta_start,
        f,
        g;
        gamma = gamma,
        adaptive = adaptive,
        minimum_gamma = minimum_gamma,
        reduce_gamma = reduce_gamma,
        increase_gamma = increase_gamma,
    )

    theta_final = _optim_project_theta(Vector{Float64}(theta_pg), theta_bound)

    final_val = _proxalg_eval_original_oracle_cached!(f, theta_final)

    g_final = _optim_get_subgradient!(
        f.C,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = f.counters,
    )

    g_norm = _optim_norm2(g_final)

    if verbose
        @printf("\nProximalAlgorithms proximal gradient finished: %s\n", method_name)
        @printf("  unscaled obj      = %.12e\n", unscaled_val.obj)
        @printf("  initial obj       = %.12e\n", initial_val.obj)
        @printf("  final obj         = %.12e\n", final_val.obj)
        @printf("  ||g_final||       = %.12e\n", g_norm)
        @printf("  iterations        = %s\n", string(iterations))
        @printf("  objective solves  = %d\n", f.counters.num_objective_solves)
        @printf("  obj cache hits    = %d / %d\n",
            f.counters.num_objective_cache_hits,
            f.counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits   = %d / %d\n",
            f.counters.num_subgradient_cache_hits,
            f.counters.num_subgradient_requests,
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

        proxalg_smooth_objective = f,
        proxalg_box_indicator = g,
        proxalg_iterations = iterations,

        cache = f.cache,
        counters = f.counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            f.counters.num_objective_cache_hits,
            f.counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            f.counters.num_subgradient_cache_hits,
            f.counters.num_subgradient_requests,
        ),
    )
end


function _proxalg_run_composite_algorithm(
    alg,
    theta_start::Vector{Float64},
    f::GScalingProxAlgSmoothObjective,
    g::GScalingProxAlgBoxIndicator;
    gamma::Float64,
    adaptive::Bool,
    minimum_gamma::Float64,
    max_backtracks::Int,
)
    Lf = 1.0 / gamma

    return alg(
        ;
        x0 = theta_start,
        f = f,
        g = g,
        Lf = Lf,
        gamma = gamma,
        adaptive = adaptive,
        minimum_gamma = minimum_gamma,
        max_backtracks = max_backtracks,
    )
end


function run_upsilon_proxalg_composite_ddfactplus(
    C,
    s::Int,
    t::Int,
    alg_constructor::Function;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    theta_bound::Float64 = 20.0,
    project_f_evals::Bool = false,

    gamma::Float64 = 1e-3,
    adaptive::Bool = false,
    minimum_gamma::Float64 = 1e-12,
    max_backtracks::Int = 20,

    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    maxiters::Int = 50,
    tol::Float64 = 1e-4,
    cache_digits::Int = 12,

    verbose::Bool = true,
    method_name::String = "ProximalAlgorithms.Composite",
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    @assert maxiters >= 1
    @assert gamma > 0.0
    @assert minimum_gamma > 0.0
    @assert max_backtracks >= 1

    theta_start = _proxalg_initial_theta(
        n;
        theta0 = theta0,
        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,
        theta_bound = theta_bound,
    )

    f = _proxalg_make_smooth_objective(
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

        theta_bound = theta_bound,
        project_f_evals = project_f_evals,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    g = GScalingProxAlgBoxIndicator(theta_bound)

    unscaled_val = _proxalg_eval_original_oracle_cached!(f, zeros(Float64, n))
    initial_val = _proxalg_eval_original_oracle_cached!(f, theta_start)

    alg = alg_constructor(
        maxit = maxiters,
        tol = tol,
        verbose = verbose,
        freq = 1,
    )

    theta_alg, iterations = _proxalg_run_composite_algorithm(
        alg,
        theta_start,
        f,
        g;
        gamma = gamma,
        adaptive = adaptive,
        minimum_gamma = minimum_gamma,
        max_backtracks = max_backtracks,
    )

    theta_final = _optim_project_theta(Vector{Float64}(theta_alg), theta_bound)

    final_val = _proxalg_eval_original_oracle_cached!(f, theta_final)

    g_final = _optim_get_subgradient!(
        f.C,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = f.counters,
    )

    g_norm = _optim_norm2(g_final)

    if verbose
        @printf("\nProximalAlgorithms method finished: %s\n", method_name)
        @printf("  unscaled obj      = %.12e\n", unscaled_val.obj)
        @printf("  initial obj       = %.12e\n", initial_val.obj)
        @printf("  final obj         = %.12e\n", final_val.obj)
        @printf("  ||g_final||       = %.12e\n", g_norm)
        @printf("  iterations        = %s\n", string(iterations))
        @printf("  objective solves  = %d\n", f.counters.num_objective_solves)
        @printf("  obj cache hits    = %d / %d\n",
            f.counters.num_objective_cache_hits,
            f.counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits   = %d / %d\n",
            f.counters.num_subgradient_cache_hits,
            f.counters.num_subgradient_requests,
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

        proxalg_smooth_objective = f,
        proxalg_box_indicator = g,
        proxalg_iterations = iterations,

        cache = f.cache,
        counters = f.counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            f.counters.num_objective_cache_hits,
            f.counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            f.counters.num_subgradient_cache_hits,
            f.counters.num_subgradient_requests,
        ),
    )
end