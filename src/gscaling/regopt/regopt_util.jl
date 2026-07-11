using LinearAlgebra
using Printf
using Statistics

import RegularizedOptimization as RO
import ShiftedProximalOperators as SPO


mutable struct GScalingRegOptObjective
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
    project_evals::Bool

    relax_knitro_outlev::Union{Nothing,Int}
    relax_knitro_opttol::Union{Nothing,Float64}
    relax_knitro_feastol::Union{Nothing,Float64}
end


struct GScalingRegOptQuadraticRegularizer{
    R <: Real,
    V <: AbstractVector{R},
}
    theta_center::V
    rho::R
end


function (h::GScalingRegOptQuadraticRegularizer)(
    theta::AbstractVector{<:Real},
)
    d = theta .- h.theta_center
    return dot(d, d) / (2.0 * h.rho)
end


mutable struct GScalingRegOptShiftedQuadratic{
    R <: Real,
    Vc <: AbstractVector{R},
    V0 <: AbstractVector{R},
    V1 <: AbstractVector{R},
    V2 <: AbstractVector{R},
} <: SPO.ShiftedProximableFunction
    h::GScalingRegOptQuadraticRegularizer{R,Vc}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2
end


function GScalingRegOptShiftedQuadratic(
    h::GScalingRegOptQuadraticRegularizer{R,Vc},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
) where {R <: Real, Vc <: AbstractVector{R}}
    sol = similar(xk)
    xsy = similar(xk)

    return GScalingRegOptShiftedQuadratic{
        R,
        Vc,
        typeof(xk),
        typeof(sj),
        typeof(sol),
    }(
        h,
        xk,
        sj,
        sol,
        shifted_twice,
        xsy,
    )
end


function SPO.shifted(
    h::GScalingRegOptQuadraticRegularizer{R,Vc},
    xk::AbstractVector{R},
) where {R <: Real, Vc <: AbstractVector{R}}
    return GScalingRegOptShiftedQuadratic(
        h,
        xk,
        zero(xk),
        false,
    )
end


function SPO.shifted(
    ψ::GScalingRegOptShiftedQuadratic{R,Vc,V0,V1,V2},
    sj::AbstractVector{R},
) where {
    R <: Real,
    Vc <: AbstractVector{R},
    V0 <: AbstractVector{R},
    V1 <: AbstractVector{R},
    V2 <: AbstractVector{R},
}
    return GScalingRegOptShiftedQuadratic(
        ψ.h,
        ψ.xk,
        sj,
        true,
    )
end


function SPO.prox!(
    y::AbstractVector{R},
    ψ::GScalingRegOptShiftedQuadratic{R,Vc,V0,V1,V2},
    q::AbstractVector{R},
    σ::R,
) where {
    R <: Real,
    Vc <: AbstractVector{R},
    V0 <: AbstractVector{R},
    V1 <: AbstractVector{R},
    V2 <: AbstractVector{R},
}
    ρ = ψ.h.rho
    θc = ψ.h.theta_center

    @inbounds for i in eachindex(y)
        # Solves:
        #
        #   min_y  (1 / (2σ)) * ||y - q||^2
        #        + (1 / (2ρ)) * ||xk + sj + y - theta_center||^2
        #
        # Closed form:
        #
        #   y = (ρ q - σ (xk + sj - theta_center)) / (ρ + σ)
        a = ψ.xk[i] + ψ.sj[i] - θc[i]
        y[i] = (ρ * q[i] - σ * a) / (ρ + σ)
    end

    return y
end


function _regopt_effective_theta(
    obj::GScalingRegOptObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = Vector{Float64}(theta_raw)

    if obj.project_evals
        theta .= _optim_project_theta(theta, obj.theta_bound)
    end

    return theta
end


function _regopt_eval_original_oracle_cached!(
    obj::GScalingRegOptObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = _regopt_effective_theta(obj, theta_raw)

    return _optim_eval_original_oracle_cached!(
        obj.cache,
        obj.C,
        theta,
        obj.s,
        obj.t;
        J1 = obj.J1,
        J0 = obj.J0,
        atol = obj.atol,
        psi_margin = obj.psi_margin,
        psi_floor = obj.psi_floor,
        t1_reformulation = obj.t1_reformulation,
        cache_digits = obj.cache_digits,
        counters = obj.counters,
        warm_start_state = obj.warm_start_state,
        relax_knitro_outlev = obj.relax_knitro_outlev,
        relax_knitro_opttol = obj.relax_knitro_opttol,
        relax_knitro_feastol = obj.relax_knitro_feastol,
    )
end


function (obj::GScalingRegOptObjective)(
    theta_raw::AbstractVector{<:Real},
)
    val = _regopt_eval_original_oracle_cached!(obj, theta_raw)
    return Float64(val.obj)
end


function _regopt_grad!(
    grad_out::AbstractVector,
    obj::GScalingRegOptObjective,
    theta_raw::AbstractVector{<:Real},
)
    theta = _regopt_effective_theta(obj, theta_raw)
    val = _regopt_eval_original_oracle_cached!(obj, theta)

    g = _optim_get_subgradient!(
        obj.C,
        val,
        obj.s,
        obj.t;
        atol = obj.atol,
        psi_margin = obj.psi_margin,
        psi_floor = obj.psi_floor,
        psi_derivative = obj.psi_derivative,
        counters = obj.counters,
    )

    grad_out .= g

    return Float64(val.obj)
end


function _regopt_make_objective(
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

    return GScalingRegOptObjective(
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
        project_evals,

        relax_knitro_outlev,
        relax_knitro_opttol,
        relax_knitro_feastol,
    )
end


function _regopt_initial_theta(
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


function _regopt_make_proximal_regularizer(
    theta_center::Vector{Float64},
    rho::Float64,
)
    if rho <= 0.0
        error("rho must be positive.")
    end

    # h(theta) = (1 / (2rho)) * ||theta - theta_center||^2
    return GScalingRegOptQuadraticRegularizer(
        copy(theta_center),
        rho,
    )
end


function _regopt_regularization_value(
    theta::AbstractVector{<:Real},
    theta_center::AbstractVector{<:Real},
    rho::Float64,
)
    d = theta .- theta_center
    return dot(d, d) / (2.0 * rho)
end


function run_upsilon_regopt_r2_prox_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    rho::Float64 = 1e3,
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_perturbation::Float64 = 0.0,
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
    max_time::Float64 = Inf,
    opt_atol::Float64 = 1e-4,
    opt_rtol::Float64 = 1e-4,
    neg_tol::Float64 = 1e-6,

    sigma_min::Float64 = eps(Float64),
    nu::Float64 = 1e-3,
    gamma_reg::Float64 = 3.0,
    eta1::Float64 = sqrt(sqrt(eps(Float64))),
    eta2::Float64 = 0.9,

    cache_digits::Int = 12,
    verbose::Bool = true,
    method_name::String = "RegularizedOptimization.R2",
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    if length(theta_center) != n
        error("theta_center must have length equal to size(C, 1).")
    end

    theta_center_projected = _optim_project_theta(copy(theta_center), theta_bound)

    theta_start = _regopt_initial_theta(
        n;
        theta0 = theta0,
        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,
        theta_bound = theta_bound,
    )

    obj = _regopt_make_objective(
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
        project_evals = project_evals,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    f(theta) = obj(theta)

    function grad_f!(g, theta)
        return _regopt_grad!(g, obj, theta)
    end

    h = _regopt_make_proximal_regularizer(theta_center_projected, rho)

    options = RO.ROSolverOptions(
        ϵa = opt_atol,
        ϵr = opt_rtol,
        neg_tol = neg_tol,
        verbose = verbose ? 1 : 0,
        maxIter = maxiters,
        maxTime = max_time,
        σmin = sigma_min,
        ν = nu,
        γ = gamma_reg,
        η1 = eta1,
        η2 = eta2,
    )

    unscaled_val = _regopt_eval_original_oracle_cached!(obj, zeros(Float64, n))
    initial_val = _regopt_eval_original_oracle_cached!(obj, theta_start)

    theta_regopt, iterations, outdict = RO.R2(
        f,
        grad_f!,
        h,
        options,
        theta_start;
        selected = 1:n,
    )

    theta_final = _optim_project_theta(Vector{Float64}(theta_regopt), theta_bound)

    final_val = _regopt_eval_original_oracle_cached!(obj, theta_final)

    g_final = _optim_get_subgradient!(
        obj.C,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = obj.counters,
    )

    reg_initial = _regopt_regularization_value(
        theta_start,
        theta_center_projected,
        rho,
    )

    reg_final = _regopt_regularization_value(
        theta_final,
        theta_center_projected,
        rho,
    )

    prox_initial_obj = initial_val.obj + reg_initial
    prox_final_obj = final_val.obj + reg_final

    g_prox_final = g_final .+ (theta_final .- theta_center_projected) ./ rho
    g_prox_norm = _optim_norm2(g_prox_final)

    if verbose
        @printf("\nRegularizedOptimization proximal subproblem finished: %s\n", method_name)
        @printf("  unscaled obj          = %.12e\n", unscaled_val.obj)
        @printf("  initial V(theta)      = %.12e\n", initial_val.obj)
        @printf("  final V(theta)        = %.12e\n", final_val.obj)
        @printf("  initial prox obj      = %.12e\n", prox_initial_obj)
        @printf("  final prox obj        = %.12e\n", prox_final_obj)
        @printf("  regularization final  = %.12e\n", reg_final)
        @printf("  ||g_final||           = %.12e\n", _optim_norm2(g_final))
        @printf("  ||g_prox_final||      = %.12e\n", g_prox_norm)
        @printf("  iterations            = %s\n", string(iterations))
        @printf("  status                = %s\n", string(get(outdict, :status, :unknown)))
        @printf("  objective solves      = %d\n", obj.counters.num_objective_solves)
        @printf("  obj cache hits        = %d / %d\n",
            obj.counters.num_objective_cache_hits,
            obj.counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits       = %d / %d\n",
            obj.counters.num_subgradient_cache_hits,
            obj.counters.num_subgradient_requests,
        )
    end

    return (
        method_name = method_name,

        theta = theta_final,
        gamma = final_val.gamma,
        psi = final_val.psi,
        lambda_min = final_val.lambda_min,
        obj = final_val.obj,
        regularization_obj = reg_final,
        prox_obj = prox_final_obj,

        g = g_final,
        g_norm = _optim_norm2(g_final),
        g_prox = g_prox_final,
        g_prox_norm = g_prox_norm,

        x = final_val.x,
        y = final_val.y,

        theta_center = theta_center_projected,
        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        initial_regularization_obj = reg_initial,
        initial_prox_obj = prox_initial_obj,
        unscaled_obj = unscaled_val.obj,

        regopt_iterations = iterations,
        regopt_outdict = outdict,
        regopt_status = get(outdict, :status, :unknown),

        regopt_objective = obj,
        cache = obj.cache,
        counters = obj.counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            obj.counters.num_objective_cache_hits,
            obj.counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            obj.counters.num_subgradient_cache_hits,
            obj.counters.num_subgradient_requests,
        ),
    )
end