using LinearAlgebra

import NonSmoothSolvers as NSS


function _nss_nsbfgs_method(;
    epsilon::Float64 = 1e-8,
)
    return NSS.NSBFGS{Float64}(
        ϵ_opt = epsilon,
    )
end


function calibrate_upsilon_nss_bfgs_ddfactplus(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    theta_bound::Float64 = 20.0,
    project_evals::Bool = false,

    nsbfgs_epsilon::Float64 = 1e-8,

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
)
    method = _nss_nsbfgs_method(
        epsilon = nsbfgs_epsilon,
    )

    return run_upsilon_nss_ddfactplus(
        C,
        s,
        t,
        method;
        J1 = J1,
        J0 = J0,
        theta0 = theta0,

        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,

        theta_bound = theta_bound,
        project_evals = project_evals,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        time_limit = time_limit,
        show_trace = show_trace,
        trace_length = trace_length,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "NonSmoothSolvers.NSBFGS",
    )
end


function solve_upsilon_prox_nss_bfgs_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    rho::Float64 = 1e-2,
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    theta_bound::Float64 = 20.0,
    project_evals::Bool = false,

    nsbfgs_epsilon::Float64 = 1e-8,

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
)
    method = _nss_nsbfgs_method(
        epsilon = nsbfgs_epsilon,
    )

    return run_upsilon_prox_nss_ddfactplus(
        C,
        theta_center,
        s,
        t,
        method;
        J1 = J1,
        J0 = J0,

        rho = rho,
        theta0 = theta0,

        theta_bound = theta_bound,
        project_evals = project_evals,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        time_limit = time_limit,
        show_trace = show_trace,
        trace_length = trace_length,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "NonSmoothSolvers.NSBFGS",
    )
end