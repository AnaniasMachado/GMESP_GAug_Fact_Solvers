using LinearAlgebra
using Statistics

import NonSmoothSolvers as NSS


function _nss_gradient_sampling_method(
    theta_start::Vector{Float64};
    epsilon_opt::Float64 = 1e-6,
    nu_opt::Float64 = 1e-6,
)
    method = NSS.GradientSampling(theta_start)

    if hasproperty(method, :ϵ_opt)
        try
            setproperty!(method, :ϵ_opt, epsilon_opt)
        catch
        end
    end

    if hasproperty(method, :ν_opt)
        try
            setproperty!(method, :ν_opt, nu_opt)
        catch
        end
    end

    return method
end


function calibrate_upsilon_nss_gradient_sampling_ddfactplus(
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

    epsilon_opt::Float64 = 1e-6,
    nu_opt::Float64 = 1e-6,

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
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    theta_start = _nss_initial_theta(
        n;
        theta0 = theta0,
        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,
        theta_bound = theta_bound,
    )

    method = _nss_gradient_sampling_method(
        theta_start;
        epsilon_opt = epsilon_opt,
        nu_opt = nu_opt,
    )

    return run_upsilon_nss_ddfactplus(
        Csym,
        s,
        t,
        method;
        J1 = J1,
        J0 = J0,
        theta0 = theta_start,

        theta_perturbation = 0.0,
        center_initial_theta = false,

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
        method_name = "NonSmoothSolvers.GradientSampling",
    )
end


function solve_upsilon_prox_nss_gradient_sampling_ddfactplus(
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

    epsilon_opt::Float64 = 1e-6,
    nu_opt::Float64 = 1e-6,

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
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    if length(theta_center) != n
        error("theta_center must have length equal to size(C, 1).")
    end

    theta_center_projected =
        _optim_project_theta(copy(theta_center), theta_bound)

    theta_start = if theta0 !== nothing
        copy(theta0)
    else
        copy(theta_center_projected)
    end

    if length(theta_start) != n
        error("theta0 must have length equal to size(C, 1).")
    end

    theta_start = _optim_project_theta(theta_start, theta_bound)

    method = _nss_gradient_sampling_method(
        theta_start;
        epsilon_opt = epsilon_opt,
        nu_opt = nu_opt,
    )

    return run_upsilon_prox_nss_ddfactplus(
        Csym,
        theta_center_projected,
        s,
        t,
        method;
        J1 = J1,
        J0 = J0,

        rho = rho,
        theta0 = theta_start,

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
        method_name = "NonSmoothSolvers.GradientSampling",
    )
end