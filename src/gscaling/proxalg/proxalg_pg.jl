using LinearAlgebra

import ProximalAlgorithms as PA
import ProximalCore
import DifferentiationInterface as DI


function calibrate_upsilon_proxalg_pg_ddfactplus(
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
)
    return run_upsilon_proxalg_pg_ddfactplus(
        C,
        s,
        t;
        J1 = J1,
        J0 = J0,
        theta0 = theta0,

        theta_perturbation = theta_perturbation,
        center_initial_theta = center_initial_theta,

        theta_bound = theta_bound,
        project_f_evals = project_f_evals,

        gamma = gamma,
        adaptive = adaptive,
        minimum_gamma = minimum_gamma,
        reduce_gamma = reduce_gamma,
        increase_gamma = increase_gamma,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        tol = tol,
        cache_digits = cache_digits,

        verbose = verbose,
        method_name = "ProximalAlgorithms.ForwardBackward",
    )
end