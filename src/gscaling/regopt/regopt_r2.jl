using LinearAlgebra

import RegularizedOptimization as RO
import ShiftedProximalOperators as SPO


function solve_upsilon_prox_regopt_r2_ddfactplus(
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
)
    return run_upsilon_regopt_r2_prox_ddfactplus(
        C,
        theta_center,
        s,
        t;
        J1 = J1,
        J0 = J0,

        rho = rho,
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
        max_time = max_time,
        opt_atol = opt_atol,
        opt_rtol = opt_rtol,
        neg_tol = neg_tol,

        sigma_min = sigma_min,
        nu = nu,
        gamma_reg = gamma_reg,
        eta1 = eta1,
        eta2 = eta2,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "RegularizedOptimization.R2",
    )
end