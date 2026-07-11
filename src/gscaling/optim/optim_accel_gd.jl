using Optim
import LineSearches


# =============================================================================
# Base GradientDescent preconditioner
# =============================================================================

function _optim_accel_gd_preconditioner()
    return GradientDescent(
        alphaguess = LineSearches.InitialPrevious(),
        linesearch = LineSearches.HagerZhang(),
    )
end


# =============================================================================
# Accelerated methods
# =============================================================================

function _optim_ngmres_gd_method(;
    wmax::Int = 5,
    nlprecon_iterations::Int = 1,
    allow_f_increases::Bool = true,
)
    return NGMRES(
        nlprecon = _optim_accel_gd_preconditioner(),
        nlpreconopts = Optim.Options(
            iterations = nlprecon_iterations,
            allow_f_increases = allow_f_increases,
        ),
        wmax = wmax,
    )
end


function _optim_oaccel_gd_method(;
    wmax::Int = 5,
    nlprecon_iterations::Int = 1,
    allow_f_increases::Bool = true,
)
    return OACCEL(
        nlprecon = _optim_accel_gd_preconditioner(),
        nlpreconopts = Optim.Options(
            iterations = nlprecon_iterations,
            allow_f_increases = allow_f_increases,
        ),
        wmax = wmax,
    )
end


# =============================================================================
# 1. Original calibration with NGMRES-accelerated GradientDescent
# =============================================================================

function calibrate_upsilon_optim_ngmres_gd_ddfactplus(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Initialization.
    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

    # NGMRES parameters.
    ngmres_wmax::Int = 5,
    ngmres_nlprecon_iterations::Int = 1,
    ngmres_allow_f_increases::Bool = true,

    # Oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Inner DDGFact+_Upsilon relaxation solver options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    # Optim.jl options.
    maxiters::Int = 50,
    grad_tol::Float64 = 1e-6,
    f_tol::Float64 = 1e-10,
    x_tol::Float64 = 1e-10,
    show_trace::Bool = true,
    store_trace::Bool = false,
    extended_trace::Bool = false,

    # Cache/output.
    cache_digits::Int = 12,
    verbose::Bool = true,
)
    method = _optim_ngmres_gd_method(
        wmax = ngmres_wmax,
        nlprecon_iterations = ngmres_nlprecon_iterations,
        allow_f_increases = ngmres_allow_f_increases,
    )

    return run_upsilon_optim_ddfactplus(
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
        use_fminbox = use_fminbox,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "Optim.NGMRES_GD",
    )
end


# =============================================================================
# 2. Proximal operator subproblem with NGMRES-accelerated GradientDescent
# =============================================================================

function solve_upsilon_prox_optim_ngmres_gd_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    # Proximal parameter.
    rho::Float64 = 1e-2,

    # Starting point for the prox subproblem.
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

    # NGMRES parameters.
    ngmres_wmax::Int = 5,
    ngmres_nlprecon_iterations::Int = 1,
    ngmres_allow_f_increases::Bool = true,

    # Oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Inner DDGFact+_Upsilon relaxation solver options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    # Optim.jl options.
    maxiters::Int = 50,
    grad_tol::Float64 = 1e-6,
    f_tol::Float64 = 1e-10,
    x_tol::Float64 = 1e-10,
    show_trace::Bool = true,
    store_trace::Bool = false,
    extended_trace::Bool = false,

    # Cache/output.
    cache_digits::Int = 12,
    verbose::Bool = true,
)
    method = _optim_ngmres_gd_method(
        wmax = ngmres_wmax,
        nlprecon_iterations = ngmres_nlprecon_iterations,
        allow_f_increases = ngmres_allow_f_increases,
    )

    return run_upsilon_prox_optim_ddfactplus(
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
        use_fminbox = use_fminbox,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "Optim.NGMRES_GD",
    )
end


# =============================================================================
# 3. Original calibration with OACCEL-accelerated GradientDescent
# =============================================================================

function calibrate_upsilon_optim_oaccel_gd_ddfactplus(
    C,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Initialization.
    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

    # OACCEL parameters.
    oaccel_wmax::Int = 5,
    oaccel_nlprecon_iterations::Int = 1,
    oaccel_allow_f_increases::Bool = true,

    # Oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Inner DDGFact+_Upsilon relaxation solver options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    # Optim.jl options.
    maxiters::Int = 50,
    grad_tol::Float64 = 1e-6,
    f_tol::Float64 = 1e-10,
    x_tol::Float64 = 1e-10,
    show_trace::Bool = true,
    store_trace::Bool = false,
    extended_trace::Bool = false,

    # Cache/output.
    cache_digits::Int = 12,
    verbose::Bool = true,
)
    method = _optim_oaccel_gd_method(
        wmax = oaccel_wmax,
        nlprecon_iterations = oaccel_nlprecon_iterations,
        allow_f_increases = oaccel_allow_f_increases,
    )

    return run_upsilon_optim_ddfactplus(
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
        use_fminbox = use_fminbox,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "Optim.OACCEL_GD",
    )
end


# =============================================================================
# 4. Proximal operator subproblem with OACCEL-accelerated GradientDescent
# =============================================================================

function solve_upsilon_prox_optim_oaccel_gd_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    # Proximal parameter.
    rho::Float64 = 1e-2,

    # Starting point for the prox subproblem.
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

    # OACCEL parameters.
    oaccel_wmax::Int = 5,
    oaccel_nlprecon_iterations::Int = 1,
    oaccel_allow_f_increases::Bool = true,

    # Oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # Inner DDGFact+_Upsilon relaxation solver options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    # Optim.jl options.
    maxiters::Int = 50,
    grad_tol::Float64 = 1e-6,
    f_tol::Float64 = 1e-10,
    x_tol::Float64 = 1e-10,
    show_trace::Bool = true,
    store_trace::Bool = false,
    extended_trace::Bool = false,

    # Cache/output.
    cache_digits::Int = 12,
    verbose::Bool = true,
)
    method = _optim_oaccel_gd_method(
        wmax = oaccel_wmax,
        nlprecon_iterations = oaccel_nlprecon_iterations,
        allow_f_increases = oaccel_allow_f_increases,
    )

    return run_upsilon_prox_optim_ddfactplus(
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
        use_fminbox = use_fminbox,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        t1_reformulation = t1_reformulation,
        atol = atol,

        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,

        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,

        cache_digits = cache_digits,
        verbose = verbose,
        method_name = "Optim.OACCEL_GD",
    )
end