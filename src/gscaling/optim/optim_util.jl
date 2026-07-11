using LinearAlgebra
using Optim
using Printf


# =============================================================================
# Basic helpers
# =============================================================================

function _optim_sym(C)
    return Symmetric(Matrix{Float64}(Matrix(C)))
end


function _optim_project_theta(theta::Vector{Float64}, theta_bound::Float64)
    if isfinite(theta_bound)
        return clamp.(theta, -theta_bound, theta_bound)
    else
        return copy(theta)
    end
end


function _optim_cache_key(theta::Vector{Float64}; digits::Int = 12)
    return join(string.(round.(theta; digits = digits)), ",")
end


function _optim_norm2(v::AbstractVector{<:Real})
    return sqrt(sum(abs2, v))
end


function _optim_cache_hit_rate(hits::Int, requests::Int)
    return requests == 0 ? 0.0 : hits / requests
end


# =============================================================================
# Cache structs
# =============================================================================

mutable struct GScalingOptimOracleValue
    obj::Float64
    g::Union{Nothing,Vector{Float64}}
    gamma::Vector{Float64}
    psi::Float64
    lambda_min::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    theta::Vector{Float64}
end


mutable struct GScalingOptimEvalCounters
    num_objective_oracle_requests::Int
    num_objective_cache_hits::Int
    num_objective_solves::Int
    num_subgradient_requests::Int
    num_subgradient_cache_hits::Int
    num_subgradient_evals::Int
end


GScalingOptimEvalCounters() = GScalingOptimEvalCounters(0, 0, 0, 0, 0, 0)


mutable struct GScalingOptimWarmStartState
    x0::Union{Nothing,Vector{Float64}}
    y0::Union{Nothing,Vector{Float64}}
end


GScalingOptimWarmStartState() = GScalingOptimWarmStartState(nothing, nothing)


# =============================================================================
# Original DDGFact+_Upsilon oracle, cached
# =============================================================================

function _optim_eval_original_oracle(
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    atol::Float64,
    x0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    y0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    psi_margin::Float64,
    psi_floor::Float64,
    t1_reformulation::Bool,
    counters::Union{Nothing,GScalingOptimEvalCounters} = nothing,

    # Inner DDGFact+_Upsilon relaxation Knitro options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,
)
    val = eval_ddfactplus_upsilon_calibration_objective(
        C,
        theta,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        x0 = x0,
        y0 = y0,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        knitro_outlev = relax_knitro_outlev,
        knitro_opttol = relax_knitro_opttol,
        knitro_feastol = relax_knitro_feastol,
    )

    if counters !== nothing
        counters.num_objective_solves += 1
    end

    return GScalingOptimOracleValue(
        Float64(val.obj),
        nothing,
        Vector{Float64}(val.gamma),
        Float64(val.psi),
        Float64(val.λmin),
        Vector{Float64}(val.x),
        Vector{Float64}(val.y),
        copy(theta),
    )
end


function _optim_eval_original_oracle_cached!(
    cache::Dict{String,GScalingOptimOracleValue},
    C::Symmetric{<:Real,<:AbstractMatrix},
    theta::Vector{Float64},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    t1_reformulation::Bool,
    cache_digits::Int,
    counters::Union{Nothing,GScalingOptimEvalCounters} = nothing,
    warm_start_state::Union{Nothing,GScalingOptimWarmStartState} = nothing,
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,
)
    theta_val = copy(theta)
    key = _optim_cache_key(theta_val; digits = cache_digits)

    if counters !== nothing
        counters.num_objective_oracle_requests += 1
    end

    if haskey(cache, key)
        if counters !== nothing
            counters.num_objective_cache_hits += 1
        end
        return cache[key]
    end

    x0 = warm_start_state === nothing ? nothing : warm_start_state.x0
    y0 = warm_start_state === nothing ? nothing : warm_start_state.y0

    val = _optim_eval_original_oracle(
        C,
        theta_val,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        x0 = x0,
        y0 = y0,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        counters = counters,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    if warm_start_state !== nothing
        warm_start_state.x0 = copy(val.x)
        warm_start_state.y0 = copy(val.y)
    end

    cache[key] = val
    return val
end


function _optim_get_subgradient!(
    C::Symmetric{<:Real,<:AbstractMatrix},
    val::GScalingOptimOracleValue,
    s::Int,
    t::Int;
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    counters::Union{Nothing,GScalingOptimEvalCounters} = nothing,
)
    if counters !== nothing
        counters.num_subgradient_requests += 1
    end

    if val.g !== nothing
        if counters !== nothing
            counters.num_subgradient_cache_hits += 1
        end
        return val.g::Vector{Float64}
    end

    val.g = Vector{Float64}(
        eval_ddfactplus_upsilon_calibration_subgradient(
            C,
            val,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
        ),
    )

    if counters !== nothing
        counters.num_subgradient_evals += 1
    end

    return val.g::Vector{Float64}
end


# =============================================================================
# Build f and g! closures for Optim.jl
# =============================================================================

function _optim_make_fg!(
    cache::Dict{String,GScalingOptimOracleValue},
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer},
    J0::AbstractVector{<:Integer},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
    psi_derivative::Bool,
    t1_reformulation::Bool,
    cache_digits::Int,
    counters::GScalingOptimEvalCounters,
    warm_start_state::GScalingOptimWarmStartState,

    # Proximal term.
    theta_center::Union{Nothing,Vector{Float64}} = nothing,
    rho::Float64 = 1.0,

    # Inner relaxation options.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,
)
    if theta_center !== nothing
        @assert rho > 0.0
    end

    function f(theta_raw::Vector{Float64})
        theta = copy(theta_raw)

        val = _optim_eval_original_oracle_cached!(
            cache,
            C,
            theta,
            s,
            t;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            t1_reformulation = t1_reformulation,
            cache_digits = cache_digits,
            counters = counters,
            warm_start_state = warm_start_state,
            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
        )

        obj = val.obj

        if theta_center !== nothing
            obj += (0.5 / rho) * sum(
                (theta[i] - theta_center[i])^2 for i in eachindex(theta)
            )
        end

        return obj
    end

    function g!(G::Vector{Float64}, theta_raw::Vector{Float64})
        theta = copy(theta_raw)

        val = _optim_eval_original_oracle_cached!(
            cache,
            C,
            theta,
            s,
            t;
            J1 = J1,
            J0 = J0,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            t1_reformulation = t1_reformulation,
            cache_digits = cache_digits,
            counters = counters,
            warm_start_state = warm_start_state,
            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
        )

        g = _optim_get_subgradient!(
            C,
            val,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            counters = counters,
        )

        G .= g

        if theta_center !== nothing
            G .+= (theta .- theta_center) ./ rho
        end

        return nothing
    end

    return f, g!
end


# =============================================================================
# Generic Optim.jl driver
# =============================================================================

function _optim_run!(
    f,
    g!,
    theta0::Vector{Float64},
    method;
    theta_bound::Float64,
    use_fminbox::Bool,
    maxiters::Int,
    grad_tol::Float64,
    f_tol::Float64,
    x_tol::Float64,
    show_trace::Bool,
    store_trace::Bool,
    extended_trace::Bool,
)
    options = Optim.Options(
        iterations = maxiters,
        g_tol = grad_tol,
        f_reltol = f_tol,
        x_abstol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,
    )

    if use_fminbox && isfinite(theta_bound)
        lower = fill(-theta_bound, length(theta0))
        upper = fill( theta_bound, length(theta0))

        return optimize(
            f,
            g!,
            lower,
            upper,
            theta0,
            Fminbox(method),
            options,
        )
    else
        return optimize(
            f,
            g!,
            theta0,
            method,
            options,
        )
    end
end


# =============================================================================
# Generic original calibration wrapper
# =============================================================================

function run_upsilon_optim_ddfactplus(
    C,
    s::Int,
    t::Int,
    method;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Initialization.
    theta_perturbation::Float64 = 1e-4,
    center_initial_theta::Bool = false,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

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
    method_name::String = string(typeof(method)),
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert all(i -> 1 <= i <= n, J0)
    @assert isempty(intersect(J1, J0))
    @assert length(J1) <= s
    @assert s <= n - length(J0)
    @assert 1 <= t <= s <= n
    @assert maxiters >= 1

    theta_start = if theta0 !== nothing
        copy(theta0)
    elseif theta_perturbation == 0.0
        zeros(n)
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

    cache = Dict{String,GScalingOptimOracleValue}()
    counters = GScalingOptimEvalCounters()
    warm_start_state = GScalingOptimWarmStartState()

    unscaled_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        zeros(n),
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    initial_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_start,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    f, g! = _optim_make_fg!(
        cache,
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
        counters = counters,
        warm_start_state = warm_start_state,
        theta_center = nothing,
        rho = 1.0,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    result = _optim_run!(
        f,
        g!,
        theta_start,
        method;
        theta_bound = theta_bound,
        use_fminbox = use_fminbox,
        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,
    )

    theta_final = Vector{Float64}(Optim.minimizer(result))
    theta_final = _optim_project_theta(theta_final, theta_bound)

    final_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_final,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    g_final = _optim_get_subgradient!(
        Csym,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = counters,
    )

    if verbose
        @printf("\nOptim original calibration finished: %s\n", method_name)
        @printf("  unscaled obj      = %.12e\n", unscaled_val.obj)
        @printf("  initial obj       = %.12e\n", initial_val.obj)
        @printf("  final obj         = %.12e\n", final_val.obj)
        @printf("  ||g_final||       = %.12e\n", _optim_norm2(g_final))
        @printf("  objective solves  = %d\n", counters.num_objective_solves)
        @printf("  obj cache hits    = %d / %d\n",
            counters.num_objective_cache_hits,
            counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits   = %d / %d\n",
            counters.num_subgradient_cache_hits,
            counters.num_subgradient_requests,
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
        g_norm = _optim_norm2(g_final),
        x = final_val.x,
        y = final_val.y,

        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        unscaled_obj = unscaled_val.obj,

        optim_result = result,
        converged = Optim.converged(result),
        iterations = Optim.iterations(result),

        cache = cache,
        counters = counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            counters.num_objective_cache_hits,
            counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            counters.num_subgradient_cache_hits,
            counters.num_subgradient_requests,
        ),
    )
end


# =============================================================================
# Generic proximal operator subproblem wrapper
# =============================================================================

function run_upsilon_prox_optim_ddfactplus(
    C,
    theta_center::Vector{Float64},
    s::Int,
    t::Int,
    method;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],

    # Proximal parameter.
    rho::Float64 = 1e-2,

    # Starting point for the prox subproblem.
    theta0::Union{Nothing,Vector{Float64}} = nothing,

    # Bounds.
    theta_bound::Float64 = 20.0,
    use_fminbox::Bool = true,

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
    method_name::String = string(typeof(method)),
)
    Csym = _optim_sym(C)
    n = size(Csym, 1)

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert all(i -> 1 <= i <= n, J0)
    @assert isempty(intersect(J1, J0))
    @assert length(J1) <= s
    @assert s <= n - length(J0)
    @assert 1 <= t <= s <= n
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

    cache = Dict{String,GScalingOptimOracleValue}()
    counters = GScalingOptimEvalCounters()
    warm_start_state = GScalingOptimWarmStartState()

    center_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_center_projected,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    initial_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_start,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    f, g! = _optim_make_fg!(
        cache,
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
        counters = counters,
        warm_start_state = warm_start_state,
        theta_center = theta_center_projected,
        rho = rho,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    result = _optim_run!(
        f,
        g!,
        theta_start,
        method;
        theta_bound = theta_bound,
        use_fminbox = use_fminbox,
        maxiters = maxiters,
        grad_tol = grad_tol,
        f_tol = f_tol,
        x_tol = x_tol,
        show_trace = show_trace,
        store_trace = store_trace,
        extended_trace = extended_trace,
    )

    theta_final = Vector{Float64}(Optim.minimizer(result))
    theta_final = _optim_project_theta(theta_final, theta_bound)

    final_val = _optim_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_final,
        s,
        t;
        J1 = J1,
        J0 = J0,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        t1_reformulation = t1_reformulation,
        cache_digits = cache_digits,
        counters = counters,
        warm_start_state = warm_start_state,
        relax_knitro_outlev = relax_knitro_outlev,
        relax_knitro_opttol = relax_knitro_opttol,
        relax_knitro_feastol = relax_knitro_feastol,
    )

    g_original_final = _optim_get_subgradient!(
        Csym,
        final_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = counters,
    )

    prox_gradient =
        g_original_final .+ (theta_final .- theta_center_projected) ./ rho

    prox_obj =
        final_val.obj +
        (0.5 / rho) * sum(
            (theta_final[i] - theta_center_projected[i])^2 for i in 1:n
        )

    initial_prox_obj =
        initial_val.obj +
        (0.5 / rho) * sum(
            (theta_start[i] - theta_center_projected[i])^2 for i in 1:n
        )

    center_prox_obj = center_val.obj

    if verbose
        @printf("\nOptim proximal subproblem finished: %s\n", method_name)
        @printf("  center V(theta)     = %.12e\n", center_val.obj)
        @printf("  initial prox obj    = %.12e\n", initial_prox_obj)
        @printf("  final prox obj      = %.12e\n", prox_obj)
        @printf("  final V(theta)      = %.12e\n", final_val.obj)
        @printf("  prox step norm      = %.12e\n", _optim_norm2(theta_final .- theta_center_projected))
        @printf("  ||prox grad||       = %.12e\n", _optim_norm2(prox_gradient))
        @printf("  objective solves    = %d\n", counters.num_objective_solves)
        @printf("  obj cache hits      = %d / %d\n",
            counters.num_objective_cache_hits,
            counters.num_objective_oracle_requests,
        )
        @printf("  grad cache hits     = %d / %d\n",
            counters.num_subgradient_cache_hits,
            counters.num_subgradient_requests,
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
        center_prox_obj = center_prox_obj,
        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        initial_prox_obj = initial_prox_obj,

        optim_result = result,
        converged = Optim.converged(result),
        iterations = Optim.iterations(result),

        cache = cache,
        counters = counters,
        objective_cache_hit_rate = _optim_cache_hit_rate(
            counters.num_objective_cache_hits,
            counters.num_objective_oracle_requests,
        ),
        subgradient_cache_hit_rate = _optim_cache_hit_rate(
            counters.num_subgradient_cache_hits,
            counters.num_subgradient_requests,
        ),
    )
end