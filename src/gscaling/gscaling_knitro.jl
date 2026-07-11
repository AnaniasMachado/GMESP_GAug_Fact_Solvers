using LinearAlgebra
using Random
using JuMP
using KNITRO
import MathOptInterface as MOI
using Printf


# =============================================================================
# DDGFactplus_Upsilon original calibration with Knitro
#
# Solves:
#
#   min_theta  DDGFactplus_Upsilon(C, s, t; gamma = exp(theta), psi(theta))
#
# subject to:
#
#   -theta_bound <= theta_i <= theta_bound
#
# This is the same oracle/subgradient structure used in the PPA code, but with
# no proximal quadratic regularization term.
# =============================================================================

function calibrate_upsilon_knitro_ddfactplus(
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

    # Original DDGFact+_Upsilon oracle options.
    psi_margin::Float64 = 1e-8,
    psi_floor::Float64 = 0.0,
    psi_derivative::Bool = true,
    t1_reformulation::Bool = true,
    atol::Float64 = 1e-10,

    # DDGFact+_Upsilon relaxation solver tolerances.
    relax_knitro_outlev::Union{Nothing,Int} = nothing,
    relax_knitro_opttol::Union{Nothing,Float64} = nothing,
    relax_knitro_feastol::Union{Nothing,Float64} = nothing,

    # Knitro calibration tolerances.
    knitro_feastol::Float64 = 1e-8,
    knitro_opttol::Float64 = 1e-6,
    knitro_xtol::Float64 = 1e-10,
    knitro_ftol::Float64 = 1e-12,
    knitro_maxtime_real::Float64 = Inf,
    knitro_algorithm::Union{Nothing,Int} = nothing,
    knitro_bar_murule::Union{Nothing,Int} = nothing,
    knitro_honorbnds::Union{Nothing,Int} = 1,
    knitro_outlev::Int = 0,

    # Cache / output.
    cache_digits::Int = 12,
    diagnostics::Bool = false,
    verbose::Bool = true,
)
    Csym = _prox_sym(C)
    n = size(Csym, 1)

    J1 = sort(unique(collect(Int, J1)))
    J0 = sort(unique(collect(Int, J0)))

    @assert all(i -> 1 <= i <= n, J1)
    @assert all(i -> 1 <= i <= n, J0)
    @assert isempty(intersect(J1, J0))
    @assert length(J1) <= s
    @assert s <= n - length(J0)
    @assert 1 <= t <= s <= n

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

    theta_start = _prox_project_theta(theta_start, theta_bound)

    cache = Dict{String,Any}()
    counters = ProxEvalCounters()

    warm_start_state = (
        x0 = Ref{Union{Nothing,Vector{Float64}}}(nothing),
        y0 = Ref{Union{Nothing,Vector{Float64}}}(nothing),
    )

    unscaled_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        zeros(Float64, n),
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

    initial_val = _prox_eval_original_oracle_cached!(
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

    initial_g = _prox_get_subgradient!(
        Csym,
        initial_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = counters,
    )

    model = Model(KNITRO.Optimizer)
    add_knitro_options!(model)

    if knitro_outlev <= 0
        set_silent(model)
    end

    _prox_set_knitro_attribute!(model, "feastol", knitro_feastol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "opttol", knitro_opttol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "xtol", knitro_xtol; verbose = verbose)
    _prox_set_knitro_attribute!(model, "ftol", knitro_ftol; verbose = verbose)

    if isfinite(knitro_maxtime_real)
        _prox_set_knitro_attribute!(
            model,
            "maxtime_real",
            knitro_maxtime_real;
            verbose = verbose,
        )
    end

    if knitro_algorithm !== nothing
        _prox_set_knitro_attribute!(model, "algorithm", knitro_algorithm; verbose = verbose)
    end

    if knitro_bar_murule !== nothing
        _prox_set_knitro_attribute!(model, "bar_murule", knitro_bar_murule; verbose = verbose)
    end

    if knitro_honorbnds !== nothing
        _prox_set_knitro_attribute!(model, "honorbnds", knitro_honorbnds; verbose = verbose)
    end

    if knitro_outlev > 0
        _prox_set_knitro_attribute!(model, "outlev", knitro_outlev; verbose = verbose)
    end

    if isfinite(theta_bound)
        @variable(model, -theta_bound <= theta_var[1:n] <= theta_bound)
    else
        @variable(model, theta_var[1:n])
    end

    for i in 1:n
        set_start_value(theta_var[i], theta_start[i])
    end

    function original_value_f(thetavals...)
        thetavec = collect(thetavals)

        if isfinite(theta_bound)
            thetavec .= clamp.(thetavec, -theta_bound, theta_bound)
        end

        val = _prox_eval_original_oracle_cached!(
            cache,
            Csym,
            thetavec,
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

        return val.obj
    end

    function original_value_grad!(gout, thetavals...)
        thetavec = collect(thetavals)

        if isfinite(theta_bound)
            thetavec .= clamp.(thetavec, -theta_bound, theta_bound)
        end

        val = _prox_eval_original_oracle_cached!(
            cache,
            Csym,
            thetavec,
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

        g = _prox_get_subgradient!(
            Csym,
            val,
            s,
            t;
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = psi_derivative,
            counters = counters,
        )

        for i in 1:n
            gout[i] = g[i]
        end

        return nothing
    end

    register(
        model,
        :original_upsilon_value,
        n,
        original_value_f,
        original_value_grad!,
    )

    @NLobjective(model, Min, original_upsilon_value(theta_var...))

    if verbose
        @printf(
            "Knitro-DDGFactplus-Upsilon init | obj = %.12e | unscaled = %.12e | psi = %.6e | ||g|| = %.3e | gamma [%.4e, %.4e]\n",
            initial_val.obj,
            unscaled_val.obj,
            initial_val.psi,
            norm(initial_g),
            minimum(initial_val.gamma),
            maximum(initial_val.gamma),
        )
        flush(stdout)
    end

    optimize!(model)

    term_stat = termination_status(model)
    primal_stat = primal_status(model)

    if primal_stat == MOI.NO_SOLUTION
        error(
            "Knitro returned no primal solution for the original calibration problem. " *
            "termination_status = $term_stat, primal_status = $primal_stat",
        )
    end

    theta_knitro = value.(theta_var)

    if isfinite(theta_bound)
        theta_knitro .= clamp.(theta_knitro, -theta_bound, theta_bound)
    end

    final_val = _prox_eval_original_oracle_cached!(
        cache,
        Csym,
        theta_knitro,
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

    final_g = _prox_get_subgradient!(
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

    # Be conservative: if Knitro returns a worse point than the initial point,
    # report the better of the two. The raw Knitro point is still returned below.
    if final_val.obj <= initial_val.obj
        best_val = final_val
        best_theta = copy(theta_knitro)
        best_source = :knitro_final
    else
        best_val = initial_val
        best_theta = copy(theta_start)
        best_source = :initial_point
    end

    best_g = _prox_get_subgradient!(
        Csym,
        best_val,
        s,
        t;
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = psi_derivative,
        counters = counters,
    )

    acceptable_status = _prox_knitro_status_is_acceptable(term_stat, primal_stat)

    objective_cache_hit_rate =
        counters.num_objective_oracle_requests == 0 ?
        0.0 :
        counters.num_objective_cache_hits / counters.num_objective_oracle_requests

    subgradient_cache_hit_rate =
        counters.num_subgradient_requests == 0 ?
        0.0 :
        counters.num_subgradient_cache_hits / counters.num_subgradient_requests

    history = diagnostics ? Any[
        (
            stage = :initial,
            obj = initial_val.obj,
            psi = initial_val.psi,
            lambda_min = initial_val.lambda_min,
            gamma_min = minimum(initial_val.gamma),
            gamma_max = maximum(initial_val.gamma),
            grad_norm = norm(initial_g),
        ),
        (
            stage = :knitro_final,
            obj = final_val.obj,
            psi = final_val.psi,
            lambda_min = final_val.lambda_min,
            gamma_min = minimum(final_val.gamma),
            gamma_max = maximum(final_val.gamma),
            grad_norm = norm(final_g),
        ),
    ] : nothing

    if verbose
        @printf(
            "Knitro-DDGFactplus-Upsilon final | obj = %.12e | raw_final = %.12e | initial = %.12e | best_source = %s | psi = %.6e | ||g|| = %.3e | status = %s | primal = %s | cache = %d | obj_solves = %d | grad_evals = %d\n",
            best_val.obj,
            final_val.obj,
            initial_val.obj,
            string(best_source),
            best_val.psi,
            norm(best_g),
            string(term_stat),
            string(primal_stat),
            length(cache),
            counters.num_objective_solves,
            counters.num_subgradient_evals,
        )
        flush(stdout)
    end

    return (
        method_name = "Knitro-DDGFactplus-Upsilon",

        gamma = best_val.gamma,
        theta = best_theta,
        psi = best_val.psi,
        lambda_min = best_val.lambda_min,
        x = best_val.x,
        y = best_val.y,
        obj = best_val.obj,

        g = best_g,
        g_norm = norm(best_g),

        initial_theta = theta_start,
        initial_obj = initial_val.obj,
        initial_gamma = initial_val.gamma,
        initial_psi = initial_val.psi,
        initial_g = initial_g,
        initial_g_norm = norm(initial_g),

        raw_knitro_theta = theta_knitro,
        raw_knitro_obj = final_val.obj,
        raw_knitro_gamma = final_val.gamma,
        raw_knitro_psi = final_val.psi,
        raw_knitro_g = final_g,
        raw_knitro_g_norm = norm(final_g),

        best_source = best_source,

        unscaled_obj = unscaled_val.obj,
        unscaled_gamma = unscaled_val.gamma,
        unscaled_psi = unscaled_val.psi,
        improved_vs_unscaled = best_val.obj < unscaled_val.obj,
        improved_vs_initial = best_val.obj < initial_val.obj,

        knitro_status = term_stat,
        knitro_primal_status = primal_stat,
        knitro_acceptable_status = acceptable_status,

        cache_size = length(cache),
        num_objective_solves = counters.num_objective_solves,
        num_subgradient_evals = counters.num_subgradient_evals,
        num_objective_oracle_requests = counters.num_objective_oracle_requests,
        num_objective_cache_hits = counters.num_objective_cache_hits,
        objective_cache_hit_rate = objective_cache_hit_rate,
        num_subgradient_requests = counters.num_subgradient_requests,
        num_subgradient_cache_hits = counters.num_subgradient_cache_hits,
        subgradient_cache_hit_rate = subgradient_cache_hit_rate,

        history = history,
    )
end