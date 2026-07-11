using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
import MathOptInterface as MOI

include("../misc/util.jl")
include("../misc/heuristics.jl")
include("../solvers/solver_knitro.jl")

include("../gscaling/gscaling_util.jl")
include("../gscaling/gscaling_bfgs.jl")
include("../gscaling/gscaling_prox.jl")

include("../misc/dual.jl")
include("../misc/var_fixing.jl")


# ============================================================
# Problem data
# ============================================================

n = 63
kappa = 1

# Full run:
# s_vals = [s for s in (kappa + 1):(n - 1)]

# Test run:
s_vals = [s for s in (kappa + 1):(kappa + 5)]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# Initial unscaled psi for DDGFact+.
psi = eigmin(Csym) - atol


# ============================================================
# DDGFact+_Upsilon relaxation solver tolerances
# Used inside each calibration oracle evaluation.
# ============================================================

relax_knitro_outlev = nothing
relax_knitro_opttol = 1e-8
relax_knitro_feastol = 1e-5


# ============================================================
# Full spectral DDGFact+_Upsilon BFGS calibration parameters
# ============================================================

bfgs_max_iter = 50
bfgs_grad_tol = 1e-2
bfgs_step_tol = 1e-8

bfgs_alpha0 = 1.0
bfgs_alpha_min = 1e-10
bfgs_alpha_decay = 0.75
bfgs_armijo_c1 = 1e-6
bfgs_max_backtracks = 50
bfgs_curvature_tol = 1e-12

psi_margin = 1e-7
psi_floor = 0.0

bfgs_max_theta_norm = 20.0
verbose_bfgs = false


# ============================================================
# PPA calibration parameters
# ============================================================

ppa_rho = 1e3

ppa_theta_perturbation = 1e-2
ppa_center_initial_theta = false
ppa_theta_bound = 20.0

# Full PPA stopping criteria, used only when k = Inf.
ppa_grad_tol = 1e-2
ppa_prox_obj_abs_tol = 1e-8
ppa_prox_step_tol = 1e-8
ppa_max_wall_time = Inf

# Knitro proximal subproblem options.
ppa_knitro_feastol = 1e-6
ppa_knitro_opttol = 1e-2
ppa_knitro_xtol = 1e-4
ppa_knitro_ftol = 1e-5

ppa_knitro_maxtime_real = Inf
ppa_knitro_algorithm = nothing
ppa_knitro_bar_murule = nothing
ppa_knitro_honorbnds = 1
ppa_knitro_outlev = 0

ppa_cache_digits = 6
ppa_diagnostics = false
verbose_ppa = false


# ============================================================
# Data collection
# ============================================================

solver = "knitro"
calib_method = "bfgs_ppa"

mkpath("results")

results_filepath =
    "results/results_$(solver)_$(calib_method)_n$(n)_kappa$(kappa).csv"

results = Any[]


# ============================================================
# Small helpers for compact CSV rows
# ============================================================

_status_string(x) = x === missing ? "missing" : string(x)

function _gamma_theta_stats(prefix::Symbol, gamma::AbstractVector, theta::AbstractVector)
    p = String(prefix)

    names = (
        Symbol("$(p)_gamma_norm"),
        Symbol("$(p)_gamma_norm_inf"),
        Symbol("$(p)_theta_norm"),
        Symbol("$(p)_theta_norm_inf"),
    )

    values = (
        norm(gamma),
        norm(gamma, Inf),
        norm(theta),
        norm(theta, Inf),
    )

    return NamedTuple{names}(values)
end

function _ppa_cache_stats(prefix::Symbol, result)
    p = String(prefix)

    names = (
        Symbol("$(p)_cache_size"),
        Symbol("$(p)_num_objective_oracle_requests"),
        Symbol("$(p)_num_objective_cache_hits"),
        Symbol("$(p)_objective_cache_hit_rate"),
        Symbol("$(p)_num_objective_solves"),
        Symbol("$(p)_num_subgradient_requests"),
        Symbol("$(p)_num_subgradient_cache_hits"),
        Symbol("$(p)_subgradient_cache_hit_rate"),
        Symbol("$(p)_num_subgradient_evals"),
    )

    values = (
        result.cache_size,
        result.num_objective_oracle_requests,
        result.num_objective_cache_hits,
        result.objective_cache_hit_rate,
        result.num_objective_solves,
        result.num_subgradient_requests,
        result.num_subgradient_cache_hits,
        result.subgradient_cache_hit_rate,
        result.num_subgradient_evals,
    )

    return NamedTuple{names}(values)
end


for s in s_vals
    t = s - kappa

    println("--------------------")
    println("s: $s")
    println("t: $t")
    flush(stdout)


    # ============================================================
    # DDGFact, non-augmented
    # ============================================================

    runtime_ddgfact = @elapsed begin
        x_ddgfact, z_ddgfact = ddfact_gmesp(
            Csym,
            s,
            t;
            atol = atol,
        )
    end


    # ============================================================
    # DDGFact+, augmented
    # ============================================================

    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(
            Csym,
            s,
            t,
            psi;
            atol = atol,
        )
    end


    # ============================================================
    # Local search
    # Needed to compute gaps.
    # ============================================================

    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end


    # ============================================================
    # Spectral bound
    # ============================================================

    runtime_spec = @elapsed begin
        z_spec = spectral_bound(Csym, t)
    end


    # ============================================================
    # DDGFact+_Upsilon, full spectral BFGS calibration
    # ============================================================

    runtime_upsilon_bfgs = @elapsed begin
        result_bfgs = calibrate_upsilon_bfgs_ddfactplus(
            Csym,
            s,
            t;
            atol = atol,
            max_iter = bfgs_max_iter,
            grad_tol = bfgs_grad_tol,
            step_tol = bfgs_step_tol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            alpha0 = bfgs_alpha0,
            alpha_min = bfgs_alpha_min,
            alpha_decay = bfgs_alpha_decay,
            armijo_c1 = bfgs_armijo_c1,
            curvature_tol = bfgs_curvature_tol,
            max_backtracks = bfgs_max_backtracks,
            max_theta_norm = bfgs_max_theta_norm,
            psi_derivative = true,
            t1_reformulation = false,

            knitro_outlev = relax_knitro_outlev,
            knitro_opttol = relax_knitro_opttol,
            knitro_feastol = relax_knitro_feastol,

            verbose = verbose_bfgs,
        )
    end


    # ============================================================
    # Shared initial theta for one-step PPA and full PPA
    # ============================================================

    Random.seed!(1)

    theta0_ppa =
        ppa_theta_perturbation == 0.0 ?
        zeros(Float64, n) :
        ppa_theta_perturbation .* randn(n)

    if ppa_center_initial_theta
        theta0_ppa .-= mean(theta0_ppa)
    end

    if isfinite(ppa_theta_bound)
        theta0_ppa .= clamp.(theta0_ppa, -ppa_theta_bound, ppa_theta_bound)
    end


    # ============================================================
    # DDGFact+_Upsilon, one PPA iteration
    # ============================================================

    println()
    println("Running one PPA iteration")
    flush(stdout)

    runtime_upsilon_ppa_one = @elapsed begin
        result_ppa_one = calibrate_upsilon_ppa_ddfactplus(
            Csym,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
            theta0 = theta0_ppa,
            k = 1,

            rho = ppa_rho,
            grad_tol = ppa_grad_tol,
            prox_obj_abs_tol = ppa_prox_obj_abs_tol,
            prox_step_tol = ppa_prox_step_tol,
            max_wall_time = ppa_max_wall_time,

            theta_perturbation = 0.0,
            center_initial_theta = false,
            theta_bound = ppa_theta_bound,

            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = true,
            t1_reformulation = false,
            atol = atol,

            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,

            knitro_feastol = ppa_knitro_feastol,
            knitro_opttol = ppa_knitro_opttol,
            knitro_xtol = ppa_knitro_xtol,
            knitro_ftol = ppa_knitro_ftol,
            knitro_maxtime_real = ppa_knitro_maxtime_real,
            knitro_algorithm = ppa_knitro_algorithm,
            knitro_bar_murule = ppa_knitro_bar_murule,
            knitro_honorbnds = ppa_knitro_honorbnds,
            knitro_outlev = ppa_knitro_outlev,

            cache_digits = ppa_cache_digits,
            diagnostics = ppa_diagnostics,
            verbose = verbose_ppa,
        )
    end


    # ============================================================
    # DDGFact+_Upsilon, full PPA until convergence
    # ============================================================

    println()
    println("Running full PPA")
    flush(stdout)

    runtime_upsilon_ppa_full = @elapsed begin
        result_ppa_full = calibrate_upsilon_ppa_ddfactplus(
            Csym,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
            theta0 = theta0_ppa,
            k = Inf,

            rho = ppa_rho,
            grad_tol = ppa_grad_tol,
            prox_obj_abs_tol = ppa_prox_obj_abs_tol,
            prox_step_tol = ppa_prox_step_tol,
            max_wall_time = ppa_max_wall_time,

            theta_perturbation = 0.0,
            center_initial_theta = false,
            theta_bound = ppa_theta_bound,

            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = true,
            t1_reformulation = false,
            atol = atol,

            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,

            knitro_feastol = ppa_knitro_feastol,
            knitro_opttol = ppa_knitro_opttol,
            knitro_xtol = ppa_knitro_xtol,
            knitro_ftol = ppa_knitro_ftol,
            knitro_maxtime_real = ppa_knitro_maxtime_real,
            knitro_algorithm = ppa_knitro_algorithm,
            knitro_bar_murule = ppa_knitro_bar_murule,
            knitro_honorbnds = ppa_knitro_honorbnds,
            knitro_outlev = ppa_knitro_outlev,

            cache_digits = ppa_cache_digits,
            diagnostics = ppa_diagnostics,
            verbose = verbose_ppa,
        )
    end


    # ============================================================
    # Store row
    # ============================================================

    bfgs_stats = _gamma_theta_stats(
        :upsilon_bfgs,
        result_bfgs.gamma,
        result_bfgs.theta,
    )

    ppa_one_stats = _gamma_theta_stats(
        :upsilon_ppa_one,
        result_ppa_one.gamma,
        result_ppa_one.theta,
    )

    ppa_full_stats = _gamma_theta_stats(
        :upsilon_ppa_full,
        result_ppa_full.gamma,
        result_ppa_full.theta,
    )

    ppa_one_cache_stats = _ppa_cache_stats(:ppa_one, result_ppa_one)
    ppa_full_cache_stats = _ppa_cache_stats(:ppa_full, result_ppa_full)

    push!(
        results,
        merge(
            (
                n = n,
                s = s,
                t = t,

                # Gaps relative to local search.
                ddgfact_gap = z_ddgfact - z_ls,
                ddgfact_plus_gap = z_ddgfact_plus - z_ls,
                upsilon_bfgs_gap = result_bfgs.obj - z_ls,
                upsilon_ppa_one_gap = result_ppa_one.obj - z_ls,
                upsilon_ppa_full_gap = result_ppa_full.obj - z_ls,
                spectral_gap = z_spec - z_ls,

                # Runtimes.
                ddgfact_runtime = runtime_ddgfact,
                ddgfact_plus_runtime = runtime_ddgfact_plus,
                upsilon_bfgs_runtime = runtime_upsilon_bfgs,
                upsilon_ppa_one_runtime = runtime_upsilon_ppa_one,
                upsilon_ppa_full_runtime = runtime_upsilon_ppa_full,
                local_search_runtime = runtime_ls,
                spectral_runtime = runtime_spec,

                # Calibration metadata.
                upsilon_bfgs_improved = result_bfgs.improved,
                upsilon_bfgs_fallback_used = result_bfgs.fallback_used,
                upsilon_bfgs_fallback_count = result_bfgs.fallback_count,
                upsilon_bfgs_final_oracle = result_bfgs.final_oracle,

                ppa_one_improved = result_ppa_one.improved,
                ppa_one_iters = result_ppa_one.ppa_iters,
                ppa_one_stop_reason = result_ppa_one.stop_reason,
                ppa_one_knitro_status =
                    _status_string(result_ppa_one.knitro_status),
                ppa_one_knitro_primal_status =
                    _status_string(result_ppa_one.knitro_primal_status),
                ppa_one_subproblem_acceptable =
                    result_ppa_one.last_subproblem_acceptable_status,

                ppa_full_improved = result_ppa_full.improved,
                ppa_full_iters = result_ppa_full.ppa_iters,
                ppa_full_stop_reason = result_ppa_full.stop_reason,
                ppa_full_knitro_status =
                    _status_string(result_ppa_full.knitro_status),
                ppa_full_knitro_primal_status =
                    _status_string(result_ppa_full.knitro_primal_status),
                ppa_full_subproblem_acceptable =
                    result_ppa_full.last_subproblem_acceptable_status,

                # Parameters worth keeping with the result row.
                relax_knitro_opttol = relax_knitro_opttol,
                relax_knitro_feastol = relax_knitro_feastol,
                bfgs_grad_tol = bfgs_grad_tol,
                bfgs_step_tol = bfgs_step_tol,
                ppa_rho = ppa_rho,
                ppa_grad_tol = ppa_grad_tol,
                ppa_prox_obj_abs_tol = ppa_prox_obj_abs_tol,
                ppa_prox_step_tol = ppa_prox_step_tol,
                ppa_knitro_opttol = ppa_knitro_opttol,
                ppa_knitro_feastol = ppa_knitro_feastol,
                ppa_knitro_xtol = ppa_knitro_xtol,
                ppa_knitro_ftol = ppa_knitro_ftol,
            ),
            bfgs_stats,
            ppa_one_stats,
            ppa_full_stats,
            ppa_one_cache_stats,
            ppa_full_cache_stats,
        ),
    )


    # ============================================================
    # Console output
    # ============================================================

    println()
    println("========== INSTANCE SUMMARY ==========")
    println("s:                                      ", s)
    println("t:                                      ", t)

    println()
    println("ddgfact_gap:                           ", z_ddgfact - z_ls)
    println("ddgfact_plus_gap:                      ", z_ddgfact_plus - z_ls)
    println("upsilon_bfgs_gap:                      ", result_bfgs.obj - z_ls)
    println("upsilon_ppa_one_gap:                   ", result_ppa_one.obj - z_ls)
    println("upsilon_ppa_full_gap:                  ", result_ppa_full.obj - z_ls)
    println("spectral_gap:                          ", z_spec - z_ls)

    println()
    println("ddgfact_runtime:                       ", runtime_ddgfact)
    println("ddgfact_plus_runtime:                  ", runtime_ddgfact_plus)
    println("upsilon_bfgs_runtime:                  ", runtime_upsilon_bfgs)
    println("upsilon_ppa_one_runtime:               ", runtime_upsilon_ppa_one)
    println("upsilon_ppa_full_runtime:              ", runtime_upsilon_ppa_full)
    println("local_search_runtime:                  ", runtime_ls)
    println("spectral_runtime:                      ", runtime_spec)

    println()
    println("norm_gamma_upsilon_bfgs:               ", norm(result_bfgs.gamma))
    println("norm_theta_upsilon_bfgs:               ", norm(result_bfgs.theta))
    println("norm_gamma_upsilon_ppa_one:            ", norm(result_ppa_one.gamma))
    println("norm_theta_upsilon_ppa_one:            ", norm(result_ppa_one.theta))
    println("norm_gamma_upsilon_ppa_full:           ", norm(result_ppa_full.gamma))
    println("norm_theta_upsilon_ppa_full:           ", norm(result_ppa_full.theta))

    println()
    println("ppa_one_cache_size:                    ", result_ppa_one.cache_size)
    println("ppa_one_objective_cache_hit_rate:      ", result_ppa_one.objective_cache_hit_rate)
    println("ppa_one_subgradient_cache_hit_rate:    ", result_ppa_one.subgradient_cache_hit_rate)
    println("ppa_one_stop_reason:                   ", result_ppa_one.stop_reason)
    println("ppa_one_iters:                         ", result_ppa_one.ppa_iters)

    println()
    println("ppa_full_cache_size:                   ", result_ppa_full.cache_size)
    println("ppa_full_objective_cache_hit_rate:     ", result_ppa_full.objective_cache_hit_rate)
    println("ppa_full_subgradient_cache_hit_rate:   ", result_ppa_full.subgradient_cache_hit_rate)
    println("ppa_full_stop_reason:                  ", result_ppa_full.stop_reason)
    println("ppa_full_iters:                        ", result_ppa_full.ppa_iters)

    println("======================================")
    println()

    flush(stdout)
end


# ============================================================
# Save CSV
# ============================================================

df_results = DataFrame(results)
CSV.write(results_filepath, df_results)

println("Saved results to: $results_filepath")