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

# Initial unscaled psi for DDGFact+
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

max_bfgs_iter = 50

grad_tol = 1e-2
step_tol = 1e-8

alpha0 = 1.0
alpha_min = 1e-10
alpha_decay = 0.75
armijo_c1 = 1e-6
max_backtracks = 50

curvature_tol = 1e-12

psi_margin = 1e-7
psi_floor = 0.0

max_theta_norm = 20.0

verbose_bfgs = false


# ============================================================
# One-step proximal calibration parameters
# ============================================================

prox_rho = 1e3

prox_theta_perturbation = 1e-2
prox_center_initial_theta = false

prox_theta_bound = 20.0

# Knitro proximal subproblem options.
prox_knitro_feastol = 1e-6
prox_knitro_opttol = 1e-2
prox_knitro_xtol = 1e-4
prox_knitro_ftol = 1e-5

prox_knitro_maxtime_real = Inf
prox_knitro_algorithm = nothing
prox_knitro_bar_murule = nothing
prox_knitro_honorbnds = 1
prox_knitro_outlev = 0

prox_cache_digits = 6
prox_diagnostics = false
verbose_prox = false


# ============================================================
# Multi-step proximal calibration parameters
# ============================================================

multi_prox_obj_abs_tol = 1e-8
multi_prox_step_tol = 1e-8
multi_prox_max_wall_time = Inf


# ============================================================
# Data collection
# ============================================================

solver = "knitro"
calib_method = "bfgs_prox_knitro"

mkpath("results")

results_filepath =
    "results/results_$(solver)_$(calib_method)_n$(n)_kappa$(kappa).csv"

results = Any[]


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
    # DDGFact+_Upsilon, full spectral BFGS calibration
    # ============================================================

    runtime_ddgfact_plus_upsilon_bfgs = @elapsed begin
        calib_result_bfgs = calibrate_upsilon_bfgs_ddfactplus(
            Csym,
            s,
            t;
            atol = atol,
            max_iter = max_bfgs_iter,
            grad_tol = grad_tol,
            step_tol = step_tol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            alpha0 = alpha0,
            alpha_min = alpha_min,
            alpha_decay = alpha_decay,
            armijo_c1 = armijo_c1,
            curvature_tol = curvature_tol,
            max_backtracks = max_backtracks,
            max_theta_norm = max_theta_norm,
            psi_derivative = true,
            t1_reformulation = false,

            knitro_outlev = relax_knitro_outlev,
            knitro_opttol = relax_knitro_opttol,
            knitro_feastol = relax_knitro_feastol,

            verbose = verbose_bfgs,
        )

        gamma_upsilon_bfgs = calib_result_bfgs.gamma
        theta_upsilon_bfgs = calib_result_bfgs.theta
        psi_upsilon_bfgs = calib_result_bfgs.psi

        x_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.x
        y_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.y
        z_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.obj
    end

    Cgamma_upsilon_bfgs = scaled_matrix(Csym, gamma_upsilon_bfgs)
    lambda_min_upsilon_bfgs = eigmin(Cgamma_upsilon_bfgs)
    feasibility_slack_upsilon_bfgs =
        lambda_min_upsilon_bfgs - psi_upsilon_bfgs


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
    # Initial theta for one-step proximal calibration
    # ============================================================

    Random.seed!(1)

    theta0_prox =
        prox_theta_perturbation == 0.0 ?
        zeros(Float64, n) :
        prox_theta_perturbation .* randn(n)

    if prox_center_initial_theta
        theta0_prox .-= mean(theta0_prox)
    end

    if isfinite(prox_theta_bound)
        theta0_prox .= clamp.(theta0_prox, -prox_theta_bound, prox_theta_bound)
    end


    # ============================================================
    # DDGFact+_Upsilon one-step proximal calibration with Knitro
    # ============================================================

    println()
    println("Running one-step proximal calibration with Knitro")
    flush(stdout)

    one_step_runtime = @elapsed begin
        calib_result_one_step =
            solve_one_step_proximal_knitro_upsilon_calibration(
                Csym,
                s,
                t;
                J1 = Int[],
                J0 = Int[],
                theta0 = theta0_prox,

                rho = prox_rho,

                theta_perturbation = 0.0,
                center_initial_theta = false,
                theta_bound = prox_theta_bound,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,
                atol = atol,

                # DDGFact+_Upsilon relaxation solve tolerances.
                relax_knitro_outlev = relax_knitro_outlev,
                relax_knitro_opttol = relax_knitro_opttol,
                relax_knitro_feastol = relax_knitro_feastol,

                # Proximal subproblem tolerances.
                knitro_feastol = prox_knitro_feastol,
                knitro_opttol = prox_knitro_opttol,
                knitro_xtol = prox_knitro_xtol,
                knitro_ftol = prox_knitro_ftol,
                knitro_maxtime_real = prox_knitro_maxtime_real,
                knitro_algorithm = prox_knitro_algorithm,
                knitro_bar_murule = prox_knitro_bar_murule,
                knitro_honorbnds = prox_knitro_honorbnds,
                knitro_outlev = prox_knitro_outlev,

                cache_digits = prox_cache_digits,
                diagnostics = prox_diagnostics,
                verbose = verbose_prox,
            )

        gamma_upsilon_one_step = calib_result_one_step.gamma
        theta_upsilon_one_step = calib_result_one_step.theta
        psi_upsilon_one_step = calib_result_one_step.psi

        x_ddgfact_plus_upsilon_one_step = calib_result_one_step.x
        y_ddgfact_plus_upsilon_one_step = calib_result_one_step.y
        z_ddgfact_plus_upsilon_one_step = calib_result_one_step.obj
    end

    Cgamma_upsilon_one_step = scaled_matrix(Csym, gamma_upsilon_one_step)
    lambda_min_upsilon_one_step = eigmin(Cgamma_upsilon_one_step)
    feasibility_slack_upsilon_one_step =
        lambda_min_upsilon_one_step - psi_upsilon_one_step


    # ============================================================
    # DDGFact+_Upsilon multi-step proximal calibration with Knitro
    # ============================================================

    println()
    println("Running multi-step proximal calibration with Knitro")
    flush(stdout)

    multi_step_runtime = @elapsed begin
        calib_result_multi_step =
            solve_proximal_knitro_upsilon_calibration(
                Csym,
                s,
                t;
                J1 = Int[],
                J0 = Int[],
                theta0 = theta0_prox,

                rho = prox_rho,

                prox_obj_abs_tol = multi_prox_obj_abs_tol,
                prox_step_tol = multi_prox_step_tol,
                max_wall_time = multi_prox_max_wall_time,

                theta_perturbation = 0.0,
                center_initial_theta = false,
                theta_bound = prox_theta_bound,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,
                atol = atol,

                # DDGFact+_Upsilon relaxation solve tolerances.
                relax_knitro_outlev = relax_knitro_outlev,
                relax_knitro_opttol = relax_knitro_opttol,
                relax_knitro_feastol = relax_knitro_feastol,

                # Proximal subproblem tolerances.
                knitro_feastol = prox_knitro_feastol,
                knitro_opttol = prox_knitro_opttol,
                knitro_xtol = prox_knitro_xtol,
                knitro_ftol = prox_knitro_ftol,
                knitro_maxtime_real = prox_knitro_maxtime_real,
                knitro_algorithm = prox_knitro_algorithm,
                knitro_bar_murule = prox_knitro_bar_murule,
                knitro_honorbnds = prox_knitro_honorbnds,
                knitro_outlev = prox_knitro_outlev,

                cache_digits = prox_cache_digits,
                diagnostics = prox_diagnostics,
                verbose = verbose_prox,
            )

        gamma_upsilon_multi_step = calib_result_multi_step.gamma
        theta_upsilon_multi_step = calib_result_multi_step.theta
        psi_upsilon_multi_step = calib_result_multi_step.psi

        x_ddgfact_plus_upsilon_multi_step = calib_result_multi_step.x
        y_ddgfact_plus_upsilon_multi_step = calib_result_multi_step.y
        z_ddgfact_plus_upsilon_multi_step = calib_result_multi_step.obj
    end

    Cgamma_upsilon_multi_step = scaled_matrix(Csym, gamma_upsilon_multi_step)
    lambda_min_upsilon_multi_step = eigmin(Cgamma_upsilon_multi_step)
    feasibility_slack_upsilon_multi_step =
        lambda_min_upsilon_multi_step - psi_upsilon_multi_step


    # ============================================================
    # Store row
    # ============================================================

    push!(
        results,
        (
            n = n,
            s = s,
            t = t,

            # Gaps
            ddgfact_gap = z_ddgfact - z_ls,
            ddgfact_plus_gap = z_ddgfact_plus - z_ls,
            ddgfact_plus_upsilon_bfgs_gap =
                z_ddgfact_plus_upsilon_bfgs - z_ls,
            ddgfact_plus_upsilon_one_step_gap =
                z_ddgfact_plus_upsilon_one_step - z_ls,
            ddgfact_plus_upsilon_multi_step_gap =
                z_ddgfact_plus_upsilon_multi_step - z_ls,
            spec_gap = z_spec - z_ls,

            # Runtimes
            ddgfact_runtime = runtime_ddgfact,
            ddgfact_plus_runtime = runtime_ddgfact_plus,
            ddgfact_plus_upsilon_bfgs_runtime =
                runtime_ddgfact_plus_upsilon_bfgs,
            ddgfact_plus_upsilon_one_step_runtime =
                one_step_runtime,
            ddgfact_plus_upsilon_multi_step_runtime =
                multi_step_runtime,
            local_search_runtime = runtime_ls,
            spectral_runtime = runtime_spec,

            # Objective values
            z_ls = z_ls,
            z_ddgfact = z_ddgfact,
            z_ddgfact_plus = z_ddgfact_plus,
            z_ddgfact_plus_upsilon_bfgs = z_ddgfact_plus_upsilon_bfgs,
            z_ddgfact_plus_upsilon_one_step =
                z_ddgfact_plus_upsilon_one_step,
            z_ddgfact_plus_upsilon_multi_step =
                z_ddgfact_plus_upsilon_multi_step,
            z_spec = z_spec,

            # Psi and feasibility diagnostics
            psi_ddgfact_plus = psi,

            psi_upsilon_bfgs = psi_upsilon_bfgs,
            lambda_min_upsilon_bfgs = lambda_min_upsilon_bfgs,
            feasibility_slack_upsilon_bfgs =
                feasibility_slack_upsilon_bfgs,

            psi_upsilon_one_step = psi_upsilon_one_step,
            lambda_min_upsilon_one_step =
                lambda_min_upsilon_one_step,
            feasibility_slack_upsilon_one_step =
                feasibility_slack_upsilon_one_step,
            
            psi_upsilon_multi_step = psi_upsilon_multi_step,
            lambda_min_upsilon_multi_step =
                lambda_min_upsilon_multi_step,
            feasibility_slack_upsilon_multi_step =
                feasibility_slack_upsilon_multi_step,

            # Gamma diagnostics: full spectral BFGS
            gamma_min_upsilon_bfgs = minimum(gamma_upsilon_bfgs),
            gamma_max_upsilon_bfgs = maximum(gamma_upsilon_bfgs),
            theta_norm_upsilon_bfgs = norm(theta_upsilon_bfgs),
            theta_norm_inf_upsilon_bfgs = norm(theta_upsilon_bfgs, Inf),

            # Gamma diagnostics: one-step proximal method
            gamma_min_upsilon_one_step =
                minimum(gamma_upsilon_one_step),
            gamma_max_upsilon_one_step =
                maximum(gamma_upsilon_one_step),
            theta_norm_upsilon_one_step =
                norm(theta_upsilon_one_step),
            theta_norm_inf_upsilon_one_step =
                norm(theta_upsilon_one_step, Inf),
            
            # Gamma diagnostics: PPA method
            gamma_min_upsilon_multi_step =
                minimum(gamma_upsilon_multi_step),
            gamma_max_upsilon_multi_step =
                maximum(gamma_upsilon_multi_step),
            theta_norm_upsilon_multi_step =
                norm(theta_upsilon_multi_step),
            theta_norm_inf_upsilon_multi_step =
                norm(theta_upsilon_multi_step, Inf),

            # One-step cache and solver diagnostics
            one_step_cache_size = calib_result_one_step.cache_size,

            one_step_num_objective_oracle_requests =
                calib_result_one_step.num_objective_oracle_requests,
            one_step_num_objective_cache_hits =
                calib_result_one_step.num_objective_cache_hits,
            one_step_objective_cache_hit_rate =
                calib_result_one_step.objective_cache_hit_rate,
            one_step_num_objective_solves =
                calib_result_one_step.num_objective_solves,

            one_step_num_subgradient_requests =
                calib_result_one_step.num_subgradient_requests,
            one_step_num_subgradient_cache_hits =
                calib_result_one_step.num_subgradient_cache_hits,
            one_step_subgradient_cache_hit_rate =
                calib_result_one_step.subgradient_cache_hit_rate,
            one_step_num_subgradient_evals =
                calib_result_one_step.num_subgradient_evals,

            one_step_subproblem_status =
                calib_result_one_step.knitro_status,
            one_step_subproblem_primal_status =
                calib_result_one_step.knitro_primal_status,
            one_step_subproblem_acceptable_status =
                calib_result_one_step.last_subproblem_acceptable_status,
            one_step_subproblem_residual_norm_inf =
                calib_result_one_step.last_subproblem_residual_norm_inf,
            
            # Multi-step cache and solver diagnostics
            multi_step_cache_size =
                calib_result_multi_step.cache_size,

            multi_step_num_objective_oracle_requests =
                calib_result_multi_step.num_objective_oracle_requests,
            multi_step_num_objective_cache_hits =
                calib_result_multi_step.num_objective_cache_hits,
            multi_step_objective_cache_hit_rate =
                calib_result_multi_step.objective_cache_hit_rate,
            multi_step_num_objective_solves =
                calib_result_multi_step.num_objective_solves,

            multi_step_num_subgradient_requests =
                calib_result_multi_step.num_subgradient_requests,
            multi_step_num_subgradient_cache_hits =
                calib_result_multi_step.num_subgradient_cache_hits,
            multi_step_subgradient_cache_hit_rate =
                calib_result_multi_step.subgradient_cache_hit_rate,
            multi_step_num_subgradient_evals =
                calib_result_multi_step.num_subgradient_evals,

            multi_step_prox_iters =
                calib_result_multi_step.prox_iters,
            multi_step_stop_reason =
                calib_result_multi_step.stop_reason,
            multi_step_subproblem_acceptable_status =
                calib_result_multi_step.last_subproblem_acceptable_status,
            multi_step_subproblem_residual_norm_inf =
                calib_result_multi_step.last_subproblem_residual_norm_inf,
            
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
            prox_knitro_opttol = prox_knitro_opttol,
            prox_knitro_feastol = prox_knitro_feastol,
            prox_knitro_xtol = prox_knitro_xtol,
            prox_knitro_ftol = prox_knitro_ftol,
        ),
    )


    # ============================================================
    # Console output
    # ============================================================

    println()
    println("========== INSTANCE SUMMARY ==========")
    println("s:                                                     ", s)
    println("t:                                                     ", t)

    println()
    println("gap_ddgfact:                                          ", z_ddgfact - z_ls)
    println("gap_ddgfact_plus:                                     ", z_ddgfact_plus - z_ls)
    println("gap_ddgfact_plus_upsilon_bfgs:                        ", z_ddgfact_plus_upsilon_bfgs - z_ls)
    println("gap_ddgfact_plus_upsilon_one_step:                    ", z_ddgfact_plus_upsilon_one_step - z_ls)
    println("gap_ddgfact_plus_upsilon_multi_step:                  ", z_ddgfact_plus_upsilon_multi_step - z_ls)
    println("gap_spec:                                             ", z_spec - z_ls)

    println()
    println("runtime_ddgfact:                                      ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                                 ", runtime_ddgfact_plus)
    println("runtime_ddgfact_plus_upsilon_bfgs:                    ", runtime_ddgfact_plus_upsilon_bfgs)
    println("runtime_ddgfact_plus_upsilon_one_step:                ", one_step_runtime)
    println("runtime_ddgfact_plus_upsilon_multi_step:              ", multi_step_runtime)
    println("runtime_ls:                                           ", runtime_ls)
    println("runtime_spec:                                         ", runtime_spec)

    println()
    println("gamma_min_upsilon_bfgs:                               ", minimum(gamma_upsilon_bfgs))
    println("gamma_max_upsilon_bfgs:                               ", maximum(gamma_upsilon_bfgs))
    println("gamma_min_upsilon_one_step:                           ", minimum(gamma_upsilon_one_step))
    println("gamma_max_upsilon_one_step:                           ", maximum(gamma_upsilon_one_step))
    println("gamma_min_upsilon_multi_step:                         ", minimum(gamma_upsilon_multi_step))
    println("gamma_max_upsilon_multi_step:                         ", maximum(gamma_upsilon_multi_step))

    println()
    println("one_step_cache_size:                                  ", calib_result_one_step.cache_size)

    println("one_step_num_objective_oracle_requests:               ", calib_result_one_step.num_objective_oracle_requests)
    println("one_step_num_objective_cache_hits:                    ", calib_result_one_step.num_objective_cache_hits)
    println("one_step_objective_cache_hit_rate:                    ", calib_result_one_step.objective_cache_hit_rate)
    println("one_step_num_objective_solves:                        ", calib_result_one_step.num_objective_solves)

    println("one_step_num_subgradient_requests:                    ", calib_result_one_step.num_subgradient_requests)
    println("one_step_num_subgradient_cache_hits:                  ", calib_result_one_step.num_subgradient_cache_hits)
    println("one_step_subgradient_cache_hit_rate:                  ", calib_result_one_step.subgradient_cache_hit_rate)
    println("one_step_num_subgradient_evals:                       ", calib_result_one_step.num_subgradient_evals)

    println("one_step_subproblem_status:                           ", calib_result_one_step.knitro_status)
    println("one_step_subproblem_primal_status:                    ", calib_result_one_step.knitro_primal_status)
    println("one_step_subproblem_acceptable_status:                ", calib_result_one_step.last_subproblem_acceptable_status)
    println("one_step_subproblem_residual_norm_inf:                ", calib_result_one_step.last_subproblem_residual_norm_inf)
    
    println()
    println("multi_step_cache_size:                                ", calib_result_multi_step.cache_size)

    println("multi_step_num_objective_oracle_requests:             ", calib_result_multi_step.num_objective_oracle_requests)
    println("multi_step_num_objective_cache_hits:                  ", calib_result_multi_step.num_objective_cache_hits)
    println("multi_step_objective_cache_hit_rate:                  ", calib_result_multi_step.objective_cache_hit_rate)
    println("multi_step_num_objective_solves:                      ", calib_result_multi_step.num_objective_solves)

    println("multi_step_num_subgradient_requests:                  ", calib_result_multi_step.num_subgradient_requests)
    println("multi_step_num_subgradient_cache_hits:                ", calib_result_multi_step.num_subgradient_cache_hits)
    println("multi_step_subgradient_cache_hit_rate:                ", calib_result_multi_step.subgradient_cache_hit_rate)
    println("multi_step_num_subgradient_evals:                     ", calib_result_multi_step.num_subgradient_evals)

    println("multi_step_prox_iters:                                ", calib_result_multi_step.prox_iters)
    println("multi_step_stop_reason:                               ", calib_result_multi_step.stop_reason)
    println("multi_step_subproblem_acceptable_status:              ", calib_result_multi_step.last_subproblem_acceptable_status)
    println("multi_step_subproblem_residual_norm_inf:              ", calib_result_multi_step.last_subproblem_residual_norm_inf)

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