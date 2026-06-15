using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
import MathOptInterface as MOI

include("util.jl")
include("heuristics.jl")
include("solver_knitro.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_prox.jl")
include("gscaling_prox_alg.jl")

include("dual.jl")
include("var_fixing.jl")


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
# One-step proximal calibration with Knitro parameters
# ============================================================

prox_rho = 1e-3

prox_theta_perturbation = 1e-2
prox_center_initial_theta = false

prox_q_bound = 20.0

prox_knitro_feastol = 1e-6
prox_knitro_opttol = 1e-2
prox_knitro_xtol = 1e-4
prox_knitro_ftol = 1e-5

prox_knitro_maxtime_real = Inf
prox_knitro_algorithm = nothing
prox_knitro_bar_murule = nothing
prox_knitro_honorbnds = 1
prox_knitro_outlev = 0

prox_cache_digits = 12
prox_diagnostics = false
verbose_prox = true


# ============================================================
# BFGS proximal calibration parameters
#
# Tolerance-style stopping, similar to Prox Knitro:
#   prox_bfgs_grad_tol ~= prox_knitro_opttol
#   prox_bfgs_step_tol ~= prox_knitro_xtol
#   prox_bfgs_obj_tol  ~= prox_knitro_ftol
# ============================================================

prox_bfgs_rho = prox_rho

prox_bfgs_max_iter = 200
prox_bfgs_grad_tol = 1e-2
prox_bfgs_step_tol = 1e-6
prox_bfgs_obj_tol = 1e-7

prox_bfgs_alpha0 = 1.0
prox_bfgs_alpha_min = 1e-10
prox_bfgs_alpha_decay = 0.5
prox_bfgs_armijo_c1 = 1e-4
prox_bfgs_max_backtracks = 40
prox_bfgs_curvature_tol = 1e-12
prox_bfgs_reset_on_bad_curvature = false

prox_bfgs_max_abs_theta = 20.0

verbose_prox_bfgs = false


# ============================================================
# Data collection
# ============================================================

solver = "knitro"
calib_method = "bfgs_prox_knitro_prox_bfgs_tol"

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
            verbose = verbose_bfgs,
        )

        gamma_upsilon_bfgs = calib_result_bfgs.gamma
        theta_upsilon_bfgs = calib_result_bfgs.theta
        psi_upsilon_bfgs = calib_result_bfgs.psi

        x_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.x
        y_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.y
        z_ddgfact_plus_upsilon_bfgs = calib_result_bfgs.obj
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
    # Initial theta for proximal calibration methods
    # ============================================================

    Random.seed!(1)

    theta0_prox =
        prox_theta_perturbation == 0.0 ?
        zeros(Float64, n) :
        prox_theta_perturbation .* randn(n)

    if prox_center_initial_theta
        theta0_prox .-= mean(theta0_prox)
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
                q0 = theta0_prox,

                rho = prox_rho,

                theta_perturbation = 0.0,
                center_initial_theta = false,
                q_bound = prox_q_bound,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,
                atol = atol,

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


    # ============================================================
    # DDGFact+_Upsilon proximal calibration with BFGS
    # ============================================================

    println()
    println("Running proximal calibration with BFGS")
    flush(stdout)

    prox_bfgs_runtime = @elapsed begin
        calib_result_prox_bfgs =
            solve_proximal_calibration_alg(
                Csym,
                s,
                t,
                theta0_prox;
                theta0 = theta0_prox,

                rho = prox_bfgs_rho,

                max_iter = prox_bfgs_max_iter,
                grad_tol = prox_bfgs_grad_tol,
                step_tol = prox_bfgs_step_tol,
                obj_tol = prox_bfgs_obj_tol,

                bfgs_alpha0 = prox_bfgs_alpha0,
                bfgs_alpha_min = prox_bfgs_alpha_min,
                bfgs_alpha_decay = prox_bfgs_alpha_decay,
                bfgs_armijo_c1 = prox_bfgs_armijo_c1,
                bfgs_max_backtracks = prox_bfgs_max_backtracks,
                bfgs_curvature_tol = prox_bfgs_curvature_tol,
                bfgs_reset_on_bad_curvature = prox_bfgs_reset_on_bad_curvature,

                max_abs_theta = prox_bfgs_max_abs_theta,

                J1 = Int[],
                J0 = Int[],
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,

                verbose = verbose_prox_bfgs,
            )

        gamma_upsilon_prox_bfgs =
            calib_result_prox_bfgs.gamma

        theta_upsilon_prox_bfgs =
            calib_result_prox_bfgs.theta

        psi_upsilon_prox_bfgs =
            calib_result_prox_bfgs.psi

        x_ddgfact_plus_upsilon_prox_bfgs =
            calib_result_prox_bfgs.x

        y_ddgfact_plus_upsilon_prox_bfgs =
            calib_result_prox_bfgs.y

        z_ddgfact_plus_upsilon_prox_bfgs =
            calib_result_prox_bfgs.obj
    end


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
            ddgfact_gap =
                z_ddgfact - z_ls,

            ddgfact_plus_gap =
                z_ddgfact_plus - z_ls,

            ddgfact_plus_upsilon_bfgs_gap =
                z_ddgfact_plus_upsilon_bfgs - z_ls,

            ddgfact_plus_upsilon_one_step_gap =
                z_ddgfact_plus_upsilon_one_step - z_ls,

            ddgfact_plus_upsilon_prox_bfgs_gap =
                z_ddgfact_plus_upsilon_prox_bfgs - z_ls,

            spectral_gap =
                z_spec - z_ls,

            # Runtimes
            ddgfact_runtime =
                runtime_ddgfact,

            ddgfact_plus_runtime =
                runtime_ddgfact_plus,

            ddgfact_plus_upsilon_bfgs_runtime =
                runtime_ddgfact_plus_upsilon_bfgs,

            ddgfact_plus_upsilon_one_step_runtime =
                one_step_runtime,

            ddgfact_plus_upsilon_prox_bfgs_runtime =
                prox_bfgs_runtime,

            local_search_runtime =
                runtime_ls,

            spectral_runtime =
                runtime_spec,

            # Objective values
            z_ls =
                z_ls,

            z_ddgfact =
                z_ddgfact,

            z_ddgfact_plus =
                z_ddgfact_plus,

            z_ddgfact_plus_upsilon_bfgs =
                z_ddgfact_plus_upsilon_bfgs,

            z_ddgfact_plus_upsilon_one_step =
                z_ddgfact_plus_upsilon_one_step,

            z_ddgfact_plus_upsilon_prox_bfgs =
                z_ddgfact_plus_upsilon_prox_bfgs,

            z_spec =
                z_spec,

            # Minimal proximal BFGS diagnostics
            prox_bfgs_prox_obj =
                calib_result_prox_bfgs.prox_obj,

            prox_bfgs_prox_grad_norm =
                calib_result_prox_bfgs.prox_grad_norm,

            prox_bfgs_iterations =
                calib_result_prox_bfgs.iterations,

            prox_bfgs_num_evals =
                calib_result_prox_bfgs.num_evals,

            prox_bfgs_best_iteration =
                calib_result_prox_bfgs.best_iteration,

            prox_bfgs_status =
                calib_result_prox_bfgs.status,
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
    println("gap_ddgfact_plus_upsilon_prox_bfgs:                   ", z_ddgfact_plus_upsilon_prox_bfgs - z_ls)
    println("gap_spec:                                             ", z_spec - z_ls)

    println()
    println("runtime_ddgfact:                                      ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                                 ", runtime_ddgfact_plus)
    println("runtime_ddgfact_plus_upsilon_bfgs:                    ", runtime_ddgfact_plus_upsilon_bfgs)
    println("runtime_ddgfact_plus_upsilon_one_step:                ", one_step_runtime)
    println("runtime_ddgfact_plus_upsilon_prox_bfgs:               ", prox_bfgs_runtime)
    println("runtime_ls:                                           ", runtime_ls)
    println("runtime_spec:                                         ", runtime_spec)

    println()
    println("prox_bfgs_status:                                     ", calib_result_prox_bfgs.status)
    println("prox_bfgs_iterations:                                 ", calib_result_prox_bfgs.iterations)
    println("prox_bfgs_num_evals:                                  ", calib_result_prox_bfgs.num_evals)
    println("prox_bfgs_best_iteration:                             ", calib_result_prox_bfgs.best_iteration)
    println("prox_bfgs_prox_obj:                                   ", calib_result_prox_bfgs.prox_obj)
    println("prox_bfgs_prox_grad_norm:                             ", calib_result_prox_bfgs.prox_grad_norm)

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