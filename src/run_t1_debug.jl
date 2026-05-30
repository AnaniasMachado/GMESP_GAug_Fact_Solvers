using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using JuMP
using Ipopt
using LBFGSB
import MathOptInterface as MOI

include("util.jl")
include("heuristics.jl")
include("solver_ipopt.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_t1.jl")
include("gscaling_bfgs_debug.jl")

# -------------------------
# Problem data
# -------------------------
n = 63

# For t = 1, we vary s directly.
# Full run:
# s_vals = [s for s in 2:(n - 2)]

# Small debug run:
s_vals = [s for s in 2:12]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10
t = 1

# Initial unscaled psi for DDGFact+
psi = eigmin(Csym) - atol

# -------------------------
# BFGS calibration parameters
# -------------------------
max_bfgs_iter = 100

grad_tol = 1e-2
step_tol = 1e-8

alpha0 = 5.0
alpha_min = 1e-10
armijo_c1 = 1e-6
max_backtracks = 30

curvature_tol = 1e-12

# Largest feasible psi will be:
# psi(gamma) = lambda_min(D_gamma^(1/2) C D_gamma^(1/2)) - psi_margin
psi_margin = 1e-7
psi_floor = 0.0

max_theta_norm = 20.0
psi_derivative = true

# Debug flags
verbose_bfgs = true
finite_diff_check = true
print_trial_steps = true
use_steepest_descent_fallback = true

# -------------------------
# Data Collection
# -------------------------
solver = "ipopt"
calib_method = "bfgs_debug_psideriv"
experiment = "t1_calibration_original_vs_reform"

mkpath("results")

results_filepath =
    "results/results_gap_$(solver)_$(calib_method)_$(experiment)_n$(n).csv"

results = []

for s in s_vals
    println("============================================================")
    println("s: $s")
    println("t: $t")
    println("============================================================")
    flush(stdout)

    result = []
    append!(result, [n, s, t])

    # -------------------------
    # DDGFact, non-augmented
    # This corresponds to psi = 0
    # -------------------------
    runtime_ddgfact = @elapsed begin
        x_ddgfact, z_ddgfact = ddfact_gmesp(
            Csym,
            s,
            t;
            atol = atol,
        )
    end

    # -------------------------
    # DDGFact+, augmented
    # This corresponds to psi < lambda_min(C)
    # -------------------------
    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(
            Csym,
            s,
            t,
            psi;
            atol = atol,
        )
    end

    # -------------------------
    # DDGFact+_Upsilon calibration
    # Debug BFGS with psi derivative, original formulation
    # -------------------------
    println("")
    println("------------------------------------------------------------")
    println("Running DEBUG BFGS: original formulation")
    println("------------------------------------------------------------")
    flush(stdout)

    Random.seed!(1)
    runtime_bfgs_original = @elapsed begin
        calib_result_bfgs_original =
            calibrate_upsilon_bfgs_ddfactplus_debug(
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
                armijo_c1 = armijo_c1,
                curvature_tol = curvature_tol,
                max_backtracks = max_backtracks,
                max_theta_norm = max_theta_norm,
                psi_derivative = psi_derivative,
                t1_reformulation = false,
                verbose = verbose_bfgs,
                finite_diff_check = finite_diff_check,
                print_trial_steps = print_trial_steps,
                use_steepest_descent_fallback = use_steepest_descent_fallback,
            )

        gamma_bfgs_original = calib_result_bfgs_original.gamma
        theta_bfgs_original = calib_result_bfgs_original.theta
        psi_bfgs_original = calib_result_bfgs_original.psi

        x_bfgs_original = calib_result_bfgs_original.x
        y_bfgs_original = calib_result_bfgs_original.y
        z_bfgs_original = calib_result_bfgs_original.obj
    end

    # -------------------------
    # DDGFact+_Upsilon calibration
    # Debug BFGS with psi derivative, t = 1 reformulation
    # -------------------------
    println("")
    println("------------------------------------------------------------")
    println("Running DEBUG BFGS: t = 1 reformulation")
    println("------------------------------------------------------------")
    flush(stdout)

    Random.seed!(1)
    runtime_bfgs_t1_reform = @elapsed begin
        calib_result_bfgs_t1_reform =
            calibrate_upsilon_bfgs_ddfactplus_debug(
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
                armijo_c1 = armijo_c1,
                curvature_tol = curvature_tol,
                max_backtracks = max_backtracks,
                max_theta_norm = max_theta_norm,
                psi_derivative = psi_derivative,
                t1_reformulation = true,
                verbose = verbose_bfgs,
                finite_diff_check = finite_diff_check,
                print_trial_steps = print_trial_steps,
                use_steepest_descent_fallback = use_steepest_descent_fallback,
            )

        gamma_bfgs_t1_reform = calib_result_bfgs_t1_reform.gamma
        theta_bfgs_t1_reform = calib_result_bfgs_t1_reform.theta
        psi_bfgs_t1_reform = calib_result_bfgs_t1_reform.psi

        x_bfgs_t1_reform = calib_result_bfgs_t1_reform.x
        y_bfgs_t1_reform = calib_result_bfgs_t1_reform.y
        z_bfgs_t1_reform = calib_result_bfgs_t1_reform.obj
    end

    # -------------------------
    # Local search
    # Used only to compute gaps
    # -------------------------
    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end

    # -------------------------
    # Feasibility diagnostics
    # -------------------------
    Cgamma_bfgs_original = scaled_matrix(Csym, gamma_bfgs_original)
    lambda_min_bfgs_original = eigmin(Cgamma_bfgs_original)
    feasibility_slack_bfgs_original =
        lambda_min_bfgs_original - psi_bfgs_original

    Cgamma_bfgs_t1_reform = scaled_matrix(Csym, gamma_bfgs_t1_reform)
    lambda_min_bfgs_t1_reform = eigmin(Cgamma_bfgs_t1_reform)
    feasibility_slack_bfgs_t1_reform =
        lambda_min_bfgs_t1_reform - psi_bfgs_t1_reform

    # -------------------------
    # Collect gaps, runtimes, and debug diagnostics
    # -------------------------
    append!(
        result,
        [
            # Gaps
            z_ddgfact - z_ls,
            z_ddgfact_plus - z_ls,
            z_bfgs_original - z_ls,
            z_bfgs_t1_reform - z_ls,

            # Runtimes
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_bfgs_original,
            runtime_bfgs_t1_reform,
            runtime_ls,

            # Objective values
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_bfgs_original,
            z_bfgs_t1_reform,

            # Comparison between calibration variants
            z_bfgs_original - z_bfgs_t1_reform,
            runtime_bfgs_original - runtime_bfgs_t1_reform,

            # Psi and feasibility diagnostics
            psi_bfgs_original,
            lambda_min_bfgs_original,
            feasibility_slack_bfgs_original,

            psi_bfgs_t1_reform,
            lambda_min_bfgs_t1_reform,
            feasibility_slack_bfgs_t1_reform,

            # Gamma diagnostics
            minimum(gamma_bfgs_original),
            maximum(gamma_bfgs_original),
            norm(theta_bfgs_original),

            minimum(gamma_bfgs_t1_reform),
            maximum(gamma_bfgs_t1_reform),
            norm(theta_bfgs_t1_reform),

            # Debug diagnostics: original formulation
            calib_result_bfgs_original.accepted_bfgs_steps,
            calib_result_bfgs_original.accepted_sd_fallback_steps,
            calib_result_bfgs_original.line_search_failures,
            calib_result_bfgs_original.curvature_resets,

            # Debug diagnostics: t = 1 reformulation
            calib_result_bfgs_t1_reform.accepted_bfgs_steps,
            calib_result_bfgs_t1_reform.accepted_sd_fallback_steps,
            calib_result_bfgs_t1_reform.line_search_failures,
            calib_result_bfgs_t1_reform.curvature_resets,
        ],
    )

    push!(results, result)

    println("")
    println("------------------------------------------------------------")
    println("Summary for s = $s")
    println("------------------------------------------------------------")

    println("z_ls:                                ", z_ls)
    println("z_ddgfact:                           ", z_ddgfact)
    println("z_ddgfact_plus:                      ", z_ddgfact_plus)
    println("z_bfgs_original:                     ", z_bfgs_original)
    println("z_bfgs_t1_reform:                    ", z_bfgs_t1_reform)

    println("gap_ddgfact:                         ", z_ddgfact - z_ls)
    println("gap_ddgfact_plus:                    ", z_ddgfact_plus - z_ls)
    println("gap_bfgs_original:                   ", z_bfgs_original - z_ls)
    println("gap_bfgs_t1_reform:                  ", z_bfgs_t1_reform - z_ls)

    println("improvement_t1_reform_over_original: ", z_bfgs_original - z_bfgs_t1_reform)

    println("runtime_ddgfact:                     ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                ", runtime_ddgfact_plus)
    println("runtime_bfgs_original:               ", runtime_bfgs_original)
    println("runtime_bfgs_t1_reform:              ", runtime_bfgs_t1_reform)
    println("runtime_saved_by_t1_reform:          ", runtime_bfgs_original - runtime_bfgs_t1_reform)

    println("psi_bfgs_original:                   ", psi_bfgs_original)
    println("psi_bfgs_t1_reform:                  ", psi_bfgs_t1_reform)

    println("debug_original_accepted_bfgs_steps:   ",
        calib_result_bfgs_original.accepted_bfgs_steps)
    println("debug_original_accepted_sd_steps:     ",
        calib_result_bfgs_original.accepted_sd_fallback_steps)
    println("debug_original_line_search_failures:  ",
        calib_result_bfgs_original.line_search_failures)
    println("debug_original_curvature_resets:      ",
        calib_result_bfgs_original.curvature_resets)

    println("debug_t1_accepted_bfgs_steps:         ",
        calib_result_bfgs_t1_reform.accepted_bfgs_steps)
    println("debug_t1_accepted_sd_steps:           ",
        calib_result_bfgs_t1_reform.accepted_sd_fallback_steps)
    println("debug_t1_line_search_failures:        ",
        calib_result_bfgs_t1_reform.line_search_failures)
    println("debug_t1_curvature_resets:            ",
        calib_result_bfgs_t1_reform.curvature_resets)

    flush(stdout)
end

results_matrix = hcat(results...)'

cols = [
    :n,
    :s,
    :t,

    # Gaps
    :ddgfact_gap,
    :ddgfact_plus_gap,
    :ddgfact_plus_upsilon_bfgs_original_gap,
    :ddgfact_plus_upsilon_bfgs_t1_reform_gap,

    # Runtimes
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :ddgfact_plus_upsilon_bfgs_original_runtime,
    :ddgfact_plus_upsilon_bfgs_t1_reform_runtime,
    :local_search_runtime,

    # Objective values
    :z_ls,
    :z_ddgfact,
    :z_ddgfact_plus,
    :z_ddgfact_plus_upsilon_bfgs_original,
    :z_ddgfact_plus_upsilon_bfgs_t1_reform,

    # Direct comparison
    :bfgs_t1_reform_improvement_over_original,
    :bfgs_t1_reform_runtime_saving,

    # Psi and feasibility diagnostics
    :psi_bfgs_original,
    :lambda_min_bfgs_original,
    :feasibility_slack_bfgs_original,

    :psi_bfgs_t1_reform,
    :lambda_min_bfgs_t1_reform,
    :feasibility_slack_bfgs_t1_reform,

    # Gamma diagnostics
    :gamma_min_bfgs_original,
    :gamma_max_bfgs_original,
    :theta_norm_bfgs_original,

    :gamma_min_bfgs_t1_reform,
    :gamma_max_bfgs_t1_reform,
    :theta_norm_bfgs_t1_reform,

    # Debug diagnostics: original formulation
    :debug_original_accepted_bfgs_steps,
    :debug_original_accepted_sd_fallback_steps,
    :debug_original_line_search_failures,
    :debug_original_curvature_resets,

    # Debug diagnostics: t = 1 reformulation
    :debug_t1_reform_accepted_bfgs_steps,
    :debug_t1_reform_accepted_sd_fallback_steps,
    :debug_t1_reform_line_search_failures,
    :debug_t1_reform_curvature_resets,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")