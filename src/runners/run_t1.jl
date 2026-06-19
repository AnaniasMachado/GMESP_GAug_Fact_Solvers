using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using JuMP
using KNITRO
using LBFGSB
import MathOptInterface as MOI

include("./misc/util.jl")
include("./misc/heuristics.jl")
include("./solvers/solver_knitro.jl")

include("./gscaling/gscaling_util.jl")
include("./gscaling/gscaling_bfgs.jl")
include("./gscaling/gscaling_t1.jl")
include("./gscaling/gscaling_params.jl")


# -------------------------
# Problem data
# -------------------------
n = 124

# For t = 1, we vary s directly.
# Full run:
# s_vals = [s for s in 2:(n - 2)]

# Test run:
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

# ============================================================
# Choose active BFGS parameter set
# ============================================================
# active_bfgs_param_set = :default
# active_bfgs_param_set = :fast
active_bfgs_param_set = :very_fast

bfgs_params = bfgs_param_sets[active_bfgs_param_set]

max_bfgs_iter = bfgs_params[:max_bfgs_iter]

grad_tol = bfgs_params[:grad_tol]
step_tol = bfgs_params[:step_tol]

alpha0 = bfgs_params[:alpha0]
alpha_min = bfgs_params[:alpha_min]
alpha_decay = bfgs_params[:alpha_decay]
armijo_c1 = bfgs_params[:armijo_c1]
max_backtracks = bfgs_params[:max_backtracks]

curvature_tol = bfgs_params[:curvature_tol]

psi_margin = bfgs_params[:psi_margin]
psi_floor = bfgs_params[:psi_floor]

max_theta_norm = bfgs_params[:max_theta_norm]
psi_derivative = bfgs_params[:psi_derivative]

use_steepest_descent_fallback = bfgs_params[:use_steepest_descent_fallback]

verbose_bfgs = bfgs_params[:verbose_bfgs]

t1_fallback_limit = max_bfgs_iter

# -------------------------
# Data Collection
# -------------------------
solver = "knitro"
calib_method = "bfgs"
bfgs_param_set = "very_fast"

mkpath("results")

results_filepath = "results/results_t1_gap_$(solver)_$(calib_method)_n$(n)_$(bfgs_param_set).csv"
results = []

for s in s_vals
    println("--------------------")
    println("s: $s")
    println("t: $t")
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

    Random.seed!(1)
    # -------------------------
    # DDGFact+_Upsilon calibration
    # BFGS with psi derivative, original formulation
    # -------------------------
    runtime_bfgs_original = @elapsed begin
        calib_result_bfgs_original =
            calibrate_upsilon_bfgs_ddfactplus(
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
                psi_derivative = psi_derivative,
                t1_reformulation = false,
                use_steepest_descent_fallback = use_steepest_descent_fallback,
                verbose = verbose_bfgs,
            )

        gamma_bfgs_original = calib_result_bfgs_original.gamma
        theta_bfgs_original = calib_result_bfgs_original.theta
        psi_bfgs_original = calib_result_bfgs_original.psi

        x_bfgs_original = calib_result_bfgs_original.x
        y_bfgs_original = calib_result_bfgs_original.y
        z_bfgs_original = calib_result_bfgs_original.obj
    end

    Random.seed!(1)
    # -------------------------
    # DDGFact+_Upsilon calibration
    # BFGS with psi derivative, t = 1 reformulation
    # -------------------------
    runtime_bfgs_t1_reform = @elapsed begin
        calib_result_bfgs_t1_reform =
            calibrate_upsilon_bfgs_ddfactplus(
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
                psi_derivative = psi_derivative,
                t1_reformulation = true,
                t1_fallback = true,
                t1_fallback_limit = t1_fallback_limit,
                use_steepest_descent_fallback = use_steepest_descent_fallback,
                verbose = verbose_bfgs,
            )

        gamma_bfgs_t1_reform = calib_result_bfgs_t1_reform.gamma
        theta_bfgs_t1_reform = calib_result_bfgs_t1_reform.theta
        psi_bfgs_t1_reform = calib_result_bfgs_t1_reform.psi

        x_bfgs_t1_reform = calib_result_bfgs_t1_reform.x
        y_bfgs_t1_reform = calib_result_bfgs_t1_reform.y
        z_bfgs_t1_reform = calib_result_bfgs_t1_reform.obj
    end

    # -------------------------
    # Optional feasibility diagnostics
    # -------------------------
    Cgamma_bfgs_original = scaled_matrix(Csym, gamma_bfgs_original)
    lambda_min_bfgs_original = eigmin(Cgamma_bfgs_original)
    feasibility_slack_bfgs_original = lambda_min_bfgs_original - psi_bfgs_original

    Cgamma_bfgs_t1_reform = scaled_matrix(Csym, gamma_bfgs_t1_reform)
    lambda_min_bfgs_t1_reform = eigmin(Cgamma_bfgs_t1_reform)
    feasibility_slack_bfgs_t1_reform = lambda_min_bfgs_t1_reform - psi_bfgs_t1_reform

    # -------------------------
    # Local search
    # Used only to compute gaps
    # -------------------------
    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end

    # -------------------------
    # Spectral bound
    # -------------------------
    runtime_spec = @elapsed begin
        z_spec = spectral_bound(Csym, t)
    end

    # -------------------------
    # Collect gaps and runtimes
    # -------------------------
    append!(
        result,
        [
            # Gaps
            z_ddgfact - z_ls,
            z_ddgfact_plus - z_ls,
            z_bfgs_original - z_ls,
            z_bfgs_t1_reform - z_ls,
            z_spec - z_ls,

            # Runtimes
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_bfgs_original,
            runtime_bfgs_t1_reform,
            runtime_ls,
            runtime_spec,

            # Objective values
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_bfgs_original,
            z_bfgs_t1_reform,
            z_spec,

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
        ],
    )

    push!(results, result)

    println("gap_ddgfact:                         ", z_ddgfact - z_ls)
    println("gap_ddgfact_plus:                    ", z_ddgfact_plus - z_ls)
    println("gap_bfgs_original:                   ", z_bfgs_original - z_ls)
    println("gap_bfgs_t1_reform:                  ", z_bfgs_t1_reform - z_ls)
    println("gap_spec:                            ", z_spec - z_ls)

    println("runtime_ddgfact:                     ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                ", runtime_ddgfact_plus)
    println("runtime_bfgs_original:               ", runtime_bfgs_original)
    println("runtime_bfgs_t1_reform:              ", runtime_bfgs_t1_reform)

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
    :spec_gap,

    # Runtimes
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :ddgfact_plus_upsilon_bfgs_original_runtime,
    :ddgfact_plus_upsilon_bfgs_t1_reform_runtime,
    :local_search_runtime,
    :spectral_runtime,

    # Objective values
    :z_ls,
    :z_ddgfact,
    :z_ddgfact_plus,
    :z_ddgfact_plus_upsilon_bfgs_original,
    :z_ddgfact_plus_upsilon_bfgs_t1_reform,
    :z_spec,

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
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")