using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Ipopt
using LBFGSB
import MathOptInterface as MOI

include("util.jl")
include("heuristics.jl")
include("solver_ipopt.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_lbfgsb.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
kappa = 5

# Full run:
# s_vals = [s for s in (kappa + 1):(n - 2)]

# Test run:
s_vals = [s for s in (kappa + 1):(kappa + 10)]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# Initial unscaled psi for DDGFact+
psi = eigmin(Csym) - atol

# -------------------------
# Custom BFGS calibration parameters
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

verbose_bfgs = false

# -------------------------
# LBFGSB calibration parameters
# -------------------------
gamma_lower = 1e-6
gamma_upper = 1e6

lbfgsb_m = 10
lbfgsb_factr = 1e7
lbfgsb_pgtol = 1e-2

lbfgsb_iprint = -1
lbfgsb_maxfun = 15_000
lbfgsb_maxiter = 200

verbose_lbfgsb = false

# -------------------------
# Data Collection
# -------------------------
solver = "ipopt"
calib_method = "bfgs_vs_bfgs_psideriv_vs_lbfgsb"

mkpath("results")

results_filepath = "results/results_gap_$(solver)_$(calib_method)_n$(n)_kappa$(kappa).csv"
results = []

for s in s_vals
    t = s - kappa

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

    # -------------------------
    # DDGFact+_Upsilon, custom BFGS calibration
    # fixed-psi subgradient
    # -------------------------
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
            armijo_c1 = armijo_c1,
            curvature_tol = curvature_tol,
            max_backtracks = max_backtracks,
            max_theta_norm = max_theta_norm,
            psi_derivative = false,
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

    Cgamma_upsilon_bfgs = scaled_matrix(Csym, gamma_upsilon_bfgs)
    lambda_min_upsilon_bfgs = eigmin(Cgamma_upsilon_bfgs)
    feasibility_slack_upsilon_bfgs = lambda_min_upsilon_bfgs - psi_upsilon_bfgs

    # -------------------------
    # DDGFact+_Upsilon, custom BFGS calibration
    # corrected subgradient including psi derivative
    # -------------------------
    runtime_ddgfact_plus_upsilon_bfgs_psideriv = @elapsed begin
        calib_result_bfgs_psideriv = calibrate_upsilon_bfgs_ddfactplus(
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
            psi_derivative = true,
            t1_reformulation = false,
            verbose = verbose_bfgs,
        )

        gamma_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.gamma
        theta_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.theta
        psi_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.psi

        x_ddgfact_plus_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.x
        y_ddgfact_plus_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.y
        z_ddgfact_plus_upsilon_bfgs_psideriv = calib_result_bfgs_psideriv.obj
    end

    Cgamma_upsilon_bfgs_psideriv = scaled_matrix(Csym, gamma_upsilon_bfgs_psideriv)
    lambda_min_upsilon_bfgs_psideriv = eigmin(Cgamma_upsilon_bfgs_psideriv)
    feasibility_slack_upsilon_bfgs_psideriv =
        lambda_min_upsilon_bfgs_psideriv - psi_upsilon_bfgs_psideriv

    # -------------------------
    # DDGFact+_Upsilon, LBFGSB calibration
    # -------------------------
    runtime_ddgfact_plus_upsilon_lbfgsb = @elapsed begin
        calib_result_lbfgsb = calibrate_upsilon_lbfgsb_ddfactplus(
            Csym,
            s,
            t;
            atol = atol,

            gamma_lower = gamma_lower,
            gamma_upper = gamma_upper,

            psi_margin = psi_margin,
            psi_floor = psi_floor,

            lbfgsb_m = lbfgsb_m,
            factr = lbfgsb_factr,
            pgtol = lbfgsb_pgtol,
            iprint = lbfgsb_iprint,
            maxfun = lbfgsb_maxfun,
            maxiter = lbfgsb_maxiter,

            psi_derivative = true,
            t1_reformulation = false,

            verbose = verbose_lbfgsb,
        )

        gamma_upsilon_lbfgsb = calib_result_lbfgsb.gamma
        theta_upsilon_lbfgsb = calib_result_lbfgsb.theta
        psi_upsilon_lbfgsb = calib_result_lbfgsb.psi

        x_ddgfact_plus_upsilon_lbfgsb = calib_result_lbfgsb.x
        y_ddgfact_plus_upsilon_lbfgsb = calib_result_lbfgsb.y
        z_ddgfact_plus_upsilon_lbfgsb = calib_result_lbfgsb.obj
    end

    Cgamma_upsilon_lbfgsb = scaled_matrix(Csym, gamma_upsilon_lbfgsb)
    lambda_min_upsilon_lbfgsb = eigmin(Cgamma_upsilon_lbfgsb)
    feasibility_slack_upsilon_lbfgsb = lambda_min_upsilon_lbfgsb - psi_upsilon_lbfgsb

    # -------------------------
    # Local search
    # -------------------------
    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end

    # -------------------------
    # Spectral bound
    # -------------------------
    runtime_spec = @elapsed begin
        z_spec = spectral_bound_solver(Csym, t)
    end

    append!(
        result,
        [
            # Gaps
            z_ddgfact - z_ls,
            z_ddgfact_plus - z_ls,
            z_ddgfact_plus_upsilon_bfgs - z_ls,
            z_ddgfact_plus_upsilon_bfgs_psideriv - z_ls,
            z_ddgfact_plus_upsilon_lbfgsb - z_ls,
            z_spec - z_ls,

            # Runtimes
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_ddgfact_plus_upsilon_bfgs,
            runtime_ddgfact_plus_upsilon_bfgs_psideriv,
            runtime_ddgfact_plus_upsilon_lbfgsb,
            runtime_ls,
            runtime_spec,

            # Objective values
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_ddgfact_plus_upsilon_bfgs,
            z_ddgfact_plus_upsilon_bfgs_psideriv,
            z_ddgfact_plus_upsilon_lbfgsb,
            z_spec,

            # Psi and feasibility diagnostics
            psi,

            psi_upsilon_bfgs,
            lambda_min_upsilon_bfgs,
            feasibility_slack_upsilon_bfgs,

            psi_upsilon_bfgs_psideriv,
            lambda_min_upsilon_bfgs_psideriv,
            feasibility_slack_upsilon_bfgs_psideriv,

            psi_upsilon_lbfgsb,
            lambda_min_upsilon_lbfgsb,
            feasibility_slack_upsilon_lbfgsb,

            # Gamma diagnostics
            minimum(gamma_upsilon_bfgs),
            maximum(gamma_upsilon_bfgs),
            norm(theta_upsilon_bfgs),

            minimum(gamma_upsilon_bfgs_psideriv),
            maximum(gamma_upsilon_bfgs_psideriv),
            norm(theta_upsilon_bfgs_psideriv),

            minimum(gamma_upsilon_lbfgsb),
            maximum(gamma_upsilon_lbfgsb),
            norm(theta_upsilon_lbfgsb),

            # Improvements over unscaled DDGFact+
            z_ddgfact_plus - z_ddgfact_plus_upsilon_bfgs,
            z_ddgfact_plus - z_ddgfact_plus_upsilon_bfgs_psideriv,
            z_ddgfact_plus - z_ddgfact_plus_upsilon_lbfgsb,

            # LBFGSB diagnostics
            calib_result_lbfgsb.eval_count,
        ],
    )

    push!(results, result)

    println("z_ls:                                ", z_ls)
    println("z_ddgfact:                           ", z_ddgfact)
    println("z_ddgfact_plus:                      ", z_ddgfact_plus)
    println("z_upsilon_bfgs:                      ", z_ddgfact_plus_upsilon_bfgs)
    println("z_upsilon_bfgs_psideriv:             ", z_ddgfact_plus_upsilon_bfgs_psideriv)
    println("z_upsilon_lbfgsb:                    ", z_ddgfact_plus_upsilon_lbfgsb)
    println("z_spec:                              ", z_spec)

    println("gap_bfgs:                            ", z_ddgfact_plus_upsilon_bfgs - z_ls)
    println("gap_bfgs_psideriv:                   ", z_ddgfact_plus_upsilon_bfgs_psideriv - z_ls)
    println("gap_lbfgsb:                          ", z_ddgfact_plus_upsilon_lbfgsb - z_ls)

    println("improvement_bfgs over DDG+:           ", z_ddgfact_plus - z_ddgfact_plus_upsilon_bfgs)
    println("improvement_bfgs_psideriv over DDG+:  ", z_ddgfact_plus - z_ddgfact_plus_upsilon_bfgs_psideriv)
    println("improvement_lbfgsb over DDG+:         ", z_ddgfact_plus - z_ddgfact_plus_upsilon_lbfgsb)

    println("runtime_bfgs:                        ", runtime_ddgfact_plus_upsilon_bfgs)
    println("runtime_bfgs_psideriv:               ", runtime_ddgfact_plus_upsilon_bfgs_psideriv)
    println("runtime_lbfgsb:                      ", runtime_ddgfact_plus_upsilon_lbfgsb)
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
    :ddgfact_plus_upsilon_bfgs_gap,
    :ddgfact_plus_upsilon_bfgs_psideriv_gap,
    :ddgfact_plus_upsilon_lbfgsb_gap,
    :spec_gap,

    # Runtimes
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :ddgfact_plus_upsilon_bfgs_runtime,
    :ddgfact_plus_upsilon_bfgs_psideriv_runtime,
    :ddgfact_plus_upsilon_lbfgsb_runtime,
    :local_search_runtime,
    :spectral_runtime,

    # Objective values
    :z_ls,
    :z_ddgfact,
    :z_ddgfact_plus,
    :z_ddgfact_plus_upsilon_bfgs,
    :z_ddgfact_plus_upsilon_bfgs_psideriv,
    :z_ddgfact_plus_upsilon_lbfgsb,
    :z_spec,

    # Psi and feasibility diagnostics
    :psi_ddgfact_plus,

    :psi_upsilon_bfgs,
    :lambda_min_upsilon_bfgs,
    :feasibility_slack_upsilon_bfgs,

    :psi_upsilon_bfgs_psideriv,
    :lambda_min_upsilon_bfgs_psideriv,
    :feasibility_slack_upsilon_bfgs_psideriv,

    :psi_upsilon_lbfgsb,
    :lambda_min_upsilon_lbfgsb,
    :feasibility_slack_upsilon_lbfgsb,

    # Gamma diagnostics
    :gamma_min_upsilon_bfgs,
    :gamma_max_upsilon_bfgs,
    :theta_norm_upsilon_bfgs,

    :gamma_min_upsilon_bfgs_psideriv,
    :gamma_max_upsilon_bfgs_psideriv,
    :theta_norm_upsilon_bfgs_psideriv,

    :gamma_min_upsilon_lbfgsb,
    :gamma_max_upsilon_lbfgsb,
    :theta_norm_upsilon_lbfgsb,

    # Improvements over DDGFact+
    :upsilon_bfgs_improvement_over_ddgfact_plus,
    :upsilon_bfgs_psideriv_improvement_over_ddgfact_plus,
    :upsilon_lbfgsb_improvement_over_ddgfact_plus,

    # Extra diagnostics
    :lbfgsb_eval_count,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")