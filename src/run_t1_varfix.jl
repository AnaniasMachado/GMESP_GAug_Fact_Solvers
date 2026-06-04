using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using JuMP
using Ipopt
using KNITRO
using LBFGSB
import MathOptInterface as MOI

include("util.jl")
include("heuristics.jl")
# include("solver_ipopt.jl")
include("solver_knitro.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_t1.jl")
include("gscaling_params.jl")

include("dual.jl")
include("var_fixing.jl")

# -------------------------
# Problem data
# -------------------------
n = 63

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

results_filepath = "results/results_t1_gap_vf_$(solver)_$(calib_method)_n$(n)_$(bfgs_param_set).csv"
results = []

for s in s_vals
    println("--------------------")
    println("s: $s")
    println("t: $t")
    flush(stdout)

    result = []
    append!(result, [n, s, t])

    # Root-node bounds
    l_root = zeros(Float64, n)
    c_root = ones(Float64, n)

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
    # DDGFact+_Upsilon, custom BFGS calibration
    # corrected subgradient including psi derivative
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
            alpha_decay = alpha_decay,
            armijo_c1 = armijo_c1,
            curvature_tol = curvature_tol,
            max_backtracks = max_backtracks,
            max_theta_norm = max_theta_norm,
            psi_derivative = true,
            t1_reformulation = true,
            t1_fallback = true,
            t1_fallback_limit = t1_fallback_limit,
            use_steepest_descent_fallback = use_steepest_descent_fallback,
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

    # -------------------------
    # Local search
    # Used only to compute gaps
    # -------------------------
    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end

    # -------------------------
    # Variable fixing at root node
    # -------------------------

    # DDGFact variable fixing, psi = 0
    runtime_vf_ddgfact = @elapsed begin
        F_ddgfact = factorize_matrix(Csym; psi = 0.0, atol = atol)

        vf_ddgfact_dual = var_fixing_DDGFact_dual(
            x_ddgfact,
            F_ddgfact,
            s,
            t,
            z_ls;
            l = l_root,
            c = c_root,
            atol = atol,
        )

        n_fixed_ddgfact_dual =
            length(union(vf_ddgfact_dual.fixing.fix_zero,
                         vf_ddgfact_dual.fixing.fix_one))
    end

    # DDGFact+ variable fixing, using psi
    runtime_vf_ddgfact_plus = @elapsed begin
        F_ddgfact_plus = factorize_matrix(Csym; psi = psi, atol = atol)

        vf_ddgfact_plus_dual = var_fixing_DDGFactplus_dual(
            x_ddgfact_plus,
            F_ddgfact_plus,
            s,
            t,
            psi,
            z_ls;
            l = l_root,
            c = c_root,
            atol = atol,
        )

        n_fixed_ddgfact_plus_dual =
            length(union(vf_ddgfact_plus_dual.fixing.fix_zero,
                         vf_ddgfact_plus_dual.fixing.fix_one))

        vf_ddgfact_plus_primal = var_fixing_DDGFactplus_primal(
            x_ddgfact_plus,
            F_ddgfact_plus,
            s,
            t,
            psi,
            z_ls;
            l = zeros(Float64, n),
            c = ones(Float64, n),
            atol = atol,
        )

        n_fixed_ddgfact_plus_primal =
            length(union(vf_ddgfact_plus_primal.fix_zero, vf_ddgfact_plus_primal.fix_one))

        fixed_ddgfact_plus_dual =
            union(vf_ddgfact_plus_dual.fixing.fix_zero,
                vf_ddgfact_plus_dual.fixing.fix_one)

        fixed_ddgfact_plus_primal =
            union(vf_ddgfact_plus_primal.fix_zero,
                vf_ddgfact_plus_primal.fix_one)

        n_fixed_ddgfact_plus_union = length(union(fixed_ddgfact_plus_dual, fixed_ddgfact_plus_primal))
    end

    # DDGFact+Upsilon variable fixing
    runtime_vf_ddgfact_plus_upsilon_bfgs = @elapsed begin
        Cgamma_upsilon_bfgs = scaled_matrix(Csym, gamma_upsilon_bfgs)

        F_ddgfact_plus_upsilon_bfgs =
            factorize_matrix(Cgamma_upsilon_bfgs; psi = psi_upsilon_bfgs, atol = atol)

        vf_ddgfact_plus_upsilon_bfgs_dual = var_fixing_DDGFactplusUpsilon_dual_strong(
            x_ddgfact_plus_upsilon_bfgs,
            y_ddgfact_plus_upsilon_bfgs,
            gamma_upsilon_bfgs,
            F_ddgfact_plus_upsilon_bfgs,
            s,
            t,
            psi_upsilon_bfgs,
            z_ls;
            l = l_root,
            c = c_root,
            atol = atol,
            silent = true,
        )

        n_fixed_ddgfact_plus_upsilon_bfgs_dual =
            length(union(vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_zero,
                        vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_one))

        vf_ddgfact_plus_upsilon_bfgs_primal = var_fixing_DDGFactplusUpsilon_primal(
            x_ddgfact_plus_upsilon_bfgs,
            y_ddgfact_plus_upsilon_bfgs,
            gamma_upsilon_bfgs,
            F_ddgfact_plus_upsilon_bfgs,
            s,
            t,
            psi_upsilon_bfgs,
            z_ls;
            l = l_root,
            c = c_root,
            atol = atol,
            silent = true,
        )

        n_fixed_ddgfact_plus_upsilon_bfgs_primal =
            length(union(vf_ddgfact_plus_upsilon_bfgs_primal.fix_zero,
                        vf_ddgfact_plus_upsilon_bfgs_primal.fix_one))

        fixed_ddgfact_plus_upsilon_bfgs_dual =
            union(vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_one)

        fixed_ddgfact_plus_upsilon_bfgs_primal =
            union(vf_ddgfact_plus_upsilon_bfgs_primal.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_primal.fix_one)

        n_fixed_ddgfact_plus_upsilon_bfgs_union = length(union(fixed_ddgfact_plus_upsilon_bfgs_dual, fixed_ddgfact_plus_upsilon_bfgs_primal))
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
            z_ddgfact_plus_upsilon_bfgs - z_ls,
            z_spec - z_ls,

            # Runtimes
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_ddgfact_plus_upsilon_bfgs,
            runtime_ls,
            runtime_spec,

            # Variable fixing counts
            n_fixed_ddgfact_dual,
            n_fixed_ddgfact_plus_dual,
            n_fixed_ddgfact_plus_primal,
            n_fixed_ddgfact_plus_union,
            n_fixed_ddgfact_plus_upsilon_bfgs_dual,
            n_fixed_ddgfact_plus_upsilon_bfgs_primal,
            n_fixed_ddgfact_plus_upsilon_bfgs_union,

            # Objective values
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_ddgfact_plus_upsilon_bfgs,
            z_spec,

            # Psi and feasibility diagnostics
            psi,

            psi_upsilon_bfgs,
            lambda_min_upsilon_bfgs,
            feasibility_slack_upsilon_bfgs,

            # Gamma diagnostics
            minimum(gamma_upsilon_bfgs),
            maximum(gamma_upsilon_bfgs),
            norm(theta_upsilon_bfgs),
        ],
    )

    push!(results, result)

    println("gap_ddgfact:                                            ", z_ddgfact - z_ls)
    println("gap_ddgfact_plus:                                       ", z_ddgfact_plus - z_ls)
    println("gap_ddgfact_plus_upsilon:                               ", z_ddgfact_plus_upsilon_bfgs - z_ls)
    println("gap_spec:                                               ", z_spec - z_ls)

    println("n_fixed_ddgfact_dual:                                   ", n_fixed_ddgfact_dual)
    println("n_fixed_ddgfact_plus_dual:                              ", n_fixed_ddgfact_plus_dual)
    println("n_fixed_ddgfact_plus_primal:                            ", n_fixed_ddgfact_plus_primal)
    println("n_fixed_ddgfact_plus_union:                             ", n_fixed_ddgfact_plus_union)
    println("n_fixed_ddgfact_plus_upsilon_bfgs_dual:                 ", n_fixed_ddgfact_plus_upsilon_bfgs_dual)
    println("n_fixed_ddgfact_plus_upsilon_bfgs_primal:               ", n_fixed_ddgfact_plus_upsilon_bfgs_primal)
    println("n_fixed_ddgfact_plus_upsilon_bfgs_union:                ", n_fixed_ddgfact_plus_upsilon_bfgs_union)

    println("runtime_ddgfact:                                        ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                                   ", runtime_ddgfact_plus)
    println("runtime_ddgfact_plus_upsilon:                           ", runtime_ddgfact_plus_upsilon_bfgs)

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
    :spec_gap,

    # Runtimes
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :ddgfact_plus_upsilon_bfgs_runtime,
    :local_search_runtime,
    :spectral_runtime,

    # Variable fixing counts
    :n_fixed_ddgfact_dual,
    :n_fixed_ddgfact_plus_dual,
    :n_fixed_ddgfact_plus_primal,
    :n_fixed_ddgfact_plus_union,
    :n_fixed_ddgfact_plus_upsilon_bfgs_dual,
    :n_fixed_ddgfact_plus_upsilon_bfgs_primal,
    :n_fixed_ddgfact_plus_upsilon_bfgs_union,

    # Objective values
    :z_ls,
    :z_ddgfact,
    :z_ddgfact_plus,
    :z_ddgfact_plus_upsilon_bfgs,
    :z_spec,

    # Psi and feasibility diagnostics
    :psi_ddgfact_plus,

    :psi_upsilon_bfgs,
    :lambda_min_upsilon_bfgs,
    :feasibility_slack_upsilon_bfgs,

    # Gamma diagnostics
    :gamma_min_upsilon_bfgs,
    :gamma_max_upsilon_bfgs,
    :theta_norm_upsilon_bfgs,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")