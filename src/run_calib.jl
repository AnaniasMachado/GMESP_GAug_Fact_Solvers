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
include("gscaling_rbfgs.jl")

include("dual.jl")
include("var_fixing.jl")


# -------------------------
# Problem data
# -------------------------
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


# -------------------------
# Custom BFGS calibration parameters
# -------------------------
max_bfgs_iter = 50

grad_tol = 1e-2
step_tol = 1e-8

alpha0 = 1.0
alpha_min = 1e-10
alpha_decay = 0.75
armijo_c1 = 1e-6
max_backtracks = 50

curvature_tol = 1e-12

# Largest feasible psi will be:
# psi(gamma) = lambda_min(D_gamma^(1/2) C D_gamma^(1/2)) - psi_margin
psi_margin = 1e-7
psi_floor = 0.0

max_theta_norm = 20.0

verbose_bfgs = false


# -------------------------
# Regularized BFGS calibration parameters
# -------------------------
max_regbfgs_iter = 100

# Initial dense Hessian approximation B0 = B0_scale * I.
regbfgs_B0_scale = 1.0

# Regularization parameter in (B + mu I)d = -g.
regbfgs_mu0 = 1e-2
regbfgs_mu_min = 1e-10
regbfgs_mu_max = 1e4
regbfgs_mu_decrease = 0.2
regbfgs_mu_increase = 5.0
regbfgs_eta1 = 0.05
regbfgs_eta2 = 0.75
regbfgs_max_inner_regularization = 20

# Dense BFGS curvature controls.
regbfgs_curvature_tol = 1e-12
regbfgs_damping_delta = 0.2
regbfgs_reset_B_on_failed_update = false

# Direction and iterate safeguards.
regbfgs_normalize_direction = false
regbfgs_max_direction_norm = 10.0
regbfgs_max_q_norm_inf = 20.0

# Armijo line search.
regbfgs_armijo_c1 = 1e-4
regbfgs_accept_tol = 1e-12
regbfgs_alpha0 = 1.0
regbfgs_alpha_min = 1e-12
regbfgs_alpha_decay = 0.5
regbfgs_max_backtracks = 30

# Nonmonotone Armijo reference.
regbfgs_nonmonotone = true
regbfgs_nonmonotone_window = 10

# Hessian safeguards.
regbfgs_project_spd = true
regbfgs_min_B_eig = 1e-8
regbfgs_max_B_eig = 1e8
regbfgs_reset_B_on_bad = true
regbfgs_max_B_norm = 1e8

regbfgs_grad_tol = 1e-2
regbfgs_step_tol = 1e-10

# If false, history and spectral diagnostics are not collected.
# The final eigenvalue-based fields will be NaN.
regbfgs_diagnostics = false

verbose_regbfgs = false


# -------------------------
# Data Collection
# -------------------------
solver = "knitro"
calib_method = "bfgs_regbfgs"

mkpath("results")

results_filepath = "results/results_gap_vf_$(solver)_$(calib_method)_n$(n)_kappa$(kappa).csv"
results = []

for s in s_vals
    t = s - kappa

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


    # -------------------------
    # DDGFact+_Upsilon, custom BFGS calibration
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
    feasibility_slack_upsilon_bfgs =
        lambda_min_upsilon_bfgs - psi_upsilon_bfgs


    # -------------------------
    # DDGFact+_Upsilon, regularized BFGS calibration
    # -------------------------
    Random.seed!(1)

    runtime_ddgfact_plus_upsilon_regbfgs = @elapsed begin
        calib_result_regbfgs = solve_regularized_bfgs_upsilon_calibration(
            Csym,
            s,
            t;
            q0 = 1e-4 .* randn(n),

            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            psi_derivative = true,
            t1_reformulation = false,

            q_bound = Inf,
            max_q_norm_inf = regbfgs_max_q_norm_inf,

            max_iter = max_regbfgs_iter,

            B0_scale = regbfgs_B0_scale,

            mu0 = regbfgs_mu0,
            mu_min = regbfgs_mu_min,
            mu_max = regbfgs_mu_max,
            mu_decrease = regbfgs_mu_decrease,
            mu_increase = regbfgs_mu_increase,
            eta1 = regbfgs_eta1,
            eta2 = regbfgs_eta2,
            max_inner_regularization = regbfgs_max_inner_regularization,

            normalize_direction = regbfgs_normalize_direction,
            max_direction_norm = regbfgs_max_direction_norm,

            armijo_c1 = regbfgs_armijo_c1,
            accept_tol = regbfgs_accept_tol,
            alpha0 = regbfgs_alpha0,
            alpha_min = regbfgs_alpha_min,
            alpha_decay = regbfgs_alpha_decay,
            max_backtracks = regbfgs_max_backtracks,

            nonmonotone = regbfgs_nonmonotone,
            nonmonotone_window = regbfgs_nonmonotone_window,

            curvature_tol = regbfgs_curvature_tol,
            damping_delta = regbfgs_damping_delta,
            reset_B_on_failed_update = regbfgs_reset_B_on_failed_update,

            project_spd = regbfgs_project_spd,
            min_B_eig = regbfgs_min_B_eig,
            max_B_eig = regbfgs_max_B_eig,
            reset_B_on_bad = regbfgs_reset_B_on_bad,
            max_B_norm = regbfgs_max_B_norm,

            grad_tol = regbfgs_grad_tol,
            step_tol = regbfgs_step_tol,

            cache_digits = 12,
            diagnostics = regbfgs_diagnostics,
            verbose = verbose_regbfgs,
        )

        gamma_upsilon_regbfgs = calib_result_regbfgs.gamma
        theta_upsilon_regbfgs = calib_result_regbfgs.theta
        psi_upsilon_regbfgs = calib_result_regbfgs.psi

        x_ddgfact_plus_upsilon_regbfgs = calib_result_regbfgs.x
        y_ddgfact_plus_upsilon_regbfgs = calib_result_regbfgs.y
        z_ddgfact_plus_upsilon_regbfgs = calib_result_regbfgs.obj

        regbfgs_num_evals = calib_result_regbfgs.num_evals
        regbfgs_cache_size = calib_result_regbfgs.cache_size
        regbfgs_final_mu = calib_result_regbfgs.final_mu

        regbfgs_final_B_norm = calib_result_regbfgs.final_B_norm
        regbfgs_final_B_min_eig = calib_result_regbfgs.final_B_min_eig
        regbfgs_final_B_max_eig = calib_result_regbfgs.final_B_max_eig

        regbfgs_final_H_norm = calib_result_regbfgs.final_H_norm
        regbfgs_final_H_min_eig = calib_result_regbfgs.final_H_min_eig
        regbfgs_final_H_max_eig = calib_result_regbfgs.final_H_max_eig
    end

    Cgamma_upsilon_regbfgs = scaled_matrix(Csym, gamma_upsilon_regbfgs)
    lambda_min_upsilon_regbfgs = eigmin(Cgamma_upsilon_regbfgs)
    feasibility_slack_upsilon_regbfgs =
        lambda_min_upsilon_regbfgs - psi_upsilon_regbfgs


    # -------------------------
    # Local search
    # Needed before variable fixing because LB = z_ls
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
            length(union(
                vf_ddgfact_dual.fixing.fix_zero,
                vf_ddgfact_dual.fixing.fix_one,
            ))
    end


    # DDGFact+ variable fixing
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
            length(union(
                vf_ddgfact_plus_dual.fixing.fix_zero,
                vf_ddgfact_plus_dual.fixing.fix_one,
            ))

        vf_ddgfact_plus_primal = var_fixing_DDGFactplus_primal(
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

        n_fixed_ddgfact_plus_primal =
            length(union(
                vf_ddgfact_plus_primal.fix_zero,
                vf_ddgfact_plus_primal.fix_one,
            ))

        fixed_ddgfact_plus_dual =
            union(
                vf_ddgfact_plus_dual.fixing.fix_zero,
                vf_ddgfact_plus_dual.fixing.fix_one,
            )

        fixed_ddgfact_plus_primal =
            union(
                vf_ddgfact_plus_primal.fix_zero,
                vf_ddgfact_plus_primal.fix_one,
            )

        n_fixed_ddgfact_plus_union =
            length(union(fixed_ddgfact_plus_dual, fixed_ddgfact_plus_primal))
    end


    # DDGFact+_Upsilon BFGS variable fixing
    runtime_vf_ddgfact_plus_upsilon_bfgs = @elapsed begin
        Cgamma_upsilon_bfgs = scaled_matrix(Csym, gamma_upsilon_bfgs)

        F_ddgfact_plus_upsilon_bfgs =
            factorize_matrix(Cgamma_upsilon_bfgs; psi = psi_upsilon_bfgs, atol = atol)

        vf_ddgfact_plus_upsilon_bfgs_dual =
            var_fixing_DDGFactplusUpsilon_dual_strong(
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
            length(union(
                vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_one,
            ))

        vf_ddgfact_plus_upsilon_bfgs_primal =
            var_fixing_DDGFactplusUpsilon_primal(
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
            length(union(
                vf_ddgfact_plus_upsilon_bfgs_primal.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_primal.fix_one,
            ))

        fixed_ddgfact_plus_upsilon_bfgs_dual =
            union(
                vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_dual.fixing.fix_one,
            )

        fixed_ddgfact_plus_upsilon_bfgs_primal =
            union(
                vf_ddgfact_plus_upsilon_bfgs_primal.fix_zero,
                vf_ddgfact_plus_upsilon_bfgs_primal.fix_one,
            )

        n_fixed_ddgfact_plus_upsilon_bfgs_union =
            length(union(
                fixed_ddgfact_plus_upsilon_bfgs_dual,
                fixed_ddgfact_plus_upsilon_bfgs_primal,
            ))
    end


    # DDGFact+_Upsilon R-BFGS variable fixing
    runtime_vf_ddgfact_plus_upsilon_regbfgs = @elapsed begin
        Cgamma_upsilon_regbfgs = scaled_matrix(Csym, gamma_upsilon_regbfgs)

        F_ddgfact_plus_upsilon_regbfgs =
            factorize_matrix(Cgamma_upsilon_regbfgs; psi = psi_upsilon_regbfgs, atol = atol)

        vf_ddgfact_plus_upsilon_regbfgs_dual =
            var_fixing_DDGFactplusUpsilon_dual_strong(
                x_ddgfact_plus_upsilon_regbfgs,
                y_ddgfact_plus_upsilon_regbfgs,
                gamma_upsilon_regbfgs,
                F_ddgfact_plus_upsilon_regbfgs,
                s,
                t,
                psi_upsilon_regbfgs,
                z_ls;
                l = l_root,
                c = c_root,
                atol = atol,
                silent = true,
            )

        n_fixed_ddgfact_plus_upsilon_regbfgs_dual =
            length(union(
                vf_ddgfact_plus_upsilon_regbfgs_dual.fixing.fix_zero,
                vf_ddgfact_plus_upsilon_regbfgs_dual.fixing.fix_one,
            ))

        vf_ddgfact_plus_upsilon_regbfgs_primal =
            var_fixing_DDGFactplusUpsilon_primal(
                x_ddgfact_plus_upsilon_regbfgs,
                y_ddgfact_plus_upsilon_regbfgs,
                gamma_upsilon_regbfgs,
                F_ddgfact_plus_upsilon_regbfgs,
                s,
                t,
                psi_upsilon_regbfgs,
                z_ls;
                l = l_root,
                c = c_root,
                atol = atol,
                silent = true,
            )

        n_fixed_ddgfact_plus_upsilon_regbfgs_primal =
            length(union(
                vf_ddgfact_plus_upsilon_regbfgs_primal.fix_zero,
                vf_ddgfact_plus_upsilon_regbfgs_primal.fix_one,
            ))

        fixed_ddgfact_plus_upsilon_regbfgs_dual =
            union(
                vf_ddgfact_plus_upsilon_regbfgs_dual.fixing.fix_zero,
                vf_ddgfact_plus_upsilon_regbfgs_dual.fixing.fix_one,
            )

        fixed_ddgfact_plus_upsilon_regbfgs_primal =
            union(
                vf_ddgfact_plus_upsilon_regbfgs_primal.fix_zero,
                vf_ddgfact_plus_upsilon_regbfgs_primal.fix_one,
            )

        n_fixed_ddgfact_plus_upsilon_regbfgs_union =
            length(union(
                fixed_ddgfact_plus_upsilon_regbfgs_dual,
                fixed_ddgfact_plus_upsilon_regbfgs_primal,
            ))
    end


    # -------------------------
    # Spectral bound
    # -------------------------
    runtime_spec = @elapsed begin
        z_spec = spectral_bound(Csym, t)
    end


    append!(
        result,
        [
            # Gaps
            z_ddgfact - z_ls,
            z_ddgfact_plus - z_ls,
            z_ddgfact_plus_upsilon_bfgs - z_ls,
            z_ddgfact_plus_upsilon_regbfgs - z_ls,
            z_spec - z_ls,

            # Runtimes
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_ddgfact_plus_upsilon_bfgs,
            runtime_ddgfact_plus_upsilon_regbfgs,
            runtime_ls,
            runtime_spec,

            # Variable fixing runtimes
            runtime_vf_ddgfact,
            runtime_vf_ddgfact_plus,
            runtime_vf_ddgfact_plus_upsilon_bfgs,
            runtime_vf_ddgfact_plus_upsilon_regbfgs,

            # Variable fixing counts
            n_fixed_ddgfact_dual,

            n_fixed_ddgfact_plus_dual,
            n_fixed_ddgfact_plus_primal,
            n_fixed_ddgfact_plus_union,

            n_fixed_ddgfact_plus_upsilon_bfgs_dual,
            n_fixed_ddgfact_plus_upsilon_bfgs_primal,
            n_fixed_ddgfact_plus_upsilon_bfgs_union,

            n_fixed_ddgfact_plus_upsilon_regbfgs_dual,
            n_fixed_ddgfact_plus_upsilon_regbfgs_primal,
            n_fixed_ddgfact_plus_upsilon_regbfgs_union,

            # Objective values
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_ddgfact_plus_upsilon_bfgs,
            z_ddgfact_plus_upsilon_regbfgs,
            z_spec,

            # Psi and feasibility diagnostics
            psi,

            psi_upsilon_bfgs,
            lambda_min_upsilon_bfgs,
            feasibility_slack_upsilon_bfgs,

            psi_upsilon_regbfgs,
            lambda_min_upsilon_regbfgs,
            feasibility_slack_upsilon_regbfgs,

            # Gamma diagnostics: BFGS
            minimum(gamma_upsilon_bfgs),
            maximum(gamma_upsilon_bfgs),
            norm(theta_upsilon_bfgs),
            norm(theta_upsilon_bfgs, Inf),

            # Gamma diagnostics: R-BFGS
            minimum(gamma_upsilon_regbfgs),
            maximum(gamma_upsilon_regbfgs),
            norm(theta_upsilon_regbfgs),
            norm(theta_upsilon_regbfgs, Inf),

            # R-BFGS diagnostics
            regbfgs_num_evals,
            regbfgs_cache_size,
            regbfgs_final_mu,
            regbfgs_final_B_norm,
            regbfgs_final_B_min_eig,
            regbfgs_final_B_max_eig,
            regbfgs_final_H_norm,
            regbfgs_final_H_min_eig,
            regbfgs_final_H_max_eig,
        ],
    )

    push!(results, result)

    println("gap_ddgfact:                                            ", z_ddgfact - z_ls)
    println("gap_ddgfact_plus:                                       ", z_ddgfact_plus - z_ls)
    println("gap_ddgfact_plus_upsilon_bfgs:                          ", z_ddgfact_plus_upsilon_bfgs - z_ls)
    println("gap_ddgfact_plus_upsilon_regbfgs:                       ", z_ddgfact_plus_upsilon_regbfgs - z_ls)
    println("gap_spec:                                               ", z_spec - z_ls)

    println("n_fixed_ddgfact_dual:                                   ", n_fixed_ddgfact_dual)

    println("n_fixed_ddgfact_plus_dual:                              ", n_fixed_ddgfact_plus_dual)
    println("n_fixed_ddgfact_plus_primal:                            ", n_fixed_ddgfact_plus_primal)
    println("n_fixed_ddgfact_plus_union:                             ", n_fixed_ddgfact_plus_union)

    println("n_fixed_ddgfact_plus_upsilon_bfgs_dual:                 ", n_fixed_ddgfact_plus_upsilon_bfgs_dual)
    println("n_fixed_ddgfact_plus_upsilon_bfgs_primal:               ", n_fixed_ddgfact_plus_upsilon_bfgs_primal)
    println("n_fixed_ddgfact_plus_upsilon_bfgs_union:                ", n_fixed_ddgfact_plus_upsilon_bfgs_union)

    println("n_fixed_ddgfact_plus_upsilon_regbfgs_dual:              ", n_fixed_ddgfact_plus_upsilon_regbfgs_dual)
    println("n_fixed_ddgfact_plus_upsilon_regbfgs_primal:            ", n_fixed_ddgfact_plus_upsilon_regbfgs_primal)
    println("n_fixed_ddgfact_plus_upsilon_regbfgs_union:             ", n_fixed_ddgfact_plus_upsilon_regbfgs_union)

    println("runtime_ddgfact:                                        ", runtime_ddgfact)
    println("runtime_ddgfact_plus:                                   ", runtime_ddgfact_plus)
    println("runtime_ddgfact_plus_upsilon_bfgs:                      ", runtime_ddgfact_plus_upsilon_bfgs)
    println("runtime_ddgfact_plus_upsilon_regbfgs:                   ", runtime_ddgfact_plus_upsilon_regbfgs)

    println("gamma_min_upsilon_bfgs:                                 ", minimum(gamma_upsilon_bfgs))
    println("gamma_max_upsilon_bfgs:                                 ", maximum(gamma_upsilon_bfgs))
    println("gamma_min_upsilon_regbfgs:                              ", minimum(gamma_upsilon_regbfgs))
    println("gamma_max_upsilon_regbfgs:                              ", maximum(gamma_upsilon_regbfgs))

    println("regbfgs_num_evals:                                      ", regbfgs_num_evals)
    println("regbfgs_final_mu:                                       ", regbfgs_final_mu)
    println("regbfgs_final_B_norm:                                   ", regbfgs_final_B_norm)
    println("regbfgs_final_B_min_eig:                                ", regbfgs_final_B_min_eig)
    println("regbfgs_final_B_max_eig:                                ", regbfgs_final_B_max_eig)
    println("regbfgs_final_H_norm:                                   ", regbfgs_final_H_norm)
    println("regbfgs_final_H_min_eig:                                ", regbfgs_final_H_min_eig)
    println("regbfgs_final_H_max_eig:                                ", regbfgs_final_H_max_eig)

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
    :ddgfact_plus_upsilon_regbfgs_gap,
    :spec_gap,

    # Runtimes
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :ddgfact_plus_upsilon_bfgs_runtime,
    :ddgfact_plus_upsilon_regbfgs_runtime,
    :local_search_runtime,
    :spectral_runtime,

    # Variable fixing runtimes
    :vf_ddgfact_runtime,
    :vf_ddgfact_plus_runtime,
    :vf_ddgfact_plus_upsilon_bfgs_runtime,
    :vf_ddgfact_plus_upsilon_regbfgs_runtime,

    # Variable fixing counts
    :n_fixed_ddgfact_dual,

    :n_fixed_ddgfact_plus_dual,
    :n_fixed_ddgfact_plus_primal,
    :n_fixed_ddgfact_plus_union,

    :n_fixed_ddgfact_plus_upsilon_bfgs_dual,
    :n_fixed_ddgfact_plus_upsilon_bfgs_primal,
    :n_fixed_ddgfact_plus_upsilon_bfgs_union,

    :n_fixed_ddgfact_plus_upsilon_regbfgs_dual,
    :n_fixed_ddgfact_plus_upsilon_regbfgs_primal,
    :n_fixed_ddgfact_plus_upsilon_regbfgs_union,

    # Objective values
    :z_ls,
    :z_ddgfact,
    :z_ddgfact_plus,
    :z_ddgfact_plus_upsilon_bfgs,
    :z_ddgfact_plus_upsilon_regbfgs,
    :z_spec,

    # Psi and feasibility diagnostics
    :psi_ddgfact_plus,

    :psi_upsilon_bfgs,
    :lambda_min_upsilon_bfgs,
    :feasibility_slack_upsilon_bfgs,

    :psi_upsilon_regbfgs,
    :lambda_min_upsilon_regbfgs,
    :feasibility_slack_upsilon_regbfgs,

    # Gamma diagnostics: BFGS
    :gamma_min_upsilon_bfgs,
    :gamma_max_upsilon_bfgs,
    :theta_norm_upsilon_bfgs,
    :theta_norm_inf_upsilon_bfgs,

    # Gamma diagnostics: R-BFGS
    :gamma_min_upsilon_regbfgs,
    :gamma_max_upsilon_regbfgs,
    :theta_norm_upsilon_regbfgs,
    :theta_norm_inf_upsilon_regbfgs,

    # R-BFGS diagnostics
    :regbfgs_num_evals,
    :regbfgs_cache_size,
    :regbfgs_final_mu,
    :regbfgs_final_B_norm,
    :regbfgs_final_B_min_eig,
    :regbfgs_final_B_max_eig,
    :regbfgs_final_H_norm,
    :regbfgs_final_H_min_eig,
    :regbfgs_final_H_max_eig,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")