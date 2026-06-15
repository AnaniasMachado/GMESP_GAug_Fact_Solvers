# =============================================================================
# test_levenberg_bfgs.jl
#
# Test Levenberg / trust-region regularized BFGS calibration for
# DDGFact+_Upsilon.
# =============================================================================

using Random
using MAT
using LinearAlgebra
using Printf
using CSV
using DataFrames


# =============================================================================
# Includes from your project
# =============================================================================

include("util.jl")
include("heuristics.jl")
include("solver_knitro.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_t1.jl")
include("gscaling_params.jl")

include("dual.jl")
include("var_fixing.jl")

include("levenberg_bfgs_util.jl")


# =============================================================================
# Instance
# =============================================================================

Random.seed!(1)

data_n = 63
k = 63

s = 2
t = 1

atol = 1e-10
psi_margin = 1e-7
psi_floor = 0.0

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
Csym = Symmetric(C)

n = size(Csym, 1)

println()
println("="^80)
println("Levenberg-BFGS DDGFact+_Upsilon calibration test")
println("="^80)
println("data_n:     ", data_n)
println("n:          ", n)
println("s:          ", s)
println("t:          ", t)
println("psi_margin: ", psi_margin)
println("="^80)


# =============================================================================
# Incumbent lower bound
# =============================================================================

runtime_ls = @elapsed begin
    x_ls, z_ls = run_all_LS(Csym, s, t)
end

S_inc = sort(findall(x_ls .> 0.5))

println()
println("="^80)
println("Incumbent")
println("="^80)
println("incumbent S:        ", S_inc)
println("incumbent lb z_ls:  ", z_ls)
println("local search time:  ", runtime_ls)
println("="^80)


# =============================================================================
# Baselines
# =============================================================================

println()
println("="^80)
println("Baseline relaxations")
println("="^80)

runtime_ddgfact = @elapsed begin
    x_ddgfact, z_ddgfact = ddfact_gmesp(
        Csym,
        s,
        t;
        J1 = Int[],
        atol = atol,
    )
end

psi_ddgfact_plus = eigmin(Csym) - atol

runtime_ddgfact_plus = @elapsed begin
    x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(
        Csym,
        s,
        t,
        psi_ddgfact_plus;
        J1 = Int[],
        atol = atol,
    )
end

println("DDGFact ub:      ", z_ddgfact)
println("DDGFact gap:     ", z_ddgfact - z_ls)
println("DDGFact time:    ", runtime_ddgfact)
println()
println("DDGFact+ psi:    ", psi_ddgfact_plus)
println("DDGFact+ ub:     ", z_ddgfact_plus)
println("DDGFact+ gap:    ", z_ddgfact_plus - z_ls)
println("DDGFact+ time:   ", runtime_ddgfact_plus)
println("="^80)


# =============================================================================
# BFGS baseline
# =============================================================================

println()
println("="^80)
println("BFGS calibration baseline")
println("="^80)

runtime_bfgs = @elapsed begin
    bfgs = calibrate_upsilon_bfgs_ddfactplus(
        Csym,
        s,
        t;
        atol = atol,
        max_iter = 50,
        grad_tol = 1e-2,
        step_tol = 1e-8,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        alpha0 = 1.0,
        alpha_min = 1e-10,
        alpha_decay = 0.75,
        armijo_c1 = 1e-6,
        curvature_tol = 1e-12,
        max_backtracks = 50,
        max_theta_norm = 20.0,
        psi_derivative = true,
        t1_reformulation = false,
        verbose = false,
    )
end

println("BFGS ub:        ", bfgs.obj)
println("BFGS gap:       ", bfgs.obj - z_ls)
println("BFGS time:      ", runtime_bfgs)
println("BFGS gamma min: ", minimum(bfgs.gamma))
println("BFGS gamma max: ", maximum(bfgs.gamma))
println("BFGS q inf:     ", norm(bfgs.theta, Inf))
println("="^80)


# =============================================================================
# Levenberg-BFGS calibration
# =============================================================================

println()
println("="^80)
println("Running Levenberg-BFGS calibration")
println("="^80)

levenberg_theta_perturbation = 1e-4

runtime_levenberg = @elapsed begin
    levenberg = solve_levenberg_bfgs_upsilon_calibration(
        Csym,
        s,
        t;
        q0 = levenberg_theta_perturbation .* randn(n),

        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = true,
        t1_reformulation = false,

        q_bound = Inf,

        max_iter = 50,

        # Initial Hessian approximation.
        B0_scale = 1.0,

        # Levenberg regularization.
        lambda0 = 1e-2,
        lambda_min = 1e-10,
        lambda_max = 1e8,
        lambda_decrease = 0.5,
        lambda_increase = 10.0,

        # Direction / trust-region safeguards.
        normalize_direction = false,
        max_direction_norm = 2.0,
        max_q_norm_inf = 20.0,

        # Line search.
        alpha0 = 1.0,
        alpha_min = 1e-12,
        alpha_decay = 0.75,
        max_backtracks = 60,
        accept_tol = 1e-12,
        use_armijo = true,
        armijo_c1 = 1e-8,

        # Optional nonmonotone acceptance.
        nonmonotone = false,
        nonmonotone_window = 5,
        nonmonotone_tol = 1e-10,

        # BFGS update.
        update_rule = :damped_bfgs,
        curvature_tol = 1e-12,
        damping_delta = 0.2,

        project_spd = true,
        min_B_eig = 1e-8,
        max_B_eig = 1e8,

        reset_B_on_bad_update = true,
        max_B_norm = 1e8,

        grad_tol = 1e-8,
        step_tol = 1e-10,

        # Inactive by default.
        best_improve_tol = 0.0,
        patience = typemax(Int),

        verbose = true,
    )
end

levenberg_gap = levenberg.obj - z_ls

println()
println("="^80)
println("Levenberg-BFGS calibration summary")
println("="^80)
println("runtime:          ", runtime_levenberg)
println("num evals:        ", levenberg.num_evals)
println("cache size:       ", levenberg.cache_size)
println("best ub:          ", levenberg.obj)
println("best gap:         ", levenberg_gap)
println("best psi:         ", levenberg.psi)
println("best lambda_min:  ", levenberg.lambda_min)
println("best min gamma:   ", minimum(levenberg.gamma))
println("best max gamma:   ", maximum(levenberg.gamma))
println("q norm:           ", norm(levenberg.q))
println("q norm inf:       ", norm(levenberg.q, Inf))
println("final lambda:     ", levenberg.final_lambda)
println("final B norm:     ", levenberg.final_B_norm)
println("="^80)


# =============================================================================
# Recheck Levenberg-BFGS best gamma directly
# =============================================================================

println()
println("="^80)
println("Rechecking Levenberg-BFGS best gamma")
println("="^80)

runtime_levenberg_recheck = @elapsed begin
    x_levenberg_recheck,
    y_levenberg_recheck,
    z_levenberg_recheck = aug_ddfact_upsilon_gmesp(
        Csym,
        levenberg.gamma,
        s,
        t,
        levenberg.psi;
        J1 = Int[],
        atol = atol,
    )
end

println("Levenberg recheck ub:      ", z_levenberg_recheck)
println("Levenberg recheck gap:     ", z_levenberg_recheck - z_ls)
println("Levenberg recheck diff:    ", z_levenberg_recheck - levenberg.obj)
println("Levenberg recheck time:    ", runtime_levenberg_recheck)
println("sum x:                     ", sum(x_levenberg_recheck))
println("sum y:                     ", sum(y_levenberg_recheck))
println("min x / max x:             ", minimum(x_levenberg_recheck), " / ", maximum(x_levenberg_recheck))
println("min y / max y:             ", minimum(y_levenberg_recheck), " / ", maximum(y_levenberg_recheck))
println("="^80)


# =============================================================================
# Save results
# =============================================================================

mkpath("results")

summary = DataFrame(
    method = [
        "DDGFact",
        "DDGFact+",
        "BFGS DDGFact+_Upsilon",
        "Levenberg-BFGS DDGFact+_Upsilon",
        "Levenberg-BFGS DDGFact+_Upsilon recheck",
    ],
    root_ub = [
        z_ddgfact,
        z_ddgfact_plus,
        bfgs.obj,
        levenberg.obj,
        z_levenberg_recheck,
    ],
    root_gap = [
        z_ddgfact - z_ls,
        z_ddgfact_plus - z_ls,
        bfgs.obj - z_ls,
        levenberg.obj - z_ls,
        z_levenberg_recheck - z_ls,
    ],
    runtime = [
        runtime_ddgfact,
        runtime_ddgfact_plus,
        runtime_bfgs,
        runtime_levenberg,
        runtime_levenberg_recheck,
    ],
    psi = [
        0.0,
        psi_ddgfact_plus,
        bfgs.psi,
        levenberg.psi,
        levenberg.psi,
    ],
    min_gamma = [
        missing,
        missing,
        minimum(bfgs.gamma),
        minimum(levenberg.gamma),
        minimum(levenberg.gamma),
    ],
    max_gamma = [
        missing,
        missing,
        maximum(bfgs.gamma),
        maximum(levenberg.gamma),
        maximum(levenberg.gamma),
    ],
    q_norm_inf = [
        missing,
        missing,
        norm(bfgs.theta, Inf),
        norm(levenberg.q, Inf),
        norm(levenberg.q, Inf),
    ],
    num_evals = [
        missing,
        missing,
        missing,
        levenberg.num_evals,
        missing,
    ],
    final_lambda = [
        missing,
        missing,
        missing,
        levenberg.final_lambda,
        levenberg.final_lambda,
    ],
    final_B_norm = [
        missing,
        missing,
        missing,
        levenberg.final_B_norm,
        levenberg.final_B_norm,
    ],
)

summary_filepath =
    "results/test_levenberg_bfgs_summary_data$(data_n)_n$(n)_s$(s)_t$(t).csv"

CSV.write(summary_filepath, summary)

history_df = DataFrame(levenberg.history)

history_filepath =
    "results/test_levenberg_bfgs_history_data$(data_n)_n$(n)_s$(s)_t$(t).csv"

CSV.write(history_filepath, history_df)

println()
println("="^80)
println("Summary")
println("="^80)
show(summary; allrows = true, allcols = true)
println()
println("="^80)
println("Saved summary to: ", summary_filepath)
println("Saved history to: ", history_filepath)