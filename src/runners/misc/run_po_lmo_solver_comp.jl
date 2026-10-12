using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Printf

include("../../misc/util.jl")
include("../../misc/heuristics.jl")
include("../../solvers/solver_knitro.jl")

include("../../gscaling/gscaling_util.jl")
include("../../gscaling/gscaling_projection_free.jl")

include("../../misc/dual.jl")


# ============================================================
# Problem data
# ============================================================

n = 90
kappa = 4

# Full run:
# s_vals = collect((kappa + 1):(n - 1))

# Test run:
s_vals = collect((kappa + 1):(kappa + 5))

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10


# ============================================================
# PO-LMO parameters
# ============================================================

pf_max_iter = 500
pf_min_iter = 100
pf_iteration_power = 1.5

pf_theta_bound = 20.0

psi_margin = 1e-7
psi_floor = 0.0
pf_psi_derivative = true

# PO-LMO parameters
pf_tau0 = 1.00
pf_tau_power = 2.0 / 3.0

pf_beta0 = 0.10
pf_beta_power = 1.0 / 6.0

pf_Lqq_hat = 0.50

# Run all three q/w LMO implementations.
w_solvers = [
    :solver,
    :dp,
    :sort,
]

# Gurobi options for :solver.
pf_gurobi_output_flag = 0
pf_gurobi_opttol = 1e-8
pf_gurobi_feastol = 1e-8

# Diagnostics must be true because operator runtimes
# are stored in the diagnostic history.
pf_diagnostics = true
pf_verbose = false


# ============================================================
# Relaxation solve parameters used for the dual gap
# ============================================================

relax_knitro_outlev = nothing
relax_knitro_opttol = 1e-8
relax_knitro_feastol = 1e-5


# ============================================================
# Output
# ============================================================

mkpath("results")

results_filepath =
    "results/minimax/results_po_lmo_lmo_comparison_n$(n)_kappa$(kappa).csv"

rows = Dict{Symbol,Any}[]


# ============================================================
# Initial theta
# ============================================================

function make_theta0(n::Int)
    Random.seed!(1)

    theta0 = 1e-2 .* randn(n)

    return clamp.(theta0, -pf_theta_bound, pf_theta_bound)
end


# ============================================================
# Fixed-variable bounds for dual construction
# ============================================================

function fixed_bounds_from_J(n::Int, J1, J0)
    l = zeros(Float64, n)
    c = ones(Float64, n)

    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    l[J1v] .= 1.0
    c[J1v] .= 1.0

    l[J0v] .= 0.0
    c[J0v] .= 0.0

    return l, c
end


# ============================================================
# Dual gap
# ============================================================

function compute_dual_gap(
    C::Symmetric{<:Real,<:AbstractMatrix},
    pf_result,
    z_ls::Float64,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
)
    n = size(C, 1)

    theta = pf_result.theta
    gamma = exp.(theta)

    psi_val, _ = max_feasible_psi(
        C,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    F = scaled_factorize_matrix(
        C,
        gamma,
        psi_val;
        atol = atol,
    )

    l, c = fixed_bounds_from_J(n, J1, J0)

    dual =
        DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
            pf_result.x,
            gamma,
            F,
            s,
            t,
            psi_val;
            yhat = pf_result.y,
            l = l,
            c = c,
            atol = atol,
            silent = true,
        )

    return dual.objective_value - z_ls
end


# ============================================================
# Warm-up
# ============================================================

s_warm = first(s_vals)
t_warm = s_warm - kappa
theta0_warm = make_theta0(n)

println("Warming up...")

for w_solver in w_solvers
    calibrate_upsilon_projection_free_ddfactplus(
        Csym,
        s_warm,
        t_warm;
        algorithm = :po_lmo,
        lmo_q_method = w_solver,
        J1 = Int[],
        J0 = Int[],
        theta0 = copy(theta0_warm),

        # Only a few iterations are needed for compilation.
        max_iter = 3,
        min_iter = 3,
        iteration_power = pf_iteration_power,

        theta_bound = pf_theta_bound,
        tau0 = pf_tau0,
        tau_power = pf_tau_power,
        beta0 = pf_beta0,
        beta_power = pf_beta_power,
        Lqq_hat = pf_Lqq_hat,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = pf_psi_derivative,
        atol = atol,

        gurobi_output_flag = pf_gurobi_output_flag,
        gurobi_opttol = pf_gurobi_opttol,
        gurobi_feastol = pf_gurobi_feastol,

        diagnostics = false,
        verbose = false,
    )
end

GC.gc()
println("Warm-up complete.")


# ============================================================
# Main loop
# ============================================================

for s in s_vals
    t = s - kappa

    println("Running n=$n, s=$s, t=$t")
    flush(stdout)

    # Local-search value is needed as the reference value
    # for the dual gap.
    _, z_ls = run_all_LS(Csym, s, t)

    theta0 = make_theta0(n)

    for w_solver in w_solvers
        println("  w_solver = $w_solver")
        flush(stdout)

        # ========================================================
        # Run 1: clean algorithm runtime
        #
        # diagnostics=false so timing is not affected by
        # operator-runtime measurements or history collection.
        # ========================================================

        result_runtime = nothing

        GC.gc()

        total_runtime = @elapsed begin
            result_runtime =
                calibrate_upsilon_projection_free_ddfactplus(
                    Csym,
                    s,
                    t;

                    algorithm = :po_lmo,
                    lmo_q_method = w_solver,

                    J1 = Int[],
                    J0 = Int[],

                    theta0 = copy(theta0),

                    max_iter = pf_max_iter,
                    min_iter = pf_min_iter,
                    iteration_power = pf_iteration_power,

                    theta_bound = pf_theta_bound,

                    tau0 = pf_tau0,
                    tau_power = pf_tau_power,

                    beta0 = pf_beta0,
                    beta_power = pf_beta_power,

                    Lqq_hat = pf_Lqq_hat,

                    psi_margin = psi_margin,
                    psi_floor = psi_floor,
                    psi_derivative = pf_psi_derivative,
                    atol = atol,

                    gurobi_output_flag = pf_gurobi_output_flag,
                    gurobi_opttol = pf_gurobi_opttol,
                    gurobi_feastol = pf_gurobi_feastol,

                    diagnostics = false,
                    verbose = false,
                )
        end


        # ========================================================
        # Run 2: instrumented run
        #
        # diagnostics=true so we can collect the time spent
        # specifically in the theta PO and q/w LMO operators.
        # ========================================================

        GC.gc()

        result_diagnostics =
            calibrate_upsilon_projection_free_ddfactplus(
                Csym,
                s,
                t;

                algorithm = :po_lmo,
                lmo_q_method = w_solver,

                J1 = Int[],
                J0 = Int[],

                theta0 = copy(theta0),

                max_iter = pf_max_iter,
                min_iter = pf_min_iter,
                iteration_power = pf_iteration_power,

                theta_bound = pf_theta_bound,

                tau0 = pf_tau0,
                tau_power = pf_tau_power,

                beta0 = pf_beta0,
                beta_power = pf_beta_power,

                Lqq_hat = pf_Lqq_hat,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = pf_psi_derivative,
                atol = atol,

                gurobi_output_flag = pf_gurobi_output_flag,
                gurobi_opttol = pf_gurobi_opttol,
                gurobi_feastol = pf_gurobi_feastol,

                diagnostics = true,
                verbose = false,
            )


        # ========================================================
        # Dual gap
        #
        # Use the instrumented run so the dual gap and operator
        # measurements correspond to the same trajectory.
        # ========================================================

        dual_gap = compute_dual_gap(
            Csym,
            result_diagnostics,
            z_ls,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
        )


        # ========================================================
        # Total operator runtimes
        # ========================================================

        theta_operator_runtime =
            sum(
                h.theta_operator_runtime
                for h in result_diagnostics.history
            )

        q_operator_runtime =
            sum(
                h.q_operator_runtime
                for h in result_diagnostics.history
            )


        # ========================================================
        # Save one row per (n,s,t,w_solver)
        # ========================================================

        push!(
            rows,
            Dict{Symbol,Any}(
                :n => n,
                :s => s,
                :t => t,
                :w_solver => string(w_solver),

                :dual_gap => dual_gap,

                :theta_operator_runtime =>
                    theta_operator_runtime,

                :q_operator_runtime =>
                    q_operator_runtime,

                :total_runtime =>
                    total_runtime,
            ),
        )


        # ========================================================
        # Console output
        # ========================================================

        @printf(
            "    dual_gap = % .6e\n",
            dual_gap,
        )

        @printf(
            "    theta operator = %.6fs\n",
            theta_operator_runtime,
        )

        @printf(
            "    q operator     = %.6fs\n",
            q_operator_runtime,
        )

        @printf(
            "    total runtime  = %.6fs\n",
            total_runtime,
        )

        @printf(
            "    operators / total = %.2f%%\n",
            100.0 *
            (
                theta_operator_runtime +
                q_operator_runtime
            ) /
            total_runtime,
        )

        println()
        flush(stdout)
    end
end


# ============================================================
# Save results
# ============================================================

df_results = DataFrame(rows)

column_order = [
    :n,
    :s,
    :t,
    :w_solver,
    :dual_gap,
    :theta_operator_runtime,
    :q_operator_runtime,
    :total_runtime,
]

df_results = df_results[:, column_order]

CSV.write(results_filepath, df_results)

println()
println("Saved results to: $results_filepath")