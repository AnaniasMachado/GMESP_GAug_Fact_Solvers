using MKL
using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
using Hypatia
import MathOptInterface as MOI

include("../../misc/util.jl")
include("../../misc/dual_glinx.jl")
include("../../misc/heuristics.jl")
include("../../solvers/solver_knitro.jl")
include("../../gscaling/gscaling_glinx.jl")


# ============================================================
# Experiment parameters
# ============================================================

n = 63
t = 2
idx = "1"

s_vals = [10, 15, 20, 25, 30]

glinx_tol = 1e-2

blas_threads = 16
MKL.set_num_threads(blas_threads)


# ============================================================
# Problem data
# ============================================================

matfile = matopen("data/generated/data$(n)_inst$(idx).mat")
# C = n == 63 ? read(matfile, "A") : read(matfile, "C")
C = read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# Choose psi < lambda_min(C)
psi = eigmin(Csym) - atol


# ============================================================
# Output
# ============================================================

results_filepath = "results/journal_ddgfactplus/results_root_gap_n$(n)_t$(t)_inst$(idx).csv"

scalings_dir = "glinx_scalings"

mkpath(dirname(results_filepath))
mkpath(scalings_dir)

results = []


# ============================================================
# Warm-up
# ============================================================

println("Warming up solvers...")

n_warmup = 9
s_warmup = 7
t_warmup = 5

C_warmup = Symmetric(Matrix(Csym[1:n_warmup, 1:n_warmup]))
psi_warmup = eigmin(C_warmup) - atol

ddfact_gmesp(C_warmup, s_warmup, t_warmup; atol=atol)
aug_ddfact_gmesp(C_warmup, s_warmup, t_warmup, psi_warmup; atol=atol)

gamma_warmup = compute_gamma_linx_armijo(C_warmup, s_warmup, t_warmup; tol=glinx_tol)
linx_gmesp(C_warmup, s_warmup, t_warmup; gamma=gamma_warmup, return_model=true)

GC.gc()

println("Warm-up complete.")


# ============================================================
# Root-node experiments
# ============================================================

for s in s_vals

    println()
    println("============================================================")
    println("n = $n, s = $s, t = $t")
    println("============================================================")

    result = []

    append!(result, [n, s, t])


    # --------------------------------------------------------
    # DDGFact
    # --------------------------------------------------------

    println("Solving DDGFact...")

    runtime_ddgfact = @elapsed begin
        x_ddgfact, z_ddgfact = ddfact_gmesp(Csym, s, t; atol=atol)
    end


    # --------------------------------------------------------
    # DDGFact+
    # --------------------------------------------------------

    println("Solving DDGFact+...")

    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(Csym, s, t, psi; atol=atol)
    end


    # --------------------------------------------------------
    # GLinx gamma
    # --------------------------------------------------------

    println("Computing optimal GLinx scaling...")

    gamma_opt = compute_gamma_linx_armijo(Csym, s, t; tol=glinx_tol)

    println("Optimal gamma = $gamma_opt")


    # --------------------------------------------------------
    # Save optimal GLinx scaling
    # --------------------------------------------------------

    scaling_filepath = joinpath(scalings_dir, "glinx_scaling_n$(n)_s$(s)_t$(t)_inst$(idx).mat")

    scaling_file = matopen(scaling_filepath, "w")
    write(scaling_file, "gamma", gamma_opt)
    close(scaling_file)

    println("Saved optimal scaling to: $scaling_filepath")


    # --------------------------------------------------------
    # Solve GLinx using fixed optimal scaling
    # --------------------------------------------------------

    println("Solving GLinx at fixed optimal gamma...")

    GC.gc()

    runtime_glinx = @elapsed begin
        x_glinx, X_glinx, z_glinx, model_glinx = linx_gmesp(Csym, s, t; gamma=gamma_opt, return_model=true)
    end

    cert_glinx = compute_glinx_dual_gap_v2(model_glinx, Vector{Float64}(x_glinx), Matrix{Float64}(X_glinx), Matrix(Csym), Float64(gamma_opt), s, t)

    glinx_dual_obj = cert_glinx.dual_obj
    glinx_duality_gap = cert_glinx.gap


    # --------------------------------------------------------
    # Local search
    # --------------------------------------------------------

    println("Running local search...")

    x_ls, z_ls = run_all_LS(Csym, s, t)


    # --------------------------------------------------------
    # Spectral bound
    # --------------------------------------------------------

    println("Computing spectral bound...")

    z_spec = spectral_bound(Csym, t)


    # --------------------------------------------------------
    # Root gaps
    # --------------------------------------------------------

    ddgfact_gap = z_ddgfact - z_ls
    ddgfact_plus_gap = z_ddgfact_plus - z_ls
    glinx_gap = z_glinx - z_ls
    spec_gap = z_spec - z_ls


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    append!(
        result,
        [
            z_ls,
            z_ddgfact,
            z_ddgfact_plus,
            z_glinx,
            z_spec,
            ddgfact_gap,
            ddgfact_plus_gap,
            glinx_gap,
            spec_gap,
            glinx_dual_obj,
            glinx_duality_gap,
            gamma_opt,
            runtime_ddgfact,
            runtime_ddgfact_plus,
            runtime_glinx,
        ],
    )

    push!(results, result)


    # --------------------------------------------------------
    # Progress information
    # --------------------------------------------------------

    println()
    println("z_ls              = $z_ls")
    println("z_ddgfact         = $z_ddgfact")
    println("z_ddgfact_plus    = $z_ddgfact_plus")
    println("z_glinx           = $z_glinx")
    println("z_spec            = $z_spec")
    println()
    println("DDGFact gap       = $ddgfact_gap")
    println("DDGFact+ gap      = $ddgfact_plus_gap")
    println("GLinx gap         = $glinx_gap")
    println("Spectral gap      = $spec_gap")
    println()
    println("DDGFact time      = $runtime_ddgfact")
    println("DDGFact+ time     = $runtime_ddgfact_plus")
    println("GLinx time        = $runtime_glinx")
    println("gamma_opt         = $gamma_opt")
end


# ============================================================
# Construct DataFrame
# ============================================================

results_matrix = hcat(results...)'

cols = [
    :n,
    :s,
    :t,

    # Objective values
    :ls_obj,
    :ddgfact_obj,
    :ddgfact_plus_obj,
    :glinx_obj,
    :spec_obj,

    # Root gaps
    :ddgfact_gap,
    :ddgfact_plus_gap,
    :glinx_gap,
    :spec_gap,

    # GLinx scaling
    :glinx_dual_obj,
    :glinx_duality_gap,
    :glinx_gamma,

    # Relaxation solution times
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
    :glinx_runtime,
]

df = DataFrame(results_matrix, cols)


# ============================================================
# Save results
# ============================================================

CSV.write(results_filepath, df)

println()
println("Saved results to: $results_filepath")