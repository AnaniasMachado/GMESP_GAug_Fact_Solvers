using MKL
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Hypatia
import MathOptInterface as MOI

include("../../misc/dual_glinx.jl")
include("../../gscaling/gscaling_glinx.jl")


# ============================================================
# Experiment parameters
# ============================================================

n = 63
kappa = 2

s_vals = [s for s in 10:10:(n - 1) if s >= kappa + 1]

blas_threads = 16
MKL.set_num_threads(blas_threads)
println("BLAS backend = ", BLAS.get_config())
println("BLAS threads = ", BLAS.get_num_threads())
println("MKL threads = ", MKL.get_num_threads())


# ============================================================
# Problem data
# ============================================================

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)


# ============================================================
# Input/output folders
# ============================================================

# Replace this with the folder containing your saved scaling MAT files.
scalings_dir = "glinx_scalings_old"

results_filepath = "results/journal_ddgfactplus/glinx_fixed_scalings_n$(n)_kappa$(kappa).csv"

mkpath(dirname(results_filepath))

results = []


# ============================================================
# Warm-up
# ============================================================

println("Warming up GLinx...")

s_warmup = first(s_vals)
t_warmup = s_warmup - kappa
scaling_filepath_warmup = joinpath(scalings_dir, "glinx_scaling_n$(n)_s$(s_warmup)_kappa$(kappa).mat")

scaling_file_warmup = matopen(scaling_filepath_warmup)
gamma_warmup = Float64(read(scaling_file_warmup, "gamma"))
close(scaling_file_warmup)

linx_gmesp(Csym,s_warmup,t_warmup;gamma=gamma_warmup,return_model=true)

GC.gc()

println("Warm-up complete.")


# ============================================================
# Fixed-scaling GLinx experiments
# ============================================================

for s in s_vals

    t = s - kappa

    println()
    println("============================================================")
    println("n = $n, kappa = $kappa, s = $s, t = $t")
    println("============================================================")

    result = []

    append!(result, [n, kappa, s, t])


    # --------------------------------------------------------
    # Read saved GLinx scaling
    # --------------------------------------------------------

    scaling_filepath = joinpath(scalings_dir, "glinx_scaling_n$(n)_s$(s)_kappa$(kappa).mat")

    println("Reading scaling from: $scaling_filepath")

    scaling_file = matopen(scaling_filepath)
    gamma_opt = read(scaling_file, "gamma")
    close(scaling_file)

    gamma_opt = Float64(gamma_opt)

    println("gamma = $gamma_opt")


    # --------------------------------------------------------
    # Solve GLinx at fixed gamma
    # --------------------------------------------------------

    println("Solving GLinx...")

    GC.gc()

    runtime_glinx = @elapsed begin
        x_glinx, X_glinx, z_glinx, model_glinx = linx_gmesp(Csym,s,t;gamma=gamma_opt,return_model=true)
    end


    # --------------------------------------------------------
    # Compute GLinx dual statistics
    # --------------------------------------------------------

    cert_glinx = compute_glinx_dual_gap_v2(model_glinx,Vector{Float64}(x_glinx),Matrix{Float64}(X_glinx),Matrix(Csym),Float64(gamma_opt),s,t)

    glinx_primal_obj = cert_glinx.primal_obj
    glinx_dual_obj = cert_glinx.dual_obj
    glinx_duality_gap = cert_glinx.gap
    glinx_dual_feasible = cert_glinx.dual_feasible
    glinx_res_x = cert_glinx.res_x
    glinx_res_mat = cert_glinx.res_mat
    glinx_min_eig_Z = cert_glinx.min_eig_Z
    glinx_min_eig_Omega = cert_glinx.min_eig_Omega
    glinx_min_eig_Theta = cert_glinx.min_eig_Theta


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    append!(
        result,
        [
            gamma_opt,
            z_glinx,
            glinx_primal_obj,
            glinx_dual_obj,
            glinx_duality_gap,
            glinx_dual_feasible,
            glinx_res_x,
            glinx_res_mat,
            glinx_min_eig_Z,
            glinx_min_eig_Omega,
            glinx_min_eig_Theta,
            runtime_glinx,
        ],
    )

    push!(results, result)


    # --------------------------------------------------------
    # Progress information
    # --------------------------------------------------------

    println()
    println("gamma                 = $gamma_opt")
    println("z_glinx               = $z_glinx")
    println("glinx_primal_obj      = $glinx_primal_obj")
    println("glinx_dual_obj        = $glinx_dual_obj")
    println("glinx_duality_gap     = $glinx_duality_gap")
    println("glinx_dual_feasible   = $glinx_dual_feasible")
    println("glinx_res_x           = $glinx_res_x")
    println("glinx_res_mat         = $glinx_res_mat")
    println("glinx_min_eig_Z       = $glinx_min_eig_Z")
    println("glinx_min_eig_Omega   = $glinx_min_eig_Omega")
    println("glinx_min_eig_Theta   = $glinx_min_eig_Theta")
    println("glinx_runtime         = $runtime_glinx")
end


# ============================================================
# Construct DataFrame
# ============================================================

results_matrix = hcat(results...)'

cols = [
    :n,
    :kappa,
    :s,
    :t,
    :glinx_gamma,
    :glinx_obj,
    :glinx_primal_obj,
    :glinx_dual_obj,
    :glinx_duality_gap,
    :glinx_dual_feasible,
    :glinx_res_x,
    :glinx_res_mat,
    :glinx_min_eig_Z,
    :glinx_min_eig_Omega,
    :glinx_min_eig_Theta,
    :glinx_runtime,
]

df = DataFrame(results_matrix, cols)


# ============================================================
# Save results
# ============================================================

CSV.write(results_filepath, df)

println()
println("Saved results to: $results_filepath")