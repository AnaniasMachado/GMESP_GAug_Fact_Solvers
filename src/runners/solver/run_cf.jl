
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

data_n = 60
t = 2

s_values = [5, 10, 15, 20, 25]

glinx_tol = 1e-2
atol = 1e-10

blas_threads = 16
MKL.set_num_threads(blas_threads)

println("MKL threads requested: ", blas_threads)


# ============================================================
# Problem data
# ============================================================

matfile = matopen(
    "data/generated/rank3/data$(data_n)_rank3.mat"
)

C_data = read(matfile, "C")
close(matfile)

C = Symmetric(Matrix{Float64}(C_data))
n = size(C, 1)

psi = eigmin(C) - atol

println("data_n = ", data_n)
println("n      = ", n)
println("t      = ", t)
println("psi    = ", psi)


# ============================================================
# Output
# ============================================================

results_dir = "results/journal_ddgfactplus"
scalings_dir = "glinx_scalings"

mkpath(results_dir)
mkpath(scalings_dir)

results_filepath = joinpath(
    results_dir,
    "results_root_gap_data$(data_n)_n$(n)_t$(t)_rank3.csv"
)

cols = [
    :data_n,
    :n,
    :s,
    :t,
    :psi,

    # Feasible objective
    :ls_obj,

    # Root upper bounds
    :ddgfact_plus_obj,
    :glinx_obj,
    :trace_obj,

    # Root gaps relative to local search
    :ddgfact_plus_gap,
    :glinx_gap,
    :trace_gap,

    # GLinx certificate and scaling
    :glinx_dual_obj,
    :glinx_duality_gap,
    :glinx_gamma,

    # Computation times
    :ddgfact_plus_runtime,
    :glinx_scaling_runtime,
    :glinx_runtime,
    :trace_runtime,
]

# Initialize CSV so completed runs are saved immediately.
CSV.write(
    results_filepath,
    DataFrame([c => Any[] for c in cols]),
)


# ============================================================
# Warm-up
# ============================================================

println()
println("Warming up solvers...")

warm_n = 9
warm_s = 7
warm_t = 5

C_warm = Symmetric(
    Matrix(C[1:warm_n, 1:warm_n])
)

psi_warm = eigmin(C_warm) - atol

aug_ddfact_gmesp(
    C_warm,
    warm_s,
    warm_t,
    psi_warm;
    atol = atol,
)

gamma_warm = compute_gamma_linx_armijo(
    C_warm,
    warm_s,
    warm_t;
    tol = glinx_tol,
)

linx_gmesp(
    C_warm,
    warm_s,
    warm_t;
    gamma = gamma_warm,
    return_model = true,
)

trace_upper_bound(
    C_warm,
    warm_s,
    warm_t,
    psi_warm,
)

GC.gc()

println("Warm-up complete.")


# ============================================================
# Root-node experiments
# ============================================================

results = []

for s in s_values

    println()
    println("="^82)
    println("GMESP root-node comparison")
    println("data_n:  $data_n")
    println("n:       $n")
    println("s:       $s")
    println("t:       $t")
    println("psi:     $psi")
    println("="^82)
    flush(stdout)

    Random.seed!(1)

    # --------------------------------------------------------
    # DDGFact+
    # --------------------------------------------------------

    println("Solving DDGFact+...")
    flush(stdout)

    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, z_ddgfact_plus =
            aug_ddfact_gmesp(
                C,
                s,
                t,
                psi;
                atol = atol,
            )
    end


    # --------------------------------------------------------
    # GLinx scaling
    # --------------------------------------------------------

    println("Computing GLinx scaling...")
    flush(stdout)

    runtime_glinx_scaling = @elapsed begin
        gamma_opt = compute_gamma_linx_armijo(
            C,
            s,
            t;
            tol = glinx_tol,
        )
    end

    println("Optimal gamma = ", gamma_opt)

    scaling_filepath = joinpath(
        scalings_dir,
        "glinx_scaling_n$(n)_s$(s)_t$(t)_rank3.mat"
    )

    scaling_file = matopen(scaling_filepath, "w")
    write(scaling_file, "gamma", gamma_opt)
    close(scaling_file)

    println("Saved GLinx scaling to: ", scaling_filepath)


    # --------------------------------------------------------
    # GLinx relaxation
    # --------------------------------------------------------

    println("Solving GLinx...")
    flush(stdout)

    GC.gc()

    runtime_glinx = @elapsed begin
        x_glinx, X_glinx, z_glinx, model_glinx =
            linx_gmesp(
                C,
                s,
                t;
                gamma = gamma_opt,
                return_model = true,
            )
    end

    cert_glinx = compute_glinx_dual_gap_v2(
        model_glinx,
        Vector{Float64}(x_glinx),
        Matrix{Float64}(X_glinx),
        Matrix(C),
        Float64(gamma_opt),
        s,
        t,
    )

    glinx_dual_obj = cert_glinx.dual_obj
    glinx_duality_gap = cert_glinx.gap


    # --------------------------------------------------------
    # Trace upper bound
    # --------------------------------------------------------

    println("Computing Trace upper bound...")
    flush(stdout)

    runtime_trace = @elapsed begin
        z_trace = trace_upper_bound(
            C,
            s,
            t,
            psi,
        )
    end


    # --------------------------------------------------------
    # Local search
    # --------------------------------------------------------

    println("Running local search...")
    flush(stdout)

    x_ls, z_ls = run_all_LS(C, s, t)


    # --------------------------------------------------------
    # Root gaps
    # --------------------------------------------------------

    ddgfact_plus_gap = z_ddgfact_plus - z_ls
    glinx_gap = z_glinx - z_ls
    trace_gap = z_trace - z_ls


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    row = [
        data_n,
        n,
        s,
        t,
        psi,

        z_ls,

        z_ddgfact_plus,
        z_glinx,
        z_trace,

        ddgfact_plus_gap,
        glinx_gap,
        trace_gap,

        glinx_dual_obj,
        glinx_duality_gap,
        gamma_opt,

        runtime_ddgfact_plus,
        runtime_glinx_scaling,
        runtime_glinx,
        runtime_trace,
    ]

    push!(results, row)

    row_df = DataFrame(
        [cols[j] => [row[j]] for j in eachindex(cols)]
    )

    CSV.write(
        results_filepath,
        row_df;
        append = true,
    )


    # --------------------------------------------------------
    # Progress information
    # --------------------------------------------------------

    println()
    println("Root-node results")
    println("-"^82)

    println("z_ls                  = ", z_ls)
    println("z_ddgfact_plus        = ", z_ddgfact_plus)
    println("z_glinx               = ", z_glinx)
    println("z_trace               = ", z_trace)

    println()
    println("DDGFact+ gap          = ", ddgfact_plus_gap)
    println("GLinx gap             = ", glinx_gap)
    println("Trace gap             = ", trace_gap)

    println()
    println("DDGFact+ time         = ", runtime_ddgfact_plus)
    println("GLinx scaling time    = ", runtime_glinx_scaling)
    println("GLinx solve time      = ", runtime_glinx)
    println("Trace time            = ", runtime_trace)

    println()
    println("GLinx dual objective  = ", glinx_dual_obj)
    println("GLinx duality gap     = ", glinx_duality_gap)
    println("GLinx gamma           = ", gamma_opt)

    println()
    println("Appended row to: ", results_filepath)
    flush(stdout)
end


# ============================================================
# Summary
# ============================================================

println()
println("="^82)
println("Root-node experiment complete.")
println("Results saved to: ", results_filepath)
println("="^82)
