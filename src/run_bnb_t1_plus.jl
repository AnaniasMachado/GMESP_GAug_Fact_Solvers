using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using JuMP
using KNITRO
using Printf
import MathOptInterface as MOI

include("util.jl")
include("heuristics.jl")
include("solver_knitro.jl")

include("gscaling_util.jl")
include("gscaling_bfgs.jl")
include("gscaling_t1.jl")
include("gscaling_params.jl")

include("dual.jl")
include("var_fixing.jl")
include("bnb_t1_plus.jl")


# -------------------------
# Problem data
# -------------------------
n = 63

# For t = 1, we vary s directly.
# Full run:
# s_vals = [s for s in 2:(n - 1)]

# Test run:
s_vals = [s for s in 2:12]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)

k = 32
C = C[1:k, 1:k]
n = copy(k)

Csym = Symmetric(C)

atol = 1e-10
t = 1

# Initial unscaled psi for DDGFact+
# If psi = nothing, solve_bnb_ddgfactplus_t1 will choose psi once at the root.
psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


# ============================================================
# B&B relaxation
# ============================================================
# Specialized B&B for DDGFactplus with t = 1.
# ============================================================
relaxation_name = "DDGFactplus_t1"


# ============================================================
# Choose B&B parameters
# ============================================================
time_limit = 600.0
verbose_bnb = false


# -------------------------
# Data Collection
# -------------------------
solver = "closed_form"

mkpath("results")

results_filepath =
    "results/results_bnb_$(solver)_$(relaxation_name)_n$(n).csv"

results = []

cols = [
    :n,
    :s,
    :t,

    # B&B relaxation choice
    :relaxation,

    # B&B gaps
    :bnb_gap,
    :bnb_root_gap,

    # Runtimes
    :bnb_runtime,
    :bnb_reported_wall_time,

    # B&B branching counts
    :n_branch_zero_bnb,
    :n_branch_one_bnb,
    :n_branch_total_bnb,

    # B&B tree diagnostics
    :bnb_nodes,
    :bnb_tree_exhausted,
    :bnb_time_limit_hit,

    # Objective values
    :z_bnb,
    :z_bnb_ub,
    :z_bnb_root_ub,

    # DDGFactplus parameter
    :psi,
]

# Create CSV with header before the loop.
CSV.write(results_filepath, DataFrame([c => Any[] for c in cols]))

println("="^82)
println("GMESP branch-and-bound experiment")
println("n:                 $n")
println("t fixed at:        $t")
println("s values:          $s_vals")
println("relaxation:        $relaxation_name")
println("solver:            $solver")
println("time limit:        $time_limit")
println("results filepath:  $results_filepath")
println("="^82)
flush(stdout)


for s in s_vals
    println("--------------------")
    println("s: $s")
    println("t: $t")
    println("relaxation: $relaxation_name")
    flush(stdout)

    result = []
    append!(
        result,
        [
            n,
            s,
            t,
            relaxation_name,
        ],
    )

    # -------------------------
    # B&B with DDGFactplus t = 1 closed form
    # -------------------------
    Random.seed!(1)

    S_best = Int[]
    st = nothing

    runtime_bnb = @elapsed begin
        S_best, st = solve_bnb_ddgfactplus_t1(
            Csym,
            s;
            psi = psi,
            time_limit = time_limit,
            verbose = verbose_bnb,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        )
    end

    n_branch_total_bnb = st.nfix0 + st.nfix1
    bnb_root_gap = st.root_ub - st.lb

    # -------------------------
    # Collect gaps, runtimes, and B&B diagnostics
    # -------------------------
    append!(
        result,
        [
            # B&B gaps
            st.gap,
            bnb_root_gap,

            # Runtimes
            runtime_bnb,
            st.wall_time,

            # B&B branching counts
            st.nfix0,
            st.nfix1,
            n_branch_total_bnb,

            # B&B tree diagnostics
            st.nodes,
            st.tree_exhausted,
            st.time_limit_hit,

            # Objective values
            st.lb,
            st.ub,
            st.root_ub,

            # DDGFactplus parameter
            st.psi,
        ],
    )

    push!(results, result)

    println("bnb_gap:                  ", st.gap)
    println("bnb_root_gap:             ", bnb_root_gap)
    println("z_bnb:                    ", st.lb)
    println("z_bnb_ub:                 ", st.ub)
    println("z_bnb_root_ub:            ", st.root_ub)

    println("bnb_nodes:                ", st.nodes)
    println("n_branch_zero_bnb:        ", st.nfix0)
    println("n_branch_one_bnb:         ", st.nfix1)
    println("n_branch_total_bnb:       ", n_branch_total_bnb)

    println("runtime_bnb:              ", runtime_bnb)
    println("bnb_reported_wall_time:   ", st.wall_time)
    println("tree_exhausted:           ", st.tree_exhausted)
    println("time_limit_hit:           ", st.time_limit_hit)
    println("psi:                      ", st.psi)

    @printf(
        "s=%2d  t=%2d  root_gap=%9.4f  gap=%9.4e  nodes=%7d  branch0=%5d  branch1=%5d  wall=%8.2fs%s\n",
        s,
        t,
        bnb_root_gap,
        st.gap,
        st.nodes,
        st.nfix0,
        st.nfix1,
        st.wall_time,
        st.time_limit_hit ? "  [TIMEOUT]" : "",
    )

    flush(stdout)

    # -------------------------
    # Write one CSV row after this B&B instance finishes
    # -------------------------
    row_df = DataFrame(
        [Symbol(cols[j]) => [result[j]] for j in eachindex(cols)]
    )

    CSV.write(
        results_filepath,
        row_df;
        append = true,
    )

    println("Appended row to: $results_filepath")
    flush(stdout)
end


# -------------------------
# Final in-memory table
# -------------------------
df = DataFrame(
    [Symbol(cols[j]) => [r[j] for r in results] for j in eachindex(cols)]
)

println("Saved results to: $results_filepath")