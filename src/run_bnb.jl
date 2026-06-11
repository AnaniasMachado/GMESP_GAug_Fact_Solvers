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
include("bnb.jl")


# -------------------------
# Problem data
# -------------------------
n = 63
kappa = 1

# Full run:
# s_vals = [s for s in (kappa + 1):(n - 1)]

# Test run:
s_vals = [s for s in (kappa + 1):(kappa + 10)]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)

k = 32
C = C[1:k, 1:k]
n = copy(k)

Csym = Symmetric(C)

atol = 1e-10

# Initial unscaled psi for DDGFact+
# If psi = nothing, solve_bnb_ddfact will choose psi at each node.
psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


# ============================================================
# Choose B&B relaxation
# ============================================================
# Options:
#   DDGFact
#   DDGFactplus
#   DDGFactplusUpsilon
# ============================================================
# relaxation = DDGFact
# relaxation = DDGFactplus
relaxation = DDGFactplusUpsilon


# ============================================================
# Choose variable fixing rule
# ============================================================
# Options:
#   :dual
#   :primal
#   :both
#
# For DDGFact, only :dual is supported.
# For DDGFactplus and DDGFactplusUpsilon, all three options are supported.
# ============================================================
fixing_rule = :dual
# fixing_rule = :primal
# fixing_rule = :both


# ============================================================
# Choose active BFGS parameter set
# ============================================================
# active_bfgs_param_set = :default
active_bfgs_param_set = :fast
# active_bfgs_param_set = :very_fast

# ============================================================
# Choose B&B parameters
# ============================================================
time_limit = 600.0
verbose_bnb = false

# Options:
#   :simple
#   :strong
#
# This is used only when relaxation = DDGFactplusUpsilon and
# fixing_rule is either :dual or :both.
upsilon_fixing = :strong


# -------------------------
# Data Collection
# -------------------------
solver = "knitro"
calib_method = "bfgs"
bfgs_param_set = String(active_bfgs_param_set)
relaxation_name = String(Symbol(relaxation))
fixing_rule_name = String(fixing_rule)
upsilon_fixing_name = String(upsilon_fixing)

mkpath("results")

results_filepath =
    "results/results_bnb_$(solver)_$(relaxation_name)_n$(n)_kappa$(kappa).csv"

results = []

cols = [
    :n,
    :s,
    :t,

    # B&B relaxation and variable fixing choices
    :relaxation,
    :fixing_rule,
    :bfgs_param_set,
    :upsilon_fixing,

    # B&B gaps
    :bnb_gap,
    :bnb_root_gap,

    # Runtimes
    :bnb_runtime,
    :bnb_reported_wall_time,

    # B&B variable fixing counts
    :n_fixed_zero_bnb,
    :n_fixed_one_bnb,
    :n_fixed_total_bnb,

    # B&B tree diagnostics
    :bnb_nodes,
    :bnb_n_int_sols,
    :bnb_tree_exhausted,
    :bnb_time_limit_hit,

    # B&B integrality-gap diagnostics
    :bnb_int_gap_max,
    :bnb_int_gap_avg,
    :bnb_int_gap_opt,

    # Objective values
    :z_bnb,
    :z_bnb_ub,
    :z_bnb_root_ub,
]

# Create CSV with header before the loop.
CSV.write(results_filepath, DataFrame([c => Any[] for c in cols]))

println("="^82)
println("GMESP branch-and-bound experiment")
println("n:                 $n")
println("kappa = s - t:     $kappa")
println("s values:          $s_vals")
println("relaxation:        $relaxation_name")
println("fixing rule:       $fixing_rule")
println("solver:            $solver")
println("time limit:        $time_limit")
println("BFGS param set:    $active_bfgs_param_set")
println("Upsilon fixing:    $upsilon_fixing")
println("results filepath:  $results_filepath")
println("="^82)


for s in s_vals
    t = s - kappa
    println("--------------------")
    println("s: $s")
    println("t: $t")
    println("relaxation: $relaxation_name")
    println("fixing_rule: $fixing_rule")
    flush(stdout)

    result = []
    append!(
        result,
        [
            n,
            s,
            t,
            relaxation_name,
            fixing_rule_name,
            bfgs_param_set,
            upsilon_fixing_name,
        ],
    )

    # -------------------------
    # B&B with selected relaxation
    # -------------------------
    Random.seed!(1)

    S_best = Int[]
    st = nothing

    runtime_bnb = @elapsed begin
        S_best, st = solve_bnb_ddfact(
            Csym,
            s,
            t;
            relaxation = relaxation,
            fixing_rule = fixing_rule,
            psi = psi,
            time_limit = time_limit,
            verbose = verbose_bnb,
            bfgs_param_set = active_bfgs_param_set,
            bfgs_param_sets = bfgs_param_sets,
            upsilon_fixing = upsilon_fixing,
            atol = atol,
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        )
    end

    n_fixed_total_bnb = st.nfix0 + st.nfix1
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

            # B&B variable fixing counts
            st.nfix0,
            st.nfix1,
            n_fixed_total_bnb,

            # B&B tree diagnostics
            st.nodes,
            st.n_int_sols,
            st.tree_exhausted,
            st.time_limit_hit,

            # B&B integrality-gap diagnostics
            st.int_gap_max,
            st.int_gap_avg,
            st.int_gap_opt,

            # Objective values
            st.lb,
            st.ub,
            st.root_ub,
        ],
    )

    push!(results, result)

    println("bnb_gap:                  ", st.gap)
    println("bnb_root_gap:             ", bnb_root_gap)
    println("z_bnb:                    ", st.lb)
    println("z_bnb_ub:                 ", st.ub)
    println("z_bnb_root_ub:            ", st.root_ub)

    println("bnb_nodes:                ", st.nodes)
    println("n_fixed_zero_bnb:         ", st.nfix0)
    println("n_fixed_one_bnb:          ", st.nfix1)
    println("n_fixed_total_bnb:        ", n_fixed_total_bnb)

    println("bnb_n_int_sols:           ", st.n_int_sols)
    println("bnb_int_gap_max:          ", st.int_gap_max)
    println("bnb_int_gap_avg:          ", st.int_gap_avg)
    println("bnb_int_gap_opt:          ", st.int_gap_opt)

    println("runtime_bnb:              ", runtime_bnb)
    println("bnb_reported_wall_time:   ", st.wall_time)
    println("tree_exhausted:           ", st.tree_exhausted)
    println("time_limit_hit:           ", st.time_limit_hit)

    @printf(
        "s=%2d  t=%2d  root_gap=%9.4f  gap=%9.4e  nodes=%7d  fix0=%5d  fix1=%5d  wall=%8.2fs%s\n",
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