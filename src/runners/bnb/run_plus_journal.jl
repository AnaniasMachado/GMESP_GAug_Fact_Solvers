using MKL
using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
using Printf
import MathOptInterface as MOI

include("../../misc/util.jl")
include("../../misc/heuristics.jl")
include("../../solvers/solver_knitro.jl")

include("../../misc/dual.jl")
include("../../misc/var_fixing.jl")

include("../../bnb/bnb_util.jl")
include("../../bnb/bnb_general.jl")


# ============================================================
# Full instance
# ============================================================

data_n = 60
k = 60

s_values = [5, 10, 15, 20, 25]
t = 2

matfile = matopen("data/generated/rank3/data$(data_n)_rank3.mat")
C_data = read(matfile, "C")
close(matfile)

C_data = Matrix{Float64}(C_data)

C = Symmetric(C_data[1:k, 1:k])
n = size(C, 1)


# ============================================================
# B&B parameters
# ============================================================

Random.seed!(1)

time_limit = 2 * 3600.0
verbose_bnb = false
atol = 1e-10

fixing_rule = :dual

psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


# ============================================================
# Threading
# ============================================================

blas_threads = 16
MKL.set_num_threads(blas_threads)

println("MKL threads requested: ", blas_threads)


# ============================================================
# Runs
# ============================================================

runs = [
    (
        name = "Trace",
        relaxation = TraceBound,
        plus_cf = false,
    ),
    # (
    #     name = "DDGFact",
    #     relaxation = DDGFact,
    #     plus_cf = false,
    # ),
    (
        name = "DDGFactplus",
        relaxation = DDGFactplus,
        plus_cf = false,
    ),
    (
        name = "DDGFactplus_CF",
        relaxation = DDGFactplus,
        plus_cf = true,
    ),
]


# ============================================================
# Warm-up on smaller principal submatrix
# ============================================================

warm_n = 9
warm_s = 7
warm_t = 5

C_warm = Symmetric(Matrix(C[1:warm_n, 1:warm_n]))

println("="^82)
println("Warming up B&B solvers...")
println("n = $warm_n, s = $warm_s, t = $warm_t")
println("="^82)
flush(stdout)

for run in runs
    println("Warming up $(run.name)...")
    flush(stdout)

    Random.seed!(1)

    solve_bnb_ddfact(
        C_warm,
        warm_s,
        warm_t;
        relaxation = run.relaxation,
        fixing_rule = run.relaxation == TraceBound ? :none : fixing_rule,
        psi = psi,
        plus_cf = run.plus_cf,
        time_limit = time_limit,
        verbose = false,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )
end

GC.gc()

println("Warm-up complete.")
println()
flush(stdout)


# ============================================================
# CSV output
# ============================================================

mkpath("results/journal_ddgfactplus")

# results_filepath =
#     "results/journal_ddgfactplus/bnb_data$(data_n)_n$(n)_s$(s)_t$(t)_rank3_trace.csv"

results_filepath =
    "results/journal_ddgfactplus/bnb_data$(data_n)_n$(n)_t$(t)_rank3.csv"

cols = [
    :data_n,
    :n,
    :s,
    :t,

    :relaxation,
    :plus_cf,
    :fixing_rule,

    :bnb_gap,
    :bnb_root_gap,

    :bnb_runtime,
    :bnb_reported_wall_time,
    :bnb_knitro_time,
    :bnb_relaxation_solve_time,
    :bnb_factorization_time,
    :bnb_bound_computation_time,
    :bnb_open_list_time,
    :bnb_node_setup_time,
    :bnb_dual_solution_time,
    :bnb_variable_fixing_direct_time,
    :bnb_variable_fixing_time,
    :bnb_variable_fixing_calls,

    :closed_form_uses,
    :closed_form_time,
    :trace_prunes,

    :n_fixed_zero_bnb,
    :n_fixed_one_bnb,
    :n_fixed_total_bnb,

    :bnb_nodes,
    :bnb_n_int_sols,
    :bnb_tree_exhausted,
    :bnb_time_limit_hit,

    :bnb_int_gap_max,
    :bnb_int_gap_avg,
    :bnb_int_gap_opt,

    :z_bnb,
    :z_bnb_ub,
    :z_bnb_root_ub,
]

if !isfile(results_filepath)
    CSV.write(
        results_filepath,
        DataFrame([c => Any[] for c in cols]),
    )
end


# ============================================================
# Full experiment
# ============================================================

results = []

for s in s_values

    println("="^82)
    println("GMESP B&B comparison")
    println("data_n:       $data_n")
    println("n:            $n")
    println("s:            $s")
    println("t:            $t")
    println("fixing_rule:  $fixing_rule")
    println("time_limit:   $time_limit")
    println("output:       $results_filepath")
    println("="^82)
    flush(stdout)

    for run in runs
        println()
        println("-"^82)
        println("Running:       $(run.name)")
        println("relaxation:    $(run.relaxation)")
        println("plus_cf:       $(run.plus_cf)")
        println("s:            $s")
        println("-"^82)
        flush(stdout)

        Random.seed!(1)

        S_best = Int[]
        st = nothing

        runtime = @elapsed begin
            S_best, st = solve_bnb_ddfact(
                C,
                s,
                t;
                relaxation = run.relaxation,
                fixing_rule = run.relaxation == TraceBound ? :none : fixing_rule,
                psi = psi,
                plus_cf = run.plus_cf,
                time_limit = time_limit,
                verbose = verbose_bnb,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )
        end

        n_fixed_zero_bnb = st.nfix0
        n_fixed_one_bnb = st.nfix1
        n_fixed_total_bnb =
            n_fixed_zero_bnb + n_fixed_one_bnb

        bnb_root_gap = st.root_ub - st.lb

        row = [
            data_n,
            n,
            s,
            t,

            run.name,
            run.plus_cf,
            String(st.fixing_rule),

            st.gap,
            bnb_root_gap,

            runtime,
            st.wall_time,
            st.knitro_time,
            st.relaxation_solve_time,
            st.factorization_time,
            st.bound_computation_time,
            st.open_list_time,
            st.node_setup_time,
            st.dual_solution_time,
            st.variable_fixing_direct_time,
            st.variable_fixing_time,
            st.variable_fixing_calls,

            st.closed_form_uses,
            st.closed_form_time,
            st.trace_prunes,

            n_fixed_zero_bnb,
            n_fixed_one_bnb,
            n_fixed_total_bnb,

            st.nodes,
            st.n_int_sols,
            st.tree_exhausted,
            st.time_limit_hit,

            st.int_gap_max,
            st.int_gap_avg,
            st.int_gap_opt,

            st.lb,
            st.ub,
            st.root_ub,
        ]

        push!(
            results,
            (
                name = run.name,
                relaxation = run.relaxation,
                plus_cf = run.plus_cf,
                s = s,
                S_best = S_best,
                st = st,
                runtime = runtime,
                row = row,
            ),
        )

        row_df = DataFrame(
            [Symbol(cols[j]) => [row[j]] for j in eachindex(cols)]
        )

        CSV.write(
            results_filepath,
            row_df;
            append = true,
        )

        println()
        println("B&B result")
        println("name:                    ", run.name)
        println("s:                       ", s)
        println("S_best:                  ", S_best)
        println("obj / lb:                ", st.lb)
        println("ub:                      ", st.ub)
        println("gap:                     ", st.gap)
        println("root_ub:                 ", st.root_ub)
        println("nodes:                   ", st.nodes)
        println("closed_form_uses:        ", st.closed_form_uses)
        println("closed_form_time:        ", st.closed_form_time)
        println("trace_prunes:            ", st.trace_prunes)
        println("wall_time:               ", st.wall_time)
        println("runtime measured:        ", runtime)
        println("knitro_time:             ", st.knitro_time)
        println("relaxation_solve_time:   ", st.relaxation_solve_time)
        println("factorization_time:      ", st.factorization_time)
        println("dual_solution_time:      ", st.dual_solution_time)
        println("variable_fixing_time:    ", st.variable_fixing_time)
        println("nfix0:                   ", st.nfix0)
        println("nfix1:                   ", st.nfix1)
        println("tree_exhausted:          ", st.tree_exhausted)
        println("time_limit_hit:          ", st.time_limit_hit)
        println("Appended row to:         ", results_filepath)
        flush(stdout)
    end
end
