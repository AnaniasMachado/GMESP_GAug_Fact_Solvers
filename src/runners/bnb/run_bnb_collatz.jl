using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using Printf

include("../misc/util.jl")
include("../misc/heuristics.jl")

include("../misc/dual.jl")
include("../misc/var_fixing.jl")

include("../bnb/bnb_collatz.jl")


# ============================================================
# Choose instance
# ============================================================
data_n = 63
k = 32

t = 1

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
C = Symmetric(C)

n = size(C, 1)


# ============================================================
# Parameters
# ============================================================
Random.seed!(1)

time_limit = 7200.0
# time_limit = 86400.0

verbose_bnb = false
report_every = 1000
atol = 1e-10

# Collatz scaling options:
#   :sqrt_diag_minus_psi
#   :perron
#   :min_of_both
scaling_rule = :min_of_both

# For :sqrt_diag_minus_psi and :min_of_both
psi = nothing
psi_margin = 1e-7
w_floor = 1e-10

# Options are accepted by the interface, but currently fixing is placeholder.
# Use :none for now.
#
#   :none
#   :dual
#   :primal
#   :both
fixing_rule = :none

# Collatz requires C >= 0 entrywise.
check_nonnegative = true


# ============================================================
# CSV output
# ============================================================
mkpath("results")

results_filepath =
    "results/bnb_t1_collatz_min_of_both_data$(data_n)_n$(n)_s2_to_$(n - 1)_heap.csv"

cols = [
    :data_n,
    :n,
    :s,
    :t,

    # Reformulation / solver choices
    :relaxation,
    :method,
    :solver_used,
    :fixing_rule,
    :scaling_rule,

    # Gaps
    :bnb_gap,
    :bnb_root_gap,

    # Runtimes
    :bnb_runtime,
    :bnb_reported_wall_time,
    :bnb_knitro_time,
    :bnb_relaxation_solve_time,
    :bnb_upsilon_calibration_time,
    :bnb_factorization_time,
    :bnb_scaling_setup_time,
    :bnb_bound_computation_time,
    :bnb_open_list_time,
    :bnb_node_setup_time,
    :bnb_dual_solution_time,
    :bnb_primal_solution_time,
    :bnb_variable_fixing_direct_time,
    :bnb_variable_fixing_time,
    :bnb_variable_fixing_calls,

    # Extra t=1 B&B timing diagnostics
    :bnb_true_obj_time,
    :bnb_incumbent_time,
    :bnb_branch_variable_time,

    # Branching counts
    :n_branch_zero_bnb,
    :n_branch_one_bnb,
    :n_branch_total_bnb,

    # True variable fixing counts
    :n_fixed_zero_bnb,
    :n_fixed_one_bnb,
    :n_fixed_total_bnb,

    # Tree diagnostics
    :bnb_nodes,
    :bnb_n_int_sols,
    :bnb_tree_exhausted,
    :bnb_time_limit_hit,

    # Integer-gap diagnostics
    :bnb_int_gap_max,
    :bnb_int_gap_avg,
    :bnb_int_gap_opt,

    # Objective values
    :z_bnb,
    :z_bnb_ub,
    :z_bnb_root_ub,

    # Collatz / scaling parameter
    :psi,

    # Best subset
    :S_best,
]

CSV.write(results_filepath, DataFrame([c => Any[] for c in cols]))


println("="^82)
println("GMESP specialized B&B test: Collatz, t = 1, fixed scaling")
println("data_n:                         $data_n")
println("k / n:                          $k / $n")
println("s range:                        2:$(n - 1)")
println("t:                              $t")
println("scaling_rule:                   $scaling_rule")
println("fixing_rule:                    $fixing_rule")
println("time_limit:                     $time_limit")
println("verbose_bnb:                    $verbose_bnb")
println("report_every:                   $report_every")
println("psi:                            $psi")
println("psi_margin:                     $psi_margin")
println("w_floor:                        $w_floor")
println("check_nonnegative:              $check_nonnegative")
println("results_filepath:               $results_filepath")
println("="^82)
flush(stdout)


# ============================================================
# Helpers
# ============================================================
function _get_stat(st, name::Symbol, default)
    return hasproperty(st, name) ? getproperty(st, name) : default
end

function _symbol_string_stat(st, name::Symbol, default::String)
    return hasproperty(st, name) ? String(Symbol(getproperty(st, name))) : default
end

function _make_row_df(cols::Vector{Symbol}, row::Vector{Any})
    if length(cols) != length(row)
        println("length(cols) = ", length(cols))
        println("length(row)  = ", length(row))

        for j in 1:max(length(cols), length(row))
            colj = j <= length(cols) ? string(cols[j]) : "<missing col>"
            rowj = j <= length(row) ? row[j] : "<missing row value>"
            println(rpad(string(j), 4), rpad(colj, 35), rowj)
        end

        error("CSV row/column length mismatch.")
    end

    return DataFrame([cols[j] => [row[j]] for j in eachindex(cols)])
end


# ============================================================
# Run t = 1 Collatz B&B for s = 2, ..., n - 1
# ============================================================
results = []

for s in 2:(n - 1)
    println()
    println("-"^82)
    println("Running Collatz t = 1 B&B")
    println("s:                              $s")
    println("scaling_rule:                   $scaling_rule")
    println("fixing_rule:                    $fixing_rule")
    flush(stdout)

    Random.seed!(1)

    S_best = Int[]
    st = nothing

    runtime = @elapsed begin
        S_best, st = solve_bnb_collatz_t1(
            C,
            s;
            psi = psi,
            scaling_rule = scaling_rule,
            fixing_rule = fixing_rule,
            time_limit = time_limit,
            verbose = verbose_bnb,
            report_every = report_every,
            atol = atol,
            psi_margin = psi_margin,
            w_floor = w_floor,
            check_nonnegative = check_nonnegative,
        )
    end

    relaxation_out =
        _symbol_string_stat(st, :relaxation, "Collatz")

    method_out =
        _symbol_string_stat(st, :method, "bnb_t1_collatz")

    solver_used_out =
        _symbol_string_stat(st, :solver_used, "t1_collatz_bnb")

    fixing_rule_out =
        _symbol_string_stat(st, :fixing_rule, String(fixing_rule))

    scaling_rule_out =
        _symbol_string_stat(st, :scaling_rule, String(scaling_rule))

    bnb_root_gap = st.root_ub - st.lb

    bnb_knitro_time =
        _get_stat(st, :knitro_time, 0.0)

    bnb_relaxation_solve_time =
        _get_stat(st, :relaxation_solve_time, missing)

    bnb_upsilon_calibration_time =
        _get_stat(st, :upsilon_calibration_time, 0.0)

    bnb_factorization_time =
        _get_stat(st, :factorization_time, 0.0)

    bnb_scaling_setup_time =
        _get_stat(st, :scaling_setup_time, 0.0)

    bnb_bound_computation_time =
        _get_stat(st, :bound_computation_time, missing)

    bnb_open_list_time =
        _get_stat(st, :open_list_time, missing)

    bnb_node_setup_time =
        _get_stat(st, :node_setup_time, missing)

    bnb_dual_solution_time =
        _get_stat(st, :dual_solution_time, 0.0)

    bnb_primal_solution_time =
        _get_stat(st, :primal_solution_time, 0.0)

    bnb_variable_fixing_direct_time =
        _get_stat(st, :variable_fixing_direct_time, 0.0)

    bnb_variable_fixing_time =
        _get_stat(st, :variable_fixing_time, 0.0)

    bnb_variable_fixing_calls =
        _get_stat(st, :variable_fixing_calls, 0)

    bnb_true_obj_time =
        _get_stat(st, :true_obj_time, missing)

    bnb_incumbent_time =
        _get_stat(st, :incumbent_time, missing)

    bnb_branch_variable_time =
        _get_stat(st, :branch_variable_time, missing)

    n_branch_zero_bnb =
        _get_stat(st, :nbranch0, 0)

    n_branch_one_bnb =
        _get_stat(st, :nbranch1, 0)

    n_branch_total_bnb =
        n_branch_zero_bnb + n_branch_one_bnb

    n_fixed_zero_bnb =
        _get_stat(st, :nfix0, 0)

    n_fixed_one_bnb =
        _get_stat(st, :nfix1, 0)

    n_fixed_total_bnb =
        n_fixed_zero_bnb + n_fixed_one_bnb

    bnb_n_int_sols =
        _get_stat(st, :n_int_sols, missing)

    bnb_int_gap_max =
        _get_stat(st, :int_gap_max, missing)

    bnb_int_gap_avg =
        _get_stat(st, :int_gap_avg, missing)

    bnb_int_gap_opt =
        _get_stat(st, :int_gap_opt, missing)

    psi_out =
        _get_stat(st, :psi, missing)

    row = Any[
        data_n,
        n,
        s,
        t,

        relaxation_out,
        method_out,
        solver_used_out,
        fixing_rule_out,
        scaling_rule_out,

        st.gap,
        bnb_root_gap,

        runtime,
        st.wall_time,
        bnb_knitro_time,
        bnb_relaxation_solve_time,
        bnb_upsilon_calibration_time,
        bnb_factorization_time,
        bnb_scaling_setup_time,
        bnb_bound_computation_time,
        bnb_open_list_time,
        bnb_node_setup_time,
        bnb_dual_solution_time,
        bnb_primal_solution_time,
        bnb_variable_fixing_direct_time,
        bnb_variable_fixing_time,
        bnb_variable_fixing_calls,

        bnb_true_obj_time,
        bnb_incumbent_time,
        bnb_branch_variable_time,

        n_branch_zero_bnb,
        n_branch_one_bnb,
        n_branch_total_bnb,

        n_fixed_zero_bnb,
        n_fixed_one_bnb,
        n_fixed_total_bnb,

        st.nodes,
        bnb_n_int_sols,
        st.tree_exhausted,
        st.time_limit_hit,

        bnb_int_gap_max,
        bnb_int_gap_avg,
        bnb_int_gap_opt,

        st.lb,
        st.ub,
        st.root_ub,

        psi_out,

        join(S_best, " "),
    ]

    row_df = _make_row_df(cols, row)

    CSV.write(
        results_filepath,
        row_df;
        append = true,
    )

    push!(
        results,
        (
            s = s,
            S_best = S_best,
            st = st,
            runtime = runtime,
            row = row,
        ),
    )

    println()
    println("B&B result")
    println("s:                                ", s)
    println("relaxation:                       ", relaxation_out)
    println("method:                           ", method_out)
    println("solver_used:                      ", solver_used_out)
    println("scaling_rule:                     ", scaling_rule_out)
    println("fixing_rule:                      ", fixing_rule_out)
    println("S_best:                           ", S_best)
    println("obj / lb:                         ", st.lb)
    println("ub:                               ", st.ub)
    println("gap:                              ", st.gap)
    println("root_ub:                          ", st.root_ub)
    println("root_gap:                         ", bnb_root_gap)
    println("nodes:                            ", st.nodes)
    println("n_int_sols:                       ", bnb_n_int_sols)
    println("branch0:                          ", n_branch_zero_bnb)
    println("branch1:                          ", n_branch_one_bnb)
    println("fix0:                             ", n_fixed_zero_bnb)
    println("fix1:                             ", n_fixed_one_bnb)
    println("wall_time:                        ", st.wall_time)
    println("runtime measured:                 ", runtime)
    println("relaxation_solve_time:            ", bnb_relaxation_solve_time)
    println("factorization_time:               ", bnb_factorization_time)
    println("scaling_setup_time:               ", bnb_scaling_setup_time)
    println("bound_computation_time:           ", bnb_bound_computation_time)
    println("open_list_time:                   ", bnb_open_list_time)
    println("node_setup_time:                  ", bnb_node_setup_time)
    println("dual_solution_time:               ", bnb_dual_solution_time)
    println("primal_solution_time:             ", bnb_primal_solution_time)
    println("variable_fixing_direct_time:      ", bnb_variable_fixing_direct_time)
    println("variable_fixing_time:             ", bnb_variable_fixing_time)
    println("variable_fixing_calls:            ", bnb_variable_fixing_calls)
    println("true_obj_time:                    ", bnb_true_obj_time)
    println("incumbent_time:                   ", bnb_incumbent_time)
    println("branch_variable_time:             ", bnb_branch_variable_time)
    println("tree_exhausted:                   ", st.tree_exhausted)
    println("time_limit_hit:                   ", st.time_limit_hit)
    println("psi:                              ", psi_out)
    println("Appended row to:                  ", results_filepath)
    flush(stdout)
end


# ============================================================
# Final in-memory table
# ============================================================
df = DataFrame(
    [cols[j] => [r.row[j] for r in results] for j in eachindex(cols)]
)


# ============================================================
# Summary
# ============================================================
println()
println("="^82)
println("Summary")
println("="^82)

for r in results
    st = r.st

    nbranch0 = _get_stat(st, :nbranch0, 0)
    nbranch1 = _get_stat(st, :nbranch1, 0)
    nfix0 = _get_stat(st, :nfix0, 0)
    nfix1 = _get_stat(st, :nfix1, 0)

    @printf(
        "s=%3d  lb=% .8f  ub=% .8f  gap=% .3e  root_ub=% .8f  nodes=%8d  br0=%7d  br1=%7d  fix0=%7d  fix1=%7d  wall=%8.2fs%s\n",
        r.s,
        st.lb,
        st.ub,
        st.gap,
        st.root_ub,
        st.nodes,
        nbranch0,
        nbranch1,
        nfix0,
        nfix1,
        st.wall_time,
        st.time_limit_hit ? "  [TIMEOUT]" : "",
    )
end

println("="^82)
println("Saved results to: $results_filepath")