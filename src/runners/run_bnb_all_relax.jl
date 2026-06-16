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
include("gscaling_prox.jl")
include("gscaling_params.jl")

include("dual.jl")
include("var_fixing.jl")

include("bnb_util.jl")
include("bnb_general.jl")
include("bnb_t1_plus.jl")


# ============================================================
# Choose instance
# ============================================================
data_n = 63
k = 32

s = 16
t = 15

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
C = Symmetric(C)

n = size(C, 1)


# ============================================================
# B&B reformulations / calibration algorithms
# ============================================================
# DDGFact and DDGFactplus do not use a calibration algorithm.
# DDGFactplusUpsilon is tested with BFGS and one-step proximal Knitro.
runs = [
    (
        relaxation = DDGFact,
        calibration_method = :none,
    ),
    (
        relaxation = DDGFactplus,
        calibration_method = :none,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        calibration_method = :bfgs,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        calibration_method = :prox_step,
    ),
]

# runs = [
#     (
#         relaxation = DDGFactplusUpsilon,
#         calibration_method = :prox_step,
#     ),
# ]


# ============================================================
# DDGFactplus t = 1 special B&B option
# ============================================================
# If true and t == 1, DDGFactplus is solved with solve_bnb_ddgfactplus_t1.
# Otherwise, DDGFactplus is solved with the general B&B.
use_t1_plus_bnb = true


# ============================================================
# General B&B parameters
# ============================================================
# Options:
#   :none
#   :dual
#   :primal
#   :both
#
# For DDGFact, only :none and :dual are supported.
fixing_rule = :primal

# Options:
#   :simple
#   :strong
#
# Used only when relaxation = DDGFactplusUpsilon and fixing_rule includes :dual.
upsilon_fixing = :strong


# ============================================================
# Upsilon calibration parameter sets
# ============================================================
# The B&B function receives one parameter dictionary for the root node and
# another one for all child/rebound nodes. These dictionaries are used only
# for relaxation = DDGFactplusUpsilon.

# -------------------------
# Calibration parameter choices
# -------------------------
# These labels are used only in this runner for reporting.
# The dictionaries come from gscaling_params.jl.

root_bfgs_param_label = :default
node_bfgs_param_label = :very_fast

root_prox_step_param_label = :root
node_prox_step_param_label = :node


function _calibration_config(method::Symbol)
    if method == :bfgs
        return (
            solver_method = :bfgs,
            root_params = copy(bfgs_param_sets[root_bfgs_param_label]),
            node_params = copy(bfgs_param_sets[node_bfgs_param_label]),
            root_label = root_bfgs_param_label,
            node_label = node_bfgs_param_label,
        )

    elseif method == :prox_step
        return (
            solver_method = :prox_step,
            root_params = copy(prox_step_param_sets[root_prox_step_param_label]),
            node_params = copy(prox_step_param_sets[node_prox_step_param_label]),
            root_label = root_prox_step_param_label,
            node_label = node_prox_step_param_label,
        )

    elseif method == :none
        # The calibration method is ignored for DDGFact and DDGFactplus,
        # but solve_bnb_ddfact still expects a supported Upsilon method.
        return (
            solver_method = :bfgs,
            root_params = Dict{Symbol,Any}(),
            node_params = Dict{Symbol,Any}(),
            root_label = :none,
            node_label = :none,
        )

    else
        error("Unknown calibration method: $method")
    end
end


# ============================================================
# Common parameters
# ============================================================
Random.seed!(1)

time_limit = 7200.0
verbose_bnb = false
atol = 1e-10

psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


# ============================================================
# CSV output
# ============================================================
mkpath("results")

results_filepath =
    "results/test_bnb_all_data$(data_n)_n$(n)_s$(s)_t$(t).csv"

cols = [
    :data_n,
    :n,
    :s,
    :t,

    # Reformulation / solver choices
    :relaxation,
    :solver_used,
    :fixing_rule,
    :calibration_method,
    :root_calibration_param_set,
    :node_calibration_param_set,
    :upsilon_fixing,
    :use_t1_plus_bnb,

    # B&B gaps
    :bnb_gap,
    :bnb_root_gap,

    # Runtimes
    :bnb_runtime,
    :bnb_reported_wall_time,
    :bnb_knitro_time,
    :bnb_relaxation_solve_time,
    :bnb_upsilon_calibration_time,
    :bnb_factorization_time,
    :bnb_bound_computation_time,
    :bnb_open_list_time,
    :bnb_node_setup_time,
    :bnb_dual_solution_time,
    :bnb_variable_fixing_direct_time,
    :bnb_variable_fixing_time,
    :bnb_variable_fixing_calls,

    # B&B variable fixing / branching counts
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

    # DDGFactplus parameter, if available
    :psi,
]

CSV.write(results_filepath, DataFrame([c => Any[] for c in cols]))


println("="^82)
println("GMESP B&B test: DDGFact, DDGFactplus, BFGS-Upsilon, prox-step-Upsilon")
println("data_n:                         $data_n")
println("n:                              $n")
println("s:                              $s")
println("t:                              $t")
println("use_t1_plus_bnb:                $use_t1_plus_bnb")
println("fixing_rule:                    $fixing_rule")
println("upsilon_fixing:                 $upsilon_fixing")
println("root_bfgs_param_set:            $root_bfgs_param_label")
println("node_bfgs_param_set:            $node_bfgs_param_label")
println("root_prox_step_param_set:       $root_prox_step_param_label")
println("node_prox_step_param_set:       $node_prox_step_param_label")
println("time_limit:                     $time_limit")
println("results_filepath:               $results_filepath")
println("="^82)
flush(stdout)


# ============================================================
# Run all reformulations / calibration algorithms
# ============================================================
results = []

for run in runs
    relaxation = run.relaxation
    requested_calibration_method = run.calibration_method
    cfg = _calibration_config(requested_calibration_method)

    relaxation_name = String(Symbol(relaxation))
    calibration_name = String(Symbol(requested_calibration_method))

    println()
    println("-"^82)
    println("Running relaxation:          $relaxation_name")
    println("calibration_method:          $calibration_name")
    println("root_calibration_param_set:  $(cfg.root_label)")
    println("node_calibration_param_set:  $(cfg.node_label)")
    flush(stdout)

    Random.seed!(1)

    S_best = Int[]
    st = nothing
    solver_used = :general

    local_fixing_rule =
        relaxation == DDGFact ?
        (fixing_rule == :none ? :none : :dual) :
        fixing_rule

    runtime = @elapsed begin
        if relaxation == DDGFactplus && t == 1 && use_t1_plus_bnb
            solver_used = :t1_plus

            S_best, st = solve_bnb_ddgfactplus_t1(
                C,
                s;
                psi = psi,
                time_limit = time_limit,
                verbose = verbose_bnb,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )
        else
            solver_used = :general

            S_best, st = solve_bnb_ddfact(
                C,
                s,
                t;
                relaxation = relaxation,
                fixing_rule = local_fixing_rule,
                psi = psi,
                time_limit = time_limit,
                verbose = verbose_bnb,
                calibration_method = cfg.solver_method,
                root_calibration_params = cfg.root_params,
                node_calibration_params = cfg.node_params,
                upsilon_fixing = upsilon_fixing,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )
        end
    end

    n_fixed_zero_bnb = hasproperty(st, :nfix0) ? st.nfix0 : 0
    n_fixed_one_bnb = hasproperty(st, :nfix1) ? st.nfix1 : 0
    n_fixed_total_bnb = n_fixed_zero_bnb + n_fixed_one_bnb

    bnb_root_gap = st.root_ub - st.lb

    bnb_n_int_sols = hasproperty(st, :n_int_sols) ? st.n_int_sols : missing
    bnb_int_gap_max = hasproperty(st, :int_gap_max) ? st.int_gap_max : missing
    bnb_int_gap_avg = hasproperty(st, :int_gap_avg) ? st.int_gap_avg : missing
    bnb_int_gap_opt = hasproperty(st, :int_gap_opt) ? st.int_gap_opt : missing

    psi_out = hasproperty(st, :psi) ? st.psi : missing

    bnb_knitro_time =
        hasproperty(st, :knitro_time) ? st.knitro_time : missing

    bnb_relaxation_solve_time =
        hasproperty(st, :relaxation_solve_time) ? st.relaxation_solve_time : missing

    bnb_upsilon_calibration_time =
        hasproperty(st, :upsilon_calibration_time) ? st.upsilon_calibration_time : missing
    
    bnb_factorization_time =
        hasproperty(st, :factorization_time) ? st.factorization_time : missing

    bnb_bound_computation_time =
        hasproperty(st, :bound_computation_time) ? st.bound_computation_time : missing

    bnb_open_list_time =
        hasproperty(st, :open_list_time) ? st.open_list_time : missing

    bnb_node_setup_time =
        hasproperty(st, :node_setup_time) ? st.node_setup_time : missing

    bnb_dual_solution_time =
        hasproperty(st, :dual_solution_time) ? st.dual_solution_time : missing

    bnb_variable_fixing_direct_time =
        hasproperty(st, :variable_fixing_direct_time) ? st.variable_fixing_direct_time : missing

    bnb_variable_fixing_time =
        hasproperty(st, :variable_fixing_time) ? st.variable_fixing_time : missing

    bnb_variable_fixing_calls =
        hasproperty(st, :variable_fixing_calls) ? st.variable_fixing_calls : missing

    calibration_method_out =
        relaxation == DDGFactplusUpsilon ?
        String(st.calibration_method) :
        "none"

    row = [
        data_n,
        n,
        s,
        t,

        relaxation_name,
        String(solver_used),
        String(local_fixing_rule),
        calibration_method_out,
        String(cfg.root_label),
        String(cfg.node_label),
        String(upsilon_fixing),
        use_t1_plus_bnb,

        st.gap,
        bnb_root_gap,

        runtime,
        st.wall_time,
        bnb_knitro_time,
        bnb_relaxation_solve_time,
        bnb_upsilon_calibration_time,
        bnb_factorization_time,
        bnb_bound_computation_time,
        bnb_open_list_time,
        bnb_node_setup_time,
        bnb_dual_solution_time,
        bnb_variable_fixing_direct_time,
        bnb_variable_fixing_time,
        bnb_variable_fixing_calls,

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
    ]

    push!(
        results,
        (
            relaxation = relaxation_name,
            calibration_method = calibration_method_out,
            root_calibration_param_set = cfg.root_label,
            node_calibration_param_set = cfg.node_label,
            solver_used = solver_used,
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
    println("relaxation:                        ", relaxation_name)
    println("solver_used:                       ", solver_used)
    println("calibration_method:                ", calibration_method_out)
    println("root_calibration_param_set:        ", cfg.root_label)
    println("node_calibration_param_set:        ", cfg.node_label)
    println("S_best:                            ", S_best)
    println("obj / lb:                          ", st.lb)
    println("ub:                                ", st.ub)
    println("gap:                               ", st.gap)
    println("root_ub:                           ", st.root_ub)
    println("nodes:                             ", st.nodes)
    println("wall_time:                         ", st.wall_time)
    println("runtime measured:                  ", runtime)

    if hasproperty(st, :knitro_time)
        println("knitro_time:                       ", st.knitro_time)
    end

    if hasproperty(st, :relaxation_solve_time)
        println("relaxation_solve_time:             ", st.relaxation_solve_time)
    end

    if hasproperty(st, :upsilon_calibration_time)
        println("upsilon_calibration_time:          ", st.upsilon_calibration_time)
    end

    if hasproperty(st, :factorization_time)
        println("factorization_time:                ", st.factorization_time)
    end

    if hasproperty(st, :bound_computation_time)
        println("bound_computation_time:            ", st.bound_computation_time)
    end

    if hasproperty(st, :open_list_time)
        println("open_list_time:                    ", st.open_list_time)
    end

    if hasproperty(st, :node_setup_time)
        println("node_setup_time:                   ", st.node_setup_time)
    end

    if hasproperty(st, :dual_solution_time)
        println("dual_solution_time:                ", st.dual_solution_time)
    end

    if hasproperty(st, :variable_fixing_direct_time)
        println("variable_fixing_direct_time:       ", st.variable_fixing_direct_time)
    end

    if hasproperty(st, :variable_fixing_time)
        println("variable_fixing_time:              ", st.variable_fixing_time)
    end

    if hasproperty(st, :variable_fixing_calls)
        println("variable_fixing_calls:             ", st.variable_fixing_calls)
    end

    println("tree_exhausted:                    ", st.tree_exhausted)
    println("time_limit_hit:                    ", st.time_limit_hit)

    if hasproperty(st, :nfix0)
        println("nfix0:                             ", st.nfix0)
    end

    if hasproperty(st, :nfix1)
        println("nfix1:                             ", st.nfix1)
    end

    if hasproperty(st, :fixing_rule)
        println("fixing_rule:                       ", st.fixing_rule)
    end

    if hasproperty(st, :upsilon_fixing)
        println("upsilon_fixing:                    ", st.upsilon_fixing)
    end

    if hasproperty(st, :psi)
        println("psi:                               ", st.psi)
    end

    println("Appended row to:                   ", results_filepath)
    flush(stdout)
end


# ============================================================
# Final in-memory table
# ============================================================
df = DataFrame(
    [Symbol(cols[j]) => [r.row[j] for r in results] for j in eachindex(cols)]
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

    @printf(
        "%-22s  calib=%-10s  root=%-12s  node=%-12s  lb=% .8f  ub=% .8f  gap=% .3e  root_ub=% .8f  nodes=%8d  wall=%8.2fs%s\n",
        r.relaxation,
        r.calibration_method,
        String(r.root_calibration_param_set),
        String(r.node_calibration_param_set),
        st.lb,
        st.ub,
        st.gap,
        st.root_ub,
        st.nodes,
        st.wall_time,
        st.time_limit_hit ? "  [TIMEOUT]" : "",
    )
end

println("="^82)
println("Saved results to: $results_filepath")