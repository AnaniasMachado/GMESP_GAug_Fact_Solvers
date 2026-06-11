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
# Choose reformulations / root and child Upsilon strategies
# ============================================================
runs = [
    (
        relaxation = DDGFact,
        root_psi0_warm_start = false,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = false,
    ),
    (
        relaxation = DDGFactplus,
        root_psi0_warm_start = false,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = false,
    ),

    # ------------------------------------------------------------
    # DDGFactplusUpsilon strategy options:
    #
    # root_psi0_warm_start:
    #   false -> root directly optimizes with psi := psi(gamma)
    #   true  -> root first optimizes with psi = 0, then uses that theta
    #            to initialize root optimization with psi := psi(gamma)
    #
    # warm_start_parent_upsilon:
    #   false -> child/rebound nodes use default BFGS initialization
    #   true  -> child/rebound nodes start from parent/rebound gamma,
    #            but still recalibrate with BFGS
    #
    # reuse_parent_upsilon:
    #   true  -> child/rebound nodes reuse parent/current gamma directly
    #            and do NOT recalibrate with BFGS
    #
    # If reuse_parent_upsilon = true, warm_start_parent_upsilon is ignored
    # for child/rebound Upsilon nodes.
    # ------------------------------------------------------------
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = false,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = false,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = false,
        warm_start_parent_upsilon = true,
        reuse_parent_upsilon = false,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = false,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = true,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = true,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = false,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = true,
        warm_start_parent_upsilon = true,
        reuse_parent_upsilon = false,
    ),
    (
        relaxation = DDGFactplusUpsilon,
        root_psi0_warm_start = true,
        warm_start_parent_upsilon = false,
        reuse_parent_upsilon = true,
    ),
]


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

# Direct root BFGS effort with psi = psi(gamma).
# Used when root_psi0_warm_start = false.
#
# Options depend on bfgs_param_sets:
#   :default
#   :fast
#   :very_fast
#   :direct
root_bfgs_param_set = :direct

# BFGS effort for the auxiliary root psi = 0 calibration.
# Used only for runs with root_psi0_warm_start = true.
root_psi0_bfgs_param_set = :default

# BFGS effort for the second root optimization when root_psi0_warm_start = true.
# This is the root optimization with psi = psi(gamma), initialized from the
# root psi = 0 solution.
root_after_psi0_bfgs_param_set = :fast

# BFGS effort at non-root nodes.
# Ignored for child/rebound Upsilon nodes when reuse_parent_upsilon = true.
bfgs_param_set = :very_fast

# Options:
#   :simple
#   :strong
#
# Used only when relaxation = DDGFactplusUpsilon and fixing_rule includes :dual.
upsilon_fixing = :strong


# ============================================================
# Common parameters
# ============================================================
Random.seed!(1)

time_limit = 7200.0
verbose_bnb = false
atol = 1e-8

psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


# ============================================================
# CSV output
# ============================================================
mkpath("results")

results_filepath =
    "results/test_bnb_all_reforms_data$(data_n)_n$(n)_s$(s)_t$(t)_upsilon_reuse_strategies.csv"

cols = [
    :data_n,
    :n,
    :s,
    :t,

    # Reformulation / solver choices
    :relaxation,
    :solver_used,
    :fixing_rule,
    :root_bfgs_param_set,
    :root_after_psi0_bfgs_param_set,
    :bfgs_param_set,
    :root_psi0_warm_start,
    :root_psi0_bfgs_param_set,
    :upsilon_fixing,
    :warm_start_parent_upsilon,
    :reuse_parent_upsilon,
    :use_t1_plus_bnb,

    # B&B gaps
    :bnb_gap,
    :bnb_root_gap,

    # Runtimes
    :bnb_runtime,
    :bnb_reported_wall_time,

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

    # Root psi=0 warm-start diagnostics
    :root_psi0_obj,
    :root_psi0_improved,

    # DDGFactplus parameter, if available
    :psi,
]

CSV.write(results_filepath, DataFrame([c => Any[] for c in cols]))


println("="^82)
println("GMESP B&B test: all reformulations and Upsilon reuse strategies")
println("n:                              $n")
println("s:                              $s")
println("t:                              $t")
println("use_t1_plus_bnb:                $use_t1_plus_bnb")
println("fixing_rule:                    $fixing_rule")
println("root_bfgs_param_set:            $root_bfgs_param_set")
println("root_after_psi0_bfgs_param_set: $root_after_psi0_bfgs_param_set")
println("bfgs_param_set:                 $bfgs_param_set")
println("root_psi0_bfgs_param_set:       $root_psi0_bfgs_param_set")
println("upsilon_fixing:                 $upsilon_fixing")
println("time_limit:                     $time_limit")
println("results_filepath:               $results_filepath")
println("="^82)
flush(stdout)


# ============================================================
# Run all reformulations / Upsilon strategies
# ============================================================
results = []

for run in runs
    relaxation = run.relaxation
    run_root_psi0_warm_start = run.root_psi0_warm_start
    run_warm_start_parent_upsilon = run.warm_start_parent_upsilon
    run_reuse_parent_upsilon = run.reuse_parent_upsilon

    relaxation_name = String(Symbol(relaxation))

    println()
    println("-"^82)
    println("Running relaxation:         $relaxation_name")
    println("root_psi0_warm_start:      $run_root_psi0_warm_start")
    println("warm_start_parent_upsilon: $run_warm_start_parent_upsilon")
    println("reuse_parent_upsilon:      $run_reuse_parent_upsilon")
    flush(stdout)

    Random.seed!(1)

    S_best = Int[]
    st = nothing
    solver_used = :general

    local_fixing_rule =
        relaxation == DDGFact ? (
            fixing_rule == :none ? :none : :dual
        ) : fixing_rule

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
                root_bfgs_param_set = root_bfgs_param_set,
                root_after_psi0_bfgs_param_set = root_after_psi0_bfgs_param_set,
                bfgs_param_set = bfgs_param_set,
                bfgs_param_sets = bfgs_param_sets,
                root_psi0_warm_start =
                    relaxation == DDGFactplusUpsilon ? run_root_psi0_warm_start : false,
                root_psi0_bfgs_param_set = root_psi0_bfgs_param_set,
                upsilon_fixing = upsilon_fixing,
                warm_start_parent_upsilon =
                    relaxation == DDGFactplusUpsilon ? run_warm_start_parent_upsilon : false,
                reuse_parent_upsilon =
                    relaxation == DDGFactplusUpsilon ? run_reuse_parent_upsilon : false,
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

    root_psi0_obj = hasproperty(st, :root_psi0_obj) ? st.root_psi0_obj : missing
    root_psi0_improved =
        hasproperty(st, :root_psi0_improved) ? st.root_psi0_improved : missing

    psi_out = hasproperty(st, :psi) ? st.psi : missing

    root_bfgs_param_set_out =
        hasproperty(st, :root_bfgs_param_set) ?
        String(st.root_bfgs_param_set) :
        String(root_bfgs_param_set)

    root_after_psi0_bfgs_param_set_out =
        hasproperty(st, :root_after_psi0_bfgs_param_set) ?
        String(st.root_after_psi0_bfgs_param_set) :
        String(root_after_psi0_bfgs_param_set)

    bfgs_param_set_out =
        hasproperty(st, :bfgs_param_set) ?
        String(st.bfgs_param_set) :
        String(bfgs_param_set)

    root_psi0_warm_start_out =
        hasproperty(st, :root_psi0_warm_start) ?
        st.root_psi0_warm_start :
        false

    root_psi0_bfgs_param_set_out =
        hasproperty(st, :root_psi0_bfgs_param_set) ?
        String(st.root_psi0_bfgs_param_set) :
        String(root_psi0_bfgs_param_set)

    warm_start_parent_upsilon_out =
        hasproperty(st, :warm_start_parent_upsilon) ?
        st.warm_start_parent_upsilon :
        false

    reuse_parent_upsilon_out =
        hasproperty(st, :reuse_parent_upsilon) ?
        st.reuse_parent_upsilon :
        false

    row = [
        data_n,
        n,
        s,
        t,

        relaxation_name,
        String(solver_used),
        String(local_fixing_rule),
        root_bfgs_param_set_out,
        root_after_psi0_bfgs_param_set_out,
        bfgs_param_set_out,
        root_psi0_warm_start_out,
        root_psi0_bfgs_param_set_out,
        String(upsilon_fixing),
        warm_start_parent_upsilon_out,
        reuse_parent_upsilon_out,
        use_t1_plus_bnb,

        st.gap,
        bnb_root_gap,

        runtime,
        st.wall_time,

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

        root_psi0_obj,
        root_psi0_improved,

        psi_out,
    ]

    push!(
        results,
        (
            relaxation = relaxation_name,
            root_psi0_warm_start = root_psi0_warm_start_out,
            warm_start_parent_upsilon = warm_start_parent_upsilon_out,
            reuse_parent_upsilon = reuse_parent_upsilon_out,
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
    println("root_psi0_warm_start:              ", root_psi0_warm_start_out)
    println("warm_start_parent_upsilon:         ", warm_start_parent_upsilon_out)
    println("reuse_parent_upsilon:              ", reuse_parent_upsilon_out)
    println("S_best:                            ", S_best)
    println("obj / lb:                          ", st.lb)
    println("ub:                                ", st.ub)
    println("gap:                               ", st.gap)
    println("root_ub:                           ", st.root_ub)
    println("nodes:                             ", st.nodes)
    println("wall_time:                         ", st.wall_time)
    println("runtime measured:                  ", runtime)
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

    if hasproperty(st, :root_bfgs_param_set)
        println("root_bfgs_param_set:               ", st.root_bfgs_param_set)
    end

    if hasproperty(st, :root_after_psi0_bfgs_param_set)
        println("root_after_psi0_bfgs_param_set:    ", st.root_after_psi0_bfgs_param_set)
    end

    if hasproperty(st, :bfgs_param_set)
        println("bfgs_param_set:                    ", st.bfgs_param_set)
    end

    if hasproperty(st, :root_psi0_bfgs_param_set)
        println("root_psi0_bfgs_param_set:          ", st.root_psi0_bfgs_param_set)
    end

    if hasproperty(st, :root_psi0_obj)
        println("root_psi0_obj:                     ", st.root_psi0_obj)
    end

    if hasproperty(st, :root_psi0_improved)
        println("root_psi0_improved:                ", st.root_psi0_improved)
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
        "%-22s  root_psi0=%-5s  child_warm=%-5s  reuse=%-5s  lb=% .8f  ub=% .8f  gap=% .3e  root_ub=% .8f  nodes=%8d  wall=%8.2fs%s\n",
        r.relaxation,
        string(r.root_psi0_warm_start),
        string(r.warm_start_parent_upsilon),
        string(r.reuse_parent_upsilon),
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