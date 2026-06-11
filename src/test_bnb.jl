using Random
using MAT
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
include("bnb_t1_plus.jl")


# ============================================================
# Choose B&B solver
# ============================================================
# Options:
#   :general
#   :t1_plus
# ============================================================
bnb_solver = :general
# bnb_solver = :t1_plus


# ============================================================
# Choose instance
# ============================================================
data_n = 63
k = 32

s = 6
t = 3

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
C = Symmetric(C)

n = size(C, 1)


# ============================================================
# General B&B parameters
# ============================================================
# Options:
#   DDGFact
#   DDGFactplus
#   DDGFactplusUpsilon
relaxation = DDGFactplus

# Options:
#   :dual
#   :primal
#   :both
#
# For DDGFact, only :dual is supported.
fixing_rule = :dual

# Options:
#   :default
#   :fast
#   :very_fast
active_bfgs_param_set = :fast

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

time_limit = 600.0
verbose_bnb = true
atol = 1e-8

psi = nothing
psi_margin = 1e-7
psi_floor = 0.0


println("="^82)
println("GMESP B&B test")
println("n:                 $n")
println("s:                 $s")
println("t:                 $t")
println("bnb_solver:        $bnb_solver")
println("="^82)
flush(stdout)


# ============================================================
# Run selected B&B
# ============================================================
S_best = Int[]
st = nothing

runtime = @elapsed begin
    if bnb_solver == :general
        S_best, st = solve_bnb_ddfact(
            C,
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
    elseif bnb_solver == :t1_plus
        if t != 1
            error("bnb_solver = :t1_plus requires t = 1.")
        end

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
        error("bnb_solver must be either :general or :t1_plus.")
    end
end


# ============================================================
# Report
# ============================================================
println()
println("="^82)
println("B&B result")
println("S_best:            ", S_best)
println("obj / lb:          ", st.lb)
println("ub:                ", st.ub)
println("gap:               ", st.gap)
println("root_ub:           ", st.root_ub)
println("nodes:             ", st.nodes)
println("wall_time:         ", st.wall_time)
println("runtime measured:  ", runtime)
println("tree_exhausted:    ", st.tree_exhausted)
println("time_limit_hit:    ", st.time_limit_hit)

if hasproperty(st, :nfix0)
    println("nfix0:             ", st.nfix0)
end

if hasproperty(st, :nfix1)
    println("nfix1:             ", st.nfix1)
end

if hasproperty(st, :relaxation)
    println("relaxation:        ", st.relaxation)
end

if hasproperty(st, :fixing_rule)
    println("fixing_rule:       ", st.fixing_rule)
end

if hasproperty(st, :psi)
    println("psi:               ", st.psi)
end

println("="^82)