using Pkg
const ROOT = @__DIR__
Pkg.activate(ROOT)
Pkg.instantiate()

using LinearAlgebra, JuMP, MAT, Ipopt, Printf

include(joinpath(ROOT, "src", "relaxations.jl"))
include(joinpath(ROOT, "src", "duals.jl"))
include(joinpath(ROOT, "src", "heuristics.jl"))
include(joinpath(ROOT, "src", "bnb_ddfact.jl"))


# data63 stores the matrix under "A"; take the leading 32x32 principal submatrix.
const DATA = "data63.mat"
const DATANAME = replace(DATA, ".mat" => "")
raw = read(matopen(joinpath(ROOT, "data", DATA)), "A")
C = Symmetric(raw[1:32, 1:32])
n = size(C, 1)

println("="^82)
println("$DATANAME, leading n=$n principal submatrix.   Regime s - t = 1,  t = 1,2,3.")
println("="^82)

# One record per instance, for the CSV.
records = NamedTuple[]

for t in 1:3
    s = t + 1
    _, st = solve_bnb_ddfact(C, s, t; time_limit = 600.0, verbose = false)
    @printf("t=%2d  s=%2d  rgap=%7.4f  gap=%7.4f  nodes=%6d  fix0=%5d  fix1=%5d  wall=%7.2fs%s\n",
            t, s, st.root_ub - st.lb, st.gap, st.nodes, st.nfix0, st.nfix1, st.wall_time,
            st.time_limit_hit ? "  [TIMEOUT]" : "")
    flush(stdout)
    push!(records, (s = s, t = t, st = st))
end

# ── Write one CSV row per instance ──────────────────────────────────────────
csv_path = joinpath(ROOT, "results.csv")
open(csv_path, "w") do io
    println(io, "data,n,s,t,obj,ub,gap,root_ub,nodes,nfix0,nfix1,",
                "n_int_sols,int_gap_max,int_gap_avg,int_gap_opt,",
                "wall_time,tree_exhausted,time_limit_hit")
    for r in records
        st = r.st
        @printf(io,
            "%s,%d,%d,%d,%.8f,%.8f,%.3e,%.8f,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.4f,%s,%s\n",
            DATANAME, n, r.s, r.t,
            st.lb, st.ub, st.gap, st.root_ub,
            st.nodes, st.nfix0, st.nfix1, st.n_int_sols,
            st.int_gap_max, st.int_gap_avg, st.int_gap_opt,
            st.wall_time, st.tree_exhausted, st.time_limit_hit)
    end
end
println("\nWrote $(length(records)) rows to $csv_path")
