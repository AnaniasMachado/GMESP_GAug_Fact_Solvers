using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Ipopt
using KNITRO

include("util.jl")
include("heuristics.jl")
include("solver_ipopt.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
s = 20
t_vals = [i for i in 1:s]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# -------------------------
# Data Collection
# -------------------------
solver = "ipopt"
df = DataFrame()
results_filepath = "results/results_gap_$(solver)_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    result = []
    append!(result, [n, s, t])

    # -------------------------
    # DDGFact, non-augmented
    # This corresponds to ψ = 0
    # -------------------------
    runtime_ddgfact = @elapsed begin
        x_ddgfact, z_ddgfact = ddfact_gmesp(
            Csym,
            s,
            t;
            atol = atol,
        )
    end

    # -------------------------
    # DDGFact+, augmented
    # This corresponds to ψ < λ_min(C)
    # -------------------------
    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(
            Csym,
            s,
            t;
            atol = atol,
        )
    end

    # -------------------------
    # Local search
    # -------------------------
    x_ls, z_ls = run_all_LS(Csym, s, t)

    # -------------------------
    # Spectral bound
    # -------------------------
    z_spec = spectral_bound_solver(Csym, t)

    append!(
        result,
        [
            z_ddgfact - z_ls,
            z_ddgfact_plus - z_ls,
            z_spec - z_ls,
            runtime_ddgfact,
            runtime_ddgfact_plus,
        ],
    )

    push!(results, result)
end

results_matrix = hcat(results...)'

cols = [
    :n,
    :s,
    :t,
    :ddgfact_gap,
    :ddgfact_plus_gap,
    :spec_gap,
    :ddgfact_runtime,
    :ddgfact_plus_runtime,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")