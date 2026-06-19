using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Ipopt
using KNITRO
import MathOptInterface as MOI

include("./misc/util.jl")
include("./misc/heuristics.jl")
include("./solvers/solver_ipopt.jl")


# -------------------------
# Problem data
# -------------------------
n = 63
kappa = 5
s_vals = [s for s in (kappa + 1):(n-2)]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# Choose ψ < λ_min(C)
psi = eigmin(Csym) - atol

# -------------------------
# Calibration parameters
# -------------------------
max_calib_iter = 50
alpha0 = 1e-3
rho = 1e4
margin = 1e-5
tau = 1e-5

# -------------------------
# Data Collection
# -------------------------
solver = "ipopt"
df = DataFrame()
results_filepath = "results/results_gap_$(solver)_n$(n)_kappa$(kappa).csv"
results = []

for s in s_vals
    t = s - kappa
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
            t,
            psi;
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