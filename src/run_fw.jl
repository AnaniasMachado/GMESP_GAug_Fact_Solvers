using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra

include("heuristics.jl")
include("util.jl")
include("fw.jl")

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

# Choose ψ < λ_min(C)
# This is the augmented case.
psi = minimum(eigvals(Csym)) - 1e-10

# -------------------------
# Compute A(ψ)
# -------------------------
At = compute_At(C, psi)

# -------------------------
# Tolerance
# -------------------------
tol = 1e-3

# -------------------------
# Method Parameters
# -------------------------
solver = "fw"

method_names = ["FW"]

methods = Dict(
    "FW" => fw_gaug_fact_from_At,
)

line_search_vals = [false, true]

# -------------------------
# Data Collection
# -------------------------
df = DataFrame()
results_filepath = "results/results_gap_$(solver)_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    result = []
    append!(result, [n, s, t])

    # -------------------------
    # Local search baseline
    # -------------------------
    x_ls, z_ls = run_all_LS(Csym, s, t)

    # -------------------------
    # Frank-Wolfe variants
    # -------------------------
    for method_name in method_names
        method = methods[method_name]

        for line_search_val in line_search_vals
            println("Running $method_name, line_search = $line_search_val")

            runtime = @elapsed begin
                x, fw_gap, k = method(
                    At,
                    s,
                    t,
                    psi;
                    tol = tol,
                    line_search = line_search_val,
                )
            end

            println("Finished $method_name, line_search = $line_search_val, k = $k, FW gap = $fw_gap, runtime = $runtime")

            obj = Gamma_t_from_At(x, At, t, psi)

            # Statistics gap relative to local search
            stat_gap = obj - z_ls

            append!(result, [stat_gap, runtime])
        end
    end

    # -------------------------
    # Simplex gap
    # -------------------------
    simplex_x = simplex_sol(At, s)
    simplex_obj = Gamma_t_from_At(simplex_x, At, t, psi)
    simplex_gap = simplex_obj - z_ls

    # -------------------------
    # Spectral bound gap
    # -------------------------
    spectral_bound_val = spectral_bound(C, t)
    spec_gap = spectral_bound_val - z_ls

    append!(result, [simplex_gap, spec_gap])

    push!(results, result)
end

results_matrix = hcat(results...)'

cols = [
    :n,
    :s,
    :t,

    :fw_gap,
    :fw_runtime,

    :fwls_gap,
    :fwls_runtime,

    :simplex_gap,
    :spec_gap,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")