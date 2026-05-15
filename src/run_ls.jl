using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra

include("util.jl")
include("heuristics.jl")

# -------------------------
# Problem data
# -------------------------
n = 124
s = 20
t_vals = [i for i in 1:s]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-5

# -------------------------
# Method Parameters
# -------------------------
init_strategies = [:Cont, :Greedy, :ReverseGreedy, :Simplex]
ls_strategies = [:FI, :FP, :BI]

# -------------------------
# Data Collection
# -------------------------
solver = "ls_by_init"
df = DataFrame()
results_filepath = "results/results_$(solver)_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    result = []
    append!(result, [n, s, t])

    best_ls_obj = -Inf

    for init_strategy in init_strategies
        println("Initial solution: $init_strategy")

        arr_init = init_heur_soln(
            Csym,
            s,
            t,
            init_strategy;
            atol = atol,
        )

        for ls_strategy in ls_strategies
            println("Running LS strategy: $ls_strategy")

            x_ls, z_ls = runLS(
                Csym,
                n,
                s,
                t,
                ls_strategy;
                atol = atol,
                arr_init = copy(arr_init),
            )

            push!(result, z_ls)

            if z_ls > best_ls_obj
                best_ls_obj = z_ls
            end
        end
    end

    push!(result, best_ls_obj)
    push!(results, result)
end

results_matrix = hcat(results...)'

cols = [
    :n,
    :s,
    :t,

    :cont_fi_obj,
    :cont_fp_obj,
    :cont_bi_obj,

    :greedy_fi_obj,
    :greedy_fp_obj,
    :greedy_bi_obj,

    :reversegreedy_fi_obj,
    :reversegreedy_fp_obj,
    :reversegreedy_bi_obj,

    :simplex_fi_obj,
    :simplex_fp_obj,
    :simplex_bi_obj,

    :best_ls_obj,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)

println("Saved results to: $results_filepath")