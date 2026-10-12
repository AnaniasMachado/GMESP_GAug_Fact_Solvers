using MKL
using MAT
using LinearAlgebra
using Printf

# ============================================================
# Experiment parameters
# ============================================================

n_vals = [63, 90]

blas_threads = 16
MKL.set_num_threads(blas_threads)


# ============================================================
# Output
# ============================================================

results_dir = "results/journal_ddgfactplus"
latex_filepath = joinpath(results_dir, "matrix_statistics.txt")

mkpath(results_dir)

results = []


# ============================================================
# Problem data and statistics
# ============================================================

for n in n_vals
    matfile = matopen("data/data$n.mat")
    C = n == 63 ? read(matfile, "A") : read(matfile, "C")
    close(matfile)

    C = Matrix{Float64}(C)
    Csym = Symmetric(C)

    # Eigenvalues
    lambda = eigvals(Csym)

    lambda_min  = minimum(lambda)
    lambda_max  = maximum(lambda)
    lambda_mean = sum(lambda) / length(lambda)

    # Spectral condition number
    cond_C = lambda_max / lambda_min

    push!(results, (
        n = n,
        lambda_min = lambda_min,
        lambda_max = lambda_max,
        lambda_mean = lambda_mean,
        cond = cond_C
    ))
end


# ============================================================
# Print results
# ============================================================

println("\nMatrix statistics")
println("=================")

for r in results
    println("\nn = $(r.n)")
    @printf("  Minimum eigenvalue : %.10e\n", r.lambda_min)
    @printf("  Maximum eigenvalue : %.10e\n", r.lambda_max)
    @printf("  Mean eigenvalue    : %.10e\n", r.lambda_mean)
    @printf("  Condition number   : %.10e\n", r.cond)
end


# ============================================================
# Write LaTeX table
# ============================================================

open(latex_filepath, "w") do io
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\begin{tabular}{crrrr}")
    println(io, "\\hline")
    println(io,
        "\$n\$ & \$\\lambda_{\\min}(C)\$ & \$\\lambda_{\\max}(C)\$ & " *
        "\$\\operatorname{mean}(\\lambda(C))\$ & \$\\kappa_2(C)\$ \\\\"
    )
    println(io, "\\hline")

    for r in results
        @printf(
            io,
            "%d & %.6e & %.6e & %.6e & %.6e \\\\\n",
            r.n,
            r.lambda_min,
            r.lambda_max,
            r.lambda_mean,
            r.cond
        )
    end

    println(io, "\\hline")
    println(io, "\\end{tabular}")
    println(io, "\\caption{Spectral statistics of the covariance matrices.}")
    println(io, "\\label{tab:matrix_statistics}")
    println(io, "\\end{table}")
end

println("\nLaTeX table written to: $latex_filepath")