using CSV
using DataFrames
using Plots

# -------------------------
# CSV files to plot
# -------------------------
csv_paths = [
    "results/results_ls_by_init_n63_s20.csv",
    "results/results_ls_by_init_n90_s20.csv",
    "results/results_ls_by_init_n124_s20.csv",
]

# -------------------------
# Output folder
# -------------------------
plots_dir = "plots"
isdir(plots_dir) || mkpath(plots_dir)

# -------------------------
# Create one plot per CSV
# -------------------------
for filepath in csv_paths
    println("Reading: $filepath")

    if !isfile(filepath)
        error("CSV file not found: $(abspath(filepath))")
    end

    df = CSV.read(filepath, DataFrame)

    # -------------------------
    # Check required columns
    # -------------------------
    required_cols = [
        :n, :s, :t,
        :cont_fi_obj, :cont_fp_obj, :cont_bi_obj,
        :greedy_fi_obj, :greedy_fp_obj, :greedy_bi_obj,
        :reversegreedy_fi_obj, :reversegreedy_fp_obj, :reversegreedy_bi_obj,
        :simplex_fi_obj, :simplex_fp_obj, :simplex_bi_obj,
    ]

    for col in required_cols
        if !(col in Symbol.(names(df)))
            error("Column $(col) not found in $filepath")
        end
    end

    # -------------------------
    # Best LS objective for each initialization
    # -------------------------
    cont_obj = max.(df.cont_fi_obj, df.cont_fp_obj, df.cont_bi_obj)
    greedy_obj = max.(df.greedy_fi_obj, df.greedy_fp_obj, df.greedy_bi_obj)
    reversegreedy_obj = max.(df.reversegreedy_fi_obj, df.reversegreedy_fp_obj, df.reversegreedy_bi_obj)
    simplex_obj = max.(df.simplex_fi_obj, df.simplex_fp_obj, df.simplex_bi_obj)

    # -------------------------
    # Gap of Simplex local optimum to best non simplex local optimum
    # -------------------------
    best_non_simplex = max.(cont_obj, greedy_obj, reversegreedy_obj)
    simplex_gap = best_non_simplex .- simplex_obj

    n_val = df.n[1]
    s_val = df.s[1]
    plot_title = "Simplex local search gap n=$(n_val) s=$(s_val)"

    p = scatter(
        df.t,
        simplex_gap;
        label = "best_z - z_simplex",
        xlabel = "t",
        ylabel = "Gap",
        title = plot_title,
        markersize = 7,
        marker = :circle,
        xticks = df.t,
        legend = :best,
    )

    outname = splitext(basename(filepath))[1] * "_simplex_gap.png"
    outpath = joinpath(plots_dir, outname)

    savefig(p, outpath)
    println("Saved plot to: $outpath")
end

println("Done.")