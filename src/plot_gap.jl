using CSV
using DataFrames
using Plots

# -------------------------
# CSV files you want to plot
# -------------------------
csv_paths = [
    "results/results_gap_ipopt_n63_s20.csv",
    "results/results_gap_ipopt_n90_s20.csv",
    "results/results_gap_ipopt_n124_s20.csv",
]

# -------------------------
# Output folder
# -------------------------
plots_dir = "plots"
isdir(plots_dir) || mkpath(plots_dir)

# -------------------------
# Create one scatter plot per CSV
# -------------------------
for filepath in csv_paths
    println("Reading: $filepath")

    if !isfile(filepath)
        error("CSV file not found: $(abspath(filepath))")
    end

    df = CSV.read(filepath, DataFrame)

    required_cols = [:t, :ddgfact_gap, :ddgfact_plus_gap, :spec_gap]
    for col in required_cols
        if !(col in Symbol.(names(df)))
            error("Column $(col) not found in $filepath")
        end
    end

    offset = 0.15
    n_val = df.n[1]
    s_val = df.s[1]
    plot_title = "Gap n=$(n_val) s=$(s_val)"

    p = scatter(
        df.t .- offset,
        df.ddgfact_gap;
        label = "DDGFact",
        xlabel = "t",
        ylabel = "Gap",
        title = plot_title,
        markersize = 5,
        marker = :circle,
        legend = :best,
    )

    scatter!(
        p,
        df.t,
        df.ddgfact_plus_gap;
        label = "DDGFact+",
        markersize = 5,
        marker = :circle,
    )

    scatter!(
        p,
        df.t .+ offset,
        df.spec_gap;
        label = "Spec",
        markersize = 5,
        marker = :circle,
    )

    outname = splitext(basename(filepath))[1] * "_scatter.png"
    outpath = joinpath(plots_dir, outname)

    savefig(p, outpath)

    println("Saved plot to: $outpath")
end

println("Done.")