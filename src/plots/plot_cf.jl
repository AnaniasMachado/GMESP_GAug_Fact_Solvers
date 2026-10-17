
using CSV
using DataFrames
using Plots
using LaTeXStrings

# ============================================================
# Experiment parameters
# ============================================================

data_n = 60
n = 60
t = 2

# ============================================================
# Plot
# ============================================================

function gen_gap_plot(data_n, n, t)

    results_filepath = joinpath(
        "results/journal_ddgfactplus",
        "results_root_gap_data$(data_n)_n$(n)_t$(t)_rank3.csv"
    )

    plots_dir = "plots/journal_ddgfactplus"
    mkpath(plots_dir)

    df = CSV.read(results_filepath, DataFrame)
    sort!(df, :s)

    default(
        linewidth = 2,
        markersize = 5,
        marker = :circle,
        markerstrokewidth = 0,
        framestyle = :box,
        grid = true,
        size = (700, 500),
        dpi = 300,
    )

    plt = plot(
        df.s,
        df.ddgfact_plus_gap;
        label = L"\mathrm{DDGFact}^{+}",
        xlabel = L"s",
        ylabel = "gap",
        title = L"n = %$(n), t = %$(t)",
    )

    plot!(
        plt,
        df.s,
        df.glinx_gap;
        label = L"\mathrm{glinx}_{\gamma}",
    )

    plot!(
        plt,
        df.s,
        df.trace_gap;
        label = L"\mathrm{Trace}",
    )

    output_filepath = joinpath(
        plots_dir,
        "root_gaps_data$(data_n)_n$(n)_t$(t)_rank3.png"
    )

    savefig(plt, output_filepath)

    println("Saved plot to: $output_filepath")
end

# ============================================================
# Generate plot
# ============================================================

gen_gap_plot(data_n, n, t)
