using CSV
using DataFrames
using Plots
using LaTeXStrings

n = [50]
kappa = [2, 4, 6, 8]

function gen_gap_plot(n, kappa)
    results_filepath =
        "results/journal_ddgfactplus/results_root_gap_n$(n)_kappa$(kappa).csv"

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
        df.ddgfact_gap;
        label = L"\mathrm{DGFact}",
        xlabel = L"s",
        ylabel = "gap",
        title = L"n = %$(n), \kappa = %$(kappa)",
    )

    plot!(
        plt,
        df.s,
        df.ddgfact_plus_gap;
        label = L"\mathrm{DGFact}^{+}",
    )

    plot!(
        plt,
        df.s,
        df.spec_gap;
        label = L"\mathrm{Spectral}",
    )

    plot!(
        plt,
        df.s,
        df.glinx_gap;
        label = L"\mathrm{glinx}_{\gamma}",
    )

    output_filepath =
        joinpath(
            plots_dir,
            "gaps_n$(n)_kappa$(kappa).png",
        )

    savefig(plt, output_filepath)

    println("Saved plot to: $output_filepath")
end

for n_val in n
    for kappa_val in kappa
        gen_gap_plot(n_val, kappa_val)
    end
end