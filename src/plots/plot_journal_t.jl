using CSV
using DataFrames
using Plots
using LaTeXStrings

n = [60]
t = [2, 4, 6, 8]

function gen_gap_plot(n, t)
    results_filepath = "results/journal_ddgfactplus/results_root_gap_n$(n)_t$(t).csv"
    plots_dir = "plots/journal_ddgfactplus"

    mkpath(plots_dir)

    df = CSV.read(results_filepath, DataFrame)
    sort!(df, :s)

    default(
        linewidth=2,
        markersize=5,
        marker=:circle,
        markerstrokewidth=0,
        framestyle=:box,
        grid=true,
        size=(700, 500),
        dpi=300,
    )

    # plt = plot(
    #     df.s,
    #     df.ddgfact_gap;
    #     label=L"\mathrm{DGFact}",
    #     xlabel=L"s",
    #     ylabel="gap",
    #     title=L"n = %$(n), t = %$(t)",
    # )

    # plot!(plt, df.s, df.ddgfact_plus_gap; label=L"\mathrm{DGFact}^{+}")
    # plot!(plt, df.s, df.spec_gap; label=L"\mathrm{Spectral}")
    # plot!(plt, df.s, df.glinx_gap; label=L"\mathrm{glinx}_{\gamma}")

    plt = plot(
        df.s,
        df.ddgfact_plus_gap;
        label=L"\mathrm{DGFact}^{+}",
        xlabel=L"s",
        ylabel="gap",
        title=L"n = %$(n), t = %$(t)",
    )

    plot!(plt, df.s, df.glinx_gap; label=L"\mathrm{glinx}_{\gamma}")

    output_filepath = joinpath(plots_dir, "gaps_n$(n)_t$(t).png")

    savefig(plt, output_filepath)

    println("Saved plot to: $output_filepath")
end

for n_val in n
    for t_val in t
        gen_gap_plot(n_val, t_val)
    end
end