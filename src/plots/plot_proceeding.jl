using CSV
using DataFrames
using Plots
using LaTeXStrings

n = 63
kappa = 8

results_filepath =
    "results/proceeding/results_gap_knitro_n$(n)_kappa$(kappa).csv"

plots_dir = "plots/proceeding"

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
    label = L"\mathrm{DDGFact}",
    xlabel = "s",
    ylabel = "Gap",
    title = L"n = %$(n), \kappa = %$(kappa)",
)

plot!(
    plt,
    df.s,
    df.ddgfact_plus_gap;
    label = L"\mathrm{DDGFact}^{+}",
)

plot!(
    plt,
    df.s,
    df.spec_gap;
    label = L"\mathrm{Spectral}",
)

output_filepath =
    joinpath(
        plots_dir,
        "gaps_n$(n)_kappa$(kappa).png",
    )

savefig(plt, output_filepath)

println("Saved plot to: $output_filepath")