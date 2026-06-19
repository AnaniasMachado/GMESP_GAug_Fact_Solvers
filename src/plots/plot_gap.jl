using CSV
using DataFrames
using Statistics
using Printf
using Plots

# ============================================================
# User inputs
# ============================================================

file_n63 = "./results/results_knitro_bfgs_prox_knitro_n63_kappa5.csv"
file_n90 = "./results/results_knitro_bfgs_prox_knitro_n90_kappa5.csv"

plots_dir = "plots"
tables_dir = "latex_tables"

mkpath(plots_dir)
mkpath(tables_dir)

# ============================================================
# Method specification
# ============================================================

const METHODS_ALL = [
    (
        plot_label = "DDGFact",
        tex_label = raw"DDGFact",
        gap = :ddgfact_gap,
        runtime = :ddgfact_runtime,
    ),
    (
        plot_label = "DDGFact+",
        tex_label = raw"DDGFact$^+$",
        gap = :ddgfact_plus_gap,
        runtime = :ddgfact_plus_runtime,
    ),
    (
        plot_label = "BFGS",
        tex_label = raw"DDGFact$^+_\Upsilon$ BFGS",
        gap = :ddgfact_plus_upsilon_bfgs_gap,
        runtime = :ddgfact_plus_upsilon_bfgs_runtime,
    ),
    (
        plot_label = "Prox Knitro",
        tex_label = raw"DDGFact$^+_\Upsilon$ Prox Knitro",
        gap = :ddgfact_plus_upsilon_one_step_gap,
        runtime = :ddgfact_plus_upsilon_one_step_runtime,
    ),
]

const METHODS_BFGS_PROX = [
    METHODS_ALL[3],
    METHODS_ALL[4],
]

# ============================================================
# Filename parsing
# ============================================================

function infer_n_from_filename(path::AbstractString)
    m = match(r"n(\d+)", basename(path))
    m === nothing && error("Could not infer n from filename: $path")
    return parse(Int, m.captures[1])
end

function infer_kappa_from_filename(path::AbstractString)
    m = match(r"kappa([^_./\\]+)", basename(path))
    m === nothing && error("Could not infer kappa from filename: $path")
    return m.captures[1]
end

# ============================================================
# Loading
# ============================================================

function load_result_table(path::AbstractString)
    df = DataFrame(
        CSV.File(
            path;
            delim = ',',
            normalizenames = true,
            silencewarnings = true,
        ),
    )

    required_cols = [
        :n,
        :s,
        :t,
        :ddgfact_gap,
        :ddgfact_plus_gap,
        :ddgfact_plus_upsilon_bfgs_gap,
        :ddgfact_plus_upsilon_one_step_gap,
        :ddgfact_runtime,
        :ddgfact_plus_runtime,
        :ddgfact_plus_upsilon_bfgs_runtime,
        :ddgfact_plus_upsilon_one_step_runtime,
    ]

    missing_cols = setdiff(required_cols, Symbol.(names(df)))
    if !isempty(missing_cols)
        error("Missing required columns in $(path): $(missing_cols)")
    end

    sort!(df, [:s, :t])

    return df
end

# ============================================================
# Aggregation
# ============================================================

function average_by_s(df::DataFrame, methods)
    sdf =
        combine(
            groupby(df, :s),
            [m.gap => mean => m.gap for m in methods]...,
        )

    sort!(sdf, :s)

    return sdf
end

function avg_gap_runtime_table(df::DataFrame, methods)
    rows = DataFrame(
        Method = String[],
        AvgGap = Float64[],
        AvgRuntime = Float64[],
    )

    for m in methods
        push!(
            rows,
            (
                Method = m.tex_label,
                AvgGap = mean(skipmissing(df[!, m.gap])),
                AvgRuntime = mean(skipmissing(df[!, m.runtime])),
            ),
        )
    end

    return rows
end

# ============================================================
# Formatting
# ============================================================

function fmt_float(x; digits::Int = 4)
    if ismissing(x)
        return "--"
    end

    xf = Float64(x)

    if isnan(xf)
        return "--"
    end

    return @sprintf("%.*f", digits, xf)
end

# ============================================================
# PNG plots using Plots.jl
# ============================================================

function make_gap_plot_png(
    df::DataFrame;
    n_value::Int,
    kappa::AbstractString,
    methods,
    filename::AbstractString,
    title_suffix::AbstractString = "",
)
    sdf = average_by_s(df, methods)

    plt =
        plot(
            xlabel = "s",
            ylabel = "Gap",
            title = "n = $(n_value), κ = $(kappa)$(title_suffix)",
            legend = :best,
            linewidth = 2.2,
            markersize = 4.5,
            framestyle = :box,
            grid = true,
            size = (850, 560),
        )

    markers = [:circle, :square, :utriangle, :diamond]

    for (j, m) in enumerate(methods)
        plot!(
            plt,
            sdf.s,
            sdf[!, m.gap],
            label = m.plot_label,
            marker = markers[mod1(j, length(markers))],
            linewidth = 2.2,
            markersize = 4.5,
        )
    end

    savefig(plt, filename)

    println("Saved plot: ", filename)

    return nothing
end

function write_all_plots(df::DataFrame; n_value::Int, kappa::AbstractString)
    filename_all =
        joinpath(plots_dir, "gaps_all_n$(n_value)_kappa$(kappa).png")

    filename_bfgs_prox =
        joinpath(plots_dir, "gaps_bfgs_vs_prox_n$(n_value)_kappa$(kappa).png")

    make_gap_plot_png(
        df;
        n_value = n_value,
        kappa = kappa,
        methods = METHODS_ALL,
        filename = filename_all,
        title_suffix = "",
    )

    make_gap_plot_png(
        df;
        n_value = n_value,
        kappa = kappa,
        methods = METHODS_BFGS_PROX,
        filename = filename_bfgs_prox,
        title_suffix = " — BFGS vs Prox Knitro",
    )

    return nothing
end

# ============================================================
# LaTeX tables saved as .txt
# ============================================================

function write_latex_avg_table_txt(
    df::DataFrame;
    n_value::Int,
    kappa::AbstractString,
    methods = METHODS_ALL,
    filename::AbstractString,
)
    tab = avg_gap_runtime_table(df, methods)

    open(filename, "w") do io
        println(io, raw"\begin{table}[!ht]")
        println(io, raw"\centering")
        println(io, raw"\footnotesize")
        println(io, "\\caption{Average gap and runtime for each upper bound for \$n = $(n_value)\$ and \$\\kappa = $(kappa)\$.}")
        println(io, "\\label{tab:avg_gap_runtime_n$(n_value)_kappa$(kappa)}")
        println(io, raw"\begin{tabular}{lrr}")
        println(io, raw"\hline")
        println(io, raw"\textbf{Upper bound} & \textbf{Average gap} & \textbf{Average runtime (s)} \\")
        println(io, raw"\hline")

        for row in eachrow(tab)
            method = row.Method
            avg_gap = fmt_float(row.AvgGap; digits = 6)
            avg_runtime = fmt_float(row.AvgRuntime; digits = 3)

            println(io, "$(method) & $(avg_gap) & $(avg_runtime) \\\\")
        end

        println(io, raw"\hline")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end

    println("Saved table: ", filename)

    return tab
end

function write_all_tables(df::DataFrame; n_value::Int, kappa::AbstractString)
    filename =
        joinpath(tables_dir, "avg_gap_runtime_n$(n_value)_kappa$(kappa).txt")

    tab =
        write_latex_avg_table_txt(
            df;
            n_value = n_value,
            kappa = kappa,
            filename = filename,
        )

    return tab
end

# ============================================================
# Main
# ============================================================

function process_file(path::AbstractString)
    df = load_result_table(path)

    n_from_file = infer_n_from_filename(path)
    kappa_from_file = infer_kappa_from_filename(path)

    n_from_table = Int(first(unique(df.n)))

    if n_from_file != n_from_table
        @warn "n inferred from filename differs from n in table" path n_from_file n_from_table
    end

    write_all_plots(
        df;
        n_value = n_from_table,
        kappa = kappa_from_file,
    )

    tab =
        write_all_tables(
            df;
            n_value = n_from_table,
            kappa = kappa_from_file,
        )

    println()
    println("Average table for n = $(n_from_table), kappa = $(kappa_from_file)")
    show(tab; allrows = true, allcols = true)
    println("\n")

    return tab
end

tab63 = process_file(file_n63)
tab90 = process_file(file_n90)