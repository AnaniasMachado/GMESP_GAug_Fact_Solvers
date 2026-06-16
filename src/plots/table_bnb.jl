# make_bnb_main_results_table.jl

using CSV
using DataFrames
using Printf

# ============================================================
# User input
# ============================================================

bnb_file = "./results/test_bnb_all_data63_n32_s16_t15.csv"

tables_dir = "latex_tables"
mkpath(tables_dir)

# ============================================================
# Helpers
# ============================================================

function infer_n_from_filename(path::AbstractString)
    m = match(r"n(\d+)", basename(path))
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function infer_kappa_from_filename(path::AbstractString)
    m = match(r"kappa([^_./\\]+)", basename(path))
    return m === nothing ? nothing : m.captures[1]
end

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

function fmt_int(x)
    if ismissing(x)
        return "--"
    end

    return string(Int(round(Float64(x))))
end

function method_name(row)
    relaxation = String(row.relaxation)

    if relaxation == "DDGFact"
        return raw"DDGFact"
    elseif relaxation == "DDGFactplus"
        return raw"DDGFact$^+$"
    elseif relaxation == "DDGFactplusUpsilon"
        calibration_method = String(row.calibration_method)

        if calibration_method == "bfgs"
            return raw"BFGS"
        elseif calibration_method == "prox_step"
            return raw"Prox Knitro"
        else
            return raw"DDGFact$^+_\Upsilon$"
        end
    else
        return relaxation
    end
end

function load_bnb_table(path::AbstractString)
    df = DataFrame(
        CSV.File(
            path;
            delim = ',',
            normalizenames = true,
            silencewarnings = true,
        ),
    )

    required_cols = [
        :data_n,
        :n,
        :s,
        :t,
        :relaxation,
        :calibration_method,
        :bnb_root_gap,
        :bnb_runtime,
        :bnb_knitro_time,
        :bnb_nodes,
        :n_fixed_total_bnb,
        :bnb_int_gap_max,
        :bnb_int_gap_avg,
    ]

    missing_cols = setdiff(required_cols, Symbol.(names(df)))
    if !isempty(missing_cols)
        error("Missing required columns in $(path): $(missing_cols)")
    end

    return df
end

function build_main_results_table(df::DataFrame)
    rows = DataFrame(
        data_n = Int[],
        n = Int[],
        s = Int[],
        t = Int[],
        Method = String[],
        RootGap = Float64[],
        Runtime = Float64[],
        KnitroTime = Float64[],
        Nodes = Int[],
        FixedVars = Int[],
        MaxIntGap = Float64[],
        AvgIntGap = Float64[],
    )

    for row in eachrow(df)
        push!(
            rows,
            (
                data_n = Int(row.data_n),
                n = Int(row.n),
                s = Int(row.s),
                t = Int(row.t),
                Method = method_name(row),
                RootGap = Float64(row.bnb_root_gap),
                Runtime = Float64(row.bnb_runtime),
                KnitroTime = Float64(row.bnb_knitro_time),
                Nodes = Int(row.bnb_nodes),
                FixedVars = Int(row.n_fixed_total_bnb),
                MaxIntGap = Float64(row.bnb_int_gap_max),
                AvgIntGap = Float64(row.bnb_int_gap_avg),
            ),
        )
    end

    sort!(rows, [:data_n, :n, :s, :t, :Method])

    return rows
end

function write_latex_main_results_table(
    tab::DataFrame;
    filename::AbstractString,
    caption::AbstractString,
    label::AbstractString,
)
    open(filename, "w") do io
        println(io, raw"\begin{table}[!ht]")
        println(io, raw"\centering")
        println(io, raw"\footnotesize")
        println(io, "\\caption{$caption}")
        println(io, "\\label{$label}")
        println(io, raw"\begin{tabular}{rrrrlrrrrrrr}")
        println(io, raw"\hline")
        println(io, raw"\textbf{data\_n} & \textbf{$n$} & \textbf{$s$} & \textbf{$t$} & \textbf{Method} & \textbf{Root gap} & \textbf{Runtime} & \textbf{Knitro time} & \textbf{Nodes} & \textbf{Fixed vars.} & \textbf{Max int. gap} & \textbf{Avg int. gap} \\")
        println(io, raw"\hline")

        for row in eachrow(tab)
            println(
                io,
                "$(row.data_n) & " *
                "$(row.n) & " *
                "$(row.s) & " *
                "$(row.t) & " *
                "$(row.Method) & " *
                "$(fmt_float(row.RootGap; digits = 6)) & " *
                "$(fmt_float(row.Runtime; digits = 3)) & " *
                "$(fmt_float(row.KnitroTime; digits = 3)) & " *
                "$(fmt_int(row.Nodes)) & " *
                "$(fmt_int(row.FixedVars)) & " *
                "$(fmt_float(row.MaxIntGap; digits = 6)) & " *
                "$(fmt_float(row.AvgIntGap; digits = 6)) \\\\",
            )
        end

        println(io, raw"\hline")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end

    println("Saved table: ", filename)

    return nothing
end

# ============================================================
# Main
# ============================================================

df = load_bnb_table(bnb_file)
tab = build_main_results_table(df)

n_from_file = infer_n_from_filename(bnb_file)
kappa_from_file = infer_kappa_from_filename(bnb_file)

n_label =
    n_from_file === nothing ? string(first(unique(df.n))) : string(n_from_file)

kappa_label =
    kappa_from_file === nothing ? "unknown" : kappa_from_file

out_file =
    joinpath(tables_dir, "bnb_main_results_n$(n_label)_kappa$(kappa_label).txt")

write_latex_main_results_table(
    tab;
    filename = out_file,
    caption = "Main branch-and-bound results for \$n = $(n_label)\$ and \$\\kappa = $(kappa_label)\$.",
    label = "tab:bnb_main_results_n$(n_label)_kappa$(kappa_label)",
)

println()
println("Main BnB results table")
show(tab; allrows = true, allcols = true)
println()