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

function infer_s_from_filename(path::AbstractString)
    m = match(r"s(\d+)", basename(path))
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function infer_t_from_filename(path::AbstractString)
    m = match(r"t(\d+)", basename(path))
    return m === nothing ? nothing : parse(Int, m.captures[1])
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

function method_order(method::AbstractString)
    if method == raw"DDGFact"
        return 1
    elseif method == raw"DDGFact$^+$"
        return 2
    elseif method == raw"BFGS"
        return 3
    elseif method == raw"Prox Knitro"
        return 4
    else
        return 100
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
        Method = String[],
        RootGap = Float64[],
        Runtime = Float64[],
        KnitroTime = Float64[],
        Nodes = Int[],
        FixedVars = Int[],
        MaxIntGap = Float64[],
        AvgIntGap = Float64[],
        _order = Int[],
    )

    for row in eachrow(df)
        method = method_name(row)

        push!(
            rows,
            (
                Method = method,
                RootGap = Float64(row.bnb_root_gap),
                Runtime = Float64(row.bnb_runtime),
                KnitroTime = Float64(row.bnb_knitro_time),
                Nodes = Int(row.bnb_nodes),
                FixedVars = Int(row.n_fixed_total_bnb),
                MaxIntGap = Float64(row.bnb_int_gap_max),
                AvgIntGap = Float64(row.bnb_int_gap_avg),
                _order = method_order(method),
            ),
        )
    end

    sort!(rows, :_order)
    select!(rows, Not(:_order))

    return rows
end

function write_latex_main_results_table(
    tab::DataFrame;
    filename::AbstractString,
    n_value::Int,
    s_value::Int,
    t_value::Int,
)
    open(filename, "w") do io
        println(io, raw"\begin{table}[!ht]")
        println(io, raw"    \centering")
        println(io, raw"    \scriptsize")
        println(io, raw"    \setlength{\tabcolsep}{3pt}")
        println(io, raw"    \renewcommand{\arraystretch}{0.95}")
        println(io, "    \\caption{Main branch-and-bound results for \$n = $(n_value)\$, \$s = $(s_value)\$ and \$t = $(t_value)\$.}")
        println(io, "    \\label{tab:bnb_main_results_n$(n_value)_s$(s_value)_t$(t_value)}")
        println(io, raw"    \begin{tabular}{l|rrr|rr|rr}")
        println(io, raw"    \hline")
        println(io, raw"    \textbf{Method}")
        println(io, raw"    & \textbf{Root gap}")
        println(io, raw"    & \textbf{Runtime}")
        println(io, raw"    & \textbf{Knitro time}")
        println(io, raw"    & \textbf{Nodes}")
        println(io, raw"    & \textbf{Fixed vars.}")
        println(io, raw"    & \textbf{Max int. gap}")
        println(io, raw"    & \textbf{Avg int. gap} \\")
        println(io, raw"    \hline")

        for row in eachrow(tab)
            println(
                io,
                "    $(row.Method) & " *
                "$(fmt_float(row.RootGap; digits = 6)) & " *
                "$(fmt_float(row.Runtime; digits = 3)) & " *
                "$(fmt_float(row.KnitroTime; digits = 3)) & " *
                "$(fmt_int(row.Nodes)) & " *
                "$(fmt_int(row.FixedVars)) & " *
                "$(fmt_float(row.MaxIntGap; digits = 6)) & " *
                "$(fmt_float(row.AvgIntGap; digits = 6)) \\\\",
            )
        end

        println(io, raw"    \hline")
        println(io, raw"    \end{tabular}")
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
s_from_file = infer_s_from_filename(bnb_file)
t_from_file = infer_t_from_filename(bnb_file)

n_value =
    n_from_file === nothing ? Int(first(unique(df.n))) : n_from_file

s_value =
    s_from_file === nothing ? Int(first(unique(df.s))) : s_from_file

t_value =
    t_from_file === nothing ? Int(first(unique(df.t))) : t_from_file

out_file =
    joinpath(
        tables_dir,
        "bnb_main_results_n$(n_value)_s$(s_value)_t$(t_value).txt",
    )

write_latex_main_results_table(
    tab;
    filename = out_file,
    n_value = n_value,
    s_value = s_value,
    t_value = t_value,
)

println()
println("Main BnB results table")
show(tab; allrows = true, allcols = true)
println()