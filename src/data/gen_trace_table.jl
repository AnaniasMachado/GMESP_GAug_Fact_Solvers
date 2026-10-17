
using CSV
using DataFrames
using Printf

# ============================================================
# Configuration
# ============================================================

data_n = 60
n = 60
t = 2

s_values = [5, 10, 15, 20, 25]

gap_decimals = 3
other_decimals = 1

csv_filepath = joinpath(
    "results/journal_ddgfactplus",
    "bnb_data$(data_n)_n$(n)_t$(t)_rank3.csv"
)

latex_dir = "results/journal_ddgfactplus/latex_tables"
mkpath(latex_dir)

latex_filepath = joinpath(
    latex_dir,
    "bnb_comparison_n$(n)_t$(t)_rank3.txt"
)

# ============================================================
# Read CSV
# ============================================================

df = CSV.read(csv_filepath, DataFrame)

required_cols = [
    :s,
    :relaxation,
    :plus_cf,
    :bnb_gap,
    :bnb_nodes,
    :n_fixed_total_bnb,
    :bnb_runtime,
    :closed_form_uses,
    :trace_prunes,
]

missing_cols = setdiff(required_cols, propertynames(df))

isempty(missing_cols) ||
    error("Missing CSV columns: $missing_cols")

# ============================================================
# Formatting
# ============================================================

function fmt_decimal(x, digits)
    ismissing(x) && return "--"
    isfinite(x) || return "--"
    return @sprintf("%.*f", digits, Float64(x))
end

function fmt_integer(x)
    ismissing(x) && return "--"
    return string(round(Int, x))
end

function stat(row, col)
    row === nothing && return "--"

    if col == :bnb_gap
        return fmt_decimal(row[col], gap_decimals)
    elseif col == :bnb_runtime
        return fmt_decimal(row[col], other_decimals)
    else
        return fmt_integer(row[col])
    end
end

# ============================================================
# Find result
# ============================================================

function get_result(df, s, method, plus_cf)
    rows = df[
        (df.s .== s) .&
        (df.relaxation .== method) .&
        (df.plus_cf .== plus_cf),
        :
    ]

    if nrow(rows) == 0
        @warn "Missing result" s method plus_cf
        return nothing
    elseif nrow(rows) > 1
        @warn "Multiple results; using last row" s method plus_cf
    end

    return rows[end, :]
end

# ============================================================
# Write LaTeX table
# ============================================================

open(latex_filepath, "w") do io

    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\resizebox{\\columnwidth}{!}{%")
    println(io, "\\begin{tabular}{c|ccc|cccc|cccccc}")
    println(io, "\\hline")

    # First header layer: bounds
    println(
        io,
        "\$s\$ & " *
        "\\multicolumn{3}{c|}{\$\\mathrm{Trace}\$} & " *
        "\\multicolumn{4}{c|}{\$\\mathrm{DDGFact}^{+}\$} & " *
        "\\multicolumn{6}{c}{\$\\mathrm{DDGFact}^{+}\\text{ (CF)}\$} \\\\"
    )

    println(io, "\\hline")

    # Second header layer: statistics
    println(
        io,
        " & Gap & Nodes & Time & " *
        "Gap & Nodes & Fixed & Time & " *
        "Gap & Nodes & Fixed & Time & CF & Prunes \\\\"
    )

    println(io, "\\hline")

    # Data rows
    for s in s_values

        trace = get_result(df, s, "Trace", false)

        ddgplus = get_result(
            df, s, "DDGFactplus", false
        )

        ddgplus_cf = get_result(
            df, s, "DDGFactplus_CF", true
        )

        values = String[
            string(s),

            # Trace
            stat(trace, :bnb_gap),
            stat(trace, :bnb_nodes),
            stat(trace, :bnb_runtime),

            # DDGFact+
            stat(ddgplus, :bnb_gap),
            stat(ddgplus, :bnb_nodes),
            stat(ddgplus, :n_fixed_total_bnb),
            stat(ddgplus, :bnb_runtime),

            # DDGFact+ (CF)
            stat(ddgplus_cf, :bnb_gap),
            stat(ddgplus_cf, :bnb_nodes),
            stat(ddgplus_cf, :n_fixed_total_bnb),
            stat(ddgplus_cf, :bnb_runtime),
            stat(ddgplus_cf, :closed_form_uses),
            stat(ddgplus_cf, :trace_prunes),
        ]

        println(io, join(values, " & ") * " \\\\")
    end

    println(io, "\\hline")
    println(io, "\\end{tabular}")
    println(io, "}")

    println(io, "\\caption{Placeholder caption.}")
    println(
        io,
        "\\label{tab:bnb_comparison_n$(n)_t$(t)_rank3}"
    )
    println(io, "\\end{table}")
end

println("Written: $latex_filepath")
