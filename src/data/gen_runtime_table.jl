using CSV
using DataFrames
using Printf

csv_dir = "results/journal_ddgfactplus/csv_tables"
latex_dir = "results/journal_ddgfactplus/latex_tables"

n_vals = [63, 90]
kappa_pairs = [(2, 4), (6, 8)]

mkpath(latex_dir)


function format_runtime(x)
    if x == 0
        return "0.00"
    elseif abs(x) >= 0.01 && abs(x) < 1000
        exponent = floor(Int, log10(abs(x)))
        decimals = max(0, 2 - exponent)
        return @sprintf("%.*f", decimals, x)
    else
        exponent = floor(Int, log10(abs(x)))
        mantissa = x / 10.0^exponent
        return @sprintf("%.2fe%d", mantissa, exponent)
    end
end


for n in n_vals
    for (kappa1, kappa2) in kappa_pairs

        csv_filepath1 = joinpath(
            csv_dir,
            "results_root_gap_n$(n)_kappa$(kappa1).csv"
        )

        csv_filepath2 = joinpath(
            csv_dir,
            "results_root_gap_n$(n)_kappa$(kappa2).csv"
        )

        df1 = CSV.read(csv_filepath1, DataFrame)
        df2 = CSV.read(csv_filepath2, DataFrame)

        nrow(df1) == nrow(df2) ||
            error("Different numbers of rows for kappa=$kappa1 and kappa=$kappa2.")

        latex_filepath = joinpath(
            latex_dir,
            "runtimes_n$(n)_kappa$(kappa1)_$(kappa2).txt"
        )

        open(latex_filepath, "w") do io
            println(io, "\\begin{table}[htbp]")
            println(io, "\\centering")
            println(io, "\\begin{tabular}{cc|ccc|ccc}")
            println(io, "\\hline")

            println(
                io,
                "\\multicolumn{2}{c|}{} & " *
                "\\multicolumn{3}{c|}{\$\\kappa = $(kappa1)\$} & " *
                "\\multicolumn{3}{c}{\$\\kappa = $(kappa2)\$} \\\\"
            )

            println(io, "\\hline")

            println(
                io,
                "\$n\$ & \$s\$ & " *
                "DDGFact & DDGFact+ & GLinx & " *
                "DDGFact & DDGFact+ & GLinx \\\\"
            )

            println(io, "\\hline")

            for i in 1:nrow(df1)
                row1 = df1[i, :]
                row2 = df2[i, :]

                row1.n == row2.n ||
                    error("n does not match in row $i.")

                row1.s == row2.s ||
                    error("s does not match in row $i.")

                ddgfact1 = format_runtime(row1.ddgfact_runtime)
                ddgfact_plus1 = format_runtime(row1.ddgfact_plus_runtime)
                glinx1 = format_runtime(row1.glinx_runtime)

                ddgfact2 = format_runtime(row2.ddgfact_runtime)
                ddgfact_plus2 = format_runtime(row2.ddgfact_plus_runtime)
                glinx2 = format_runtime(row2.glinx_runtime)

                println(
                    io,
                    "$(row1.n) & $(row1.s) & " *
                    "$ddgfact1 & $ddgfact_plus1 & $glinx1 & " *
                    "$ddgfact2 & $ddgfact_plus2 & $glinx2 \\\\"
                )
            end

            println(io, "\\hline")
            println(io, "\\end{tabular}")

            println(
                io,
                "\\caption{Runtimes (in seconds) for \$n = $(n)\$ and " *
                "\$\\kappa \\in \\{$(kappa1),$(kappa2)\\}\$.}"
            )

            println(
                io,
                "\\label{tab:runtimes_n$(n)_kappa$(kappa1)_$(kappa2)}"
            )

            println(io, "\\end{table}")
        end

        println("Written: $latex_filepath")
    end
end