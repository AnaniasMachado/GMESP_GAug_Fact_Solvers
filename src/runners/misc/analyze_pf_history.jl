using CSV
using DataFrames
using Plots

# ============================================================
# Settings
# ============================================================

n = 63
kappa = 4
run_tag = "projection_free"

history_filepath =
    "results/results_$(run_tag)_history_n$(n)_kappa$(kappa).csv"

plots_dir = "plots"
mkpath(plots_dir)

# ============================================================
# Load PF diagnostic history
# ============================================================

df = CSV.read(history_filepath, DataFrame)

df.method = string.(df.method)

sort!(df, [:method, :s, :t, :iter])

println("Loaded PF history:")
println("  rows = ", nrow(df))
println("  methods = ", unique(df.method))

# ============================================================
# Check required columns
# ============================================================

required_cols = Symbol[
    :method,
    :s,
    :t,
    :iter,
    :state_obj,
    :actual_obj,
]

missing_cols = setdiff(required_cols, propertynames(df))

if !isempty(missing_cols)
    error(
        "Missing columns in history CSV: $(missing_cols). " *
        "Rerun the main runner with the updated diagnostics."
    )
end

# ============================================================
# Save two scatter plots per algorithm
#
# For each algorithm:
#   1. state objective history
#   2. actual Knitro objective history
#
# Each (s,t) instance is plotted as a separate series.
# ============================================================

for method_name in unique(df.method)
    dfm = filter(row -> row.method == method_name, df)

    # ------------------------------------------------------------
    # State objective plot
    # ------------------------------------------------------------

    p_state = plot(
        xlabel = "Iteration",
        ylabel = "State objective",
        title = "$(method_name): state objective history",
        legend = :topright,
    )

    for sdf in groupby(dfm, [:s, :t])
        s_val = first(sdf.s)
        t_val = first(sdf.t)

        scatter!(
            p_state,
            sdf.iter,
            sdf.state_obj;
            label = "s=$(s_val), t=$(t_val)",
            markersize = 2,
            markerstrokewidth = 0,
        )
    end

    savepath_state = joinpath(
        plots_dir,
        "scatter_state_obj_by_instance_$(method_name)_n$(n)_kappa$(kappa).png",
    )

    savefig(p_state, savepath_state)

    println("Saved plot: ", savepath_state)

    # ------------------------------------------------------------
    # Actual objective plot
    # ------------------------------------------------------------

    p_actual = plot(
        xlabel = "Iteration",
        ylabel = "Actual objective",
        title = "$(method_name): actual objective history",
        legend = :topright,
    )

    for sdf in groupby(dfm, [:s, :t])
        s_val = first(sdf.s)
        t_val = first(sdf.t)

        scatter!(
            p_actual,
            sdf.iter,
            sdf.actual_obj;
            label = "s=$(s_val), t=$(t_val)",
            markersize = 2,
            markerstrokewidth = 0,
        )
    end

    savepath_actual = joinpath(
        plots_dir,
        "scatter_actual_obj_by_instance_$(method_name)_n$(n)_kappa$(kappa).png",
    )

    savefig(p_actual, savepath_actual)

    println("Saved plot: ", savepath_actual)
end