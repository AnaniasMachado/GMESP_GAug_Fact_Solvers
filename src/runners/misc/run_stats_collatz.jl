using CSV
using DataFrames
using Printf

# ============================================================
# Input / output
# ============================================================
input_filepath = "./results/bnb_t1_collatz_data63_n32_s2_to_31.csv"

output_filepath = replace(
    input_filepath,
    ".csv" => "_enumeration_ratios.csv",
)

# ============================================================
# Read table
# ============================================================
df = CSV.read(input_filepath, DataFrame)

# ============================================================
# Helpers
# ============================================================
function _safe_binomial(n::Integer, s::Integer)
    return binomial(BigInt(n), BigInt(s))
end

function _ratio_float(a, b)
    return Float64(BigFloat(a) / BigFloat(b))
end

function _percent_float(a, b)
    return 100.0 * _ratio_float(a, b)
end

# ============================================================
# Compute enumeration comparisons
# ============================================================
df.enumeration = [
    _safe_binomial(row.n, row.s)
    for row in eachrow(df)
]

df.nodes_over_enumeration = [
    _ratio_float(row.bnb_nodes, row.enumeration)
    for row in eachrow(df)
]

df.nodes_over_enumeration_percent = [
    _percent_float(row.bnb_nodes, row.enumeration)
    for row in eachrow(df)
]

df.int_sols_over_enumeration = [
    _ratio_float(row.bnb_n_int_sols, row.enumeration)
    for row in eachrow(df)
]

df.int_sols_over_enumeration_percent = [
    _percent_float(row.bnb_n_int_sols, row.enumeration)
    for row in eachrow(df)
]

df.nodes_minus_enumeration = [
    BigInt(row.bnb_nodes) - row.enumeration
    for row in eachrow(df)
]

df.nodes_better_than_enumeration = [
    BigInt(row.bnb_nodes) < row.enumeration
    for row in eachrow(df)
]

df.int_sols_better_than_enumeration = [
    BigInt(row.bnb_n_int_sols) < row.enumeration
    for row in eachrow(df)
]

# ============================================================
# Compact display table
# ============================================================
summary_cols = [
    :s,
    :enumeration,
    :bnb_nodes,
    :bnb_n_int_sols,
    :nodes_over_enumeration_percent,
    :int_sols_over_enumeration_percent,
    :nodes_minus_enumeration,
    :nodes_better_than_enumeration,
]

summary_df = df[:, summary_cols]

println()
println("="^100)
println("BnB nodes versus full enumeration")
println("="^100)

for row in eachrow(summary_df)
    @printf(
        "s=%2d  enum=%12s  nodes=%10d  int_sols=%10d  nodes/enum=%8.4f%%  int/enum=%8.4f%%  nodes-enum=%12s  better=%s\n",
        row.s,
        string(row.enumeration),
        row.bnb_nodes,
        row.bnb_n_int_sols,
        row.nodes_over_enumeration_percent,
        row.int_sols_over_enumeration_percent,
        string(row.nodes_minus_enumeration),
        string(row.nodes_better_than_enumeration),
    )
end

println("="^100)

# ============================================================
# Aggregate statistics
# ============================================================
n_rows = nrow(df)

n_nodes_better = count(df.nodes_better_than_enumeration)
n_int_better = count(df.int_sols_better_than_enumeration)

worst_nodes_ratio_idx = argmax(df.nodes_over_enumeration)
best_nodes_ratio_idx = argmin(df.nodes_over_enumeration)

println()
println("Aggregate statistics")
println("-"^100)
println("rows:                                      ", n_rows)
println("nodes better than enumeration:             ", n_nodes_better, " / ", n_rows)
println("integer solutions better than enumeration: ", n_int_better, " / ", n_rows)

println()
println("Best node ratio:")
@printf(
    "s=%d, nodes/enum=%.6f%%, nodes=%d, enum=%s\n",
    df.s[best_nodes_ratio_idx],
    df.nodes_over_enumeration_percent[best_nodes_ratio_idx],
    df.bnb_nodes[best_nodes_ratio_idx],
    string(df.enumeration[best_nodes_ratio_idx]),
)

println()
println("Worst node ratio:")
@printf(
    "s=%d, nodes/enum=%.6f%%, nodes=%d, enum=%s\n",
    df.s[worst_nodes_ratio_idx],
    df.nodes_over_enumeration_percent[worst_nodes_ratio_idx],
    df.bnb_nodes[worst_nodes_ratio_idx],
    string(df.enumeration[worst_nodes_ratio_idx]),
)

# ============================================================
# Save enriched table
# ============================================================
CSV.write(output_filepath, df)

println()
println("Saved enriched table to:")
println(output_filepath)