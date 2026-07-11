using DataFrames
using Statistics
using CSV

input_path = "./results/bnb_data63_n32_s16_t12.csv"
output_path = "./results/bnb_calib_stats.csv"

df = CSV.read(input_path, DataFrame)

df_calib = filter(row -> !ismissing(row.recalibrate_k) && row.recalibrate_k != "", df)

summary_by_method = combine(
    groupby(df_calib, [:calibration_method]),

    :bnb_upsilon_calibration_time => sum => :total_calibration_time,
    :bnb_nodes => sum => :total_nodes,

    :bnb_gap => mean => :avg_final_bnb_gap,
    :bnb_int_gap_max => mean => :avg_int_gap_max,
    :bnb_int_gap_avg => mean => :avg_int_gap_avg,
) |> x -> transform(
    x,
    [:total_calibration_time, :total_nodes] =>
        ByRow((t, nodes) -> t / nodes) =>
        :avg_calibration_time_per_node
)

println(summary_by_method)

CSV.write(output_path, summary_by_method)