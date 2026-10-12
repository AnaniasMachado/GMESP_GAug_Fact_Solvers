
using MKL
using Random
using CSV
using DataFrames
using MAT
using LinearAlgebra
using Printf
using JuMP
using KNITRO
using Hypatia
import MathOptInterface as MOI

include("../misc/util.jl")
include("../misc/dual_glinx.jl")
include("../misc/heuristics.jl")
include("../solvers/solver_knitro.jl")
include("../gscaling/gscaling_glinx.jl")

# ============================================================
# Generate instance
# ============================================================

rng = MersenneTwister(237)
A = 2 * rand(rng, 60, 3) .- 1
D = Diagonal([1.0, 1.0, 0.5])
C = Symmetric(I + 3 * A * D * A')

n = size(C, 1)
t = 2
s_list = 10:5:25

psi_min = eigmin(C)
atol = 1e-10
glinx_tol = 1e-2

MKL.set_num_threads(16)

output_dir = "data/generated/rank3"
mkpath(output_dir)

matwrite(joinpath(output_dir, "data60_rank3.mat"), Dict("C" => Matrix(C), "A" => A, "n" => n, "t" => t, "lambda_min" => psi_min, "seed" => 237))

# ============================================================
# Compare DDGFact+ and GLINX_gamma
# ============================================================

rows = NamedTuple[]

for s in s_list
    valid_min, _, _ = closed_form_general(C, s, t, psi_min; atol = atol)

    if valid_min
        z_min = trace_upper_bound(C, s, t, psi_min)
        method_min = "closed_form"
    else
        _, z_min = aug_ddfact_gmesp(C, s, t, psi_min; atol = atol)
        method_min = "KNITRO"
    end

    valid_max, psi_max, _, _ = largest_closed_form_certificate_psi(C, s, t; atol = atol)

    if valid_max
        z_max = trace_upper_bound(C, s, t, psi_max)
        method_max = "closed_form"
    else
        psi_max = psi_min
        z_max = z_min
        method_max = method_min
    end

    gamma_opt = compute_gamma_linx_armijo(C, s, t; tol = glinx_tol)
    _, _, z_glinx, _ = linx_gmesp(C, s, t; gamma = gamma_opt, return_model = true)

    push!(rows, (
        s = s,
        psi_min = psi_min,
        closed_form_min = valid_min,
        ddgfact_plus_min = z_min,
        method_min = method_min,
        psi_max = psi_max,
        closed_form_max = valid_max,
        ddgfact_plus_max = z_max,
        method_max = method_max,
        glinx_bound = z_glinx,
        difference_min = z_glinx - z_min,
        difference_max = z_glinx - z_max,
    ))

    @printf("\ns = %d\n", s)
    @printf("  psi_min = %.10f | closed form = %s | DDGFact+ = %.10f (%s)\n",
            psi_min, string(valid_min), z_min, method_min)
    @printf("  psi_max = %.10f | closed form = %s | DDGFact+ = %.10f (%s)\n",
            psi_max, string(valid_max), z_max, method_max)
    @printf("  GLINX = %.10f | difference_min = %+.10f | difference_max = %+.10f\n",
            z_glinx, z_glinx - z_min, z_glinx - z_max)
end

results = DataFrame(rows)

filepath = joinpath(output_dir, "root_comparison.csv")
CSV.write(filepath, results)

println("\nSaved: $filepath")
