using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
import MathOptInterface as MOI

include("../misc/util.jl")
include("../misc/heuristics.jl")
include("../solvers/solver_knitro.jl")

include("../gscaling/gscaling_util.jl")
include("../gscaling/gscaling_bfgs.jl")
include("../gscaling/gscaling_prox.jl")
include("../gscaling/gscaling_knitro.jl")

include("../misc/dual.jl")
include("../misc/var_fixing.jl")


# ============================================================
# Problem data
# ============================================================

n = 63
kappa = 4

# Full run:
# s_vals = [s for s in (kappa + 1):(n - 1)]

# Test run:
s_vals = [s for s in (kappa + 1):(kappa + 5)]

matfile = matopen("data/data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
Csym = Symmetric(C)

atol = 1e-10

# Initial unscaled psi for DDGFact+.
psi = eigmin(Csym) - atol


# ============================================================
# Method selection
#
# Comment out any line below to skip that method.
# The order here is also the order of the gap/runtime columns.
# ============================================================

selected_methods = Symbol[
    :ddgfact,
    :ddgfact_plus,
    :spectral,

    :custom_bfgs,
    :ppa_one,

    :knitro,
]

run_method(m::Symbol) = m in selected_methods


# ============================================================
# DDGFact+_Upsilon relaxation solver tolerances
# Used inside each calibration oracle evaluation.
# ============================================================

relax_knitro_outlev = nothing
relax_knitro_opttol = 1e-8
relax_knitro_feastol = 1e-5


# ============================================================
# Custom BFGS calibration parameters
# ============================================================

bfgs_max_iter = 50
bfgs_grad_tol = 1e-2
bfgs_step_tol = 1e-8

bfgs_alpha0 = 1.0
bfgs_alpha_min = 1e-10
bfgs_alpha_decay = 0.75
bfgs_armijo_c1 = 1e-6
bfgs_max_backtracks = 50
bfgs_curvature_tol = 1e-12

psi_margin = 1e-7
psi_floor = 0.0

bfgs_max_theta_norm = 20.0
verbose_bfgs = false


# ============================================================
# One-step PPA calibration parameters
# ============================================================

ppa_rho = 1e3

ppa_theta_perturbation = 1e-2
ppa_center_initial_theta = false
ppa_theta_bound = 20.0

ppa_grad_tol = 1e-2
ppa_prox_obj_abs_tol = 1e-8
ppa_prox_step_tol = 1e-8
ppa_max_wall_time = Inf

ppa_knitro_feastol = 1e-6
ppa_knitro_opttol = 1e-2
ppa_knitro_xtol = 1e-4
ppa_knitro_ftol = 1e-5

ppa_knitro_maxtime_real = Inf
ppa_knitro_algorithm = nothing
ppa_knitro_bar_murule = nothing
ppa_knitro_honorbnds = 1
ppa_knitro_outlev = 0

ppa_cache_digits = 6
ppa_diagnostics = false
verbose_ppa = false


# ============================================================
# Direct Knitro calibration parameters
#
# Solves:
#   min_theta V(theta)
#
# where V(theta) = DDGFact+_Upsilon calibration objective.
# ============================================================

knitro_calib_theta_bound = 20.0

knitro_calib_feastol = 1e-6
knitro_calib_opttol = 1e-2
knitro_calib_xtol = 1e-4
knitro_calib_ftol = 1e-5

knitro_calib_maxtime_real = Inf
knitro_calib_algorithm = nothing
knitro_calib_bar_murule = nothing
knitro_calib_honorbnds = 1
knitro_calib_outlev = 0

knitro_calib_cache_digits = 6
knitro_calib_diagnostics = false
verbose_knitro_calib = false


# ============================================================
# Data collection
# ============================================================

run_tag = "knitro_calib"

mkpath("results")

results_filepath = "results/results_$(run_tag)_n$(n)_kappa$(kappa).csv"

rows = Dict{Symbol,Any}[]


# ============================================================
# Helpers
# ============================================================

function _make_theta0_calib(n::Int)
    Random.seed!(1)

    theta0_calib =
        ppa_theta_perturbation == 0.0 ?
        zeros(Float64, n) :
        ppa_theta_perturbation .* randn(n)

    if ppa_center_initial_theta
        theta0_calib .-= mean(theta0_calib)
    end

    if isfinite(ppa_theta_bound)
        theta0_calib .= clamp.(theta0_calib, -ppa_theta_bound, ppa_theta_bound)
    end

    return theta0_calib
end


function _needs_theta0()
    return any(
        run_method(m) for m in (
            :ppa_one,

            :knitro_calib,
        )
    )
end


function _print_method_line(m::Symbol, gap_by_method, runtime_by_method)
    @printf(
        "  %-22s gap = % .6e    runtime = %.2fs\n",
        string(m),
        gap_by_method[m],
        runtime_by_method[m],
    )
    return nothing
end


# ============================================================
# Main loop
# ============================================================

for s in s_vals
    t = s - kappa

    println("Running n=$n, s=$s, t=$t")
    flush(stdout)

    obj_by_method = Dict{Symbol,Float64}()
    gap_by_method = Dict{Symbol,Float64}()
    runtime_by_method = Dict{Symbol,Float64}()

    # ============================================================
    # Local search, always needed to compute gaps
    # ============================================================

    runtime_ls = @elapsed begin
        x_ls, z_ls = run_all_LS(Csym, s, t)
    end

    theta0_calib = _needs_theta0() ? _make_theta0_calib(n) : nothing


    # ============================================================
    # DDGFact, non-augmented
    # ============================================================

    if run_method(:ddgfact)
        runtime_by_method[:ddgfact] = @elapsed begin
            x_ddgfact, z_ddgfact = ddfact_gmesp(
                Csym,
                s,
                t;
                atol = atol,
            )
        end

        obj_by_method[:ddgfact] = z_ddgfact
    end


    # ============================================================
    # DDGFact+, augmented
    # ============================================================

    if run_method(:ddgfact_plus)
        runtime_by_method[:ddgfact_plus] = @elapsed begin
            x_ddgfact_plus, z_ddgfact_plus = aug_ddfact_gmesp(
                Csym,
                s,
                t,
                psi;
                atol = atol,
            )
        end

        obj_by_method[:ddgfact_plus] = z_ddgfact_plus
    end


    # ============================================================
    # Spectral bound
    # ============================================================

    if run_method(:spectral)
        runtime_by_method[:spectral] = @elapsed begin
            z_spec = spectral_bound(Csym, t)
        end

        obj_by_method[:spectral] = z_spec
    end


    # ============================================================
    # DDGFact+_Upsilon, custom BFGS calibration
    # ============================================================

    if run_method(:custom_bfgs)
        runtime_by_method[:custom_bfgs] = @elapsed begin
            result_custom_bfgs = calibrate_upsilon_bfgs_ddfactplus(
                Csym,
                s,
                t;
                atol = atol,
                max_iter = bfgs_max_iter,
                grad_tol = bfgs_grad_tol,
                step_tol = bfgs_step_tol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
                alpha0 = bfgs_alpha0,
                alpha_min = bfgs_alpha_min,
                alpha_decay = bfgs_alpha_decay,
                armijo_c1 = bfgs_armijo_c1,
                curvature_tol = bfgs_curvature_tol,
                max_backtracks = bfgs_max_backtracks,
                max_theta_norm = bfgs_max_theta_norm,
                psi_derivative = true,
                t1_reformulation = false,

                knitro_outlev = relax_knitro_outlev,
                knitro_opttol = relax_knitro_opttol,
                knitro_feastol = relax_knitro_feastol,

                verbose = verbose_bfgs,
            )
        end

        obj_by_method[:custom_bfgs] = result_custom_bfgs.obj
    end


    # ============================================================
    # DDGFact+_Upsilon, one PPA iteration with Knitro subproblem
    # ============================================================

    if run_method(:ppa_one)
        runtime_by_method[:ppa_one] = @elapsed begin
            result_ppa_one = calibrate_upsilon_ppa_ddfactplus(
                Csym,
                s,
                t;
                J1 = Int[],
                J0 = Int[],
                theta0 = theta0_calib,
                k = 1,

                rho = ppa_rho,
                grad_tol = ppa_grad_tol,
                prox_obj_abs_tol = ppa_prox_obj_abs_tol,
                prox_step_tol = ppa_prox_step_tol,
                max_wall_time = ppa_max_wall_time,

                theta_perturbation = 0.0,
                center_initial_theta = false,
                theta_bound = ppa_theta_bound,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,
                atol = atol,

                relax_knitro_outlev = relax_knitro_outlev,
                relax_knitro_opttol = relax_knitro_opttol,
                relax_knitro_feastol = relax_knitro_feastol,

                knitro_feastol = ppa_knitro_feastol,
                knitro_opttol = ppa_knitro_opttol,
                knitro_xtol = ppa_knitro_xtol,
                knitro_ftol = ppa_knitro_ftol,
                knitro_maxtime_real = ppa_knitro_maxtime_real,
                knitro_algorithm = ppa_knitro_algorithm,
                knitro_bar_murule = ppa_knitro_bar_murule,
                knitro_honorbnds = ppa_knitro_honorbnds,
                knitro_outlev = ppa_knitro_outlev,

                cache_digits = ppa_cache_digits,
                diagnostics = ppa_diagnostics,
                verbose = verbose_ppa,
            )
        end

        obj_by_method[:ppa_one] = result_ppa_one.obj
    end


    # ============================================================
    # DDGFact+_Upsilon, direct Knitro calibration
    #
    # Solves:
    #   min_theta V(theta)
    #
    # using the DDGFact+_Upsilon objective and subgradient callbacks.
    # ============================================================

    if run_method(:knitro)
        runtime_by_method[:knitro] = @elapsed begin
            result_knitro_calib = calibrate_upsilon_knitro_ddfactplus(
                Csym,
                s,
                t;
                J1 = Int[],
                J0 = Int[],
                theta0 = theta0_calib,

                theta_perturbation = 0.0,
                center_initial_theta = false,
                theta_bound = knitro_calib_theta_bound,

                psi_margin = psi_margin,
                psi_floor = psi_floor,
                psi_derivative = true,
                t1_reformulation = false,
                atol = atol,

                relax_knitro_outlev = relax_knitro_outlev,
                relax_knitro_opttol = relax_knitro_opttol,
                relax_knitro_feastol = relax_knitro_feastol,

                knitro_feastol = knitro_calib_feastol,
                knitro_opttol = knitro_calib_opttol,
                knitro_xtol = knitro_calib_xtol,
                knitro_ftol = knitro_calib_ftol,
                knitro_maxtime_real = knitro_calib_maxtime_real,
                knitro_algorithm = knitro_calib_algorithm,
                knitro_bar_murule = knitro_calib_bar_murule,
                knitro_honorbnds = knitro_calib_honorbnds,
                knitro_outlev = knitro_calib_outlev,

                cache_digits = knitro_calib_cache_digits,
                diagnostics = knitro_calib_diagnostics,
                verbose = verbose_knitro_calib,
            )
        end

        obj_by_method[:knitro] = result_knitro_calib.obj
    end


    # ============================================================
    # Compute gaps
    # ============================================================

    for m in selected_methods
        if haskey(obj_by_method, m)
            gap_by_method[m] = obj_by_method[m] - z_ls
        end
    end


    # ============================================================
    # Build row with explicit logical order
    #
    # Order:
    #   n, s, t,
    #   all gaps,
    #   all runtimes.
    # ============================================================

    row = Dict{Symbol,Any}()

    row[:n] = n
    row[:s] = s
    row[:t] = t

    for m in selected_methods
        if haskey(gap_by_method, m)
            row[Symbol("$(m)_gap")] = gap_by_method[m]
        end
    end

    for m in selected_methods
        if haskey(runtime_by_method, m)
            row[Symbol("$(m)_runtime")] = runtime_by_method[m]
        end
    end

    row[:local_search_runtime] = runtime_ls

    push!(rows, row)


    # ============================================================
    # Minimal console output
    # ============================================================

    println("done s=$s t=$t")
    @printf("  local_search_runtime = %.2fs\n", runtime_ls)

    for m in selected_methods
        if haskey(gap_by_method, m) && haskey(runtime_by_method, m)
            _print_method_line(m, gap_by_method, runtime_by_method)
        end
    end

    println()
    flush(stdout)
end


# ============================================================
# Build DataFrame with explicit column order
# ============================================================

gap_cols = Symbol[]
runtime_cols = Symbol[]

for m in selected_methods
    push!(gap_cols, Symbol("$(m)_gap"))
end

for m in selected_methods
    push!(runtime_cols, Symbol("$(m)_runtime"))
end

push!(runtime_cols, :local_search_runtime)

column_order = vcat(
    [:n, :s, :t],
    gap_cols,
    runtime_cols,
)

df_results = DataFrame(rows)

df_results = df_results[:, column_order]

CSV.write(results_filepath, df_results)

println("Saved results to: $results_filepath")