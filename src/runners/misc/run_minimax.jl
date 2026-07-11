using Random
using MAT
using CSV
using DataFrames
using LinearAlgebra
using JuMP
using KNITRO
using Printf
import MathOptInterface as MOI

include("../../misc/util.jl")
include("../../misc/heuristics.jl")
include("../../solvers/solver_knitro.jl")

include("../../gscaling/gscaling_util.jl")
include("../../gscaling/gscaling_bfgs.jl")
include("../../gscaling/gscaling_prox.jl")
include("../../gscaling/gscaling_projection_free.jl")

include("../../misc/dual.jl")
include("../../misc/var_fixing.jl")


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

    :pf_lmo_lmo,
    :pf_lmo_po,
    :pf_po_lmo,
]

run_method(m::Symbol) = m in selected_methods

pf_methods = Symbol[
    :pf_lmo_lmo,
    :pf_lmo_po,
    :pf_po_lmo,
]

selected_pf_methods = Symbol[m for m in pf_methods if run_method(m)]


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
# Projection-free calibration parameters
# ============================================================

pf_max_iter = 500
pf_min_iter = 100
pf_iteration_power = 1.5

pf_theta_bound = 20.0

pf_psi_derivative = true

# Algorithm 1: LMO-LMO
pf_lmo_lmo_tau0 = 0.10
pf_lmo_lmo_tau_power = 1.05
pf_lmo_lmo_beta0 = 0.18
pf_lmo_lmo_beta_power = 1.0 / 4.0
pf_lmo_lmo_Lqq_hat = 0.52

# Algorithm 2: LMO-PO
pf_lmo_po_tau0 = 0.0525
pf_lmo_po_tau_power = 1.00
pf_lmo_po_beta0 = 0.55
pf_lmo_po_beta_power = 0.40
pf_lmo_po_Lqq_hat = 0.65

# Algorithm 3: PO-LMO
pf_po_lmo_tau0 = 1.00
pf_po_lmo_tau_power = 2.0 / 3.0
pf_po_lmo_beta0 = 0.10
pf_po_lmo_beta_power = 1.0 / 6.0
pf_po_lmo_Lqq_hat = 0.50

# Gurobi options for LMO/projection over the (x,y) polytope.
pf_gurobi_output_flag = 0
pf_gurobi_opttol = 1e-8
pf_gurobi_feastol = 1e-8

pf_diagnostics = true
pf_verbose = false


# ============================================================
# Data collection
# ============================================================

run_tag = "projection_free"

mkpath("results")

results_filepath = "results/results_$(run_tag)_n$(n)_kappa$(kappa).csv"
pf_history_filepath = "results/results_$(run_tag)_history_n$(n)_kappa$(kappa).csv"

rows = Dict{Symbol,Any}[]
pf_history_rows = Dict{Symbol,Any}[]


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
            :pf_lmo_lmo,
            :pf_lmo_po,
            :pf_po_lmo,
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


function _pf_common_kwargs(theta0_calib)
    return (
        J1 = Int[],
        J0 = Int[],
        theta0 = theta0_calib,

        max_iter = pf_max_iter,
        min_iter = pf_min_iter,
        iteration_power = pf_iteration_power,

        theta_bound = pf_theta_bound,

        psi_margin = psi_margin,
        psi_floor = psi_floor,
        psi_derivative = pf_psi_derivative,
        atol = atol,

        gurobi_output_flag = pf_gurobi_output_flag,
        gurobi_opttol = pf_gurobi_opttol,
        gurobi_feastol = pf_gurobi_feastol,

        diagnostics = pf_diagnostics,
        verbose = pf_verbose,
    )
end


function _print_pf_summary(label::Symbol, result)
    println("$(label) algorithm = ", result.algorithm)
    println("$(label) tau0 = ", result.tau0)
    println("$(label) tau_power = ", result.tau_power)
    println("$(label) beta0 = ", result.beta0)
    println("$(label) beta_power = ", result.beta_power)
    println("$(label) Lqq_hat = ", result.Lqq_hat)

    println("$(label) max_iter = ", result.max_iter)
    println("$(label) adaptive_max_iter = ", result.adaptive_max_iter)
    println("$(label) min_iter = ", result.min_iter)
    println("$(label) iteration_power = ", result.iteration_power)
    println("$(label) free_n = ", result.free_n)
    println("$(label) fixed_n = ", result.fixed_n)

    println("$(label) iterations = ", result.iterations)
    println("$(label) stop_reason = ", result.stop_reason)

    if !isempty(result.history)
        println("$(label) last history = ", last(result.history))
    end

    return nothing
end


function _append_pf_history!(
    pf_history_rows,
    label::Symbol,
    result,
    n::Int,
    s::Int,
    t::Int,
)
    for h in result.history
        row = Dict{Symbol,Any}()

        row[:n] = n
        row[:s] = s
        row[:t] = t
        row[:method] = string(label)

        row[:max_iter] = result.max_iter
        row[:adaptive_max_iter] = result.adaptive_max_iter
        row[:min_iter] = result.min_iter
        row[:iteration_power] = result.iteration_power
        row[:free_n] = result.free_n
        row[:fixed_n] = result.fixed_n

        for name in propertynames(h)
            value = getproperty(h, name)
            row[name] = value isa Symbol ? string(value) : value
        end

        push!(pf_history_rows, row)
    end

    return nothing
end


function _fixed_bounds_from_J(n::Int, J1, J0)
    l = zeros(Float64, n)
    c = ones(Float64, n)

    J1v = sort(unique(collect(Int, J1)))
    J0v = sort(unique(collect(Int, J0)))

    l[J1v] .= 1.0
    c[J1v] .= 1.0

    l[J0v] .= 0.0
    c[J0v] .= 0.0

    return l, c
end


function _pf_primal_dual_gaps(
    C::Symmetric{<:Real,<:AbstractMatrix},
    pf_result,
    z_ls::Float64,
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    psi_margin::Float64,
    psi_floor::Float64,
    atol::Float64,
    relax_knitro_outlev,
    relax_knitro_opttol,
    relax_knitro_feastol,
)
    n = size(C, 1)

    theta = pf_result.theta
    gamma = exp.(theta)

    psi_val, lambda_min_val = max_feasible_psi(
        C,
        gamma;
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    F = scaled_factorize_matrix(
        C,
        gamma,
        psi_val;
        atol = atol,
    )

    x_primal, y_primal, primal_obj = aug_ddfact_upsilon_gmesp(
        C,
        gamma,
        s,
        t,
        psi_val;
        J1 = J1,
        x0 = pf_result.x,
        y0 = pf_result.y,
        atol = atol,
        knitro_outlev = relax_knitro_outlev,
        knitro_opttol = relax_knitro_opttol,
        knitro_feastol = relax_knitro_feastol,
    )

    l, c = _fixed_bounds_from_J(n, J1, J0)

    dual = DGFactplusUpsilon_dual_solution_from_DDGFactplusUpsilon_xy(
        pf_result.x,
        gamma,
        F,
        s,
        t,
        psi_val;
        yhat = pf_result.y,
        l = l,
        c = c,
        atol = atol,
        silent = true,
    )

    dual_obj = dual.objective_value

    return (
        primal_gap = primal_obj - z_ls,
        dual_gap = dual_obj - z_ls,

        primal_obj = primal_obj,
        dual_obj = dual_obj,

        theta = theta,
        gamma = gamma,
        psi = psi_val,
        lambda_min = lambda_min_val,

        dual_variables = (
            Theta = dual.Theta,
            upsilon = dual.upsilon,
            nu = dual.nu,
            eta = dual.eta,
            rho = dual.rho,
            tau = dual.tau,
            alpha = dual.alpha,
        ),
    )
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

    dual_gap_by_method = Dict{Symbol,Float64}()
    dual_variables_by_method = Dict{Symbol,Any}()

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
    # Projection-free, Algorithm 1: LMO-LMO
    # ============================================================

    if run_method(:pf_lmo_lmo)
        runtime_by_method[:pf_lmo_lmo] = @elapsed begin
            result_pf_lmo_lmo = calibrate_upsilon_projection_free_ddfactplus(
                Csym,
                s,
                t;
                _pf_common_kwargs(theta0_calib)...,

                algorithm = :lmo_lmo,

                tau0 = pf_lmo_lmo_tau0,
                tau_power = pf_lmo_lmo_tau_power,

                beta0 = pf_lmo_lmo_beta0,
                beta_power = pf_lmo_lmo_beta_power,

                Lqq_hat = pf_lmo_lmo_Lqq_hat,
            )
        end

        eval_pf_lmo_lmo = _pf_primal_dual_gaps(
            Csym,
            result_pf_lmo_lmo,
            z_ls,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            atol = atol,
            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
        )

        obj_by_method[:pf_lmo_lmo] = eval_pf_lmo_lmo.primal_obj
        gap_by_method[:pf_lmo_lmo] = eval_pf_lmo_lmo.primal_gap
        dual_gap_by_method[:pf_lmo_lmo] = eval_pf_lmo_lmo.dual_gap
        dual_variables_by_method[:pf_lmo_lmo] = eval_pf_lmo_lmo.dual_variables

        _print_pf_summary(:pf_lmo_lmo, result_pf_lmo_lmo)

        _append_pf_history!(
            pf_history_rows,
            :pf_lmo_lmo,
            result_pf_lmo_lmo,
            n,
            s,
            t,
        )
    end


    # ============================================================
    # Projection-free, Algorithm 2: LMO-PO
    # ============================================================

    if run_method(:pf_lmo_po)
        runtime_by_method[:pf_lmo_po] = @elapsed begin
            result_pf_lmo_po = calibrate_upsilon_projection_free_ddfactplus(
                Csym,
                s,
                t;
                _pf_common_kwargs(theta0_calib)...,

                algorithm = :lmo_po,

                tau0 = pf_lmo_po_tau0,
                tau_power = pf_lmo_po_tau_power,

                beta0 = pf_lmo_po_beta0,
                beta_power = pf_lmo_po_beta_power,

                Lqq_hat = pf_lmo_po_Lqq_hat,
            )
        end

        eval_pf_lmo_po = _pf_primal_dual_gaps(
            Csym,
            result_pf_lmo_po,
            z_ls,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            atol = atol,
            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
        )

        obj_by_method[:pf_lmo_po] = eval_pf_lmo_po.primal_obj
        gap_by_method[:pf_lmo_po] = eval_pf_lmo_po.primal_gap
        dual_gap_by_method[:pf_lmo_po] = eval_pf_lmo_po.dual_gap
        dual_variables_by_method[:pf_lmo_po] = eval_pf_lmo_po.dual_variables

        _print_pf_summary(:pf_lmo_po, result_pf_lmo_po)

        _append_pf_history!(
            pf_history_rows,
            :pf_lmo_po,
            result_pf_lmo_po,
            n,
            s,
            t,
        )
    end


    # ============================================================
    # Projection-free, Algorithm 3: PO-LMO
    # ============================================================

    if run_method(:pf_po_lmo)
        runtime_by_method[:pf_po_lmo] = @elapsed begin
            result_pf_po_lmo = calibrate_upsilon_projection_free_ddfactplus(
                Csym,
                s,
                t;
                _pf_common_kwargs(theta0_calib)...,

                algorithm = :po_lmo,

                tau0 = pf_po_lmo_tau0,
                tau_power = pf_po_lmo_tau_power,

                beta0 = pf_po_lmo_beta0,
                beta_power = pf_po_lmo_beta_power,

                Lqq_hat = pf_po_lmo_Lqq_hat,
            )
        end

        eval_pf_po_lmo = _pf_primal_dual_gaps(
            Csym,
            result_pf_po_lmo,
            z_ls,
            s,
            t;
            J1 = Int[],
            J0 = Int[],
            psi_margin = psi_margin,
            psi_floor = psi_floor,
            atol = atol,
            relax_knitro_outlev = relax_knitro_outlev,
            relax_knitro_opttol = relax_knitro_opttol,
            relax_knitro_feastol = relax_knitro_feastol,
        )

        obj_by_method[:pf_po_lmo] = eval_pf_po_lmo.primal_obj
        gap_by_method[:pf_po_lmo] = eval_pf_po_lmo.primal_gap
        dual_gap_by_method[:pf_po_lmo] = eval_pf_po_lmo.dual_gap
        dual_variables_by_method[:pf_po_lmo] = eval_pf_po_lmo.dual_variables

        _print_pf_summary(:pf_po_lmo, result_pf_po_lmo)

        _append_pf_history!(
            pf_history_rows,
            :pf_po_lmo,
            result_pf_po_lmo,
            n,
            s,
            t,
        )
    end


    # ============================================================
    # Compute gaps
    # ============================================================

    for m in selected_methods
        if haskey(obj_by_method, m) && !haskey(gap_by_method, m)
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

    for m in selected_pf_methods
        if haskey(dual_gap_by_method, m)
            row[Symbol("$(m)_dual_gap")] = dual_gap_by_method[m]
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

    for m in selected_pf_methods
        if haskey(dual_gap_by_method, m)
            @printf(
                "  %-22s dual_gap = % .6e\n",
                string(m),
                dual_gap_by_method[m],
            )
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
pf_dual_gap_cols = Symbol[]

for m in selected_methods
    push!(gap_cols, Symbol("$(m)_gap"))
end

for m in selected_methods
    push!(runtime_cols, Symbol("$(m)_runtime"))
end

for m in selected_pf_methods
    push!(pf_dual_gap_cols, Symbol("$(m)_dual_gap"))
end

push!(runtime_cols, :local_search_runtime)

column_order = vcat(
    [:n, :s, :t],
    gap_cols,
    pf_dual_gap_cols,
    runtime_cols,
)

df_results = DataFrame(rows)

df_results = df_results[:, column_order]

CSV.write(results_filepath, df_results)

println("Saved results to: $results_filepath")

df_pf_history = DataFrame(pf_history_rows)

if nrow(df_pf_history) > 0
    CSV.write(pf_history_filepath, df_pf_history)
    println("Saved PF history to: $pf_history_filepath")
else
    println("No PF history saved because no PF diagnostics were collected.")
end