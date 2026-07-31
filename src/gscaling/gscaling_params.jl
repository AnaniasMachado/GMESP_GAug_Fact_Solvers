# ============================================================
# BFGS parameter sets
# ============================================================
bfgs_param_sets = Dict(
    :default => Dict{Symbol,Any}(
        :max_bfgs_iter => 100,

        :grad_tol => 1e-2,
        :step_tol => 1e-8,

        :alpha0 => 1.0,
        :alpha_min => 1e-10,
        :alpha_decay => 0.75,
        :armijo_c1 => 1e-6,
        :max_backtracks => 50,

        :curvature_tol => 1e-12,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,

        :max_theta_norm => 20.0,
        :psi_derivative => true,

        :use_steepest_descent_fallback => true,

        :knitro_outlev => nothing,
        :knitro_opttol => 1e-8,
        :knitro_feastol => 1e-5,

        :verbose_bfgs => false,
    ),

    :fast => Dict{Symbol,Any}(
        :max_bfgs_iter => 50,

        :grad_tol => 1e-2,
        :step_tol => 1e-8,

        :alpha0 => 1.0,
        :alpha_min => 1e-10,
        :alpha_decay => 0.75,
        :armijo_c1 => 1e-6,
        :max_backtracks => 50,

        :curvature_tol => 1e-12,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,

        :max_theta_norm => 20.0,
        :psi_derivative => true,

        :use_steepest_descent_fallback => true,

        :knitro_outlev => nothing,
        :knitro_opttol => 1e-8,
        :knitro_feastol => 1e-5,

        :verbose_bfgs => false,
    ),

    :very_fast => Dict{Symbol,Any}(
        :max_bfgs_iter => 10,

        :grad_tol => 1e-2,
        :step_tol => 1e-8,

        :alpha0 => 1.0,
        :alpha_min => 1e-10,
        :alpha_decay => 0.75,
        :armijo_c1 => 1e-8,
        :max_backtracks => 50,

        :curvature_tol => 1e-12,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,

        :max_theta_norm => 20.0,
        :psi_derivative => true,

        :use_steepest_descent_fallback => true,

        :knitro_outlev => nothing,
        :knitro_opttol => 1e-8,
        :knitro_feastol => 1e-5,

        :verbose_bfgs => false,
    ),
)


# ============================================================
# One-step PPA parameter sets
#
# These are for:
#
#     calibrate_upsilon_ppa_ddfactplus(...; k = 1, ...)
#
# ============================================================
ppa_one_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :k => 1,

        :rho => 1e3,

        :grad_tol => 1e-2,
        :prox_obj_abs_tol => 1e-8,
        :prox_step_tol => 1e-8,
        :max_wall_time => Inf,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :theta_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-5,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 6,
        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :k => 1,

        :rho => 1e3,

        :grad_tol => 1e-2,
        :prox_obj_abs_tol => 1e-8,
        :prox_step_tol => 1e-8,
        :max_wall_time => Inf,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        # Fixed typo: was :q_bound.
        :theta_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-5,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 6,
        :diagnostics => false,
        :verbose => false,
    ),
)


# ============================================================
# Full PPA parameter sets
#
# These are for:
#
#     calibrate_upsilon_ppa_ddfactplus(...; k = Inf, ...)
#
# Full PPA stops when either:
#
#     ||grad V(theta)|| <= grad_tol
#
# or:
#
#     prox_obj_change <= prox_obj_abs_tol
#     and step_norm <= prox_step_tol
#
# ============================================================
ppa_full_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :k => Inf,

        :rho => 1e3,

        :grad_tol => 1e-2,
        :prox_obj_abs_tol => 1e-8,
        :prox_step_tol => 1e-8,
        :max_wall_time => Inf,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :theta_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-5,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 6,
        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :k => Inf,

        :rho => 1e3,

        :grad_tol => 1e-2,
        :prox_obj_abs_tol => 1e-8,
        :prox_step_tol => 1e-8,
        :max_wall_time => Inf,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :theta_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-5,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 6,
        :diagnostics => false,
        :verbose => false,
    ),
)


# ============================================================
# Projection-free LMO-LMO parameter sets
#
# These are for:
#
#     calibrate_upsilon_projection_free_ddfactplus(...; algorithm = :lmo_lmo, ...)
#
# ============================================================
pf_lmo_lmo_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :algorithm => :lmo_lmo,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 250,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 0.10,
        :tau_power => 1.05,

        :beta0 => 0.18,
        :beta_power => 1.0 / 4.0,

        :Lqq_hat => 0.52,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :algorithm => :lmo_lmo,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 50,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 0.10,
        :tau_power => 1.05,

        :beta0 => 0.18,
        :beta_power => 1.0 / 4.0,

        :Lqq_hat => 0.52,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),
)


# ============================================================
# Projection-free LMO-PO parameter sets
#
# These are for:
#
#     calibrate_upsilon_projection_free_ddfactplus(...; algorithm = :lmo_po, ...)
#
# ============================================================
pf_lmo_po_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :algorithm => :lmo_po,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 250,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 0.0525,
        :tau_power => 1.00,

        :beta0 => 0.55,
        :beta_power => 0.40,

        :Lqq_hat => 0.65,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :algorithm => :lmo_po,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 50,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 0.0525,
        :tau_power => 1.00,

        :beta0 => 0.55,
        :beta_power => 0.40,

        :Lqq_hat => 0.65,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),
)


# ============================================================
# Projection-free PO-LMO parameter sets
#
# These are for:
#
#     calibrate_upsilon_projection_free_ddfactplus(...; algorithm = :po_lmo, ...)
#
# ============================================================
pf_po_lmo_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :algorithm => :po_lmo,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 250,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 1.00,
        :tau_power => 2.0 / 3.0,

        :beta0 => 0.10,
        :beta_power => 1.0 / 6.0,

        :Lqq_hat => 0.50,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :algorithm => :po_lmo,
        :lmo_q_solver => false,

        :max_iter => 500,
        :min_iter => 50,
        :iteration_power => 1.5,

        :theta_bound => 20.0,

        :tau0 => 1.00,
        :tau_power => 2.0 / 3.0,

        :beta0 => 0.10,
        :beta_power => 1.0 / 6.0,

        :Lqq_hat => 0.50,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,

        :gurobi_output_flag => 0,
        :gurobi_opttol => 1e-8,
        :gurobi_feastol => 1e-8,

        :diagnostics => false,
        :verbose => false,
    ),
)