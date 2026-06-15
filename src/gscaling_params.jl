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

        :verbose_bfgs => false,
    ),

    :direct => Dict{Symbol,Any}(
        :max_bfgs_iter => 150,

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

        :verbose_bfgs => false,
    ),
)


# ============================================================
# Regularized BFGS parameter sets
# ============================================================
rbfgs_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :max_iter => 200,

        :B0_scale => 1.0,

        :mu0 => 1e-2,
        :mu_min => 1e-10,
        :mu_max => 1e4,
        :mu_decrease => 0.2,
        :mu_increase => 5.0,
        :eta1 => 0.05,
        :eta2 => 0.75,
        :max_inner_regularization => 20,

        :curvature_tol => 1e-12,
        :damping_delta => 0.2,
        :reset_B_on_failed_update => false,

        :normalize_direction => false,
        :max_direction_norm => 10.0,
        :max_q_norm_inf => 20.0,

        :armijo_c1 => 1e-4,
        :accept_tol => 1e-12,
        :alpha0 => 1.0,
        :alpha_min => 1e-12,
        :alpha_decay => 0.5,
        :max_backtracks => 30,

        :nonmonotone => true,
        :nonmonotone_window => 10,

        :project_spd => true,
        :min_B_eig => 1e-8,
        :max_B_eig => 1e8,
        :reset_B_on_bad => true,
        :max_B_norm => 1e8,

        :grad_tol => 1e-2,
        :step_tol => 1e-10,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :cache_digits => 12,
        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :max_iter => 30,

        :B0_scale => 1.0,

        :mu0 => 1e-2,
        :mu_min => 1e-10,
        :mu_max => 1e4,
        :mu_decrease => 0.2,
        :mu_increase => 5.0,
        :eta1 => 0.05,
        :eta2 => 0.75,
        :max_inner_regularization => 20,

        :curvature_tol => 1e-12,
        :damping_delta => 0.2,
        :reset_B_on_failed_update => false,

        :normalize_direction => false,
        :max_direction_norm => 10.0,
        :max_q_norm_inf => 20.0,

        :armijo_c1 => 1e-4,
        :accept_tol => 1e-12,
        :alpha0 => 1.0,
        :alpha_min => 1e-12,
        :alpha_decay => 0.5,
        :max_backtracks => 30,

        :nonmonotone => true,
        :nonmonotone_window => 10,

        :project_spd => false,
        :min_B_eig => 1e-8,
        :max_B_eig => 1e8,
        :reset_B_on_bad => true,
        :max_B_norm => 1e8,

        :grad_tol => 1e-2,
        :step_tol => 1e-10,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :cache_digits => 12,
        :diagnostics => false,
        :verbose => false,
    ),
)


# ============================================================
# One-step proximal Knitro parameter sets
# ============================================================
prox_step_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :rho => 1e-3,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :q_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 12,
        :diagnostics => false,
        :verbose => false,
    ),

    :node => Dict{Symbol,Any}(
        :rho => 1e-3,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :q_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :knitro_feastol => 1e-6,
        :knitro_opttol => 1e-2,
        :knitro_xtol => 1e-4,
        :knitro_ftol => 1e-5,

        :knitro_maxtime_real => Inf,
        :knitro_algorithm => nothing,
        :knitro_bar_murule => nothing,
        :knitro_honorbnds => 1,
        :knitro_outlev => 0,

        :cache_digits => 12,
        :diagnostics => false,
        :verbose => false,
    ),
)