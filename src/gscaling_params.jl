# ============================================================
# BFGS parameter sets
# ============================================================
bfgs_param_sets = Dict(
    :default => Dict(
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

    :fast => Dict(
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

    :very_fast => Dict(
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

    :direct => Dict(
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