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

        :knitro_outlev => nothing,
        :knitro_opttol => 1e-8,
        :knitro_feastol => 1e-8,

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

        :knitro_outlev => nothing,
        :knitro_opttol => 1e-8,
        :knitro_feastol => 1e-8,

        :verbose_bfgs => false,
    ),
)


# ============================================================
# One-step proximal Knitro parameter sets
# ============================================================
prox_step_param_sets = Dict(
    :root => Dict{Symbol,Any}(
        :rho => 1e3,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :theta_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-8,

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
        :rho => 1e3,

        :theta_perturbation => 1e-2,
        :center_initial_theta => false,

        :q_bound => 20.0,

        :psi_margin => 1e-7,
        :psi_floor => 0.0,
        :psi_derivative => true,
        :t1_reformulation => false,

        :relax_knitro_outlev => nothing,
        :relax_knitro_opttol => 1e-8,
        :relax_knitro_feastol => 1e-8,

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