import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import importlib.util
import sys

# Import from dapd.py (has utilities)
spec_dapd = importlib.util.spec_from_file_location("dapd", "dapd.py")
dapd = importlib.util.module_from_spec(spec_dapd)
sys.modules["dapd"] = dapd
spec_dapd.loader.exec_module(dapd)

# Import from dapdbo.py (has d_apdb_unconstrained for unconstrained optimization)
spec_dapdbo = importlib.util.spec_from_file_location("dapdbo", "dapdbo.py")
dapdbo = importlib.util.module_from_spec(spec_dapdbo)
sys.modules["dapdbo"] = dapdbo
spec_dapdbo.loader.exec_module(dapdbo)

# Import from aldo.py (has aldo_qcqp_merely_convex)
spec_aldo = importlib.util.spec_from_file_location("aldo", "aldo.py")
aldo = importlib.util.module_from_spec(spec_aldo)
sys.modules["aldo"] = aldo
spec_aldo.loader.exec_module(aldo)

# Import QP with L1 problem generation functions from utils
from utils import generate_feasible_qp_l1, solve_qp_l1_ground_truth

# --------------------------
# Main comparison function
# --------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # Problem configuration
    main_seed = 42
    solver_seed = 789
    
    rng = np.random.default_rng(main_seed)
    
    # Problem size
    n = 20
    N = 12
    E = 24
    
    # Algorithm parameters
    gamma = None
    max_iter = 3000
    c_alpha = 0.1
    c_beta = 0.1
    c_c = 0.1
    zeta = 1.0
    rho_shrink = 0.9
    delta = 0.1
    verbose_every = 10
    initial_scale = 10
    initialization_mode = "independent"
    
    # Generate problem
    print("="*80)
    print("Generating QP Problem with L1 Regularization")
    print("="*80)
    
    # Use generate_feasible_qp_l1 (with linear terms from (x - bar{x}^i)^T Q^i (x - bar{x}^i))
    Q_list, q_list, lambda_l1, constant_list = generate_feasible_qp_l1(n, N, rng)
    
    # Aggregate objective for ground truth solving: 
    # A0 = (1/N) * sum(Q_i), b0 = (1/N) * sum(q_i), constant = (1/N) * sum(c_i)
    A0_agg = np.mean(Q_list, axis=0)
    b0_agg = np.mean(q_list, axis=0)
    constant_agg = np.mean(constant_list)
    print(f"Generated problem: n={n}, lambda_l1={lambda_l1}")
    print(f"Constant term (aggregated): {constant_agg:.6f}")
    
    # Generate network
    print("\nGenerating network topology...")
    network_type = "small_world"
    neighbors_list = dapd.generate_small_world_network(N, E, seed=main_seed)
    print(f"Network type: {network_type}, N={N}, E={E}")
    
    # Solve ground truth
    print("\n" + "="*80)
    print("Solving Ground Truth")
    print("="*80)
    # solve_qp_l1_ground_truth directly solves the centralized problem:
    #   min sum_i [(1/2) x^T Q^i x + (q^i)^T x + c^i] + 1.0 * ||x||_1
    # This is the same scale as the algorithms compute.
    x_star, f_star = solve_qp_l1_ground_truth(A0_agg, b0_agg, lambda_l1, neighbors_list, 
                                               constant_term=constant_agg, use_centralized_scale=True)
    print(f"Ground truth (centralized scale): f* = {f_star:.6f}")
    print(f"x* norm: {np.linalg.norm(x_star):.6f}")
    print(f"x* l1 norm: {np.linalg.norm(x_star, 1):.6f}")
    
    # Compute per-node Lipschitz constants L_{f_i} = ||Q_i||_2
    L_f_i_list = []
    for i, Q_i in enumerate(Q_list):
        L_val = np.linalg.norm(Q_i, ord=2)
        if L_val < 1e-12:
            L_val = 1e-12  # Prevent division by zero
        L_f_i_list.append(L_val)
    max_L_f_i = max(L_f_i_list)
    min_L_f_i = min(L_f_i_list)
    mean_L_f_i = float(np.mean(L_f_i_list))
    print(f"L_f_i_list: {L_f_i_list}")
    
    # D-APD: tau_i = 1 / L_{f_i} for each node
    tau_dapd_list = [1.0 / L_val for L_val in L_f_i_list]
    
    # D-APDB: tau_i = tau_multiplier / L_{f_i} for each node
    tau_multiplier = 1  # Hyperparameter for D-APDB
    tau_dapdbo_list = [tau_multiplier / L_val for L_val in L_f_i_list]
    
    # ALDO: alpha_init = constant / max{L_{f_i}}
    alpha_init_constant = 10.0  # Hyperparameter for ALDO
    alpha_init_aldo = alpha_init_constant / max_L_f_i
    
    # Compute gamma for D-APD (use largest tau_i for conservative bound)
    temp_d_max = max(len(neighbors_list[i]) for i in range(N))
    tau_dapd_max = max(tau_dapd_list)
    temp_sigma0_max = zeta * tau_dapd_max
    computed_gamma_dapd = 1.0 / (2.0 * temp_d_max * temp_sigma0_max * N * ((2.0 / c_alpha) + (1.0 / c_c)))
    
    print(f"\nComputed parameters:")
    print(f"  L_obj (aggregated) = {dapd.compute_lipschitz_constant(A0_agg, b0_agg):.6f}")
    print(f"  L_f_i stats -> min: {min_L_f_i:.6f}, max: {max_L_f_i:.6f}, mean: {mean_L_f_i:.6f}")
    print(f"\nInitial step sizes:")
    print(f"  D-APD: tau_i = 1 / L_f_i (min={min(tau_dapd_list):.6e}, max={max(tau_dapd_list):.6e})")
    print(f"  D-APDB: tau_i = {tau_multiplier} / L_f_i (min={min(tau_dapdbo_list):.6e}, max={max(tau_dapdbo_list):.6e})")
    print(f"  ALDO: alpha_init = {alpha_init_constant} / max{{L_f_i}} = {alpha_init_aldo:.6e} (alpha_init_constant = {alpha_init_constant})")
    print(f"\nOther parameters:")
    print(f"  D-APD gamma = {computed_gamma_dapd:.6e}")
    
    # ==================== Run Multiple Simulations ====================
    num_simulations = 1
    print(f"\n" + "="*80)
    print(f"Running {num_simulations} Simulations (i.i.d. random initializations)")
    print("="*80)
    
    # Storage for all simulation results
    all_hist_dapd = []
    all_hist_dapdbo = []
    all_hist_aldo = []
    all_x_bar_dapd = []
    all_x_bar_dapdbo = []
    all_x_bar_aldo = []
    
    for sim_idx in range(num_simulations):
        sim_seed = solver_seed + sim_idx  # Different seed for each simulation
        print(f"\n--- Simulation {sim_idx + 1}/{num_simulations} (seed={sim_seed}) ---")
        
        # Generate new initial points for each simulation (i.i.d. random)
        initial_points_matrix = initial_scale * np.random.default_rng(sim_seed).standard_normal((N, n))
        initial_points_list = [initial_points_matrix[i, :].copy() for i in range(N)]
        
        # Run D-APD
        if sim_idx == 0:
            print("Running D-APD Solver...")
        x_bar_dapd, hist_dapd = dapd.d_apd_qcqp_merely_convex(
            A0_agg, b0_agg, None, [([], [], []) for _ in range(N)], None, None,
            Q_list=Q_list, q_list=q_list,  # Node-specific objectives
            N=N, max_iter=max_iter, seed=sim_seed,
            c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_c,
            zeta=zeta, tau=None, gamma=computed_gamma_dapd,
            verbose_every=verbose_every if sim_idx == 0 else 0, initial_scale=initial_scale,
            phi_star=f_star, tol=1e-8, normalize_consensus_error=False,
            use_optimal_consensus_error=False, x_star=x_star,
            neighbors_list=neighbors_list, initialization_mode=initialization_mode,
            B_theta=[0.0] * N, lambda_l1=lambda_l1, initial_points=initial_points_list,
            tau_list=tau_dapd_list,
            constant_list=constant_list  # Add constant terms
        )
        all_hist_dapd.append(hist_dapd)
        all_x_bar_dapd.append(x_bar_dapd)
        
        # Run D-APDB (unconstrained version)
        if sim_idx == 0:
            print("Running D-APDB Solver...")
        x_bar_dapdbo, hist_dapdbo, _ = dapdbo.d_apdb_unconstrained(
            N=N, n=n, max_iter=max_iter, seed=sim_seed,
            c_alpha=c_alpha, c_varsigma=c_c, c_gamma=None,  # c_gamma will be computed automatically
            rho_shrink=rho_shrink, delta=delta,
            verbose_every=verbose_every if sim_idx == 0 else 0,
            initial_scale=initial_scale,
            phi_star=f_star, tol=1e-8, normalize_consensus_error=False,
            use_optimal_consensus_error=False, x_star=x_star,
            neighbors_list=neighbors_list, initialization_mode=initialization_mode,
            lambda_l1=lambda_l1, initial_points=initial_points_list,
            Q_list=Q_list, q_list=q_list, tau_list=tau_dapdbo_list,
            constant_list=constant_list  # Add constant terms
        )
        all_hist_dapdbo.append(hist_dapdbo)
        all_x_bar_dapdbo.append(x_bar_dapdbo)
        
        # Run ALDO
        if sim_idx == 0:
            print("Running ALDO Solver...")
        x_bar_aldo, hist_aldo = aldo.aldo_qcqp_merely_convex(
            A0_agg, b0_agg, 0.0, None, None,
            N=N, max_iter=max_iter, seed=sim_seed,
            alpha_init=None, alpha_init_constant=alpha_init_constant, delta=delta, c=0.3,
            verbose_every=verbose_every if sim_idx == 0 else 0, initial_scale=initial_scale,
            phi_star=f_star, tol=1e-8, normalize_consensus_error=False,
            use_optimal_consensus_error=False, x_star=x_star,
            neighbors_list=neighbors_list, initialization_mode=initialization_mode,
            lambda_l1=lambda_l1, initial_points=initial_points_list,
            Q_list=Q_list, q_list=q_list, constant_list=constant_list  # Add linear and constant terms
        )
        all_hist_aldo.append(hist_aldo)
        all_x_bar_aldo.append(x_bar_aldo)
        
        if sim_idx == 0:
            print(f"D-APD result: ||x_bar - x*|| = {np.linalg.norm(x_bar_dapd - x_star):.6f}")
            print(f"D-APDB result: ||x_bar - x*|| = {np.linalg.norm(x_bar_dapdbo - x_star):.6f}")
            print(f"ALDO result: ||x_bar - x*|| = {np.linalg.norm(x_bar_aldo - x_star):.6f}")
    
    print(f"\nCompleted {num_simulations} simulations")
    
    # ==================== Aggregate Results Across Simulations ====================
    print("\n" + "="*80)
    print("Aggregating Results Across Simulations")
    print("="*80)
    
    def aggregate_histories(all_histories, metric_idx):
        """
        Aggregate a specific metric across all simulations by aligning gradient calls.
        
        Parameters:
        -----------
        all_histories : list of lists
            Each element is a history list from one simulation
        metric_idx : int
            Index of the metric in history tuple (0=obj, 1=maxV, 2=cons, 3=avgV, 4=subopt, 5=grad_calls, 6=backtrack)
            
        Returns:
        --------
        grad_calls_grid : np.ndarray
            Common grid of gradient calls (x-axis)
        mean_values : np.ndarray
            Mean values across simulations
        std_values : np.ndarray
            Standard deviation across simulations
        """
        # Extract gradient calls and metric values for each simulation
        all_grad_calls = []
        all_metric_values = []
        
        for hist in all_histories:
            grad_calls = np.array([h[5] for h in hist])  # Average gradient calls per node
            metric_vals = np.array([h[metric_idx] for h in hist])
            all_grad_calls.append(grad_calls)
            all_metric_values.append(metric_vals)
        
        # Find common gradient calls grid (union of all gradient call sequences)
        all_grad_calls_flat = np.concatenate(all_grad_calls)
        min_grad = np.min(all_grad_calls_flat)
        max_grad = np.max(all_grad_calls_flat)
        
        # Create a fine grid for interpolation
        # Use the maximum length among all simulations to determine grid density
        max_len = max(len(gc) for gc in all_grad_calls)
        num_points = min(max_len * 2, 2000)  # Reasonable upper limit
        grad_calls_grid = np.linspace(min_grad, max_grad, num_points)
        
        # Interpolate each simulation's metric values onto the common grid
        interpolated_values = []
        for grad_calls, metric_vals in zip(all_grad_calls, all_metric_values):
            # Use forward fill for values beyond the last point
            if len(grad_calls) > 1:
                # Interpolate using numpy
                interp_vals = np.interp(grad_calls_grid, grad_calls, metric_vals)
            else:
                # Single point - use constant value
                interp_vals = np.full_like(grad_calls_grid, metric_vals[0])
            interpolated_values.append(interp_vals)
        
        # Compute mean and std across simulations
        interpolated_array = np.array(interpolated_values)
        mean_values = np.nanmean(interpolated_array, axis=0)
        std_values = np.nanstd(interpolated_array, axis=0)
        
        return grad_calls_grid, mean_values, std_values
    
    # Aggregate all metrics for all three algorithms
    # D-APD history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
    grad_calls_dapd, objs_dapd_mean, objs_dapd_std = aggregate_histories(all_hist_dapd, 0)
    _, cons_dapd_mean, cons_dapd_std = aggregate_histories(all_hist_dapd, 2)
    _, subopt_dapd_mean, subopt_dapd_std = aggregate_histories(all_hist_dapd, 4)
    _, x_bar_norm_sq_dapd_mean, x_bar_norm_sq_dapd_std = aggregate_histories(all_hist_dapd, 6)  # Index 6 for x_bar_norm_sq
    _, cons_err_sq_sum_dapd_mean, cons_err_sq_sum_dapd_std = aggregate_histories(all_hist_dapd, 7)  # Index 7 for cons_err_sq_sum
    _, tau_dapd_mean, tau_dapd_std = aggregate_histories(all_hist_dapd, 8)  # Index 8 for avg_tau
    
    # D-APDB history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
    grad_calls_dapdbo, objs_dapdbo_mean, objs_dapdbo_std = aggregate_histories(all_hist_dapdbo, 0)
    _, cons_dapdbo_mean, cons_dapdbo_std = aggregate_histories(all_hist_dapdbo, 2)
    _, subopt_dapdbo_mean, subopt_dapdbo_std = aggregate_histories(all_hist_dapdbo, 4)
    _, backtrack_dapdbo_mean, backtrack_dapdbo_std = aggregate_histories(all_hist_dapdbo, 6)  # Index 6 for backtrack iterations
    _, x_bar_norm_sq_dapdbo_mean, x_bar_norm_sq_dapdbo_std = aggregate_histories(all_hist_dapdbo, 7)  # Index 7 for x_bar_norm_sq
    _, cons_err_sq_sum_dapdbo_mean, cons_err_sq_sum_dapdbo_std = aggregate_histories(all_hist_dapdbo, 8)  # Index 8 for cons_err_sq_sum
    _, tau_dapdbo_mean, tau_dapdbo_std = aggregate_histories(all_hist_dapdbo, 9)  # Index 9 for avg_tau
    
    # ALDO/global-DATOS history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, alpha)
    grad_calls_aldo, objs_aldo_mean, objs_aldo_std = aggregate_histories(all_hist_aldo, 0)
    _, cons_aldo_mean, cons_aldo_std = aggregate_histories(all_hist_aldo, 2)
    _, subopt_aldo_mean, subopt_aldo_std = aggregate_histories(all_hist_aldo, 4)
    _, backtrack_aldo_mean, backtrack_aldo_std = aggregate_histories(all_hist_aldo, 6)  # Index 6 for backtrack iterations
    _, x_bar_norm_sq_aldo_mean, x_bar_norm_sq_aldo_std = aggregate_histories(all_hist_aldo, 7)  # Index 7 for x_bar_norm_sq
    _, cons_err_sq_sum_aldo_mean, cons_err_sq_sum_aldo_std = aggregate_histories(all_hist_aldo, 8)  # Index 8 for cons_err_sq_sum
    _, alpha_aldo_mean, alpha_aldo_std = aggregate_histories(all_hist_aldo, 9)  # Index 9 for alpha
    
    # Compute relative errors
    # Relative suboptimality: |f(x_bar) - f*| / |f*|
    f_star_abs = abs(f_star)
    rel_subopt_dapd = subopt_dapd_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_mean, np.nan)
    rel_subopt_dapd_std = subopt_dapd_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_std, np.nan)
    rel_subopt_dapdbo = subopt_dapdbo_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_mean, np.nan)
    rel_subopt_dapdbo_std = subopt_dapdbo_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_std, np.nan)
    rel_subopt_aldo = subopt_aldo_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_mean, np.nan)
    rel_subopt_aldo_std = subopt_aldo_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_std, np.nan)
    
    # Relative consensus error: ||x_i^k - x_bar^k||^2 / (N * ||x_bar^k||^2)
    rel_cons_dapd = cons_err_sq_sum_dapd_mean / (N * np.maximum(x_bar_norm_sq_dapd_mean, 1e-12))
    rel_cons_dapdbo = cons_err_sq_sum_dapdbo_mean / (N * np.maximum(x_bar_norm_sq_dapdbo_mean, 1e-12))
    rel_cons_aldo = cons_err_sq_sum_aldo_mean / (N * np.maximum(x_bar_norm_sq_aldo_mean, 1e-12))
    rel_cons_dapd_std = np.abs(rel_cons_dapd) * np.sqrt(
        np.maximum((cons_err_sq_sum_dapd_std / np.maximum(cons_err_sq_sum_dapd_mean, 1e-12))**2 +
                   (x_bar_norm_sq_dapd_std / np.maximum(x_bar_norm_sq_dapd_mean, 1e-12))**2, 0)
    )
    rel_cons_dapdbo_std = np.abs(rel_cons_dapdbo) * np.sqrt(
        np.maximum((cons_err_sq_sum_dapdbo_std / np.maximum(cons_err_sq_sum_dapdbo_mean, 1e-12))**2 +
                   (x_bar_norm_sq_dapdbo_std / np.maximum(x_bar_norm_sq_dapdbo_mean, 1e-12))**2, 0)
    )
    rel_cons_aldo_std = np.abs(rel_cons_aldo) * np.sqrt(
        np.maximum((cons_err_sq_sum_aldo_std / np.maximum(cons_err_sq_sum_aldo_mean, 1e-12))**2 +
                   (x_bar_norm_sq_aldo_std / np.maximum(x_bar_norm_sq_aldo_mean, 1e-12))**2, 0)
    )
    
    # Final comparison (averaged over simulations)
    print(f"\nFinal Results (averaged over {num_simulations} simulations):")
    avg_dist_dapd = np.mean([np.linalg.norm(x - x_star) for x in all_x_bar_dapd])
    avg_dist_dapdbo = np.mean([np.linalg.norm(x - x_star) for x in all_x_bar_dapdbo])
    avg_dist_aldo = np.mean([np.linalg.norm(x - x_star) for x in all_x_bar_aldo])
    print(f"  D-APD:    avg ||x - x*|| = {avg_dist_dapd:.6f}, avg f(x) = {objs_dapd_mean[-1]:.6f}, avg subopt = {subopt_dapd_mean[-1]:.6e}")
    print(f"  D-APDB: avg ||x - x*|| = {avg_dist_dapdbo:.6f}, avg f(x) = {objs_dapdbo_mean[-1]:.6f}, avg subopt = {subopt_dapdbo_mean[-1]:.6e}")
    print(f"  ALDO:     avg ||x - x*|| = {avg_dist_aldo:.6f}, avg f(x) = {objs_aldo_mean[-1]:.6f}, avg subopt = {subopt_aldo_mean[-1]:.6e}")
    print(f"  Ground truth: f* = {f_star:.6f}")
    
    # Plotting
    import datetime
    import os
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "main-qp-experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    network_info = f"SW_N{N}_E{E}" if network_type == "small_world" else f"UNK_N{N}"
    subfolder_name = f"{network_info}_n{n}_l1{lambda_l1:.4f}_init{initialization_mode}_seed{main_seed}_{solver_seed}_nsims{num_simulations}_{timestamp}"
    subfolder_path = os.path.join(output_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    base_filename = f"qp_comparison_{subfolder_name}"
    
    # 1. Objective Function
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax1.plot(grad_calls_dapd, objs_dapd_mean, lw=2, label='D-APD (mean)', color='blue')
    ax1.fill_between(grad_calls_dapd, objs_dapd_mean - objs_dapd_std, objs_dapd_mean + objs_dapd_std, 
                      alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB with shaded region
    ax1.plot(grad_calls_dapdbo, objs_dapdbo_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax1.fill_between(grad_calls_dapdbo, objs_dapdbo_mean - objs_dapdbo_std, objs_dapdbo_mean + objs_dapdbo_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    # ALDO with shaded region
    ax1.plot(grad_calls_aldo, objs_aldo_mean, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
    ax1.fill_between(grad_calls_aldo, objs_aldo_mean - objs_aldo_std, objs_aldo_mean + objs_aldo_std, 
                      alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
    ax1.axhline(f_star, color='k', ls=':', alpha=0.5, label='$\\varphi^*$')
    ax1.set_title(f'Objective Function with L1 Regularization (N={N}, n={n}, λ={lambda_l1}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Average Number of Gradient Calls per Node')
    ax1.set_ylabel('Objective Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filename1 = os.path.join(subfolder_path, f"{base_filename}_objective.png")
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename1}")
    plt.close()
    
    # 2. Consensus Error
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax2.plot(grad_calls_dapd, cons_dapd_mean, lw=2, label='D-APD (mean)', color='blue')
    ax2.fill_between(grad_calls_dapd, cons_dapd_mean - cons_dapd_std, cons_dapd_mean + cons_dapd_std, 
                      alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB with shaded region
    ax2.plot(grad_calls_dapdbo, cons_dapdbo_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax2.fill_between(grad_calls_dapdbo, cons_dapdbo_mean - cons_dapdbo_std, cons_dapdbo_mean + cons_dapdbo_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    # ALDO with shaded region
    ax2.plot(grad_calls_aldo, cons_aldo_mean, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
    ax2.fill_between(grad_calls_aldo, cons_aldo_mean - cons_aldo_std, cons_aldo_mean + cons_aldo_std, 
                      alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
    ax2.set_title(f'Consensus Error (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Number of Gradient Calls per Node')
    ax2.set_ylabel('Consensus Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    filename2 = os.path.join(subfolder_path, f"{base_filename}_consensus.png")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename2}")
    plt.close()
    
    # 3. Absolute Suboptimality
    if not all(np.isnan(subopt_dapd_mean)) and not all(np.isnan(subopt_dapdbo_mean)) and not all(np.isnan(subopt_aldo_mean)):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # D-APD with shaded region
        ax3.plot(grad_calls_dapd, subopt_dapd_mean, lw=2, label='D-APD (mean)', color='blue')
        ax3.fill_between(grad_calls_dapd, subopt_dapd_mean - subopt_dapd_std, subopt_dapd_mean + subopt_dapd_std, 
                          alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        # D-APDB with shaded region
        ax3.plot(grad_calls_dapdbo, subopt_dapdbo_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax3.fill_between(grad_calls_dapdbo, subopt_dapdbo_mean - subopt_dapdbo_std, subopt_dapdbo_mean + subopt_dapdbo_std, 
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        # ALDO with shaded region
        ax3.plot(grad_calls_aldo, subopt_aldo_mean, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
        ax3.fill_between(grad_calls_aldo, subopt_aldo_mean - subopt_aldo_std, subopt_aldo_mean + subopt_aldo_std, 
                          alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
        ax3.set_title(f'Absolute Suboptimality (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Number of Gradient Calls per Node')
        ax3.set_ylabel('Suboptimality')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        filename3 = os.path.join(subfolder_path, f"{base_filename}_suboptimality.png")
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename3}")
        plt.close()
    
    # 4. Relative Suboptimality: |f(x_bar) - f*| / |f*|
    if not all(np.isnan(rel_subopt_dapd)) and not all(np.isnan(rel_subopt_dapdbo)) and not all(np.isnan(rel_subopt_aldo)):
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.semilogy(grad_calls_dapd, rel_subopt_dapd, lw=2, label='D-APD (mean)', color='blue')
        ax4.fill_between(grad_calls_dapd,
                         np.maximum(rel_subopt_dapd - rel_subopt_dapd_std, 1e-12),
                         rel_subopt_dapd + rel_subopt_dapd_std,
                         alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        ax4.semilogy(grad_calls_dapdbo, rel_subopt_dapdbo, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax4.fill_between(grad_calls_dapdbo,
                         np.maximum(rel_subopt_dapdbo - rel_subopt_dapdbo_std, 1e-12),
                         rel_subopt_dapdbo + rel_subopt_dapdbo_std,
                         alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax4.semilogy(grad_calls_aldo, rel_subopt_aldo, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
        ax4.fill_between(grad_calls_aldo,
                         np.maximum(rel_subopt_aldo - rel_subopt_aldo_std, 1e-12),
                         rel_subopt_aldo + rel_subopt_aldo_std,
                         alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
        ax4.set_title(f'Relative Suboptimality: $|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Average Number of Gradient Calls per Node')
        ax4.set_ylabel('Relative Suboptimality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        filename4 = os.path.join(subfolder_path, f"{base_filename}_relative_suboptimality.png")
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename4}")
        plt.close()
    
    # 4b. Log Relative Suboptimality
    if not all(np.isnan(rel_subopt_dapd)) and not all(np.isnan(rel_subopt_dapdbo)) and not all(np.isnan(rel_subopt_aldo)):
        log_rel_subopt_dapd = np.log(rel_subopt_dapd + 1.0)
        log_rel_subopt_dapdbo = np.log(rel_subopt_dapdbo + 1.0)
        log_rel_subopt_aldo = np.log(rel_subopt_aldo + 1.0)
        log_rel_subopt_dapd_std = rel_subopt_dapd_std / (rel_subopt_dapd + 1.0)
        log_rel_subopt_dapdbo_std = rel_subopt_dapdbo_std / (rel_subopt_dapdbo + 1.0)
        log_rel_subopt_aldo_std = rel_subopt_aldo_std / (rel_subopt_aldo + 1.0)
        fig4b, ax4b = plt.subplots(figsize=(10, 6))
        ax4b.plot(grad_calls_dapd, log_rel_subopt_dapd, lw=2, label='D-APD (mean)', color='blue')
        ax4b.fill_between(grad_calls_dapd,
                          log_rel_subopt_dapd - log_rel_subopt_dapd_std,
                          log_rel_subopt_dapd + log_rel_subopt_dapd_std,
                          alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        ax4b.plot(grad_calls_dapdbo, log_rel_subopt_dapdbo, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax4b.fill_between(grad_calls_dapdbo,
                          log_rel_subopt_dapdbo - log_rel_subopt_dapdbo_std,
                          log_rel_subopt_dapdbo + log_rel_subopt_dapdbo_std,
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax4b.plot(grad_calls_aldo, log_rel_subopt_aldo, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
        ax4b.fill_between(grad_calls_aldo,
                          log_rel_subopt_aldo - log_rel_subopt_aldo_std,
                          log_rel_subopt_aldo + log_rel_subopt_aldo_std,
                          alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
        ax4b.set_title(f'Log Relative Suboptimality: $\\log((|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|) + 1)$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax4b.set_xlabel('Average Number of Gradient Calls per Node')
        ax4b.set_ylabel('$\\log((|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|) + 1)$')
        ax4b.legend()
        ax4b.grid(True, alpha=0.3)
        plt.tight_layout()
        filename4b = os.path.join(subfolder_path, f"{base_filename}_log_relative_suboptimality.png")
        plt.savefig(filename4b, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename4b}")
        plt.close()

    # 5. Relative Consensus Error: ||x_i^k - x_bar^k||^2 / (N * ||x_bar^k||^2)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.semilogy(grad_calls_dapd, rel_cons_dapd, lw=2, label='D-APD (mean)', color='blue')
    ax5.fill_between(grad_calls_dapd,
                     np.maximum(rel_cons_dapd - rel_cons_dapd_std, 1e-12),
                     rel_cons_dapd + rel_cons_dapd_std,
                     alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    ax5.semilogy(grad_calls_dapdbo, rel_cons_dapdbo, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax5.fill_between(grad_calls_dapdbo,
                     np.maximum(rel_cons_dapdbo - rel_cons_dapdbo_std, 1e-12),
                     rel_cons_dapdbo + rel_cons_dapdbo_std,
                     alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    ax5.semilogy(grad_calls_aldo, rel_cons_aldo, lw=2, label='global-DATOS (mean)', color='green', linestyle=':')
    ax5.fill_between(grad_calls_aldo,
                     np.maximum(rel_cons_aldo - rel_cons_aldo_std, 1e-12),
                     rel_cons_aldo + rel_cons_aldo_std,
                     alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
    ax5.set_title(f'Relative Consensus Error: $\\|x_i^k - \\bar{{x}}^k\\|^2/(N\\|\\bar{{x}}^k\\|^2)$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Average Number of Gradient Calls per Node')
    ax5.set_ylabel('Relative Consensus Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    filename5 = os.path.join(subfolder_path, f"{base_filename}_relative_consensus.png")
    plt.savefig(filename5, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename5}")
    plt.close()
    
    # 6. Backtrack Iterations (D-APDB and ALDO) - Focus on first 10-20 iterations
    # Compute backtracking statistics from raw histories
    print("\n" + "="*80)
    print("Backtracking Statistics")
    print("="*80)
    
    # D-APDB: Count iterations where backtracking occurred (backtrack_iters > 0)
    dapdbo_backtrack_counts = []
    for hist in all_hist_dapdbo:
        # hist format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, ...)
        backtrack_iters = [h[6] for h in hist[1:]]  # Skip initial point (index 0)
        num_backtrack = sum(1 for b in backtrack_iters if b > 0)
        total_iters = len(backtrack_iters)
        dapdbo_backtrack_counts.append((num_backtrack, total_iters))
    
    avg_backtrack_dapdbo = np.mean([c[0] for c in dapdbo_backtrack_counts])
    avg_total_dapdbo = np.mean([c[1] for c in dapdbo_backtrack_counts])
    print(f"D-APDB: {avg_backtrack_dapdbo:.1f} / {avg_total_dapdbo:.1f} outer iterations had backtracking ({100*avg_backtrack_dapdbo/avg_total_dapdbo:.1f}%)")
    
    # global-DATOS: Count iterations where backtracking occurred (backtrack_iters > 0)
    aldo_backtrack_counts = []
    for hist in all_hist_aldo:
        # hist format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, ...)
        backtrack_iters = [h[6] for h in hist[1:]]  # Skip initial point (index 0)
        num_backtrack = sum(1 for b in backtrack_iters if b > 0)
        total_iters = len(backtrack_iters)
        aldo_backtrack_counts.append((num_backtrack, total_iters))
    
    avg_backtrack_aldo = np.mean([c[0] for c in aldo_backtrack_counts])
    avg_total_aldo = np.mean([c[1] for c in aldo_backtrack_counts])
    print(f"global-DATOS: {avg_backtrack_aldo:.1f} / {avg_total_aldo:.1f} outer iterations had backtracking ({100*avg_backtrack_aldo/avg_total_aldo:.1f}%)")
    
    # Plot backtracking iterations - Use iteration number as x-axis (like main-qcqp.py)
    focus_iterations = 20  # Focus on first 20 iterations
    
    # Extract backtrack values for each simulation and align by iteration number
    # D-APDB
    all_backtrack_dapdbo = []
    for hist in all_hist_dapdbo:
        backtrack_vals = np.array([h[6] for h in hist])  # Index 6 for backtrack iterations
        all_backtrack_dapdbo.append(backtrack_vals)
    
    # global-DATOS
    all_backtrack_aldo = []
    for hist in all_hist_aldo:
        backtrack_vals = np.array([h[6] for h in hist])  # Index 6 for backtrack iterations
        all_backtrack_aldo.append(backtrack_vals)
    
    # Align by iteration number (pad shorter histories with last value)
    max_iterations = min(max(len(b) for b in all_backtrack_dapdbo + all_backtrack_aldo), focus_iterations)
    
    # D-APDB aligned
    aligned_backtrack_dapdbo = []
    for backtrack_vals in all_backtrack_dapdbo:
        if len(backtrack_vals) >= max_iterations:
            aligned_backtrack_dapdbo.append(backtrack_vals[:max_iterations])
        else:
            padded = np.pad(backtrack_vals, (0, max_iterations - len(backtrack_vals)), mode='edge')
            aligned_backtrack_dapdbo.append(padded)
    aligned_array_dapdbo = np.array(aligned_backtrack_dapdbo)
    backtrack_dapdbo_iter_mean = np.mean(aligned_array_dapdbo, axis=0)
    backtrack_dapdbo_iter_std = np.std(aligned_array_dapdbo, axis=0)
    
    # global-DATOS aligned
    aligned_backtrack_aldo = []
    for backtrack_vals in all_backtrack_aldo:
        if len(backtrack_vals) >= max_iterations:
            aligned_backtrack_aldo.append(backtrack_vals[:max_iterations])
        else:
            padded = np.pad(backtrack_vals, (0, max_iterations - len(backtrack_vals)), mode='edge')
            aligned_backtrack_aldo.append(padded)
    aligned_array_aldo = np.array(aligned_backtrack_aldo)
    backtrack_aldo_iter_mean = np.mean(aligned_array_aldo, axis=0)
    backtrack_aldo_iter_std = np.std(aligned_array_aldo, axis=0)
    
    # Plot
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    iterations = np.arange(max_iterations)
    
    # D-APDB with shaded region
    ax6.plot(iterations, backtrack_dapdbo_iter_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--', marker='o', markersize=4)
    ax6.fill_between(iterations, 
                      backtrack_dapdbo_iter_mean - backtrack_dapdbo_iter_std, 
                      backtrack_dapdbo_iter_mean + backtrack_dapdbo_iter_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    
    # global-DATOS with shaded region
    ax6.plot(iterations, backtrack_aldo_iter_mean, lw=2, label='global-DATOS (mean)', color='green', linestyle=':', marker='s', markersize=4)
    ax6.fill_between(iterations, 
                      backtrack_aldo_iter_mean - backtrack_aldo_iter_std, 
                      backtrack_aldo_iter_mean + backtrack_aldo_iter_std, 
                      alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
    
    ax6.set_title(f'Total Backtrack Iterations per Iteration (First {max_iterations} iterations, N={N}, n={n})', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Iteration Number')
    ax6.set_ylabel('Total Backtrack Iterations (All Nodes)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    filename6 = os.path.join(subfolder_path, f"{base_filename}_backtrack.png")
    plt.savefig(filename6, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename6}")
    plt.close()
    
    # 7. Step Size (tau/alpha) Evolution
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    # D-APD: tau (constant, no backtracking)
    ax7.semilogy(grad_calls_dapd, tau_dapd_mean, lw=2, label='D-APD: $\\bar{\\tau}$ (mean)', color='blue')
    ax7.fill_between(grad_calls_dapd,
                     np.maximum(tau_dapd_mean - tau_dapd_std, 1e-12),
                     tau_dapd_mean + tau_dapd_std,
                     alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB: tau (adaptive with backtracking)
    ax7.semilogy(grad_calls_dapdbo, tau_dapdbo_mean, lw=2, label='D-APDB: $\\bar{\\tau}$ (mean)', color='red', linestyle='--')
    ax7.fill_between(grad_calls_dapdbo,
                     np.maximum(tau_dapdbo_mean - tau_dapdbo_std, 1e-12),
                     tau_dapdbo_mean + tau_dapdbo_std,
                     alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    # global-DATOS: alpha (adaptive with backtracking)
    ax7.semilogy(grad_calls_aldo, alpha_aldo_mean, lw=2, label='global-DATOS: $\\alpha$ (mean)', color='green', linestyle=':')
    ax7.fill_between(grad_calls_aldo,
                     np.maximum(alpha_aldo_mean - alpha_aldo_std, 1e-12),
                     alpha_aldo_mean + alpha_aldo_std,
                     alpha=0.2, color='green', label=f'global-DATOS (±1 std, {num_simulations} sims)')
    ax7.set_title(f'Step Size Evolution (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Average Number of Gradient Calls per Node')
    ax7.set_ylabel('Step Size ($\\tau$ or $\\alpha$)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    filename7 = os.path.join(subfolder_path, f"{base_filename}_stepsize.png")
    plt.savefig(filename7, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename7}")
    plt.close()
    
    print(f"\nAll figures saved in folder: {subfolder_path}")

