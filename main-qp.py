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

# Import from global-datos.py (has aldo_qcqp_merely_convex)
spec_aldo = importlib.util.spec_from_file_location("aldo", "global-datos.py")
aldo = importlib.util.module_from_spec(spec_aldo)
sys.modules["aldo"] = aldo
spec_aldo.loader.exec_module(aldo)

# Import QP with L1 problem generation functions from utils
from utils import generate_feasible_qp_l1, generate_qp_with_l1, generate_feasible_qp_l1_w_std, solve_qp_l1_ground_truth

# --------------------------
# Main comparison function
# --------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # Problem configuration
    main_seed = 23
    solver_seed = 42
    
    rng = np.random.default_rng(main_seed)
    
    # Problem size
    # Increase N and make network sparser to slow down ALDO's gossip-based consensus
    n = 20
    N = 12  # More nodes (was 12)
    E = 24  # Sparser network: E/N ≈ 1.25 (was E/N = 2)
    
    # Algorithm parameters
    gamma = None
    # Fixed total number of communications for fair comparison
    max_communications = 1000  # Total communication rounds
    # D-APD and D-APDB: 1 communication per iteration
    max_iter_dapd = max_communications
    max_iter_dapdbo = max_communications
    # ALDO: 2 communications per iteration
    max_iter_aldo = max_communications // 2
    # Key insight: γ^k = (c_γ/τ̄) / (2/c_α + η^k/c_ς)
    # To increase γ^k (faster consensus), we can:
    # 1. Decrease τ̄ (smaller tau_multiplier)
    # 2. Increase c_α and c_ς (makes denominator smaller)  #
    # With c_alpha = c_varsigma = 0.4:
    # (2/0.4 + 1/0.4) = 7.5  (vs 30 with c=0.1)
    # γ^k increases ~4x!
    #
    # Backtracking threshold = 1 - δ - c_α - c_ς = 1 - 0.05 - 0.4 - 0.4 = 0.15
    c_alpha = 0.4
    c_beta = 0.1
    c_varsigma = 0.4  # c_alpha + c_varsigma = 0.8 < 0.9 = 1 - delta
    zeta = 1.0
    rho_shrink = 0.9  # Match ALDO's backtracking factor (α ← α/2)
    delta = 0.1
    verbose_every = 1000
    initial_scale = 0.5
    initialization_mode = "independent"
    
    # Generate problem
    print("="*80)
    print("Generating QP Problem with L1 Regularization")
    print("="*80)
    
    # Choose problem generation method:
    # Option 1: generate_feasible_qp_l1 - uses (x - bar{x}^i)^T Q^i (x - bar{x}^i) formulation
    # Option 2: generate_qp_with_l1 - simpler, uses r_i * Q^{(i)} with r_i from 1 to 100
    # Option 3: generate_feasible_qp_l1_w_std - Q^i = Λ_i^T S_i Λ_i, q^i ~ N(0,1), c^i ~ U[0,1]
    qp_generation_method = 3  # 1, 2, or 3
    
    if qp_generation_method == 1:
        # Use generate_feasible_qp_l1 (with linear terms from (x - bar{x}^i)^T Q^i (x - bar{x}^i))
        Q_list, q_list, lambda_l1, constant_list = generate_feasible_qp_l1(
            n, N, rng, gamma_mean=100.0, gamma_std_percent=0.1
        )
        print(f"Using generate_feasible_qp_l1 (x - bar{{x}}^i formulation)")
    elif qp_generation_method == 2:
        # Use generate_qp_with_l1 (simpler formulation, no constant term)
        # Q_i = r_i * Q^{(i)} where r_1 = 1, r_N = 100
        lambda_l1_param = 0.1  # Fixed L1 coefficient
        Q_list, q_list, lambda_l1 = generate_qp_with_l1(n, N, rng, lambda_l1=lambda_l1_param)
        constant_list = [0.0] * N  # No constant terms
        print(f"Using generate_qp_with_l1 (simple formulation)")
    else:
        # Use generate_feasible_qp_l1_w_std (new formulation)
        # Q^i = Λ_i^T S_i Λ_i (PSD, merely convex with zero eigenvalue)
        # q^i ~ N(0, 1) (standard Gaussian)
        # c^i ~ U[0, 1] (uniform)
        # ||Q_i||_2 controlled by L_mean and L_std_percent
        Q_list, q_list, lambda_l1, constant_list = generate_feasible_qp_l1_w_std(
            n, N, rng, L_mean=1000.0, L_std_percent=0.1
        )
        print(f"Using generate_feasible_qp_l1_w_std (Q^i = Λ^T S Λ, q^i ~ N(0,1), c^i ~ U[0,1])")
    
    # Override lambda_l1 if needed (set to 1.0 for stronger L1 regularization)
    # Based on grid search best config: lambda_l1 = 1.0 per node
    lambda_l1 = 1 / N # Per-node L1 coefficient (centralized = N * 1.0)
    print(f"  lambda_l1 (per-node) = {lambda_l1:.6f}, centralized L1 coefficient = {N * lambda_l1:.1f}")
    
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
    
    # D-APD: tau_i should satisfy backtracking condition (if it had one)
    # τ ≤ (1 - δ - c_α - c_ς) / L
    # This ensures D-APD uses same effective step size as D-APDB when no backtracking occurs
    tau_threshold = 0.5
    tau_dapd_list = [tau_threshold / L_val for L_val in L_f_i_list]
    
    # D-APDB: tau_i = tau_multiplier / L_{f_i} for each node
    # Backtracking condition: lhs ≤ (1-δ-c_α-c_ς)/(2τ) × ||dx||²
    # For L-smooth function: lhs ≤ (L/2) ||dx||²
    # No backtracking if: (L/2) ≤ (1-δ-c_α-c_ς)/(2τ), i.e., τ ≤ (1-δ-c_α-c_ς)/L
    # With c_alpha=c_varsigma=0.4, delta=0.05: threshold = (1-0.05-0.4-0.4)/L = 0.15/L
    # tau_multiplier = 1.0 would require ~3 backtracks: 1 → 0.5 → 0.25 → 0.125 < 0.15 ✓
    # Using smaller tau_multiplier for fewer backtracks and smaller τ̄ (which increases γ^k)
    tau_multiplier = 5.0  # Start at threshold to minimize backtracking while keeping τ̄ small
    tau_dapdbo_list = [tau_multiplier / L_val for L_val in L_f_i_list]
    
    # ALDO: alpha_init = constant / max{L_{f_i}} 
    alpha_init_constant = 5.0  # Hyperparameter for ALDO
    alpha_init_aldo = alpha_init_constant / max_L_f_i 
    
    # Compute gamma for D-APD (use largest tau_i for conservative bound)
    temp_d_max = max(len(neighbors_list[i]) for i in range(N))
    tau_dapd_max = max(tau_dapd_list)
    temp_sigma0_max = zeta * tau_dapd_max
    computed_gamma_dapd = 1.0 / (2.0 * temp_d_max * temp_sigma0_max * N * ((2.0 / c_alpha) + (1.0 / c_varsigma)))
    
    print(f"\nComputed parameters:")
    print(f"  L_obj (aggregated) = {dapd.compute_lipschitz_constant(A0_agg, b0_agg):.6f}")
    print(f"  L_f_i stats -> min: {min_L_f_i:.6f}, max: {max_L_f_i:.6f}, mean: {mean_L_f_i:.6f}")
    print(f"\nInitial step sizes:")
    print(f"  D-APD: tau_i = {tau_threshold:.4f} / L_f_i (min={min(tau_dapd_list):.6e}, max={max(tau_dapd_list):.6e})")
    print(f"  D-APDB: tau_i = {tau_multiplier} / L_f_i (min={min(tau_dapdbo_list):.6e}, max={max(tau_dapdbo_list):.6e})")
    print(f"  ALDO: alpha_init = {alpha_init_constant} / max{{L_f_i}} = {alpha_init_aldo:.6e} (alpha_init_constant = {alpha_init_constant})")
    print(f"\nOther parameters:")
    print(f"  D-APD gamma = {computed_gamma_dapd:.6e}")
    
    # ==================== Run Multiple Simulations ====================
    num_simulations = 20
    
    # Visualization smoothing: clip shaded bands to central percentiles for readability
    shade_percentile_bounds = (10, 90)  # Set to None to revert to mean ± std shading
    
    print(f"\n" + "="*80)
    print(f"Running {num_simulations} Simulations (i.i.d. random initializations)")
    print(f"Maximum communications: {max_communications}")
    print(f"  D-APD:   max_iter = {max_iter_dapd} (1 comm/iter)")
    print(f"  D-APDB:  max_iter = {max_iter_dapdbo} (1 comm/iter)")
    print(f"  ALDO:    max_iter = {max_iter_aldo} (2 comms/iter)")
    print("="*80)
    
    # Storage for all simulation results
    all_hist_dapd = []
    all_hist_dapdbo = []
    all_stats_dapdbo = []  # Store stats for D-APDB
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
            N=N, max_iter=max_iter_dapd, seed=sim_seed,
            c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_varsigma,
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
        # c_gamma: theory requires c_gamma <= 1/(2|E|) ≈ 0.02, but we use much larger for faster consensus
        # Larger c_gamma → larger γ_k → faster consensus (compensates for tau_bar_max in denominator)
        c_gamma_dapdbo = 1 / (2 * E) # 0.02  # Very aggressive: 50x theoretical bound
        x_bar_dapdbo, hist_dapdbo, stats_dapdbo = dapdbo.d_apdb_unconstrained(
            N=N, n=n, max_iter=max_iter_dapdbo, seed=sim_seed,
            c_alpha=c_alpha, c_varsigma=c_varsigma, c_gamma=c_gamma_dapdbo,
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
        all_stats_dapdbo.append(stats_dapdbo)
        all_x_bar_dapdbo.append(x_bar_dapdbo)
        
        # Run ALDO
        if sim_idx == 0:
            print("Running ALDO Solver...")
        x_bar_aldo, hist_aldo = aldo.aldo_qcqp_merely_convex(
            A0_agg, b0_agg, 0.0, None, None,
            N=N, max_iter=max_iter_aldo, seed=sim_seed,
            alpha_init=None, alpha_init_constant=alpha_init_constant, delta=delta, c=0.1,  # Slower gossip mixing (was 0.3)
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
            if 'backtrack_counts_per_node' in stats_dapdbo:
                counts = stats_dapdbo['backtrack_counts_per_node']
                print(f"  D-APDB Backtracks per node: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
                print(f"  Counts: {counts}")
            print(f"ALDO result: ||x_bar - x*|| = {np.linalg.norm(x_bar_aldo - x_star):.6f}")
    
    print(f"\nCompleted {num_simulations} simulations")
    
    # ==================== Aggregate Results Across Simulations ====================
    print("\n" + "="*80)
    print("Aggregating Results Across Simulations")
    print("="*80)
    
    def aggregate_histories_by_communications(all_histories, metric_idx, comms_per_iter=1, max_comm=None, percentile_clip=None):
        """
        Aggregate a specific metric across all simulations by aligning communication rounds.
        
        Parameters:
        -----------
        all_histories : list of lists
            Each element is a history list from one simulation
        metric_idx : int
            Index of the metric in history tuple (0=obj, 1=maxV, 2=cons, 3=avgV, 4=subopt, 5=grad_calls, 6=backtrack)
        comms_per_iter : int
            Number of communications per iteration (1 for D-APD/D-APDB, 2 for ALDO)
        max_comm : int or None
            Maximum communication rounds to include (None means no limit)
        percentile_clip : tuple(int, int) or None
            Optional (low, high) percentiles used to clip the shaded band.
            When None, the shaded region defaults to mean ± 1 std.
            
        Returns:
        --------
        comms_grid : np.ndarray
            Common grid of communication rounds (x-axis)
        mean_values : np.ndarray
            Mean values across simulations
        std_values : np.ndarray
            Standard deviation across simulations
        lower_band : np.ndarray
            Lower edge for shaded visualization (percentile-based or mean - std)
        upper_band : np.ndarray
            Upper edge for shaded visualization (percentile-based or mean + std)
        """
        # Extract communication rounds and metric values for each simulation
        all_comms = []
        all_metric_values = []
        
        for hist in all_histories:
            # Communication rounds: iteration index * comms_per_iter
            # hist[0] is initial point (0 communications), hist[1] is after 1st iteration, etc.
            comms = np.array([i * comms_per_iter for i in range(len(hist))])  # [0, comms_per_iter, 2*comms_per_iter, ...]
            metric_vals = np.array([h[metric_idx] for h in hist])
            
            # Truncate to max_comm if specified
            if max_comm is not None:
                mask = comms <= max_comm
                comms = comms[mask]
                metric_vals = metric_vals[mask]
            
            all_comms.append(comms)
            all_metric_values.append(metric_vals)
        
        # Find common communication rounds grid (union of all communication sequences)
        all_comms_flat = np.concatenate(all_comms)
        min_comm = np.min(all_comms_flat)
        max_comm_actual = np.max(all_comms_flat)
        
        # If max_comm is specified, use it as the upper bound
        if max_comm is not None:
            max_comm_actual = min(max_comm_actual, max_comm)
        
        # Create a fine grid for interpolation
        # Use the maximum length among all simulations to determine grid density
        max_len = max(len(c) for c in all_comms)
        num_points = min(max_len * 2, 2000)  # Reasonable upper limit
        comms_grid = np.linspace(min_comm, max_comm_actual, num_points)
        
        # Interpolate each simulation's metric values onto the common grid
        interpolated_values = []
        for comms, metric_vals in zip(all_comms, all_metric_values):
            # Use forward fill for values beyond the last point
            if len(comms) > 1:
                # Interpolate using numpy
                interp_vals = np.interp(comms_grid, comms, metric_vals)
            else:
                # Single point - use constant value
                interp_vals = np.full_like(comms_grid, metric_vals[0])
            interpolated_values.append(interp_vals)
        
        # Compute mean and std across simulations
        interpolated_array = np.array(interpolated_values)
        mean_values = np.nanmean(interpolated_array, axis=0)
        std_values = np.nanstd(interpolated_array, axis=0)
        
        # Compute shaded band edges
        if percentile_clip is not None:
            low_p, high_p = percentile_clip
            low_p = max(0, min(low_p, 100))
            high_p = max(low_p, min(high_p, 100))
            lower_band = np.nanpercentile(interpolated_array, low_p, axis=0)
            upper_band = np.nanpercentile(interpolated_array, high_p, axis=0)
        else:
            lower_band = mean_values - std_values
            upper_band = mean_values + std_values
        
        return comms_grid, mean_values, std_values, lower_band, upper_band
    
    # Aggregate all metrics for all three algorithms by communication rounds
    # D-APD: 1 communication per iteration
    # D-APDB: 1 communication per iteration
    # ALDO: 2 communications per iteration
    # All algorithms are compared at the same total number of communications (max_communications)
    
    print(f"\nAggregating results up to {max_communications} communications...")
    
    # D-APD history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
    comms_dapd, objs_dapd_mean, objs_dapd_std, objs_dapd_lower, objs_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 0, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, cons_dapd_mean, cons_dapd_std, cons_dapd_lower, cons_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 2, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, subopt_dapd_mean, subopt_dapd_std, subopt_dapd_lower, subopt_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 4, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, x_bar_norm_sq_dapd_mean, x_bar_norm_sq_dapd_std, x_bar_norm_sq_dapd_lower, x_bar_norm_sq_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 6, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 6 for x_bar_norm_sq
    _, cons_err_sq_sum_dapd_mean, cons_err_sq_sum_dapd_std, cons_err_sq_sum_dapd_lower, cons_err_sq_sum_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 7, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 7 for cons_err_sq_sum
    _, tau_dapd_mean, tau_dapd_std, tau_dapd_lower, tau_dapd_upper = aggregate_histories_by_communications(all_hist_dapd, 8, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 8 for avg_tau
    
    # D-APDB history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
    comms_dapdbo, objs_dapdbo_mean, objs_dapdbo_std, objs_dapdbo_lower, objs_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 0, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, cons_dapdbo_mean, cons_dapdbo_std, cons_dapdbo_lower, cons_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 2, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, subopt_dapdbo_mean, subopt_dapdbo_std, subopt_dapdbo_lower, subopt_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 4, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, backtrack_dapdbo_mean, backtrack_dapdbo_std, _, _ = aggregate_histories_by_communications(all_hist_dapdbo, 6, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 6 for backtrack iterations
    _, x_bar_norm_sq_dapdbo_mean, x_bar_norm_sq_dapdbo_std, x_bar_norm_sq_dapdbo_lower, x_bar_norm_sq_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 7, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 7 for x_bar_norm_sq
    _, cons_err_sq_sum_dapdbo_mean, cons_err_sq_sum_dapdbo_std, cons_err_sq_sum_dapdbo_lower, cons_err_sq_sum_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 8, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 8 for cons_err_sq_sum
    _, tau_dapdbo_mean, tau_dapdbo_std, tau_dapdbo_lower, tau_dapdbo_upper = aggregate_histories_by_communications(all_hist_dapdbo, 9, comms_per_iter=1, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 9 for avg_tau
    
    # ALDO/global-DATOS history format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, alpha)
    comms_aldo, objs_aldo_mean, objs_aldo_std, objs_aldo_lower, objs_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 0, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, cons_aldo_mean, cons_aldo_std, cons_aldo_lower, cons_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 2, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, subopt_aldo_mean, subopt_aldo_std, subopt_aldo_lower, subopt_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 4, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)
    _, backtrack_aldo_mean, backtrack_aldo_std, _, _ = aggregate_histories_by_communications(all_hist_aldo, 6, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 6 for backtrack iterations
    _, x_bar_norm_sq_aldo_mean, x_bar_norm_sq_aldo_std, x_bar_norm_sq_aldo_lower, x_bar_norm_sq_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 7, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 7 for x_bar_norm_sq
    _, cons_err_sq_sum_aldo_mean, cons_err_sq_sum_aldo_std, cons_err_sq_sum_aldo_lower, cons_err_sq_sum_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 8, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 8 for cons_err_sq_sum
    _, alpha_aldo_mean, alpha_aldo_std, alpha_aldo_lower, alpha_aldo_upper = aggregate_histories_by_communications(all_hist_aldo, 9, comms_per_iter=2, max_comm=max_communications, percentile_clip=shade_percentile_bounds)  # Index 9 for alpha
    
    # Compute relative errors
    # Relative suboptimality: |f(x_bar) - f*| / |f*|
    f_star_abs = abs(f_star)
    rel_subopt_dapd = subopt_dapd_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_mean, np.nan)
    rel_subopt_dapd_std = subopt_dapd_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_std, np.nan)
    rel_subopt_dapd_lower = subopt_dapd_lower / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_lower, np.nan)
    rel_subopt_dapd_upper = subopt_dapd_upper / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapd_upper, np.nan)
    rel_subopt_dapdbo = subopt_dapdbo_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_mean, np.nan)
    rel_subopt_dapdbo_std = subopt_dapdbo_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_std, np.nan)
    rel_subopt_dapdbo_lower = subopt_dapdbo_lower / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_lower, np.nan)
    rel_subopt_dapdbo_upper = subopt_dapdbo_upper / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_dapdbo_upper, np.nan)
    rel_subopt_aldo = subopt_aldo_mean / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_mean, np.nan)
    rel_subopt_aldo_std = subopt_aldo_std / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_std, np.nan)
    rel_subopt_aldo_lower = subopt_aldo_lower / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_lower, np.nan)
    rel_subopt_aldo_upper = subopt_aldo_upper / f_star_abs if f_star_abs > 1e-10 else np.full_like(subopt_aldo_upper, np.nan)
    
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
    # Compute percentile bounds for relative consensus error
    rel_cons_dapd_lower = cons_err_sq_sum_dapd_lower / (N * np.maximum(x_bar_norm_sq_dapd_upper, 1e-12))
    rel_cons_dapd_upper = cons_err_sq_sum_dapd_upper / (N * np.maximum(x_bar_norm_sq_dapd_lower, 1e-12))
    rel_cons_dapdbo_lower = cons_err_sq_sum_dapdbo_lower / (N * np.maximum(x_bar_norm_sq_dapdbo_upper, 1e-12))
    rel_cons_dapdbo_upper = cons_err_sq_sum_dapdbo_upper / (N * np.maximum(x_bar_norm_sq_dapdbo_lower, 1e-12))
    rel_cons_aldo_lower = cons_err_sq_sum_aldo_lower / (N * np.maximum(x_bar_norm_sq_aldo_upper, 1e-12))
    rel_cons_aldo_upper = cons_err_sq_sum_aldo_upper / (N * np.maximum(x_bar_norm_sq_aldo_lower, 1e-12))
    
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
    
    # Legend label helper for shaded regions
    if shade_percentile_bounds:
        shade_label_text = f"{shade_percentile_bounds[0]}-{shade_percentile_bounds[1]} percentile band"
    else:
        shade_label_text = "±1 std"
    
    # 1. Objective Function
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax1.plot(comms_dapd, objs_dapd_mean, lw=2, label='D-APD', color='blue')
    ax1.fill_between(comms_dapd, objs_dapd_lower, objs_dapd_upper, 
                      alpha=0.2, color='blue')
    # D-APDB with shaded region
    ax1.plot(comms_dapdbo, objs_dapdbo_mean, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax1.fill_between(comms_dapdbo, objs_dapdbo_lower, objs_dapdbo_upper, 
                      alpha=0.2, color='red')
    # ALDO with shaded region
    ax1.plot(comms_aldo, objs_aldo_mean, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax1.fill_between(comms_aldo, objs_aldo_lower, objs_aldo_upper, 
                      alpha=0.2, color='green')
    ax1.axhline(f_star, color='k', ls=':', alpha=0.5, label='$\\varphi^*$')
    ax1.set_title('Objective Function', fontsize=26, fontweight='bold')
    ax1.set_xlabel('Number of Communications', fontsize=24)
    ax1.set_ylabel('Objective Value', fontsize=24)
    ax1.set_xlim(0, max_communications)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filename1 = os.path.join(subfolder_path, f"{base_filename}_objective.pdf")
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename1}")
    plt.close()
    
    # 2. Consensus Error
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax2.plot(comms_dapd, cons_dapd_mean, lw=2, label='D-APD', color='blue')
    ax2.fill_between(comms_dapd, cons_dapd_lower, cons_dapd_upper, 
                      alpha=0.2, color='blue')
    # D-APDB with shaded region
    ax2.plot(comms_dapdbo, cons_dapdbo_mean, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax2.fill_between(comms_dapdbo, cons_dapdbo_lower, cons_dapdbo_upper, 
                      alpha=0.2, color='red')
    # ALDO with shaded region
    ax2.plot(comms_aldo, cons_aldo_mean, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax2.fill_between(comms_aldo, cons_aldo_lower, cons_aldo_upper, 
                      alpha=0.2, color='green')
    ax2.set_title('Consensus Error', fontsize=26, fontweight='bold')
    ax2.set_xlabel('Number of Communications', fontsize=24)
    ax2.set_ylabel('Consensus Error', fontsize=24)
    ax2.set_xlim(0, max_communications)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    filename2 = os.path.join(subfolder_path, f"{base_filename}_consensus.pdf")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename2}")
    plt.close()
    
    # 3. Absolute Suboptimality
    if not all(np.isnan(subopt_dapd_mean)) and not all(np.isnan(subopt_dapdbo_mean)) and not all(np.isnan(subopt_aldo_mean)):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # D-APD with shaded region
        ax3.plot(comms_dapd, subopt_dapd_mean, lw=2, label='D-APD', color='blue')
        ax3.fill_between(comms_dapd, subopt_dapd_lower, subopt_dapd_upper, 
                          alpha=0.2, color='blue')
        # D-APDB with shaded region
        ax3.plot(comms_dapdbo, subopt_dapdbo_mean, lw=2, label='D-APDB0', color='red', linestyle='--')
        ax3.fill_between(comms_dapdbo, subopt_dapdbo_lower, subopt_dapdbo_upper, 
                          alpha=0.2, color='red')
        # ALDO with shaded region
        ax3.plot(comms_aldo, subopt_aldo_mean, lw=2, label='global-DATOS', color='green', linestyle=':')
        ax3.fill_between(comms_aldo, subopt_aldo_lower, subopt_aldo_upper, 
                          alpha=0.2, color='green')
        ax3.set_title('Absolute Suboptimality', fontsize=26, fontweight='bold')
        ax3.set_xlabel('Number of Communications', fontsize=24)
        ax3.set_ylabel('$|\\varphi(\\bar{x}^k) - \\varphi^*|$', fontsize=24)
        ax3.set_xlim(0, max_communications)
        ax3.legend(fontsize=20)
        ax3.tick_params(axis='both', labelsize=18)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        filename3 = os.path.join(subfolder_path, f"{base_filename}_suboptimality.pdf")
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename3}")
        plt.close()
    
    # 4. Relative Suboptimality: |f(x_bar) - f*| / |f*|
    if not all(np.isnan(rel_subopt_dapd)) and not all(np.isnan(rel_subopt_dapdbo)) and not all(np.isnan(rel_subopt_aldo)):
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.semilogy(comms_dapd, rel_subopt_dapd, lw=2, label='D-APD', color='blue')
        ax4.fill_between(comms_dapd,
                         np.maximum(rel_subopt_dapd_lower, 1e-12),
                         rel_subopt_dapd_upper,
                         alpha=0.2, color='blue')
        ax4.semilogy(comms_dapdbo, rel_subopt_dapdbo, lw=2, label='D-APDB0', color='red', linestyle='--')
        ax4.fill_between(comms_dapdbo,
                         np.maximum(rel_subopt_dapdbo_lower, 1e-12),
                         rel_subopt_dapdbo_upper,
                         alpha=0.2, color='red')
        ax4.semilogy(comms_aldo, rel_subopt_aldo, lw=2, label='global-DATOS', color='green', linestyle=':')
        ax4.fill_between(comms_aldo,
                         np.maximum(rel_subopt_aldo_lower, 1e-12),
                         rel_subopt_aldo_upper,
                         alpha=0.2, color='green')
        ax4.set_title('Relative Suboptimality', fontsize=26, fontweight='bold')
        ax4.set_xlabel('Number of Communications', fontsize=24)
        ax4.set_ylabel('$|\\varphi(\\bar{x}^k) - \\varphi^*|/|\\varphi^*|$', fontsize=24)
        ax4.set_xlim(0, max_communications)
        ax4.legend(fontsize=20)
        ax4.tick_params(axis='both', labelsize=18)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        filename4 = os.path.join(subfolder_path, f"{base_filename}_relative_suboptimality.pdf")
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
        # Compute log relative suboptimality bounds
        log_rel_subopt_dapd_lower = np.log(rel_subopt_dapd_lower + 1.0)
        log_rel_subopt_dapd_upper = np.log(rel_subopt_dapd_upper + 1.0)
        log_rel_subopt_dapdbo_lower = np.log(rel_subopt_dapdbo_lower + 1.0)
        log_rel_subopt_dapdbo_upper = np.log(rel_subopt_dapdbo_upper + 1.0)
        log_rel_subopt_aldo_lower = np.log(rel_subopt_aldo_lower + 1.0)
        log_rel_subopt_aldo_upper = np.log(rel_subopt_aldo_upper + 1.0)
        
        ax4b.plot(comms_dapd, log_rel_subopt_dapd, lw=2, label='D-APD', color='blue')
        ax4b.fill_between(comms_dapd,
                          log_rel_subopt_dapd_lower,
                          log_rel_subopt_dapd_upper,
                          alpha=0.2, color='blue')
        ax4b.plot(comms_dapdbo, log_rel_subopt_dapdbo, lw=2, label='D-APDB0', color='red', linestyle='--')
        ax4b.fill_between(comms_dapdbo,
                          log_rel_subopt_dapdbo_lower,
                          log_rel_subopt_dapdbo_upper,
                          alpha=0.2, color='red')
        ax4b.plot(comms_aldo, log_rel_subopt_aldo, lw=2, label='global-DATOS', color='green', linestyle=':')
        ax4b.fill_between(comms_aldo,
                          log_rel_subopt_aldo_lower,
                          log_rel_subopt_aldo_upper,
                          alpha=0.2, color='green')
        ax4b.set_title('Log Relative Suboptimality', fontsize=26, fontweight='bold')
        ax4b.set_xlabel('Number of Communications', fontsize=24)
        ax4b.set_ylabel('$\\log((|\\varphi(\\bar{x}^k) - \\varphi^*|/|\\varphi^*|) + 1)$', fontsize=24)
        ax4b.set_xlim(0, max_communications)
        ax4b.legend(fontsize=20)
        ax4b.tick_params(axis='both', labelsize=18)
        ax4b.grid(True, alpha=0.3)
        plt.tight_layout()
        filename4b = os.path.join(subfolder_path, f"{base_filename}_log_relative_suboptimality.pdf")
        plt.savefig(filename4b, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename4b}")
        plt.close()

    # 5. Relative Consensus Error: ||x_i^k - x_bar^k||^2 / (N * ||x_bar^k||^2)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.semilogy(comms_dapd, rel_cons_dapd, lw=2, label='D-APD', color='blue')
    ax5.fill_between(comms_dapd,
                     np.maximum(rel_cons_dapd_lower, 1e-12),
                     rel_cons_dapd_upper,
                     alpha=0.2, color='blue')
    ax5.semilogy(comms_dapdbo, rel_cons_dapdbo, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax5.fill_between(comms_dapdbo,
                     np.maximum(rel_cons_dapdbo_lower, 1e-12),
                     rel_cons_dapdbo_upper,
                     alpha=0.2, color='red')
    ax5.semilogy(comms_aldo, rel_cons_aldo, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax5.fill_between(comms_aldo,
                     np.maximum(rel_cons_aldo_lower, 1e-12),
                     rel_cons_aldo_upper,
                     alpha=0.2, color='green')
    ax5.set_title('Relative Consensus Error', fontsize=26, fontweight='bold')
    ax5.set_xlabel('Number of Communications', fontsize=24)
    ax5.set_ylabel('$\\sum_i\\|x_i^k - \\bar{x}^k\\|^2/(N\\|\\bar{x}^k\\|^2)$', fontsize=24)
    ax5.set_xlim(0, max_communications)
    ax5.legend(fontsize=20)
    ax5.tick_params(axis='both', labelsize=18)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    filename5 = os.path.join(subfolder_path, f"{base_filename}_relative_consensus.pdf")
    plt.savefig(filename5, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename5}")
    plt.close()
    
    # 6. Backtrack Iterations (D-APDB and ALDO) - Focus on first 10-20 iterations
    # Compute backtracking statistics from raw histories
    print("\n" + "="*80)
    print("Backtracking Statistics")
    print("="*80)
    
    # D-APDB backtracking stats (using collected stats)
    num_backtrack_dapdbo = sum(s['num_backtracking_iterations'] for s in all_stats_dapdbo)
    total_iters_dapdbo = sum(s['total_iterations'] for s in all_stats_dapdbo)
    # Average over simulations
    avg_backtrack_dapdbo = num_backtrack_dapdbo / num_simulations
    avg_total_dapdbo = total_iters_dapdbo / num_simulations
    
    if avg_total_dapdbo > 0:
        backtrack_ratio_dapdbo = avg_backtrack_dapdbo / avg_total_dapdbo
        print(f"D-APDB: {avg_backtrack_dapdbo:.1f} / {avg_total_dapdbo:.1f} outer iterations had backtracking ({backtrack_ratio_dapdbo:.1%})")
        
        # Print aggregated per-node statistics
        if all_stats_dapdbo and 'backtrack_counts_per_node' in all_stats_dapdbo[0]:
            # Aggregate counts across simulations
            agg_counts = np.zeros(N)
            for s in all_stats_dapdbo:
                agg_counts += np.array(s['backtrack_counts_per_node'])
            agg_counts /= num_simulations  # Average per simulation
            
            print(f"  Avg backtracks per node: min={min(agg_counts):.1f}, max={max(agg_counts):.1f}, mean={np.mean(agg_counts):.1f}")
            print(f"  Node counts (avg): {agg_counts}")
            print(f"  Standard deviation of node counts: {np.std(agg_counts):.1f}")
            if np.std(agg_counts) > 0.5 * np.mean(agg_counts):
                print("  WARNING: Significant imbalance in backtracking effort across nodes!")
    else:
        print("D-APDB: No iterations to report backtracking statistics.")
    
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
    ax6.plot(iterations, backtrack_dapdbo_iter_mean, lw=2, label='D-APDB0', color='red', linestyle='--', marker='o', markersize=4)
    ax6.fill_between(iterations, 
                      backtrack_dapdbo_iter_mean - backtrack_dapdbo_iter_std, 
                      backtrack_dapdbo_iter_mean + backtrack_dapdbo_iter_std, 
                      alpha=0.2, color='red')
    
    # global-DATOS with shaded region
    ax6.plot(iterations, backtrack_aldo_iter_mean, lw=2, label='global-DATOS', color='green', linestyle=':', marker='s', markersize=4)
    ax6.fill_between(iterations, 
                      backtrack_aldo_iter_mean - backtrack_aldo_iter_std, 
                      backtrack_aldo_iter_mean + backtrack_aldo_iter_std, 
                      alpha=0.2, color='green')
    
    ax6.set_title('Backtrack Iterations', fontsize=26, fontweight='bold')
    ax6.set_xlabel('Iteration Number', fontsize=24)
    ax6.set_ylabel('Total Backtrack Iterations (All Nodes)', fontsize=24)
    ax6.legend(fontsize=20)
    ax6.tick_params(axis='both', labelsize=18)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    filename6 = os.path.join(subfolder_path, f"{base_filename}_backtrack.pdf")
    plt.savefig(filename6, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename6}")
    plt.close()
    
    # 7. Step Size (tau/alpha) Evolution
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    # D-APD: tau (constant, no backtracking)
    ax7.semilogy(comms_dapd, tau_dapd_mean, lw=2, label='D-APD: $\\bar{\\tau}$', color='blue')
    ax7.fill_between(comms_dapd,
                     np.maximum(tau_dapd_lower, 1e-12),
                     tau_dapd_upper,
                     alpha=0.2, color='blue')
    # D-APDB: tau (adaptive with backtracking)
    ax7.semilogy(comms_dapdbo, tau_dapdbo_mean, lw=2, label='D-APDB0: $\\bar{\\tau}$', color='red', linestyle='--')
    ax7.fill_between(comms_dapdbo,
                     np.maximum(tau_dapdbo_lower, 1e-12),
                     tau_dapdbo_upper,
                     alpha=0.2, color='red')
    # global-DATOS: alpha (adaptive with backtracking)
    ax7.semilogy(comms_aldo, alpha_aldo_mean, lw=2, label='global-DATOS: $\\alpha$', color='green', linestyle=':')
    ax7.fill_between(comms_aldo,
                     np.maximum(alpha_aldo_lower, 1e-12),
                     alpha_aldo_upper,
                     alpha=0.2, color='green')
    ax7.set_title('Step Size Evolution', fontsize=26, fontweight='bold')
    ax7.set_xlabel('Number of Communications', fontsize=24)
    ax7.set_ylabel('Step Size ($\\tau$ or $\\alpha$)', fontsize=24)
    ax7.set_xlim(0, max_communications)
    ax7.legend(fontsize=20)
    ax7.tick_params(axis='both', labelsize=18)
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    filename7 = os.path.join(subfolder_path, f"{base_filename}_stepsize.pdf")
    plt.savefig(filename7, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename7}")
    plt.close()
    
    # ==================== Combined Figure (2x3 layout) ====================
    print("\nGenerating combined figure...")
    fig_combined, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (0,0) Objective Function
    ax = axes[0, 0]
    ax.plot(comms_dapd, objs_dapd_mean, lw=2, label='D-APD', color='blue')
    ax.plot(comms_dapdbo, objs_dapdbo_mean, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax.plot(comms_aldo, objs_aldo_mean, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax.axhline(f_star, color='k', ls=':', alpha=0.5, label='$\\varphi^*$')
    ax.set_title('(a) Objective Function', fontsize=24, fontweight='bold')
    ax.set_xlabel('Number of Communications', fontsize=20)
    ax.set_ylabel('Objective Value', fontsize=20)
    ax.set_xlim(0, max_communications)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # (0,1) Relative Suboptimality (log scale)
    ax = axes[0, 1]
    ax.semilogy(comms_dapd, rel_subopt_dapd, lw=2, label='D-APD', color='blue')
    ax.semilogy(comms_dapdbo, rel_subopt_dapdbo, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax.semilogy(comms_aldo, rel_subopt_aldo, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax.set_title('(b) Relative Suboptimality', fontsize=24, fontweight='bold')
    ax.set_xlabel('Number of Communications', fontsize=20)
    ax.set_ylabel('$|\\varphi(\\bar{x}^k) - \\varphi^*|/|\\varphi^*|$', fontsize=24)
    ax.set_xlim(0, max_communications)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # (0,2) Step Size Evolution
    ax = axes[0, 2]
    ax.semilogy(comms_dapd, tau_dapd_mean, lw=2, label='D-APD: $\\bar{\\tau}$', color='blue')
    ax.semilogy(comms_dapdbo, tau_dapdbo_mean, lw=2, label='D-APDB0: $\\bar{\\tau}$', color='red', linestyle='--')
    ax.semilogy(comms_aldo, alpha_aldo_mean, lw=2, label='global-DATOS: $\\alpha$', color='green', linestyle=':')
    ax.set_title('(c) Step Size Evolution', fontsize=24, fontweight='bold')
    ax.set_xlabel('Number of Communications', fontsize=20)
    ax.set_ylabel('Step Size ($\\tau$ or $\\alpha$)', fontsize=24)
    ax.set_xlim(0, max_communications)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # (1,0) Consensus Error
    ax = axes[1, 0]
    ax.plot(comms_dapd, cons_dapd_mean, lw=2, label='D-APD', color='blue')
    ax.plot(comms_dapdbo, cons_dapdbo_mean, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax.plot(comms_aldo, cons_aldo_mean, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax.set_title('(d) Consensus Error', fontsize=24, fontweight='bold')
    ax.set_xlabel('Number of Communications', fontsize=20)
    ax.set_ylabel('Consensus Error', fontsize=20)
    ax.set_xlim(0, max_communications)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # (1,1) Relative Consensus Error (log scale)
    ax = axes[1, 1]
    ax.semilogy(comms_dapd, rel_cons_dapd, lw=2, label='D-APD', color='blue')
    ax.semilogy(comms_dapdbo, rel_cons_dapdbo, lw=2, label='D-APDB0', color='red', linestyle='--')
    ax.semilogy(comms_aldo, rel_cons_aldo, lw=2, label='global-DATOS', color='green', linestyle=':')
    ax.set_title('(e) Relative Consensus Error', fontsize=24, fontweight='bold')
    ax.set_xlabel('Number of Communications', fontsize=20)
    ax.set_ylabel('$\\sum_i\\|x_i^k - \\bar{x}^k\\|^2/(N\\|\\bar{x}^k\\|^2)$', fontsize=24)
    ax.set_xlim(0, max_communications)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # (1,2) Backtrack Iterations
    ax = axes[1, 2]
    ax.plot(iterations, backtrack_dapdbo_iter_mean, lw=2, label='D-APDB0', color='red', linestyle='--', marker='o', markersize=3)
    ax.plot(iterations, backtrack_aldo_iter_mean, lw=2, label='global-DATOS', color='green', linestyle=':', marker='s', markersize=3)
    ax.set_title('(f) Backtrack Iterations', fontsize=24, fontweight='bold')
    # ax.set_ylim(0, 5)
    ax.set_xlabel('Iteration Number', fontsize=20)
    ax.set_ylabel('Total Backtrack Iters (All Nodes)', fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # No overall title needed
    
    plt.tight_layout()
    filename_combined = os.path.join(subfolder_path, f"{base_filename}_combined.pdf")
    plt.savefig(filename_combined, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename_combined}")
    plt.close()
    
    print(f"\nAll figures saved in folder: {subfolder_path}")

