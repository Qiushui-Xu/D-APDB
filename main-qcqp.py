import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import importlib.util
import sys
import warnings
# Filter out ECOS deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")

# Import QCQP problem generation function from utils
from utils import generate_feasible_qcqp, generate_feasible_qcqp_l1, generate_initial_point_with_violation, solve_qcqp_l1_ground_truth

# Import from dapd.py (has d_apd_qcqp_merely_convex and other utilities)
spec_dapd = importlib.util.spec_from_file_location("dapd", "dapd.py")
dapd = importlib.util.module_from_spec(spec_dapd)
sys.modules["dapd"] = dapd
spec_dapd.loader.exec_module(dapd)

# Import from dapdb.py (has d_apdb_qcqp_merely_convex)
spec_dapdb = importlib.util.spec_from_file_location("dapdb", "dapdb.py")
dapdb = importlib.util.module_from_spec(spec_dapdb)
sys.modules["dapdb"] = dapdb
spec_dapdb.loader.exec_module(dapdb)

# --------------------------
# Main comparison function
# --------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # ==================== Problem Configuration ====================
    # Configuration for random seeds (use same seeds for both solvers)
    main_seed = 2025  # Seed for problem generation
    solver_seed = 789  # Seed for algorithm initialization (must be same for both solvers)
    
    rng = np.random.default_rng(main_seed)  # Use configured seed
    
    # Problem size
    n = 20   # dimension of decision variable
    N = 12   # number of agents
    m = N    # number of constraints equals number of nodes
    
    E = 24   # number of edges in network
    
    # Box domain: For L1 regularized problems, no box constraints
    # Set to None to disable box constraints
    box_lo, box_hi = None, None
    
    # Algorithm parameters (common for both solvers)
    
    gamma = None  # Will be computed from formula for D-APDB, computed for D-APD
    max_iter = 3000
    c_alpha = 0.1
    c_beta = 0.1
    c_c = 0.1
    zeta = 1.0
    rho_shrink = 0.9
    delta = 0.1  # Only for D-APDB
    tau_multiplier = 20  # Multiplier for D-APDB: tau_dapdb = tau_dapd * tau_multiplier
    # verbose_every: print initial step size and progress every N iterations
    # Set to 500 to print initial step size and then every 500 iterations
    verbose_every = 500  
    initial_scale = 10.0
    initialization_mode = "independent"  # Must be same for both solvers
    
    # ==================== Generate Network ====================
    print("="*80)
    print("Generating Network Topology")
    print("="*80)
    network_type = "small_world"
    neighbors_list = dapd.generate_small_world_network(N, E, seed=main_seed)  # Use main_seed for network
    print(f"Network type: {network_type}, N={N}, E={E}")
    
    # ==================== Generate Problem ====================
    print("\n" + "="*80)
    print("Generating QCQP Problem with L1 Regularization")
    print("="*80)
    
    # Generate feasible QCQP problem with L1 regularization and node-specific objectives/constraints
    # Pass neighbors_list to ensure phi* > 1 is verified with consensus constraints
    print("Generating feasible QCQP problem with L1 regularization and node-specific objectives...")
    Q_list, lambda_l1, A_list, b_list, c_list = generate_feasible_qcqp_l1(n, N, rng, neighbors_list=neighbors_list)
    print(f"Generated problem: n={n}, N={N}, lambda_l1={lambda_l1:.6e}")
    
    # Aggregate objective for ground truth solving: A0 = sum(Q_i), b0 = 0 (no linear term)
    # Total objective: ||x||_1 + (1/2) * sum_i x^T Q^i x = ||x||_1 + (1/2) * x^T (sum Q^i) x
    A0_agg = np.sum(Q_list, axis=0)
    b0_agg = np.zeros(n)  # No linear term in new formulation
    q_list = None  # No q_list in new formulation
    
    # ==================== Solve Ground Truth ====================
    print("\n" + "="*80)
    print("Solving Ground Truth (with L1 regularization)")
    print("="*80)
    # Use solve_qcqp_l1_ground_truth for L1 regularized problem
    # Note: Centralized objective is ||x||_1 + (1/2) * x^T Q_agg x, so lambda_l1 = 1.0
    # Centralized solver doesn't need network information
    x_star, f_star = solve_qcqp_l1_ground_truth(A0_agg, 1.0, A_list, b_list, c_list, neighbors_list=None, verbose=True)
    
    # ==================== Ensure At Least 1-2 Tight Constraints ====================
    # Compute constraint values at optimal solution
    # Constraint: g_i(x) = (1/2) x^T A_i x + b_i^T x - c_i <= 0
    constraint_values = []
    for i in range(N):
        g_i_value = 0.5 * x_star @ (A_list[i] @ x_star) + b_list[i] @ x_star - c_list[i]
        constraint_values.append(g_i_value)
    
    constraint_values = np.array(constraint_values)
    
    # A constraint is tight if g_i(x*) = 0 (within numerical tolerance)
    tolerance = 1e-6
    tight_constraints = np.abs(constraint_values) <= tolerance
    num_tight = np.sum(tight_constraints)
    
    # Target: at least 1-2 tight constraints
    min_tight_constraints = 1
    max_tight_constraints = 2
    
    if num_tight < min_tight_constraints:
        print(f"\nAdjusting constraints to ensure at least {min_tight_constraints} tight constraint(s)...")
        # Select constraints to make tight: choose the ones with constraint values closest to 0
        # (i.e., the most negative values, which are closest to being tight)
        num_to_adjust = min(max_tight_constraints, N)
        # Sort by constraint value (ascending, most negative first)
        sorted_indices = np.argsort(constraint_values)
        # Select the top constraints (closest to 0, i.e., least negative)
        indices_to_adjust = sorted_indices[:num_to_adjust]
        
        print(f"  Adjusting constraints: {indices_to_adjust.tolist()}")
        
        # Adjust c_i for selected constraints to make them tight at x*
        # g_i(x*) = 0.5 * x*^T A_i x* + b_i^T x* - c_i = 0
        # So: c_i = 0.5 * x*^T A_i x* + b_i^T x*
        for i in indices_to_adjust:
            # Compute the value that would make constraint i tight
            c_i_tight = 0.5 * x_star @ (A_list[i] @ x_star) + b_list[i] @ x_star
            old_c_i = c_list[i]
            c_list[i] = c_i_tight
            print(f"    Constraint {i}: c_i = {old_c_i:.6e} -> {c_i_tight:.6e}")
        
        # Re-solve ground truth with adjusted constraints
        print(f"  Re-solving ground truth with adjusted constraints...")
        x_star, f_star = solve_qcqp_l1_ground_truth(A0_agg, 1.0, A_list, b_list, c_list, neighbors_list=neighbors_list, verbose=False)
        
        # Recompute constraint values
        constraint_values = []
        for i in range(N):
            g_i_value = 0.5 * x_star @ (A_list[i] @ x_star) + b_list[i] @ x_star - c_list[i]
            constraint_values.append(g_i_value)
        constraint_values = np.array(constraint_values)
        tight_constraints = np.abs(constraint_values) <= tolerance
        num_tight = np.sum(tight_constraints)
    
    # ==================== Check Tight Constraints ====================
    # Also check constraints that are "almost tight" (within larger tolerance)
    tolerance_loose = 1e-3
    almost_tight = (np.abs(constraint_values) <= tolerance_loose) & (~tight_constraints)
    num_almost_tight = np.sum(almost_tight)
    
    print(f"\n{'='*80}")
    print(f"Ground Truth Solution:")
    print(f"  phi* = {f_star:.6e}")
    print(f"  ||x*|| = {np.linalg.norm(x_star):.6f}")
    print(f"\nConstraint Analysis at Optimal Solution:")
    print(f"  Total constraints: {N}")
    print(f"  Tight constraints (|g_i(x*)| <= {tolerance:.0e}): {num_tight}")
    print(f"  Almost tight (|g_i(x*)| <= {tolerance_loose:.0e}): {num_almost_tight}")
    print(f"  Active constraints (g_i(x*) >= 0): {np.sum(constraint_values >= -tolerance)}")
    print(f"  Inactive constraints (g_i(x*) < -{tolerance:.0e}): {np.sum(constraint_values < -tolerance)}")
    print(f"\n  Constraint values at x*:")
    for i in range(N):
        status = "TIGHT" if tight_constraints[i] else ("ALMOST" if almost_tight[i] else "INACTIVE")
        print(f"    Constraint {i:2d}: g_i(x*) = {constraint_values[i]:12.6e} [{status}]")
    print(f"{'='*80}")
    
    # ==================== Prepare Per-Node Constraints ====================
    # Each node gets exactly one constraint: constraint i
    pernode_constraints = []
    for i in range(N):
        A_i = [A_list[i]]  # Node i gets constraint i
        b_i = [b_list[i]]
        c_i = [c_list[i]]
        pernode_constraints.append((A_i, b_i, c_i))
    
    print(f"\nConstraint allocation: Each node gets exactly one constraint")
    
    # ==================== Compute Parameters ====================
    # Compute Lipschitz constants for D-APD (using aggregated objective)
    # Note: L1 term doesn't affect Lipschitz constant of smooth part
    L_obj = dapd.compute_lipschitz_constant(A0_agg, b0_agg)
    L_constraints = dapd.compute_constraint_lipschitz_constants(A_list, b_list)
    
    # Compute dual variable bound (using aggregated objective)
    # Note: For L1 regularized problems, we may need different bound computation
    # For now, use the same method (L1 term doesn't affect dual variable bound computation)
    # For L1 problems without box constraints, pass None for box_lo/box_hi
    try:
        B_theta, x_slater, slater_gaps = dapd.compute_dual_variable_bound(
            A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi
        )
    except Exception as e:
        print(f"Failed to compute dual variable bound: {e}")
        B_theta = 1000.0
    
    # ==================== Compute Initial Tau ====================
    # Compute initial tau_list using D-APD's method (compute_initial_tau_per_node)
    # D-APD will use this tau_list directly
    # D-APDB will use tau_list * tau_multiplier
    print(f"\nComputing initial tau_list (using D-APD's method)...")
    
    # Compute Lipschitz constant for objective
    if Q_list is not None:
        # For node-specific objectives with L1 regularization
        A0_for_tau = np.sum(Q_list, axis=0)  # sum Q^i
        b0_for_tau = np.zeros(n)  # No linear term
        L_obj = dapd.compute_lipschitz_constant(A0_for_tau, b0_for_tau)
        # Compute node-specific L_f_i = ||Q^i||_2
        L_f_i_list = [np.linalg.norm(Q_i, ord=2) for Q_i in Q_list]
        print(f"  L_f_i_list: {L_f_i_list}")
    else:
        A0_for_tau = A0_agg
        b0_for_tau = b0_agg
        L_obj = dapd.compute_lipschitz_constant(A0_for_tau, b0_for_tau)
        L_f_i_list = None
    
    # Compute initial tau for each node using compute_initial_tau_per_node (D-APD's method)
    tau_list_dapd, tau_components_list = dapd.compute_initial_tau_per_node(
        A0_for_tau, b0_for_tau, pernode_constraints, box_lo, box_hi,
        L_obj, N,
        c_alpha=c_alpha, c_beta=c_beta, c_c=c_c,
        delta=0.1, zeta=zeta,
        L_f_i_list=L_f_i_list
    )
    
    # For D-APDB, multiply D-APD's tau_list by tau_multiplier
    tau_list_dapdb = [tau_i * tau_multiplier for tau_i in tau_list_dapd]
    
    print(f"  D-APD tau_list: min={min(tau_list_dapd):.6e}, max={max(tau_list_dapd):.6e}, mean={np.mean(tau_list_dapd):.6e}")
    print(f"  D-APDB tau_list: min={min(tau_list_dapdb):.6e}, max={max(tau_list_dapdb):.6e}, mean={np.mean(tau_list_dapdb):.6e} (tau_multiplier = {tau_multiplier})")
    print(f"  gamma_k is computed each iteration in both algorithms")
    
    # ==================== Run Multiple Simulations ====================
    num_simulations = 20
    print(f"\n" + "="*80)
    print(f"Running {num_simulations} Simulations (i.i.d. random initializations)")
    print("="*80)
    
    # Storage for all simulation results
    all_hist_dapd = []
    all_hist_dapdb = []
    all_x_bar_dapd = []
    all_x_bar_dapdb = []
    all_backtrack_stats = []  # Store backtracking statistics for D-APDB
    
    for sim_idx in range(num_simulations):
        sim_seed = solver_seed + sim_idx  # Different seed for each simulation
        print(f"\n--- Simulation {sim_idx + 1}/{num_simulations} (seed={sim_seed}) ---")
        
        # Generate initial points with constraint violation > 1 and large objective value
        # Use larger scale to ensure initial objective is larger than optimal value
        sim_rng = np.random.default_rng(sim_seed)
        # For L1 problems, optimal x* has norm around 40-45, so use larger scale (50-60) 
        # to generate initial points with larger objective values
        initial_scale_for_violation = 50.0  # Larger scale to get larger objective values
        
        if initialization_mode == "independent":
            # Generate independent initial points, each with violation > 1
            initial_points_list = []
            for i in range(N):
                # Each node gets its own constraint (pernode_constraints[i])
                A_list_i, b_list_i, c_list_i = pernode_constraints[i]
                x_init_i, max_viol_i = generate_initial_point_with_violation(
                    n, A_list_i, b_list_i, c_list_i, box_lo, box_hi, sim_rng, 
                    target_violation=1.1, initial_scale=initial_scale_for_violation
                )
                initial_points_list.append(x_init_i)
            if sim_idx == 0:
                print(f"Generated initial points with constraint violations > 1.1 (scale={initial_scale_for_violation})")
        elif initialization_mode == "connected":
            # Generate one initial point and use it for all nodes in connected components
            # Use the first node's constraints to generate the point
            A_list_0, b_list_0, c_list_0 = pernode_constraints[0]
            x_init_0, max_viol_0 = generate_initial_point_with_violation(
                n, A_list_0, b_list_0, c_list_0, box_lo, box_hi, sim_rng, 
                target_violation=1.1, initial_scale=initial_scale_for_violation
            )
            initial_points_list = [x_init_0.copy() for _ in range(N)]
            if sim_idx == 0:
                print(f"Generated initial point with constraint violation > 1.1 (scale={initial_scale_for_violation}, used for all nodes)")
        else:
            # Fallback to random initialization
            initial_points_list = [initial_scale * sim_rng.standard_normal(n) for _ in range(N)]
        
        # Only print verbose output for the first simulation to avoid clutter
        # This ensures we see the initial step size print only once
        sim_verbose = verbose_every if sim_idx == 0 else 0

        # Run D-APD
        if sim_idx == 0:
            print("Running D-APD Solver...")
        # For D-APD, use the computed tau_list_dapd
        # Note: New formulation has L1 regularization, so pass lambda_l1
        x_bar_dapd, hist_dapd = dapd.d_apd_qcqp_merely_convex(
            A0_agg, b0_agg, 0.0, pernode_constraints, box_lo, box_hi,
            Q_list=Q_list, q_list=None,  # Node-specific objectives (no q_list in new formulation)
            N=N, max_iter=max_iter, seed=sim_seed,
            c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_c,
            zeta=zeta, tau=None, gamma=None,  # tau=None, but tau_list is provided
            verbose_every=sim_verbose, initial_scale=initial_scale,
            phi_star=f_star, tol=1e-8, normalize_consensus_error=False,
            use_optimal_consensus_error=False, x_star=x_star,
            neighbors_list=neighbors_list, initialization_mode=initialization_mode,
            initial_points=initial_points_list,
            lambda_l1=lambda_l1,  # Pass L1 regularization coefficient
            tau_list=tau_list_dapd  # Use D-APD's tau_list
        )
        all_hist_dapd.append(hist_dapd)
        all_x_bar_dapd.append(x_bar_dapd)
        
        # Run D-APDB
        # Note: E_use_gradient_form parameter controls Term 5 in E_i^k:
        #   - False (default): 2(φ_i(x) - φ_i(x_i^k) - <∇φ_i(x_i^k), x - x_i^k>)
        #   - True: <∇φ_i(x) - ∇φ_i(x_i^k), x - x_i^k>
        # To use gradient form, add: E_use_gradient_form=True
        if sim_idx == 0:
            print("Running D-APDB Solver...")
        # For D-APDB, use tau_list_dapdb = tau_list_dapd * tau_multiplier
        # Note: New formulation has L1 regularization, so pass lambda_l1
        x_bar_dapdb, hist_dapdb, stats_dapdb = dapdb.d_apdb_qcqp_merely_convex(
            A0_agg, b0_agg, 0.0, pernode_constraints, box_lo, box_hi,
            Q_list=Q_list, q_list=None,  # Node-specific objectives (no q_list in new formulation)
            N=N, max_iter=max_iter, seed=sim_seed,
            c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_c,
            zeta=zeta, tau_bar=None, gamma=gamma,  # tau_bar=None, but tau_list is provided
            rho_shrink=rho_shrink, delta=delta,
            phi_star=f_star, verbose_every=sim_verbose,
            initial_scale=initial_scale, neighbors_list=neighbors_list,
            initialization_mode=initialization_mode,
            initial_points=initial_points_list,
            lambda_l1=lambda_l1,  # Pass L1 regularization coefficient
            E_use_gradient_form=True,
            tau_list=tau_list_dapdb  # Use D-APD's tau_list * tau_multiplier
        )
        all_hist_dapdb.append(hist_dapdb)
        all_x_bar_dapdb.append(x_bar_dapdb)
        
        # Store backtracking statistics
        all_backtrack_stats.append(stats_dapdb)
        
        # Only print progress for first simulation
        if sim_idx == 0:
            print(f"D-APD result: ||x_bar - x*|| = {np.linalg.norm(x_bar_dapd - x_star):.6f}")
            print(f"D-APDB result: ||x_bar - x*|| = {np.linalg.norm(x_bar_dapdb - x_star):.6f}")
        elif (sim_idx + 1) % 5 == 0:
            # Print progress every 5 simulations
            print(f"  Completed {sim_idx + 1}/{num_simulations} simulations...")
    
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
    
    # Aggregate all metrics
    grad_calls_dapd, objs_dapd_mean, objs_dapd_std = aggregate_histories(all_hist_dapd, 0)
    _, maxV_dapd_mean, maxV_dapd_std = aggregate_histories(all_hist_dapd, 1)
    _, cons_dapd_mean, cons_dapd_std = aggregate_histories(all_hist_dapd, 2)
    _, avgV_dapd_mean, avgV_dapd_std = aggregate_histories(all_hist_dapd, 3)
    _, subopt_dapd_mean, subopt_dapd_std = aggregate_histories(all_hist_dapd, 4)
    _, x_bar_norm_sq_dapd_mean, x_bar_norm_sq_dapd_std = aggregate_histories(all_hist_dapd, 6)  # Index 6 for x_bar_norm_sq
    _, cons_err_sq_sum_dapd_mean, cons_err_sq_sum_dapd_std = aggregate_histories(all_hist_dapd, 7)  # Index 7 for cons_err_sq_sum
    
    grad_calls_dapdb, objs_dapdb_mean, objs_dapdb_std = aggregate_histories(all_hist_dapdb, 0)
    _, maxV_dapdb_mean, maxV_dapdb_std = aggregate_histories(all_hist_dapdb, 1)
    _, cons_dapdb_mean, cons_dapdb_std = aggregate_histories(all_hist_dapdb, 2)
    _, avgV_dapdb_mean, avgV_dapdb_std = aggregate_histories(all_hist_dapdb, 3)
    _, subopt_dapdb_mean, subopt_dapdb_std = aggregate_histories(all_hist_dapdb, 4)
    # For backtracking, aggregate by iteration number (not gradient calls)
    # Extract backtrack values for each simulation
    all_backtrack_values = []
    max_iterations = 0
    for hist in all_hist_dapdb:
        backtrack_vals = np.array([h[6] for h in hist])  # Index 6 for backtrack iterations
        all_backtrack_values.append(backtrack_vals)
        max_iterations = max(max_iterations, len(backtrack_vals))
    
    # Align by iteration number (pad shorter histories with last value)
    max_iterations = min(max_iterations, 20)  # Limit to first 20 iterations
    aligned_backtrack = []
    for backtrack_vals in all_backtrack_values:
        if len(backtrack_vals) >= max_iterations:
            aligned_backtrack.append(backtrack_vals[:max_iterations])
        else:
            # Pad with last value if history is shorter
            padded = np.pad(backtrack_vals, (0, max_iterations - len(backtrack_vals)), mode='edge')
            aligned_backtrack.append(padded)
    
    # Compute mean and std across simulations for each iteration
    aligned_array = np.array(aligned_backtrack)
    backtrack_dapdb_mean = np.mean(aligned_array, axis=0)
    backtrack_dapdb_std = np.std(aligned_array, axis=0)
    
    _, x_bar_norm_sq_dapdb_mean, x_bar_norm_sq_dapdb_std = aggregate_histories(all_hist_dapdb, 7)  # Index 7 for x_bar_norm_sq
    _, cons_err_sq_sum_dapdb_mean, cons_err_sq_sum_dapdb_std = aggregate_histories(all_hist_dapdb, 8)  # Index 8 for cons_err_sq_sum
    
    # Compute relative errors
    # Relative suboptimality: |f(x_bar) - f*| / |f*|
    f_star_abs = abs(f_star)
    if f_star_abs > 1e-10:
        rel_subopt_dapd = subopt_dapd_mean / f_star_abs
        rel_subopt_dapd_std = subopt_dapd_std / f_star_abs  # Standard deviation of relative suboptimality
        rel_subopt_dapdb = subopt_dapdb_mean / f_star_abs
        rel_subopt_dapdb_std = subopt_dapdb_std / f_star_abs  # Standard deviation of relative suboptimality
        
        # Log relative suboptimality: log((|varphi(x_bar^k) - optimal| / |optimal|) + 1)
        # = log(rel_subopt + 1)
        log_rel_subopt_dapd = np.log(rel_subopt_dapd + 1.0)
        log_rel_subopt_dapdb = np.log(rel_subopt_dapdb + 1.0)
        
        # Compute std for log relative suboptimality using error propagation
        # For z = log(a + 1) where a = rel_subopt, dz/da = 1/(a + 1)
        # So std(z) ≈ std(a) / (a + 1)
        log_rel_subopt_dapd_std = rel_subopt_dapd_std / (rel_subopt_dapd + 1.0)
        log_rel_subopt_dapdb_std = rel_subopt_dapdb_std / (rel_subopt_dapdb + 1.0)
    else:
        rel_subopt_dapd = np.full_like(subopt_dapd_mean, np.nan)
        rel_subopt_dapd_std = np.full_like(subopt_dapd_std, np.nan)
        rel_subopt_dapdb = np.full_like(subopt_dapdb_mean, np.nan)
        rel_subopt_dapdb_std = np.full_like(subopt_dapdb_std, np.nan)
        log_rel_subopt_dapd = np.full_like(subopt_dapd_mean, np.nan)
        log_rel_subopt_dapd_std = np.full_like(subopt_dapd_std, np.nan)
        log_rel_subopt_dapdb = np.full_like(subopt_dapdb_mean, np.nan)
        log_rel_subopt_dapdb_std = np.full_like(subopt_dapdb_std, np.nan)
    
    # Relative consensus error: ||x_i^k - x_bar^k||^2 / (N * ||x_bar^k||^2)
    # Compute mean and std for relative consensus error
    x_bar_norm_sq_dapd_denom = N * np.maximum(x_bar_norm_sq_dapd_mean, 1e-12)
    x_bar_norm_sq_dapdb_denom = N * np.maximum(x_bar_norm_sq_dapdb_mean, 1e-12)
    rel_cons_dapd = cons_err_sq_sum_dapd_mean / x_bar_norm_sq_dapd_denom
    rel_cons_dapdb = cons_err_sq_sum_dapdb_mean / x_bar_norm_sq_dapdb_denom
    
    # Compute std for relative consensus error using error propagation
    # For ratio z = a/b, std(z) ≈ |z| * sqrt((std(a)/a)^2 + (std(b)/b)^2)
    # But we need to be careful with near-zero values
    rel_cons_dapd_std = np.abs(rel_cons_dapd) * np.sqrt(
        np.maximum((cons_err_sq_sum_dapd_std / np.maximum(cons_err_sq_sum_dapd_mean, 1e-12))**2 + 
                   (x_bar_norm_sq_dapd_std / np.maximum(x_bar_norm_sq_dapd_mean, 1e-12))**2, 0)
    )
    rel_cons_dapdb_std = np.abs(rel_cons_dapdb) * np.sqrt(
        np.maximum((cons_err_sq_sum_dapdb_std / np.maximum(cons_err_sq_sum_dapdb_mean, 1e-12))**2 + 
                   (x_bar_norm_sq_dapdb_std / np.maximum(x_bar_norm_sq_dapdb_mean, 1e-12))**2, 0)
    )
    
    # Relative constraint violation: max_i ||(g_i(x_bar^k))_+|| / max_i ||(g_i(x_bar^0))_+||
    # Get initial max violation from first simulation
    maxV_dapd_init = all_hist_dapd[0][0][1] if len(all_hist_dapd[0]) > 0 and len(all_hist_dapd[0][0]) > 1 else 1.0
    maxV_dapdb_init = all_hist_dapdb[0][0][1] if len(all_hist_dapdb[0]) > 0 and len(all_hist_dapdb[0][0]) > 1 else 1.0
    # Average initial max violation across all simulations
    maxV_dapd_init = np.mean([hist[0][1] for hist in all_hist_dapd if len(hist) > 0])
    maxV_dapdb_init = np.mean([hist[0][1] for hist in all_hist_dapdb if len(hist) > 0])
    maxV_dapd_init_std = np.std([hist[0][1] for hist in all_hist_dapd if len(hist) > 0])
    maxV_dapdb_init_std = np.std([hist[0][1] for hist in all_hist_dapdb if len(hist) > 0])
    
    rel_maxV_dapd = maxV_dapd_mean / np.maximum(maxV_dapd_init, 1e-12)
    rel_maxV_dapdb = maxV_dapdb_mean / np.maximum(maxV_dapdb_init, 1e-12)
    
    # Compute std for relative constraint violation using error propagation
    # For ratio z = a/b, std(z) ≈ |z| * sqrt((std(a)/a)^2 + (std(b)/b)^2)
    rel_maxV_dapd_std = np.abs(rel_maxV_dapd) * np.sqrt(
        np.maximum((maxV_dapd_std / np.maximum(maxV_dapd_mean, 1e-12))**2 + 
                   (maxV_dapd_init_std / np.maximum(maxV_dapd_init, 1e-12))**2, 0)
    )
    rel_maxV_dapdb_std = np.abs(rel_maxV_dapdb) * np.sqrt(
        np.maximum((maxV_dapdb_std / np.maximum(maxV_dapdb_mean, 1e-12))**2 + 
                   (maxV_dapdb_init_std / np.maximum(maxV_dapdb_init, 1e-12))**2, 0)
    )
    
    # Final comparison (using first simulation for reference)
    print(f"\n{'='*80}")
    print(f"Final Results (averaged over {num_simulations} simulations):")
    print(f"{'='*80}")
    avg_dist_dapd = np.mean([np.linalg.norm(x - x_star) for x in all_x_bar_dapd])
    avg_dist_dapdb = np.mean([np.linalg.norm(x - x_star) for x in all_x_bar_dapdb])
    print(f"  D-APD:  avg ||x - x*|| = {avg_dist_dapd:.6f}, avg f(x) = {objs_dapd_mean[-1]:.6f}, avg subopt = {subopt_dapd_mean[-1]:.6e}")
    print(f"  D-APDB: avg ||x - x*|| = {avg_dist_dapdb:.6f}, avg f(x) = {objs_dapdb_mean[-1]:.6f}, avg subopt = {subopt_dapdb_mean[-1]:.6e}")
    
    # Backtracking statistics
    if 'all_backtrack_stats' in locals() and len(all_backtrack_stats) > 0:
        num_backtrack_list = [stats['num_backtracking_iterations'] for stats in all_backtrack_stats]
        total_iter_list = [stats['total_iterations'] for stats in all_backtrack_stats]
        backtrack_ratio_list = [stats['backtracking_ratio'] for stats in all_backtrack_stats]
        
        avg_num_backtrack = np.mean(num_backtrack_list)
        avg_total_iter = np.mean(total_iter_list)
        avg_backtrack_ratio = np.mean(backtrack_ratio_list)
        
        print(f"\n  D-APDB Backtracking Statistics:")
        print(f"    Average number of iterations with eta^k > 1: {avg_num_backtrack:.1f} / {avg_total_iter:.1f}")
        print(f"    Average backtracking ratio: {avg_backtrack_ratio:.2%}")
        print(f"    Min/Max backtracking iterations: {min(num_backtrack_list)} / {max(num_backtrack_list)}")
    
    print(f"\n  Ground truth: f* = {f_star:.6e}")
    print(f"{'='*80}")
    
    # ==================== Plotting (4 Separate Figures) ====================
    import datetime
    import os
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory and subfolder
    output_dir = "main-qcqp-experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    if network_type == "small_world":
        network_info = f"SW_N{N}_E{E}"
    else:
        network_info = f"UNK_N{N}"
    
    # Create subfolder with timestamp and parameters
    subfolder_name = f"{network_info}_n{n}_init{initialization_mode}_seed{main_seed}_{solver_seed}_nsims{num_simulations}_{timestamp}"
    subfolder_path = os.path.join(output_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    base_filename = f"qcqp_comparison_{subfolder_name}"
    
    # 1. Objective Function
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax1.plot(grad_calls_dapd, objs_dapd_mean, lw=2, label='D-APD (mean)', color='blue')
    ax1.fill_between(grad_calls_dapd, objs_dapd_mean - objs_dapd_std, objs_dapd_mean + objs_dapd_std, 
                      alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB with shaded region
    ax1.plot(grad_calls_dapdb, objs_dapdb_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax1.fill_between(grad_calls_dapdb, objs_dapdb_mean - objs_dapdb_std, objs_dapdb_mean + objs_dapdb_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    ax1.axhline(f_star, color='k', ls=':', alpha=0.5, label='$\\varphi^*$')
    ax1.set_title(f'Objective Function: $\\varphi(\\bar{{x}}^k)$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
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
    ax2.plot(grad_calls_dapdb, cons_dapdb_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax2.fill_between(grad_calls_dapdb, cons_dapdb_mean - cons_dapdb_std, cons_dapdb_mean + cons_dapdb_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    ax2.set_title(f'Consensus Error: $\\frac{{1}}{{N}}\\sum_{{i=1}}^N \\|x_i^k - \\bar{{x}}^k\\|$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Number of Gradient Calls per Node')
    ax2.set_ylabel('Consensus Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    filename2 = os.path.join(subfolder_path, f"{base_filename}_consensus.png")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename2}")
    plt.close()
    
    # 3. Constraint Violations
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    all_zero_dapd = np.all(maxV_dapd_mean < 1e-12)
    all_zero_dapdb = np.all(maxV_dapdb_mean < 1e-12)
    
    if all_zero_dapd and all_zero_dapdb:
        # All constraints satisfied - use linear scale
        ax3.plot(grad_calls_dapd, maxV_dapd_mean, lw=2, label='D-APD (mean)', color='blue', marker='.', markersize=2)
        ax3.fill_between(grad_calls_dapd, maxV_dapd_mean - maxV_dapd_std, maxV_dapd_mean + maxV_dapd_std, 
                          alpha=0.2, color='blue')
        ax3.plot(grad_calls_dapdb, maxV_dapdb_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--', marker='.', markersize=2)
        ax3.fill_between(grad_calls_dapdb, maxV_dapdb_mean - maxV_dapdb_std, maxV_dapdb_mean + maxV_dapdb_std, 
                          alpha=0.2, color='red')
        ax3.set_yscale('linear')
        ax3.axhline(0, color='gray', ls=':', alpha=0.3)
        ax3.set_title(f'Constraint Violations: $\\max_{{i \\in N}} \\{{g_i(\\bar{{x}}^k)_+\\}}$ (All Satisfied) (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    else:
        # Some violations exist - use log scale
        maxV_dapd_plot = np.maximum(maxV_dapd_mean, 1e-12)
        maxV_dapdb_plot = np.maximum(maxV_dapdb_mean, 1e-12)
        ax3.semilogy(grad_calls_dapd, maxV_dapd_plot, lw=2, label='D-APD (mean)', color='blue')
        ax3.fill_between(grad_calls_dapd, 
                          np.maximum(maxV_dapd_mean - maxV_dapd_std, 1e-12), 
                          maxV_dapd_mean + maxV_dapd_std, 
                          alpha=0.2, color='blue')
        ax3.semilogy(grad_calls_dapdb, maxV_dapdb_plot, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax3.fill_between(grad_calls_dapdb, 
                          np.maximum(maxV_dapdb_mean - maxV_dapdb_std, 1e-12), 
                          maxV_dapdb_mean + maxV_dapdb_std, 
                          alpha=0.2, color='red')
        ax3.set_title(f'Constraint Violations: $\\max_{{i \\in N}} \\{{g_i(\\bar{{x}}^k)_+\\}}$ (Log Scale) (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    
    ax3.set_xlabel('Average Number of Gradient Calls per Node')
    ax3.set_ylabel('Max Violation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    filename3 = os.path.join(subfolder_path, f"{base_filename}_constraint_violation.png")
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename3}")
    plt.close()
    
    # 4. Absolute Suboptimality
    if not all(np.isnan(subopt_dapd_mean)) and not all(np.isnan(subopt_dapdb_mean)):
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        # D-APD with shaded region
        ax4.plot(grad_calls_dapd, subopt_dapd_mean, lw=2, label='D-APD (mean)', color='blue')
        ax4.fill_between(grad_calls_dapd, subopt_dapd_mean - subopt_dapd_std, subopt_dapd_mean + subopt_dapd_std, 
                          alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        # D-APDB with shaded region
        ax4.plot(grad_calls_dapdb, subopt_dapdb_mean, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax4.fill_between(grad_calls_dapdb, subopt_dapdb_mean - subopt_dapdb_std, subopt_dapdb_mean + subopt_dapdb_std, 
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax4.set_title(f'Absolute Suboptimality: $|\\varphi(\\bar{{x}}^k) - \\varphi^*|$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Average Number of Gradient Calls per Node')
        ax4.set_ylabel('Suboptimality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        filename4 = os.path.join(subfolder_path, f"{base_filename}_suboptimality.png")
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename4}")
        plt.close()
    
    # 5. Relative Suboptimality: |f(x_bar) - f*| / |f*|
    if not all(np.isnan(rel_subopt_dapd)) and not all(np.isnan(rel_subopt_dapdb)):
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        # D-APD with shaded region
        ax5.semilogy(grad_calls_dapd, rel_subopt_dapd, lw=2, label='D-APD (mean)', color='blue')
        ax5.fill_between(grad_calls_dapd, 
                          np.maximum(rel_subopt_dapd - rel_subopt_dapd_std, 1e-12), 
                          rel_subopt_dapd + rel_subopt_dapd_std, 
                          alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        # D-APDB with shaded region
        ax5.semilogy(grad_calls_dapdb, rel_subopt_dapdb, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax5.fill_between(grad_calls_dapdb, 
                          np.maximum(rel_subopt_dapdb - rel_subopt_dapdb_std, 1e-12), 
                          rel_subopt_dapdb + rel_subopt_dapdb_std, 
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax5.set_title(f'Relative Suboptimality: $|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Average Number of Gradient Calls per Node')
        ax5.set_ylabel('Relative Suboptimality')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        filename5 = os.path.join(subfolder_path, f"{base_filename}_relative_suboptimality.png")
        plt.savefig(filename5, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename5}")
        plt.close()
    
    # 5b. Log Relative Suboptimality: log((|varphi(x_bar^k) - optimal| / |optimal|) + 1)
    if not all(np.isnan(rel_subopt_dapd)) and not all(np.isnan(rel_subopt_dapdb)):
        fig5b, ax5b = plt.subplots(figsize=(10, 6))
        # D-APD with shaded region
        ax5b.plot(grad_calls_dapd, log_rel_subopt_dapd, lw=2, label='D-APD (mean)', color='blue')
        ax5b.fill_between(grad_calls_dapd, 
                          log_rel_subopt_dapd - log_rel_subopt_dapd_std, 
                          log_rel_subopt_dapd + log_rel_subopt_dapd_std, 
                          alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
        # D-APDB with shaded region
        ax5b.plot(grad_calls_dapdb, log_rel_subopt_dapdb, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
        ax5b.fill_between(grad_calls_dapdb, 
                          log_rel_subopt_dapdb - log_rel_subopt_dapdb_std, 
                          log_rel_subopt_dapdb + log_rel_subopt_dapdb_std, 
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax5b.set_title(f'Log Relative Suboptimality: $\\log((|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|) + 1)$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax5b.set_xlabel('Average Number of Gradient Calls per Node')
        ax5b.set_ylabel('$\\log((|\\varphi(\\bar{{x}}^k) - \\varphi^*|/|\\varphi^*|) + 1)$')
        ax5b.legend()
        ax5b.grid(True, alpha=0.3)
        plt.tight_layout()
        filename5b = os.path.join(subfolder_path, f"{base_filename}_log_relative_suboptimality.png")
        plt.savefig(filename5b, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename5b}")
        plt.close()
    
    # 6. Relative Consensus Error: ||x_i^k - x_bar^k||^2 / (N * ||x_bar^k||^2)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax6.semilogy(grad_calls_dapd, rel_cons_dapd, lw=2, label='D-APD (mean)', color='blue')
    ax6.fill_between(grad_calls_dapd, 
                      np.maximum(rel_cons_dapd - rel_cons_dapd_std, 1e-12), 
                      rel_cons_dapd + rel_cons_dapd_std, 
                      alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB with shaded region
    ax6.semilogy(grad_calls_dapdb, rel_cons_dapdb, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax6.fill_between(grad_calls_dapdb, 
                      np.maximum(rel_cons_dapdb - rel_cons_dapdb_std, 1e-12), 
                      rel_cons_dapdb + rel_cons_dapdb_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    ax6.set_title(f'Relative Consensus Error: $\\|x_i^k - \\bar{{x}}^k\\|^2/(N\\|\\bar{{x}}^k\\|^2)$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Average Number of Gradient Calls per Node')
    ax6.set_ylabel('Relative Consensus Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    filename6 = os.path.join(subfolder_path, f"{base_filename}_relative_consensus.png")
    plt.savefig(filename6, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename6}")
    plt.close()
    
    # 7. Relative Constraint Violation: max_i ||(g_i(x_bar^k))_+|| / max_i ||(g_i(x_bar^0))_+||
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    # D-APD with shaded region
    ax7.semilogy(grad_calls_dapd, rel_maxV_dapd, lw=2, label='D-APD (mean)', color='blue')
    ax7.fill_between(grad_calls_dapd, 
                      np.maximum(rel_maxV_dapd - rel_maxV_dapd_std, 1e-12), 
                      rel_maxV_dapd + rel_maxV_dapd_std, 
                      alpha=0.2, color='blue', label=f'D-APD (±1 std, {num_simulations} sims)')
    # D-APDB with shaded region
    ax7.semilogy(grad_calls_dapdb, rel_maxV_dapdb, lw=2, label='D-APDB (mean)', color='red', linestyle='--')
    ax7.fill_between(grad_calls_dapdb, 
                      np.maximum(rel_maxV_dapdb - rel_maxV_dapdb_std, 1e-12), 
                      rel_maxV_dapdb + rel_maxV_dapdb_std, 
                      alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
    ax7.set_title(f'Relative Constraint Violation: $\\max_{{i\\in\\mathcal{{N}}}} \\|(g_i(\\bar{{x}}^k))_+\\|/\\max_{{i\\in\\mathcal{{N}}}} \\|(g_i(\\bar{{x}}^0))_+\\|$ (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Average Number of Gradient Calls per Node')
    ax7.set_ylabel('Relative Constraint Violation')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    filename7 = os.path.join(subfolder_path, f"{base_filename}_relative_constraint_violation.png")
    plt.savefig(filename7, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename7}")
    plt.close()
    
    # 8. Backtrack Iterations (D-APDB only) - First 20 iterations only
    if not all(np.isnan(backtrack_dapdb_mean)):
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        # Limit to first 20 iterations
        max_points = min(20, len(backtrack_dapdb_mean))
        # Use iteration numbers (0, 1, 2, ..., max_points-1) as x-axis
        iterations = np.arange(max_points)
        backtrack_dapdb_mean_limited = backtrack_dapdb_mean[:max_points]
        backtrack_dapdb_std_limited = backtrack_dapdb_std[:max_points]
        
        # D-APDB with shaded region
        ax8.plot(iterations, backtrack_dapdb_mean_limited, lw=2, label='D-APDB (mean)', color='red', linestyle='--', marker='o', markersize=4)
        ax8.fill_between(iterations, 
                          backtrack_dapdb_mean_limited - backtrack_dapdb_std_limited, 
                          backtrack_dapdb_mean_limited + backtrack_dapdb_std_limited, 
                          alpha=0.2, color='red', label=f'D-APDB (±1 std, {num_simulations} sims)')
        ax8.set_title(f'Total Backtrack Iterations per Iteration (First 20 Iterations) (N={N}, n={n}, {num_simulations} simulations)', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Iteration Number')
        ax8.set_ylabel('Total Backtrack Iterations (All Nodes)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        filename8 = os.path.join(subfolder_path, f"{base_filename}_backtrack.png")
        plt.savefig(filename8, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename8}")
        plt.close()
    
    print(f"\nAll figures saved in folder: {subfolder_path}")
    print(f"Subfolder path: {subfolder_path}")

