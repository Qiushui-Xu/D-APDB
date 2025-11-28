import numpy as np
import importlib.util
import sys
import time
from itertools import product
import csv

# Import solvers
spec_dapd = importlib.util.spec_from_file_location("dapd", "dapd.py")
dapd = importlib.util.module_from_spec(spec_dapd)
sys.modules["dapd"] = dapd
spec_dapd.loader.exec_module(dapd)

spec_dapdbo = importlib.util.spec_from_file_location("dapdbo", "dapdbo.py")
dapdbo = importlib.util.module_from_spec(spec_dapdbo)
sys.modules["dapdbo"] = dapdbo
spec_dapdbo.loader.exec_module(dapdbo)

# Import utils
from utils import generate_feasible_qp_l1_w_std, solve_qp_l1_ground_truth

def run_grid_search():
    print("="*80)
    print("Grid Search for D-APDB Advantage")
    print("="*80)
    
    # Fixed parameters
    n = 20
    N = 12
    E = 24  # Sparse network
    seed = 42
    max_iter = 1000
    rng = np.random.default_rng(seed)
    
    # Network topology (fixed)
    neighbors_list = dapd.generate_small_world_network(N, E, seed=seed)
    
    # Algorithm constants (fixed to reasonable values)
    c_alpha = 0.4
    c_beta = 0.1
    c_varsigma = 0.4
    delta = 0.1
    zeta = 1.0
    rho_shrink = 0.9
    
    # Parameter grid
    grid = {
        'L_mean': [10.0, 100.0, 1000.0],           # Condition number / difficulty
        'L_std_percent': [0.1, 1.0, 2.0],          # Heterogeneity (0.1=uniform, 2.0=very heterogeneous)
        'tau_multiplier': [1.0, 5.0, 20.0],        # Initial step size aggression
        'init_mode': ['independent', 'connected'], # Initial consensus state
        'lambda_l1': [0.1, 1.0]                    # Regularization strength
    }
    
    results = []
    
    # Generate all combinations
    keys = grid.keys()
    combinations = list(product(*grid.values()))
    total_combs = len(combinations)
    
    print(f"Total combinations to test: {total_combs}")
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        print(f"\nTesting combo {i+1}/{total_combs}: {params}")
        
        # 1. Generate Problem
        # Use generate_feasible_qp_l1_w_std for controlled heterogeneity
        Q_list, q_list, _, constant_list = generate_feasible_qp_l1_w_std(
            n, N, rng, L_mean=params['L_mean'], L_std_percent=params['L_std_percent']
        )
        
        # Use the lambda from grid
        lambda_l1 = params['lambda_l1'] / N  # Per-node lambda
        
        # Aggregate for ground truth
        A0_agg = np.mean(Q_list, axis=0)
        b0_agg = np.mean(q_list, axis=0)
        constant_agg = np.mean(constant_list)
        
        # Solve Ground Truth
        x_star, f_star = solve_qp_l1_ground_truth(
            A0_agg, b0_agg, lambda_l1, neighbors_list, 
            constant_term=constant_agg, use_centralized_scale=True
        )
        
        # Initial points
        initial_scale = 1.0
        if params['init_mode'] == 'independent':
            initial_points_list = [initial_scale * rng.standard_normal(n) for _ in range(N)]
        else: # connected (all same)
            x_init = initial_scale * rng.standard_normal(n)
            initial_points_list = [x_init.copy() for _ in range(N)]
            
        # Compute Lipschitz constants
        L_f_i_list = [np.linalg.norm(Q, ord=2) for Q in Q_list]
        
        # --- Run D-APD (Benchmark: Conservative Step Size) ---
        # Use a safe theoretical step size for D-APD to represent "standard" performance
        tau_dapd_list = [0.1 / L for L in L_f_i_list]
        
        # Compute gamma for D-APD
        tau_dapd_max = max(tau_dapd_list)
        temp_d_max = max(len(neighbors_list[idx]) for idx in range(N))
        temp_sigma0_max = zeta * tau_dapd_max
        computed_gamma_dapd = 1.0 / (2.0 * temp_d_max * temp_sigma0_max * N * ((2.0 / c_alpha) + (1.0 / c_varsigma)))
        
        x_bar_dapd, hist_dapd = dapd.d_apd_qcqp_merely_convex(
            A0_agg, b0_agg, None, [([], [], []) for _ in range(N)], None, None,
            Q_list=Q_list, q_list=q_list,
            N=N, max_iter=max_iter, seed=seed,
            c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_varsigma,
            zeta=zeta, tau=None, gamma=computed_gamma_dapd,
            verbose_every=0, initial_scale=initial_scale,
            phi_star=f_star, tol=1e-6,
            neighbors_list=neighbors_list, initialization_mode=params['init_mode'],
            B_theta=[0.0] * N, lambda_l1=lambda_l1, initial_points=initial_points_list,
            tau_list=tau_dapd_list,
            constant_list=constant_list
        )
        
        # --- Run D-APDB (Aggressive Initial Step Size) ---
        tau_dapdbo_list = [params['tau_multiplier'] / L for L in L_f_i_list]
        c_gamma_dapdbo = 1 / (2 * E) # Large gamma coefficient
        
        x_bar_dapdbo, hist_dapdbo, stats_dapdbo = dapdbo.d_apdb_unconstrained(
            N=N, n=n, max_iter=max_iter, seed=seed,
            c_alpha=c_alpha, c_varsigma=c_varsigma, c_gamma=c_gamma_dapdbo,
            rho_shrink=rho_shrink, delta=delta,
            verbose_every=0, initial_scale=initial_scale,
            phi_star=f_star, tol=1e-6,
            neighbors_list=neighbors_list, initialization_mode=params['init_mode'],
            lambda_l1=lambda_l1, initial_points=initial_points_list,
            Q_list=Q_list, q_list=q_list, tau_list=tau_dapdbo_list,
            constant_list=constant_list
        )
        
        # --- Evaluate ---
        # 1. Final Suboptimality
        subopt_dapd = hist_dapd[-1][4]
        subopt_dapdbo = hist_dapdbo[-1][4]
        
        # 2. Iterations to reach 1e-3 accuracy (relative to f_star)
        target_acc = 1e-3 * max(abs(f_star), 1.0)
        
        def get_iters_to_acc(hist, target):
            for i, h in enumerate(hist):
                if h[4] <= target:
                    return i
            return max_iter
            
        iters_dapd = get_iters_to_acc(hist_dapd, target_acc)
        iters_dapdbo = get_iters_to_acc(hist_dapdbo, target_acc)
        
        # 3. Backtracking activity
        backtrack_ratio = stats_dapdbo['backtracking_ratio']
        
        # Score: Positive if D-APDB is better (fewer iterations or lower final error)
        iter_improvement = iters_dapd - iters_dapdbo
        subopt_ratio = subopt_dapd / max(subopt_dapdbo, 1e-12)
        
        print(f"  D-APD: {iters_dapd} iters, Final Subopt: {subopt_dapd:.2e}")
        print(f"  D-APDB: {iters_dapdbo} iters, Final Subopt: {subopt_dapdbo:.2e}, BT Ratio: {backtrack_ratio:.2%}")
        print(f"  Improvement: {iter_improvement} iters")
        
        res = params.copy()
        res.update({
            'dapd_iters': iters_dapd,
            'dapdbo_iters': iters_dapdbo,
            'dapd_subopt': subopt_dapd,
            'dapdbo_subopt': subopt_dapdbo,
            'iter_diff': iter_improvement,
            'bt_ratio': backtrack_ratio,
            'L_min': min(L_f_i_list),
            'L_max': max(L_f_i_list)
        })
        results.append(res)

    # Sort results by iteration difference (descending) - D-APDB wins when positive
    results_sorted_iter = sorted(results, key=lambda x: x['iter_diff'], reverse=True)
    
    print("\n" + "="*80)
    print("Top 5 Configurations where D-APDB wins (by iteration count):")
    print("="*80)
    
    print(f"{'L_mean':>8} {'L_std%':>8} {'tau_mult':>10} {'init_mode':>12} {'lambda':>8} {'dapd_it':>8} {'dapdbo_it':>10} {'bt_ratio':>10}")
    print("-" * 90)
    for res in results_sorted_iter[:5]:
        print(f"{res['L_mean']:>8.1f} {res['L_std_percent']:>8.1f} {res['tau_multiplier']:>10.1f} {res['init_mode']:>12} {res['lambda_l1']:>8.2f} {res['dapd_iters']:>8} {res['dapdbo_iters']:>10} {res['bt_ratio']:>10.2%}")
    
    print("\n" + "="*80)
    print("Top 5 Configurations where D-APDB wins (by final suboptimality ratio):")
    print("="*80)
    
    # Add subopt_ratio to each result
    for res in results:
        res['subopt_ratio'] = res['dapd_subopt'] / max(res['dapdbo_subopt'], 1e-12)
    
    results_sorted_subopt = sorted(results, key=lambda x: x['subopt_ratio'], reverse=True)
    
    print(f"{'L_mean':>8} {'L_std%':>8} {'tau_mult':>10} {'init_mode':>12} {'lambda':>8} {'dapd_subopt':>12} {'dapdbo_subopt':>14} {'ratio':>8}")
    print("-" * 100)
    for res in results_sorted_subopt[:5]:
        print(f"{res['L_mean']:>8.1f} {res['L_std_percent']:>8.1f} {res['tau_multiplier']:>10.1f} {res['init_mode']:>12} {res['lambda_l1']:>8.2f} {res['dapd_subopt']:>12.2e} {res['dapdbo_subopt']:>14.2e} {res['subopt_ratio']:>8.2f}")
    
    # Save to CSV
    with open('grid_search_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print("\nFull results saved to grid_search_results.csv")
    
    # Print best configuration summary
    if results_sorted_iter[0]['iter_diff'] > 0:
        best = results_sorted_iter[0]
        print("\n" + "="*80)
        print("BEST CONFIGURATION (D-APDB wins by iteration count):")
        print("="*80)
        print(f"  L_mean = {best['L_mean']}")
        print(f"  L_std_percent = {best['L_std_percent']}")
        print(f"  tau_multiplier = {best['tau_multiplier']}")
        print(f"  init_mode = '{best['init_mode']}'")
        print(f"  lambda_l1 = {best['lambda_l1']}")
        print(f"  D-APD iterations: {best['dapd_iters']}, D-APDB iterations: {best['dapdbo_iters']}")
        print(f"  Improvement: {best['iter_diff']} iterations faster")
    else:
        print("\n" + "="*80)
        print("WARNING: D-APDB did not outperform D-APD in any configuration by iteration count.")
        print("="*80)

if __name__ == "__main__":
    run_grid_search()
