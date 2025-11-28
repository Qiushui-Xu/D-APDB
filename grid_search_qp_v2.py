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

spec_aldo = importlib.util.spec_from_file_location("aldo", "aldo.py")
aldo = importlib.util.module_from_spec(spec_aldo)
sys.modules["aldo"] = aldo
spec_aldo.loader.exec_module(aldo)

# Import utils
from utils import (
    generate_feasible_qp_l1, 
    generate_feasible_qp_l1_w_std, 
    solve_qp_l1_ground_truth
)

def run_grid_search():
    print("="*80)
    print("Grid Search for D-APDB Advantage (vs D-APD AND global-DATOS)")
    print("="*80)
    
    # Fixed parameters
    n = 20
    N = 12
    E = 24  # Sparse network
    seed = 42
    max_iter = 3000  # Increased for better convergence
    tol = 1e-4       # Relaxed tolerance
    
    # Network topology (fixed)
    neighbors_list = dapd.generate_small_world_network(N, E, seed=seed)
    
    # Parameter grid - focused on finding D-APDB advantage
    grid = {
        # Problem generation
        'problem_type': [1, 3],  # 1=generate_feasible_qp_l1, 3=generate_feasible_qp_l1_w_std
        
        # Problem parameters - higher heterogeneity to show backtracking advantage
        'L_or_gamma_mean': [100.0, 1000.0, 5000.0],
        'std_percent': [0.5, 1.0, 2.0],  # Higher heterogeneity
        
        # L1 regularization
        'lambda_l1': [0.1, 1.0, 5.0],
        
        # Step size multiplier: tau_i = tau_multiplier / L_i
        # Keep <= 2.0 to avoid divergence (larger values cause overflow)
        'tau_multiplier': [1.5, 2.0],
        
        # Algorithm parameters for D-APDB
        'c_alpha': [0.1, 0.3],
        'c_varsigma': [0.1, 0.3],
        'rho_shrink': [0.5, 0.8],
        
        # Initialization - independent shows more advantage
        'init_mode': ['independent'],
        'initial_scale': [1.0, 5.0],  # Larger initial scale
    }
    
    results = []
    
    # Generate all combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    total_combs = len(combinations)
    
    print(f"Total combinations to test: {total_combs}")
    print("This may take a while...")
    
    # Filter out invalid combinations (c_alpha + c_varsigma must be < 0.9 for delta=0.1)
    valid_combinations = []
    for values in combinations:
        params = dict(zip(keys, values))
        if params['c_alpha'] + params['c_varsigma'] < 0.9:  # 1 - delta = 0.9
            valid_combinations.append(values)
    
    print(f"Valid combinations (c_alpha + c_varsigma < 0.9): {len(valid_combinations)}")
    
    for i, values in enumerate(valid_combinations):
        params = dict(zip(keys, values))
        
        # Create new RNG for each combination for reproducibility
        rng = np.random.default_rng(seed)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\nTesting combo {i+1}/{len(valid_combinations)}: {params}")
        
        try:
            # 1. Generate Problem based on type
            if params['problem_type'] == 1:
                # generate_feasible_qp_l1: uses gamma_mean for Q generation
                Q_list, q_list, _, constant_list = generate_feasible_qp_l1(
                    n, N, rng, 
                    gamma_mean=params['L_or_gamma_mean'], 
                    gamma_std_percent=params['std_percent']
                )
            else:
                # generate_feasible_qp_l1_w_std: uses L_mean for ||Q_i||_2
                Q_list, q_list, _, constant_list = generate_feasible_qp_l1_w_std(
                    n, N, rng, 
                    L_mean=params['L_or_gamma_mean'], 
                    L_std_percent=params['std_percent']
                )
            
            # Use the lambda from grid (per-node)
            lambda_l1 = params['lambda_l1']
            
            # Compute Lipschitz constants
            L_f_i_list = [np.linalg.norm(Q, ord=2) for Q in Q_list]
            max_L_f_i = max(L_f_i_list)
            
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
            initial_scale = params['initial_scale']
            initial_points_list = [initial_scale * rng.standard_normal(n) for _ in range(N)]
            
            # Algorithm constants
            c_alpha = params['c_alpha']
            c_varsigma = params['c_varsigma']
            c_beta = 0.1
            delta = 0.1
            zeta = 1.0
            rho_shrink = params['rho_shrink']
            
            # --- Run D-APD ---
            # Use same initial step size as D-APDB for fair comparison
            # D-APD uses fixed step size, D-APDB can adapt via backtracking
            tau_multiplier = params['tau_multiplier']
            tau_dapd_list = [tau_multiplier / L for L in L_f_i_list]
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
                phi_star=f_star, tol=tol,
                neighbors_list=neighbors_list, initialization_mode=params['init_mode'],
                B_theta=[0.0] * N, lambda_l1=lambda_l1, initial_points=initial_points_list,
                tau_list=tau_dapd_list,
                constant_list=constant_list
            )
            
            # --- Run D-APDB ---
            # Same initial step size as D-APD, but D-APDB can adapt via backtracking
            tau_dapdbo_list = [tau_multiplier / L for L in L_f_i_list]
            c_gamma_dapdbo = 1 / (2 * E)
            
            x_bar_dapdbo, hist_dapdbo, stats_dapdbo = dapdbo.d_apdb_unconstrained(
                N=N, n=n, max_iter=max_iter, seed=seed,
                c_alpha=c_alpha, c_varsigma=c_varsigma, c_gamma=c_gamma_dapdbo,
                rho_shrink=rho_shrink, delta=delta,
                verbose_every=0, initial_scale=initial_scale,
                phi_star=f_star, tol=tol,
                neighbors_list=neighbors_list, initialization_mode=params['init_mode'],
                lambda_l1=lambda_l1, initial_points=initial_points_list,
                Q_list=Q_list, q_list=q_list, tau_list=tau_dapdbo_list,
                constant_list=constant_list
            )
            
            # --- Run global-DATOS (ALDO) ---
            alpha_init_aldo = 10.0 / max_L_f_i
            
            x_bar_aldo, hist_aldo = aldo.aldo_qcqp_merely_convex(
                A0_agg, b0_agg, None,  # A0, b0, c0
                None, None,            # box_lo, box_hi
                N=N, max_iter=max_iter, seed=seed,
                alpha_init=alpha_init_aldo,
                verbose_every=0, initial_scale=initial_scale,
                phi_star=f_star, tol=tol,
                neighbors_list=neighbors_list, initialization_mode=params['init_mode'],
                lambda_l1=lambda_l1, initial_points=initial_points_list,
                Q_list=Q_list, q_list=q_list, constant_list=constant_list
            )
            
            # --- Evaluate ---
            # Final Suboptimality
            subopt_dapd = hist_dapd[-1][4]
            subopt_dapdbo = hist_dapdbo[-1][4]
            subopt_aldo = hist_aldo[-1][4]
            
            # Check for numerical instability (NaN or Inf)
            if np.isnan(subopt_dapd) or np.isinf(subopt_dapd):
                print(f"  Skipped: D-APD diverged (subopt={subopt_dapd})")
                continue
            if np.isnan(subopt_dapdbo) or np.isinf(subopt_dapdbo):
                print(f"  Skipped: D-APDB diverged (subopt={subopt_dapdbo})")
                continue
            if np.isnan(subopt_aldo) or np.isinf(subopt_aldo):
                print(f"  Skipped: ALDO diverged (subopt={subopt_aldo})")
                continue
            
            # Iterations to reach target accuracy
            target_acc = tol * max(abs(f_star), 1.0)
            
            def get_iters_to_acc(hist, target):
                for idx, h in enumerate(hist):
                    # Skip NaN/Inf values
                    if np.isnan(h[4]) or np.isinf(h[4]):
                        continue
                    if h[4] <= target:
                        return idx
                return max_iter
                
            iters_dapd = get_iters_to_acc(hist_dapd, target_acc)
            iters_dapdbo = get_iters_to_acc(hist_dapdbo, target_acc)
            iters_aldo = get_iters_to_acc(hist_aldo, target_acc)
            
            # Backtracking activity
            backtrack_ratio = stats_dapdbo['backtracking_ratio']
            
            # Calculate improvements
            iter_improvement_vs_dapd = iters_dapd - iters_dapdbo
            iter_improvement_vs_aldo = iters_aldo - iters_dapdbo
            
            # D-APDB wins if it's faster than BOTH
            dapdbo_wins = (iters_dapdbo < iters_dapd) and (iters_dapdbo < iters_aldo)
            
            res = params.copy()
            res.update({
                'dapd_iters': iters_dapd,
                'dapdbo_iters': iters_dapdbo,
                'aldo_iters': iters_aldo,
                'dapd_subopt': subopt_dapd,
                'dapdbo_subopt': subopt_dapdbo,
                'aldo_subopt': subopt_aldo,
                'iter_diff_vs_dapd': iter_improvement_vs_dapd,
                'iter_diff_vs_aldo': iter_improvement_vs_aldo,
                'dapdbo_wins': dapdbo_wins,
                'bt_ratio': backtrack_ratio,
                'L_min': min(L_f_i_list),
                'L_max': max(L_f_i_list),
                'f_star': f_star
            })
            results.append(res)
            
            if dapdbo_wins:
                print(f"  *** D-APDB WINS! *** D-APD: {iters_dapd}, D-APDB: {iters_dapdbo}, ALDO: {iters_aldo}")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Filter results where D-APDB wins against BOTH
    winning_results = [r for r in results if r['dapdbo_wins']]
    
    print("\n" + "="*80)
    print(f"SUMMARY: D-APDB won in {len(winning_results)}/{len(results)} configurations")
    print("="*80)
    
    if winning_results:
        # Sort by total improvement (vs D-APD + vs ALDO)
        winning_results.sort(key=lambda x: x['iter_diff_vs_dapd'] + x['iter_diff_vs_aldo'], reverse=True)
        
        print("\nTop 10 Configurations where D-APDB beats BOTH D-APD and global-DATOS:")
        print("-" * 160)
        header = f"{'type':>4} {'L/gamma':>8} {'std%':>6} {'λ_l1':>5} {'τ_mult':>7} {'c_α':>5} {'c_ς':>5} {'ρ':>4} {'scale':>6} | {'D-APD':>6} {'D-APDB':>7} {'ALDO':>6} | {'vs_APD':>7} {'vs_ALDO':>8}"
        print(header)
        print("-" * 160)
        
        for res in winning_results[:10]:
            row = f"{res['problem_type']:>4} {res['L_or_gamma_mean']:>8.0f} {res['std_percent']:>6.1f} {res['lambda_l1']:>5.1f} {res['tau_multiplier']:>7.1f} {res['c_alpha']:>5.1f} {res['c_varsigma']:>5.1f} {res['rho_shrink']:>4.1f} {res['initial_scale']:>6.1f} | {res['dapd_iters']:>6} {res['dapdbo_iters']:>7} {res['aldo_iters']:>6} | +{res['iter_diff_vs_dapd']:>6} +{res['iter_diff_vs_aldo']:>7}"
            print(row)
        
        # Best configuration
        best = winning_results[0]
        print("\n" + "="*80)
        print("BEST CONFIGURATION (D-APDB beats both by largest margin):")
        print("="*80)
        print(f"  problem_type = {best['problem_type']} ({'generate_feasible_qp_l1' if best['problem_type']==1 else 'generate_feasible_qp_l1_w_std'})")
        print(f"  L_or_gamma_mean = {best['L_or_gamma_mean']}")
        print(f"  std_percent = {best['std_percent']}")
        print(f"  lambda_l1 = {best['lambda_l1']}")
        print(f"  tau_multiplier = {best['tau_multiplier']}")
        print(f"  c_alpha = {best['c_alpha']}")
        print(f"  c_varsigma = {best['c_varsigma']}")
        print(f"  rho_shrink = {best['rho_shrink']}")
        print(f"  initial_scale = {best['initial_scale']}")
        print(f"  init_mode = '{best['init_mode']}'")
        print(f"\nResults:")
        print(f"  D-APD iterations: {best['dapd_iters']}")
        print(f"  D-APDB iterations: {best['dapdbo_iters']}")
        print(f"  global-DATOS iterations: {best['aldo_iters']}")
        print(f"  Improvement vs D-APD: {best['iter_diff_vs_dapd']} iterations faster")
        print(f"  Improvement vs global-DATOS: {best['iter_diff_vs_aldo']} iterations faster")
        print(f"  Backtracking ratio: {best['bt_ratio']:.2%}")
    else:
        print("\nNo configuration found where D-APDB beats both D-APD and global-DATOS.")
        print("Showing top 5 configurations where D-APDB has best combined performance:")
        
        # Sort by minimum of the two improvements (want both to be positive and large)
        results.sort(key=lambda x: min(x['iter_diff_vs_dapd'], x['iter_diff_vs_aldo']), reverse=True)
        
        print("-" * 160)
        header = f"{'type':>4} {'L/gamma':>8} {'std%':>6} {'λ_l1':>5} {'τ_mult':>7} {'c_α':>5} {'c_ς':>5} {'ρ':>4} {'scale':>6} | {'D-APD':>6} {'D-APDB':>7} {'ALDO':>6} | {'vs_APD':>7} {'vs_ALDO':>8}"
        print(header)
        print("-" * 160)
        
        for res in results[:5]:
            sign_apd = '+' if res['iter_diff_vs_dapd'] >= 0 else ''
            sign_aldo = '+' if res['iter_diff_vs_aldo'] >= 0 else ''
            row = f"{res['problem_type']:>4} {res['L_or_gamma_mean']:>8.0f} {res['std_percent']:>6.1f} {res['lambda_l1']:>5.1f} {res['tau_multiplier']:>7.1f} {res['c_alpha']:>5.1f} {res['c_varsigma']:>5.1f} {res['rho_shrink']:>4.1f} {res['initial_scale']:>6.1f} | {res['dapd_iters']:>6} {res['dapdbo_iters']:>7} {res['aldo_iters']:>6} | {sign_apd}{res['iter_diff_vs_dapd']:>6} {sign_aldo}{res['iter_diff_vs_aldo']:>7}"
            print(row)
    
    # Save to CSV
    csv_filename = 'grid_search_results_v2.csv'
    with open(csv_filename, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nFull results saved to {csv_filename}")

if __name__ == "__main__":
    run_grid_search()