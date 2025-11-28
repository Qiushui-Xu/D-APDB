import numpy as np
import importlib.util
import sys

# Import solvers
spec_dapd = importlib.util.spec_from_file_location("dapd", "dapd.py")
dapd = importlib.util.module_from_spec(spec_dapd)
sys.modules["dapd"] = dapd
spec_dapd.loader.exec_module(dapd)

spec_dapdbo = importlib.util.spec_from_file_location("dapdbo", "dapdbo.py")
dapdbo = importlib.util.module_from_spec(spec_dapdbo)
sys.modules["dapdbo"] = dapdbo
spec_dapdbo.loader.exec_module(dapdbo)

from utils import generate_feasible_qp_l1_w_std, solve_qp_l1_ground_truth

def test():
    print("Starting test...")
    n = 20
    N = 12
    E = 24
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Test with moderate parameters
    L_mean = 100.0
    L_std_percent = 1.0
    
    print("Generating problem...")
    Q_list, q_list, _, constant_list = generate_feasible_qp_l1_w_std(
        n, N, rng, L_mean=L_mean, L_std_percent=L_std_percent
    )
    print("Problem generated.")
    
    lambda_l1 = 0.1 / N
    A0_agg = np.mean(Q_list, axis=0)
    b0_agg = np.mean(q_list, axis=0)
    constant_agg = np.mean(constant_list)
    neighbors_list = dapd.generate_small_world_network(N, E, seed=seed)
    
    print("Solving ground truth...")
    x_star, f_star = solve_qp_l1_ground_truth(
        A0_agg, b0_agg, lambda_l1, neighbors_list, 
        constant_term=constant_agg, use_centralized_scale=True
    )
    print(f"Ground truth solved: {f_star}")

    # Test D-APDB
    print("Testing D-APDB...")
    L_f_i_list = [np.linalg.norm(Q, ord=2) for Q in Q_list]
    tau_multiplier = 5.0
    tau_list = [tau_multiplier / L for L in L_f_i_list]
    c_gamma = 1 / (2 * E)
    
    initial_points_list = [rng.standard_normal(n) for _ in range(N)]
    
    dapdbo.d_apdb_unconstrained(
        N=N, n=n, max_iter=100, seed=seed,
        c_alpha=0.4, c_varsigma=0.4, c_gamma=c_gamma,
        rho_shrink=0.9, delta=0.1,
        verbose_every=10, initial_scale=1.0,
        phi_star=f_star, tol=1e-6,
        neighbors_list=neighbors_list, initialization_mode='independent',
        lambda_l1=lambda_l1, initial_points=initial_points_list,
        Q_list=Q_list, q_list=q_list, tau_list=tau_list,
        constant_list=constant_list
    )
    print("Test finished successfully.")

if __name__ == "__main__":
    test()

