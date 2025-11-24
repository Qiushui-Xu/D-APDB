"""
测试 D-APD 和 D-APDB 在每次迭代时是否一致
"""
import numpy as np
import sys
import importlib.util

# Import from dapd.py
spec_dapd = importlib.util.spec_from_file_location("dapd", "dapd.py")
dapd = importlib.util.module_from_spec(spec_dapd)
sys.modules["dapd"] = dapd
spec_dapd.loader.exec_module(dapd)

# Import from dapdb.py
spec_dapdb = importlib.util.spec_from_file_location("dapdb", "dapdb.py")
dapdb = importlib.util.module_from_spec(spec_dapdb)
sys.modules["dapdb"] = dapdb
spec_dapdb.loader.exec_module(dapdb)

from utils import generate_feasible_qcqp, generate_initial_point_with_violation

# Problem configuration
main_seed = 42
solver_seed = 789
rng = np.random.default_rng(main_seed)
n = 20
N = 12
box_lo, box_hi = -5.0, 5.0
max_iter = 10  # Only test first 10 iterations
c_alpha = 0.1
c_beta = 0.1
c_c = 0.1
zeta = 1.0
initial_scale = 5.0
initialization_mode = "independent"

# Generate network
neighbors_list = dapd.generate_small_world_network(N, 24, seed=main_seed)

# Generate problem
print("Generating problem...")
Q_list, q_list, A_list, b_list, c_list = generate_feasible_qcqp(n, N, rng, box_lo, box_hi, neighbors_list=neighbors_list)
A0_agg = np.sum(Q_list, axis=0)
b0_agg = np.sum(q_list, axis=0)

# Solve ground truth
x_star, f_star = dapd.solve_qcqp_ground_truth(A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi, neighbors_list, constant_term=0.0)

# Prepare per-node constraints
pernode_constraints = []
for i in range(N):
    A_i = [A_list[i]]
    b_i = [b_list[i]]
    c_i = [c_list[i]]
    pernode_constraints.append((A_i, b_i, c_i))

# Compute initial tau_list
L_obj = dapd.compute_lipschitz_constant(A0_agg, b0_agg)
L_f_i_list = [np.linalg.norm(Q_i, ord=2) for Q_i in Q_list]
tau_list, _ = dapd.compute_initial_tau_per_node(
    A0_agg, b0_agg, pernode_constraints, box_lo, box_hi,
    L_obj, N,
    c_alpha=c_alpha, c_beta=c_beta, c_c=c_c,
    delta=0.1, zeta=zeta,
    L_f_i_list=L_f_i_list
)

# Generate initial points
sim_rng = np.random.default_rng(solver_seed)
initial_points_list = []
for i in range(N):
    A_list_i, b_list_i, c_list_i = pernode_constraints[i]
    x_init_i, max_viol_i = generate_initial_point_with_violation(
        n, A_list_i, b_list_i, c_list_i, box_lo, box_hi, sim_rng, target_violation=1.1
    )
    initial_points_list.append(x_init_i)

print(f"\nInitial tau_list: {tau_list}")
print(f"Testing first {max_iter} iterations...\n")

# Run D-APD
print("Running D-APD...")
x_bar_dapd, hist_dapd = dapd.d_apd_qcqp_merely_convex(
    A0_agg, b0_agg, 0.0, pernode_constraints, box_lo, box_hi,
    Q_list=Q_list, q_list=q_list,
    N=N, max_iter=max_iter, seed=solver_seed,
    c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_c,
    zeta=zeta, tau=None, gamma=None,
    verbose_every=1, initial_scale=initial_scale,
    phi_star=f_star, tol=1e-8, normalize_consensus_error=False,
    use_optimal_consensus_error=False, x_star=x_star,
    neighbors_list=neighbors_list, initialization_mode=initialization_mode,
    initial_points=initial_points_list,
    tau_list=tau_list
)

# Run D-APDB
print("\nRunning D-APDB...")
x_bar_dapdb, hist_dapdb, _ = dapdb.d_apdb_qcqp_merely_convex(
    A0_agg, b0_agg, 0.0, pernode_constraints, box_lo, box_hi,
    Q_list=Q_list, q_list=q_list,
    N=N, max_iter=max_iter, seed=solver_seed,
    c_alpha=c_alpha, c_beta=c_beta, c_varsigma=c_c,
    zeta=zeta, tau_bar=None, gamma=None,
    rho_shrink=0.5, delta=0.1,
    phi_star=f_star, verbose_every=1,
    initial_scale=initial_scale, neighbors_list=neighbors_list,
    initialization_mode=initialization_mode,
    initial_points=initial_points_list,
    E_use_gradient_form=True,
    tau_list=tau_list
)

# Compare results
print("\n" + "="*80)
print("Comparison of results:")
print("="*80)
print(f"\nNumber of iterations: D-APD={len(hist_dapd)}, D-APDB={len(hist_dapdb)}")

# Compare objective values
print("\nObjective function values:")
print("Iter | D-APD obj      | D-APDB obj     | Difference")
print("-" * 60)
for i in range(min(len(hist_dapd), len(hist_dapdb))):
    obj_dapd = hist_dapd[i][0]
    obj_dapdb = hist_dapdb[i][0]
    diff = abs(obj_dapd - obj_dapdb)
    print(f"{i:4d} | {obj_dapd:14.6e} | {obj_dapdb:14.6e} | {diff:14.6e}")

# Compare x_bar
print(f"\nFinal x_bar difference: {np.linalg.norm(x_bar_dapd - x_bar_dapdb):.6e}")
print(f"Final x_bar D-APD norm: {np.linalg.norm(x_bar_dapd):.6e}")
print(f"Final x_bar D-APDB norm: {np.linalg.norm(x_bar_dapdb):.6e}")

# Check if they are identical
if np.allclose(x_bar_dapd, x_bar_dapdb, rtol=1e-10, atol=1e-10):
    print("\n✓ SUCCESS: x_bar values are identical!")
else:
    print("\n✗ DIFFERENCE: x_bar values differ")
    max_diff_idx = np.argmax(np.abs(x_bar_dapd - x_bar_dapdb))
    print(f"  Maximum difference at index {max_diff_idx}: {np.abs(x_bar_dapd - x_bar_dapdb)[max_diff_idx]:.6e}")

# Check objective values
obj_diffs = [abs(hist_dapd[i][0] - hist_dapdb[i][0]) for i in range(min(len(hist_dapd), len(hist_dapdb)))]
max_obj_diff = max(obj_diffs) if obj_diffs else 0
if max_obj_diff < 1e-10:
    print("✓ SUCCESS: Objective values are identical across all iterations!")
else:
    print(f"✗ DIFFERENCE: Maximum objective difference: {max_obj_diff:.6e}")

