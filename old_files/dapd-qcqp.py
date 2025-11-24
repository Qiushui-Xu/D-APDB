import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Import shared utility functions
from utils import (
    compute_lipschitz_constant,
    compute_constraint_lipschitz_constants,
    compute_jacobian_bound_for_node_i,
    compute_dual_variable_bound_for_node_i,
    find_slater_point_for_node,
    compute_dual_variable_bound,
    find_slater_point,
    compute_constraint_gradient_bound,
    compute_initial_tau_per_node,
    compute_initial_tau,
    random_orthonormal,
    rand_psd_merely_convex,
    solve_qcqp_ground_truth,
    generate_small_world_network,
    generate_fully_connected_network,
    get_neighbors,
    sum_neighbor_diffs,
    generate_feasible_qcqp
)

# --------------------------
# Distributed APD (D-APD) for convex QCQP + box
# --------------------------

# Removed duplicate utility functions - now imported from utils.py
# The following functions were moved to utils.py:
# - compute_lipschitz_constant
# - compute_constraint_lipschitz_constants
# - compute_jacobian_bound_for_node_i
# - find_slater_point_for_node
# - compute_dual_variable_bound_for_node_i
# - compute_dual_variable_bound
# - find_slater_point
# - compute_constraint_gradient_bound
# - compute_initial_tau_per_node
# - compute_initial_tau
# - random_orthonormal
# - rand_psd_merely_convex
# - solve_qcqp_ground_truth
# - generate_small_world_network
# - generate_fully_connected_network
# - get_neighbors
# - sum_neighbor_diffs

# --------------------------
# Distributed APD (D-APD) for convex QCQP + box
# --------------------------
def d_apd_qcqp_merely_convex(A0, b0, pernode_constraints,
                              box_lo, box_hi,
                              N=3, max_iter=2000, seed=0,
                              c_alpha=0.1, c_beta=0.1, c_c=0.1,
                              zeta=1.0, tau=0.08, gamma=None,
                              verbose_every=200, initial_scale=1.0,
                              f_star=None, tol=1e-8, normalize_consensus_error=False,
                              use_optimal_consensus_error=False, x_star=None,
                              neighbors_list=None, initialization_mode="connected"):
    """
    pernode_constraints: list of length N
      pernode_constraints[i] = (A_list_i, b_list_i, c_list_i)
      with Aj PSD (merely convex). Box handled via prox (clip).
    """
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    
    # Use provided network topology or generate fully connected network as default
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    d_max = max(len(neighbors_list[i]) for i in range(N))  # maximum degree

    # Local smooth objective: f_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
    def grad_f_i(x):
        return (A0 @ x + b0) / N

    # Local convex constraints for node i:
    def g_i_of(i, x):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        return np.array([0.5 * x @ (Aik @ x) + bik @ x + cik
                         for Aik, bik, cik in zip(A_list_i, b_list_i, c_list_i)])

    # Jacobian^T * theta for node i:
    # ∇ g_{i,k}(x) = A_{i,k} x + b_{i,k}
    def jacT_theta_i(i, x, theta):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        v = np.zeros_like(x)
        for th, Aik, bik, _ in zip(theta, A_list_i, b_list_i, c_list_i):
            v += th * (Aik @ x + bik)
        return v

    # Prox for box: projection onto X=[box_lo, box_hi]^n
    def prox_primal(v, tau):
        return np.clip(v, box_lo, box_hi)
    

    # Dual projection onto R_+^{m_i}
    def proj_R_plus(theta):
        return np.maximum(theta, 0.0)

    # Init states - initialize based on mode
    if initialization_mode == "connected":
        # Connected mode: nodes with edges have the same initial values
        # Group nodes by connected components
        visited = [False] * N
        node_groups = []
        
        for i in range(N):
            if not visited[i]:
                # Start a new connected component
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        # Add all neighbors to stack
                        for neighbor in neighbors_list[node]:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                node_groups.append(component)
        
        # Initialize each connected component with the same random value
        x = [None] * N
        for group in node_groups:
            x_init = rng.uniform(box_lo, box_hi, size=n) * initial_scale
            for node in group:
                x[node] = x_init.copy()
    else:
        # Independent mode: each node has independent initial values
        x = [rng.uniform(box_lo, box_hi, size=n) * initial_scale for _ in range(N)]
    
    # Initialize dual variables
    theta = []
    for i in range(N):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        m_i = len(A_list_i)
        theta.append(np.zeros(m_i))
    
    # Initialize auxiliary variables
    s = [np.zeros(n) for _ in range(N)]
    q = [np.zeros(n) for _ in range(N)]
    
    # Compute initial tau if not provided
    if tau is None:
        # Compute Lipschitz constants
        L_obj = compute_lipschitz_constant(A0, b0)
        L_constraints = []
        for i in range(N):
            A_list_i, b_list_i, c_list_i = pernode_constraints[i]
            L_list_i = compute_constraint_lipschitz_constants(A_list_i, b_list_i)
            L_constraints.append(L_list_i)
        
        # Compute dual variable bound
        # Aggregate all constraints for global bound computation
        A_list_all = []
        b_list_all = []
        c_list_all = []
        for i in range(N):
            A_list_i, b_list_i, c_list_i = pernode_constraints[i]
            A_list_all.extend(A_list_i)
            b_list_all.extend(b_list_i)
            c_list_all.extend(c_list_i)
        
        B_theta, _, _ = compute_dual_variable_bound(A0, b0, A_list_all, b_list_all, c_list_all, box_lo, box_hi)
        
        # Compute initial tau using legacy function
        tau, _ = compute_initial_tau(
            A0, b0, A_list_all, b_list_all, c_list_all, box_lo, box_hi,
            B_theta, L_obj, L_constraints, N,
            c_alpha=c_alpha, c_beta=c_beta, c_c=c_c, delta=0.1, zeta=zeta
        )
    
    # Initialize step sizes
    tau_i = tau
    sigma_i = zeta * tau_i
    
    # Initialize algorithm parameters
    alpha_i = c_alpha / tau_i
    beta_i = c_beta / tau_i
    
    # Compute gamma if not provided
    if gamma is None:
        d_max = max(len(neighbors_list[i]) for i in range(N))
        gamma = 1.0 / (2.0 * d_max + 1.0)
    
    # History
    hist = []
    
    # Gradient call tracking
    grad_calls_per_node = [0] * N
    
    # Main loop
    for k in range(max_iter):
        # Update for each node
        for i in range(N):
            # Compute gradient
            grad_calls_per_node[i] += 1  # Node i counts its own gradient call: grad_f_i
            x_next = prox_primal(x[i] - tau_i * (grad_f_i(x[i]) + q[i]), tau_i)
            # Note: g_i_of computes constraint function values, not gradients, so we don't count it
            theta_next = proj_R_plus(theta[i] + sigma_i * g_i_of(i, x_next))
            
            # Update auxiliary variables
            s[i] = s[i] + gamma * sum_neighbor_diffs(x, i, neighbors_list)
            q[i] = jacT_theta_i(i, x_next, theta_next) + s[i]
            
            # Update states
            x[i] = x_next
            theta[i] = theta_next
        
        # Compute metrics every verbose_every iterations
        if (k + 1) % verbose_every == 0 or k == 0:
            # Compute average objective
            obj = np.mean([0.5 * x[i] @ (A0 @ x[i]) + b0 @ x[i] for i in range(N)]) / N
            
            # Compute constraint violations
            max_viol = 0.0
            cons_err = 0.0
            avg_viol = 0.0
            for i in range(N):
                A_list_i, b_list_i, c_list_i = pernode_constraints[i]
                g_vals = g_i_of(i, x[i])
                max_viol = max(max_viol, np.max(np.maximum(g_vals, 0.0)))
                cons_err += np.sum(np.maximum(g_vals, 0.0))
                avg_viol += np.mean(np.maximum(g_vals, 0.0))
            cons_err /= N
            avg_viol /= N
            
            # Compute suboptimality if f_star is provided
            subopt = None
            if f_star is not None:
                subopt = abs(obj - f_star)
            
            # Compute average gradient calls
            avg_grad_calls = sum(grad_calls_per_node) / N
            
            hist.append((obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls))
            
            if verbose_every > 0:
                print(f"Iter {k+1}: obj={obj:.6e}, max_viol={max_viol:.6e}, cons_err={cons_err:.6e}, avg_grad_calls={avg_grad_calls:.1f}")
            
            # Check convergence
            if max_viol < tol and (f_star is None or subopt < tol):
                break
    
    return x, theta, hist

# Removed duplicate utility functions - now imported from utils.py
# The following functions were moved to utils.py and are imported at the top:
# - compute_lipschitz_constant
# - compute_constraint_lipschitz_constants
# - compute_jacobian_bound_for_node_i
# - find_slater_point_for_node
# - compute_dual_variable_bound_for_node_i
# - compute_dual_variable_bound
# - find_slater_point
# - compute_constraint_gradient_bound
# - compute_initial_tau_per_node
# - compute_initial_tau
# - random_orthonormal
# - rand_psd_merely_convex
# - solve_qcqp_ground_truth
# - generate_small_world_network
# - generate_fully_connected_network
# - get_neighbors
# - sum_neighbor_diffs
# - generate_feasible_qcqp
