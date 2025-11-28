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
# Distributed APDB with Backtracking for QCQP + box
# --------------------------
def d_apdb_qcqp_merely_convex(A0, b0, c0, pernode_constraints,
                               box_lo, box_hi,
                               N=3, max_iter=2000, seed=0,
                               c_alpha=0.1, c_beta=0.1, c_varsigma=0.1,
                               zeta=1.0, tau_bar=None, gamma=None,
                               rho_shrink=0.5, delta=0.1,
                               verbose_every=200, initial_scale=5.0,
                               phi_star=None, tol=1e-8, normalize_consensus_error=False,
                               use_optimal_consensus_error=False, x_star=None,
                               neighbors_list=None, initialization_mode="connected",
                               B_theta=None, lambda_l1=None, initial_points=None,
                               E_use_gradient_form=False, tau_multiplier=0.005,
                               Q_list=None, q_list=None, tau_list=None,
                               constant_list=None):
    """
    Distributed APD with Backtracking (D-APDB) for QCQP
    
    pernode_constraints: list of length N
      pernode_constraints[i] = (A_list_i, b_list_i, c_list_i)
      with Aj PSD (merely convex). Box handled via prox (clip).
    
    Parameters:
    - c_alpha, c_beta, c_varsigma: constants > 0 with c_alpha + c_beta + c_varsigma < 1/2
    - tau_bar: initial step size parameter (if None, will use compute_initial_tau_per_node * tau_multiplier)
    - rho_i: contraction coefficient in (0,1) for backtracking (same for all nodes if scalar)
    - delta: parameter for backtracking condition
    - B_theta: bound for dual variables (if None, will compute or use large value)
    - E_use_gradient_form: bool, default False
        If False: Term 5 in E_i^k uses 2(φ_i(x) - φ_i(x_i^k) - <∇φ_i(x_i^k), x - x_i^k>)
        If True: Term 5 in E_i^k uses 2<∇φ_i(x) - ∇φ_i(x_i^k), x - x_i^k>
    - tau_multiplier: float, default 10.0
        Multiplier for initial tau computed from formula. 
        If tau_bar is None, tau_bar = min(compute_initial_tau_per_node) * tau_multiplier
    """
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    
    # Use provided network topology or generate fully connected network
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    
    # Compute node degrees d_i
    d_i = [len(neighbors_list[i]) for i in range(N)]
    
    # Local smooth objective: φ_i(x) = (1/2) x^T Q_i x + q_i^T x
    # If Q_list and q_list are provided, use node-specific objectives
    # Otherwise, use aggregated objective: φ_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
    # Note: The full objective is φ_i(x) + r_i(x), where r_i is the non-smooth term.
    # In QCQP, r_i(x) = I_{[box_lo, box_hi]^n}(x) (indicator function of box constraint).
    # If r_i == 0 (no non-smooth term), then prox_{τ r_i}(x) = x.
    if Q_list is not None:
        if q_list is not None:
            # Node-specific objectives with linear terms: φ_i(x) = (1/2) x^T Q_i x + q_i^T x
            def phi_i(i, x):
                return 0.5 * x @ (Q_list[i] @ x) + q_list[i] @ x
            
            def grad_phi_i(i, x):
                return Q_list[i] @ x + q_list[i]
        else:
            # Node-specific objectives without linear terms: φ_i(x) = (1/2) x^T Q_i x
            def phi_i(i, x):
                return 0.5 * x @ (Q_list[i] @ x)
            
            def grad_phi_i(i, x):
                return Q_list[i] @ x
    else:
        # Aggregated objective: φ_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
        def phi_i(i, x):
            return (0.5 * x @ (A0 @ x) + b0 @ x) / N
    
        def grad_phi_i(i, x):
            return (A0 @ x + b0) / N

    # Local convex constraints for node i
    # Local convex constraints for node i:
    # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
    def g_i_of(i, x):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        return np.array([0.5 * x @ (Aik @ x) + bik @ x - cik
                         for Aik, bik, cik in zip(A_list_i, b_list_i, c_list_i)])

    # Jacobian^T * theta for node i
    def jacT_theta_i(i, x, theta):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        v = np.zeros_like(x)
        for th, Aik, bik, _ in zip(theta, A_list_i, b_list_i, c_list_i):
            v += th * (Aik @ x + bik)
        return v
    
    # Merit function E_i^k for backtracking
    def E_i_k(i, x, theta, x_k, theta_k, tau_tilde_k, sigma_tilde_k, 
              alpha_k, beta_k, alpha_tilde_kp1, beta_tilde_kp1, varsigma_tilde_kp1, eta_i_k,
              use_gradient_form=False):
        """
        E_i^k(x,θ) = 
          - (1/τ̃_i^k - η_i^k(α_i^k + β_i^k) - ς̃_i^{k+1}) ||x - x_i^k||^2 
          - (1/σ̃_i^k) ||θ - θ_i^k||^2
          + (2/α̃_i^{k+1}) ||J g_i(x)^T (θ - θ_i^k)||^2 
          + (1/β̃_i^{k+1}) ||(J g_i(x) - J g_i(x_i^k))^T θ_i^k||^2
          + Term 5 (controlled by use_gradient_form):
            - If False: 2(φ_i(x) - φ_i(x_i^k) - <∇φ_i(x_i^k), x - x_i^k>)
            - If True: 2<∇φ_i(x) - ∇φ_i(x_i^k), x - x_i^k>
        """
        # Term 1: - (1/τ̃_i^k - η_i^k(α_i^k + β_i^k) - ς̃_i^{k+1}) ||x - x_i^k||^2
        dx = x - x_k
        dx_norm_sq = np.dot(dx, dx)
        term1_coeff = -(1.0 / tau_tilde_k - eta_i_k * (alpha_k + beta_k) - varsigma_tilde_kp1)
        term1 = term1_coeff * dx_norm_sq
        
        # Term 2: - (1/σ̃_i^k) ||θ - θ_i^k||^2
        dtheta = theta - theta_k
        dtheta_norm_sq = np.dot(dtheta, dtheta)
        term2 = -(1.0 / sigma_tilde_k) * dtheta_norm_sq
        
        # Term 3: + (2/α̃_i^{k+1}) ||J g_i(x)^T (θ - θ_i^k)||^2
        jac_g_x_dtheta = jacT_theta_i(i, x, dtheta)
        term3 = (2.0 / alpha_tilde_kp1) * np.dot(jac_g_x_dtheta, jac_g_x_dtheta)
        
        # Term 4: + (1/β̃_i^{k+1}) ||(J g_i(x) - J g_i(x_i^k))^T θ_i^k||^2
        jac_g_x_theta_k = jacT_theta_i(i, x, theta_k)
        jac_g_xk_theta_k = jacT_theta_i(i, x_k, theta_k)
        jac_diff = jac_g_x_theta_k - jac_g_xk_theta_k
        term4 = (1.0 / beta_tilde_kp1) * np.dot(jac_diff, jac_diff)
        
        # Term 5: Controlled by use_gradient_form
        if use_gradient_form:
            # Use gradient form: 2*<∇φ_i(x) - ∇φ_i(x_i^k), x - x_i^k>
            grad_f_x = grad_phi_i(i, x)
            grad_f_xk = grad_phi_i(i, x_k)
            grad_diff = grad_f_x - grad_f_xk
            term5 = 2 * np.dot(grad_diff, dx)
        else:
            # Use function value form: 2 * (φ_i(x) - φ_i(x_i^k) - <∇φ_i(x_i^k), x - x_i^k>)
            f_x = phi_i(i, x)
            f_xk = phi_i(i, x_k)
            grad_f_xk = grad_phi_i(i, x_k)
            term5 = 2 * (f_x - f_xk - np.dot(grad_f_xk, dx))
        
        return term1 + term2 + term3 + term4 + term5

    # Proximal operator for non-smooth term r_i:
    # prox_{τ r_i}(v) = argmin_{w∈R^n} {τ r_i(w) + (1/2)||w - v||^2}
    #
    # Helper to project onto the box when bounds exist. When lambda_l1 > 0 the
    # L1 formulation may not include box constraints; in that case box_lo/box_hi
    # can be None and no projection is applied.
    def _project_box(vec):
        lo = box_lo if box_lo is not None else -np.inf
        hi = box_hi if box_hi is not None else np.inf
        if np.isneginf(lo) and np.isposinf(hi):
            return vec
        return np.clip(vec, lo, hi)

    # If lambda_l1 is None (default QCQP case):
    #   r_i(x) = I_{[box_lo, box_hi]^n}(x)
    # If lambda_l1 > 0 (L1 regularization case, no box if bounds are None):
    #   r_i(x) = lambda_l1 * ||x||_1
    #   Note: lambda_l1 = 1/N is the per-node L1 coefficient
    # Uses L1 soft-thresholding when lambda_l1 > 0
    def prox_r_i(v, tau):
        if lambda_l1 is not None and lambda_l1 > 0:
            # Soft-thresholding for lambda_l1*||·||_1
            # Note: lambda_l1 = 1/N is the per-node L1 coefficient
            # prox_{τ * lambda_l1 * ||·||_1}(v) = sign(v) * max(|v| - τ*lambda_l1, 0)
            threshold = tau * lambda_l1
            v_soft = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
            # For L1 problems, don't apply box constraints (box_lo/box_hi should be None)
            # Only apply box if explicitly provided (for mixed L1+box problems)
            if box_lo is None and box_hi is None:
                return v_soft
            else:
                return _project_box(v_soft)
        else:
            return _project_box(v)
    
    # Alias for backward compatibility
    prox_primal = prox_r_i

    # Dual projection onto R_+ ∩ B_i (if B_theta is provided, also project to ball)
    def proj_dual(theta, B_i=None):
        theta_proj = np.maximum(theta, 0.0)
        if B_i is not None and B_i > 0:
            theta_norm = np.linalg.norm(theta_proj)
            if theta_norm > B_i:
                theta_proj = theta_proj * (B_i / theta_norm)
        return theta_proj

    def _sample_initial_vec():
        if box_lo is not None and box_hi is not None:
            return initial_scale * rng.uniform(box_lo, box_hi, size=n)
        return initial_scale * rng.standard_normal(n)

    # Init states - initialize based on mode
    if initial_points is not None:
        if len(initial_points) != N:
            raise ValueError("initial_points must have length N")
        x = [initial_points[i].copy() for i in range(N)]
    elif initialization_mode == "connected":
        visited = [False] * N
        node_groups = []
        
        for i in range(N):
            if not visited[i]:
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        for neighbor in neighbors_list[node]:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                node_groups.append(component)
        
        x = []
        for group in node_groups:
            x_init_group = _sample_initial_vec()
            for node in group:
                x.append(x_init_group.copy())
                
    elif initialization_mode == "independent":
        x = [_sample_initial_vec() for _ in range(N)]
    
    # Initialize according to new pseudocode:
    # (x_i^{-1}, θ_i^{-1}) = (x_i^0, θ_i^0)
    x_prev = [xi.copy() for xi in x]
    theta = [np.zeros(len(pernode_constraints[i][0])) for i in range(N)]
    theta_prev = [theta[i].copy() for i in range(N)]
    
    # s_i^0 = 0
    s = [np.zeros(n) for _ in range(N)]
    s_prev = [np.zeros(n) for _ in range(N)]
    
    # Compute initial tau for each node
    # If tau_list is provided, use it directly (for using same tau as D-APD)
    # If tau_bar is provided, use it for all nodes; otherwise compute from formula
    computed_tau_per_node = False
    if tau_list is not None:
        # Use provided tau_list (e.g., from D-APD)
        if len(tau_list) != N:
            raise ValueError(f"tau_list must have length N={N}, got {len(tau_list)}")
        tau_list_per_node = list(tau_list)
        tau_bar = min(tau_list_per_node)
        computed_tau_per_node = True
        if verbose_every:
            print(f"\n[D-APDB] Using provided tau_list (same as D-APD):")
            print(f"  tau_bar = {tau_bar:.6e} (min of tau_list)")
            print(f"  tau_i values: min={min(tau_list_per_node):.6e}, max={max(tau_list_per_node):.6e}, mean={np.mean(tau_list_per_node):.6e}")
            print(f"  Individual tau_i values:")
            for i in range(N):
                print(f"    Node {i}: tau_i = {tau_list_per_node[i]:.6e}")
    elif tau_bar is None:
        if Q_list is not None and q_list is not None:
            # For node-specific objectives: tau_i = tau_multiplier / L_{\varphi_i}
            # where L_{\varphi_i} = ||Q_i||_2 is the Lipschitz constant of φ_i(x) = (1/2) x^T Q_i x + q_i^T x
            tau_list_per_node = []
            L_phi_i_list = []
            for i in range(N):
                # Compute L_{\varphi_i} = ||Q_i||_2
                L_phi_i = np.linalg.norm(Q_list[i], ord=2)
                L_phi_i_list.append(L_phi_i)
                # tau_i = tau_multiplier / L_{\varphi_i}
                if L_phi_i > 1e-10:
                    tau_i = tau_multiplier / L_phi_i
                else:
                    tau_i = tau_multiplier / 1e-10  # Avoid division by zero
                tau_list_per_node.append(tau_i)
            # tau_bar = min(tau_i)
            tau_bar = min(tau_list_per_node)
            computed_tau_per_node = True
            if verbose_every:
                print(f"\n[D-APDB] Computed tau_bar from node-specific objectives: {tau_bar:.6e} = min(tau_i)")
                print(f"  tau_multiplier = {tau_multiplier}")
                print(f"  L_phi_i values: min={min(L_phi_i_list):.6e}, max={max(L_phi_i_list):.6e}, mean={np.mean(L_phi_i_list):.6e}")
                print(f"  tau_i values: min={min(tau_list_per_node):.6e}, max={max(tau_list_per_node):.6e}, mean={np.mean(tau_list_per_node):.6e}")
                print(f"  Individual tau_i values:")
                for i in range(N):
                    print(f"    Node {i}: L_phi_i = {L_phi_i_list[i]:.6e}, tau_i = {tau_list_per_node[i]:.6e}")
        else:
            # For aggregated objectives, use compute_initial_tau_per_node
            L_obj = compute_lipschitz_constant(A0, b0)
            A0_for_tau = A0
            b0_for_tau = b0
            tau_list_per_node, _ = compute_initial_tau_per_node(
                A0_for_tau, b0_for_tau, pernode_constraints, box_lo, box_hi,
                L_obj, N, c_alpha=c_alpha, c_beta=c_beta, c_c=c_varsigma,
                delta=delta, zeta=zeta
            )
            # Use the minimum tau multiplied by tau_multiplier as tau_bar
            tau_bar = min(tau_list_per_node) * tau_multiplier
            # Apply tau_multiplier to each node's tau_i
            tau_list_per_node = [tau_i * tau_multiplier for tau_i in tau_list_per_node]
            computed_tau_per_node = True
            if verbose_every:
                print(f"\n[D-APDB] Computed tau_bar from formula: {tau_bar:.6e} (min(tau_i) * {tau_multiplier})")
                print(f"  tau_i values (after multiplier): min={min(tau_list_per_node):.6e}, max={max(tau_list_per_node):.6e}, mean={np.mean(tau_list_per_node):.6e}")
                print(f"  tau_multiplier = {tau_multiplier}")
                print(f"  Individual tau_i values:")
                for i in range(N):
                    print(f"    Node {i}: tau_i = {tau_list_per_node[i]:.6e}")
    
    # Initialize step sizes according to pseudocode:
    # τ_i^{-1} = τ̄_i, σ_i^{-1} = ζ_i τ̄_i
    # τ_i^0 = τ̄_i, σ_i^0 = ζ_i τ̄_i
    # But if we computed per-node tau, use those values (already multiplied by tau_multiplier)
    zeta_i = [zeta] * N
    if computed_tau_per_node:
        # Use computed per-node tau values (already multiplied by tau_multiplier)
        tau_list = tau_list_per_node.copy()
        tau_prev = tau_list_per_node.copy()
    else:
        # Use provided tau_bar for all nodes
        tau_list = [tau_bar] * N
        tau_prev = [tau_bar] * N
        if verbose_every:
            print(f"\n[D-APDB] Using provided tau_bar for all nodes: tau_bar = {tau_bar:.6e}")
            print(f"  All {N} nodes use the same tau_i = {tau_bar:.6e}")
    sigma_prev = [zeta_i[i] * tau_list[i] for i in range(N)]
    
    # α_i^0 = c_α/τ_i^{-1}, β_i^0 = c_β/τ_i^{-1}, ς_i^0 = c_ς/τ_i^{-1}
    # Use per-node tau values if computed, otherwise use tau_bar
    if computed_tau_per_node:
        alpha_list = [c_alpha / tau_list[i] for i in range(N)]
        beta_list = [c_beta / tau_list[i] for i in range(N)]
        varsigma_list = [c_varsigma / tau_list[i] for i in range(N)]
    else:
        alpha_list = [c_alpha / tau_bar for _ in range(N)]
        beta_list = [c_beta / tau_bar for _ in range(N)]
        varsigma_list = [c_varsigma / tau_bar for _ in range(N)]
    
    # q_i^0 = J g_i(x_i^0)^T θ_i^0 + Σ_{j∈N_i}(s_i^0 - s_j^0)
    # q_i^{-1} = q_i^0
    q = []
    q_prev = []
    for i in range(N):
        q_i_0 = jacT_theta_i(i, x[i], theta[i])
        Ni = get_neighbors(i, neighbors_list)
        for j in Ni:
            q_i_0 += s[i] - s[j]  # s_i^0 - s_j^0, but s_i^0 = 0, so this is 0 - 0 = 0
        q.append(q_i_0)
        q_prev.append(q_i_0.copy())
    
    # τ_max^0 = max_{i∈N} {τ_i^0}
    tau_max_0 = max(tau_list)
    
    # Compute B_i for each node i using Slater point
    # If B_theta is provided, use it; otherwise compute for each node
    if B_theta is None:
        # Compute B_i for each node using its local constraints
        if verbose_every:
            print("Computing B_i for each node using Slater point...")
        B_i_list = []
        for i in range(N):
            A_list_i, b_list_i, c_list_i = pernode_constraints[i]
            B_i, x_slater_i = compute_dual_variable_bound_for_node_i(
                A0, b0, A_list_i, b_list_i, c_list_i, box_lo, box_hi
            )
            B_i_list.append(B_i)
            if verbose_every and i < 5:  # Print first 5 nodes
                print(f"  Node {i}: B_i = {B_i:.6f}")
    else:
        # Use provided B_theta
        if isinstance(B_theta, (list, np.ndarray)):
            B_i_list = list(B_theta)
        else:
            B_i_list = [B_theta] * N
    
    if verbose_every:
        if initialization_mode == "connected":
            print(f"Connected initialization mode:")
            print(f"  Number of connected components: {len(node_groups)}")
            for i, group in enumerate(node_groups):
                print(f"  Component {i}: nodes {group}")
        else:
            print(f"Independent initialization mode:")
            print(f"  All {N} nodes initialized independently")
        print(f"  tau_bar = {tau_bar:.6f}")
        print(f"  tau_max^0 = {tau_max_0:.6f}")
        print(f"  B_i values: min={min(B_i_list):.6f}, max={max(B_i_list):.6f}, mean={np.mean(B_i_list):.6f}")

    hist = []
    
    # Track gradient calls per node: in distributed setting, each node counts its own gradient calls
    # grad_calls_per_node[i] = total gradient calls (grad_f_i) made by node i so far
    # Note: We only count grad_f_i calls, not g_i_of calls (which compute constraint function values, not gradients)
    grad_calls_per_node = [0] * N
    
    # Track number of times eta^k > 1 (i.e., backtracking occurred)
    num_backtracking_iterations = 0

    # Record initial objective function value (before first iteration)
    x_bar_init = sum(x) / N
    # Compute constraint violations at each node's initial point
    # Check violations at node i's point x[i] for node i's constraints
    all_vals_init = []
    for i in range(N):
        all_vals_init.extend(g_i_of(i, x[i]))
    all_vals_init = np.array(all_vals_init)
    max_viol_init = float(np.maximum(all_vals_init, 0).max()) if all_vals_init.size > 0 else 0.0
    avg_viol_init = float(np.maximum(all_vals_init, 0).mean()) if all_vals_init.size > 0 else 0.0
    if Q_list is not None:
        if q_list is not None:
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) + q_i @ x_bar_init for Q_i, q_i in zip(Q_list, q_list))
            # Add constant terms if provided
            if constant_list is not None:
                obj_init += sum(constant_list)
        else:
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) for Q_i in Q_list)
    else:
        obj_init = 0.5 * x_bar_init @ (A0 @ x_bar_init) + b0 @ x_bar_init
    if lambda_l1 is not None and lambda_l1 > 0:
        # For centralized objective, use 1.0 for L1 coefficient
        obj_init += 1.0 * np.linalg.norm(x_bar_init, 1)
    subopt_init = abs(obj_init - phi_star) if phi_star is not None else np.nan
    cons_err_init = sum(np.linalg.norm(x[i] - x_bar_init) for i in range(N)) / N
    x_bar_norm_sq_init = np.dot(x_bar_init, x_bar_init)
    cons_err_sq_sum_init = sum(np.dot(x[i] - x_bar_init, x[i] - x_bar_init) for i in range(N))
    hist.append((obj_init, max_viol_init, cons_err_init, avg_viol_init, subopt_init, 0.0, 0, x_bar_norm_sq_init, cons_err_sq_sum_init))

    for k in range(max_iter):
        # η^k = 1 (reset at start of each iteration)
        eta_k = 1.0
        
        # Backtracking for each node
        tau_tilde = [None] * N
        sigma_tilde = [None] * N
        eta_i_k_list = [None] * N
        alpha_tilde_kp1_list = [None] * N
        beta_tilde_kp1_list = [None] * N
        varsigma_tilde_kp1_list = [None] * N
        p_tilde = [None] * N
        x_tilde_kp1 = [None] * N
        theta_tilde_kp1 = [None] * N
        grad_phi_i_cached = [None] * N  # Cache gradients computed in backtracking loop
        
        # Track total backtrack iterations across all nodes
        total_backtrack_iterations = 0
        
        for i in range(N):
            # τ̃_i^k = τ_i^{k-1}
            tau_tilde_i = tau_prev[i]
            
            # Track backtracking iterations for node i
            backtrack_iterations_i = 0
            
            # Compute gradient once before backtracking loop (x[i] doesn't change during backtracking)
            grad_phi_i_x_i = grad_phi_i(i, x[i])
            grad_calls_per_node[i] += 1  # Node i counts its own gradient call: grad_f_i
            
            # Backtracking loop
            while True:
                # σ̃_i^k = ζ_i τ̃_i^k, η_i^k = τ_i^{k-1}/τ̃_i^k
                sigma_tilde_i = zeta_i[i] * tau_tilde_i
                eta_i_k = tau_prev[i] / tau_tilde_i
                
                # α̃_i^{k+1} = c_α/τ̃_i^k, β̃_i^{k+1} = c_β/τ̃_i^k, ς̃_i^{k+1} = c_ς/τ̃_i^k
                alpha_tilde_kp1 = c_alpha / tau_tilde_i
                beta_tilde_kp1 = c_beta / tau_tilde_i
                varsigma_tilde_kp1 = c_varsigma / tau_tilde_i
                
                # p̃_i^k = q_i^k + η_i^k(q_i^k - q_i^{k-1})
                p_tilde_i = q[i] + eta_i_k * (q[i] - q_prev[i])
                
                # x̃_i^{k+1} = prox_{τ̃_i^k r_i}(x_i^k - τ̃_i^k(∇φ_i(x_i^k) + p̃_i^k))
                # where prox_{τ r_i}(v) = argmin_{w∈R^n} {τ r_i(w) + (1/2)||w - v||^2}
                # Note: grad_phi_i(x[i]) is computed once before the loop and reused
                x_tilde_kp1_i = prox_r_i(
                    x[i] - tau_tilde_i * (grad_phi_i_x_i + p_tilde_i),
                    tau_tilde_i
                )
                
                # θ̃_i^{k+1} = Π_{θ_i∈K_i^*∩B_i}(θ_i^k + σ̃_i^k g_i(x̃_i^{k+1}))
                # Note: g_i_of computes constraint function values, not gradients, so we don't count it
                theta_tilde_kp1_i = proj_dual(
                    theta[i] + sigma_tilde_i * g_i_of(i, x_tilde_kp1_i),
                    B_i_list[i]
                )
                
                # Check backtracking condition
                dx_tilde = x_tilde_kp1_i - x[i]
                dx_tilde_norm_sq = np.dot(dx_tilde, dx_tilde)
                
                # Check if node i has constraints
                has_constraints_i = len(pernode_constraints[i][0]) > 0
                
                if has_constraints_i:
                    # With constraints: use E_i^k condition
                    # E_i^k(x̃_i^{k+1}, θ̃_i^{k+1}) <= -δ/τ̃_i^k ||x̃_i^{k+1} - x_i^k||^2 
                    #                                  - δ/σ̃_i^k ||θ̃_i^{k+1} - θ_i^k||^2
                    E_val = E_i_k(i, x_tilde_kp1_i, theta_tilde_kp1_i, x[i], theta[i],
                                  tau_tilde_i, sigma_tilde_i,
                                  alpha_list[i], beta_list[i],
                                  alpha_tilde_kp1, beta_tilde_kp1, varsigma_tilde_kp1, eta_i_k,
                                  use_gradient_form=E_use_gradient_form)
                    
                    dtheta_tilde = theta_tilde_kp1_i - theta[i]
                    dtheta_tilde_norm_sq = np.dot(dtheta_tilde, dtheta_tilde)
                    
                    rhs = -(delta / tau_tilde_i) * dx_tilde_norm_sq - (delta / sigma_tilde_i) * dtheta_tilde_norm_sq
                    condition_satisfied = (E_val <= rhs)
                else:
                    # Without constraints: use simplified condition
                    # f_i(x̃_i^{k+1}) - f_i(x_i^k) - <∇f_i(x_i^k), x̃_i^{k+1} - x_i^k> 
                    #     <= (1/(2τ̃_i^k)) * (1 - δ - c_α - c_ς) * ||x̃_i^{k+1} - x_i^k||^2
                    f_tilde = phi_i(i, x_tilde_kp1_i)
                    f_k = phi_i(i, x[i])
                    # grad_phi_i_x_i is already computed before the backtracking loop
                    lhs = f_tilde - f_k - np.dot(grad_phi_i_x_i, dx_tilde)
                    rhs = (1.0 / (2.0 * tau_tilde_i)) * (1.0 - delta - c_alpha - c_varsigma) * dx_tilde_norm_sq
                    condition_satisfied = (lhs <= rhs)
                
                if condition_satisfied:
                    # Condition satisfied, break
                    tau_tilde[i] = tau_tilde_i
                    sigma_tilde[i] = sigma_tilde_i
                    eta_i_k_list[i] = eta_i_k
                    alpha_tilde_kp1_list[i] = alpha_tilde_kp1
                    beta_tilde_kp1_list[i] = beta_tilde_kp1
                    varsigma_tilde_kp1_list[i] = varsigma_tilde_kp1
                    p_tilde[i] = p_tilde_i
                    x_tilde_kp1[i] = x_tilde_kp1_i
                    theta_tilde_kp1[i] = theta_tilde_kp1_i
                    grad_phi_i_cached[i] = grad_phi_i_x_i  # Cache the gradient for reuse
                    break
                else:
                    # Condition not satisfied, shrink step size
                    backtrack_iterations_i += 1
                    tau_tilde_i = rho_shrink * tau_tilde_i
            
            # Accumulate backtrack iterations for node i
            total_backtrack_iterations += backtrack_iterations_i
        
        # η^k = max_{i∈N} η_i^k (max-consensus step)
        eta_k = max(eta_i_k_list)
        
        # Track if backtracking occurred (eta^k > 1)
        if eta_k > 1.0:
            num_backtracking_iterations += 1
        
        # γ^k = (Σ_{i∈N} τ_max^0 d_i)^{-1} (2/c_α + η^k/c_ς)^{-1}
        #     = 1 / ((Σ_{i∈N} τ_max^0 d_i) * (2/c_α + η^k/c_ς))
        sum_tau_max_0_d = sum(tau_max_0 * d_i[i] for i in range(N))
        denominator = sum_tau_max_0_d * ((2.0 / c_alpha) + (eta_k / c_varsigma))
        gamma_k = 1.0 / denominator
        
        # Update for each node
        for i in range(N):
            # Step 1: Update step sizes based on global eta^k
            # τ_i^k = τ_i^{k-1}/η^k, σ_i^k = ζ_i τ_i^k
            tau_list[i] = tau_prev[i] / eta_k
            sigma_i_k = zeta_i[i] * tau_list[i]
            
            # Step 2: Update consensus variable s_i^{k+1}
            # s_i^{k+1} = s_i^k + γ^k((1+η^k)x_i^k - η^k x_i^{k-1})
            s[i] = s[i] + gamma_k * ((1 + eta_k) * x[i] - eta_k * x_prev[i])
            
            # Step 3: Compute p_i^k using global eta^k
            # p_i^k = q_i^k + η^k(q_i^k - q_i^{k-1})
            p_i_k = q[i] + eta_k * (q[i] - q_prev[i])
            
            # Step 4: Update primal and dual variables
            # IMPORTANT: Save x_prev[i] BEFORE updating x[i]!
            x_prev_i_old = x[i].copy()  # Save old x[i] for x_prev update
            theta_prev_i_old = theta[i].copy()  # Save old theta[i] for theta_prev update
            
            if eta_k > 1.0:  # At least one node did backtracking (η^k > 1)
                # Recompute with updated step size τ_i^k = τ_i^{k-1}/η^k and p_i^k based on global η^k
                # x_i^{k+1} = prox_{τ_i^k r_i}(x_i^k - τ_i^k(∇φ_i(x_i^k) + p_i^k))
                # where prox_{τ r_i}(v) = argmin_{w∈R^n} {τ r_i(w) + (1/2)||w - v||^2}
                # Note: We reuse the cached gradient from backtracking loop (no additional gradient call needed)
                x[i] = prox_r_i(
                    x_prev_i_old - tau_list[i] * (grad_phi_i_cached[i] + p_i_k),
                    tau_list[i]
                )
                # θ_i^{k+1} = Π_{θ_i∈K_i^*∩B_i}(θ_i^k + σ_i^k g_i(x_i^{k+1}))
                # Note: g_i_of computes constraint function values, not gradients, so we don't count it
                theta[i] = proj_dual(
                    theta_prev_i_old + sigma_i_k * g_i_of(i, x[i]),
                    B_i_list[i]
                )
            else:  # No backtracking occurred (η^k = 1)
                # All nodes used their original step sizes, so we can use the values
                # computed in backtracking (which were computed with τ_i^{k-1} = τ_i^k)
                x[i] = x_tilde_kp1[i]
                theta[i] = theta_tilde_kp1[i]
            
            # q_i^{k+1} = J g_i(x_i^{k+1})^T θ_i^{k+1} + Σ_{j∈N_i}(s_i^{k+1} - s_j^{k+1})
            
            # α_i^{k+1} = c_α/τ_i^k, β_i^{k+1} = c_β/τ_i^k, ς_i^{k+1} = c_ς/τ_i^k
            alpha_list[i] = c_alpha / tau_list[i]
            beta_list[i] = c_beta / tau_list[i]
            varsigma_list[i] = c_varsigma / tau_list[i]
            
            # Update previous values for next iteration
            # x_prev[i] should be the OLD x[i] (before this iteration's update)
            x_prev[i] = x_prev_i_old
            theta_prev[i] = theta_prev_i_old
            s_prev[i] = s[i].copy()
            tau_prev[i] = tau_list[i]
            
        for i in range(N):
            q_prev[i] = q[i].copy()
            q[i] = jacT_theta_i(i, x[i], theta[i])
            Ni = get_neighbors(i, neighbors_list)
            for j in Ni:
                q[i] += s[i] - s[j]
        
        # Metrics
        x_bar = sum(x) / N
        
        if use_optimal_consensus_error and x_star is not None:
            x_star_norm = np.linalg.norm(x_star)
            if x_star_norm > 1e-10:
                cons_err = max(np.linalg.norm(x[i] - x_star) for i in range(N)) / x_star_norm
            else:
                cons_err = max(np.linalg.norm(x[i] - x_star) for i in range(N))
        else:
            cons_err = sum(np.linalg.norm(x[i] - x_bar) for i in range(N)) / N
            if normalize_consensus_error:
                x_bar_norm = np.linalg.norm(x_bar)
                cons_err = cons_err / max(x_bar_norm, 1e-10)
        
        # Compute constraint violations
        # For distributed setting, we should check violations at each node's point x[i],
        # not just at x_bar, because each node has its own constraints
        all_vals = []
        for i in range(N):
            # Check violations at node i's point x[i] for node i's constraints
            all_vals.extend(g_i_of(i, x[i]))
        all_vals = np.array(all_vals)
        
        max_viol = float(np.maximum(all_vals, 0).max()) if all_vals.size > 0 else 0.0
        avg_viol = float(np.maximum(all_vals, 0).mean()) if all_vals.size > 0 else 0.0
        # Objective: φ(x) + r(x)
        # If Q_list is provided, use node-specific objectives
        # If Q_list and q_list are both provided, use node-specific objectives with linear terms
        # Otherwise, use aggregated objective
        if Q_list is not None:
            if q_list is not None:
                # Node-specific with linear terms: φ(x) = sum_i [(1/2) x^T Q_i x + q_i^T x + c_i]
                obj = sum(0.5 * x_bar @ (Q_i @ x_bar) + q_i @ x_bar for Q_i, q_i in zip(Q_list, q_list))
                # Add constant terms if provided
                if constant_list is not None:
                    obj += sum(constant_list)
            else:
                # Node-specific without linear terms: φ(x) = sum_i [(1/2) x^T Q_i x]
                obj = sum(0.5 * x_bar @ (Q_i @ x_bar) for Q_i in Q_list)
        else:
            # Aggregated: φ(x) = 0.5 * x^T A0 x + b0^T x (no constant term)
            obj = 0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar
        if lambda_l1 is not None and lambda_l1 > 0:
            # For L1 regularized problems:
            # - Per-node objective: (1/N) * ||x||_1 + (1/2) * x^T Q^i x
            # - Centralized objective: ||x||_1 + (1/2) * sum_i x^T Q^i x
            # So for centralized objective, L1 coefficient should be 1.0, not lambda_l1
            # lambda_l1 = 1/N is the per-node coefficient
            obj += 1.0 * np.linalg.norm(x_bar, 1)  # Use 1.0 for centralized objective
        subopt = abs(obj - phi_star) if phi_star is not None else np.nan

        # In distributed setting, use average gradient calls across all nodes
        # This represents the average computational cost per node
        avg_grad_calls = sum(grad_calls_per_node) / N
        
        # Compute x_bar norm for relative consensus error calculation
        x_bar_norm_sq = np.dot(x_bar, x_bar)
        
        # Compute sum of squared consensus errors: sum_i ||x_i - x_bar||^2
        cons_err_sq_sum = sum(np.dot(x[i] - x_bar, x[i] - x_bar) for i in range(N))
        
        # Store history with average gradient call count and total backtrack iterations
        # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum)
        hist.append((obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum))

        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | eta {eta_k:.3f}"
            if phi_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)

        if phi_star is not None:
            if max(subopt, avg_viol) <= tol:
                break

    # Return x_bar, hist, and statistics
    # Statistics: (num_backtracking_iterations, total_iterations)
    stats = {
        'num_backtracking_iterations': num_backtracking_iterations,
        'total_iterations': len(hist) - 1,  # Subtract 1 for initial point
        'backtracking_ratio': num_backtracking_iterations / max(len(hist) - 1, 1)
    }
    
    return x_bar, hist, stats
