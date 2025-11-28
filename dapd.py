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
def d_apd_qcqp_merely_convex(A0, b0, c0, pernode_constraints,
                              box_lo, box_hi,
                              N=3, max_iter=2000, seed=0,
                              c_alpha=0.1, c_beta=0.1, c_varsigma=0.1,
                              zeta=1.0, tau=0.08, gamma=None,
                              verbose_every=200, initial_scale=1.0,
                              phi_star=None, tol=1e-8, normalize_consensus_error=False,
                              use_optimal_consensus_error=False, x_star=None,
                              neighbors_list=None, initialization_mode="connected",
                              B_theta=None, lambda_l1=None, initial_points=None,
                              Q_list=None, q_list=None, tau_list=None,
                              constant_list=None):
    """
    Distributed APD (D-APD) for convex QCQP with box constraints.
    
    Parameters:
    -----------
    pernode_constraints: list of length N
      pernode_constraints[i] = (A_list_i, b_list_i, c_list_i)
      with Aj PSD (merely convex). Box handled via prox (clip).
    gamma : float, optional (deprecated)
        This parameter is ignored. gamma_k is computed each iteration using:
        gamma_k = 1.0 / ((Σ_{i∈N} τ_max^0 d_i) * (2/c_α + η^k/c_ς))
        where η^k = 1.0 (fixed, no backtracking in D-APD)
    
    Returns:
    --------
    x_bar : np.ndarray
        Average solution across all nodes
    hist : list
        Convergence history
    """
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    
    # Use provided network topology or generate fully connected network as default
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    d_max = max(len(neighbors_list[i]) for i in range(N))  # maximum degree

    # Compute node degrees d_i for each node
    d_i = [len(neighbors_list[i]) for i in range(N)]

    # Local smooth objective: φ_i(x) = (1/2) x^T Q_i x + q_i^T x
    # If Q_list and q_list are provided, use node-specific objectives
    # Otherwise, use aggregated objective: φ_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
    # Note: The full objective is φ_i(x) + r_i(x), where r_i is the non-smooth term.
    # In QCQP, r_i(x) = I_{[box_lo, box_hi]^n}(x) (indicator function of box constraint).
    # If r_i == 0 (no non-smooth term), then prox_{τ r_i}(x) = x.
    if Q_list is not None and q_list is not None:
        # Node-specific objectives: φ_i(x) = (1/2) x^T Q_i x + q_i^T x
        def phi_i(i, x):
            return 0.5 * x @ (Q_list[i] @ x) + q_list[i] @ x
        
        def grad_phi_i(i, x):
            return Q_list[i] @ x + q_list[i]
    else:
        # Aggregated objective: φ_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
        def phi_i(i, x):
            return (0.5 * x @ (A0 @ x) + b0 @ x) / N
        
        def grad_phi_i(i, x):
            return (A0 @ x + b0) / N

    # Local convex constraints for node i:
    # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
    def g_i_of(i, x):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        return np.array([0.5 * x @ (Aik @ x) + bik @ x - cik
                         for Aik, bik, cik in zip(A_list_i, b_list_i, c_list_i)])

    # Jacobian^T * theta for node i:
    # ∇ g_{i,k}(x) = A_{i,k} x + b_{i,k}
    def jacT_theta_i(i, x, theta):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        v = np.zeros_like(x)
        for th, Aik, bik, _ in zip(theta, A_list_i, b_list_i, c_list_i):
            v += th * (Aik @ x + bik)
        return v

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
            return rng.uniform(box_lo, box_hi, size=n) * initial_scale
        return initial_scale * rng.standard_normal(n)

    # Init states - initialize based on mode
    if initial_points is not None:
        if len(initial_points) != N:
            raise ValueError("initial_points must have length N")
        x = [initial_points[i].copy() for i in range(N)]
    elif initialization_mode == "connected":
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
            x_init = _sample_initial_vec()
            for node in group:
                x[node] = x_init.copy()
    else:
        # Independent mode: each node has independent initial values
        x = [_sample_initial_vec() for _ in range(N)]
    
    # Initialize dual variables
    theta = []
    for i in range(N):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        m_i = len(A_list_i)
        theta.append(np.zeros(m_i))
    
    # Initialize previous values (for momentum terms)
    # (x_i^{-1}, θ_i^{-1}) = (x_i^0, θ_i^0)
    x_prev = [xi.copy() for xi in x]
    theta_prev = [theta[i].copy() for i in range(N)]
    
    # Initialize auxiliary variables
    s = [np.zeros(n) for _ in range(N)]
    s_prev = [np.zeros(n) for _ in range(N)]
    
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
        
    # Compute initial tau per node if not provided
    # If tau_list is provided, use it directly (for using same tau as D-APDB or pre-computed)
    if tau_list is not None:
        # Use provided tau_list
        if len(tau_list) != N:
            raise ValueError(f"tau_list must have length N={N}, got {len(tau_list)}")
        if verbose_every:
            print(f"\n[D-APD] Using provided tau_list:")
            print(f"  tau_i values: min={min(tau_list):.6e}, max={max(tau_list):.6e}, mean={np.mean(tau_list):.6e}")
            print(f"  Individual tau_i values:")
            for i in range(N):
                print(f"    Node {i}: tau_i = {tau_list[i]:.6e}")
    elif tau is None:
        # Compute Lipschitz constant for objective
        # Compute Lipschitz constant using aggregated or node-specific objectives
        if Q_list is not None and q_list is not None:
            # For node-specific objectives, compute L_obj from aggregated objective
            # Total objective: sum_i [(1/2) x^T Q_i x + q_i^T x] = (1/2) x^T (sum Q_i) x + (sum q_i)^T x
            A0_agg = np.sum(Q_list, axis=0)  # sum Q_i
            b0_agg = np.sum(q_list, axis=0)  # sum q_i
            L_obj = compute_lipschitz_constant(A0_agg, b0_agg)
        else:
            L_obj = compute_lipschitz_constant(A0, b0)
        
        # Compute initial tau for each node using compute_initial_tau_per_node
        # Use aggregated A0 and b0 if node-specific objectives are provided
        L_f_i_list = None
        if Q_list is not None and q_list is not None:
            # Use aggregated objective for compute_initial_tau_per_node (for other parameters)
            A0_for_tau = np.sum(Q_list, axis=0)  # sum Q_i
            b0_for_tau = np.sum(q_list, axis=0)  # sum q_i
            # Compute node-specific L_f_i = ||Q_i||_2
            L_f_i_list = [np.linalg.norm(Q_i, ord=2) for Q_i in Q_list]
        else:
            A0_for_tau = A0
            b0_for_tau = b0
            # L_f_i_list remains None, will default to L_obj/N
        
        tau_list, tau_components_list = compute_initial_tau_per_node(
            A0_for_tau, b0_for_tau, pernode_constraints, box_lo, box_hi,
            L_obj, N,
            c_alpha=c_alpha, c_beta=c_beta, c_c=c_varsigma,
            delta=0.1, zeta=zeta,
            L_f_i_list=L_f_i_list
        )
        
        if verbose_every:
            print(f"\n[D-APD] Computed initial tau per node:")
            print(f"  tau_i values: min={min(tau_list):.6e}, max={max(tau_list):.6e}, mean={np.mean(tau_list):.6e}")
            print(f"  Individual tau_i values:")
            for i in range(N):
                print(f"    Node {i}: tau_i = {tau_list[i]:.6e}")
    else:
        # If tau is provided, use it for all nodes
        tau_list = [tau] * N
        if verbose_every:
            print(f"\n[D-APD] Using provided tau for all nodes: tau = {tau:.6e}")
            print(f"  All {N} nodes use the same tau_i = {tau:.6e}")
    
    # If tau_list was provided, it's already set above
    
    # Initialize step sizes per node (each node has its own tau_i)
    # tau_i is now a list: tau_list[i] is the step size for node i
    sigma_list = [zeta * tau_list[i] for i in range(N)]
    
    # Initialize algorithm parameters per node
    alpha_list = [c_alpha / tau_list[i] for i in range(N)]
    beta_list = [c_beta / tau_list[i] for i in range(N)]
    
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
    
    # History
    hist = []
    
    # Gradient call tracking
    grad_calls_per_node = [0] * N
    
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
    # Objective: φ(x) + r(x)
    # If Q_list is provided, use node-specific objectives
    # If Q_list and q_list are both provided, use node-specific objectives with linear terms
    # Otherwise, use aggregated objective
    if Q_list is not None:
        if q_list is not None:
            # Node-specific with linear terms: φ(x) = sum_i [(1/2) x^T Q_i x + q_i^T x + c_i]
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) + q_i @ x_bar_init for Q_i, q_i in zip(Q_list, q_list))
            # Add constant terms if provided
            if constant_list is not None:
                obj_init += sum(constant_list)
        else:
            # Node-specific without linear terms: φ(x) = sum_i [(1/2) x^T Q_i x]
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) for Q_i in Q_list)
    else:
        # Aggregated: φ(x) = 0.5 * x^T A0 x + b0^T x (no constant term)
        obj_init = 0.5 * x_bar_init @ (A0 @ x_bar_init) + b0 @ x_bar_init
    if lambda_l1 is not None and lambda_l1 > 0:
        # For L1 regularized problems:
        # - Per-node objective: (1/N) * ||x||_1 + (1/2) * x^T Q^i x
        # - Centralized objective: ||x||_1 + (1/2) * sum_i x^T Q^i x
        # So for centralized objective, L1 coefficient should be 1.0, not lambda_l1
        obj_init += 1.0 * np.linalg.norm(x_bar_init, 1)  # Use 1.0 for centralized objective
    subopt_init = abs(obj_init - phi_star) if phi_star is not None else np.nan
    cons_err_init = sum(np.linalg.norm(x[i] - x_bar_init) for i in range(N)) / N
    x_bar_norm_sq_init = np.dot(x_bar_init, x_bar_init)
    cons_err_sq_sum_init = sum(np.dot(x[i] - x_bar_init, x[i] - x_bar_init) for i in range(N))
    avg_tau_init = np.mean(tau_list)
    hist.append((obj_init, max_viol_init, cons_err_init, avg_viol_init, subopt_init, 0.0, x_bar_norm_sq_init, cons_err_sq_sum_init, avg_tau_init))
    
    # Main loop
    for k in range(max_iter):
        # η^k = 1 (fixed, since D-APD has no backtracking)
        eta_k = 1.0
        
        # γ^k = (Σ_{i∈N} τ_max^0 d_i)^{-1} (2/c_α + η^k/c_ς)^{-1}
        #     = 1 / ((Σ_{i∈N} τ_max^0 d_i) * (2/c_α + η^k/c_ς))
        sum_tau_max_0_d = sum(tau_max_0 * d_i[i] for i in range(N))
        denominator = sum_tau_max_0_d * ((2.0 / c_alpha) + (eta_k / c_varsigma))
        gamma_k = 1.0 / denominator
        
        # Update for each node
        for i in range(N):
            # Step 1: Update consensus variable s_i^{k+1}
            # s_i^{k+1} = s_i^k + γ^k((1+η^k)x_i^k - η^k x_i^{k-1})
            s[i] = s[i] + gamma_k * ((1 + eta_k) * x[i] - eta_k * x_prev[i])
            
            # Step 2: Compute p_i^k using global eta^k
            # p_i^k = q_i^k + η^k(q_i^k - q_i^{k-1})
            p_i_k = q[i] + eta_k * (q[i] - q_prev[i])
            
            # Step 3: Update primal and dual variables
            # x_i^{k+1} = prox_{τ_i^k r_i}(x_i^k - τ_i^k(∇f_i(x_i^k) + p_i^k))
            # where prox_{τ r_i}(v) = argmin_{w∈R^n} {τ r_i(w) + (1/2)||w - v||^2}
            # Use per-node step size tau_list[i]
            grad_calls_per_node[i] += 1  # Node i counts its own gradient call: grad_phi_i
            x_next = prox_r_i(x[i] - tau_list[i] * (grad_phi_i(i, x[i]) + p_i_k), tau_list[i])
            # Note: g_i_of computes constraint function values, not gradients, so we don't count it
            theta_next = proj_dual(theta[i] + sigma_list[i] * g_i_of(i, x_next), B_i_list[i])
            
            # Step 4: Update q_i^{k+1}
            # q_i^{k+1} = J g_i(x_i^{k+1})^T θ_i^{k+1} + Σ_{j∈N_i}(s_i^{k+1} - s_j^{k+1})
            q_prev[i] = q[i].copy()
            
            # Update states and previous values
            x_prev[i] = x[i].copy()
            theta_prev[i] = theta[i].copy()
            s_prev[i] = s[i].copy()
            x[i] = x_next
            theta[i] = theta_next
        # Phase 2: Update q for each node (after all s have been updated)
        for i in range(N):
            q[i] = jacT_theta_i(i, x[i], theta[i])  # Use updated x[i], theta[i]
            Ni = get_neighbors(i, neighbors_list)
            for j in Ni:
                q[i] += s[i] - s[j]
        
        # Metrics (compute every iteration for convergence check)
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
        
        # Compute average tau across all nodes
        avg_tau = np.mean(tau_list)
        
        # Store history with average gradient call count
        # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
        hist.append((obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, x_bar_norm_sq, cons_err_sq_sum, avg_tau))
            
        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | eta {eta_k:.3f}"
            if phi_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)
            
            # Check convergence
        if phi_star is not None:
            if max(subopt, avg_viol) <= tol:
                break
    
    # Return average solution x_bar and history
    x_bar = sum(x) / N
    return x_bar, hist

