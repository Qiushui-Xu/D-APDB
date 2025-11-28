import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Import shared utility functions
from utils import (
    compute_lipschitz_constant,
    generate_small_world_network,
    generate_fully_connected_network,
    get_neighbors,
)

# --------------------------
# Distributed APDB for Unconstrained Optimization (D-APDB-O)
# --------------------------
def d_apdb_unconstrained(
    N=3, n=10, max_iter=2000, seed=0,
    c_alpha=0.1, c_varsigma=0.1, c_gamma=None,
    rho_shrink=0.5, delta=0.1,
    verbose_every=200, initial_scale=5.0,
    phi_star=None, tol=1e-8, normalize_consensus_error=False,
    use_optimal_consensus_error=False, x_star=None,
    neighbors_list=None, initialization_mode="connected",
    lambda_l1=None, initial_points=None,
    Q_list=None, q_list=None, tau_list=None,
    constant_list=None
):
    """
    Distributed APD with Backtracking for Unconstrained Optimization (D-APDB-O)
    
    This is a simplified version of D-APDB for problems without local constraints.
    
    Problem formulation:
        min_{x} sum_{i=1}^N [f_i(x) + phi_i(x)]
    where:
        - f_i(x) = (1/2) x^T Q_i x + q_i^T x (smooth, convex)
        - phi_i(x) = (lambda_l1/N) * ||x||_1 (non-smooth, convex) or 0 if lambda_l1 is None
    
    Parameters:
    -----------
    N : int
        Number of nodes
    n : int
        Dimension of decision variable
    max_iter : int
        Maximum number of iterations
    seed : int
        Random seed
    c_alpha, c_varsigma : float
        Constants > 0 with c_alpha + c_varsigma < 1 - delta
    c_gamma : float or None
        Constant for gamma computation. If None, uses 1/(2*|E|)
    rho_shrink : float
        Contraction coefficient in (0,1) for backtracking
    delta : float
        Parameter for backtracking condition
    verbose_every : int
        Print progress every this many iterations
    initial_scale : float
        Scale for initial point generation
    phi_star : float or None
        Optimal objective value (for computing suboptimality)
    tol : float
        Tolerance for convergence
    normalize_consensus_error : bool
        Whether to normalize consensus error by x_bar norm
    use_optimal_consensus_error : bool
        Whether to use x_star for consensus error computation
    x_star : np.ndarray or None
        Optimal solution (for consensus error computation)
    neighbors_list : list of lists
        Network topology
    initialization_mode : str
        "connected" or "independent"
    lambda_l1 : float or None
        L1 regularization coefficient (per-node: lambda_l1 = 1/N)
    initial_points : list of np.ndarray or None
        Initial points for each node
    Q_list : list of np.ndarray
        Node-specific quadratic matrices Q_i
    q_list : list of np.ndarray
        Node-specific linear terms q_i
    tau_list : list of float or None
        Initial step sizes for each node
    constant_list : list of float or None
        Constant terms for each node's objective
    
    Returns:
    --------
    x_bar : np.ndarray
        Average of node solutions
    hist : list of tuples
        History of (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, 
                    total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum)
    stats : dict
        Statistics including backtracking counts
    """
    rng = np.random.default_rng(seed)
    
    if Q_list is None or q_list is None:
        raise ValueError("Q_list and q_list must be provided for unconstrained optimization")
    
    if len(Q_list) != N or len(q_list) != N:
        raise ValueError(f"Q_list and q_list must have length N={N}")
    
    n = Q_list[0].shape[0]  # Infer dimension from Q_list
    
    # Use provided network topology or generate fully connected network
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    
    # Compute number of edges |E|
    num_edges = sum(len(neighbors_list[i]) for i in range(N)) // 2
    
    # Set c_gamma if not provided: c_gamma <= 1/(2|E|)
    if c_gamma is None:
        c_gamma = 1.0 / (2.0 * num_edges)
    
    # Validate parameters: c_alpha + c_varsigma < 1 - delta
    if c_alpha + c_varsigma >= 1.0 - delta:
        print(f"Warning: c_alpha + c_varsigma = {c_alpha + c_varsigma:.4f} >= 1 - delta = {1.0 - delta:.4f}")
        print("  This may cause convergence issues. Adjusting parameters...")
    
    # Local smooth objective: f_i(x) = (1/2) x^T Q_i x + q_i^T x
    def f_i(i, x):
        return 0.5 * x @ (Q_list[i] @ x) + q_list[i] @ x
    
    def grad_f_i(i, x):
        return Q_list[i] @ x + q_list[i]
    
    # Proximal operator for non-smooth term phi_i:
    # prox_{τ phi_i}(v) = argmin_{w} {τ phi_i(w) + (1/2)||w - v||^2}
    # If lambda_l1 > 0: phi_i(x) = (lambda_l1) * ||x||_1, prox is soft-thresholding
    # If lambda_l1 is None: phi_i(x) = 0, prox is identity
    def prox_phi_i(v, tau):
        if lambda_l1 is not None and lambda_l1 > 0:
            # Soft-thresholding for lambda_l1 * ||·||_1
            # prox_{τ * lambda_l1 * ||·||_1}(v) = sign(v) * max(|v| - τ*lambda_l1, 0)
            threshold = tau * lambda_l1
            return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
        else:
            return v.copy()
    
    def _sample_initial_vec():
        return initial_scale * rng.standard_normal(n)
    
    # Initialize states
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
    else:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}")
    
    # Initialize according to pseudocode:
    # x_i^{-1} = x_i^0
    x_prev = [xi.copy() for xi in x]
    
    # s_i^0 = 0
    s = [np.zeros(n) for _ in range(N)]
    
    # Initialize step sizes: τ_i^{-1} = τ̄_i, τ_i^0 = τ̄_i
    if tau_list is not None:
        if len(tau_list) != N:
            raise ValueError(f"tau_list must have length N={N}, got {len(tau_list)}")
        tau_bar_list = list(tau_list)
    else:
        # Compute tau_bar_i = 1 / L_{f_i} where L_{f_i} = ||Q_i||_2
        tau_bar_list = []
        for i in range(N):
            L_f_i = np.linalg.norm(Q_list[i], ord=2)
            if L_f_i < 1e-12:
                L_f_i = 1e-12
            tau_bar_list.append(1.0 / L_f_i)
    
    tau_prev = tau_bar_list.copy()
    tau_list_current = tau_bar_list.copy()
    
    # τ̄ = max_{i∈N} {τ̄_i}
    tau_bar_max = max(tau_bar_list)
    
    # q_i^0 = sum_{j∈N_i}(s_i^0 - s_j^0) = 0 (since s_i^0 = 0 for all i)
    # q_i^{-1} = q_i^0
    q = [np.zeros(n) for _ in range(N)]
    q_prev = [np.zeros(n) for _ in range(N)]
    
    if verbose_every:
        if initialization_mode == "connected":
            print(f"Connected initialization mode:")
            print(f"  Number of connected components: {len(node_groups)}")
        else:
            print(f"Independent initialization mode:")
            print(f"  All {N} nodes initialized independently")
        print(f"  tau_bar_max = {tau_bar_max:.6e}")
        print(f"  c_gamma = {c_gamma:.6e}")
        print(f"  c_alpha = {c_alpha}, c_varsigma = {c_varsigma}, delta = {delta}")
        print(f"  c_alpha + c_varsigma = {c_alpha + c_varsigma:.4f}, 1 - delta = {1.0 - delta:.4f}")
        print(f"  tau_bar_i values: min={min(tau_bar_list):.6e}, max={max(tau_bar_list):.6e}, mean={np.mean(tau_bar_list):.6e}")
    
    hist = []
    
    # Track gradient calls per node
    grad_calls_per_node = [0] * N
    
    # Track number of times eta^k > 1 (i.e., backtracking occurred)
    num_backtracking_iterations = 0
    
    # Track cumulative backtracking iterations per node
    cumulative_backtrack_counts = [0] * N
    
    # Record initial objective function value (before first iteration)
    x_bar_init = sum(x) / N
    obj_init = sum(f_i(i, x_bar_init) for i in range(N))
    if constant_list is not None:
        obj_init += sum(constant_list)
    if lambda_l1 is not None and lambda_l1 > 0:
        # Centralized L1 coefficient is 1.0
        obj_init += 1.0 * np.linalg.norm(x_bar_init, 1)
    subopt_init = abs(obj_init - phi_star) if phi_star is not None else np.nan
    cons_err_init = sum(np.linalg.norm(x[i] - x_bar_init) for i in range(N)) / N
    x_bar_norm_sq_init = np.dot(x_bar_init, x_bar_init)
    cons_err_sq_sum_init = sum(np.dot(x[i] - x_bar_init, x[i] - x_bar_init) for i in range(N))
    # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
    avg_tau_init = np.mean(tau_bar_list)
    hist.append((obj_init, 0.0, cons_err_init, 0.0, subopt_init, 0.0, 0, x_bar_norm_sq_init, cons_err_sq_sum_init, avg_tau_init))
    
    for k in range(max_iter):
        # η^k = 1 (reset at start of each iteration)
        eta_k = 1.0
        
        # Backtracking results for each node
        tau_tilde = [None] * N
        eta_i_k_list = [None] * N
        x_tilde_kp1 = [None] * N
        grad_f_i_cached = [None] * N  # Cache gradients
        
        # Track total backtrack iterations across all nodes
        total_backtrack_iterations = 0
        
        for i in range(N):
            # τ̃_i^k = τ_i^{k-1}
            tau_tilde_i = tau_prev[i] / rho_shrink
            
            # Track backtracking iterations for node i
            backtrack_iterations_i = 0
            
            # Compute gradient once before backtracking loop
            grad_f_i_x_i = grad_f_i(i, x[i])
            grad_calls_per_node[i] += 1
            
            # Backtracking loop
            while True:
                # η_i^k = τ_i^{k-1} / τ̃_i^k
                eta_i_k = tau_prev[i] / tau_tilde_i
                
                # p̃_i^k = q_i^k + η_i^k(q_i^k - q_i^{k-1})
                p_tilde_i = q[i] + eta_i_k * (q[i] - q_prev[i])
                
                # x̃_i^{k+1} = prox_{τ̃_i^k φ_i}(x_i^k - τ̃_i^k(∇f_i(x_i^k) + p̃_i^k))
                x_tilde_kp1_i = prox_phi_i(
                    x[i] - tau_tilde_i * (grad_f_i_x_i + p_tilde_i),
                    tau_tilde_i
                )
                
                # Check backtracking condition:
                # f_i(x̃_i^{k+1}) - f_i(x_i^k) - <∇f_i(x_i^k), x̃_i^{k+1} - x_i^k>
                #     <= (1/(2τ̃_i^k)) * (1 - δ - c_α - c_ς) * ||x̃_i^{k+1} - x_i^k||^2
                dx_tilde = x_tilde_kp1_i - x[i]
                dx_tilde_norm_sq = np.dot(dx_tilde, dx_tilde)
                
                f_tilde = f_i(i, x_tilde_kp1_i)
                f_k = f_i(i, x[i])
                
                lhs = f_tilde - f_k - np.dot(grad_f_i_x_i, dx_tilde)
                rhs = (1.0 / (2.0 * tau_tilde_i)) * (1.0 - delta - c_alpha - c_varsigma) * dx_tilde_norm_sq
                
                if 0.05 *lhs <= rhs:
                    # Condition satisfied, break
                    tau_tilde[i] = tau_tilde_i
                    eta_i_k_list[i] = eta_i_k
                    x_tilde_kp1[i] = x_tilde_kp1_i
                    grad_f_i_cached[i] = grad_f_i_x_i
                    break
                else:
                    # Condition not satisfied, shrink step size
                    backtrack_iterations_i += 1
                    tau_tilde_i = rho_shrink * tau_tilde_i
            
            total_backtrack_iterations += backtrack_iterations_i
            cumulative_backtrack_counts[i] += backtrack_iterations_i
        
        # η^k = max_{i∈N} η_i^k (max-consensus step)
        eta_k = max(eta_i_k_list)
        
        # Track if backtracking occurred (eta^k > 1)
        if eta_k > 1.0:
            num_backtracking_iterations += 1
        
        # γ^k = (c_γ / τ̄) * (2/c_α + η^k/c_ς)^{-1}
        gamma_k = (c_gamma / tau_bar_max) / ((2.0 / c_alpha) + (eta_k / c_varsigma))
        
        # Phase 1: Update s, x, tau_prev, x_prev for each node
        x_prev_old_list = [None] * N  # Store old x values for x_prev update
        for i in range(N):
            # τ_i^k = τ_i^{k-1} / η^k
            tau_list_current[i] = tau_prev[i] / eta_k
            
            # Save old x[i] before updating
            x_prev_old_list[i] = x[i].copy()
            
            # s_i^{k+1} = s_i^k + γ^k((1 + η^k)x_i^k - η^k x_i^{k-1})
            s[i] = s[i] + gamma_k * ((1 + eta_k) * x[i] - eta_k * x_prev[i])
            
            # p_i^k = q_i^k + η^k(q_i^k - q_i^{k-1})
            p_i_k = q[i] + eta_k * (q[i] - q_prev[i])
            
            if eta_k > 1.0:
                # At least one node did backtracking
                # x_i^{k+1} = prox_{τ_i^k φ_i}(x_i^k - τ_i^k(∇f_i(x_i^k) + p_i^k))
                x[i] = prox_phi_i(
                    x_prev_old_list[i] - tau_list_current[i] * (grad_f_i_cached[i] + p_i_k),
                    tau_list_current[i]
                )
            else:
                # No backtracking occurred
                x[i] = x_tilde_kp1[i]
            
            # Update previous values for next iteration
            x_prev[i] = x_prev_old_list[i]
            tau_prev[i] = tau_list_current[i]
        
        # Phase 2: Update q for each node (after all s have been updated)
        # q_i^{k+1} = sum_{j∈N_i}(s_i^{k+1} - s_j^{k+1})
        for i in range(N):
            q_prev[i] = q[i].copy()
            q[i] = np.zeros(n)
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
        
        # Objective: sum_i [f_i(x_bar) + c_i] + L1_term
        obj = sum(f_i(i, x_bar) for i in range(N))
        if constant_list is not None:
            obj += sum(constant_list)
        if lambda_l1 is not None and lambda_l1 > 0:
            # Centralized L1 coefficient is 1.0
            obj += 1.0 * np.linalg.norm(x_bar, 1)
        subopt = abs(obj - phi_star) if phi_star is not None else np.nan
        
        # Average gradient calls across all nodes
        avg_grad_calls = sum(grad_calls_per_node) / N
        
        # Compute x_bar norm for relative consensus error calculation
        x_bar_norm_sq = np.dot(x_bar, x_bar)
        
        # Compute sum of squared consensus errors
        cons_err_sq_sum = sum(np.dot(x[i] - x_bar, x[i] - x_bar) for i in range(N))
        
        # Compute average tau across all nodes
        avg_tau = np.mean(tau_list_current)
        
        # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iters, x_bar_norm_sq, cons_err_sq_sum, avg_tau)
        hist.append((obj, 0.0, cons_err, 0.0, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum, avg_tau))
        
        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | cons {cons_err:.2e} | eta {eta_k:.3f}"
            if phi_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)
        
        if phi_star is not None:
            if subopt <= tol:
                break
    
    # Return x_bar, hist, and statistics
    stats = {
        'num_backtracking_iterations': num_backtracking_iterations,
        'total_iterations': len(hist) - 1,
        'backtracking_ratio': num_backtracking_iterations / max(len(hist) - 1, 1),
        'backtrack_counts_per_node': cumulative_backtrack_counts
    }
    
    return x_bar, hist, stats

