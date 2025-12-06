import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Import shared utility functions
from utils import (
    generate_fully_connected_network,
    generate_small_world_network,
    build_gossip_matrix,
    compute_lipschitz_constant
)

# --------------------------
# Linesearch function
# --------------------------

def linesearch_with_grad(alpha, f, grad_f_x1, x1, x2, d, delta):
    """
    Linesearch algorithm for backtracking step size (with gradient at x1 provided)
    
    Parameters:
    -----------
    alpha : float
        Initial step size
    f : callable
        Function f(x) to evaluate
    grad_f_x1 : np.ndarray
        Gradient at x1 (already computed)
    x1 : np.ndarray
        Point x_1
    x2 : np.ndarray
        Point x_2
    d : np.ndarray
        Direction vector d
    delta : float
        Backtracking parameter in (0, 1]
        
    Returns:
    --------
    alpha_plus : float
        Final step size after backtracking
    num_backtrack : int
        Number of backtracking iterations (t - 1, where t is the number of evaluations)
    """
    alpha_plus = alpha
    x_plus = x2 + alpha_plus * d
    t = 1
    
    # Compute f(x1) once (grad_f_x1 is already provided)
    f_x1 = f(x1)
    
    while True:
        f_x_plus = f(x_plus)
        
        # Check backtracking condition:
        # f(x_plus) > f(x1) + <grad_f(x1), x_plus - x1> + (delta/(2*alpha_plus)) * ||x_plus - x1||^2
        dx = x_plus - x1
        dx_norm_sq = np.dot(dx, dx)
        
        rhs = f_x1 + np.dot(grad_f_x1, dx) + (delta / (2 * alpha_plus)) * dx_norm_sq
        
        if f_x_plus <= rhs:
            # Condition satisfied
            break
        else:
            # Shrink step size
            alpha_plus = alpha_plus * 0.9
            x_plus = x2 + alpha_plus * d
            t = t + 1
    
    # Number of backtracking iterations = t - 1 (subtract 1 because first evaluation doesn't count as backtrack)
    num_backtrack = max(0, t - 1)
    return alpha_plus, num_backtrack

# --------------------------
# Decentralized Adaptive Three Operator Splitting (global_DATOS)
# --------------------------

def aldo_qcqp_merely_convex(A0, b0, c0,
                             box_lo, box_hi,
                             N=3, max_iter=2000, seed=0,
                             alpha_init=None, alpha_init_constant=1.0, delta=0.9, c=0.33,
                             verbose_every=200, initial_scale=1.0,
                             phi_star=None, tol=1e-8, normalize_consensus_error=False,
                             use_optimal_consensus_error=False, x_star=None,
                             neighbors_list=None, initialization_mode="connected",
                             lambda_l1=None, initial_points=None,
                             Q_list=None, q_list=None, constant_list=None):
    """
    Decentralized Adaptive Three Operator Splitting (global_DATOS) for QCQP
    
    Parameters:
    -----------
    A0 : np.ndarray
        Quadratic coefficient matrix of the objective function (n x n)
    b0 : np.ndarray
        Linear coefficient vector of the objective function (n,)
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    N : int
        Number of nodes
    max_iter : int
        Maximum number of iterations
    seed : int
        Random seed
    alpha_init : float, optional
        Initial step size α^{-1} (if None, will be computed as alpha_init_constant / max{L_f_i})
    alpha_init_constant : float
        Constant for computing initial step size: alpha_init = alpha_init_constant / max{L_f_i}
        Only used if alpha_init is None
    delta : float
        Backtracking parameter in (0, 1]
    c : float
        Gossip matrix mixing parameter in (0, 0.5)
    verbose_every : int
        Print frequency
    initial_scale : float
        Initial value scaling factor
    phi_star : float, optional
        Optimal objective function value
    tol : float
        Convergence tolerance
    normalize_consensus_error : bool
        Whether to normalize consensus error
    use_optimal_consensus_error : bool
        Whether to use optimal consensus error definition
    x_star : np.ndarray, optional
        Optimal solution
    neighbors_list : list, optional
        Network topology
    initialization_mode : str
        "connected" or "independent"
        
    Returns:
    --------
    x_bar : np.ndarray
        Average solution
    hist : list
        Convergence history
    """
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    
    # Use provided network topology or generate fully connected network
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    
    # Build gossip matrix W
    W = build_gossip_matrix(neighbors_list, c=c)
    
    if verbose_every:
        print(f"Gossip matrix W (c={c}):")
        print(W)
        print(f"W eigenvalues: {np.linalg.eigvals(W)}")
    
    # Local smooth objective: if Q_list is provided, use node-specific objectives
    # φ_i(x) = 0.5 x^T Q_i x + q_i^T x + c_i (if q_list and constant_list are provided)
    # otherwise fall back to aggregated objective φ_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x + c0)
    if Q_list is not None:
        if len(Q_list) != N:
            raise ValueError(f"Q_list must have length N={N}, got {len(Q_list)}")
        
        if q_list is not None:
            # Node-specific with linear terms: φ_i(x) = 0.5 x^T Q_i x + q_i^T x + c_i
            def phi_i_node(i, x):
                val = 0.5 * x @ (Q_list[i] @ x) + q_list[i] @ x
                if constant_list is not None:
                    val += constant_list[i]
                return val
            
            def grad_phi_i_node(i, x):
                return Q_list[i] @ x + q_list[i]
        else:
            # Node-specific without linear terms: φ_i(x) = 0.5 x^T Q_i x
            def phi_i_node(i, x):
                return 0.5 * x @ (Q_list[i] @ x)
            
            def grad_phi_i_node(i, x):
                return Q_list[i] @ x
    else:
        def phi_i_node(i, x):
            return (0.5 * x @ (A0 @ x) + b0 @ x + c0) / N
        
        def grad_phi_i_node(i, x):
            return (A0 @ x + b0) / N
    
    # Helper to project onto box when bounds exist (can be None for L1-only case)
    def _project_box(vec):
        lo = box_lo if box_lo is not None else -np.inf
        hi = box_hi if box_hi is not None else np.inf
        if np.isneginf(lo) and np.isposinf(hi):
            return vec
        return np.clip(vec, lo, hi)
    
    # Determine centralized L1 coefficient (default to 1/N if not provided)
    lambda_coeff = lambda_l1 if lambda_l1 is not None else (1.0 / N)
    
    # Proximal operator for non-smooth term r_i:
    # If lambda_coeff == 0 -> r_i(x) = I_{[box_lo, box_hi]^n}(x)
    # If lambda_coeff > 0  -> r_i(x) = lambda_coeff * ||x||_1 (+ indicator if box bounds provided)
    # Uses L1 soft-thresholding when lambda_coeff > 0
    def prox_box(v, alpha):
        if lambda_coeff > 0:
            # Soft-thresholding for lambda_coeff * ||·||_1
            threshold = alpha * lambda_coeff
            v_soft = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
            return _project_box(v_soft)
        return _project_box(v)
    
    def _sample_initial_matrix():
        if box_lo is not None and box_hi is not None:
            return _project_box(initial_scale * rng.uniform(box_lo, box_hi, size=(N, n)))
        return initial_scale * rng.standard_normal((N, n))
    
    # Initialize states
    # X^0: N x n matrix, each row is x_i^0
    if initial_points is not None:
        initial_points_arr = np.asarray(initial_points)
        if initial_points_arr.shape != (N, n):
            raise ValueError(f"initial_points must have shape (N, n), got {initial_points_arr.shape}")
        X = initial_points_arr.copy()
    elif initialization_mode == "connected":
        # Connected mode: nodes with edges have the same initial values
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
        
        X = np.zeros((N, n))
        for group in node_groups:
            if box_lo is not None and box_hi is not None:
                x_init_group = initial_scale * rng.uniform(box_lo, box_hi, size=n)
            else:
                x_init_group = initial_scale * rng.standard_normal(n)
            for node in group:
                X[node, :] = x_init_group.copy()
        
        if verbose_every:
            print(f"Connected initialization mode:")
            print(f"  Number of connected components: {len(node_groups)}")
            
    elif initialization_mode == "independent":
        # Independent mode: all nodes have independent initial values
        X = _sample_initial_matrix()
        
        if verbose_every:
            print(f"Independent initialization mode:")
            print(f"  All {N} nodes initialized independently")
    else:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}")
    
    # S^0: N x n matrix
    S = np.zeros((N, n))
    
    # D^0: N x n matrix, initialized to 0
    D = np.zeros((N, n))
    
    # α^{-1}: initial step size
    # If alpha_init is not provided, compute it as constant / max{L_f_i}
    if alpha_init is None:
        # Compute Lipschitz constant for objective function
        L_obj = compute_lipschitz_constant(A0, b0)
        # For each node i, f_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
        # L_f_i = L_obj / N (same for all nodes)
        L_f_i = L_obj / N
        # max{L_f_i} = L_f_i (since all nodes have the same L_f_i)
        max_L_f_i = L_f_i
        # Compute initial step size: alpha_init = constant / max{L_f_i}
        alpha_init = alpha_init_constant / max_L_f_i
        if verbose_every:
            print(f"Computed alpha_init from formula: {alpha_init:.6f} = {alpha_init_constant} / {max_L_f_i:.6f}")
            print(f"  L_obj = {L_obj:.6f}, L_f_i = {L_f_i:.6f}, alpha_init_constant = {alpha_init_constant}")
    
    alpha_prev = alpha_init
    
    # Track gradient calls per node
    grad_calls_per_node = [0] * N
    
    hist = []
    
    # Record initial objective function value (before first iteration)
    x_bar_init = np.mean(X, axis=0)
    if Q_list is not None:
        if q_list is not None:
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) + q_i @ x_bar_init for Q_i, q_i in zip(Q_list, q_list))
            if constant_list is not None:
                obj_init += sum(constant_list)
        else:
            obj_init = sum(0.5 * x_bar_init @ (Q_i @ x_bar_init) for Q_i in Q_list)
    else:
        obj_init = 0.5 * x_bar_init @ (A0 @ x_bar_init) + b0 @ x_bar_init + c0
    # For centralized objective, L1 coefficient should be 1.0
    obj_init += 1.0 * np.linalg.norm(x_bar_init, 1)
    subopt_init = abs(obj_init - phi_star) if phi_star is not None else np.nan
    cons_err_init = sum(np.linalg.norm(X[i, :] - x_bar_init) for i in range(N)) / N
    x_bar_norm_sq_init = np.dot(x_bar_init, x_bar_init)
    cons_err_sq_sum_init = sum(np.dot(X[i, :] - x_bar_init, X[i, :] - x_bar_init) for i in range(N))
    # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum, alpha)
    hist.append((obj_init, 0.0, cons_err_init, 0.0, subopt_init, 0.0, 0, x_bar_norm_sq_init, cons_err_sq_sum_init, alpha_init))
    
    for k in range(max_iter):
        # (S.1) Communication Step
        # X^{k+1/2} = W * X^k
        X_half = W @ X
        
        # D^{k+1/2} = W * (∇F(X^k) + S^k + D^k)
        # ∇F(X^k) is N x n matrix where row i is grad_phi_i(x_i^k)
        grad_F = np.zeros((N, n))
        for i in range(N):
            grad_calls_per_node[i] += 1  # Count gradient call
            grad_F[i, :] = grad_phi_i_node(i, X[i, :])
        
        D_half = W @ (grad_F + S + D)
        
        # (S.2) Decentralized line-search
        # Each agent updates α_bar_i^k
        alpha_bar_list = []
        total_backtrack_iterations = 0  # Track total backtrack iterations across all nodes
        for i in range(N):
            x_i_k = X[i, :]
            x_i_half = X_half[i, :]
            d_i_half = -D_half[i, :]  # Note: direction is -d_i^{k+1/2}
            
            # Linesearch(α^{k-1}, φ_i, x_i^k, x_i^{k+1/2}, -d_i^{k+1/2}, δ)
            # Use already computed gradient grad_F[i, :] = grad_phi_i(x_i^k)
            alpha_bar_i, num_backtrack_i = linesearch_with_grad(
                alpha_prev,
                lambda x, node=i: phi_i_node(node, x),
                grad_F[i, :], x_i_k, x_i_half, d_i_half, delta
            )
            alpha_bar_list.append(alpha_bar_i)
            total_backtrack_iterations += num_backtrack_i
        
        # (S.3) Global min-consensus
        # α^k = min_{i in [N]} α_bar_i^k
        alpha_k = min(alpha_bar_list)
        
        # (S.4) Updates of the primal and dual variables
        # X^{k+1} = prox_{α^k R}(X^{k+1/2} - α^k * D^{k+1/2} + α^k * S^k)
        X_next = np.zeros_like(X)
        for i in range(N):
            X_next[i, :] = prox_box(
                X_half[i, :] - alpha_k * D_half[i, :] + alpha_k * S[i, :],
                alpha_k
            )
        
        # S^{k+1} = S^k + (1/α^k) * (X^{k+1/2} - X^{k+1} - α^k * D^{k+1/2})
        S_next = S + (1.0 / alpha_k) * (X_half - X_next - alpha_k * D_half)
        
        # D^{k+1} = D^{k+1/2} + (1/α^k) * (X^k - X^{k+1/2} - α^k * ∇F(X^k) - α^k * S^k)
        D_next = D_half + (1.0 / alpha_k) * (
            X - X_half - alpha_k * grad_F - alpha_k * S
        )
        
        # Update variables
        X = X_next
        S = S_next
        D = D_next
        alpha_prev = alpha_k
        
        # Metrics on network average
        x_bar = np.mean(X, axis=0)
        
        # Compute consensus error
        if use_optimal_consensus_error and x_star is not None:
            x_star_norm = np.linalg.norm(x_star)
            if x_star_norm > 1e-10:
                cons_err = max(np.linalg.norm(X[i, :] - x_star) for i in range(N)) / x_star_norm
            else:
                cons_err = max(np.linalg.norm(X[i, :] - x_star) for i in range(N))
        else:
            cons_err = np.mean([np.linalg.norm(X[i, :] - x_bar) for i in range(N)])
            if normalize_consensus_error:
                x_bar_norm = np.linalg.norm(x_bar)
                cons_err = cons_err / max(x_bar_norm, 1e-10)
        
        # Constraint violations (not applicable for ALDO - constraints handled via prox)
        max_viol = 0.0
        avg_viol = 0.0
        # Objective: φ(x) + r(x)
        # φ(x) = sum_i [(1/2) x^T Q_i x + q_i^T x + c_i] (if Q_list is provided)
        # r(x) = ||x||_1 (centralized L1 coefficient is 1.0, not lambda_l1 = 1/N)
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
            obj = 0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar + c0
        # For centralized objective, L1 coefficient should be 1.0, not lambda_l1 = 1/N
        obj += 1.0 * np.linalg.norm(x_bar, 1)
        subopt = abs(obj - phi_star) if phi_star is not None else np.nan
        
        # Average gradient calls per node
        avg_grad_calls = sum(grad_calls_per_node) / N
        
        # Compute x_bar norm for relative consensus error calculation
        x_bar_norm_sq = np.dot(x_bar, x_bar)
        
        # Compute sum of squared consensus errors: sum_i ||x_i - x_bar||^2
        cons_err_sq_sum = sum(np.dot(X[i, :] - x_bar, X[i, :] - x_bar) for i in range(N))
        
        # Store history
        # History format: (obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum, alpha)
        hist.append((obj, max_viol, cons_err, avg_viol, subopt, avg_grad_calls, total_backtrack_iterations, x_bar_norm_sq, cons_err_sq_sum, alpha_k))
        
        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | alpha {alpha_k:.6f}"
            if phi_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)
        
        # Optional stopping
        if phi_star is not None:
            if max(subopt, avg_viol) <= tol:
                break
    
    return x_bar, hist

