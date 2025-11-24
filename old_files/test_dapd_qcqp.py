import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --------------------------
# 生成随机正交矩阵
def random_orthonormal(n, rng):
    M = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)  # square, Q is orthonormal
    return Q

def rand_psd_merely_convex(n, rng, lo=0.0, hi=100.0):
    """Return U^T S U with S diagonal in [0,hi], forced to include at least one 0."""
    U = random_orthonormal(n, rng)
    diag = rng.uniform(lo, hi, size=n)
    # force at least one exact zero (merely convex)
    idx = rng.integers(0, n)
    diag[idx] = 0.0
    #强制merely convex
    S = np.diag(diag)
    A = U.T @ S @ U
    # A = U^T S U
    # numerical symmetrization
    return 0.5 * (A + A.T)

# --------------------------
# Ground truth (centralized)
# --------------------------
# 使用mosek求解
def solve_qcqp_ground_truth(A0, b0, A_list, b_list, c_list, box_lo, box_hi, neighbors_list=None):
    """
    min 0.5 x^T A0 x + b0^T x
    s.t. 0.5 x^T Aj x + bj^T x + cj <= 0,  j=1..m
         box_lo <= x <= box_hi
         x_i == x_j for all edges (i,j) in the network (consensus constraints)
    """
    n = A0.shape[0]
    N = len(A_list)  # Number of nodes
    
    # Create variables for each node
    x_vars = [cp.Variable(n) for _ in range(N)]
    
    # Objective: sum of local objectives
    obj = 0
    for i in range(N):
        obj += 0.5 * cp.quad_form(x_vars[i], A0) + b0 @ x_vars[i]
    obj = obj / N  # Average objective
    
    cons = []
    
    # Box constraints for each node
    for i in range(N):
        cons.append(x_vars[i] >= box_lo)
        cons.append(x_vars[i] <= box_hi)
    
    # Local constraints for each node
    for i in range(N):
        Aj, bj, cj = A_list[i], b_list[i], c_list[i]
        cons.append(0.5 * cp.quad_form(x_vars[i], Aj) + bj @ x_vars[i] + cj <= 0)
    
    # Consensus constraints: x_i == x_j for connected nodes
    if neighbors_list is not None:
        for i in range(N):
            for j in neighbors_list[i]:
                if i < j:  # Avoid duplicate constraints
                    cons.append(x_vars[i] == x_vars[j])
    
    prob = cp.Problem(cp.Minimize(obj), cons)

    # Try solvers in order of preference with detailed error reporting
    solver_names = ["MOSEK", "ECOS", "SCS"]
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
    last_error = None
    
    for solver_name, solver in zip(solver_names, solvers):
        try:
            print(f"Trying solver: {solver_name}")
            prob.solve(solver=solver, verbose=False)  # Disable verbose to avoid pipe errors
            print(f"Solver {solver_name} status: {prob.status}")
            if prob.status in ("optimal", "optimal_inaccurate"):
                print(f"Solver {solver_name} succeeded with objective value: {prob.value}")
                # Return the consensus solution (all x_vars should be equal)
                return x_vars[0].value.copy(), prob.value
            else:
                print(f"Solver {solver_name} failed with status: {prob.status}")
        except Exception as e:
            print(f"Solver {solver_name} raised exception: {e}")
            last_error = e
            continue

    # If all solvers failed, provide detailed error information
    error_msg = f"All solvers failed. Last solver status: {prob.status}"
    if last_error:
        error_msg += f". Last exception: {last_error}"
    
    # Check if the problem might be infeasible
    print("\nProblem diagnostics:")
    print(f"Problem size: n={n}, m={len(A_list)}")
    print(f"Box constraints: [{box_lo}, {box_hi}]")
    print(f"Objective matrix A0 eigenvalues (min, max): {np.linalg.eigvals(A0).min():.2e}, {np.linalg.eigvals(A0).max():.2e}")
    
    # Check constraint feasibility by trying a simpler problem
    print("Testing feasibility with relaxed constraints...")
    x_feas = cp.Variable(n)
    feas_prob = cp.Problem(cp.Minimize(0), [x_feas >= box_lo, x_feas <= box_hi])
    feas_prob.solve(solver=cp.ECOS, verbose=False)
    print(f"Box constraints feasibility: {feas_prob.status}")
    
    raise RuntimeError(error_msg)

# --------------------------
# Small-world network generation and helper functions
# --------------------------
def generate_small_world_network(N, E, seed=42):
    """
    Generate a small-world network G = (N, E) where:
    - First create a random cycle over N nodes (N edges)
    - Then add remaining E-N edges uniformly at random
    """
    rng = np.random.default_rng(seed)
    
    # Initialize adjacency list
    neighbors = [[] for _ in range(N)]
    
    # Step 1: Create a random cycle (N edges)
    # Randomly permute nodes to create the cycle
    cycle_order = rng.permutation(N)
    for i in range(N):
        current = cycle_order[i]
        next_node = cycle_order[(i + 1) % N]
        neighbors[current].append(next_node)
        neighbors[next_node].append(current)
    
    # Step 2: Add remaining E-N edges uniformly at random
    remaining_edges = E - N
    added_edges = 0
    max_attempts = remaining_edges * 10  # Prevent infinite loop
    attempts = 0
    
    while added_edges < remaining_edges and attempts < max_attempts:
        attempts += 1
        # Randomly select two nodes
        i, j = rng.choice(N, 2, replace=False)
        
        # Check if edge already exists
        if j not in neighbors[i]:
            neighbors[i].append(j)
            neighbors[j].append(i)
            added_edges += 1
    
    return neighbors

def generate_fully_connected_network(N):
    """
    Generate a fully connected network where every node is connected to every other node.
    For N nodes, this creates N(N-1)/2 edges.
    """
    neighbors = [[] for _ in range(N)]
    
    # Connect every node to every other node
    for i in range(N):
        for j in range(N):
            if i != j:  # Don't connect node to itself
                neighbors[i].append(j)
    
    return neighbors

def get_neighbors(i, neighbors_list):
    """Get neighbors of node i"""
    return neighbors_list[i]


def sum_neighbor_diffs(x_list, i, neighbors_list):
    """Sum of differences with neighbors for node i"""
    Ni = get_neighbors(i, neighbors_list)
    return sum(x_list[i] - x_list[j] for j in Ni)

# --------------------------
# Distributed APD (D-APD) for convex QCQP + box
# --------------------------
def d_apd_qcqp_merely_convex(A0, b0, pernode_constraints,
                              box_lo, box_hi,
                              N=3, max_iter=2000, seed=0,
                              c_alpha=0.1, c_beta=0.1, c_c=0.1,
                              zeta=1.0, tau=0.08, gamma=0.2,
                              verbose_every=200, initial_scale=1.0,
                              f_star=None, tol=1e-8):
    """
    pernode_constraints: list of length N
      pernode_constraints[i] = (A_list_i, b_list_i, c_list_i)
      with Aj PSD (merely convex). Box handled via prox (clip).
    """
    rng = np.random.default_rng(seed)
    n = A0.shape[0]
    
    # Generate fully connected network
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

    # Prox for constraints: projection onto feasible set for a single node
    # For QCQP with box constraints, we need to project onto the intersection of
    # box constraints and the node's local quadratic constraints
    def prox_primal_i(i, v, tau):
        """
        Proximal operator for the constraint indicator function for node i.
        For QCQP, this should project onto the feasible set defined by:
        - Box constraints: box_lo <= x <= box_hi
        - Node i's local quadratic constraints: g_i(x) <= 0
        
        Since exact projection onto the intersection is computationally expensive,
        we use a two-step approach:
        1. First project onto box constraints
        2. Then use a few iterations of gradient projection to handle node i's quadratic constraints
        """
        # Step 1: Project onto box constraints
        # x_box = np.clip(v, box_lo, box_hi)
        
        # Step 2: Handle node i's quadratic constraints using gradient projection
        # x_current = x_box.copy()
        x_current = v.copy()
        max_inner_iter = 50  # Limit inner iterations for efficiency
        
        # Get node i's constraints
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        
        for inner_iter in range(max_inner_iter):
            # Check constraint violations for node i only
            max_violation = 0.0
            total_grad = np.zeros_like(x_current)
            
            for Aik, bik, cik in zip(A_list_i, b_list_i, c_list_i):
                constraint_val = 0.5 * x_current @ (Aik @ x_current) + bik @ x_current + cik
                if constraint_val > 0:  # Constraint violated
                    max_violation = max(max_violation, constraint_val)
                    # Gradient of violated constraint
                    grad_constraint = Aik @ x_current + bik
                    total_grad += grad_constraint
            
            # If no violations or violations are small, stop
            if max_violation < 1e-6:
                break
                
            # Take a small step in the direction of constraint satisfaction
            step_size = 0.1 * tau  # Use a fraction of the main step size
            x_current = x_current - step_size * total_grad
            
            # Re-project onto box constraints
            # x_current = np.clip(x_current, box_lo, box_hi)
        
        return x_current
    

    # Dual projection onto R_+^{m_i}
    def proj_R_plus(theta):
        return np.maximum(theta, 0.0)
    

    # Init states - initialize with consensus (all x_i are equal)
    x_init = initial_scale * rng.standard_normal(n)
    # x_init = np.zeros(n)
    x = [x_init.copy() for _ in range(N)]  # All nodes start with the same x
    x_prev = [xi.copy() for xi in x]
    
    # Print initial x for debugging
    if verbose_every:
        print(f"Initial x (all nodes): {x_init}")
        print(f"Initial x norm: {np.linalg.norm(x_init):.6f}")
        print(f"Initial x range: [{np.min(x_init):.6f}, {np.max(x_init):.6f}]")
    
    # Parameters for each variable group (same for all variables in the same group)
    tau_list = [tau] * N  # Each node has the same step size for its variable group
    zeta_i = [zeta] * N  # Each node has the same zeta for its variable group
    # θ init
    theta = [np.zeros(len(pernode_constraints[i][0])) for i in range(N)]
    theta_prev = [np.zeros(len(pernode_constraints[i][0])) for i in range(N)]
    sigma0_max = max(zeta_i[i] * tau_list[i] for i in range(N))
    eta = 1.0  # eta is a constant, always equal to 1

    def gamma_k():
        return gamma  # 1.0 / (2.0 * d_max * sigma0_max * N * ((2.0 / c_alpha) + (eta / c_c)))
    
    # Initialize s as zero vectors
    s = [np.zeros(n) for _ in range(N)]
    s_prev = [np.zeros(n) for _ in range(N)]

    hist = []

    # 记录每步每节点六个量的二范数
    norms_history = {
        's_diff_term': [],  # list over iters of list over nodes
        'grad_term': [],
        'p_i': [],
        'x_grad': [],
        'x_update': [],
        's_update': [],
    }
    
    # 记录每步的均值统计
    means_history = {
        's_diff_term': [],  # list over iters of scalar means
        'grad_term': [],
        'p_i': [],
        'x_grad': [],
        'x_update': [],
        's_update': [],
    }
    
    # 记录梯度方向相关量
    direction_history = {
        'x_grad_p_i_cosine': [],  # cosine between x_grad and p_i for each node
        'grad_consistency': [],   # consistency of gradient directions across nodes
        'p_consistency': [],      # consistency of p_i directions across nodes
    }

    for k in range(max_iter):
        gamma = gamma_k()

        # ---- Standard APD update ----
        # 本次迭代，逐节点的六个量的二范数
        iter_s_diff_terms = []
        iter_grad_terms = []
        iter_p_is = []
        iter_x_grads = []
        iter_x_updates = []
        iter_s_updates = []
        
        # 本次迭代，存储向量用于方向计算
        iter_x_grad_vectors = []
        iter_p_i_vectors = []
        for i in range(N):
            # Compute p_i according to the correct formula:
            # p_i^k ← ∑_{j∈N_i} ((1+η^k)(s_i^k - s_j^k) - η^k(s_i^{k-1} - s_j^{k-1}))
            #        + (1+η^k)∇g_i(x_i^k)^T θ_i^k - η^k∇g_i(x_i^{k-1})^T θ_i^{k-1}
            
            # First term: sum over neighbors of s differences
            s_diff_term = np.zeros(n)
            Ni = get_neighbors(i, neighbors_list)
            for j in Ni:
                s_diff_term += (1 + eta) * (s[i] - s[j]) - eta * (s_prev[i] - s_prev[j])
            
            # Second term: gradient terms
            grad_term = (1 + eta) * jacT_theta_i(i, x[i], theta[i]) \
                       - eta * jacT_theta_i(i, x_prev[i], theta_prev[i])
            
            p_i = s_diff_term + grad_term

            tau_i = tau_list[i]
            sigma_i = zeta_i[i] * tau_i

            x_grad = grad_f_i(x[i])
            tmp_update = x_grad + p_i
            x_next = prox_primal_i(i, x[i] - tau_i * tmp_update, tau_i)
            x_update = x_next - x[i]
            # x_next = x[i] - x_update
            theta_next = proj_R_plus(theta[i] + sigma_i * g_i_of(i, x_next))
            s_update = ((1 + eta) * x[i] - eta * x_prev[i])
            s_next = s[i] + gamma * s_update

            # 记录本节点本步的六个量的二范数
            iter_s_diff_terms.append(float(np.linalg.norm(s_diff_term)))
            iter_grad_terms.append(float(np.linalg.norm(grad_term)))
            iter_p_is.append(float(np.linalg.norm(p_i)))
            iter_x_grads.append(float(np.linalg.norm(x_grad)))
            iter_x_updates.append(float(np.linalg.norm(x_update)))
            iter_s_updates.append(float(np.linalg.norm(s_update)))
            
            # 存储向量用于方向计算
            iter_x_grad_vectors.append(x_grad.copy())
            iter_p_i_vectors.append(p_i.copy())

            x_prev[i] = x[i]
            theta_prev[i] = theta[i]
            s_prev[i] = s[i]
            x[i] = x_next
            theta[i] = theta_next
            s[i] = s_next

        # 追加到整体时间序列
        norms_history['s_diff_term'].append(iter_s_diff_terms)
        norms_history['grad_term'].append(iter_grad_terms)
        norms_history['p_i'].append(iter_p_is)
        norms_history['x_grad'].append(iter_x_grads)
        norms_history['x_update'].append(iter_x_updates)
        norms_history['s_update'].append(iter_s_updates)
        
        # 计算并记录均值
        means_history['s_diff_term'].append(np.mean(iter_s_diff_terms))
        means_history['grad_term'].append(np.mean(iter_grad_terms))
        means_history['p_i'].append(np.mean(iter_p_is))
        means_history['x_grad'].append(np.mean(iter_x_grads))
        means_history['x_update'].append(np.mean(iter_x_updates))
        means_history['s_update'].append(np.mean(iter_s_updates))
        
        # 计算梯度方向相关量
        # 1. x_grad 与 p_i 的夹角余弦（每个节点）
        x_grad_p_cosines = []
        for i in range(N):
            x_grad_norm = np.linalg.norm(iter_x_grad_vectors[i])
            p_i_norm = np.linalg.norm(iter_p_i_vectors[i])
            if x_grad_norm > 1e-12 and p_i_norm > 1e-12:
                cosine = np.dot(iter_x_grad_vectors[i], iter_p_i_vectors[i]) / (x_grad_norm * p_i_norm)
                x_grad_p_cosines.append(cosine)
            else:
                x_grad_p_cosines.append(0.0)
        
        # 2. 梯度方向一致性（节点间梯度方向的相似性）
        grad_consistency = 0.0
        if N > 1:
            total_pairs = 0
            for i in range(N):
                for j in range(i+1, N):
                    norm_i = np.linalg.norm(iter_x_grad_vectors[i])
                    norm_j = np.linalg.norm(iter_x_grad_vectors[j])
                    if norm_i > 1e-12 and norm_j > 1e-12:
                        cosine_ij = np.dot(iter_x_grad_vectors[i], iter_x_grad_vectors[j]) / (norm_i * norm_j)
                        grad_consistency += cosine_ij
                        total_pairs += 1
            if total_pairs > 0:
                grad_consistency /= total_pairs
        
        # 3. p_i 方向一致性
        p_consistency = 0.0
        if N > 1:
            total_pairs = 0
            for i in range(N):
                for j in range(i+1, N):
                    norm_i = np.linalg.norm(iter_p_i_vectors[i])
                    norm_j = np.linalg.norm(iter_p_i_vectors[j])
                    if norm_i > 1e-12 and norm_j > 1e-12:
                        cosine_ij = np.dot(iter_p_i_vectors[i], iter_p_i_vectors[j]) / (norm_i * norm_j)
                        p_consistency += cosine_ij
                        total_pairs += 1
            if total_pairs > 0:
                p_consistency /= total_pairs
        
        direction_history['x_grad_p_i_cosine'].append(x_grad_p_cosines)
        direction_history['grad_consistency'].append(grad_consistency)
        direction_history['p_consistency'].append(p_consistency)


        # Metrics on network average
        x_bar = sum(x) / N
        cons_err = np.sqrt(sum(np.linalg.norm(x[i] - x_bar) ** 2 for i in range(N)) / N)
        # Violations (global)
        all_vals = []
        for i in range(N):
            all_vals.extend(g_i_of(i, x_bar))
        all_vals = np.array(all_vals)
        
        # # Debug: print all_vals to check constraint violations
        # if verbose_every and (k % 10 == 0 or k == max_iter - 1):
        #     print(f"  Debug - all_vals at iter {k}: {all_vals}")
        #     print(f"  Debug - all_vals shape: {all_vals.shape}")
        #     print(f"  Debug - all_vals > 0: {all_vals[all_vals > 0]}")
        
        max_viol = float(np.maximum(all_vals, 0).max()) if all_vals.size > 0 else 0.0
        avg_viol = float(np.maximum(all_vals, 0).mean()) if all_vals.size > 0 else 0.0
        
        # Debug: print max_viol calculation
        # if verbose_every and (k % 10 == 0 or k == max_iter - 1):
        #     print(f"  Debug - max_viol at iter {k}: {max_viol}")
        #     print(f"  Debug - np.maximum(all_vals, 0): {np.maximum(all_vals, 0)}")
        #     print(f"  Debug - np.maximum(all_vals, 0).max(): {np.maximum(all_vals, 0).max()}")
        obj = 0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar
        subopt = abs(obj - f_star) / max(1-8, abs(f_star)) if f_star is not None else np.nan

        hist.append((obj, max_viol, cons_err, avg_viol, subopt))

        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | eta {eta:.3f}"
            if f_star is not None:
                msg += f" | rel subopt {subopt:.2e}"
            print(msg)

        # Optional stopping (per论文条件的改良版)
        if f_star is not None:
            if max(subopt, avg_viol) <= tol:
                break

    return x_bar, hist, norms_history, means_history, direction_history

# --------------------------
# Generate feasible QCQP problem
# --------------------------
def generate_feasible_qcqp(n, m, rng, box_lo=-10.0, box_hi=10.0):
    """
    Generate a QCQP problem that is guaranteed to be feasible.
    """
    # Generate objective: strongly convex to ensure boundedness
    A0 = rand_psd_merely_convex(n, rng, lo=1.0, hi=10.0)  # Ensure positive eigenvalues
    b0 = rng.standard_normal(n) * 0.1  # Small linear term
    
    # Generate constraints that are known to be feasible
    A_list, b_list, c_list = [], [], []
    
    # Create a feasible point first
    x_feasible = rng.uniform(box_lo * 0.3, box_hi * 0.3, size=n)  # Point well inside the box
    
    # Add constraints using the correct method: A_j = Γ_j^T S_j Γ_j
    for i in range(m):
        # Generate A_j = Γ_j^T S_j Γ_j where Γ_j is random orthonormal and S_j is diagonal
        # Aj = rand_psd_merely_convex(n, rng, lo=0.0, hi=100.0)  # This gives Γ^T S Γ form
        Aj = rand_psd_merely_convex(n, rng, lo=1.0, hi=10.0)  # This gives Γ^T S Γ form
        
        # Generate random linear term and constant
        bj = rng.standard_normal(n) * 0.1
        cj = rng.uniform(-1.0, 1.0)
        
        # Verify this constraint is satisfied at x_feasible
        constraint_value = 0.5 * x_feasible @ (Aj @ x_feasible) + bj @ x_feasible + cj
        if constraint_value > 0:
            # Adjust cj to make it feasible
            cj = cj - constraint_value - 0.1  # Add small margin
        
        A_list.append(Aj)
        b_list.append(bj)
        c_list.append(cj)
    
    return A0, b0, A_list, b_list, c_list

# --------------------------
# Build a merely-convex QCQP instance & run
# --------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(1)

    # Problem size (paper setup) - reduced for testing
    n = 10   # reduced from 1000 for faster testing
    N = 3      # number of agents (fully connected triangle network)
    m = N      # number of constraints equals number of nodes
    E = 3      # number of edges in triangle (N*(N-1)/2 = 3*2/2 = 3)

    # Box domain X = [-10, 10]^n
    box_lo, box_hi = -10.0, 10.0

    # Generate feasible QCQP problem
    print("Generating feasible QCQP problem...")
    A0, b0, A_list, b_list, c_list = generate_feasible_qcqp(n, m, rng, box_lo, box_hi)
    print(f"Generated problem: n={n}, m={m}, box=[{box_lo}, {box_hi}]")
    
    # Generate and display network topology
    print(f"Generating fully connected triangle network: N={N}, E={E}")
    neighbors_list = generate_fully_connected_network(N)
    print("Network topology:")
    for i in range(N):
        print(f"  Node {i}: neighbors = {neighbors_list[i]}")
    print(f"Maximum degree: {max(len(neighbors_list[i]) for i in range(N))}")
    print(f"Total edges: {sum(len(neighbors_list[i]) for i in range(N)) // 2}")  # Each edge counted twice
    

    # Centralized ground truth with consensus constraints
    print("Solving centralized ground truth with consensus constraints...")
    x_star, f_star = solve_qcqp_ground_truth(A0, b0, A_list, b_list, c_list, box_lo, box_hi, neighbors_list)
    print("Ground truth: f* =", f_star)
    print(f"Optimal solution x*: {x_star}")
    print(f"x* norm: {np.linalg.norm(x_star):.6f}")
    print(f"x* range: [{np.min(x_star):.6f}, {np.max(x_star):.6f}]")

    # Each node is responsible for ALL variables (full variable set)
    # All nodes update all dimensions of the decision variable
    print("Variable allocation:")
    for i in range(N):
        print(f"  Node {i}: all variables [0 to {n-1}] (size: {n})")
    
    # Each node is responsible for exactly one constraint (its own constraint)
    # Node i is responsible for constraint i (g_i(x))
    pernode_constraints = []
    for i in range(N):
        # Each node gets exactly one constraint: constraint i
        A_i = [A_list[i]]  # Node i gets constraint i
        b_i = [b_list[i]]
        c_i = [c_list[i]]
        pernode_constraints.append((A_i, b_i, c_i))
    
    # Display constraint allocation
    print("Constraint allocation:")
    for i in range(N):
        print(f"  Node {i}: constraint {i} (g_{i}(x))")

    # Run distributed solver
    # n = 10, tau = 0.08
    # n = 100, tau = 0.01
    # tau = 0.08
    # gamma = 0.2
    taus = [0.08, 0.05, 0.02, 0.01, 0.005, 0.001]
    gammas = [0.2, 0.1, 0.05, 0.02]
    fig, axes = plt.subplots(len(taus), len(gammas), figsize=(16, 24))
    max_iter = 2000
    for i,tau in enumerate(taus):
        for j, gamma in enumerate(gammas):
            x_bar, hist, norms_history, means_history, direction_history = d_apd_qcqp_merely_convex(
                A0, b0, pernode_constraints,
                box_lo=box_lo, box_hi=box_hi,
                N=N, max_iter=max_iter, seed=42,
                c_alpha=0.1, c_beta=0.1, c_c=0.1,
                zeta=1.0, tau=tau, gamma=gamma,
                verbose_every=200, initial_scale=1.0,
                f_star=f_star, tol=1e-8
            )

            # Final comparison
            err_x = np.linalg.norm(x_bar - x_star)
            obj_bar = 0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar
            # global violations at x_bar
            all_vals = np.array([0.5 * x_bar @ (Aj @ x_bar) + bj @ x_bar + cj for Aj, bj, cj in zip(A_list, b_list, c_list)])
            max_viol = float(np.maximum(all_vals, 0).max()) if all_vals.size > 0 else 0.0
            avg_viol = float(np.maximum(all_vals, 0).mean()) if all_vals.size > 0 else 0.0

            print("\nD-APD result:")
            print("||x_bar - x*|| =", err_x)
            print("f(x_bar)       =", obj_bar, " | f* =", f_star, " | rel subopt =", abs(obj_bar - f_star)/max(1.0,abs(f_star)))
            print("max violation  =", max_viol, " | avg violation =", avg_viol)

            # ---- Simple plots (optional; with n=1000 curves still fine) ----
            iters = np.arange(len(hist))
            objs = [h[0] for h in hist]
            maxV = [h[1] for h in hist]
            cons = [h[2] for h in hist]
            avgV = [h[3] for h in hist]
            relsub = [h[4] for h in hist]

            ax00 = axes[i, j]
            # ax00.plot(iters, objs, lw=2)
            # ax00.axhline(f_star, color='k', ls='--', alpha=0.5)
            # ax00.set_title(f'Objective, tau={tau}, gamma={gamma}')
            # ax00.set_ylim(f_star - 0.1, 1)
            # ax00.semilogy(iters, cons, lw=2)
            # ax00.set_title(f'Cons err, tau={tau}, gamma={gamma}')
            ax00.semilogy(iters, maxV, lw=2)
            ax00.set_title(f'Max vio, tau={tau}, gamma={gamma}')
            # ax00.semilogy(iters, avgV, lw=2)
            # ax00.set_title(f'Avg vio, tau={tau}, gamma={gamma}')
            ax00.grid(True, alpha=0.3)

plt.savefig(f'dapd_grid_search_maxV_plots_10var_prox_{max_iter}iters.png', dpi=300, bbox_inches='tight')