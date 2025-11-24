import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --------------------------
# Utilities: random orthonormal & PSD with zero eigenvalues (merely convex)
# --------------------------

def compute_lipschitz_constant(A, b=None):
    """
    计算二次函数 f(x) = 0.5 * x^T * A * x + b^T * x 的Lipschitz常数
    """
    A_sym = 0.5 * (A + A.T)
    eigenvals = np.linalg.eigvals(A_sym)
    L = np.max(np.abs(eigenvals))
    return L

def compute_constraint_lipschitz_constants(A_list, b_list):
    """
    计算所有约束函数的Lipschitz常数
    """
    L_list = []
    for A_j, b_j in zip(A_list, b_list):
        L_j = compute_lipschitz_constant(A_j, b_j)
        L_list.append(L_j)
    return L_list

def compute_dual_variable_bound(A0, b0, A_list, b_list, c_list, box_lo, box_hi, x_slater=None):
    """
    使用Slater点计算对偶变量theta的界B
    """
    if x_slater is None:
        # Try to find a Slater point by random search
        n = A0.shape[0]
        rng = np.random.default_rng(123)
        x_slater = None
        for _ in range(1000):
            x_candidate = rng.uniform(box_lo, box_hi, n)
            all_feasible = True
            for Aj, bj, cj in zip(A_list, b_list, c_list):
                val = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate + cj
                if val >= 0:
                    all_feasible = False
                    break
            if all_feasible:
                x_slater = x_candidate
                break
        
        if x_slater is None:
            # If no Slater point found, use optimization to find one
            # Return a conservative bound
            return 1e-10
    
    # Compute bound
    f_slater = 0.5 * x_slater @ (A0 @ x_slater) + b0 @ x_slater
    max_gap = 0.0
    for Aj, bj, cj in zip(A_list, b_list, c_list):
        g_val = 0.5 * x_slater @ (Aj @ x_slater) + bj @ x_slater + cj
        if g_val < 0:
            gap = abs(f_slater / g_val)
            max_gap = max(max_gap, gap)
    
    return max_gap

def get_neighbors(i, neighbors_list):
    """
    Get the set of neighbors of node i
    """
    return neighbors_list[i]

def generate_fully_connected_network(N):
    """
    Generate a fully connected network with N nodes
    """
    neighbors_list = []
    for i in range(N):
        neighbors = [j for j in range(N) if j != i]
        neighbors_list.append(neighbors)
    return neighbors_list

def generate_small_world_network(N, E, seed=0):
    """
    Generate a small-world network with N nodes and E edges
    """
    rng = np.random.default_rng(seed)
    neighbors_list = [[] for _ in range(N)]
    
    # Initialize as a ring
    for i in range(N):
        neighbors_list[i].append((i + 1) % N)
        neighbors_list[(i + 1) % N].append(i)
    
    # Add random edges
    edges_added = N  # Start with ring edges
    while edges_added < E:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i != j and j not in neighbors_list[i]:
            neighbors_list[i].append(j)
            neighbors_list[j].append(i)
            edges_added += 1
    
    return neighbors_list

# --------------------------
# Ground truth (centralized)
# --------------------------
def solve_qcqp_ground_truth(A0, b0, A_list, b_list, c_list, box_lo, box_hi, neighbors_list=None):
    """
    min 0.5 x^T A0 x + b0^T x
    s.t. 0.5 x^T Aj x + bj^T x + cj <= 0,  j=1..m
         box_lo <= x <= box_hi
         x_i == x_j for all edges (i,j) in the network (consensus constraints)
    """
    n = A0.shape[0]
    N = len(A_list)
    
    x_vars = [cp.Variable(n) for _ in range(N)]
    
    # Objective: sum of local objectives
    obj = 0
    for i in range(N):
        obj += 0.5 * cp.quad_form(x_vars[i], A0) + b0 @ x_vars[i]
    obj = obj / N
    
    cons = []
    
    # Box constraints for each node
    for i in range(N):
        cons.append(x_vars[i] >= box_lo)
        cons.append(x_vars[i] <= box_hi)
    
    # Local constraints for each node
    for i in range(N):
        Aj, bj, cj = A_list[i], b_list[i], c_list[i]
        cons.append(0.5 * cp.quad_form(x_vars[i], Aj) + bj @ x_vars[i] + cj <= 0)
    
    # Consensus constraints
    if neighbors_list is not None:
        for i in range(N):
            for j in neighbors_list[i]:
                if i < j:
                    cons.append(x_vars[i] == x_vars[j])
    
    prob = cp.Problem(cp.Minimize(obj), cons)
    
    solver_names = ["MOSEK", "ECOS", "SCS"]
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
    
    for solver_name, solver in zip(solver_names, solvers):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                x_star = x_vars[0].value
                f_star = prob.value
                return x_star, f_star
        except Exception as e:
            continue
    
    raise RuntimeError("All solvers failed to solve the QCQP problem")

# --------------------------
# Distributed APDB with Backtracking for QCQP + box
# --------------------------
def d_apdb_qcqp_merely_convex(A0, b0, pernode_constraints,
                               box_lo, box_hi,
                               N=3, max_iter=2000, seed=0,
                               c_alpha=0.1, c_beta=0.1, c_c=0.1,
                               zeta=1.0, tau_init=0.08, gamma=None,
                               rho_shrink=0.5, delta=0.1,
                               verbose_every=200, initial_scale=5.0,
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
    
    # Use provided network topology or generate fully connected network
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    d_max = max(len(neighbors_list[i]) for i in range(N))

    # Local smooth objective: f_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x)
    def grad_f_i(x):
        return (A0 @ x + b0) / N

    # Local convex constraints for node i
    def g_i_of(i, x):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        return np.array([0.5 * x @ (Aik @ x) + bik @ x + cik
                         for Aik, bik, cik in zip(A_list_i, b_list_i, c_list_i)])

    # Jacobian^T * theta for node i
    def jacT_theta_i(i, x, theta):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        v = np.zeros_like(x)
        for th, Aik, bik, _ in zip(theta, A_list_i, b_list_i, c_list_i):
            v += th * (Aik @ x + bik)
        return v
    
    # Merit function P_i^k for backtracking
    def merit_function_P(i, x_plus, theta_plus, x_k, theta_k, tau_k, sigma_k, alpha_k, beta_k, alpha_kp1, beta_kp1, c_plus, eta):
        """
        P_i^k = -\frac{1}{2}\Big(\frac{1}{2\tau_i^k}-\eta^k(\alpha_i^k + \beta_i^k)-c_i^{k+1}\Big) \|x_i^{k+1}-x_i^k\|^2
               - \frac{1}{2\sigma_i^k } \|\theta_i^{k+1}-\theta_i^k\|^2
               + \frac{1}{2\alpha_i^{k+1}}\|\langle \J g_i(x_i^{k+1}), \theta_i^{k+1} - \theta_i^k \rangle\|^2
               + \frac{1}{2\beta_i^{k+1}}\|\langle \J g_i(x_i^{k+1}) - \J g_i(x_i^k), \theta_i^k \rangle\|^2
        """
        # Term 1: -\frac{1}{2}\Big(\frac{1}{2\tau_i^k}-\eta^k(\alpha_i^k + \beta_i^k)-c_i^{k+1}\Big) \|x_i^{k+1}-x_i^k\|^2
        dx = x_plus - x_k
        dx_norm_sq = np.dot(dx, dx)
        term1_coeff = -0.5 * (1.0 / (2.0 * tau_k) - eta * (alpha_k + beta_k) - c_plus)
        term1 = term1_coeff * dx_norm_sq
        
        # Term 2: - \frac{1}{2\sigma_i^k } \|\theta_i^{k+1}-\theta_i^k\|^2
        dtheta = theta_plus - theta_k
        dtheta_norm_sq = np.dot(dtheta, dtheta)
        term2 = -0.5 * dtheta_norm_sq / sigma_k
        
        # Term 3: + \frac{1}{2\alpha_i^{k+1}}\|\langle \J g_i(x_i^{k+1}), \theta_i^{k+1} - \theta_i^k \rangle\|^2
        # \J g_i(x_i^{k+1}) * (theta_i^{k+1} - theta_i^k) = jacT_theta_i(i, x_plus, theta_plus - theta_k)
        jac_g_xplus_dtheta = jacT_theta_i(i, x_plus, dtheta)
        term3 = 0.5 * np.dot(jac_g_xplus_dtheta, jac_g_xplus_dtheta) / alpha_kp1
        
        # Term 4: + \frac{1}{2\beta_i^{k+1}}\|\langle \J g_i(x_i^{k+1}) - \J g_i(x_i^k), \theta_i^k \rangle\|^2
        # \J g_i(x_i^{k+1}) * theta_i^k - \J g_i(x_i^k) * theta_i^k = jacT_theta_i(i, x_plus, theta_k) - jacT_theta_i(i, x_k, theta_k)
        jac_g_xplus_theta_k = jacT_theta_i(i, x_plus, theta_k)
        jac_g_xk_theta_k = jacT_theta_i(i, x_k, theta_k)
        jac_diff = jac_g_xplus_theta_k - jac_g_xk_theta_k
        term4 = 0.5 * np.dot(jac_diff, jac_diff) / beta_kp1
        
        return term1 + term2 + term3 + term4

    # Prox for box
    def prox_primal(v, tau):
        return np.clip(v, box_lo, box_hi)

    # Dual projection onto R_+
    def proj_R_plus(theta):
        return np.maximum(theta, 0.0)

    # Init states - initialize based on mode
    if initialization_mode == "connected":
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
            x_init_group = initial_scale * rng.standard_normal(n)
            for node in group:
                x.append(x_init_group.copy())
                
    elif initialization_mode == "independent":
        x = [initial_scale * rng.standard_normal(n) for _ in range(N)]
    
    x_prev = [xi.copy() for xi in x]
    
    if verbose_every:
        if initialization_mode == "connected":
            print(f"Connected initialization mode:")
            print(f"  Number of connected components: {len(node_groups)}")
            for i, group in enumerate(node_groups):
                print(f"  Component {i}: nodes {group}")
        else:
            print(f"Independent initialization mode:")
            print(f"  All {N} nodes initialized independently")
    
    # Parameters
    tau_list = [tau_init] * N
    zeta_i = [zeta] * N
    theta = [np.zeros(len(pernode_constraints[i][0])) for i in range(N)]
    theta_prev = [np.zeros(len(pernode_constraints[i][0])) for i in range(N)]
    sigma0_max = max(zeta_i[i] * tau_list[i] for i in range(N))
    eta = 1.0

    def gamma_k():
        """
        Compute gamma based on formula:
        gamma^k = (2 * d_max * sigma_max^0 * |N| * (2/c_alpha + eta_k/c_c))^(-1)
        """
        return 1.0 / (2.0 * d_max * sigma0_max * N * ((2.0 / c_alpha) + (eta / c_c)))
    
    # Initialize gamma based on formula (eta=1.0 initially)
    # Note: gamma will be updated at the start of each iteration based on current eta
    # User-provided gamma (if any) is ignored - always computed from formula
    gamma_user_provided = gamma  # Store for informational purposes only
    gamma = gamma_k()  # Always use formula-based value
    
    s = [np.zeros(n) for _ in range(N)]
    s_prev = [np.zeros(n) for _ in range(N)]

    hist = []

    for k in range(max_iter):
        # Update gamma based on current eta according to formula:
        # gamma^k = (2 * d_max * sigma_max^0 * |N| * (2/c_alpha + eta_k/c_c))^(-1)
        # Gamma is always computed from formula, ignoring any user-provided value
        gamma = gamma_k()
        
        # Print gamma info at first iteration
        if k == 0 and verbose_every:
            print(f"Gamma information:")
            print(f"  gamma always computed from formula (ignoring user-provided value if any)")
            print(f"  gamma at iteration 0: {gamma:.10f}")
            if gamma_user_provided is not None:
                print(f"  (user-provided gamma {gamma_user_provided:.10f} was ignored)")
            print(f"  d_max: {d_max}")
            print(f"  sigma0_max: {sigma0_max}")
            print(f"  N: {N}")
            print(f"  c_alpha: {c_alpha}")
            print(f"  c_c: {c_c}")
            print(f"  eta (initial): {eta}")

        # ---- Backtracking ----
        tau_new = [None] * N
        ell_k = [0] * N
        backtracking_failed_count = [0] * N  # Track how many times condition fails per node
        
        for i in range(N):
            # Compute p_i
            s_diff_term = np.zeros(n)
            Ni = get_neighbors(i, neighbors_list)
            for j in Ni:
                s_diff_term += (1 + eta) * (s[i] - s[j]) - eta * (s_prev[i] - s_prev[j])
            
            grad_term = (1 + eta) * jacT_theta_i(i, x[i], theta[i]) \
                       - eta * jacT_theta_i(i, x_prev[i], theta_prev[i])
            
            p_i = s_diff_term + grad_term
            
            # Backtracking line search
            t = tau_list[i]
            lcnt = 0
            
            # Reference point: x_i^k, theta_i^k (current iteration)
            x_k = x[i]
            theta_k = theta[i]
            tau_k = tau_list[i]  # This is tau_i^{k-1} at the start of iteration k
            sigma_k = zeta_i[i] * tau_k
            
            # Compute alpha_i^k, beta_i^k, c_i^k using tau_i^{k-1} (previous iteration's tau)
            # alpha_i^k = c_alpha / tau_i^{k-1}
            # beta_i^k = c_beta / tau_i^{k-1}
            # c_i^k = c_c / tau_i^{k-1}
            alpha_i_k = c_alpha / tau_k  # tau_k is tau_i^{k-1} here
            beta_i_k = c_beta / tau_k
            c_i_k = c_c / tau_k
            
            while True:
                sigma = zeta_i[i] * t
                x_plus = prox_primal(x[i] - t * (grad_f_i(x[i]) + p_i), t)
                theta_plus = proj_R_plus(theta[i] + sigma * g_i_of(i, x_plus))
                
                # Compute alpha_i^{k+1}, beta_i^{k+1}, c_i^{k+1} using tau_i^k (current trial step size t)
                # alpha_i^{k+1} = c_alpha / tau_i^k
                # beta_i^{k+1} = c_beta / tau_i^k
                # c_i^{k+1} = c_c / tau_i^k
                # Note: tau_i^k is the trial step size t in backtracking
                alpha_i_kp1 = c_alpha / t
                beta_i_kp1 = c_beta / t
                c_i_kp1 = c_c / t
                
                # Merit function P_i^k based backtracking condition
                # Condition: P_i^k <= -\frac{\delta}{2\tau_i^k} \|x_i^{k+1} - x_i^k\|^2
                #            -\frac{\delta}{2\sigma_i^k} \|\theta_i^{k+1} - \theta_i^k\|^2
                
                # Compute merit function at new point (x_plus, theta_plus)
                # tau_i^k is the trial step size t
                sigma_k_trial = zeta_i[i] * t
                P_plus = merit_function_P(i, x_plus, theta_plus, x_k, theta_k,
                                          t, sigma_k_trial, alpha_i_k, beta_i_k, 
                                          alpha_i_kp1, beta_i_kp1, c_i_kp1, eta)
                
                # Compute decrement terms
                dx_plus = x_plus - x_k
                dtheta_plus = theta_plus - theta_k
                dx_plus_norm_sq = np.dot(dx_plus, dx_plus)
                dtheta_plus_norm_sq = np.dot(dtheta_plus, dtheta_plus)
                
                # Right-hand side of the condition: only the negative penalty terms
                # Use tau_i^k (trial step size t) and corresponding sigma_i^k
                rhs = -(delta / (2.0 * t)) * dx_plus_norm_sq - (delta / (2.0 * sigma_k_trial)) * dtheta_plus_norm_sq
                
                # Check backtracking condition
                if P_plus <= rhs:
                    tau_new[i] = t
                    ell_k[i] = lcnt
                    break
                else:
                    # Backtracking condition failed
                    backtracking_failed_count[i] += 1
                    t *= rho_shrink
                    lcnt += 1
        
        # ---- Update ----
        for i in range(N):
            # Compute p_i again (same as in backtracking)
            s_diff_term = np.zeros(n)
            Ni = get_neighbors(i, neighbors_list)
            for j in Ni:
                s_diff_term += (1 + eta) * (s[i] - s[j]) - eta * (s_prev[i] - s_prev[j])
            
            grad_term = (1 + eta) * jacT_theta_i(i, x[i], theta[i]) \
                       - eta * jacT_theta_i(i, x_prev[i], theta_prev[i])
            
            p_i = s_diff_term + grad_term
            
            tau_i = tau_new[i]
            sigma_i = zeta_i[i] * tau_i

            x_next = prox_primal(x[i] - tau_i * (grad_f_i(x[i]) + p_i), tau_i)
            theta_next = proj_R_plus(theta[i] + sigma_i * g_i_of(i, x_next))
            s_next = s[i] + gamma * ((1 + eta) * x[i] - eta * x_prev[i])

            x_prev[i] = x[i]
            theta_prev[i] = theta[i]
            s_prev[i] = s[i]
            x[i] = x_next
            theta[i] = theta_next
            s[i] = s_next
            tau_list[i] = tau_i

        # Update eta based on backtracking according to formula:
        # eta_k = max_{i in N} {rho_i^(-ell_{i,k})}
        # Since rho_shrink is the same for all nodes, this becomes:
        # eta_k = max_{i in N} {(rho_shrink)^(-ell_{i,k})}
        eta_values = [rho_shrink ** (-ell_k[i]) for i in range(N)]
        eta = max(eta_values)
        
        # Note: gamma will be updated at the start of next iteration based on the new eta value

        # Backtracking statistics
        max_ell = max(ell_k) if ell_k else 0
        total_failed = sum(backtracking_failed_count)
        nodes_with_backtracking = sum(1 for ell in ell_k if ell > 0)
        
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
        
        all_vals = []
        for i in range(N):
            all_vals.extend(g_i_of(i, x_bar))
        all_vals = np.array(all_vals)
        
        max_viol = float(np.maximum(all_vals, 0).max()) if all_vals.size > 0 else 0.0
        avg_viol = float(np.maximum(all_vals, 0).mean()) if all_vals.size > 0 else 0.0
        obj = 0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar
        subopt = abs(obj - f_star) if f_star is not None else np.nan

        hist.append((obj, max_viol, cons_err, avg_viol, subopt))

        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | eta {eta:.3f}"
            if f_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)
            # Print backtracking statistics
            if total_failed > 0 or max_ell > 0:
                print(f"        Backtracking: max_ell={max_ell}, total_failed={total_failed}, nodes_with_bt={nodes_with_backtracking}")
                if verbose_every <= 50 or k == max_iter - 1:  # Show detailed info if verbose_every is small or last iteration
                    ell_str = " ".join([f"ell[{i}]={ell_k[i]}" for i in range(N) if ell_k[i] > 0])
                    if ell_str:
                        print(f"        ell_k (nonzero): {ell_str}")
                    if total_failed > 0:
                        failed_str = " ".join([f"failed[{i}]={backtracking_failed_count[i]}" for i in range(N) if backtracking_failed_count[i] > 0])
                        if failed_str:
                            print(f"        failed_count (nonzero): {failed_str}")

        if f_star is not None:
            if max(subopt, avg_viol) <= tol:
                break

    return x_bar, hist

# --------------------------
# Run main test
# --------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # Problem parameters
    n = 20
    m = 12
    N = 12
    E = 24
    
    # Algorithm parameters (matching dapd-qcqp.py)
    tau_init = 0.1  # Same as dapd-qcqp.py computed tau_initial
    gamma = None  # Will be computed from formula based on eta (ignored if provided)
    max_iter = 2000   # Same as dapd-qcqp.py
    c_alpha = 0.1
    c_beta = 0.1
    c_c = 0.1
    zeta = 1.0
    rho_shrink = 0.5
    delta = 0.1
    verbose_every = 10  # Output every 10 iterations
    initial_scale = 5.0
    seed = 456
    dapd_seed = 789
    
    # Generate problem
    rng = np.random.default_rng(seed)
    network_type = "small_world"
    neighbors_list = generate_small_world_network(N, E, seed=seed)
    
    # Generate a simple QCQP problem
    A0 = np.eye(n) * 0.1
    b0 = rng.standard_normal(n) * 0.01
    
    pernode_constraints = []
    for i in range(N):
        A_list_i = [np.eye(n) * 0.5]
        b_list_i = [rng.standard_normal(n) * 0.1]
        c_list_i = [-0.5]
        pernode_constraints.append((A_list_i, b_list_i, c_list_i))
    
    box_lo, box_hi = -5.0, 5.0
    
    print("Solving ground truth...")
    
    # Prepare per-node constraints
    A_list_all = []
    b_list_all = []
    c_list_all = []
    for i in range(N):
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        A_list_all.extend(A_list_i)
        b_list_all.extend(b_list_i)
        c_list_all.extend(c_list_i)
    
    x_star, f_star = solve_qcqp_ground_truth(
        A0, b0, A_list_all, b_list_all, c_list_all, 
        box_lo, box_hi, neighbors_list
    )
    
    print(f"Ground truth: f* = {f_star:.6f}")
    
    # Run D-APDB with backtracking
    print("\nRunning D-APDB with backtracking...")
    x_bar, hist = d_apdb_qcqp_merely_convex(
        A0, b0, pernode_constraints, box_lo, box_hi,
        N=N, max_iter=max_iter, seed=dapd_seed,
        c_alpha=c_alpha, c_beta=c_beta, c_c=c_c,
        zeta=zeta, tau_init=tau_init, gamma=gamma,
        rho_shrink=rho_shrink, delta=delta,
        f_star=f_star, verbose_every=verbose_every,
        initial_scale=initial_scale, neighbors_list=neighbors_list,
        initialization_mode="independent"
    )
    
    print(f"\nD-APDB result: ||x_bar - x*|| = {np.linalg.norm(x_bar - x_star):.6f}")
    print(f"f(x_bar) = {0.5 * x_bar @ (A0 @ x_bar) + b0 @ x_bar:.6f} | f* = {f_star:.6f}")
    
    # ---- Plotting ----
    iters = np.arange(len(hist))
    objs = [h[0] for h in hist]
    maxV = [h[1] for h in hist]
    cons = [h[2] for h in hist]
    avgV = [h[3] for h in hist]
    subopt = [h[4] for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Add overall title
    gamma_str = "auto" if gamma is None else f"{gamma:.4f}"
    fig.suptitle(f'D-APDB QCQP Convergence Results (N={N}, n={n}, tau_init={tau_init:.4f}, gamma={gamma_str})', 
                 fontsize=14, fontweight='bold')
    
    # 1. Objective function
    axes[0,0].plot(iters, objs, lw=2)
    axes[0,0].axhline(f_star, color='k', ls='--', alpha=0.5)
    axes[0,0].set_title('Objective Function: $f(\\bar{x}^k) = \\frac{1}{N}\\sum_{i=1}^N f_i(x_i^k)$')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Consensus error with formula
    axes[0,1].plot(iters, cons, lw=2)
    axes[0,1].set_title('Consensus Error: $\\frac{1}{N}\\sum_{i=1}^N \\|x_i^k - \\bar{x}^k\\|$')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Constraint violations with formula
    axes[1,0].semilogy(iters, [max(v, 1e-10) for v in maxV], lw=2, label='max viol')
    axes[1,0].set_title('Constraint Violations: $\\max_{i \\in N} \\{g_i(\\bar{x}^k)_+\\}$')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Suboptimality with formula
    if not all(np.isnan(subopt)):
        axes[1,1].plot(iters, subopt, lw=2)
        axes[1,1].set_title('Absolute Suboptimality: $|f(\\bar{x}^k) - f^*|$')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].axis('off')

    # Save figure
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if network_type == "small_world":
        network_info = f"SW_N{N}_E{E}"
    else:
        network_info = f"UNK_N{N}"
    
    # Use gamma_str for filename as well
    gamma_filename = "auto" if gamma is None else f"{gamma:.3f}"
    filename = f"qcqp-experiments/qcqp_dapdb_convergence_{network_info}_n{n}_initindependent_tau{tau_init:.3f}_gamma{gamma_filename}_seed{seed}_{dapd_seed}_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as: {filename}")
    plt.show()
