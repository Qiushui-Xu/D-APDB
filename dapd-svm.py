import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import datetime

# --------------------------
# SVM Data Generation
# --------------------------

def generate_svm_data(s=900, n_features=2, seed=42):
    """
    生成SVM问题的数据
    
    根据论文描述：
    - 生成s=900个样本点
    - 特征向量x_t从二维多元高斯分布生成
    - 协方差矩阵Σ = [[1, 0], [0, 2]]
    - 均值向量为m1=[-1,-1]^T或m2=[1,1]^T，等概率选择
    - 标签y_t ∈ {-1, +1}
    
    Parameters:
    -----------
    s : int
        总样本数
    n_features : int
        特征维度（论文中为2）
    seed : int
        随机种子
        
    Returns:
    --------
    X : np.ndarray, shape (s, n_features)
        特征矩阵
    y : np.ndarray, shape (s,)
        标签向量
    """
    rng = np.random.default_rng(seed)
    
    # 协方差矩阵
    cov_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    
    # 均值向量
    mean1 = np.array([-1.0, -1.0])
    mean2 = np.array([1.0, 1.0])
    
    # 生成数据
    X = np.zeros((s, n_features))
    y = np.zeros(s)
    
    for i in range(s):
        # 等概率选择均值
        if rng.random() < 0.5:
            mean = mean1
            label = -1
        else:
            mean = mean2
            label = 1
            
        # 生成样本
        X[i] = rng.multivariate_normal(mean, cov_matrix)
        y[i] = label
    
    return X, y

def partition_data(X, y, N, test_ratio=0.33, seed=42):
    """
    将数据分割为训练集和测试集，并将训练集分配给N个节点
    
    Parameters:
    -----------
    X : np.ndarray, shape (s, n_features)
        特征矩阵
    y : np.ndarray, shape (s,)
        标签向量
    N : int
        节点数量
    test_ratio : float
        测试集比例
    seed : int
        随机种子
        
    Returns:
    --------
    X_train : np.ndarray
        训练特征矩阵
    y_train : np.ndarray
        训练标签向量
    X_test : np.ndarray
        测试特征矩阵
    y_test : np.ndarray
        测试标签向量
    data_partitions : list
        每个节点的数据分区 [(X_i, y_i), ...]
    """
    rng = np.random.default_rng(seed)
    s = len(X)
    
    # 分割训练集和测试集
    indices = np.arange(s)
    rng.shuffle(indices)
    
    n_test = int(s * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    # 将训练数据分配给N个节点
    n_train = len(train_indices)
    data_partitions = []
    
    for i in range(N):
        # 计算每个节点的数据范围
        start_idx = i * n_train // N
        end_idx = (i + 1) * n_train // N
        
        X_i = X_train[start_idx:end_idx]
        y_i = y_train[start_idx:end_idx]
        
        data_partitions.append((X_i, y_i))
    
    return X_train, y_train, X_test, y_test, data_partitions

# --------------------------
# Network Generation
# --------------------------

def generate_small_world_network(N, E, seed=42):
    """
    生成小世界网络
    """
    rng = np.random.default_rng(seed)
    
    # 初始化邻接表
    neighbors = [[] for _ in range(N)]
    
    # 步骤1：创建随机环（N条边）
    cycle_order = rng.permutation(N)
    for i in range(N):
        current = cycle_order[i]
        next_node = cycle_order[(i + 1) % N]
        neighbors[current].append(next_node)
        neighbors[next_node].append(current)
    
    # 步骤2：随机添加剩余的E-N条边
    remaining_edges = E - N
    added_edges = 0
    max_attempts = remaining_edges * 10
    attempts = 0
    
    while added_edges < remaining_edges and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(N, 2, replace=False)
        
        if j not in neighbors[i]:
            neighbors[i].append(j)
            neighbors[j].append(i)
            added_edges += 1
    
    return neighbors

def generate_fully_connected_network(N):
    """
    生成全连接网络
    """
    neighbors = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                neighbors[i].append(j)
    
    return neighbors

# --------------------------
# Ground Truth (Centralized SVM)
# --------------------------

def solve_svm_ground_truth(X_train, y_train, C, neighbors_list=None, data_partitions=None, N=None):
    """
    使用集中式方法求解分布式SVM问题作为ground truth
    
    分布式SVM目标函数：
    min_{w,b,ξ} (1/2)∑_{i∈N}||w_i||² + N*C * ∑_{i∈N} ∑_{t∈S_t} ξ_t
    s.t. y_t(w_i^T x_t + b_i) ≥ 1 - ξ_t, ξ_t ≥ 0, t ∈ S_t, i ∈ N
         w_i = w_j, b_i = b_j, (i,j) ∈ E
    
    Parameters:
    -----------
    X_train : np.ndarray, shape (n_train, n_features)
        训练特征矩阵
    y_train : np.ndarray, shape (n_train,)
        训练标签向量
    C : float
        正则化参数
    neighbors_list : list, optional
        网络拓扑结构
    data_partitions : list, optional
        数据分区，如果提供则使用分布式目标函数
    N : int, optional
        节点数量
        
    Returns:
    --------
    w_star : np.ndarray, shape (n_features,)
        最优权重向量
    b_star : float
        最优偏置
    f_star : float
        最优目标函数值
    """
    n_features = X_train.shape[1]
    n_train = len(X_train)
    
    if data_partitions is not None and N is not None:
        # 使用分布式SVM目标函数
        # 为每个节点创建变量
        w_vars = [cp.Variable(n_features) for _ in range(N)]
        b_vars = [cp.Variable() for _ in range(N)]
        
        # 为每个节点的数据创建松弛变量
        xi_vars = []
        for i in range(N):
            X_i, y_i = data_partitions[i]
            n_i = len(X_i)
            xi_vars.append(cp.Variable(n_i))
        
        # 分布式目标函数：(1/2)∑_{i∈N}||w_i||² + N*C * ∑_{i∈N} ∑_{t∈S_t} ξ_t
        objective = 0
        for i in range(N):
            objective += 0.5 * cp.sum_squares(w_vars[i])  # (1/2)||w_i||²
            objective += N * C * cp.sum(xi_vars[i])       # N*C * ∑_{t∈S_t} ξ_t
        
        objective = cp.Minimize(objective)
        
        # 约束条件
        constraints = []
        
        # 软间隔约束：y_t(w_i^T x_t + b_i) ≥ 1 - ξ_t, ξ_t ≥ 0
        for i in range(N):
            X_i, y_i = data_partitions[i]
            n_i = len(X_i)
            for j in range(n_i):
                constraints.append(y_i[j] * (X_i[j] @ w_vars[i] + b_vars[i]) >= 1 - xi_vars[i][j])
                constraints.append(xi_vars[i][j] >= 0)
        
        # 一致性约束：w_i = w_j, b_i = b_j for all (i,j) ∈ E
        if neighbors_list is not None:
            for i in range(N):
                for j in neighbors_list[i]:
                    if i < j:  # 避免重复约束
                        constraints.append(w_vars[i] == w_vars[j])
                        constraints.append(b_vars[i] == b_vars[j])
        
        # 求解
        prob = cp.Problem(objective, constraints)
        
    else:
        # 使用标准单节点SVM目标函数（向后兼容）
        w = cp.Variable(n_features)
        b = cp.Variable()
        xi = cp.Variable(n_train)
        
        # 目标函数：min (1/2)||w||² + C*sum(ξ)
        objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
        
        # 约束条件
        constraints = []
        for i in range(n_train):
            constraints.append(y_train[i] * (X_train[i] @ w + b) >= 1 - xi[i])
            constraints.append(xi[i] >= 0)
        
        # 求解
        prob = cp.Problem(objective, constraints)
    
    # 尝试不同的求解器
    solvers = [cp.ECOS, cp.SCS, cp.MOSEK]
    solver_names = ["ECOS", "SCS", "MOSEK"]
    
    for solver_name, solver in zip(solver_names, solvers):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                if data_partitions is not None and N is not None:
                    # 分布式情况：返回第一个节点的解（所有节点应该相同）
                    w_star = w_vars[0].value
                    b_star = b_vars[0].value
                    f_star = prob.value
                    # 计算所有松弛变量的最优值
                    xi_star = np.concatenate([xi_vars[i].value for i in range(N)])
                else:
                    # 单节点情况
                    w_star = w.value
                    b_star = b.value
                    f_star = prob.value
                    xi_star = xi.value
                return w_star, b_star, xi_star, f_star
        except Exception as e:
            print(f"Solver {solver_name} failed: {e}")
            continue
    
    raise RuntimeError("All solvers failed to solve the SVM problem")

# --------------------------
# Distributed APD for SVM
# --------------------------

def d_apd_svm(X_train, y_train, data_partitions, C, N, max_iter=2000, seed=0,
              c_alpha=0.1, c_beta=0.1, c_c=0.1, zeta=1.0, tau=0.01, gamma=None,
              verbose_every=200, initial_scale=1.0, f_star=None, tol=1e-8,
              neighbors_list=None, initialization_mode="independent"):
    """
    分布式APD算法求解SVM问题
    
    Parameters:
    -----------
    X_train : np.ndarray
        训练特征矩阵
    y_train : np.ndarray
        训练标签向量
    data_partitions : list
        每个节点的数据分区
    C : float
        正则化参数
    N : int
        节点数量
    max_iter : int
        最大迭代次数
    seed : int
        随机种子
    c_alpha, c_beta, c_c : float
        算法参数
    zeta : float
        对偶步长参数
    tau : float
        原始步长参数
    gamma : float, optional
        一致性参数，如果为None则自动计算
    verbose_every : int
        打印频率
    initial_scale : float
        初始值缩放因子
    f_star : float, optional
        最优目标函数值
    tol : float
        收敛容差
    neighbors_list : list, optional
        网络拓扑结构
    initialization_mode : str
        初始化模式："connected" 或 "independent"
        
    Returns:
    --------
    w_bar : np.ndarray
        平均权重向量
    b_bar : float
        平均偏置
    hist : list
        收敛历史
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    
    # 使用提供的网络拓扑或生成全连接网络
    if neighbors_list is None:
        neighbors_list = generate_fully_connected_network(N)
    d_max = max(len(neighbors_list[i]) for i in range(N))
    
    # 局部目标函数：f_i(w_i, b_i, xi_i) = 0.5 * ||w_i||² + N*C * Σ_{t∈S_t} ξ_t
    def grad_f_i(w_i, b_i, xi_i, i):
        X_i, y_i = data_partitions[i]
        n_i = len(X_i)
        
        # 梯度计算
        grad_w = w_i  # 来自正则化项：∂f/∂w_i = w_i
        grad_b = 0.0
        grad_xi = np.full(n_i, N * C)  # 来自松弛变量项：∂f/∂ξ_t = N*C
        
        return grad_w, grad_b, grad_xi
    
    # 局部约束函数：g_i(w_i, b_i, ξ_i) = [y_t(w_i^T x_t + b_i) + ξ_t - 1, -ξ_t]
    def g_i_of(i, w_i, b_i, xi_i):
        X_i, y_i = data_partitions[i]
        n_i = len(X_i)
        
        constraints = []
        for j in range(n_i):
            # 软间隔约束：y_t(w_i^T x_t + b_i) + ξ_t - 1 ≥ 0
            margin = y_i[j] * (X_i[j] @ w_i + b_i)
            constraints.append(margin + xi_i[j] - 1)
            # 非负约束：ξ_t ≥ 0
            constraints.append(-xi_i[j])
        
        return np.array(constraints)
    
    # 约束函数的雅可比矩阵转置乘以对偶变量
    def jacT_theta_i(i, w_i, b_i, xi_i, theta_i):
        X_i, y_i = data_partitions[i]
        n_i = len(X_i)
        
        grad_w = np.zeros(n_features)
        grad_b = 0.0
        grad_xi = np.zeros(n_i)
        
        for j in range(n_i):
            # 对软间隔约束的贡献
            grad_w += theta_i[2*j] * (y_i[j] * X_i[j])
            grad_b += theta_i[2*j] * y_i[j]
            grad_xi[j] += theta_i[2*j]
            
            # 对非负约束的贡献
            grad_xi[j] -= theta_i[2*j + 1]
        
        return grad_w, grad_b, grad_xi
    
    # 对偶投影到非负象限
    def proj_R_plus(theta):
        return np.maximum(theta, 0.0)
    
    # 初始化
    if initialization_mode == "connected":
        # 连通模式：有边的节点具有相同的初始值
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
        
        w = []
        b = []
        xi = []
        for group in node_groups:
            w_init_group = initial_scale * rng.standard_normal(n_features)
            b_init_group = initial_scale * rng.standard_normal()
            for node in group:
                X_i, y_i = data_partitions[node]
                n_i = len(X_i)
                w.append(w_init_group.copy())
                b.append(b_init_group)
                xi.append(np.zeros(n_i))
        
        if verbose_every:
            print(f"Connected initialization mode:")
            print(f"  Number of connected components: {len(node_groups)}")
            for i, group in enumerate(node_groups):
                print(f"  Component {i}: nodes {group}")
                
    elif initialization_mode == "independent":
        # 独立模式：所有节点具有独立的初始值
        w = []
        b = []
        xi = []
        for i in range(N):
            w.append(initial_scale * rng.standard_normal(n_features))
            b.append(initial_scale * rng.standard_normal())
            X_i, y_i = data_partitions[i]
            n_i = len(X_i)
            xi.append(np.zeros(n_i))
        
        if verbose_every:
            print(f"Independent initialization mode:")
            print(f"  All {N} nodes initialized independently")
            
    elif initialization_mode == "zero":
        # 零初始化模式：所有节点从零开始
        w = []
        b = []
        xi = []
        for i in range(N):
            w.append(np.zeros(n_features))
            b.append(0.0)
            X_i, y_i = data_partitions[i]
            n_i = len(X_i)
            xi.append(np.zeros(n_i))
        
        if verbose_every:
            print(f"Zero initialization mode:")
            print(f"  All {N} nodes initialized to zero")
            
    elif initialization_mode == "data_based":
        # 基于数据的初始化模式：根据数据统计特性初始化
        w = []
        b = []
        xi = []
        
        # 计算全局数据统计
        all_X = np.vstack([X_i for X_i, _ in data_partitions])
        all_y = np.hstack([y_i for _, y_i in data_partitions])
        
        # 计算类别中心
        pos_mask = all_y == 1
        neg_mask = all_y == -1
        
        if np.any(pos_mask) and np.any(neg_mask):
            pos_center = np.mean(all_X[pos_mask], axis=0)
            neg_center = np.mean(all_X[neg_mask], axis=0)
            
            # 初始化权重为连接两个类别中心的向量
            w_init = pos_center - neg_center
            # 归一化权重
            w_norm = np.linalg.norm(w_init)
            if w_norm > 0:
                w_init = w_init / w_norm * initial_scale
            
            # 初始化偏置为两个类别中心的中点
            b_init = -np.dot(w_init, (pos_center + neg_center) / 2)
        else:
            # 如果只有一个类别，使用零初始化
            w_init = np.zeros(n_features)
            b_init = 0.0
        
        for i in range(N):
            w.append(w_init.copy())
            b.append(b_init)
            X_i, y_i = data_partitions[i]
            n_i = len(X_i)
            xi.append(np.zeros(n_i))
        
        if verbose_every:
            print(f"Data-based initialization mode:")
            print(f"  w_init = {w_init}")
            print(f"  b_init = {b_init:.6f}")
            print(f"  All {N} nodes initialized with data-based values")
            
    else:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}. Use 'connected', 'independent', 'zero', or 'data_based'")
    
    # 保存前一步的值
    w_prev = [w_i.copy() for w_i in w]
    b_prev = [b_i for b_i in b]
    xi_prev = [xi_i.copy() for xi_i in xi]
    
    # 参数设置
    tau_list = [tau] * N
    zeta_i = [zeta] * N
    sigma0_max = max(zeta_i[i] * tau_list[i] for i in range(N))
    eta = 1.0
    
    def gamma_k():
        return 1.0 / (2.0 * d_max * sigma0_max * N * ((2.0 / c_alpha) + (eta / c_c)))
    
    if gamma is None:
        gamma = gamma_k()
    
    # 初始化对偶变量
    theta = []
    theta_prev = []
    for i in range(N):
        X_i, y_i = data_partitions[i]
        n_i = len(X_i)
        theta.append(np.zeros(2 * n_i))  # 每个样本有两个约束
        theta_prev.append(np.zeros(2 * n_i))
    
    # 初始化一致性变量
    s_w = [np.zeros(n_features) for _ in range(N)]
    s_b = [0.0 for _ in range(N)]
    s_w_prev = [np.zeros(n_features) for _ in range(N)]
    s_b_prev = [0.0 for _ in range(N)]
    
    hist = []
    
    for k in range(max_iter):
        # 标准APD更新
        for i in range(N):
            # 计算p_i
            # 一致性项
            s_w_diff = np.zeros(n_features)
            s_b_diff = 0.0
            Ni = neighbors_list[i]
            for j in Ni:
                s_w_diff += (1 + eta) * (s_w[i] - s_w[j]) - eta * (s_w_prev[i] - s_w_prev[j])
                s_b_diff += (1 + eta) * (s_b[i] - s_b[j]) - eta * (s_b_prev[i] - s_b_prev[j])
            
            # 梯度项
            grad_w_term, grad_b_term, grad_xi_term = jacT_theta_i(i, w[i], b[i], xi[i], theta[i])
            grad_w_term_prev, grad_b_term_prev, grad_xi_term_prev = jacT_theta_i(i, w_prev[i], b_prev[i], xi_prev[i], theta_prev[i])
            
            p_w = s_w_diff + (1 + eta) * grad_w_term - eta * grad_w_term_prev
            p_b = s_b_diff + (1 + eta) * grad_b_term - eta * grad_b_term_prev
            p_xi = (1 + eta) * grad_xi_term - eta * grad_xi_term_prev
            
            # 目标函数梯度
            grad_w_f, grad_b_f, grad_xi_f = grad_f_i(w[i], b[i], xi[i], i)
            
            # 更新原始变量
            tau_i = tau_list[i]
            sigma_i = zeta_i[i] * tau_i
            
            w_next = w[i] - tau_i * (grad_w_f + p_w)
            b_next = b[i] - tau_i * (grad_b_f + p_b)
            
            # 对于松弛变量，使用约束违反情况来更新
            # 而不是简单的梯度下降
            X_i, y_i = data_partitions[i]
            margins = y_i * (X_i @ w_next + b_next)
            xi_next = np.maximum(0, 1 - margins)  # 根据约束违反情况计算xi
            
            # 更新对偶变量
            theta_next = proj_R_plus(theta[i] + sigma_i * g_i_of(i, w_next, b_next, xi_next))
            
            # 更新一致性变量
            s_w_next = s_w[i] + gamma * ((1 + eta) * w[i] - eta * w_prev[i])
            s_b_next = s_b[i] + gamma * ((1 + eta) * b[i] - eta * b_prev[i])
            
            # 保存前一步的值
            w_prev[i] = w[i]
            b_prev[i] = b[i]
            xi_prev[i] = xi[i]
            theta_prev[i] = theta[i]
            s_w_prev[i] = s_w[i]
            s_b_prev[i] = s_b[i]
            
            # 更新当前值
            w[i] = w_next
            b[i] = b_next
            xi[i] = xi_next
            theta[i] = theta_next
            s_w[i] = s_w_next
            s_b[i] = s_b_next
        
        # 计算网络平均值
        w_bar = sum(w) / N
        b_bar = sum(b) / N
        
        # 计算xi_bar（所有节点松弛变量的平均值）
        # 在分布式SVM中，每个节点的xi对应不同的数据样本
        # xi_bar应该是所有松弛变量的平均值，用于分析整体约束违反情况
        all_xi = np.concatenate(xi)
        xi_bar = all_xi  # 直接使用所有松弛变量的拼接，因为每个样本对应一个松弛变量
        
        # 计算一致性误差：max_{(i,j) ∈ E} ||[w_i^T b_i]^T - [w_j^T b_j]^T||
        max_consensus_error = 0.0
        for i in range(N):
            for j in neighbors_list[i]:
                if j > i:  # 避免重复计算
                    # 计算 [w_i^T b_i]^T - [w_j^T b_j]^T 的范数
                    w_diff = w[i] - w[j]
                    b_diff = b[i] - b[j]
                    # 拼接成 [w_diff^T b_diff]^T 并计算范数
                    combined_diff = np.concatenate([w_diff, [b_diff]])
                    consensus_error = np.linalg.norm(combined_diff)
                    max_consensus_error = max(max_consensus_error, consensus_error)
        cons_err = max_consensus_error
        
        # 计算约束违反
        all_violations = []
        for i in range(N):
            violations = g_i_of(i, w_bar, b_bar, xi[i])
            all_violations.extend(violations)
        all_violations = np.array(all_violations)
        
        max_viol = float(np.maximum(all_violations, 0).max()) if all_violations.size > 0 else 0.0
        avg_viol = float(np.maximum(all_violations, 0).mean()) if all_violations.size > 0 else 0.0
        
        # 计算目标函数值
        # 分布式SVM目标函数：min (1/2)∑_{i∈N}||w_i||² + N*C * ∑_{i∈N} ∑_{t∈S_t} ξ_t
        # 当所有节点收敛到相同解时：= (1/2)N||w||² + N*C * ∑_{t∈S_t} ξ_t
        # 使用实际的松弛变量xi_bar
        obj = 0.5 * N * np.linalg.norm(w_bar)**2 + N * C * np.sum(xi_bar)
        subopt = abs(obj - f_star) if f_star is not None else np.nan
        
        hist.append((obj, max_viol, cons_err, avg_viol, subopt))
        
        if verbose_every and (k % verbose_every == 0 or k == max_iter - 1):
            msg = f"iter {k:5d} | obj {obj:.6e} | maxV {max_viol:.2e} | avgV {avg_viol:.2e} | cons {cons_err:.2e} | eta {eta:.3f}"
            if f_star is not None:
                msg += f" | abs subopt {subopt:.2e}"
            print(msg)
        
        # 可选停止条件
        if f_star is not None:
            if max(subopt, avg_viol) <= tol:
                break
    
    return w_bar, b_bar, xi_bar, hist

# --------------------------
# Visualization
# --------------------------

def plot_svm_results(X_train, y_train, X_test, y_test, w_bar, b_bar, w_star, b_star, 
                     hist, f_star, C, N, tau, gamma, network_type, seed, timestamp):
    """
    绘制SVM结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 添加总标题
    fig.suptitle(f'D-APD SVM Results (N={N}, C={C}, tau={tau:.4f}, gamma={gamma:.4f})', 
                 fontsize=16, fontweight='bold')
    
    # 提取收敛历史数据
    iters = np.arange(len(hist))
    objs = [h[0] for h in hist]
    maxV = [h[1] for h in hist]  # feasibility violation
    cons = [h[2] for h in hist]  # consensus violation
    relsub = [h[4] for h in hist]  # suboptimality
    
    # 1. Consensus Violation
    axes[0,0].semilogy(iters, cons, lw=2, color='blue')
    axes[0,0].set_title('Consensus Violation\nmax_{(i,j)∈E} ||[w_i^T b_i]^T - [w_j^T b_j]^T||')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Consensus Error')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Feasibility Violation
    axes[0,1].semilogy(iters, maxV, lw=2, color='red')
    axes[0,1].set_title('Feasibility Violation\nmax_{i,t} max(0, 1-y_t(w_i^T x_t + b_i))')
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Max Constraint Violation')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Suboptimality
    if not np.isnan(relsub).all():
        axes[1,0].semilogy(iters, relsub, lw=2, color='green')
        axes[1,0].set_title('Suboptimality\n|f(w_bar, b_bar) - f*|')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('|f(x) - f*|')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].axis('off')
    
    # 4. Classification Comparison - Ground Truth vs D-APD
    plot_classification_comparison(axes[1,1], X_train, y_train, w_star, b_star, w_bar, b_bar)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f"svm-experiments/svm_convergence_{network_type}_N{N}_C{C}_tau{tau:.3f}_gamma{gamma:.3f}_seed{seed}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as: {filename}")
    plt.show()

def plot_classification_comparison(ax, X, y, w_star, b_star, w_bar, b_bar):
    """
    绘制Ground Truth和D-APD的分类对比图
    """
    # 绘制数据点
    colors = ['red' if label == -1 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=20, label='Data points')
    
    # 绘制决策边界
    if len(w_star) == 2 and len(w_bar) == 2:  # 二维情况
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx = np.linspace(x_min, x_max, 100)
        yy = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(xx, yy)
        
        # Ground Truth决策函数值
        Z_star = w_star[0] * XX + w_star[1] * YY + b_star
        
        # D-APD决策函数值
        Z_bar = w_bar[0] * XX + w_bar[1] * YY + b_bar
        
        # 绘制Ground Truth决策边界（实线）
        contour_star = ax.contour(XX, YY, Z_star, levels=[0], colors=['black'], 
                                 linestyles=['-'], linewidths=2, alpha=0.8)
        
        # 绘制D-APD决策边界（虚线）
        contour_bar = ax.contour(XX, YY, Z_bar, levels=[0], colors=['orange'], 
                                linestyles=['--'], linewidths=2, alpha=0.8)
        
        # 绘制Ground Truth间隔边界
        ax.contour(XX, YY, Z_star, levels=[-1, 1], colors=['black'], 
                  linestyles=[':'], linewidths=1, alpha=0.5)
        
        # 绘制D-APD间隔边界
        ax.contour(XX, YY, Z_bar, levels=[-1, 1], colors=['orange'], 
                  linestyles=[':'], linewidths=1, alpha=0.5)
        
        # 创建图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Ground Truth'),
            Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='D-APD')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Classification Comparison\nGround Truth vs D-APD\nf(w,b) = (1/2)N||w||² + N*C*∑_t ξ_t')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

def plot_svm_classification(ax, X, y, w, b, title):
    """
    绘制SVM分类结果
    """
    # 绘制数据点
    colors = ['red' if label == -1 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=20)
    
    # 绘制决策边界
    if len(w) == 2:  # 二维情况
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx = np.linspace(x_min, x_max, 100)
        yy = np.linspace(y_min, y_max, 100)
        XX, YY = np.meshgrid(xx, yy)
        
        # 决策函数值
        Z = w[0] * XX + w[1] * YY + b
        
        # 绘制决策边界和间隔
        ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
                  linestyles=['--', '-', '--'], alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

# --------------------------
# Main execution
# --------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # 配置参数
    main_seed = 42
    dapd_seed = 789
    
    # 问题参数
    s = 900  # 总样本数
    n_features = 2  # 特征维度
    N = 12  # 节点数量
    C = 2.0  # 正则化参数
    E = 24  # 边数量
    
    # 算法参数
    tau = 0.001
    gamma = 0.001
    max_iter = 5000
    
    # 网络类型
    network_type = "small_world"  # "fully_connected" 或 "small_world"
    
    # 初始化模式
    initialization_mode = "independent"  # "connected", "independent", "zero", 或 "data_based"
    
    print("="*60)
    print("Distributed APD for Linear SVM")
    print("="*60)
    print(f"Problem parameters:")
    print(f"  Total samples: {s}")
    print(f"  Features: {n_features}")
    print(f"  Nodes: {N}")
    print(f"  Regularization parameter C: {C}")
    print(f"  Network type: {network_type}")
    print(f"  Initialization mode: {initialization_mode}")
    
    # 生成数据
    print("\nGenerating SVM data...")
    X, y = generate_svm_data(s=s, n_features=n_features, seed=main_seed)
    X_train, y_train, X_test, y_test, data_partitions = partition_data(
        X, y, N, test_ratio=0.33, seed=main_seed)
    
    print(f"Data generated:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Samples per node: {[len(X_i) for X_i, _ in data_partitions]}")
    
    # 生成网络拓扑
    if network_type == "fully_connected":
        neighbors_list = generate_fully_connected_network(N)
    elif network_type == "small_world":
        neighbors_list = generate_small_world_network(N, E, seed=dapd_seed)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    print(f"\nNetwork topology:")
    for i in range(N):
        print(f"  Node {i}: neighbors = {neighbors_list[i]}")
    print(f"Maximum degree: {max(len(neighbors_list[i]) for i in range(N))}")
    
    # 求解ground truth
    print("\nSolving ground truth...")
    w_star, b_star, xi_star, f_star = solve_svm_ground_truth(X_train, y_train, C, neighbors_list, data_partitions, N)
    print(f"Ground truth:")
    print(f"  w* = {w_star}")
    print(f"  b* = {b_star:.6f}")
    print(f"  xi* stats: min={np.min(xi_star):.6f}, max={np.max(xi_star):.6f}, mean={np.mean(xi_star):.6f}")
    print(f"  f* = {f_star:.6f}")
    
    # 运行分布式算法
    print("\nRunning D-APD...")
    w_bar, b_bar, xi_bar, hist = d_apd_svm(
        X_train, y_train, data_partitions, C, N, max_iter=max_iter, seed=dapd_seed,
        tau=tau, gamma=gamma, verbose_every=200, initial_scale=1.0,
        f_star=f_star, tol=1e-8, neighbors_list=neighbors_list,
        initialization_mode=initialization_mode
    )
    
    # 结果比较
    print("\nResults comparison:")
    print(f"D-APD result:")
    print(f"  w_bar = {w_bar}")
    print(f"  b_bar = {b_bar:.6f}")
    print(f"  xi_bar stats: min={np.min(xi_bar):.6f}, max={np.max(xi_bar):.6f}, mean={np.mean(xi_bar):.6f}")
    print(f"  ||w_bar - w*|| = {np.linalg.norm(w_bar - w_star):.6f}")
    print(f"  |b_bar - b*| = {abs(b_bar - b_star):.6f}")
    print(f"  ||xi_bar - xi*|| = {np.linalg.norm(xi_bar - xi_star):.6f}")
    
    # 计算最终目标函数值
    # 使用与集中式相同的目标函数：(1/2)N||w||² + N*C * ∑_{t=1}^{n} ξ_t
    # 使用实际的松弛变量xi_bar
    final_obj = 0.5 * N * np.linalg.norm(w_bar)**2 + N * C * np.sum(xi_bar)
    print(f"  Final objective: {final_obj:.6f}")
    print(f"  Optimal objective: {f_star:.6f}")
    print(f"  Absolute suboptimality: {abs(final_obj - f_star):.6f}")
    
    # 绘制结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_svm_results(X_train, y_train, X_test, y_test, w_bar, b_bar, w_star, b_star,
                     hist, f_star, C, N, tau, gamma, network_type, main_seed, timestamp)
