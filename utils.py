"""
Utility functions for distributed optimization algorithms.
This module contains shared functions used by dapd.py, dapdb.py, and aldo.py.
"""

import numpy as np
import cvxpy as cp


# --------------------------
# Lipschitz constant computation
# --------------------------

def compute_lipschitz_constant(A, b=None):
    """
    Compute the Lipschitz constant for quadratic function f(x) = 0.5 * x^T * A * x + b^T * x
    
    For quadratic function f(x) = 0.5 * x^T * A * x + b^T * x,
    the gradient is ∇f(x) = A * x + b
    
    If A is symmetric, then the Lipschitz constant L = ||A||_2 = λ_max(A)
    where λ_max is the maximum eigenvalue of A
    
    Parameters:
    -----------
    A : np.ndarray
        Quadratic coefficient matrix (n x n)
    b : np.ndarray, optional
        Linear coefficient vector (n,)
        
    Returns:
    --------
    L : float
        Lipschitz constant
    """
    # Ensure A is symmetric
    A_sym = 0.5 * (A + A.T)
    
    # Compute maximum eigenvalue
    eigenvals = np.linalg.eigvals(A_sym)
    L = np.max(np.abs(eigenvals))
    
    return L


def compute_constraint_lipschitz_constants(A_list, b_list):
    """
    Compute Lipschitz constants for all constraint functions
    
    For constraint g_j(x) = 0.5 * x^T * A_j * x + b_j^T * x + c_j
    the gradient is ∇g_j(x) = A_j * x + b_j
    
    Parameters:
    -----------
    A_list : list of np.ndarray
        List of quadratic coefficient matrices for constraints
    b_list : list of np.ndarray  
        List of linear coefficient vectors for constraints
        
    Returns:
    --------
    L_list : list of float
        List of Lipschitz constants for each constraint
    """
    L_list = []
    for A_j, b_j in zip(A_list, b_list):
        L_j = compute_lipschitz_constant(A_j, b_j)
        L_list.append(L_j)
    
    return L_list


# --------------------------
# Jacobian bound computation
# --------------------------

def compute_jacobian_bound_for_node_i(A_list_i, b_list_i, box_lo, box_hi):
    """
    Compute the bound C_{g_i} for the Jacobian of constraint function g_i for node i
    
    For node i's constraint function g_i(x) = [g_{i,1}(x), ..., g_{i,m_i}(x)]^T,
    where g_{i,j}(x) = 0.5 * x^T * A_{i,j} * x + b_{i,j}^T * x + c_{i,j}
    
    The Jacobian matrix is:
    J g_i(x) = [A_{i,1}*x + b_{i,1}, A_{i,2}*x + b_{i,2}, ..., A_{i,m_i}*x + b_{i,m_i}]
    
    This is an n × m_i matrix, where each column is A_{i,j}*x + b_{i,j}
    
    We need to find C_{g_i} > 0 such that for all x ∈ dom φ_i (box constraint),
    we have ||J g_i(x)|| ≤ C_{g_i}
    
    Here we use the Frobenius norm (or 2-norm) of the matrix:
    ||J g_i(x)||_F^2 = Σ_j ||A_{i,j}*x + b_{i,j}||^2
    
    For box constraint box_lo ≤ x ≤ box_hi, we can find an upper bound:
    ||A_{i,j}*x + b_{i,j}|| ≤ ||A_{i,j}||_2 * ||x|| + ||b_{i,j}||
    ≤ ||A_{i,j}||_2 * sqrt(n) * max(|box_lo|, |box_hi|) + ||b_{i,j}||
    
    Therefore:
    C_{g_i} = sqrt(Σ_j (||A_{i,j}||_2 * sqrt(n) * max(|box_lo|, |box_hi|) + ||b_{i,j}||)^2)
    
    Parameters:
    -----------
    A_list_i : list of np.ndarray
        List of quadratic coefficient matrices for node i's constraints
    b_list_i : list of np.ndarray
        List of linear coefficient vectors for node i's constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
        
    Returns:
    --------
    C_g_i : float
        Jacobian bound C_{g_i} for node i
    """
    n = A_list_i[0].shape[0] if A_list_i else 0
    m_i = len(A_list_i)
    
    if m_i == 0:
        return 0.0
    
    # Compute the diameter of the box (or use default if unbounded)
    if box_lo is None or box_hi is None:
        # For unbounded case, use a reasonable default bound
        box_diameter = 10.0  # Default bound
    else:
        box_diameter = max(abs(box_lo), abs(box_hi))
    max_norm_x = np.sqrt(n) * box_diameter
    
    # Compute the maximum norm for each column
    max_col_norms = []
    for A_ij, b_ij in zip(A_list_i, b_list_i):
        # Compute the spectral norm (2-norm) of A_{i,j}
        A_ij_norm = np.linalg.norm(A_ij, ord=2)
        
        # Compute the 2-norm of b_{i,j}
        b_ij_norm = np.linalg.norm(b_ij, ord=2)
        
        # Compute the upper bound of ||A_{i,j}*x + b_{i,j}||
        max_col_norm = A_ij_norm * max_norm_x + b_ij_norm
        max_col_norms.append(max_col_norm)
    
    # Use the upper bound of the Frobenius norm
    # ||J g_i(x)||_F ≤ sqrt(Σ_j (max_col_norm_j)^2)
    C_g_i_frobenius = np.sqrt(sum(norm**2 for norm in max_col_norms))
    
    # For the 2-norm (operator norm), we have:
    # ||J g_i(x)||_2 ≤ ||J g_i(x)||_F ≤ sqrt(m_i) * max_j(max_col_norm_j)
    C_g_i_operator = max(max_col_norms) * np.sqrt(m_i) if m_i > 0 else 0.0
    
    # Return the Frobenius norm bound
    C_g_i = C_g_i_frobenius
    
    return C_g_i


def compute_constraint_gradient_bound(A_list, b_list, box_lo, box_hi):
    """
    Compute the bound C_{g_i} for constraint function gradient
    
    For quadratic constraint g_i(x) = 0.5 * x^T * A_i * x + b_i^T * x + c_i
    the gradient is ∇g_i(x) = A_i * x + b_i
    
    Under box constraint box_lo <= x <= box_hi, the bound for gradient is:
    ||∇g_i(x)|| <= ||A_i * x + b_i|| <= ||A_i|| * ||x|| + ||b_i||
    
    Since ||x|| <= sqrt(n) * max(|box_lo|, |box_hi|)
    we have C_{g_i} = ||A_i|| * sqrt(n) * max(|box_lo|, |box_hi|) + ||b_i||
    
    Parameters:
    -----------
    A_list : list of np.ndarray
        List of quadratic coefficient matrices for constraints
    b_list : list of np.ndarray
        List of linear coefficient vectors for constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
        
    Returns:
    --------
    C_list : list of float
        List of bounds for each constraint function gradient
    """
    n = A_list[0].shape[0] if A_list else 0
    C_list = []
    
    # Compute the diameter of the box (or use default if unbounded)
    if box_lo is None or box_hi is None:
        # For unbounded case, use a reasonable default bound
        box_diameter = 10.0  # Default bound
    else:
        box_diameter = max(abs(box_lo), abs(box_hi))
    max_norm_x = np.sqrt(n) * box_diameter
    
    for A_i, b_i in zip(A_list, b_list):
        # Compute the spectral norm (maximum singular value) of matrix A_i
        A_i_norm = np.linalg.norm(A_i, ord=2)
        
        # Compute the 2-norm of vector b_i
        b_i_norm = np.linalg.norm(b_i, ord=2)
        
        # Compute gradient bound
        C_i = A_i_norm * max_norm_x + b_i_norm
        C_list.append(C_i)
    
    return C_list


# --------------------------
# Slater point computation
# --------------------------

def find_slater_point_for_node(A0, b0, A_list_i, b_list_i, c_list_i, box_lo, box_hi, max_attempts=1000):
    """
    Find a point satisfying Slater condition for node i
    
    Parameters:
    -----------
    A0 : np.ndarray
        Quadratic coefficient matrix of the objective function
    b0 : np.ndarray
        Linear coefficient vector of the objective function
    A_list_i : list of np.ndarray
        List of quadratic coefficient matrices for node i's constraints
    b_list_i : list of np.ndarray
        List of linear coefficient vectors for node i's constraints
    c_list_i : list of float
        List of constant terms for node i's constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    max_attempts : int
        Maximum number of attempts
        
    Returns:
    --------
    x_slater : np.ndarray or None
        Found Slater point, or None if not found
    """
    n = A0.shape[0]
    rng = np.random.default_rng(123)
    
    for _ in range(max_attempts):
        # Generate point: use box if provided, otherwise use standard normal
        if box_lo is None or box_hi is None:
            x_candidate = rng.standard_normal(n) * 5.0  # Use reasonable scale
        else:
            x_candidate = rng.uniform(box_lo, box_hi, n)
        all_feasible = True
        for Aj, bj, cj in zip(A_list_i, b_list_i, c_list_i):
            val = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate + cj
            if val >= 0:
                all_feasible = False
                break
        if all_feasible:
            return x_candidate
    
    return None


def find_slater_point(A0, b0, A_list, b_list, c_list, box_lo, box_hi, max_attempts=100):
    """
    Find a point satisfying Slater condition
    
    Parameters:
    -----------
    A0, b0, A_list, b_list, c_list, box_lo, box_hi : same as compute_dual_variable_bound
    max_attempts : int
        Maximum number of attempts
        
    Returns:
    --------
    x_slater : np.ndarray or None
        Found Slater point, or None if not found
    """
    n = A0.shape[0]
    m = len(A_list)
    
    # Try to find a point inside the box that satisfies all constraints
    for attempt in range(max_attempts):
        # Randomly generate a point inside the box (or unbounded if box_lo/box_hi are None)
        if box_lo is None or box_hi is None:
            x_candidate = np.random.standard_normal(n) * 5.0  # Use reasonable scale
        else:
            x_candidate = np.random.uniform(box_lo * 0.5, box_hi * 0.5, size=n)
        
        # Check all constraints
        all_satisfied = True
        for j in range(m):
            Aj, bj, cj = A_list[j], b_list[j], c_list[j]
            constraint_value = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate + cj
            if constraint_value >= 0:
                all_satisfied = False
                break
        
        if all_satisfied:
            return x_candidate
    
    # If random search fails, try using optimization method
    print("Random search did not find Slater point, trying optimization method...")
    
    # Use cvxpy to find a point satisfying constraints
    try:
        x_var = cp.Variable(n)
        
        # Objective: minimize constraint violation degree
        obj = 0
        for j in range(m):
            Aj, bj, cj = A_list[j], b_list[j], c_list[j]
            obj += cp.maximum(0.5 * cp.quad_form(x_var, Aj) + bj @ x_var + cj, 0)
        
        # Add box constraints only if provided
        constraints = []
        if box_lo is not None:
            constraints.append(x_var >= box_lo)
        if box_hi is not None:
            constraints.append(x_var <= box_hi)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status in ("optimal", "optimal_inaccurate"):
            x_slater = x_var.value.copy()
            # Verify Slater condition
            slater_gaps = np.zeros(m)
            for j in range(m):
                Aj, bj, cj = A_list[j], b_list[j], c_list[j]
                slater_gaps[j] = 0.5 * x_slater @ (Aj @ x_slater) + bj @ x_slater + cj
            
            if np.all(slater_gaps < 0):
                return x_slater
            else:
                print(f"Point found by optimization does not satisfy strict Slater condition, constraint violations: {slater_gaps}")
                return x_slater  # Still return even if strict condition is not satisfied
    except Exception as e:
        print(f"Optimization method failed: {e}")
    
    return None


# --------------------------
# Dual variable bound computation
# --------------------------

def compute_dual_variable_bound_for_node_i(A0, b0, A_list_i, b_list_i, c_list_i, 
                                            box_lo, box_hi, x_slater=None):
    """
    Compute the bound B_i for dual variable theta for node i
    
    Compute the dual variable bound for node i using Slater point:
    B_i = max_j |f(x_slater) / g_{i,j}(x_slater)|
    where g_{i,j} is the j-th constraint function of node i
    
    Parameters:
    -----------
    A0 : np.ndarray
        Quadratic coefficient matrix of the objective function (n x n)
    b0 : np.ndarray
        Linear coefficient vector of the objective function (n,)
    A_list_i : list of np.ndarray
        List of quadratic coefficient matrices for node i's constraints
    b_list_i : list of np.ndarray
        List of linear coefficient vectors for node i's constraints
    c_list_i : list of float
        List of constant terms for node i's constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    x_slater : np.ndarray, optional
        Slater point, if None then automatically search for one
        
    Returns:
    --------
    B_i : float
        Bound for dual variable theta for node i
    x_slater : np.ndarray
        Slater point used
    """
    n = A0.shape[0]
    m_i = len(A_list_i)
    
    # If no Slater point is provided, try to find one
    if x_slater is None:
        x_slater = find_slater_point_for_node(A0, b0, A_list_i, b_list_i, c_list_i, 
                                               box_lo, box_hi)
        if x_slater is None:
            # If no Slater point found, return a conservative bound
            return 100.0, None
    
    # Compute the objective function value at the Slater point
    f_slater = 0.5 * x_slater @ (A0 @ x_slater) + b0 @ x_slater
    
    # Compute the estimate of the dual variable bound
    # Use the violation degree of each constraint of node i to estimate the bound
    B_candidates = []
    for j in range(m_i):
        Aj, bj, cj = A_list_i[j], b_list_i[j], c_list_i[j]
        g_val = 0.5 * x_slater @ (Aj @ x_slater) + bj @ x_slater + cj
        if g_val < 0:  # Constraint is satisfied
            B_j = abs(f_slater / g_val)
            B_candidates.append(B_j)
        else:  # Constraint is violated, use a larger bound
            B_j = abs(f_slater) * 10.0  # Use a moderate multiplier
            B_candidates.append(B_j)
    
    B_i = max(B_candidates) if B_candidates else 10.0
    
    return B_i, x_slater


def compute_dual_variable_bound(A0, b0, A_list, b_list, c_list, box_lo, box_hi, x_slater=None):
    """
    Compute the bound B for dual variable theta using Slater point
    
    For QCQP problem:
    min 0.5 * x^T * A0 * x + b0^T * x
    s.t. 0.5 * x^T * Aj * x + bj^T * x + cj <= 0,  j=1..m
         box_lo <= x <= box_hi
    
    If there exists a Slater point x_slater, i.e.:
    - box_lo <= x_slater <= box_hi
    - 0.5 * x_slater^T * Aj * x_slater + bj^T * x_slater + cj < 0, for all j
    
    Then the bound for dual variable theta can be estimated as:
    B = max_j |f(x_slater) / g_j(x_slater)|
    where f(x) is the objective function and g_j(x) is the j-th constraint function
    
    Parameters:
    -----------
    A0 : np.ndarray
        Quadratic coefficient matrix of the objective function (n x n)
    b0 : np.ndarray
        Linear coefficient vector of the objective function (n,)
    A_list : list of np.ndarray
        List of quadratic coefficient matrices for constraints
    b_list : list of np.ndarray
        List of linear coefficient vectors for constraints
    c_list : list of float
        List of constant terms for constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    x_slater : np.ndarray, optional
        Slater point, if None then automatically search for one
        
    Returns:
    --------
    B : float
        Bound for dual variable theta
    x_slater : np.ndarray
        Slater point used
    slater_gaps : np.ndarray
        Constraint violation degree at Slater point for each constraint (negative values indicate satisfaction)
    """
    n = A0.shape[0]
    m = len(A_list)
    
    # If no Slater point is provided, try to find one
    if x_slater is None:
        x_slater = find_slater_point(A0, b0, A_list, b_list, c_list, box_lo, box_hi)
        if x_slater is None:
            raise ValueError("Cannot find Slater point, problem may not satisfy Slater condition")
    
    # Verify Slater condition
    slater_gaps = np.zeros(m)
    for j in range(m):
        Aj, bj, cj = A_list[j], b_list[j], c_list[j]
        slater_gaps[j] = 0.5 * x_slater @ (Aj @ x_slater) + bj @ x_slater + cj
    
    # Check if Slater condition is satisfied
    if np.any(slater_gaps >= 0):
        print(f"Warning: Provided point does not satisfy Slater condition")
        print(f"Constraint violation degree: {slater_gaps}")
        # Still compute bound even if strict Slater condition is not satisfied
    
    # Compute objective function value at Slater point
    f_slater = 0.5 * x_slater @ (A0 @ x_slater) + b0 @ x_slater
    
    # Compute estimate of dual variable bound
    # Use violation degree of each constraint to estimate the bound
    B_candidates = []
    for j in range(m):
        if slater_gaps[j] < 0:  # Constraint is satisfied
            B_j = abs(f_slater / slater_gaps[j])
            B_candidates.append(B_j)
        else:  # Constraint is violated, use a larger bound
            B_j = abs(f_slater) * 10.0  # Use a moderate multiplier
            B_candidates.append(B_j)
    
    B = max(B_candidates) if B_candidates else 10.0
    
    return B, x_slater, slater_gaps


# --------------------------
# Initial tau computation
# --------------------------

def compute_initial_tau_per_node(A0, b0, pernode_constraints, box_lo, box_hi,
                                  L_obj, N, c_alpha=0.1, c_beta=0.1, c_c=0.1,
                                  delta=0.1, zeta=1.0, L_f_i_list=None):
    """
    Compute the initial tau parameter for each node
    
    According to the formula:
    \hat\tau_i = min{
      (-L_{f_i} + sqrt(L_{f_i}^2 + 4(1-(δ+c))L_{g_i}^2 B_i^2 / c_β)) / (2L_{g_i}^2 B_i^2 / c_β),
      1/C_{g_i} * sqrt(c_α(1-δ) / (2ζ_i))
    }
    
    where:
    - c = c_alpha + c_beta + c_c (where c_c is c_varsigma)
    - L_{f_i} = L_obj / N (if L_f_i_list is None) OR L_f_i_list[i] (if provided)
    - L_{g_i} = node-specific Lipschitz constant for node i's constraints
    - B_i = dual variable bound for node i
    - C_{g_i} = Jacobian bound for node i's constraints
    - ζ_i = zeta parameter for node i
    
    Parameters:
    -----------
    ...
    L_f_i_list : list of float, optional
        List of Lipschitz constants for each node's objective function.
        If provided, uses L_f_i_list[i] instead of L_obj / N.
        
    Returns:
    --------
    ...
    """
    c = c_alpha + c_beta + c_c
    
    if L_f_i_list is None:
        # Use average Lipschitz constant if per-node values not provided
        L_f_i_values = [L_obj / N] * N
    else:
        # Use provided per-node Lipschitz constants
        if len(L_f_i_list) != N:
            raise ValueError(f"Length of L_f_i_list ({len(L_f_i_list)}) must match N ({N})")
        L_f_i_values = L_f_i_list
    
    tau_list = []
    tau_components_list = []
    
    for i in range(N):
        L_f_i = L_f_i_values[i]
        A_list_i, b_list_i, c_list_i = pernode_constraints[i]
        
        # Compute L_{g_i}: node-specific Lipschitz constant for node i's constraints
        # If node i has a single constraint, use its Lipschitz constant directly
        # If node i has multiple constraints, compute appropriately for that node
        if len(A_list_i) == 1:
            # Single constraint: use its Lipschitz constant directly
            L_g_i = compute_lipschitz_constant(A_list_i[0], b_list_i[0])
        elif len(A_list_i) > 1:
            # Multiple constraints: compute node-specific value
            # For now, use the maximum (can be changed if needed)
            L_g_i_list = []
            for A_ij, b_ij in zip(A_list_i, b_list_i):
                L_ij = compute_lipschitz_constant(A_ij, b_ij)
                L_g_i_list.append(L_ij)
            L_g_i = max(L_g_i_list)
        else:
            # No constraints
            L_g_i = 0.0
        
        # Compute B_i: dual variable bound for node i
        B_i, _ = compute_dual_variable_bound_for_node_i(
            A0, b0, A_list_i, b_list_i, c_list_i, box_lo, box_hi
        )
        
        # Compute C_{g_i}: Jacobian bound for node i
        C_g_i = compute_jacobian_bound_for_node_i(A_list_i, b_list_i, box_lo, box_hi)
        
        # Compute term1: (-L_{f_i} + sqrt(L_{f_i}^2 + 4(1-(δ+c))L_{g_i}^2 B_i^2 / c_β)) / (2L_{g_i}^2 B_i^2 / c_β)
        # Formula: term1 = (-L_{f_i} + sqrt(L_{f_i}^2 + 4(1-(δ+c))L_{g_i}^2 B_i^2 / c_β)) / (2L_{g_i}^2 B_i^2 / c_β)
        if L_g_i > 0 and B_i > 0:
            # Compute discriminant: L_{f_i}^2 + 4(1-(δ+c))L_{g_i}^2 B_i^2 / c_β
            discriminant = L_f_i**2 + 4 * (1 - (delta + c)) * L_g_i**2 * B_i**2 / c_beta
            if discriminant >= 0:
                sqrt_term = np.sqrt(discriminant)
                # Numerator: -L_{f_i} + sqrt(L_{f_i}^2 + 4(1-(δ+c))L_{g_i}^2 B_i^2 / c_β)
                term1_numerator = -L_f_i + sqrt_term
                # Denominator: 2L_{g_i}^2 B_i^2 / c_β
                term1_denominator = 2 * L_g_i**2 * B_i**2 / c_beta
                if term1_denominator > 0:
                    term1 = term1_numerator / term1_denominator
                else:
                    term1 = np.inf
            else:
                term1 = np.inf
        else:
            term1 = np.inf
        
        # Compute term2: 1/C_{g_i} * sqrt(c_α(1-δ) / (2ζ_i))
        # Formula: term2 = 1 / C_{g_i} * sqrt(c_α(1-δ) / (2ζ_i))
        if C_g_i > 0 and zeta > 0:
            # Compute: sqrt(c_α(1-δ) / (2ζ_i))
            sqrt_arg = c_alpha * (1 - delta) / (2.0 * zeta)
            if sqrt_arg >= 0:
                term2 = (1.0 / C_g_i) * np.sqrt(sqrt_arg)
            else:
                term2 = np.inf
        else:
            term2 = np.inf
        
        # Select minimum value
        tau_i = min(term1, term2)
        
        # Save components for debugging
        tau_components = {
            'L_f_i': L_f_i,
            'L_g_i': L_g_i,
            'B_i': B_i,
            'C_g_i': C_g_i,
            'c': c,
            'term1': term1,
            'term2': term2,
            'tau_i': tau_i
        }
        
        tau_list.append(tau_i)
        tau_components_list.append(tau_components)
    
    return tau_list, tau_components_list


def compute_initial_tau(A0, b0, A_list, b_list, c_list, box_lo, box_hi, 
                       B_theta, L_obj, L_constraints, N,
                       c_alpha=0.1, c_beta=0.1, c_c=0.1, delta=0.1, zeta=1.0):
    """
    Compute the initial tau parameter (legacy function for backward compatibility)
    
    This function is kept for backward compatibility but now uses the new formula.
    It computes tau for each node and returns the average.
    
    Parameters:
    -----------
    A0, b0, A_list, b_list, c_list, box_lo, box_hi : problem parameters
    B_theta : float
        Dual variable bound (not used in new formula, kept for compatibility)
    L_obj : float
        Lipschitz constant of objective function
    L_constraints : list of float
        List of Lipschitz constants for constraint functions (not used in new formula)
    N : int
        Number of nodes
    c_alpha, c_beta, c_c : float
        Algorithm parameters
    delta : float
        Backtracking parameter
    zeta : float
        Dual step size parameter
        
    Returns:
    --------
    tau_initial : float
        Average initial tau value across all nodes
    tau_components : dict
        Values of various components in tau calculation
    """
    # Convert to pernode_constraints format
    pernode_constraints = []
    for i in range(N):
        # Each node gets constraint i
        A_i = [A_list[i]]
        b_i = [b_list[i]]
        c_i = [c_list[i]]
        pernode_constraints.append((A_i, b_i, c_i))
    
    # Use new per-node computation
    tau_list, tau_components_list = compute_initial_tau_per_node(
        A0, b0, pernode_constraints, box_lo, box_hi,
        L_obj, N, c_alpha=c_alpha, c_beta=c_beta, c_c=c_c,
        delta=delta, zeta=zeta
    )
    
    # Return average for backward compatibility
    tau_initial = np.mean(tau_list)
    
    # Aggregate components for backward compatibility
    tau_components = {
        'tau_list': tau_list,
        'tau_initial': tau_initial,
        'tau_components_list': tau_components_list
    }
    
    return tau_initial, tau_components


# --------------------------
# Random matrix generation
# --------------------------

def random_orthonormal(n, rng):
    """Generate a random orthonormal matrix"""
    M = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)  # square, Q is orthonormal
    return Q


def rand_psd_merely_convex(n, rng, lo=0.0, hi=100.0):
    """
    Return U^T S U with S diagonal in [0,hi], forced to include at least one 0.
    This creates a merely convex (not strongly convex) PSD matrix.
    """
    U = random_orthonormal(n, rng)
    diag = rng.uniform(lo, hi, size=n)
    # Force at least one exact zero (merely convex)
    idx = rng.integers(0, n)
    diag[idx] = 0.0
    # Force merely convex
    S = np.diag(diag)
    A = U.T @ S @ U
    # Numerical symmetrization
    return 0.5 * (A + A.T)


def rand_psd_merely_convex_std(n, rng, mean=50.0, std_ratio=0.4):
    """
    Return U^T S U with S diagonal generated from normal distribution,
    forced to include at least one 0. This creates a merely convex (not strongly convex) PSD matrix.
    
    The diagonal elements are generated from a normal distribution with specified mean and std,
    where std = mean * std_ratio (std_ratio is a percentage of mean).
    Values are then truncated to be non-negative (since PSD matrices require non-negative eigenvalues).
    At least one diagonal element is forced to be 0 to ensure merely convex property.
    
    Parameters:
    -----------
    n : int
        Dimension of the matrix
    rng : np.random.Generator
        Random number generator
    mean : float
        Mean of the normal distribution for diagonal elements
    std_ratio : float
        Ratio of standard deviation to mean (e.g., 0.4 means std = 0.4 * mean, i.e., 40% of mean)
        
    Returns:
    --------
    A : np.ndarray
        A merely convex PSD matrix of shape (n, n)
    """
    U = random_orthonormal(n, rng)
    # Compute std as a percentage of mean
    std = mean * std_ratio
    # Generate diagonal elements from normal distribution
    diag = rng.normal(mean, std, size=n)
    # Truncate negative values to 0 (PSD matrices require non-negative eigenvalues)
    diag = np.maximum(diag, 0.0)
    # Force at least one exact zero (merely convex)
    idx = rng.integers(0, n)
    diag[idx] = 0.0
    # Construct PSD matrix
    S = np.diag(diag)
    A = U.T @ S @ U
    # Numerical symmetrization
    return 0.5 * (A + A.T)


# --------------------------
# Network generation
# --------------------------

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
        i, j = rng.choice(N, 2, replace=False)
        if j not in neighbors[i]:
            neighbors[i].append(j)
            neighbors[j].append(i)
            added_edges += 1
    
    return neighbors


def get_neighbors(i, neighbors_list):
    """Get neighbors of node i"""
    return neighbors_list[i]


def sum_neighbor_diffs(x_list, i, neighbors_list):
    """Sum of differences with neighbors for node i"""
    Ni = get_neighbors(i, neighbors_list)
    return sum(x_list[i] - x_list[j] for j in Ni)


def generate_feasible_qcqp(n, N, rng, box_lo=-5.0, box_hi=5.0, neighbors_list=None, max_seed_attempts=10):
    """
    Generate a feasible QCQP problem with node-specific objective functions and constraints.
    
    Objective function: sum_{i in N} f_i(x) = (1/2) x^T Q_i x + q_i^T x
    where Q_i = r_i * Q^{(i)} with r_1 = 1 <= r_2 <= ... <= r_N = 10
    
    Constraint for node i: g_i(x) = (1/2) x^T A_i x + b_i^T x - c_i <= 0
    where ||A_i||_2 = 1
    
    This function ensures:
    1. sum_i Q_i is positive definite (tries different seeds if not)
    2. Optimal value phi* > 1 (by adjusting b_i, the linear term in constraints)
    3. At least 1-2 constraints are tight at optimal solution (by adjusting c_i)
    
    Note: We adjust b_i (constraint linear terms) to indirectly affect phi* by changing the feasible region.
    This is less direct than adjusting q_i, but allows us to control the optimal value through constraint geometry.
    
    Parameters:
    -----------
    n : int
        Dimension of decision variable
    N : int
        Number of nodes
    rng : np.random.Generator
        Random number generator
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    neighbors_list : list, optional
        Network topology for consensus constraints. If provided, will be used when solving
        to ensure phi* > 1. If None, solves without consensus constraints.
    max_seed_attempts : int
        Maximum number of seed attempts to find positive definite sum_i Q_i
        
    Returns:
    --------
    Q_list : list of np.ndarray
        List of node-specific quadratic coefficient matrices Q_i (n x n)
    q_list : list of np.ndarray
        List of node-specific linear coefficient vectors q_i (n,)
    A_list : list of np.ndarray
        List of node-specific constraint matrices A_i (n x n), one per node
    b_list : list of np.ndarray
        List of node-specific constraint linear terms b_i (n,), one per node
    c_list : list of float
        List of node-specific constraint constants c_i, one per node
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")
        
        # Step 1: Generate Q_i matrices and ensure sum_i Q_i is positive definite
        # Try different seeds if sum_i Q_i is not positive definite
        original_seed = rng.bit_generator.state['state']['state']
        Q_list = None
        Q_copies = None
        r_list = None
        
        for seed_attempt in range(max_seed_attempts):
            # Generate N i.i.d. copies Q^{(i)} with ||Q^{(i)}||_2 = 1
            Q_copies = []
            for i in range(N):
                Q_i = rand_psd_merely_convex(n, rng, lo=0.1, hi=10.0)
                Q_i_norm = np.linalg.norm(Q_i, ord=2)
                if Q_i_norm > 1e-10:
                    Q_i = Q_i / Q_i_norm  # Normalize to ||Q^{(i)}||_2 = 1
                Q_copies.append(Q_i)
            
            # Set Q_i = r_i * Q^{(i)} where 1 = r_1 <= r_2 <= ... <= r_N = 10
            r_list = np.linspace(1.0, 10.0, N)  # Linear spacing from 1 to 10
            Q_list = [r_list[i] * Q_copies[i] for i in range(N)]
            
            # Check if sum_i Q_i is positive definite
            A0_agg = np.sum(Q_list, axis=0)
            # Ensure symmetry
            A0_agg = 0.5 * (A0_agg + A0_agg.T)
            eigenvals = np.linalg.eigvals(A0_agg)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval > 1e-8:  # Positive definite
                break
            else:
                # Try next seed - advance the random state
                if seed_attempt < max_seed_attempts - 1:
                    # Advance random state by generating some random numbers
                    _ = rng.standard_normal(n * N)  # Advance state
        
        if min_eigenval <= 1e-8:
            raise RuntimeError(f"Could not generate positive definite sum_i Q_i after {max_seed_attempts} attempts. "
                             f"Min eigenvalue: {min_eigenval:.2e}")
        
        # Step 2: Generate initial q_i (will be adjusted later to ensure phi* > 1)
        q_list = [rng.standard_normal(n) * 0.1 for _ in range(N)]
    
    # Step 5: Generate A_i with ||A_i||_2 = 1 for each node
    A_list = []
    for i in range(N):
        A_i = rand_psd_merely_convex(n, rng, lo=0.1, hi=10.0)
        A_i_norm = np.linalg.norm(A_i, ord=2)
        if A_i_norm > 1e-10:
            A_i = A_i / A_i_norm  # Normalize to ||A_i||_2 = 1
        A_list.append(A_i)
    
    # Step 6: Generate b_i and c_i for constraints
    # Constraint: g_i(x) = (1/2) x^T A_i x + b_i^T x - c_i <= 0
    # We need to ensure feasibility for a common x that satisfies ALL constraints
    # (due to consensus constraints, all nodes must use the same x)
    b_list = []
    c_list = []
    
    # Create a feasible point that will satisfy all constraints
    # Use a point near the center of the box to ensure it's feasible
    x_feasible = rng.uniform(box_lo * 0.2, box_hi * 0.2, size=n)
    
    # Generate b_i for each node
    for i in range(N):
        b_i = rng.standard_normal(n) * 0.1
        b_list.append(b_i)
    
    # Now set c_i for each node to ensure ALL constraints are satisfied at x_feasible
    # We need: g_i(x_feasible) = 0.5 * x_feasible^T A_i x_feasible + b_i^T x_feasible - c_i <= 0
    # So: c_i >= 0.5 * x_feasible^T A_i x_feasible + b_i^T x_feasible
    # We set c_i to be slightly larger to ensure strict feasibility
    for i in range(N):
        constraint_value = 0.5 * x_feasible @ (A_list[i] @ x_feasible) + b_list[i] @ x_feasible
        # Set c_i such that g_i(x_feasible) <= 0 with a safety margin
        # Since constraint is: 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        # We need: c_i >= constraint_value, so set c_i = constraint_value + margin
        c_i = constraint_value + 0.5  # Add larger margin to ensure feasibility
        c_list.append(c_i)
    
    # Step 7: Test feasibility using solve_qcqp_ground_truth
    # First, verify that the problem is feasible with consensus constraints
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")
        
        # Aggregate objective for testing
        A0_agg = np.sum(Q_list, axis=0)
        b0_agg = np.sum(q_list, axis=0)
        
        # Test feasibility: try to solve the problem
        feasible = False
        max_feasibility_attempts = 5
        for feas_attempt in range(max_feasibility_attempts):
            try:
                x_test, phi_test = solve_qcqp_ground_truth(
                    A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi, 
                    neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                )
                feasible = True
                break
            except RuntimeError as e:
                # Problem is infeasible, relax constraints by increasing c_i
                if feas_attempt < max_feasibility_attempts - 1:
                    # Increase all c_i to make constraints more relaxed
                    relaxation = 1.0 + feas_attempt * 0.5  # Gradually increase relaxation
                    for i in range(N):
                        c_list[i] = c_list[i] + relaxation
                else:
                    # Last attempt: use very relaxed constraints
                    for i in range(N):
                        c_list[i] = c_list[i] + 5.0
        
        if not feasible:
            # Final attempt with very relaxed constraints
            for i in range(N):
                c_list[i] = c_list[i] + 10.0
            try:
                x_test, phi_test = solve_qcqp_ground_truth(
                    A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi, 
                    neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                )
                feasible = True
            except RuntimeError as e:
                raise RuntimeError(f"Could not generate feasible QCQP problem after multiple attempts: {e}")
    
    # Step 8: Adjust b_i (linear term in constraints) to ensure phi* > 1
    # Strategy: Adjust b_i to change constraint geometry, which affects the feasible region
    # and indirectly influences the optimal value phi*
    target_min = 1.1
    max_adjustment_iterations = 30
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")
            
            # Use the feasible solution from feasibility test
            x_star_temp = x_test
            phi_star_temp = phi_test
            
            # Strategy: Find a point with objective value > 1, then adjust constraints to make it optimal
            # This is more direct than iteratively adjusting constraints
            print(f"  Finding point with objective > {target_min:.6e} and adjusting constraints...")
            
            # Step 1: Find a point with objective value >= target_min
            # Since sum_i Q_i is positive definite, we can find points with high objective values
            # by moving away from the unconstrained minimizer in the direction of increasing objective
            
            # Find unconstrained minimizer: x_min = - (sum Q_i)^{-1} (sum q_i)
            try:
                A0_agg_inv = np.linalg.inv(A0_agg)
                x_min = -A0_agg_inv @ b0_agg
                x_min = np.clip(x_min, box_lo, box_hi)
                phi_min = 0.5 * x_min @ (A0_agg @ x_min) + b0_agg @ x_min
            except:
                # If inversion fails, use current optimal point as reference
                x_min = x_star_temp.copy()
                phi_min = phi_star_temp
            
            # Find a point with objective >= target_min
            # Strategy: Move from x_min in a direction that increases the objective
            # We can use the gradient at x_min, or try multiple directions
            x_target = None
            phi_target = None
            
            # Try gradient direction first
            grad_at_min = A0_agg @ x_min + b0_agg
            grad_norm = np.linalg.norm(grad_at_min)
            
            if grad_norm > 1e-10:
                direction = grad_at_min / grad_norm
                # Try different step sizes to find a point with high objective
                for step_mult in [1.0, 2.0, 5.0, 10.0, 20.0]:
                    x_candidate = x_min + direction * step_mult
                    x_candidate = np.clip(x_candidate, box_lo, box_hi)
                    phi_candidate = 0.5 * x_candidate @ (A0_agg @ x_candidate) + b0_agg @ x_candidate
                    if phi_candidate >= target_min:
                        x_target = x_candidate
                        phi_target = phi_candidate
                        break
            
            # If gradient direction didn't work, try random directions
            if x_target is None:
                for attempt in range(20):
                    direction = rng.standard_normal(n)
                    direction = direction / np.linalg.norm(direction)
                    for step_mult in [1.0, 2.0, 5.0, 10.0]:
                        x_candidate = x_min + direction * step_mult
                        x_candidate = np.clip(x_candidate, box_lo, box_hi)
                        phi_candidate = 0.5 * x_candidate @ (A0_agg @ x_candidate) + b0_agg @ x_candidate
                        if phi_candidate >= target_min:
                            x_target = x_candidate
                            phi_target = phi_candidate
                            break
                    if x_target is not None:
                        break
            
            # If still no point found, adjust q_i to shift the objective function
            if x_target is None:
                print(f"  Could not find point with phi >= {target_min:.6e}, adjusting q_i to shift objective...")
                # Adjust q_i to make the objective function have higher values
                delta_needed = target_min - phi_min + 0.2
                # Add a constant vector to q_i to shift the objective
                x_min_norm_sq = np.dot(x_min, x_min)
                if x_min_norm_sq > 1e-10:
                    adjustment_q = (delta_needed / (N * x_min_norm_sq)) * x_min
                else:
                    adjustment_q = rng.standard_normal(n) * (delta_needed / (N * n))
                
                for i in range(N):
                    q_list[i] = q_list[i] + adjustment_q
                
                # Recompute aggregated objective
                b0_agg = np.sum(q_list, axis=0)
                # Now try again to find a point with high objective
                x_min = -A0_agg_inv @ b0_agg if 'A0_agg_inv' in locals() else x_star_temp
                x_min = np.clip(x_min, box_lo, box_hi)
                phi_min = 0.5 * x_min @ (A0_agg @ x_min) + b0_agg @ x_min
                
                if phi_min < target_min:
                    # Try gradient direction again
                    grad_at_min = A0_agg @ x_min + b0_agg
                    grad_norm = np.linalg.norm(grad_at_min)
                    if grad_norm > 1e-10:
                        direction = grad_at_min / grad_norm
                        for step_mult in [1.0, 2.0, 5.0, 10.0]:
                            x_candidate = x_min + direction * step_mult
                            x_candidate = np.clip(x_candidate, box_lo, box_hi)
                            phi_candidate = 0.5 * x_candidate @ (A0_agg @ x_candidate) + b0_agg @ x_candidate
                            if phi_candidate >= target_min:
                                x_target = x_candidate
                                phi_target = phi_candidate
                                break
            
            # Step 2: Adjust constraints (b_i and c_i) to make x_target optimal
            if x_target is not None and phi_target >= target_min:
                print(f"  Found target point with phi = {phi_target:.6e}, adjusting constraints...")
                
                # Strategy: Adjust constraints so that x_target becomes the optimal solution
                # For x_target to be optimal, it must:
                # 1. Be feasible: g_i(x_target) <= 0 for all i
                # 2. Satisfy KKT conditions: grad_phi(x_target) + sum_i lambda_i * grad_g_i(x_target) = 0
                #    where lambda_i >= 0 are dual variables for tight constraints
                
                # First, make all constraints tight at x_target (or slightly satisfied)
                # This ensures x_target is on the boundary of feasible region
                for i in range(N):
                    # Set c_i so that g_i(x_target) = 0 (tight constraint)
                    c_i_new = 0.5 * x_target @ (A_list[i] @ x_target) + b_list[i] @ x_target
                    c_list[i] = c_i_new
                
                # Now adjust b_i to ensure KKT conditions are satisfied
                # KKT condition: grad_phi(x_target) + sum_i lambda_i * (A_i @ x_target + b_i) = 0
                # We want to adjust b_i so that the gradient of constraints aligns with the gradient of objective
                grad_phi_at_target = A0_agg @ x_target + b0_agg
                
                # For each constraint, adjust b_i to help satisfy KKT conditions
                # We want: A_i @ x_target + b_i to be in the direction opposite to grad_phi
                # This helps make x_target optimal
                for i in range(N):
                    # Current gradient of constraint i at x_target
                    grad_g_i = A_list[i] @ x_target + b_list[i]
                    
                    # We want grad_g_i to be in a direction that helps satisfy KKT
                    # Adjust b_i to align grad_g_i with -grad_phi (scaled)
                    # This is a heuristic to make x_target closer to optimal
                    grad_phi_norm = np.linalg.norm(grad_phi_at_target)
                    if grad_phi_norm > 1e-10:
                        # Adjust b_i to make constraint gradient align better with objective gradient
                        # This is a simplified approach - in practice, we'd need to solve for dual variables
                        adjustment = -grad_phi_at_target * 0.1 / N  # Small adjustment
                        b_list[i] = b_list[i] + adjustment
                        
                        # Recompute c_i to keep constraint tight at x_target
                        c_list[i] = 0.5 * x_target @ (A_list[i] @ x_target) + b_list[i] @ x_target
                
                # Re-solve to verify x_target is optimal
                b0_agg = np.sum(q_list, axis=0)
                try:
                    x_star_new, phi_star_new = solve_qcqp_ground_truth(
                        A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                        neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                    )
                    
                    if phi_star_new >= target_min:
                        print(f"  ✓ Success: phi* = {phi_star_new:.6e} >= {target_min:.6e}")
                        x_star_temp = x_star_new
                        phi_star_temp = phi_star_new
                    else:
                        # If not optimal, try iterative refinement
                        print(f"  Target point not optimal yet (phi* = {phi_star_new:.6e}), refining constraints...")
                        x_star_temp = x_star_new
                        phi_star_temp = phi_star_new
                        
                        # Iterative refinement: adjust constraints to push optimal solution towards target
                        for refine_iter in range(10):
                            if phi_star_temp >= target_min:
                                break
                            
                            # Adjust b_i to shift constraints towards x_target
                            grad_phi = A0_agg @ x_star_temp + b0_agg
                            grad_norm = np.linalg.norm(grad_phi)
                            if grad_norm > 1e-10:
                                direction = grad_phi / grad_norm
                                # Adjust b_i to make constraints more favorable at x_target
                                adjustment = (x_target - x_star_temp) * 0.1
                                for i in range(N):
                                    b_list[i] = b_list[i] + adjustment
                                    # Keep constraint tight at x_target
                                    c_list[i] = 0.5 * x_target @ (A_list[i] @ x_target) + b_list[i] @ x_target
                            
                            # Re-solve
                            b0_agg = np.sum(q_list, axis=0)
                            x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                                A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                                neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                            )
                            
                except RuntimeError as e:
                    # If solving fails, relax constraints slightly
                    for i in range(N):
                        c_list[i] = c_list[i] + 0.5
                    b0_agg = np.sum(q_list, axis=0)
                    x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                        A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                        neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                    )
            else:
                # Fallback: use iterative adjustment of b_i
                print(f"  Could not find suitable target point, using iterative b_i adjustment...")
                # Use the previous iterative method as fallback
                for adjust_iter in range(max_adjustment_iterations):
                    b0_agg = np.sum(q_list, axis=0)
                    try:
                        x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                            A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                            neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                        )
                    except RuntimeError as e:
                        for i in range(N):
                            c_list[i] = c_list[i] + 2.0
                        x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                            A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                            neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                        )
                    
                    if phi_star_temp >= target_min:
                        break
                    
                    delta = target_min - phi_star_temp + 0.1
                    grad_phi = A0_agg @ x_star_temp + b0_agg
                    grad_norm = np.linalg.norm(grad_phi)
                    if grad_norm > 1e-10:
                        direction = grad_phi / grad_norm
                        adjustment = direction * max(2.0, abs(delta)) / N
                        for i in range(N):
                            b_list[i] = b_list[i] + adjustment
                            c_list[i] = c_list[i] + 0.5
                else:
                    # If gradient is near zero, adjust b_i in a way that shifts constraints
                    # Use a strategy that makes constraints more restrictive
                    base_adjustment = max(1.0, abs(delta))
                    adjustment_magnitude = base_adjustment / N
                    for i in range(N):
                        # Make a random adjustment to shift constraints
                        direction = rng.standard_normal(n)
                        direction = direction / np.linalg.norm(direction)
                        adjustment = direction * adjustment_magnitude
                        b_list[i] = b_list[i] + adjustment
                        # Relax c_i to maintain feasibility
                        c_list[i] = c_list[i] + 0.3
                
                # Re-solve to check new optimal value
                b0_agg = np.sum(q_list, axis=0)
                try:
                    x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                        A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                        neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                    )
                except RuntimeError as e:
                    # If solving fails, slightly relax constraints
                    for i in range(N):
                        c_list[i] = c_list[i] + 1.0
                    x_star_temp, phi_star_temp = solve_qcqp_ground_truth(
                        A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                        neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                    )
            
            # Final verification: ensure phi* > 1
            b0_agg = np.sum(q_list, axis=0)
            x_star_final, phi_star_final = solve_qcqp_ground_truth(
                A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                neighbors_list=neighbors_list, verbose=False, constant_term=0.0
            )
            
            if phi_star_final < target_min:
                # Last resort: make much larger adjustments to b_i
                delta = target_min - phi_star_final + 0.3
                grad_phi = A0_agg @ x_star_final + b0_agg
                grad_phi_norm = np.linalg.norm(grad_phi)
                if grad_phi_norm > 1e-10:
                    direction = grad_phi / grad_phi_norm
                    # Use much larger adjustment
                    base_adjustment = max(2.0, abs(delta) * 2.0)
                    adjustment_magnitude = base_adjustment / N
                    for i in range(N):
                        # Adjust opposite to gradient to tighten constraints
                        adjustment = -direction * adjustment_magnitude
                        b_list[i] = b_list[i] + adjustment
                        c_list[i] = c_list[i] + 0.5  # Also relax c_i significantly
                else:
                    # Random direction with large magnitude
                    base_adjustment = max(2.0, abs(delta) * 2.0)
                    adjustment_magnitude = base_adjustment / N
                    for i in range(N):
                        direction = rng.standard_normal(n)
                        direction = direction / np.linalg.norm(direction)
                        adjustment = direction * adjustment_magnitude
                        b_list[i] = b_list[i] + adjustment
                        c_list[i] = c_list[i] + 0.5
                b0_agg = np.sum(q_list, axis=0)
                x_star_final, phi_star_final = solve_qcqp_ground_truth(
                    A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                    neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                )
                
    except Exception as e:
        # If solving fails completely, use a simple heuristic
        print(f"Warning: Could not solve ground truth to verify phi* > 1, using heuristic adjustment: {e}")
        # Make small adjustments to b_i
        for i in range(N):
            b_list[i] = b_list[i] + rng.standard_normal(n) * 0.01
        # Also relax constraints as a safety measure
        for i in range(N):
            c_list[i] = c_list[i] + 2.0
    
    # Step 9: Ensure at least 1-2 constraints are tight at optimal solution
    # Solve to get optimal solution, then adjust some constraints to be tight
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")
            
            A0_agg = np.sum(Q_list, axis=0)
            b0_agg = np.sum(q_list, axis=0)
            
            # Get optimal solution
            x_star_final, phi_star_final = solve_qcqp_ground_truth(
                A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                neighbors_list=neighbors_list, verbose=False, constant_term=0.0
            )
            
            # Compute constraint values at optimal solution
            # Constraint: g_i(x) = (1/2) x^T A_i x + b_i^T x - c_i <= 0
            constraint_values = []
            for i in range(N):
                g_i_value = 0.5 * x_star_final @ (A_list[i] @ x_star_final) + b_list[i] @ x_star_final - c_list[i]
                constraint_values.append(g_i_value)
            
            constraint_values = np.array(constraint_values)
            
            # Check how many constraints are tight (within tolerance)
            tolerance = 1e-6
            tight_mask = np.abs(constraint_values) <= tolerance
            num_tight = np.sum(tight_mask)
            
            # Target: at least 1-2 tight constraints
            target_num_tight = min(2, N)  # At least 1-2, but not more than N
            # Ensure at least 1 tight constraint, preferably 2
            min_target = 1
            max_target = min(2, N)
            
            if num_tight < min_target:
                # Select constraints to make tight
                # Choose constraints that are closest to being tight (smallest |g_i(x*)|)
                # but not already tight
                constraint_abs_values = np.abs(constraint_values)
                # Exclude already tight constraints
                candidate_indices = np.where(~tight_mask)[0]
                
                if len(candidate_indices) > 0:
                    # Sort by absolute constraint value (closest to 0 first)
                    sorted_indices = candidate_indices[np.argsort(constraint_abs_values[candidate_indices])]
                    # Try to make 2 constraints tight, but at least 1
                    num_to_make_tight = max(1, min(max_target - num_tight, len(sorted_indices)))
                    indices_to_make_tight = sorted_indices[:num_to_make_tight]
                    
                    # Adjust c_i for selected constraints to make them tight
                    # For constraint i to be tight: g_i(x*) = 0.5 * x*^T A_i x* + b_i^T x* - c_i = 0
                    # So: c_i = 0.5 * x*^T A_i x* + b_i^T x*
                    for i in indices_to_make_tight:
                        c_i_new = 0.5 * x_star_final @ (A_list[i] @ x_star_final) + b_list[i] @ x_star_final
                        c_list[i] = c_i_new
                    
                    # Re-solve to verify feasibility and phi* > 1
                    # Note: Making constraints tighter might make the problem infeasible or change the optimal solution
                    # So we need to re-solve and potentially adjust
                    max_retry = 5
                    x_star_current = x_star_final.copy()
                    for retry in range(max_retry):
                        try:
                            x_star_new, phi_star_new = solve_qcqp_ground_truth(
                                A0_agg, b0_agg, A_list, b_list, c_list, box_lo, box_hi,
                                neighbors_list=neighbors_list, verbose=False, constant_term=0.0
                            )
                            
                            # Verify phi* > 1
                            if phi_star_new > target_min:
                                # Success: verify tight constraints
                                constraint_values_new = []
                                for i in range(N):
                                    g_i_value = 0.5 * x_star_new @ (A_list[i] @ x_star_new) + b_list[i] @ x_star_new - c_list[i]
                                    constraint_values_new.append(g_i_value)
                                constraint_values_new = np.array(constraint_values_new)
                                tight_mask_new = np.abs(constraint_values_new) <= tolerance
                                num_tight_new = np.sum(tight_mask_new)
                                
                                if num_tight_new >= min_target:  # At least 1 tight constraint
                                    # If we have at least 1 but want 2, try to add one more
                                    if num_tight_new < max_target and retry < max_retry - 1:
                                        # Try to make one more constraint tight
                                        candidate_indices_new = np.where(~tight_mask_new)[0]
                                        if len(candidate_indices_new) > 0:
                                            constraint_abs_values_new = np.abs(constraint_values_new)
                                            sorted_indices_new = candidate_indices_new[np.argsort(constraint_abs_values_new[candidate_indices_new])]
                                            if len(sorted_indices_new) > 0:
                                                i_additional = sorted_indices_new[0]
                                                c_i_new = 0.5 * x_star_new @ (A_list[i_additional] @ x_star_new) + b_list[i_additional] @ x_star_new
                                                c_list[i_additional] = c_i_new
                                                x_star_current = x_star_new.copy()
                                                # Continue to next iteration to verify
                                                continue
                                    # Success: we have at least min_target tight constraints
                                    break
                                else:
                                    # If still no tight constraints, try again with different constraints
                                    if retry < max_retry - 1:
                                        # Use the new optimal solution
                                        x_star_current = x_star_new.copy()
                                        constraint_values = constraint_values_new
                                        tight_mask = tight_mask_new
                                        num_tight = num_tight_new
                                        candidate_indices = np.where(~tight_mask)[0]
                                        if len(candidate_indices) > 0:
                                            constraint_abs_values = np.abs(constraint_values)
                                            sorted_indices = candidate_indices[np.argsort(constraint_abs_values[candidate_indices])]
                                            num_to_make_tight = max(1, min(max_target - num_tight, len(sorted_indices)))
                                            indices_to_make_tight = sorted_indices[:num_to_make_tight]
                                            for i in indices_to_make_tight:
                                                c_i_new = 0.5 * x_star_current @ (A_list[i] @ x_star_current) + b_list[i] @ x_star_current
                                                c_list[i] = c_i_new
                                    else:
                                        # Last retry: if still no tight constraints, at least ensure feasibility
                                        break
                            else:
                                # phi* <= target_min, need to adjust q_i again
                                if retry < max_retry - 1:
                                    # Adjust q_i to increase phi*
                                    scale_factor = target_min / phi_star_new * 1.1
                                    for i in range(N):
                                        q_list[i] = q_list[i] * scale_factor
                                    b0_agg = np.sum(q_list, axis=0)
                        except RuntimeError as e:
                            # If solving fails, slightly relax the constraints we just made tight
                            if retry < max_retry - 1:
                                for i in indices_to_make_tight:
                                    c_list[i] = c_list[i] - 0.1  # Slightly relax
                            else:
                                # Last retry failed, keep original constraints
                                break
    except Exception as e:
        # If ensuring tight constraints fails, continue with original constraints
        # This is not critical, so we just warn and continue
        pass
    
    return Q_list, q_list, A_list, b_list, c_list


def generate_feasible_qcqp_l1(n, N, rng, neighbors_list=None, max_adjustment_iterations=30):
    """
    Generate a feasible QCQP problem with L1 regularization and node-specific objectives/constraints.
    
    Centralized objective: ||x||_1 + (1/2) * sum_{i in N} x^T Q^i x
    Per-node objective: (1/N) * ||x||_1 + (1/2) * x^T Q^i x
    
    where Q^i = V^i * Gamma^i * (V^i)^T:
    - V^i is a random orthonormal matrix (V^i * (V^i)^T = I)
    - Gamma^i is a diagonal matrix with:
      * First diagonal element: 5 * i (i from 1 to N)
      * Last two diagonal elements: 0
      * Third-to-last diagonal element: 1
      * Remaining n-4 elements: uniformly sampled from [1, 5*i]
      * All elements sorted in decreasing order
    
    Per-node constraint: (1/2) * (x - bar{x}^i)^T * A^i * (x - bar{x}^i) <= 1
    
    where A^i = U^i * R^i * (U^i)^T:
    - U^i is a random orthonormal matrix (U^i * (U^i)^T = I)
    - R^i is a diagonal matrix with:
      * First diagonal element: 1/4
      * Last diagonal element: 1/16
      * Remaining n-2 elements: uniformly sampled from [1/16, 1/4]
      * All elements sorted in decreasing order
    
    and bar{x}^i is defined as:
    - bar{x}_j^i = 10 + xi_j^i
    - xi_j^i is uniformly sampled from [-1/(2*sqrt(n)), 1/(2*sqrt(n))]
    
    This function ensures:
    1. Optimal value phi* > 1
    2. At least 1-2 constraints are tight at optimal solution
    
    Parameters:
    -----------
    n : int
        Dimension of decision variable
    N : int
        Number of nodes
    rng : np.random.Generator
        Random number generator
    neighbors_list : list, optional
        Network topology for consensus constraints. If provided, will be used when solving.
    max_adjustment_iterations : int
        Maximum number of iterations to adjust constraints for phi* > 1 and tight constraints
        
    Returns:
    --------
    Q_list : list of np.ndarray
        List of node-specific quadratic coefficient matrices Q^i (n x n)
    lambda_l1 : float
        L1 regularization coefficient (1/N)
    A_list : list of np.ndarray
        List of node-specific constraint matrices A^i (n x n), one per node
    b_list : list of np.ndarray
        List of node-specific constraint linear terms b_i (n,), one per node
    c_list : list of float
        List of node-specific constraint constants c_i, one per node
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*ECOS.*")
        
        # L1 regularization coefficient
        lambda_l1 = 1.0 / N
        
        # Step 1: Generate Q^i = V^i * Gamma^i * (V^i)^T for each node i
        Q_list = []
        for i in range(N):
            node_idx = i + 1  # i from 1 to N
            
            # Generate random orthonormal matrix V^i
            V_i = random_orthonormal(n, rng)
            
            # Generate Gamma^i diagonal elements
            gamma_diag = np.zeros(n)
            
            if n >= 4:
                # First element: 5 * i
                gamma_diag[0] = 5.0 * node_idx
                
                # Last two elements: 0
                gamma_diag[n-1] = 0.0
                gamma_diag[n-2] = 0.0
                
                # Third-to-last element: 1
                gamma_diag[n-3] = 1.0
                
                # Remaining n-4 elements: uniformly sampled from [1, 5*i]
                if n > 4:
                    gamma_diag[1:n-3] = rng.uniform(1.0, 5.0 * node_idx, size=n-4)
            elif n == 3:
                # For n=3: first=5*i, last two=0,1
                gamma_diag[0] = 5.0 * node_idx
                gamma_diag[1] = 1.0
                gamma_diag[2] = 0.0
            elif n == 2:
                # For n=2: first=5*i, last=0
                gamma_diag[0] = 5.0 * node_idx
                gamma_diag[1] = 0.0
            else:  # n == 1
                # For n=1: just 5*i
                gamma_diag[0] = 5.0 * node_idx
            
            # Sort in decreasing order
            gamma_diag = np.sort(gamma_diag)[::-1]
            
            # Construct Gamma^i
            Gamma_i = np.diag(gamma_diag)
            
            # Construct Q^i = V^i * Gamma^i * (V^i)^T
            Q_i = V_i @ Gamma_i @ V_i.T
            # Ensure symmetry
            Q_i = 0.5 * (Q_i + Q_i.T)
            
            Q_list.append(Q_i)
        
        # Step 2: Generate A^i = U^i * R^i * (U^i)^T and bar{x}^i for each node i
        A_list = []
        b_list = []
        c_list = []
        x_bar_list = []
        
        for i in range(N):
            # Generate random orthonormal matrix U^i
            U_i = random_orthonormal(n, rng)
            
            # Generate R^i diagonal elements
            r_diag = np.zeros(n)
            
            if n >= 2:
                # First element: 1/4
                r_diag[0] = 0.25
                
                # Last element: 1/16
                r_diag[n-1] = 1.0 / 16.0
                
                # Remaining n-2 elements: uniformly sampled from [1/16, 1/4]
                if n > 2:
                    r_diag[1:n-1] = rng.uniform(1.0/16.0, 0.25, size=n-2)
            else:  # n == 1
                # For n=1: just 1/4
                r_diag[0] = 0.25
            
            # Sort in decreasing order
            r_diag = np.sort(r_diag)[::-1]
            
            # Construct R^i
            R_i = np.diag(r_diag)
            
            # Construct A^i = U^i * R^i * (U^i)^T
            A_i = U_i @ R_i @ U_i.T
            # Ensure symmetry
            A_i = 0.5 * (A_i + A_i.T)
            
            A_list.append(A_i)
            
            # Generate bar{x}^i
            # bar{x}_j^i = 2 + xi_j^i, where xi_j^i ~ Uniform[-1/(2*sqrt(n)), 1/(2*sqrt(n))]
            xi_i = rng.uniform(-1.0/(2.0*np.sqrt(n)), 1.0/(2.0*np.sqrt(n)), size=n)
            x_bar_i = 2 + xi_i
            x_bar_list.append(x_bar_i)
            
            # Convert constraint (1/2) * (x - bar{x}^i)^T * A^i * (x - bar{x}^i) <= 1
            # to standard form: (1/2) * x^T A^i x + b_i^T x - c_i <= 0
            # Expanding: (1/2) * x^T A^i x - x^T A^i bar{x}^i + (1/2) * (bar{x}^i)^T A^i bar{x}^i <= 1
            # Standard form: (1/2) * x^T A^i x + b_i^T x - c_i <= 0
            # where: b_i = -A^i bar{x}^i
            #        c_i = 1 - (1/2) * (bar{x}^i)^T A^i bar{x}^i
            b_i = -A_i @ x_bar_i
            c_i = 1.0 - 0.5 * (x_bar_i @ (A_i @ x_bar_i))
            
            b_list.append(b_i)
            c_list.append(c_i)
        
        # Step 3: Solve initial problem to get optimal value
        # Aggregate objective: sum_i Q^i
        Q_agg = np.sum(Q_list, axis=0)
        Q_agg = 0.5 * (Q_agg + Q_agg.T)  # Ensure symmetry
        
        # Step 4: Ensure optimal value > 1 and tight constraints
        target_min = 1.1
        
        try:
            # Solve the problem with L1 regularization
            # Note: Centralized objective is ||x||_1 + (1/2) * x^T Q_agg x
            # So L1 coefficient in solver should be 1, not lambda_l1
            x_star_temp, phi_star_temp = solve_qcqp_l1_ground_truth(
                Q_agg, 1.0, A_list, b_list, c_list,  # Use 1.0 for centralized L1 coefficient
                neighbors_list=neighbors_list, verbose=False
            )
            
            # Adjust constraints to ensure phi* > 1
            if phi_star_temp < target_min:
                print(f"  Initial phi* = {phi_star_temp:.6e} < {target_min:.6e}, adjusting constraints...")
                
                # Strategy: Adjust c_i to make constraints tighter, which may increase optimal value
                # We can also adjust x_bar^i to shift constraint centers
                for adjust_iter in range(max_adjustment_iterations):
                    # Adjust x_bar^i to shift constraints
                    # Move x_bar^i away from current optimal solution to make constraints tighter
                    for i in range(N):
                        # Adjust x_bar^i slightly
                        adjustment = (x_star_temp - x_bar_list[i]) * 0.1
                        x_bar_list[i] = x_bar_list[i] + adjustment
                        
                        # Recompute b_i and c_i
                        b_list[i] = -A_list[i] @ x_bar_list[i]
                        c_list[i] = 1.0 - 0.5 * (x_bar_list[i] @ (A_list[i] @ x_bar_list[i]))
                    
                    # Re-solve
                    x_star_temp, phi_star_temp = solve_qcqp_l1_ground_truth(
                        Q_agg, 1.0, A_list, b_list, c_list,  # Use 1.0 for centralized L1 coefficient
                        neighbors_list=neighbors_list, verbose=False
                    )
                    
                    if phi_star_temp >= target_min:
                        print(f"  ✓ Success: phi* = {phi_star_temp:.6e} >= {target_min:.6e}")
                        break
                
                # If still not > 1, try making constraints tighter by reducing c_i
                if phi_star_temp < target_min:
                    print(f"  Still phi* = {phi_star_temp:.6e} < {target_min:.6e}, making constraints tighter...")
                    for adjust_iter in range(max_adjustment_iterations):
                        # Reduce c_i to make constraints tighter
                        for i in range(N):
                            c_list[i] = c_list[i] * 0.95  # Reduce by 5%
                        
                        # Re-solve
                        try:
                            x_star_temp, phi_star_temp = solve_qcqp_l1_ground_truth(
                                Q_agg, 1.0, A_list, b_list, c_list,  # Use 1.0 for centralized L1 coefficient
                                neighbors_list=neighbors_list, verbose=False
                            )
                            if phi_star_temp >= target_min:
                                print(f"  ✓ Success: phi* = {phi_star_temp:.6e} >= {target_min:.6e}")
                                break
                        except RuntimeError:
                            # If infeasible, relax constraints slightly
                            for i in range(N):
                                c_list[i] = c_list[i] * 1.1
                            break
            
            # Step 5: Ensure at least 1-2 tight constraints
            # Compute constraint values at optimal solution
            constraint_values = []
            for i in range(N):
                # Constraint: (1/2) * x^T A^i x + b_i^T x - c_i <= 0
                g_i_value = 0.5 * x_star_temp @ (A_list[i] @ x_star_temp) + b_list[i] @ x_star_temp - c_list[i]
                constraint_values.append(g_i_value)
            
            constraint_values = np.array(constraint_values)
            
            # Check how many constraints are tight (within tolerance)
            tolerance = 1e-6
            tight_mask = np.abs(constraint_values) <= tolerance
            num_tight = np.sum(tight_mask)
            
            # Target: at least 1-2 tight constraints
            min_target = 1
            max_target = min(2, N)
            
            if num_tight < min_target:
                print(f"  Only {num_tight} tight constraints, adjusting to have at least {min_target}...")
                # Select constraints to make tight
                constraint_abs_values = np.abs(constraint_values)
                candidate_indices = np.where(~tight_mask)[0]
                
                if len(candidate_indices) > 0:
                    sorted_indices = candidate_indices[np.argsort(constraint_abs_values[candidate_indices])]
                    num_to_make_tight = max(1, min(max_target - num_tight, len(sorted_indices)))
                    indices_to_make_tight = sorted_indices[:num_to_make_tight]
                    
                    # Adjust c_i to make constraints tight at current optimal solution
                    for i in indices_to_make_tight:
                        # For constraint to be tight: (1/2) * x*^T A^i x* + b_i^T x* - c_i = 0
                        # So: c_i = (1/2) * x*^T A^i x* + b_i^T x*
                        c_list[i] = 0.5 * x_star_temp @ (A_list[i] @ x_star_temp) + b_list[i] @ x_star_temp
                    
                    # Re-solve to verify feasibility and phi* > 1
                    max_retry = 5
                    for retry in range(max_retry):
                        try:
                            x_star_new, phi_star_new = solve_qcqp_l1_ground_truth(
                                Q_agg, 1.0, A_list, b_list, c_list,  # Use 1.0 for centralized L1 coefficient
                                neighbors_list=neighbors_list, verbose=False
                            )
                            
                            if phi_star_new >= target_min:
                                # Verify tight constraints
                                constraint_values_new = []
                                for i in range(N):
                                    g_i_value = 0.5 * x_star_new @ (A_list[i] @ x_star_new) + b_list[i] @ x_star_new - c_list[i]
                                    constraint_values_new.append(g_i_value)
                                constraint_values_new = np.array(constraint_values_new)
                                tight_mask_new = np.abs(constraint_values_new) <= tolerance
                                num_tight_new = np.sum(tight_mask_new)
                                
                                if num_tight_new >= min_target:
                                    print(f"  ✓ Success: {num_tight_new} tight constraints, phi* = {phi_star_new:.6e}")
                                    break
                        except RuntimeError:
                            if retry < max_retry - 1:
                                # Slightly relax constraints
                                for i in indices_to_make_tight:
                                    c_list[i] = c_list[i] + 0.01
                            else:
                                break
                
        except Exception as e:
            print(f"Warning: Could not fully verify/adjust problem: {e}")
            # Continue with generated problem
        
        return Q_list, lambda_l1, A_list, b_list, c_list


def solve_qcqp_l1_ground_truth(Q_agg, lambda_l1, A_list, b_list, c_list, neighbors_list=None, verbose=True):
    """
    Solve QCQP problem with L1 regularization using centralized solver.
    
    This is a centralized solver, so it uses a single variable x (not per-node variables).
    The problem is:
    min ||x||_1 + (1/2) * x^T Q_agg x
    s.t. (1/2) * x^T A_i x + b_i^T x - c_i <= 0,  i=1..N
    
    Note: This is a centralized problem, so we don't need network topology information.
    The neighbors_list parameter is kept for backward compatibility but is not used.
    
    Parameters:
    -----------
    Q_agg : np.ndarray
        Aggregated quadratic coefficient matrix (n x n)
    lambda_l1 : float
        L1 regularization coefficient (should be 1.0 for centralized objective)
    A_list : list of np.ndarray
        List of constraint matrices A_i (n x n)
    b_list : list of np.ndarray
        List of constraint linear terms b_i (n,)
    c_list : list of float
        List of constraint constants c_i
    neighbors_list : list, optional
        Not used (kept for backward compatibility). Centralized solver doesn't need network info.
    verbose : bool, default=True
        If False, suppress print statements
        
    Returns:
    --------
    x_star : np.ndarray
        Optimal solution
    f_star : float
        Optimal objective value
    """
    n = Q_agg.shape[0]
    N = len(A_list)
    
    # Centralized solver: use a single variable x (not per-node variables)
    x_var = cp.Variable(n)
    
    # Objective: ||x||_1 + (1/2) * x^T Q_agg x
    obj = lambda_l1 * cp.norm(x_var, 1) + 0.5 * cp.quad_form(x_var, Q_agg)
    
    cons = []
    
    # All constraints: each node's constraint must be satisfied by the single variable x
    for i in range(N):
        cons.append(0.5 * cp.quad_form(x_var, A_list[i]) + b_list[i] @ x_var - c_list[i] <= 0)
    
    prob = cp.Problem(cp.Minimize(obj), cons)
    
    # Try solvers in order of preference
    solver_names = ["MOSEK", "ECOS", "SCS"]
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
    last_error = None
    
    for solver_name, solver in zip(solver_names, solvers):
        try:
            if verbose:
                print(f"Trying solver: {solver_name}")
            prob.solve(solver=solver, verbose=False)
            if verbose:
                print(f"Solver {solver_name} status: {prob.status}")
            if prob.status in ("optimal", "optimal_inaccurate"):
                if verbose:
                    print(f"Solver {solver_name} succeeded with objective value: {prob.value}")
                return x_var.value.copy(), prob.value
            else:
                if verbose:
                    print(f"Solver {solver_name} failed with status: {prob.status}")
        except Exception as e:
            if verbose:
                print(f"Solver {solver_name} raised exception: {e}")
            last_error = e
            continue
    
    error_msg = f"All solvers failed. Last solver status: {prob.status}"
    if last_error:
        error_msg += f". Last exception: {last_error}"
    raise RuntimeError(error_msg)


def generate_initial_point_near_feasible(n, A_list, b_list, c_list, box_lo, box_hi, rng, 
                                         max_attempts=1000, use_constraint_centers=False):
    """
    Generate an initial point near the feasible region.
    
    For L1-regularized QCQP problems, constraints are of the form:
    (1/2) * (x - bar{x}^i)^T * A^i * (x - bar{x}^i) <= 1
    
    This function tries to generate a point near the constraint centers bar{x}^i
    or find a feasible point using optimization.
    
    Parameters:
    -----------
    n : int
        Dimension of decision variable
    A_list : list of np.ndarray
        List of quadratic coefficient matrices for constraints
    b_list : list of np.ndarray
        List of linear coefficient vectors for constraints
    c_list : list of float
        List of constant terms for constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    rng : np.random.Generator
        Random number generator
    max_attempts : int
        Maximum number of attempts to find a feasible point
    use_constraint_centers : bool
        If True, try to reconstruct constraint centers bar{x}^i from b_i and use them
        as reference points. For L1 problems, b_i = -A^i * bar{x}^i, so bar{x}^i = -A^i^{-1} * b_i
        
    Returns:
    --------
    x_init : np.ndarray
        Initial point near the feasible region
    max_violation : float
        Maximum constraint violation at the generated point (should be small or negative)
    """
    m = len(A_list)
    
    # Strategy 1: Try to find a feasible point using optimization
    try:
        import cvxpy as cp
        x_var = cp.Variable(n)
        
        # Objective: minimize constraint violation
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        obj = 0
        for j in range(m):
            Aj, bj, cj = A_list[j], b_list[j], c_list[j]
            obj += cp.maximum(0.5 * cp.quad_form(x_var, Aj) + bj @ x_var - cj, 0)
        
        # Add box constraints only if provided
        constraints = []
        if box_lo is not None:
            constraints.append(x_var >= box_lo)
        if box_hi is not None:
            constraints.append(x_var <= box_hi)
        
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status in ("optimal", "optimal_inaccurate"):
            x_feasible = x_var.value.copy()
            # Check constraint violations
            # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
            violations = []
            for j in range(m):
                Aj, bj, cj = A_list[j], b_list[j], c_list[j]
                g_val = 0.5 * x_feasible @ (Aj @ x_feasible) + bj @ x_feasible - cj
                violation = max(0.0, g_val)
                violations.append(violation)
            max_violation = max(violations) if violations else 0.0
            
            # If we found a feasible point (or very close to feasible), return it
            if max_violation < 0.1:  # Allow small violation for numerical reasons
                return x_feasible, max_violation
    except Exception as e:
        # If optimization fails, continue with other strategies
        pass
    
    # Strategy 2: Use constraint centers if available
    if use_constraint_centers and m > 0:
        # For L1 problems: b_i = -A^i * bar{x}^i, so bar{x}^i = -A^i^{-1} * b_i
        constraint_centers = []
        for i in range(m):
            try:
                A_i = A_list[i]
                b_i = b_list[i]
                # Check if A_i is invertible
                if np.linalg.cond(A_i) < 1e12:  # Reasonable condition number
                    x_bar_i = -np.linalg.solve(A_i, b_i)
                    constraint_centers.append(x_bar_i)
            except:
                continue
        
        if len(constraint_centers) > 0:
            # Use average of constraint centers as reference point
            x_ref = np.mean(constraint_centers, axis=0)
            
            # Generate point near the reference point
            # Add small random perturbation
            if box_lo is None or box_hi is None:
                perturbation = rng.standard_normal(n) * 0.1  # Small perturbation
            else:
                # Scale perturbation based on box size
                box_size = box_hi - box_lo
                perturbation = rng.standard_normal(n) * (box_size * 0.05)
            
            x_candidate = x_ref + perturbation
            
            # Check constraint violations
            # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
            violations = []
            for j in range(m):
                Aj, bj, cj = A_list[j], b_list[j], c_list[j]
                g_val = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate - cj
                violation = max(0.0, g_val)
                violations.append(violation)
            max_violation = max(violations) if violations else 0.0
            
            return x_candidate, max_violation
    
    # Strategy 3: Random search for feasible point
    for attempt in range(max_attempts):
        if box_lo is None or box_hi is None:
            # For unbounded case, use smaller scale to stay near feasible region
            x_candidate = rng.standard_normal(n) * 2.0  # Smaller scale
        else:
            # Generate point in a smaller region of the box
            x_candidate = rng.uniform(box_lo * 0.3, box_hi * 0.3, size=n)
        
        # Compute constraint violations
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        violations = []
        for j in range(m):
            Aj, bj, cj = A_list[j], b_list[j], c_list[j]
            g_val = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate - cj
            violation = max(0.0, g_val)
            violations.append(violation)
        
        max_violation = max(violations) if violations else 0.0
        
        # Accept if violation is small (near feasible)
        if max_violation < 1.0:
            return x_candidate, max_violation
    
    # Fallback: return a point with moderate violation
    if box_lo is None or box_hi is None:
        x_fallback = rng.standard_normal(n) * 3.0
    else:
        x_fallback = rng.uniform(box_lo * 0.5, box_hi * 0.5, size=n)
    
    violations = []
    for j in range(m):
        Aj, bj, cj = A_list[j], b_list[j], c_list[j]
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        g_val = 0.5 * x_fallback @ (Aj @ x_fallback) + bj @ x_fallback - cj
        violation = max(0.0, g_val)
        violations.append(violation)
    max_violation = max(violations) if violations else 0.0
    
    return x_fallback, max_violation


def generate_initial_point_with_violation(n, A_list, b_list, c_list, box_lo, box_hi, rng, 
                                         target_violation=1.1, max_attempts=1000, 
                                         initial_scale=None, target_objective_ratio=None):
    """
    Generate an initial point with constraint violation > target_violation
    and optionally with large objective function value.
    
    Parameters:
    -----------
    n : int
        Dimension of decision variable
    A_list : list of np.ndarray
        List of quadratic coefficient matrices for constraints
    b_list : list of np.ndarray
        List of linear coefficient vectors for constraints
    c_list : list of float
        List of constant terms for constraints
    box_lo : float
        Lower bound of variables
    box_hi : float
        Upper bound of variables
    rng : np.random.Generator
        Random number generator
    target_violation : float
        Target minimum constraint violation
    max_attempts : int
        Maximum number of attempts to find a point with sufficient violation
    initial_scale : float, optional
        Scale factor for initial point generation. If None, uses default (10.0 for unbounded, 
        or box range for bounded). For larger objective values, use larger scale (e.g., 50.0).
    target_objective_ratio : float, optional
        If provided, try to generate a point with objective value >= target_objective_ratio * optimal_value.
        This requires Q_list to be passed separately. If None, only ensures constraint violation.
        
    Returns:
    --------
    x_init : np.ndarray
        Initial point with constraint violation > target_violation
    max_violation : float
        Maximum constraint violation at the generated point
    """
    m = len(A_list)
    
    # Determine initial scale
    if initial_scale is None:
        if box_lo is None or box_hi is None:
            initial_scale = 10.0  # Default for unbounded
        else:
            initial_scale = max(abs(box_lo), abs(box_hi))  # Use box range
    
    for attempt in range(max_attempts):
        # Generate a random point in the box (or unbounded if box_lo/box_hi are None)
        if box_lo is None or box_hi is None:
            # For unbounded case, use standard normal distribution with specified scale
            x_candidate = rng.standard_normal(n) * initial_scale
        else:
            # For bounded case, scale the box range
            x_candidate = rng.uniform(box_lo * (initial_scale / 10.0), box_hi * (initial_scale / 10.0), size=n)
        
        # Compute constraint violations
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        violations = []
        for j in range(m):
            Aj, bj, cj = A_list[j], b_list[j], c_list[j]
            g_val = 0.5 * x_candidate @ (Aj @ x_candidate) + bj @ x_candidate - cj
            violation = max(0.0, g_val)  # Only positive values count as violations
            violations.append(violation)
        
        max_violation = max(violations) if violations else 0.0
        
        if max_violation > target_violation:
            return x_candidate, max_violation
    
    # If we couldn't find a point with sufficient violation, try to create one
    # by starting from a feasible point and moving away from it
    if box_lo is None or box_hi is None:
        x_feasible = rng.standard_normal(n) * (initial_scale * 0.2)  # Use smaller scale for feasible point
    else:
        x_feasible = rng.uniform(box_lo * 0.2, box_hi * 0.2, size=n)
    
    # Find a constraint to violate
    for j in range(m):
        Aj, bj, cj = A_list[j], b_list[j], c_list[j]
        # Try to find a direction that increases violation
        # We can use the gradient of the constraint: grad g_j(x) = Aj @ x + bj
        grad_g = Aj @ x_feasible + bj
        
        # Move in a direction that increases the constraint value
        # Scale the movement to achieve target violation
        scale = target_violation * 2.0  # Use a larger scale to ensure violation > target
        if np.linalg.norm(grad_g) > 1e-10:
            direction = grad_g / np.linalg.norm(grad_g)
        else:
            # If gradient is too small, use a random direction
            direction = rng.standard_normal(n)
            direction = direction / np.linalg.norm(direction)
        
        x_violating = x_feasible + scale * direction
        
        # Clip to box constraints (if provided)
        if box_lo is not None and box_hi is not None:
            x_violating = np.clip(x_violating, box_lo, box_hi)
        
        # Check violation
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        g_val = 0.5 * x_violating @ (Aj @ x_violating) + bj @ x_violating - cj
        violation = max(0.0, g_val)
        
        if violation > target_violation:
            return x_violating, violation
    
    # Last resort: return a point at the boundary with large violation
    if box_lo is None or box_hi is None:
        x_boundary = rng.standard_normal(n) * (initial_scale * 2.0)  # Use larger scale for violation
    else:
        x_boundary = np.full(n, box_hi * 0.9)  # Near upper boundary
    violations = []
    for j in range(m):
        Aj, bj, cj = A_list[j], b_list[j], c_list[j]
        # Constraint: g_i(x) = 0.5 * x^T A_i x + b_i^T x - c_i <= 0
        g_val = 0.5 * x_boundary @ (Aj @ x_boundary) + bj @ x_boundary - cj
        violation = max(0.0, g_val)
        violations.append(violation)
    max_violation = max(violations) if violations else 0.0
    
    return x_boundary, max_violation


def build_gossip_matrix(neighbors_list, c=0.3):
    """
    Build gossip matrix W using the definition: W = I - L/(d_max + 1),
    where L is the Laplacian matrix of the graph, and d_max is the maximum degree.
    
    Laplacian matrix L definition:
    - L_ii = d_i (degree of node i)
    - L_ij = -1 if (i,j) is an edge, L_ij = 0 otherwise
    
    This yields a matrix W that is:
    - Symmetric: W = W^T (because L is symmetric)
    - Row stochastic: each row sums to 1
    - Compliant with graph G: w_ij > 0 for (i,j) ∈ E, w_ij = 0 otherwise, w_ii > 0
    
    Parameters:
    -----------
    neighbors_list : list of list
        neighbors_list[i] contains the neighbors of node i
    c : float
        Mixing parameter (optional, kept for backward compatibility, not used)
        
    Returns:
    --------
    W : np.ndarray
        Gossip matrix of shape (N, N)
    """
    N = len(neighbors_list)
    
    # Compute degrees (excluding self-loops)
    degrees = np.array([len(neighbors_list[i]) for i in range(N)])
    d_max = np.max(degrees)
    
    # Build Laplacian matrix L
    # L_ii = d_i (degree of node i)
    # L_ij = -1 if (i,j) is an edge, L_ij = 0 otherwise
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = degrees[i]  # Diagonal: degree of node i
        for j in neighbors_list[i]:
            L[i, j] = -1.0  # Off-diagonal: -1 for edges
    
    # Build gossip matrix: W = I - L/(d_max + 1)
    I = np.eye(N)
    W = I - L / (d_max + 1)
    
    # Verify properties
    assert np.allclose(W, W.T, atol=1e-10), "Matrix is not symmetric"
    assert np.allclose(np.sum(W, axis=1), 1.0, atol=1e-10), "Rows do not sum to 1"
    assert np.all(W.diagonal() > 0), "Diagonal elements are not all positive"
    
    # Verify graph compliance
    for i in range(N):
        for j in neighbors_list[i]:
            assert W[i, j] > 0, f"Edge weight w[{i},{j}] is not positive"
        # Check that non-edge entries are zero
        for j in range(N):
            if j != i and j not in neighbors_list[i]:
                assert abs(W[i, j]) < 1e-10, f"Non-edge weight w[{i},{j}] is not zero"
    
    return W


# --------------------------
# Ground truth solver
# --------------------------

def solve_qcqp_ground_truth(A0, b0, A_list, b_list, c_list, box_lo, box_hi, neighbors_list=None, verbose=True, constant_term=0.0):
    """
    Solve QCQP problem using centralized solver (MOSEK/ECOS/SCS)
    
    min 0.5 x^T A0 x + b0^T x + constant_term
    s.t. 0.5 x^T Aj x + bj^T x + cj <= 0,  j=1..m
         box_lo <= x <= box_hi
         x_i == x_j for all edges (i,j) in the network (consensus constraints)
    
    Parameters:
    -----------
    verbose : bool, default=True
        If False, suppress print statements
    constant_term : float, default=0.0
        Constant term added to the objective function
    """
    n = A0.shape[0]
    N = len(A_list)  # Number of nodes
    
    # Create variables for each node
    x_vars = [cp.Variable(n) for _ in range(N)]
    
    # Objective: sum of local objectives + constant term
    # Note: A0 and b0 should be aggregated as: A0 = sum_i Q_i, b0 = sum_i q_i
    # With consensus constraints (x_i == x_j), all x_vars[i] are equal to x
    # So: sum_i [0.5 x^T A0 x + b0^T x] = N * (0.5 x^T (sum Q_i) x + (sum q_i)^T x)
    # But we want: sum_i f_i(x) = 0.5 x^T (sum Q_i) x + (sum q_i)^T x
    # Therefore: obj = (1/N) * sum_i [0.5 x^T A0 x + b0^T x] = 0.5 x^T (sum Q_i) x + (sum q_i)^T x
    obj = 0
    for i in range(N):
        obj += 0.5 * cp.quad_form(x_vars[i], A0) + b0 @ x_vars[i]
    obj = obj / N + constant_term  # Divide by N to get correct total objective
    
    cons = []
    
    # Box constraints for each node
    for i in range(N):
        cons.append(x_vars[i] >= box_lo)
        cons.append(x_vars[i] <= box_hi)
    
    # Local constraints for each node
    # Constraint: g_i(x) = (1/2) x^T A_i x + b_i^T x - c_i <= 0
    for i in range(N):
        Aj, bj, cj = A_list[i], b_list[i], c_list[i]
        cons.append(0.5 * cp.quad_form(x_vars[i], Aj) + bj @ x_vars[i] - cj <= 0)
    
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
            if verbose:
                print(f"Trying solver: {solver_name}")
            prob.solve(solver=solver, verbose=False)  # Disable verbose to avoid pipe errors
            if verbose:
                print(f"Solver {solver_name} status: {prob.status}")
            if prob.status in ("optimal", "optimal_inaccurate"):
                if verbose:
                    print(f"Solver {solver_name} succeeded with objective value: {prob.value}")
                # Return the consensus solution (all x_vars should be equal)
                return x_vars[0].value.copy(), prob.value
            else:
                if verbose:
                    print(f"Solver {solver_name} failed with status: {prob.status}")
        except Exception as e:
            if verbose:
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
# QP with L1 regularization problem generation
# --------------------------

def generate_qp_with_l1(n, N, rng, lambda_l1=0.1):
    """
    Generate a QP problem with l1 regularization and no quadratic constraints.
    Uses node-specific objective functions.
    
    Objective function: sum_{i in N} f_i(x) = (1/2) x^T Q_i x + q_i^T x
    where Q_i = r_i * Q^{(i)} with r_1 = 1 <= r_2 <= ... <= r_N = 100
    
    Parameters:
    -----------
    n : int
        Dimension of decision variable
    N : int
        Number of nodes
    rng : np.random.Generator
        Random number generator
    lambda_l1 : float
        L1 regularization parameter
        
    Returns:
    --------
    Q_list : list of np.ndarray
        List of node-specific quadratic coefficient matrices Q_i (n x n)
    q_list : list of np.ndarray
        List of node-specific linear coefficient vectors q_i (n,)
    lambda_l1 : float
        L1 regularization parameter
    """
    # Step 1: Generate base matrix Q with ||Q||_2 = 1
    Q_base = rand_psd_merely_convex(n, rng, lo=0.1, hi=10.0)
    Q_base_norm = np.linalg.norm(Q_base, ord=2)
    if Q_base_norm > 1e-10:
        Q_base = Q_base / Q_base_norm  # Normalize to ||Q||_2 = 1
    
    # Step 2: Generate N i.i.d. copies Q^{(i)} with ||Q^{(i)}||_2 = 1
    Q_copies = []
    for i in range(N):
        Q_i = rand_psd_merely_convex(n, rng, lo=0.1, hi=10.0)
        Q_i_norm = np.linalg.norm(Q_i, ord=2)
        if Q_i_norm > 1e-10:
            Q_i = Q_i / Q_i_norm  # Normalize to ||Q^{(i)}||_2 = 1
        Q_copies.append(Q_i)
    
    # Step 3: Set Q_i = r_i * Q^{(i)} where 1 = r_1 <= r_2 <= ... <= r_N = 100
    r_list = np.linspace(1.0, 100.0, N)  # Linear spacing from 1 to 100
    Q_list = [r_list[i] * Q_copies[i] for i in range(N)]
    
    # Step 4: Generate q_i for each node
    q_list = [rng.standard_normal(n) * 0.1 for _ in range(N)]
    
    return Q_list, q_list, lambda_l1


def solve_qp_l1_ground_truth(A0, b0, lambda_l1, neighbors_list=None):
    """
    Solve QP with l1 regularization using centralized solver
    
    min 0.5 * x^T * A0 * x + b0^T * x + lambda_l1 * ||x||_1
    s.t. x_i == x_j for all edges (i,j) in the network (consensus constraints)
    
    Note: The objective is f(x) + r(x) where
    f(x) = 0.5 * x^T * A0 * x + b0^T * x and r(x) = lambda_l1 * ||x||_1
    
    Parameters:
    -----------
    A0 : np.ndarray
        Quadratic coefficient matrix (n x n)
    b0 : np.ndarray
        Linear coefficient vector (n,)
    lambda_l1 : float
        L1 regularization parameter
    neighbors_list : list, optional
        Network topology for consensus constraints
        
    Returns:
    --------
    x_star : np.ndarray
        Optimal solution
    f_star : float
        Optimal objective value
    """
    n = A0.shape[0]
    N = len(neighbors_list) if neighbors_list is not None else 1
    
    # Create variables for each node
    x_vars = [cp.Variable(n) for _ in range(N)]
    
    # Objective: sum of local objectives with l1 regularization
    # f_i(x) = (1/N)*(0.5 x^T A0 x + b0^T x), r(x) = lambda_l1 * ||x||_1
    # Full objective: (1/N) * (0.5 x^T A0 x + b0^T x) + lambda_l1 * ||x||_1
    # Since we have consensus constraints (x_i == x_j), we can use x_vars[0] for the L1 term
    obj = 0
    for i in range(N):
        obj += 0.5 * cp.quad_form(x_vars[i], A0) + b0 @ x_vars[i]
    obj = obj / N  # Average of local smooth objectives
    obj += lambda_l1 * cp.norm(x_vars[0], 1)  # L1 regularization (same for all nodes due to consensus)
    
    cons = []
    
    # Consensus constraints: x_i == x_j for connected nodes
    if neighbors_list is not None:
        for i in range(N):
            for j in neighbors_list[i]:
                if i < j:  # Avoid duplicate constraints
                    cons.append(x_vars[i] == x_vars[j])
    
    prob = cp.Problem(cp.Minimize(obj), cons)
    
    # Try solvers in order of preference
    solver_names = ["MOSEK", "ECOS", "SCS"]
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
    last_error = None
    
    for solver_name, solver in zip(solver_names, solvers):
        try:
            print(f"Trying solver: {solver_name}")
            prob.solve(solver=solver, verbose=False)
            print(f"Solver {solver_name} status: {prob.status}")
            if prob.status in ("optimal", "optimal_inaccurate"):
                print(f"Solver {solver_name} succeeded with objective value: {prob.value}")
                # Return the consensus solution
                return x_vars[0].value.copy(), prob.value
            else:
                print(f"Solver {solver_name} failed with status: {prob.status}")
        except Exception as e:
            print(f"Solver {solver_name} raised exception: {e}")
            last_error = e
            continue
    
    error_msg = f"All solvers failed. Last solver status: {prob.status}"
    if last_error:
        error_msg += f". Last exception: {last_error}"
    raise RuntimeError(error_msg)


def build_empty_constraints(num_nodes: int):
    """
    Build per-node constraint tuples with no local constraints (for L1 problems).
    Each tuple is (A_list_i, b_list_i, c_list_i); we use empty lists.
    """
    return [([], [], []) for _ in range(num_nodes)]

