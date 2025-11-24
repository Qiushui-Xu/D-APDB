import cvxpy as cp
import numpy as np
# min 0.5 x^T Q x  s.t.  ||A x + b||_2 <= c^T x + d  (SOCP 形式)
n = 5
Q = cp.diag([1,2,3,4,5])      # PSD -> 凸
A = 0.2 * np.random.randn(3,n)
b = np.zeros(3); c = np.ones(n); d = 1.0

x = cp.Variable(n)
obj = 0.5*cp.quad_form(x, Q)
con = [cp.norm(A@x + b, 2) <= c@x + d]  # 二范数约束是凸 QCQP
prob = cp.Problem(cp.Minimize(obj), con)
prob.solve(solver=cp.MOSEK)   
print("status:", prob.status, "opt:", prob.value)
