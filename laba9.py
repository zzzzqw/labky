import numpy as np

def build_variant6(n, p, r, t, bcoef):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 6.5 * (i + 1) ** (p / 3.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = bcoef * np.exp(-r / ((i + j + 2) ** t))
    f = np.array([5.0 * (i + 1) ** (t / 2.0) for i in range(n)], dtype=float)
    return A, f

def residual_inf(A, x, b):
    return np.linalg.norm(A @ x - b, np.inf)

def jacobi(A, b, tol=1e-10, kmax=200000):
    D = np.diag(np.diag(A))
    R = A - D
    Dinv = np.diag(1.0 / np.diag(D))
    x = np.zeros_like(b)
    for k in range(1, kmax + 1):
        x_new = Dinv @ (b - R @ x)
        if np.linalg.norm(x_new - x, np.inf) < tol:
            x = x_new
            break
        x = x_new
    return x, k

def sor(A, b, omega, tol=1e-10, kmax=200000):
    n = len(b)
    x = np.zeros(n, dtype=float)
    for k in range(1, kmax + 1):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i, i]
        if residual_inf(A, x, b) < tol:
            break
    return x, k

def omega_opt_from_jacobi(A):
    D = np.diag(np.diag(A))
    BJ = np.eye(A.shape[0]) - np.linalg.inv(D) @ A
    rho = max(abs(np.linalg.eigvals(BJ)))
    if rho >= 1.0:
        return 1.2
    return 2.0 / (1.0 + np.sqrt(1.0 - rho**2))

# ---- параметры варианта ----
n, p, r, t, bcoef = 8, 3.0, 0.8, 1.0, 0.05
A, f = build_variant6(n, p, r, t, bcoef)

# ---- эталон LU ----
x_lu = np.linalg.solve(A, f)
r_lu = residual_inf(A, x_lu, f)

# ---- Якоби ----
x_j, kj = jacobi(A, f, tol=1e-12)
r_j = residual_inf(A, x_j, f)

# ---- SOR ----
omega = omega_opt_from_jacobi(A)
x_s, ks = sor(A, f, omega, tol=1e-12)
r_s = residual_inf(A, x_s, f)

# ---- вывод ----
print(f"Вариант 6 | n={n}, p={p}, r={r}, t={t}, b={bcoef}")
print(f"Оптимальное ω ≈ {omega:.5f}")
print(f"LU:     невязка = {r_lu:.3e}")
print(f"Якоби:  итераций = {kj:5d}, невязка = {r_j:.3e}")
print(f"SOR:    итераций = {ks:5d}, невязка = {r_s:.3e}")
print(f"||x_LU - x_J||∞ = {np.linalg.norm(x_lu - x_j, np.inf):.3e}")
print(f"||x_LU - x_S||∞ = {np.linalg.norm(x_lu - x_s, np.inf):.3e}")