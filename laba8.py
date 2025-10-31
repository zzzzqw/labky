import numpy as np

# ---- функции LU, Зейделя и построения матрицы ----
def lu_compact(A):
    A = A.astype(float).copy()
    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            s = sum(A[i, k]*A[k, j] for k in range(i))
            A[i, j] -= s
        for j in range(i+1, n):
            s = sum(A[j, k]*A[k, i] for k in range(i))
            if abs(A[i, i]) < 1e-15:
                raise ZeroDivisionError("zero pivot")
            A[j, i] = (A[j, i]-s)/A[i, i]
    return A

def lu_solve(LU, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(LU[i, :i], y[:i])
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(LU[i, i+1:], x[i+1:])) / LU[i, i]
    return x

def gauss_seidel(A, b, tol=1e-10, kmax=10000):
    n = len(b)
    x = np.zeros(n)
    for k in range(kmax):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k+1
    return x, kmax

def build_variant6(n, p, r, t):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 8.5 * (i + 1) ** (p / 3)
        for j in range(n):
            if i != j:
                sgn = -1 if (i + j) % 2 else 1
                A[i, j] = sgn * 1e-3 * np.exp(-r / (abs(i - j) ** t))
    b = np.array([7.5 * (i + 1) ** (t / 2) for i in range(n)])
    return A, b

# ---- параметры ----
n, p, r, t = 8, 3.0, 2.5, 1.25
A, b = build_variant6(n, p, r, t)

# ---- решения ----
LU = lu_compact(A.copy())
x_lu = lu_solve(LU, b)
x_gs, k = gauss_seidel(A, b)

# ---- результаты ----
res_lu = np.linalg.norm(A @ x_lu - b, np.inf)
res_gs = np.linalg.norm(A @ x_gs - b, np.inf)
diff = np.linalg.norm(x_lu - x_gs, np.inf)

print(f" n={n}, p={p}, r={r}, t={t}")
print(f"LU:     невязка = {res_lu:.3e}")
print(f"Зейдель: итераций = {k}, невязка = {res_gs:.3e}")
print(f"||x_LU - x_GS||∞ = {diff:.3e}")
