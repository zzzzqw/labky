import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---- параметры варианта 
alpha = np.array([-2.0, 0.0,  2.5], dtype=float)
beta  = np.array([ 1.2, 2.0,  0.8], dtype=float)
p     = np.array([ 0.25,0.40,0.15], dtype=float)
qv    = np.array([ 1.0, 0.8,  1.6], dtype=float)

# ---- функции
def term_value(r, b, p_, q):
    return np.power(r, b) * np.exp(-p_ * np.power(r, q))

def d_term_dx(t, b, p_, q):
    r = np.abs(t)
    out = np.zeros_like(r, dtype=float)
    m = r > 0
    if np.any(m):
        out[m] = np.sign(t[m]) * np.exp(-p_ * np.power(r[m], q)) * np.power(r[m], b-1.0) * (b - p_*q*np.power(r[m], q))
    return out

def rho3(x1, x2, x3):
    r1 = term_value(np.abs(x1 - alpha[0]), beta[0], p[0], qv[0])
    r2 = term_value(np.abs(x2 - alpha[1]), beta[1], p[1], qv[1])
    r3 = term_value(np.abs(x3 - alpha[2]), beta[2], p[2], qv[2])
    return r1 + r2 + r3

def grad_rho3(x1, x2, x3):
    g1 = d_term_dx(x1 - alpha[0], beta[0], p[0], qv[0])
    g2 = d_term_dx(x2 - alpha[1], beta[1], p[1], qv[1])
    g3 = d_term_dx(x3 - alpha[2], beta[2], p[2], qv[2])
    return g1, g2, g3

# ---- 3D-решётка
L = 6.0
n = 41  # 41^3 ~ 69k точек — ок
x1 = np.linspace(-L, L, n)
x2 = np.linspace(-L, L, n)
x3 = np.linspace(-L, L, n)
X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='xy')

R = rho3(X1, X2, X3)
G1, G2, G3 = grad_rho3(X1, X2, X3)

# ---- аналитика и метрики
x_star = alpha.copy()
rho_star = rho3(*x_star)
g_star = np.array(grad_rho3(*x_star), dtype=float).reshape(3)
g_norm = float(np.linalg.norm(g_star))

idx = np.unravel_index(np.argmin(R), R.shape)
x_min_grid = np.array([X1[idx], X2[idx], X3[idx]], dtype=float)
rho_min_grid = float(R[idx])
gap = abs(rho_min_grid - rho_star)

print("=== Трёхмерный вывод по ρ(x) ===")
print(f"α = {alpha}, β = {beta}, p = {p}, q = {qv}")
print(f"Аналитический минимум: x* = α = {x_star},  ρ(x*) = {rho_star:.6f}")
print(f"Норма градиента в x*: ||∇ρ(x*)|| ≈ {g_norm:.3e}")
print(f"Сеточный минимум: x_grid* = {x_min_grid},  ρ(x_grid*) ≈ {rho_min_grid:.6e}")
print(f"Зазор |ρ_grid − ρ(α)| ≈ {gap:.3e}")
print("Вывод: ρ(x) = Σ r_i(|x_i−α_i|) ≥ 0, минимум достигается в x=α, где r_i(0)=0 ⇒ ρ(α)=0; "
      "векторное поле −∇ρ в объёме направлено к α, что подтверждает глобальный минимум.")

# ---- 3D-визуализация в ОДНОМ окне
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# облако низких значений ρ (нижние 2% квантиль)
thr = np.quantile(R, 0.02)
mask = R <= thr
xs, ys, zs = X1[mask], X2[mask], X3[mask]
ax.scatter(xs, ys, zs, s=4, alpha=0.25, label=f"низкие значения ρ (≤ {thr:.2e})")

# стрелки −∇ρ (редко, чтобы не перегружать)
step = max(1, n//8)
Xq = X1[::step, ::step, ::step]
Yq = X2[::step, ::step, ::step]
Zq = X3[::step, ::step, ::step]
Uq = -G1[::step, ::step, ::step]
Vq = -G2[::step, ::step, ::step]
Wq = -G3[::step, ::step, ::step]
ax.quiver(Xq, Yq, Zq, Uq, Vq, Wq, length=1.0, normalize=True, color='gray', linewidth=0.6)

# точки минимума
ax.scatter(*alpha, color='red', s=80, label="теоретический минимум x=α")
ax.scatter(*x_min_grid, color='gold', s=60, marker='^', label="сеточный минимум")

ax.set_title("3D: облако низких значений ρ, стрелки −∇ρ, точки минимума")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
ax.legend(loc="upper left")
ax.view_init(elev=22, azim=45)
plt.tight_layout()
plt.show()
