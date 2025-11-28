import math
import matplotlib.pyplot as plt

a = 1.0
L = 1.0
N = 100
h = L / N

c = 0.9
tau = c * h / a

T = 1.0
kmax = int(T / tau)

def u_exact(x, t):
    return math.sin(math.pi * x) * math.cos(math.pi * t)

x = [j * h for j in range(N + 1)]

u_prev = [0.0] * (N + 1)
u_curr = [0.0] * (N + 1)
u_next = [0.0] * (N + 1)

for j in range(N + 1):
    u_prev[j] = u_exact(x[j], 0.0)

t1 = tau
for j in range(N + 1):
    u_curr[j] = u_exact(x[j], t1)

u_prev[0] = 0.0
u_prev[N] = 0.0
u_curr[0] = 0.0
u_curr[N] = 0.0

for n in range(1, kmax):
    for j in range(1, N):
        u_next[j] = (2.0 * u_curr[j]
                     - u_prev[j]
                     + c * c * (u_curr[j + 1] - 2.0 * u_curr[j] + u_curr[j - 1]))
    u_next[0] = 0.0
    u_next[N] = 0.0
    u_prev, u_curr, u_next = u_curr, u_next, u_prev

t_final = kmax * tau

max_err = 0.0
for j in range(N + 1):
    exact = u_exact(x[j], t_final)
    err = abs(u_curr[j] - exact)
    if err > max_err:
        max_err = err

print("Параметры сетки:")
print("N =", N)
print("h =", h)
print("tau =", tau)
print("число шагов по времени kmax =", kmax)
print("финальное время t_final =", t_final)
print()
print("Максимальная погрешность по сети (u_num - u_exact_max) =", max_err)
print()
print("Несколько значений численного решения и точного решения при t = t_final:")
for j in range(0, N + 1, 20):
    exact = u_exact(x[j], t_final)
    print(f"x = {x[j]:.3f}, u_num = {u_curr[j]:.6f}, u_exact = {exact:.6f}")

y_exact = [u_exact(xj, t_final) for xj in x]

plt.plot(x, u_curr, label="Численное решение")
plt.plot(x, y_exact, linestyle="--", label="Точное решение")
plt.xlabel("x")
plt.ylabel("u(x, t_final)")
plt.title("Сравнение численного и точного решений")
plt.legend()
plt.grid(True)
plt.show()