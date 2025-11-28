import numpy as np
import matplotlib.pyplot as plt

c = 1.0
L = 1.0
Nx = 200
h = L / Nx

T = 1.0
tau = 0.003
Nt = int(T / tau)

lam = c * tau / h
print("lambda =", lam)

x = np.linspace(0.0, L, Nx + 1)

u = np.zeros_like(x)
u[(x >= 0.2) & (x <= 0.4)] = 1.0

u_new = np.zeros_like(u)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_xlim(0.0, L)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Уравнение переноса: t = 0.0")

for n in range(Nt):
    u_new[0] = 0.0
    u_new[1:] = (1.0 - lam) * u[1:] + lam * u[:-1]

    u, u_new = u_new, u

    if n % 5 == 0:
        t = (n + 1) * tau
        line.set_ydata(u)
        ax.set_title(f"Уравнение переноса: t = {t:.3f}")
        plt.pause(0.001)

plt.ioff()
plt.show()
