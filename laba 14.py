import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 0.2
N = 50
Nt = 2000
h = L / N
tau = T / Nt

x = np.linspace(0, L, N + 1)

p_true = np.sin(np.pi * x)

def forward(p):
    u = np.zeros((Nt + 1, N + 1))
    for n in range(Nt):
        u_n = u[n]
        u_next = u[n + 1]
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_next[1:-1] = (
            u_n[1:-1]
            + tau * (
                (u_n[2:] - 2.0 * u_n[1:-1] + u_n[:-2]) / h**2
                + p[1:-1]
            )
        )
    return u

u_true = forward(p_true)
g = u_true[-1].copy()

def adjoint(u, g):
    w = np.zeros((Nt + 1, N + 1))
    w[-1] = u[-1] - g
    for n in range(Nt, 0, -1):
        w_n = w[n]
        w_prev = w[n - 1]
        w_prev[0] = 0.0
        w_prev[-1] = 0.0
        w_prev[1:-1] = (
            w_n[1:-1]
            + tau * (
                (w_n[2:] - 2.0 * w_n[1:-1] + w_n[:-2]) / h**2
            )
        )
    return w

def gradient(p, alpha):
    u = forward(p)
    w = adjoint(u, g)
    grad = tau * np.sum(w, axis=0) + alpha * p
    return grad, u

p = np.zeros_like(x)
alpha = 1.0    # регуляризация (можно менять)
n_iters = 40
history = []

for k in range(n_iters):
    grad, u = gradient(p, alpha)
    p -= 0.5 * grad
    J = 0.5 * np.sum((u[-1] - g)**2) * h + 0.5 * alpha * np.sum(p**2) * h
    history.append(J)
    print(f"iter {k+1}, J = {J:.6e}")

u_rec = u

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(x, p_true, label="p_true")
plt.plot(x, p, label="p_rec", linestyle="--")
plt.title("Источник p(x) с регуляризацией")
plt.xlabel("x")
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(x, g, label="g(x)=u_true")
plt.plot(x, u_rec[-1], label="u_rec", linestyle="--")
plt.title("Конечное состояние")
plt.xlabel("x")
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
plt.semilogy(history)
plt.title("Спуск функционала Jα(p)")
plt.xlabel("итерация")
plt.grid(True)

plt.tight_layout()
plt.show()
