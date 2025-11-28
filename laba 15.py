import numpy as np
import matplotlib.pyplot as plt

l = 1.0
T = 0.5

N = 100
h = 2*l / N

M = 200
tau = T / M

z = np.linspace(-l, l, N+1)

def f(t, z):
    return np.exp(-t) * (np.pi**2 - 1) * np.sin(np.pi * z)

def u_exact(t, z):
    return np.exp(-t) * np.sin(np.pi * z)

u = np.zeros((M+1, N+1))
u[0] = u_exact(0, z)

for n in range(M):
    t_next = (n+1)*tau

    A = np.zeros(N+1)
    B = np.zeros(N+1)
    C = np.zeros(N+1)
    d = np.zeros(N+1)

    B[0] = 1.0
    B[-1] = 1.0
    d[0] = 0.0
    d[-1] = 0.0

    for i in range(1, N):
        A[i] = -tau / h**2
        C[i] = -tau / h**2
        B[i] = 1 + 2*tau/h**2
        d[i] = u[n, i] + tau * f(t_next, z[i])

    alpha = np.zeros(N+1)
    beta = np.zeros(N+1)

    alpha[1] = -C[0] / B[0]
    beta[1]  = d[0] / B[0]

    for i in range(1, N):
        denom = B[i] + A[i]*alpha[i]
        alpha[i+1] = -C[i] / denom
        beta[i+1]  = (d[i] - A[i]*beta[i]) / denom

    u[n+1, N] = d[N] / B[N]
    for i in reversed(range(N)):
        u[n+1, i] = alpha[i+1]*u[n+1, i+1] + beta[i+1]

u_num = u[-1]
u_ex = u_exact(T, z)
err = np.abs(u_num - u_ex)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(z, u_ex, label="u_exact")
plt.plot(z, u_num, '--', label="u_num (implicit)")
plt.title("Неявная схема (метод прогонки)")
plt.xlabel("z")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(z, err)
plt.title("Ошибка |u_num - u_exact|")
plt.xlabel("z")
plt.grid(True)

plt.tight_layout()
plt.show()
