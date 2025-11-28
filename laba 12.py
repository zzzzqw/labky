import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 50
M = 50
hx = 1.0 / N
hy = 1.0 / M

x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, M+1)
X, Y = np.meshgrid(x, y)

f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

u = np.zeros((M+1, N+1))

eps = 1e-5
max_iter = 5000

for k in range(max_iter):
    max_err = 0.0
    
    for j in range(1, M):
        for i in range(1, N):
            u_old = u[j, i]
            u[j, i] = 0.25 * (
                u[j+1, i] + u[j-1, i] +
                u[j, i+1] + u[j, i-1] -
                hx**2 * f[j, i]
            )
            max_err = max(max_err, abs(u[j, i] - u_old))

    if max_err < eps:
        print("Converged at iteration:", k+1)
        break

u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u - u_exact)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, u, cmap="viridis")
ax.set_title("Численное решение u(x,y)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, error, cmap="inferno")
ax2.set_title("Абсолютная ошибка |u - u_exact|")

plt.show()
