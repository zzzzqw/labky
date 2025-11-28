import numpy as np
import matplotlib.pyplot as plt

N = 200            
h = 1.0 / N
x = np.linspace(0, 1, N+1)

f = np.zeros(N+1)
i0 = N // 2
f[i0] = 1.0 / h      

A = np.zeros((N+1, N+1))

for i in range(1, N):
    A[i, i-1] = -1
    A[i, i]   =  2
    A[i, i+1] = -1

A[0,0] = 1
A[N,N] = 1
f[0] = 0
f[N] = 0

u = np.linalg.solve(A, f) * (h*h)

plt.figure(figsize=(8,5))
plt.plot(x, u, label="Численное решение")
plt.axvline(0.5, color='r', linestyle='--', label="Источник δ(x-0.5)")
plt.title("Решение задачи  -u''(x)=δ(x-0.5),  u(0)=u(1)=0")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.show()
