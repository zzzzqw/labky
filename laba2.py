import math
import random
import numpy as np
import matplotlib.pyplot as plt

j = random.randint(1, 4)
k = random.randint(1, 4)
m = random.randint(1, 4)
print(f"Параметры: j={j}, k={k}, m={m}")

def f(x: float) -> float:
    return (1 - x)**(1/j) - math.tan(math.pi * (x**m) / 4)**k

def fprime(x: float) -> float:
    term1 = -(1/j) * (1 - x)**(1/j - 1)
    ang = math.pi * (x**m) / 4
    d_ang = (math.pi / 4) * m * x**(m-1) if not (x == 0 and m == 1) else 0.0
    term2 = k * (math.tan(ang)**(k-1)) * (1/math.cos(ang))**2 * d_ang
    return term1 - term2

a, b = None, None
grid = np.linspace(0.0, 1.0, 1001)
prev_x, prev_f = grid[0], f(grid[0])
for x in grid[1:]:
    try:
        fx = f(x)
    except Exception:
        prev_x, prev_f = x, np.nan
        continue
    if np.isfinite(prev_f) and np.isfinite(fx) and prev_f * fx < 0:
        a, b = prev_x, x
        break
    prev_x, prev_f = x, fx

if a is None:
    print("На [0,1] смены знака не нашли — корень не обнаружен.")
else:
    print(f"Отрезок со сменой знака: [{a:.6f}, {b:.6f}]")

    #(биссекция)
    def bisection(a, b, eps=1e-6):
        fa, fb = f(a), f(b)
        while (b - a) > 2*eps:
            c = (a + b) / 2
            fc = f(c)
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        c = (a + b) / 2
        return c, f(c)

    # Ньютон 
    def newton(x0, eps=1e-6, max_iter=50):
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            dfx = fprime(x)
            if abs(dfx) < 1e-12:
                break
            x_new = x - fx/dfx
            if abs(x_new - x) < eps:
                x = x_new
                break
            x = x_new
        return x, f(x)

    xb, fb = bisection(a, b)
    xn, fn = newton((a+b)/2)

    print(f"[Биссекция] x ≈ {xb:.8f}, f(x) ≈ {fb:.2e}")
    print(f"[Ньютон]    x ≈ {xn:.8f}, f(x) ≈ {fn:.2e}")

    xs = np.linspace(0, 1, 1000)
    ys = []
    for xi in xs:
        try:
            ys.append(f(xi))
        except Exception:
            ys.append(np.nan)

    plt.figure(figsize=(10, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(xs, ys, label="f(x)")
    plt.scatter([xb], [f(xb)], s=70, label=f"Биссекция: {xb:.4f}")
    plt.scatter([xn], [f(xn)], s=70, label=f"Ньютон: {xn:.4f}")
    plt.xlim(0, 1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("f(x) и найденные корни")
    plt.grid(True)
    plt.legend()
    plt.show()
