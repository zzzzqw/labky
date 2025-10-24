import numpy as np
import matplotlib.pyplot as plt

# --- Ввод параметров ---
k = int(input("Введите k (1..4) [например 2]: ").strip() or "2")
m = int(input("Введите m (1..4) [например 2]: ").strip() or "2")
a = float(input("Введите a (по умолчанию 0.0): ").strip() or "0.0")
b = float(input("Введите b (по умолчанию 1.0): ").strip() or "1.0")
n = int(input("Введите n (20..100, например 50): ").strip() or "50")

assert 1 <= k <= 4 and 1 <= m <= 4, "k,m должны быть в {1..4}"
assert n >= 2 and b > a, "нужно n≥2 и b>a"

# --- Функция варианта 6 ---
def y_fun(x, k, m):
    """y(x) = tan^k( (π/4) * x^(1/m) )"""
    u = (np.pi / 4.0) * np.power(x, 1.0 / m)
    return np.tan(u) ** k

# --- Узлы и значения ---
x = np.linspace(a, b, n + 1)
h = (b - a) / n
y = y_fun(x, k, m)
x_half = x[:-1] + 0.5 * h  # полуузлы

# --- Функция: квадратичная интерполяция Ньютона по трём узлам ---
def newton_quad_value(x0, x1, x2, y0, y1, y2, xp):
    f01 = (y1 - y0) / (x1 - x0)
    f12 = (y2 - y1) / (x2 - x1)
    f012 = (f12 - f01) / (x2 - x0)
    dx0 = xp - x0
    dx1 = xp - x1
    return y0 + f01 * dx0 + f012 * dx0 * dx1

# --- Интерполяция в полуузлах ---
P_half = np.empty_like(x_half)
for i in range(n):
    i0, i1, i2 = i - 1, i, i + 1
    if i0 < 0:
        i0, i1, i2 = 0, 1, 2
    if i2 > n:
        i0, i1, i2 = n - 2, n - 1, n
    P_half[i] = newton_quad_value(
        x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], x_half[i]
    )

# --- Ошибки в полуузлах ---
y_half = y_fun(x_half, k, m)
err_half = y_half - P_half
e_max = np.max(np.abs(err_half))
rms = np.sqrt(np.mean(err_half**2))

print("\n=== Результаты (Задание 3, вариант 6) ===")
print(f"k={k}, m={m}, [a,b]=[{a},{b}], n={n}, h={(b-a)/n:.5f}")
print(f"e_max = {e_max:.3e}")
print(f"RMS   = {rms:.3e}")

# --- Для гладкого графика построим «лоскутный» интерполянт ---
xx = np.linspace(a, b, 2000)
PP = np.empty_like(xx)
seg = np.minimum(np.floor((xx - a) / h).astype(int), n - 1)
for j, i in enumerate(seg):
    i0, i1, i2 = i - 1, i, i + 1
    if i0 < 0:
        i0, i1, i2 = 0, 1, 2
    if i2 > n:
        i0, i1, i2 = n - 2, n - 1, n
    PP[j] = newton_quad_value(x[i0], x[i1], x[i2], y[i0], y[i1], y[i2], xx[j])

YY = y_fun(xx, k, m)
ERR = YY - PP

# --- Безопасная подпись функции (без \tfrac и \big) ---
label_func = f"y(x)=tan^{k}((π/4)*x^(1/{m}))"

# --- Визуализация ---
plt.figure(figsize=(10, 6))
plt.plot(xx, YY, lw=2, label=label_func)
plt.plot(xx, PP, '--', lw=2, label="Интерполяция Ньютона (2-й порядок)")
plt.plot(x, y, 'ko', ms=4, label="узлы x_i")
plt.plot(x_half, P_half, 'rs', ms=3, label="P(x_{i+0.5})")
plt.plot(xx, ERR, 'm-', lw=1.2, label="ошибка: y(x) − P(x)")

plt.title(
    f"Интерполирование (Ньютон 2-го порядка) — Вариант 6  |  k={k}, m={m}\n"
    f"e_max={e_max:.2e}, RMS={rms:.2e}   |   [a,b]=[{a},{b}], n={n}"
)
plt.xlabel("x")
plt.ylabel("значение / ошибка")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
