import numpy as np
import matplotlib.pyplot as plt

# --- Параметры сетки (можно менять) ---
a = float(input("Введите a (например 0.0): ").strip() or "0.0")
b = float(input("Введите b (например 1.0): ").strip() or "1.0")
n = int(input("Введите n (>=4, например 100): ").strip() or "100")
assert n >= 4, "n должно быть ≥ 4"

# --- Функция и точные производные ---
def y_fun(x):    # y(x) = x cosh x
    return x * np.cosh(x)

def dy_true(x):  # y'(x) = cosh x + x sinh x
    return np.cosh(x) + x * np.sinh(x)

def d2y_true(x): # y''(x) = 2 sinh x + x cosh x
    return 2.0 * np.sinh(x) + x * np.cosh(x)

# --- Сетка и значения функции ---
x = np.linspace(a, b, n + 1)
h = (b - a) / n
f = y_fun(x)

# --- ЧИСЛЕННАЯ ПЕРВАЯ ПРОИЗВОДНАЯ (O(h^2)) ---
d1 = np.empty_like(f)
# левый край
d1[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*h)
# центральная формула
d1[1:-1] = (f[2:] - f[:-2]) / (2*h)
# правый край
d1[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2*h)

# --- ЧИСЛЕННАЯ ВТОРАЯ ПРОИЗВОДНАЯ (O(h^2)) ---
d2 = np.empty_like(f)
# левый край
d2[0] = (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / (h*h)
# центральная формула
d2[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / (h*h)
# правый край
d2[-1] = (2*f[-1] - 5*f[-2] + 4*f[-3] - f[-4]) / (h*h)

# --- Точные значения и ошибки ---
dy_ex  = dy_true(x)
d2y_ex = d2y_true(x)

e1 = dy_ex  - d1
e2 = d2y_ex - d2
e1_max, e1_rms = np.max(np.abs(e1)), np.sqrt(np.mean(e1**2))
e2_max, e2_rms = np.max(np.abs(e2)), np.sqrt(np.mean(e2**2))

print("\n=== Результаты (Задание 5, вариант 6) ===")
print(f"[a,b]=[{a},{b}], n={n}, h={h:.4e}")
print(f"Первая  производная: e_max = {e1_max:.3e}, RMS = {e1_rms:.3e}")
print(f"Вторая   производная: e_max = {e2_max:.3e}, RMS = {e2_rms:.3e}")

# --- ВИЗУАЛИЗАЦИЯ — ОДНО ОКНО, 3 ПОДПЛОТА ---
import matplotlib.ticker as mticker
fig = plt.figure(figsize=(10, 8))

# 1) y'(x)
ax1 = plt.subplot(3,1,1)
ax1.plot(x, dy_ex, lw=2, label="y'(x) — точная")
ax1.plot(x, d1,  '--', lw=2, label="y'(x) — численная (O(h²))")
ax1.set_title(f"Численное дифференцирование y(x)=x·cosh(x)  |  h={h:.2e}")
ax1.set_ylabel("y'(x)")
ax1.grid(True); ax1.legend()

# 2) y''(x)
ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(x, d2y_ex, lw=2, label="y''(x) — точная")
ax2.plot(x, d2,    '--', lw=2, label="y''(x) — численная (O(h²))")
ax2.set_ylabel("y''(x)")
ax2.grid(True); ax2.legend()

# 3) Ошибки
ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(x, e1, lw=1.6, label="ошибка y' : exact − num")
ax3.plot(x, e2, lw=1.6, label="ошибка y'': exact − num")
ax3.axhline(0, color='k', lw=1)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))
ax3.set_xlabel("x")
ax3.set_ylabel("ошибка")
ax3.grid(True); ax3.legend()
ax3.set_title(f"e1_max={e1_max:.2e}, RMS={e1_rms:.2e}   |   e2_max={e2_max:.2e}, RMS={e2_rms:.2e}")

plt.tight_layout()
plt.show()
