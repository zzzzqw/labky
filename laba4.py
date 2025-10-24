import numpy as np
import matplotlib.pyplot as plt

# --- Ввод параметра q пользователем ---
q = float(input("Введите параметр q (1, 2 или 3): "))

# --- Настройки сетки ---
a, b = 0.0, 1.0      # отрезок аппроксимации
n = 50               # количество разбиений
x = np.linspace(a, b, n + 1)

# --- Функция варианта 6 ---
def y_fun(x, q):
    return np.cos(np.power(x, q))

# --- Усреднение и система МНК ---
def mean(v): return np.mean(v)

def fit_quadratic_mls(x, y):
    """Решает систему МНК для φ(x)=c0+c1x+c2x²"""
    m1, m2, m3, m4 = mean(x), mean(x**2), mean(x**3), mean(x**4)
    K01, K11, K21 = mean(y), mean(y*x), mean(y*x**2)
    A = np.array([[1, m1, m2],
                  [m1, m2, m3],
                  [m2, m3, m4]], float)
    b = np.array([K01, K11, K21], float)
    c0, c1, c2 = np.linalg.solve(A, b)
    return c0, c1, c2

def phi(x, c0, c1, c2):
    return c0 + c1*x + c2*(x**2)

# --- Расчёт ---
y = y_fun(x, q)
c0, c1, c2 = fit_quadratic_mls(x, y)

xx = np.linspace(a, b, 1000)
yy = y_fun(xx, q)
pp = phi(xx, c0, c1, c2)
err = yy - pp

e_max = np.max(np.abs(err))
rms = np.sqrt(np.mean(err**2))

# --- Вывод результатов в консоль ---
print("\nРезультаты аппроксимации y(x)=cos(x^q), q=", q)
print(f"c0 = {c0:.8f}")
print(f"c1 = {c1:.8f}")
print(f"c2 = {c2:.8f}")
print(f"e_max = {e_max:.3e}")
print(f"RMS    = {rms:.3e}")

# --- Визуализация (всё в одном окне) ---
plt.figure(figsize=(10, 6))

plt.plot(xx, yy, 'b-', lw=2, label=f"y(x)=cos(x^{int(q)})")
plt.plot(xx, pp, 'r--', lw=2, label="φ(x)=c0+c1x+c2x²")
plt.plot(xx, err, 'm-', lw=1.5, label="ошибка y−φ(x)")

plt.title(f"МНК-аппроксимация y(x)=cos(x^{int(q)}) на [0,1]\n"
          f"e_max={e_max:.2e}, RMS={rms:.2e}")
plt.xlabel("x")
plt.ylabel("значение / ошибка")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
