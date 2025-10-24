import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x / (math.log(x) ** 2)

a, b = 1.5, 3.0

#метод прямоугольников
def rectangles_midpoint(f, a, b, n):
    h = (b - a) / n
    s = 0
    x_mid = []
    y_mid = []
    for i in range(n):
        xi = a + (i + 0.5) * h
        s += f(xi)
        x_mid.append(xi)
        y_mid.append(f(xi))
    I = h * s
    return I, x_mid, y_mid, h

#квдратура гаусса
def gauss11(f, a, b):
    t = [0.010885671, 0.056468700, 0.134923997, 0.240451935, 0.365228422, 0.500000000,
         1-0.365228422, 1-0.240451935, 1-0.134923997, 1-0.056468700, 1-0.010885671]
    A = [0.027834284, 0.062790185, 0.093145105, 0.116596882, 0.131402272, 0.136462543,
         0.131402272, 0.116596882, 0.093145105, 0.062790185, 0.027834284]
    s = 0
    for ti, Ai in zip(t, A):
        x = a + (b - a) * ti
        s += Ai * f(x)
    return (b - a) * s

#вычисления
n = 10
I_rect, x_mid, y_mid, h = rectangles_midpoint(f, a, b, n)
I_gauss = gauss11(f, a, b)

print(f"Метод прямоугольников (n={n}): I ≈ {I_rect:.9f}")
print(f"Формула Гаусса (m=11):        I ≈ {I_gauss:.9f}")

#график
x = np.linspace(a, b, 400)
y = [f(xi) for xi in x]

plt.figure(figsize=(10,6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x / (ln x)^2')
plt.fill_between(x, y, color='lightblue', alpha=0.3)

# рисуем прямоугольники (по серединам)
for xi, yi in zip(x_mid, y_mid):
    plt.bar(xi, yi, width=h, color='orange', alpha=0.4, align='center', edgecolor='red')

plt.title("Метод прямоугольников и функция f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()