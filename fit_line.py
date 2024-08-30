# Linear Solver
def my_linfit(x, y):
    a_num = sum(x * y) - sum(x) * sum(y)
    a_den = sum(x**2) - (sum(x))**2
    a = a_num / a_den

    b = sum(y) - a * sum(x)

    return a, b

# Main
import matplotlib.pyplot as plt
import numpy as np

x = np.random.uniform(-2, 5, 10)
y = np.random.uniform(0, 3, 10)

a, b = my_linfit(x, y)

plt.plot(x, y, 'kx')
xp = np.arange(-2, 5, 0.1)
plt.plot(xp, a * xp + b, 'r-')

print(f"My fit: a={a} and b={b}")
plt.show()
