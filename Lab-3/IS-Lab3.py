import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 1, 1/22)
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

c1, r1 = 0.2, 0.15
c2, r2 = 0.85, 0.15

def rbf(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

phi1 = rbf(x, c1, r1)
phi2 = rbf(x, c2, r2)

phi = np.vstack([phi1, phi2, np.ones_like(x)]).T

weights = np.random.rand(3)

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred = phi @ weights
    error = y - y_pred
    weights += learning_rate * phi.T @ error

w1, w2, w0 = weights
y_final = phi @ weights

print(y_final)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', label="Desired Output")
plt.plot(x, y_final, 'x--', label="RBF Network Output")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.title("Radial Basis Function Network Approximation")
plt.legend()
plt.grid()
plt.show()
