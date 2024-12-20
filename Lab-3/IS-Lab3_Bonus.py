import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 1, 1/22)
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

c1, r1 = 0.3, 0.1
c2, r2 = 0.7, 0.1

def rbf(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

def compute_gradients(x, error, weights, c, r, phi):
    grad_c = (error * weights * (x - c) / (r ** 2)) * phi
    grad_r = (error * weights * ((x - c) ** 2) / (r ** 3)) * phi
    return grad_c, grad_r

learning_rate_c = 0.01
learning_rate_r = 0.01

weights = np.random.rand(3)

learning_rate_w = 0.01
epochs = 1000

for epoch in range(epochs):
    phi1 = rbf(x, c1, r1)
    phi2 = rbf(x, c2, r2)
    phi = np.vstack([phi1, phi2, np.ones_like(x)]).T
    y_pred = phi @ weights
    error = y - y_pred
    weights += learning_rate_w * phi.T @ error

    for i in range(len(x)):
        grad_c1, grad_r1 = compute_gradients(x[i], error[i], weights[0], c1, r1, phi1[i])
        c1 += learning_rate_c * grad_c1
        r1 += learning_rate_r * grad_r1

        grad_c2, grad_r2 = compute_gradients(x[i], error[i], weights[1], c2, r2, phi2[i])
        c2 += learning_rate_c * grad_c2
        r2 += learning_rate_r * grad_r2

phi1 = rbf(x, c1, r1)
phi2 = rbf(x, c2, r2)
phi = np.vstack([phi1, phi2, np.ones_like(x)]).T
y_final = phi @ weights

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', label="Desired Output")
plt.plot(x, y_final, 'x--', label="RBF Network Output")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.title("Radial Basis Function Network Approximation with Adaptive Centers and Spreads")
plt.legend()
plt.grid()
plt.show()
