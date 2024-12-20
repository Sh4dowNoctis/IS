import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    x = np.linspace(0.1, 1, 20)
    y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2
    return x, y

# Activation function (hyperbolic tangent)
def tanh(x):
    return np.tanh(x)

# Derivative of activation function
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Parameters
input_size = 1
hidden_size = 4
output_size = 1
learning_rate = 0.05
epochs = 10000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

# Training data
x_train, y_train = generate_data()
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# To store error over epochs for plotting
errors = []

# Training loop with backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(x_train, W1) + b1
    hidden_output = tanh(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    output = output_input  # Linear activation for output layer

    # Compute the error
    error = y_train - output
    errors.append(np.mean(np.abs(error)))  # Storing error for visualization

    # Backpropagation
    output_delta = error  # Derivative of linear activation is 1
    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * tanh_derivative(hidden_output)

    # Update weights and biases
    W2 += learning_rate * hidden_output.T.dot(output_delta)
    b2 += learning_rate * np.sum(output_delta, axis=0)
    W1 += learning_rate * x_train.T.dot(hidden_delta)
    b1 += learning_rate * np.sum(hidden_delta, axis=0)

# Testing
predicted_outputs = []
for x in x_train:
    hidden_output = tanh(np.dot(x, W1) + b1)
    output = np.dot(hidden_output, W2) + b2
    predicted_outputs.append(output[0])

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(errors, label="Training Error")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training Error Over Time")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_train, y_train, label="Target Function", marker='o', linestyle="--")
plt.plot(x_train, predicted_outputs, label="Predicted Function", marker='x')
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.title("Target vs. Predicted Outputs")
plt.legend()

plt.tight_layout()
plt.show()
