import numpy as np

def calc_stats(x, y):
    classes = np.unique(y)
    stats = {}
    for c in classes:
        features = x[y == c]
        stats[c] = {
            "mean": np.mean(features, axis=0),
            "var": np.var(features, axis=0)
        }
    return stats

def gaussian(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def calc_priors(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    priors = {c: counts[i] / total for i, c in enumerate(classes)}
    return priors

def predict(x, stats, priors):
    probabilities = {}
    for c, stat in stats.items():
        probabilities[c] = priors[c]
        for i in range(len(x)):
            probabilities[c] *= gaussian(x[i], stat["mean"][i], stat["var"][i])
    return max(probabilities, key=probabilities.get)

def predict_all(X, stats, priors):
    return [predict(x, stats, priors) for x in X]


data = []

with open("Data.txt", "r") as file:
    for line in file:
        row = [float(value) for value in line.strip().split(",")]
        data.append(row)
data_array = np.array(data)


x = data_array[:, :2]
y = data_array[:, 2]

# Training
stats = calc_stats(x, y)
priors = calc_priors(y)

def convert_to_native(predictions):
    return [float(pred) for pred in predictions]

predictions = convert_to_native(predict_all(x, stats, priors))

print("Predictions:", predictions)
print("Actual:", y)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")