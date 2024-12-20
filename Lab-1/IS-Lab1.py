import numpy as np
import random

def Training(w, learningRate, e, x):
    if e == 0:
        return w
    return (w + learningRate * e * x)

def TotalError(errors):
    if len(errors) == 0:
        return 1
    errors = np.abs(errors)
    return np.sum(errors)
    
w1 = random.random()
w2 = random.random()
b  = random.random()
learningRate = 0.2
y = 0

errors = []
data = []

with open("Data.txt", "r") as file:
    for line in file:
        row = [float(value) for value in line.strip().split(",")]
        data.append(row)

data_array = np.array(data)

random_indices = random.sample(range(len(data_array)), 5)
trainingRows = data_array[random_indices]
testingRows = np.delete(data_array, random_indices, axis=0)

i = 0
while ( i < len(trainingRows)
    or TotalError(errors) != 0):
    if (i >= len(trainingRows)):
        errors = []
        i = 0
    x1, x2, d = trainingRows[i]
    v = x1 * w1 + x2 * w2 + b

    if v > 1:
        y = 1
    else:
        y = -1

    e = float(d) - y
    
    errors.append(e)
    w1 = Training(w1, learningRate, e, x1)
    w2 = Training(w2, learningRate, e, x2)
    b  = Training(b,  learningRate, e, 1)


    i += 1


print("After training of our perceptron we tested it multiple images, here are the results:")
z = 0
while z < len(testingRows):
    x1, x2, d = testingRows[z]
    v = x1 * w1 + x2 * w2 + b

    if v > 1:
        y = 1
    else:
        y = -1
    e = float(d) - y
    z += 1
    if (d == 1):
        print(f'Image nb{z} is supposed to be an apple, the result is: {y == d}')
    else:
        print(f'Image nb{z} is supposed to be an pear, the result is: {y == d}')
