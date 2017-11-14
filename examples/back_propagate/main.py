import numpy as np

# Sigmoid function
def sigmoid(x):
  return 1/(1 + np.exp(-x))

# Derivative
def sigmoid_dev(x):
  return x * (1 - x)

# Input Data
input = np.array([
  [0, 0, 1],
  [1, 1, 1],
  [1, 0, 1],
  [0, 1, 1]
])

# Output Data
output = np.array([
  [0],
  [1],
  [1],
  [0]
])

np.random.seed(1)

# Input weight & Bias
Weight = 2 * np.random.random((3, 1)) - 1
bias = 1

# Learning rate
l_rate = 0.01

# Train
for i in range(60000):
  # Predict output
  # y = W.x + b
  predict_output = sigmoid(np.dot(input, Weight) + bias)

  # Calulate error
  error = predict_output - output
  
  # Show accuracy
  if (i % 10000) == 0:
    print('Error: %s', np.mean(np.abs(error)))
  
  # Calulate delta -> what direction is the target value?
  delta = error * sigmoid_dev(predict_output) * l_rate
  Weight += np.dot(input.T, delta)

# Test
predict = sigmoid(np.dot([[0,1,0]], Weight) + bias)
print(predict)
