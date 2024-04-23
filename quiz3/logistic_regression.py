import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  

def logistic_regression(xs, ys, alpha, num_iteration):
    #set initial values
    theta = np.zeros(xs.shape[1])
    bias = 0

    for epoch in range(num_iteration):
        for i in range(xs.shape[0]):
            x = xs[i]
            y = ys[i]

            prediction = sigmoid(np.dot(theta.T, x) + bias)

            #compute gradients
            gradient_theta = x * (y - prediction)

            gradient_bias = y - prediction

            theta = theta - alpha * gradient_theta
            bias = bias - alpha * gradient_bias
    
    def model(x, theta=theta, bias=bias):
        return sigmoid(np.dot(theta.T, x) + bias)
    return model



#Test

# The input has one dimension.

# Training data:
xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1)) # an n-by-1 matrix of inputs
ys = np.array([0, 0, 0, 1, 1, 1]) # n-element vector of outputs

# Fitting a model
model = logistic_regression(xs, ys, 0.05, 10000)

test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))

# 0.00
# 0.00
# 0.00
# 0.02
# 0.11
# 0.43
# 0.83
# 0.97
# 0.99
# 1.00
# 1.00
# 1.00
# 1.00

xs = np.array(
    [0.50,0.75,1.00,1.25,1.50,
     1.75,1.75,2.00,2.25,2.50,
     2.75,3.00,3.25,3.50,4.00,
     4.25,4.50,4.75,5.00,5.50]).reshape((-1, 1))

ys = np.array([0,0,0,0,0,
               0,1,0,1,0,
               1,0,1,0,1,
               1,1,1,1,1])

model = logistic_regression(xs, ys, 0.02, 5000)
sse = 0
output = []
expected = [0.02, 0.03, 0.07, 0.14, 0.25, 0.42, 0.60, 0.77, 0.87, 0.94, 0.97, 0.99]
for i, x in enumerate(np.arange(0, 6, 0.5).reshape(-1,1)): 
    output.append(np.array(model(x)).item())
    sse += (expected[i] - output[-1]) ** 2

tolerance = 1e-3
if sse / len(expected) < tolerance:
    print("OK")
else:
    print("The error is too high.")
    print("The expected output is: ", expected)
    print("The output of the trained model is:", output)

# OK


