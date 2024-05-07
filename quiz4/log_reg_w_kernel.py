import numpy as np

def monomial_kernel(d):
    def k(x, y, d=d):
        phi_x_y = 0
        prod_xy = np.dot(x.T,y)

        for n in range(d+1):
            phi_x_y += (prod_xy ** n)

        return phi_x_y
        
    return k

def rbf_kernel(sigma):
    def k(x, y, sigma=sigma):
        numerator = np.linalg.norm(x - y) **2
        denominator = 2 * (sigma ** 2)
        
        return np.exp(-numerator / denominator)

    return k

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

def logistic_regression_with_kernel(X, y, k, alpha, iterations):
    theta = np.zeros(X.shape[1])

    for _ in range(iterations):
        kernel = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            x = X[i]
            z = y[i]
         
            s_z = z * k(theta, x)
            kernel = sigmoid(s_z) 

            #compute gradients
            theta += alpha * kernel

    def model(x, theta=theta):
        return sigmoid(np.dot(theta, x))
    return model



# The underlining function is y = 2x - 1, and examples are positive if y >= 0 
# With m >= 1 we should be able to converge to the exact function
# training_examples = np.array([
#     (0.99, 1),
#     (0.58, 1),
#     (0.01, 0),
#     (0.33, 0),
#     (0.09, 0)
# ])

# X = training_examples[:,[0]]
# y = training_examples[:,1]

# h = logistic_regression_with_kernel(X, y, monomial_kernel(1), 0.1, 500)

# inputs = np.array([0.88, 0.15, 0.82, 0.07, 0.52])

# print(f"{'x' : ^5}{'label' : >5}{'true': >5}")
# for x in inputs:
#     print(f"{x : <5}{int(h(x)) : ^5}{int(2 * x - 1 >= 0) : ^5}")
#   x  label true
# 0.88   1    1
# 0.15   0    0
# 0.82   1    1
# 0.07   0    0
# 0.52   1    1

np.random.seed(0xc05c401)
# # Some random samples
X = 3 * np.random.random((100, 2)) - 1.5

# # Examples with be + if they live in the unit circle and - otherwise
y = np.array([x[0]**2 + x[1]**2 < 1 for x in X]) 

# # Quadratic monomials should find the separating plane in question 3
h = logistic_regression_with_kernel(X, y, monomial_kernel(2), 0.01, 750)

np.random.seed(0xbee5)
# Unseen inputs
test_inputs =  3 * np.random.random((10, 2)) - 1.5

print("h(x) label")
for x in test_inputs:
    output = '+' if h(x) else '-'
    true = '+' if x[0]**2 + x[1]**2 < 1 else '-'
    print(f"{output: >2}{true: >6}")
# h(x) label
#  -     -
#  +     +
#  -     -
#  +     +
#  -     -
#  +     +
#  -     -
#  -     -
#  +     +
#  -     -