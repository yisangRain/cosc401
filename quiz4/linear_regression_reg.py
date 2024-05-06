import numpy as np

def linear_regression(xs, ys, basis_functions=None, penalty=0):

    if basis_functions == []:
        return [np.average(ys)]
    
    if basis_functions != None:
        matrice = []

        for f in basis_functions:
            temp = [f(x) for x in xs]
            matrice.append(np.array(temp))

        xs = np.array(matrice).T
    
    ones_column = np.ones((xs.shape[0], 1))
    xs_with_intercept = np.concatenate((ones_column, xs), axis=1)
    
    # Calculate the normal equations: θ = (X^T*X + lambda*identity)^-1 * X^T * y
    left_matrix = np.linalg.inv(np.dot(xs_with_intercept.T, xs_with_intercept) + (penalty * np.identity(xs_with_intercept.shape[1])))
    right_matrix = np.dot(xs_with_intercept.T, ys)
    theta = np.dot(left_matrix, right_matrix)
    
    return theta
    

xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)

print(linear_regression(xs, ys), end="\n\n")

with np.printoptions(precision=5, suppress=True):
    print(linear_regression(xs, ys, penalty=0.1))	

# [1. 2.]

# [0.98113 1.99963]

xs = np.arange(-1, 1, 0.1).reshape(-1, 1)
m, n = xs.shape
# Some true function plus some noise:
ys = (xs**2 - 3*xs + 2 + np.random.normal(0, 0.5, (m, 1))).ravel()
print(ys)

functions = [lambda x: x[0], lambda x: x[0]**2, lambda x: x[0]**3, lambda x: x[0]**4,
      lambda x: x[0]**5, lambda x: x[0]**6, lambda x: x[0]**7, lambda x: x[0]**8]

# for penalty in [0, 0.01, 0.1, 1, 10]:
for penalty in [0]:
    with np.printoptions(precision=5, suppress=True):
        print(linear_regression(xs, ys, basis_functions=functions, penalty=penalty)
              .reshape((-1, 1)), end="\n\n")

# [[  2.37235]
#  [ -1.90772]
#  [ -5.70136]
#  [ -6.15984]
#  [ 41.15948]
#  [  7.49754]
#  [-73.36604]
#  [ -2.5122 ]
#  [ 39.33253]]

# [[ 2.17341]
#  [-2.61201]
#  [ 1.74149]
#  [-1.47769]
#  [-0.25131]
#  [ 0.65549]
#  [-1.13712]
#  [-0.38722]
#  [ 0.33897]]

# [[ 2.21863]
#  [-2.61111]
#  [ 1.23183]
#  [-1.00527]
#  [ 0.12226]
#  [-0.1989 ]
#  [-0.33258]
#  [-0.01852]
#  [-0.34059]]

# [[ 2.18272]
#  [-2.15183]
#  [ 0.72762]
#  [-0.93658]
#  [ 0.23555]
#  [-0.45143]
#  [ 0.02779]
#  [-0.23089]
#  [-0.05509]]

# [[ 1.55112]
#  [-1.06524]
#  [ 0.55608]
#  [-0.62681]
#  [ 0.34315]
#  [-0.4415 ]
#  [ 0.2563 ]
#  [-0.34422]
#  [ 0.21365]]
