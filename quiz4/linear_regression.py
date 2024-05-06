import numpy as np

def base_linear_regression(xs, ys):
    """
    Perform linear regression using the closed-form solution.
    
    Parameters:
    xs (numpy array): The design matrix, an n×d array representing the input part of the training data.
    ys (numpy array): A one-dimensional array with n elements, representing the output part of the training data.
    
    Returns:
    numpy array: A one-dimensional array θ, with d + 1 elements, containing the least-squares regression coefficients.
    """
    # Add a column of ones to the design matrix for the intercept
    ones_column = np.ones((xs.shape[0], 1))
    xs_with_intercept = np.concatenate((ones_column, xs), axis=1)
    
    # Calculate the normal equations: θ = (X^T*X)^-1 * X^T * y
    #Page 15 of the course book
    left_matrix = np.linalg.inv(np.dot(xs_with_intercept.T, xs_with_intercept))
    right_matrix = np.dot(xs_with_intercept.T, ys)
    theta = np.dot(left_matrix, right_matrix)
    
    return theta


def linear_regression(xs, ys, basis_functions=None):
    if basis_functions == None:
        return base_linear_regression(xs, ys)
    
    if len(basis_functions) == 0:
        return [np.average(ys)]

    matrice = []

    for f in basis_functions:
        temp = [f(x) for x in xs]
        matrice.append(np.array(temp))

    new_xs = np.array(matrice).T

    return base_linear_regression(new_xs, ys)


    




xs = np.arange(5).reshape((-1, 1))
ys = np.array([3, 6, 11, 18, 27])
# Can you see y as a function of x? [hint: it's quadratic.]
functions = [lambda x: x[0], lambda x: x[0] ** 2]
print(linear_regression(xs, ys, functions))

# [3. 2. 1.]


xs = np.array([[1, 2, 3, 4],
               [6, 2, 9, 1]]).T
ys = np.array([7, 5, 14, 8])
print(linear_regression(xs, ys, []) == np.average(ys))

# [ True]