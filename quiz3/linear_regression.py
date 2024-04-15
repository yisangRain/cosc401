"""
The previous function is limited to a single feature variable; often, we need to consider multiple features.

Define a function linear_regression(xs, ys) that takes two numpy arrays as inputs. 
The first parameter, xs is the design matrix, an n×d array representing the input part of the training data. 
The second parameter, ys is a one-dimensional array with n elements, representing the output part of the training data. 
The function should return a one-dimensional array θ, with d + 1 elements, containing the least-squares regression 
coefficients for the features, with the first "extra" value being the intercept.

Use numpy for calculations, but avoid using any built-in least-squares solvers. 
Instead, apply the closed-form solution (normal equations) provided in the lecture notes.
"""
import numpy as np

def linear_regression(xs, ys):
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


# Test
xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
print(linear_regression(xs, ys))

#[1. 2.]

xs = np.array([[1, 2, 3, 4],
               [6, 2, 9, 1]]).T
ys = np.array([7, 5, 14, 8]).T
print(linear_regression(xs, ys))

# [-1.  2.  1.]





