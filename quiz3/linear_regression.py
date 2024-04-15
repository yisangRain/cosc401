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

