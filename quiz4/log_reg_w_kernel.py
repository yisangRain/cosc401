import numpy as np
import warnings

warnings.filterwarnings('ignore')

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

    n_samples, _ = X.shape
    bias = 0
    kernel_matrix = np.zeros((n_samples, n_samples))
    beta = np.zeros(n_samples)

    #create kernel matrix
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i][j] = k(X[i], X[j])
    
    for _ in range(iterations):
        for i in range(n_samples):
            total = 0
            for j in range(n_samples):
                total += beta[j] * kernel_matrix[i][j]
            total += bias
            sigmoid_value = sigmoid(total)
            t = y[i]

            beta += kernel_matrix[i] * alpha * (t - sigmoid_value)
                     
            bias += (alpha * (t - (sigmoid_value))) 

    def model(x, beta=beta, bias=bias, k=k, ref=X):
        z = np.sum([k(ref[i], x) * beta[i] for i in range(ref.shape[0])]) + bias 
        return round(sigmoid(z))
    return model


def test1():

    # The underlining function is y = 2x - 1, and examples are positive if y >= 0 
    # With m >= 1 we should be able to converge to the exact function
    training_examples = np.array([
        (0.99, 1),
        (0.58, 1),
        (0.01, 0),
        (0.33, 0),
        (0.09, 0)
    ])

    X = training_examples[:,[0]]
    y = training_examples[:,1]

    h = logistic_regression_with_kernel(X, y, monomial_kernel(1), 0.1, 500)

    inputs = np.array([0.88, 0.15, 0.82, 0.07, 0.52])
    # inputs = np.array([0.88])

    print(f"{'x' : ^5}{'label' : >5}{'true': >5}")
    for x in inputs:
        print(f"{x : <5}{int(h(x)) : ^5}{int(2 * x - 1 >= 0) : ^5}")
        # print("value:", h(x))
    #   x  label true
    # 0.88   1    1
    # 0.15   0    0
    # 0.82   1    1
    # 0.07   0    0
    # 0.52   1    1

def test2():

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
    # # h(x) label
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

def test3():
    training_examples = np.array([
        (-0.84, 0),
        (0.61, 0),
        (0.43, 1),
        (-0.85, 0),
        (0.56, 0),
        (0.34, 1)
    ])

    X = training_examples[:,[0]]
    y = training_examples[:,1]

    # The underlining function is y = x^3 - x^2 - x/4 + 1/4, and examples are positive if y >= 0
    # With m >= 3 we should be able to converge to the exact function

    h = logistic_regression_with_kernel(X, y, monomial_kernel(3), 0.1, 900)

    inputs = np.array([0.34, -0.72, 0.84, -0.35, 0.3])

    print(f"{'x' : ^5}{'label' : >5}{'true': >5}")
    for x in inputs:
        print(f"{x : <5}{int(h(x)) : ^5}{int(x**3 - x**2 - 0.25*x + 0.25 >= 0) : ^5}")
    #   x  label true↩
    # 0.34   1    1↩
    # -0.72  0    0↩
    # 0.84   0    0↩
    # -0.35  1    1↩
    # 0.3    1    1

def test4():
    	
    f = lambda x, y, z, w:  int(x*y*z - y**2*z*w/4 + x**4*w**3/8- y*w/2 >= 0)

    training_examples = [
        ([0.254, 0.782, 0.254, 0.569], 0),
        ([0.237, 0.026, 0.237, 0.638], 0),
        ([0.814, 0.18, 0.814, 0.707], 1),
        ([0.855, 0.117, 0.855, 0.669], 1),
        ([0.776, 0.643, 0.776, 0.628], 1),
        ([0.701, 0.71, 0.701, 0.982], 0),
        ([0.443, 0.039, 0.443, 0.356], 1),
        ([0.278, 0.105, 0.278, 0.158], 0),
        ([0.394, 0.203, 0.394, 0.909], 0),
        ([0.83, 0.197, 0.83, 0.779], 1),
        ([0.277, 0.415, 0.277, 0.357], 0),
        ([0.683, 0.117, 0.683, 0.455], 1),
        ([0.421, 0.631, 0.421, 0.015], 1)
    ]

    X, y = map(np.array, zip(*training_examples))

    h = logistic_regression_with_kernel(X, y, monomial_kernel(10), 0.01, 4000)

    test_examples = [
        ([0.157, 0.715, 0.787, 0.644], 0),
        ([0.79, 0.279, 0.761, 0.886], 1),
        ([0.903, 0.544, 0.138, 0.925], 0),
        ([0.129, 0.01, 0.493, 0.658], 0),
        ([0.673, 0.526, 0.672, 0.489], 1),
        ([0.703, 0.716, 0.088, 0.674], 0),
        ([0.276, 0.174, 0.69, 0.358], 1),
        ([0.199, 0.812, 0.825, 0.653], 0),
        ([0.332, 0.721, 0.148, 0.541], 0),
        ([0.51, 0.956, 0.023, 0.249], 0)
    ]
    print(f"{'x' : ^30}{'prediction' : ^11}{'true' : ^6}")
    for x, y in test_examples:
        print(f"{str(x) : ^30}{int(h(x)) : ^11}{y : ^6}")
    #             x               prediction  true
    # [0.157, 0.715, 0.787, 0.644]      0       0
    # [0.79, 0.279, 0.761, 0.886]       1       1
    # [0.903, 0.544, 0.138, 0.925]      0       0
    # [0.129, 0.01, 0.493, 0.658]       0       0
    # [0.673, 0.526, 0.672, 0.489]      1       1
    # [0.703, 0.716, 0.088, 0.674]      0       0
    # [0.276, 0.174, 0.69, 0.358]       1       1
    # [0.199, 0.812, 0.825, 0.653]      0       0
    # [0.332, 0.721, 0.148, 0.541]      0       0
    # [0.51, 0.956, 0.023, 0.249]       0       0

test1()
test2()
test3()
test4()