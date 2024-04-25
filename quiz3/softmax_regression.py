import numpy as np

def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z))).T


def one_hot_encoding(ys):
    if len(ys) == 0:
        return np.array([])
    k = np.max(ys)
    array = []
    for y in ys:
        row = np.zeros(k + 1, dtype=int)
        row[y] = 1
        array.append(row)
    return np.array(array).reshape(len(ys),k+1)
        

def softmax_regression(xs, ys, learning_rate, num_iterations):
   
    tau = one_hot_encoding(ys)
    # number of feature = xs.shape[1], number of classes = tau.shape[1]
    theta = np.zeros((xs.shape[1], tau.shape[1]))
    bias = np.zeros(tau.shape[1])

    for _ in range(num_iterations):
        total_error = []
        total_gradient = []

        for i in range(xs.shape[0]):
            z = np.dot(xs[i], theta) + bias
            o = softmax(z)
            error = o - tau[i]
            gradient = error * xs[i]
            total_error.append(error)
            total_gradient.append(gradient)
        
            theta = theta + (learning_rate * np.sum(total_gradient)) 
            bias = bias + (learning_rate * np.sum(total_error)) 

    print(theta, bias)
    def model(xs, theta=theta, bias=bias):
        z = np.dot(xs, theta) + bias
        soft = softmax(z)
        return soft.argmax(axis=1)
    
    return model

            

#Test

training_data = np.array([
    (0.17, 0),
    (0.79, 0),
    (2.66, 2),
    (2.81, 2),
    (1.58, 1),
    (1.86, 1),
    (2.97, 2),
    (2.70, 2),
    (1.64, 1),
    (1.68, 1)
])

xs = training_data[:,0].reshape((-1, 1)) # a 2D n-by-1 array
ys = training_data[:,1].astype(int)      # a 1D array of length n

h = softmax_regression(xs, ys, 0.05, 750)

test_inputs = [(1.30, 1), (2.25, 2), (0.97, 0), (1.07, 1), (1.51, 1)]
print(f"{'prediction':^10}{'true':^10}")
for x, y in test_inputs:
    print(f"{h(x):^10}{y:^10}")
# prediction   true   
#     1         1     
#     2         2     
#     0         0     
#     1         1     
#     1         1 