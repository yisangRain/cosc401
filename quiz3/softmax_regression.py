import numpy as np

def softmax(z):
    total = sum(np.exp(x) for x in z)
    return np.array([(np.exp(k)/total) for k in z])


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
    theta = np.zeros(xs.shape[1])
    bias = 0
    tau = one_hot_encoding(ys)
    
    for _ in range(num_iterations):
        z = np.append(np.dot(theta, xs.T),bias)

        o = softmax(z)

        grad = np.dot(o - tau, xs)

        theta += learning_rate * grad
        bias += np.mean(learning_rate * (o - tau))
    

    def model(xs, theta=theta, bias=bias):
        return one_hot_encoding(np.append(softmax(np.dot(theta, xs.T), bias)))
    
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