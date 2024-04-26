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
    i, j = xs.shape
    xs = np.hstack((np.ones((i,1)), xs))

    y_hot_encoding = one_hot_encoding(ys)

    #initialise theta with r1 = num. class, r2 = num. features + bias column
    theta = np.zeros((y_hot_encoding.shape[1], j + 1))

    #.......stochastic hahahhahahhahahahhahahahhaha fk
    for _ in range(num_iterations):
        z = np.dot(theta, xs.T)
        predict = softmax(z)

        #gradient
        gradient = np.dot(predict - y_hot_encoding.T, xs)

        #update theta (+ bias)
        theta -= learning_rate * gradient
    
    def model(xs, theta=theta):
        #add bias column to the xs
        xs = np.append(1, xs)
        z = np.dot(theta, xs)
        predict = softmax(z)
        return np.argmax(predict)
    
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

# h = softmax_regression(xs, ys, 0.05, 750)

# test_inputs = [(1.30, 1), (2.25, 2), (0.97, 0), (1.07, 1), (1.51, 1)]
# print(f"{'prediction':^10}{'true':^10}")
# for x, y in test_inputs:
#     print(f"{h(x):^10}{y:^10}")
# prediction   true   
#     1         1     
#     2         2     
#     0         0     
#     1         1     
#     1         1 

print(xs)
print(xs.shape)
i, j = xs.shape
xs = np.hstack((np.ones((i,1)), xs))
print("xs", xs)
y_hot_encoding = one_hot_encoding(ys)
print("yhot", y_hot_encoding.T)
theta = np.zeros((y_hot_encoding.shape[1], j + 1))
print(theta)

z = np.dot(theta, xs.T)
print(z)

predict = softmax(z)
print("predict", predict)

gradient = np.dot(predict - y_hot_encoding.T, xs)
print(gradient)

