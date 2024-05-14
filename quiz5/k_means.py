import numpy as np

def k_means(dataset, centroids):
    n, d = dataset.shape
    centroids = np.array(centroids)
    c_num, _ = centroids.shape

    # set up empty list to contain new centroids
    new_c = np.zeros(centroids.shape)

    # record the previous centroid
    prev = np.zeros(centroids.shape)

    # Loop while centroid and new centroids are not same (have not converged)
    while np.not_equal(prev, centroids).any():

        # Find argmin distance
        c = []
        for i in range(n):
            # find the index of the centroid that gave the minimum distance and into a list of indices
            distance = [np.linalg.norm(dataset[i] - centroid) ** 2 for centroid in centroids]
            c.append(np.argmin(distance))
        
        for j in range(c_num):
            # Calculate the new centroids
            top = np.zeros(d)
            bottom = 0

            for k in range(n):
                if np.equal(centroids[c[k]], centroids[j]).all():
                    top += dataset[k]
                    bottom += 1
                
            new_c[j] = top/bottom
        
        # replace old centroid with the new one and record the old old one
        prev, centroids = centroids, new_c
    
    return tuple(centroids)

#Test

dataset = np.array([
    [0.1, 0.1],
    [0.2, 0.2],
    [0.8, 0.8],
    [0.9, 0.9]
])
centroids = (np.array([0., 0.]), np.array([1., 1.]))
for c in k_means(dataset, centroids):
    print(c)
# [0.15 0.15]
# [0.85 0.85]

dataset = np.array([
    [0.125, 0.125],
    [0.25, 0.25],
    [0.75, 0.75],
    [0.875, 0.875],
    [2.000, -1.00]
])
centroids = (np.array([0., 1.]), np.array([1., 0.]))
for c in k_means(dataset, centroids):
    print(c)
# [0.5 0.5]
# [ 2. -1.]

dataset = np.array([
    [0.1, 0.3],
    [0.4, 0.6],
    [0.1, 0.2],
    [0.2, 0.1]
])
centroids = (np.array([2., 5.]),)
for c in k_means(dataset, centroids):
    print(c)
# [0.2 0.3]

import sklearn.datasets
import sklearn.utils

wine = sklearn.datasets.load_wine()
data, target = sklearn.utils.shuffle(wine.data, wine.target, random_state=0)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]

centroids = (
    np.array([13.0, 2.2, 2.4, 18.1, 107.9, 2.6,
              2.5, 0.3, 1.6, 5.2, 1.0, 3.0, 964.0]),
    np.array([14.5, 1.8, 2.5, 17.0, 106.0, 2.9,
              3.0, 0.3, 2.0, 6.6, 1.1, 3.0, 1300.0]),
    np.array([12.0, 3.1, 2.3, 20.7, 92.8, 2.0,
              1.6, 0.4, 1.0, 4.7, 0.9, 2.0, 550.9])
)
for c in k_means(train_data, centroids):
    print(c)

# [1.34071739e+01 2.39152174e+00 2.40934783e+00 1.85521739e+01↩
#  1.08782609e+02 2.46326087e+00 2.24630435e+00 3.27608696e-01↩
#  1.69978261e+00 5.24847826e+00 9.58391304e-01 2.86695652e+00↩
#  9.12782609e+02]↩
# [1.38507407e+01 1.77851852e+00 2.48777778e+00 1.69259259e+01↩
#  1.05629630e+02 2.94148148e+00 3.13666667e+00 2.98888889e-01↩
#  2.00703704e+00 6.27518519e+00 1.10296296e+00 3.00222222e+00↩
#  1.30877778e+03]↩
# [1.25874000e+01 2.49150000e+00 2.32650000e+00 2.06210000e+01↩
#  9.43400000e+01 2.04410000e+00 1.63370000e+00 3.96400000e-01↩
#  1.43350000e+00 4.64879999e+00 9.19100000e-01 2.38020000e+00↩
#  5.27070000e+02]

