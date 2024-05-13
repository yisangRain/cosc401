import numpy as np

def k_means(dataset, centroids, iteration = 500 ):
  
    n, d = dataset.shape
    k = len(centroids)

    centroids = list(centroids)
    new = np.zeros((k, len(centroids[0])))

    i = 0

    while np.not_equal(centroids, new).any() and i < iteration:

        c = np.zeros(n)
        
        for i in range(n):
            distance = [np.linalg.norm(dataset[i] - centroid) ** 2 for centroid in centroids]
            c[i] = np.argmin(distance)

        for j in range(k):        
            top = np.zeros(d)
            bottom = 0
            for i in range(n):
                if np.equal(c[i], centroids[j]).all():
                    top += dataset[i] 
                    bottom += 1

            new[j] = top / bottom
        
        centroids = new

        i += 1
        
    return centroids

                
            


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