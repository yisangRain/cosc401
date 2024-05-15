import random, collections, numpy as np

def k_means(dataset, centroids):
    n, _ = dataset.shape
    centroids = np.array(centroids)
    c_num, _ = centroids.shape

    # Loop while centroid and new centroids are not same (have not converged)
    while True:

        # Find argmin distance
        c = [[] for _ in range(c_num)]
        for i in range(n):
            # find the index of the centroid that gave the minimum distance and into a list of indices
            distance = [np.linalg.norm(dataset[i] - centroid) ** 2 for centroid in centroids]
            c[np.argmin(distance)].append(dataset[i])
        
        new_c = [np.mean(data, axis=0) for data in c]
        
        # replace old centroid with the new one and record the old one
        if np.equal(centroids, new_c).all():
            break

        centroids = new_c
    
    return tuple(centroids)

def distance(c1, c2):

    min_dist = []

    for c in c1:
        min_dist.append(np.min([(d - c) ** 2 for d in c2]))
    
    return min_dist

def goodness(centroids, dataset):
    # goodness = separation / compactness

    n, _ = centroids.shape

    # Separation
    min_dist = []

    for i in range(n):
        for j in range(n):
            if i != j:
                min_dist.append(np.min((centroids[i] - centroids[j]) ** 2))
    
    sep = np.mean(min_dist, axis=0)

    
    
    cpt


    pass

def k_means_random_restart(dataset, k, restarts, seed=None):
    random.seed(seed)
    Model = collections.namedtuple('Model', 'goodness, centroids')
    models = []
    for _ in range(restarts):
        centroids = k_means(dataset, random.sample([x for x in dataset], k=k))
        models.append(Model(goodness(centroids, dataset), centroids))
    return max(models, key=lambda m: m.goodness).centroids

# Test

def test1():
    dataset = np.array([
    [0.1, 0.1],
    [0.2, 0.2],
    [0.8, 0.8],
    [0.9, 0.9]
    ])
    centroids = k_means_random_restart(dataset, k=2, restarts=5, seed=0)

    for c in sorted([f"{x:.3}" for x in centroid] for centroid in centroids):
        print("  ".join(c))
    # 0.15  0.15
    # 0.85  0.85

def test2():
    import sklearn.datasets
    import sklearn.utils

    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]

    centroids = k_means_random_restart(iris.data, k=3, restarts=10, seed=0)


    # We suggest you check which centroid each 
    # element in test_data is closest to, then see test_target.
    # Note cluster 0 -> label 1
    #      cluster 1 -> label 2
    #      cluster 2 -> label 0

    for c in sorted([f"{x:.2}" for x in centroid] for centroid in centroids):
        print("  ".join(c))
    # 5.0  3.4  1.5  0.25
    # 5.9  2.7  4.4  1.4
    # 6.9  3.1  5.7  2.1

test1()
test2()