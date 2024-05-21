import random
import numpy as np

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size, seed=0):
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        # your code
        sample = np.array(random.choices(self.dataset, weights=self.weights, k=self.sample_size))
        return sample
    

# Test

dataset = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]])

wbs = weighted_bootstrap(dataset, [1, 1, 1, 1, 1], 5)
sample = next(wbs)
print(type(sample))
print(sample)
print()
print(next(wbs))

print("\nWith new weights:")
wbs.weights = [1, 1, 1000, 1, 1]
print(next(wbs))
print()
print(next(wbs))
# <class 'numpy.ndarray'>
# [[5]
#  [4]
#  [3]
#  [2]
#  [3]]

# [[3]
#  [4]
#  [2]
#  [3]
#  [3]]

# With new weights:
# [[3]
#  [3]
#  [3]
#  [3]
#  [3]]

# [[3]
#  [3]
#  [3]
#  [3]
#  [3]]