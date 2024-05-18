import random
import numpy as np

def bootstrap(dataset, sample_size, seed=0):
    random.seed(seed)
    # add code if necessary
    while True:
        sample = random.choices(dataset, k=sample_size)
        yield np.array(sample)



# Test

dataset = np.array([[1, 0, 2, 3],
                    [2, 3, 0, 0],
                    [4, 1, 2, 0],
                    [3, 2, 1, 0]])
ds_gen = bootstrap(dataset, 3)

for _ in range(5):
    print(next(ds_gen), end="\n\n")

ds = next(ds_gen)
print(type(ds))
print(ds.dtype != object)

# [[3 2 1 0]
#  [3 2 1 0]
#  [2 3 0 0]]

# [[2 3 0 0]
#  [4 1 2 0]
#  [2 3 0 0]]

# [[3 2 1 0]
#  [2 3 0 0]
#  [2 3 0 0]]

# [[4 1 2 0]
#  [3 2 1 0]
#  [4 1 2 0]]

# [[2 3 0 0]
#  [3 2 1 0]
#  [4 1 2 0]]

# <class 'numpy.ndarray'>
# True