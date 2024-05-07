import numpy as np
import itertools
import math

def monomial_kernel(d):
    
    def k(x, y, dimension = d):
        m = len(x)
    
        phi_list = [lambda x: 1]

        combi = set()

        for i in range(1, dimension+1):
            combi = combi.union(set(itertools.product(range(m), repeat=i)))
            
        for j in list(combi):
            def temp_function(z, combination = j):
                subtotal = 1
                for index in combination:
                    subtotal *= (z[index])
                return subtotal
            phi_list.append(temp_function)


        phi_x = np.array([p(x) for p in phi_list])
        phi_y = np.array([p(y) for p in phi_list])

        return np.dot(phi_x, phi_y)

    return k 

#Tests
# The monomial kernel of order 1 is just fitting a bias
k = monomial_kernel(1)
x = np.array([1])
y = np.array([2])
print(k(x, y) == 1 + (x * y).item())
# True

# The feature map in the question example
phi = lambda z: [1, z[0], z[1], z[0]*z[1], z[1]*z[0], z[0]**2, z[1]**2]
k = monomial_kernel(2)
x = np.array([1, 2])
y = np.array([3, 0.5])
print(k(x, y) == np.dot(phi(x), phi(y)))
# True

d = 10
k = monomial_kernel(d)
def phi(z): # Probably a nicer way to do this...
    features = range(len(z))
    mapped = []
    for bits in range(d + 1):
        for indices in itertools.product(features, repeat=bits):
            mapped.append(math.prod(z[j] for j in indices))
    return mapped

x = np.array([0.15, -0.6, 7.1])
y = np.array([-3.1, 0.01, -0.24])
print(np.isclose(k(x, y), np.dot(phi(x), phi(y))))

# Will timeout if feature map is computed
d = 100
k = monomial_kernel(d)
x = 0.1 * np.ones(100)
y = -0.02 * np.ones(100)
print(np.isclose(k(x, y), 5 / 6))
# True