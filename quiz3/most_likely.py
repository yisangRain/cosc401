"""
Given a collection of normal distributions represented by a number of (mu, sigma) pairs, 
write a function most_likely(sample, distributions) that returns a tuple (mu, sigma) 
containing the distribution that the observations in sample were most likely drawn from.
"""
from statistics import NormalDist
from random import seed
from itertools import product
from math import exp, pi, sqrt

def likelihood(sample, mu, sigma):

    likelihood = 1
    
    for x in sample:
        # left half of the equation
        left = 1 / (sigma * sqrt(2 * pi))
        #right half of the equation
        right = exp((-((x - mu)**2)) / (2 * sigma ** 2))
        likelihood *= (left * right)

    return likelihood

def most_likely(samples, distributions):
    d_list = list(distributions)
    l = [likelihood(samples, mu, sigma) for mu, sigma in d_list]

    return d_list[l.index(max(l))]
    


#Tests
samples = [0.1]
distributions = [(0, 1), (-2, 3)]
mu, sigma = most_likely(samples, distributions)
print(f"Sample most likely has mean {mu} and standard deviation {sigma}")	

# Sample most likely has mean 0 and standard deviation 1

samples = [0.5]
distributions = [(0, 1), (0, 0.5)]
mu, sigma = most_likely(samples, distributions)
print(f"Sample most likely has mean {mu} and standard deviation {sigma}")

# Sample most likely has mean 0 and standard deviation 0.5


seed(0x5eeded)
mus = range(-10, 11)
sigmas = range(1, 3)
distributions = product(mus, sigmas)
true = NormalDist(mu=3.1, sigma=1.9)
# print(list(distributions))
mu, sigma = most_likely(true.samples(20), distributions)
print(f"Sample most likely has mean {mu} and standard deviation {sigma}")

	

# Sample most likely has mean 3 and standard deviation 2