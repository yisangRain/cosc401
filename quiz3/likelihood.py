"""
 Given the assumption that a number of real observations in a collection samples are independent and identically distributed, write a function likelihood(sample, mu, sigma) that returns the probability that the observations in sample are drawn from a normal distribution with mean mu and standard deviation sigma.

    Hint:

    The probability density function for an normal distribution is:

    fX(x;μ,σ2)=1σ2π√exp(−(x−μ)22σ2)
"""
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

#Test
mu = 0
sigma = 1
samples = [0.2]
print(f"{likelihood(samples, mu, sigma):.4f}")
# 0.3910

mu = 0
sigma = 1
samples = [-2.2]
print(f"{likelihood(samples, mu, sigma):.4f}")
# 0.0355