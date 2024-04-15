import numpy as np

def max_log_likelihood_estimator(samples):
    n = len(samples)
    
    #mu
    mu = (1/n) * np.sum(samples)
    
    #sigma
    sigma = np.sqrt((1/n) * np.sum((samples - mu) ** 2))
    
    return (mu, sigma)






#Tests
samples = [-0.5, 0.5]
mu, sigma = max_log_likelihood_estimator(samples)
print(mu == 0, sigma == 0.5)
# True True

samples = np.full(100, -0.25)
mu, sigma = max_log_likelihood_estimator(samples)
print(mu == -0.25, sigma == 0)
# True True