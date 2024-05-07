import numpy as np

def rbf_kernel(sigma):
    def k(x, y, sigma=sigma):
        numerator = np.linalg.norm(x - y) **2
        denominator = 2 * (sigma ** 2)
        
        return np.exp(-numerator / denominator)

    return k


#Test

x = np.array([0.1])
y = np.array([-0.25])
k = rbf_kernel(1)
print(f"RBF: {k(x, y):.6f}")
# RBF: 0.940588