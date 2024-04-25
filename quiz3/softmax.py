import numpy as np

def softmax(z):
    total = sum(np.exp(x) for x in z)
    return np.array([(np.exp(k)/total) for k in z])


#Test
np.set_printoptions(precision=3, suppress=True)

z = np.array([1, -1])
print(softmax(z))
# [0.881 0.119]

np.set_printoptions(precision=3, suppress=True)

z = np.array([-1, -1, 2, -1])
print(softmax(z))
# [0.043 0.043 0.87  0.043]

np.set_printoptions(precision=3, suppress=True)

z = np.array([-1, -1, -1, -1])
print(softmax(z))
# [0.25 0.25 0.25 0.25]