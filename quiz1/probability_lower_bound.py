import math

def probability_lower_bound(test_outcomes, deviation):
    return 1- 2 * math.exp(-2 * len(test_outcomes) * deviation ** 2)

print(probability_lower_bound([True, False] * 500, 0.05))
# 0.986524106001829