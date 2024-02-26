from itertools import product

def input_space(domains):
    return product(*domains)

domains = [
{0, 1, 2},
{True, False},
{'a', 'b', 'c'},
]

for element in sorted(input_space(domains)):
    print(element)
# (0, False)
# (0, True)
# (1, False)
# (1, True)
# (2, False)
# (2, True)