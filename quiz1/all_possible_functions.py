from itertools import product


def all_possible_functions(X):
    temp = []
    for item in X:
        temp.append({(item, True), (item, False)})
    sets = product(*temp)
    functions = []
    for i in sets:
        def hypothesis(x, i=i): # i=i to avoid clusure capturing same i 
            result = False
            for combination in i:
                if x == combination[0] and combination[1] is True:
                    result = True
            return result
        functions.append(hypothesis)
    return functions

X = {"green", "purple"} # an input space with two elements
F = all_possible_functions(X)

# # Let's store the image of each function in F as a tuple
images = set()
for h in F:
    images.add(tuple(h(x) for x in X))

for image in sorted(images):
    print(image)
    
# (False, False)
# (False, True)
# (True, False)
# (True, True)
    
X = {1, 2, 3}
F = all_possible_functions(X)
print(len(F))
# 8