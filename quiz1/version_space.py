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


def version_space(H, D):
    vs = []
    for h in H:
        consistency = True
        for d in D:
            x = d[0]
            y = d[1]
            
            if h(x) == y & consistency:
                consistency = True
            else:
                consistency = False
        if consistency == True:
            vs.append(h)
    return vs


X = {"green", "purple"} # an input space with two elements
D = {("green", True)} # the training data is a subset of X * {True, False}
F = all_possible_functions(X)
H = F # H must be a subset of (or equal to) F

VS = version_space(H, D)

print(len(VS))

for h in VS:
    for x, y in D:
        if h(x) != y:
            print("You have a hypothesis in VS that does not agree with the set D!")
            break
    else:
        continue
    break
else:
    print("OK")    
# 2
# OK
    
D = {
    ((False, True), False),
    ((True, True), True),
}

def h1(x): return True
def h2(x): return False
def h3(x): return x[0] and x[1]
def h4(x): return x[0] or x[1]
def h5(x): return x[0]
def h6(x): return x[1]

H = {h1, h2, h3, h4, h5, h6}

VS = version_space(H, D)
print(sorted(h.__name__ for h in VS))
# ['h3', 'h5']