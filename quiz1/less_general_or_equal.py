'''
return False if x is accepted by ha, but not by hb

i.e. 
    ha(x): return x % 2 == 0
    hb(x): return x % 3 == 0

if x == 2, 
    ha(2) = True
    hb(2) = False

for this case, ha is more general (accepts 2) than hb (doesn't accept 2)
'''
def less_general_or_equal(ha, hb, X):
    for x in X:
        if ha(x) and not hb(x):
            return False
    return True



X = list(range(1000))

def h2(x): return x % 2 == 0
def h3(x): return x % 3 == 0
def h6(x): return x % 6 == 0

H = [h2, h3, h6]

for ha in H:
    for hb in H:
        print(ha.__name__, "<=", hb.__name__, "?", less_general_or_equal(ha, hb, X))

# h2 <= h2 ? True
# h2 <= h3 ? False
# h2 <= h6 ? False
# h3 <= h2 ? False
# h3 <= h3 ? True
# h3 <= h6 ? False
# h6 <= h2 ? True
# h6 <= h3 ? True
# h6 <= h6 ? True