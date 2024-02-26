def less_general_or_equal(ha, hb, X):




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