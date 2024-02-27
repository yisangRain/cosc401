import itertools


def decode(code):
    ax, ay, bx, by = code
    def decoded(xy):
        x, y = xy
        if (x >= min(ax, bx) and x <= max(ax, bx)):
            if (y >= min(ay, by) and y <= max(ay, by)):
                return True
        return False
    return decoded

h = decode((-1, -1, 1, 1))

for x in itertools.product(range(-2, 3), repeat=2):
    print(x, h(x))
# x 
# (-2, -2) False
# (-2, -1) False
# (-2, 0) False
# (-2, 1) False
# (-2, 2) False
# (-1, -2) False
# (-1, -1) True
# (-1, 0) True
# (-1, 1) True
# (-1, 2) False
# (0, -2) False
# (0, -1) True
# (0, 0) True
# (0, 1) True
# (0, 2) False
# (1, -2) False
# (1, -1) True
# (1, 0) True
# (1, 1) True
# (1, 2) False
# (2, -2) False
# (2, -1) False
# (2, 0) False
# (2, 1) False
# (2, 2) False