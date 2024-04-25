import numpy

def one_hot_encoding(ys):
    if len(ys) == 0:
        return numpy.array([])
    k = numpy.max(ys)
    array = []
    for y in ys:
        row = numpy.zeros(k + 1, dtype=int)
        row[y] = 1
        array.append(row)
    return numpy.array(array).reshape(len(ys),k+1)


#Test
ys = numpy.array([0, 1, 0, 2, 1])
print(one_hot_encoding(ys))

# [[1 0 0]
#  [0 1 0]
#  [1 0 0]
#  [0 0 1]
#  [0 1 0]]

ys = numpy.array([])
print(one_hot_encoding(ys))
# []

ys = numpy.array([0])
print(one_hot_encoding(ys))
# [[1]]