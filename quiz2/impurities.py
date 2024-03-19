import math

def pk_calculator(dataset):
    # Find and record each k (classification)
    k_set = set()
    for data in dataset:
        k_set.add(data[1])

    pk_list = []
    Qm = len(dataset) # total number of training data at node

    for k in list(k_set):
        n = 0
        # find sum of each classification's occurance
        for d in dataset:
            if d[1] == k:
                n += 1
        # Calculate pm per k, and record as a tuple (k, pmk)
        pk_list.append((k, n/Qm ))

    return pk_list

def misclassification(dataset):
    pk_list = pk_calculator(dataset)

    # find max pk 
    max_pk = max(pk_list, key=lambda x: x[1])[1]

    # calculate misclassification error
    return 1- max_pk

def gini(dataset): 
    pk_list = pk_calculator(dataset)
    
    # accumulator
    gini_impurity = 0

    # sum each k's pk * (1 - pk)
    for item in pk_list:
        _, pk = item
        gini_impurity += (pk * (1-pk))

    return gini_impurity


def entropy(dataset):
    pk_list = pk_calculator(dataset)

    # accumulator
    sums = 0

    for item in pk_list:
        _, pk = item
        sums += (pk * math.log(pk, 2))

    return sums * - 1


# data = [
#     ((False, False), False),
#     ((False, True), True),
#     ((True, False), True),
#     ((True, True), False)
# ]
# print("{:.4f}".format(misclassification(data)))
# print("{:.4f}".format(gini(data)))
# print("{:.4f}".format(entropy(data)))
# 0.5000
# 0.5000
# 1.0000


# data = [
#     ((0, 1, 2), 1),
#     ((0, 2, 1), 2),
#     ((1, 0, 2), 1),
#     ((1, 2, 0), 3),
#     ((2, 0, 1), 3),
#     ((2, 1, 0), 3)
# ]
# print("{:.4f}".format(misclassification(data)))
# print("{:.4f}".format(gini(data)))
# print("{:.4f}".format(entropy(data)))
# 0.5000
# 0.6111
# 1.4591


	
# dataset = []
# with open("quiz2/car.data") as f: #no car data, ha
#     for line in f.readlines():
#         out, *data = line.split(",")
#         dataset.append((data, out))
# print("{:.4f}".format(misclassification(dataset)))
# print("{:.4f}".format(gini(dataset)))
# print("{:.4f}".format(entropy(dataset)))
# 0.7500
# 0.7500
# 2.0000