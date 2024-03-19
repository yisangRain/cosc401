import math

class DTNode:
    def __init__(self, decision, children=None):
        # either a function or a value
        self.decision = decision
        self.children = children


    def predict(self, inputObject):
        if self.children == None:
            return self.decision
        else:
            try:
                return self.children[self.decision(inputObject)].predict(inputObject)
            except:
                return self.children[0].predict(inputObject) #resolves underfitting
        

    def leaves(self):
        if self.children == None:
            return 1
        else:
            leaf_number = 0
            for child in self.children:
                leaf_number += child.leaves()
            return leaf_number

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

def partition_by_feature_value(dataset, feature_index):
    partition = []
    keys = []

    for data in dataset:
        features, _ = data
        key = features[feature_index]

        if len(partition) == 0:
            partition.append([data])
            keys.append(key)
        else:
            try:
                i = keys.index(key)
                partition[i].append(data)
            except:
                partition.append([data])
                keys.append(key)

    def separation(x, keyset=keys, i=feature_index):
        return keyset.index(x[i])
    
    return separation, partition 
            
def calculate_information_gain(dataset, criterion):
    """
    Calculates information gain per feature based on the given criterion function
    A helper function
    Returns list of information gain with same index as the feature
    """

    c_dataset = criterion(dataset)

    info_gain_list = []

    for i in range(0, len(dataset[0][0])):
        processed = []
        for data in dataset:
            features, classification = data
            processed.append(((features[i]), classification))
        
        info_gain_list.append((i, c_dataset - criterion(processed)))
    
    return info_gain_list


def create_tree(dataset, order):
    if len(order) == 0:
        return DTNode(dataset[0][1])

    else:
        current_i = order.pop()
        separation, partition = partition_by_feature_value(dataset.copy(), current_i)
        children = [create_tree(p, order.copy()) for p in partition]
        return DTNode(separation, children)




def train_tree(dataset, criterion): 
    #calculate pk_list, to get total number of classes and their frequency
    pk_list = pk_calculator(dataset)

    #if there is only one class within the dataset
    if len(pk_list) == 1:
        #return leaf of the sole class, and a function to pick that class
        return DTNode(pk_list[0][0], lambda x: x[1] == pk_list[0][0])  

    # if there are no features
    elif len(dataset[0][0]) == 0:
        # find the most frequent class
        max_k = max(pk_list, key=lambda x: x[1])[0]
        return DTNode(max_k, lambda y: y[1] ==  max_k)
    
    else:
        # calculate information gain per feature
        info = calculate_information_gain(dataset, criterion)
        # create a new list from info in ascending order
        desc_info = sorted(info, key=lambda x: x[1])

        order = [item[0] for item in desc_info]
        order.reverse()
        print(order)
        return create_tree(dataset, order)



# dataset = [
#   ((True, True), False),
#   ((True, False), True),
#   ((False, True), True),
#   ((False, False), False)
# ]
# t = train_tree(dataset, misclassification)
# print(t.predict((True, False)))
# print(t.predict((False, False)))

# True
# False
    
dataset = [
    (("Sunny",    "Hot",  "High",   "Weak"),   False),
    (("Sunny",    "Hot",  "High",   "Strong"), False),
    (("Overcast", "Hot",  "High",   "Weak"),   True),
    (("Rain",     "Mild", "High",   "Weak"),   True),
    (("Rain",     "Cool", "Normal", "Weak"),   True),
    (("Rain",     "Cool", "Normal", "Strong"), False),
    (("Overcast", "Cool", "Normal", "Strong"), True),
    (("Sunny",    "Mild", "High",   "Weak"),   False),
    (("Sunny",    "Cool", "Normal", "Weak"),   True),
    (("Rain",     "Mild", "Normal", "Weak"),   True),
    (("Sunny",    "Mild", "Normal", "Strong"), True),
    (("Overcast", "Mild", "High",   "Strong"), True),
    (("Overcast", "Hot",  "Normal", "Weak"),   True),
    (("Rain",     "Mild", "High",   "Strong"), False),
]
t = train_tree(dataset, misclassification)
print(t.leaves())
print(t.predict(("Overcast", "Cool", "Normal", "Strong")))
print(t.predict(("Sunny", "Cool", "Normal", "Strong")))

# True
# True