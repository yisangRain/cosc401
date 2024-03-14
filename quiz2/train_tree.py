from impurities import *
from partition import *
from dtnode import DTNode

def train_tree(dataset, criterion, used_index=[]): 
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
        







dataset = [
  ((True, True), False),
  ((True, False), True),
  ((False, True), True),
  ((False, False), False)
]
t = train_tree(dataset, misclassification)
print(t.predict((True, False)))
print(t.predict((False, False)))
# True
# False