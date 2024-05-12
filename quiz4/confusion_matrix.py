from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        elements = [self.true_positive, self.false_negative,
                   self.false_positive, self.true_negative]
        return ("{:>{width}} " * 2 + "\n" + "{:>{width}} " * 2).format(
                    *elements, width=max(len(str(e)) for e in elements))
    

def confusion_matrix(classifier, dataset):
    conf_mat = [0, 0, 0, 0]
    for i in range(len(dataset)):
        features, prediction = dataset[i]
        actual = classifier(features)
        
        if prediction == True: 
            if actual == True:
                # TP
                conf_mat[0] += 1
            elif actual == False:
                # FP
                conf_mat[1] += 1
        elif prediction == False:
            if actual == True:
                # FN
                conf_mat[2] += 1
            elif actual == False:
                # FN
                conf_mat[3] += 1
    
    return ConfusionMatrix(conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3])


#Test

dataset = [
    ((0.8, 0.2), 1),
    ((0.4, 0.3), 1),
    ((0.1, 0.35), 0),
]
print(confusion_matrix(lambda x: 1, dataset))
print()
print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))
# 2 0
# 1 0

# 2 0
# 0 1