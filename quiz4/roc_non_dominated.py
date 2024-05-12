from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):
    
    def tpr(self):
        p = self.true_positive + self.false_negative
        return self.true_positive / p

    def fpr(self):
        n = self.false_positive + self.true_negative
        return self.false_positive/ n
    
def roc_non_dominated(classifiers):
    non_dom = []

    for name, classifier in classifiers:
        tpr = classifier.tpr()
        fpr = classifier.fpr()

        dominated = False

        for _, c in classifiers:
            if (c.tpr() > tpr):
                if (c.fpr() < fpr):
                    dominated = True
                    break
        
        if dominated == False:
            non_dom.append((name, classifier))
        
    return non_dom



#Test

# Example similar to the lecture notes

classifiers = [
    ("h1", ConfusionMatrix(60, 40, 
                            20, 80)),
    ("h2", ConfusionMatrix(40, 60, 
                              30, 70)),
    ("h3", ConfusionMatrix(80, 20, 
                             50, 50)),
]
print(sorted(label for (label, _) in roc_non_dominated(classifiers)))
# ['h1', 'h3']
