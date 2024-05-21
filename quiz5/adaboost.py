import numpy as np
import random
from collections import Counter

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size, seed=0):
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        # your code
        sample = np.array(random.choices(self.dataset, weights=self.weights, k=self.sample_size))
        return sample
    
def voting_ensemble(classifiers):
    def ve(x, classifiers = classifiers):
        votes = Counter([classifier(x) for classifier in classifiers])
        common = votes.most_common()
        vote = common[0][1]
        ties = []
        for key, value in common:
            if value == vote:
                ties.append(key)
            else:
                break
        
        return sorted(ties)[0]

    return ve

def adaboost(learner, dataset, n_models):
    n, _ = dataset.shape
    weights = np.ones(n)
    bootstrap = weighted_bootstrap(dataset, weights, n)

    models = []

    terminate = False

    for _ in range(n_models):
        sample = bootstrap.__next__()
        model = learner(sample)
        models.append(model)
        for i in range(n):
            predicted = model(sample[i][:-1]) 
            actual = sample[i][-1]
            error = np.abs(predicted - actual)
            if error == 0 or error >= 0.5:
                terminate = True
                break

            else:
                bootstrap.weights[i] *= error / (1 - error)

        if terminate == True:
            break

        bootstrap.weights = np.linalg.norm(bootstrap.weights)
    
    return voting_ensemble(models)
    

# Test

import sklearn.datasets
import sklearn.utils
import sklearn.linear_model

digits = sklearn.datasets.load_digits()
data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def linear_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]

boosted = adaboost(linear_learner, dataset, 10)
for (v, c) in zip(test_data, test_target):
    print(int(boosted(v)), c)
# 6 6
# 1 1
# 0 0
# 2 2
# 1 1