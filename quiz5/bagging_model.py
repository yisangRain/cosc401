import numpy as np
import random
from collections import Counter 


def bootstrap(dataset, sample_size, seed=0):
    random.seed(seed)
    # add code if necessary
    while True:
        sample = random.choices(dataset, k=sample_size)
        yield np.array(sample)


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


def bagging_model(learner, dataset, n_models, sample_size):
    sample_gen = bootstrap(dataset, sample_size)
    models = []
    for _ in range(n_models):
        models.append(learner(next(sample_gen)))
    ve = voting_ensemble(models)
    return ve

# Test

import sklearn.datasets
import sklearn.utils
import sklearn.tree

iris = sklearn.datasets.load_iris()
data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=1)
train_data, train_target = data[:-5, :], target[:-5]
test_data, test_target = data[-5:, :], target[-5:]
dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

def tree_learner(dataset):
    features, target = dataset[:, :-1], dataset[:, -1]
    model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
    return lambda v: model.predict(np.array([v]))[0]

bagged = bagging_model(tree_learner, dataset, 50, len(dataset)//2)
# Note that we get the first one wrong!
for (v, c) in zip(test_data, test_target):
    print(int(bagged(v)), c)
# 1 2
# 2 2
# 1 1
# 2 2
# 0 0