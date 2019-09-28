import numpy as np

from scipy.stats import mode


def testModel(forest, features, labels):

    print('Testing model...')

    n, forestSize = labels.size, len(forest)

    votes = np.zeros([n, forestSize])
    for k in range(forestSize):
        votes[:, k] = forest[k].predict(features)

    prediction = mode(votes, axis=1)[0][:, 0]
    accuracy = 1 - np.count_nonzero(labels - prediction) / n

    print('Accuracy of full model is', accuracy)

