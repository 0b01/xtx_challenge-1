import numpy as np

from scipy.stats import mode


def testModel(forrest, features, labels):

    print('Testing model...')

    n, forrestSize = y.size, len(forrest)

    votes = np.zeros([n, forrestSize])
    for k in range(forrestSize):
        votes[:, k] = forrest[k].predict(features)

    prediction = mode(votes, axis=1)[0][:, 0]
    accuracy = 1 - np.count_nonzero(labels - prediction) / n

    print('Accuracy of full model is', accuracy)

