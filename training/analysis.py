import numpy as np

from scipy.stats import mode


def testModel(forrest, features, y):

    print('Testing model...')

    n, forrestSize = y.size, len(forrest)

    votes = np.zeros([n, forrestSize])
    for k in range(forrestSize):
        votes[:, k] = forrest[k].predict(features)

    print(votes)
    prediction = mode(votes, axis=1)[0]
    accuracy = np.sum(prediction) / n

    print('Accuracy of model is', accuracy)

