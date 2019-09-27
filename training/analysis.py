import numpy as np

from scipy.stats import mode


def testModel(forrest, features, y):

    print('Testing model...')

    n, forrestSize = y.size, len(forrest)

    votes = np.zeros([n, forrestSize])
    for k in range(forrestSize):
        votes[:, k] = forrest[k].predict(features)

    prediction = mode(votes, axis=1)[0]
    
    print(prediction.shape, y.shape)
    accuracy = 1 - np.count_nonzero(y - prediction) / n

    print('Accuracy of model is', accuracy)

