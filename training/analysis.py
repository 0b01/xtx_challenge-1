import numpy as np

from scipy.stats import mode


def testModel(forrest, features, y):
    forrestSize = len(forrest)
    n = y.size
    prediction = np.zeros(n)
    for k in range(forrestSize):


    accuracy = np.sum(prediction) / n

    return accuracy


