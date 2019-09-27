import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tools import featuresSort, featuresQuery, featuresSample, modelSave, modelLoad, preProcessData
from analysis import testModel


# -------------------------------------- OPERATION -------------------------------------- #

trainBool = True

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

# -------------------------------------- PRE-PROCESS DATA -------------------------------------- #

featuresTrain, labelsTrain, featuresTest, labelsTest = preProcessData(data)
yTest = np.heaviside(labelsTest, 1)

# -------------------------------------- TRAINING MODEL -------------------------------------- #

if trainBool:
    print('Training...')
    forrestSize, forrest = 10, []

    classes = np.unique(labelsTrain)
    for k in range(forrestSize):
        print('-- Training model', k + 1, '...')
        clf = LinearSVC(random_state=0, dual=False, max_iter=400)
        featuresTrainSpl, labelsTrainSpl = featuresSample(featuresTrain, labelsTrain)
        # featuresSort(
        yTrainSpl = np.heaviside(labelsTrainSpl, 1)     # labels
        clf.fit(featuresTrainSpl, yTrainSpl)
        print('Score for this tree is ', clf.score(featuresTest, yTest))
        forrest.append(clf)

    print('Done!')

    outfile = './forrest.pkl'
    modelSave(forrest, outfile)

else:
    infile = './forrest.pkl'
    forrest = modelLoad(infile)


# -------------------------------------- TESTING MODEL -------------------------------------- #

testModel(forrest, featuresTest, yTest)
