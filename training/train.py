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

# -------------------------------------- TRAINING MODEL -------------------------------------- #

if trainBool:
    print('Training...')
    forrestSize, forrest = 10, []

    for k in range(forrestSize):
        print('-- Training model', k + 1, '...')
        clf = LinearSVC(class_weight='balanced', random_state=0, dual=False, max_iter=1000)
        featuresTrainSpl, labelsTrainSpl = featuresSample(featuresTrain, labelsTrain, pct=0.1)
        # featuresSort(
        clf.fit(featuresTrainSpl, labelsTrainSpl)
        print('Score for this tree is ', clf.score(featuresTest, labelsTest))
        forrest.append(clf)

    print('Done!')

    outfile = './forrest.pkl'
    modelSave(forrest, outfile)

else:
    infile = './forrest.pkl'
    forrest = modelLoad(infile)


# -------------------------------------- TESTING MODEL -------------------------------------- #

testModel(forrest, featuresTest, labelsTest)
