import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tools import featuresSample, modelSave, modelLoad, preProcessData, labelEncode
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
labelsTestEncoded = labelEncode(labelsTest)

# -------------------------------------- TRAINING MODEL -------------------------------------- #

if trainBool:
    print('Training...')
    forrestSize, forrest = 10, []

    for k in range(forrestSize):
        print('-- Training model', k + 1, '...')
        clf = DecisionTreeClassifier()
        featuresTrainSpl, labelsTrainSpl = featuresSample(featuresTrain, labelsTrain, pct=0.2)
        labelsTrainSplEncoded = labelEncode(labelsTrainSpl)
        clf.fit(featuresTrainSpl, labelsTrainSpl.astype(int))
        print('Score for this machine is ', clf.score(featuresTest, labelsTest.astype(int)))
        forrest.append(clf)

    print('Done!')

    outfile = './forrest.pkl'
    modelSave(forrest, outfile)

else:
    infile = './machine.pkl'
    forrest = modelLoad(infile)


# -------------------------------------- TESTING MODEL -------------------------------------- #

testModel(forrest, featuresTest, labelsTest.astype(int))
