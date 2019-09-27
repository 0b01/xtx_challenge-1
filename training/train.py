import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tools import featuresSort, featuresQuery, featuresSample, modelSave, preProcessData
from analysis import testModel

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

# -------------------------------------- PRE-PROCESS DATA -------------------------------------- #

featuresTrain, labelsTrain, featuresTest, labelsTest = preProcessData(data)

# -------------------------------------- TRAINING MODEL -------------------------------------- #

print('Training...')
clf = LinearSVC(random_state=0, dual=False, max_iter=400)
forrestSize, forrest = 10, []

classes = np.unique(labelsTrain)
for k in range(forrestSize):
    featuresTrainSpl, labelsTrainSpl = featuresSort(featuresSample(featuresTrain, labelsTrain))
    yTrainSpl = np.heaviside(labelsTrainSpl, 1)     # labels
    forrest.append(clf.fit(featuresTrainSpl, yTrainSpl))

print('Done!')

# -------------------------------------- SAVE MODEL -------------------------------------- #

outfile = './forrest.pkl'
modelSave(forrest, outfile)

# -------------------------------------- TESTING MODEL -------------------------------------- #

yTest = np.heaviside(labelsTest, 1)
testModel(forrest, featuresTest, yTest)

