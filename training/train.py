import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from tools import dataSample, dataDisplay, modelSave, preProcessData
# from analysis import rndFeatSplMachina

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

# -------------------------------------- PRE-PROCESS DATA -------------------------------------- #

featuresTrain, labelsTrain, featuresTest, labelsTest = preProcessData(data)

# -------------------------------------- DATA OBSERVATION -------------------------------------- #

# dataDisplay(features, labels)

# -------------------------------------- TRAINING MODEL -------------------------------------- #

print('Training...')

clf = LinearSVC(random_state=0, dual=False, max_iter=400)

tree = []
m = 10

for k in range(m):
    y = np.heaviside(labels, 1)
    decision = clf.fit(features, y)


print(decision.score(features, y))

print('Done!')

# -------------------------------------- SAVE MODEL -------------------------------------- #

# -------------------------------------- TESTING MODEL -------------------------------------- #

prediction = np.zeros(labelsTest.size)
for i in range(prediction.size):
    for k in range(m):

