import numpy as np
import pandas as pd

from tools import dataSample, dataOrder, dataDisplay, modelSave
from analysis import rndFeatSmplMachina

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

p = 10
features = np.zeros([data[:, 0].size, p])
t = 10
m0 = 4
m1 = 3

# -------------------------------------- DATA OBSERVATION -------------------------------------- #

# dataDisplay(features, labels)

# -------------------------------------- TRAINING MODEL -------------------------------------- #

print('Building Random Feature Sampling Machine...')
masterMachina = []
for k in range(t):
    dataSpl = dataSample(data, pct=0.2)
    askRate, askSize, bidRate, bidSize, labels = dataOrder(dataSpl)

    features[:, 0] = askSize[:, 0]  # askSize0
    features[:, 1] = np.nanmax(askSize, axis=1)  # askSizePtl
    askSizePrbDst = (askSize.T / np.nansum(askSize, axis=1)).T
    features[:, 2] = - np.nansum(askSizePrbDst * np.log(askSizePrbDst), axis=1)  # askSizeEntropy
    features[:, 3] = askRate[:, 0]  # askRate0
    features[:, 4] = np.choose(np.nanargmax(askSize, axis=1), askRate.T)  # askRatePtl

    features[:, 5] = bidSize[:, 0]  # bidSize0
    features[:, 6] = np.nanmax(bidSize, axis=1)  # bidSizePtl
    bidSizePrbDst = (bidSize.T / np.nansum(bidSize, axis=1)).T
    features[:, 7] = - np.nansum(bidSizePrbDst * np.log(bidSizePrbDst), axis=1)  # bidSizeEntropy
    features[:, 8] = bidRate[:, 0]  # bidRate0
    features[:, 9] = np.choose(np.nanargmax(bidSize, axis=1), bidRate.T)  # bidRatePtl

    subMachina = rndFeatSmplMachina(k, features, labels, m0, m1)
    subMachina.run()
    masterMachina.append(subMachina)

print('Done!')

# -------------------------------------- SAVE MODEL -------------------------------------- #

modelSave(masterMachina, './model.pkl')

# -------------------------------------- TESTING MODEL -------------------------------------- #

# print('Testing Random Feature Sampling Machine...')
# test_machine(face_data_testing, randsmp, n_p)
# print('Done!')
