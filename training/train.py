import numpy as np
import pandas as pd

from tools import optimizeMetric, dataSample, dataOrder, dataDisplay, modelSave

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

# -------------------------------------- DATA ORDERING -------------------------------------- #

dataSpl = dataSample(data)

# -------------------------------------- DATA ORDERING -------------------------------------- #

askRate, askSize, bidRate, bidSize, labels = dataOrder(dataSpl)

# -------------------------------------- PRE-PROCESS DATA -------------------------------------- #

p = 10
features = np.zeros([labels.size, p])

print('Pre-processing Data...')

features[:, 0] = askSize[:, 0]   # askSize0
features[:, 1] = np.nanmax(askSize, axis=1)    # askSizePtl
askSizePrbDst = (askSize.T / np.nansum(askSize, axis=1)).T
features[:, 2] = - np.nansum(askSizePrbDst * np.log(askSizePrbDst), axis=1)     # askSizeEntropy
features[:, 3] = askRate[:, 0]   # askRate0
features[:, 4] = np.choose(np.nanargmax(askSize, axis=1), askRate.T)   # askRatePtl

features[:, 5] = bidSize[:, 0]   # bidSize0
features[:, 6] = np.nanmax(bidSize, axis=1)    # bidSizePtl
bidSizePrbDst = (bidSize.T / np.nansum(bidSize, axis=1)).T
features[:, 7] = - np.nansum(bidSizePrbDst * np.log(bidSizePrbDst), axis=1)     # bidSizeEntropy
features[:, 8] = bidRate[:, 0]  # bidRate0
features[:, 9] = np.choose(np.nanargmax(bidSize, axis=1), bidRate.T)    # bidRatePtl

print('Data pre-processed!')

# -------------------------------------- DATA OBSERVATION -------------------------------------- #

# dataDisplay(features, labels)

# -------------------------------------- TRAINING -------------------------------------- #

aMat, featuresMean, featuresStd = optimizeMetric(features, labels, alpha=1e-5, maxIter=40, tol=1e-4)

# -------------------------------------- SAVE MODEL -------------------------------------- #

modelSave(aMat, featuresMean, featuresStd)

print('All done!!')
