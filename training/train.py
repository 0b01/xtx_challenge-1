import numpy as np
import pandas as pd

from .tools import optimizeMetric

# -------------------------------------- LOAD DATA -------------------------------------- #

print('Loading Data...')
# df = pd.read_csv('../data.csv')
df = pd.read_csv('../data-training.csv')
data = df.values
print('Data loaded!')

y = data[:, -1]
n = y.size

# -------------------------------------- DATA ORDERING -------------------------------------- #

print('Organizing Data...')

labels, labelsCntr = np.unique(y, return_counts=True)
print(labels, labelsCntr)


askRate = data[:, 0:15]
askSize = data[:, 15:30]
bidRate = data[:, 30:45]
bidSize = data[:, 45:60]

print('Data Organized!')

# -------------------------------------- PRE-PROCESS DATA -------------------------------------- #

p = 10
features = np.zeros([n, p])

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

mu = np.mean(features, axis=1)

print('Data pre-processed!')

# -------------------------------------- DATA OBSERVATION -------------------------------------- #
# from tabulate import tabulate

# ticks = ['Day', 'askRate0', 'askRate14', 'bidRate0', 'bidRate14', 'y']
# rows = []

# for i in range(n):
#     if y[i] == -0.25:
#         rows.append([i+1, askRate0[i], askSize0[i], bidRate0[i], bidSize0[i], y[i]])

# print(tabulate(rows, headers=ticks))

# -------------------------------------- TRAINING -------------------------------------- #

aMat, , = optimizeMetric(features, y, )

# clusterMedians = np.zeros
# for i in range(clusters.size):
