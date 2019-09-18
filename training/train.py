import numpy as np
import pandas as pd

from tabulate import tabulate

# -------- LOAD DATA -------- #

df = pd.read_csv('../data.csv')
rawData = df.values

y = rawData[:, -1]
askRate = rawData[:, 0:15]
askSize = rawData[:, 15:30]
bidRate = rawData[:, 30:45]
bidSize = rawData[:, 45:60]

# askRateMean = np.nanmean(askRate, axis=1)
# askRateStd = np.nanstd(askRate, axis=1, ddof=1)
askRateMedian = np.nanmedian(askRate, axis=1)
askRateIQR = np.nanquantile(askRate, 0.75, axis=1) - np.nanquantile(askRate, 0.25, axis=1)

# askSizeMean = np.nanmean(askSize, axis=1)
# askSizeStd = np.nanstd(askSize, axis=1, ddof=1)
askSizeMedian = np.nanmedian(askSize, axis=1)
askSizeIQR = np.nanquantile(askSize, 0.75, axis=1) - np.nanquantile(askSize, 0.25, axis=1)

# bidRateMean = np.nanmean(bidRate, axis=1)
# bidRateStd = np.nanstd(bidRate, axis=1, ddof=1)
bidRateMedian = np.nanmedian(bidRate, axis=1)
bidRateIQR = np.nanquantile(bidRate, 0.75, axis=1) - np.nanquantile(bidRate, 0.25, axis=1)

bidSizeMean = np.nanmean(bidSize, axis=1)
bidSizeStd = np.nanstd(bidSize, axis=1, ddof=1)
# bidSizeMedian = np.nanmedian(bidSize, axis=1)

# -------- DATA OBSERVATION -------- #

ticks = ['Day', 'askRateMean', 'askRateStd', 'askSizeMean', 'askSizeStd',
                'bidRateMean', 'bidRateStd', 'bidSizeMean', 'bidSizeStd', 'Change']
rows = []
n = y.size
for i in range(n):
        rows.append([i+1, askRateMean[i], askRateStd[i], askSizeMean[i], askSizeStd[i],
                          bidRateMean[i], bidRateStd[i], bidSizeMean[i], bidSizeStd[i], y[i]])

print(tabulate(rows, headers=ticks))

# -------- TRAINING -------- #

clusters = np.array([-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# clusterMedians = np.zeros
# for i in range(clusters.size):
