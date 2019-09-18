import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data.csv')

data = df.values

y = data[:, -1]
n = y.size
askRate = data[:, 0:15]
askSize = data[:, 15:30]
bidRate = data[:, 30:45]
bidSize = data[:, 45:60]

ticks = ['askRate', 'askSize', 'bidRate', 'bidSize']
x = np.arange(5)

for i in range(n):
    cols = np.array([np.nanmean(askRate[i, :]), np.nanmean(askSize[i, :]),
                     np.nanmean(bidRate[i, :]), np.nanmean(bidSize[i, :]), y[i]])
    print(cols)
    fig, ax = plt.subplots()
    ax.plot(x, cols)
    plt.show()
