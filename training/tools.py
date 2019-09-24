import numpy as np

from tabulate import tabulate


# Compute gradient of sum of distances between points of dissimilar classes. Returns g and dg
def computeGrad(features, labels, aMat):
    rows, cols = features.shape

    g, dg = 0, np.zeros(cols)
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                vectorDiff = features[i, :] - features[j, :]
                vectorDist = np.sqrt(vectorDiff.dot(aMat.dot(vectorDiff.T)))
                g += vectorDist
                dg += y2 / vectorDist

    g = g / rows
    dg = dg / (2 * rows)

    return g, dg


# Compute sum of element-wise square of differences between points of similar classes.
def compute_sim_feat(features, labels):
    rows, cols = features.shape

    y2 = np.zeros(cols)
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] == labels[j]:
                y2 += np.square(features[i, :] - features[j, :])
            else:
                break

    return y2


# Iterative Projection steps 1 & 2.
def iterativeProjection(aMat, y2, f, fThreshold, maxIter=50, tol=1e-3):
    print('--- Undergoing Iterative Projection...')

    eps, nIter = 1, 0
    while eps > tol and nIter < maxIter:
        # First Part
        lam = (f - fThreshold) / (y2.dot(y2))
        aMatNxt = aMat - y2 * lam

        # Second Part
        w, vMat = np.linalg.eigh(aMatNxt)

        idx = np.transpose(np.argwhere(w < 0))[0]
        w[idx] = 0

        aMatNxt = vMat.dot(np.diag(w).dot(vMat.T))
        eps = np.linalg.norm(aMatNxt - aMat, ord='fro') / np.linalg.norm(aMatNxt, ord='fro')
        aMat = aMatNxt

        nIter += 1

    print('Iterative Projection Done!')

    return aMat


# Gradient ascent algorithm with Iterative Projection.
def optimizeMetric(features, labels, alpha, maxIter=40, tol=1e-3, tol_f=1e-3, fThreshold=1):
    print('Training Metric...')

    featuresMean = np.mean(features, axis=1)
    features = features - featuresMean

    rows, cols = features.shape

    aMat = np.eye(cols) / cols

    y2 = compute_sim_feat(features, labels)
    eps, nIter, g = 1, 0, 0.0
    while eps > tol and nIter < maxIter:
        f = aMat.dot(y2)
        print('-> g(A) = %.2f' % g, '/ f(A) = %.2f' % f)
        aMat = iterativeProjection(aMat, y2, f, fThreshold)

        print('-- Ascending...')
        g, dg = computeGrad(features, labels, aMat)

        aMatNxt = aMat + alpha * dg
        eps = np.linalg.norm(aMatNxt - aMat, ord='fro') / np.linalg.norm(aMatNxt, ord='fro')
        aMat = aMatNxt

        nIter += 1

    print('Metric trained!')

    return aMat, featuresMean, nIter


def dataOrder(data):
    print('Ordering Data...')

    idx = np.argsort(data[:, -1])
    dataOrd = data[idx, :]

    print('Data Ordered!')

    return dataOrd[:, 0:15], dataOrd[:, 15:30], dataOrd[:, 30:45], dataOrd[:, 45:60], dataOrd[:, -1]


def dataDisplay(features, labels):
    print('Formatting Data for display...')

    n = labels.size

    cols = ['label', 'askSize0', 'askSizePtl', 'askSizeEntropy', 'askRate0', 'askRatePtl',
            'bidSize0', 'bidSizePtl', 'bidSizeEntropy', 'bidRate0', 'bidRatePtl']
    rows = []

    for i in range(n):
        rows.append([labels[i], features[i, 0], features[i, 1], features[i, 2], features[i, 3], features[i, 4],
                     features[i, 5], features[i, 6], features[i, 7], features[i, 8], features[i, 9]])

    print(tabulate(rows, headers=cols))
