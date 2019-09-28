import numpy as np
import pickle

from tabulate import tabulate


# Compute gradient of sum of distances between points of dissimilar classes. Returns g and dg
def computeGrad(features, labels, aMat, delta=1e-5):
    print('--- Undergoing Gradient Ascent...')

    rows, cols = features.shape
    g, dg = 0, np.zeros([cols, cols])
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                vectorDiff = features[i, :] - features[j, :]
                vectorDist = np.maximum(np.sqrt(vectorDiff.dot(aMat.dot(vectorDiff.T))), delta)
                g += vectorDist
                dg += np.outer(vectorDiff, vectorDiff) / vectorDist

    g = g / rows
    dg = dg / (2 * rows)

    return g, dg


# Compute sum of element-wise square of differences between points of similar classes.
def computeSimilarityMatrix(features, labels):
    print('--- Computing Similarity Matrix...')

    rows, cols = features.shape

    simMat = np.zeros([cols, cols])
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] == labels[j]:
                vectorDiff = features[i, :] - features[j, :]
                simMat += np.outer(vectorDiff, vectorDiff)
            else:
                break

    simMat = simMat / rows

    print('-- Similarity Matrix Computed!')

    return simMat


# Iterative Projection steps 1 & 2.
def iterativeProjection(aMat, simMat, fThreshold=1, maxIter=200, tol=1e-4):
    print('--- Undergoing Iterative Projection...')

    w, vMat = np.linalg.eigh(aMat)
    simMatOrthoDiag = np.diag(np.transpose(vMat).dot(simMat.dot(vMat)))

    f, converged, nIter = 0.0, False, 0
    while ~converged and nIter < maxIter:
        f = w.dot(simMatOrthoDiag)
        rho = np.maximum((f - fThreshold) / (simMatOrthoDiag.dot(simMatOrthoDiag)), 0)

        wNxt = np.maximum(w - rho * simMatOrthoDiag, 0)

        eps = np.linalg.norm(wNxt - w) / np.linalg.norm(wNxt)
        if eps < tol:
            converged = True

        w = wNxt

        nIter += 1

    aMat = vMat.dot(np.diag(w).dot(vMat.T))

    print('--- Iterative Projection Done!')

    return aMat, f


# Gradient ascent algorithm with Iterative Projection.
def optimizeMetric(features, labels, alpha, fThreshold=1, maxIter=40, tol=1e-3):

    print('Standardizing features...')
    featuresMean = np.mean(features, axis=0)
    features = features - featuresMean

    featuresStd = np.std(features, axis=0, ddof=1)
    features = np.divide(features, featuresStd)
    print('Features standardized!')

    print('Training Metric:')
    rows, cols = features.shape
    aMat = np.eye(cols) / cols

    simMat = computeSimilarityMatrix(features, labels)
    g, converged, nIter = 0.0, False, 0
    while ~converged and nIter < maxIter:

        aMat, f = iterativeProjection(aMat, simMat, fThreshold)

        g, dg = computeGrad(features, labels, aMat)
        aMatNxt = aMat + alpha * dg     # Gradient Update

        eps = np.linalg.norm(aMatNxt - aMat, ord='fro') / np.linalg.norm(aMatNxt, ord='fro')
        if eps < tol:
            converged = True

        aMat = aMatNxt
        nIter += 1

        print('--> At step ', nIter, ': Difference g(A) = %.2f' % g, '/ Similarity f(A) = %.2f' % f)

    print('Metric trained!')

    return aMat, featuresMean, featuresStd


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


def preProcessData(data, ratio=0.8):
    p = 14
    n = data[:, 0].size
    features = np.zeros([n, p])

    askRate, askSize, bidRate, bidSize, labels = data[:, 0:15], data[:, 15:30], \
                                                 data[:, 30:45], data[:, 45:60], data[:, -1]

    print('Pre-processing Data...')

    features[:, 0] = askSize[:, 0]                                                      # askSize0
    features[:, 1] = np.nanmax(askSize, axis=1)                                         # askSizePtl
    askSizePrbDst = (askSize.T / np.nansum(askSize, axis=1)).T
    features[:, 2] = - np.nansum(askSizePrbDst * np.log(askSizePrbDst), axis=1)         # askSizeEntropy
    features[:, 3] = np.nanmedian(askSize, axis=1)                                      # askSizeMedian
    features[:, 4] = np.nanmedian(np.abs(askSize.T - features[:, 3].T).T, axis=1)       # askSizeMAD
    features[:, 5] = askRate[:, 0]                                                      # askRate0
    features[:, 6] = np.choose(np.nanargmax(askSize, axis=1), askRate.T)                # askRatePtl

    features[:, 7] = bidSize[:, 0]                                                      # bidSize0
    features[:, 8] = np.nanmax(bidSize, axis=1)                                         # bidSizePtl
    bidSizePrbDst = (bidSize.T / np.nansum(bidSize, axis=1)).T
    features[:, 9] = - np.nansum(bidSizePrbDst * np.log(bidSizePrbDst), axis=1)         # bidSizeEntropy
    features[:, 10] = np.nanmedian(bidSize, axis=1)                                     # bidSizeMedian
    features[:, 11] = np.nanmedian(np.abs(bidSize.T - features[:, 10].T).T, axis=1)     # bidSizeMAD
    features[:, 12] = bidRate[:, 0]                                                     # bidRate0
    features[:, 13] = np.choose(np.nanargmax(bidSize, axis=1), bidRate.T)               # bidRatePtl

    print('Data pre-processed!')

    print('Partitioning Data...')

    idx = np.random.permutation(np.arange(n))
    border = int(np.ceil(ratio * n))
    featuresTrain, labelsTrain = features[idx[0:border], :], labels[idx[0:border]]
    featuresTest, labelsTest = features[idx[border:], :], labels[idx[border:]]

    return featuresTrain, labelsTrain, featuresTest, labelsTest


def featuresSort(features, labels):
    idx = np.argsort(labels)
    featuresSorted, labelsSorted = features[idx, :], labels[idx]

    return featuresSorted, labelsSorted


def featuresQuery(features, labels, labels0, labels1):
    p = features.shape[1]

    features0 = np.empty([0, p])
    for i in range(labels0.size):
        idx = np.argwhere(labels == labels0[i])
        features0 = np.append(features0, [features[idx, :]], axis=0)

    features1 = np.empty([0, p])
    for i in range(labels1.size):
        idx = np.argwhere(labels == labels1[i])
        features1 = np.append(features1, [features[idx, :]], axis=0)

    return features0, features1


def featuresSample(features, labels, pct=0.2):

    print('Sampling Features...')
    n = features[:, 0].size

    idx = np.random.permutation(np.arange(n))[0:int(np.ceil(n * pct))]
    featuresSpl, labelsSpl = features[idx, :], labels[idx]

    print('Data sampled!')

    return featuresSpl, labelsSpl


def labelEncode(labels):
    n = labels.size
    labelsEncoded = 4 * labels

    return labelEncoded


def modelSave(obj, filename):
    print('Saving Model...')

    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)

    print('Model saved!')


def modelLoad(filename):
    print('Loading Model...')

    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)

    print('Model loaded!')

    return obj
