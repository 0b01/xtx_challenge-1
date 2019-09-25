import numpy as np

from tabulate import tabulate


# Compute gradient of sum of distances between points of dissimilar classes. Returns g and dg
def computeGrad(features, labels, aMat):
    print('--- Undergoing Gradient Ascent...')

    rows, cols = features.shape
    delta = 1e-5
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
    print('Computing Similarity Matrix...')

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

    print('Similarity Matrix Computed!')

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

    print('Iterative Projection Done!')

    return aMat, f


# Gradient ascent algorithm with Iterative Projection.
def optimizeMetric(features, labels, alpha, fThreshold=1, maxIter=40, tol=1e-3):
    print('Training Metric...')

    featuresMean = np.mean(features, axis=0)
    print(featuresMean.shape)
    features = features - featuresMean

    featuresStd = np.std(features, axis=0, ddof=1)
    features = np.divide(features, featuresStd)

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

        print('-> At step ', nIter, ': Difference g(A) = %.2f' % g, '/ Similarity f(A) = %.2f' % f)

        nIter += 1

    print('Metric trained!')

    return aMat, featuresMean, featuresStd


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


def modelSave(aMat, featuresMean, featuresStd):
    print('Saving Model...')

    np.save('./matrix.npy', aMat)
    np.save('./features_mean.npy', featuresMean)
    np.save('./features_std.npy', featuresStd)

    print('Model saved!')
