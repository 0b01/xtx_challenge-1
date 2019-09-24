import numpy as np


# Compute gradient of sum of distances between points of dissimilar classes. Returns g and dg
def compute_dg(features, labels, quad_u):
    rows, cols = features.shape

    g, dg = 0, np.zeros(cols)
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                y2 = np.square(features[i, :] - features[j, :])
                distance = np.sqrt(quad_u.dot(y2))
                g += distance
                dg += y2 / distance

    dg = dg / 2

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
def iterativeProjection(quad_u, y2, f, obj_f):

    lam = (f - obj_f) / (y2.dot(y2))
    quad_u_nxt = quad_u - y2 * lam

    idx = np.transpose(np.argwhere(quad_u_nxt < 0))[0]
    quad_u_nxt[idx] = 0

    return quad_u_nxt


# Gradient ascent algorithm with Iterative Projection.
def gradAscentWithIterativeProjection(features, labels, alpha, maxIter=20, tol=1e-1, tol_f=1e-3, obj_f=1):
    cols = features.shape[1]

    quad_u = np.ones(cols) / cols

    y2 = compute_sim_feat(features, labels)
    eps, nIter, g = 1, 0, 0.0
    while eps > tol and nIter < maxIter:
        f = quad_u.dot(y2)
        print('g = %.2f' % g, '/ f = %.2f' % f)
        print('Projecting...')
        while np.abs(f - obj_f) > tol_f:
            quad_u = iterativeProjection(quad_u, y2, f, obj_f)
            f = quad_u.dot(y2)

        print('Ascending...')
        g, dg = compute_dg(features, labels, quad_u)

        quad_u_nxt = quad_u + alpha * dg
        eps = np.linalg.norm(quad_u_nxt - quad_u) / np.linalg.norm(quad_u)
        quad_u = quad_u_nxt

        nIter += 1

    gMat = np.sqrt(np.diag(quad_u))

    return gMat, nIter


def optimizeMetric(features, labels, rho, alpha, lbda, maxIter=20, tol=1e-1, tol_f=1e-3, obj_f=1):

    mu = np.mean(features, axis=1)
    n, p = features.shape

    aMat = np.ones(p)
    oMat, bMat = np.zeros(p), np.zeros(p)
    zMat = aMat - bMat

    lbda = lbda / rho

    nIter = 0
    converged = False
    while ~converged and nIter < maxIter:

        aMat = gradAscentWithIterativeProjection(features, labels, alpha)

        bMat = np.sign(aMat + zMat) * np.maximum((aMat + zMat) - lbda, oMat)

        zMatNxt = zMat + aMat - bMat

        eps = np.linalg.norm(zMatNxt - zMat, ord='fro') / np.linalg.norm(zMatNxt, ord='fro')
        if eps < tol:
            converged = True

        nIter += 1
        zMat = zMatNxt

    return aMat, mu, nIter


def dataOrder(data):
    labels, labelsIdx = np.unique(y, return_index=True)

    dataOrd = 1

    return dataOrd[:, 0:15], dataOrd[:, 15:30], dataOrd[:, 30:45], dataOrd[:, 45:60], dataOrd[:, -1]
