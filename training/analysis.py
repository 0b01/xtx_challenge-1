import numpy as np


def eigenOrder(covMat, m=None, tol=1e-3):
    w, vMat = np.linalg.eigh(covMat)

    idx = np.argsort(w)[::-1]

    if type(m) is np.ndarray:
        v = np.real_if_close(vMat[:, idx], tol=tol)[:, m]
    else:
        v = np.real_if_close(vMat[:, idx], tol=tol)[:, 0:m]

    return v


def lda(features, labels, mLDA=None, mPCA=None):
    rows, cols = features.shape

    featuresMean = np.mean(features, axis=0)
    covMatWithin = np.zeros([cols, cols])

    setClasses = np.unique(labels)
    nClasses = setClasses.size

    featuresMeanClasses = np.zeros([nClasses, cols])
    for i in range(nClasses):
        idx = np.transpose(np.argwhere(labels == setClasses[i]))[0]

        if idx.size == 0:
            featuresMeanClasses[i, :] = np.zeros(rows)
        else:
            featuresMeanClasses[i, :] = np.mean(features[idx, :], axis=0)
            covMatWithin += (features[idx, :] - featuresMeanClasses[i, :]).dot((features[idx, :] - featuresMeanClasses[i, :]).T)

    covMatBetween = (featuresMeanClasses - featuresMean).dot((featuresMeanClasses - featuresMean).T)
    covMatTotal = (features - featuresMean).dot((features - featuresMean).T)

    if mPCA is None:
        mPCA = nClasses - 1
        if mLDA is None:
            mLDA = int(mPCA * 2 / 3)
    elif mLDA is None and type(mPCA) is not np.ndarray:
        if mPCA > nClasses - 1:
            mLDA = nClasses - 1
        else:
            mLDA = mPCA

    if type(mPCA) is np.ndarray:
        i_set = np.transpose(np.argwhere(mPCA > cols - 1 - nClasses))[0]
        mPCA[i_set] = (cols - 1 - nClasses) * np.ones(i_set.size)
        if mLDA is None:
            if mPCA.size > nClasses - 1:
                mLDA = nClasses - 1
            else:
                mLDA = mPCA.size

    uMat = (features - featuresMean).dot(eigenOrder(covMatTotal, m=mPCA))

    covMatLDA = np.linalg.inv(uMat.T.dot(covMatWithin.dot(uMat))).dot(uMat.T.dot(covMatBetween.dot(uMat)))
    wMat = uMat.dot(eigenOrder(covMatLDA, m=mLDA))

    featuresProj = wMat.T.dot(features - featuresMean)

    return featuresProj, wMat, featuresMean


class rndFeatSplMachina:
    def __init__(self, modelId, features, labels, m0, m1):
        self.modelId = modelId
        self.features = features
        self.labels = labels
        self.m0 = m0
        self.m1 = m1
        self.featuresProj = None
        self.wMat = None
        self.featuresMean = None

    def run(self):
        print('-- Building Random Feature Sampling Machine', self.modelId, '...')

        array = np.random.permutation(np.arange(self.m0, self.features.shape[1] - self.n_p))
        m_ar = np.concatenate((np.arange(self.m0), array[0:self.m1]), axis=None)
        self.featuresProj, self.wMat, self.featuresMean = lda(self.features, self.labels, mPCA=m_ar)

        print('-- Sub-model', self.modelId, 'done!')

