import numpy as np


def pca_lda(features, labels, mLDA=None, mPCA=None):
    rows, cols = features.shape

    featuresMean = np.mean(features, axis=0)

    covMatWithin = np.zeros([cols, cols])
    mu_cluster = np.zeros([, cols])
    for k in range(0, n_p):
        k_set = np.transpose(np.argwhere(labels == k))[0]

        if k_set.size == 0:
            mu_cluster[:, k] = np.zeros(rows)
        else:
            mu_cluster[:, k] = np.mean(features[:, k_set], axis=1)

        if k_set.size > 1:
            mu_w = mu_cluster[k, :]
            covMatWithin += (features[:, k_set] - mu_w[:, None]).dot((features[:, k_set] - mu_w[:, None]).T)

    covMatBetween = (mu_cluster - featuresMean[:, None]).dot((mu_cluster - featuresMean).T)

    if mPCA is None:
        mPCA = n_p - 1
        if mLDA is None:
            mLDA = int(mPCA * 2 / 3)
    elif mLDA is None and type(mPCA) is not np.ndarray:
        if mPCA > n_p - 1:
            mLDA = n_p - 1
        else:
            mLCA = mPCA

    if type(mPCA) is np.ndarray:
        i_set = np.transpose(np.argwhere(mPCA > cols - 1 - n_p))[0]
        mPCA[i_set] = (cols - 1 - n_p) * np.ones(i_set.size)
        if mLDA is None:
            if m_pca.size > n_p - 1:
                m_lda = n_p - 1
            else:
                m_lda = m_pca.size

    covMatTotal = np.cov(features)
    u = (features - featuresMean[:, None]).dot(eigen_order(st, m=m_pca))

    covMatLDA = np.linalg.inv(u.T.dot(covMatWithin.dot(u))).dot(u.T.dot(covMatBetween.dot(u)))
    w = u.dot(eigen_order(covMatLDA, m=m_lda))

    featuresProj = w.T.dot(features - featuresMean[:, None])

    return featuresProj, w, featuresMean


class rndFeatSplMachina:
    def __init__(self, model_id, features, labels, m0, m1):
        self.model_id = model_id
        self.features = features
        self.labels = labels
        self.m0 = m0
        self.m1 = m1
        self.data_train_proj = None
        self.w = None
        self.mu = None

    def run(self):
        print('Building Random Feature Sampling Machine ', self.model_id, '...')
        array = np.random.permutation(np.arange(self.m0, self.features.shape[1] - self.n_p))
        m_ar = np.concatenate((np.arange(self.m0), array[0:self.m1]), axis=None)
        self.data_train_proj, self.w, self.mu = pca_lda(self.features, self.labels, mPCA=m_ar)
        print('sub-model', self.model_id, 'done!')