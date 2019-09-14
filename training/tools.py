import numpy as np


def gauss_kernel(rsdMat, sigma):
    a = np.diag(rsdMat.dot(np.transpose(rsdMat)))
    g = np.exp(- a / (2 * np.square(sigma)))

    return g


def mcc_pca(dataMat):
    n = dataMat.shape[0]
    p = dataMat.shape[1]
    idMat = np.eye(p)
