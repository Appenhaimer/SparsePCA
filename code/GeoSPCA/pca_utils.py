import numpy as np
from sklearn.decomposition import PCA

def subA(sup, M):
    index = [i for i in range(sup.shape[0]) if sup[i] > 0.5]
    return M[:, index]

def disP(subA, ncA):
    pca = PCA(n_components=ncA)
    diff = subA - pca.inverse_transform(pca.fit_transform(subA))
    objval = np.linalg.norm(pca.singular_values_) ** 2
    norm = np.zeros(subA.shape[1])
    for i in range(subA.shape[1]):
        norm[i] = np.linalg.norm(diff[:, i]) ** 2
    return norm, objval 