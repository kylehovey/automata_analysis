import numpy as np
from sklearn.decomposition import PCA
import json

if __name__ == "__main__":
    data = np.load('./rule_data.npz')
    targets = np.load('./target.npz')

    N, dimensions = data.shape

    pca = PCA(n_components=dimensions)
    pca.fit(data)

    variance_ratios = pca.explained_variance_ratio_
    components = pca.components_

    np.save("variance_ratios.npy", variance_ratios)
    np.save("pca_components.npy", components)
