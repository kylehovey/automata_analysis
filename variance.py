import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    variances = np.load('./variance_ratios.npy')
    components = np.load('./pca_components.npy')

    plt.bar(np.arange(len(variances)), variances)
    plt.xlabel("Component Number")
    plt.ylabel("Variance contribution")
    plt.yscale("log")
    plt.savefig("./component_variance.png")
