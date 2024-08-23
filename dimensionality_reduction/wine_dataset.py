"""
wine_dataset.py
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# Sklearn general tools
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

# Components for PCA
PCA_N = 3
GAUS_N = 3

# Verbosity?
VERBOSE = False

# Show all the columns (13 features)
pd.set_option('display.max_columns', None)


def load_data(verbose=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the wine dataset
    wine: Bunch = load_wine(as_frame=True)
    if verbose: print(wine.DESCR)
    return wine.data, wine.target


def reduce_dimensionality(X, verbose=False) -> pd.DataFrame:
    pca = PCA(n_components=PCA_N)
    pcomponents = pca.fit_transform(X)

    if verbose:
        print(f"Principal components: {pca.components_}")
        print(f"Explained variance: {pca.explained_variance_}")
        print(f"Variance ratio: {pca.explained_variance_ratio_}")

    return pd.DataFrame(
        data=pcomponents,
        columns=[f"PC{i}" for i in range(1, PCA_N + 1)]
    )


def plot_pca(df: pd.DataFrame) -> None:
    sns.pairplot(data=df, hue="target")
    plt.savefig("./dimensionality_reduction/output/" + "wine_pca_visualization.png")


# Load data
X, y = load_data(verbose=VERBOSE)

# Scale the data
X = StandardScaler().fit_transform(X)

# Apply PCA
X = reduce_dimensionality(X, verbose=VERBOSE)

# Plot PCA pairplot
plot_pca(pd.concat([X, y], axis=1))