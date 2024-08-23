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

# PCA & Gaussian
# We can only plot in two dimensions, so these parameters can't be tweaked for this specific script.
PCA_N = 2
GAUSSIAN_N = 3

# Verbosity?
VERBOSE = False

# Show all the columns (13 features)
pd.set_option('display.max_columns', None)


def main():
    # Load data
    X, y = load_data(verbose=VERBOSE)

    # Scale the data
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    X_pca = reduce_dimensionality(X, verbose=VERBOSE)

    # Combine PCA results with target
    df_pca = pd.concat([X_pca, pd.Series(y, name='target')], axis=1)

    # Plot PCA pairplot
    plot_pca(df_pca)

    # Split the prepared dataset
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    gm = fit_gmm(X_train, verbose=VERBOSE)

    # Example call to plot_gmm with the trained GMM and training data
    plot_gmm(gm, X_train, y_train)


def load_data(verbose=False) -> tuple[pd.DataFrame, pd.Series]:
    # Load the wine dataset
    wine: Bunch = load_wine(as_frame=True)
    if verbose:
        print(wine.DESCR)
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
    sns.pairplot(data=df, hue="target", palette='coolwarm')
    plt.savefig("./dimensionality_reduction/output/wine_pca_visualization.png")


def fit_gmm(X, verbose=False) -> GaussianMixture:
    # To make sure we can still visualize the Gaussian Mixture model, we use two components
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(X)

    if verbose:
        print(f"Per-sample average log-likelihood of the given data X: {gm.score(X)}")
        print(f"GMM weights: {gm.weights_}")
        print(f"GMM means: {gm.means_}")
        print(f"GMM covariances: {gm.covariances_}")
        print(f"GMM precisions: {gm.precisions_}")
        print(f"GMM converged? {gm.converged_}")
        print(f"Nr. of iterations: {gm.n_iter_}")

    return gm


def plot_gmm(gmm: GaussianMixture, X_train: pd.DataFrame, y_train: np.ndarray, title: str = "GMM Clustering"):
    # Use only the first two principal components for visualization
    X_train_2d = X_train.iloc[:, :2]

    # Create a mesh grid for the contour plot
    x = np.linspace(X_train_2d.iloc[:, 0].min() - 1, X_train_2d.iloc[:, 0].max() + 1, 100)
    y = np.linspace(X_train_2d.iloc[:, 1].min() - 1, X_train_2d.iloc[:, 1].max() + 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T

    # Compute the log-likelihood of each point in the grid
    Z = -gmm.score_samples(XX)  # Directly pass XX without adding a third dimension
    Z = Z.reshape(X_grid.shape)

    # Plot the contour of the GMM's density estimate
    plt.figure(figsize=(8, 6))
    plt.contour(X_grid, Y_grid, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
    plt.colorbar(label='Negative Log-Likelihood')

    # Plot the training data points
    plt.scatter(X_train_2d.iloc[:, 0], X_train_2d.iloc[:, 1], c=y_train, s=20, cmap='coolwarm', edgecolors='k')

    # Plot the GMM centers (only the first two dimensions)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='x', s=100, color='black', label='GMM Centers')

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
