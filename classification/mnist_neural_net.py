"""
mnist_neural_net.py

Classify images of handwritten digits.
"""

# Used for the comparison dataframe
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Define the hyperparameters. They need to match the model of course!
HYPERPARAMETERS = {
    'learning_rate_init': (0.01, 0.1, 1),
    'activation': ('tanh', 'relu'),
    'solver': ('lbfgs', 'adam')
}

# Number of images to use (0-7000)
N_IMAGES = 10000

def main():
    # Load the X and y data from the dataset
    X, y = load_data()
    X = StandardScaler().fit_transform(X)

    # Create the multilayer perceptron
    mlp = MLPClassifier(solver="lbfgs", activation="relu", learning_rate_init=0.01)

    # Pass the model and hyperparameters, then try different hyperparameters with
    # 5-fold cross validation
    clf = GridSearchCV(mlp, HYPERPARAMETERS, cv=2, verbose=1)
    clf.fit(X, y)

    # Compare the classifiers
    compare_models(clf)


def load_data():
    """ Load MNIST dataset from a library"""
    mnist = fetch_openml('mnist_784', version=1)
    return mnist["data"][:N_IMAGES], mnist["target"][:N_IMAGES]


def compare_models(clf):
    # Extract the results into a DataFrame
    results_df = pd.DataFrame(clf.cv_results_)

    # Sort by rank of test score
    results_df = results_df.sort_values(by="mean_test_score", ascending=False)

    # Display the comparison DataFrame
    print("\nComparison of different models and their scores:")
    print(results_df.to_string(index=False))

    # Summarize the top N models
    top_n = 10
    print(f"\nTop {top_n} models based on test score:")
    print(results_df.head(top_n).to_string(index=False))

    # Store the results
    results_df.to_csv("./output/mnist_results.csv")


if __name__ == "__main__":
    main()
