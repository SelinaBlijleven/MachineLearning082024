"""
iris_svc_finetuning.py

In a realistic scenario, we can not test all possible values for all hyperparameters.

We define the hyperparameters we want to test for. In this case we will be trying two different kernels to see which
fits our dataset best. We will also be trying 10 different regularization values C for a total of 20 options.
"""
import pandas as pd

# Tools
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV
# ML Algorithm
from sklearn.svm import SVC

# Define the hyperparameters. They need to match the model of course!
HYPERPARAMETERS = {
    'kernel': ('linear', 'rbf'),
    'C': range(1, 111, 10)
}


def main():
    # Load the data, no pre-processing needed
    print("Loading data...")
    iris = load_iris()
    # Fit multiple classifiers
    print("Fitting model with best hyperparameters...")
    clf = fit_model(SVC(), iris)
    # Compare the models
    print("Results time!")
    compare_models(clf)


def fit_model(model: BaseEstimator, dataset: Bunch):
    # We now create a gridsearcher
    clf = GridSearchCV(model, HYPERPARAMETERS)

    # Fit the classifiers (this will take a while)
    clf.fit(dataset.data, dataset.target)

    return clf


def compare_models(clf):
    # Extract the results into a DataFrame
    results_df = pd.DataFrame(clf.cv_results_)

    # Sort by rank of test score
    results_df = results_df.sort_values(by="rank_test_score")

    # Select relevant columns for comparison
    # Select relevant columns for comparison based on available keys
    comparison_columns = [
        "rank_test_score",  # Rank of the test score
        "mean_test_score",  # Mean cross-validated score of the test set
        "std_test_score",  # Standard deviation of the cross-validated score
        "mean_fit_time",  # Mean time spent fitting the model
        "std_fit_time",  # Standard deviation of the fit time
        "param_C",  # Hyperparameter C (SVM regularization parameter)
        "param_kernel",  # Kernel type used in SVM
        "params"  # The complete parameter setting
    ]

    # Display the comparison DataFrame
    comparison_df = results_df[comparison_columns]
    print("\nComparison of different models and their scores:")
    print(comparison_df.to_string(index=False))

    # Summarize the top N models
    top_n = 10
    print(f"\nTop {top_n} models based on test score:")
    print(comparison_df.head(top_n).to_string(index=False))


if __name__ == "__main__":
    main()
