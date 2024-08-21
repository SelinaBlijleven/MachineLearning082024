"""
regression_evaluation_metrics.py

Evaluation for regression algorithms.
We use PredictionErrorDisplay to visualize the predicted values vs. actual values in a scatterplot.
We also calculate the residuals (positive/negative differences) between the predicted and actual value, to
visualize the differences and their distribution.

https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import PredictionErrorDisplay
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)

# Load the Diabetes dataset
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

# Create a model
lr = LinearRegression()

# Use cross-validation prediction! (10 folds, averaged results)
y_pred = cross_val_predict(lr, X, y, cv=10)

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="actual_vs_predicted",
    subsample=100,
    ax=axs[0],
    random_state=0,
)
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="residual_vs_predicted",
    subsample=100,
    ax=axs[1],
    random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting cross-validated predictions")
plt.tight_layout()
plt.show()