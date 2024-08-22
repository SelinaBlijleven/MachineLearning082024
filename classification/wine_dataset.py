"""
wine_dataset.py
"""
import pandas as pd

# Sklearn general tools
from sklearn.datasets import load_wine
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', None)

# Load the wine dataset
wine_dataset: Bunch = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(wine_dataset.data, wine_dataset.target, test_size=0.3)

# Train a Naive Bayes classifier
GNBC = GaussianNB()
GNBC.fit(X_train, y_train)

# Prediction and evaluation
y_pred = GNBC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy:.2f}")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Train an LDA classifier
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"LDA Accuracy: {accuracy:.2f}")