"""
data_kfolds.py

Chapter: Preprocessing
Topic: Splitting the data

Split the data into K folds, with their own training and test sets.
"""
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns

# Set the target column
TARGET = 'species'

# Load the Penguins dataset: in this case we prepare the
# dataset to predict the species
df = sns.load_dataset('penguins')

# Drop rows with any missing values for the demo
df = df.dropna(how="any")

# Select the target variable, then remove it from the features
# (we do not want to use the target column in the features)
y = df[TARGET]
X = df.drop([TARGET], axis=1)

# Drop rows with missing values in the selected features
X = X.dropna()

# Handle categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Make a KFold splitter (using Sklearn class)
kf = KFold(n_splits=2)

# We use the KFolder splitter to split the features and the target variables.
# Then we iterate over them, using an enumeration to number the folds, and
# display the row indices used in each sample of our data.
for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")