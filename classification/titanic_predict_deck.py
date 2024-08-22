"""
titanic_predict_deck.py

Warning: create a folder called 'output' in the script directory first.

During the titanic exercise, we found that the missing deck values were hard to handle.
This script will attempt to predict the deck values.

On the results: most classifiers have a weak performance on this task, generally
confusing the decks right above and below the target deck with the target.
Best results so far seem to come from the RandomForest, which might benefit from fine-tuning.

However, if we analyse the Titanic dataset and its properties, we don't seem to have enough
information to predict this. We do have the 'class' column, which is more generalized than the deck column.
This column ís in fact complete and is perhaps better to use for our prediction.
"""

# The basics
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Pre-processing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Got that algorithm
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Choose your fighter: "LogisticRegression", "KNN", "SVM", "GaussianNB", "DecisionTree", "RandomForest", "MLP"
MODEL = "DecisionTree"


def main() -> None:
    """ Main program flow """
    # Retrieve the dataset
    df = load_data()
    X, y = prepare_data(df)

    # Split the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the model
    model = fit_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict the y values and print some info about our training accuracy etc.
    print_model_results(y_pred, y_test)
    plot_confusion_matrix(model, y_test, y_pred)


def load_data() -> pd.DataFrame:
    """ Load and return the dataset """
    return sns.load_dataset("titanic")


def prepare_data(df: pd.DataFrame):
    """ Prepare the data for the ML algorithm """
    # Drop missing decks
    deck_info = df.dropna(subset=["deck"])

    # Select a subset of features for the prediction
    feats = deck_info[['class', 'fare', 'sibsp', 'parch']]

    # Convert the categorical variables and set X and y
    X = pd.get_dummies(feats)
    y = deck_info['deck']

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X, y)

    return X, y


def fit_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """ Fit the model of choice, as defined in the global constant. """
    # Create the desired classifier
    if MODEL == "LogisticRegression":
        clf = LogisticRegression()
    elif MODEL == "KNN":
        clf = KNeighborsClassifier()
    elif MODEL == "GaussianNB":
        clf = GaussianNB()
    elif MODEL == "DecisionTree":
        clf = DecisionTreeClassifier()
    elif MODEL == "RandomForest":
        clf = RandomForestClassifier()
    elif MODEL == "MLP":
        clf = MLPClassifier()
    # Default: SVM
    else:
        clf = SVC(kernel='linear')

    # Fit the classifier
    clf.fit(X_train, y_train)

    # Return the classifier
    return clf


def print_model_results(y_pred: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """ Print the results of the model """
    # Print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(clf: BaseEstimator, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """ Plot and save a confusion matrix """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"./output/titanic_cm_{MODEL}.png")


# If we are running this script, we do the whole thing.
if __name__ == "__main__":
    main()
