import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
#predicted = model.predict([X_test[6]])

#print("Actual Value:", y_test[6])
#print("Predicted Value:", predicted[0])

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

labels = [0, 1, 2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
ax = sns.heatmap(cm, yticklabels=labels)
plt.show()

#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#disp.plot()

print(model.predict([X_test[6]]))
print(model.predict_log_proba([X_test[6]]))
print(model.predict_joint_log_proba([X_test[6]]))