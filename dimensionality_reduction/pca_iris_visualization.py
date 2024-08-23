"""
pca_iris_visualization.py

An example of a PCA application that is used to visualize the Iris dataset.
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Tweak the number of components for PCA here. There are 4 features, so should be 1-3
N_COMPONENTS = 2

# Load the dataset
df = sns.load_dataset("iris")

# Load the desired features
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Get the features as numpy array
X = df.loc[:, features].values

# Get the target column
y = df.loc[:, ['species']].values

# Transform the feature columns using the StandardScaler
# with a mean of 0 and a variance of 1.
X = StandardScaler().fit_transform(X)
#print(X[:10])

# PCA with two components
pca = PCA(n_components=N_COMPONENTS)
pcomponents = pca.fit_transform(X)
principalDF = pd.DataFrame(
    data=pcomponents,
    columns=[f"PC{i}" for i in range(1, N_COMPONENTS+1)]
)

print(f"Principal components: {pca.components_}")
print(f"Explained variance: {pca.explained_variance_}")
print(f"Variance ratio: {pca.explained_variance_ratio_}")
print(principalDF.head(5))

# Concatenate the target column onto the principal features DF
finalDf = pd.concat([principalDF, df[['species']]], axis=1)

# Use a pairplot to visualize different subsets of dimensions
sns.pairplot(data=finalDf, hue="species", palette="ocean")
plt.show()
