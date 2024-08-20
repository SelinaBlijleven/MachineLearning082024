"""
data_cleaning_missing_values.py

Chapter: Preprocessing
Topic: Missing values

Explore some different possibilities to fill in missing values in Pandas.
"""

import seaborn as sns

df = sns.load_dataset("penguins")

# Remove rows with any missing values
df_cleaned = df.dropna()

# Remove rows with missing values in a specific column
df_cleaned2 = df.dropna(subset=['species'])

# Fill missing values with a specific value (e.g., 0)
df_filled = df.fillna(0)

# Fill missing values with the mean of the column (numerical values only)
numdf = df[["body_mass_g", "bill_depth_mm", "bill_length_mm", "flipper_length_mm"]]
numdf_filled = numdf.fillna(numdf.mean())

# Don't forget to print your results
#print(df_cleaned)
#print(df_cleaned2)
#print(df_filled)
#print(numdf_filled)
