import pandas as pd
import seaborn as sns

# Load the Titanic dataset
df = sns.load_dataset("titanic")


# Option 1: apply-function
# First we write the function to determine the outcome
def impute_age(row: pd.Series) -> float:
    """
    This function receives one row to transform and
    returns one value to store in the new 'age_imputed'
    column.
    :param      row:    Row from the dataframe (we need multiple columns)
    :return:    float:  Value for the age in this row
    """
    # Import the global variable age means, so we can use them accordingly
    global age_means

    # If the row is missing an age, add the mean for a woman/child/man
    if pd.isnull(row['age']):
        return age_means[row['who']]
    # If we already have an age, we leave it alone
    else:
        return row['age']


# Calculate the mean age for each "who" group
age_means = df.groupby("who")["age"].mean()
#  Apply the function written above to fill in the missing ages
#df["age_imputed"] = df.apply(impute_age, axis=1)

# Option 2: one-liner
# Borrowed from stack overflow answer: https://stackoverflow.com/a/74233664
df['age_imputed'] = df['age'].fillna(df.groupby("who")["age"].transform('mean'))

# Print the columns we touched to test
print(df[["who", "age", "age_imputed"]].tail(10))
