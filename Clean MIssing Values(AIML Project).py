import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Sample dataset with missing values
data = {
    'Age': [25, 30, np.nan, 28, 35, np.nan],
    'Salary': [50000, 54000, 58000, np.nan, 62000, 60000],
    'Experience': [1, 3, 5, np.nan, 7, 8]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Simple Mean Imputation
df_simple = df.copy()
df_simple['Age'].fillna(df_simple['Age'].mean(), inplace=True)
df_simple['Salary'].fillna(df_simple['Salary'].mean(), inplace=True)

print("\nAfter Mean Imputation:")
print(df_simple)

# Advanced KNN Imputation
imputer = KNNImputer(n_neighbors=2)
df_knn = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nAfter KNN Imputation:")
print(df_knn)