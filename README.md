# AIML-Project
CLEAN MISSING VALUES  
Project on Handling Missing Values  
  
1.Introduction  
Missing values occur when no data is stored for a variable. They may arise due to human errors, device failures, or incomplete data collection. Improper handling of missing values can lead to biased results, information loss, and poor machine learning model performance.  
1. Types of Missing Data  
•	MCAR (Missing Completely at Random) – independent of all variables .• MAR (Missing at Random) – related to observed data.   
•	MNAR (Missing Not at Random) – related to unobserved/missing values themselves.  
  
  
  
2. Imputation Methods  
Traditional Methods: - Deletion: Removing rows/columns with missing values. - Mean/Median/Mode  
Imputation: Simple, preserves size but may bias. - LOCF/NOCB: Forward/backward fill in time series. - Hot Deck Imputation: Use values from similar records. Advanced Methods: - KNN Imputation: Uses nearest neighbors. - Regression & ML Models: Predict missing values. - Random Forests, SVM: Handle non-linear relations. - MICE: Iterative imputation. - Multiple Imputation: Creates multiple plausible datasets. Deep Learning & Future Methods: - Autoencoders & GANs (GAIN). - Time Series Models (ARIMA, RNNs). - Federated & Reinforcement Learning.  
3. Evaluation Metrics  
For regression: MSE, RMSE, MAE, MAPE. For classification: Accuracy, Precision, Recall, F1-score.  
4. Applications  
• Healthcare – patient records. • Finance – forecasting & stock predictions. • Social Sciences – survey analysis.  
5. Challenges  
- Risk of bias when missingness ≠ random. - High computational cost for advanced methods. Curse of dimensionality in large datasets. - Ethical fairness and transparency issues.  
  
6. Best Practical Approach  
Step 1: Analyze missingness type (MCAR, MAR, MNAR). Step 2: Apply mean/median imputation for simple numeric datasets. Step 3: For complex/high-dimensional data → use KNN/MICE. Step 4: For sensitive applications → prefer Multiple Imputation or ML-based models. Step 5: Evaluate with MSE/RMSE or accuracy metrics.  
  
  
7. Code Execution Example  
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

8. Conclusion  
 For small/simple datasets: Mean/Median imputation is efficient. - For complex or critical datasets: ML and Deep Learning methods are more reliable. Future research should focus on scalable, ethical, and adaptive imputation techniques.  


