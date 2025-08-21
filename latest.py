import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# === Load data ===
df1 = pd.read_excel("Student Experiment Results - Period 1-2.xlsx")
df2 = pd.read_excel("Student Survey Results - Period 1.xlsx")

# === Merge datasets ===
df = pd.merge(df1, df2, on="Participant ID", how="inner")

import statsmodels.api as sm
import patsy

# === Final regression variables
target_var = "TWT+Sospeso [=AW2+AX2]{Periods 1+2}"
regression_vars = ["Group_x", "Total Allowance", "Study Program Category", "Honesty_Humility"]

# Standardise honesty_humility variable
df["Honesty_Humility"] = (df["Honesty_Humility"] - df["Honesty_Humility"].mean()) / df["Honesty_Humility"].std()

# === Drop rows with missing data
df_reg = df[[target_var] + regression_vars].dropna()

# === Regression formula with 3 categoricals + 1 continuous
formula = (
    f'Q("{target_var}") ~ C(Q("Group_x")) + C(Q("Total Allowance")) + '
    f'C(Q("Study Program Category")) + Q("Honesty_Humility")'
)

# === Create design matrices
y, X = patsy.dmatrices(formula, data=df_reg, return_type="dataframe")

# === Run the regression
model = sm.OLS(y, X).fit()

# === Print detailed summary
print(model.summary())

df["Observed_Behavior"] = df["TWT+Sospeso [=AW2+AX2]{Periods 1+2}"]

# Start with the intercept
df["Predicted_Behavior"] = 3.5694

# Group_x dummies
df["Predicted_Behavior"] += (
    df["Group_x"].map({
        "MidSub": 0.88,
        "NoSub": -0.91
    }).fillna(0)  # Base group gets 0
)

# Total Allowance dummies
df["Predicted_Behavior"] += (
    df["Total Allowance"].map({
        32: -0.42,
        72: -0.74,
        128: 3.54,
        200: 3.78
    }).fillna(0)  # Base level gets 0
)

# Study Program Category dummies
df["Predicted_Behavior"] += (
    df["Study Program Category"].map({
        "Incoming": -6.88,
        "Law 5-year Program": -2.00,
        "UG 3-year Program": -2.12
    }).fillna(0)  # Base program gets 0
)

# Add continuous predictor
df["Predicted_Behavior"] += 0.34 * df["Honesty_Humility"]


df["First_Case_Decision_Score"] = 0.25 * df["Observed_Behavior"] + 0.75 * df["Predicted_Behavior"]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["First_Case_Decision_Score_norm"] = scaler.fit_transform(df[["First_Case_Decision_Score"]])

print("First Case Decision Score:")
# Can you print all values please in the table?
print(df[["First_Case_Decision_Score_norm"]].head())
print("\nFirst Case Decision Score Summary:")
# Print summary statistics for First Case Decision Score
print(df[["First_Case_Decision_Score_norm"]].describe())
print(df[["First_Case_Decision_Score_norm"]])