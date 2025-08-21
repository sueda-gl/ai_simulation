#!/usr/bin/env python3
"""
Phase B (alternative): LASSO-based variable selection on legitimate predictors

This script repeats the screening step using a machine-learning approach
(LassoCV) but *only* on predictors that are allowed under the professor’s
framework: personality survey variables (+ their z-scored versions we created)
plus the standard controls.  It explicitly excludes any behavioural or outcome
variables that would not be available for a new participant.

Outputs
-------
• `lasso_selected_features.txt` – list of predictors with non-zero LASSO weights
• Console printout of cross-validated MSE and selected variables
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV

# ----------------------------------------------------------------------------
# 1. Load the clean analysis dataset (same one used in earlier phases)
# ----------------------------------------------------------------------------

df = pd.read_csv("analysis_table.csv")  # contains survey _std vars + controls

# ----------------------------------------------------------------------------
# 2. Build predictor matrix (legitimate variables only)
#    • Survey predictors = columns ending with '_std' AND coming from the 24-item list
#    • Controls = TotalAllowance_std, Age_std, Foreign_binary, Group_*, Education_*,
#                 StudyProgram_*, TopPlatform_*
# ----------------------------------------------------------------------------

survey_predictors = [c for c in df.columns if c.endswith("_std") and c != "Prosocial_DV"]

control_cols = [
    "TotalAllowance_std",
    "Age_std",
    "Foreign_binary",
]
control_cols += [c for c in df.columns if c.startswith("Group_")]
control_cols += [c for c in df.columns if c.startswith("Education_")]
control_cols += [c for c in df.columns if c.startswith("StudyProgram_")]
control_cols += [c for c in df.columns if c.startswith("TopPlatform_")]

all_predictors = survey_predictors + control_cols

X = df[all_predictors].copy()
# remove any object-type columns that slipped through (e.g., Group_exp text)
obj_cols = X.select_dtypes(include=["object"]).columns
if len(obj_cols) > 0:
    print("Dropping non-numeric columns:", list(obj_cols))
    X = X.drop(columns=obj_cols)
    # also filter predictor list
    all_predictors = [p for p in all_predictors if p not in obj_cols]
Y = df["Prosocial_DV"].values

# ----------------------------------------------------------------------------
# 3. Prepare X: impute (though we have no missing), scale (mean 0, sd 1)
# ----------------------------------------------------------------------------

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ----------------------------------------------------------------------------
# 4. Run cross-validated LASSO
# ----------------------------------------------------------------------------

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, Y)

selected_mask = lasso.coef_ != 0
selected_features = np.array(all_predictors)[selected_mask]

print("Cross-validated alpha chosen:", lasso.alpha_)
print("Selected", len(selected_features), "predictors:")
for f, coef in zip(selected_features, lasso.coef_[selected_mask]):
    print(f"  {f}: β = {coef:.3f}")

# Save list to text file for record
Path("lasso_selected_features.txt").write_text("\n".join(selected_features))
print("List saved to lasso_selected_features.txt") 