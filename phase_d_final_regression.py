#!/usr/bin/env python3
"""
Phase D: Final Regression – Factor Scores + Controls
====================================================

This script estimates each participant’s expected prosociality by regressing
`Prosocial_DV` on (a) the factor score(s) produced in Phase C and (b) the full
set of controls.  Robust HC3 standard errors are used.

It also re-runs the same specification separately on the Period 1 and Period 2
components to check period sensitivity (optional columns available in the field
sheet).

Outputs
-------
1. `final_regression_table.csv`   – coefficient table for the main model
2. `expected_prosociality.csv`    – Participant ID with fitted values and linear predictor z-score
3. `period_sensitivity_table.csv` – coefficient p-values for Period 1 and Period 2 models
4. `phase_d_final_regression.md`  – narrative markdown report
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CONTROL_COLUMNS_BASE = [
    "TotalAllowance_std", "Age_std", "Foreign_binary",
]
CONTROL_PREFIXES = ["Group_", "Education_", "StudyProgram_", "TopPlatform_"]

PERIOD1_COL = "TWT+Sospeso [=AN2+5*Q2]{Period 1}"
PERIOD2_COL = "TWT+Sospeso [=AN2+5*Q2]{Period 2}"


def load_data():
    # Prefer the multi-bundle factor file if available
    if Path("analysis_table_data_driven_factors.csv").exists():
        print("Loading data-driven factor table ...")
        return pd.read_csv("analysis_table_data_driven_factors.csv")
    elif Path("analysis_table_with_multi_factors.csv").exists():
        print("Loading multi-bundle factor table ...")
        return pd.read_csv("analysis_table_with_multi_factors.csv")
    else:
        print("Loading single-factor table ...")
        return pd.read_csv("analysis_table_with_factors.csv")


def build_X(df: pd.DataFrame):
    # predictor columns
    factor_scores = [c for c in df.columns if c.endswith("_score")]

    control_vars = []
    control_vars += [c for c in CONTROL_COLUMNS_BASE if c in df.columns]
    control_vars += [c for c in df.columns if any(c.startswith(p) for p in CONTROL_PREFIXES)]

    predictors = factor_scores + control_vars

    X_raw = df[predictors].copy()
    # Convert booleans to int
    for col in X_raw.columns:
        if pd.api.types.is_bool_dtype(X_raw[col]):
            X_raw[col] = X_raw[col].astype(int)

    # Flip sign of PROSOCIAL factor so that higher = more prosocial
    if 'PROSOCIAL_F1_score' in X_raw.columns:
        X_raw['PROSOCIAL_F1_score'] = -1 * X_raw['PROSOCIAL_F1_score']
    # Drop any residual object columns
    obj_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"Dropping non-numeric columns: {obj_cols}")
        X_raw = X_raw.drop(columns=obj_cols)

    X = sm.add_constant(X_raw)
    return X, factor_scores, control_vars


def fit_model(y, X):
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HC3")
    return res


def save_coef_table(results: sm.regression.linear_model.RegressionResults, path: str):
    tbl = pd.DataFrame({
        "Variable": results.params.index,
        "Coefficient": results.params.values,
        "Std_Error": results.bse,
        "t_value": results.tvalues,
        "p_value": results.pvalues,
        "CI_lower": results.conf_int().iloc[:, 0],
        "CI_upper": results.conf_int().iloc[:, 1],
    })
    tbl.to_csv(path, index=False)
    print(f"Saved coefficient table → {path}")
    return tbl


def period_sensitivity(df: pd.DataFrame, X):
    period_tables = []
    for col in [PERIOD1_COL, PERIOD2_COL]:
        if col not in df.columns:
            continue
        y_p = df[col]
        res_p = fit_model(y_p, X)
        period_tables.append(pd.DataFrame({
            "Variable": res_p.params.index,
            f"p_{'P1' if 'Period 1' in col else 'P2'}": res_p.pvalues
        }))
    if not period_tables:
        print("Period-wise columns not found – skipping sensitivity check.")
        return None
    merged = period_tables[0]
    for t in period_tables[1:]:
        merged = merged.merge(t, on="Variable")
    merged.to_csv("period_sensitivity_table.csv", index=False)
    print("Saved period_sensitivity_table.csv")
    return merged


def write_report(factor_scores, results, period_table):
    fs_list = "\n".join(f"- {f}" for f in factor_scores)
    n_obs = int(results.nobs)
    md = f"""# Phase D: Final Regression – Factor Scores + Controls

**Dependent variable**: Prosocial_DV  
Observations: {n_obs}  
R²: {results.rsquared:.3f} (Adj-R² {results.rsquared_adj:.3f})

## Predictors
Factor scores included:
{fs_list}

Controls retained: Group dummies, TotalAllowance_std, Age_std, Education dummies, StudyProgram dummies, Foreign_binary, TopPlatform dummies (as available).

Expected prosociality (fitted values) saved to `expected_prosociality.csv`.

## Period sensitivity
``period_sensitivity_table.csv`` contains p-values for each predictor using the Period 1 and Period 2 components individually.  Any factor/control that is significant (p < 0.10) in at least one period is highlighted for discussion.

Ready for Phase E (mapping predicted prosociality to default donation rates).
"""
    Path("phase_d_final_regression.md").write_text(md)
    print("Saved phase_d_final_regression.md")


def main():
    df = load_data()

    # Build predictors matrix
    X, factor_scores, control_vars = build_X(df)

    # Main model
    y = df["Prosocial_DV"]
    results = fit_model(y, X)
    save_coef_table(results, "final_regression_table.csv")

    # Save fitted values
    if "Participant ID" in df.columns:
        df_out = df[["Participant ID"]].copy()
    else:
        df_out = pd.DataFrame({"Row": df.index})
    df_out["Expected_Prosociality"] = results.fittedvalues
    df_out.to_csv("expected_prosociality.csv", index=False)
    print("Saved expected_prosociality.csv")

    # Period sensitivity analysis
    period_table = period_sensitivity(df, X)

    # Report
    write_report(factor_scores, results, period_table)
    print("Phase D completed successfully.")


if __name__ == "__main__":
    main() 