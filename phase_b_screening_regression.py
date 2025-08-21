#!/usr/bin/env python3
"""
Phase B: Screening Regression Analysis
=====================================

This script fits a single OLS model with robust (HC3) standard errors that includes:
  • All 24 standardized survey predictors prepared in Phase A
  • All required controls (Group dummies, Total Allowance, Age, Education, Study-Program, Foreign, TopPlatform)

Outputs
-------
1. `screening_regression_table.csv` – full coefficient table with robust s.e.
2. `significant_predictors.png` – visual of survey predictors with p < 0.10
3. `phase_b_screening_regression.md` – markdown report summarising kept vs dropped predictors

Kept predictors = survey items with p-value < 0.10 (controls are retained regardless).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


SIGNIFICANCE_LEVEL = 0.10  # 10% level per professor's guidance


def load_analysis_data(path: str = "analysis_table.csv") -> pd.DataFrame:
    print("Loading analysis dataset…")
    df = pd.read_csv(path)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df


def build_X_y(df: pd.DataFrame):
    """Return y (DV) and X (predictors incl. intercept)."""
    y = df["Prosocial_DV"]

    # Survey predictors have the `_std` suffix (excluding the DV itself)
    survey_predictors = [c for c in df.columns if c.endswith("_std") and c != "Prosocial_DV"]

    # Controls
    control_vars = []
    control_vars += [c for c in df.columns if c.startswith("Group_")]
    control_vars += [c for c in ["TotalAllowance_std", "Age_std", "Foreign_binary"] if c in df.columns]
    control_vars += [c for c in df.columns if c.startswith("Education_")]
    control_vars += [c for c in df.columns if c.startswith("StudyProgram_")]
    control_vars += [c for c in df.columns if c.startswith("TopPlatform_")]

    all_predictors = survey_predictors + control_vars

    # Keep only numeric predictors; convert bool to int
    X_raw = df[all_predictors].copy()
    # Remove exact duplicate columns (same name appears twice)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()]

    # Convert boolean dtypes to int
    for col in X_raw.columns:
        if pd.api.types.is_bool_dtype(X_raw[col]):
            X_raw[col] = X_raw[col].astype(int)
    # Remove exact duplicate column names (possible from CSV save) and columns ending with '.1'
    dup_cols = [c for c in X_raw.columns if c.endswith('.1')]
    if dup_cols:
        print(f"Dropping duplicate suffixed columns: {dup_cols}")
        X_raw = X_raw.drop(columns=dup_cols)

    # Drop non-numeric (object) columns if any slipped through
    non_numeric_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        X_raw = X_raw.drop(columns=non_numeric_cols)

    X = sm.add_constant(X_raw)

    print(f"Survey predictors: {len(survey_predictors)}")
    print(f"Control variables: {len(control_vars)}")
    print(f"Total predictors: {len(all_predictors)}")
    return y, X, survey_predictors, control_vars


def run_regression(y, X):
    print("Fitting OLS model with HC3 robust s.e.…")
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")
    print("Done. R² = {:.3f}, adj-R² = {:.3f}".format(results.rsquared, results.rsquared_adj))
    return results


def analyse_significance(results, survey_predictors):
    records = []
    for pred in survey_predictors:
        coef = results.params[pred]
        pval = results.pvalues[pred]
        se = results.bse[pred]
        cil, ciu = results.conf_int().loc[pred]
        records.append({
            "predictor": pred,
            "coef": coef,
            "p_value": pval,
            "std_err": se,
            "ci_lower": cil,
            "ci_upper": ciu,
            "significant": pval < SIGNIFICANCE_LEVEL,
        })
    survey_df = pd.DataFrame(records).sort_values("p_value")
    kept = survey_df[survey_df.significant]["predictor"].tolist()
    dropped = survey_df[~survey_df.significant]["predictor"].tolist()
    print(f"Kept {len(kept)} / {len(survey_predictors)} survey predictors (p < {SIGNIFICANCE_LEVEL})")
    return kept, dropped, survey_df


def save_regression_table(results: sm.regression.linear_model.RegressionResults, path: str):
    tbl = pd.DataFrame({
        "Variable": results.params.index,
        "Coefficient": results.params.values,
        "Std_Error": results.bse.values,
        "t_value": results.tvalues.values,
        "p_value": results.pvalues.values,
        "CI_lower": results.conf_int().iloc[:, 0].values,
        "CI_upper": results.conf_int().iloc[:, 1].values,
    })
    tbl.to_csv(path, index=False)
    print(f"Saved regression table → {path}")


def plot_significant(survey_df: pd.DataFrame, out_path: str):
    sig_df = survey_df[survey_df.significant].copy()
    if sig_df.empty:
        print("No significant survey predictors to plot.")
        return
    sig_df = sig_df.sort_values("coef")
    plt.figure(figsize=(10, max(4, 0.4 * len(sig_df))))
    y_pos = np.arange(len(sig_df))
    plt.errorbar(sig_df["coef"], y_pos,
                 xerr=[sig_df["coef"] - sig_df["ci_lower"], sig_df["ci_upper"] - sig_df["coef"]],
                 fmt="o", capsize=4)
    plt.yticks(y_pos, sig_df["predictor"].str.replace("_std", "", regex=False))
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Coefficient estimate (robust)")
    plt.title("Significant survey predictors (p < 0.10)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved coefficient plot → {out_path}")


def write_markdown_report(results, kept, dropped, md_path: str):
    kept_md = "\n".join(f"- {k.replace('_std', '')}" for k in kept) if kept else "_None_"
    first_10_dropped = dropped[:10]
    dropped_md = "\n".join(f"- {d.replace('_std', '')}" for d in first_10_dropped)
    if len(dropped) > 10:
        dropped_md += f"\n- … and {len(dropped) - 10} others"
    md = f"""# Phase B: Screening Regression

**Model**: OLS with HC3 robust s.e.  
Observations: {int(results.nobs)}  
R²: {results.rsquared:.3f} (Adj. R² {results.rsquared_adj:.3f})  
F-statistic p-value: {results.f_pvalue:.4g}

## Kept survey predictors (p < 0.10)
{kept_md}

## Dropped survey predictors
{dropped_md if dropped_md else '_None_'}

All controls (Group, TotalAllowance, Age, Education, StudyProgram, Foreign, TopPlatform) are retained regardless of significance, per the professor’s instructions.

Files generated:
- `screening_regression_table.csv`
- `significant_predictors.png`

Ready to proceed to Phase C (Exploratory Factor Analysis) on the kept predictors plus the DV.
"""
    Path(md_path).write_text(md)
    print(f"Saved markdown report → {md_path}")


def main():
    df = load_analysis_data()
    y, X, survey_preds, control_vars = build_X_y(df)
    results = run_regression(y, X)
    kept, dropped, survey_df = analyse_significance(results, survey_preds)
    save_regression_table(results, "screening_regression_table.csv")
    plot_significant(survey_df, "significant_predictors.png")
    write_markdown_report(results, kept, dropped, "phase_b_screening_regression.md")


if __name__ == "__main__":
    main() 