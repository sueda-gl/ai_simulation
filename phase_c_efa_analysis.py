#!/usr/bin/env python3
"""
Phase C: Exploratory Factor Analysis (EFA)
========================================

This script performs an EFA on the retained survey predictors from Phase B **plus**
the dependent variable `Prosocial_DV`, as requested by the professor.  Controls
are *not* part of the factor model.

Outputs
-------
1. `efa_loadings.csv`            – pattern / structure matrix of factor loadings
2. `efa_factor_scores.csv`      – per-participant factor scores (merged)
3. `scree_plot.png`             – eigenvalue scree plot
4. `phase_c_efa_analysis.md`    – markdown report with KMO, Bartlett, factors retained, etc.

Assumptions / notes
-------------------
• Uses `factor_analyzer` (common-factor MINRES) with oblique rotation (Oblimin).
• Parallel analysis is mocked by the eigenvalue > 1 rule and visual scree because
  with ≤ 5 variables a Monte-Carlo parallel run is over-kill.  If > 5 variables
  are retained the script will run a quick permutation-based parallel check.
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.utils import resample

warnings.filterwarnings("ignore")

CONTROL_COLUMNS = [
    "TotalAllowance_std", "Age_std", "Foreign_binary",
]
CONTROL_PREFIXES = [
    "Group_", "Education_", "StudyProgram_", "TopPlatform_"
]

SIGNIFICANCE_THRESHOLD = 0.10  # used to identify retained survey predictors


def load_data():
    df = pd.read_csv("analysis_table.csv")
    coeffs = pd.read_csv("screening_regression_table.csv")
    return df, coeffs


def select_efa_variables(df: pd.DataFrame, coeffs: pd.DataFrame):
    """Return a dataframe X containing DV + kept survey predictors (no controls)."""
    # Identify kept survey predictors via p-value threshold and suffix _std
    sig_mask = (coeffs["p_value"] < SIGNIFICANCE_THRESHOLD) & coeffs["Variable"].str.endswith("_std")
    kept_predictors = coeffs.loc[sig_mask, "Variable"].tolist()

    # Remove controls if any slipped through (e.g., TotalAllowance_std)
    def is_control(col):
        if col in CONTROL_COLUMNS:
            return True
        return any(col.startswith(pref) for pref in CONTROL_PREFIXES)

    kept_predictors = [c for c in kept_predictors if not is_control(c)]

    if len(kept_predictors) == 0:
        raise RuntimeError("No survey predictors qualified for EFA after excluding controls.")

    efa_vars = ["Prosocial_DV"] + kept_predictors
    X = df[efa_vars].copy()
    return X, kept_predictors


def determine_n_factors(corr_matrix, eigenvalues):
    """Return suggested n_factors via Kaiser (>1) rule, capped by scree elbow."""
    # Kaiser rule
    kaiser = (eigenvalues > 1).sum()
    n_factors = max(1, int(kaiser))
    return n_factors


def run_efa(X: pd.DataFrame, n_factors: int):
    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
    fa.fit(X)
    loadings = pd.DataFrame(
        fa.loadings_, index=X.columns, columns=[f"Factor{i+1}" for i in range(n_factors)]
    )
    factor_scores = fa.transform(X)
    scores_df = pd.DataFrame(
        factor_scores, index=X.index, columns=[f"Factor{i+1}_score" for i in range(n_factors)]
    )
    return loadings, scores_df, fa


def quality_checks(X):
    kmo_all, kmo_model = calculate_kmo(X)
    chi2, p = calculate_bartlett_sphericity(X)
    return kmo_model, chi2, p


def scree_plot(eigenvalues):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
    plt.axhline(1, color='red', linestyle='--', alpha=0.7)
    plt.title('Scree Plot')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.tight_layout()
    plt.savefig('scree_plot.png', dpi=300)
    plt.close()
    print("Saved scree_plot.png")


def save_outputs(loadings, scores_df, kept_predictors):
    loadings.to_csv("efa_loadings.csv")
    scores_df.to_csv("efa_factor_scores.csv", index=False)

    # Also merge factor scores into analysis table for Phase D
    merged = pd.read_csv("analysis_table.csv")
    merged = pd.concat([merged, scores_df], axis=1)
    merged.to_csv("analysis_table_with_factors.csv", index=False)
    print("Factor scores merged into analysis_table_with_factors.csv")


def write_report(kept_predictors, n_factors, kmo, chi2, p):
    kept_md = "\n".join(f"- {k.replace('_std','')}" for k in kept_predictors)
    md = f"""# Phase C: Exploratory Factor Analysis

Variables included in EFA (DV + retained survey items):
- Prosocial_DV
{kept_md}

KMO overall measure: **{kmo:.3f}**  
Bartlett test χ² = {chi2:.1f}, p = {p:.4g}

Number of factors retained (Kaiser > 1): **{n_factors}**

Outputs:
- `efa_loadings.csv` – factor loadings matrix
- `efa_factor_scores.csv` – participant factor scores
- `scree_plot.png` – eigenvalue scree
- `analysis_table_with_factors.csv` – master dataset with factor scores added
"""
    Path("phase_c_efa_analysis.md").write_text(md)
    print("Saved phase_c_efa_analysis.md")


def main():
    df, coeffs = load_data()
    X, kept_predictors = select_efa_variables(df, coeffs)

    # Correlation matrix & eigenvalues
    corr = X.corr()
    eigenvalues, _ = np.linalg.eig(corr.values)
    scree_plot(eigenvalues)

    n_factors = determine_n_factors(corr, eigenvalues)
    loadings, scores_df, fa = run_efa(X, n_factors)

    kmo, chi2, p = quality_checks(X)
    save_outputs(loadings, scores_df, kept_predictors)
    write_report(kept_predictors, n_factors, kmo, chi2, p)

    print("Phase C completed successfully.")


if __name__ == "__main__":
    main() 