#!/usr/bin/env python3
"""
Phase C (revised): Bundle-wise Exploratory Factor Analyses
=========================================================

The professor asked that we *not* restrict the EFA to only the p<0.10 survivors.
Instead, we form conceptual bundles of related survey indicators and run an EFA
**within each bundle plus the dependent variable Prosocial_DV**.
Controls remain outside.

Bundles implemented
-------------------
1. Prosocial orientation      → SVO_type, Prosocial_Motivation, Dictator_1
2. Honesty / Kindness         → Honesty_Humility, Integrity_Honesty, Kindness, Agreeable
3. Big‐Five (task-oriented)   → ExtraversionBig5, OpennessBig5, ConscientiousnessBig5, NeuroticismBig5
4. Affect / Well-being        → PosAffect, LifeSatis, SubHappy
5. Beliefs / Ideology         → ClassSystemJust, EconomicBelief, SupportEquality, Egalitarianism,
                                 SocialOrientation, HumanVS, RSDO

For each bundle we:
• Subset the dataframe to DV + bundle items (already z-scored)  
• Inspect the correlation matrix; if <2 non-DV variables correlate >|0.2| with
  anything, we abort that bundle (too weak).  
• Otherwise run common-factor EFA (MINRES, oblimin).  
  – Retain factors with eigenvalue > 1 (max 2 to keep parsimony).  
• Save loadings and compute Bartlett regression scores.  
• Prefix score columns with the bundle code (e.g. `PROSOCIAL_F1`).

Outputs
--------
• `efa_loadings_*.csv` one per bundle  
• `efa_factor_scores.csv`  – all factor scores concatenated  
• `bundle_corr_heatmap_*.png`  – correlation heatmaps  
• `phase_c_multi_efa.md` – markdown report  
• `analysis_table_with_multi_factors.csv`  – master data incl. all new scores
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

warnings.filterwarnings("ignore")

# --------------------------- Bundle specification ---------------------------
BUNDLES = {
    "PROSOCIAL": ["SVO_type_std", "Prosocial_Motivation_std", "Dictator_1_std"],
    "HONESTY": ["Honesty_Humility_std", "Integrity_Honesty_std", "Kindness_std", "Agreeable_std"],
    "BIGFIVE": ["ExtraversionBig5_std", "OpennessBig5_std", "ConscientiousnessBig5_std", "NeuroticismBig5_std"],
    "AFFECT": ["PosAffect_std", "LifeSatis_std", "SubHappy_std"],
    "IDEOLOGY": ["ClassSystemJust_std", "EconomicBelief_std", "SupportEquality_std", "Egalitarianism_std",
                 "SocialOrientation_std", "HumanVS_std", "RSDO_std"],
}
DV = "Prosocial_DV"
MAX_FACTORS = 2

# --------------------------- Helper functions ------------------------------

def scree_plot(eigenvalues, out_path):
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    plt.axhline(1, ls="--", c="red", alpha=0.7)
    plt.title("Scree")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_heatmap(corr, out_path):
    plt.figure(figsize=(corr.shape[0] * 0.6 + 1, corr.shape[0] * 0.6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------- Main workflow ---------------------------------

def run_bundle(bundle_code, cols, df):
    report_lines = []
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:  # need at least DV + 2 indicators
        report_lines.append(f"*{bundle_code}* skipped – fewer than 2 indicators present (found {available}).")
        return None, None, report_lines

    X_raw = df[[DV] + available].dropna()
    # Standardize each column (z-score). This puts DV and indicators on common scale
    X = (X_raw - X_raw.mean()) / X_raw.std(ddof=0)

    corr = X.corr()
    heat_path = f"bundle_corr_heatmap_{bundle_code}.png"
    save_heatmap(corr, heat_path)

    # Simple correlation strength check
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    strong_pairs = (upper.abs() > 0.2).sum().sum()
    if strong_pairs < 1:
        report_lines.append(f"*{bundle_code}* correlations too weak (<0.2). Skipped.")
        return None, None, report_lines

    # EFA
    eigenvalues, _ = np.linalg.eig(corr.values)
    n_factors = int(min(MAX_FACTORS, (eigenvalues > 1).sum())) or 1

    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
    fa.fit(X)

    loadings = pd.DataFrame(
        fa.loadings_, index=X.columns, columns=[f"{bundle_code}_F{i+1}" for i in range(n_factors)]
    )
    loadings.to_csv(f"efa_loadings_{bundle_code}.csv")

    scores = fa.transform(X)  # regression scoring by default
    score_df = pd.DataFrame(
        scores, index=X.index, columns=[f"{bundle_code}_F{i+1}_score" for i in range(n_factors)]
    )
    # Standardize factor scores
    score_df = (score_df - score_df.mean()) / score_df.std(ddof=0)

    # Diagnostics
    kmo_all, kmo_overall = calculate_kmo(X)
    if kmo_overall < 0.6:
        report_lines.append(f"*{bundle_code}* KMO={kmo_overall:.2f} (<0.6). Bundle skipped.")
        return None, None, report_lines
    chi2, p_val = calculate_bartlett_sphericity(X)
    scree_plot(eigenvalues, f"scree_{bundle_code}.png")

    report_lines.append(f"### {bundle_code}\n"
                        f"Items used: {', '.join(available)}\n"
                        f"N = {len(X)} rows (complete-case within bundle)\n"
                        f"KMO = {kmo_overall:.2f}, Bartlett p = {p_val:.4g}\n"
                        f"Factors retained = {n_factors}\n")
    return loadings, score_df, report_lines


def main():
    df = pd.read_csv("analysis_table.csv")

    all_scores = []
    md_sections = ["# Phase C (revised): Bundle-wise EFA"]

    for code, cols in BUNDLES.items():
        loadings, score_df, lines = run_bundle(code, cols, df)
        md_sections.extend(lines)
        if score_df is not None:
            all_scores.append(score_df)

    if not all_scores:
        raise RuntimeError("No bundles produced factor scores – aborting.")

    scores_concat = pd.concat(all_scores, axis=1)
    scores_concat.to_csv("efa_factor_scores.csv", index=False)

    # Merge with master dataset
    merged = pd.concat([df, scores_concat], axis=1)
    merged.to_csv("analysis_table_with_multi_factors.csv", index=False)

    md_sections.append("\nOutputs generated:\n")
    md_sections.extend([f"- efa_loadings_{c}.csv" for c in BUNDLES.keys()])
    md_sections.append("- efa_factor_scores.csv (all bundles concatenated)")
    md_sections.append("- analysis_table_with_multi_factors.csv (master with scores)")
    Path("phase_c_multi_efa.md").write_text("\n".join(md_sections))
    print("Bundle-wise EFA completed – see phase_c_multi_efa.md for details.")


if __name__ == "__main__":
    main() 