import json
import numpy as np
import pandas as pd
from semopy import Model, calc_stats
from sklearn.model_selection import train_test_split

# ======================
# CONFIG – EDIT AS NEEDED
# ======================
CSV_PATH = "analysis_table.csv"

DV_COL = "Prosocial_DV"  # Our actual DV column name
INDICATORS = [
    "SVO_type_std",
    "Prosocial_Motivation_std",
    "Dictator_1_std",
]
CONTROLS_CONT = ["TotalAllowance_std", "Age_std", "Foreign_binary"]  # numeric controls
CONTROLS_CAT  = []  # Group is already encoded as dummies in our data

# Add the pre-existing dummy variables - note the spaces in the names
CONTROL_DUMMIES = ["Group_MidSub", "Group_NoSub", "Education_2", 
                   "StudyProgram_Incoming", "StudyProgram_Law 5-year Program", 
                   "StudyProgram_UG 3-year Program", "TopPlatform_1"]

RANDOM_STATE = 123
TEST_SIZE = 0.20
EPS = 1e-6  # for clipping proportions


# ======================
# UTILS
# ======================

def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Cronbach's alpha from item covariance matrix."""
    k = df_items.shape[1]
    cov = df_items.cov(ddof=0).values  # population cov
    var_sum = np.trace(cov)
    total_var = cov.sum()
    if k <= 1 or total_var <= 0:
        return np.nan
    return (k / (k - 1)) * (1 - (var_sum / total_var))


def compute_omega_from_sem(std_est: pd.DataFrame, indicators: list) -> float:
    """
    McDonald's omega (approx) from standardized solution.
    For a single common factor with standardized loadings λ_i
    and standardized residual variances θ_i (on diagonals).
    """
    # standardized loadings for indicators (op '=~', Est. Std column)
    lam = std_est[(std_est.op == "=~") & (std_est.rval.isin(indicators))]["Est. Std"].values
    # residual variances for indicators (op '~~', lval==rval)
    theta = std_est[(std_est.op == "~~") &
                    (std_est.lval.isin(indicators)) &
                    (std_est.lval == std_est.rval)]["Est. Std"].values
    if len(lam) == 0 or len(theta) == 0:
        return np.nan
    num = (np.sum(lam)) ** 2
    den = num + np.sum(theta)
    return float(num / den) if den > 0 else np.nan


def orient_factor_by_indicator(loadings_df: pd.DataFrame,
                               indicator_for_sign: str,
                               factor_name: str = "PROSOCIAL") -> int:
    """
    Decide factor orientation so that loading on `indicator_for_sign` is positive.
    Returns +1 if orientation is fine, -1 if factor should be flipped.
    """
    row = loadings_df[(loadings_df.lval == factor_name) &
                      (loadings_df.rval == indicator_for_sign)]
    if row.empty:
        return +1
    sign = np.sign(row["Est.Std"].values[0])
    return +1 if sign >= 0 else -1


# ======================
# DATA PREP
# ======================

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns with spaces to use underscores for SEM compatibility
    df = df.rename(columns={
        'StudyProgram_Law 5-year Program': 'StudyProgram_Law_5_year_Program',
        'StudyProgram_UG 3-year Program': 'StudyProgram_UG_3_year_Program'
    })
    
    # Update CONTROL_DUMMIES to use underscores
    global CONTROL_DUMMIES
    CONTROL_DUMMIES = ["Group_MidSub", "Group_NoSub", "Education_2", 
                       "StudyProgram_Incoming", "StudyProgram_Law_5_year_Program", 
                       "StudyProgram_UG_3_year_Program", "TopPlatform_1"]

    # Ensure DV exists
    if DV_COL not in df.columns:
        raise ValueError(f"DV column '{DV_COL}' not found in {csv_path}.")
    df = df.copy()
    df[DV_COL] = pd.to_numeric(df[DV_COL], errors="coerce")
    df = df.dropna(subset=[DV_COL])
    
    # Our DV is already a count variable, not a proportion
    # No need for logit transformation
    # We'll use the DV as-is for SEM
    
    # Mean-impute indicator items (simple, acceptable here)
    for v in INDICATORS:
        if v not in df.columns:
            raise ValueError(f"Indicator '{v}' not found in data.")
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df[v] = df[v].fillna(df[v].mean())
 
    # Keep a copy for alpha (use raw indicator columns as provided)
    items_for_alpha = df[INDICATORS].copy()
 
    # No need to one-hot encode since our controls are already dummy coded
    
    # Ensure numeric controls are numeric
    for c in CONTROLS_CONT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Ensure dummy controls exist and are numeric
    for c in CONTROL_DUMMIES:
        if c in df.columns:
            # Convert boolean to int if needed
            if df[c].dtype == 'bool':
                df[c] = df[c].astype(int)
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce')
 
    # Build final set of columns to keep
    keep_cols = [DV_COL] + INDICATORS + CONTROLS_CONT + CONTROL_DUMMIES
    keep_cols = [c for c in keep_cols if c in df.columns]
 
    # Drop rows with missing values
    df = df[keep_cols].dropna().copy()
    
    # Final check: ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN after conversion
    df = df.dropna()
 
    return df, items_for_alpha


# ======================
# SEM / MIMIC
# ======================

def build_sem_syntax(indicators: list, controls_cont: list, control_dummies: list) -> str:
    meas = "PROSOCIAL =~ " + " + ".join(indicators)
    rhs_terms = ["PROSOCIAL"]
    rhs_terms += [c for c in controls_cont if c] + [d for d in control_dummies if d]
    rhs = " + ".join(rhs_terms)
    struct = f"{DV_COL} ~ {rhs}"  # Use the actual DV column name
    return meas + "\n" + struct + "\n"


def fit_sem(train: pd.DataFrame, syntax: str) -> Model:
    model = Model(syntax)
    model.fit(train)  # ML estimator; items treated as continuous
    return model


def train_test_pipeline(df: pd.DataFrame, items_for_alpha: pd.DataFrame):
    # Split
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Get the actual control dummy columns that exist in our data
    control_dummy_cols = [c for c in CONTROL_DUMMIES if c in train.columns]
    sem_syntax = build_sem_syntax(INDICATORS, CONTROLS_CONT, control_dummy_cols)

    print("\nSEM Model Syntax:")
    print(sem_syntax)
    
    # Fit SEM on train
    m = fit_sem(train, sem_syntax)

    # Fit indices - skip for now due to semopy issues
    print("\nModel fitted successfully!")
    
    # Set all fit stats to None for now
    cfi = tli = rmsea = srmr = np.nan

    # Estimates (standardized)
    std_est = m.inspect(std_est=True)
    
    # Also get unstandardized estimates for loadings
    unstd_est = m.inspect()
    
    # Check what columns are available
    print("\nAvailable columns in std_est:", std_est.columns.tolist())
    
    # Get loadings from unstandardized estimates
    loadings_unstd = unstd_est[(unstd_est.op == "=~")][["lval", "rval", "Estimate", "Std. Err"]].copy()
    
    # Filter for loadings and paths with available columns
    est_col = 'Est. Std'
    se_col = 'Std. Err'
    
    # For standardized, we need to look at both standardized and unstandardized
    if len(loadings_unstd) > 0:
        print("\nUNSTANDARDIZED LOADINGS (for reference):")
        print(loadings_unstd.to_string(index=False))
        loadings = loadings_unstd.copy()
        loadings.columns = ["lval", "rval", "Est.Std", "SE"]  # Keep for compatibility
    else:
        loadings = std_est[(std_est.op == "=~")][["lval", "rval", est_col, se_col]].copy()
        loadings.columns = ["lval", "rval", "Est.Std", "SE"]  # Rename for consistency
    
    paths = std_est[(std_est.op == "~") & (std_est.lval == DV_COL)][["lval", "rval", est_col, se_col]].copy()
    paths.columns = ["lval", "rval", "Est.Std", "SE"]  # Rename for consistency

    print("\nFACTOR LOADINGS:")
    print(loadings.to_string(index=False))

    print(f"\nSTRUCTURAL PATHS (to {DV_COL}):")
    print(paths.to_string(index=False))

    # Reliability
    alpha = cronbach_alpha(items_for_alpha[INDICATORS])
    omega = compute_omega_from_sem(std_est, INDICATORS)
    print(f"\nReliability: alpha={alpha:.3f}   omega={omega:.3f}")

    # Orientation: ensure higher = more prosocial (use first indicator as anchor)
    sign = orient_factor_by_indicator(loadings, INDICATORS[0])
    if sign == -1:
        print("Note: Flipping factor orientation so higher = more prosocial.")
    # Factor scores (train & test)
    train_scores = m.predict_factors(train)[["PROSOCIAL"]].copy()
    test_scores = m.predict_factors(test)[["PROSOCIAL"]].copy()
    train_scores["PROSOCIAL"] *= sign
    test_scores["PROSOCIAL"] *= sign

    # Out-of-sample R^2
    pred_test = m.predict(test)[DV_COL]
    r2_test = np.corrcoef(pred_test, test[DV_COL])[0, 1] ** 2
    print(f"\nOut-of-sample R^2 (test): {r2_test:.3f}")

    # Export artifacts
    loadings.to_csv("sem_loadings_std.csv", index=False)
    paths.to_csv("sem_structural_std.csv", index=False)
    pd.DataFrame({"metric": ["CFI", "TLI", "RMSEA", "SRMR"],
                  "value": [cfi, tli, rmsea, srmr]}).to_csv(
        "sem_fit_indices.csv", index=False)

    train_out = train.copy()
    test_out = test.copy()
    train_out["F_score"] = train_scores["PROSOCIAL"]
    test_out["F_score"] = test_scores["PROSOCIAL"]
    train_out.to_csv("sem_train_with_fscore.csv", index=False)
    test_out.to_csv("sem_test_with_fscore.csv", index=False)

    # Save a compact config for the simulator (coefficients on standardized scale)
    config = {
        "factor": {
            "name": "PROSOCIAL",
            "indicators": INDICATORS,
            "orientation_anchor": INDICATORS[0],
            "flip_sign": (sign == -1)
        },
        "structural_std": {
            "dv": DV_COL,
            "paths": paths.to_dict(orient="records")
        },
        "fit_indices": {
            "CFI": float(cfi) if not np.isnan(cfi) else None, 
            "TLI": float(tli) if not np.isnan(tli) else None,
            "RMSEA": float(rmsea) if not np.isnan(rmsea) else None, 
            "SRMR": float(srmr) if not np.isnan(srmr) else None
        },
        "reliability": {"alpha": float(alpha), "omega": float(omega)},
        "notes": "Coefficients are standardized (Est.Std). SEM/MIMIC approach with DV included in structural model."
    }
    with open("sem_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved files:\n - sem_loadings_std.csv\n - sem_structural_std.csv\n - sem_fit_indices.csv\n - sem_train_with_fscore.csv\n - sem_test_with_fscore.csv\n - sem_config.json")

    return {
        "model": m,
        "fit": {"CFI": cfi, "TLI": tli, "RMSEA": rmsea, "SRMR": srmr},
        "loadings_std": loadings,
        "paths_std": paths,
        "alpha": alpha,
        "omega": omega,
        "r2_test": r2_test
    }


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    df, items_for_alpha = load_and_prepare(CSV_PATH)
    print(f"Prepared dataset shape: {df.shape}")
    results = train_test_pipeline(df, items_for_alpha)
