import pandas as pd
import numpy as np
import json
import statsmodels.formula.api as smf

# ---------- CONFIG ----------
SURVEY_PATH = "/Users/suedagul/<sdg/Student Survey Results - Period 1.xlsx"
EXPER_PATH  = "/Users/suedagul/<sdg/Student Experiment Results - Period 1-2.xlsx"
ID_COL      = "Participant ID"

# Column names (aligned with actual headers)
DV_COL      = "TWT+Sospeso [=AW2+AX2]{Periods 1+2}"  # observed prosocial behavior
HH_COL      = "Honesty_Humility"
INCOME_COL  = "IndividualIncome"           # annual € per person (net)
PROG_COL    = "Study Program Category"     # study program/category

# Income quintile edges (Eurostat-style). Adjust if you have updated values.
BIN_EDGES = [-np.inf, 12617, 18030, 23567, 31487, np.inf]
BIN_LABELS = [1, 2, 3, 4, 5]               # Q1..Q5

# Policy weights to try (share on observed behavior in the blend)
WEIGHTS = [0.25, 0.50, 0.75]
DEFAULT_W = 0.50

# Noise scale (fraction of RMSE mapped to 0–100)
NOISE_SCALE = 0.5

# Random seed for reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# ========== 1) Load & merge ==========
survey = pd.read_excel(SURVEY_PATH)
exper  = pd.read_excel(EXPER_PATH)

for need, name in [(survey, "survey_data.xlsx"), (exper, "experiment_results.xlsx")]:
    if ID_COL not in need.columns:
        raise ValueError(f"'{ID_COL}' is missing in {name}.")

survey_keep = [c for c in [ID_COL, HH_COL, INCOME_COL, PROG_COL] if c in survey.columns]
exper_keep  = [c for c in [ID_COL, DV_COL, PROG_COL] if c in exper.columns]

survey = survey[survey_keep].copy()
exper  = exper[exper_keep].copy()

df = pd.merge(survey, exper, on=ID_COL, how="inner", suffixes=("_survey", "_exp"))

# robust coalesce across possible program columns
prog_series = None
for col in [f"{PROG_COL}_exp", f"{PROG_COL}_survey", PROG_COL]:
    if col in df.columns:
        prog_series = df[col] if prog_series is None else prog_series.combine_first(df[col])

df["study_prog"] = prog_series

df = df.rename(columns={
    DV_COL: "obs_behav",
    HH_COL: "hh",
    INCOME_COL: "income_eur"
})

# clean up any suffix leftovers
for c in [f"{PROG_COL}_exp", f"{PROG_COL}_survey", PROG_COL]:
    if c in df.columns and c != "study_prog":
        df.drop(columns=[c], inplace=True)

# ========== 2) Income bins ==========
# Map the 6 survey categories to 5 quintiles
# Original: 1,2,3,4,5,6 -> Map to: 1,2,3,4,5
income_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}  # Combine categories 5&6 into Q5
df["income_cat"] = df["income_eur"].map(income_mapping).astype("Int64")

lower_bounds = np.array(BIN_EDGES[:-1])
upper_bounds = np.array(BIN_EDGES[1:])

bin_lows  = {i+1: lower_bounds[i] for i in range(5)}
bin_highs = {i+1: upper_bounds[i] for i in range(5)}

def draw_within_bin(cat):
    if pd.isna(cat):
        return np.nan
    low, high = bin_lows[int(cat)], bin_highs[int(cat)]
    if np.isneginf(low):
        low = 0.0
    if np.isposinf(high):
        hi95 = np.nanpercentile(df["income_eur"], 95) if df["income_eur"].notna().any() else 31488
        high = max(31488.0, hi95) * 1.6
    return rng.uniform(low, high)

df["income_draw"] = df["income_cat"].apply(draw_within_bin)

# ========== 3) Standardize HH ==========
hh_mean = float(df["hh"].mean())
hh_sd   = float(df["hh"].std())
if hh_sd and hh_sd > 0:
    df["z_hh"] = (df["hh"] - hh_mean) / hh_sd
else:
    df["z_hh"] = 0.0
    hh_sd = 1.0

# ========== 4) Fit deployment regression ==========
df["income_cat"] = pd.Categorical(df["income_cat"], categories=BIN_LABELS, ordered=True)
prog_levels = sorted(pd.Series(df["study_prog"], dtype="string").dropna().unique().tolist())
df["study_prog"] = pd.Categorical(df["study_prog"], categories=prog_levels, ordered=True)

baseline_income = df["income_cat"].cat.categories[0] if len(df["income_cat"].cat.categories) else None
baseline_prog   = df["study_prog"].cat.categories[0]   if len(df["study_prog"].cat.categories) else None

model = smf.ols("obs_behav ~ z_hh + C(income_cat) + C(study_prog)", data=df).fit()

df["other_score"] = model.fittedvalues
rmse = float(np.sqrt(model.mse_resid))

# ========== 5) Scale to 0–100 ==========
y_min, y_max = float(df["obs_behav"].min()), float(df["obs_behav"].max())

def to0100(x, lo, hi):
    span = hi - lo
    if isinstance(x, pd.Series):
        return 100 * (x - lo) / span if span > 0 else pd.Series(50.0, index=x.index)
    return 100 * (x - lo) / span if span > 0 else 50.0

df["obs_0100"]   = to0100(df["obs_behav"],   y_min, y_max)
o_min, o_max     = float(df["other_score"].min()), float(df["other_score"].max())
df["other_0100"] = to0100(df["other_score"], o_min, o_max)

# ========== 6) Anchors ==========
for w in WEIGHTS:
    df[f"anchor_{int(w*100)}"] = w*df["obs_0100"] + (1-w)*df["other_0100"]

# ========== 7) Noise ==========
sigma_0100 = 100 * rmse / (y_max - y_min) if (y_max - y_min) > 0 else 5.0
for w in WEIGHTS:
    base_col = f"anchor_{int(w*100)}"
    noise = pd.Series(rng.normal(0, NOISE_SCALE * sigma_0100, size=len(df)), index=df.index)
    pred = (df[base_col] + noise).clip(0, 100)
    df[f"pred_{int(w*100)}"] = pred

# ========== 8) Save outputs ==========
out_cols = [ID_COL, "obs_behav", "obs_0100", "other_0100"] + \
           [f"anchor_{int(w*100)}" for w in WEIGHTS] + \
           [f"pred_{int(w*100)}" for w in WEIGHTS] + \
           ["income_eur", "income_cat", "income_draw", "z_hh", "study_prog"]

df[out_cols].to_excel("phase2_outputs.xlsx", index=False)

# ========== 9) Persist parameters ==========

def json_safe_edges(edges):
    safe = []
    for x in edges:
        if np.isneginf(x):
            safe.append("-inf")
        elif np.isposinf(x):
            safe.append("inf")
        else:
            safe.append(float(x))
    return safe

params = {
    "hh_mean": hh_mean,
    "hh_sd": hh_sd,
    "income_bin_edges": json_safe_edges(BIN_EDGES),
    "income_bin_labels": BIN_LABELS,
    "baseline_income_cat": None if baseline_income is None else int(baseline_income),
    "baseline_study_prog": None if baseline_prog is None else str(baseline_prog),
    "y_min": y_min,
    "y_max": y_max,
    "other_min": o_min,
    "other_max": o_max,
    "weights_tried": WEIGHTS,
    "weight_default": DEFAULT_W,
    "noise_scale": NOISE_SCALE,
    "sigma_rmse": rmse,
    "sigma_0100": float(sigma_0100),
    "study_prog_levels": [str(c) for c in df["study_prog"].cat.categories],
    "model_params": {k: float(v) for k, v in model.params.items()}
}

with open("phase2_params.json", "w") as f:
    json.dump(params, f, indent=2)

with open("phase2_model_summary.txt", "w") as f:
    f.write(model.summary().as_text())

print("Phase 2 complete. Files written: 'phase2_outputs.xlsx', 'phase2_params.json', 'phase2_model_summary.txt'") 