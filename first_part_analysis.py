import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr


# --- Paths / constants
SURVEY_PATH = "/Users/suedagul/<sdg/Student Survey Results - Period 1.xlsx"
EXPER_PATH  = "/Users/suedagul/<sdg/Student Experiment Results - Period 1-2.xlsx"
SURVEY_SHEET = 0
EXPER_SHEET  = 0

ID_COL = "Participant ID"
DV = "TWT+Sospeso [=AW2+AX2]{Periods 1+2}"   # observed prosocial behavior

# Column groups (match Stata lists)
design_cols = ["Group", "Total Allowance", "Study Program Category"]
demo_cols   = ["Age","Education","Foreign","Religious","ReligiousService",
               "IndividualIncome","TopPlatform"]
lab_cols    = ["Dictator_1"]
trait_cols1 = ["SVO_type","Prosocial_Motivation","Agreeable","Integrity_Honesty",
               "Kindness","Honesty_Humility","NeuroticismBig5","ExtraversionBig5",
               "OpennessBig5","ConscientiousnessBig5","HumanVS","PosAffect","LifeSatis","SubHappy"]
belief_cols = ["ClassSystemJust","EconomicBelief","SupportEquality","SocialOrientation","Egalitarianism"]
misc_cols   = ["RSDO"]

# =========================
# 1) Load & merge
# =========================
survey = pd.read_excel(SURVEY_PATH, sheet_name=SURVEY_SHEET)
exper  = pd.read_excel(EXPER_PATH,  sheet_name=EXPER_SHEET)

if ID_COL not in survey.columns or ID_COL not in exper.columns:
    raise ValueError(f"'{ID_COL}' must exist in BOTH files to merge.")

# Keep only needed cols (but allow extras)
survey_keep = [c for c in ([ID_COL] + demo_cols + trait_cols1 + belief_cols + misc_cols + ["studyprogramcategory"]) if c in survey.columns]
exper_keep  = [c for c in ([ID_COL] + design_cols + lab_cols + [DV]) if c in exper.columns]

survey = survey[survey_keep].copy()
exper  = exper[exper_keep].copy()

# Merge; suffix to avoid collisions
df = pd.merge(survey, exper, on=ID_COL, how="inner", suffixes=("_survey", "_exp"))

# Helper to coalesce columns from multiple sources into one
def coalesce_to(df, outcol, candidates):
    present = [c for c in candidates if c in df.columns]
    if not present:
        return
    base = df[present[0]].copy()
    for c in present[1:]:
        base = base.combine_first(df[c])
    df[outcol] = base

# Coalesce overlapping columns (prefer experiment values for design/DV)
coalesce_to(df, "Group",                 ["Group_exp","Group_survey"])
coalesce_to(df, "Total Allowance",       ["Total Allowance_exp","Total Allowance_survey"])
coalesce_to(df, "Study Program Category", ["Study Program Category_exp","Study Program Category_survey"])
coalesce_to(df, "Dictator_1",            ["Dictator_1_exp","Dictator_1_survey"])
coalesce_to(df, DV,                      [f"{DV}_exp", f"{DV}_survey"])

# Drop the suffixed duplicates to clean up
drop_cols = [c for c in df.columns if c.endswith("_exp") or c.endswith("_survey")]
df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# 2) Descriptives (Stata: summarize)
# =========================
all_cols = [DV] + design_cols + demo_cols + lab_cols + trait_cols1 + belief_cols + misc_cols
all_cols = [c for c in all_cols if c in df.columns]
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

desc = df[num_cols].describe()  # mean, std, min, 25%, 50%, 75%, max (closest to Stata summarize)
print("\n=== DESCRIPTIVES (summarize; numeric only) ===")
print(desc)

# =========================
# 3) Correlations (Stata: correlate)
# =========================
def corr_with_pvals(dataframe):
    cols = dataframe.columns
    n = len(cols)
    R = pd.DataFrame(np.nan, index=cols, columns=cols)
    P = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in range(n):
        for j in range(i, n):
            s1, s2 = dataframe[cols[i]], dataframe[cols[j]]
            ok = s1.notna() & s2.notna()
            if ok.sum() >= 3:
                r, p = pearsonr(s1[ok], s2[ok])
                R.iloc[i, j] = R.iloc[j, i] = r
                P.iloc[i, j] = P.iloc[j, i] = p
    return R, P

R, P = corr_with_pvals(df[num_cols])
print("\n=== CORRELATION MATRIX (pearson r; pairwise) ===")
print(R.round(3))

# =========================
# 4) Kitchen-sink regressions (diagnostic)
# =========================
# Build categorical helpers
if "Group" in df.columns:
    df["groupcat"] = df["Group"].astype("category")
if "Study Program Category" in df.columns:
    df["studyprogramcategorycat"] = df["Study Program Category"].astype("category")

# ---- Kitchen sink #1 (matches first Stata regress)
form1_terms = []
if "groupcat" in df.columns: form1_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form1_terms.append("Q('Total Allowance')")
for c in ["Age","Education"]:
    if c in df.columns: form1_terms.append(c)
if "studyprogramcategorycat" in df.columns: form1_terms.append("C(studyprogramcategorycat)")
for c in ["Foreign","Religious","ReligiousService","IndividualIncome","TopPlatform",
          "Dictator_1","SVO_type","Prosocial_Motivation","Agreeable","Integrity_Honesty",
          "Kindness","Honesty_Humility","NeuroticismBig5","ExtraversionBig5","OpennessBig5",
          "ConscientiousnessBig5","HumanVS","PosAffect","LifeSatis","SubHappy",
          "ClassSystemJust","EconomicBelief","SupportEquality","SocialOrientation","Egalitarianism","RSDO"]:
    if c in df.columns: form1_terms.append(c)

form1 = f"Q('{DV}') ~ " + " + ".join(form1_terms)
m1 = smf.ols(formula=form1, data=df, missing="drop").fit()  # classical SEs
print("\n=== KITCHEN SINK #1 ===")
print(m1.summary())

# ---- Kitchen sink #2 (trimmed pre-factor; matches second Stata regress)
form2_terms = []
if "groupcat" in df.columns: form2_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form2_terms.append("Q('Total Allowance')")
if "Age" in df.columns: form2_terms.append("Age")
if "studyprogramcategorycat" in df.columns: form2_terms.append("C(studyprogramcategorycat)")
for c in ["Foreign","Religious","ReligiousService","IndividualIncome","TopPlatform",
          "Dictator_1","SVO_type","Prosocial_Motivation","Integrity_Honesty","Kindness",
          "Honesty_Humility","NeuroticismBig5","ExtraversionBig5","OpennessBig5",
          "ConscientiousnessBig5","HumanVS","PosAffect","LifeSatis",
          "ClassSystemJust","EconomicBelief","SupportEquality","SocialOrientation","Egalitarianism"]:
    if c in df.columns: form2_terms.append(c)

form2 = f"Q('{DV}') ~ " + " + ".join(form2_terms)
m2 = smf.ols(formula=form2, data=df, missing="drop").fit()
print("\n=== KITCHEN SINK #2 (trimmed, pre-factor) ===")
print(m2.summary())

# =========================
# 5) EFA — belief/justice/equality block → f1
#     Stata: factor ..., mineigen(1); alpha ...; predict f1
# =========================
belief_items = [c for c in belief_cols if c in df.columns]

def cronbach_alpha(frame_raw):
    frame = frame_raw.dropna()
    if frame.shape[1] < 2 or frame.shape[0] < 2:
        return np.nan
    k = frame.shape[1]
    item_vars = frame.var(axis=0, ddof=1)
    total_var = frame.sum(axis=1).var(ddof=1)
    return (k/(k-1)) * (1 - item_vars.sum()/total_var)

if len(belief_items) >= 2:
    # z-scores for FA logic; alpha on RAW items (Stata-like)
    belief_raw = df[belief_items]
    belief_z = belief_raw.apply(lambda s: (s - s.mean())/s.std())

    # Unrotated principal factor; choose #factors by eigenvalues > 1
    fa0 = FactorAnalyzer(rotation=None, method="principal")
    fa0.fit(belief_z.dropna())
    eigs, _ = fa0.get_eigenvalues()
    n_factors = int((eigs > 1).sum()) or 1

    fa1 = FactorAnalyzer(n_factors=n_factors, rotation=None, method="principal")
    fa1.fit(belief_z.dropna())

    alpha_belief = cronbach_alpha(belief_raw)
    print(f"\n=== EFA (belief block) ===\nEigenvalues: {np.round(eigs,3)}"
          f"\nkept factors: {n_factors}\nAlpha (raw): {alpha_belief:.3f}")

    # Factor scores: no imputation; scores missing if any item missing (Stata behavior)
    valid = belief_z.dropna()
    scores_b = pd.Series(np.nan, index=belief_z.index, dtype=float)
    scores_b.loc[valid.index] = fa1.transform(valid)[:, 0]  # regression scores
    df["f1"] = scores_b
else:
    print("\n[Warn] Not enough belief items to form f1; skipping.")
    df["f1"] = np.nan

# =========================
# 6) Regression with f1
#     Stata: regress DV i.groupcat totalallowance age i.studyprogramcategorycat ... f1
# =========================
form3_terms = []
if "groupcat" in df.columns: form3_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form3_terms.append("Q('Total Allowance')")
for c in ["Age"]:
    if c in df.columns: form3_terms.append(c)
if "studyprogramcategorycat" in df.columns: form3_terms.append("C(studyprogramcategorycat)")
for c in ["Foreign","Religious","ReligiousService","IndividualIncome","TopPlatform",
          "Dictator_1","SVO_type","Prosocial_Motivation","Integrity_Honesty","Kindness",
          "Honesty_Humility","NeuroticismBig5","ExtraversionBig5","OpennessBig5",
          "ConscientiousnessBig5","HumanVS","PosAffect","LifeSatis","f1"]:
    if c in df.columns: form3_terms.append(c)

form3 = f"Q('{DV}') ~ " + " + ".join(form3_terms)
m3 = smf.ols(formula=form3, data=df, missing="drop").fit()
print("\n=== MODEL with f1 ===")
print(m3.summary())

# =========================
# 7) EFA — prosocial/HH block → f2
#     Stata: factor svo_type prosocial_motivation integrity_honesty kindness honesty_humility, mineigen(1)
#            alpha ... ; predict f2
# =========================
pros_items_all = ["SVO_type","Prosocial_Motivation","Integrity_Honesty","Kindness","Honesty_Humility"]
prosocial_items = [c for c in pros_items_all if c in df.columns]

if len(prosocial_items) >= 2:
    pros_raw = df[prosocial_items]
    pros_z = pros_raw.apply(lambda s: (s - s.mean())/s.std())

    fa0b = FactorAnalyzer(rotation=None, method="principal")
    fa0b.fit(pros_z.dropna())
    eigs_b, _ = fa0b.get_eigenvalues()
    n_factors_b = int((eigs_b > 1).sum()) or 1

    fa2 = FactorAnalyzer(n_factors=n_factors_b, rotation=None, method="principal")
    fa2.fit(pros_z.dropna())

    alpha_prosocial = cronbach_alpha(pros_raw)
    print(f"\n=== EFA (prosocial block) ===\nEigenvalues: {np.round(eigs_b,3)}"
          f"\nkept factors: {n_factors_b}\nAlpha (raw): {alpha_prosocial:.3f}")

    valid_p = pros_z.dropna()
    scores_p = pd.Series(np.nan, index=pros_z.index, dtype=float)
    scores_p.loc[valid_p.index] = fa2.transform(valid_p)[:, 0]
    df["f2"] = scores_p
else:
    print("\n[Warn] Not enough prosocial items to form f2; skipping.")
    df["f2"] = np.nan

# =========================
# 8) Regression with f1 + f2
#     Stata: regress DV i.groupcat totalallowance age i.studyprogramcategorycat ... f2 ... f1
# =========================
form4_terms = []
if "groupcat" in df.columns: form4_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form4_terms.append("Q('Total Allowance')")
for c in ["Age"]:
    if c in df.columns: form4_terms.append(c)
if "studyprogramcategorycat" in df.columns: form4_terms.append("C(studyprogramcategorycat)")
for c in ["Foreign","Religious","ReligiousService","IndividualIncome","TopPlatform",
          "Dictator_1","f2","NeuroticismBig5","ExtraversionBig5","OpennessBig5",
          "ConscientiousnessBig5","HumanVS","PosAffect","LifeSatis","f1"]:
    if c in df.columns: form4_terms.append(c)

form4 = f"Q('{DV}') ~ " + " + ".join(form4_terms)
m4 = smf.ols(formula=form4, data=df, missing="drop").fit()
print("\n=== MODEL with f1 + f2 ===")
print(m4.summary())

# =========================
# 9) Parsimony model
#     Stata: regress DV i.groupcat totalallowance i.studyprogramcategorycat foreign religious individualincome topplatform dictator_1 f2 extraversionbig5 conscientiousnessbig5 humanvs lifesatis f1
# =========================
form5_terms = []
if "groupcat" in df.columns: form5_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form5_terms.append("Q('Total Allowance')")  # continuous here
if "studyprogramcategorycat" in df.columns: form5_terms.append("C(studyprogramcategorycat)")
for c in ["Foreign","Religious","IndividualIncome","TopPlatform","Dictator_1","f2",
          "ExtraversionBig5","ConscientiousnessBig5","HumanVS","LifeSatis","f1"]:
    if c in df.columns: form5_terms.append(c)

form5 = f"Q('{DV}') ~ " + " + ".join(form5_terms)
m5 = smf.ols(formula=form5, data=df, missing="drop").fit()
print("\n=== PARSIMONY MODEL ===")
print(m5.summary())

# =========================
# 10) Structural-only and Structural+HH
#     Stata: regress DV i.groupcat i.totalallowance i.studyprogramcategorycat
#            regress DV i.groupcat i.totalallowance i.studyprogramcategorycat honesty_humility
# =========================
form6_terms = []
if "groupcat" in df.columns: form6_terms.append("C(groupcat)")
if "Total Allowance" in df.columns: form6_terms.append("C(Q('Total Allowance'))")
if "studyprogramcategorycat" in df.columns: form6_terms.append("C(studyprogramcategorycat)")

form6 = f"Q('{DV}') ~ " + " + ".join(form6_terms)
m6 = smf.ols(formula=form6, data=df, missing="drop").fit()
print("\n=== STRUCTURAL-ONLY MODEL ===")
print(m6.summary())

form7_terms = form6_terms.copy()
if "Honesty_Humility" in df.columns:
    form7_terms = form7_terms + ["Honesty_Humility"]

form7 = f"Q('{DV}') ~ " + " + ".join(form7_terms)
m7 = smf.ols(formula=form7, data=df, missing="drop").fit()
print("\n=== STRUCTURAL+HH MODEL ===")
print(m7.summary())

# =========================
# 11) Save outputs
# =========================
with pd.ExcelWriter("first_part_outputs.xlsx", engine="xlsxwriter") as writer:
    desc.to_excel(writer, sheet_name="summarize")
    R.round(3).to_excel(writer, sheet_name="corr_r")
    P.round(3).to_excel(writer, sheet_name="corr_p")
    out_scores = df[[ID_COL]].copy()
    if "f1" in df.columns: out_scores["f1"] = df["f1"]
    if "f2" in df.columns: out_scores["f2"] = df["f2"]
    out_scores.to_excel(writer, sheet_name="factor_scores", index=False)

print("\nDone. Outputs saved to 'first_part_outputs.xlsx'.") 