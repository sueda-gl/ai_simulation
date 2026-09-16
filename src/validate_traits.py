# src/validate_traits.py
import pandas as pd, sys
from pathlib import Path
from src.build_master_traits import get_master_trait_list

SURVEY_PATH     = Path(__file__).resolve().parents[1] / "data" / "Student Survey Results - Period 1.xlsx"
EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "data" / "Student Experiment Results - Period 1-2.xlsx"
# Per-participant standard deviation in the number of actions per cycle over the eight
# experiment cycle-weeks ('stdactions'). The experiment workbook only carries per-
# participant totals, so this variable is taken from the professor's Stata file
# (Stata_File_Decision4_290826.dta, identical in the 050626 file), keyed by
# Participant ID. Used by Decision 4's Flexibility mechanism (doc Section 5).
STDACTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "stata_stdactions.csv"
# The professor's ONE fixed per-participant continuous income. Every income-using result
# in his documents (Decision 4 WTP / Risk-Taking, Decision 1/2 continuous modes) is
# computed with this column, which is identical across Stata_File_Decision4_050926.dta,
# Stata_File_Decision 2_260626 - CORRECTED.dta and Stata_File_Decision 1.dta. Attaching
# it here makes the Research Baseline / Research Specification populations (which run
# these original participants) reproduce the Stata file exactly; Copula populations have
# no Participant ID and keep the Page-1 generated income (see decisions/income_utils.py).
# float_precision="round_trip" is required: pandas' default CSV float parser is not
# correctly rounded and loses the last ULP on two of the 280 values.
INCOMES_PATH = Path(__file__).resolve().parents[1] / "data" / "stata_incomes.csv"

traits = get_master_trait_list()

survey     = pd.read_excel(SURVEY_PATH, sheet_name=0)
experiment = pd.read_excel(EXPERIMENT_PATH, sheet_name=0)
merged     = survey.merge(experiment, on="Participant ID", how="inner",
                          suffixes=("_survey", "_experiment"))
if STDACTIONS_PATH.exists():
    stdactions = pd.read_csv(STDACTIONS_PATH)
    merged = merged.merge(stdactions, on="Participant ID", how="left")
if INCOMES_PATH.exists():
    incomes = pd.read_csv(INCOMES_PATH, float_precision="round_trip")
    merged = merged.merge(incomes[["Participant ID", "income"]],
                          on="Participant ID", how="left")
    n_missing_income = int(merged["income"].isna().sum())
    if n_missing_income:
        print(f"⚠️  {n_missing_income}/{len(merged)} participants have no income in "
              f"{INCOMES_PATH.name}; their income will be generated from the Page-1 "
              "distribution instead.")

missing = [c for c in traits if c not in merged.columns]
if missing:
    print("❌  Missing columns:", missing)
    sys.exit(1)
print("✅  All required traits found.")