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
# Participant ID. Used by Decision 4's Cognitive Flexibility mechanism (doc Section 5).
STDACTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "stata_stdactions.csv"

traits = get_master_trait_list()

survey     = pd.read_excel(SURVEY_PATH, sheet_name=0)
experiment = pd.read_excel(EXPERIMENT_PATH, sheet_name=0)
merged     = survey.merge(experiment, on="Participant ID", how="inner",
                          suffixes=("_survey", "_experiment"))
if STDACTIONS_PATH.exists():
    stdactions = pd.read_csv(STDACTIONS_PATH)
    merged = merged.merge(stdactions, on="Participant ID", how="left")

missing = [c for c in traits if c not in merged.columns]
if missing:
    print("❌  Missing columns:", missing)
    sys.exit(1)
print("✅  All required traits found.")