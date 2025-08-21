# src/validate_traits.py
import pandas as pd, sys
from pathlib import Path
from src.build_master_traits import get_master_trait_list

SURVEY_PATH     = Path(__file__).resolve().parents[1] / "data" / "Student Survey Results - Period 1.xlsx"
EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "data" / "Student Experiment Results - Period 1-2.xlsx"

traits = get_master_trait_list()

survey     = pd.read_excel(SURVEY_PATH, sheet_name=0)
experiment = pd.read_excel(EXPERIMENT_PATH, sheet_name=0)
merged     = survey.merge(experiment, on="Participant ID", how="inner",
                          suffixes=("_survey", "_experiment"))

missing = [c for c in traits if c not in merged.columns]
if missing:
    print("❌  Missing columns:", missing)
    sys.exit(1)
print("✅  All required traits found.")