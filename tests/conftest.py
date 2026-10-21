"""
Fixtures and helpers shared by the test modules.

Only what several modules had as IDENTICAL copies lives here; a module whose copy
differed keeps its own (a module-level fixture of the same name overrides the one
below). Plain helpers are imported explicitly - ``from conftest import REPO`` - the
way the modules already import ``browser_sim`` and each other.
"""
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.seam.config_repo import get_config_repo

# --------------------------------------------------------------------------- paths
# The repository root, in the two forms the modules use.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # str
ROOT = Path(__file__).resolve().parents[1]                           # pathlib.Path

APP_FILE = str(ROOT / "app_enhanced_new.py")
DECISIONS_YAML = os.path.join(REPO, "config", "decisions.yaml")
D4_VERIFICATION_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")


def load_decisions_yaml():
    """config/decisions.yaml, freshly parsed (every call returns a new dict)."""
    with open(DECISIONS_YAML) as f:
        return yaml.safe_load(f)


def start_app():
    """The real app (app_enhanced_new.py) in Streamlit's AppTest, not yet run."""
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP_FILE, default_timeout=300)


# ------------------------------------------------------- reference files outside the repo
DTA_SKIP_ENV = "COOPECON_ALLOW_DTA_SKIP"


def require_reference_file(path, constant):
    """Fail when the reference file at ``path`` is missing, naming the file and the
    constant that points at it. Skip instead only when COOPECON_ALLOW_DTA_SKIP=1."""
    if os.path.exists(path):
        return
    message = (f"reference file missing: {path!r} - put the file there or update "
               f"{constant} (set {DTA_SKIP_ENV}=1 to skip instead)")
    if os.environ.get(DTA_SKIP_ENV) == "1":
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


# ------------------------------------------------- Decision 4 reference extract (280 rows)
@pytest.fixture(scope="module")
def gold():
    return pd.read_csv(D4_VERIFICATION_CSV)


@pytest.fixture(scope="module")
def sim_config(gold):
    # egen std semantics: sample SD (ddof=1) over the reference file's income column
    return {
        "income_stats": {
            "mean": float(gold["income"].mean()),
            "sd": float(gold["income"].std(ddof=1)),
        }
    }


# ------------------------------------------------------------------- the run seam
@pytest.fixture(scope="module")
def repo():
    return get_config_repo()


@pytest.fixture(scope="module")
def default_values():
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    return DEFAULT_DECISION_VALUES
