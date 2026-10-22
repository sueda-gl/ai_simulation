"""The results modules after the split of app/components.py and transaction_viz.py.

* ``app/components.py`` no longer exists; nothing in app_enhanced_new.py or under app/,
  tests/ or scripts/ imports it.
* ``viz_helpers.yes_no_pie`` replaced four token-identical copies of the Y/N pie;
  it must keep drawing exactly the figure those copies drew (their ``px.pie`` call,
  re-typed here, on a golden Disclose Income frame).
"""
import ast
import gzip
import io

import numpy as np
import pandas as pd

from conftest import ROOT

def _imports_of_app_components(path):
    """The names `path` imports from app.components (empty set: none)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module == "app.components":
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Import) and any(a.name == "app.components" for a in node.names):
            names.add("*")
    return names


def test_nothing_imports_app_components_and_it_is_gone():
    assert not (ROOT / "app" / "components.py").exists()
    files = ([ROOT / "app_enhanced_new.py"] + sorted((ROOT / "app").rglob("*.py"))
             + sorted((ROOT / "tests").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")))
    importers = [str(p.relative_to(ROOT)) for p in files
                 if "__pycache__" not in p.parts and _imports_of_app_components(p)]
    assert importers == []


def _same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return type(a) is type(b) and a == b


def test_yes_no_pie_is_the_figure_the_four_copies_drew():
    import plotly.express as px
    from app.pages.results.visualizations.viz_helpers import yes_no_pie

    golden = ROOT / "tests/golden/single_disclose_income_baseline/categorical.frame.json.gz"
    df = pd.read_json(io.StringIO(gzip.decompress(golden.read_bytes()).decode("utf-8")),
                      orient="split")
    value_counts = df["disclose_income"].value_counts()
    assert set(value_counts.index) == {"Y", "N"}
    # the expression each of the four copies spelled out
    copy = px.pie(values=value_counts.values, names=value_counts.index,
                  color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'})
    assert _same(yes_no_pie(value_counts).to_dict(), copy.to_dict())
