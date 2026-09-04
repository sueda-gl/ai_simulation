"""
Import-boundary test: `src/` is the engine and must not depend on the UI.

Rule under test: importing ANY module under `src/` must never drag `streamlit`
or the `app` package into `sys.modules`. The engine is the layer the CLI
(`scripts/run_simulation.py`), the Monte-Carlo subprocess and the tests import;
if it reaches back into `app/` it can only run inside a Streamlit script context,
and a stray `try: from app... except ImportError` silently changes engine
behaviour depending on whether Streamlit happened to be installed/importable.

Each module is imported in a FRESH subprocess (a plain interpreter, no pytest
plugins, no already-imported `app`/`streamlit` from a sibling test) and the
child reports its own `sys.modules` back. Doing it in-process would be
meaningless: another test in the same session may already have imported
Streamlit.

"Module starting with 'app'" is checked as the *top-level package* `app`
(`app` itself or `app.<something>`), not as a raw string prefix -- a raw prefix
would false-positive on unrelated third-party distributions such as `appnope`
(pulled in by matplotlib on macOS) or `appdirs`.

The subprocess check only sees IMPORT-TIME leakage. A `from app... import ...`
hidden inside a function body would pass it while still breaking the boundary
the first time that function runs outside Streamlit, so a second, static AST
check forbids the import statement itself anywhere in a src/ file.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

# Prefer the project's own venv interpreter (that is what the app and the CLI
# run under); fall back to the interpreter running pytest.
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

MARKER = "__IMPORT_BOUNDARY__"

# Executed by the child interpreter. Imports the module named on argv, then
# reports every UI module that ended up in sys.modules as a result.
PROBE = (
    "import sys, json, importlib\n"
    "name = sys.argv[1]\n"
    "importlib.import_module(name)\n"
    "leaked = sorted(\n"
    "    m for m in list(sys.modules)\n"
    "    if m == 'streamlit' or m.startswith('streamlit.')\n"
    "    or m == 'app' or m.startswith('app.')\n"
    ")\n"
    "sys.stdout.write('\\n' + %r + json.dumps(leaked) + '\\n')\n" % MARKER
)


def _discover_src_modules():
    """Every importable module/package under src/, as dotted names."""
    modules = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PROJECT_ROOT)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]          # src/decisions/__init__.py -> src.decisions
        else:
            parts[-1] = parts[-1][:-3]  # strip .py
        if not parts:
            continue
        modules.append(".".join(parts))
    return modules


SRC_MODULES = _discover_src_modules()


def _import_in_fresh_subprocess(module_name):
    """Import `module_name` in a clean interpreter.

    Returns (returncode, leaked_modules|None, stdout, stderr).
    `leaked_modules` is None when the import itself failed.
    """
    env = dict(os.environ)
    # The engine is imported as `src.<...>` from the project root, exactly the
    # way scripts/run_simulation.py and the app do it.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [PYTHON, "-c", PROBE, module_name],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    leaked = None
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            leaked = json.loads(line[len(MARKER):])
    return proc.returncode, leaked, proc.stdout, proc.stderr


def _tail(text, n=40):
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:])


def _brief(leaked, n=12):
    """Streamlit drags in ~250 submodules; show the head plus a count."""
    if len(leaked) <= n:
        return str(leaked)
    return f"{leaked[:n]} ... (+{len(leaked) - n} more, {len(leaked)} total)"


def test_src_modules_were_discovered():
    """Guard against the walk silently finding nothing (then everything 'passes')."""
    assert len(SRC_MODULES) >= 20, f"only found {SRC_MODULES}"
    for expected in ("src.orchestrator", "src.orchestrator_baseline",
                     "src.orchestrator_doc_mode", "src.trait_engine",
                     "src.decisions.rejected_transaction_defaults"):
        assert expected in SRC_MODULES, f"{expected} missing from {SRC_MODULES}"


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_src_module_does_not_import_ui(module_name):
    """No module under src/ may pull in streamlit or the app package."""
    rc, leaked, out, err = _import_in_fresh_subprocess(module_name)

    if rc != 0:
        # Only src/orchestrator_depvar.py is allowed to be unimportable, and
        # only because it cannot import without streamlit. Anything else is a
        # real breakage and must fail the test.
        if module_name == "src.orchestrator_depvar" and "streamlit" in (err or ""):
            pytest.skip(
                "src.orchestrator_depvar cannot be imported without streamlit; "
                f"skipped per spec. stderr tail:\n{_tail(err)}"
            )
        pytest.fail(
            f"importing {module_name} in a fresh interpreter failed "
            f"(rc={rc}).\nstderr:\n{_tail(err)}\nstdout:\n{_tail(out)}"
        )

    assert leaked is not None, (
        f"probe produced no {MARKER} line for {module_name}.\n"
        f"stdout:\n{_tail(out)}\nstderr:\n{_tail(err)}"
    )
    assert leaked == [], (
        f"importing {module_name} pulled UI modules into sys.modules: "
        f"{_brief(leaked)}.\n"
        "src/ must not depend on streamlit or app/ - move the shared code into "
        "src/ (e.g. src/engine/, src/data/) instead of importing back into app/."
    )


# The four modules that historically reached back into
# app.pages.decision_execution for default decision values
# (src/orchestrator_baseline.py:17-18 and the three decision modules).
# Called out explicitly so a regression names the culprit even if the
# parametrized walk above is changed.
@pytest.mark.parametrize(
    "module_name",
    [
        "src.decisions.rejected_transaction_option",
        "src.decisions.final_donation_rate",
        "src.decisions.rejected_bid_value",
        "src.orchestrator_baseline",
    ],
)
def test_known_app_importers_are_clean(module_name):
    rc, leaked, out, err = _import_in_fresh_subprocess(module_name)
    assert rc == 0, (
        f"importing {module_name} failed (rc={rc}).\nstderr:\n{_tail(err)}"
    )
    assert leaked == [], (
        f"{module_name} still imports the UI layer: {_brief(leaked)}. It must read its "
        "defaults from the engine/config, not from app.pages.decision_execution."
    )


# --- static check: no UI import statement anywhere in src/, even a lazy one ---

def _ui_import_statements(path):
    """Every `import streamlit`/`import app...` in a file, at ANY nesting depth.

    ast.walk descends into function and class bodies, so this also catches the
    deferred `try: from app.pages.decision_execution import ...` pattern that a
    fresh-import subprocess can never observe.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # relative import: cannot reach app/streamlit
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in ("streamlit", "app"):
                hits.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {name}")
    return hits


@pytest.mark.parametrize(
    "rel_path",
    [
        str(p.relative_to(PROJECT_ROOT))
        for p in sorted(SRC_ROOT.rglob("*.py"))
        if "__pycache__" not in p.parts
    ],
)
def test_src_file_has_no_ui_import_statement(rel_path):
    hits = _ui_import_statements(PROJECT_ROOT / rel_path)
    assert hits == [], (
        "src/ file imports the UI layer (a lazy import inside a function counts - "
        "it breaks the first time the engine runs outside Streamlit):\n  "
        + "\n  ".join(hits)
    )
