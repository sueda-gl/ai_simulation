# tests/test_session_key_registry.py
"""The session-key registry is the truth about ``st.session_state``.

This test walks every ``app/`` module with the ``ast`` module, collects every
form in which a session-state key can be named, and holds the result against
``app.state.registry.REGISTRY``:

* every key the code touches has a registry entry (no undocumented keys);
* every registry entry is touched by the code (no dead entries);
* families written with an f-string (``f"{decision_name}_default_value"``) are
  matched by their pattern entry;
* every key marked as an engine input is one ``app/seam`` actually names.

The scan is deliberately syntactic - it never imports the app, so it stays
green without Streamlit and cannot be fooled by an import-time side effect.
"""

import ast
import collections
import os
import warnings

import pytest

from app.state.registry import (
    CATEGORIES,
    DYNAMIC_KEY_VARIABLES,
    PREFIX_DELETE_PREFIXES,
    REGISTRY,
    lookup,
    matching_families,
    pattern_regex,
    specificity,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")

# ``st.session_state.<attr>`` where <attr> is one of these is a Mapping method,
# not a key.
MAPPING_METHODS = {
    "get", "pop", "update", "setdefault", "keys", "items", "values", "clear",
    "to_dict",
}

# ``key=`` is a session-state key only on a Streamlit call; ``sorted(key=...)``
# and ``list.sort(key=...)`` take a callable of the same keyword name.
WIDGET_CALL_PREFIX = "st."


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #

class KeyUse:
    """Every place one key (or one key family) is named."""

    __slots__ = ("key", "modes", "forms", "sites")

    def __init__(self, key):
        self.key = key
        self.modes = set()      # 'r' | 'w' | 'del' | 'widget'
        self.forms = set()      # 'attribute' | 'subscript' | 'ss.get' | 'key=' ...
        self.sites = []         # (relpath, lineno)

    def add(self, mode, form, rel, line):
        self.modes.add(mode)
        self.forms.add(form)
        self.sites.append((rel, line))

    def where(self, limit=4):
        seen, out = set(), []
        for rel, line in sorted(self.sites):
            if rel in seen:
                continue
            seen.add(rel)
            out.append(f"{rel}:{line}")
            if len(out) == limit:
                break
        return ", ".join(out)


class ScanResult:
    def __init__(self):
        self.keys = {}                       # key/pattern -> KeyUse
        self.dynamic = collections.Counter()  # variable name -> count
        self.prefix_deletes = set()          # 'dd_' ...
        self.wipe_all = []                   # (relpath, lineno)

    def use(self, key):
        return self.keys.setdefault(key, KeyUse(key))


def _is_session_state(node):
    """``st.session_state``"""
    return (isinstance(node, ast.Attribute)
            and node.attr == "session_state"
            and isinstance(node.value, ast.Name)
            and node.value.id == "st")


def _literal_specs(node):
    """Every key string ``node`` can evaluate to, or [] when it is dynamic.

    A constant yields itself; an f-string yields its pattern with each
    placeholder rendered as ``{expr}``; a conditional yields both branches.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("{" + ast.unparse(part.value).strip() + "}")
        return ["".join(out)]
    if isinstance(node, ast.IfExp):
        return _literal_specs(node.body) + _literal_specs(node.orelse)
    return []


class _ScopeIndex:
    """Names in one scope (a module, or one function) that stand for a key.

    Two bindings count: ``k = "literal"`` / ``k = f"..."`` (every such binding
    above the use, because the name is often assigned in several branches of an
    if/elif before the lookup) and ``for k in ("a", f"b_{x}")`` (the loop
    variable stands for every element).  Dict literals are indexed too, for the
    seeding loops in ``models.initialize_session_state``.

    One index is built per function and per module; a use resolves against its
    own function first and the module last, so a loop variable in one helper
    cannot pretend to name the keys of a same-named variable in another.
    """

    def __init__(self, tree):
        self.assigned = collections.defaultdict(list)   # name -> [(line, spec)]
        self.looped = collections.defaultdict(list)     # name -> [spec]
        self.dicts = {}                                 # name -> ast.Dict
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                for spec in _literal_specs(node.value):
                    self.assigned[name].append((node.lineno, spec))
                if isinstance(node.value, ast.Dict):
                    self.dicts[name] = node.value
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                    for elt in node.value.elts:
                        for spec in _literal_specs(elt):
                            self.looped[name].append(spec)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                if isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
                    for elt in node.iter.elts:
                        for spec in _literal_specs(elt):
                            self.looped[node.target.id].append(spec)
                elif isinstance(node.iter, ast.Name):
                    self.looped[node.target.id] += self.looped.get(node.iter.id, [])
        for name in self.assigned:
            self.assigned[name].sort()

    def resolve(self, node, line):
        """Key strings ``node`` can name, plus the variable name when dynamic."""
        specs = _literal_specs(node)
        if specs:
            return specs, None
        if isinstance(node, ast.Name):
            above = [s for ln, s in self.assigned.get(node.id, []) if ln <= line]
            if above:
                # EVERY binding above the use, not just the nearest: the name is
                # often assigned in several branches of an if/elif before the
                # lookup, and a registry that missed one of those keys would be
                # wrong in the direction that matters.
                out = list(dict.fromkeys(above))
                out += self.looped.get(node.id, [])
                return out, None
            looped = self.looped.get(node.id)
            if looped:
                return list(looped), None
            any_binding = self.assigned.get(node.id)
            if any_binding:
                return [any_binding[0][1]], None
            return [], node.id
        return [], ast.unparse(node).strip()


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel, index, result):
        self.rel, self.result = rel, result
        self._scopes = [index]          # innermost last
        self._stored, self._deleted = set(), set()

    def _resolve(self, node, line):
        """Ask each enclosing scope, innermost first."""
        dynamic = ast.unparse(node).strip()
        for index in reversed(self._scopes):
            specs, dynamic = index.resolve(node, line)
            if specs:
                return specs, None
        return [], dynamic

    def _enter_function(self, node):
        self._scopes.append(_ScopeIndex(node))
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    # -- assignment / deletion bookkeeping ---------------------------------- #
    def _mark(self, targets, bucket):
        for target in targets:
            for node in ast.walk(target):
                bucket.add(id(node))

    def visit_Assign(self, node):
        self._mark(node.targets, self._stored)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._mark([node.target], self._stored)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._mark([node.target], self._stored)
        self.generic_visit(node)

    def visit_Delete(self, node):
        self._mark(node.targets, self._deleted)
        self.generic_visit(node)

    def _mode(self, node):
        if id(node) in self._deleted:
            return "del"
        return "w" if id(node) in self._stored else "r"

    # -- emit --------------------------------------------------------------- #
    def _emit(self, key_node, form, mode, line):
        specs, dynamic = self._resolve(key_node, line)
        if dynamic is not None:
            self.result.dynamic[dynamic] += 1
            return
        for spec in specs:
            self.result.use(spec).add(mode, form, self.rel, line)

    # -- the access forms ---------------------------------------------------- #
    def visit_Attribute(self, node):
        # st.session_state.<key>
        if _is_session_state(node.value) and node.attr not in MAPPING_METHODS:
            self.result.use(node.attr).add(
                self._mode(node), "attribute", self.rel, node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # st.session_state[<key>]
        if _is_session_state(node.value):
            self._emit(node.slice, "subscript", self._mode(node), node.lineno)
        self.generic_visit(node)

    def visit_Compare(self, node):
        # <key> in st.session_state
        for op, other in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and _is_session_state(other):
                self._emit(node.left, "contains", "r", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        # st.session_state.get / .pop / .setdefault
        if isinstance(func, ast.Attribute) and _is_session_state(func.value) \
                and func.attr in ("get", "pop", "setdefault") and node.args:
            mode = {"get": "r", "pop": "del", "setdefault": "w"}[func.attr]
            self._emit(node.args[0], "ss." + func.attr, mode, node.lineno)
        # getattr / hasattr / setattr / delattr(st.session_state, <key>)
        if isinstance(func, ast.Name) \
                and func.id in ("getattr", "hasattr", "setattr", "delattr") \
                and len(node.args) >= 2 and _is_session_state(node.args[0]):
            mode = {"getattr": "r", "hasattr": "r",
                    "setattr": "w", "delattr": "del"}[func.id]
            self._emit(node.args[1], func.id, mode, node.lineno)
        # widget key=
        name = ast.unparse(func).strip()
        if name.startswith(WIDGET_CALL_PREFIX):
            for kw in node.keywords:
                if kw.arg == "key":
                    self._emit(kw.value, "key=", "widget", node.lineno)
        self.generic_visit(node)

    def visit_For(self, node):
        # for k, v in <dict literal name>.items():  ... st.session_state[k] = v
        it = node.iter
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute)
                and it.func.attr == "items"
                and isinstance(it.func.value, ast.Name)
                and isinstance(node.target, ast.Tuple) and node.target.elts
                and isinstance(node.target.elts[0], ast.Name)):
            var = node.target.elts[0].id
            source = None
            for index in reversed(self._scopes):
                source = index.dicts.get(it.func.value.id)
                if source is not None:
                    break
            seeds_state = any(
                _is_session_state(sub.value)
                and isinstance(sub.slice, ast.Name) and sub.slice.id == var
                for sub in ast.walk(node) if isinstance(sub, ast.Subscript))
            if source is not None and seeds_state:
                for entry in source.keys:
                    if isinstance(entry, ast.Constant) and isinstance(entry.value, str):
                        self.result.use(entry.value).add(
                            "w", "dict-seed", self.rel, entry.lineno)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self._bulk(node)
        self.generic_visit(node)

    def _bulk(self, node):
        """``[k for k in st.session_state.keys() if k.startswith('dd_')]``"""
        for gen in node.generators:
            it = gen.iter
            if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute)
                    and it.func.attr == "keys" and _is_session_state(it.func.value)):
                continue
            if not gen.ifs:
                self.result.wipe_all.append((self.rel, node.lineno))
                continue
            for cond in gen.ifs:
                for call in ast.walk(cond):
                    if isinstance(call, ast.Call) \
                            and isinstance(call.func, ast.Attribute) \
                            and call.func.attr == "startswith" and call.args:
                        for spec in _literal_specs(call.args[0]):
                            self.result.prefix_deletes.add(spec)


def _parse(path):
    """Compile one module to an AST.

    ``ast.parse`` re-raises the module's own SyntaxWarnings (there is a docstring
    under app/pages/decision_tabs/ with an invalid ``\\_`` escape).  Those belong
    to the app, not to this test, so reading a file here does not repeat them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        with open(path, encoding="utf-8") as handle:
            return ast.parse(handle.read(), filename=path)


def scan_app(app_dir=APP, root=ROOT):
    """Every session-state key named anywhere under ``app/``."""
    result = ScanResult()
    for dirpath, dirnames, filenames in os.walk(app_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, root)
            tree = _parse(path)
            _Visitor(rel, _ScopeIndex(tree), result).visit(tree)
    return result


@pytest.fixture(scope="module")
def scan():
    return scan_app()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_every_key_in_the_code_has_a_registry_entry(scan):
    """No key may be introduced without describing it in the registry."""
    missing = {}
    for key, use in scan.keys.items():
        if lookup(key) is None:
            missing[key] = use.where()
    assert not missing, (
        "session-state keys with no app/state/registry.py entry:\n"
        + "\n".join(f"  {k!r}  ({w})" for k, w in sorted(missing.items())))


def test_every_registry_entry_is_used_by_the_code(scan):
    """No dead entries: a key that left the code must leave the registry."""
    used = set(scan.keys)
    matched = set()
    for key in used:
        spec = lookup(key)
        if spec is not None:
            matched.add(spec.name)
    dead = [spec.name for spec in REGISTRY if spec.name not in matched]
    assert not dead, (
        "registry entries no key in app/ matches (delete them):\n"
        + "\n".join(f"  {name!r}" for name in sorted(dead)))


def test_registry_is_internally_consistent():
    names = [spec.name for spec in REGISTRY]
    duplicates = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not duplicates, f"duplicate registry names: {duplicates}"

    for spec in REGISTRY:
        assert spec.category in CATEGORIES, (
            f"{spec.name!r}: unknown category {spec.category!r}")
        assert spec.initialised_by, f"{spec.name!r}: no initialiser recorded"
        assert isinstance(spec.engine_input, bool), spec.name
        assert ("{" in spec.name) == spec.is_family, (
            f"{spec.name!r}: is_family disagrees with the name")
        if spec.is_family:
            pattern_regex(spec.name)  # must compile


def test_lookup_is_unambiguous(scan):
    """No key may be covered by two equally specific families.

    Exact names win over patterns and a more specific pattern wins over a
    looser one, so overlap is fine - a *tie* is not, because then which entry
    describes the key depends on registry order.
    """
    ties = {}
    for key in scan.keys:
        hits = matching_families(key)
        if len(hits) < 2:
            continue
        top = specificity(hits[0].name)
        tied = [spec.name for spec in hits if specificity(spec.name) == top]
        if len(tied) > 1:
            ties[key] = tied
    assert not ties, (
        "keys matched by two equally specific family patterns:\n"
        + "\n".join(f"  {k!r} <- {v}" for k, v in sorted(ties.items())))


def test_widget_keys_are_declared_as_widgets(scan):
    """Anything passed as ``key=`` to a Streamlit call is a widget key."""
    wrong = []
    for key, use in scan.keys.items():
        if "widget" not in use.modes:
            continue
        spec = lookup(key)
        if spec is not None and not spec.is_widget:
            wrong.append(f"{key!r} -> entry {spec.name!r} ({use.where(2)})")
    assert not wrong, (
        "keys bound to a Streamlit widget but not marked is_widget:\n  "
        + "\n  ".join(sorted(wrong)))


def _modes_by_spec(scan):
    """Aggregate the scanned access modes per registry entry."""
    modes = collections.defaultdict(set)
    for key, use in scan.keys.items():
        spec = lookup(key)
        if spec is not None:
            modes[spec.name] |= use.modes
    return modes


def test_declared_widget_keys_really_are_widget_keys(scan):
    """And the other way round, so the flag cannot rot."""
    modes = _modes_by_spec(scan)
    wrong = [spec.name for spec in REGISTRY
             if spec.is_widget and spec.name in modes
             and "widget" not in modes[spec.name]]
    assert not wrong, (
        "entries marked is_widget that no Streamlit `key=` binds:\n  "
        + "\n  ".join(sorted(wrong)))


def test_write_only_keys_are_the_documented_ones(scan):
    """Keys nothing in app/ reads back - the doc lists them as candidates."""
    modes = _modes_by_spec(scan)
    observed = {name for name, seen in modes.items() if seen and seen <= {"w", "del"}}
    declared = {spec.name for spec in REGISTRY if spec.write_only}
    assert observed == declared, (
        "the write-but-never-read set moved.\n"
        f"  newly write-only: {sorted(observed - declared)}\n"
        f"  no longer write-only: {sorted(declared - observed)}\n"
        "Update REGISTRY (write_only=) and docs/migration/session-state.md.")


def _seam_key_vocabulary():
    """Every key string named anywhere under ``app/seam/``.

    The plan builder reads a :class:`SessionSnapshot`, not ``st.session_state``,
    and some names are built by a local helper (``key("intercept")``), so the
    honest check is that the seam mentions the key at all.
    """
    seam = os.path.join(APP, "seam")
    vocabulary = set()
    for dirpath, dirnames, filenames in os.walk(seam):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            tree = _parse(os.path.join(dirpath, filename))
            for node in ast.walk(tree):
                vocabulary.update(_literal_specs(node))
    return vocabulary


def test_engine_input_keys_are_named_by_the_seam():
    """`engine_input=True` means the plan builder reads it from the snapshot."""
    vocabulary = _seam_key_vocabulary()
    resolved = set()
    patterns = []
    for word in vocabulary:
        spec = lookup(word)
        if spec is not None:
            resolved.add(spec.name)
        if "{" in word:
            # the seam builds the name itself (f"{prefix}_scale_factor"); the
            # pattern covers whichever registry entries it can produce
            patterns.append(pattern_regex(word))
    missing = sorted(
        spec.name for spec in REGISTRY
        if spec.engine_input and spec.name not in resolved
        and not any(p.fullmatch(spec.name) for p in patterns))
    assert not missing, (
        "entries marked engine_input=True that app/seam never names:\n  "
        + "\n  ".join(missing))


def test_dynamic_key_variables_are_the_known_helpers(scan):
    """Keys named through a variable stay a short, reviewed list."""
    unknown = {name: count for name, count in scan.dynamic.items()
               if name not in DYNAMIC_KEY_VARIABLES}
    assert not unknown, (
        "session-state keys named through an unreviewed variable "
        f"(document it in DYNAMIC_KEY_VARIABLES once you know what it holds): "
        f"{unknown}")


def test_prefix_deletes_are_registered(scan):
    assert scan.prefix_deletes == set(PREFIX_DELETE_PREFIXES), (
        f"prefix bulk-deletes moved: code={sorted(scan.prefix_deletes)} "
        f"registry={sorted(PREFIX_DELETE_PREFIXES)}")


def test_scan_finds_the_app(scan):
    """Guard against a scanner that silently matches nothing."""
    assert len(scan.keys) > 150
    assert "sim_params" in scan.keys
    assert scan.wipe_all, "the Clear Results wipe-all disappeared from the scan"
