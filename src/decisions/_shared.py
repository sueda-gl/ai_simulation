# src/decisions/_shared.py
"""
Helpers shared by the modelled decisions (Decisions 1, 2, 3 and 4).

config/decisions.yaml is the ONLY source of every constant. A decision reads its
block strictly through :func:`require`; a key the block lacks raises
:class:`MissingDecisionKey` (a ``KeyError``) naming the decision and the dotted key
path. No helper here carries a fallback value.

The few module-level names the app and the tests import by name (for example
``PRIORITY_SEQUENCES`` and ``SIGMA_OVERALL`` in Decision 4, ``DOC_COEFFICIENTS`` in
Decision 3) are read from the file once, at import, through :func:`file_block` - the
one loader in ``src/config.py`` - so they can never drift from it. A decision never
reads its run-time values from them: those come from ``params``, the patched block.
"""

from typing import Any, Mapping, Optional

from src.config import DEFAULT_DECISIONS_PATH, read_yaml_file


class MissingDecisionKey(KeyError):
    """A key the decision needs is absent from its config/decisions.yaml block."""

    def __str__(self) -> str:   # KeyError quotes its argument; keep the message readable
        return self.args[0] if self.args else ""


def require(block: Any, decision: str, *path: Any, prefix: Optional[str] = None) -> Any:
    """``block[path[0]][path[1]]...`` with no default.

    ``block`` is the decision's params dict, or one of its sub-blocks - then ``prefix``
    is the sub-block's dotted path from the decision root (used only in the message).
    A missing step, or a node that is not a mapping, raises :class:`MissingDecisionKey`
    naming the decision and the full dotted key path. A present key is returned as is,
    whatever its value.
    """
    node = block
    walked = [prefix] if prefix else []
    for key in path:
        walked.append(str(key))
        if not isinstance(node, Mapping) or key not in node:
            raise MissingDecisionKey(
                f"{decision}: key '{'.'.join(walked)}' is missing from its config/decisions.yaml "
                "block (there is no module fallback: add it to the file or to the run's patch)")
        node = node[key]
    return node


def file_block(decision: str) -> dict:
    """The decision's block of config/decisions.yaml AS SHIPPED (a deep copy)."""
    block = read_yaml_file(DEFAULT_DECISIONS_PATH).get(decision)
    if not isinstance(block, dict):
        raise MissingDecisionKey(f"{decision}: no block in {DEFAULT_DECISIONS_PATH}")
    return block


def z_score_trait(value: float, trait_name: str, z_params: Mapping, decision: str) -> float:
    """(value - mean) / sd with the trait's mean/sd taken from ``z_params[trait_name]``
    (the decision's ``z_scoring`` block). A missing trait, mean or sd raises. An sd of 0
    gives 0.0.
    """
    mean = require(z_params, decision, trait_name, 'mean', prefix='z_scoring')
    sd = require(z_params, decision, trait_name, 'sd', prefix='z_scoring')
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def categorical_intercept(params: Mapping, level: int, decision: str) -> float:
    """Per-level intercept for categorical income.

    Reads ``params['categorical_intercepts']`` accepting ``'level_N'``, ``'N'`` or ``N``
    keys, in that order. A missing block or level raises.
    """
    cfg = require(params, decision, 'categorical_intercepts')
    if not isinstance(cfg, Mapping):
        raise MissingDecisionKey(f"{decision}: 'categorical_intercepts' is not a mapping")
    for key in (f'level_{level}', str(level), level):
        if key in cfg:
            return float(cfg[key])
    raise MissingDecisionKey(
        f"{decision}: key 'categorical_intercepts.level_{level}' is missing from its "
        "config/decisions.yaml block (there is no module fallback)")


def normalise_income_mode(mode: Any, default: str = 'categorical') -> str:
    """'continuous' or 'categorical' from a UI / YAML spelling ('Continuous only', ...).

    The string is searched for the mode that is NOT ``default``; any other string gives
    ``default``. Decisions 1, 2 and 3 default to 'categorical' (the string must contain
    'continuous'); Decision 4 defaults to 'continuous' (it must contain 'categorical').
    """
    other = 'continuous' if default == 'categorical' else 'categorical'
    return other if other in str(mode).lower() else default
