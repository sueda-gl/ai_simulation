# src/engine/__init__.py
"""
The single simulation engine shared by every population mode.

* :mod:`src.engine.profile`     - ModeProfile / PROFILES: the genuine per-mode differences
* :mod:`src.engine.core`        - Engine: the one two-pass agent/decision loop
* :mod:`src.engine.vendors`     - vendor attribute generation (setup RNG)
* :mod:`src.engine.sampling`    - Research-mode participant sampling
* :mod:`src.engine.postprocess` - result post-processing (global transaction ids)
* :mod:`src.engine.log`         - prefixed console logging

``src.orchestrator``, ``src.orchestrator_doc_mode`` and ``src.orchestrator_baseline``
are thin subclasses of :class:`src.engine.core.Engine` bound to one profile each.
"""

from src.engine.profile import ModeProfile, PROFILES

__all__ = ["ModeProfile", "PROFILES"]
