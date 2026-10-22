# app/pages/results/run_context.py
"""
What the results page knows about the run it is showing.

The page decides its layout - which comparison grid, which sections, which
frame feeds the agent details / export, which decisions were executed - from
``st.session_state._run_metadata`` (written by ``app.simulation.run_full_simulation``
from the RunPlan that was actually executed) and from the result frames in
``st.session_state.simulation_results``.  It never consults the live Page-1 /
Page-2 keys (``population_mode``, ``income_spec_mode``, ``di_income_mode``,
``dd_income_mode``, ``rtd_income_mode``, ``decision_params.selected_decisions``),
which describe the NEXT run the user is configuring, not this one.

Every run-shape question the results layer used to answer from those keys is a
property here; the semantics are the ones the page had when it read those keys:

* ``custom_decisions`` / ``default_decisions`` are the run's own lists (an
  individual run is ``custom == (decision,)`` with no defaults; a complete run
  has every default decision listed; a DD-only run still reports
  ``('disclose_documents',)``);
* ``is_compare_all`` / ``is_compare_both`` come from the effective modes the
  plan ran with (they are what the result keys were built from);
* ``income_type`` normalises the effective income mode case-insensitively, as
  the comparison grids always did;
* ``param(name)`` / ``periods`` / ``platform_markup`` ... are the Page 1
  parameters the run USED (``RunMetadata.simulation_params``), ``rtd_run_element``
  the Decision 4 element its run button named and ``anchor_observed_weight`` the
  donation anchor weight it ran with - never ``st.session_state.sim_params`` or the
  live keys, which a Page 1 / Page 2 edit after the run has moved on.  Metadata
  from before these were recorded (2026-10-22) falls back to the live values, the
  only ones such a session has;
* ``default_setting(decision, name, live_value)`` is the default-decision setting
  the run used (``RunMetadata.default_decision_settings``: the Overview tab's
  probability, parameter selection, priority template, option or value), never
  the live Overview key; metadata that does not record them falls back to
  ``live_value``, the live key the caller read, as above.
"""
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from app.models import ALL_DECISIONS
from app.reports.timestamps import DEFAULT_DURATION_HOURS, DEFAULT_PERIODS
from app.state.saved_configs import _file_value, RunOnScreen


def _as_plain_dict(sim_params: Any) -> Dict[str, Any]:
    """A SimulationParameters object (or any attribute bag) as a plain dict - the
    fallback for a session whose run metadata predates ``simulation_params``."""
    if sim_params is None:
        return {}
    if dataclasses.is_dataclass(sim_params) and not isinstance(sim_params, type):
        return dataclasses.asdict(sim_params)
    return {k: v for k, v in vars(sim_params).items() if not k.startswith("_")}

# Result keys of a "Compare all" run, in the order the page probes them.
COMPARE_ALL_KEYS: Tuple[str, ...] = (
    "copula_categorical", "copula_continuous",
    "research_spec_categorical", "research_spec_continuous",
    "research_baseline_categorical", "research_baseline_continuous",
)

# Individual runs that offer a "Use This Config" button (Decision 4's renders under
# its detailed results, the others in their overview cells).
_SELECTABLE_INDIVIDUAL_RUNS = (
    ("donation_default",),
    ("disclose_income",),
    ("disclose_documents",),
    ("rejected_transaction_defaults",),
)


@dataclass(frozen=True)
class RunContext:
    """Immutable view of ``_run_metadata`` + ``simulation_results`` for one render."""

    results: Dict[str, pd.DataFrame]
    is_comparison: bool
    result_keys: Tuple[str, ...]
    num_results: int
    effective_income_mode: str
    effective_population_mode: str
    custom_decisions: Tuple[str, ...]
    default_decisions: Tuple[str, ...]
    seed: Optional[int]
    n_agents: Optional[int]
    # {result_key: {decision: 'categorical' | 'continuous'}} - what each income-dependent
    # decision actually ran with (absent in metadata from before 2026-10-07)
    decision_income_modes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    rtd_compare_both_fallback: bool = False
    # ISO 8601 moment the run was started (``RunMetadata.started_at``); None when the
    # metadata does not carry it
    started_at: Optional[str] = None
    # The Page 1 parameters the run used: SimulationParameters as a plain dict
    # (``RunMetadata.simulation_params``); {} when nothing is known
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    # The Decision 4 element the run button named ('ttp' ... 'flexibility',
    # 'aggregation', or None for a whole-decision / complete run)
    rtd_run_element: Optional[str] = None
    # The donation anchor weight the run used (observed share; predicted = 1 - it)
    anchor_observed_weight: float = field(default_factory=lambda: _file_value('donation_default', 'anchor_weights', 'observed'))
    # The default-decision settings the run used ({decision: {'type': ..., ...}},
    # ``RunMetadata.default_decision_settings``); None when the metadata does not
    # record them (``default_setting`` then returns the caller's live value)
    default_decision_settings: Optional[Dict[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------ factory
    @classmethod
    def from_session(cls) -> "RunContext":
        """Build the context from the session's ``_run_metadata`` and result frames.

        ``_run_metadata`` is written together with ``simulation_results`` by every
        run; should it ever be absent (a foreign session), the shape is derived
        from the result keys and the run's own decision lists instead - never
        from the live Page-1 / Page-2 mode keys.
        """
        results = st.session_state.get("simulation_results") or {}
        if not isinstance(results, dict):
            results = {}
        metadata = st.session_state.get("_run_metadata") or {}

        result_keys = tuple(metadata.get("result_keys", list(results.keys())))

        if "effective_population_mode" in metadata:
            population_mode = metadata["effective_population_mode"]
        else:
            population_mode = "Compare all" if any(k in COMPARE_ALL_KEYS for k in result_keys) else ""

        if "effective_income_mode" in metadata:
            income_mode = metadata["effective_income_mode"]
        else:
            has_cat = any(str(k).endswith("categorical") for k in result_keys)
            has_cont = any(str(k).endswith("continuous") for k in result_keys)
            if has_cat and has_cont:
                income_mode = "Compare both"
            elif has_cont:
                income_mode = "continuous only"
            else:
                income_mode = "categorical only"

        custom = metadata.get("custom_decisions", st.session_state.get("custom_decisions", []) or [])
        default = metadata.get("default_decisions", st.session_state.get("default_decisions", []) or [])

        # The run's Page 1 parameters, Decision 4 element and anchor weight; metadata
        # written before they were recorded falls back to the live values
        if "simulation_params" in metadata:
            simulation_params = dict(metadata["simulation_params"] or {})
        else:
            simulation_params = _as_plain_dict(st.session_state.get("sim_params"))
        if "rtd_run_element" in metadata:
            rtd_run_element = metadata["rtd_run_element"]
        else:
            rtd_run_element = st.session_state.get("rtd_run_element")
        anchor = metadata.get("anchor_observed_weight")
        if anchor is None:
            anchor = st.session_state.get("anchor_observed_weight",
                                          _file_value('donation_default', 'anchor_weights', 'observed'))

        return cls(
            results=results,
            is_comparison=bool(metadata.get("is_comparison", len(result_keys) > 1)),
            result_keys=result_keys,
            num_results=int(metadata.get("num_results", len(result_keys))),
            effective_income_mode=str(income_mode),
            effective_population_mode=str(population_mode),
            custom_decisions=tuple(custom),
            default_decisions=tuple(default),
            seed=metadata.get("seed"),
            n_agents=metadata.get("n_agents", st.session_state.get("n_agents")),
            decision_income_modes={str(k): dict(v) for k, v in
                                   (metadata.get("decision_income_modes") or {}).items()},
            rtd_compare_both_fallback=bool(metadata.get("rtd_compare_both_fallback", False)),
            started_at=metadata.get("started_at") or None,
            simulation_params=simulation_params,
            rtd_run_element=rtd_run_element,
            anchor_observed_weight=float(anchor),
            default_decision_settings=metadata.get("default_decision_settings"),
        )

    # --------------------------------------------------- the run's Page 1 values
    def param(self, name: str, default: Any = None) -> Any:
        """The Page 1 parameter ``name`` (a SimulationParameters attribute) the run
        used; ``default`` when the run did not record it."""
        return self.simulation_params.get(name, default)

    @property
    def periods(self) -> int:
        return int(self.param("periods", DEFAULT_PERIODS))

    @property
    def duration_hours(self) -> float:
        return float(self.param("duration_hours", DEFAULT_DURATION_HOURS))

    @property
    def market_price(self) -> float:
        return float(self.param("market_price", 100.0))

    @property
    def platform_markup(self) -> float:
        return float(self.param("platform_markup", 0.1))

    @property
    def price_range(self) -> float:
        return float(self.param("price_range", 0.25))

    @property
    def vendor_price_min(self) -> float:
        return float(self.param("vendor_price_min", 50.0))

    @property
    def vendor_price_max(self) -> float:
        return float(self.param("vendor_price_max", 150.0))

    @property
    def anchor_predicted_weight(self) -> float:
        return 1 - self.anchor_observed_weight

    # ------------------------------------------ the run's default-decision settings
    def default_setting(self, decision: str, name: str, live_value: Any = None) -> Any:
        """``name`` ('probability_y', 'selected_params', 'priority_template',
        'selected_option' or 'value') of the default-decision settings the run gave
        ``decision`` - not the live Overview key, which a change after the run has
        moved.  Metadata that does not record them falls back to ``live_value``:
        the live key as the caller read it, what the page showed before."""
        recorded = (self.default_decision_settings or {}).get(decision) or {}
        return recorded[name] if name in recorded else live_value

    # ------------------------------------------------- what "Use This Config" pins
    @property
    def run_on_screen(self) -> RunOnScreen:
        """The run's seed, agent count, population and income mode, handed to the
        pin store (``app.state.saved_configs``), which takes them as arguments
        instead of reaching back into this page."""
        return RunOnScreen(seed=self.seed, n_agents=self.n_agents,
                           population_mode=self.effective_population_mode,
                           income_mode=self.effective_income_mode)

    # ----------------------------------------------------------- base time
    @property
    def base_time(self) -> datetime:
        """The base time of every export's timestamp column: the moment the run was
        started.  Re-exporting the same run later therefore gives the same timestamp
        strings.  Metadata without ``started_at`` (a foreign session) falls back to
        the former anchor, today's midnight."""
        if self.started_at:
            return datetime.fromisoformat(self.started_at)
        from app.reports.timestamps import midnight_of
        return midnight_of(datetime.now())

    # ---------------------------------------------------------- run shape
    @property
    def is_compare_all(self) -> bool:
        """The run compared the three population modes (3xN grid)."""
        return self.effective_population_mode == "Compare all"

    @property
    def is_compare_both(self) -> bool:
        """The run compared the two income specifications (categorical + continuous)."""
        return self.effective_income_mode == "Compare both"

    @property
    def is_comparison_mode(self) -> bool:
        return self.is_compare_all or self.is_compare_both

    @property
    def income_type(self) -> str:
        """'continuous' / 'categorical' - the single-income key suffix (case-insensitive)."""
        return "continuous" if "continuous" in str(self.effective_income_mode).lower() else "categorical"

    @property
    def has_compare_all_results(self) -> bool:
        return any(k in self.results for k in COMPARE_ALL_KEYS)

    # ---------------------------------------------------------- decisions
    @property
    def has_combined_simulation(self) -> bool:
        """A complete run: at least one decision ran with default values."""
        return len(self.default_decisions) > 0

    def is_individual_run(self, decision: str) -> bool:
        """The run executed exactly this one decision with custom parameters."""
        return self.custom_decisions == (decision,) and not self.default_decisions

    @property
    def is_individual_decision_run(self) -> bool:
        """Exactly one custom decision and no defaults, whichever decision it is."""
        return len(self.custom_decisions) == 1 and not self.default_decisions

    @property
    def should_enable_selection(self) -> bool:
        """An individual donation / DI / DD / Decision 4 run offers 'Use This Config'."""
        return self.custom_decisions in _SELECTABLE_INDIVIDUAL_RUNS and not self.default_decisions

    @property
    def executed_decisions(self) -> Tuple[str, ...]:
        """Every decision the run executed, in ALL_DECISIONS (chronological) order."""
        executed = set(self.custom_decisions) | set(self.default_decisions)
        return tuple(d for d in ALL_DECISIONS if d in executed)

    @property
    def num_decisions(self) -> int:
        """The '- Decisions: N' line of the parameter summary: the decisions this run
        executed, not the live selection."""
        return len(self.custom_decisions) + len(self.default_decisions)

    def decision_income_label(self, decision: str) -> Optional[str]:
        """The income mode ``decision`` actually ran with, in the tabs' wording
        ('Categorical only' / 'Continuous only'; 'Compare both' when its result keys
        ran different modes), or None when the run did not record it."""
        modes = {m[decision] for m in self.decision_income_modes.values() if decision in m}
        if not modes:
            return None
        if len(modes) > 1:
            return "Compare both"
        return f"{next(iter(modes)).capitalize()} only"

    # ------------------------------------------------------------- frames
    @property
    def first_df(self) -> pd.DataFrame:
        return next(iter(self.results.values()), pd.DataFrame())

    @property
    def primary_df(self) -> pd.DataFrame:
        """The frame behind 'Individual Agent Details' and the export section.

        Compare all + Compare both: the first present of the six compare-all keys
        (categorical population modes first); Compare all with one income
        specification: the first present ``<population>_<income_type>`` key;
        Compare both alone: categorical before continuous; otherwise the first
        result.  Falls back to the first result when the probed keys are absent.
        """
        if self.is_compare_all:
            if self.is_compare_both:
                keys = ["copula_categorical", "research_spec_categorical", "research_baseline_categorical",
                        "copula_continuous", "research_spec_continuous", "research_baseline_continuous"]
            else:
                income_type = self.income_type
                keys = [f"copula_{income_type}", f"research_spec_{income_type}",
                        f"research_baseline_{income_type}"]
            df = next((self.results[k] for k in keys if k in self.results), pd.DataFrame())
        elif self.is_compare_both:
            df = next((self.results[k] for k in ["categorical", "continuous"] if k in self.results), pd.DataFrame())
        else:
            df = self.first_df

        if df.empty and self.results:
            df = self.first_df
        return df


__all__ = ["COMPARE_ALL_KEYS", "RunContext"]
