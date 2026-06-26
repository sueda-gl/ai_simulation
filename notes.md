# Implementation Notes — Disclose Documents (Decision 2)

## 1. `consumedtransferssospeso2periods` (the stochastic σ source) — RECONSTRUCTED

This variable **CAN be reconstructed bit-for-bit** from the raw per-period experiment files
(reproduces the professor's `Stata_File_Decision 2_260326 - Final.dta` column 280/280). An earlier
note claimed it could not be — that was wrong, because it was computed from the **combined**
`Student Experiment Results - Period 1-2.xlsx`, whose consumption columns are actually **Period 1
only** (`Number of Sospeso Concumed` + `Number of Transfers Consumed` = 268 / max 11 / 90 non-zero,
identical to the standalone Period 1 file). The professor's variable is the **two-period** total:
**351 / max 13 / 125 non-zero**.

**Reconstruction recipe** (`src/build_dd_sigma.py` → `python -m src.build_dd_sigma`):

```
consumed[i] = (Sospeso Consumed + Transfers Consumed) from Period 1   [master row order]
            + (Sospeso Consumed + Transfers Consumed) from Period 2   [aligned by ROW POSITION]
```

- Source columns per the experiment codebook (`decisions.pdf`): "Claim (consume) a Sospeso" →
  `Number of Sospeso Concumed`; "Consume a received transfer" → `Number of Transfers Consumed`.
- Files: `data/Student Experiment Results - Period 1-2.xlsx` (Period-1 consumption + allowance level
  + master row order) and `data/Student Experiment Results - Period 2.xlsx` (Period-2 consumption,
  269 rows — 11 participants inactive in Period 2, contributing 0). **The Period 2 file is required;
  the repo previously shipped only the combined file (Period 1).**

**⚠ Participant-ID merge caveat.** Period 2 is merged onto the master by **ROW POSITION**, NOT by
`Participant ID`. The Period 2 export's `Participant ID` column is offset relative to its data rows,
so joining the two periods by `Participant ID` gives **89/280 per-participant mismatches** (the
total 351 is unchanged — the same 269 Period-2 values are just attributed to *adjacent*
participants, e.g. 736↔738, 741↔742). The professor's Stata merge matched by position; we replicate
the positional merge to reproduce his validated `disclosedoc_*` output. This is likely a latent
bug in the upstream data prep (the per-participant Period-2 attribution is arguably wrong) but it
does not affect the aggregate model or σ — **flag to the professor.**

**Derived σ** (= `sd(consumed / max(consumed))`, ddof=1; matches `config/decisions.yaml` to ~1e-8):
  - `σ_overall = 0.16066`
  - per allowance-level σ (quintile strategy):
    TA12 = 0.1690, TA32 = 0.1910, TA72 = 0.1078, TA128 = 0.1436, TA200 = 0.1563

The σ constants stay **frozen in `config/decisions.yaml`** (consistent with the rest of the repo);
`src/build_dd_sigma.py` re-derives and self-validates them from the raw files, so the provenance is
reproducible and the `.dta` is no longer needed.

## 2. Continuous income mode = a fresh distribution each run, confirmed against frozen Stata data

- In **continuous** mode every run draws a **fresh per-agent income** from the configured
  distribution (lognormal within each allowance-level percentile band, via
  `get_agent_income` in `src/decisions/income_utils.py`). So continuous results vary in
  distribution run-to-run (~36–39%) **by design** — this mirrors how Disclose Income works.
- To **confirm correctness against Stata**, we feed the **frozen income realization**
  (`data/stata_incomes.csv`, identical to the professor's `income` column) through the
  pipeline. With that frozen income the model reproduces the professor's continuous table
  **exactly: 102/280 = 36.43%, every cell**.
- **Categorical** mode needs no income and is deterministic: reproduces **110/280 = 39.29%**.
- Algebraic note: `z_picont = −z_income` exactly (the `max_income` term cancels in
  standardization), so continuous DD reuses the same runtime `income_stats` that Disclose
  Income already computes in Pass 1 of the orchestrator.

## Validation status (against the professor's `Stata_File_Decision 2` .dta)

The Python reproduction matches the professor's Stata output **participant-by-participant**:
`disclosedoc_cont` 280/280, `disclosedoc_categorical` 280/280; all intermediate columns
(z-scores, picont, weighted_dd, dd_deterministic) agree to ~1e-7 (float32 precision).
