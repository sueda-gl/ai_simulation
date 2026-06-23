# Implementation Notes — Disclose Documents (Decision 2)

## 1. `consumedtransferssospeso2periods` (the stochastic σ source)

This variable **cannot be reconstructed from the data files in this repo**, so I take it
**directly from Stata** (the professor's `Stata_File_Decision 2_260326 - Final.dta`).

- It is **NOT** `Number of Sospeso Concumed` + `Number of Transfers Consumed`.
  That sum = 268 / max 11 / 90 non-zero participants; the professor's variable =
  **351 / max 13 / 125 non-zero**.
- The component columns in the professor's `.dta` match our experiment xlsx **exactly per
  participant** (max diff = 0), and **no combination** of the summary count columns
  reproduces it (verified by brute force). It is aggregated from the **raw cycle/
  transaction-level consumption logs over both periods**, which are not present in this repo.
- **Decision: use it directly from Stata.** The σ constant is read from his
  `constranssospeso2periods01` column:
  - `σ_overall = 0.16066` (ddof = 1)
  - per income/allowance group σ (if quintile strategy is used):
    TA12 = 0.1690, TA32 = 0.1910, TA72 = 0.1078, TA128 = 0.1436, TA200 = 0.1563
  We do **not** recompute it from raw data. To regenerate from scratch we would need the
  professor's raw consumption logs / his derivation of `consumedtransferssospeso2periods`.

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
