* ============================================================
* Step 5: Full Pipeline - Identical to Documentation Section 7
* Using the exact variable names and commands from the spec
* ============================================================

* Load fresh data
import delimited "/Users/suedagul/<sdg/data/stata_verification.csv", clear

* ============================================================
* Section 5: Variable Standardisation
* ============================================================

// 1. Create standardized (z-score) versions of each variable
egen z_agreeable = std(agreeable)
egen z_honesty_humility = std(honesty_humility)
egen z_neuroticism = std(neuroticism)
egen z_extraversion = std(extraversion)
egen z_openness = std(openness)
egen z_twt_sospeso = std(twt_sospeso)

// Religiosity composite (equal weight after standardization)
summarize religious_service
local rs_min = r(min)
local rs_max = r(max)
gen religiousservice_01 = (religious_service - `rs_min') / (`rs_max' - `rs_min')

// Create composite by averaging the two variables
gen religious_composite_raw = (religious_affiliation + religiousservice_01) / 2

// Standardize the composite variable for use in regression
egen z_religious_composite = std(religious_composite_raw)

* ============================================================
* Section 7: Constructing the Prosocial Behavior Anchor
* ============================================================

gen weighted_prosocial = ///
    (0.023776 * z_agreeable) + ///
    (0.016537 * z_openness) + ///
    (0.0295482 * z_honesty_humility) + ///
    (0.0677157 * z_religious_composite)

// Standardize the composite to match the observed z-score scale
egen z_weighted_prosocial = std(weighted_prosocial)

gen anchored_prosocial_behavior = 0.25 * z_twt_sospeso + 0.75 * z_weighted_prosocial

* ============================================================
* Section 7: Direct Effect - Categorical Income
* ============================================================

gen weighted_disclosure_noincome = (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility)

gen weighted_disclosure_categorical = cond(allowance_level > 4, ///
    0.0089094 - 0.0234673 + (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility), ///
    cond(allowance_level > 3, ///
    0.0089094 - 0.0121239 + (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility), ///
    cond(allowance_level > 2, ///
    0.0089094 - 0.0065954 + (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility), ///
    cond(allowance_level > 1, ///
    0.0089094 - 0.0033691 + (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility), ///
    0.0089094 + (0.00674934 * z_extraversion) + (0.0173732 * z_neuroticism) + (0.0295482 * z_honesty_humility)))))

* ============================================================
* Section 7: Composite Z-Scoring
* ============================================================

// Standardize the composite to match the observed z-score scale
egen z_weighted_categorical = std(weighted_disclosure_categorical)
egen z_anchored_prosocial_behavior = std(anchored_prosocial_behavior)

* ============================================================
* Section 7: Income High Indicator
* ============================================================

gen income_high_cat = (allowance_level > 3)

* ============================================================
* Section 7: Final Equation
* ============================================================

gen beta0 = 0.1
gen fs_deterministic_categorical = beta0 + 0.50 * z_weighted_categorical + 0.50 * z_anchored_prosocial_behavior * income_high_cat

* ============================================================
* Section 7: Disclose Decision
* ============================================================

gen disclose_categorical = (fs_deterministic_categorical > 0) if !missing(fs_deterministic_categorical)

* ============================================================
* RESULTS
* ============================================================

display " "
display "============================================"
display "  OVERALL DISCLOSURE RATE (Categorical)"
display "  Spec reference: 165/280 = 58.9%"
display "============================================"
tab disclose_categorical

display " "
display "============================================"
display "  DISCLOSURE BY INCOME LEVEL"
display "  Spec reference:"
display "  Level 1 (12):  73.7%"
display "  Level 2 (32):  59.0%"
display "  Level 3 (72):  63.4%"
display "  Level 4 (128): 57.1%"
display "  Level 5 (200): 40.7%"
display "============================================"
tab allowance_level disclose_categorical, row

display " "
display "============================================"
display "  DI_i DISTRIBUTION"
display "============================================"
summarize fs_deterministic_categorical, detail

display " "
display "============================================"
display "  DI_i BY INCOME LEVEL"
display "============================================"
bysort allowance_level: summarize fs_deterministic_categorical

display " "
display "============================================"
display "  COMPOSITE Z-SCORING STATISTICS"
display "  (SDs should match Python config)"
display "============================================"
display "--- weighted_prosocial ---"
summarize weighted_prosocial
display "--- weighted_disclosure_categorical ---"
summarize weighted_disclosure_categorical
display "--- anchored_prosocial_behavior ---"
summarize anchored_prosocial_behavior

* Export
export delimited using "/Users/suedagul/<sdg/data/stata_step5_results.csv", replace
display " "
display "Results exported to stata_step5_results.csv"
