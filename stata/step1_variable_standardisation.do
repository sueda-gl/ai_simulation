* ============================================================
* Step 1: Variable Standardisation
* From Spec Section 5 - using the exact commands from documentation
* ============================================================

* Load fresh data
import delimited "/Users/suedagul/<sdg/data/stata_verification.csv", clear

display "============================================"
display "  STEP 1: VARIABLE STANDARDISATION"
display "  (Section 5 of Model Specification)"
display "============================================"

* ============================================================
* 1a. Raw variable summary statistics (BEFORE standardisation)
* ============================================================

display " "
display "--- RAW VARIABLE STATISTICS (before z-scoring) ---"
summarize agreeable honesty_humility neuroticism extraversion openness twt_sospeso

* ============================================================
* 1b. Z-score each variable (per spec Section 5)
*     Spec command: egen z_variable = std(variable)
* ============================================================

// 1. Create standardized (z-score) versions of each variable
egen z_agreeable = std(agreeable)
egen z_honesty_humility = std(honesty_humility)
egen z_neuroticism = std(neuroticism)
egen z_extraversion = std(extraversion)
egen z_openness = std(openness)
egen z_twt_sospeso = std(twt_sospeso)

display " "
display "--- Z-SCORED VARIABLE STATISTICS (should be mean~0, sd~1) ---"
summarize z_agreeable z_honesty_humility z_neuroticism z_extraversion z_openness z_twt_sospeso

* ============================================================
* 1c. Religiosity Composite (per spec Section 5)
*     Step 1: Scale ReligiousService to 0-1 range
*     Step 2: Average with ReligiousAffiliation (equal weights)
*     Step 3: Z-score the composite
* ============================================================

// Religiosity composite (equal weight after standardization)
summarize religious_service
local rs_min = r(min)
local rs_max = r(max)
gen religiousservice_01 = (religious_service - `rs_min') / (`rs_max' - `rs_min')

// Create composite by averaging the two variables
gen religious_composite_raw = (religious_affiliation + religiousservice_01) / 2

// Standardize the composite variable for use in regression
egen z_religious_composite = std(religious_composite_raw)

display " "
display "--- RELIGIOSITY COMPOSITE ---"
display "  ReligiousService min = `rs_min'"
display "  ReligiousService max = `rs_max'"
summarize religious_affiliation religiousservice_01 religious_composite_raw z_religious_composite

* ============================================================
* 1d. Report: First 10 participants' z-scores for comparison
* ============================================================

display " "
display "--- FIRST 10 PARTICIPANTS Z-SCORES ---"
list participant_id z_agreeable z_honesty_humility z_neuroticism z_extraversion z_openness z_religious_composite z_twt_sospeso in 1/10

* ============================================================
* 1e. Report: Key statistics for comparison with Python
* ============================================================

display " "
display "============================================"
display "  REPORT: Mean and SD of raw variables"
display "  (these should match Python config z_scoring values)"
display "============================================"

foreach var in agreeable honesty_humility neuroticism extraversion openness twt_sospeso religious_composite_raw {
    quietly summarize `var'
    display "  `var':"
    display "    mean = " r(mean)
    display "    sd   = " r(sd)
    display " "
}

* Export z-scored data for Python comparison
export delimited using "/Users/suedagul/<sdg/data/stata_step1_zscores.csv", replace
display "Exported z-scored data to stata_step1_zscores.csv"
