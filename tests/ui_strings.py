"""
On-screen wording the tests pin, in one place.

These are copies of what the app shows, not the app's own source of truth: when the
app's wording changes on purpose, change it here and every test that pins it follows.
"""

# --------------------------------------------------------------------------- Decision 4
# Element subtitles: drawn under "4. Rejected Transaction Defaults" on the results page
# of an element-only run (the element's Decision 4 sub-tab number and name).
D4_SUBTITLE_TTP = "4.1 Options List Length (Tendency to Plan)"
D4_SUBTITLE_LOYALTY = "4.2 Loyalty Ranking"
D4_SUBTITLE_WTP = "4.3 Willingness-to-Pay Ranking"
D4_SUBTITLE_RISK_TAKING = "4.4 Risk-Taking Ranking"
D4_SUBTITLE_FLEXIBILITY = "4.5 Flexibility Ranking"
D4_SUBTITLE_AGGREGATION = "4.6 Integrated Default List (Rank Aggregation)"

# Group titles of a "Compare both" income layout.
CATEGORICAL_TREATMENT = "Categorical Income Treatment"
CONTINUOUS_TREATMENT = "Continuous Income Treatment"

# Download-button labels.
DOWNLOAD_D4_ALL_ELEMENTS = "📊 Download Decision 4 Excel (all elements)"
DOWNLOAD_D4_AGENT_LEVEL = "📊 Download Decision 4 Agent-Level Excel"
DOWNLOAD_OPTIONS_LIST_LENGTH = "📊 Download Options List Length Excel"
DOWNLOAD_LOYALTY = "📊 Download Loyalty Ranking Excel"
DOWNLOAD_WTP = "📊 Download Willingness-to-Pay Ranking Excel"
DOWNLOAD_RISK_TAKING = "📊 Download Risk-Taking Ranking Excel"
DOWNLOAD_FLEXIBILITY = "📊 Download Flexibility Ranking Excel"
DOWNLOAD_INTEGRATED = "📊 Download Integrated Default List Excel"

# The "Use This Config" button once its configuration is selected (an Unselect button).
SELECTED_LABEL = "✅ Selected"

# ------------------------------------------------------------------------- Monte Carlo
# The caption a Monte-Carlo study shows when it repeats a pinned configuration.
SAVED_CONFIG_CAPTION_PREFIX = "🔑 Using saved config"
MC_PINNED_CAPTION_280_BASELINE = ("🔑 Using saved config: agents 280, population Research Baseline; "
                                  "seeds vary per run")
MC_PINNED_CAPTION_1500_SPEC = ("🔑 Using saved config: agents 1,500, population Research "
                               "Specification; seeds vary per run")

# ------------------------------------------------------------------------- navigation
NEXT_TO_PAGE2 = "Next: Decision Parameters"            # Page 1 -> Page 2
BACK_TO_PAGE1 = "Back to Common Parameters"            # Page 2 / results -> Page 1
BACK_TO_PAGE2 = "← Back to Decision Parameters"        # results -> Page 2
VIEW_RESULTS = "View Results"                          # Page 2 -> results (once results exist)

# ------------------------------------------------------------------------- Page 1 options
EXECUTION_LIVE = "Live Simulation"                     # Execution Mode radio
ANALYSIS_MONTE_CARLO = "Monte-Carlo Study"             # Analysis Mode radio

# ------------------------------------------------------------------- decision-tab options
# Income-mode radio options: Disclose Income, Disclose Documents and Decision 4 use the
# capitalised labels, the Donation Default tab its own lower-case ones.
COMPARE_BOTH = "Compare both"                          # every income-mode radio
CONTINUOUS_ONLY = "Continuous only"                    # DI / DD / Decision 4
CATEGORICAL_ONLY = "Categorical only"                  # DI / DD / Decision 4 (their default)

# Results page: the button that pins a configuration.
USE_THIS_CONFIG = "🎯 Use This Config"
