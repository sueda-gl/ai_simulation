"""
Decision 4 (Rejected Transaction Defaults) configuration tab - UI text / structure
tests for the professor's 2026-09 review round.

The tab must:
  - bin Flexibility with the plain `Flexibility15_i` formula over AnchoredFlexibility
    (no "z" markers anywhere, no standardization tooltip);
  - map segment s to the LAST s options of the mechanism's priority sequence
    (segment 5 = "Highest 20%" gets the full sequence, segment 1 only Option 5),
    in a table whose column titles render in black (HTML table, not st.dataframe);
  - print the five option explanations as regular black markdown, not gray captions;
  - carry no "Advanced: stochastic anchor" section and no aggregation enable checkbox;
  - phrase the aggregation sub-tab with the professor's wording (per agent, the two
    rules, the plain-language Kendall-tau paragraph, Phase 1 / Phase 2) and offer the
    "Run Integrated Default List Only" button (key rtd_run_aggregation_btn).

The AppTest renders the Decision 4 tab exactly as page 2 renders it (the pattern of
tests/test_rtd_batch4_ui.py); no simulation is started, so the run is cheap.
"""
import pytest

from app.pages.decision_tabs import rejected_transaction as rtd_tab


def _tab_script():
    """Page 2's Decision 4 configuration tab, rendered on its own."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()


@pytest.fixture(scope="module")
def tab():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_tab_script)
    at.run(timeout=300)
    assert not at.exception
    return at


def _md(at):
    return "\n".join(str(m.value) for m in at.markdown)


def _latex(at):
    return "\n".join(str(l.value) for l in at.latex)


def _captions(at):
    return "\n".join(str(c.value) for c in at.caption)


def _helps(at):
    return "\n".join(str(getattr(e, 'help', '') or '') for e in list(at.number_input)
                     + list(at.checkbox) + list(at.slider) + list(at.radio))


# ---------------------------------------------------------------------------
# 1. Flexibility formulas: no z markers, no standardization tooltip
# ---------------------------------------------------------------------------
def test_flexibility_binning_formula_has_no_z(tab):
    tex = _latex(tab)
    assert (r"Flexibility15_i = \left\lfloor 1 + (5 - 0.0001) \times"
            r" \frac{AnchoredFlexibility_i - \min(AnchoredFlexibility)}"
            r"{\max(AnchoredFlexibility) - \min(AnchoredFlexibility)}"
            r" \right\rfloor \in \{1,\dots,5\}") in tex
    # the mirror formulas of the other elements keep the same shape
    for name in ('Loyalty15_i', 'WTP15_i', 'RiskTaking15_i'):
        assert name in tex
    # no "z" marker anywhere around the anchored score
    for forbidden in (r"z_{Anchored", "z_Anchored", r"z_{obs\_Flex", r"z_{calc\_Flex",
                      "z_anchored_flexibility"):
        assert forbidden not in tex, forbidden
    assert "z_Anchored" not in _md(tab)


def test_anchored_flexibility_line_and_plain_sentence(tab):
    tex = _latex(tab)
    assert (r"AnchoredFlexibility_i = W_{OFlex} \times ObservedFlexibility_i"
            r" + W_{CFlex} \times Flexibility_i"
            r" = 0.25 \times ObservedFlexibility_i + 0.75 \times Flexibility_i") in tex
    md = _md(tab)
    assert "ObservedFlexibility is the observed flexibility variable stdactions" in md
    # the professor: do not mark the flexibility formulas as standardized
    assert "standardized" not in md.split("ObservedFlexibility is")[1][:400]
    assert "All variables are standardized prior to calculation" not in _helps(tab)


# ---------------------------------------------------------------------------
# 2. Segment mapping table: mirrored direction + black column titles
# ---------------------------------------------------------------------------
def test_segment_mapping_direction_is_mirrored():
    """Segment s gets the LAST s options: segment 5 the full sequence, segment 1 only
    the sequence's last option."""
    df = rtd_tab._segment_mapping_df([2, 4, 3, 1, 5], 'Flexibility')
    lists = df['Priority list for rejected transaction options'].tolist()
    assert lists == ['5', '1 > 5', '3 > 1 > 5', '4 > 3 > 1 > 5', '2 > 4 > 3 > 1 > 5']
    assert df['Options list length'].tolist() == [1, 2, 3, 4, 5]
    labels = df['Flexibility score segment'].tolist()
    assert labels[0] == '1 (Lowest 20% of Flexibility score segment)'
    assert labels[4] == '5 (Highest 20% of Flexibility score segment)'


def test_mapping_tables_rendered_with_black_headers(tab):
    md = _md(tab)
    # rendered as an HTML table (st.dataframe headers are gray)
    assert 'color:#000000;font-weight:600' in md
    assert not list(tab.dataframe), "no st.dataframe on the tab (gray column titles)"
    # every ranking sub-tab's mapping table, mirrored direction
    for element, seq in (('Loyalty', [3, 1, 4, 5, 2]),
                         ('Willingness-to-Pay', [3, 2, 1, 4, 5]),
                         ('Risk-Taking', [4, 2, 1, 3, 5]),
                         ('Flexibility', [2, 4, 3, 1, 5])):
        assert f'{element} score segment' in md
        assert f'5 (Highest 20% of {element} score segment)' in md
        assert f'1 (Lowest 20% of {element} score segment)' in md
        assert ' > '.join(str(o) for o in seq) in md        # segment 5 = full sequence
        assert f'>{seq[-1]}</td>' in md                     # segment 1 = last option only


def test_other_tab_tables_also_black_headed(tab):
    """The aggregation inputs table and (in categorical income mode) the income
    effects table use the same black-header rendering."""
    md = _md(tab)
    assert 'Priority sequence' in md and 'Mechanism' in md
    assert 'Option 3 > Option 1 > Option 4 > Option 5 > Option 2' in md   # loyalty inputs

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_tab_script)
    at.session_state['rtd_income_mode'] = 'Categorical only'
    at.run(timeout=300)
    assert not at.exception
    cat_md = _md(at)
    assert 'Quintile' in cat_md and 'β_income_q' in cat_md
    assert 'color:#000000;font-weight:600' in cat_md
    assert not list(at.dataframe)


# ---------------------------------------------------------------------------
# 3. Option explanations in black markdown, not gray captions
# ---------------------------------------------------------------------------
def test_option_labels_are_markdown_not_captions(tab):
    md, captions = _md(tab), _captions(tab)
    for label in ("Option 1: higher price category, same vendor",
                  "Option 2: other vendor at lower PN price",
                  "Option 3: current vendor at PN price",
                  "Option 4: place a bid",
                  "Option 5: forgo the transaction"):
        assert label in md, label
        assert label not in captions, label
    # one block per ranking sub-tab (4), no bold
    assert md.count("Option 4: place a bid") == 4
    assert "**Option 1" not in md


# ---------------------------------------------------------------------------
# 4. Stochastic component wording
# ---------------------------------------------------------------------------
def test_stochastic_component_text(tab):
    md = _md(tab)
    assert (
        "If stochastic enabled: AnchoredFlexibility_i ~ Normal(μ = anchor, σ) where "
        "the anchor is the continuous AnchoredFlexibility_i score and σ = base σ × "
        "coefficient (overall or per budget level); the drawn values are re-rescaled "
        "over the population and re-binned into the 1-5 Flexibility segment."
    ) in md
    # the other four elements are phrased identically with their own score names
    for score, bins in (('TTP_i', 'the 0-5 options list length'),
                        ('Loyalty_i', 'the 1-5 Loyalty segment'),
                        ('WTP_i', 'the 1-5 WTP segment'),
                        ('RiskTaking_i', 'the 1-5 Risk-Taking segment')):
        assert (
            f"If stochastic enabled: {score} ~ Normal(μ = anchor, σ) where the anchor "
            f"is the continuous {score} score and σ = base σ × coefficient (overall or "
            f"per budget level); the drawn values are re-rescaled over the population "
            f"and re-binned into {bins}."
        ) in md, score
    assert "(or the binned segment" not in md
    assert "`" not in md.split("If stochastic enabled")[1][:400]


# ---------------------------------------------------------------------------
# 5. No "Advanced: stochastic anchor" remnant anywhere on the tab
# ---------------------------------------------------------------------------
def test_no_advanced_stochastic_anchor_section(tab):
    haystack = _md(tab) + _captions(tab) + _helps(tab)
    for forbidden in ("Advanced: stochastic anchor", "Advanced: Stochastic Anchor",
                      "binned anchor", "Binned anchor"):
        assert forbidden not in haystack, forbidden
    assert not any(str(e.label).lower().startswith("advanced") for e in tab.expander)
    assert not any(str(k).startswith('rtd_anchor_') for k in tab.session_state.filtered_state)
    src = __import__('inspect').getsource(rtd_tab)
    assert 'rtd_anchor_' not in src and 'stochastic anchor' not in src.lower()


# ---------------------------------------------------------------------------
# 6. Aggregation sub-tab text + run button + no enable checkbox
# ---------------------------------------------------------------------------
def test_aggregation_subtab_wording(tab):
    md = _md(tab)
    assert "6. Integrated Default List (Rank Aggregation)" in [str(t.label) for t in tab.tabs]

    assert ("priority list of the five options per agent") in md
    assert "all sub-decision mechanisms receive equal weight" in md
    assert "following two rules:" in md
    assert "1. the integrated list is truncated to the Options List Length" in md
    assert ("2. every option listed after Option 5 (forgo the transaction) is dropped."
            in md)
    assert "Both rules apply to the integrated ranking only." in md
    # dropped clause / dropped caption / renamed construct
    assert "there can be no subsequent default option" not in md
    assert "Each customer's list is the tail" not in md + _captions(tab)
    assert "customer" not in md.lower()
    assert "cognitive flexibility" not in md.lower()
    # no bold inside the explanations
    for bold in ("**one ", "**truncated", "**after Option 5**", "**smallest",
                 "**Phase 1", "**Phase 2", "**Schulze**", "**Copeland**",
                 "**Spearman footrule**", "**at random**"):
        assert bold not in md, bold


def test_aggregation_kendall_tau_and_phases(tab):
    md = _md(tab)
    assert (
        "The distance between two rankings is measured by the Kendall-tau distance: "
        "the number of option pairs that the two rankings order differently. With five "
        "options there are 10 pairs; a pair (x, y) counts 1 when one ranking places x "
        "above y and the other places y above x, and 0 otherwise. The integrated "
        "ranking is the ordering of the five options whose total Kendall-tau distance "
        "to the four mechanism rankings is smallest (Kemeny-Young); it is found by "
        "checking all 120 possible orderings."
    ) in md
    assert r"\arg\min_{\pi \in S_5}" in _latex(tab)
    assert ("After applying the Kemeny-Young method, remaining ranking ties are "
            "resolved in two phases:") in md
    assert ("Phase 1 - If Kemeny returns several equally good orderings, the Schulze "
            "(2011) strongest-paths ordering is used to produce an initial ranking.") in md
    assert ("Phase 2 - leftover ties. After applying Kemeny and Schulze, remaining ties "
            "are resolved using Copeland (pairwise wins minus losses) and then by "
            "Spearman footrule (smallest total positional displacement). Any remaining "
            "ties are resolved by randomization which avoids any systematic bias.") in md
    assert "With few input rankings ties are pervasive" not in md


def test_stage_shares_expander_and_constant_removed(tab):
    assert not hasattr(rtd_tab, 'DOC_STAGE_SHARES')
    labels = [str(e.label) for e in tab.expander]
    assert not any("Share of cases settled at each stage" in l for l in labels)
    assert "100,000" not in _md(tab)


def test_aggregation_enable_checkbox_removed_but_flag_true(tab):
    checkbox_keys = [c.key for c in tab.checkbox]
    assert 'rtd_tab_aggregation_enabled' not in checkbox_keys
    assert "Integrate the mechanism rankings into one default list" not in _md(tab)
    assert "Settings" not in [str(m.value) for m in tab.markdown]
    # app/simulation.py still reads the flag; it is always on now
    assert tab.session_state['rtd_aggregation_enabled'] is True


def test_aggregation_run_button(tab):
    btn = tab.button(key='rtd_run_aggregation_btn')
    assert btn is not None
    assert str(btn.label) == "🔬 Run Integrated Default List Only"
    md = _md(tab)
    assert ("This run presents the integrated default list results together with the "
            "tie-resolution statistics: the share of agents with initial ties after "
            "Kemeny and the stage at which the ties were settled.") in md
    # the five per-element run buttons are untouched
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert tab.button(key=f'rtd_run_{mech}_btn') is not None


def _stubbed_run_tab_script():
    """The tab with run_individual_decision stubbed out: clicking a Run button records
    the call instead of starting a simulation (AppTest scripts must be self-contained -
    from_function re-executes the function's own source)."""
    import streamlit as st
    import app.pages.decision_execution as dex
    from app.models import initialize_session_state

    st.session_state.setdefault('_runs', [])
    dex.run_individual_decision = lambda name: st.session_state['_runs'].append(name)

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()


def test_aggregation_run_button_sets_element_flag():
    """Clicking the button flags rtd_run_element = 'aggregation' for the results page
    and triggers the same individual Decision 4 run as the per-element buttons."""
    from streamlit.testing.v1 import AppTest
    import app.pages.decision_execution as dex

    original = dex.run_individual_decision
    try:
        at = AppTest.from_function(_stubbed_run_tab_script)
        at.run(timeout=300)
        assert not at.exception
        at.button(key='rtd_run_aggregation_btn').click().run(timeout=300)
        assert not at.exception
        assert at.session_state['rtd_run_element'] == 'aggregation'
        assert at.session_state['_runs'] == ['rejected_transaction_defaults']
    finally:
        dex.run_individual_decision = original


# ---------------------------------------------------------------------------
# 7. "Running the whole decision" paragraph
# ---------------------------------------------------------------------------
def test_running_the_whole_decision_paragraph(tab):
    md = _md(tab)
    assert "##### Running the whole decision" in md
    para = next(str(m.value) for m in tab.markdown
                if str(m.value).startswith("Run Rejected Transaction Defaults Only presents"))
    assert "every element's results" in para
    assert "score distribution and the option allocation" in para
    assert "percentage share of each first integrated default option" in para
    assert ("Kemeny-Young consensus of the Loyalty, Willingness-to-Pay, Risk-Taking "
            "and Flexibility rankings, truncated to the agent's Options List Length "
            "and cut after Option 5") in para
    assert "Run Integrated Default List Only button" in para
    assert ("The complete simulation presents only the default list length and the "
            "first integrated option per agent.") in para
    assert "The Excel files contain all element variables" in para
    assert "**" not in para
    assert "customer" not in para.lower()


# ---------------------------------------------------------------------------
# 8. TTP intercept help text: standardized-scale wording
# ---------------------------------------------------------------------------
def test_intercept_help_text_is_standardized_scale(tab):
    help_text = str(tab.number_input(key='rtd_tab_intercept_ttp').help)
    assert "β₀ baseline for this element (research default 0.0500)" in help_text
    assert ("The intercept shifts the element's standardized score by β and thereby "
            "the allocation across the segment boundaries") in help_text
    assert "raw" not in help_text.lower()
    for mech in ('loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert "standardized score by β" in str(
            tab.number_input(key=f'rtd_tab_intercept_{mech}').help)
