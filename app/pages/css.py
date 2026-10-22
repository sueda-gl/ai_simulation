# app/pages/css.py
"""
The global CSS of the app (header classes, metric cards, the Decision 4 sub-tab
labels), injected once per script run by app_enhanced_new.py.
"""


def get_css_styles():
    """Return the CSS styles for the application"""
    return """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.page-header {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #34495e;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
}
.subsection-header {
    font-size: 1.1rem;
    font-weight: 500;
    color: #2c3e50;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
.stAlert {
    margin-top: 1rem;
}
.navigation-buttons {
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.param-group {
    margin-bottom: 2rem;
    padding: 1rem;
    border-left: 4px solid #3498db;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-radius: 0 0.5rem 0.5rem 0;
}
/* Decision 4 sub-decision mechanism tabs (st.container(key="rtd_subtabs") on the
   Decision 4 tab): larger bold labels (requested 2026-09-17). The label markdown may
   be wrapped in a p/span/div depending on the Streamlit build, so the rule targets the
   tab and its descendants - FONT properties only, which cannot disturb the layout. */
.st-key-rtd_subtabs [data-baseweb="tab"],
.st-key-rtd_subtabs [data-baseweb="tab"] * {
    font-size: 1.15rem;
    font-weight: 700;
}

</style>
"""
