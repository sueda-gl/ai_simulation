# app/pages/navigation.py
"""
Navigation functions for the Enhanced AI Agent Simulation.
"""
import streamlit as st


def restore_original_session_state():
    """Restore original population and income mode values if they were overridden.
    
    IMPORTANT:
    Only restore when navigating AWAY FROM the results page.
    This prevents unintended resets when moving between Page 1 and Page 2.
    """
    # Only restore when coming from results
    current_page = st.session_state.get('page', None)
    if current_page != 'results':
        return
    
    if hasattr(st.session_state, '_original_population_mode'):
        st.session_state.population_mode = st.session_state._original_population_mode
        st.session_state.income_spec_mode = st.session_state._original_income_spec_mode
        delattr(st.session_state, '_original_population_mode')
        delattr(st.session_state, '_original_income_spec_mode')


def go_to_page1():
    restore_original_session_state()
    st.session_state.page = 'page1'


def go_to_page2():
    restore_original_session_state()
    st.session_state.page = 'page2'


def go_to_results():
    st.session_state.page = 'results'


def render_navigation(current_page):
    """Render navigation buttons based on current page"""
    st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
    
    if current_page == 'page1':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            st.button("Next: Decision Parameters →", type="primary", on_click=go_to_page2, use_container_width=True)
    
    elif current_page == 'page2':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("← Back to Common Parameters", on_click=go_to_page1, use_container_width=True)
        with col3:
            # Show "Go to Results" button if results exist
            if st.session_state.simulation_results is not None or st.session_state.mc_results is not None:
                st.button("View Results →", type="primary", on_click=go_to_results, use_container_width=True)
    
    elif current_page == 'results':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("← Back to Decision Parameters", on_click=go_to_page2, use_container_width=True)
        with col3:
            st.button("Back to Common Parameters →", on_click=go_to_page1, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
