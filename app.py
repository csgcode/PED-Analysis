import streamlit as st
from src.ui_components import (
    render_what_if_analysis,
    render_promo_planner,
    render_inventory_forecaster,
    render_portfolio_analysis
)
from src.utils import load_all_models, load_ped_results

# --- Page Configuration ---
st.set_page_config(
    page_title="SME Price Optimization Tool",
    page_icon="💡",
    layout="wide"
)

# --- Load Models and Data ---
# This is done once and cached for performance
try:
    models = load_all_models()
    ped_results = load_ped_results()
    st.session_state['models_loaded'] = True
    st.session_state['models'] = models
    st.session_state['ped_results'] = ped_results
except FileNotFoundError:
    st.error("Model files not found. Please run the `model_trainer.py` script first.")
    st.stop()


# --- App Header ---
st.title(" pragmatic Price Optimization for E-commerce SMEs")
st.markdown("""
This tool is a practical demonstration of the hybrid analytical framework developed in the MSc Data Science dissertation. 
It combines predictive forecasting with strategic insights to help SMEs make data-driven pricing decisions.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
use_case = st.sidebar.radio("Choose a Use Case:", [
    "📈 'What-If' Weekly Price Setting",
    "🎁 Strategic Promotional Planning",
    "📦 Proactive Inventory Management",
    "📊 Strategic Product Portfolio Analysis"
])

# --- Main Content Area ---
# The content of this area will change based on the sidebar selection.
if use_case == "📈 'What-If' Weekly Price Setting":
    render_what_if_analysis()
elif use_case == "🎁 Strategic Promotional Planning":
    render_promo_planner()
elif use_case == "📦 Proactive Inventory Management":
    render_inventory_forecaster()
elif use_case == "📊 Strategic Product Portfolio Analysis":
    render_portfolio_analysis()
