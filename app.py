import streamlit as st
from src.ui_components import (
    render_what_if_analysis,
    render_promo_planner,
    render_inventory_forecaster,
    render_portfolio_analysis
)
from src.utils import load_all_models, load_ped_results, load_feature_means, load_price_stats

# --- Page Configuration ---
st.set_page_config(
    page_title="SME Price Optimization Tool",
    page_icon="�",
    layout="wide"
)

try:
    if 'models_loaded' not in st.session_state:
        st.session_state['models'] = load_all_models()
        st.session_state['ped_results'] = load_ped_results()
        st.session_state['feature_means'] = load_feature_means()
        st.session_state['price_stats'] = load_price_stats() 
        st.session_state['models_loaded'] = True
except FileNotFoundError as e:
    st.error(f"Model file not found. Please ensure all .joblib files are in the 'saved_models' directory. Error: {e}")
    st.stop()


# --- App Header ---
st.title("Price Optimization for E-commerce SMEs")
st.markdown("""
This tool is a practical demonstration of the hybrid analytical framework developed. 
It combines predictive forecasting with strategic insights to help SMEs make data-driven pricing decisions.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
use_case = st.sidebar.radio("Choose a Use Case:", [
    "📊 Strategic Product Portfolio Analysis",
    "🎁 Strategic Promotional Planning",
    "📈 'What-If' Weekly Price Setting",
    "📦 Proactive Inventory Management",
])

# --- Main Content Area ---
if use_case == "📈 'What-If' Weekly Price Setting":
    render_what_if_analysis()
elif use_case == "🎁 Strategic Promotional Planning":
    render_promo_planner()
elif use_case == "📦 Proactive Inventory Management":
    render_inventory_forecaster()
elif use_case == "📊 Strategic Product Portfolio Analysis":
    render_portfolio_analysis()

