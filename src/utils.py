import joblib
import os
import streamlit as st

MODEL_DIR = 'saved_models'
SELECTED_PRODUCTS = [
    'WHITE HANGING HEART T-LIGHT HOLDER',
    'REGENCY CAKESTAND 3 TIER',
    'JUMBO BAG RED RETROSPOT'
]

@st.cache_data # Use Streamlit's caching to load models only once
def load_model(product_name):
    """Loads a single pre-trained model file from disk."""
    safe_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f'random_forest_model_{safe_name}.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_all_models():
    """Loads all models for the selected products into a dictionary."""
    models = {}
    for product in SELECTED_PRODUCTS:
        models[product] = load_model(product)
    return models

@st.cache_data
def load_ped_results():
    """Loads the saved Price Elasticity of Demand results."""
    ped_path = os.path.join(MODEL_DIR, 'ped_results.joblib')
    if not os.path.exists(ped_path):
        raise FileNotFoundError(f"PED results file not found at {ped_path}")
    ped_results = joblib.load(ped_path)
    return ped_results
