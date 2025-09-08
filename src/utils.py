import joblib
import os
import streamlit as st

MODEL_DIR = 'saved_models'
SELECTED_PRODUCTS = [
    'WHITE HANGING HEART T-LIGHT HOLDER',
    'REGENCY CAKESTAND 3 TIER',
    'JUMBO BAG RED RETROSPOT'
]

@st.cache_data
def load_model(product_name):
    """Loads a single pre-trained model file from disk."""
    safe_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f'random_forest_model_{safe_name}.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

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
    return joblib.load(ped_path)

@st.cache_data
def load_feature_means():
    """Loads the saved feature means for app placeholders."""
    means_path = os.path.join(MODEL_DIR, 'feature_means.joblib')
    if not os.path.exists(means_path):
        raise FileNotFoundError(f"Feature means file not found at {means_path}")
    return joblib.load(means_path)

@st.cache_data
def load_price_stats():
    """Loads the saved price stats for UI defaults."""
    stats_path = os.path.join(MODEL_DIR, 'price_stats.joblib')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Price stats file not found at {stats_path}")
    return joblib.load(stats_path)

