# ==============================================================================
#
# DS7010 MSc Dissertation: Consolidated Analysis Script
#
# Author: Gokul Chirayath Sudheer
#
# Project: PRICE ELASTICITY ANALYSIS AND PRAGMATIC PRICE OPTIMIZATION FOR E-COMMERCE SME
#
# Description:
# This script contains the complete, end-to-end code for the dissertation.
# It performs all stages of the analysis and ML pipelines.
# Intented to be used in Notebooks
#
# ==============================================================================

print("--- Installing required libraries ---")
# !pip install pandas numpy statsmodels scikit-learn xgboost matplotlib seaborn joblib -q
print("Libraries installed successfully.")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')

# --- Global Settings & Configuration ---
SELECTED_PRODUCTS = [
    'WHITE HANGING HEART T-LIGHT HOLDER',
    'REGENCY CAKESTAND 3 TIER',
    'JUMBO BAG RED RETROSPOT'
]
PRODUCT_COSTS = {
    'WHITE HANGING HEART T-LIGHT HOLDER': 1.00,
    'REGENCY CAKESTAND 3 TIER': 5.50,
    'JUMBO BAG RED RETROSPOT': 0.75
}
# --- IMPORTANT: UPDATE THIS PATH ---
FILE_PATH = "online_retail_II_kaggle.csv" #

# Define output directories for models and plots
MODEL_DIR = 'saved_models'
OUTPUT_DIR = 'tmp/outputs'

SAVE_MODEL = True

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


# --- Data Loading and Full Preparation Pipeline ---
print("\n--- Starting Data Preparation Pipeline ---")
try:
    raw_df = pd.read_csv(FILE_PATH, encoding='cp1252') # Added encoding for compatibility
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: '{FILE_PATH}' file not found. Please update the FILE_PATH variable in the script.")
    exit()

# Cleaning
raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
raw_df.dropna(subset=['Customer ID'], inplace=True)
raw_df = raw_df[(raw_df['Quantity'] > 0) & (raw_df['Price'] > 0)]
df_selected = raw_df[raw_df['Description'].isin(SELECTED_PRODUCTS)]

# Aggregation
print("Aggregating data to a weekly level...")
df_selected['Date'] = df_selected['InvoiceDate'].dt.date
product_daily = df_selected.groupby(['Description', 'Date']).agg(
    Quantity=('Quantity', 'sum'),
    Price=('Price', 'mean')
).reset_index()
product_daily['Date'] = pd.to_datetime(product_daily['Date'])
product_daily['Week_of_Year'] = product_daily['Date'].dt.isocalendar().week
product_daily['Year'] = product_daily['Date'].dt.year
product_daily['Month'] = product_daily['Date'].dt.month
weekly_data = product_daily.groupby(['Description', 'Year', 'Week_of_Year', 'Month']).agg(
    Quantity=('Quantity', 'sum'),
    Weekly_Avg_Price=('Price', 'mean')
).reset_index()

# Explicitly convert data types to prevent plotting errors
weekly_data['Week_of_Year'] = weekly_data['Week_of_Year'].astype('int64')
weekly_data['Year'] = weekly_data['Year'].astype('int64')
weekly_data['Month'] = weekly_data['Month'].astype('int64')
weekly_data.sort_values(by=['Description', 'Year', 'Week_of_Year'], inplace=True)

# Outlier Handling and Feature Engineering
print("Applying outlier handling and advanced feature engineering...")
all_final_data = []
for product in SELECTED_PRODUCTS:
    product_df = weekly_data[weekly_data['Description'] == product].copy()
    Q1 = product_df['Quantity'].quantile(0.25)
    Q3 = product_df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = (product_df['Quantity'] > upper_bound).sum()
    if outliers_count > 0:
        print(f"   - For '{product}', found and capped {outliers_count} outliers.")
    product_df['Quantity'] = np.where(product_df['Quantity'] > upper_bound, upper_bound, product_df['Quantity'])
    product_df['Quantity_Last_Week'] = product_df['Quantity'].shift(1)
    product_df['Quantity_4_Week_MA'] = product_df['Quantity'].rolling(window=4).mean().shift(1)
    product_df['Is_Holiday_Season'] = product_df['Month'].apply(lambda x: 1 if x in [11, 12] else 0)
    all_final_data.append(product_df)

final_df = pd.concat(all_final_data)
final_df.dropna(inplace=True)
print("Data preparation complete.")
print("-" * 60)

# --- Helper Function for Saving Models ---
def save_model_joblib(model_object, product_name, model_prefix):
    """Saves a model object to a file using joblib."""
    safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    model_filename = os.path.join(MODEL_DIR, f'{model_prefix}_{safe_product_name}.joblib')
    joblib.dump(model_object, model_filename)
    print(f"\n   >>> Model for '{product_name}' saved successfully to: {model_filename} <<<")


def plot_eda(data, product_name):
    print(f"\n--- Generating EDA Plots for: {product_name} ---")
    data['Date'] = pd.to_datetime(data['Year'].astype(str) + data['Week_of_Year'].astype(str) + '1', format='%Y%W%w')
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f'Exploratory Data Analysis for {product_name}', fontsize=16)
    sns.scatterplot(data=data, x='Weekly_Avg_Price', y='Quantity', ax=axes[0], alpha=0.6)
    axes[0].set_title('Weekly Average Price vs. Quantity Sold')
    sns.lineplot(data=data, x='Date', y='Quantity', ax=axes[1], errorbar=None)
    axes[1].set_title('Weekly Demand Over Time')
    sns.histplot(data=data, x='Weekly_Avg_Price', ax=axes[2], bins=15, kde=True)
    axes[2].set_title('Distribution of Weekly Average Prices')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_eda_plots.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"   - EDA plots saved to {plot_filename}")


def plot_advanced_eda(data, product_name):
    """
    Generates and saves more advanced EDA plots for a given product,
    including seasonality and price trend analysis.
    """
    print(f"--- Generating Advanced EDA Plots for: {product_name} ---")

    data['Date'] = pd.to_datetime(data['Year'].astype(str) + data['Week_of_Year'].astype(str) + '1', format='%Y%W%w')

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Advanced EDA for {product_name}', fontsize=16)

    # Plot 1: Monthly Sales Boxplot to show seasonality
    sns.boxplot(data=data, x='Month', y='Quantity', ax=axes[0])
    axes[0].set_title('Distribution of Weekly Sales by Month (Seasonality)')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Weekly Quantity Sold')
    axes[0].set_xticks(range(12)) # to ensure all 12 months are labelled if data is sparse
    axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Plot 2: Price Over Time Line Plot to show pricing strategy
    sns.lineplot(data=data, x='Date', y='Weekly_Avg_Price', ax=axes[1], errorbar=None)
    axes[1].set_title('Weekly Average Price Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Average Price (£)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_advanced_eda_plots.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"   - Advanced EDA plots saved to {plot_filename}")

def run_correlation_analysis(data, product_name):
    """
    Calculates and plots a correlation matrix for the features.
    """
    print(f"\n--- Performing Correlation Analysis for: {product_name} ---")

    # Select only numeric features for correlation
    numeric_cols = data.select_dtypes(include=np.number)

    # --- 1. Pearson Correlation (measures linear relationships) ---
    pearson_corr = numeric_cols.corr(method='pearson')
    print("\nPearson Correlation with Target ('Quantity'):")
    print(pearson_corr['Quantity'].sort_values(ascending=False))

    # --- 2. Spearman Correlation (measures monotonic relationships) ---
    spearman_corr = numeric_cols.corr(method='spearman')
    print("\nSpearman Correlation with Target ('Quantity'):")
    print(spearman_corr['Quantity'].sort_values(ascending=False))

    # Plot and save the Pearson heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Feature Correlation Matrix (Pearson) for {product_name}')
    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_correlation_heatmap.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"\n   - Pearson correlation heatmap saved to {plot_filename}")

def plot_model_diagnostics(model, product_name):
    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_regress_exog(model, 'log_Price', fig=fig)
    plt.suptitle(f'Log-Log Model Regression Diagnostics for {product_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_log_model_diagnostics.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"   - OLS diagnostic plots saved to {plot_filename}")


def run_elasticity_model(data, product_name, save_model=False):
    print("\n--- Model 1: Log-Log Regression (Explanatory) ---")
    log_data = data.copy()
    log_data['log_Quantity'] = np.log1p(log_data['Quantity'])
    log_data['log_Price'] = np.log1p(log_data['Weekly_Avg_Price'])

    # Define the model formula
    formula = 'log_Quantity ~ log_Price + C(Month) + C(Week_of_Year)'
    model = smf.ols(formula, data=log_data).fit()

    # Extract key metrics
    ped = model.params.get('log_Price', None)
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj

    print(f"   - Price Elasticity of Demand (PED): {ped:.3f}")
    print(f"   - R-squared: {r_squared:.3f}")
    print(f"   - Adjusted R-squared: {adj_r_squared:.3f}")

    plot_model_diagnostics(model, product_name)

    if save_model:
        save_model_joblib(model, product_name, 'log_log_model')

    return model


def tune_and_run_random_forest(product_data, product_name, save_model=False):
    """
    This function now only trains and tunes the RF model, returning the best one.
    """
    print(f"\n--- Model: Tuning Random Forest Regressor for {product_name} ---")
    features = ['Weekly_Avg_Price', 'Month', 'Week_of_Year', 'Quantity_Last_Week', 'Quantity_4_Week_MA', 'Is_Holiday_Season']
    X = product_data[features]
    y = product_data['Quantity']

    # We train on the full dataset for the final model to be deployed
    X_train, y_train = X, y

    # A more focused parameter grid for faster tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='r2')
    grid_search.fit(X_train, y_train)

    print(f"   - Best parameters found: {grid_search.best_params_}")
    print(f"   - Best cross-validated R-squared on training data: {grid_search.best_score_:.3f}")

    best_rf_model = grid_search.best_estimator_
    
    if save_model:
        save_model_joblib(best_rf_model, product_name, 'random_forest_model')
        
    return best_rf_model


def tune_and_run_xgboost(product_data, product_name, save_model=False):
    print(f"\n--- Model: Tuning XGBoost Regressor for {product_name} ---")
    features = ['Weekly_Avg_Price', 'Month', 'Week_of_Year', 'Quantity_Last_Week', 'Quantity_4_Week_MA', 'Is_Holiday_Season']
    X = product_data[features]
    y = product_data['Quantity']
    X_train, y_train = X, y # Train on full data for final model

    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 1.0]
    }
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgbr, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=0, scoring='r2')
    grid_search_xgb.fit(X_train, y_train)

    print(f"   - Best parameters found: {grid_search_xgb.best_params_}")
    best_xgb_model = grid_search_xgb.best_estimator_
    
    if save_model:
        save_model_joblib(best_xgb_model, product_name, 'xgboost_model')

    return best_xgb_model


def plot_model_comparison(y_test, y_pred_rf, y_pred_xgb, product_name):
    plt.figure(figsize=(10, 8))
    sns.regplot(x=y_test, y=y_pred_rf, scatter_kws={'alpha':0.6}, label=f'Random Forest (R2={r2_score(y_test, y_pred_rf):.3f})', line_kws={'color':'blue'})
    sns.regplot(x=y_test, y=y_pred_xgb, scatter_kws={'alpha':0.6}, label=f'XGBoost (R2={r2_score(y_test, y_pred_xgb):.3f})', line_kws={'color':'green'})
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Quantity Sold (Test Set)'); plt.ylabel('Predicted Quantity Sold')
    plt.title(f'Model Prediction Comparison for {product_name}'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_model_comparison_plot.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"\n   - Comparative prediction plot saved to {plot_filename}")


def static_price_optimization(ped, cost):
    if ped is None or not isinstance(ped, (int, float)) or ped >= -1:
        return "Cannot optimize: Demand is inelastic or PED is not valid."
    return cost / (1 + (1 / ped))


def apply_psychological_pricing(price):
    if isinstance(price, (int, float)) and price > 0:
        return int(price) + 0.99
    return price


print("\n--- Starting Final Model Training and Saving Pipeline ---")

ped_results = {}

for product in SELECTED_PRODUCTS:
    print(f"\n{'='*60}\nProcessing Product: {product}\n{'='*60}")
    product_data = final_df[final_df['Description'] == product].copy()
    if product_data.empty:
        print(f"No data available for {product} after processing. Skipping.")
        continue
    
    # plot_eda(product_data, product)
    # plot_advanced_eda(product_data, product)
    # run_correlation_analysis(product_data, product)


    # --- 1. Train and save the best predictive model (Random Forest) ---
    tune_and_run_random_forest(product_data, product, save_model=SAVE_MODEL)
    tune_and_run_xgboost(product_data, product, save_model=SAVE_MODEL)
    
    # --- 2. Run the explanatory model to get PED for strategic context ---
    log_model = run_elasticity_model(product_data, product_name=product) # Not saving this one by default
    ped_value = log_model.params.get('log_Price', None)
    ped_results[product] = round(ped_value, 2) if ped_value is not None else "N/A"
    print(f"   - Calculated PED for strategic context: {ped_results[product]}")


# Save the PED results to a file for the app to use
ped_filename = os.path.join(MODEL_DIR, 'ped_results.joblib')
joblib.dump(ped_results, ped_filename)
print(f"\n{'='*60}\nStrategic PED values saved to: {ped_filename}")


print(f"\n{'='*60}\nFull Training and Saving Process Complete. All models saved to '{MODEL_DIR}'.\n{'='*60}")

