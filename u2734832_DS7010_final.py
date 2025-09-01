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
# !pip install pandas numpy statsmodels scikit-learn xgboost matplotlib seaborn -q
print("Libraries installed successfully.")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
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
OUTPUT_DIR = 'outputs'
FILE_PATH = "ENTER_FILE_PATH_LOCATION"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- Data Loading and Full Preparation Pipeline ---
print("\n--- Starting Data Preparation Pipeline ---")
try:
    raw_df = pd.read_csv(FILE_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: '{FILE_PATH}' file not found. Please upload the file.")
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
    Calculates and plots a correlation matrix for the features. Includes
    both Pearson (linear) and Spearman (monotonic) correlations.
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


def run_elasticity_model(data, product_name):
    """
    Runs the Log-Log regression, prints key metrics including both R-squared
    and Adjusted R-squared, and generates diagnostic plots.
    """
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

    return model


def tune_and_run_random_forest(product_data, product_name):
    print(f"\n--- Model 2: Tuned Random Forest Regressor (Predictive) ---")
    features = ['Weekly_Avg_Price', 'Month', 'Week_of_Year', 'Quantity_Last_Week', 'Quantity_4_Week_MA', 'Is_Holiday_Season']
    X = product_data[features]
    y = product_data['Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5]}
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='r2')
    grid_search.fit(X_train, y_train)
    print(f"   - Best parameters found: {grid_search.best_params_}")
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)
    r2, rmse, mae = r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)
    print(f"   - Final R-squared: {r2:.3f}\n   - Final RMSE: {rmse:.3f}\n   - Final MAE: {mae:.3f}")
    importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind='barh')
    plt.title(f'TUNED RF Feature Importance for {product_name}')
    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_tuned_rf_feature_importance.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"   - Feature importance plot saved to {plot_filename}")
    return y_test, y_pred

def tune_and_run_xgboost(product_data, product_name):
    print(f"\n--- Model 3: Tuned XGBoost Regressor (Predictive) ---")
    features = ['Weekly_Avg_Price', 'Month', 'Week_of_Year', 'Quantity_Last_Week', 'Quantity_4_Week_MA', 'Is_Holiday_Season']
    X = product_data[features]
    y = product_data['Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1], 'subsample': [0.7, 1.0]}
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgbr, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=0, scoring='r2')
    grid_search_xgb.fit(X_train, y_train)
    print(f"   - Best parameters found: {grid_search_xgb.best_params_}")
    best_xgb_model = grid_search_xgb.best_estimator_
    y_pred = best_xgb_model.predict(X_test)
    r2, rmse, mae = r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)
    print(f"   - Final R-squared: {r2:.3f}\n   - Final RMSE: {rmse:.3f}\n   - Final MAE: {mae:.3f}")
    importances = pd.Series(best_xgb_model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind='barh')
    plt.title(f'TUNED XGBoost Feature Importance for {product_name}')
    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f'{product_name}_tuned_xgb_feature_importance.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"   - Feature importance plot saved to {plot_filename}")
    return y_test, y_pred

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
    if ped is None or not isinstance(ped, (int, float)) or ped >= -1: return "Cannot optimize: Demand is inelastic or PED is not valid."
    return cost / (1 + (1 / ped))

def apply_psychological_pricing(price):
    if isinstance(price, (int, float)) and price > 0: return int(price) + 0.99
    return price

print("\n--- Starting Full Analysis Pipeline for Selected Products ---")

for product in SELECTED_PRODUCTS:
    print(f"\n{'='*60}\nAnalyzing Product: {product}\n{'='*60}")
    product_data = final_df[final_df['Description'] == product].copy()
    if product_data.empty:
        print(f"No data available for {product} after processing. Skipping.")
        continue
    plot_eda(product_data, product)
    plot_advanced_eda(product_data, product) # ADD THIS LINE

    run_correlation_analysis(product_data, product)
    log_model = run_elasticity_model(product_data, product)
    y_test_rf, y_pred_rf = tune_and_run_random_forest(product_data, product)
    y_test_xgb, y_pred_xgb = tune_and_run_xgboost(product_data, product)
    plot_model_comparison(y_test_rf, y_pred_rf, y_pred_xgb, product)
    print("\n--- Price Optimization Recommendations (from Log-Log Model) ---")
    ped_value = log_model.params.get('log_Price', None)
    cost_value = PRODUCT_COSTS.get(product, np.mean(product_data['Weekly_Avg_Price']) * 0.4)
    optimal_price = static_price_optimization(ped_value, cost_value)
    final_price = apply_psychological_pricing(optimal_price)
    print(f"   - Static Optimal Price: £{optimal_price:.2f}" if isinstance(optimal_price, float) else f"   - {optimal_price}")
    if isinstance(final_price, float): print(f"   - Recommended Price (Psychologically Adjusted): £{final_price:.2f}")

print(f"\n{'='*60}\nFull Analysis Complete. All outputs saved to '{OUTPUT_DIR}'.\n{'='*60}")