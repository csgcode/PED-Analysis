import streamlit as st
import pandas as pd
import datetime

def render_what_if_analysis():
    """Renders the UI for Use Case 1: 'What-If' Weekly Price Setting."""
    st.header("📈 'What-If' Weekly Price Setting")
    st.markdown("Use this tool to forecast weekly sales for a product at different potential price points.")

    # --- User Inputs ---
    models = st.session_state['models']
    product_list = list(models.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Select a Product:", product_list, key="uc1_product")
    with col2:
        target_date = st.date_input("Select a Target Week:", datetime.date.today(), key="uc1_date")

    st.subheader("Enter Potential Prices to Compare")
    price_col1, price_col2, price_col3 = st.columns(3)
    with price_col1:
        price1 = st.number_input("Price 1 (£):", min_value=0.01, value=9.99, step=0.50)
    with price_col2:
        price2 = st.number_input("Price 2 (£):", min_value=0.01, value=10.99, step=0.50)
    with price_col3:
        price3 = st.number_input("Price 3 (£):", min_value=0.01, value=11.99, step=0.50)

    # --- Prediction Logic ---
    if st.button("Forecast Sales", key="uc1_forecast"):
        model = models[selected_product]
        potential_prices = [price1, price2, price3]
        results = []

        # Prepare features from date
        month = target_date.month
        week_of_year = target_date.isocalendar()[1]
        is_holiday = 1 if month in [11, 12] else 0
        
        # For lagged features, we make a reasonable assumption to use the mean of the historical data
        # In a real-world app, you might fetch the most recent data.
        quantity_last_week_placeholder = 200 # A reasonable placeholder
        quantity_4_week_ma_placeholder = 200 # A reasonable placeholder

        for price in potential_prices:
            features = pd.DataFrame({
                'Weekly_Avg_Price': [price],
                'Month': [month],
                'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [quantity_last_week_placeholder],
                'Quantity_4_Week_MA': [quantity_4_week_ma_placeholder],
                'Is_Holiday_Season': [is_holiday]
            })
            prediction = model.predict(features)[0]
            revenue = prediction * price
            results.append({
                "Proposed Price": f"£{price:.2f}",
                "Predicted Weekly Sales (Units)": f"~{int(prediction)}",
                "Estimated Weekly Revenue": f"~£{revenue:,.2f}"
            })

        st.subheader("Forecasted Results")
        results_df = pd.DataFrame(results)
        st.table(results_df)

        # Find best option
        best_revenue_str = results_df["Estimated Weekly Revenue"].str.replace('~£', '').str.replace(',', '').astype(float).idxmax()
        best_option = results_df.loc[best_revenue_str]
        st.success(f"**Recommendation:** The price of **{best_option['Proposed Price']}** is forecast to generate the highest revenue ({best_option['Estimated Weekly Revenue']}).")


def render_promo_planner():
    """Renders the UI for Use Case 2: Strategic Promotional Planning."""
    st.header("🎁 Strategic Promotional Planning")
    st.markdown("Analyze a product's price sensitivity and simulate the impact of promotional discounts.")

    # --- Strategic Insight ---
    models = st.session_state['models']
    ped_results = st.session_state['ped_results']
    product_list = list(models.keys())
    
    selected_product = st.selectbox("Select a Product:", product_list, key="uc2_product")
    
    ped_value = ped_results.get(selected_product, "N/A")
    elasticity_text = "**Highly Elastic**" if ped_value < -2 else "**Elastic**" if ped_value < -1 else "**Inelastic**"

    with st.container(border=True):
        st.subheader(f"Strategic Insight for: {selected_product}")
        st.metric(label="Price Elasticity of Demand (PED)", value=ped_value)
        st.markdown(f"**Interpretation:** This product is {elasticity_text}. Demand is sensitive to price changes, making it a good candidate for promotions.")

    # --- Tactical Simulation ---
    st.subheader("Simulate a Promotion")
    col1, col2 = st.columns(2)
    with col1:
        promo_date = st.date_input("Select a Promotional Week:", datetime.date.today(), key="uc2_date")
    
    promo_price1 = st.number_input("Promotional Price 1 (£):", min_value=0.01, value=1.49, step=0.10, key="uc2_p1")
    promo_price2 = st.number_input("Promotional Price 2 (£):", min_value=0.01, value=1.29, step=0.10, key="uc2_p2")

    if st.button("Simulate Promotion", key="uc2_simulate"):
        # This logic is very similar to Use Case 1
        model = models[selected_product]
        potential_prices = [promo_price1, promo_price2]
        results = []
        month = promo_date.month
        week_of_year = promo_date.isocalendar()[1]
        is_holiday = 1 if month in [11, 12] else 0
        quantity_last_week_placeholder = 200
        quantity_4_week_ma_placeholder = 200

        for price in potential_prices:
            features = pd.DataFrame({
                'Weekly_Avg_Price': [price], 'Month': [month], 'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [quantity_last_week_placeholder], 'Quantity_4_Week_MA': [quantity_4_week_ma_placeholder],
                'Is_Holiday_Season': [is_holiday]
            })
            prediction = model.predict(features)[0]
            revenue = prediction * price
            results.append({
                "Promotional Price": f"£{price:.2f}", "Predicted Sales": f"~{int(prediction)}", "Estimated Revenue": f"~£{revenue:,.2f}"
            })
        
        st.subheader("Promotional Forecast")
        st.table(pd.DataFrame(results))

def render_inventory_forecaster():
    """Renders the UI for Use Case 3: Proactive Inventory Management."""
    st.header("📦 Proactive Inventory Management")
    st.markdown("Forecast total demand over the next several weeks to inform your inventory planning.")

    models = st.session_state['models']
    product_list = list(models.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Select a Product:", product_list, key="uc3_product")
    with col2:
        weeks_to_forecast = st.slider("Weeks to Forecast:", min_value=1, max_value=12, value=4, key="uc3_weeks")

    planned_price = st.number_input("Enter Planned Average Price for this Period (£):", min_value=0.01, value=12.50, step=0.25, key="uc3_price")

    if st.button("Forecast Demand", key="uc3_forecast"):
        model = models[selected_product]
        weekly_predictions = []
        total_demand = 0
        start_date = datetime.date.today()

        for i in range(weeks_to_forecast):
            target_date = start_date + datetime.timedelta(weeks=i)
            month = target_date.month
            week_of_year = target_date.isocalendar()[1]
            is_holiday = 1 if month in [11, 12] else 0
            quantity_last_week_placeholder = 200
            quantity_4_week_ma_placeholder = 200

            features = pd.DataFrame({
                'Weekly_Avg_Price': [planned_price], 'Month': [month], 'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [quantity_last_week_placeholder], 'Quantity_4_Week_MA': [quantity_4_week_ma_placeholder],
                'Is_Holiday_Season': [is_holiday]
            })
            prediction = int(model.predict(features)[0])
            total_demand += prediction
            weekly_predictions.append({"Week": f"Week {i+1}", "Forecasted Sales (Units)": f"~{prediction}"})
        
        st.metric(label=f"Total Estimated Demand for next {weeks_to_forecast} weeks", value=f"~{total_demand:,} Units")
        st.subheader("Weekly Breakdown")
        st.table(pd.DataFrame(weekly_predictions))
        st.warning("Note: This forecast assumes the planned price is constant and uses historical averages for recent sales trends.")


def render_portfolio_analysis():
    """Renders the UI for Use Case 4: Strategic Product Portfolio Analysis."""
    st.header("📊 Strategic Product Portfolio Analysis")
    st.markdown("Compare the price sensitivity across your product portfolio to inform high-level strategy.")
    st.info("Price Elasticity of Demand (PED) measures how much the quantity demanded of a good responds to a change in its price. A more negative number means demand is more elastic (i.e., more sensitive to price changes).")

    ped_results = st.session_state['ped_results']
    
    portfolio_data = []
    for product, ped in ped_results.items():
        if ped < -2:
            implication = "**Extremely Elastic:** Volume-driven. Highly price-sensitive. Ideal for promotions."
        elif ped < -1:
            implication = "**Elastic:** Responds well to discounts, but less sensitive than the others."
        else:
            implication = "**Inelastic:** Less sensitive to price changes. May handle a price increase."
        
        portfolio_data.append({
            "Product Name": product,
            "Price Elasticity (PED)": ped,
            "Strategic Implication": implication
        })

    st.table(pd.DataFrame(portfolio_data))
