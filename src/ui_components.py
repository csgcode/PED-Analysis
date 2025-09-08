import streamlit as st
import pandas as pd
import datetime

def render_what_if_analysis():
    """Renders the UI for Use Case 1 with data-driven price defaults."""
    st.header("📈 'What-If' Weekly Price Setting")
    st.markdown("Use this tool to forecast weekly sales for a product at different potential price points.")

    models = st.session_state['models']
    feature_means = st.session_state['feature_means']
    price_stats = st.session_state['price_stats']
    product_list = list(models.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Select a Product:", product_list, key="uc1_product")
    with col2:
        target_date = st.date_input("Select a Target Week:", datetime.date.today(), key="uc1_date")
    
    avg_price = price_stats[selected_product]['avg_price']

    st.subheader("Enter Potential Prices to Compare")
    price_col1, price_col2, price_col3 = st.columns(3)
    with price_col1:
        price1 = st.number_input("Price 1 (£):", min_value=0.01, value=round(avg_price * 0.95, 2), step=0.50)
    with price_col2:
        price2 = st.number_input("Price 2 (£):", min_value=0.01, value=round(avg_price, 2), step=0.50)
    with price_col3:
        price3 = st.number_input("Price 3 (£):", min_value=0.01, value=round(avg_price * 1.05, 2), step=0.50)

    if st.button("Forecast Sales", key="uc1_forecast"):
        model = models[selected_product]
        potential_prices = [price1, price2, price3]
        results = []

        month = target_date.month
        week_of_year = target_date.isocalendar()[1]
        is_holiday = 1 if month in [11, 12] else 0
        means = feature_means[selected_product]
        
        for price in potential_prices:
            features = pd.DataFrame({
                'Weekly_Avg_Price': [price], 'Month': [month], 'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [means['Quantity_Last_Week']],
                'Quantity_4_Week_MA': [means['Quantity_4_Week_MA']],
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

        best_revenue_str = results_df["Estimated Weekly Revenue"].str.replace('~£', '').str.replace(',', '').astype(float).idxmax()
        best_option = results_df.loc[best_revenue_str]
        st.success(f"**Recommendation:** The price of **{best_option['Proposed Price']}** is forecast to generate the highest revenue ({best_option['Estimated Weekly Revenue']}).")


def render_promo_planner():
    """Renders the UI for Use Case 2 with data-driven price defaults."""
    st.header("🎁 Strategic Promotional Planning")
    st.markdown("Analyze a product's price sensitivity and simulate the impact of promotional discounts.")
    
    models = st.session_state['models']
    ped_results = st.session_state['ped_results']
    feature_means = st.session_state['feature_means']
    price_stats = st.session_state['price_stats']
    product_list = list(models.keys())
    
    selected_product = st.selectbox("Select a Product:", product_list, key="uc2_product")
    
    avg_price = price_stats[selected_product]['avg_price']
    ped_value = ped_results.get(selected_product, "N/A")
    elasticity_text = "**Highly Elastic**" if isinstance(ped_value, float) and ped_value < -2 else "**Elastic**" if isinstance(ped_value, float) and ped_value < -1 else "**Inelastic**"

    with st.container(border=True):
        st.subheader(f"Strategic Insight for: {selected_product}")
        st.metric(label="Price Elasticity of Demand (PED)", value=ped_value)
        st.markdown(f"**Interpretation:** This product is {elasticity_text}, making it a good candidate for promotions.")

    st.subheader("Simulate a Promotion")
    promo_date = st.date_input("Select a Promotional Week:", datetime.date.today(), key="uc2_date")
    
    price_col1, price_col2 = st.columns(2)
    with price_col1:
        promo_price1 = st.number_input("Promotional Price 1 (£):", min_value=0.01, value=round(avg_price * 0.9, 2), step=0.10)
    with price_col2:
        promo_price2 = st.number_input("Promotional Price 2 (£):", min_value=0.01, value=round(avg_price * 0.8, 2), step=0.10)

    if st.button("Simulate Promotion", key="uc2_simulate"):
        model = models[selected_product]
        potential_prices = [promo_price1, promo_price2]
        results = []

        month = promo_date.month
        week_of_year = promo_date.isocalendar()[1]
        is_holiday = 1 if month in [11, 12] else 0
        means = feature_means[selected_product]

        for price in potential_prices:
            features = pd.DataFrame({
                'Weekly_Avg_Price': [price], 'Month': [month], 'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [means['Quantity_Last_Week']],
                'Quantity_4_Week_MA': [means['Quantity_4_Week_MA']],
                'Is_Holiday_Season': [is_holiday]
            })
            prediction = model.predict(features)[0]
            revenue = prediction * price
            results.append({
                "Promotional Price": f"£{price:.2f}",
                "Predicted Sales": f"~{int(prediction)}",
                "Estimated Revenue": f"~£{revenue:,.2f}"
            })
        
        st.subheader("Promotional Forecast")
        st.table(pd.DataFrame(results))


def render_inventory_forecaster():
    """Renders the UI for Use Case 3 with data-driven price defaults."""
    st.header("📦 Proactive Inventory Management")
    st.markdown("Forecast total demand over the next several weeks to inform your inventory planning.")

    models = st.session_state['models']
    feature_means = st.session_state['feature_means']
    price_stats = st.session_state['price_stats']
    product_list = list(models.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Select a Product:", product_list, key="uc3_product")
    with col2:
        weeks_to_forecast = st.slider("Weeks to Forecast:", 1, 12, 4, key="uc3_weeks")
    
    avg_price = price_stats[selected_product]['avg_price']
    planned_price = st.number_input("Planned Average Price (£):", 0.01, value=round(avg_price, 2), step=0.25, key="uc3_price")

    if st.button("Forecast Demand", key="uc3_forecast"):
        model = models[selected_product]
        means = feature_means[selected_product]
        weekly_predictions = []
        total_demand = 0
        start_date = datetime.date.today()

        for i in range(weeks_to_forecast):
            target_date = start_date + datetime.timedelta(weeks=i)
            month = target_date.month
            week_of_year = target_date.isocalendar()[1]
            is_holiday = 1 if month in [11, 12] else 0
            
            features = pd.DataFrame({
                'Weekly_Avg_Price': [planned_price], 'Month': [month], 'Week_of_Year': [week_of_year],
                'Quantity_Last_Week': [means['Quantity_Last_Week']],
                'Quantity_4_Week_MA': [means['Quantity_4_Week_MA']],
                'Is_Holiday_Season': [is_holiday]
            })
            prediction = int(model.predict(features)[0])
            total_demand += prediction
            weekly_predictions.append({"Week": f"Week {i+1} (from {target_date.strftime('%d %b')})", "Forecasted Sales (Units)": f"~{prediction}"})
        
        st.metric(f"Total Estimated Demand (Next {weeks_to_forecast} weeks)", f"~{total_demand:,} Units")
        
        st.subheader("Weekly Breakdown")
        st.table(pd.DataFrame(weekly_predictions))
        st.warning("Note: Forecast assumes a constant price and uses historical averages for recent sales trends.")


def render_portfolio_analysis():
    """Renders the UI for Use Case 4."""
    st.header("📊 Strategic Product Portfolio Analysis")
    st.markdown("Compare price sensitivity across your portfolio to inform high-level strategy.")
    st.info("Price Elasticity of Demand (PED) measures how quantity demanded responds to a price change. A more negative number means demand is more sensitive to price.")
    
    ped_results = st.session_state['ped_results']
    portfolio_data = []

    for product, ped in ped_results.items():
        implication = ""
        if isinstance(ped, float):
            if ped < -2:
                implication = "**Extremely Elastic:** Volume-driven. Ideal for promotions."
            elif ped < -1:
                implication = "**Elastic:** Responds well to discounts."
            else:
                implication = "**Inelastic:** Less sensitive to price. May handle a price increase."
        
        portfolio_data.append({
            "Product Name": product, 
            "Price Elasticity (PED)": ped, 
            "Strategic Implication": implication
        })

    st.table(pd.DataFrame(portfolio_data))

