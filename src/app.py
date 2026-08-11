import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.api_client import get_api_client

import re

def strip_emoji(text: str) -> str:
    """Remove emoji/symbol characters from LLM-generated prose for cleaner display.
    Does not touch underlying data values (RiskTier, AlertStatus, etc.),
    only the narrative text shown to the user."""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
        "\U00002600-\U000027BF"  # misc symbols, dingbats
        "\U0001F1E6-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="RI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .tab-description {
        font-size: 1.25rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .risk-extreme { color: #4d0000; font-weight: bold; }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── API client ─────────────────────────────────────
api = get_api_client()


# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.markdown("## Retail Intelligence")
st.sidebar.markdown("---")

selected_tab = st.sidebar.radio(
    "Navigate",
    ["Churn Prediction", "Demand Forecast", "AI Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** UCI Online Retail (UK, 2010-2011)")
st.sidebar.markdown("**Model:** XGBoost + SHAP")
st.sidebar.markdown("**LLM:** Ollama Llama3 (local)")


_sidebar_backtest = api.get_backtest_results()
_sidebar_summary = api.get_demand_summary()
_sidebar_products = pd.DataFrame(_sidebar_summary.get('products', []))
_sidebar_unreliable = _sidebar_products[_sidebar_products['AlertLevel'] == 'unreliable']['StockCode'].tolist()

_sidebar_folds = pd.DataFrame(_sidebar_backtest.get('fold_details', []))
_sidebar_folds = _sidebar_folds.rename(columns={
    'stock_code': 'StockCode', 'prophet_mape': 'Prophet_MAPE',
    'beat_naive': 'BeatNaive', 'beat_seasonal': 'BeatSeasonal'
})
_sidebar_folds_ok = _sidebar_folds[~_sidebar_folds['StockCode'].isin(_sidebar_unreliable)]

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
if len(_sidebar_folds_ok) > 0:
    st.sidebar.markdown(f"Demand MAPE: {_sidebar_folds_ok['Prophet_MAPE'].median():.1f}% (median)")
    st.sidebar.markdown(
        f"Prophet win rate: {_sidebar_folds_ok['BeatNaive'].mean():.0%} vs naive, "
        f"{_sidebar_folds_ok['BeatSeasonal'].mean():.0%} vs seasonal"
    )

# ══════════════════════════════════════════════════════════════════
# TAB 1: CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════

if selected_tab == "Churn Prediction":

    st.markdown('<p class="main-header">Customer Churn Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="tab-description">Identify at-risk customers before they leave — '
                'with SHAP-explained reasons and revenue impact.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
 
    # Load data via API
    risk_table = api.get_all_customers()
    feature_importance_data = api.get_feature_importance()
    feature_importance = pd.DataFrame(feature_importance_data.get("features", []))

    # ── Row 1: KPI metrics ─────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    total_customers = len(risk_table)
    extreme_risk = risk_table[risk_table['RiskTier'] == '⚫ Extreme Risk']
    high_risk = risk_table[risk_table['RiskTier'] == '🔴 High Risk']
    medium_risk = risk_table[risk_table['RiskTier'] == '🟡 Medium Risk']
    total_revenue_at_risk = risk_table['RevenueAtRisk'].sum()

    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}")

    with col2:
        st.metric(
            label="Extreme Risk",
            value=f"{len(extreme_risk):,}",
            delta=f"{len(extreme_risk)/total_customers:.1%} of base",
            delta_color="off"
        )

    with col3:
        st.metric(
            label="High Risk",
            value=f"{len(high_risk):,}",
            delta=f"{len(high_risk)/total_customers:.1%} of base",
            delta_color="off"
        )

    with col4:
        st.metric(
            label="Medium Risk",
            value=f"{len(medium_risk):,}",
            delta=f"{len(medium_risk)/total_customers:.1%} of base",
            delta_color="off"
        )

    with col5:
        st.metric(label="Revenue at Risk", value=f"${total_revenue_at_risk:,.0f}")

    st.markdown("---")
    
    top_20pct_n = max(int(len(risk_table) * 0.2), 1)
    top_20pct_revenue = risk_table.nlargest(top_20pct_n, 'RevenueAtRisk')['RevenueAtRisk'].sum()
    st.caption(
        f"Your top 20% highest-risk-by-revenue customers ({top_20pct_n:,} people) "
        f"represent ${top_20pct_revenue:,.0f} in revenue at risk — "
        f"focus retention efforts here first."
    )
    
    # ── Row 2: Charts ──────────────────────────────────────────────
    st.subheader("Risk Overview")

    tier_color_map = {
        '⚫ Extreme Risk': '#4d0000',
        '🔴 High Risk': '#d62728',
        '🟡 Medium Risk': '#ff7f0e',
        '🟢 Low Risk': '#2ca02c'
    }

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        tier_counts = risk_table['RiskTier'].value_counts().reset_index()
        tier_counts.columns = ['RiskTier', 'Count']

        fig_pie = px.pie(
            tier_counts, names='RiskTier', values='Count',
            title='Customer Risk Distribution',
            color='RiskTier', color_discrete_map=tier_color_map,
            hole=0.4
        )
        fig_pie.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_c2:
        tier_revenue = risk_table.groupby('RiskTier')['RevenueAtRisk'].sum().reset_index()

        fig_revenue = px.bar(
            tier_revenue, x='RiskTier', y='RevenueAtRisk',
            title='Revenue at Risk by Tier ($)',
            color='RiskTier', color_discrete_map=tier_color_map
        )
        fig_revenue.update_layout(
            height=300, margin=dict(t=40, b=0),
            showlegend=False, yaxis_title="Revenue at Risk ($)"
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col_c3:
        fig_hist = px.histogram(
            risk_table, x='ChurnProbability', nbins=20,
            title='Churn Probability Distribution',
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            height=300, margin=dict(t=40, b=0),
            xaxis_title="Churn Probability", yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # ── Row: Feature importance + scatter ──────────────────────────
    st.markdown("---")
    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        fig_shap = px.bar(
            feature_importance.head(7),
            x='MeanAbsSHAP', y='Feature',
            orientation='h',
            title='SHAP Feature Importance (Global)',
            color='MeanAbsSHAP',
            color_continuous_scale='RdYlGn_r'
        )
        fig_shap.update_layout(
            height=350, margin=dict(t=40, b=0),
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_fi2:
        fig_scatter = px.scatter(
            risk_table, x='DaysActive', y='Monetary',
            color='RiskTier',
            title='Customer Map: Tenure vs Spend',
            color_discrete_map=tier_color_map,
            opacity=0.6,
            labels={'DaysActive': 'Days Active', 'Monetary': 'Total Spend ($)'}
        )
        fig_scatter.update_layout(height=350, margin=dict(t=40, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 3: Filters ─────────────────────────────────────────────
    st.subheader("Filter Customers")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        risk_filter = st.multiselect(
            "Risk Tier",
            options=['⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk'],
            default=['⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk']
        )

    with col_f2:
        min_prob = st.slider(
            "Minimum Churn Probability",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )

    with col_f3:
        min_revenue = st.number_input(
            "Minimum Revenue at Risk ($)",
            min_value=0, value=0, step=10
        )

    filtered_table = risk_table[
        (risk_table['RiskTier'].isin(risk_filter)) &
        (risk_table['ChurnProbability'] >= min_prob) &
        (risk_table['RevenueAtRisk'] >= min_revenue)
    ].copy()

    st.caption(f"Showing {len(filtered_table):,} customers "
               f"(filtered from {total_customers:,})")

    st.markdown("---")

    # ── Row 4: Customer risk table ─────────────────────────────────
    st.subheader("Customer Risk Table")

    display_cols = {
        'RiskTier': 'Risk Tier',
        'ChurnProbability': 'Churn Prob.',
        'CLV': 'CLV ($)',
        'RevenueAtRisk': 'Revenue at Risk ($)',
        'DaysActive': 'Days Active',
        'Frequency': 'Orders',
        'Monetary': 'Total Spend ($)',
        'TopChurnDriver': 'Top Risk Factor',
        'Explanation': 'AI Explanation'
    }

    display_df = filtered_table[list(display_cols.keys())].rename(columns=display_cols)

    display_df['Churn Prob.'] = display_df['Churn Prob.'].apply(lambda x: f"{x:.1%}")
    display_df['CLV ($)'] = display_df['CLV ($)'].apply(lambda x: f"${x:,.2f}")
    display_df['Revenue at Risk ($)'] = display_df['Revenue at Risk ($)'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_df, use_container_width=True, height=400)
    csv = filtered_table.to_csv(index=False)
    st.download_button(
        label="Download filtered customer list",
        data=csv,
        file_name="at_risk_customers.csv",
        mime="text/csv"
    )

    # ── Row 5: Individual customer deep dive ───────────────────────
    st.markdown("---")
    st.subheader("Individual Customer Deep Dive")

    selected_customer = st.selectbox(
        "Select a customer to analyze",
        options=filtered_table.index.tolist(),
        format_func=lambda x: f"Customer {x} — "
                               f"{filtered_table.loc[x, 'RiskTier']} — "
                               f"Prob: {filtered_table.loc[x, 'ChurnProbability']:.1%}"
    )

    if selected_customer is not None:
        customer = filtered_table.loc[selected_customer]

        # ── AI Explanation — full width, standalone ─────────────
        st.markdown("**AI Explanation**")
        st.info(customer['Explanation'])

        st.markdown("---")

        # ── Risk drivers (left, wider) + Customer profile (right) ─
        col_d1, col_d2 = st.columns([2, 1])

        with col_d1:
            st.markdown("**Primary Risk Factor**")
            st.warning(f"{customer['TopChurnDriver']} — {customer['TopDriverDirection']}")

            customer_detail = api.get_customer_detail(selected_customer)
            top_shap_factors = customer_detail.get('top_shap_factors', [])

            if top_shap_factors:
                driver_df = pd.DataFrame(top_shap_factors)
                driver_df['Direction'] = driver_df['shap_value'].apply(
                    lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
                )
                driver_df = driver_df.rename(columns={'feature': 'Feature', 'shap_value': 'SHAP Value'})

                fig_driver = px.bar(
                    driver_df, x='SHAP Value', y='Feature',
                    orientation='h', color='Direction',
                    color_discrete_map={
                        'Increases Risk': '#d62728',
                        'Decreases Risk': '#2ca02c'
                    },
                    title="This Customer's Risk Drivers"
                )

                fig_driver.update_layout(
                    height=320, margin=dict(t=40, b=0),
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=True
                )

                st.plotly_chart(fig_driver, use_container_width=True)

        with col_d2:
            st.markdown("**Customer Profile**")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write(f"**Risk Tier:** {customer['RiskTier']}")
            st.write(f"**Churn Probability:** {customer['ChurnProbability']:.1%}")
            st.write(f"**Days Active:** {customer['DaysActive']:.0f}")
            st.write(f"**Total Orders:** {customer['Frequency']:.0f}")
            st.write(f"**Total Spend:** ${customer['Monetary']:,.2f}")
            st.write(f"**Estimated CLV:** ${customer['CLV']:,.2f}")
            st.write(f"**Revenue at Risk:** ${customer['RevenueAtRisk']:,.2f}")
            st.write("")
            
            
        # ── Moved out of col_d2 — now spans full width ──────────────
        st.markdown("---")
        st.subheader("AI-Generated Insights")

        col_ai1, col_ai2 = st.columns(2, gap="large")

        with col_ai1:
            if st.button("Generate live AI explanation"):
                with st.spinner("Generating explanation..."):
                    explanation_data = api.explain_churn(selected_customer)
                    st.session_state['live_explanation'] = explanation_data.get('explanation', 'Explanation unavailable')
                    st.session_state['live_explanation_customer'] = selected_customer

            if (st.session_state.get('live_explanation_customer') == selected_customer
                    and 'live_explanation' in st.session_state):
                st.success(st.session_state['live_explanation'])

        with col_ai2:
            if customer['RiskTier'] in ['⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk']:
                if st.button("Generate win-back email"):
                    with st.spinner("Generating personalized win-back email..."):
                        st.session_state['winback'] = api.generate_winback(selected_customer)
                        st.session_state['winback_customer'] = selected_customer

                if (st.session_state.get('winback_customer') == selected_customer
                        and 'winback' in st.session_state):
                    winback = st.session_state['winback']
                    st.write(f"**Favorite product:** {winback.get('favorite_product') or 'Unknown'}")
                    st.write(f"**Recommended incentive:** {winback.get('recommended_incentive', '')}")
                    st.write(f"**Urgency:** {winback.get('urgency', '')}")
                    st.markdown(f"**Subject:** {winback.get('email_subject', '')}")
                    st.text_area("Email body", value=winback.get('email_body', ''), height=150)
            else:
                st.caption("Win-back email available for Extreme, High, and Medium risk customers.")


elif selected_tab == "AI Insights":

    st.markdown('<p class="main-header">AI Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="tab-description">AI-generated summaries and reports drawn directly '
                'from your churn and demand data.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Customer Retention Summary ─────────────────────────────────
    st.subheader("Customer Retention Summary")

    if st.button("Generate churn segment summaries"):
        with st.spinner("Generating customer summaries..."):
            result = api.get_segment_summaries()
            st.session_state['segment_summaries'] = result.get('summaries', {})

    if 'segment_summaries' in st.session_state:
        for tier, summary in st.session_state['segment_summaries'].items():
            with st.expander(tier):
                st.write(strip_emoji(summary))

    st.markdown("---")

    # ── Inventory Status Briefing ───────────────────────────────────
    st.subheader("Inventory Status Briefing")

    if st.button("Generate daily inventory briefing"):
        with st.spinner("Generating inventory briefing..."):
            briefing_data = api.get_briefing()
            st.session_state['inventory_briefing'] = briefing_data.get('inventory_briefing', 'Briefing unavailable')

    if 'inventory_briefing' in st.session_state:
        st.info(st.session_state['inventory_briefing'])

    st.markdown("---")

    # ── Weekly report ─────────────────────────────────────────────
    st.subheader("Weekly Demand Report")

    if st.button("Generate this week's demand report"):
        with st.spinner("Generating weekly report..."):
            report_data = api.get_weekly_report()
            st.session_state['weekly_report'] = report_data.get('report', 'Report unavailable')

    if 'weekly_report' in st.session_state:
        st.info(st.session_state['weekly_report'])

    st.markdown("---")

    # ── Natural language query interface ─────────────────────────
    st.subheader("Ask a Question")

    example_questions = [
        "Which customers are at highest risk of leaving?",
        "What products need urgent reordering?",
        "How much revenue is at risk from churning customers?",
        "Give me an overall business health summary"
    ]

    selected_example = st.selectbox(
        "Or pick an example question:",
        options=["Type your own question below..."] + example_questions
    )

    default_question = selected_example if selected_example != "Type your own question below..." else ""

    user_question = st.text_input(
        "Your question:",
        value=default_question,
        placeholder="e.g. Which customers should I focus on retaining this week?"
    )

    if st.button("Ask") and user_question:
        with st.spinner("Analyzing your data..."):
            st.session_state['query_result'] = api.query(user_question)

    if 'query_result' in st.session_state:
        result = st.session_state['query_result']
        st.markdown("**Answer:**")
        st.success(result.get('answer', 'No answer returned'))
        st.caption(f"Data source used: {result.get('route', 'UNKNOWN')}")
    

elif selected_tab == "Demand Forecast":

    st.markdown('<p class="main-header">Demand & Inventory Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="tab-description">Forecast product demand with confidence intervals — '
                'and know exactly when to reorder before stockouts happen.</p>', unsafe_allow_html=True)
 
    
    # ── Load demand data via API ────────────────────────────────
    demand_summary = api.get_demand_summary()
    alerts = pd.DataFrame(demand_summary.get('products', []))

    backtest_data = api.get_backtest_results()
    fold_details = pd.DataFrame(backtest_data.get('fold_details', []))
    # Rename API's snake_case fold fields back to original names used below
    fold_rename = {
        'stock_code': 'StockCode', 'fold': 'Fold',
        'prophet_mae': 'Prophet_MAE', 'prophet_mape': 'Prophet_MAPE',
        'naive_mae': 'Naive_MAE', 'seasonal_mae': 'Seasonal_MAE',
        'beat_naive': 'BeatNaive', 'beat_seasonal': 'BeatSeasonal'
    }
    folds = fold_details.rename(columns=fold_rename)
    # API already nulls out non-finite MAPE values, so drop those instead
    # of filtering by DataQualityFlag=='OK' (that column isn't in the API response)
    unreliable_codes = alerts[alerts['AlertLevel'] == 'unreliable']['StockCode'].tolist()
    folds_ok = folds[~folds['StockCode'].isin(unreliable_codes)].copy()


    col1, col2, col3, col4 = st.columns(4)

    critical_count = demand_summary.get('stockout_risk_count', 0)
    warning_count = demand_summary.get('reorder_soon_count', 0)
    unreliable_count = demand_summary.get('unreliable_count', 0)
    total_products = demand_summary.get('total_products', 0)
    avg_mape = folds_ok['Prophet_MAPE'].mean() if len(folds_ok) > 0 else 0

    with col1:
        st.metric(label="Products Forecasted", value=f"{total_products}")

    with col2:
        st.metric(
            label="Stockout Risk",
            value=f"{critical_count}",
            delta="Needs immediate action" if critical_count > 0 else "All clear",
            delta_color="off"
        )

    with col3:
        st.metric(
            label="Reorder Soon",
            value=f"{warning_count}",
            delta="Order within 2 weeks" if warning_count > 0 else "All clear",
            delta_color="off"
        )

    with col4:
        st.metric(
            label="Avg Forecast MAPE",
            value=f"{avg_mape:.1f}%",
            delta=f"Walk-forward, {unreliable_count} products excluded",
            delta_color="off"
        )

    st.markdown("---")
    
    st.subheader("Demand Overview")

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        status_counts = alerts['AlertStatus'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        fig_pie = px.pie(
            status_counts, names='Status', values='Count',
            title='Alert Status Distribution',
            color='Status',
            color_discrete_map={
                '🔴 Stockout Risk': '#d62728',
                '🟡 Reorder Soon': '#ff7f0e',
                '🟢 Adequate': '#2ca02c',
                '⚪ Unreliable Forecast': '#7f7f7f'
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent')
        fig_pie.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_c2:
        reorder_products = alerts[
            (alerts['SuggestedReorderQty'] > 0) & (alerts['AlertLevel'] != 'unreliable')
        ].sort_values('SuggestedReorderQty', ascending=False)

        if len(reorder_products) > 0:
            reorder_products = reorder_products.copy()
            reorder_products['StockCode'] = reorder_products['StockCode'].astype(str)
            fig_reorder = px.bar(
                reorder_products.head(10),
                x='StockCode', y='SuggestedReorderQty',
                title='Suggested Reorder Quantities',
                color='AlertStatus',
                color_discrete_map={
                    '🔴 Stockout Risk': '#d62728',
                    '🟡 Reorder Soon': '#ff7f0e'
                },
                labels={'SuggestedReorderQty': 'Units to Reorder'}
            )
            fig_reorder.update_xaxes(type='category')
            fig_reorder.update_layout(height=300, margin=dict(t=40, b=0), showlegend=False)
            st.plotly_chart(fig_reorder, use_container_width=True)
        else:
            st.info("No reorders currently needed")

    with col_c3:
        fig_mape = px.histogram(
            folds_ok, x='Prophet_MAPE', nbins=20,
            title='Forecast Error Distribution (MAPE %)',
            labels={'Prophet_MAPE': 'MAPE (%)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_mape.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_mape, use_container_width=True)

    st.markdown("---")

    st.subheader("Inventory Alerts")

    alert_filter = st.multiselect(
        "Filter by Alert Status",
        options=['🔴 Stockout Risk', '🟡 Reorder Soon',
                 '🟢 Adequate', '⚪ Unreliable Forecast'],
        default=['🔴 Stockout Risk', '🟡 Reorder Soon']
    )

    filtered_alerts = alerts[alerts['AlertStatus'].isin(alert_filter)]

    alert_display_cols = {
        'AlertStatus': 'Status',
        'StockCode': 'Product Code',
        'Description': 'Product Name',
        'EstimatedCurrentStock': 'Current Stock',
        'ForecastedDemand_Reorder': 'Forecasted (3wk)',
        'ReorderPoint': 'Reorder Point',
        'SuggestedReorderQty': 'Reorder Qty',
        'AlertMessage': 'Action Required'
    }

    display_alerts = filtered_alerts[list(alert_display_cols.keys())].rename(
        columns=alert_display_cols
    )

    for col in ['Current Stock', 'Forecasted (3wk)', 'Reorder Point', 'Reorder Qty']:
        display_alerts[col] = display_alerts[col].apply(lambda x: f"{x:.0f}")

    st.caption("Note: Current Stock is a synthetic estimate — no real inventory "
               "data exists in this dataset.")

    st.dataframe(display_alerts, use_container_width=True, height=350)

    csv = filtered_alerts.to_csv(index=False)
    st.download_button(
        label="Download alert list",
        data=csv,
        file_name="inventory_alerts.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.subheader("Product Forecast Deep Dive")

    products = alerts['StockCode'].tolist()
    product_descriptions = dict(zip(alerts['StockCode'], alerts['Description']))

    selected_product = st.selectbox(
        "Select a product to view forecast",
        options=products,
        format_func=lambda x: f"{x} — {str(product_descriptions.get(x, ''))[:40]}"
    )

    if selected_product:
        product_detail = api.get_product_detail(selected_product)

        historical = pd.DataFrame(product_detail.get('historical', []))
        forecast = pd.DataFrame(product_detail.get('forecast', []))

        if len(historical) > 0:
            historical['date'] = pd.to_datetime(historical['date'])
        if len(forecast) > 0:
            forecast['date'] = pd.to_datetime(forecast['date'])

        is_unreliable = product_detail.get('alert_level') == 'unreliable'

        if is_unreliable:
            st.warning(f"This product is flagged as low-reliability — "
                       f"forecast shown for reference only, "
                       f"not recommended for automated reorder decisions.")

        # ── Forecast chart — full width ──────────────────────────
        fig = go.Figure()

        if len(historical) > 0:
            fig.add_trace(go.Scatter(
                x=historical['date'], y=historical['actual'],
                mode='lines', name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))

        if len(forecast) > 0:
            fig.add_trace(go.Scatter(
                x=forecast['date'], y=forecast['forecast'],
                mode='lines', name='Forecast',
                line=dict(color='#d62728', width=2, dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['date'], forecast['date'][::-1]]),
                y=pd.concat([forecast['upper'], forecast['lower'][::-1]]),
                fill='toself', fillcolor='rgba(214,39,40,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))

        if len(historical) > 0:
            last_historical = historical['date'].max()
            fig.add_vline(x=last_historical, line_dash='dot', line_color='gray',
                         annotation_text='Forecast start')

        fig.update_layout(
            title=f"Demand Forecast: {selected_product}",
            xaxis_title='Week', yaxis_title='Units Sold', height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("**Product Summary**")
            st.write(f"**Status:** {product_detail.get('alert_status', 'Unknown')}")

            if not is_unreliable:
                st.write(f"**Est. Stock:** {product_detail.get('current_stock', 0):.0f} units")
                st.write(f"**Lead-time Forecast:** {product_detail.get('forecasted_demand_lead', 0):.0f} units")
                st.write(f"**3-wk (Reorder-cycle) Forecast:** {product_detail.get('forecasted_demand_reorder', 0):.0f} units")
                st.write(f"**Safety Stock Buffer:** {product_detail.get('safety_stock', 0):.0f} units")
                st.write(f"**Reorder Point:** {product_detail.get('reorder_point', 0):.0f} units "
                          f"*(= lead-time forecast + safety stock)*")

                if product_detail.get('suggested_reorder_qty', 0) > 0:
                    st.warning(f"**Suggested Reorder:** {product_detail.get('suggested_reorder_qty', 0):.0f} units "
                                f"*(= 3-wk forecast + safety stock − current stock)*")
                    
        with col_p2:
            st.markdown("**Forecast Statistics**")
            st.write("Next 8 weeks:")
            if len(forecast) > 0:
                st.write(f"- Min forecast: {forecast['forecast'].min():.0f}")
                st.write(f"- Max forecast: {forecast['forecast'].max():.0f}")
                st.write(f"- Total expected: {forecast['forecast'].sum():.0f}")
            else:
                st.write("- No forecast data available")
                
            st.write("")   
            st.write("") 
            st.markdown("**Action Required:**")
            st.info(product_detail.get('alert_message', ''))
                
            
    st.markdown("---")
    st.subheader("Model Validation: Walk-Forward Backtesting")

    st.caption(
        "5-fold walk-forward validation — each fold trains on past data "
        "only and tests on the next 4 weeks. This mirrors real-world "
        "deployment where the model is retrained weekly."
    )

    product_summary_data = backtest_data.get('product_summary', [])
    wf_summary = pd.DataFrame(product_summary_data)
    wf_rename = {
        'stock_code': 'StockCode',
        'win_rate_vs_naive': 'WinRateVsNaive',
        'win_rate_vs_seasonal': 'WinRateVsSeasonal'
    }
    wf_summary = wf_summary.rename(columns=wf_rename)

    col_wf1, col_wf2 = st.columns(2)

    with col_wf1:
        wf_summary_sorted = wf_summary.sort_values('WinRateVsNaive', ascending=False).copy()
        wf_summary_sorted['StockCode'] = wf_summary_sorted['StockCode'].astype(str)
        fig_win = px.bar(
            wf_summary_sorted,
            x='StockCode', y='WinRateVsNaive',
            title='Prophet Win Rate vs Naive Baseline',
            labels={'WinRateVsNaive': 'Win Rate (% of folds)', 'StockCode': 'Product'},
            color='WinRateVsNaive', color_continuous_scale='RdYlGn'
        )
        fig_win.update_xaxes(type='category')
        fig_win.add_hline(y=0.5, line_dash='dash', line_color='gray', annotation_text='50%')
        fig_win.update_layout(
            height=450, coloraxis_showscale=False,
            yaxis=dict(tickformat='.0%', dtick=0.1, range=[0, 0.9]),
            xaxis_tickangle=-45, bargap=0.1
        )
        st.plotly_chart(fig_win, use_container_width=True)
        
        
    with col_wf2:
        mae_comparison = pd.DataFrame({
            'Model': ['Prophet', 'Naive Baseline', 'Seasonal Naive'],
            'Mean MAE': [
                folds_ok['Prophet_MAE'].mean(),
                folds_ok['Naive_MAE'].mean(),
                folds_ok['Seasonal_MAE'].mean()
            ]
        })

        fig_mae = px.bar(
            mae_comparison, x='Model', y='Mean MAE',
            title='Average MAE: Prophet vs Baselines',
            color='Model',
            color_discrete_map={
                'Prophet': '#1f77b4',
                'Naive Baseline': '#d62728',
                'Seasonal Naive': '#ff7f0e'
            },
            text='Mean MAE'
        )
        fig_mae.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_mae.update_layout(
            height=450, showlegend=False,
            yaxis=dict(title='Mean Absolute Error (units)', dtick=100, range=[0, 600]),
            bargap=0.5
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    col_b1, col_b2, col_b3 = st.columns(3)

    overall_win_rate = folds_ok['BeatNaive'].mean()
    overall_seasonal_win = folds_ok['BeatSeasonal'].mean()
    median_mape = folds_ok['Prophet_MAPE'].median()

    with col_b1:
        st.metric("Win Rate vs Naive", f"{overall_win_rate:.0%}", delta="of all folds", delta_color='off')
    with col_b2:
        st.metric("Win Rate vs Seasonal", f"{overall_seasonal_win:.0%}", delta="of all folds", delta_color='off')
    with col_b3:
        st.metric("Median MAPE", f"{median_mape:.1f}%", delta="across all products and folds", delta_color='off')
    
