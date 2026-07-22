import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
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

# ── Load data ──────────────────────────────────────────────────────
@st.cache_data
def load_risk_table():
    return pd.read_csv('data/processed/customer_risk_table.csv')

@st.cache_data
def load_feature_importance():
    return pd.read_csv('data/processed/feature_importance.csv')

@st.cache_data
def load_shap_values():
    return pd.read_csv('data/processed/shap_values.csv')

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

# ══════════════════════════════════════════════════════════════════
# TAB 1: CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════

if selected_tab == "Churn Prediction":

    st.markdown('<p class="main-header">Customer Churn Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown("Identify at-risk customers before they leave — "
                "with SHAP-explained reasons and revenue impact.")
    st.markdown("---")

    # Load data
    risk_table = load_risk_table()
    feature_importance = load_feature_importance()
    shap_values_df = load_shap_values()

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
            delta=f"{len(extreme_risk)/total_customers:.1%} of base"
        )

    with col3:
        st.metric(
            label="High Risk",
            value=f"{len(high_risk):,}",
            delta=f"{len(high_risk)/total_customers:.1%} of base"
        )

    with col4:
        st.metric(
            label="Medium Risk",
            value=f"{len(medium_risk):,}",
            delta=f"{len(medium_risk)/total_customers:.1%} of base"
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

        col_d1, col_d2 = st.columns([1, 2])

        with col_d1:
            st.markdown("**Customer Profile**")
            st.write(f"**Risk Tier:** {customer['RiskTier']}")
            st.write(f"**Churn Probability:** {customer['ChurnProbability']:.1%}")
            st.write(f"**Days Active:** {customer['DaysActive']:.0f}")
            st.write(f"**Total Orders:** {customer['Frequency']:.0f}")
            st.write(f"**Total Spend:** ${customer['Monetary']:,.2f}")
            st.write(f"**Estimated CLV:** ${customer['CLV']:,.2f}")
            st.write(f"**Revenue at Risk:** ${customer['RevenueAtRisk']:,.2f}")

        with col_d2:
            st.markdown("**AI Explanation**")
            st.info(customer['Explanation'])
            st.markdown("**Primary Risk Factor**")
            st.warning(f"{customer['TopChurnDriver']} — {customer['TopDriverDirection']}")
            
            
    # Top drivers bar chart for this specific customer
            customer_shap_row = shap_values_df.iloc[customer['ShapRowIndex']]

            driver_df = pd.DataFrame({
                'Feature': [col.replace('shap_', '') for col in shap_values_df.columns],
                'SHAP Value': customer_shap_row.values
            }).sort_values('SHAP Value', key=abs, ascending=False).head(6)

            driver_df['Direction'] = driver_df['SHAP Value'].apply(
                lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
            )

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
                height=300, margin=dict(t=40, b=0),
                yaxis={'categoryorder': 'total ascending'},
                showlegend=True
            )
            st.plotly_chart(fig_driver, use_container_width=True)

elif selected_tab == "Demand Forecast":
    st.markdown('<p class="main-header">Demand Forecasting</p>', unsafe_allow_html=True)
    st.info("Coming in Week 5 — Demand forecasting module under construction")

elif selected_tab == "AI Insights":
    st.markdown('<p class="main-header">AI Insights</p>', unsafe_allow_html=True)
    st.info("Coming in Week 6 — LLM narration layer under construction")