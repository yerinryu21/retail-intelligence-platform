import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

def calculate_safety_stock(forecast_std: float,
                            lead_time_weeks: int = 2,
                            service_level: float = 0.95) -> float:
    """
    Safety stock = Z * sigma * sqrt(lead_time)
    """
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(service_level, 1.645)
    safety_stock = z * forecast_std * np.sqrt(lead_time_weeks)
    return max(0, safety_stock)


def build_alert_system(forecasts_df: pd.DataFrame,
                        fold_results: pd.DataFrame,
                        lead_time_weeks: int = 2,
                        reorder_point_weeks: int = 3) -> pd.DataFrame:
    """
    Builds the inventory alert table.

    NOTE: CurrentStock is synthetic throughout -- no real inventory data
    exists in this dataset (documented Week 4 limitation). Any alert here
    inherits that limitation and is illustrative, not a real stock feed.

    Products with DataQualityFlag != 'OK' are labeled "Unreliable Forecast"
    regardless of computed thresholds, matching Week 4's Day 3 fix.
    """

    future = forecasts_df[forecasts_df['IsFuture'] == True].copy()
    future['ds'] = pd.to_datetime(future['ds'])

    error_std = fold_results.groupby('StockCode')['Prophet_MAE'].std().fillna(
        fold_results['Prophet_MAE'].std()
    )

    next_reorder_weeks = future.groupby('StockCode').head(reorder_point_weeks)
    next_lead_weeks = future.groupby('StockCode').head(lead_time_weeks)

    demand_reorder = next_reorder_weeks.groupby('StockCode').agg(
        ForecastedDemand_Reorder=('yhat', 'sum'),
        DemandLower_Reorder=('yhat_lower', 'sum'),
        DemandUpper_Reorder=('yhat_upper', 'sum'),
        NextForecastDate=('ds', 'min')
    ).reset_index()

    demand_lead = next_lead_weeks.groupby('StockCode').agg(
        ForecastedDemand_Lead=('yhat', 'sum'),
        DemandUpper_Lead=('yhat_upper', 'sum')
    ).reset_index()

    alerts = demand_reorder.merge(demand_lead, on='StockCode')

    descriptions = forecasts_df[['StockCode', 'Description']].drop_duplicates()
    alerts = alerts.merge(descriptions, on='StockCode')

    # Carry DataQualityFlag through -- checked before any computed status
    flags = forecasts_df[['StockCode', 'DataQualityFlag']].drop_duplicates()
    alerts = alerts.merge(flags, on='StockCode')

    # CurrentStock: synthetic proxy (historical weekly mean x 3 weeks)
    historical = forecasts_df[~forecasts_df['IsFuture']]
    weekly_mean = historical.groupby('StockCode')['yhat'].mean()
    alerts['EstimatedCurrentStock'] = alerts['StockCode'].map(weekly_mean * 3)

    alerts['ForecastErrorStd'] = alerts['StockCode'].map(error_std).fillna(
        error_std.mean()
    )
    alerts['SafetyStock'] = alerts['ForecastErrorStd'].apply(
        lambda x: calculate_safety_stock(x, lead_time_weeks)
    )

    alerts['ReorderPoint'] = (
        alerts['ForecastedDemand_Lead'] + alerts['SafetyStock']
    )

    alerts['SuggestedReorderQty'] = (
        alerts['ForecastedDemand_Reorder'] +
        alerts['SafetyStock'] -
        alerts['EstimatedCurrentStock']
    ).clip(lower=0).round(0)

    def assign_alert(row):
        # Override first: low-reliability products get an explicit,
        # honest label instead of a computed (and likely meaningless) status.
        if row['DataQualityFlag'] != 'OK':
            return ('⚪ Unreliable Forecast',
                    f"Sparse/spike-driven demand pattern -- forecast not "
                    f"reliable enough for automated reorder decisions. "
                    f"Manual review recommended.",
                    'unreliable')

        stock = row['EstimatedCurrentStock']
        reorder_pt = row['ReorderPoint']
        demand_reorder = row['ForecastedDemand_Reorder']
        demand_upper = row['DemandUpper_Reorder']

        if stock < reorder_pt:
            return ('🔴 Stockout Risk',
                    f"Stock ({stock:.0f}) below reorder point ({reorder_pt:.0f}). "
                    f"Order {row['SuggestedReorderQty']:.0f} units immediately.",
                    'critical')
        elif stock < demand_reorder:
            return ('🟡 Reorder Soon',
                    f"Stock ({stock:.0f}) below {reorder_point_weeks}-week "
                    f"forecast ({demand_reorder:.0f}). "
                    f"Order {row['SuggestedReorderQty']:.0f} units within "
                    f"{lead_time_weeks} weeks.",
                    'warning')
        elif stock > demand_upper * 2:
            overstock = stock - demand_upper
            return ('🟣 Overstock',
                    f"Stock ({stock:.0f}) exceeds 2x maximum forecast "
                    f"({demand_upper:.0f}). Consider promotions or returns. "
                    f"Excess: ~{overstock:.0f} units.",
                    'info')
        else:
            return ('🟢 Adequate',
                    f"Stock ({stock:.0f}) adequate for forecasted demand "
                    f"({demand_reorder:.0f} units over {reorder_point_weeks} weeks).",
                    'ok')

    alert_results = alerts.apply(
        lambda row: pd.Series(assign_alert(row),
                              index=['AlertStatus', 'AlertMessage', 'AlertLevel']),
        axis=1
    )

    alerts = pd.concat([alerts, alert_results], axis=1)

    urgency_order = {'critical': 0, 'warning': 1, 'info': 2, 'ok': 3, 'unreliable': 4}
    alerts['UrgencyRank'] = alerts['AlertLevel'].map(urgency_order)
    alerts = alerts.sort_values('UrgencyRank')

    print("\n📦 Inventory Alert Summary:")
    print(f"{'='*60}")
    print("NOTE: EstimatedCurrentStock is synthetic -- no real inventory data exists.")

    for status in ['🔴 Stockout Risk', '🟡 Reorder Soon',
                    '🟣 Overstock', '🟢 Adequate', '⚪ Unreliable Forecast']:
        count = (alerts['AlertStatus'] == status).sum()
        print(f"  {status}: {count} products")

    print(f"\nDetailed alerts (critical/warning only):")
    for _, row in alerts[alerts['AlertLevel'].isin(['critical', 'warning'])].iterrows():
        print(f"\n  {row['AlertStatus']} — {row['StockCode']}")
        print(f"  {row['Description'][:50]}")
        print(f"  → {row['AlertMessage']}")

    alerts.to_csv('data/processed/inventory_alerts.csv', index=False)
    print(f"\n✅ Alerts saved to data/processed/inventory_alerts.csv")

    return alerts

def plot_alert_dashboard(alerts: pd.DataFrame,
                          save_dir: str = 'notebooks/demand_plots') -> None:
    """Alert visualizations. Unreliable-forecast products included in the
    distribution pie (for a true picture) but excluded from quantity/stock
    charts, since their numbers aren't meaningful."""

    os.makedirs(save_dir, exist_ok=True)

    # ── Plot 1: Alert status distribution (all products, including unreliable) ──
    status_counts = alerts['AlertStatus'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']

    fig1 = px.pie(
        status_counts, names='Status', values='Count',
        title='Inventory Alert Status Distribution',
        color='Status',
        color_discrete_map={
            '🔴 Stockout Risk': '#d62728',
            '🟡 Reorder Soon': '#ff7f0e',
            '🟣 Overstock': '#9467bd',
            '🟢 Adequate': '#2ca02c',
            '⚪ Unreliable Forecast': '#7f7f7f'
        },
        hole=0.4
    )
    fig1.update_layout(height=400)
    fig1.write_html(f'{save_dir}/alert_distribution.html')
    fig1.show()

    # ── Plot 2: Stock vs demand vs reorder point (OK products only) ──
    ok_alerts = alerts[alerts['AlertLevel'] != 'unreliable'].copy()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name='Estimated Current Stock', x=ok_alerts['StockCode'],
        y=ok_alerts['EstimatedCurrentStock'], marker_color='#1f77b4'
    ))
    fig2.add_trace(go.Bar(
        name='Forecasted Demand (3wk)', x=ok_alerts['StockCode'],
        y=ok_alerts['ForecastedDemand_Reorder'], marker_color='#ff7f0e'
    ))
    fig2.add_trace(go.Scatter(
        name='Reorder Point', x=ok_alerts['StockCode'],
        y=ok_alerts['ReorderPoint'], mode='markers',
        marker=dict(symbol='line-ew', size=20, color='red', line=dict(width=3, color='red'))
    ))
    fig2.update_layout(
        barmode='group',
        title='Current Stock vs Forecasted Demand vs Reorder Point (OK products only)',
        xaxis_title='Product', yaxis_title='Units', height=500
    )
    fig2.write_html(f'{save_dir}/stock_vs_demand.html')
    fig2.show()

    # ── Plot 3: Suggested reorder quantities ──
    reorder_needed = ok_alerts[ok_alerts['SuggestedReorderQty'] > 0].copy()

    if len(reorder_needed) > 0:
        fig3 = px.bar(
            reorder_needed.sort_values('SuggestedReorderQty', ascending=False),
            x='StockCode', y='SuggestedReorderQty',
            title='Suggested Reorder Quantities by Product',
            color='AlertStatus',
            color_discrete_map={'🔴 Stockout Risk': '#d62728', '🟡 Reorder Soon': '#ff7f0e'},
            labels={'SuggestedReorderQty': 'Units to Reorder'}
        )
        fig3.update_layout(height=400)
        fig3.write_html(f'{save_dir}/reorder_quantities.html')
        fig3.show()

    print("✅ Alert visualizations saved")


if __name__ == "__main__":
    forecasts = pd.read_csv('data/processed/all_forecasts.csv')
    forecasts['ds'] = pd.to_datetime(forecasts['ds'])

    fold_results = pd.read_csv('data/processed/walkforward_folds.csv')

    alerts = build_alert_system(forecasts, fold_results)
    plot_alert_dashboard(alerts)

    print(f"\n✅ Alert system complete")
    print(f"Total products: {len(alerts)}")
    print(f"Critical alerts: {(alerts['AlertLevel']=='critical').sum()}")
    print(f"Warning alerts: {(alerts['AlertLevel']=='warning').sum()}")
    print(f"Unreliable forecasts: {(alerts['AlertLevel']=='unreliable').sum()}")