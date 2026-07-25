import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

def calculate_uncertainty_metrics(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate uncertainty metrics per product.
    
    NOTE: RelativeUncertainty is calculated against each product's historical
    mean demand, not the point forecast (yhat). yhat can be exactly 0 after
    clipping negative predictions (see forecast_pipeline.py), which would
    otherwise make RelativeUncertainty silently report 0% for exactly the
    products with the most unstable forecasts (e.g. product 23166).
    """
    
    historical = forecasts_df[forecasts_df['IsFuture'] == False].copy()
    future = forecasts_df[forecasts_df['IsFuture'] == True].copy()
    
    historical_means = historical.groupby('StockCode')['yhat'].mean()
    historical_means = historical_means.rename('HistoricalMeanDemand')
    
    future['IntervalWidth'] = future['yhat_upper'] - future['yhat_lower']
    future = future.merge(historical_means, on='StockCode', how='left')
    
    future['RelativeUncertainty'] = future.apply(
        lambda x: x['IntervalWidth'] / x['HistoricalMeanDemand']
        if x['HistoricalMeanDemand'] > 0 else np.nan,
        axis=1
    )
    
    uncertainty_summary = (
        future.groupby(['StockCode', 'Description', 'DataQualityFlag'])
        .agg(
            MeanForecast=('yhat', 'mean'),
            MeanIntervalWidth=('IntervalWidth', 'mean'),
            MeanRelativeUncertainty=('RelativeUncertainty', 'mean'),
            MinForecast=('yhat_lower', 'min'),
            MaxForecast=('yhat_upper', 'max')
        )
        .reset_index()
        .sort_values('MeanRelativeUncertainty', ascending=False)
    )
    
    uncertainty_summary['MeanRelativeUncertainty'] = (
        uncertainty_summary['MeanRelativeUncertainty'] * 100
    ).round(1)
    
    uncertainty_summary['MeanForecast'] = uncertainty_summary['MeanForecast'].round(1)
    uncertainty_summary['MeanIntervalWidth'] = uncertainty_summary['MeanIntervalWidth'].round(1)
    
    print("Forecast Uncertainty by Product:")
    print(f"{'StockCode':<12} {'Flag':<12} {'Mean Forecast':<15} "
          f"{'Interval Width':<18} {'Relative Uncertainty'}")
    print("-" * 80)
    for _, row in uncertainty_summary.iterrows():
        flag_short = 'LOW-REL' if 'Low' in row['DataQualityFlag'] else 'OK'
        print(f"{row['StockCode']:<12} {flag_short:<12} {row['MeanForecast']:<15.1f} "
              f"{row['MeanIntervalWidth']:<18.1f} {row['MeanRelativeUncertainty']:.1f}%")
    
    return uncertainty_summary

def plot_uncertainty_comparison(forecasts_df: pd.DataFrame,
                                 n_products: int = 6,
                                 save_dir: str = 'notebooks/demand_plots') -> None:
    """
    Plot side-by-side forecast comparison for multiple products
    showing historical data, point forecast, and confidence intervals.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    products = forecasts_df['StockCode'].unique()[:n_products]
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            forecasts_df[forecasts_df['StockCode']==p]['Description'].iloc[0][:35]
            for p in products
        ]
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
    
    for (row, col), product in zip(positions, products):
        product_data = forecasts_df[forecasts_df['StockCode'] == product].copy()
        product_data['ds'] = pd.to_datetime(product_data['ds'])
        
        historical = product_data[~product_data['IsFuture']]
        future = product_data[product_data['IsFuture']]
        
        fig.add_trace(
            go.Scatter(
                x=historical['ds'], y=historical['yhat'],
                mode='lines', name='Historical',
                line=dict(color='#1f77b4', width=1.5),
                showlegend=(row==1 and col==1)
            ), row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=future['ds'], y=future['yhat'],
                mode='lines', name='Forecast',
                line=dict(color='#d62728', width=2, dash='dash'),
                showlegend=(row==1 and col==1)
            ), row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=pd.concat([future['ds'], future['ds'][::-1]]),
                y=pd.concat([future['yhat_upper'], 
                             future['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(214,39,40,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=(row==1 and col==1)
            ), row=row, col=col
        )
        
        last_historical_date = historical['ds'].max()
        fig.add_vline(
            x=last_historical_date,
            line_dash="dot",
            line_color="gray",
            row=row, col=col
        )
    
    fig.update_layout(
        height=900,
        title_text="Multi-Product Demand Forecast with 95% Confidence Intervals",
        title_font_size=16
    )
    
    fig.write_html(f'{save_dir}/multi_product_forecast.html')
    print(f"Saved interactive plot: {save_dir}/multi_product_forecast.html")
    fig.show()

def calculate_inventory_risk(forecasts_df: pd.DataFrame,
                              current_stock: dict = None) -> pd.DataFrame:
    """
    Calculate inventory risk for each product based on forecast uncertainty.
    
    NOTE: products flagged as low-reliability (see DataQualityFlag) get an
    explicit 'Unreliable Forecast' status instead of a computed risk level.
    Their yhat is often clipped to exactly 0 (see forecast_pipeline.py),
    which would otherwise compute as 'Adequate' - the opposite of the truth.
    """
    
    future = forecasts_df[forecasts_df['IsFuture'] == True].copy()
    
    next_4_weeks = future.groupby('StockCode').head(4)
    
    demand_summary = next_4_weeks.groupby('StockCode').agg(
        ForecastedDemand=('yhat', 'sum'),
        DemandLower=('yhat_lower', 'sum'),
        DemandUpper=('yhat_upper', 'sum')
    ).reset_index()
    
        
    if current_stock is None:
        historical = forecasts_df[~forecasts_df['IsFuture']]
        weekly_mean = historical.groupby('StockCode')['yhat'].mean()
        # Match the 4-week forecast horizon used above, rather than 3 -
        # using a shorter stock assumption than the demand window
        # structurally guarantees most products look understocked
        demand_summary['CurrentStock'] = demand_summary['StockCode'].map(
            weekly_mean * 4
        )
    else:
        demand_summary['CurrentStock'] = demand_summary['StockCode'].map(current_stock)
    
    descriptions = (
        forecasts_df[['StockCode', 'Description', 'DataQualityFlag']]
        .drop_duplicates()
    )
    demand_summary = demand_summary.merge(descriptions, on='StockCode')
    
    def assess_risk(row):
        if row['DataQualityFlag'] != 'OK':
            return 'Unreliable Forecast', 'Data too sparse/spike-driven for a trustworthy demand estimate'
        
        stock = row['CurrentStock']
        demand_low = row['DemandLower']
        demand_mid = row['ForecastedDemand']
        demand_high = row['DemandUpper']
        
        if stock < demand_low:
            return 'Stockout Risk', 'High probability of running out'
        elif stock < demand_mid:
            return 'Low Stock', 'May run short if demand is average'
        elif stock > demand_high * 2:
            return 'Overstock Risk', 'Excess inventory - cash flow impact'
        else:
            return 'Adequate', 'Stock level appropriate for forecast'
    
    risk_results = demand_summary.apply(
        lambda row: pd.Series(assess_risk(row), 
                              index=['RiskStatus', 'RiskReason']),
        axis=1
    )
    
    demand_summary = pd.concat([demand_summary, risk_results], axis=1)
    demand_summary = demand_summary.sort_values('ForecastedDemand', ascending=False)
    
    print("\nInventory Risk Assessment (Next 4 Weeks):")
    print(f"{'Product':<10} {'Flag':<10} {'Current Stock':<15} {'Forecast':<12} "
          f"{'Range':<20} {'Status'}")
    print("-" * 90)
    
    for _, row in demand_summary.iterrows():
        flag_short = 'LOW-REL' if 'Low' in row['DataQualityFlag'] else 'OK'
        print(
            f"{row['StockCode']:<10} "
            f"{flag_short:<10} "
            f"{row['CurrentStock']:<15.0f} "
            f"{row['ForecastedDemand']:<12.0f} "
            f"{row['DemandLower']:.0f}-{row['DemandUpper']:.0f}{'':>10} "
            f"{row['RiskStatus']}"
        )
    
    demand_summary.to_csv('data/processed/inventory_risk.csv', index=False)
    print("\nSaved to data/processed/inventory_risk.csv")
    
    return demand_summary

if __name__ == "__main__":
    forecasts = pd.read_csv('data/processed/all_forecasts.csv')
    forecasts['ds'] = pd.to_datetime(forecasts['ds'])
    
    uncertainty = calculate_uncertainty_metrics(forecasts)
    uncertainty.to_csv('data/processed/uncertainty_metrics.csv', index=False)
    
    plot_uncertainty_comparison(forecasts)
    
    inventory_risk = calculate_inventory_risk(forecasts)
    
    print("\nUncertainty analysis complete")