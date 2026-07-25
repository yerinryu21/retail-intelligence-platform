import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

def prepare_prophet_data(demand_df: pd.DataFrame, 
                          stock_code: str) -> pd.DataFrame:
    """
    Prepare a single product's data for Prophet.
    Prophet requires exactly two columns: ds (date) and y (value).
    """
    product_df = demand_df[demand_df['StockCode'] == stock_code].copy()
    product_df = product_df.sort_values('Week')
    
    prophet_df = product_df[['Week', 'TotalQuantity']].rename(
        columns={'Week': 'ds', 'TotalQuantity': 'y'}
    )
    
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df[prophet_df['y'] >= 0]
    
    print(f"Product {stock_code}:")
    print(f"  Weeks of data: {len(prophet_df)}")
    print(f"  Date range: {prophet_df['ds'].min().date()} "
          f"to {prophet_df['ds'].max().date()}")
    print(f"  Mean weekly demand: {prophet_df['y'].mean():.1f} units")
    print(f"  Max weekly demand: {prophet_df['y'].max():.0f} units")
    print(f"  Zero-demand weeks: {(prophet_df['y']==0).sum()}")
    
    return prophet_df

def train_prophet_model(prophet_df: pd.DataFrame,
                        product_code: str,
                        forecast_weeks: int = 8) -> tuple:
    """
    Train a Prophet model and generate forecast.
    
    NOTE: yearly_seasonality, weekly_seasonality, and UK holidays are all
    disabled. Reasons:
    - yearly_seasonality: only ~1 year of history, cannot distinguish a
      real annual pattern from a one-time event
    - UK holidays: most holidays occur exactly once in the training data,
      so effects are single-occurrence estimates, not real patterns
    - weekly_seasonality: every training date falls on the same day of
      week (Monday), confirmed via demand['Week'].dt.day_name().unique().
      Prophet fit a full day-of-week curve with zero real variation to
      constrain it, and forecast dates (which default to Sunday-ending
      via make_future_dataframe freq='W') landed on days the model had
      never seen — causing forecasts to crash toward the curve's
      unconstrained minimum (e.g. 84077: historical mean ~910, forecast
      ~100). Only trend is left, since it's the only component supported
      by weekly-aggregated, single-weekday data.
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        growth='linear',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    print(f"\nFitting Prophet model for {product_code}...")
    model.fit(prophet_df)
    print("Model fitted")
    
    future = model.make_future_dataframe(
        periods=forecast_weeks,
        freq='W-MON'  # match your data's weekly cadence (all training dates are Mondays)
    )
    
    print(f"Forecasting {forecast_weeks} weeks ahead...")
    forecast = model.predict(future)
    
    print(f"Forecast generated")
    print(f"\nForecast for next {forecast_weeks} weeks:")
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
    print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))
    
    return model, forecast, prophet_df

def plot_forecast(model, 
                  forecast: pd.DataFrame, 
                  prophet_df: pd.DataFrame,
                  product_code: str,
                  save_dir: str = 'notebooks/demand_plots') -> None:
    """Plot Prophet forecast with components"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig1 = model.plot(forecast, figsize=(14, 6))
    plt.title(f"Demand Forecast: {product_code}", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{product_code}_forecast.png', dpi=150)
    plt.close()
    print(f"Saved: {product_code}_forecast.png")
    
    fig2 = model.plot_components(forecast, figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{product_code}_components.png', dpi=150)
    plt.close()
    print(f"Saved: {product_code}_components.png")
    
def run_single_product_forecast():
    """Run complete Prophet forecast for one product"""
    
    demand = pd.read_csv('data/processed/weekly_demand.csv')
    demand['Week'] = pd.to_datetime(demand['Week'])
    
    product_completeness = (
        demand.groupby('StockCode')['TotalQuantity']
        .apply(lambda x: (x > 0).mean())
        .sort_values(ascending=False)
    )
    best_product = product_completeness.index[0]
    
    print(f"Running forecast for: {best_product}")
    print(f"Description: {demand[demand['StockCode']==best_product]['Description'].iloc[0]}")
    
    prophet_df = prepare_prophet_data(demand, best_product)
    
    model, forecast, prophet_df = train_prophet_model(
        prophet_df, best_product, forecast_weeks=8
    )
    
    plot_forecast(model, forecast, prophet_df, best_product)
    
    print(f"\nSingle product forecast complete")
    return model, forecast, prophet_df, best_product

if __name__ == "__main__":
    model, forecast, prophet_df, product_code = run_single_product_forecast()