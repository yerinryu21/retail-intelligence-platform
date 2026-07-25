import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import os
import warnings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
warnings.filterwarnings('ignore')

from src.module2_demand.prophet_model import prepare_prophet_data, train_prophet_model

def assess_forecastability(prophet_df: pd.DataFrame) -> dict:
    """
    Flag products where the demand pattern is too sparse/spiky for
    Prophet's trend + weekly seasonality to produce a meaningful forecast.

    A product dominated by one or two huge one-time orders (common in this
    dataset from wholesale/reseller buyers) has no repeating pattern to
    learn from - Prophet will still produce a number, but it reflects the
    shape of one outlier event, not expected future demand.
    """
    zero_pct = (prophet_df['y'] == 0).mean()
    mean_val = prophet_df['y'].mean()
    max_val = prophet_df['y'].max()
    spike_ratio = max_val / mean_val if mean_val > 0 else 0

    
    is_low_reliability = (zero_pct > 0.40) and (spike_ratio > 15)

    return {
        'ZeroWeekPct': round(zero_pct * 100, 1),
        'SpikeRatio': round(spike_ratio, 1),
        'DataQualityFlag': 'Low reliability - sparse/spike-driven' if is_low_reliability else 'OK'
    }

def run_multi_product_forecast(demand_df: pd.DataFrame,
                                forecast_weeks: int = 8,
                                save_models: bool = True) -> dict:
    """
    Run Prophet forecast for all top products.
    
    Returns: dictionary of {stock_code: {model, forecast, prophet_df}}
    """
    
    products = demand_df['StockCode'].unique()
    print(f"Running forecasts for {len(products)} products...")
    print(f"Forecast horizon: {forecast_weeks} weeks\n")
    
    results = {}
    failed_products = []
    
    for i, stock_code in enumerate(products):
        print(f"[{i+1}/{len(products)}] Forecasting {stock_code}...")
        
        try:
            prophet_df = prepare_prophet_data(demand_df, stock_code)
            
            if len(prophet_df) < 20:
                print(f"  Skipping - insufficient data ({len(prophet_df)} weeks)")
                failed_products.append(stock_code)
                continue
            
            model, forecast, prophet_df = train_prophet_model(
                prophet_df, stock_code, forecast_weeks
            )

            quality = assess_forecastability(prophet_df)

            results[stock_code] = {
                'model': model,
                'forecast': forecast,
                'prophet_df': prophet_df,
                'description': demand_df[
                    demand_df['StockCode'] == stock_code
                ]['Description'].iloc[0],
                'quality': quality
            }
            
            print(f"  Done ({quality['DataQualityFlag']})")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            failed_products.append(stock_code)
    
    print(f"\nForecast pipeline complete:")
    print(f"  Successful: {len(results)}")
    print(f"  Failed/skipped: {len(failed_products)}")
    
    if failed_products:
        print(f"  Failed products: {failed_products}")
    
    all_forecasts = []
    
    for stock_code, result in results.items():
        forecast = result['forecast'].copy()
        forecast['StockCode'] = stock_code
        forecast['Description'] = result['description']
        forecast['DataQualityFlag'] = result['quality']['DataQualityFlag']
        forecast['ZeroWeekPct'] = result['quality']['ZeroWeekPct']
        
        forecast = forecast[[
            'StockCode', 'Description', 'ds',
            'yhat', 'yhat_lower', 'yhat_upper',
            'trend', 'trend_lower', 'trend_upper',
            'DataQualityFlag', 'ZeroWeekPct'
        ]]
        
        last_date = result['prophet_df']['ds'].max()
        forecast['IsFuture'] = forecast['ds'] > last_date
        
        all_forecasts.append(forecast)
    
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    
    # Clip negative predictions to 0 - demand can't be negative.
    forecasts_df['yhat'] = forecasts_df['yhat'].clip(lower=0)
    forecasts_df['yhat_lower'] = forecasts_df['yhat_lower'].clip(lower=0)
    
    os.makedirs('data/processed', exist_ok=True)
    forecasts_df.to_csv('data/processed/all_forecasts.csv', index=False)
    print(f"All forecasts saved to data/processed/all_forecasts.csv")
    
    if save_models:
        os.makedirs('models/prophet_models', exist_ok=True)
        for stock_code, result in results.items():
            model_path = f'models/prophet_models/{stock_code}.pkl'
            joblib.dump(result['model'], model_path)
        print(f"{len(results)} Prophet models saved to models/prophet_models/")
    
    return results, forecasts_df

def analyze_seasonality_patterns(results: dict) -> pd.DataFrame:
    """
    Extract and compare seasonality patterns across all products.
    """
    
    seasonality_data = []
    
    for stock_code, result in results.items():
        forecast = result['forecast']
        
        if 'yearly' in forecast.columns:
            yearly_range = forecast['yearly'].max() - forecast['yearly'].min()
        else:
            yearly_range = 0
        
        if 'weekly' in forecast.columns:
            weekly_range = forecast['weekly'].max() - forecast['weekly'].min()
        else:
            weekly_range = 0
        
        trend_start = forecast['trend'].iloc[0]
        trend_end = forecast['trend'].iloc[-1]
        trend_direction = 'Growing' if trend_end > trend_start else 'Declining'

        # Express change relative to the product's historical mean demand,
        # not trend_start - trend_start is often near 0 for low-volume
        # products, which caused percentages like 28,000%+.
        historical_mean = result['prophet_df']['y'].mean()
        trend_change_units = trend_end - trend_start
        trend_pct_of_mean = (
            (trend_change_units / historical_mean * 100)
            if historical_mean > 0 else 0
        )
        
        seasonality_data.append({
            'StockCode': stock_code,
            'Description': result['description'][:40],
            'YearlySeasonalityRange': round(yearly_range, 2),
            'WeeklySeasonalityRange': round(weekly_range, 2),
            'TrendDirection': trend_direction,
            'TrendChangeUnits': round(trend_change_units, 1),
            'TrendChangeVsMean%': round(trend_pct_of_mean, 1),
            'DataQualityFlag': result['quality']['DataQualityFlag']
        })
    
    seasonality_df = pd.DataFrame(seasonality_data)
    seasonality_df = seasonality_df.sort_values(
        'WeeklySeasonalityRange', ascending=False
    )
    
    print("\nSeasonality Analysis:")
    print(seasonality_df.to_string(index=False))
    
    seasonality_df.to_csv('data/processed/seasonality_analysis.csv', index=False)
    print("Saved to data/processed/seasonality_analysis.csv")
    
    return seasonality_df

if __name__ == "__main__":
    demand = pd.read_csv('data/processed/weekly_demand.csv')
    demand['Week'] = pd.to_datetime(demand['Week'])
    
    results, forecasts_df = run_multi_product_forecast(demand, forecast_weeks=8)
    seasonality_df = analyze_seasonality_patterns(results)
    
    print(f"\nPipeline complete")
    print(f"Total forecast rows: {len(forecasts_df):,}")