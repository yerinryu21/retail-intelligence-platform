import pandas as pd
import numpy as np
from typing import Tuple

def naive_baseline_forecast(prophet_df: pd.DataFrame,
                             forecast_weeks: int = 8) -> pd.DataFrame:
    """
    Naive baseline: forecast = last observed value (repeated).
    """
    
    last_value = prophet_df['y'].iloc[-1]
    last_date = prophet_df['ds'].iloc[-1]
    
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=forecast_weeks,
        freq='W-MON'  # match training cadence, same as prophet_model.py
    )
    
    naive_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat_naive': last_value,
        'yhat_naive_lower': last_value * 0.7,
        'yhat_naive_upper': last_value * 1.3
    })
    
    return naive_forecast

def seasonal_naive_baseline(prophet_df: pd.DataFrame,
                              forecast_weeks: int = 8) -> pd.DataFrame:
    """
    Seasonal naive baseline: forecast = same week from last year.
    
    NOTE: with only ~1 year of history, this baseline is on shaky ground for
    the same reason yearly_seasonality was disabled in prophet_model.py - the
    "same week last year" is really just one specific historical week, not a
    validated seasonal pattern. Kept as a baseline anyway since it's still a
    fair, simple comparison point (and its own limitation is transparent -
    it literally IS one specific historical data point, no hidden modeling).
    """
    
    last_date = prophet_df['ds'].iloc[-1]
    
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=forecast_weeks,
        freq='W-MON'
    )
    
    seasonal_forecasts = []
    
    for future_date in future_dates:
        same_week_last_year = future_date - pd.Timedelta(weeks=52)
        
        date_diffs = abs(prophet_df['ds'] - same_week_last_year)
        closest_idx = date_diffs.idxmin()
        seasonal_value = prophet_df.loc[closest_idx, 'y']
        
        seasonal_forecasts.append({
            'ds': future_date,
            'yhat_seasonal': seasonal_value,
            'yhat_seasonal_lower': seasonal_value * 0.7,
            'yhat_seasonal_upper': seasonal_value * 1.3
        })
    
    return pd.DataFrame(seasonal_forecasts)

def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mape(actual: pd.Series, predicted: pd.Series, min_actual: float = 5) -> float:
    """
    Mean Absolute Percentage Error.
    Excludes weeks where actual demand is below min_actual (default 5 units),
    not just exactly 0 - MAPE is mathematically unstable when dividing by
    very small numbers, producing extreme, misleading percentages even when
    the absolute forecast error is small (e.g. predicting 380 vs actual 2 
    units inflates MAPE by ~19,000% for that single week alone).
    """
    mask = actual >= min_actual
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def run_baseline_comparison_single(prophet_df: pd.DataFrame,
                                    prophet_forecast: pd.DataFrame,
                                    stock_code: str,
                                    test_weeks: int = 8) -> dict:
    """
    Compare Prophet against both baselines on held-out test weeks.
    Simplified comparison - full walk-forward validation is Week 5.
    """
    
    train = prophet_df.iloc[:-test_weeks]
    test = prophet_df.iloc[-test_weeks:]
    
    if len(test) == 0 or len(train) < 10:
        return None
    
    prophet_test_preds = prophet_forecast[
        prophet_forecast['ds'].isin(test['ds'])
    ]['yhat'].values
    
    if len(prophet_test_preds) != len(test):
        return None
    
    naive_pred = train['y'].iloc[-1]
    naive_preds = np.full(len(test), naive_pred)
    
    seasonal_naive = seasonal_naive_baseline(train, forecast_weeks=test_weeks)
    seasonal_preds = seasonal_naive['yhat_seasonal'].values[:len(test)]
    
    actual = test['y'].values
    
    results = {
        'StockCode': stock_code,
        'Prophet_MAE': calculate_mae(actual, prophet_test_preds),
        'Prophet_RMSE': calculate_rmse(actual, prophet_test_preds),
        'Prophet_MAPE': calculate_mape(pd.Series(actual), 
                                        pd.Series(prophet_test_preds)),
        'Naive_MAE': calculate_mae(actual, naive_preds),
        'Naive_RMSE': calculate_rmse(actual, naive_preds),
        'Naive_MAPE': calculate_mape(pd.Series(actual), pd.Series(naive_preds)),
        'SeasonalNaive_MAE': calculate_mae(actual, seasonal_preds),
        'SeasonalNaive_RMSE': calculate_rmse(actual, seasonal_preds),
        'SeasonalNaive_MAPE': calculate_mape(pd.Series(actual), 
                                               pd.Series(seasonal_preds))
    }
    
    results['BeatNaive'] = results['Prophet_MAE'] < results['Naive_MAE']
    results['BeatSeasonalNaive'] = (
        results['Prophet_MAE'] < results['SeasonalNaive_MAE']
    )
    
    return results

if __name__ == "__main__":
    import joblib
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    demand = pd.read_csv('data/processed/weekly_demand.csv')
    demand['Week'] = pd.to_datetime(demand['Week'])
    forecasts = pd.read_csv('data/processed/all_forecasts.csv')
    forecasts['ds'] = pd.to_datetime(forecasts['ds'])
    
    products = demand['StockCode'].unique()
    all_results = []
    
    print("Running baseline comparison across all products...")
    
    for stock_code in products:
        model_path = f'models/prophet_models/{stock_code}.pkl'
        if not os.path.exists(model_path):
            continue
        
        from src.module2_demand.prophet_model import prepare_prophet_data
        prophet_df = prepare_prophet_data(demand, stock_code)
        
        if len(prophet_df) < 20:
            continue
        
        product_forecast = forecasts[forecasts['StockCode'] == stock_code]
        
        result = run_baseline_comparison_single(
            prophet_df, product_forecast, stock_code, test_weeks=8
        )
        
        if result:
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    # Bring the data quality flag along so we can see if it correlates
    # with baseline performance
    quality_flags = forecasts[['StockCode', 'DataQualityFlag']].drop_duplicates()
    results_df = results_df.merge(quality_flags, on='StockCode', how='left')
    
    print(f"\nBaseline Comparison Results:")
    print(f"Products where Prophet beats naive baseline: "
          f"{results_df['BeatNaive'].sum()}/{len(results_df)}")
    print(f"Products where Prophet beats seasonal naive: "
          f"{results_df['BeatSeasonalNaive'].sum()}/{len(results_df)}")
    
    print(f"\nAverage metrics across all products:")
    metric_cols = ['Prophet_MAE', 'Naive_MAE', 'SeasonalNaive_MAE',
                   'Prophet_MAPE', 'Naive_MAPE', 'SeasonalNaive_MAPE']
    print(results_df[metric_cols].mean().round(2))
    
    results_df.to_csv('data/processed/baseline_comparison.csv', index=False)
    print("\nSaved to data/processed/baseline_comparison.csv")
    
    reliable_results = results_df[results_df['DataQualityFlag'] == 'OK']
    print(f"\nSame metrics, OK products only (excluding {len(results_df) - len(reliable_results)} flagged):")
    print(reliable_results[metric_cols].mean().round(2))
    print(f"\nBeat naive baseline (OK only): {reliable_results['BeatNaive'].sum()}/{len(reliable_results)}")
    print(f"Beat seasonal naive (OK only): {reliable_results['BeatSeasonalNaive'].sum()}/{len(reliable_results)}")