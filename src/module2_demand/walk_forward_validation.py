import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
warnings.filterwarnings('ignore')

from src.module2_demand.prophet_model import prepare_prophet_data
from src.module2_demand.baseline_models import (
    calculate_mae, calculate_rmse, calculate_mape,
    seasonal_naive_baseline
)

# Week 4 finding: actuals below this threshold make MAPE blow up
# (product 21915: 3517% MAPE from a single near-zero week). Same 5-unit
# threshold Week 4 landed on in Day 5 — keep it consistent here.
MAPE_ZERO_THRESHOLD = 5


def load_data_quality_flags(forecasts_path: str = 'data/processed/all_forecasts.csv') -> pd.Series:
    """
    Pull DataQualityFlag per StockCode from Week 4's all_forecasts.csv.
    One flag per product, so drop_duplicates on StockCode is enough.
    """
    forecasts = pd.read_csv(forecasts_path)
    flags = forecasts[['StockCode', 'DataQualityFlag']].drop_duplicates()
    flags['StockCode'] = flags['StockCode'].astype(str)
    return flags.set_index('StockCode')['DataQualityFlag']


def walk_forward_validation_single(prophet_df: pd.DataFrame,
                                    stock_code: str,
                                    n_folds: int = 5,
                                    test_size: int = 4,
                                    min_train_size: int = 20) -> pd.DataFrame:
    """
    Walk-forward validation for a single product.

    Parameters:
    - prophet_df: dataframe with ds and y columns
    - stock_code: product identifier for labeling
    - n_folds: number of validation folds
    - test_size: weeks in each test window
    - min_train_size: minimum weeks needed to train Prophet

    Returns: dataframe with error metrics per fold per model
    """

    total_weeks = len(prophet_df)

    min_required = min_train_size + (n_folds * test_size)
    if total_weeks < min_required:
        print(f"  ⚠️ {stock_code}: insufficient data "
              f"({total_weeks} weeks, need {min_required})")
        return None

    fold_results = []

    for fold in range(n_folds):

        test_end_idx = total_weeks - (n_folds - fold - 1) * test_size
        test_start_idx = test_end_idx - test_size
        train_end_idx = test_start_idx

        if train_end_idx < min_train_size:
            continue

        train = prophet_df.iloc[:train_end_idx].copy()
        test = prophet_df.iloc[test_start_idx:test_end_idx].copy()

        if len(test) == 0:
            continue

        fold_start_date = test['ds'].iloc[0].date()
        fold_end_date = test['ds'].iloc[-1].date()

        try:
            # Week 4 final model: TREND ONLY.
            # yearly_seasonality/weekly_seasonality/holidays all confirmed
            # as noise for this dataset (single training weekday, ~1yr
            # history, single-occurrence holidays) — don't reintroduce them.
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                growth='linear'
            )
            model.fit(train)

            # Week 4 Day 2 fix: training dates are Monday-anchored, so
            # future dates must be too, or forecast dates won't align
            # with actual test dates for comparison.
            future = model.make_future_dataframe(
                periods=len(test), freq='W-MON'
            )
            forecast = model.predict(future)

            prophet_preds_df = forecast[forecast['ds'].isin(test['ds'])]

            # Guard: if dates still don't align, this catches it loudly
            # instead of silently truncating arrays of mismatched length.
            if len(prophet_preds_df) != len(test):
                print(f"  ❌ Fold {fold+1} date mismatch for {stock_code}: "
                      f"{len(prophet_preds_df)} predictions vs {len(test)} test rows")
                continue

            prophet_preds = np.clip(prophet_preds_df['yhat'].values, 0, None)

        except Exception as e:
            print(f"  ❌ Fold {fold+1} failed for {stock_code}: {e}")
            continue

        actual = test['y'].values

        naive_pred = train['y'].iloc[-1]
        naive_preds = np.full(len(test), naive_pred)

        seasonal = seasonal_naive_baseline(train, forecast_weeks=len(test))
        seasonal_preds = seasonal['yhat_seasonal'].values[:len(test)]

        fold_result = {
            'StockCode': stock_code,
            'Fold': fold + 1,
            'TestStart': fold_start_date,
            'TestEnd': fold_end_date,
            'TrainWeeks': len(train),
            'TestWeeks': len(test),

            'Prophet_MAE': calculate_mae(actual, prophet_preds),
            'Prophet_RMSE': calculate_rmse(actual, prophet_preds),
            'Prophet_MAPE': calculate_mape(
                pd.Series(actual), pd.Series(prophet_preds),
                min_actual=MAPE_ZERO_THRESHOLD
            ),

            'Naive_MAE': calculate_mae(actual, naive_preds),
            'Naive_RMSE': calculate_rmse(actual, naive_preds),
            'Naive_MAPE': calculate_mape(
                pd.Series(actual), pd.Series(naive_preds),
                min_actual=MAPE_ZERO_THRESHOLD
            ),

            'Seasonal_MAE': calculate_mae(actual, seasonal_preds),
            'Seasonal_RMSE': calculate_rmse(actual, seasonal_preds),
            'Seasonal_MAPE': calculate_mape(
                pd.Series(actual), pd.Series(seasonal_preds),
                min_actual=MAPE_ZERO_THRESHOLD
            ),

            'BeatNaive': calculate_mae(actual, prophet_preds) < \
                         calculate_mae(actual, naive_preds),
            'BeatSeasonal': calculate_mae(actual, prophet_preds) < \
                            calculate_mae(actual, seasonal_preds),

            'MeanActualDemand': np.mean(actual),
        }

        fold_results.append(fold_result)

        print(f"  Fold {fold+1}: Train={len(train)}wk | "
              f"Test={fold_start_date}→{fold_end_date} | "
              f"Prophet MAE={fold_result['Prophet_MAE']:.1f} | "
              f"Naive MAE={fold_result['Naive_MAE']:.1f} | "
              f"Beat={'✅' if fold_result['BeatNaive'] else '❌'}")

    if not fold_results:
        return None

    return pd.DataFrame(fold_results)


def run_walk_forward_all_products(demand_df: pd.DataFrame,
                                   quality_flags: pd.Series,
                                   n_folds: int = 5,
                                   test_size: int = 4) -> pd.DataFrame:
    """
    Run walk-forward validation for all products.
    """

    products = demand_df['StockCode'].unique()
    all_fold_results = []

    print(f"Running walk-forward validation:")
    print(f"  Products: {len(products)}")
    print(f"  Folds: {n_folds}")
    print(f"  Test window: {test_size} weeks per fold")
    print(f"  Total model fits: ~{len(products) * n_folds}")
    print(f"  Estimated time: 20-40 minutes\n")

    for i, stock_code in enumerate(products):
        print(f"[{i+1}/{len(products)}] {stock_code}")

        prophet_df = prepare_prophet_data(demand_df, stock_code)

        fold_df = walk_forward_validation_single(
            prophet_df, stock_code, n_folds, test_size
        )

        if fold_df is not None:
            all_fold_results.append(fold_df)

    if not all_fold_results:
        print("❌ No valid results produced")
        return pd.DataFrame()

    results_df = pd.concat(all_fold_results, ignore_index=True)

    # Carryover note #3: DataQualityFlag must travel with any new table,
    # merged in immediately so it can't accidentally be left out of a
    # later summary/average.
    results_df['StockCode'] = results_df['StockCode'].astype(str)
    results_df['DataQualityFlag'] = results_df['StockCode'].map(quality_flags)

    unflagged = results_df['DataQualityFlag'].isna().sum()
    if unflagged > 0:
        print(f"  ⚠️ {unflagged} rows have no DataQualityFlag match — check StockCode types/values")

    results_df.to_csv('data/processed/walkforward_folds.csv', index=False)
    print(f"\n✅ Fold-level results saved: {len(results_df)} rows")

    return results_df


def summarize_walkforward_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fold-level results into a per-product summary CSV.
    Filters to DataQualityFlag == 'OK' -- summary numbers should never
    include low-reliability products in an average.
    """

    folds_ok = fold_results[fold_results['DataQualityFlag'] == 'OK'].copy()

    summary = folds_ok.groupby('StockCode').agg(
        Prophet_MAE_mean=('Prophet_MAE', 'mean'),
        Prophet_MAE_median=('Prophet_MAE', 'median'),
        Prophet_MAE_std=('Prophet_MAE', 'std'),
        Prophet_MAPE_mean=('Prophet_MAPE', 'mean'),
        Prophet_MAPE_median=('Prophet_MAPE', 'median'),

        Naive_MAE_mean=('Naive_MAE', 'mean'),
        Seasonal_MAE_mean=('Seasonal_MAE', 'mean'),

        WinRateVsNaive=('BeatNaive', 'mean'),
        WinRateVsSeasonal=('BeatSeasonal', 'mean'),

        Folds=('Fold', 'count'),
        MeanActualDemand=('MeanActualDemand', 'mean')
    ).reset_index().round(2)

    summary['ImprovementVsNaive%'] = (
        (summary['Naive_MAE_mean'] - summary['Prophet_MAE_mean']) /
        summary['Naive_MAE_mean'] * 100
    ).round(1)

    summary['ImprovementVsSeasonal%'] = (
        (summary['Seasonal_MAE_mean'] - summary['Prophet_MAE_mean']) /
        summary['Seasonal_MAE_mean'] * 100
    ).round(1)

    summary = summary.sort_values('Prophet_MAE_mean')

    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION SUMMARY (OK products only)")
    print("="*70)

    print(f"\nProducts summarized: {len(summary)}")
    print(f"Folds per product: {folds_ok['Fold'].max()}")

    print(f"\nOverall Prophet MAE: mean={summary['Prophet_MAE_mean'].mean():.2f}, "
          f"median={summary['Prophet_MAE_median'].median():.2f}")
    print(f"Naive MAE mean:    {summary['Naive_MAE_mean'].mean():.2f}")
    print(f"Seasonal MAE mean: {summary['Seasonal_MAE_mean'].mean():.2f}")

    print(f"\nWin rates (mean across products):")
    print(f"  vs Naive baseline:    {summary['WinRateVsNaive'].mean()*100:.1f}%")
    print(f"  vs Seasonal naive:    {summary['WinRateVsSeasonal'].mean()*100:.1f}%")

    summary.to_csv('data/processed/walkforward_summary.csv', index=False)
    print("\n✅ Summary saved to data/processed/walkforward_summary.csv")

    return summary

    
if __name__ == "__main__":
    demand = pd.read_csv('data/processed/weekly_demand.csv')
    demand['Week'] = pd.to_datetime(demand['Week'])

    quality_flags = load_data_quality_flags()

    fold_results = run_walk_forward_all_products(
        demand, quality_flags, n_folds=5, test_size=4
    )

    summary = summarize_walkforward_results(fold_results)