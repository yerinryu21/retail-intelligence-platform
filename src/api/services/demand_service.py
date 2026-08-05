"""
Demand forecasting service layer.

Data note: all_forecasts.csv is a weekly time series (historical + future
rows split by IsFuture) joined to inventory_alerts.csv (one row per product,
current snapshot) via StockCode. inventory_alerts.csv has no DemandLower_Lead
column — only an upper bound exists for the lead-time horizon.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def get_demand_summary(alerts_df: pd.DataFrame,
                        folds_df: pd.DataFrame) -> Dict:
    """Calculate demand and inventory summary statistics"""

    return {
        "total_products": len(alerts_df),
        "stockout_risk_count": int((alerts_df['AlertLevel'] == 'critical').sum()),
        "reorder_soon_count": int((alerts_df['AlertLevel'] == 'warning').sum()),
        "adequate_count": int((alerts_df['AlertLevel'] == 'ok').sum()),
        "unreliable_count": int((alerts_df['AlertLevel'] == 'unreliable').sum()),
        "avg_forecast_mape": float(folds_df['Prophet_MAPE'].mean()),
        "prophet_win_rate_vs_naive": float(folds_df['BeatNaive'].mean()),
        "products": alerts_df.to_dict(orient='records')
    }


def get_product_forecast(forecasts_df: pd.DataFrame,
                          alerts_df: pd.DataFrame,
                          folds_df: pd.DataFrame,
                          stock_code: str) -> Dict:
    """
    Get complete forecast + alert + backtest detail for a single product.
    Raises ValueError if the product is missing from forecasts or alerts —
    all 20 tracked products are confirmed present in both sources, so a
    missing product here indicates a real data problem, not an expected gap.
    """

    product_forecasts = forecasts_df[forecasts_df['StockCode'] == stock_code].copy()

    if len(product_forecasts) == 0:
        raise ValueError(f"Product {stock_code} not found in forecasts")

    product_alert = alerts_df[alerts_df['StockCode'] == stock_code]

    if len(product_alert) == 0:
        raise ValueError(f"Product {stock_code} not found in inventory alerts")

    product_folds = folds_df[folds_df['StockCode'] == stock_code]

    historical = product_forecasts[~product_forecasts['IsFuture']]
    future = product_forecasts[product_forecasts['IsFuture']]

    historical_records = [
        {
            "date": str(row['ds'].date() if hasattr(row['ds'], 'date') else row['ds']),
            "actual": float(row['yhat']),
            "trend": float(row.get('trend', row['yhat']))
        }
        for _, row in historical.iterrows()
    ]

    forecast_records = [
        {
            "date": str(row['ds'].date() if hasattr(row['ds'], 'date') else row['ds']),
            "forecast": float(max(0, row['yhat'])),
            "lower": float(max(0, row['yhat_lower'])),
            "upper": float(max(0, row['yhat_upper']))
        }
        for _, row in future.iterrows()
    ]

    model_metrics = {}
    if len(product_folds) > 0:
        model_metrics = {
            "mean_mae": float(product_folds['Prophet_MAE'].mean()),
            "mean_mape": float(product_folds['Prophet_MAPE'].mean()),
            "win_rate_vs_naive": float(product_folds['BeatNaive'].mean()),
            "win_rate_vs_seasonal": float(product_folds['BeatSeasonal'].mean()),
            "folds_evaluated": int(len(product_folds))
        }

    alert_row = product_alert.iloc[0]

    return {
        "stock_code": stock_code,
        "description": str(product_forecasts['Description'].iloc[0]),
        "historical": historical_records[-20:],  # last 20 weeks
        "forecast": forecast_records,
        "model_metrics": model_metrics,
        "alert_status": str(alert_row.get('AlertStatus', 'Unknown')),
        "alert_level": str(alert_row.get('AlertLevel', 'ok')),
        "alert_message": str(alert_row.get('AlertMessage', '')),
        "current_stock": float(alert_row.get('EstimatedCurrentStock', 0)),
        "suggested_reorder_qty": float(alert_row.get('SuggestedReorderQty', 0)),
        "reorder_point": float(alert_row.get('ReorderPoint', 0)),
        "safety_stock": float(alert_row.get('SafetyStock', 0)),
        "forecasted_demand_reorder": float(alert_row.get('ForecastedDemand_Reorder', 0)),
        "demand_lower_reorder": float(alert_row.get('DemandLower_Reorder', 0)),
        "demand_upper_reorder": float(alert_row.get('DemandUpper_Reorder', 0)),
        "forecasted_demand_lead": float(alert_row.get('ForecastedDemand_Lead', 0)),
        "demand_upper_lead": float(alert_row.get('DemandUpper_Lead', 0)),
        "urgency_rank": int(alert_row.get('UrgencyRank', 0)) if pd.notna(alert_row.get('UrgencyRank')) else None
    }