"""
Demand forecasting API endpoints.

Endpoints:
- GET /demand/summary — inventory status overview
- GET /demand/alerts — all product alerts
- GET /demand/products — list all tracked products
- GET /demand/products/{stock_code} — single product forecast (partial data tolerant)
- GET /demand/backtest — walk-forward validation results
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import numpy as np
import pandas as pd

from src.api.dependencies import (
    get_forecasts, get_alerts, get_folds, get_walkforward_summary
)

from src.api.services.demand_service import (
    get_demand_summary, get_product_forecast
)

from src.api.models.demand import (
    DemandSummaryResponse,
    AlertsSummaryResponse,
    ProductListResponse,
    ProductDetailResponse
)

router = APIRouter(prefix="/demand", tags=["Demand Forecasting"])


@router.get("/summary", response_model=DemandSummaryResponse)
async def get_summary():
    """Get demand and inventory overview statistics."""
    alerts = get_alerts()
    folds = get_folds()
    return get_demand_summary(alerts, folds)


@router.get("/alerts", response_model=AlertsSummaryResponse)
async def get_alerts_list(
    alert_level: Optional[str] = Query(
        None,
        description="Filter by level: critical, warning, ok, unreliable"
    ),
    sort_by: str = Query("UrgencyRank", description="Sort column")
):
    """Get all inventory alerts with optional filtering, sorted by urgency by default."""

    alerts = get_alerts()
    filtered = alerts.copy()

    if alert_level:
        filtered = filtered[filtered['AlertLevel'] == alert_level]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by)

    results = []
    for _, row in filtered.iterrows():
        results.append({
            "stock_code": str(row.get('StockCode', '')),
            "description": str(row.get('Description', ''))[:60],
            "alert_status": str(row.get('AlertStatus', '')),
            "alert_level": str(row.get('AlertLevel', '')),
            "alert_message": str(row.get('AlertMessage', '')),
            "current_stock": float(row.get('EstimatedCurrentStock', 0)),
            "forecasted_demand": float(row.get('ForecastedDemand_Reorder', 0)),
            "reorder_point": float(row.get('ReorderPoint', 0)),
            "suggested_reorder_qty": float(row.get('SuggestedReorderQty', 0))
        })

    return {
        "total": len(results),
        "critical": int((alerts['AlertLevel'] == 'critical').sum()),
        "warning": int((alerts['AlertLevel'] == 'warning').sum()),
        "alerts": results
    }


@router.get("/products", response_model=ProductListResponse)
async def get_products():
    """List all products being tracked with their current alert status."""

    alerts = get_alerts()

    products = []
    for _, row in alerts.iterrows():
        products.append({
            "stock_code": str(row.get('StockCode', '')),
            "description": str(row.get('Description', ''))[:60],
            "alert_status": str(row.get('AlertStatus', '')),
            "alert_level": str(row.get('AlertLevel', '')),
            "forecasted_demand_4wk": float(row.get('ForecastedDemand_Reorder', 0)),
            "current_stock": float(row.get('EstimatedCurrentStock', 0))
        })

    return {
        "total_products": len(products),
        "products": products
    }


@router.get("/products/{stock_code}", response_model=ProductDetailResponse)
async def get_product_detail(stock_code: str):
    """
    Get complete forecast detail for a specific product.

    Returns 404 if the product is missing from either forecasts or
    inventory alerts — all 20 tracked products are confirmed present
    in both sources, so a missing product indicates a data problem.
    """

    forecasts = get_forecasts()
    alerts = get_alerts()
    folds = get_folds()

    try:
        result = get_product_forecast(forecasts, alerts, folds, stock_code.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve forecast: {str(e)}")

    return result


@router.get("/backtest")
async def get_backtest_results(
    stock_code: Optional[str] = Query(None, description="Filter to specific product")
):
    """
    Get walk-forward validation backtest results, including both
    fold-level detail and per-product summary stats (walkforward_summary.csv).
    """

    import numpy as np

    folds = get_folds()
    wf_summary = get_walkforward_summary()

    if stock_code:
        folds = folds[folds['StockCode'] == stock_code.upper()]
        wf_summary = wf_summary[wf_summary['StockCode'] == stock_code.upper()]

    if len(folds) == 0:
        raise HTTPException(status_code=404, detail=f"No backtest data found for {stock_code}")

    def safe_float(value):
        """Convert NaN/Inf to None so the response is valid JSON"""
        f = float(value)
        return f if np.isfinite(f) else None

    finite_mape = folds['Prophet_MAPE'].replace([np.inf, -np.inf], np.nan).dropna()

    fold_details = []
    for _, row in folds.iterrows():
        fold_details.append({
            "stock_code": str(row.get('StockCode', '')),
            "fold": int(row.get('Fold', 0)),
            "test_start": str(row.get('TestStart', '')),
            "test_end": str(row.get('TestEnd', '')),
            "prophet_mae": safe_float(row.get('Prophet_MAE', 0)),
            "prophet_mape": safe_float(row.get('Prophet_MAPE', 0)),
            "naive_mae": safe_float(row.get('Naive_MAE', 0)),
            "seasonal_mae": safe_float(row.get('Seasonal_MAE', 0)),
            "beat_naive": bool(row.get('BeatNaive', False)),
            "beat_seasonal": bool(row.get('BeatSeasonal', False))
        })

    product_summary = []
    for _, row in wf_summary.iterrows():
        product_summary.append({
            "stock_code": str(row.get('StockCode', '')),
            "prophet_mae_mean": safe_float(row.get('Prophet_MAE_mean', 0)),
            "prophet_mae_median": safe_float(row.get('Prophet_MAE_median', 0)),
            "prophet_mape_mean": safe_float(row.get('Prophet_MAPE_mean', 0)),
            "prophet_mape_median": safe_float(row.get('Prophet_MAPE_median', 0)),
            "naive_mae_mean": safe_float(row.get('Naive_MAE_mean', 0)),
            "seasonal_mae_mean": safe_float(row.get('Seasonal_MAE_mean', 0)),
            "win_rate_vs_naive": safe_float(row.get('WinRateVsNaive', 0)),
            "win_rate_vs_seasonal": safe_float(row.get('WinRateVsSeasonal', 0)),
            "folds": int(row.get('Folds', 0)) if pd.notna(row.get('Folds')) else None,
            "mean_actual_demand": safe_float(row.get('MeanActualDemand', 0)),
            "improvement_vs_naive_pct": safe_float(row.get('ImprovementVsNaive%', 0)),
            "improvement_vs_seasonal_pct": safe_float(row.get('ImprovementVsSeasonal%', 0))
        })

    return {
        "overall_metrics": {
            "avg_prophet_mae": safe_float(folds['Prophet_MAE'].mean()),
            "avg_naive_mae": safe_float(folds['Naive_MAE'].mean()),
            "avg_seasonal_mae": safe_float(folds['Seasonal_MAE'].mean()),
            "prophet_win_rate_vs_naive": safe_float(folds['BeatNaive'].mean()),
            "prophet_win_rate_vs_seasonal": safe_float(folds['BeatSeasonal'].mean()),
            "median_mape_excl_inf": safe_float(finite_mape.median()) if len(finite_mape) > 0 else None,
            "total_folds": len(folds),
            "folds_with_infinite_mape": int(np.isinf(folds['Prophet_MAPE']).sum())
        },
        "fold_details": fold_details,
        "product_summary": product_summary
    }