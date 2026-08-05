"""
Pydantic models for demand forecasting endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class AlertLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    OK = "ok"
    UNRELIABLE = "unreliable"

class AlertStatus(str, Enum):
    STOCKOUT = "🔴 Stockout Risk"
    REORDER = "🟡 Reorder Soon"
    ADEQUATE = "🟢 Adequate"
    UNRELIABLE = "⚪ Unreliable Forecast"

class ProductForecastResponse(BaseModel):
    """Forecast for a single product"""
    stock_code: str
    description: str
    forecasted_demand: float
    demand_lower: float
    demand_upper: float
    current_stock: float
    alert_status: AlertStatus
    alert_level: AlertLevel
    alert_message: str
    suggested_reorder_qty: float
    reorder_point: float
    trend_direction: Optional[str] = None

class DemandSummaryResponse(BaseModel):
    """Overall demand and inventory summary"""
    total_products: int
    stockout_risk_count: int
    reorder_soon_count: int
    adequate_count: int
    unreliable_count: int
    avg_forecast_mape: float
    prophet_win_rate_vs_naive: float
    products: List[Dict]

class ForecastTimeSeriesResponse(BaseModel):
    """Time series forecast for a single product"""
    stock_code: str
    description: str
    historical: List[Dict]
    forecast: List[Dict]
    model_metrics: Dict