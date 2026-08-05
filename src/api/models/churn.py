"""
Pydantic models for churn prediction endpoints.
These define the shape of requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskTier(str, Enum):
    EXTREME = "⚫ Extreme Risk"
    HIGH = "🔴 High Risk"
    MEDIUM = "🟡 Medium Risk"
    LOW = "🟢 Low Risk"

class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction"""
    frequency: float = Field(..., description="Number of orders placed", ge=0)
    monetary: float = Field(..., description="Total spend in GBP", ge=0)
    avg_order_value: float = Field(..., description="Average order value", ge=0)
    unique_products: float = Field(..., description="Unique products purchased", ge=0)
    avg_quantity: float = Field(..., description="Average quantity per order", ge=0)
    days_active: float = Field(..., description="Days as active customer", ge=0)
    orders_per_day: float = Field(..., description="Order frequency rate", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "frequency": 8,
                "monetary": 320.50,
                "avg_order_value": 40.06,
                "unique_products": 15,
                "avg_quantity": 3.2,
                "days_active": 180,
                "orders_per_day": 0.044
            }
        }

class ChurnPredictionResponse(BaseModel):
    """Response from churn prediction endpoint"""
    customer_index: Optional[int] = None
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_tier: RiskTier
    predicted_churn: bool
    top_risk_factors: List[Dict[str, Any]]
    explanation: str
    clv_estimate: Optional[float] = None
    revenue_at_risk: Optional[float] = None

class CustomerRiskTableResponse(BaseModel):
    """Response for customer risk table endpoint"""
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    extreme_risk_count: int
    total_revenue_at_risk: float
    customers: List[Dict]
    page: int
    page_size: int
    total_pages: int

class ChurnSummaryResponse(BaseModel):
    """High-level churn summary statistics"""
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    extreme_risk_count: int
    total_revenue_at_risk: float
    avg_churn_probability: float
    top_churn_driver: str
    model_pr_auc: Optional[float] = None