"""
Pydantic models for LLM narration endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class QueryRoute(str, Enum):
    CHURN = "CHURN"
    DEMAND = "DEMAND"
    BOTH = "BOTH"
    UNKNOWN = "UNKNOWN"

class NLQueryRequest(BaseModel):
    """Natural language query request"""
    question: str = Field(
        ..., 
        description="Plain-English question about customers or inventory",
        min_length=5,
        max_length=500
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Which customers are at highest risk of churning?"
            }
        }

class NLQueryResponse(BaseModel):
    """Natural language query response"""
    question: str
    route: QueryRoute
    answer: str

class WinBackRequest(BaseModel):
    """Win-back message generation request"""
    customer_index: int = Field(
        ..., 
        description="Index of customer in the risk table"
    )

class WinBackResponse(BaseModel):
    """Win-back message response"""
    customer_index: int
    risk_tier: str
    churn_probability: float
    explanation: str
    email_subject: str
    email_body: str
    recommended_incentive: str
    urgency: str

class DailyBriefingResponse(BaseModel):
    """Daily business briefing response"""
    churn_briefing: str
    inventory_briefing: str
    high_risk_customers: int
    stockout_products: int

class ChurnExplanationRequest(BaseModel):
    """Request for churn explanation"""
    customer_index: int

class ChurnExplanationResponse(BaseModel):
    """Churn explanation response"""
    customer_index: int
    churn_probability: float
    risk_tier: str
    explanation: str