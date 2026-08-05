"""
Health check endpoints.
These are the first endpoints you build and the first you test.
A health check tells you:
- Is the API running?
- Are the models loaded?
- Is Ollama responding?
"""

from fastapi import APIRouter
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.api.config import settings
from src.api.dependencies import (
    get_churn_model, get_alerts, get_risk_table
)

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    """Basic health check — is the API running?"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.API_VERSION
    }

@router.get("/models")
async def models_health():
    """Check that all ML models and data are loaded and responding"""
    
    status = {}
    
    try:
        model = get_churn_model()
        status["churn_model"] = {
            "status": "loaded",
            "type": type(model).__name__
        }
    except Exception as e:
        status["churn_model"] = {"status": "error", "message": str(e)}
    
    try:
        risk_table = get_risk_table()
        status["customer_risk_table"] = {
            "status": "loaded",
            "rows": len(risk_table)
        }
    except Exception as e:
        status["customer_risk_table"] = {"status": "error", "message": str(e)}
    
    try:
        alerts = get_alerts()
        status["inventory_alerts"] = {
            "status": "loaded",
            "rows": len(alerts)
        }
    except Exception as e:
        status["inventory_alerts"] = {"status": "error", "message": str(e)}
    
    all_healthy = all(v.get("status") == "loaded" for v in status.values())
    
    return {
        "overall": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": status
    }

@router.get("/data")
async def data_health():
    """Check all data files are present"""
    
    data_files = {
        "customer_risk_table": str(settings.CUSTOMER_RISK_TABLE),
        "shap_values": str(settings.SHAP_VALUES),
        "all_forecasts": str(settings.ALL_FORECASTS),
        "inventory_alerts": str(settings.INVENTORY_ALERTS),
        "walkforward_folds": str(settings.WALKFORWARD_FOLDS)
    }
    
    status = {}
    for name, path in data_files.items():
        status[name] = {
            "exists": os.path.exists(path),
            "path": path
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "data_files": status
    }