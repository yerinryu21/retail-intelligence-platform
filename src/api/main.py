"""
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.config import settings
from src.api.routers import health, churn, demand, llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models and data on startup to reduce first-request latency"""
    print("Starting Retail Intelligence Platform API...")
    print(f"Version: {settings.API_VERSION}")
    print("Pre-loading models and data...")
    
    try:
        from src.api.dependencies import (
            get_risk_table, get_alerts, get_churn_model,
            get_shap_explainer, get_forecasts, get_folds
        )
        get_risk_table()
        get_alerts()
        get_churn_model()
        get_shap_explainer()
        get_forecasts()
        get_folds()
        print("All models and data pre-loaded successfully")
    except Exception as e:
        print(f"Pre-loading failed: {e}")
        print("Models will be loaded on first request")
    
    yield  # app runs here
    
    print("Shutting down API...")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(churn.router)
app.include_router(demand.router)
app.include_router(llm.router)

@app.get("/")
async def root():
    return {
        "message": "Retail Intelligence Platform API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )