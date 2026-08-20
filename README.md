# Retail Intelligence Platform

## Problem Statement
This system addresses two critical retail challenges: predicting which customers are likely to churn so interventions can be made before they leave, and forecasting product demand to optimize inventory levels and reduce stockouts or overstock. By adding LLM-powered narration, the platform turns raw model outputs into plain-English insights accessible to users without a data science background, all while keeping data processing entirely on-premise.

## Overview
The Retail Intelligence Platform is an end-to-end machine learning system that predicts customer churn and forecasts product demand. It integrates supervised learning models (XGBoost for churn, Prophet for demand) with explainability tools (SHAP) and a local LLM (Ollama/Llama3) to generate actionable business narratives. The platform exposes its functionality via a versioned FastAPI backend, which is consumed by a three-tab Streamlit dashboard for interactive exploration.

## Architecture
The system follows a layered architecture where processed data files feed into a FastAPI backend, which serves REST endpoints to a Streamlit frontend:

```text
customer_risk_table.csv, shap_values.csv, all_forecasts.csv,
inventory_alerts.csv, walkforward_folds.csv, walkforward_summary.csv,
seasonality_analysis.csv, clean_retail.csv, model .pkl files
              │
              ▼
   FastAPI Backend (src/api/)
   config → dependencies (cached loaders) → services → routers
              │
      ┌───────┼────────┬─────────────┐
      ▼       ▼        ▼             ▼
  /health   /churn   /demand       /llm
              │
              ▼
   RetailAPIClient (src/api_client.py)
              │
              ▼
   Streamlit Dashboard (3 tabs, HTTP-driven)
```

## What It Does

**Churn Prediction (XGBoost + SHAP)**:  
Identifies customers at risk of churn and quantifies that risk with a probability score. Uses SHAP values to explain the key drivers behind each prediction (e.g., low engagement, low spend), enabling targeted retention strategies.

**Demand Forecasting (Prophet)**:  
Produces weekly demand forecasts for the top 20 products, validated via walk-forward backtesting that simulates real-world weekly retraining. Provides point forecasts and confidence intervals to guide restocking and promotional planning.

**LLM Narration (Ollama/Llama3, local)**:  
Converts numerical model outputs into plain-English reports and answers—such as weekly demand summaries, churn explanations, and win-back suggestions—entirely on-premise. No customer or business data leaves the system, ensuring privacy and eliminating reliance on external APIs.

## Project Structure
```
retail-intelligence/
├── src/
│   ├── api/                  # FastAPI backend: config, dependencies, services, routers
│   ├── module1_churn/        # Churn prediction: features, modeling, SHAP
│   ├── module2_demand/       # Demand forecasting: features, Prophet modeling
│   └── module3_llm/          # LLM narration: prompt templates, narrators
├── data/
│   └── processed/            # Cleaned data, features, forecasts, model outputs
├── models/                   # Trained model pickles, SHAP explainers, thresholds
├── notebooks/                # Jupyter notebooks for exploration and validation
├── docs/                     # Documentation (including DEVLOG.md)
└── tests/                    # Unit tests
```

## Key Results

| Metric                  | Value     |
|-------------------------|-----------|
| **Churn Model**         | XGBoost (tuned) |
| Churn PR-AUC            | 0.604     |
| Churn F1 Score          | 0.64      |
| Churn Decision Threshold| 0.20      |
| **Demand Forecast**     |           |
| Win Rate vs Naive       | 47.8%     |
| Win Rate vs Seasonal Naive| 70.0%   |
| Median MAPE             | 73.4%     |

## Tech Stack

- **Modeling**: XGBoost, SHAP, Prophet
- **LLM**: LangChain, Ollama (Llama3)
- **API**: FastAPI
- **Frontend**: Streamlit
- **Data**: pandas

## Setup Instructions

1. **Create the conda environment**:
   ```bash
   conda create -n retail_intelligence python=3.10
   conda activate retail_intelligence
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas xgboost shap prophet streamlit fastapi langchain ollama
   ```

3. **Verify the setup** (optional):
   ```bash
   python src/shared/test_setup.py
   ```

4. **Run the FastAPI backend**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

5. **Run the Streamlit app** (in a separate terminal):
   ```bash
   streamlit run src/app.py
   ```

## Documentation

For the full week-by-week development log — every bug found and fixed along the way — see [docs/DEVLOG.md](docs/DEVLOG.md).