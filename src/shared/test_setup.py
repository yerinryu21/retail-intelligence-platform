# Test 1 — pandas
import pandas as pd
import numpy as np
print("✅ pandas and numpy working")

# Test 2 — scikit-learn + xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
print("✅ scikit-learn and xgboost working")

# Test 3 — SHAP
import shap
print("✅ SHAP working")

# Test 4 — Prophet
from prophet import Prophet
print("✅ Prophet working")

# Test 5 — Ollama connection
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3")
response = llm.invoke("Say hello in exactly 5 words.")
print(f"✅ Ollama working: {response}")

# Test 6 — Streamlit import
import streamlit
print("✅ Streamlit working")

# Test 7 — Plotly
import plotly.express as px
print("✅ Plotly working")

print("\n🎉 All systems ready. Day 1 complete.")