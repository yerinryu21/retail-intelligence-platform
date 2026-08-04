import pandas as pd
from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.demand_narrator import DemandNarrator

client = RetailLLMClient()
narrator = DemandNarrator(client)

alerts = pd.read_csv('data/processed/inventory_alerts.csv')
seasonality = pd.read_csv('data/processed/seasonality_analysis.csv')
merged = alerts.merge(seasonality[['StockCode', 'TrendDirection']], on='StockCode', how='left')

# Test 1: a normal, reliable product with a stockout alert
sample = merged[merged['StockCode'] == '21915'].iloc[0]
result = narrator.explain_product_forecast(sample, sample['TrendDirection'])
print("Reliable product (21915):")
print(result)
print()

# Test 2: the unreliable product
sample_unreliable = merged[merged['StockCode'] == '23166'].iloc[0]
result_unreliable = narrator.explain_product_forecast(sample_unreliable, sample_unreliable['TrendDirection'])
print("Unreliable product (23166):")
print(result_unreliable)

print()
print("Daily briefing:")
briefing = narrator.generate_daily_briefing(alerts)
print(briefing)

print()
print("Weekly report:")
forecasts = pd.read_csv('data/processed/all_forecasts.csv')
folds = pd.read_csv('data/processed/walkforward_folds.csv')
weekly_report = narrator.generate_weekly_report(forecasts, folds)
print(weekly_report)