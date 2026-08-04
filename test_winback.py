import pandas as pd
from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.winback_generator import WinBackGenerator

client = RetailLLMClient()
generator = WinBackGenerator(client)

risk_table = pd.read_csv('data/processed/customer_risk_table.csv')
shap_values_df = pd.read_csv('data/processed/shap_values.csv')
retail_df = pd.read_csv('data/processed/clean_retail.csv')

feature_cols = ['Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts',
                 'AvgQuantity', 'DaysActive', 'OrdersPerDay']

clv_median = risk_table['CLV'].median()

high_risk = risk_table[risk_table['RiskTier'] == '🔴 High Risk']
sample_idx = high_risk.index[0]
customer_row = risk_table.loc[sample_idx]
shap_row = shap_values_df.iloc[sample_idx] if sample_idx < len(shap_values_df) else pd.Series()

result = generator.generate_for_customer(customer_row, shap_row, feature_cols, clv_median, retail_df)

print(f"Favorite product: {result['favorite_product']}")
print(f"Subject: {result['email_subject']}")
print(f"Body: {result['email_body']}")