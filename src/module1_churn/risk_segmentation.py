import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data
from src.module1_churn.shap_explainability import generate_natural_language_explanation

def assign_risk_tier(probability: float, threshold: float = 0.20) -> str:
    """Assign risk tier based on churn probability, tied to the actual
    business threshold — same logic as generate_natural_language_explanation,
    so RiskTier can never contradict PredictedChurn for the same customer."""
    if probability < threshold:
        return "🟢 Low Risk"
    elif probability < 0.5:
        return "🟡 Medium Risk"
    elif probability < 0.7:
        return "🔴 High Risk"
    else:
        return "⚫ Extreme Risk"

def build_customer_risk_table() -> pd.DataFrame:
    """
    Build complete customer risk table with:
    - Churn probability and risk tier
    - Key RFM features for context
    - CLV estimate
    - Top churn driver (from SHAP)
    - Natural language explanation
    """
    
    print("Building customer risk table...")
    
    # Load everything
    data = prepare_churn_data()
    X_test = data['X_test']
    y_test = data['y_test']
    feature_cols = data['feature_cols']
    
    model = joblib.load('models/churn_model_tuned.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    optimal_threshold = float(np.load('models/optimal_threshold.npy'))
    
    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # SHAP values
    shap_values = explainer(X_test)
    
    # ── Build main table ───────────────────────────────────────────
    risk_table = X_test.copy().reset_index(drop=True)
    risk_table['ChurnProbability'] = y_prob
    risk_table['PredictedChurn'] = y_pred
    risk_table['ActualChurn'] = y_test.reset_index(drop=True)
    risk_table['RiskTier'] = risk_table['ChurnProbability'].apply(
    lambda p: assign_risk_tier(p, optimal_threshold)
)
    
    # ── Add CLV estimate ───────────────────────────────────────────
    risk_table['CLV'] = risk_table['Monetary'] * 1.2
    risk_table['RevenueAtRisk'] = risk_table.apply(
        lambda x: x['CLV'] * x['ChurnProbability'], axis=1
    )
    
    # ── Add top SHAP driver per customer ──────────────────────────
    shap_array = shap_values.values
    top_driver_indices = np.argmax(np.abs(shap_array), axis=1)
    top_driver_names = [feature_cols[i] for i in top_driver_indices]
    top_driver_shap = [shap_array[i, top_driver_indices[i]] 
                       for i in range(len(top_driver_indices))]
    
    risk_table['TopChurnDriver'] = top_driver_names
    risk_table['TopDriverDirection'] = [
        'Increases risk' if v > 0 else 'Decreases risk' 
        for v in top_driver_shap
    ]
    
    # ── Generate natural language explanations ─────────────────────
    print("Generating natural language explanations...")
    explanations = []
    for i in range(len(risk_table)):
        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'value': X_test.iloc[i].values,
            'shap_value': shap_array[i]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        top_churn_drivers = shap_df[shap_df['shap_value'] > 0].head(3)
        top_retention_factors = shap_df[shap_df['shap_value'] < 0].head(3)
        
        explanation_dict = {
            'churn_probability': y_prob[i],
            'top_churn_drivers': top_churn_drivers,
            'top_retention_factors': top_retention_factors
        }
        
        nl = generate_natural_language_explanation(explanation_dict, optimal_threshold)
        explanations.append(nl)
    
    risk_table['Explanation'] = explanations
    
    # ── Sort by revenue at risk ────────────────────────────────────
    risk_table = risk_table.sort_values('RevenueAtRisk', ascending=False)
    risk_table = risk_table.reset_index(drop=True)
    
    # ── Summary ───────────────────────────────────────────────────
    print(f"\n📊 Customer Risk Summary:")
    tier_summary = risk_table['RiskTier'].value_counts()
    for tier, count in tier_summary.items():
        tier_customers = risk_table[risk_table['RiskTier'] == tier]
        revenue = tier_customers['RevenueAtRisk'].sum()
        print(f"   {tier}: {count:,} customers — ${revenue:,.2f} revenue at risk")
    
    print(f"\n💰 Total revenue at risk: ${risk_table['RevenueAtRisk'].sum():,.2f}")
    
    # ── Save ──────────────────────────────────────────────────────
    risk_table.to_csv('data/processed/customer_risk_table.csv', index=False)
    print("✅ Saved to data/processed/customer_risk_table.csv")
    
    return risk_table

if __name__ == "__main__":
    risk_table = build_customer_risk_table()
    print("\nSample output:")
    print(risk_table[['RiskTier', 'ChurnProbability', 'CLV', 
                       'RevenueAtRisk', 'TopChurnDriver', 'Explanation']].head(10)
    )