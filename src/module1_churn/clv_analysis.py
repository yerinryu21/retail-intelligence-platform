import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data

def calculate_clv_and_revenue_at_risk():
    """
    Calculate Customer Lifetime Value and revenue at risk from churn.
    This is the business translation layer that makes the ML output meaningful.
    """

    data = prepare_churn_data()
    X_test = data['X_test']
    y_test = data['y_test']

    model = joblib.load('models/churn_model_tuned.pkl')
    optimal_threshold = np.load('models/optimal_threshold.npy')

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Simple CLV formula: average monthly spend × expected remaining months
    customers = pd.read_csv('data/processed/customer_features.csv')
    customers['CustomerID'] = customers['CustomerID'].astype(str)

    # Assume average customer lifetime of 12 months
    customers['MonthlySpend'] = customers['Monetary'] / 12
    customers['CLV'] = customers['MonthlySpend'] * 12

    # Add predictions to test set (X_test.index maps back to the original customers table)
    test_customers = customers.loc[X_test.index].copy()
    test_customers['ChurnProbability'] = y_prob
    test_customers['PredictedChurn'] = y_pred

    # Revenue at risk: sum CLV of predicted churners
    predicted_churners = test_customers[test_customers['PredictedChurn'] == 1]
    revenue_at_risk = predicted_churners['CLV'].sum()

    # Top 20% highest risk customers
    top_20_pct = test_customers.nlargest(
        int(len(test_customers) * 0.2),
        'ChurnProbability'
    )
    top_20_revenue = top_20_pct['CLV'].sum()

    print(f"\n Business Impact Analysis:")
    print(f"   Optimal threshold used: {optimal_threshold:.2f}")
    print(f"   Predicted churners: {len(predicted_churners):,}")
    print(f"   Revenue at risk: ${revenue_at_risk:,.2f}")
    print(f"   Top 20% highest risk customers: {len(top_20_pct):,}")
    print(f"   Revenue at risk (top 20%): ${top_20_revenue:,.2f}")
    print(f"\n   → If we retain even 50% of predicted churners:")
    print(f"     Potential revenue saved: ${revenue_at_risk * 0.5:,.2f}")

    # Save results
    os.makedirs('data/processed', exist_ok=True)
    test_customers.to_csv('data/processed/churn_predictions.csv', index=False)
    print("\n✅ Predictions with CLV saved to data/processed/churn_predictions.csv")

    return test_customers

if __name__ == "__main__":
    predictions = calculate_clv_and_revenue_at_risk()
    print(predictions[['CustomerID', 'ChurnProbability', 'PredictedChurn', 'CLV']].head(10))