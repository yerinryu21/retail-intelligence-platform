import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data

def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values for a dataset using TreeExplainer.
    TreeExplainer is the fastest and most accurate explainer for XGBoost.
    """
    print("Computing SHAP values...")
    
    # Fix for shap 0.49.1 + XGBoost 2.x compatibility bug:
    # XGBoost serializes base_score as a bracketed string like '[5E-1]'.
    # shap 0.49.1's internal loader calls float() on it directly, which
    # fails. shap 0.50+ fixes this, but that requires Python 3.11+, so
    # instead we patch shap's own float() usage to handle the bracket
    # format, scoped only to shap's tree module.
    import shap.explainers._tree as _shap_tree
    import ast
    _builtin_float = float
    def _patched_float(x):
        if isinstance(x, str) and x.strip().startswith('['):
            parsed = ast.literal_eval(x)
            return _builtin_float(parsed[0] if isinstance(parsed, (list, tuple)) else parsed)
        return _builtin_float(x)
    _shap_tree.float = _patched_float
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    print(f"✅ SHAP values computed for {len(X)} customers")
    return shap_values, explainer

def plot_global_feature_importance(shap_values, 
                                    X: pd.DataFrame,
                                    save_dir: str = 'notebooks/shap_plots'):
    """
    Plot global SHAP feature importance — which features matter most
    across all customers in the dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ── Plot 1: Bar plot — mean absolute SHAP values ───────────────
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/global_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: global_bar.png")
    
    # ── Plot 2: Beeswarm plot — distribution of SHAP values ────────
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Value Distribution per Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/global_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: global_beeswarm.png")
    
    # ── Plot 3: SHAP heatmap — patterns across customers ───────────
    plt.figure(figsize=(12, 6))
    shap.plots.heatmap(shap_values[:100], show=False)
    plt.title("SHAP Heatmap (First 100 customers)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/global_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: global_heatmap.png")
    
    return save_dir

def get_feature_importance_table(shap_values, 
                                  X: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean dataframe of feature importances from SHAP values.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'MeanAbsSHAP': mean_abs_shap,
        'Rank': range(1, len(X.columns) + 1)
    }).sort_values('MeanAbsSHAP', ascending=False).reset_index(drop=True)
    
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    print("\n📊 Global Feature Importance:")
    print(importance_df.to_string(index=False))
    
    return importance_df

def explain_single_customer(customer_idx: int,
                             X: pd.DataFrame,
                             shap_values,
                             y_prob: np.ndarray,
                             threshold: float = 0.5,
                             save_dir: str = 'notebooks/shap_plots') -> dict:
    """
    Generate a complete SHAP explanation for a single customer.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    customer_features = X.iloc[customer_idx]
    customer_shap = shap_values[customer_idx]
    churn_probability = y_prob[customer_idx]
    predicted_churn = int(churn_probability >= threshold)
    
    # ── Waterfall plot ───────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(customer_shap, show=False)
    plt.title(f"Customer {customer_idx} — Churn Probability: {churn_probability:.1%}", 
              fontsize=13)
    plt.tight_layout()
    
    plot_path = f'{save_dir}/customer_{customer_idx}_waterfall.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Plain-text explanation ──────────────────────────────────
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'value': customer_features.values,
        'shap_value': customer_shap.values
    }).sort_values('shap_value', key=abs, ascending=False)
    
    top_churn_drivers = shap_df[shap_df['shap_value'] > 0].head(3)
    top_retention_factors = shap_df[shap_df['shap_value'] < 0].head(3)
    
    explanation = {
        'customer_idx': customer_idx,
        'churn_probability': churn_probability,
        'predicted_churn': predicted_churn,
        'top_churn_drivers': top_churn_drivers,
        'top_retention_factors': top_retention_factors,
        'feature_shap_df': shap_df,
        'plot_path': plot_path
    }
    
    print(f"\n{'='*55}")
    print(f"Customer {customer_idx} Analysis")
    print(f"{'='*55}")
    print(f"Churn Probability: {churn_probability:.1%}")
    print(f"Prediction: {'⚠️ HIGH RISK' if predicted_churn else '✅ LOW RISK'}")
    
    print(f"\n🔴 Top factors INCREASING churn risk:")
    for _, row in top_churn_drivers.iterrows():
        print(f"   {row['feature']}: {row['value']:.2f} "
              f"(SHAP: +{row['shap_value']:.4f})")
    
    print(f"\n🟢 Top factors DECREASING churn risk:")
    for _, row in top_retention_factors.iterrows():
        print(f"   {row['feature']}: {row['value']:.2f} "
              f"(SHAP: {row['shap_value']:.4f})")
    
    return explanation


def generate_natural_language_explanation(explanation: dict, threshold: float = 0.20) -> str:
    """
    Convert SHAP explanation dictionary into a plain-English sentence.
    Template-based for now — replaced by Ollama LLM in Week 6.
    """
    prob = explanation['churn_probability']
    drivers = explanation['top_churn_drivers']
    retention = explanation['top_retention_factors']
    
    # risk_level is derived FROM predicted_churn directly, not from a
    # separate set of cutoffs — this guarantees it can never contradict
    # the HIGH RISK / LOW RISK flag shown elsewhere for the same customer.
        
    is_flagged = prob >= threshold
    if not is_flagged:
        risk_level = "low"
    elif prob < 0.5:
        risk_level = "moderate"
    elif prob < 0.7:
        risk_level = "high"
    else:
        risk_level = "extreme"
    
    explanation_text = f"This customer has a {risk_level} churn risk ({prob:.1%}). "
    
    if risk_level == "low":
        if len(retention) > 0:
            top_retention = retention.iloc[0]
            explanation_text += (
                f"This is primarily due to their {top_retention['feature']} "
                f"(value: {top_retention['value']:.1f}), which strongly "
                f"supports retention. "
            )
            if len(drivers) > 0:
                minor_driver = drivers.iloc[0]
                explanation_text += (
                    f"Their {minor_driver['feature']} (value: "
                    f"{minor_driver['value']:.1f}) is a minor contributing "
                    f"risk factor, but it's outweighed by the retention "
                    f"factors above."
                )
        else:
            explanation_text += "No significant risk factors were identified for this customer."
    else:
        if len(drivers) == 0:
            explanation_text += "No strong individual risk factors were identified, though the overall profile suggests risk."
        else:
            top_driver = drivers.iloc[0]
            explanation_text += (
                f"The primary driver is their {top_driver['feature']} "
                f"(value: {top_driver['value']:.1f}), which increases "
                f"churn likelihood. "
            )
            if len(drivers) > 1:
                second_driver = drivers.iloc[1]
                explanation_text += (
                    f"Additionally, their {second_driver['feature']} "
                    f"(value: {second_driver['value']:.1f}) contributes "
                    f"to this risk."
                )
            if risk_level in ("moderate", "high") and len(retention) > 0:
                top_retention = retention.iloc[0]
                explanation_text += (
                    f" This is partially offset by their "
                    f"{top_retention['feature']} (value: "
                    f"{top_retention['value']:.1f}), which supports retention."
                )
    
    return explanation_text

def run_global_shap_analysis():
    """Main function — runs full global SHAP analysis"""
    
    # Load data and model
    data = prepare_churn_data()
    X_test = data['X_test']
    y_test = data['y_test']
    feature_cols = data['feature_cols']
    
    model = joblib.load('models/churn_model_tuned.pkl')
    
    # Compute SHAP values
    shap_values, explainer = compute_shap_values(model, X_test)
    
    # Save explainer for later use in Streamlit
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    print("✅ SHAP explainer saved to models/shap_explainer.pkl")
    
    # Global plots
    plot_global_feature_importance(shap_values, X_test)
    
    # Feature importance table
    importance_df = get_feature_importance_table(shap_values, X_test)
    importance_df.to_csv('data/processed/feature_importance.csv', index=False)
    print("✅ Feature importance table saved")
    
    # Save SHAP values for later use
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=[f'shap_{col}' for col in X_test.columns]
    )
    shap_df.to_csv('data/processed/shap_values.csv', index=False)
    print("✅ SHAP values saved to data/processed/shap_values.csv")
    
    return shap_values, explainer, importance_df

if __name__ == "__main__":
    shap_values, explainer, importance_df = run_global_shap_analysis()