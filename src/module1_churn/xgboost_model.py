import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data
from src.shared.evaluation import evaluate_model, compare_models, plot_precision_recall_curve

def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost.
    Formula: count(negative class) / count(positive class)
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    weight = neg / pos
    print(f"scale_pos_weight: {weight:.2f} (neg:{neg} / pos:{pos})")
    return weight

def train_xgboost_models():
    """Train XGBoost with two imbalance handling strategies"""

    data = prepare_churn_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    results = []
    models = {}

    # ── Strategy 1: scale_pos_weight ──────────────────────────────
    print("\n--- Strategy 1: XGBoost with scale_pos_weight ---")

    spw = calculate_scale_pos_weight(y_train)

    xgb_weighted = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='aucpr',
        verbosity=0
    )
    xgb_weighted.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    pred_weighted = xgb_weighted.predict(X_test)
    prob_weighted = xgb_weighted.predict_proba(X_test)[:, 1]

    result = evaluate_model("XGBoost (scale_pos_weight)",
                           y_test, pred_weighted, prob_weighted)
    results.append(result)
    models['xgb_weighted'] = xgb_weighted

    # ── Strategy 2: SMOTE ─────────────────────────────────────────
    print("\n--- Strategy 2: XGBoost with SMOTE ---")

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {pd.Series(y_train_smote).value_counts().to_dict()}")

    xgb_smote = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='aucpr',
        verbosity=0
    )
    xgb_smote.fit(X_train_smote, y_train_smote)

    pred_smote = xgb_smote.predict(X_test)
    prob_smote = xgb_smote.predict_proba(X_test)[:, 1]

    result = evaluate_model("XGBoost (SMOTE)",
                           y_test, pred_smote, prob_smote)
    results.append(result)
    models['xgb_smote'] = xgb_smote

    # ── Compare all models ─────────────────────────────────────────
    comparison = compare_models(results)

    # ── Plot PR curves for both ────────────────────────────────────
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(10, 6))
    for name, prob in [("scale_pos_weight", prob_weighted),
                        ("SMOTE", prob_smote)]:
        precision, recall, _ = precision_recall_curve(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        plt.plot(recall, precision, label=f'XGBoost {name} (PR-AUC={pr_auc:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('XGBoost: Comparing Imbalance Handling Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('notebooks/xgboost_comparison.png')
    plt.show()

    # ── Save best model ────────────────────────────────────────────
    best_model_name = max(results, key=lambda x: x['pr_auc'])['model']
    print(f"\n✅ Best model: {best_model_name}")

    best_model = xgb_weighted if 'scale_pos_weight' in best_model_name else xgb_smote

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model.pkl')
    joblib.dump(data['scaler'], 'models/churn_scaler.pkl')
    print("✅ Best model saved to models/churn_model.pkl")

    return best_model, data, results

if __name__ == "__main__":
    model, data, results = train_xgboost_models()