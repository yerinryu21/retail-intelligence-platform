import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data
from src.shared.evaluation import evaluate_model

def tune_xgboost():
    """Hyperparameter tuning for XGBoost churn model"""

    data = prepare_churn_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # ── Calculate class weight ─────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # ── Parameter grid ─────────────────────────────────────────────
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # ── Base model ─────────────────────────────────────────────────
    base_model = XGBClassifier(
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='aucpr',
        verbosity=0
    )

    # ── Cross-validation strategy ──────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── PR-AUC scorer ──────────────────────────────────────────────
    # NOTE: newer sklearn versions replaced needs_proba= with response_method=
    pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')

    # ── Randomized search ─────────────────────────────────────────
    print("Starting hyperparameter search (this may take 10-20 minutes)...")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring=pr_auc_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    # ── Results ───────────────────────────────────────────────────
    print(f"\n✅ Best PR-AUC (cross-validated): {search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    # ── Evaluate best model on test set ───────────────────────────
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    result = evaluate_model("XGBoost (Tuned)", y_test, y_pred, y_prob)

    # ── Save tuned model ───────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model_tuned.pkl')
    print("\n✅ Tuned model saved to models/churn_model_tuned.pkl")

    return best_model, result, search.best_params_

if __name__ == "__main__":
    model, result, best_params = tune_xgboost()