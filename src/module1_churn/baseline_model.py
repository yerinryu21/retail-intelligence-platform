import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data
from src.shared.evaluation import evaluate_model, compare_models, plot_precision_recall_curve

def train_baseline_models():
    """Train and evaluate baseline models"""

    data = prepare_churn_data()
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']

    results = []

    print("\nTraining Dummy Classifier (majority class baseline)...")
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]

    result = evaluate_model("Dummy (Majority Class)", y_test, dummy_pred, dummy_prob)
    results.append(result)

    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    result = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
    results.append(result)

    plot_precision_recall_curve(
        "Logistic Regression", y_test, lr_prob,
        save_path='notebooks/lr_precision_recall.png'
    )

    comparison = compare_models(results)

    print("\n✅ Baseline training complete")
    print("Goal: XGBoost should beat Logistic Regression on F1 and PR-AUC")

    return lr, data

if __name__ == "__main__":
    lr_model, data = train_baseline_models()