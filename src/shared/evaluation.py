import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
    average_precision_score, f1_score,
    precision_score, recall_score
)

def evaluate_model(model_name: str,
                   y_true: pd.Series,
                   y_pred: np.ndarray,
                   y_prob: np.ndarray) -> dict:
    """
    Comprehensive evaluation of a churn classification model.
    Returns a dictionary of all metrics.
    """

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\nCore Metrics:")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}  ← most important for imbalanced data")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Active', 'Churned']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  True Negatives (correctly predicted Active):  {tn}")
    print(f"  False Positives (Active predicted as Churned): {fp}")
    print(f"  False Negatives (Churned missed):              {fn}")
    print(f"  True Positives (correctly predicted Churned): {tp}")

    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def plot_precision_recall_curve(model_name: str,
                                 y_true: pd.Series,
                                 y_prob: np.ndarray,
                                 save_path: str = None):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compare_models(results: list) -> pd.DataFrame:
    """Compare multiple model results in a table"""
    df = pd.DataFrame(results)
    df = df.set_index('model')
    df = df[['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']]
    df = df.round(4)
    print("\n📊 Model Comparison:")
    print(df.to_string())
    return df


# ── Standalone sanity check ──────────────────────────────────────
# This block only runs if you execute this file directly
# (python src/shared/evaluation.py). It's NOT part of the real
# pipeline — just confirms the functions work using fake data.
if __name__ == "__main__":
    print("Running evaluation.py sanity check with dummy data...\n")

    rng = np.random.default_rng(42)
    y_true_fake = pd.Series(rng.integers(0, 2, size=100))
    y_prob_fake = rng.random(100)
    y_pred_fake = (y_prob_fake >= 0.5).astype(int)

    evaluate_model("Sanity Check Model", y_true_fake, y_pred_fake, y_prob_fake)
    print("\n✅ evaluation.py functions work correctly.")