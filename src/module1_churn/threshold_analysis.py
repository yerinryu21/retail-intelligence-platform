import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module1_churn.data_preparation import prepare_churn_data

def analyze_thresholds(
    cost_false_positive: float = 5.0,
    cost_false_negative: float = 50.0
):
    """
    Analyze different classification thresholds and their business cost.
    """

    data = prepare_churn_data()
    X_test = data['X_test']
    y_test = data['y_test']

    model = joblib.load('models/churn_model_tuned.pkl')
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)

        results.append({
            'threshold': round(threshold, 2),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'total_cost': total_cost
        })

    results_df = pd.DataFrame(results)

    best_f1_threshold = results_df.loc[results_df['f1'].idxmax(), 'threshold']
    best_cost_threshold = results_df.loc[results_df['total_cost'].idxmin(), 'threshold']

    print(f"Threshold with best F1:           {best_f1_threshold}")
    print(f"Threshold with lowest business cost: {best_cost_threshold}")
    print(f"\nFull threshold analysis:")
    print(results_df[['threshold', 'precision', 'recall', 'f1', 'total_cost']].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df['threshold'], results_df['precision'], label='Precision')
    axes[0].plot(results_df['threshold'], results_df['recall'], label='Recall')
    axes[0].plot(results_df['threshold'], results_df['f1'], label='F1', linewidth=2)
    axes[0].axvline(best_f1_threshold, color='green', linestyle='--',
                    label=f'Best F1 threshold ({best_f1_threshold})')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics vs Classification Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df['threshold'], results_df['total_cost'],
                color='red', linewidth=2)
    axes[1].axvline(best_cost_threshold, color='blue', linestyle='--',
                    label=f'Min cost threshold ({best_cost_threshold})')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Total Business Cost ($)')
    axes[1].set_title(f'Business Cost vs Threshold\n(FP cost=${cost_false_positive}, FN cost=${cost_false_negative})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/threshold_analysis.png')
    plt.show()

    optimal_threshold = best_cost_threshold
    np.save('models/optimal_threshold.npy', optimal_threshold)
    print(f"\n✅ Optimal threshold {optimal_threshold} saved")

    return results_df, optimal_threshold

if __name__ == "__main__":
    results_df, optimal_threshold = analyze_thresholds(
        cost_false_positive=5.0,
        cost_false_negative=50.0
    )