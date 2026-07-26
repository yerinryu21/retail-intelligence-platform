import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def build_final_evaluation_table() -> pd.DataFrame:
    """
    Build comprehensive evaluation table combining:
    - Prophet vs baseline metrics
    - Uncertainty metrics
    - Seasonality analysis
    - Inventory risk
    """
    
    baseline = pd.read_csv('data/processed/baseline_comparison.csv')
    uncertainty = pd.read_csv('data/processed/uncertainty_metrics.csv')
    seasonality = pd.read_csv('data/processed/seasonality_analysis.csv')
    inventory = pd.read_csv('data/processed/inventory_risk.csv')
    
    # baseline already carries DataQualityFlag (added Day 4) - avoid duplicate
    # columns on merge by dropping it from the others where we don't need it twice
    eval_table = baseline.merge(
        uncertainty[['StockCode', 'MeanRelativeUncertainty', 'MeanForecast']], 
        on='StockCode', how='left'
    )
    eval_table = eval_table.merge(
        seasonality[['StockCode', 'TrendDirection', 'TrendChangeUnits', 'TrendChangeVsMean%']],
        on='StockCode', how='left'
    )
    eval_table = eval_table.merge(
        inventory[['StockCode', 'RiskStatus', 'Description']],
        on='StockCode', how='left'
    )
    
    # Improvement % over baselines - safe here since Naive_MAE/SeasonalNaive_MAE
    # are computed from real demand values, not the yhat-clipping-to-zero issue
    # that affected earlier metrics (see Day 3 notes)
    eval_table['ImprovementOverNaive%'] = (
        (eval_table['Naive_MAE'] - eval_table['Prophet_MAE']) 
        / eval_table['Naive_MAE'] * 100
    ).round(1)
    
    eval_table['ImprovementOverSeasonal%'] = (
        (eval_table['SeasonalNaive_MAE'] - eval_table['Prophet_MAE']) 
        / eval_table['SeasonalNaive_MAE'] * 100
    ).round(1)
    
    eval_table = eval_table.sort_values('Prophet_MAE')
    
    numeric_cols = ['Prophet_MAE', 'Naive_MAE', 'SeasonalNaive_MAE',
                    'Prophet_MAPE', 'Naive_MAPE', 'SeasonalNaive_MAPE',
                    'MeanRelativeUncertainty', 'MeanForecast']
    
    for col in numeric_cols:
        if col in eval_table.columns:
            eval_table[col] = eval_table[col].round(2)
    
    os.makedirs('data/processed', exist_ok=True)
    eval_table.to_csv('data/processed/final_evaluation_table.csv', index=False)
    print("Saved to data/processed/final_evaluation_table.csv")
    
    return eval_table

def print_evaluation_summary(eval_table: pd.DataFrame) -> None:
    """Print a clean summary of model performance, split by data quality flag"""
    
    reliable = eval_table[eval_table['DataQualityFlag'] == 'OK']
    flagged = eval_table[eval_table['DataQualityFlag'] != 'OK']
    
    print("\n" + "="*70)
    print("DEMAND FORECASTING MODEL EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nOverall:")
    print(f"  Products evaluated: {len(eval_table)} ({len(reliable)} OK, {len(flagged)} flagged)")
    
    print(f"\nWin rate (OK products only, n={len(reliable)}):")
    print(f"  Prophet beats naive baseline: "
          f"{reliable['BeatNaive'].sum()}/{len(reliable)} products")
    print(f"  Prophet beats seasonal naive: "
          f"{reliable['BeatSeasonalNaive'].sum()}/{len(reliable)} products")
    
    print(f"\nAverage Error Metrics (OK products only):")
    print(f"  Prophet  - MAE: {reliable['Prophet_MAE'].mean():.2f} | "
          f"MAPE: {reliable['Prophet_MAPE'].mean():.1f}%")
    print(f"  Naive    - MAE: {reliable['Naive_MAE'].mean():.2f} | "
          f"MAPE: {reliable['Naive_MAPE'].mean():.1f}%")
    print(f"  Seasonal - MAE: {reliable['SeasonalNaive_MAE'].mean():.2f} | "
          f"MAPE: {reliable['SeasonalNaive_MAPE'].mean():.1f}%")
    
    print(f"\nAverage Improvement (OK products only):")
    print(f"  Over naive baseline:    "
          f"{reliable['ImprovementOverNaive%'].mean():.1f}%")
    print(f"  Over seasonal naive:    "
          f"{reliable['ImprovementOverSeasonal%'].mean():.1f}%")
    
    print(f"\nBest performing products (lowest Prophet MAE, OK only):")
    best = reliable.nsmallest(3, 'Prophet_MAE')
    for _, row in best.iterrows():
        print(f"  {row['StockCode']}: MAE={row['Prophet_MAE']:.1f}, "
              f"MAPE={row['Prophet_MAPE']:.1f}%, "
              f"Trend={row['TrendDirection']}")
    
    print(f"\nWorst performing products (highest Prophet MAE, OK only):")
    worst = reliable.nlargest(3, 'Prophet_MAE')
    for _, row in worst.iterrows():
        print(f"  {row['StockCode']}: MAE={row['Prophet_MAE']:.1f}, "
              f"MAPE={row['Prophet_MAPE']:.1f}%, "
              f"Trend={row['TrendDirection']}, "
              f"BeatNaive={row['BeatNaive']}")
    
    print(f"\nMost uncertain products (highest relative uncertainty, OK only):")
    uncertain = reliable.nlargest(3, 'MeanRelativeUncertainty')
    for _, row in uncertain.iterrows():
        print(f"  {row['StockCode']}: "
              f"Uncertainty={row['MeanRelativeUncertainty']:.1f}% of mean, "
              f"Inventory={row['RiskStatus']}")
    
    if len(flagged) > 0:
        print(f"\nFlagged products (excluded from averages above):")
        for _, row in flagged.iterrows():
            print(f"  {row['StockCode']}: {row['DataQualityFlag']}, "
                  f"Prophet_MAE={row['Prophet_MAE']:.1f} (unreliable - not comparable)")
    
    print("\n" + "="*70)
    
    print(f"\nAverage Improvement (OK products only):")
    print(f"  Over naive baseline    - mean: {reliable['ImprovementOverNaive%'].mean():.1f}%  |  median: {reliable['ImprovementOverNaive%'].median():.1f}%")
    print(f"  Over seasonal naive    - mean: {reliable['ImprovementOverSeasonal%'].mean():.1f}%  |  median: {reliable['ImprovementOverSeasonal%'].median():.1f}%")

def plot_final_evaluation(eval_table: pd.DataFrame,
                           save_dir: str = 'notebooks/demand_plots') -> None:
    """Generate final evaluation visualizations, split by data quality"""
    
    os.makedirs(save_dir, exist_ok=True)
    reliable = eval_table[eval_table['DataQualityFlag'] == 'OK']
    
    fig1 = go.Figure()
    x = reliable['StockCode']
    
    fig1.add_trace(go.Bar(name='Prophet', x=x, y=reliable['Prophet_MAE'], marker_color='#1f77b4'))
    fig1.add_trace(go.Bar(name='Naive Baseline', x=x, y=reliable['Naive_MAE'], marker_color='#d62728'))
    fig1.add_trace(go.Bar(name='Seasonal Naive', x=x, y=reliable['SeasonalNaive_MAE'], marker_color='#ff7f0e'))
    
    fig1.update_layout(
        barmode='group',
        title='MAE by Product: Prophet vs Baselines (OK products only, lower is better)',
        xaxis_title='Product',
        yaxis_title='Mean Absolute Error (units)',
        height=500
    )
    
    fig1.write_html(f'{save_dir}/mae_comparison.html')
    fig1.show()
    
    fig2 = px.scatter(
        reliable,
        x='ImprovementOverNaive%',
        y='ImprovementOverSeasonal%',
        text='StockCode',
        title='Prophet Improvement: vs Naive (x-axis) vs Seasonal Naive (y-axis)',
        labels={
            'ImprovementOverNaive%': 'Improvement over Naive (%)',
            'ImprovementOverSeasonal%': 'Improvement over Seasonal Naive (%)'
        },
        color='TrendDirection',
        color_discrete_map={'Growing': 'green', 'Declining': 'red'}
    )
    fig2.add_hline(y=0, line_dash='dash', line_color='gray')
    fig2.add_vline(x=0, line_dash='dash', line_color='gray')
    fig2.update_traces(textposition='top center')
    fig2.update_layout(height=500)
    fig2.show()
    
    mape_df = pd.DataFrame({
        'Model': ['Prophet', 'Naive Baseline', 'Seasonal Naive'],
        'Mean MAPE (%)': [
            reliable['Prophet_MAPE'].mean(),
            reliable['Naive_MAPE'].mean(),
            reliable['SeasonalNaive_MAPE'].mean()
        ]
    })
    
    fig3 = px.bar(
        mape_df, x='Model', y='Mean MAPE (%)',
        title='Average MAPE Across OK Products (lower is better)',
        color='Model',
        color_discrete_map={'Prophet': '#1f77b4', 'Naive Baseline': '#d62728', 'Seasonal Naive': '#ff7f0e'},
        text='Mean MAPE (%)'
    )
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(height=400)
    fig3.show()

if __name__ == "__main__":
    eval_table = build_final_evaluation_table()
    print_evaluation_summary(eval_table)
    plot_final_evaluation(eval_table)