#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gn_predict_match_from_id_frequency.py

Analyzes the relationship between gn term frequency in PMC and GPT-4 normalization accuracy.
Includes binning, ROC curve, precision-recall curve, Zipf plot, correlation, and regression.

Author: Daniel Hier
Date: 2025-04-01
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(input_path):
    return pd.read_csv(input_path)

def bin_data(df):
    df_sorted = df.sort_values(by='gn_id_pmc', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    df_sorted['Bin'] = pd.qcut(df_sorted['Rank'], q=20, labels=False) + 1
    return df_sorted

def plot_bin_summary(df_sorted, output_path):
    bin_means = df_sorted.groupby('Bin').agg(
        mean_gn_match=('gn_match', 'mean'),
        mean_gn_id_pmc=('gn_id_pmc', 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Bin', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Mean gn_match', color='blue', fontweight='bold', fontsize=12)
    ax1.plot(bin_means['Bin'], bin_means['mean_gn_match'], color='blue', linewidth=2, marker='o', markersize=8)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(1, 21))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean gn_id_pmc (Log Scale)', color='green', fontweight='bold', fontsize=12)
    ax2.set_yscale('log')
    ax2.plot(bin_means['Bin'], bin_means['mean_gn_id_pmc'], color='green', linewidth=2, marker='o', markersize=8)
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title('Mean gn_match and gn_id_pmc by Bin (Log Scale on Y2)', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_logistic_regression(df, output_path, low_freq_thresh=10, sample_rate=1):
    df = df.copy()
    df = df[df['gn_match'].isin([0, 1])]

    # --- Undersample where gn_id_pmc < low_freq_thresh ---
    df_high = df[df['gn_id_pmc'] >= low_freq_thresh]
    df_low = df[df['gn_id_pmc'] < low_freq_thresh].iloc[::sample_rate]
    df = pd.concat([df_high, df_low])

    # --- Compute log-transformed frequency ---
    df['log_freq'] = np.log1p(df['gn_id_pmc'])  # log1p to avoid log(0)
    X = sm.add_constant(df['log_freq'])  # Add intercept
    y = df['gn_match']

    # --- Logistic regression model ---
    logit_model = sm.Logit(y, X).fit(disp=False)
    df['predicted_proba'] = logit_model.predict(X)

    # --- Adaptive Jittering ---
    np.random.seed(42)
    jitter_strength_log = 0.3
    jitter_strength_proba = 0.05

    df['jittered_log_freq'] = df['log_freq'] + np.random.normal(
        0, jitter_strength_log / (df['gn_id_pmc'] + 1), size=len(df)
    )
    df['jittered_proba'] = df['predicted_proba'] + np.random.normal(
        0, jitter_strength_proba, size=len(df)
    )
    df['jittered_proba'] = np.clip(df['jittered_proba'], 0, 1)

    # --- Colors for match ---
    colors = df['gn_match'].map({1: 'green', 0: 'red'})

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df['jittered_log_freq'], df['jittered_proba'],
                c=colors, s=15, alpha=0.6)

    # --- Regression curve ---
    sorted_df = df.sort_values(by='log_freq')
    plt.plot(sorted_df['log_freq'], sorted_df['predicted_proba'],
             color='blue', linewidth=2, label='Logistic Regression Fit')
    
    # --- Vertical line at undersampling threshold ---
    undersample_line = np.log1p(low_freq_thresh)
    plt.axvline(x=undersample_line, color='gray', linestyle='--', linewidth=1.5, label='Undersampling Threshold')
    
    # --- Legend handles ---
    plt.scatter([], [], color='green', label='Match (1)', s=30)
    plt.scatter([], [], color='red', label='No Match (0)', s=30)
    
    # --- Labels and layout ---
    plt.xlabel('Log1p(gn_id_pmc)', fontweight='bold')
    plt.ylabel('Predicted Probability of Match', fontweight='bold')
    plt.title('Logistic Regression: Predicting Gene Name Match from Frequency', fontweight='bold')
    plt.ylim(0, 1.05)  # <-- Y-axis extended to handle values > 1.0
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.scatter(df['jittered_log_freq'], df['jittered_proba'],
                c=colors, s=15, alpha=0.6)
    
    plt.plot(sorted_df['log_freq'], sorted_df['predicted_proba'],
             color='blue', linewidth=2, label='Logistic Regression Fit')
    
    plt.axvline(x=undersample_line, color='gray', linestyle='--', linewidth=1.5, label='Undersampling Threshold')
    plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Ceiling = 1.0')
    
    plt.xlabel('Log1p(gn_id_pmc)', fontweight='bold')
    plt.ylabel('Predicted Probability of Match', fontweight='bold')
    plt.title('Logistic Regression: Predicting Gene Name Match from Frequency', fontweight='bold')
    plt.ylim(0, 1.2)
    plt.grid(True)
    plt.legend(loc='lower right')
    
    
    plt.show()
    plt.close()




    # Print model summary
    print(logit_model.summary())

    return df

def plot_zipf(df, output_path, csv_output_path=None):
    """
    Plot a Zipf plot of gene name (gn) term frequency vs. rank (log-log),
    colored by match status and undersampled for long tail.

    Parameters:
        df (pd.DataFrame): Data with 'gn_id_pmc' and 'gn_match' columns.
        output_path (Path): File path to save the plot.
        csv_output_path (Path, optional): If provided, saves filtered DataFrame.
    """
    # --- Sort and Rank ---
    df_sorted = df.sort_values(by='gn_id_pmc', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1

    # --- Undersample ---
    df_filtered = pd.concat([
        df_sorted[df_sorted['rank'] <= 2000],
        df_sorted[df_sorted['rank'] > 2000][::10]
    ])

    print(f"Total rows: {len(df_sorted)}")
    print(f"Filtered rows after undersampling: {len(df_filtered)}")

    # --- Drop rows with NaN in gn_match ---
    df_filtered = df_filtered[df_filtered['gn_match'].notna()]

    # --- Log Transforms ---
    epsilon = 0.9
    df_filtered['log_rank'] = np.log10(df_filtered['rank'])
    df_filtered['log_freq'] = np.log10(df_filtered['gn_id_pmc'] + epsilon)

    # --- Jitter ---
    np.random.seed(42)
    jitter_strength = 0.1
    df_filtered['log_rank_jittered'] = df_filtered['log_rank'] + np.random.normal(0, jitter_strength, len(df_filtered))
    df_filtered['log_freq_jittered'] = df_filtered['log_freq'] + np.random.normal(0, jitter_strength, len(df_filtered))




    # --- Drop rows with missing or unexpected gn_match values ---
    df_filtered = df_filtered[df_filtered['gn_match'].isin([0, 1])]
    
    # --- Color and Transparency ---
    color_map = df_filtered['gn_match'].map({1: 'green', 0: 'red'})
    alpha_map = df_filtered['gn_match'].map({1: 0.8, 0: 0.4})


  

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered['log_rank_jittered'], df_filtered['log_freq_jittered'],
                c=color_map, alpha=alpha_map, s=12, edgecolor='none')

    # Optional trend line
    z = np.polyfit(df_filtered['log_rank'], df_filtered['log_freq'], 1)
    p = np.poly1d(z)
    plt.plot(df_filtered['log_rank'], p(df_filtered['log_rank']), "k--", alpha=0.4, label='Trend Line')

    # --- Formatting ---
    plt.xlabel('Log10(Rank)', fontweight='bold')
    plt.ylabel('Log10(Frequency + 0.9)', fontweight='bold')
    plt.title('Zipf Plot: gn ID Frequency vs. Rank (Jittered & Undersampled)', fontweight='bold')
    plt.grid(True)

    # --- Legend ---
    handles = [
        plt.Line2D([], [], color='green', marker='o', linestyle='', label='Match (1)', markersize=6),
        plt.Line2D([], [], color='red', marker='o', linestyle='', label='No Match (0)', markersize=6),
        plt.Line2D([], [], color='black', linestyle='--', label='Trend Line')
    ]
    plt.legend(handles=handles, loc='upper right')

    # --- Save plot ---
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Optional CSV output ---
    if csv_output_path:
        df_filtered.to_csv(csv_output_path, index=False)

def run_regression(df):
    X = df[['gn_id_pmc', 'gn_id_pmc']]  # Duplicate col for placeholder multivariate
    y = df['gn_match']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model.intercept_, model.coef_, r2


def compute_and_plot_roc(df, output_path, score_col, label_col='gn_match'):
    fpr, tpr, thresholds = roc_curve(df[label_col], df[score_col])
    roc_auc = auc(fpr, tpr)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    cutpoint = thresholds[optimal_idx]

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', label=f'Cutpoint = {cutpoint:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label_col.replace("_match", "")} Normalization Prediction')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return cutpoint, roc_auc


def compute_and_plot_pr(df, output_path, score_col, label_col):
    precision, recall, _ = precision_recall_curve(df[label_col], df[score_col])
    ap = average_precision_score(df[label_col], df[score_col])

    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Normalization')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return ap
def main():
    # --- Setup paths ---
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    input_path = repo_root / "data" / "gn_with_matches.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Load and inspect data ---
    df = load_data(input_path)
    print(df.info())
    print(f"Looking for file at: {input_path}")
    
    # --- Bin and summarize ---
    df_sorted = bin_data(df)
    plot_bin_summary(df_sorted, results_dir / "gn_id_pmc_analysis_logscale_ac.png")

    # --- Zipf plot ---
    plot_zipf(df, results_dir / "zipf_plot_gn_id_pmc.png")

    # --- Correlation ---
    corr = df['gn_match'].corr(df['gn_id_pmc'])
    print(f"Correlation with gn_id_pmc: {corr:.3f}")
    
    # --- Logistic regression plot ---
    df = plot_logistic_regression(df, results_dir / "logit_plot_gn.png")

    # --- Multiple regression ---
    intercept, coef, r2 = run_regression(df)
    print("Multiple Regression Results:")
    print(f"Intercept: {intercept:.4f}")
    print(f"Coefficient for gn_id_pmc: {coef[0]:.4f}")
    print(f"R² Score: {r2:.4f}")

    # --- ROC curve ---
    cutpoint, auc_score = compute_and_plot_roc(
        df,
        output_path=results_dir / "roc_curve_gn_match.png",
        score_col='predicted_proba',
        label_col='gn_match'
    )
    print(f"ROC AUC: {auc_score:.3f}, Optimal Cutpoint: {cutpoint:.2f}")

    # --- Precision-Recall curve ---
    ap = compute_and_plot_pr(
        df,
        output_path=results_dir / "pr_curve_gn_match.png",
        score_col='predicted_proba',
        label_col='gn_match'
    )
    print(f"Average Precision (AP): {ap:.3f}")

# If running from terminal or script, use this:
if __name__ == "__main__":
    main()