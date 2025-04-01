#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:08:33 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:43:12 2024

@author: danielhier

This script calculates the ROC curve and AUC for predicting whether GPT-4 correctly matches
a protein name to its gene name, using the gene name (GN) frequency in PMC as the predictor.

It uses Youden’s J statistic to identify the optimal cutpoint, and saves a plot of the ROC curve.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Make paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'GN_with_matches.csv')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '..', 'results', 'roc_gn_pmc_id.png')

# Step 1: Load the CSV
df = pd.read_csv(INPUT_FILE)

# Step 2: Clean data — remove rows with NaN or inf in relevant columns
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['PMC_ID', 'MATCH'])

# Step 3: Create binary target — MATCH > 0 → 1, else 0
df['MATCH_BINARY'] = (df['MATCH'] > 0).astype(int)

# Step 4: Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(df['MATCH_BINARY'], df['PMC_ID'])
roc_auc = auc(fpr, tpr)

# Step 5: Identify optimal threshold (Youden’s J)
j_scores = tpr - fpr
optimal_idx = j_scores.argmax()
optimal_threshold = thresholds[optimal_idx]

# Step 6: Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, label=f'Optimal cutoff = {optimal_threshold:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
plt.title('ROC Curve for MATCH Prediction Using GN PMC Frequency', fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()

# Step 7: Save plot to file
plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight')
plt.show()

# Optional: Print diagnostic info
print(f"AUC: {roc_auc:.4f}")
print(f"Optimal Threshold (Youden's J): {optimal_threshold:.2f}")
print(f"ROC curve saved to: {OUTPUT_FILE}")