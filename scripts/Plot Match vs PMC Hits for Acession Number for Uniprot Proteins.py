#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:35:08 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:57:43 2024
@author: danielhier

This script analyzes whether the frequency of protein UniProt Accession Numbers (AC) in PMC predicts
the accuracy of mapping protein names to their accession number. The proteins are ranked by AC frequency
and divided into 20 bins. The script then calculates the mean match rate per bin and plots accuracy vs. frequency.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Make the script path-agnostic by using relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'AC_with_PMC_HITS_to_MATCH.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'results', 'pmc_id_count_ac_analysis.png')

# Step 1: Load the CSV file
df = pd.read_csv(DATA_PATH)

# Step 2: Sort the DataFrame by 'PMC_ID_COUNT_AC'
df_sorted = df.sort_values(by='PMC_ID_COUNT_AC', ascending=False).reset_index(drop=True)

# Step 3: Assign rank and divide into 20 bins
df_sorted['Rank'] = df_sorted.index + 1
df_sorted['Bin'] = pd.qcut(df_sorted['Rank'], q=20, labels=False) + 1  # Bins 1–20

# Step 4: Compute bin-level means
bin_means = df_sorted.groupby('Bin').agg(
    mean_MATCH=('MATCH', 'mean'),
    mean_PMC_ID_COUNT_AC=('PMC_ID_COUNT_AC', 'mean')
).reset_index()

# Step 5: Pearson correlation
pearson_r, p_value = pearsonr(df['PMC_ID_COUNT_AC'], df['MATCH'])
r_squared = pearson_r ** 2

# Step 6: Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot MATCH (accuracy)
ax1.set_xlabel('Bin', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy', color='blue', fontweight='bold', fontsize=12)
ax1.plot(bin_means['Bin'], bin_means['mean_MATCH'],
         color='blue', linewidth=2, marker='o', markersize=8)
ax1.tick_params(axis='y', labelcolor='blue')

# Plot PMC ID counts on second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Count of Accession Number (AC) in PMC', color='green', fontweight='bold', fontsize=12)
ax2.plot(bin_means['Bin'], bin_means['mean_PMC_ID_COUNT_AC'],
         color='green', linewidth=2, marker='o', markersize=8)
ax2.tick_params(axis='y', labelcolor='green')

# Final plot tweaks
ax1.set_xticks(range(1, 21))
plt.title('Match Rate vs. Accession Frequency by Bin', fontweight='bold', fontsize=14)
fig.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight')
plt.show()

# Console output
print(f"Pearson r between AC frequency and MATCH: {pearson_r:.3f}")
print(f"R-squared: {r_squared:.3f}")
print(f"P-value: {p_value:.3e}")
print(f"Plot saved to: {OUTPUT_PATH}")