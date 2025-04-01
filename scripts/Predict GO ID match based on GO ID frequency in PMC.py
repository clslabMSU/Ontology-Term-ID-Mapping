#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:45:03 2025

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting Match Accuracy Using GO ID Frequency

This script performs logistic regression to predict whether GPT-4 correctly matched
a GO term to its GO ID, based on:

    - GO_ID_PMC_count
    - GO_Term_PMC_count
    - Count (term frequency)

It assumes the input data is in data/go_terms_with_normalizations_complete.csv
and outputs model metrics to the console.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import norm

# Construct path to the CSV file relative to the script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'go_terms_with_normalizations_complete.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(data_path)

# Define features and target
X = df[['GO_ID_PMC_count', 'GO_Term_PMC_count', 'Count']]
y = df['Match']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_normalized, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test_normalized)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display coefficients and intercept
coef = log_reg.coef_[0]
intercept = log_reg.intercept_[0]
features = ['GO_ID_PMC_count', 'GO_Term_PMC_count', 'Count']

print("\nLogistic Regression Coefficients:")
for feature, weight in zip(features, coef):
    print(f"{feature}: {weight:.4f}")
print(f"Intercept: {intercept:.4f}")

# Generate Probit Curve (vary GO_ID_PMC_count while holding others constant)
GO_ID_range = np.linspace(X_train['GO_ID_PMC_count'].min(), X_train['GO_ID_PMC_count'].max(), 500).reshape(-1, 1)
GO_ID_range_normalized = scaler.transform(np.hstack([
    GO_ID_range,
    np.full_like(GO_ID_range, X_train['GO_Term_PMC_count'].mean()),
    np.full_like(GO_ID_range, X_train['Count'].mean())
]))

y_prob = log_reg.predict_proba(GO_ID_range_normalized)[:, 1]
y_probit = norm.ppf(y_prob)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(y_probit, y_prob, color='blue', label='Probit-Probability Curve')
plt.xlabel('Probit')
plt.ylabel('Probability')
plt.title('Probit vs. Probability Curve for Logistic Regression Model')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()