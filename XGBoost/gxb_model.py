import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Add parent directory to path for shared utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils

# Load and encode dataset
csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')
df = pd.read_csv(csv_path)
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])

y = df['Churn']
X = df.drop('Churn', axis=1)

# Hyperparameters selected via GridSearchCV optimising F1 score
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.2,
    scale_pos_weight=3,
    subsample=0.8,
    colsample_bytree=1,
    random_state=42
)

# Train and evaluate over 5 random splits; collect Brier scores
br_values = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42 + i
    )

    start = time.perf_counter()
    xgb.fit(X_train, y_train)
    end = time.perf_counter()

    preds = xgb.predict(X_test)
    time_needed = end - start

    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    xgb_brier = brier_score_loss(y_test, y_prob_xgb)
    br_values.append(xgb_brier)

brier_mean = pd.Series(br_values).mean()

# Fit Platt (sigmoid) and isotonic calibrators on the last test fold
platt_calibrator = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
iso_calibrator = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
platt_calibrator.fit(X_test, y_test)
iso_calibrator.fit(X_test, y_test)

# Compute calibrated probabilities and Brier scores
prob_platt = platt_calibrator.predict_proba(X_test)[:, 1]
prob_iso = iso_calibrator.predict_proba(X_test)[:, 1]
brier_platt = brier_score_loss(y_test, prob_platt)
brier_iso = brier_score_loss(y_test, prob_iso)

print(f"Original Brier (mean over 5 splits): {brier_mean:.4f}")
print(f"Platt (Sigmoid) Brier: {brier_platt:.4f}")
print(f"Isotonic Brier: {brier_iso:.4f}")

# Build reliability (calibration) curves for all three variants
prob_true_orig, prob_pred_orig = calibration_curve(y_test, y_prob_xgb, n_bins=10)
prob_true_platt, prob_pred_platt = calibration_curve(y_test, prob_platt, n_bins=10)
prob_true_iso, prob_pred_iso = calibration_curve(y_test, prob_iso, n_bins=10)

# Plot reliability diagram
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Ideal)")
plt.plot(prob_pred_orig, prob_true_orig, "s-", color="red", alpha=0.8,
         label=f"Original XGBoost (Brier: {brier_mean:.4f})")
plt.plot(prob_pred_platt, prob_true_platt, "^-", color="blue", alpha=0.8,
         label=f"Platt Scaling (Brier: {brier_platt:.4f})")
plt.plot(prob_pred_iso, prob_true_iso, "o-", color="green", alpha=0.8,
         label=f"Isotonic Regression (Brier: {brier_iso:.4f})")

plt.ylabel("Actual Fraction of Churners (True Probability)", fontsize=12)
plt.xlabel("Mean Predicted Probability (Model Confidence)", fontsize=12)
plt.title("Reliability Diagram: Original vs. Calibrated XGBoost", fontsize=14, pad=15)
plt.legend(loc="upper left", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("calibration_comparison.png", dpi=300, bbox_inches='tight')
plt.show()