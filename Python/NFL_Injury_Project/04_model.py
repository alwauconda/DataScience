# 04_model.py
# Step 4: Train and evaluate ML models on the preprocessed data
# Run after 03_preprocess.py. Outputs evaluation metrics and feature importance charts.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading model-ready data...")
df = pd.read_csv("data/model_ready.csv")
print(f"  {len(df):,} rows | Injury rate: {df['injured'].mean():.2%}")

# ── Define features and target ────────────────────────────────────────────────
DROP_COLS = ["injured", "gsis_id", "position", "position_y"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].fillna(0)
y = df["injured"]

print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── Train / test split ────────────────────────────────────────────────────────
# Train on 2017-2021, test on 2022-2023 (temporal split — more realistic)
train_mask = df["season"] <= 2021
test_mask  = df["season"] >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train):,} rows ({y_train.mean():.2%} injury rate)")
print(f"Test:  {len(X_test):,} rows  ({y_test.mean():.2%} injury rate)")

# ── Apply SMOTE to handle class imbalance ─────────────────────────────────────
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Balanced training set: {len(X_train_bal):,} rows ({y_train_bal.mean():.2%} injury rate)")

# ── Helper: evaluate and save results ────────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_te, y_pred, target_names=["Not Injured", "Injured"]))
    print(f"  ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred,
        display_labels=["Not Injured", "Injured"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/{fname}_confusion_matrix.png")
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax, name=name)
    ax.set_title(f"{name} — ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}_roc_curve.png")
    plt.close()

    return model

# ── 1. Logistic Regression (baseline) ────────────────────────────────────────
print("\nTraining Logistic Regression (baseline)...")
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
lr_model = evaluate_model(
    "Logistic Regression",
    lr_pipeline,
    X_train_bal, y_train_bal,
    X_test, y_test
)

# ── 2. Random Forest ──────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_train_bal, y_train_bal,
    X_test, y_test
)

# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model = evaluate_model(
    "XGBoost",
    XGBClassifier(n_estimators=100, random_state=42,
                  eval_metric="logloss", verbosity=0),
    X_train_bal, y_train_bal,
    X_test, y_test
)

# ── Feature Importance (Random Forest) ───────────────────────────────────────
print("\nGenerating feature importance chart...")
rf = rf_model  # already fitted
importances = pd.Series(
    rf.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Top 15 Feature Importances — Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()
print("  Saved → outputs/feature_importance.png")

print("\n✅ Modeling complete! Check /outputs for charts and metrics.")
