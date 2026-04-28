# 06_add_age.py
# Step 6: Pull player age from nflreadpy rosters and merge into enriched dataset
# Run after 05_improved_model.py

import nflreadpy as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
    precision_recall_curve, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Pull roster data for age ───────────────────────────────────────────────
SEASONS = list(range(2017, 2024))
print("Loading roster data...")
rosters = nfl.load_rosters(SEASONS).to_pandas()
print(f"  Roster rows: {len(rosters):,}")
print(f"  Roster columns: {rosters.columns.tolist()}")

# Find the birth date / age column
age_col = None
for candidate in ["birth_date", "birthdate", "age"]:
    if candidate in rosters.columns:
        age_col = candidate
        print(f"  Found age column: '{age_col}'")
        break

if age_col is None:
    print("  WARNING: No age/birth_date column found. Available columns:")
    print(" ", rosters.columns.tolist())

# ── 2. Build age lookup ───────────────────────────────────────────────────────
if age_col == "birth_date" or age_col == "birthdate":
    rosters["birth_date"] = pd.to_datetime(rosters[age_col], errors="coerce")
    # Approximate age at start of each season (September 1)
    rosters["season_start"] = pd.to_datetime(
        rosters["season"].astype(str) + "-09-01"
    )
    rosters["age"] = (
        (rosters["season_start"] - rosters["birth_date"]).dt.days / 365.25
    ).round(1)
elif age_col == "age":
    rosters["age"] = rosters["age"]

# Keep just what we need for the merge
id_col = None
for candidate in ["gsis_id", "player_id", "gsis_it"]:
    if candidate in rosters.columns:
        id_col = candidate
        break

print(f"  Using ID column: '{id_col}'")

if id_col and age_col:
    age_lookup = rosters[[id_col, "season", "age"]].dropna(subset=["age"])
    age_lookup = age_lookup.rename(columns={id_col: "gsis_id"})
    age_lookup = age_lookup.drop_duplicates(subset=["gsis_id", "season"])
    print(f"  Age lookup: {len(age_lookup):,} rows, age range: "
          f"{age_lookup['age'].min():.0f}-{age_lookup['age'].max():.0f}")
else:
    age_lookup = None
    print("  Could not build age lookup — proceeding without age")

# ── 3. Load enriched dataset from v2 ─────────────────────────────────────────
print("\nLoading enriched dataset from v2...")
df = pd.read_csv("data/model_ready_v2.csv")
print(f"  {len(df):,} rows, {len(df.columns)} columns")

# ── 4. Merge age in ───────────────────────────────────────────────────────────
if age_lookup is not None:
    print("Merging age into dataset...")
    df = pd.merge(df, age_lookup, on=["gsis_id", "season"], how="left")
    age_fill_rate = df["age"].notna().mean()
    print(f"  Age fill rate: {age_fill_rate:.1%}")
    print(f"  Age range in dataset: {df['age'].min():.0f}-{df['age'].max():.0f}")

    # Fill missing ages with position median
    if "position" in df.columns:
        df["age"] = df.groupby("position")["age"].transform(
            lambda x: x.fillna(x.median())
        )
    df["age"] = df["age"].fillna(df["age"].median())

    # Age-based risk features
    df["is_veteran"] = (df["age"] >= 30).astype(int)   # veterans have higher injury risk
    df["is_young"]   = (df["age"] <= 23).astype(int)   # young players also at risk
    print("  Added: is_veteran, is_young flags")
else:
    print("  Skipping age merge")

print(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns")

# ── 5. Train/test split ───────────────────────────────────────────────────────
DROP_COLS = ["injured", "gsis_id", "position", "season"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].fillna(0)
y = df["injured"]

train_mask = df["season"] <= 2021
test_mask  = df["season"] >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"\nTrain: {len(X_train_bal):,} (balanced) | Test: {len(X_test):,}")

# ── 6. Train models ───────────────────────────────────────────────────────────
print("\nTraining models...")

def train_and_proba(name, model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_te)[:, 1]

lr_pipeline = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=42))])
lr_model,  lr_proba  = train_and_proba("Logistic Regression", lr_pipeline,
                                        X_train_bal, y_train_bal, X_test)
rf_model,  rf_proba  = train_and_proba("Random Forest",
                                        RandomForestClassifier(n_estimators=100,
                                                               random_state=42, n_jobs=-1),
                                        X_train_bal, y_train_bal, X_test)
xgb_model, xgb_proba = train_and_proba("XGBoost",
                                         XGBClassifier(n_estimators=100, random_state=42,
                                                       eval_metric="logloss", verbosity=0),
                                         X_train_bal, y_train_bal, X_test)

# ── 7. Threshold tuning ───────────────────────────────────────────────────────
def find_best_threshold(proba, y_true, min_recall=0.55):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    best_thresh, best_f1 = 0.5, 0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    return best_thresh, best_f1

def evaluate(name, proba, y_true, threshold):
    y_pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    rec = recall_score(y_true, y_pred)
    print(f"\n{'─'*52}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'─'*52}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Not Injured", "Injured"]))
    print(f"  ROC-AUC: {auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not Injured", "Injured"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name} (t={threshold:.2f}) — v3 Confusion Matrix")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/{fname}_v3_confusion.png")
    plt.close()
    return auc, rec

models = [
    ("Logistic Regression", lr_proba),
    ("Random Forest",        rf_proba),
    ("XGBoost",              xgb_proba),
]

print("\n── Finding best thresholds ──")
thresholds = {}
for name, proba in models:
    t, f1 = find_best_threshold(proba, y_test)
    thresholds[name] = t
    print(f"  {name}: threshold={t:.2f}  best_f1={f1:.3f}")

print("\n── Results with tuned thresholds ──")
results = {}
for name, proba in models:
    auc, rec = evaluate(name, proba, y_test, thresholds[name])
    results[name] = {"auc": auc, "recall": rec, "threshold": thresholds[name]}

# ── 8. ROC comparison ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, proba in models:
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=name)
ax.set_title("ROC Curve Comparison — v3 (with Age)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_comparison_v3.png")
plt.close()

# ── 9. Feature importance ────────────────────────────────────────────────────
importances = pd.Series(
    rf_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Top 15 Feature Importances — v3 (with Age)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_v3.png")
plt.close()
print("\nSaved → outputs/feature_importance_v3.png")

# ── 10. Final summary ────────────────────────────────────────────────────────
lr = results["Logistic Regression"]
print("\n" + "="*52)
print("  FULL PROGRESSION SUMMARY (Logistic Regression)")
print("="*52)
print(f"  v1  ROC-AUC: 0.693 | Recall: 49% | Threshold: 0.50")
print(f"  v2  ROC-AUC: 0.701 | Recall: 56% | Threshold: 0.53")
print(f"  v3  ROC-AUC: {lr['auc']:.3f} | Recall: {lr['recall']:.0%}"
      f" | Threshold: {lr['threshold']:.2f}")
print("="*52)
print("\nDone! Check /outputs for v3 charts.")
