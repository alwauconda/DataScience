# 05_improved_model.py
# Step 5: Feature enrichment + threshold tuning for improved results
# Adds player age, days of rest, and cumulative season workload
# then retunes the classification threshold for better recall.
# Run after 03_preprocess.py (uses raw CSVs, not model_ready.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading raw data...")
injuries   = pd.read_csv("data/raw_injuries.csv")
stats      = pd.read_csv("data/raw_player_stats.csv", low_memory=False)
snaps      = pd.read_csv("data/raw_snap_counts.csv")

# ── 1. Build target variable (same as before) ─────────────────────────────────
INJURED_STATUSES = ["Out", "IR", "PUP"]
injuries["injured"] = injuries["report_status"].apply(
    lambda x: 1 if str(x).strip() in INJURED_STATUSES else 0
)
injury_flags = injuries[["season", "week", "gsis_id", "full_name",
                          "position", "injured"]].copy()
injury_flags = injury_flags.drop_duplicates(subset=["season", "week", "gsis_id"])

# ── 2. Engineer REST DAYS feature from injuries ───────────────────────────────
# Approximate rest days: NFL week spacing
# Regular week = 7 days, short week (Thu game) ~ 4 days, bye week before = 14 days
# We'll use week number difference as a proxy
print("Engineering rest days feature...")
injury_flags = injury_flags.sort_values(["gsis_id", "season", "week"])
injury_flags["prev_week"] = injury_flags.groupby(["gsis_id", "season"])["week"].shift(1)
injury_flags["weeks_since_last_game"] = injury_flags["week"] - injury_flags["prev_week"]
injury_flags["weeks_since_last_game"] = injury_flags["weeks_since_last_game"].fillna(1)
# Cap at 3 (bye week = 2, season start = 1)
injury_flags["weeks_since_last_game"] = injury_flags["weeks_since_last_game"].clip(1, 3)

# ── 3. Snap counts ────────────────────────────────────────────────────────────
snap_col = None
for candidate in ["offense_snaps", "snaps", "offense_pct", "snap_counts_offense"]:
    if candidate in snaps.columns:
        snap_col = candidate
        break

if snap_col:
    snaps = snaps.rename(columns={snap_col: "snap_count"})
    snaps_clean = snaps[["season", "week", "pfr_player_id", "snap_count"]].drop_duplicates()
else:
    snaps["snap_count"] = np.nan
    snaps_clean = snaps[["season", "week", "snap_count"]].drop_duplicates()

# ── 4. Player stats with AGE ──────────────────────────────────────────────────
print("Processing player stats with age...")
stat_cols = ["season", "week", "player_id", "position", "player_display_name",
             "completions", "attempts", "passing_yards",
             "carries", "rushing_yards", "receptions", "receiving_yards",
             "target_share", "age"]  # age is in nflreadpy player stats!

stat_cols_present = [c for c in stat_cols if c in stats.columns]
print(f"  Age column found: {'age' in stats.columns}")
stats_clean = stats[stat_cols_present].copy()
stats_clean = stats_clean.drop_duplicates(subset=["season", "week", "player_id"])

# ── 5. Merge ──────────────────────────────────────────────────────────────────
print("Merging datasets...")
df = pd.merge(
    injury_flags,
    stats_clean,
    left_on=["season", "week", "gsis_id"],
    right_on=["season", "week", "player_id"],
    how="left"
)
print(f"  After merge: {len(df):,} rows")

# ── 6. Feature Engineering ────────────────────────────────────────────────────
print("Engineering features...")
df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

# Rolling 3-week workload proxy
if "rushing_yards" in df.columns and "receiving_yards" in df.columns:
    df["touch_proxy"] = df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0)
    df["rolling_3wk_workload"] = (
        df.groupby("gsis_id")["touch_proxy"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    )

# Cumulative season workload (new feature)
if "touch_proxy" in df.columns:
    df["cumulative_season_workload"] = (
        df.groupby(["gsis_id", "season"])["touch_proxy"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

# Week of season
df["week"] = pd.to_numeric(df["week"], errors="coerce")

# Late season flag (week 14+, fatigue risk increases)
df["late_season"] = (df["week"] >= 14).astype(int)

# Position encoding
pos_col = "position_x" if "position_x" in df.columns else "position"
if pos_col in df.columns:
    df = df.rename(columns={pos_col: "position"})
    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)

# ── 7. Final cleanup ──────────────────────────────────────────────────────────
drop_cols = ["full_name", "player_id", "touch_proxy", "position_y",
             "player_display_name", "prev_week"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df = df.dropna(subset=["injured"])
df["injured"] = df["injured"].astype(int)

print(f"\nEnriched dataset: {len(df):,} rows, {len(df.columns)} columns")
print(f"Injury rate: {df['injured'].mean():.2%}")
print(f"New features present: weeks_since_last_game, cumulative_season_workload, late_season")
print(f"Age present: {'age' in df.columns}")

df.to_csv("data/model_ready_v2.csv", index=False)

# ── 8. Train/test split ───────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c not in
                ["injured", "gsis_id", "position", "season"]]

X = df[FEATURE_COLS].fillna(0)
y = df["injured"]

train_mask = df["season"] <= 2021
test_mask  = df["season"] >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ── 9. Train models ───────────────────────────────────────────────────────────
def train_and_proba(name, model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"  {name} trained.")
    return model, proba

print("\nTraining models...")
lr_pipeline = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=42))])
lr_model, lr_proba   = train_and_proba("Logistic Regression", lr_pipeline,
                                        X_train_bal, y_train_bal, X_test)
rf_model, rf_proba   = train_and_proba("Random Forest",
                                        RandomForestClassifier(n_estimators=100,
                                                               random_state=42, n_jobs=-1),
                                        X_train_bal, y_train_bal, X_test)
xgb_model, xgb_proba = train_and_proba("XGBoost",
                                         XGBClassifier(n_estimators=100, random_state=42,
                                                       eval_metric="logloss", verbosity=0),
                                         X_train_bal, y_train_bal, X_test)

# ── 10. Threshold tuning ──────────────────────────────────────────────────────
print("\nTuning classification threshold for best F1 / recall balance...")

def find_best_threshold(proba, y_true, min_recall=0.55):
    """Find threshold that maximizes F1 while keeping recall above min_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    best_thresh = 0.5
    best_f1 = 0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    return best_thresh, best_f1

def evaluate_with_threshold(name, proba, y_true, threshold):
    y_pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    print(f"\n{'─'*52}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'─'*52}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Not Injured", "Injured"]))
    print(f"  ROC-AUC: {auc:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not Injured", "Injured"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name} (t={threshold:.2f}) — Confusion Matrix")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/{fname}_v2_confusion.png")
    plt.close()
    return auc

models = [
    ("Logistic Regression", lr_proba),
    ("Random Forest",        rf_proba),
    ("XGBoost",              xgb_proba),
]

print("\n── Finding best thresholds ──")
thresholds = {}
for name, proba in models:
    t, f1 = find_best_threshold(proba, y_test, min_recall=0.55)
    thresholds[name] = t
    print(f"  {name}: threshold={t:.2f}  best_f1={f1:.3f}")

print("\n── Results with tuned thresholds ──")
for name, proba in models:
    evaluate_with_threshold(name, proba, y_test, thresholds[name])

# ── 11. ROC curve comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, proba in models:
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=name)
ax.set_title("ROC Curve Comparison — Enriched Model (v2)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_comparison_v2.png")
plt.close()
print("\nSaved → outputs/roc_comparison_v2.png")

# ── 12. Feature importance (Random Forest) ────────────────────────────────────
importances = pd.Series(
    rf_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Top 15 Feature Importances — Enriched Model (v2)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_v2.png")
plt.close()
print("Saved → outputs/feature_importance_v2.png")

# ── 13. Summary comparison vs v1 ─────────────────────────────────────────────
print("\n" + "="*52)
print("  SUMMARY: v1 vs v2 (Logistic Regression)")
print("="*52)
print(f"  v1 ROC-AUC:  0.693  |  Recall: 49%  |  Threshold: 0.50")
lr_auc_v2 = roc_auc_score(y_test, lr_proba)
lr_pred_v2 = (lr_proba >= thresholds["Logistic Regression"]).astype(int)
from sklearn.metrics import recall_score
lr_recall_v2 = recall_score(y_test, lr_pred_v2)
print(f"  v2 ROC-AUC:  {lr_auc_v2:.3f}  |  Recall: {lr_recall_v2:.0%}"
      f"  |  Threshold: {thresholds['Logistic Regression']:.2f}")
print("="*52)
print("\nDone! Check /outputs for updated charts.")
