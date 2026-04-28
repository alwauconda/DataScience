# 03_preprocess.py
# Step 3: Clean, merge, and engineer features from raw data
# Run after 01_load_data.py. Outputs a single model-ready CSV.

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading raw data...")
injuries   = pd.read_csv("data/raw_injuries.csv")
stats      = pd.read_csv("data/raw_player_stats.csv", low_memory=False)
snaps      = pd.read_csv("data/raw_snap_counts.csv")

print(f"  Injuries:     {len(injuries):,} rows")
print(f"  Player stats: {len(stats):,} rows")
print(f"  Snap counts:  {len(snaps):,} rows")

# ── 1. Build injury target variable ───────────────────────────────────────────
# report_status values include: Questionable, Doubtful, Out, Full, Limited, etc.
# We define "injured" as appearing on the injury report at all (not Full/None)
print("\nBuilding target variable...")

INJURED_STATUSES = ["Out", "IR", "PUP"]

injuries["injured"] = injuries["report_status"].apply(
    lambda x: 1 if str(x).strip() in INJURED_STATUSES else 0
)

# Keep only the columns we need from injuries
injury_flags = injuries[["season", "week", "gsis_id", "full_name", "position", "injured"]].copy()
injury_flags = injury_flags.drop_duplicates(subset=["season", "week", "gsis_id"])

print(f"  Injury rate: {injury_flags['injured'].mean():.2%} of player-weeks flagged as injured")

# ── 2. Clean snap counts ──────────────────────────────────────────────────────
print("\nProcessing snap counts...")

# Identify the snap column — nflreadpy may use offense_snaps or snaps
snap_col = None
for candidate in ["offense_snaps", "snaps", "offense_pct", "snap_counts_offense"]:
    if candidate in snaps.columns:
        snap_col = candidate
        break

if snap_col is None:
    print("  WARNING: Could not find snap count column. Available columns:")
    print(" ", snaps.columns.tolist())
    snaps["snap_count"] = np.nan
else:
    print(f"  Using snap column: '{snap_col}'")
    snaps = snaps.rename(columns={snap_col: "snap_count"})

snaps_clean = snaps[["season", "week", "pfr_player_id", "snap_count"]].copy()
snaps_clean = snaps_clean.drop_duplicates(subset=["season", "week", "pfr_player_id"])

# ── 3. Clean player stats ─────────────────────────────────────────────────────
print("\nProcessing player stats...")

# Keep relevant columns if they exist
stat_cols = ["season", "week", "player_id", "position", "age",
             "completions", "attempts", "passing_yards",
             "carries", "rushing_yards", "receptions", "receiving_yards"]

stat_cols_present = [c for c in stat_cols if c in stats.columns]
stats_clean = stats[stat_cols_present].copy()
stats_clean = stats_clean.drop_duplicates(subset=["season", "week", "player_id"])

# ── 4. Merge datasets ─────────────────────────────────────────────────────────
print("\nMerging datasets...")

# Merge injury flags with player stats on gsis_id / player_id
df = pd.merge(
    injury_flags,
    stats_clean,
    left_on=["season", "week", "gsis_id"],
    right_on=["season", "week", "player_id"],
    how="left"
)

print(f"  After merging stats: {len(df):,} rows")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
print("\nEngineering features...")

# Sort for rolling calculations
df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

# Rolling 3-week snap count workload (requires snap merge — approximate via stats for now)
# We'll use rushing yards + receiving yards as a workload proxy if snaps unavailable
if "rushing_yards" in df.columns and "receiving_yards" in df.columns:
    df["touch_proxy"] = df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0)
    df["rolling_3wk_workload"] = (
        df.groupby("gsis_id")["touch_proxy"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    )

# Week of season (fatigue increases late in season)
df["week"] = pd.to_numeric(df["week"], errors="coerce")

# Position encoding (one-hot)
if "position_x" in df.columns:
    df = df.rename(columns={"position_x": "position"})
elif "position_y" in df.columns:
    df = df.rename(columns={"position_y": "position"})

if "position" in df.columns:
    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, position_dummies], axis=1)

# ── 6. Final cleanup ──────────────────────────────────────────────────────────
print("\nFinalizing dataset...")

# Drop columns we don't need for modeling
drop_cols = ["full_name", "player_id", "touch_proxy"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Drop rows with no target variable
df = df.dropna(subset=["injured"])
df["injured"] = df["injured"].astype(int)

print(f"  Final dataset: {len(df):,} rows, {len(df.columns)} columns")
print(f"  Injury rate:   {df['injured'].mean():.2%}")
print(f"  Columns:       {df.columns.tolist()}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_csv("data/model_ready.csv", index=False)
print("\nSaved → data/model_ready.csv")
print("\nPreprocessing complete! Run 04_model.py next.")
