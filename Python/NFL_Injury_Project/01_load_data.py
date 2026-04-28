# 01_load_data.py
# Step 1: Pull raw NFL injury and player stats data using nflreadpy
# Run this first to download and save raw data locally.

import nflreadpy as nfl
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS = list(range(2017, 2024))   # Training window: 2017–2023
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Injury Reports ─────────────────────────────────────────────────────────
print("Loading injury data...")
injuries = nfl.load_injuries(SEASONS).to_pandas()
print(f"  Rows: {len(injuries):,} | Columns: {injuries.columns.tolist()}")
injuries.to_csv(f"{OUTPUT_DIR}/raw_injuries.csv", index=False)
print("  Saved → data/raw_injuries.csv")

# ── 2. Weekly Player Stats ────────────────────────────────────────────────────
print("\nLoading weekly player stats...")
player_stats = nfl.load_player_stats(SEASONS).to_pandas()
print(f"  Rows: {len(player_stats):,} | Columns: {player_stats.columns.tolist()}")
player_stats.to_csv(f"{OUTPUT_DIR}/raw_player_stats.csv", index=False)
print("  Saved → data/raw_player_stats.csv")

# ── 3. Snap Counts ────────────────────────────────────────────────────────────
print("\nLoading snap counts...")
snaps = nfl.load_snap_counts(SEASONS).to_pandas()
print(f"  Rows: {len(snaps):,} | Columns: {snaps.columns.tolist()}")
snaps.to_csv(f"{OUTPUT_DIR}/raw_snap_counts.csv", index=False)
print("  Saved → data/raw_snap_counts.csv")

# ── 4. Quick Sanity Check ─────────────────────────────────────────────────────
print("\n── Injury Data Preview ──")
print(injuries.head())
print("\n── Player Stats Preview ──")
print(player_stats.head())
print("\n── Snap Counts Preview ──")
print(snaps.head())

print("\nDone! All raw data saved to /data")
