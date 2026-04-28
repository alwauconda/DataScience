# 02_eda.py
# Step 2: Exploratory Data Analysis
# Run after 01_load_data.py. Produces charts saved to /outputs/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load saved data ───────────────────────────────────────────────────────────
injuries   = pd.read_csv("data/raw_injuries.csv")
snaps      = pd.read_csv("data/raw_snap_counts.csv")

print("Injury columns:", injuries.columns.tolist())
print("Snap count columns:", snaps.columns.tolist())

# ── 1. Injuries by Position ───────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
injury_by_pos = (
    injuries[injuries["report_status"].notna()]
    .groupby("position")
    .size()
    .sort_values(ascending=False)
    .head(15)
)
sns.barplot(x=injury_by_pos.index, y=injury_by_pos.values, palette="Reds_r")
plt.title("Injury Report Entries by Position (Top 15)")
plt.xlabel("Position")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/injuries_by_position.png")
plt.show()
print("Saved → outputs/injuries_by_position.png")

# ── 2. Injuries by Season ─────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
injury_by_season = injuries.groupby("season").size()
sns.lineplot(x=injury_by_season.index, y=injury_by_season.values, marker="o", color="firebrick")
plt.title("Total Injury Report Entries by Season")
plt.xlabel("Season")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/injuries_by_season.png")
plt.show()
print("Saved → outputs/injuries_by_season.png")

# ── 3. Injury Status Breakdown ────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
status_counts = injuries["report_status"].value_counts()
sns.barplot(x=status_counts.index, y=status_counts.values, palette="coolwarm")
plt.title("Injury Report Status Breakdown")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/injury_status_breakdown.png")
plt.show()
print("Saved → outputs/injury_status_breakdown.png")

# ── 4. Snap Count Distribution ────────────────────────────────────────────────
if "offense_snaps" in snaps.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(snaps["offense_snaps"].dropna(), bins=40, color="steelblue")
    plt.title("Distribution of Offensive Snap Counts per Game")
    plt.xlabel("Snap Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/snap_count_distribution.png")
    plt.show()
    print("Saved → outputs/snap_count_distribution.png")

print("\nEDA complete! Charts saved to /outputs")
