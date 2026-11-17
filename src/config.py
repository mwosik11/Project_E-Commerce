"""Shared configuration for the Group 8 temporal customer-state project.

Every notebook imports from here so snapshot dates, paths and seeds stay identical
across phases (proposal Section 3.8: fixed seeds, consistent pipeline).
"""
from pathlib import Path
import pandas as pd

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "online_retail_II.xlsx"              # original UCI file (2 sheets)
GROUP_CLEAN_XLSX = ROOT / "online_retail_II_cleaned.xlsx"  # group's earlier static cleaning (reference only)
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
for _p in (INTERIM, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Reproducibility --------------------------------------------------------
RANDOM_STATE = 42

# ---- Cleaning rules (Section 3.2) -------------------------------------------
# Product rows are kept only when StockCode is 5 digits + up to 2 letters.
# This drops POST, DOT, M, C2, BANK CHARGES, ADJUST*, AMAZONFEE, TEST*, gift_*, DCGS*, ...
PRODUCT_STOCKCODE_PATTERN = r"^\d{5}[A-Za-z]{0,2}$"

# ---- Temporal design (Section 3.3) ------------------------------------------
SNAPSHOTS = {
    "S1": "2010-03-31",
    "S2": "2010-06-30",
    "S3": "2010-09-30",
    "S4": "2010-12-31",
    "S5": "2011-03-31",
    "S6": "2011-06-30",
    "S7": "2011-09-30",
    "S8": "2011-12-09",
}
SNAPSHOT_DATES = {k: pd.Timestamp(v) for k, v in SNAPSHOTS.items()}
SNAPSHOT_ORDER = list(SNAPSHOTS)                      # ["S1", ..., "S8"]
SNAPSHOT_INDEX = {s: i + 1 for i, s in enumerate(SNAPSHOT_ORDER)}

# Window for "recent activity" features (orders / spend in the last N days before t)
RECENT_WINDOW_DAYS = 90

# Origins t used for prediction (target = Segment_{t+1})
TRAIN_ORIGINS = ["S1", "S2", "S3", "S4", "S5"]
TEST_ORIGINS = ["S6", "S7"]
HISTORY_ELIGIBLE_FROM = "S3"   # Config B needs Segment_{t-1} and Segment_{t-2}

# ---- State construction (Section 3.4) ---------------------------------------
CLUSTER_FEATURES = ["Recency", "Frequency", "Monetary"]   # states are defined by RFM only
SKEW_THRESHOLD = 1.0          # log1p-transform a clustering input if |skew| on training rows > this
K_CANDIDATES = list(range(2, 11))
N_STATES = 4                  # chosen in notebook 02 (Elbow method, cross-checked with Silhouette, Prediction Strength and stability across windows/seeds)
# Names are attached to the *ordered* state ids (0 = lowest value score zF + zM - zR)
STATE_NAMES = {
    0: "Dormant Low-Value",
    1: "Lapsing Mid-Value",
    2: "Recent Low-Frequency",
    3: "Active High-Value",
}
STATE_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100"}  # 4 categorical colours checked for distinguishability

# ---- Prediction dataset (Sections 3.5-3.6) ----------------------------------
CURRENT_FEATURES = ["Recency", "Frequency", "Monetary", "AOV", "ProductVariety",
                    "Tenure", "RecentOrders", "RecentSpend"]
DURATION_CAP = 5            # StateDuration is capped at the longest run observable at the last training origin (S5)
CONFIG_A = CURRENT_FEATURES + ["Segment_t"]
CONFIG_B = CONFIG_A + ["Segment_t-1", "Segment_t-2", "StateDuration"]
CATEGORICAL_FEATURES = ["Segment_t", "Segment_t-1", "Segment_t-2"]
TARGET = "Segment_t+1"
# Expanding-window tuning folds on the history-eligible training origins (S3-S5); test origins untouched
TUNING_FOLDS = [
    {"fold": 1, "train_origins": ["S3"], "val_origins": ["S4"]},
    {"fold": 2, "train_origins": ["S3", "S4"], "val_origins": ["S5"]},
]
