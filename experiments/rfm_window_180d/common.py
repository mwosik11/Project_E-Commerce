"""Shared setup for the 180-day rolling-window RFM experiment.

This experiment does NOT modify the main pipeline. It reuses the modules in `src/` and changes one thing only:
Frequency and Monetary used to build states (and as prediction features) are computed over the 180 days
before each snapshot instead of cumulatively. Recency, AOV, ProductVariety, Tenure, RecentOrders, RecentSpend are unchanged.
"""
import sys
from pathlib import Path
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_DAYS = 180
# Number of states: by default the Elbow/Kneedle rule (k = 4). Set the environment variable WINDOW_K (e.g. 3) to run
# a sensitivity analysis with another k; results then go to separate folders with the suffix _k<K>.
import os as _os
WINDOW_K = int(_os.environ.get("WINDOW_K", "0")) or None
_SUF = f"_k{WINDOW_K}" if WINDOW_K else ""
DATA, MODELS, CACHE = EXP / "data", EXP / ("models" + _SUF), EXP / "cache"
DATA_K = DATA / f"k{WINDOW_K}" if WINDOW_K else DATA          # k-dependent data (panel with states, dataset, predictions)
TUNE_CACHE = CACHE / ("tuning_raw" + _SUF)
TABLES, FIGURES = EXP / ("outputs" + _SUF) / "tables", EXP / ("outputs" + _SUF) / "figures"
if WINDOW_K:                                                   # must be set before importing modeling / bootstrap
    import config as _cfg
    _cfg.N_STATES = WINDOW_K
    _cfg.STATE_NAMES = {0: "Inactive (no purchase in 180 days)", 1: "Occasional", 2: "Active"} if WINDOW_K == 3 else {i: f"State {i}" for i in range(WINDOW_K)}
    _cfg.STATE_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#eda100"} if WINDOW_K == 3 else _cfg.STATE_COLORS
MAIN_TABLES = ROOT / "outputs" / "tables"          # results of the main design (read only, for comparison)
for _p in (DATA, DATA_K, MODELS, CACHE, TUNE_CACHE, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#9a9a92", "xtick.color": "#5c5c57", "ytick.color": "#5c5c57",
                     "font.size": 10, "lines.linewidth": 2})
