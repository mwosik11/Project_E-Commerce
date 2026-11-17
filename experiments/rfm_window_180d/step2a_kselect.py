"""Step 2a — Compute the k-selection indices (cached) for one run: python step2a_kselect.py <run_id 0..5>."""
from common import *
import sys as _s, pandas as pd
from config import TRAIN_ORIGINS, RANDOM_STATE
from states import fit_state_model
from k_selection import evaluate

RUNS = [(TRAIN_ORIGINS, RANDOM_STATE), (TRAIN_ORIGINS, 7), (TRAIN_ORIGINS, 2024),
        (["S1", "S2", "S3"], RANDOM_STATE), (["S2", "S3", "S4"], RANDOM_STATE), (["S3", "S4", "S5"], RANDOM_STATE)]
panel = pd.read_parquet(DATA / "panel_window180.parquet")

def run_window(snaps, seed, metrics=("internal", "ps")):
    m = fit_state_model(panel, snaps, k=2)                    # only used to get the transformation for this window
    Z = m.transform(panel[panel["snapshot"].isin(snaps)])
    parts = [evaluate(Z, "-".join(snaps), met, seed, CACHE / "k_selection_raw").set_index("k")
             .drop(columns=["window", "seed", "metric_group", "n"]) for met in metrics]
    return pd.concat(parts, axis=1)

if __name__ == "__main__":
    rid = int(_s.argv[1])
    snaps, seed = RUNS[rid]
    run_window(snaps, seed, ("internal", "ps", "boot") if rid == 0 else ("internal", "ps"))
    print("done run", rid)
