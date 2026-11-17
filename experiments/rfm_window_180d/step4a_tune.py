"""Step 4a — Tune one (model, configuration) with the same parameter grid as Phase 4: python step4a_tune.py <model> <A|B>."""
from common import *
import sys as _s, pickle
import modeling as M
folds = pickle.load(open(DATA_K / "tuning_folds_window180.pkl", "rb"))
df = M.tune(_s.argv[1], _s.argv[2], folds, TUNE_CACHE)
print(_s.argv[1], _s.argv[2], len(df), "rows; best mean Macro-F1", round(df.groupby("params")["macro_f1"].mean().max(), 4))
