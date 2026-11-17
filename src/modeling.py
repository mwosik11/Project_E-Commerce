"""Next-state classifiers, temporal tuning and evaluation (proposal Sections 3.5-3.7).

* Same preprocessing, grid, folds and metric for Config A and Config B; the only
  difference is the feature list (controlled ablation, H1).
* Tuning folds are expanding windows over the history-eligible training origins.
  Inside each fold the RFM states are re-fitted on the fold's past snapshots only
  (S1 .. last training target), then the dataset is rebuilt - validation labels
  never shape the states (Section 3.6).
* Class imbalance: class weighting (balanced vs. none) is a tuned hyper-parameter for every
  model, chosen by validation Macro-F1 (balanced weights for XGBoost via sample weights).
"""
from __future__ import annotations
import itertools, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (CURRENT_FEATURES, CONFIG_A, CONFIG_B, CATEGORICAL_FEATURES, TARGET, N_STATES,
                    SNAPSHOT_ORDER, RANDOM_STATE, TUNING_FOLDS)
from states import fit_state_model
from dataset import build_dataset

CONFIGS = {"A": CONFIG_A, "B": CONFIG_B}
LABELS = list(range(N_STATES))

GRIDS = {
    "LogReg": {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "RandomForest": {"max_depth": [8, 16, None], "min_samples_leaf": [1, 10], "max_features": ["sqrt", 0.5],
                     "class_weight": [None, "balanced"]},
    "XGBoost": {"n_estimators": [200, 500], "max_depth": [3, 5], "learning_rate": [0.03, 0.1], "min_child_weight": [1, 5],
                "class_weight": [None, "balanced"]},
}


def grid(model: str):
    keys, vals = zip(*GRIDS[model].items())
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def make_model(model: str, features: list, params: dict) -> Pipeline:
    params = dict(params)
    cw = params.pop("class_weight", None)
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    num = [c for c in features if c not in CATEGORICAL_FEATURES]
    ohe = OneHotEncoder(categories=[LABELS] * len(cat), handle_unknown="ignore", sparse_output=False)
    if model == "LogReg":
        num_tf = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
        clf = LogisticRegression(C=params["C"], class_weight=cw, max_iter=3000, random_state=RANDOM_STATE)
    elif model == "RandomForest":
        num_tf = "passthrough"
        clf = RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=4, random_state=RANDOM_STATE, **params)
    elif model == "XGBoost":
        num_tf = "passthrough"
        clf = XGBClassifier(objective="multi:softprob", num_class=N_STATES, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=4, random_state=RANDOM_STATE, eval_metric="mlogloss", **params)
    else:
        raise ValueError(model)
    pre = ColumnTransformer([("num", num_tf, num), ("cat", ohe, cat)])
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.balanced_ = cw == "balanced"
    return pipe


def fit(model: str, pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    if model == "XGBoost" and getattr(pipe, "balanced_", False):
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def to_xy(df: pd.DataFrame, features: list):
    X = df[features].copy()
    for c in X.columns:
        X[c] = X[c].astype(float)
    return X, df[TARGET].astype(int).values


# ---------------------------------------------------------------- folds
def fold_data(base_panel: pd.DataFrame, fold: dict, k: int = N_STATES):
    """Re-fit the states on the fold's past snapshots and rebuild the dataset."""
    last_train_target = SNAPSHOT_ORDER[SNAPSHOT_ORDER.index(fold["train_origins"][-1]) + 1]
    state_snaps = SNAPSHOT_ORDER[: SNAPSHOT_ORDER.index(last_train_target) + 1]
    sm = fit_state_model(base_panel, state_snaps, k=k)
    pnl = base_panel.copy()
    pnl["Segment"] = sm.predict(pnl)
    ds = build_dataset(pnl)
    ds = ds[ds["history_eligible"]]
    tr = ds[ds["origin"].isin(fold["train_origins"])]
    va = ds[ds["origin"].isin(fold["val_origins"])]
    return tr, va, state_snaps


# ---------------------------------------------------------------- metrics
def benchmark_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    majority = int(train[TARGET].astype(int).mode().iat[0])
    return {"Majority class": np.full(len(test), majority),
            "Persistence": test["Segment_t"].astype(int).values}


def metrics(y, pred, proba=None, segment_t=None) -> dict:
    out = {"macro_f1": f1_score(y, pred, average="macro", labels=LABELS, zero_division=0),
           "weighted_f1": f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0),
           "accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred)}
    out["macro_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS) if proba is not None else np.nan
    if segment_t is not None:
        ch = y != segment_t
        out["changers_n"] = int(ch.sum())
        out["changers_accuracy"] = accuracy_score(y[ch], pred[ch])
        out["changers_macro_f1"] = f1_score(y[ch], pred[ch], average="macro", labels=LABELS, zero_division=0)
        out["stayers_accuracy"] = accuracy_score(y[~ch], pred[~ch])
    return out


def per_class(y, pred) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=LABELS)


# ---------------------------------------------------------------- tuning
def tune(model: str, config: str, folds_cache: dict, cache_dir, recompute=False) -> pd.DataFrame:
    """Evaluate every grid point on every fold; cached per (model, config)."""
    import hashlib
    sig = hashlib.md5((repr(GRIDS[model]) + repr(CONFIGS[config])).encode()).hexdigest()[:8]
    f = cache_dir / f"{model}_{config}_{sig}.csv"           # grid/feature change -> new cache file
    if f.exists() and not recompute:
        return pd.read_csv(f)
    rows = []
    feats = CONFIGS[config]
    for params in grid(model):
        for fid, (tr, va) in folds_cache.items():
            Xtr, ytr = to_xy(tr, feats); Xva, yva = to_xy(va, feats)
            t0 = time.time()
            pipe = fit(model, make_model(model, feats, params), Xtr, ytr)
            pred = pipe.predict(Xva)
            rows.append({"model": model, "config": config, "fold": fid, **{f"param_{k}": v for k, v in params.items()},
                         "params": repr(params), "macro_f1": f1_score(yva, pred, average="macro", labels=LABELS, zero_division=0),
                         "accuracy": accuracy_score(yva, pred), "fit_seconds": time.time() - t0})
    df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def best_params(tuning: pd.DataFrame) -> tuple[dict, float]:
    agg = tuning.groupby("params")["macro_f1"].mean().sort_values(ascending=False)
    import ast
    return ast.literal_eval(agg.index[0]), float(agg.iloc[0])
