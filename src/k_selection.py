"""Choosing the number of customer states k with the Elbow method (proposal Section 3.4).

The group's primary method is the **Elbow** method: plot inertia (within-cluster sum of squares, WCSS)
against k and pick the "elbow" - the point after which adding a cluster no longer reduces inertia much.
Because reading an elbow by eye is subjective, it is also located with two numeric (reproducible) rules:

* Kneedle (Satopaa et al., 2011) - the point on the inertia curve furthest from the straight line joining the smallest and
  largest k (after rescaling both axes to [0, 1]).
* Second difference - the k where the inertia curve bends most sharply: largest inertia(k-1) - 2*inertia(k) + inertia(k+1).

The indices below are supporting evidence only (they do not overturn the Elbow decision):
* Silhouette (highest), Davies-Bouldin (lowest), Calinski-Harabasz (highest).
* Prediction Strength (Tibshirani & Walther, 2005) - do clusters found on one half of the data reappear in
  the other half? Rule: k is reproducible if PS(k) >= 0.8.
* Bootstrap stability (mean ARI between refits on bootstrap samples).
* Smallest cluster >= 5 % of the training rows.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)


def _km(k, seed, n_init=10):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed)


def internal_indices(Z, ks, seed=42, sil_samples=5, sil_size=6000):
    out = {}
    for k in ks:
        km = _km(k, seed, n_init=20).fit(Z)
        lab = km.labels_
        sil = np.mean([silhouette_score(Z, lab, sample_size=min(sil_size, len(Z)), random_state=seed + s)
                       for s in range(sil_samples)])
        out[k] = {"inertia": km.inertia_, "silhouette": sil,
                  "davies_bouldin": davies_bouldin_score(Z, lab),
                  "calinski_harabasz": calinski_harabasz_score(Z, lab),
                  "smallest_state_share": np.bincount(lab).min() / len(lab)}
    return out


def elbow_table(inertia) -> "pd.DataFrame":
    """Elbow table from inertia by k: absolute/relative reduction and second difference."""
    ine = pd.Series(inertia).sort_index()
    out = pd.DataFrame({"inertia": ine})
    out["drop"] = -ine.diff()                          # reduction relative to k-1
    out["drop_pct"] = out["drop"] / ine.shift(1) * 100  # % reduction relative to k-1
    out["second_diff"] = ine.shift(1) - 2 * ine + ine.shift(-1)   # "bend" at k (large = clear elbow)
    return out


def kneedle_k(inertia) -> int:
    """Kneedle elbow: the point furthest from the chord joining the two ends (convex, decreasing curve)."""
    ine = pd.Series(inertia).sort_index()
    ks = ine.index.values.astype(float)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (ine.values - ine.values[-1]) / (ine.values[0] - ine.values[-1])
    return int(ks[np.argmax((1 - x) - y)])


def second_diff_k(inertia) -> int:
    """Elbow by the largest second difference (only k with both neighbours are considered)."""
    sd = elbow_table(inertia)["second_diff"].dropna()
    return int(sd.idxmax())


def prediction_strength(Z, ks, n_splits=20, seed=42):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ks}
    for s in range(n_splits):
        idx = rng.permutation(len(Z))
        A, Bt = Z[idx[: len(Z) // 2]], Z[idx[len(Z) // 2:]]
        for k in ks:
            km_a = _km(k, seed + s).fit(A)
            lab_b = _km(k, seed + 100 + s).fit_predict(Bt)       # clusters of the test half
            pred_b = km_a.predict(Bt)                             # test points classified by the centroids of the training half
            strengths = []
            for c in range(k):
                m = lab_b == c
                n = m.sum()
                if n < 2:
                    strengths.append(0.0); continue
                counts = np.bincount(pred_b[m], minlength=k)
                strengths.append((counts * (counts - 1)).sum() / (n * (n - 1)))
            out[k].append(min(strengths))
    return {k: {"prediction_strength": float(np.mean(v)), "ps_sd": float(np.std(v))} for k, v in out.items()}


def bootstrap_stability(Z, ks, n_boot=10, seed=42):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(len(Z), len(Z), replace=True) for _ in range(n_boot)]
    out = {}
    for k in ks:
        ref = _km(k, seed, n_init=20).fit(Z).labels_
        aris = [adjusted_rand_score(ref, _km(k, seed + b + 1, n_init=5).fit(Z[i]).predict(Z))
                for b, i in enumerate(boots)]
        out[k] = {"stability_ARI": float(np.mean(aris)), "stability_ARI_sd": float(np.std(aris))}
    return out


# ---- cached runs + decision procedure --------------------------------------
import pandas as pd

METRIC_FUNCS = {"internal": internal_indices,
                "ps": prediction_strength, "boot": bootstrap_stability}
PS_THRESHOLD = 0.80        # Tibshirani & Walther (2005): threshold for reproducible clusters
MIN_STATE_SHARE = 0.05     # size: every state >= 5 % of the training rows


def evaluate(Z, window: str, metric: str, seed: int, cache_dir, ks=range(2, 11), recompute=False) -> pd.DataFrame:
    """Compute one group of indices for one (window, seed) and cache the results as CSV."""
    f = cache_dir / f"{window}_{metric}_{seed}.csv"
    if f.exists() and not recompute:
        return pd.read_csv(f)
    res = METRIC_FUNCS[metric](Z, list(ks), seed=seed)
    df = pd.DataFrame(res).T.rename_axis("k").reset_index().assign(
        window=window, seed=seed, metric_group=metric, n=len(Z))
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def decide(table: pd.DataFrame) -> dict:
    """Apply the pre-specified rules to one run (rows indexed by k)."""
    t = table.sort_index()
    k_knee = kneedle_k(t["inertia"])
    k_sd = second_diff_k(t["inertia"])
    replicable = t.index[t["prediction_strength"] >= PS_THRESHOLD]
    return {"k_elbow_kneedle": k_knee,
            "k_elbow_2nd_diff": k_sd,
            "k_largest_PS>=0.8": int(replicable.max()),
            "elbow_k_is_replicable": bool(t.loc[k_knee, "prediction_strength"] >= PS_THRESHOLD),
            "elbow_k_min_share_ok": (bool(t.loc[k_knee, "smallest_state_share"] >= MIN_STATE_SHARE)
                                     if "smallest_state_share" in t else None)}
