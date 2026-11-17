"""RFM customer-state model (proposal Section 3.4).

The model is fitted ONCE on a set of training snapshots and then frozen:
log1p transform (decided from training skewness) -> StandardScaler -> K-Means centroids.
Any later snapshot is transformed with the same parameters and mapped to the nearest
centroid, so a state means the same thing in every period and test data never
influences the segmentation.

Phase 4 re-uses `fit_state_model` inside each expanding-window fold so that state
assignment is learned only from the past part of the fold (Section 3.6).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_FEATURES, SKEW_THRESHOLD, RANDOM_STATE


@dataclass
class StateModel:
    k: int
    fitted_on: list
    log_cols: list
    scaler: StandardScaler
    kmeans: KMeans
    order: np.ndarray                          # order[raw_label] -> ordered state id
    centers_z: pd.DataFrame = field(default=None)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[CLUSTER_FEATURES].astype(float).copy()
        for c in self.log_cols:
            X[c] = np.log1p(X[c])
        return self.scaler.transform(X.values)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Nearest frozen centroid, returned as ordered state id (0..k-1)."""
        return self.order[self.kmeans.predict(self.transform(df))]

    def distances(self, df: pd.DataFrame) -> np.ndarray:
        """Distance of each row to every centroid, columns in ordered-state order."""
        d = self.kmeans.transform(self.transform(df))
        out = np.empty_like(d)
        out[:, self.order] = d
        return out


def fit_state_model(panel: pd.DataFrame, snapshots: list, k: int,
                    random_state: int = RANDOM_STATE, n_init: int = 20) -> StateModel:
    train = panel[panel["snapshot"].isin(snapshots)]
    X = train[CLUSTER_FEATURES].astype(float).copy()
    log_cols = [c for c in CLUSTER_FEATURES if abs(X[c].skew()) > SKEW_THRESHOLD]
    for c in log_cols:
        X[c] = np.log1p(X[c])
    scaler = StandardScaler().fit(X.values)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(scaler.transform(X.values))
    # Deterministic, interpretable ordering: value score = zF + zM - zR (low -> high)
    c = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    score = c["Frequency"] + c["Monetary"] - c["Recency"]
    order = np.empty(k, dtype=int)
    order[np.argsort(score.values)] = np.arange(k)
    centers = c.copy()
    centers.index = order
    centers = centers.sort_index()
    centers.index.name = "state"
    return StateModel(k=k, fitted_on=list(snapshots), log_cols=log_cols, scaler=scaler,
                      kmeans=km, order=order, centers_z=centers)
