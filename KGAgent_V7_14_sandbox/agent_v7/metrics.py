# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd

# ---------- Basic math helpers ----------

def zscore(X: np.ndarray, axis: int = 0, eps: float = 1e-9) -> np.ndarray:
    mu = np.nanmean(X, axis=axis, keepdims=True)
    sd = np.nanstd(X, axis=axis, keepdims=True) + eps
    return (X - mu) / sd

def tfidf(df_long: pd.DataFrame, col_item: str, col_feat: str, col_value: str) -> pd.DataFrame:
    """
    item = region, feat = subclass
    TF = value / sum(value per item)
    IDF = log((1+N)/(1+df)) + 1
    """
    pivot = df_long.pivot_table(index=col_item, columns=col_feat, values=col_value, aggfunc="sum", fill_value=0.0)
    X = pivot.to_numpy().astype(float)
    # TF
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    TF = X / row_sums
    # IDF
    df_cnt = (X > 0).sum(axis=0, keepdims=True)
    N = X.shape[0]
    IDF = np.log((1.0 + N) / (1.0 + df_cnt)) + 1.0
    M = TF * IDF
    return pd.DataFrame(M, index=pivot.index, columns=pivot.columns)

def cosine_distance_matrix(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    S = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

def jensen_shannon_distance_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Rows as distributions; add eps then normalize
    Xp = X + eps
    Xp /= Xp.sum(axis=1, keepdims=True)
    def _kl(p, q):
        return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)
    def _js(p, q):
        m = 0.5 * (p + q)
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        pi = np.repeat(Xp[i, :][None, :], n, axis=0)
        D[i, :] = np.sqrt(_js(pi, Xp))
    np.fill_diagonal(D, 0.0)
    return D

def pearson_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    x = x.astype(float); y = y.astype(float)
    xm = x - x.mean(); ym = y - y.mean()
    num = (xm * ym).sum()
    den = (np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()) + eps)
    return float(num / den)

# ---------- Domain builders ----------

MORPH_FEATURES = [
    "axonal_length",
    "dendritic_length",
    "axonal_branches",
    "dendritic_branches",
    "axonal_maximum_branch_order",
    "dendritic_maximum_branch_order",
]

def build_morph_matrix(morph_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
    if not morph_rows:
        return pd.DataFrame(columns=MORPH_FEATURES), np.zeros((0, len(MORPH_FEATURES)))
    df = pd.DataFrame(morph_rows)
    # Ensure region column
    if "region" not in df.columns:
        # adapt if r.name present
        for k in ["r.name", "name"]:
            if k in df.columns:
                df = df.rename(columns={k: "region"})
                break
    # keep features
    for f in MORPH_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df[MORPH_FEATURES] = df[MORPH_FEATURES].astype(float).fillna(0.0)
    # aggregate by region (mean)
    agg = df.groupby("region", as_index=True)[MORPH_FEATURES].mean()
    Xz = zscore(agg.to_numpy())
    return agg, Xz

def build_mole_matrix(mole_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
    if not mole_rows:
        return pd.DataFrame(columns=["region","subclass","value"]), np.zeros((0,0))
    df = pd.DataFrame(mole_rows)
    # standardize naming
    for a in ["r.acronym","r.name"]:
        if a in df.columns and "region" not in df.columns:
            df = df.rename(columns={a: "region"})
    for a in ["s.name","subclass_name"]:
        if a in df.columns and "subclass" not in df.columns:
            df = df.rename(columns={a: "subclass"})
    if "value" not in df.columns:
        # guess value column from known aliases
        for a in ["pct_cells","weight","pct","count"]:
            if a in df.columns:
                df = df.rename(columns={a: "value"})
                break
        if "value" not in df.columns:
            df["value"] = 0.0
    df["value"] = df["value"].astype(float).fillna(0.0)
    # TF-IDF normalization
    tf = tfidf(df, col_item="region", col_feat="subclass", col_value="value")
    return tf, tf.to_numpy()

def qc_stats(D: np.ndarray) -> Optional[Dict[str, Any]]:
    if D is None or D.size == 0:
        return None
    tri = D[np.triu_indices(D.shape[0], k=1)]
    if tri.size == 0:
        return None
    return {
        "min": float(np.min(tri)),
        "max": float(np.max(tri)),
        "mean": float(np.mean(tri)),
        "median": float(np.median(tri)),
        "std": float(np.std(tri)),
        "n_pairs": int(tri.size)
    }

def align_modalities(morph_df: pd.DataFrame, mole_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    reg_m = set(morph_df.index.tolist())
    reg_g = set(mole_df.index.tolist())
    inter = sorted(reg_m.intersection(reg_g))
    return inter, morph_df.loc[inter], mole_df.loc[inter]  # may be empty

def mismatch_index_matrix(Dm: np.ndarray, Dg: np.ndarray) -> Optional[np.ndarray]:
    if Dm is None or Dg is None:
        return None
    if Dm.shape != Dg.shape or Dm.size == 0:
        return None
    return np.abs(Dg - Dm)

def list_top_pairs(Dm: np.ndarray, Dg: np.ndarray, regions: List[str], morph_max=0.25, mole_min=0.6, topk=30) -> List[Dict[str, Any]]:
    if Dm is None or Dg is None or len(regions) == 0:
        return []
    n = len(regions)
    items = []
    for i in range(n):
        for j in range(i+1, n):
            dm = float(Dm[i, j])
            dg = float(Dg[i, j])
            if dm <= morph_max and dg >= mole_min:
                items.append({"region_a": regions[i], "region_b": regions[j], "morph_dist": dm, "mole_dist": dg, "delta": float(dg - dm)})
    items.sort(key=lambda x: (-x["delta"], x["morph_dist"]))
    return items[:topk]

def compute_projection_strength(edges: List[Dict[str, Any]]) -> Dict[str, float]:
    if not edges:
        return {}
    df = pd.DataFrame(edges)
    # standardize names
    for k in ["src","source","source_region","src_region"]:
        if k in df.columns and "source" not in df.columns:
            df = df.rename(columns={k:"source"})
    for k in ["dst","target","target_region","dst_region"]:
        if k in df.columns and "target" not in df.columns:
            df = df.rename(columns={k:"target"})
    if "strength" not in df.columns:
        for k in ["weight","count"]:
            if k in df.columns:
                df = df.rename(columns={k:"strength"})
                break
    if "strength" not in df.columns:
        df["strength"] = 1.0
    df["strength"] = df["strength"].astype(float).fillna(0.0)
    out = df.groupby("source")["strength"].sum().to_dict()
    return out

def subclass_projection_correlation(mole_df: pd.DataFrame, proj_strength: Dict[str, float]) -> pd.DataFrame:
    if mole_df is None or mole_df.empty or not proj_strength:
        return pd.DataFrame(columns=["subclass","pearson_r","n_regions"])
    common = sorted(set(mole_df.index).intersection(set(proj_strength.keys())))
    if not common:
        return pd.DataFrame(columns=["subclass","pearson_r","n_regions"])
    X = mole_df.loc[common]  # regions x subclasses
    y = np.array([proj_strength[r] for r in common], dtype=float)
    rows = []
    for sc in X.columns:
        r = pearson_corr(X[sc].to_numpy(), y)
        rows.append({"subclass": sc, "pearson_r": r, "n_regions": len(common)})
    df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    return df
