from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def drop_near_zero_variance(df: pd.DataFrame, eps: float = 1e-9) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with near-zero variance (or single unique value).
    Returns filtered df and list of removed columns.
    """
    to_drop = []
    for c in df.columns:
        series = df[c].dropna()
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            to_drop.append(c)
            continue
        var = float(series.var()) if len(series) > 1 else 0.0
        if var <= eps:
            to_drop.append(c)
    return df.drop(columns=to_drop, errors="ignore"), to_drop

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + 1e-12)

def l2_normalize(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return pd.DataFrame(arr / norms, index=df.index, columns=df.columns)

def top_mismatch_pairs(morph_df: pd.DataFrame, mol_df: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    """
    Demonstration-only pure compute: find pairs with minimal morph distance and maximal mol distance.
    Assumes morph_df rows index: region, columns: features
            mol_df rows index: region, columns: subclass/proportions
    """
    # Align indices
    common = morph_df.index.intersection(mol_df.index)
    m = morph_df.loc[common].copy()
    g = mol_df.loc[common].copy()

    # Artifact control
    m, dropped = drop_near_zero_variance(m)
    m = zscore(m.fillna(0.0))
    g = l2_normalize(g.fillna(0.0))

    if m.shape[1] == 0:
        raise ValueError("All morphology features have near-zero variance; cannot compute distances.")

    # Pairwise distances
    X = m.to_numpy(dtype=float)
    Y = g.to_numpy(dtype=float)
    # Morphology: smaller better (similar), use L2
    morph_dist = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    # Molecular: larger better (different), use L2 between unit vectors in mol space
    mol_dist = np.sqrt(((Y[:, None, :] - Y[None, :, :])**2).sum(axis=2))

    # Score: small morph + large mol -> maximize (mol - morph); or simply filter near-0 morph then sort by mol
    n = len(common)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            md = morph_dist[i, j]
            gd = mol_dist[i, j]
            score = gd - md
            pairs.append((common[i], common[j], float(md), float(gd), float(score)))
    out = pd.DataFrame(pairs, columns=["r1","r2","morph_dist","mol_dist","score"])
    out = out.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    return out
