# -*- coding: utf-8 -*-
from typing import Optional, List
import os
import numpy as np

def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception as e:
        print(f"[viz] Matplotlib not available: {e}")
        return False

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def scatter_morph_vs_mole(Dm, Dg, out_path: str, title: str = "Morph vs Molecular distances"):
    if not _safe_import_matplotlib():
        return None
    import matplotlib.pyplot as plt
    ensure_dir(out_path)
    x = Dm[np.triu_indices(Dm.shape[0], 1)]
    y = Dg[np.triu_indices(Dg.shape[0], 1)]
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xlabel("Morphological distance (1 - cosine)")
    plt.ylabel("Molecular distance (1 - cosine/JSD)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def heatmap_matrix(M, labels: Optional[List[str]], out_path: str, title: str = "Matrix heatmap"):
    if not _safe_import_matplotlib():
        return None
    import matplotlib.pyplot as plt
    ensure_dir(out_path)
    plt.figure(figsize=(6,5))
    plt.imshow(M, aspect='auto', interpolation='nearest')
    plt.colorbar(label="value")
    if labels and len(labels) <= 30:
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def bar_top_corr(df, k: int, out_path: str, title: str = "Top subclass vs projection correlation"):
    if not _safe_import_matplotlib():
        return None
    import matplotlib.pyplot as plt
    ensure_dir(out_path)
    top = df.head(k)
    plt.figure(figsize=(7,5))
    plt.barh(top["subclass"][::-1], top["pearson_r"][::-1])
    plt.xlabel("Pearson r (projection strength)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
