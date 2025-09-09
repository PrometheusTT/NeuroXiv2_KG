from __future__ import annotations
from typing import Any, Dict
import pandas as pd

class Workspace:
    """
    In-memory workspace storing dataframes and lightweight objects.
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._counter = 0

    # Dataframe handling
    def put_df(self, key: str, df: pd.DataFrame):
        self._store[key] = df

    def get_df(self, key: str) -> pd.DataFrame:
        if key not in self._store:
            raise KeyError(f"Workspace has no key '{key}'")
        obj = self._store[key]
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Key '{key}' is not a DataFrame")
        return obj

    def put(self, key: str, obj: Any):
        self._store[key] = obj

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def alloc_query_df_key(self) -> str:
        self._counter += 1
        return f"df.query.{self._counter}"

    def keys(self):
        return list(self._store.keys())

    def preview(self) -> str:
        lines = []
        for k,v in self._store.items():
            if hasattr(v, "shape"):
                try:
                    shape = v.shape
                except Exception:
                    shape = "N/A"
                lines.append(f"{k} : DataFrame shape={shape}")
            else:
                short = str(v)
                if len(short) > 120:
                    short = short[:117] + "..."
                lines.append(f"{k} : {short}")
        return "\n".join(lines) if lines else "(empty)"

    def to_meta(self) -> dict:
        meta = {}
        for k,v in self._store.items():
            if isinstance(v, pd.DataFrame):
                meta[k] = {
                    "type": "DataFrame",
                    "shape": list(v.shape),
                    "columns": list(v.columns),
                }
            else:
                s = str(v)
                if len(s) > 200:
                    s = s[:197] + "..."
                meta[k] = {"type":"object","repr": s}
        return meta
