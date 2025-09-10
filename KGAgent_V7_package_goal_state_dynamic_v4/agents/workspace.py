from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import json


class Workspace:
    """
    In-memory workspace storing dataframes and lightweight objects.
    Enhanced to better support context retention between agent steps.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._counter = 0
        self._context_memory: List[str] = []  # Tracks important context information

    # Dataframe handling
    def put_df(self, key: str, df: pd.DataFrame, description: Optional[str] = None):
        """Store a dataframe with optional description for better context tracking."""
        self._store[key] = df
        if description:
            self._add_context(f"DataFrame '{key}': {description}")

    def get_df(self, key: str) -> pd.DataFrame:
        if key not in self._store:
            raise KeyError(f"Workspace has no key '{key}'")
        obj = self._store[key]
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Key '{key}' is not a DataFrame")
        return obj

    def put(self, key: str, obj: Any, importance: str = "normal"):
        """
        Store an object with importance level.
        Importance levels: "critical" (key result), "high", "normal", "low"
        """
        self._store[key] = obj

        # For critical/high importance items, add to context memory
        if importance in ["critical", "high"]:
            value_str = str(obj)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            self._add_context(f"{importance.upper()}: '{key}' = {value_str}")

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def alloc_query_df_key(self) -> str:
        self._counter += 1
        return f"df.query.{self._counter}"

    def add_important_result(self, key: str, value: Any, description: str):
        """Explicitly store an important result with description for reasoning."""
        self.put(key, value, importance="critical")
        self._add_context(f"CRITICAL RESULT: {description} -> '{key}' = {value}")

    def _add_context(self, context: str):
        """Add information to context memory, keeping the most recent/important items."""
        self._context_memory.append(context)
        if len(self._context_memory) > 20:  # Keep only the most recent context items
            self._context_memory = self._context_memory[-20:]

    def keys(self):
        return list(self._store.keys())

    def preview(self) -> str:
        """Enhanced preview showing regular items and highlighting critical context."""
        lines = []

        # First show critical context information
        if self._context_memory:
            lines.append("--- CRITICAL CONTEXT ---")
            lines.extend(self._context_memory)
            lines.append("--- WORKSPACE ITEMS ---")

        # Then show regular workspace items
        for k, v in self._store.items():
            if hasattr(v, "shape"):
                try:
                    shape = v.shape
                except Exception:
                    shape = "N/A"
                lines.append(f"{k} : DataFrame shape={shape}")
                # For small dataframes, show a preview
                if hasattr(v, "shape") and v.shape[0] <= 5:
                    try:
                        preview = v.head().to_string()
                        lines.append(f"  Preview:\n{preview}")
                    except:
                        pass
            else:
                short = str(v)
                if len(short) > 120:
                    short = short[:117] + "..."
                lines.append(f"{k} : {short}")

        return "\n".join(lines) if lines else "(empty)"

    def to_meta(self) -> dict:
        meta = {
            "items": {},
            "context_memory": self._context_memory
        }

        for k, v in self._store.items():
            if isinstance(v, pd.DataFrame):
                meta["items"][k] = {
                    "type": "DataFrame",
                    "shape": list(v.shape),
                    "columns": list(v.columns),
                }
                # For small dataframes, include the actual data
                if v.shape[0] <= 10:
                    try:
                        meta["items"][k]["data"] = v.to_dict('records')
                    except:
                        pass
            else:
                s = str(v)
                if len(s) > 200:
                    s = s[:197] + "..."
                meta["items"][k] = {"type": "object", "repr": s}

        return meta