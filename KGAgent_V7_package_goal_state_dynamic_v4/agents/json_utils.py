from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy/Pandas types to Python native types for JSON serialization.
    """
    # Handle NumPy/Pandas scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.ndarray, np.bool_)):
        return obj.tolist()
    if pd.__version__ >= '1.0.0' and isinstance(obj, pd.NA):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    if isinstance(obj, pd.Series):
        return obj.tolist()

    # Handle dict, list and tuples
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Return unchanged if not a special type
    return obj


def safe_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON string, handling NumPy and Pandas types.
    """
    return json.dumps(convert_numpy_types(obj), ensure_ascii=False, indent=2)


def safe_save_json(obj: Any, filepath: str) -> None:
    """
    Safely save an object to a JSON file, handling NumPy and Pandas types.
    """
    with open(filepath, "w") as f:
        json.dump(convert_numpy_types(obj), f, ensure_ascii=False, indent=2)