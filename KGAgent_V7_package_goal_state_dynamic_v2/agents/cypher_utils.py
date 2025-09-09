import re

READONLY_PATTERNS = [
    r"(?i)\bCREATE\b",
    r"(?i)\bMERGE\b",
    r"(?i)\bDELETE\b",
    r"(?i)\bSET\b",
    r"(?i)\bDROP\b",
    r"(?i)\bREMOVE\b",
    r"(?i)\bCALL\s+db\.msql\b", # just in case
]

def sanitize_query(q: str) -> str:
    """Reject non-readonly queries by regex checks. Return stripped cypher if safe."""
    q = q.strip().rstrip(";")
    for pat in READONLY_PATTERNS:
        if re.search(pat, q):
            raise ValueError("Only read-only Cypher is allowed by guardrails.")
    return q

def extract_json_block(s: str):
    # lenient extraction of the first {...} JSON-looking block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None
