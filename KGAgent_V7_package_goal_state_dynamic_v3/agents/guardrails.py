from __future__ import annotations
from typing import Dict, Tuple

# Minimal guardrails: allow only a subset of tools given intent.
# We avoid embedding domain heuristics; classification of intent is handled by LLM.
INTENT_TOOLS = {
    "generic": {"get_schema", "execute_cypher", "save_as", "compute_mismatch", "final"},
    "mismatch": {"get_schema", "execute_cypher", "save_as", "compute_mismatch", "final"},
    "car3_analysis": {"get_schema", "execute_cypher", "save_as", "final"},
}

def validate_action(intent: str, action: str) -> Tuple[bool, str]:
    allow = INTENT_TOOLS.get(intent, INTENT_TOOLS["generic"])
    if action in allow:
        return True, ""
    return False, f"Action '{action}' not allowed under intent '{intent}'. Allowed: {sorted(list(allow))}"
