from __future__ import annotations
from typing import List, Dict

class ArgumentGraph:
    def __init__(self):
        self.claims: List[str] = []
        self.evidences: List[Dict] = []

    def add_claim(self, claim: str):
        if claim not in self.claims:
            self.claims.append(claim)

    def add_evidence(self, claim: str, evidence_content: str, source_query: str, strength: str="Moderate", reasoning_path: str=""):
        self.add_claim(claim)
        self.evidences.append({
            "claim": claim,
            "evidence_content": evidence_content,
            "source_query": source_query,
            "strength": strength,
            "reasoning_path": reasoning_path,
        })

    def to_meta(self):
        return {"claims": self.claims, "evidences": self.evidences}
