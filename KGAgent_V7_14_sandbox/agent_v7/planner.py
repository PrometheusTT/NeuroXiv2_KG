# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import json
from .llm import LLM

OPS_SCHEMA = {
    "fetch_morph_table": {"desc": "Return rows of region-level morphology features."},
    "fetch_region_subclass_table": {"desc": "Return (region, subclass, value) long table."},
    "fetch_projection_edges": {"desc": "Return directed edges between regions with weight/strength/count."},
    "compute_modality_matrices": {"desc": "Build morphology & molecular matrices, compute pairwise distances, QC."},
    "mismatch_pairs": {"desc": "List region pairs with similar morphology but different molecular profiles; configurable thresholds."},
    "analyze_subclass_projection_corr": {"desc": "Correlate per-region subclass abundance vs projection out-strength."},
    "rank_regions_by_subclass": {"desc": "Rank regions by a given subclass keyword occurrence (case-insensitive).", "args": {"keyword": "str"}},
}

PLAN_PROMPT = """You are a planning agent that outputs JSON only.
Given a scientific question, produce a plan as:
{{
  "steps": [
    {{"op": "...", "args": {{}} }},
    ...
  ]
}}
You can choose from these operators:
{ops_schema}

General rules:
- Use compute_modality_matrices only if the question requires cross-modality comparison.
- Use mismatch_pairs only if the question explicitly asks for similar morphology but different molecular features (or mismatch).
- Use fetch_projection_edges when question mentions projection/connectivity/target/innervation.
- Use analyze_subclass_projection_corr if question links subclasses to connectivity or prediction of connections.
- Always start with fetch_morph_table and fetch_region_subclass_table to keep analysis grounded in KG data.
Output JSON only.
"""

class Planner:
    def __init__(self, openai_api_key: str = "", planner_model: str = "gpt-4o", summarizer_model: str = "gpt-4o"):
        self.llm = LLM(openai_api_key=openai_api_key, model=planner_model)

    def plan(self, question: str) -> Dict[str, Any]:
        # Try LLM first
        content = None
        if self.llm:
            prompt = PLAN_PROMPT.format(ops_schema=json.dumps(OPS_SCHEMA, ensure_ascii=False, indent=2))
            messages = [
                {"role": "system", "content": "You are a rigorous planning agent that outputs JSON only."},
                {"role": "user", "content": prompt + "\n\nQuestion:\n" + question}
            ]
            content = self.llm.chat(messages, temperature=0.0)

        if content:
            try:
                js = json.loads(content)
                if isinstance(js, dict) and "steps" in js:
                    return js
            except Exception:
                pass

        # Fallback heuristic (generic, no problem-specific overfitting)
        steps: List[Dict[str, Any]] = [{"op": "fetch_morph_table"}, {"op": "fetch_region_subclass_table"}]
        qlower = question.lower()
        if any(k in qlower for k in ["projection", "connect", "target", "innervat"]):
            steps.append({"op": "fetch_projection_edges"})
            if any(k in qlower for k in ["predict", "associated", "correlat"]):
                steps.append({"op": "analyze_subclass_projection_corr"})
        if all(k in qlower for k in ["similar", "morpholog", "molecular"]) or "mismatch" in qlower:
            steps.append({"op": "compute_modality_matrices"})
            steps.append({"op": "mismatch_pairs"})
        # Allow keyword-guided ranking without hardcoding any specific subclass
        for kw in ["car3", "microglia", "astro", "oligo", "pyramidal"]:
            if kw in qlower:
                steps.append({"op": "rank_regions_by_subclass", "args": {"keyword": kw}})
                break
        return {"steps": steps}
