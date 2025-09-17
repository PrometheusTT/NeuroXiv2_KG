
from typing import Dict, Any, List, Tuple
import difflib

from .schema_cache import SchemaCache


def suggest_prop(schema: SchemaCache, label: str, prop: str) -> str:
    """Return best matching existing property for a label when prop doesn't exist."""
    if not schema.has_label(label):
        return prop
    candidates = list(schema.node_props[label].keys())
    match = difflib.get_close_matches(prop, candidates, n=1, cutoff=0.6)
    return match[0] if match else prop


def validate_and_fix_cypher(schema: SchemaCache, query: str) -> str:
    """
    Weak validator that tries to replace missing properties by closest matches per label.
    Only handles simple patterns like (n:Label) and n.prop references.
    """
    # 1) collect aliases -> labels
    import re
    alias_to_label = {}
    for m in re.finditer(r"\((\w+):([A-Za-z_][A-Za-z0-9_]*)\)", query):
        alias, label = m.group(1), m.group(2)
        alias_to_label[alias] = label

    # 2) find alias.prop occurrences
    def repl(match):
        alias, prop = match.group(1), match.group(2)
        label = alias_to_label.get(alias)
        if not label or schema.has_prop(label, prop):
            return f"{alias}.{prop}"
        fixed = suggest_prop(schema, label, prop)
        return f"{alias}.{fixed}"

    query = re.sub(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", repl, query)
    return query


# Few-shot templates you can expand for your KG
TEMPLATES = {
    "REGION_MORPH_TOPK": """
MATCH (r:Region)
WHERE r.axonal_length IS NOT NULL AND r.dendritic_length IS NOT NULL
RETURN r.region_id AS region_id, r.acronym AS acronym,
       r.axonal_length AS axon, r.dendritic_length AS dend,
       CASE WHEN r.dendritic_length>0 THEN r.axonal_length/r.dendritic_length ELSE 999 END AS ratio
ORDER BY ratio DESC
LIMIT 50
""",
    "PROJECT_TO_SUMMARY": """
MATCH (a:Region)-[p:PROJECT_TO]->(b:Region)
RETURN a.region_id AS src_id, a.acronym AS src, b.region_id AS dst_id, b.acronym AS dst,
       avg(coalesce(p.weight, p.strength, 0)) AS avg_weight,
       sum(coalesce(p.count, 1)) AS n_edges
ORDER BY avg_weight DESC
LIMIT 100
""",
    "REGION_SUBCLASS_DIST": """
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
RETURN r.region_id AS region_id, r.acronym AS region, s.name AS subclass, coalesce(h.pct_cells, h.weight, h.count, 0) AS value
LIMIT 2000
""",
    "REGION_MORPH_PROPS": """
MATCH (r:Region)
RETURN r.region_id AS region_id, r.acronym AS acronym,
       r.axonal_length AS axonal_length,
       r.dendritic_length AS dendritic_length,
       r.axonal_branches AS axonal_branches,
       r.dendritic_branches AS dendritic_branches,
       r.dendritic_maximum_branch_order AS dendritic_maximum_branch_order
LIMIT 500
"""
}


def example_planner_prompt(schema: SchemaCache) -> str:
    return f"""
You are planning Cypher queries to analyze a neuroscience KG.
Schema (excerpt):
{schema.summary_text()}

Respond in JSON with this shape:
{{
  "cypher_attempts": [
    {{"purpose": "short rationale", "query": "MATCH ... RETURN ... LIMIT 50"}}
  ],
  "analysis_plan": "how you will combine results and what metrics to compute"
}}

Constraints:
- Only READ queries.
- Prefer Region/Subclass and relations PROJECT_TO, HAS_SUBCLASS.
- Use properties you see in the schema. If unsure, prefer axonal_length, dendritic_length, *_branches, *_maximum_branch_order.
- Always include LIMIT.
"""
