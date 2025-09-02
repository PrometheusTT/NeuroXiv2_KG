
import json
import logging
from typing import Any, Dict, List, Tuple

from .neo4j_exec import Neo4jExec
from .schema_cache import SchemaCache
from .operators import example_planner_prompt, validate_and_fix_cypher
from .llm import LLMClient, ToolSpec
from .tools_stats import mismatch_index, basic_stats

logger = logging.getLogger(__name__)


class KGAgentV7:
    """
    Autonomic KG + CoT agent:
    1) Loads schema, plans queries with an LLM (JSON).
    2) Validates/fixes Cypher against schema and executes robustly.
    3) Computes metrics (e.g., mismatch index) as tools.
    4) Reflects and iterates up to N rounds, then summarizes.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str, database: str,
                 openai_api_key: str = None,
                 planner_model: str = "gpt-5", summarizer_model: str = "gpt-4o"):
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()
        with self.db.driver.session(database=database) as s:
            self.schema.load_from_db(s)
        self.llm = LLMClient(api_key=openai_api_key, planner_model=planner_model, summarizer_model=summarizer_model)

        # Expose tools to the LLM if/when needed
        self.tools = [
            ToolSpec(
                name="get_schema",
                description="Return labels, relationships and properties summary.",
                parameters={"type": "object", "properties": {}}
            ),
            ToolSpec(
                name="neo4j_query",
                description="Run a read-only Cypher query. Must include LIMIT.",
                parameters={"type": "object", "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "object"}
                }, "required": ["query"]}
            ),
            ToolSpec(
                name="compute_mismatch_index",
                description="Compute | morph_dist - subclass_dist | given four vectors and a metric.",
                parameters={"type": "object", "properties": {
                    "morph_vec_a": {"type": "object"},
                    "morph_vec_b": {"type": "object"},
                    "subclass_vec_a": {"type": "object"},
                    "subclass_vec_b": {"type": "object"},
                    "metric": {"type": "string", "enum": ["L1", "L2", "COS"]}
                }, "required": ["morph_vec_a","morph_vec_b","subclass_vec_a","subclass_vec_b"]}
            )
        ]

    # tool router for Responses API path
    def _tool_router(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_schema":
            return {
                "labels": list(self.schema.node_props.keys()),
                "relationships": list(self.schema.rel_types.keys()),
                "properties": self.schema.node_props
            }
        if name == "neo4j_query":
            q = validate_and_fix_cypher(self.schema, args["query"])
            return self.db.run(q, args.get("params"))
        if name == "compute_mismatch_index":
            val = mismatch_index(**args)
            return {"value": val}
        raise ValueError(f"Unknown tool: {name}")

    def _plan(self, question: str) -> Dict[str, Any]:
        system = "You are a rigorous KG data analyst. Return ONLY JSON."
        user = example_planner_prompt(self.schema) + "\n\nUser question:\n" + question

        plan_text = self.llm.run_planner_json(system, user)

        try:
            data = json.loads(plan_text)
        except Exception:
            data = {"cypher_attempts": [], "analysis_plan": ""}

        # 兜底：若还是空，给一个最小可执行 attempt，避免全空跑不下去
        if not data.get("cypher_attempts"):
            data["cypher_attempts"] = [{
                "purpose": "Get region-subclass distribution as a starting point",
                "query": (
                    "MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)\n"
                    "RETURN r.acronym AS region, s.name AS subclass, "
                    "coalesce(h.pct_cells, h.weight, h.count, 0.0) AS value\n"
                    "LIMIT 500"
                )
            }]
            data.setdefault("analysis_plan", "Bootstrap with subclass distributions; follow-up to compute mismatch.")
        return data

    def _execute_attempts(self, attempts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        for i, att in enumerate(attempts, 1):
            q = validate_and_fix_cypher(self.schema, att.get("query", ""))
            res = self.db.run(q)
            results.append({"idx": i, "purpose": att.get("purpose"), "query": res["query"],
                            "success": res["success"], "rows": res["rows"], "data": res["data"][:30], "t": res["t"]})
        return results

    def _reflect(self, question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        obs_lines = []
        for r in results:
            head = r["data"][:3]
            obs_lines.append(f"[Attempt {r['idx']}] rows={r['rows']} t={r['t']:.2f}s purpose={r['purpose']}\nHEAD={head}")
        obs = "\n\n".join(obs_lines) or "No data."

        system = "You are a scientist that reflects on evidence quality and plans next queries."
        user = f"""Question:\n{question}

Observations:
{obs}

Decide whether we have enough to answer. If not, produce a new JSON:
{{
  "continue": true/false,
  "reason": "...",
  "new_attempts": [
    {{"purpose":"...", "query":"MATCH ... RETURN ... LIMIT 50"}}
  ],
  "metrics_to_compute": [{{"type":"mismatch","regions":["MOp","CLA"],"metric":"L1"}}]
}}
If enough, set "continue": false and write a short final answer draft in "draft".
"""
        text = self.llm.run_with_tools(system, user, tools=[], tool_router=lambda n, a: {})
        try:
            data = json.loads(text)
        except Exception:
            data = {"continue": False, "reason": "parse-failed", "draft": text, "new_attempts": [], "metrics_to_compute": []}
        return data

    def _compute_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for m in metrics:
            if m.get("type") == "mismatch":
                # naive example: pull vectors for two regions from prior-friendly queries
                regions = m.get("regions", [])
                metric = m.get("metric", "L1")
                if len(regions) != 2:
                    continue
                rA, rB = regions
                # subclass distributions
                q_sub = """
MATCH (r:Region {acronym:$acr})-[h:HAS_SUBCLASS]->(s:Subclass)
RETURN s.name AS subclass, coalesce(h.pct_cells, h.weight, h.count, 0.0) AS value
LIMIT 1000
"""
                dA = self.db.run(q_sub, {"acr": rA})["data"]
                dB = self.db.run(q_sub, {"acr": rB})["data"]
                subA = {d["subclass"]: float(d["value"]) for d in dA}
                subB = {d["subclass"]: float(d["value"]) for d in dB}
                # morphology vectors
                q_m = """
MATCH (r:Region {acronym:$acr})
RETURN r.axonal_length AS axonal_length, r.dendritic_length AS dendritic_length,
       r.axonal_branches AS axonal_branches, r.dendritic_branches AS dendritic_branches,
       r.dendritic_maximum_branch_order AS dendritic_maximum_branch_order
LIMIT 1
"""
                mA = self.db.run(q_m, {"acr": rA})["data"]
                mB = self.db.run(q_m, {"acr": rB})["data"]
                morphA = {k: float(v) for k, v in (mA[0] if mA else {}).items() if v is not None}
                morphB = {k: float(v) for k, v in (mB[0] if mB else {}).items() if v is not None}
                val = mismatch_index(morphA, morphB, subA, subB, metric=metric)
                out.append({"type": "mismatch", "regions": regions, "metric": metric, "value": val})
        return out

    def answer(self, question: str, max_rounds: int = 2) -> Dict[str, Any]:
        plan = self._plan(question)
        attempts = plan.get("cypher_attempts", [])
        all_results: List[Dict[str, Any]] = []
        metrics_all: List[Dict[str, Any]] = []

        for rnd in range(1, max_rounds + 1):
            exec_res = self._execute_attempts(attempts)
            all_results.extend(exec_res)

            reflect = self._reflect(question, exec_res)
            metrics = self._compute_metrics(reflect.get("metrics_to_compute", []))
            metrics_all.extend(metrics)

            if not reflect.get("continue", False):
                draft = reflect.get("draft", "")
                summary = self.llm.summarize(draft or json.dumps({"results": exec_res, "metrics": metrics}, ensure_ascii=False))
                return {
                    "rounds": rnd,
                    "plan": plan,
                    "results": all_results,
                    "metrics": metrics_all,
                    "final": summary
                }
            attempts = reflect.get("new_attempts", []) or []

        # If max rounds reached, summarize what we have
        summary = self.llm.summarize(json.dumps({"results": all_results, "metrics": metrics_all}, ensure_ascii=False))
        return {
            "rounds": max_rounds,
            "plan": plan,
            "results": all_results,
            "metrics": metrics_all,
            "final": summary
        }
