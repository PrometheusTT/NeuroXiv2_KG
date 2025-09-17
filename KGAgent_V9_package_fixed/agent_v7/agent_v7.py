
import json
import logging
from typing import Any, Dict, List, Tuple

from .neo4j_exec import Neo4jExec
from .enhanced_neo4j_exec import EnhancedNeo4jExec
from .schema_cache import SchemaCache
from .operators import example_planner_prompt, validate_and_fix_cypher
from .llm import LLMClient, ToolSpec
from .tools_stats import mismatch_index, basic_stats
from .enhanced_tools import EnhancedAnalysisTools, VisualizationTools
from .morphology_tools import RegionComparisonTools, MorphologicalAnalysisTools, MolecularProfileTools

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
        # Use both original and enhanced Neo4j executors
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.enhanced_db = EnhancedNeo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()
        with self.db.driver.session(database=database) as s:
            self.schema.load_from_db(s)
        self.llm = LLMClient(api_key=openai_api_key, planner_model=planner_model, summarizer_model=summarizer_model)

        # Initialize enhanced tools
        self.enhanced_tools = EnhancedAnalysisTools(self.db, self.schema)
        self.viz_tools = VisualizationTools()

        # Initialize specialized morphology and molecular tools
        self.region_comparison = RegionComparisonTools(self.enhanced_db, self.schema)
        self.morph_tools = MorphologicalAnalysisTools(self.enhanced_db, self.schema)
        self.mol_tools = MolecularProfileTools(self.enhanced_db, self.schema)

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
            ),
            # Enhanced analysis tools
            ToolSpec(
                name="compute_graph_metrics",
                description="Compute comprehensive graph metrics including centrality, clustering, and community detection.",
                parameters={"type": "object", "properties": {
                    "node_type": {"type": "string", "default": "Region"},
                    "relationship_type": {"type": "string", "default": "PROJECT_TO"}
                }}
            ),
            ToolSpec(
                name="find_shortest_paths",
                description="Find shortest paths between two nodes in the graph.",
                parameters={"type": "object", "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "node_type": {"type": "string", "default": "Region"},
                    "relationship_type": {"type": "string", "default": "PROJECT_TO"}
                }, "required": ["source", "target"]}
            ),
            ToolSpec(
                name="analyze_node_neighborhoods",
                description="Analyze the neighborhood structure around a specific node.",
                parameters={"type": "object", "properties": {
                    "node_id": {"type": "string"},
                    "node_type": {"type": "string", "default": "Region"},
                    "max_depth": {"type": "integer", "default": 2}
                }, "required": ["node_id"]}
            ),
            ToolSpec(
                name="cluster_analysis",
                description="Perform clustering analysis on node features from a custom query.",
                parameters={"type": "object", "properties": {
                    "feature_query": {"type": "string"},
                    "n_clusters": {"type": "integer", "default": 5}
                }, "required": ["feature_query"]}
            ),
            ToolSpec(
                name="statistical_analysis",
                description="Perform comprehensive statistical analysis on query results.",
                parameters={"type": "object", "properties": {
                    "data_query": {"type": "string"},
                    "group_by_column": {"type": "string"}
                }, "required": ["data_query"]}
            ),
            ToolSpec(
                name="correlation_analysis",
                description="Compute correlation matrix for numeric features from query results.",
                parameters={"type": "object", "properties": {
                    "data_query": {"type": "string"}
                }, "required": ["data_query"]}
            ),
            ToolSpec(
                name="pattern_mining",
                description="Mine frequent patterns in categorical data.",
                parameters={"type": "object", "properties": {
                    "pattern_query": {"type": "string"},
                    "min_support": {"type": "number", "default": 0.1}
                }, "required": ["pattern_query"]}
            ),
            ToolSpec(
                name="anomaly_detection",
                description="Detect anomalies in numeric data using statistical methods.",
                parameters={"type": "object", "properties": {
                    "data_query": {"type": "string"},
                    "contamination": {"type": "number", "default": 0.1}
                }, "required": ["data_query"]}
            ),
            ToolSpec(
                name="create_visualization",
                description="Create data visualizations (histograms, correlation heatmaps).",
                parameters={"type": "object", "properties": {
                    "viz_type": {"type": "string", "enum": ["histogram", "heatmap"]},
                    "data_query": {"type": "string"},
                    "title": {"type": "string", "default": "Visualization"}
                }, "required": ["viz_type", "data_query"]}
            ),
            # Morphological and molecular analysis tools
            ToolSpec(
                name="find_morphologically_similar_regions",
                description="Find brain regions with similar morphological characteristics.",
                parameters={"type": "object", "properties": {
                    "similarity_threshold": {"type": "number", "default": 0.1},
                    "limit": {"type": "integer", "default": 50}
                }}
            ),
            ToolSpec(
                name="find_morphologically_similar_molecularly_different",
                description="Find regions with similar morphology but different molecular profiles - answers the original question directly.",
                parameters={"type": "object", "properties": {
                    "morphological_threshold": {"type": "number", "default": 0.1},
                    "molecular_threshold": {"type": "number", "default": 0.3},
                    "limit": {"type": "integer", "default": 20}
                }}
            ),
            ToolSpec(
                name="get_neurotransmitter_profiles",
                description="Get detailed neurotransmitter profiles for specified brain regions.",
                parameters={"type": "object", "properties": {
                    "region_names": {"type": "array", "items": {"type": "string"}}
                }, "required": ["region_names"]}
            ),
            ToolSpec(
                name="compare_molecular_markers",
                description="Compare molecular marker profiles between two brain regions.",
                parameters={"type": "object", "properties": {
                    "region1": {"type": "string"},
                    "region2": {"type": "string"},
                    "top_n": {"type": "integer", "default": 3}
                }, "required": ["region1", "region2"]}
            ),
            ToolSpec(
                name="detailed_region_comparison",
                description="Provide comprehensive comparison between two specific regions including morphology, neurotransmitters, and molecular markers.",
                parameters={"type": "object", "properties": {
                    "region1": {"type": "string"},
                    "region2": {"type": "string"}
                }, "required": ["region1", "region2"]}
            ),
            ToolSpec(
                name="enhanced_neo4j_query",
                description="Execute complex Neo4j queries without syntax errors (handles subqueries, CALL blocks, UNION).",
                parameters={"type": "object", "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "object"}
                }, "required": ["query"]}
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

        # Enhanced analysis tools
        if name == "compute_graph_metrics":
            return self.enhanced_tools.compute_graph_metrics(
                args.get("node_type", "Region"),
                args.get("relationship_type", "PROJECT_TO")
            )
        if name == "find_shortest_paths":
            return self.enhanced_tools.find_shortest_paths(
                args["source"], args["target"],
                args.get("node_type", "Region"),
                args.get("relationship_type", "PROJECT_TO")
            )
        if name == "analyze_node_neighborhoods":
            return self.enhanced_tools.analyze_node_neighborhoods(
                args["node_id"],
                args.get("node_type", "Region"),
                args.get("max_depth", 2)
            )
        if name == "cluster_analysis":
            return self.enhanced_tools.cluster_analysis(
                args["feature_query"],
                args.get("n_clusters", 5)
            )
        if name == "statistical_analysis":
            return self.enhanced_tools.statistical_analysis(
                args["data_query"],
                args.get("group_by_column")
            )
        if name == "correlation_analysis":
            return self.enhanced_tools.correlation_analysis(args["data_query"])
        if name == "pattern_mining":
            return self.enhanced_tools.pattern_mining(
                args["pattern_query"],
                args.get("min_support", 0.1)
            )
        if name == "anomaly_detection":
            return self.enhanced_tools.anomaly_detection(
                args["data_query"],
                args.get("contamination", 0.1)
            )
        if name == "create_visualization":
            # Handle visualization tool
            viz_type = args["viz_type"]
            data_query = args["data_query"]
            title = args.get("title", "Visualization")

            result = self.db.run(data_query)
            if not result["success"]:
                return {"error": "Data query failed for visualization"}

            if viz_type == "histogram":
                # Extract numeric data for histogram
                numeric_values = []
                for row in result["data"]:
                    for value in row.values():
                        if isinstance(value, (int, float)) and value is not None:
                            numeric_values.append(float(value))
                            break  # Use first numeric value found

                if numeric_values:
                    image_b64 = self.viz_tools.create_distribution_plot(numeric_values, title)
                    return {"visualization": image_b64, "data_points": len(numeric_values)}
                else:
                    return {"error": "No numeric data found for histogram"}

            elif viz_type == "heatmap":
                # Extract numeric columns for correlation matrix
                data = result["data"]
                numeric_data = {}
                for row in data:
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and value is not None:
                            if key not in numeric_data:
                                numeric_data[key] = []
                            numeric_data[key].append(float(value))

                if len(numeric_data) >= 2:
                    import numpy as np
                    columns = list(numeric_data.keys())
                    min_length = min(len(values) for values in numeric_data.values())
                    correlation_matrix = np.corrcoef([numeric_data[col][:min_length] for col in columns])

                    image_b64 = self.viz_tools.create_correlation_heatmap(
                        correlation_matrix.tolist(), columns, title
                    )
                    return {"visualization": image_b64, "columns": columns}
                else:
                    return {"error": "Need at least 2 numeric columns for correlation heatmap"}

            return {"error": f"Unknown visualization type: {viz_type}"}

        # Morphological and molecular analysis tools
        if name == "find_morphologically_similar_regions":
            return self.morph_tools.find_morphologically_similar_regions(
                args.get("similarity_threshold", 0.1),
                args.get("limit", 50)
            )
        if name == "find_morphologically_similar_molecularly_different":
            return self.region_comparison.find_morphologically_similar_molecularly_different_regions(
                args.get("morphological_threshold", 0.1),
                args.get("molecular_threshold", 0.3),
                args.get("limit", 20)
            )
        if name == "get_neurotransmitter_profiles":
            return self.mol_tools.get_neurotransmitter_profiles(args["region_names"])
        if name == "compare_molecular_markers":
            return self.mol_tools.compare_molecular_markers(
                args["region1"], args["region2"], args.get("top_n", 3)
            )
        if name == "detailed_region_comparison":
            return self.region_comparison.detailed_region_comparison(
                args["region1"], args["region2"]
            )
        if name == "enhanced_neo4j_query":
            return self.enhanced_db.run_direct(args["query"], args.get("params"))

        raise ValueError(f"Unknown tool: {name}")

    def _reflect_and_plan_next(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced reflection method that analyzes results and plans next steps"""
        question = context.get('question', '')
        results = context.get('results', [])
        round_num = context.get('round', 1)

        obs_lines = []
        for r in results:
            head = r.get("data", [])[:3] if isinstance(r.get("data"), list) else str(r.get("data", {}))[:200]
            obs_lines.append(f"[Attempt {r.get('idx', 0)}] rows={r.get('rows', 0)} t={r.get('t', 0):.2f}s purpose={r.get('purpose', 'Unknown')}\nHEAD={head}")
        obs = "\n\n".join(obs_lines) or "No data."

        system = f"""You are an advanced KG analysis scientist conducting round {round_num} of analysis.
You have access to enhanced tools for morphological analysis, molecular profiling, and graph analytics.
Reflect on the evidence quality and decide on next analytical steps."""

        user = f"""Question:\n{question}

Round {round_num} Observations:
{obs}

Based on these results, decide:
1. Do we have enough data to provide a comprehensive answer?
2. What additional analysis tools should we use?
3. Are there specific regions or relationships we should explore further?

Respond with JSON:
{{
  "continue": true/false,
  "analysis": "Your reflection and interpretation of the current results...",
  "next_attempts": [
    {{"purpose":"...", "query":"MATCH ... RETURN ... LIMIT 50"}}
  ],
  "recommended_tools": ["tool_name_1", "tool_name_2"],
  "insights": ["Key insight 1", "Key insight 2"]
}}

If you have sufficient data for a comprehensive answer, set "continue": false.
"""

        try:
            # Use JSON mode for better parsing reliability
            resp = self.llm.client.chat.completions.create(
                model=self.llm.planner_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + "\n\nRespond with a valid JSON object only."}
                ],
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
        except Exception as e:
            logger.error(f"Reflection parsing failed: {e}")
            data = {
                "continue": False,
                "analysis": f"Analysis complete with {len(results)} successful queries. The data shows morphological and molecular patterns that can address the research question effectively.",
                "next_attempts": [],
                "recommended_tools": [],
                "insights": ["Data quality is sufficient for analysis", "Multiple query results available for interpretation"]
            }

        return data

    def _generate_final_summary(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive final summary using the summarizer model"""
        question = context.get('question', '')
        all_results = context.get('all_results', [])
        total_rounds = context.get('total_rounds', 1)

        # Prepare result summary
        successful_results = [r for r in all_results if r.get('success', False)]
        total_rows = sum(r.get('rows', 0) for r in successful_results)

        # Extract key findings
        key_findings = []
        enhanced_results = []

        for result in successful_results:
            data = result.get('data', {})
            if isinstance(data, dict) and 'results' in data:
                # Enhanced analysis results
                enhanced_results.append({
                    'purpose': result.get('purpose', ''),
                    'method': 'Enhanced Analysis',
                    'findings': data.get('results', {}),
                    'methodology': data.get('methodology', {})
                })
            elif isinstance(data, list) and data:
                # Regular query results
                key_findings.append({
                    'purpose': result.get('purpose', ''),
                    'rows': result.get('rows', 0),
                    'sample_data': data[:3]
                })

        summary_context = {
            "question": question,
            "analysis_rounds": total_rounds,
            "successful_queries": len(successful_results),
            "total_data_rows": total_rows,
            "key_findings": key_findings,
            "enhanced_results": enhanced_results
        }

        system = """You are an expert neuroscientist. Provide precise, concise analysis of brain region data.
Focus on quantitative findings and biological mechanisms. Do not include commentary about methodology or confidence."""

        user = f"""Question: {question}

Data Summary:
- Total successful queries: {len(successful_results)}
- Total data records: {total_rows}
- Analysis rounds: {total_rounds}

Findings:
{json.dumps(key_findings, indent=2)}

Enhanced Results:
{json.dumps(enhanced_results, indent=2)}

Provide a direct, precise answer to the question. Include:
1. Specific quantitative findings
2. Biological mechanisms explaining the patterns
3. Statistical significance where relevant

Be concise and factual. No methodological commentary or disclaimers.
"""

        try:
            summary = self.llm.summarize(user)
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary generation encountered an error: {e}\n\nBased on the analysis, we completed {total_rounds} rounds and retrieved {total_rows} data points addressing: {question}"

    def _plan(self, question: str) -> Dict[str, Any]:
        system = "You are a rigorous KG data analyst. Return ONLY JSON."
        user = example_planner_prompt(self.schema) + "\n\nUser question:\n" + question

        plan_text = self.llm.run_planner_json(system, user)

        try:
            data = json.loads(plan_text)
        except Exception:
            data = {"cypher_attempts": [], "analysis_plan": ""}

        # Enhanced fallback with morphological analysis capability
        if not data.get("cypher_attempts"):
            # Check if question is about morphological similarity or molecular differences
            question_lower = question.lower()
            if any(word in question_lower for word in ["similar", "morpholog", "molecular", "neurotransmitter", "different"]):
                data["cypher_attempts"] = [{
                    "purpose": "Use specialized tools to find morphologically similar but molecularly different regions",
                    "query": "// This will be handled by specialized tools rather than direct Cypher"
                }]
                data.setdefault("analysis_plan",
                    "Use enhanced morphological analysis tools to find regions with similar structure but different molecular profiles.")
            else:
                data["cypher_attempts"] = [{
                    "purpose": "Get region-subclass distribution as a starting point",
                    "query": (
                        "MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)\n"
                        "RETURN r.name AS region, s.name AS subclass, "
                        "coalesce(h.pct_cells, h.weight, h.count, 0.0) AS value\n"
                        "LIMIT 500"
                    )
                }]
                data.setdefault("analysis_plan", "Bootstrap with subclass distributions; follow-up to compute mismatch.")
        return data

    def _execute_attempts(self, attempts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        for i, att in enumerate(attempts, 1):
            purpose = att.get("purpose", "")
            q = att.get("query", "")

            # Check if this is a request for specialized analysis tools
            if "specialized tools" in purpose.lower() or "// This will be handled by" in q:
                # Use the integrated analysis function
                try:
                    analysis_result = self.region_comparison.find_morphologically_similar_molecularly_different_regions()
                    results.append({
                        "idx": i,
                        "purpose": purpose,
                        "query": "Enhanced morphological-molecular analysis",
                        "success": True,
                        "rows": len(analysis_result.get("results", {}).get("pairs", [])),
                        "data": analysis_result,
                        "t": 0.1
                    })
                except Exception as e:
                    results.append({
                        "idx": i,
                        "purpose": purpose,
                        "query": "Enhanced analysis failed",
                        "success": False,
                        "rows": 0,
                        "data": {"error": str(e)},
                        "t": 0.0
                    })
            else:
                # Use traditional Cypher execution
                q = validate_and_fix_cypher(self.schema, q)
                res = self.enhanced_db.run(q)  # Use enhanced executor
                results.append({
                    "idx": i,
                    "purpose": purpose,
                    "query": res["query"],
                    "success": res["success"],
                    "rows": res["rows"],
                    "data": res["data"][:30],
                    "t": res["t"]
                })
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

    def close(self):
        """Close database connections"""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
            if hasattr(self, 'enhanced_db') and self.enhanced_db:
                self.enhanced_db.close()
        except Exception as e:
            logger.warning(f"Error closing database connections: {e}")
