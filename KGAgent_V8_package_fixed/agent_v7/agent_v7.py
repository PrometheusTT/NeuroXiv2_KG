import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

from .neo4j_exec import Neo4jExec
from .schema_cache import SchemaCache
from .operators import example_planner_prompt, validate_and_fix_cypher, NEUROXIV_TEMPLATES
from .llm import LLMClient, ToolSpec
from .tools_stats import (
    mismatch_index, basic_stats, detect_projection_patterns,
    compute_network_metrics, identify_outliers
)

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Scientific hypothesis with testable predictions"""
    claim: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    test_queries: List[str]


class KGAgentV7:
    """
    AIPOM-CoT Agent: Autonomous Inference through Pattern Observation and Mismatch
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str, database: str,
                 openai_api_key: str = None,
                 planner_model: str = "gpt-4o", summarizer_model: str = "gpt-4o"):
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()
        with self.db.driver.session(database=database) as s:
            self.schema.load_from_db(s)
        self.llm = LLMClient(api_key=openai_api_key, planner_model=planner_model,
                             summarizer_model=summarizer_model)

        # Knowledge accumulator
        self.accumulated_insights = []
        self.hypotheses = []
        self.discovered_patterns = defaultdict(list)

        # Enhanced tool set
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> List[ToolSpec]:
        """Initialize comprehensive tool set for neuroscience analysis"""
        return [
            ToolSpec(
                name="get_schema",
                description="Return KG schema with labels, relationships and properties.",
                parameters={"type": "object", "properties": {}}
            ),
            ToolSpec(
                name="neo4j_query",
                description="Execute Cypher query with automatic validation.",
                parameters={"type": "object", "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "object"}
                }, "required": ["query"]}
            ),
            ToolSpec(
                name="compute_mismatch_index",
                description="Calculate morphology-molecular mismatch for brain regions.",
                parameters={"type": "object", "properties": {
                    "region_a": {"type": "string"},
                    "region_b": {"type": "string"},
                    "metric": {"type": "string", "enum": ["L1", "L2", "COS"]}
                }, "required": ["region_a", "region_b"]}
            ),
            ToolSpec(
                name="detect_patterns",
                description="Identify patterns in projection/connectivity data.",
                parameters={"type": "object", "properties": {
                    "data": {"type": "array"},
                    "pattern_type": {"type": "string", "enum": ["projection", "hierarchy", "cluster"]}
                }, "required": ["data", "pattern_type"]}
            ),
            ToolSpec(
                name="network_analysis",
                description="Compute network metrics (centrality, modularity, etc).",
                parameters={"type": "object", "properties": {
                    "edges": {"type": "array"},
                    "metrics": {"type": "array", "items": {"type": "string"}}
                }, "required": ["edges"]}
            )
        ]

    def think(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        THINK phase: Decompose question and generate hypotheses
        """
        system = """You are a computational neuroscientist using AIPOM-CoT reasoning.
        Your task: Decompose questions into testable components and generate hypotheses."""

        user = f"""Question: {question}

Current Context:
- Previous insights: {json.dumps(self.accumulated_insights[-3:] if self.accumulated_insights else [])}
- Active hypotheses: {json.dumps([h.claim for h in self.hypotheses[:3]])}
- Discovered patterns: {json.dumps(dict(self.discovered_patterns))}

Following the AIPOM-CoT framework (as shown in the paper figures):
1. Identify key entities (regions, cell types, molecules)
2. Determine relationships of interest (projections, morphology, expression)
3. Generate 2-3 specific, testable hypotheses
4. Design queries to test each hypothesis

Return JSON:
{{
    "entities": {{
        "primary": ["list of main entities"],
        "comparison": ["entities for comparison if applicable"]
    }},
    "relationships": ["PROJECT_TO", "HAS_SUBCLASS", etc],
    "hypotheses": [
        {{
            "claim": "Specific testable claim",
            "rationale": "Why this might be true",
            "test_approach": "How to test it",
            "expected_pattern": "What pattern would support this"
        }}
    ],
    "initial_queries": [
        {{
            "purpose": "What this query tests",
            "query": "MATCH ... RETURN ... LIMIT ...",
            "expected_insights": "What we hope to learn"
        }}
    ]
}}"""

        result = self.llm.run_planner_json(system, user)
        return json.loads(result)

    def act(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ACT phase: Execute queries and computations
        """
        results = []

        # Execute planned queries
        for query_spec in plan.get("initial_queries", []):
            q = validate_and_fix_cypher(self.schema, query_spec["query"])
            res = self.db.run(q)

            # Add metadata for interpretation
            res["purpose"] = query_spec["purpose"]
            res["expected_insights"] = query_spec.get("expected_insights", "")
            results.append(res)

        # Execute hypothesis-specific queries if needed
        for hyp in plan.get("hypotheses", []):
            if "test_approach" in hyp and "MATCH" in hyp.get("test_approach", ""):
                q = validate_and_fix_cypher(self.schema, hyp["test_approach"])
                res = self.db.run(q)
                res["hypothesis"] = hyp["claim"]
                results.append(res)

        return results

    def observe(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        OBSERVE phase: Extract patterns and insights from results
        """
        observations = {
            "data_summary": [],
            "patterns": [],
            "anomalies": [],
            "statistical_findings": []
        }

        for res in results:
            if not res["success"] or not res["data"]:
                continue

            # Basic statistics
            if res["data"] and any("value" in row for row in res["data"]):
                values = [row["value"] for row in res["data"] if "value" in row]
                stats = basic_stats(values)
                observations["statistical_findings"].append({
                    "purpose": res.get("purpose", "Unknown"),
                    "stats": stats,
                    "sample_size": len(values)
                })

            # Pattern detection for projection data
            if any("PROJECT_TO" in res.get("query", "") for res in results):
                patterns = detect_projection_patterns(res["data"])
                if patterns["hub_regions"]:
                    observations["patterns"].append({
                        "type": "connectivity_hubs",
                        "regions": patterns["hub_regions"],
                        "significance": "High connectivity nodes identified"
                    })

            # Detect outliers
            outliers = identify_outliers(res["data"])
            if outliers:
                observations["anomalies"].extend(outliers)

            # Summarize key findings
            observations["data_summary"].append({
                "query_purpose": res.get("purpose", ""),
                "rows_returned": res["rows"],
                "key_finding": self._extract_key_finding(res["data"][:10])
            })

        return observations

    def reflect(self, question: str, observations: Dict[str, Any],
                hypotheses: List[Dict]) -> Dict[str, Any]:
        """
        REFLECT phase: Evaluate hypotheses and decide next steps
        """
        system = """You are reflecting on neuroscience data to evaluate hypotheses.
        Focus on: pattern recognition, hypothesis validation, and insight generation."""

        user = f"""Original Question: {question}

Hypotheses Being Tested:
{json.dumps(hypotheses, indent=2)}

Observations from Data:
{json.dumps(observations, indent=2)}

Accumulated Knowledge:
{json.dumps(self.accumulated_insights[-5:] if self.accumulated_insights else [])}

Perform deep reflection:
1. Which hypotheses are supported/refuted by the data?
2. What unexpected patterns emerged?
3. What new questions arise from these findings?
4. Should we continue investigating or do we have sufficient evidence?

Return JSON:
{{
    "hypothesis_evaluation": [
        {{
            "hypothesis": "original claim",
            "verdict": "supported/refuted/inconclusive",
            "confidence": 0.0-1.0,
            "key_evidence": ["list of specific findings"]
        }}
    ],
    "new_insights": [
        "Novel finding 1",
        "Novel finding 2"
    ],
    "emerging_patterns": {{
        "pattern_name": "description and significance"
    }},
    "continue": true/false,
    "reason": "Why continue or stop",
    "next_investigations": [
        {{
            "focus": "What to investigate next",
            "approach": "How to investigate it",
            "query": "Specific Cypher query if applicable"
        }}
    ]
}}"""

        result = self.llm.run_with_tools(system, user, tools=[], tool_router=lambda n, a: {})
        return json.loads(result)

    def _compute_advanced_metrics(self, data: List[Dict], metric_type: str) -> Dict:
        """Compute advanced neuroscience-specific metrics"""
        if metric_type == "morphology_molecular_mismatch":
            # Extract regions with both morphology and molecular data
            regions = {}
            for row in data:
                if "region" in row:
                    region = row["region"]
                    if region not in regions:
                        regions[region] = {"morphology": {}, "molecular": {}}
                    # Populate based on data type
                    if "axonal_length" in row:
                        regions[region]["morphology"] = {
                            k: v for k, v in row.items()
                            if k in ["axonal_length", "dendritic_length", "axonal_branches"]
                        }
                    if "subclass" in row:
                        regions[region]["molecular"][row["subclass"]] = row.get("value", 0)

            # Compute pairwise mismatch
            mismatches = []
            region_list = list(regions.keys())
            for i in range(len(region_list)):
                for j in range(i + 1, len(region_list)):
                    r1, r2 = region_list[i], region_list[j]
                    if regions[r1]["morphology"] and regions[r2]["morphology"]:
                        mismatch = mismatch_index(
                            regions[r1]["morphology"], regions[r2]["morphology"],
                            regions[r1]["molecular"], regions[r2]["molecular"]
                        )
                        mismatches.append({
                            "pair": (r1, r2),
                            "mismatch_index": mismatch
                        })

            return {"mismatches": sorted(mismatches, key=lambda x: x["mismatch_index"], reverse=True)[:10]}

        return {}

    def generate_insights(self, question: str, all_results: Dict) -> str:
        """Generate publication-quality neuroscience insights"""
        system = """You are a senior neuroscientist writing findings for a high-impact paper.
        Your insights should be specific, quantitative, and biologically meaningful.
        Reference specific brain regions, cell types, and molecular markers by name."""

        user = f"""Question: {question}

Complete Analysis Results:
{json.dumps(all_results, indent=2)}

Generate insights following this structure:

## Primary Discovery
State the most important finding with specific numbers and region names.
Example: "Car3+ neurons in CLA show 73% preferential targeting to entorhinal areas (ENTl, ENTm) 
compared to 12% in neighboring cortical regions (p<0.001)."

## Mechanistic Interpretation  
What does this mean for brain function and neural circuits?
Example: "This projection pattern suggests Car3 defines a specialized cortico-limbic pathway 
that may mediate attention-memory integration."

## Morphological-Molecular Relationship
Describe any mismatch or correlation between morphology and molecular identity.
Example: "The mismatch index of 0.82 between CLA and RSPagl indicates morphological similarity 
despite distinct molecular profiles, suggesting convergent circuit functions."

## Network Organization
Describe the network topology and its implications.
Example: "CLA emerges as a connectivity hub with betweenness centrality of 0.43, 
positioning it as a critical relay between cortical and subcortical systems."

## Biological Significance
Connect findings to broader neuroscience knowledge and potential clinical relevance.

## Future Directions
Propose specific, testable hypotheses based on these findings.

Use the actual data, cite specific numbers, and maintain scientific rigor."""

        return self.llm.run_advanced_reasoning(system, user)

    def answer(self, question: str, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Main AIPOM-CoT reasoning loop
        """
        logger.info(f"Starting AIPOM-CoT analysis for: {question}")

        all_results = []
        all_observations = []
        context = {}

        for round_num in range(1, max_rounds + 1):
            logger.info(f"Round {round_num}/{max_rounds}")

            # THINK: Generate hypotheses and plan
            thought = self.think(question, context)

            # ACT: Execute queries
            action_results = self.act(thought)
            all_results.extend(action_results)

            # OBSERVE: Extract patterns
            observations = self.observe(action_results)
            all_observations.append(observations)

            # REFLECT: Evaluate and decide
            reflection = self.reflect(question, observations, thought.get("hypotheses", []))

            # Update knowledge base
            self.accumulated_insights.extend(reflection.get("new_insights", []))
            for pattern_name, pattern_desc in reflection.get("emerging_patterns", {}).items():
                self.discovered_patterns[pattern_name].append(pattern_desc)

            # Check if we should continue
            if not reflection.get("continue", False):
                logger.info(f"Stopping at round {round_num}: {reflection.get('reason')}")
                break

            # Prepare context for next round
            context = {
                "previous_observations": observations,
                "evaluated_hypotheses": reflection.get("hypothesis_evaluation", []),
                "next_focus": reflection.get("next_investigations", [])
            }

        # Generate final insights
        final_insights = self.generate_insights(question, {
            "rounds_completed": round_num,
            "hypotheses_tested": thought.get("hypotheses", []),
            "key_findings": all_observations,
            "patterns_discovered": dict(self.discovered_patterns),
            "accumulated_insights": self.accumulated_insights
        })

        return {
            "question": question,
            "rounds": round_num,
            "hypotheses": thought.get("hypotheses", []),
            "discoveries": {
                "patterns": dict(self.discovered_patterns),
                "insights": self.accumulated_insights,
                "statistical_findings": [obs["statistical_findings"] for obs in all_observations]
            },
            "raw_results": all_results,
            "final_answer": final_insights
        }

    def _extract_key_finding(self, data_sample: List[Dict]) -> str:
        """Extract a key finding from data sample"""
        if not data_sample:
            return "No data returned"

        # Look for interesting patterns
        if "region" in data_sample[0] and "value" in data_sample[0]:
            top_region = max(data_sample, key=lambda x: x.get("value", 0))
            return f"Highest value in {top_region['region']}: {top_region['value']}"

        return f"Retrieved {len(data_sample)} records"