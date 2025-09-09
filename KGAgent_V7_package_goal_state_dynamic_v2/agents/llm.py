# Complete llm.py with all methods implemented

from __future__ import annotations
import os, json, httpx, re
from typing import List, Dict, Any, Optional
from .cypher_utils import extract_json_block

STEP_SYSTEM_PROMPT = """
You are an advanced data analysis agent with expertise in knowledge graph querying and scientific data interpretation.

CRITICAL RULES:
1. **Schema Adherence**: You MUST strictly follow the database schema provided. NEVER invent labels, relationships, or properties.
2. **Evidence-Based Reasoning**: Every conclusion must be traceable to specific data queries and results.
3. **Systematic Problem Solving**: Break complex problems into logical sub-queries.
4. **Self-Correction**: When queries fail, diagnose the root cause and adjust your approach.
5. **NO PREMATURE FINALIZATION**: Do NOT use the 'final' action until you have actually queried data and found answers.

Core Decision Process:
1) **Understand Current State**: Review the User Goal, Workspace State (including SCHEMA), and History.
2) **Diagnose Previous Results**: If the last action failed or returned unexpected results, identify WHY.
3) **Plan Next Action**: Choose the single most logical next step based on evidence.
4) **Generate Valid Arguments**: Ensure all Cypher queries match the schema EXACTLY.

For Car3 Analysis:
- First, find which regions have Car3 subclass data
- Query for the proportion/percentage of Car3 in each region
- Identify the region with highest Car3 proportion
- Then query for morphology and projection data for that specific region

Failure Handling Protocol:
If the previous action resulted in an error or '0 rows returned':
  a) Diagnose Cause: Formulate a hypothesis for why the failure occurred.
     - Schema Mismatch: Check if labels/properties exist
     - Empty Result: Data might not exist or filters too restrictive
  b) Generate Solution Strategy:
     - For Schema Mismatch: Review schema and rewrite query
     - For Empty Result: Try CONTAINS instead of =, remove filters progressively
  c) Execute Strategy: Formulate the action that implements the chosen strategy

Available Actions:
- get_schema: Get database structure (already done if schema shown)
- execute_cypher: Run a Cypher query to get data
- save_as: Save a dataframe with a name
- compute_mismatch: Compute morphology-molecular mismatch
- final: ONLY use when you have found and analyzed the requested data

Output MUST be valid JSON with these keys:
{
  "thought": "Your current analysis",
  "reasoning_chain": ["Step 1 of reasoning", "Step 2", "etc"],
  "action": "action_name",
  "args": {"key": "value"}
}
""".strip()

CAR3_SPECIFIC_PROMPT = """
The user is asking about Car3 transcriptome data. Based on the schema:
1. Car3 is likely a subclass that regions can have
2. You need to find regions and their Car3 proportions/percentages
3. Identify the region with the highest Car3 proportion
4. Analyze that region's morphology and projections

Start by querying for Car3-related data in the knowledge graph.
""".strip()

REPORT_SYSTEM_PROMPT = """
You are an expert scientific writer. Generate a comprehensive analysis report based on the data gathered.

Structure your report with:
1. **Executive Summary**: Direct answer to the user's question (2-3 sentences)
2. **Key Findings**: Bullet points of main discoveries with specific data points
3. **Methodology**: Brief description of analysis approach
4. **Data Evidence**: Specific results that support conclusions
5. **Interpretation**: Scientific interpretation of the findings
6. **Limitations**: Any caveats or data gaps

Keep the report under 400 words but ensure it fully answers the original question with concrete evidence.
""".strip()

INTENT_PROMPT = """
Classify the user's goal into one of: [mismatch, car3_analysis, generic].
- mismatch: Analysis of morphology-molecular mismatches
- car3_analysis: Analysis related to Car3 transcriptome data
- generic: General knowledge graph exploration

Return only the classification label.
""".strip()


class LLM:
    def __init__(self, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = float(os.environ.get("OPENAI_TIMEOUT", "60"))
        self.step_count = 0  # Track steps to prevent premature finalization

    def decide_step(self, user_goal: str, ws_state: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a decision with anti-premature-finalization logic.
        """
        self.step_count = len(history)

        # Check if this is about Car3
        is_car3_query = "car3" in user_goal.lower()

        # Build enhanced prompt
        system_prompt = STEP_SYSTEM_PROMPT
        if is_car3_query:
            system_prompt += "\n\n" + CAR3_SPECIFIC_PROMPT

        # Build context
        context_parts = [f"User Goal:\n{user_goal}"]

        # Add guidance based on history
        if len(history) == 0:
            context_parts.append("\nThis is your FIRST action. Start by understanding what data is available.")
            if is_car3_query:
                context_parts.append("Look for Car3-related data in the graph.")
        elif len(history) < 3:
            context_parts.append("\nYou are in the EARLY stages. Focus on data discovery and querying.")
            context_parts.append("DO NOT finalize yet - you haven't gathered enough data.")

        # Check if we have actually found data
        has_data = any(
            h.get("result", {}).get("ok") and
            h.get("result", {}).get("rows", 0) > 0
            for h in history
            if h.get("action") == "execute_cypher"
        )

        if not has_data:
            context_parts.append("\nIMPORTANT: You haven't successfully retrieved any data yet. Keep querying!")

        context_parts.append(f"\nWorkspace State:\n{ws_state}")
        context_parts.append(f"\nHistory:\n{json.dumps(history, ensure_ascii=False, indent=2)}")

        # For Car3 queries, add specific guidance
        if is_car3_query and not has_data:
            context_parts.append("""

NEXT STEPS for Car3 Analysis:
1. If you haven't found Car3 data yet, try queries like:
   - MATCH (r:Region)-[:HAS_SUBCLASS]->(s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN r, s
   - MATCH (s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN s
   - MATCH (n) WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS 'Car3') RETURN n LIMIT 10
2. Once you find Car3 data, identify the region with highest proportion
3. Then query for that region's morphology and projections
""")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(context_parts)}
        ]

        txt = self._chat(messages)
        data = self._parse_json_robust(txt)

        # Prevent premature finalization
        if data.get("action") == "final" and self.step_count < 2:
            print("[LLM] Preventing premature finalization, redirecting to data query")
            data = {
                "thought": "Need to query for data before finalizing",
                "reasoning_chain": ["Too early to finalize", "Must first query for Car3 data"],
                "action": "execute_cypher",
                "args": {
                    "query": "MATCH (s:Subclass) WHERE s.name CONTAINS 'Car3' OR ANY(prop IN keys(s) WHERE toString(s[prop]) CONTAINS 'Car3') RETURN s LIMIT 20"
                }
            }

        return data

    def decide_step_with_reasoning(
            self,
            user_goal: str,
            ws_state: str,
            history: List[Dict[str, Any]],
            schema_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Enhanced decision making with schema awareness.
        """
        self.step_count = len(history)

        # Check for Car3 query
        is_car3_query = "car3" in user_goal.lower()

        # Format schema for inclusion
        schema_str = self._format_schema_context(schema_context) if schema_context else "No schema available"

        # Build system prompt
        system_prompt = STEP_SYSTEM_PROMPT
        if is_car3_query:
            system_prompt += "\n\n" + CAR3_SPECIFIC_PROMPT

        # Build context with schema
        context = f"""
DATABASE SCHEMA:
{schema_str}

User Goal:
{user_goal}

Progress Check:
- Steps taken: {len(history)}
- Data retrieved: {self._check_data_retrieved(history)}
- Current stage: {self._determine_stage(history)}

Workspace State:
{ws_state}

Previous Actions and Results:
{self._format_history_with_insights(history)}
"""

        # Add specific guidance for Car3 if no data yet
        if is_car3_query and not self._has_car3_data(history):
            context += """

IMPORTANT for Car3 Analysis:
You need to find Car3 transcriptome data first. Based on the schema, try:
1. Look for Subclass nodes with Car3 in the name or properties
2. Find the relationship between Region and Subclass
3. Query for proportions/percentages of Car3 per region
4. Identify the region with highest Car3 proportion
5. Then analyze that region's morphology and projections
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        txt = self._chat(messages)
        data = self._parse_json_robust(txt)

        # Ensure reasoning_chain exists
        if "reasoning_chain" not in data or not data["reasoning_chain"]:
            data["reasoning_chain"] = [data.get("thought", "Direct action")]

        # Prevent premature finalization
        if data.get("action") == "final" and not self._has_sufficient_data(history, user_goal):
            print("[LLM] Preventing premature finalization - insufficient data")
            return self._generate_next_query(user_goal, history, schema_context)

        return data

    def infer_intent(self, user_goal: str) -> str:
        """
        Basic intent inference.
        """
        goal_lower = user_goal.lower()
        if "mismatch" in goal_lower:
            return "mismatch"
        if "car3" in goal_lower:
            return "car3_analysis"
        return "generic"

    def infer_intent_with_schema(self, user_goal: str, schema_summary: Optional[Dict]) -> str:
        """
        Schema-aware intent inference.
        """
        messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user",
             "content": f"User Goal: {user_goal}\n\nSchema: {self._format_schema_context(schema_summary)}"}
        ]

        txt = self._chat(messages).strip().lower()

        if "mismatch" in txt:
            return "mismatch"
        if "car3" in txt:
            return "car3_analysis"
        return "generic"

    def final_report(self, user_goal: str, ws_state: str, history: List[Dict[str, Any]]) -> str:
        """
        Generate final report based on analysis history.
        """
        # Extract key findings from history
        key_findings = self._extract_key_findings(history)

        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
User Goal: {user_goal}

Workspace State:
{ws_state}

Key Findings from Analysis:
{json.dumps(key_findings, ensure_ascii=False, indent=2)}

Full Analysis History:
{json.dumps(history, ensure_ascii=False, indent=2)}

Generate a comprehensive report that directly answers the user's question based on the data gathered.
If insufficient data was found, explain what was attempted and what prevented finding the answer.
"""}
        ]

        txt = self._chat(messages)
        return txt.strip()

    def generate_comprehensive_report(
            self,
            user_goal: str,
            workspace_state: str,
            history: List[Dict[str, Any]],
            schema_summary: Optional[Dict]
    ) -> str:
        """
        Generate comprehensive final report with schema context.
        """
        # Extract key findings
        key_findings = self._extract_key_findings(history)
        schema_str = self._format_schema_context(schema_summary) if schema_summary else "No schema available"

        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
User Goal: {user_goal}

Database Schema:
{schema_str}

Workspace State:
{workspace_state}

Key Findings:
{json.dumps(key_findings, ensure_ascii=False, indent=2)}

Analysis Steps:
{self._summarize_analysis_steps(history)}

Generate a comprehensive report answering the user's question with concrete evidence from the data.
"""}
        ]

        txt = self._chat(messages)
        return txt.strip()

    # -------------------- Helper Methods --------------------

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat request to LLM API.
        """
        if not self.api_key:
            raise RuntimeError("No OPENAI_API_KEY provided. Set the environment variable or use TraceLLM for testing.")

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1  # Low temperature for consistency
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] API call failed: {e}")
            raise

    def _parse_json_robust(self, s: str) -> Dict[str, Any]:
        """
        Robust JSON parsing with multiple fallback strategies.
        """
        # First try: extract JSON block
        block = extract_json_block(s)
        if block:
            try:
                return json.loads(block)
            except:
                pass

        # Second try: clean and parse
        cleaned = s.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except:
            pass

        # Third try: find JSON-like structure
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, s, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if "action" in result:  # Valid action object
                    return result
            except:
                continue

        # Fourth try: extract from text patterns
        result = self._extract_from_text(s)
        if result:
            return result

        # Last resort: default to a safe action
        print(f"[LLM] Warning: Could not parse JSON from response, using default action")
        return {
            "thought": "Failed to parse LLM response, attempting to query for Car3 data",
            "reasoning_chain": ["JSON parsing failed", "Defaulting to data query"],
            "action": "execute_cypher",
            "args": {
                "query": "MATCH (n) WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS 'Car3') RETURN n LIMIT 10"
            }
        }

    def _extract_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract action components from unstructured text.
        """
        result = {}

        # Look for thought
        thought_patterns = [
            r'"thought"\s*:\s*"([^"]*)"',
            r'thought:\s*"([^"]*)"',
            r'Thought:\s*(.+?)(?:\n|$)'
        ]
        for pattern in thought_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["thought"] = match.group(1).strip()
                break

        # Look for action
        action_patterns = [
            r'"action"\s*:\s*"([^"]*)"',
            r'action:\s*"([^"]*)"',
            r'Action:\s*(\w+)'
        ]
        for pattern in action_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["action"] = match.group(1).strip()
                break

        # Look for query in args
        query_patterns = [
            r'"query"\s*:\s*"([^"]*)"',
            r'query:\s*"([^"]*)"',
            r'MATCH\s+.+?RETURN\s+.+?(?=\n|$)'
        ]
        for pattern in query_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if "args" not in result:
                    result["args"] = {}
                result["args"]["query"] = match.group(1) if pattern.startswith('"') else match.group(0)
                break

        # Default reasoning chain
        if "thought" in result:
            result["reasoning_chain"] = [result["thought"]]

        # Only return if we have at least an action
        if "action" in result:
            if "args" not in result:
                result["args"] = {}
            return result

        return None

    def _check_data_retrieved(self, history: List[Dict[str, Any]]) -> str:
        """
        Check what data has been successfully retrieved.
        """
        successful_queries = [
            h for h in history
            if h.get("action") == "execute_cypher"
               and h.get("result", {}).get("ok")
               and h.get("result", {}).get("rows", 0) > 0
        ]

        if not successful_queries:
            return "No data retrieved yet"

        total_rows = sum(h.get("result", {}).get("rows", 0) for h in successful_queries)
        return f"{len(successful_queries)} successful queries, {total_rows} total rows"

    def _determine_stage(self, history: List[Dict[str, Any]]) -> str:
        """
        Determine the current stage of analysis.
        """
        if len(history) == 0:
            return "Initial - no actions taken yet"
        elif len(history) < 3:
            return "Early exploration - gathering initial data"
        elif self._has_car3_data(history):
            return "Data found - ready for analysis"
        else:
            return "Searching - looking for relevant data"

    def _has_car3_data(self, history: List[Dict[str, Any]]) -> bool:
        """
        Check if Car3 data has been found.
        """
        for h in history:
            if h.get("action") == "execute_cypher" and h.get("result", {}).get("ok"):
                thought = h.get("thought", "").lower()
                query = h.get("args", {}).get("query", "").lower()
                if ("car3" in thought or "car3" in query) and h.get("result", {}).get("rows", 0) > 0:
                    return True
        return False

    def _has_sufficient_data(self, history: List[Dict[str, Any]], user_goal: str) -> bool:
        """
        Check if we have sufficient data to answer the user's question.
        """
        # Need at least one successful query with data
        successful_queries = [
            h for h in history
            if h.get("action") == "execute_cypher"
               and h.get("result", {}).get("ok")
               and h.get("result", {}).get("rows", 0) > 0
        ]

        if not successful_queries:
            return False

        # For Car3 queries, need specific checks
        if "car3" in user_goal.lower():
            # Need both Car3 data and morphology/projection data
            has_car3 = any("car3" in str(h).lower() for h in successful_queries)
            has_morphology = any(
                any(word in str(h).lower() for word in ["morphology", "volume", "surface", "density", "projection"])
                for h in successful_queries
            )
            return has_car3 and (has_morphology or len(successful_queries) >= 3)

        # For other queries, need at least 2 successful queries
        return len(successful_queries) >= 2

    def _generate_next_query(self, user_goal: str, history: List[Dict[str, Any]], schema_context: Optional[Dict]) -> \
    Dict[str, Any]:
        """
        Generate the next logical query based on current progress.
        """
        if "car3" in user_goal.lower():
            if not self._has_car3_data(history):
                # Try different approaches to find Car3
                queries_tried = [
                    h.get("args", {}).get("query", "")
                    for h in history
                    if h.get("action") == "execute_cypher"
                ]

                # Progressive query strategies
                if not any("Subclass" in q for q in queries_tried):
                    return {
                        "thought": "Looking for Car3 in Subclass nodes",
                        "reasoning_chain": [
                            "Need to find Car3 transcriptome data",
                            "Checking Subclass nodes for Car3"
                        ],
                        "action": "execute_cypher",
                        "args": {
                            "query": "MATCH (s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN s"
                        }
                    }
                elif not any("HAS_SUBCLASS" in q for q in queries_tried):
                    return {
                        "thought": "Looking for regions with Car3 subclass",
                        "reasoning_chain": [
                            "Need to find regions with Car3",
                            "Using HAS_SUBCLASS relationship"
                        ],
                        "action": "execute_cypher",
                        "args": {
                            "query": "MATCH (r:Region)-[:HAS_SUBCLASS]->(s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN r.name as region, s.name as subclass, s"
                        }
                    }
                else:
                    return {
                        "thought": "Broad search for anything containing Car3",
                        "reasoning_chain": [
                            "Previous targeted searches didn't find Car3",
                            "Trying broad search across all properties"
                        ],
                        "action": "execute_cypher",
                        "args": {
                            "query": "MATCH (n) WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS 'Car3') RETURN n, labels(n) as labels LIMIT 20"
                        }
                    }

        # Default fallback
        return {
            "thought": "Continuing data exploration",
            "reasoning_chain": ["Need more data to answer the question"],
            "action": "execute_cypher",
            "args": {
                "query": "MATCH (n) RETURN labels(n) as labels, count(*) as count ORDER BY count DESC LIMIT 10"
            }
        }

    def _format_schema_context(self, schema: Optional[Dict]) -> str:
        """
        Format schema information for prompt inclusion.
        """
        if not schema:
            return "No schema information available."

        lines = []

        if "labels" in schema and schema["labels"]:
            lines.append(f"Node Labels: {', '.join(str(l) for l in schema['labels'])}")

        if "relationships" in schema and schema["relationships"]:
            lines.append(f"Relationship Types: {', '.join(str(r) for r in schema['relationships'])}")

        if "sample_patterns" in schema and schema["sample_patterns"]:
            lines.append("\nSample Graph Patterns:")
            for pattern in schema["sample_patterns"][:5]:
                from_labels = pattern.get("from_labels", ["?"])
                relationship = pattern.get("relationship", "?")
                to_labels = pattern.get("to_labels", ["?"])
                if isinstance(from_labels, list):
                    from_str = ":".join(from_labels)
                else:
                    from_str = str(from_labels)
                if isinstance(to_labels, list):
                    to_str = ":".join(to_labels)
                else:
                    to_str = str(to_labels)
                lines.append(f"  ({from_str})-[:{relationship}]->({to_str})")

        return "\n".join(lines) if lines else "Schema structure unclear."

    def _format_history_with_insights(self, history: List[Dict[str, Any]]) -> str:
        """
        Format history with emphasis on insights and patterns.
        """
        if not history:
            return "No previous actions taken."

        formatted = []
        for entry in history[-5:]:  # Last 5 actions for context
            step = entry.get("step", "?")
            action = entry.get("action", "?")
            thought = entry.get("thought", "")
            result = entry.get("result", {})

            formatted.append(f"Step {step}: {action}")
            if thought:
                formatted.append(f"  Thought: {thought}")

            if action == "execute_cypher":
                query = entry.get("args", {}).get("query", "")
                if query:
                    formatted.append(f"  Query: {query[:100]}...")

            if result.get("ok"):
                if "rows" in result:
                    formatted.append(f"  Result: Success - {result['rows']} rows returned")
                    if result.get("column_info"):
                        cols = list(result["column_info"].keys())
                        formatted.append(f"  Columns: {cols}")
                else:
                    formatted.append(f"  Result: Success")
            else:
                formatted.append(f"  Result: Failed - {result.get('error', 'Unknown error')}")
                if result.get("message"):
                    formatted.append(f"  Error: {result['message'][:100]}")

        return "\n".join(formatted)

    def _extract_key_findings(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key findings from successful queries.
        """
        findings = []

        for entry in history:
            if entry.get("action") == "execute_cypher" and entry.get("result", {}).get("ok"):
                if entry.get("result", {}).get("rows", 0) > 0:
                    finding = {
                        "step": entry.get("step"),
                        "query_purpose": entry.get("thought", ""),
                        "rows_found": entry.get("result", {}).get("rows"),
                        "columns": list(
                            entry.get("result", {}).get("column_info", {}).keys()) if "column_info" in entry.get(
                            "result", {}) else []
                    }
                    findings.append(finding)

        return findings

    def _summarize_analysis_steps(self, history: List[Dict[str, Any]]) -> str:
        """
        Summarize the analysis steps taken.
        """
        steps = []
        for entry in history:
            if entry.get("action") == "execute_cypher":
                result = entry.get("result", {})
                if result.get("ok"):
                    steps.append(f"- {entry.get('thought', 'Query executed')}: {result.get('rows', 0)} rows found")
                else:
                    steps.append(f"- {entry.get('thought', 'Query attempted')}: Failed")
            elif entry.get("action") == "save_as":
                steps.append(f"- Saved data as: {entry.get('args', {}).get('name', 'unknown')}")
            elif entry.get("action") == "compute_mismatch":
                steps.append(f"- Computed mismatch analysis")

        return "\n".join(steps) if steps else "No analysis steps completed"


# Enhanced TraceLLM for testing
class TraceLLM(LLM):
    """
    Trace-based mock LLM for testing without API calls.
    """

    def __init__(self, steps: List[Dict[str, Any]]):
        super().__init__(model="trace")
        self._steps = steps
        self._i = 0

    @classmethod
    def from_file(cls, path: str) -> "TraceLLM":
        """
        Load trace steps from a JSON file.
        """
        import json
        with open(path, "r") as f:
            steps = json.load(f)
        return cls(steps)

    def decide_step(self, user_goal: str, ws_state: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Return pre-recorded step.
        """
        if self._i >= len(self._steps):
            return {
                "thought": "Trace exhausted - finalizing",
                "reasoning_chain": ["No more trace steps available"],
                "action": "final",
                "args": {"answer": "Analysis complete (trace mode)"}
            }

        out = self._steps[self._i]
        self._i += 1

        # Ensure required fields
        if "reasoning_chain" not in out:
            out["reasoning_chain"] = [out.get("thought", "Trace step")]
        if "args" not in out:
            out["args"] = {}

        return out

    def decide_step_with_reasoning(
            self,
            user_goal: str,
            ws_state: str,
            history: List[Dict[str, Any]],
            schema_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Return pre-recorded step with reasoning.
        """
        return self.decide_step(user_goal, ws_state, history)

    def infer_intent(self, user_goal: str) -> str:
        """
        Infer intent from goal text.
        """
        goal_lower = user_goal.lower()
        if "mismatch" in goal_lower:
            return "mismatch"
        if "car3" in goal_lower:
            return "car3_analysis"
        return "generic"

    def infer_intent_with_schema(self, user_goal: str, schema_summary: Optional[Dict]) -> str:
        """
        Infer intent with schema context.
        """
        return self.infer_intent(user_goal)

    def final_report(self, user_goal: str, ws_state: str, history: List[Dict[str, Any]]) -> str:
        """
        Generate trace-mode final report.
        """
        return f"[TRACE MODE] Completed {len(history)} steps.\nGoal: {user_goal}\nFinal workspace state:\n{ws_state[:500]}"

    def generate_comprehensive_report(
            self,
            user_goal: str,
            workspace_state: str,
            history: List[Dict[str, Any]],
            schema_summary: Optional[Dict]
    ) -> str:
        """
        Generate trace-mode comprehensive report.
        """
        successful_queries = sum(
            1 for h in history
            if h.get("action") == "execute_cypher"
            and h.get("result", {}).get("ok")
        )

        total_rows = sum(
            h.get("result", {}).get("rows", 0)
            for h in history
            if h.get("action") == "execute_cypher"
            and h.get("result", {}).get("ok")
        )

        return f"""[TRACE MODE REPORT]

Goal: {user_goal}

Analysis Summary:
- Total steps: {len(history)}
- Successful queries: {successful_queries}
- Total rows retrieved: {total_rows}

Final Workspace:
{workspace_state[:1000]}

Note: This is a trace-mode execution using pre-recorded steps.
"""

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Override _chat to prevent API calls in trace mode.
        """
        # Return a default JSON response for trace mode
        return json.dumps({
            "thought": "Trace mode - no API call",
            "reasoning_chain": ["Using trace steps"],
            "action": "execute_cypher",
            "args": {"query": "MATCH (n) RETURN n LIMIT 1"}
        })