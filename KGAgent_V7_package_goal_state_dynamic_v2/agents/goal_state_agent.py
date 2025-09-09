# Fixed goal_state_agent.py with better initialization and error handling

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
from .workspace import Workspace
from .argument_graph import ArgumentGraph
from .neo4j_exec import Neo4jExec
from .guardrails import validate_action
from .tools_compute import top_mismatch_pairs


class GoalDrivenAgent:
    """
    Enhanced Goal-driven agent with proper initialization and error handling.
    """

    def __init__(self, llm, workspace: Workspace, argument_graph: ArgumentGraph):
        self.llm = llm
        self.ws = workspace
        self.ag = argument_graph
        self.db = Neo4jExec()
        self.intent = None
        self.history: List[Dict[str, Any]] = []
        self.schema_loaded = False
        self.schema_summary = None

    def run(self, user_goal: str, max_steps: int = 12) -> Dict[str, Any]:
        """
        Main execution loop with mandatory schema pre-loading and proper error handling.
        """
        # CRITICAL: Schema-First Approach - Always load schema before any reasoning
        print("[Agent] Loading database schema...")
        self._preload_schema()

        # Infer intent with schema context
        try:
            if hasattr(self.llm, 'infer_intent_with_schema'):
                self.intent = self.llm.infer_intent_with_schema(user_goal, self.schema_summary)
            else:
                self.intent = self.llm.infer_intent(user_goal)
        except Exception as e:
            print(f"[Agent] Intent inference failed: {e}, using generic")
            self.intent = "generic"

        print(f"[Agent] Intent classified as: {self.intent}")
        print(f"[Agent] Starting analysis for: {user_goal}")

        # Main execution loop
        for step in range(1, max_steps + 1):
            print(f"\n[Agent] Step {step}/{max_steps}")

            # Include schema context in every decision
            ws_state = self._get_enriched_workspace_state()

            try:
                # Use the enhanced method if available, otherwise fall back
                if hasattr(self.llm, 'decide_step_with_reasoning'):
                    decision = self.llm.decide_step_with_reasoning(
                        user_goal=user_goal,
                        ws_state=ws_state,
                        history=self.history,
                        schema_context=self.schema_summary
                    )
                else:
                    decision = self.llm.decide_step(
                        user_goal=user_goal,
                        ws_state=ws_state,
                        history=self.history
                    )
            except Exception as e:
                print(f"[Agent] LLM decision error: {e}")
                self.history.append({
                    "step": step,
                    "thought": f"LLM error: {str(e)}",
                    "action": "final",
                    "args": {"answer": f"Error in decision making: {str(e)}"},
                    "error": str(e)
                })
                break

            # Validate decision structure
            if not isinstance(decision, dict):
                print(f"[Agent] Invalid decision format: {decision}")
                decision = {
                    "thought": "Invalid LLM response format",
                    "action": "final",
                    "args": {"answer": "Error: Invalid response format from LLM"}
                }

            # Extract decision components with defaults
            thought = decision.get("thought", "").strip()
            reasoning_chain = decision.get("reasoning_chain", [thought] if thought else ["No reasoning provided"])
            action = decision.get("action", "").strip()
            args = decision.get("args", {})

            # Ensure args is a dictionary
            if not isinstance(args, dict):
                args = {}

            print(f"[Agent] Thought: {thought}")
            print(f"[Agent] Action: {action}")

            # Handle empty action
            if not action:
                print("[Agent] Warning: Empty action received, attempting to continue...")
                action = "get_schema"  # Default safe action
                args = {}

            # Validate action against guardrails
            ok, reason = validate_action(self.intent, action)
            if not ok:
                print(f"[Agent] Action rejected by guardrails: {reason}")
                self.history.append({
                    "step": step,
                    "thought": thought,
                    "reasoning_chain": reasoning_chain,
                    "action": action,
                    "args": args,
                    "result": {
                        "ok": False,
                        "error": "GuardrailRejected",
                        "message": reason
                    }
                })
                continue

            # Execute action
            result = self._apply_action(action, args)

            # Enhanced result analysis for better self-correction
            result_analysis = self._analyze_result(result, action, args)

            print(f"[Agent] Result: {'Success' if result.get('ok') else 'Failed'}")
            if not result.get('ok'):
                print(f"[Agent] Error: {result.get('error')} - {result.get('message')}")

            self.history.append({
                "step": step,
                "thought": thought,
                "reasoning_chain": reasoning_chain,
                "action": action,
                "args": args,
                "result": self._summarize_result(result),
                "result_analysis": result_analysis
            })

            # Check for final action
            if action == "final":
                final_answer = args.get("answer", "")
                if not final_answer and len(self.history) <= 2:
                    # If finalizing too early without an answer, force continuation
                    print("[Agent] Warning: Attempting to finalize without proper analysis")
                    # Remove the premature final from history
                    self.history.pop()
                    continue

                return {
                    "steps_used": step,
                    "final_answer": final_answer,
                    "history": self.history,
                    "schema_summary": self.schema_summary
                }

        # Generate comprehensive final report
        print("\n[Agent] Generating final report...")
        try:
            if hasattr(self.llm, 'generate_comprehensive_report'):
                report = self.llm.generate_comprehensive_report(
                    user_goal,
                    self._get_enriched_workspace_state(),
                    self.history,
                    self.schema_summary
                )
            else:
                report = self.llm.final_report(
                    user_goal,
                    self._get_enriched_workspace_state(),
                    self.history
                )
        except Exception as e:
            report = f"Unable to synthesize final report: {e}"

        return {
            "steps_used": len(self.history),
            "final_answer": report,
            "history": self.history,
            "schema_summary": self.schema_summary
        }

    def _preload_schema(self):
        """
        Mandatory schema loading before any operations.
        """
        schema_result = self.db.get_schema()
        if schema_result.get("ok"):
            self.ws.put("schema.info", schema_result.get("schema_info"))

            # Get detailed schema information
            labels_result = self.db.run_cypher("CALL db.labels()")
            relationships_result = self.db.run_cypher("CALL db.relationshipTypes()")

            # Build comprehensive schema summary
            self.schema_summary = {
                "basic_info": schema_result.get("schema_info"),
                "labels": [],
                "relationships": [],
                "sample_patterns": [],
                "raw_schema": schema_result.get("raw_schema", {})
            }

            if labels_result.get("ok"):
                labels_df = labels_result.get("df", pd.DataFrame())
                if not labels_df.empty:
                    # Handle different possible column names
                    if 'label' in labels_df.columns:
                        self.schema_summary["labels"] = labels_df['label'].tolist()
                    elif labels_df.shape[1] > 0:
                        self.schema_summary["labels"] = labels_df.iloc[:, 0].tolist()

            if relationships_result.get("ok"):
                rels_df = relationships_result.get("df", pd.DataFrame())
                if not rels_df.empty:
                    # Handle different possible column names
                    if 'relationshipType' in rels_df.columns:
                        self.schema_summary["relationships"] = rels_df['relationshipType'].tolist()
                    elif rels_df.shape[1] > 0:
                        self.schema_summary["relationships"] = rels_df.iloc[:, 0].tolist()

            # Try to get sample patterns
            sample_query = """
            MATCH (n)-[r]->(m) 
            RETURN DISTINCT labels(n) as from_labels, 
                   type(r) as relationship, 
                   labels(m) as to_labels 
            LIMIT 10
            """
            sample_result = self.db.run_cypher(sample_query)
            if sample_result.get("ok") and len(sample_result.get("df", pd.DataFrame())) > 0:
                self.schema_summary["sample_patterns"] = sample_result["df"].to_dict('records')

            self.schema_loaded = True
            self.ws.put("schema.summary", self.schema_summary)

            print(f"[Agent] Schema loaded successfully:")
            print(f"  - Labels: {self.schema_summary.get('labels', [])}")
            print(f"  - Relationships: {self.schema_summary.get('relationships', [])}")
            print(f"  - Sample patterns: {len(self.schema_summary.get('sample_patterns', []))} found")
        else:
            self.schema_summary = {"error": "Failed to load schema", "details": schema_result}
            self.ws.put("schema.error", schema_result)
            print(f"[Agent] Warning: Failed to load schema: {schema_result.get('message')}")

    def _get_enriched_workspace_state(self) -> str:
        """
        Get workspace state with schema context prominently displayed.
        """
        base_state = self.ws.preview()

        schema_section = "\n=== DATABASE SCHEMA ===\n"
        if self.schema_loaded and self.schema_summary:
            schema_section += f"Labels: {self.schema_summary.get('labels', [])}\n"
            schema_section += f"Relationships: {self.schema_summary.get('relationships', [])}\n"
            if self.schema_summary.get('sample_patterns'):
                schema_section += "Sample Patterns:\n"
                for pattern in self.schema_summary['sample_patterns'][:5]:
                    schema_section += f"  {pattern}\n"
        else:
            schema_section += "Schema not loaded or unavailable\n"

        return schema_section + "\n=== WORKSPACE DATA ===\n" + base_state

    def _analyze_result(self, result: Dict[str, Any], action: str, args: Dict) -> Dict[str, Any]:
        """
        Analyze action results to provide better feedback for self-correction.
        """
        analysis = {
            "success": result.get("ok", False),
            "insights": []
        }

        if action == "execute_cypher" and not result.get("ok"):
            error_msg = result.get("message", "")

            # Detect schema mismatch errors
            if "UnknownLabelWarning" in error_msg or "UnknownLabel" in error_msg:
                analysis["insights"].append("Query uses non-existent label. Check schema.labels")
            if "UnknownPropertyKeyWarning" in error_msg or "UnknownProperty" in error_msg:
                analysis["insights"].append("Query uses non-existent property. Review schema patterns")
            if "SyntaxError" in error_msg:
                analysis["insights"].append("Cypher syntax error. Review query structure")

        elif action == "execute_cypher" and result.get("ok"):
            rows = result.get("rows", 0)
            if rows == 0:
                analysis["insights"].append("Query valid but returned no results. Consider:")
                analysis["insights"].append("- Relaxing filter conditions")
                analysis["insights"].append("- Checking if data exists for this criteria")
                analysis["insights"].append("- Using CONTAINS instead of exact match")
            else:
                analysis["insights"].append(f"Successfully retrieved {rows} rows")

        return analysis

    def _apply_action(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions with enhanced error handling and feedback.
        """
        if action == "get_schema":
            # Return cached schema if available
            if self.schema_loaded:
                return {
                    "ok": True,
                    "schema_info": self.schema_summary,
                    "message": "Schema already loaded"
                }
            return self.db.get_schema()

        if action == "execute_cypher":
            q = args.get("query", "")
            params = args.get("params", {})

            if not q:
                return {
                    "ok": False,
                    "error": "EmptyQuery",
                    "message": "No query provided"
                }

            print(f"[Agent] Executing Cypher: {q[:100]}...")

            # Execute query
            out = self.db.run_cypher(q, params)

            if out.get("ok"):
                df = out["df"]
                key = self.ws.alloc_query_df_key()
                self.ws.put_df(key, df)
                self.ws.put("last_df_key", key)
                out = dict(out)
                out["df_key"] = key

                # Add data profiling for better understanding
                if len(df) > 0:
                    column_info = {}
                    for col in df.columns:
                        try:
                            # 尝试计算 nunique
                            nunique_val = df[col].nunique()
                        except TypeError:
                            # 如果列内容不可哈希（如 dict），则标记为 N/A
                            nunique_val = "N/A (unhashable type)"

                        column_info[col] = {
                            "dtype": str(df[col].dtype),
                            "nunique": nunique_val,
                            "sample": df[col].iloc[0] if len(df) > 0 else None
                        }
                    out["column_info"] = column_info
                    print(f"[Agent] Query returned {len(df)} rows with columns: {list(df.columns)}")
                else:
                    print("[Agent] Query returned 0 rows")
            else:
                print(f"[Agent] Query failed: {out.get('message', 'Unknown error')}")

            return out

        if action == "save_as":
            src = args.get("df_key") or self.ws.get("last_df_key")
            dst = args.get("name")
            if not dst:
                return {"ok": False, "error": "BadArgs", "message": "Missing 'name' for save_as"}
            if not src:
                return {"ok": False, "error": "BadArgs", "message": "No dataframe to save"}
            try:
                df = self.ws.get_df(src)
                self.ws.put_df(dst, df)

                # Add to argument graph for evidence tracking
                self.ag.add_evidence(
                    claim=f"Data saved as {dst}",
                    evidence_content=f"DataFrame with {len(df)} rows and columns: {list(df.columns)}",
                    source_query=f"save_as from {src}",
                    strength="Strong"
                )

                print(f"[Agent] Saved dataframe as '{dst}' with {len(df)} rows")
                return {"ok": True, "saved": dst, "rows": len(df), "columns": list(df.columns)}
            except Exception as e:
                return {"ok": False, "error": "SaveError", "message": str(e)}

        if action == "compute_mismatch":
            morph_key = args.get("morph_df")
            mol_key = args.get("mol_df")
            if not morph_key or not mol_key:
                return {
                    "ok": False,
                    "error": "BadArgs",
                    "message": "compute_mismatch requires morph_df and mol_df keys"
                }

            try:
                # Get dataframes with proper index handling
                morph = self._prepare_dataframe_for_compute(morph_key)
                mol = self._prepare_dataframe_for_compute(mol_key)

                # Perform computation
                out_df = top_mismatch_pairs(morph, mol, topk=int(args.get("topk", 10)))

                # Store result
                key = "df.mismatch.pairs"
                self.ws.put_df(key, out_df)

                # Track in argument graph
                self.ag.add_claim("Mismatch analysis completed")
                self.ag.add_evidence(
                    claim="Mismatch analysis completed",
                    evidence_content=f"Top {len(out_df)} mismatch pairs identified",
                    source_query=f"compute_mismatch({morph_key}, {mol_key})",
                    strength="Strong",
                    reasoning_path="Computed using morphological and molecular distance metrics"
                )

                print(f"[Agent] Computed {len(out_df)} mismatch pairs")
                return {
                    "ok": True,
                    "df_key": key,
                    "rows": len(out_df),
                    "top_pair": out_df.iloc[0].to_dict() if len(out_df) > 0 else None
                }
            except Exception as e:
                return {"ok": False, "error": "ComputeError", "message": str(e)}

        if action == "final":
            # Track final answer in argument graph
            answer = args.get("answer", "")
            self.ag.add_claim("Final answer provided")
            self.ag.add_evidence(
                claim="Final answer provided",
                evidence_content=answer[:200] if answer else "(empty answer)",
                source_query="final_synthesis",
                strength="Strong",
                reasoning_path=f"Synthesized from {len(self.history)} steps of analysis"
            )
            return {"ok": True}

        return {"ok": False, "error": "UnknownAction", "message": f"Action '{action}' not recognized"}

    def _prepare_dataframe_for_compute(self, key: str) -> pd.DataFrame:
        """
        Prepare dataframe for computation with proper index handling.
        """
        df = self.ws.get_df(key)

        # Check if first column should be index
        if df.index.name is None and len(df.columns) > 0:
            # Assume first column is identifier if it looks like one
            first_col = df.columns[0]
            if df[first_col].dtype == 'object' and df[first_col].nunique() == len(df):
                df = df.set_index(first_col)

        return df

    def _summarize_result(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create concise result summary for history tracking.
        """
        if not isinstance(res, dict):
            return {"ok": False, "error": "BadResult"}

        if not res.get("ok"):
            return res

        # Create summary
        out = dict(res)

        # Summarize large data structures
        if "df" in out:
            out["df"] = f"<DataFrame rows={out.get('rows', '?')} cols={len(out['df'].columns) if 'df' in out else '?'}>"

        if "schema_info" in out and isinstance(out["schema_info"], dict):
            out["schema_info"] = "<Schema loaded successfully>"

        return out