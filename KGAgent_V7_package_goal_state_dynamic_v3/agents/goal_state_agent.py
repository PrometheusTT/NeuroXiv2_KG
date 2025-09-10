from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import pandas as pd
import numpy as np
import re
from .llm import LLM
from .workspace import Workspace
from .argument_graph import ArgumentGraph
from .guardrails import validate_action
from .json_utils import convert_numpy_types


class GoalDrivenAgent:
    """
    Enhanced goal-driven agent with improved reasoning coherence and robustness.
    """

    def __init__(self, llm: LLM, workspace: Workspace, argument_graph: ArgumentGraph):
        self.llm = llm
        self.ws = workspace
        self.ag = argument_graph
        self.step_count = 0
        self.max_retries = 3
        self.history = []
        self.schema_context = None

    def run(self, user_goal: str, max_steps: int = 12) -> Dict[str, Any]:
        """
        Run the agent with the given user goal up to max_steps.
        """
        self.step_count = 0
        self.history = []
        start_time = time.time()

        # Determine the user's intent
        intent = self.llm.infer_intent(user_goal)
        print(f"[Agent] Inferred intent: {intent}")

        # First action is always to get schema for context
        if self.schema_context is None:
            schema_action = {
                "step": 0,
                "thought": "Need to understand the database schema first",
                "reasoning_chain": ["First step is to get schema information"],
                "action": "get_schema",
                "args": {}
            }
            schema_result = self._apply_action(schema_action)
            # Don't count schema fetching toward step limit
            if schema_result.get("ok"):
                self.schema_context = schema_result.get("raw_schema")

                # Re-classify intent with schema context if available
                if self.schema_context:
                    intent = self.llm.infer_intent_with_schema(user_goal, self.schema_context)
                    print(f"[Agent] Refined intent with schema: {intent}")

        for step in range(1, max_steps + 1):
            self.step_count = step

            # Get workspace state
            ws_state = self.ws.preview()

            print(f"\n[Agent] Step {step}/{max_steps}: Deciding action...")

            # Get next action from LLM
            try:
                decision = self.llm.decide_step_with_reasoning(
                    user_goal=user_goal,
                    ws_state=ws_state,
                    history=self.history,
                    schema_context=self.schema_context
                )
            except Exception as e:
                print(f"[Agent] Error getting next step: {e}")
                decision = {
                    "thought": f"Error deciding next step: {str(e)}",
                    "reasoning_chain": [f"LLM error: {str(e)}", "Using fallback strategy"],
                    "action": "execute_cypher",
                    "args": {"query": "MATCH (n) RETURN labels(n) as labels, count(*) as count LIMIT 5"}
                }

            # Add step metadata
            decision["step"] = step

            # Validate action against intent
            valid, message = validate_action(intent, decision.get("action", ""))
            if not valid:
                print(f"[Agent] Warning: {message}")
                decision["thought"] = f"GUARDRAIL VIOLATED: {message}. Choosing alternative action."
                decision["action"] = "execute_cypher"  # Default to a safe action
                decision["args"] = {"query": "MATCH (n) RETURN labels(n) as labels, count(*) as count LIMIT 5"}

            print(f"[Agent] Action: {decision.get('action')}")
            print(f"[Agent] Thought: {decision.get('thought')}")

            # Check for finalization
            if decision.get("action") == "final":
                print(f"[Agent] Final answer after {step} steps")
                # Get final answer
                answer = decision.get("args", {}).get("answer")
                if not answer:
                    # Generate final report
                    answer = self.llm.generate_comprehensive_report(
                        user_goal=user_goal,
                        workspace_state=ws_state,
                        history=self.history,
                        schema_summary=self.schema_context
                    )
                    decision["args"] = {"answer": answer}

                # Add to history
                self.history.append(decision)

                return {
                    "final_answer": answer,
                    "steps_used": step,
                    "time_used": time.time() - start_time,
                    "history": self.history
                }

            # Apply action and get result
            try:
                result = self._apply_action(decision)
                decision["result"] = result
            except Exception as e:
                error_msg = f"Error executing action: {str(e)}"
                print(f"[Agent] {error_msg}")
                decision["result"] = {"ok": False, "error": "ExecutionError", "message": error_msg}

            # Add to history
            self.history.append(decision)

            # Special handling for successful Cypher queries that find Car3 data
            if (intent == "car3_analysis" and
                    decision.get("action") == "execute_cypher" and
                    decision.get("result", {}).get("ok") and
                    decision.get("result", {}).get("rows", 0) > 0):

                # Check if this query is finding Car3 regions
                query = decision.get("args", {}).get("query", "").lower()
                thought = decision.get("thought", "").lower()

                if ("car3" in query or "car3" in thought) and "proportion" in query.lower():
                    # Try to identify and save top Car3 region for future reference
                    try:
                        df = decision.get("result", {}).get("df")
                        if isinstance(df, pd.DataFrame) and "proportion" in df.columns and "region" in df.columns:
                            # Find region with highest Car3 proportion
                            top_region = df.loc[df["proportion"].idxmax()]
                            region_name = top_region["region"]
                            proportion = top_region["proportion"]

                            # Save as critical context
                            self.ws.add_important_result(
                                "top_car3_region",
                                region_name,
                                f"Region with highest Car3 proportion ({proportion})"
                            )

                            print(f"[Agent] Saved critical result: top_car3_region = {region_name}")
                    except Exception as e:
                        print(f"[Agent] Failed to extract top Car3 region: {e}")

        # If we get here, we've reached max steps without finalization
        print(f"[Agent] Reached max steps ({max_steps}) without finalization. Generating final answer.")

        # Generate final report
        final_answer = self.llm.generate_comprehensive_report(
            user_goal=user_goal,
            workspace_state=self.ws.preview(),
            history=self.history,
            schema_summary=self.schema_context
        )

        return {
            "final_answer": final_answer,
            "steps_used": max_steps,
            "time_used": time.time() - start_time,
            "history": self.history
        }

    def _apply_action(self, action_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the given action and return the result.
        Enhanced to handle data type issues and improve error reporting.
        """
        action_type = action_obj.get("action")
        args = action_obj.get("args", {})

        if action_type == "get_schema":
            # Try multiple import approaches to handle different package structures
            try:
                from .neo4j_exec import Neo4jExec
            except ImportError:
                try:
                    from agents.neo4j_exec import Neo4jExec
                except ImportError:
                    # Absolute import as last resort
                    from neo4j_exec import Neo4jExec

            executor = Neo4jExec()
            result = executor.get_schema()
            executor.close()
            return result

        elif action_type == "execute_cypher":
            # Try multiple import approaches to handle different package structures
            try:
                from .neo4j_exec import Neo4jExec
            except ImportError:
                try:
                    from agents.neo4j_exec import Neo4jExec
                except ImportError:
                    # Absolute import as last resort
                    from neo4j_exec import Neo4jExec

            query = args.get("query", "")
            params = args.get("params", {})

            # Add safety check to prevent returning entire nodes (causes unhashable type errors)
            if "RETURN" in query and re.search(r'\bRETURN\s+\w+\b(?!\.)(?!\s+as)', query, re.IGNORECASE):
                # Could be returning an entire node - check more carefully
                return_clause = re.search(r'\bRETURN\s+(.+?)(?:$|WHERE|LIMIT|ORDER)', query, re.IGNORECASE)
                if return_clause:
                    returns = return_clause.group(1).strip()
                    # If it's just a variable without a property accessor, it's returning a whole node
                    if re.search(r'\b\w+\b(?!\.)(?!\s+as)', returns):
                        print(f"[Agent] Warning: Query returns entire nodes, which may cause errors.")
                        print(f"[Agent] Original query: {query}")
                        # Try to fix the query by selecting node properties instead
                        try:
                            executor = Neo4jExec()
                            if executor.connected:
                                # Just get some basic query to return schema info as a fallback
                                query = "MATCH (n) RETURN labels(n) as labels, count(*) as count LIMIT 5"
                            executor.close()
                        except:
                            pass

            # Execute the query
            retry_count = 0
            executor = Neo4jExec()

            while retry_count < self.max_retries:
                try:
                    result = executor.run_cypher(query, params)

                    # Handle successful results
                    if result.get("ok") and "df" in result:
                        # Save results to workspace
                        key = self.ws.alloc_query_df_key()

                        # Get description from thought for better context
                        description = action_obj.get("thought", "Cypher query result")

                        # Convert any unhashable columns to strings to prevent errors
                        df = result["df"]
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                # Check for unhashable types (like dictionaries)
                                try:
                                    # This will fail if column has unhashable types
                                    df[col].nunique()
                                except TypeError:
                                    print(f"[Agent] Converting unhashable column {col} to strings")
                                    # Convert to string representation
                                    df[col] = df[col].astype(str)

                        # Store result
                        self.ws.put_df(key, df, description)
                        result["workspace_key"] = key

                        # Extract column info for better understanding
                        result["column_info"] = {col: str(df[col].dtype) for col in df.columns}

                    # If there was an error, add suggestions for next attempt
                    if not result.get("ok"):
                        if "raw_error" not in result and "message" in result:
                            result["raw_error"] = result["message"]

                        error_msg = result.get("message", "Unknown error")

                        # Add specific suggestions based on error type
                        if "no such property" in error_msg.lower():
                            result["suggestions"] = [
                                "Check the property name in the schema",
                                "Try a broader query without the specific property",
                                "Use CONTAINS for fuzzy property matching"
                            ]
                        elif "syntax error" in error_msg.lower():
                            result["suggestions"] = [
                                "Fix Cypher syntax",
                                "Simplify the query",
                                "Check for missing commas or parentheses"
                            ]

                    executor.close()
                    return result

                except Exception as e:
                    retry_count += 1
                    print(f"[Agent] Query execution error (attempt {retry_count}/{self.max_retries}): {e}")
                    if retry_count >= self.max_retries:
                        executor.close()
                        return {
                            "ok": False,
                            "error": "ExecutionError",
                            "message": str(e),
                            "suggestions": [
                                "Simplify the query",
                                "Check the schema",
                                "Try a different approach"
                            ]
                        }
                    # Brief pause before retry
                    time.sleep(1)

            executor.close()
            return {"ok": False, "error": "MaxRetriesExceeded", "message": "Failed after multiple attempts"}

        elif action_type == "save_as":
            # Get the dataframe
            source = args.get("source")
            name = args.get("name")

            if not source or not name:
                return {"ok": False, "error": "InvalidArguments", "message": "Both 'source' and 'name' are required"}

            try:
                df = self.ws.get_df(source)
                self.ws.put(name, df)
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": "SaveError", "message": str(e)}

        elif action_type == "compute_mismatch":
            from .tools_compute import top_mismatch_pairs  # Fixed import path

            morph_df_key = args.get("morph_df")
            mol_df_key = args.get("mol_df")
            topk = args.get("topk", 10)

            if not morph_df_key or not mol_df_key:
                return {"ok": False, "error": "InvalidArguments", "message": "Both morph_df and mol_df are required"}

            try:
                morph_df = self.ws.get_df(morph_df_key)
                mol_df = self.ws.get_df(mol_df_key)

                # Compute mismatches
                result_df = top_mismatch_pairs(morph_df, mol_df, topk)

                # Save result
                key = "df.mismatch.result"
                self.ws.put_df(key, result_df, "Mismatch computation result")

                return {
                    "ok": True,
                    "rows": len(result_df),
                    "df": result_df,
                    "workspace_key": key
                }
            except Exception as e:
                return {"ok": False, "error": "ComputeError", "message": str(e)}

        elif action_type == "final":
            # Just validate the answer exists
            answer = args.get("answer")
            if not answer:
                return {"ok": False, "error": "InvalidArguments", "message": "Final answer is required"}
            return {"ok": True}

        else:
            return {"ok": False, "error": "InvalidAction", "message": f"Unknown action type: {action_type}"}