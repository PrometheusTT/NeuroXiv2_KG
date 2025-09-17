"""
Enhanced reasoning agent that integrates CoT, KG-guided reasoning, self-evaluation,
and dynamic tool orchestration for powerful think-act-observe-reflect capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .enhanced_neo4j_exec import EnhancedNeo4jExec
from .schema_cache import SchemaCache
from .llm import LLMClient, ToolSpec
from .reasoning_engine import EnhancedCoTReasoningEngine, ReasoningState, Thought, Action, Observation, Reflection
from .kg_guided_reasoning import KGGuidedReasoningEngine, KGReasoning
from .self_evaluation import SelfEvaluationEngine, QualityMetrics, ReasoningError
from .dynamic_orchestration import DynamicToolOrchestrator, GoalType, ExecutionPlan
from .morphology_tools import RegionComparisonTools, MorphologicalAnalysisTools, MolecularProfileTools
from .enhanced_tools import EnhancedAnalysisTools, VisualizationTools
from .tools_stats import mismatch_index, basic_stats

logger = logging.getLogger(__name__)


class EnhancedReasoningAgent:
    """
    Advanced reasoning agent with integrated CoT, KG-guided reasoning, self-evaluation,
    and dynamic tool orchestration. Implements sophisticated think-act-observe-reflect loops.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str, database: str,
                 openai_api_key: str = None,
                 planner_model: str = "gpt-4", summarizer_model: str = "gpt-4o"):

        # Core infrastructure
        self.enhanced_db = EnhancedNeo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()
        with self.enhanced_db.driver.session(database=database) as s:
            self.schema.load_from_db(s)
        self.llm = LLMClient(api_key=openai_api_key, planner_model=planner_model, summarizer_model=summarizer_model)

        # Specialized analysis tools
        self.region_comparison = RegionComparisonTools(self.enhanced_db, self.schema)
        self.morph_tools = MorphologicalAnalysisTools(self.enhanced_db, self.schema)
        self.mol_tools = MolecularProfileTools(self.enhanced_db, self.schema)
        self.enhanced_tools = EnhancedAnalysisTools(self.enhanced_db, self.schema)
        self.viz_tools = VisualizationTools()

        # Define available tools
        self.tools = self._define_tool_registry()

        # Initialize reasoning components
        self.cot_reasoning = EnhancedCoTReasoningEngine(
            self.llm, self.enhanced_db, self.schema, self.tools
        )
        self.kg_reasoning = KGGuidedReasoningEngine(self.enhanced_db, self.schema)
        self.self_evaluation = SelfEvaluationEngine(self.llm)
        self.orchestrator = DynamicToolOrchestrator(self.tools)

        # Initialize KG knowledge
        self.kg_reasoning.initialize_graph_knowledge()

        # Connect tool router to reasoning engine
        self.cot_reasoning._tool_router = self._execute_tool

        logger.info("Enhanced reasoning agent initialized successfully")

    def reason_through_question(self, question: str, max_iterations: int = 12) -> Dict[str, Any]:
        """
        Main reasoning method that implements sophisticated think-act-observe-reflect loop
        with KG guidance, self-evaluation, and dynamic adaptation.
        """
        logger.info(f"🧠 Starting enhanced reasoning for: {question}")

        # Initialize reasoning state
        self.cot_reasoning.reset_state(question)
        reasoning_state = self.cot_reasoning.reasoning_state

        # Track reasoning progress
        reasoning_trace = {
            "question": question,
            "iterations": [],
            "kg_guidance_history": [],
            "corrections_applied": [],
            "quality_progression": []
        }

        for iteration in range(1, max_iterations + 1):
            logger.info(f"🔄 Reasoning iteration {iteration}/{max_iterations}")

            iteration_data = {
                "iteration": iteration,
                "thought": None,
                "kg_guidance": None,
                "action_plan": None,
                "observation": None,
                "reflection": None,
                "quality_assessment": None,
                "corrections": []
            }

            try:
                # STEP 1: KG-GUIDED THINKING
                kg_guidance = self.kg_reasoning.guide_reasoning_path(question, reasoning_state)
                iteration_data["kg_guidance"] = kg_guidance
                reasoning_trace["kg_guidance_history"].append(kg_guidance)

                context = self._build_reasoning_context(reasoning_state, kg_guidance)
                thought = self.cot_reasoning.think(context)
                iteration_data["thought"] = thought
                logger.info(f"💭 THOUGHT: {thought.content[:100]}...")

                # STEP 2: DYNAMIC ACTION PLANNING
                current_goal = self._infer_current_goal(question, reasoning_state, kg_guidance)
                orchestration_plan = self.orchestrator.plan_tool_sequence(
                    reasoning_state, kg_guidance, current_goal, question
                )
                iteration_data["action_plan"] = orchestration_plan

                # Execute planned action
                action = self._create_action_from_plan(orchestration_plan, thought)
                observation = self.cot_reasoning.observe(action)
                iteration_data["observation"] = observation
                logger.info(f"👁️  OBSERVATION: {observation.success}, {len(observation.insights)} insights")

                # STEP 3: ADAPTIVE ORCHESTRATION
                adapted_plan, adaptation_decision = self.orchestrator.adapt_execution(
                    orchestration_plan, observation, reasoning_state
                )

                if adaptation_decision["action_taken"] != "continue":
                    logger.info(f"🔄 ADAPTATION: {adaptation_decision['action_taken']} - {adaptation_decision['reason']}")

                # STEP 4: DEEP REFLECTION
                reflection = self.cot_reasoning.reflect(observation)
                iteration_data["reflection"] = reflection
                logger.info(f"🤔 REFLECTION: {reflection.content[:100]}...")

                # STEP 5: SELF-EVALUATION AND CORRECTION
                quality_metrics, errors = self.self_evaluation.evaluate_reasoning_state(reasoning_state, question)
                iteration_data["quality_assessment"] = {
                    "metrics": quality_metrics,
                    "errors": errors
                }
                reasoning_trace["quality_progression"].append(quality_metrics)

                logger.info(f"📊 QUALITY: {quality_metrics.overall_quality:.3f}, ERRORS: {len(errors)}")

                # Apply corrections if needed
                if errors:
                    corrections = self.self_evaluation.suggest_corrections(errors, reasoning_state)
                    applied_corrections = self._apply_corrections(corrections, reasoning_state)
                    iteration_data["corrections"] = applied_corrections
                    reasoning_trace["corrections_applied"].extend(applied_corrections)

                    if applied_corrections:
                        logger.info(f"🔧 Applied {len(applied_corrections)} corrections")

                # STEP 6: TERMINATION DECISION
                should_continue, termination_reason = self._should_continue_enhanced_reasoning(
                    reasoning_state, quality_metrics, iteration, max_iterations
                )

                reasoning_trace["iterations"].append(iteration_data)

                if not should_continue:
                    logger.info(f"🏁 Stopping reasoning: {termination_reason}")
                    break

            except Exception as e:
                logger.error(f"Error in reasoning iteration {iteration}: {e}")
                iteration_data["error"] = str(e)
                reasoning_trace["iterations"].append(iteration_data)
                break

        # FINAL SYNTHESIS
        final_answer = self._synthesize_enhanced_answer(reasoning_state, reasoning_trace, question)

        logger.info(f"✅ Enhanced reasoning completed after {len(reasoning_trace['iterations'])} iterations")

        return final_answer

    def _define_tool_registry(self) -> List[ToolSpec]:
        """Define comprehensive tool registry for the enhanced agent."""
        return [
            # Core tools
            ToolSpec(
                name="enhanced_neo4j_query",
                description="Execute complex Neo4j queries without syntax errors",
                parameters={"type": "object", "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "object"}
                }, "required": ["query"]}
            ),

            # Morphological analysis tools
            ToolSpec(
                name="find_morphologically_similar_regions",
                description="Find brain regions with similar morphological characteristics",
                parameters={"type": "object", "properties": {
                    "similarity_threshold": {"type": "number", "default": 0.1},
                    "limit": {"type": "integer", "default": 50}
                }}
            ),

            ToolSpec(
                name="find_morphologically_similar_molecularly_different",
                description="Find regions with similar morphology but different molecular profiles",
                parameters={"type": "object", "properties": {
                    "morphological_threshold": {"type": "number", "default": 0.1},
                    "molecular_threshold": {"type": "number", "default": 0.3},
                    "limit": {"type": "integer", "default": 20}
                }}
            ),

            # Molecular analysis tools
            ToolSpec(
                name="get_neurotransmitter_profiles",
                description="Get detailed neurotransmitter profiles for brain regions",
                parameters={"type": "object", "properties": {
                    "region_names": {"type": "array", "items": {"type": "string"}}
                }, "required": ["region_names"]}
            ),

            ToolSpec(
                name="compare_molecular_markers",
                description="Compare molecular marker profiles between brain regions",
                parameters={"type": "object", "properties": {
                    "region1": {"type": "string"},
                    "region2": {"type": "string"},
                    "top_n": {"type": "integer", "default": 3}
                }, "required": ["region1", "region2"]}
            ),

            ToolSpec(
                name="detailed_region_comparison",
                description="Comprehensive comparison between two specific regions",
                parameters={"type": "object", "properties": {
                    "region1": {"type": "string"},
                    "region2": {"type": "string"}
                }, "required": ["region1", "region2"]}
            ),

            # Network analysis tools
            ToolSpec(
                name="compute_graph_metrics",
                description="Compute comprehensive graph metrics",
                parameters={"type": "object", "properties": {
                    "node_type": {"type": "string", "default": "Region"},
                    "relationship_type": {"type": "string", "default": "PROJECT_TO"}
                }}
            ),

            ToolSpec(
                name="analyze_node_neighborhoods",
                description="Analyze neighborhood structure around specific nodes",
                parameters={"type": "object", "properties": {
                    "node_id": {"type": "string"},
                    "node_type": {"type": "string", "default": "Region"},
                    "max_depth": {"type": "integer", "default": 2}
                }, "required": ["node_id"]}
            ),

            # Statistical analysis tools
            ToolSpec(
                name="statistical_analysis",
                description="Perform comprehensive statistical analysis",
                parameters={"type": "object", "properties": {
                    "data_query": {"type": "string"},
                    "group_by_column": {"type": "string"}
                }, "required": ["data_query"]}
            ),

            ToolSpec(
                name="correlation_analysis",
                description="Compute correlation matrix for numeric features",
                parameters={"type": "object", "properties": {
                    "data_query": {"type": "string"}
                }, "required": ["data_query"]}
            )
        ]

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return results."""
        try:
            logger.debug(f"🔧 Executing tool: {tool_name}")

            # Route to appropriate tool implementation
            if tool_name == "enhanced_neo4j_query":
                return self.enhanced_db.run_direct(parameters["query"], parameters.get("params"))

            elif tool_name == "find_morphologically_similar_regions":
                return self.morph_tools.find_morphologically_similar_regions(
                    parameters.get("similarity_threshold", 0.1),
                    parameters.get("limit", 50)
                )

            elif tool_name == "find_morphologically_similar_molecularly_different":
                return self.region_comparison.find_morphologically_similar_molecularly_different_regions(
                    parameters.get("morphological_threshold", 0.1),
                    parameters.get("molecular_threshold", 0.3),
                    parameters.get("limit", 20)
                )

            elif tool_name == "get_neurotransmitter_profiles":
                return self.mol_tools.get_neurotransmitter_profiles(parameters["region_names"])

            elif tool_name == "compare_molecular_markers":
                return self.mol_tools.compare_molecular_markers(
                    parameters["region1"], parameters["region2"], parameters.get("top_n", 3)
                )

            elif tool_name == "detailed_region_comparison":
                return self.region_comparison.detailed_region_comparison(
                    parameters["region1"], parameters["region2"]
                )

            elif tool_name == "compute_graph_metrics":
                return self.enhanced_tools.compute_graph_metrics(
                    parameters.get("node_type", "Region"),
                    parameters.get("relationship_type", "PROJECT_TO")
                )

            elif tool_name == "analyze_node_neighborhoods":
                return self.enhanced_tools.analyze_node_neighborhoods(
                    parameters["node_id"],
                    parameters.get("node_type", "Region"),
                    parameters.get("max_depth", 2)
                )

            elif tool_name == "statistical_analysis":
                return self.enhanced_tools.statistical_analysis(
                    parameters["data_query"],
                    parameters.get("group_by_column")
                )

            elif tool_name == "correlation_analysis":
                return self.enhanced_tools.correlation_analysis(parameters["data_query"])

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {"error": str(e), "tool": tool_name}

    def _build_reasoning_context(self, reasoning_state: ReasoningState, kg_guidance: KGReasoning) -> str:
        """Build rich context for reasoning step."""
        context_parts = []

        # Add KG guidance
        if kg_guidance.central_entities:
            context_parts.append(f"Central entities to consider: {', '.join(kg_guidance.central_entities[:5])}")

        if kg_guidance.promising_paths:
            path_descriptions = [f"{p.start_node}->{p.end_node}" for p in kg_guidance.promising_paths[:3]]
            context_parts.append(f"Promising exploration paths: {', '.join(path_descriptions)}")

        if kg_guidance.information_gaps:
            context_parts.append(f"Information gaps: {', '.join(kg_guidance.information_gaps[:3])}")

        # Add reasoning state context
        if reasoning_state.working_hypotheses:
            context_parts.append(f"Current hypotheses: {', '.join(reasoning_state.working_hypotheses)}")

        if reasoning_state.confirmed_facts:
            context_parts.append(f"Confirmed facts: {', '.join(reasoning_state.confirmed_facts)}")

        # Add recent insights
        if reasoning_state.observations:
            recent_insights = []
            for obs in reasoning_state.observations[-2:]:
                recent_insights.extend(obs.insights)
            if recent_insights:
                context_parts.append(f"Recent insights: {', '.join(recent_insights[:3])}")

        return " | ".join(context_parts)

    def _infer_current_goal(self, question: str, reasoning_state: ReasoningState,
                           kg_guidance: KGReasoning) -> GoalType:
        """Infer the current reasoning goal based on context."""
        question_lower = question.lower()

        # Pattern-based goal inference
        if "similar" in question_lower and "different" in question_lower:
            return GoalType.FIND_SIMILARITIES if len(reasoning_state.observations) < 2 else GoalType.FIND_DIFFERENCES

        elif "similar" in question_lower:
            return GoalType.FIND_SIMILARITIES

        elif "different" in question_lower or "contrast" in question_lower:
            return GoalType.FIND_DIFFERENCES

        elif "relationship" in question_lower or "connect" in question_lower:
            return GoalType.EXPLORE_RELATIONSHIPS

        elif len(reasoning_state.observations) < 3:
            return GoalType.GATHER_EVIDENCE

        elif len(reasoning_state.working_hypotheses) > 0:
            return GoalType.VERIFY_HYPOTHESIS

        else:
            return GoalType.SYNTHESIZE_INFORMATION

    def _create_action_from_plan(self, plan: ExecutionPlan, thought: Thought) -> Action:
        """Create an action from the orchestration plan."""
        if not plan or not plan.primary_sequence or not plan.primary_sequence.tools:
            # Fallback action
            return Action(
                tool_name="enhanced_neo4j_query",
                parameters={"query": "MATCH (r:Region) RETURN r.name LIMIT 10"},
                rationale="Fallback action due to planning failure",
                expected_outcome="Basic region information"
            )

        tool_name = plan.primary_sequence.tools[0]
        tool_spec = next((tool for tool in self.tools if tool.name == tool_name), None)

        if not tool_spec:
            return Action(
                tool_name="enhanced_neo4j_query",
                parameters={"query": "MATCH (r:Region) RETURN r.name LIMIT 10"},
                rationale="Tool not found in registry",
                expected_outcome="Basic information"
            )

        # Generate appropriate parameters based on tool and context
        parameters = self._generate_tool_parameters(tool_name, thought)

        return Action(
            tool_name=tool_name,
            parameters=parameters,
            rationale=f"Selected based on orchestration plan for {plan.primary_sequence.goal.value}",
            expected_outcome="Relevant data for current reasoning step"
        )

    def _generate_tool_parameters(self, tool_name: str, thought: Thought) -> Dict[str, Any]:
        """Generate appropriate parameters for a tool based on thought content."""
        # Default parameters with intelligent adjustment based on thought content
        defaults = {
            "find_morphologically_similar_regions": {"similarity_threshold": 0.1, "limit": 20},
            "get_neurotransmitter_profiles": {"region_names": ["ACAv", "ORBvl", "IC", "LGv"]},
            "compare_molecular_markers": {"region1": "ACAv", "region2": "ORBvl", "top_n": 3},
            "enhanced_neo4j_query": {"query": "MATCH (r:Region) RETURN r.name, r.axonal_length, r.dendritic_length LIMIT 10"},
            "compute_graph_metrics": {"node_type": "Region", "relationship_type": "PROJECT_TO"}
        }

        base_params = defaults.get(tool_name, {})

        # Adjust parameters based on thought content
        thought_content = thought.content.lower()

        if "strict" in thought_content or "precise" in thought_content:
            if "similarity_threshold" in base_params:
                base_params["similarity_threshold"] = 0.05

        if "comprehensive" in thought_content or "detailed" in thought_content:
            if "limit" in base_params:
                base_params["limit"] = min(base_params["limit"] * 2, 100)

        return base_params

    def _apply_corrections(self, corrections: List, reasoning_state: ReasoningState) -> List[Dict[str, Any]]:
        """Apply reasoning corrections and return what was applied."""
        applied_corrections = []

        for correction in corrections[:3]:  # Apply top 3 corrections
            try:
                if correction.strategy.value == "reduce_confidence":
                    # Reduce overall confidence
                    adjustment = correction.new_parameters.get("confidence_adjustment", -0.1)
                    reasoning_state.confidence_score = max(0.1, reasoning_state.confidence_score + adjustment)
                    applied_corrections.append({
                        "type": "confidence_adjustment",
                        "adjustment": adjustment,
                        "new_confidence": reasoning_state.confidence_score
                    })

                elif correction.strategy.value == "gather_more_evidence":
                    # Flag need for more evidence
                    reasoning_state.knowledge_gaps.append("Need additional evidence before conclusions")
                    applied_corrections.append({
                        "type": "evidence_requirement",
                        "description": "Added requirement for more evidence"
                    })

                elif correction.strategy.value == "refocus_scope":
                    # Clear recent thoughts that may be off-topic
                    if len(reasoning_state.thoughts) > 2:
                        reasoning_state.thoughts = reasoning_state.thoughts[:-1]
                    applied_corrections.append({
                        "type": "scope_refocus",
                        "description": "Removed potentially off-topic recent thought"
                    })

            except Exception as e:
                logger.warning(f"Failed to apply correction {correction.strategy}: {e}")

        return applied_corrections

    def _should_continue_enhanced_reasoning(self, reasoning_state: ReasoningState,
                                          quality_metrics: QualityMetrics,
                                          iteration: int, max_iterations: int) -> Tuple[bool, str]:
        """Enhanced decision logic for continuing reasoning."""

        # Quality-based stopping
        if quality_metrics.overall_quality >= 0.85:
            return False, "High quality threshold achieved"

        # Evidence sufficiency
        if (len(reasoning_state.observations) >= 3 and
            len(reasoning_state.confirmed_facts) >= 2 and
            quality_metrics.evidence_strength > 0.7):
            return False, "Sufficient evidence gathered with good quality"

        # Maximum iterations
        if iteration >= max_iterations:
            return False, "Maximum iterations reached"

        # Stagnation detection
        if (iteration > 6 and
            len(reasoning_state.observations) >= 4 and
            quality_metrics.efficiency_score < 0.2):
            return False, "Reasoning appears to be stagnating"

        # Error accumulation
        if iteration > 4 and quality_metrics.logical_consistency < 0.3:
            return False, "Too many logical inconsistencies detected"

        # Continue reasoning
        return True, "Continue reasoning"

    def _synthesize_enhanced_answer(self, reasoning_state: ReasoningState,
                                   reasoning_trace: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Synthesize comprehensive final answer with full reasoning trace."""

        # Extract key insights from all observations
        all_insights = []
        for obs in reasoning_state.observations:
            all_insights.extend(obs.insights)

        # Get final quality assessment
        final_quality, final_errors = self.self_evaluation.evaluate_reasoning_state(reasoning_state, question)

        # Calibrate final confidence
        calibrated_confidence = self.self_evaluation.calibrate_confidence(reasoning_state, final_quality)

        # Generate comprehensive answer using LLM synthesis
        synthesis_prompt = f"""
        Question: {question}

        Complete Reasoning Process:
        - Total iterations: {len(reasoning_trace['iterations'])}
        - Key insights discovered: {all_insights[:10]}
        - Hypotheses explored: {reasoning_state.working_hypotheses}
        - Facts confirmed: {reasoning_state.confirmed_facts}

        Quality Assessment:
        - Overall quality: {final_quality.overall_quality:.3f}
        - Logical consistency: {final_quality.logical_consistency:.3f}
        - Evidence strength: {final_quality.evidence_strength:.3f}
        - Reasoning depth: {final_quality.reasoning_depth:.3f}

        Synthesize a comprehensive, well-structured answer that:
        1. Directly addresses the original question
        2. Integrates insights from all reasoning steps
        3. Provides clear evidence and methodology
        4. Acknowledges limitations and uncertainties
        5. Suggests follow-up investigations if appropriate
        """

        final_synthesis = self.llm.run_with_tools(
            system="You are an expert synthesizer. Create comprehensive, evidence-based answers from reasoning traces.",
            user=synthesis_prompt,
            tools=[],
            tool_router=lambda n, a: {}
        )

        return {
            "question": question,
            "answer": final_synthesis,
            "reasoning_summary": {
                "total_iterations": len(reasoning_trace['iterations']),
                "thoughts_generated": len(reasoning_state.thoughts),
                "observations_made": len(reasoning_state.observations),
                "reflections_completed": len(reasoning_state.reflections),
                "corrections_applied": len(reasoning_trace['corrections_applied']),
                "working_hypotheses": reasoning_state.working_hypotheses,
                "confirmed_facts": reasoning_state.confirmed_facts,
                "knowledge_gaps": reasoning_state.knowledge_gaps
            },
            "quality_assessment": {
                "overall_quality": final_quality.overall_quality,
                "logical_consistency": final_quality.logical_consistency,
                "evidence_strength": final_quality.evidence_strength,
                "reasoning_depth": final_quality.reasoning_depth,
                "coherence_score": final_quality.coherence_score,
                "calibrated_confidence": calibrated_confidence,
                "errors_detected": len(final_errors),
                "error_types": [error.error_type.value for error in final_errors]
            },
            "methodology": {
                "reasoning_approach": "think-act-observe-reflect with KG guidance",
                "tools_used": list(set(action.tool_name for action in reasoning_state.actions)),
                "kg_guidance_steps": len(reasoning_trace['kg_guidance_history']),
                "self_corrections": len(reasoning_trace['corrections_applied']),
                "adaptive_orchestration": "dynamic tool selection and sequencing"
            },
            "evidence_trail": {
                "observations": [
                    {
                        "step": i+1,
                        "success": obs.success,
                        "insights": obs.insights,
                        "surprises": obs.surprises
                    } for i, obs in enumerate(reasoning_state.observations)
                ],
                "key_insights": all_insights[:10],
                "supporting_evidence": reasoning_state.confirmed_facts
            },
            "reasoning_trace": reasoning_trace,
            "limitations": {
                "knowledge_gaps": reasoning_state.knowledge_gaps,
                "reasoning_errors": [error.description for error in final_errors],
                "confidence_level": calibrated_confidence,
                "scope_limitations": "Analysis limited to available knowledge graph data"
            }
        }

    def close(self):
        """Clean up resources."""
        try:
            self.enhanced_db.close()
        except:
            pass