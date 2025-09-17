"""
Dynamic tool orchestration system for adaptive reasoning.
Intelligently selects and sequences tools based on context, goals, and learned experience.
"""

import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import heapq

from .reasoning_engine import ReasoningState, Thought, Action, Observation
from .kg_guided_reasoning import KGReasoning

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools for intelligent orchestration."""
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    EXPLORATION = "exploration"
    COMPUTATION = "computation"
    VISUALIZATION = "visualization"
    EVALUATION = "evaluation"


class GoalType(Enum):
    """Types of reasoning goals."""
    FIND_SIMILARITIES = "find_similarities"
    FIND_DIFFERENCES = "find_differences"
    EXPLORE_RELATIONSHIPS = "explore_relationships"
    GATHER_EVIDENCE = "gather_evidence"
    VERIFY_HYPOTHESIS = "verify_hypothesis"
    SYNTHESIZE_INFORMATION = "synthesize_information"
    RESOLVE_CONTRADICTION = "resolve_contradiction"


@dataclass
class ToolCapability:
    """Describes a tool's capabilities and characteristics."""
    name: str
    category: ToolCategory
    input_types: Set[str] = field(default_factory=set)
    output_types: Set[str] = field(default_factory=set)
    cost: float = 1.0  # Computational/time cost
    reliability: float = 0.8
    complexity: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    typical_use_cases: List[GoalType] = field(default_factory=list)


@dataclass
class ToolSequence:
    """Represents a sequence of tools to achieve a goal."""
    sequence_id: str
    tools: List[str]
    goal: GoalType
    estimated_effectiveness: float = 0.0
    estimated_cost: float = 0.0
    success_probability: float = 0.0
    prerequisites_met: bool = True
    context_relevance: float = 0.0


@dataclass
class ExecutionPlan:
    """Complete execution plan for tool orchestration."""
    primary_sequence: ToolSequence
    alternative_sequences: List[ToolSequence] = field(default_factory=list)
    contingency_plans: Dict[str, ToolSequence] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)


class DynamicToolOrchestrator:
    """
    Dynamic tool orchestration system that adaptively selects and sequences tools
    based on reasoning context, goals, and learned experience.
    """

    def __init__(self, tool_registry):
        self.tools = tool_registry
        self.tool_capabilities = {}
        self.execution_history = []
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.context_tool_mapping = defaultdict(Counter)

        # Initialize tool capabilities
        self._initialize_tool_capabilities()

        # Learning weights for different factors
        self.learning_weights = {
            "historical_success": 0.3,
            "context_relevance": 0.25,
            "goal_alignment": 0.2,
            "resource_efficiency": 0.15,
            "risk_assessment": 0.1
        }

    def plan_tool_sequence(self, reasoning_state: ReasoningState, kg_reasoning: KGReasoning,
                          current_goal: GoalType, question: str) -> ExecutionPlan:
        """
        Plan optimal tool sequence based on current context and goals.
        """
        # Analyze current context
        context = self._analyze_current_context(reasoning_state, kg_reasoning, question)

        # Generate possible tool sequences
        candidate_sequences = self._generate_candidate_sequences(current_goal, context)

        # Evaluate and rank sequences
        evaluated_sequences = self._evaluate_sequences(candidate_sequences, context, reasoning_state)

        # Select primary and alternative sequences
        primary_sequence = evaluated_sequences[0] if evaluated_sequences else None
        alternative_sequences = evaluated_sequences[1:4]  # Top 3 alternatives

        # Generate contingency plans
        contingency_plans = self._generate_contingency_plans(primary_sequence, context)

        return ExecutionPlan(
            primary_sequence=primary_sequence,
            alternative_sequences=alternative_sequences,
            contingency_plans=contingency_plans,
            resource_requirements=self._estimate_resource_requirements(primary_sequence),
            success_metrics=self._define_success_metrics(current_goal)
        )

    def adapt_execution(self, plan: ExecutionPlan, observation: Observation,
                       reasoning_state: ReasoningState) -> Tuple[ExecutionPlan, Dict[str, Any]]:
        """
        Adapt execution plan based on observed results and current state.
        """
        adaptation_decision = {
            "action_taken": "continue",
            "reason": "",
            "modifications": []
        }

        # Analyze observation quality
        observation_quality = self._assess_observation_quality(observation)

        if not observation.success:
            # Tool failed - switch to alternative or contingency
            adaptation_decision["action_taken"] = "switch_to_alternative"
            adaptation_decision["reason"] = f"Tool failure: {observation.result.get('error', 'Unknown error')}"

            # Try alternative sequence
            if plan.alternative_sequences:
                new_primary = plan.alternative_sequences[0]
                plan.primary_sequence = new_primary
                plan.alternative_sequences = plan.alternative_sequences[1:]
                adaptation_decision["modifications"].append("Switched to alternative sequence")

        elif observation_quality < 0.3:
            # Poor quality results - consider different approach
            adaptation_decision["action_taken"] = "modify_approach"
            adaptation_decision["reason"] = "Low quality results"

            # Suggest more targeted tool
            better_tool = self._suggest_better_tool(plan.primary_sequence.tools[0], observation, reasoning_state)
            if better_tool:
                adaptation_decision["modifications"].append(f"Suggested better tool: {better_tool}")

        elif len(observation.surprises) > len(observation.insights):
            # Many surprises - might need different strategy
            adaptation_decision["action_taken"] = "reassess_strategy"
            adaptation_decision["reason"] = "Unexpected results requiring strategy reassessment"

        # Update learning from this execution step
        self._update_learning(plan, observation, reasoning_state)

        return plan, adaptation_decision

    def recommend_next_tool(self, reasoning_state: ReasoningState, kg_reasoning: KGReasoning,
                           current_goal: GoalType) -> Dict[str, Any]:
        """
        Recommend the next best tool based on current context.
        """
        context = self._analyze_current_context(reasoning_state, kg_reasoning, "")

        # Get tool recommendations based on different factors
        goal_based_tools = self._get_goal_aligned_tools(current_goal)
        context_based_tools = self._get_context_relevant_tools(context, reasoning_state)
        success_based_tools = self._get_historically_successful_tools(current_goal, context)

        # Combine recommendations with weights
        tool_scores = defaultdict(float)

        for tool in goal_based_tools:
            tool_scores[tool] += self.learning_weights["goal_alignment"]

        for tool, relevance in context_based_tools:
            tool_scores[tool] += relevance * self.learning_weights["context_relevance"]

        for tool, success_rate in success_based_tools:
            tool_scores[tool] += success_rate * self.learning_weights["historical_success"]

        # Apply risk and efficiency adjustments
        for tool, score in tool_scores.items():
            capability = self.tool_capabilities.get(tool)
            if capability:
                # Efficiency bonus
                efficiency_bonus = (1.0 / capability.cost) * self.learning_weights["resource_efficiency"]
                # Risk penalty
                risk_penalty = (1.0 - capability.reliability) * self.learning_weights["risk_assessment"]
                tool_scores[tool] += efficiency_bonus - risk_penalty

        # Get top recommendation
        if tool_scores:
            top_tool = max(tool_scores.items(), key=lambda x: x[1])
            tool_name, score = top_tool

            return {
                "recommended_tool": tool_name,
                "confidence": min(score, 1.0),
                "reasoning": self._explain_tool_recommendation(tool_name, score, context),
                "parameters": self._suggest_tool_parameters(tool_name, reasoning_state, kg_reasoning),
                "alternatives": [tool for tool, _ in sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)[1:4]]
            }

        return {"recommended_tool": None, "reason": "No suitable tool found"}

    def learn_from_execution(self, tool_name: str, parameters: Dict[str, Any],
                           observation: Observation, context: Dict[str, Any],
                           goal: GoalType):
        """
        Learn from tool execution to improve future recommendations.
        """
        execution_record = {
            "tool": tool_name,
            "parameters": parameters,
            "success": observation.success,
            "quality": self._assess_observation_quality(observation),
            "context": context,
            "goal": goal,
            "insights_generated": len(observation.insights),
            "surprises": len(observation.surprises)
        }

        self.execution_history.append(execution_record)

        # Update success/failure patterns
        if observation.success and self._assess_observation_quality(observation) > 0.5:
            self.success_patterns[goal].append({
                "tool": tool_name,
                "context_signature": self._create_context_signature(context),
                "effectiveness": self._assess_observation_quality(observation)
            })
        else:
            self.failure_patterns[goal].append({
                "tool": tool_name,
                "context_signature": self._create_context_signature(context),
                "failure_reason": observation.result.get("error", "Poor quality results")
            })

        # Update context-tool mapping
        context_signature = self._create_context_signature(context)
        if observation.success:
            self.context_tool_mapping[context_signature][tool_name] += 1

    def _initialize_tool_capabilities(self):
        """Initialize capabilities for all available tools."""

        # Define capabilities for each tool type
        tool_definitions = {
            "enhanced_neo4j_query": ToolCapability(
                name="enhanced_neo4j_query",
                category=ToolCategory.DATA_RETRIEVAL,
                input_types={"cypher_query", "parameters"},
                output_types={"graph_data", "structured_results"},
                cost=0.5,
                reliability=0.9,
                complexity=0.3,
                strengths=["flexible_querying", "direct_database_access"],
                limitations=["requires_cypher_knowledge"],
                typical_use_cases=[GoalType.GATHER_EVIDENCE, GoalType.EXPLORE_RELATIONSHIPS]
            ),

            "find_morphologically_similar_regions": ToolCapability(
                name="find_morphologically_similar_regions",
                category=ToolCategory.ANALYSIS,
                input_types={"similarity_threshold", "morphological_features"},
                output_types={"similarity_pairs", "morphological_data"},
                cost=1.2,
                reliability=0.85,
                complexity=0.7,
                strengths=["specialized_morphology_analysis", "quantitative_similarity"],
                limitations=["limited_to_morphological_data"],
                typical_use_cases=[GoalType.FIND_SIMILARITIES, GoalType.GATHER_EVIDENCE]
            ),

            "get_neurotransmitter_profiles": ToolCapability(
                name="get_neurotransmitter_profiles",
                category=ToolCategory.DATA_RETRIEVAL,
                input_types={"region_names"},
                output_types={"neurotransmitter_data", "molecular_profiles"},
                cost=0.8,
                reliability=0.9,
                complexity=0.4,
                strengths=["molecular_data_retrieval", "neurotransmitter_focus"],
                limitations=["limited_to_neurotransmitter_data"],
                typical_use_cases=[GoalType.GATHER_EVIDENCE, GoalType.FIND_DIFFERENCES]
            ),

            "compare_molecular_markers": ToolCapability(
                name="compare_molecular_markers",
                category=ToolCategory.COMPARISON,
                input_types={"region_pairs", "marker_types"},
                output_types={"comparison_results", "similarity_metrics"},
                cost=1.0,
                reliability=0.8,
                complexity=0.6,
                strengths=["molecular_comparison", "quantitative_metrics"],
                limitations=["requires_region_pairs"],
                typical_use_cases=[GoalType.FIND_DIFFERENCES, GoalType.VERIFY_HYPOTHESIS]
            ),

            "compute_graph_metrics": ToolCapability(
                name="compute_graph_metrics",
                category=ToolCategory.COMPUTATION,
                input_types={"node_type", "relationship_type"},
                output_types={"network_metrics", "centrality_measures"},
                cost=1.5,
                reliability=0.75,
                complexity=0.8,
                strengths=["network_analysis", "centrality_computation"],
                limitations=["computationally_expensive"],
                typical_use_cases=[GoalType.EXPLORE_RELATIONSHIPS, GoalType.GATHER_EVIDENCE]
            ),

            "statistical_analysis": ToolCapability(
                name="statistical_analysis",
                category=ToolCategory.ANALYSIS,
                input_types={"data_query", "statistical_parameters"},
                output_types={"statistical_results", "distributions"},
                cost=0.7,
                reliability=0.85,
                complexity=0.5,
                strengths=["statistical_rigor", "data_summarization"],
                limitations=["requires_numerical_data"],
                typical_use_cases=[GoalType.VERIFY_HYPOTHESIS, GoalType.GATHER_EVIDENCE]
            )
        }

        self.tool_capabilities = tool_definitions

    def _analyze_current_context(self, reasoning_state: ReasoningState,
                                kg_reasoning: KGReasoning, question: str) -> Dict[str, Any]:
        """Analyze current reasoning context."""
        return {
            "reasoning_step": reasoning_state.current_step,
            "evidence_level": len(reasoning_state.observations),
            "hypothesis_count": len(reasoning_state.working_hypotheses),
            "knowledge_gaps": reasoning_state.knowledge_gaps,
            "central_entities": kg_reasoning.central_entities if kg_reasoning else [],
            "question_type": self._classify_question_type(question),
            "complexity_level": self._assess_question_complexity(question),
            "domain_focus": self._identify_domain_focus(question)
        }

    def _generate_candidate_sequences(self, goal: GoalType, context: Dict[str, Any]) -> List[ToolSequence]:
        """Generate candidate tool sequences for the given goal."""
        sequences = []

        # Goal-based sequence templates
        sequence_templates = {
            GoalType.FIND_SIMILARITIES: [
                ["find_morphologically_similar_regions", "statistical_analysis"],
                ["enhanced_neo4j_query", "find_morphologically_similar_regions"],
                ["compute_graph_metrics", "statistical_analysis"]
            ],

            GoalType.FIND_DIFFERENCES: [
                ["get_neurotransmitter_profiles", "compare_molecular_markers"],
                ["enhanced_neo4j_query", "compare_molecular_markers"],
                ["find_morphologically_similar_regions", "compare_molecular_markers"]
            ],

            GoalType.EXPLORE_RELATIONSHIPS: [
                ["compute_graph_metrics", "enhanced_neo4j_query"],
                ["enhanced_neo4j_query", "statistical_analysis"],
                ["analyze_node_neighborhoods", "compute_graph_metrics"]
            ],

            GoalType.GATHER_EVIDENCE: [
                ["enhanced_neo4j_query"],
                ["get_neurotransmitter_profiles"],
                ["find_morphologically_similar_regions"]
            ]
        }

        templates = sequence_templates.get(goal, [["enhanced_neo4j_query"]])

        for i, template in enumerate(templates):
            sequence = ToolSequence(
                sequence_id=f"{goal.value}_{i}",
                tools=template,
                goal=goal,
                context_relevance=self._assess_sequence_context_relevance(template, context)
            )
            sequences.append(sequence)

        return sequences

    def _evaluate_sequences(self, sequences: List[ToolSequence], context: Dict[str, Any],
                           reasoning_state: ReasoningState) -> List[ToolSequence]:
        """Evaluate and rank tool sequences."""

        for sequence in sequences:
            # Estimate effectiveness based on historical success
            effectiveness = self._estimate_sequence_effectiveness(sequence, context)

            # Estimate cost
            cost = sum(self.tool_capabilities[tool].cost for tool in sequence.tools
                      if tool in self.tool_capabilities)

            # Estimate success probability
            success_prob = self._estimate_success_probability(sequence, context, reasoning_state)

            sequence.estimated_effectiveness = effectiveness
            sequence.estimated_cost = cost
            sequence.success_probability = success_prob

        # Sort by composite score
        def composite_score(seq):
            return (seq.estimated_effectiveness * seq.success_probability) / (seq.estimated_cost + 0.1)

        sequences.sort(key=composite_score, reverse=True)
        return sequences

    def _generate_contingency_plans(self, primary_sequence: ToolSequence,
                                  context: Dict[str, Any]) -> Dict[str, ToolSequence]:
        """Generate contingency plans for potential failures."""
        contingencies = {}

        if not primary_sequence:
            return contingencies

        # Fallback for tool failures
        for i, tool in enumerate(primary_sequence.tools):
            fallback_tools = self._get_alternative_tools(tool, context)
            if fallback_tools:
                contingency_sequence = ToolSequence(
                    sequence_id=f"contingency_{tool}",
                    tools=[fallback_tools[0]] + primary_sequence.tools[i+1:],
                    goal=primary_sequence.goal
                )
                contingencies[f"tool_failure_{tool}"] = contingency_sequence

        return contingencies

    def _assess_observation_quality(self, observation: Observation) -> float:
        """Assess quality of observation results."""
        if not observation.success:
            return 0.1

        quality_score = 0.5  # Base score for success

        # Bonus for insights
        insight_bonus = min(len(observation.insights) * 0.15, 0.3)
        quality_score += insight_bonus

        # Penalty for surprises (unexpected results might indicate issues)
        surprise_penalty = min(len(observation.surprises) * 0.1, 0.2)
        quality_score -= surprise_penalty

        # Bonus for data richness
        if "data" in observation.result:
            data = observation.result["data"]
            if isinstance(data, list) and len(data) > 0:
                data_bonus = min(len(data) / 50.0, 0.2)
                quality_score += data_bonus

        return max(0.1, min(1.0, quality_score))

    def _suggest_better_tool(self, current_tool: str, observation: Observation,
                           reasoning_state: ReasoningState) -> Optional[str]:
        """Suggest a better tool based on poor results."""

        if current_tool not in self.tool_capabilities:
            return None

        current_capability = self.tool_capabilities[current_tool]
        same_category_tools = [
            name for name, cap in self.tool_capabilities.items()
            if cap.category == current_capability.category and name != current_tool
        ]

        # Choose tool with higher reliability in the same category
        if same_category_tools:
            best_alternative = max(
                same_category_tools,
                key=lambda t: self.tool_capabilities[t].reliability
            )
            return best_alternative

        return None

    def _update_learning(self, plan: ExecutionPlan, observation: Observation, reasoning_state: ReasoningState):
        """Update learning based on execution results."""
        if not plan.primary_sequence:
            return

        tool_used = plan.primary_sequence.tools[0] if plan.primary_sequence.tools else None
        if not tool_used:
            return

        # Record execution outcome
        outcome = {
            "tool": tool_used,
            "success": observation.success,
            "quality": self._assess_observation_quality(observation),
            "reasoning_step": reasoning_state.current_step,
            "goal": plan.primary_sequence.goal
        }

        self.execution_history.append(outcome)

        # Adjust tool reliability based on performance
        if tool_used in self.tool_capabilities:
            current_reliability = self.tool_capabilities[tool_used].reliability
            learning_rate = 0.05

            if observation.success and self._assess_observation_quality(observation) > 0.6:
                # Successful execution - increase reliability slightly
                new_reliability = min(0.95, current_reliability + learning_rate)
            else:
                # Poor execution - decrease reliability slightly
                new_reliability = max(0.3, current_reliability - learning_rate)

            self.tool_capabilities[tool_used].reliability = new_reliability

    def _get_goal_aligned_tools(self, goal: GoalType) -> List[str]:
        """Get tools aligned with specific goal."""
        aligned_tools = []
        for tool_name, capability in self.tool_capabilities.items():
            if goal in capability.typical_use_cases:
                aligned_tools.append(tool_name)
        return aligned_tools

    def _get_context_relevant_tools(self, context: Dict[str, Any],
                                   reasoning_state: ReasoningState) -> List[Tuple[str, float]]:
        """Get tools relevant to current context with relevance scores."""
        relevant_tools = []

        domain_focus = context.get("domain_focus", "general")
        question_type = context.get("question_type", "general")

        for tool_name, capability in self.tool_capabilities.items():
            relevance = 0.5  # Base relevance

            # Domain alignment
            if "morphological" in domain_focus and "morphological" in capability.name:
                relevance += 0.3
            if "molecular" in domain_focus and "molecular" in capability.name:
                relevance += 0.3
            if "network" in domain_focus and capability.category == ToolCategory.COMPUTATION:
                relevance += 0.2

            # Question type alignment
            if question_type == "comparison" and capability.category == ToolCategory.COMPARISON:
                relevance += 0.2
            if question_type == "analysis" and capability.category == ToolCategory.ANALYSIS:
                relevance += 0.2

            relevant_tools.append((tool_name, relevance))

        return sorted(relevant_tools, key=lambda x: x[1], reverse=True)

    def _get_historically_successful_tools(self, goal: GoalType,
                                         context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get historically successful tools for this goal and context."""
        successful_tools = defaultdict(list)

        context_signature = self._create_context_signature(context)

        # Analyze success patterns
        for success_record in self.success_patterns[goal]:
            if success_record["context_signature"] == context_signature:
                successful_tools[success_record["tool"]].append(success_record["effectiveness"])

        # Calculate success rates
        tool_success_rates = []
        for tool, effectiveness_scores in successful_tools.items():
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            tool_success_rates.append((tool, avg_effectiveness))

        return sorted(tool_success_rates, key=lambda x: x[1], reverse=True)

    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for the context for pattern matching."""
        key_elements = [
            context.get("question_type", "general"),
            context.get("domain_focus", "general"),
            context.get("complexity_level", "medium")
        ]
        return "_".join(key_elements)

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked."""
        question_lower = question.lower()

        if "similar" in question_lower and "different" in question_lower:
            return "comparison"
        elif "similar" in question_lower or "alike" in question_lower:
            return "similarity"
        elif "different" in question_lower or "distinct" in question_lower:
            return "difference"
        elif "relationship" in question_lower or "connect" in question_lower:
            return "relationship"
        elif "analyze" in question_lower or "examine" in question_lower:
            return "analysis"
        else:
            return "general"

    def _assess_question_complexity(self, question: str) -> str:
        """Assess complexity level of the question."""
        complexity_indicators = [
            "comprehensive", "detailed", "complex", "multiple", "various",
            "significant", "analyze", "compare", "relationship", "pattern"
        ]

        question_lower = question.lower()
        complexity_count = sum(1 for indicator in complexity_indicators
                             if indicator in question_lower)

        if complexity_count >= 3:
            return "high"
        elif complexity_count >= 1:
            return "medium"
        else:
            return "low"

    def _identify_domain_focus(self, question: str) -> str:
        """Identify the domain focus of the question."""
        question_lower = question.lower()

        if "morphological" in question_lower or "structural" in question_lower:
            return "morphological"
        elif "molecular" in question_lower or "neurotransmitter" in question_lower:
            return "molecular"
        elif "network" in question_lower or "connectivity" in question_lower:
            return "network"
        else:
            return "general"

    def _assess_sequence_context_relevance(self, tools: List[str], context: Dict[str, Any]) -> float:
        """Assess how relevant a tool sequence is to the current context."""
        if not tools:
            return 0.0

        relevance_scores = []
        for tool in tools:
            if tool in self.tool_capabilities:
                capability = self.tool_capabilities[tool]
                # Simple relevance based on category matching
                base_relevance = 0.5

                domain_focus = context.get("domain_focus", "general")
                if domain_focus in capability.strengths:
                    base_relevance += 0.3

                relevance_scores.append(base_relevance)
            else:
                relevance_scores.append(0.2)

        return sum(relevance_scores) / len(relevance_scores)

    def _estimate_sequence_effectiveness(self, sequence: ToolSequence, context: Dict[str, Any]) -> float:
        """Estimate effectiveness of a tool sequence."""
        if not sequence.tools:
            return 0.1

        effectiveness_factors = []

        # Historical success rate
        goal = sequence.goal
        context_signature = self._create_context_signature(context)

        for tool in sequence.tools:
            # Base effectiveness from tool reliability
            base_effectiveness = self.tool_capabilities.get(tool, type('', (), {"reliability": 0.5})()).reliability

            # Adjust based on historical success for this context and goal
            historical_success = 0.5  # Default
            for record in self.success_patterns[goal]:
                if record["tool"] == tool and record["context_signature"] == context_signature:
                    historical_success = max(historical_success, record["effectiveness"])

            combined_effectiveness = (base_effectiveness + historical_success) / 2
            effectiveness_factors.append(combined_effectiveness)

        return sum(effectiveness_factors) / len(effectiveness_factors)

    def _estimate_success_probability(self, sequence: ToolSequence, context: Dict[str, Any],
                                    reasoning_state: ReasoningState) -> float:
        """Estimate probability of sequence success."""
        base_probability = 0.7

        # Adjust based on prerequisites
        if not sequence.prerequisites_met:
            base_probability *= 0.5

        # Adjust based on reasoning state
        if reasoning_state.current_step > 6:  # Late in reasoning
            base_probability *= 0.8

        # Adjust based on tool reliability
        if sequence.tools:
            avg_reliability = sum(
                self.tool_capabilities.get(tool, type('', (), {"reliability": 0.5})()).reliability
                for tool in sequence.tools
            ) / len(sequence.tools)
            base_probability = (base_probability + avg_reliability) / 2

        return min(0.95, max(0.1, base_probability))

    def _estimate_resource_requirements(self, sequence: ToolSequence) -> Dict[str, Any]:
        """Estimate resource requirements for sequence execution."""
        if not sequence or not sequence.tools:
            return {}

        total_cost = sum(
            self.tool_capabilities.get(tool, type('', (), {"cost": 1.0})()).cost
            for tool in sequence.tools
        )

        max_complexity = max(
            self.tool_capabilities.get(tool, type('', (), {"complexity": 0.5})()).complexity
            for tool in sequence.tools
        )

        return {
            "estimated_time_units": total_cost,
            "complexity_level": max_complexity,
            "parallel_execution_possible": len(sequence.tools) > 1,
            "memory_requirements": "medium" if max_complexity > 0.7 else "low"
        }

    def _define_success_metrics(self, goal: GoalType) -> List[str]:
        """Define success metrics for different goal types."""
        metrics_map = {
            GoalType.FIND_SIMILARITIES: ["similarity_pairs_found", "similarity_confidence", "evidence_quality"],
            GoalType.FIND_DIFFERENCES: ["differences_identified", "difference_significance", "statistical_support"],
            GoalType.EXPLORE_RELATIONSHIPS: ["relationships_discovered", "network_insights", "connectivity_patterns"],
            GoalType.GATHER_EVIDENCE: ["evidence_quantity", "evidence_quality", "insight_generation"],
            GoalType.VERIFY_HYPOTHESIS: ["hypothesis_support", "statistical_significance", "alternative_explanations"],
            GoalType.SYNTHESIZE_INFORMATION: ["synthesis_completeness", "coherence_score", "insight_integration"]
        }

        return metrics_map.get(goal, ["general_effectiveness", "result_quality"])

    def _explain_tool_recommendation(self, tool_name: str, score: float, context: Dict[str, Any]) -> str:
        """Explain why a particular tool was recommended."""
        if tool_name not in self.tool_capabilities:
            return "Tool recommendation based on general suitability"

        capability = self.tool_capabilities[tool_name]

        explanation_parts = [
            f"Tool {tool_name} recommended with confidence {score:.2f}"
        ]

        # Add strength-based reasoning
        if capability.strengths:
            explanation_parts.append(f"Strengths: {', '.join(capability.strengths[:2])}")

        # Add context alignment
        domain_focus = context.get("domain_focus", "general")
        if domain_focus != "general":
            explanation_parts.append(f"Aligns with {domain_focus} domain focus")

        # Add reliability info
        explanation_parts.append(f"Reliability: {capability.reliability:.2f}")

        return ". ".join(explanation_parts)

    def _suggest_tool_parameters(self, tool_name: str, reasoning_state: ReasoningState,
                                kg_reasoning: KGReasoning) -> Dict[str, Any]:
        """Suggest appropriate parameters for the recommended tool."""
        # Default parameters based on tool type and context
        parameter_suggestions = {
            "find_morphologically_similar_regions": {
                "similarity_threshold": 0.1,
                "limit": 20
            },
            "get_neurotransmitter_profiles": {
                "region_names": kg_reasoning.central_entities[:5] if kg_reasoning else ["unknown"]
            },
            "compare_molecular_markers": {
                "top_n": 3
            },
            "enhanced_neo4j_query": {
                "query": "MATCH (n) RETURN n LIMIT 10"
            },
            "compute_graph_metrics": {
                "node_type": "Region",
                "relationship_type": "PROJECT_TO"
            }
        }

        return parameter_suggestions.get(tool_name, {})

    def _get_alternative_tools(self, tool: str, context: Dict[str, Any]) -> List[str]:
        """Get alternative tools that can serve similar purpose."""
        if tool not in self.tool_capabilities:
            return []

        original_capability = self.tool_capabilities[tool]
        alternatives = []

        for other_tool, other_capability in self.tool_capabilities.items():
            if (other_tool != tool and
                other_capability.category == original_capability.category and
                len(other_capability.output_types & original_capability.output_types) > 0):
                alternatives.append(other_tool)

        # Sort by reliability
        alternatives.sort(key=lambda t: self.tool_capabilities[t].reliability, reverse=True)
        return alternatives[:3]