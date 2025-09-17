"""
Knowledge Graph guided reasoning system.
Uses KG structure, schema, and relationships to intelligently guide reasoning paths.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx

from .reasoning_engine import ReasoningState, Thought, Action

logger = logging.getLogger(__name__)


@dataclass
class KGExplorationPath:
    """Represents a path of exploration through the knowledge graph."""
    path_id: str
    start_node: str
    end_node: str
    relationships: List[str]
    node_types: List[str]
    path_length: int
    relevance_score: float = 0.0
    exploration_value: float = 0.0
    evidence_strength: float = 0.0


@dataclass
class KGReasoning:
    """KG-specific reasoning insights."""
    central_entities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)
    information_gaps: List[str] = field(default_factory=list)
    promising_paths: List[KGExplorationPath] = field(default_factory=list)
    schema_constraints: Dict[str, Any] = field(default_factory=dict)
    reasoning_depth: int = 1


class KGGuidedReasoningEngine:
    """
    Knowledge Graph guided reasoning that uses graph structure to make intelligent decisions.
    Integrates with CoT reasoning to provide KG-aware guidance.
    """

    def __init__(self, db_executor, schema_cache):
        self.db = db_executor
        self.schema = schema_cache
        self.graph_structure = None
        self.entity_importance = {}
        self.relationship_weights = {}
        self.reasoning_history = []

    def initialize_graph_knowledge(self):
        """Initialize understanding of the graph structure and importance metrics."""
        logger.info("Initializing KG structure understanding...")

        # Get basic graph statistics
        self.graph_structure = self._analyze_graph_structure()

        # Compute entity importance (centrality measures)
        self.entity_importance = self._compute_entity_importance()

        # Analyze relationship patterns
        self.relationship_weights = self._analyze_relationship_patterns()

        logger.info("KG structure analysis completed")

    def guide_reasoning_path(self, question: str, current_state: ReasoningState) -> KGReasoning:
        """
        Analyze question and current state to provide KG-guided reasoning suggestions.
        """
        # Parse question for KG entities and concepts
        entities_mentioned = self._extract_entities_from_question(question)
        concepts_mentioned = self._extract_concepts_from_question(question)

        # Analyze current reasoning state for KG context
        explored_entities = self._extract_explored_entities(current_state)

        # Find central entities relevant to the question
        central_entities = self._find_relevant_central_entities(entities_mentioned, concepts_mentioned)

        # Identify promising exploration paths
        promising_paths = self._identify_exploration_paths(
            entities_mentioned, explored_entities, central_entities
        )

        # Identify information gaps based on KG structure
        information_gaps = self._identify_information_gaps(
            question, entities_mentioned, explored_entities
        )

        # Analyze schema constraints relevant to the question
        schema_constraints = self._analyze_relevant_schema(entities_mentioned, concepts_mentioned)

        return KGReasoning(
            central_entities=central_entities,
            key_relationships=self._identify_key_relationships(entities_mentioned),
            information_gaps=information_gaps,
            promising_paths=promising_paths,
            schema_constraints=schema_constraints,
            reasoning_depth=self._recommend_reasoning_depth(question)
        )

    def suggest_next_action(self, kg_reasoning: KGReasoning, current_state: ReasoningState) -> Dict[str, Any]:
        """
        Suggest the next best action based on KG structure and reasoning state.
        """
        suggestions = []

        # Suggest exploration of promising paths
        for path in kg_reasoning.promising_paths[:3]:
            suggestions.append({
                "action_type": "explore_path",
                "tool_name": "enhanced_neo4j_query",
                "parameters": self._generate_path_exploration_query(path),
                "rationale": f"Explore relationship path from {path.start_node} to {path.end_node}",
                "expected_value": path.exploration_value,
                "priority": "high" if path.exploration_value > 0.7 else "medium"
            })

        # Suggest addressing information gaps
        for gap in kg_reasoning.information_gaps[:2]:
            suggestions.append({
                "action_type": "fill_gap",
                "tool_name": self._suggest_tool_for_gap(gap),
                "parameters": self._generate_gap_filling_query(gap),
                "rationale": f"Address information gap: {gap}",
                "expected_value": 0.6,
                "priority": "medium"
            })

        # Suggest exploring central entities if not yet covered
        unexplored_central = [e for e in kg_reasoning.central_entities
                            if e not in self._extract_explored_entities(current_state)]
        for entity in unexplored_central[:2]:
            suggestions.append({
                "action_type": "explore_entity",
                "tool_name": "analyze_node_neighborhoods",
                "parameters": {"node_id": entity, "max_depth": 2},
                "rationale": f"Explore central entity: {entity}",
                "expected_value": self.entity_importance.get(entity, 0.5),
                "priority": "high" if self.entity_importance.get(entity, 0) > 0.8 else "medium"
            })

        # Sort by priority and expected value
        suggestions.sort(key=lambda x: (x["priority"] == "high", x["expected_value"]), reverse=True)

        return {
            "top_suggestion": suggestions[0] if suggestions else None,
            "all_suggestions": suggestions,
            "reasoning_guidance": self._generate_reasoning_guidance(kg_reasoning, current_state)
        }

    def evaluate_reasoning_quality(self, reasoning_state: ReasoningState) -> Dict[str, Any]:
        """
        Evaluate the quality of current reasoning from a KG perspective.
        """
        explored_entities = self._extract_explored_entities(reasoning_state)

        # Coverage analysis
        total_relevant_entities = len(self._find_relevant_central_entities(
            explored_entities, self._extract_concepts_from_thoughts(reasoning_state.thoughts)
        ))
        coverage_score = len(explored_entities) / max(total_relevant_entities, 1)

        # Depth analysis
        max_depth_reached = self._analyze_exploration_depth(reasoning_state)

        # Coherence analysis
        coherence_score = self._analyze_reasoning_coherence(reasoning_state)

        # Completeness analysis
        completeness_score = self._analyze_reasoning_completeness(reasoning_state)

        return {
            "coverage_score": coverage_score,
            "depth_score": min(max_depth_reached / 3.0, 1.0),  # Normalize to 3 levels
            "coherence_score": coherence_score,
            "completeness_score": completeness_score,
            "overall_quality": (coverage_score + coherence_score + completeness_score) / 3,
            "recommendations": self._generate_quality_recommendations(
                coverage_score, max_depth_reached, coherence_score, completeness_score
            )
        }

    def _analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the basic structure of the knowledge graph."""
        try:
            # Get node type distribution
            node_query = """
            MATCH (n)
            RETURN labels(n) as node_types, count(n) as count
            ORDER BY count DESC
            """
            node_result = self.db.run_direct(node_query)

            # Get relationship type distribution
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """
            rel_result = self.db.run_direct(rel_query)

            # Get connectivity patterns
            connectivity_query = """
            MATCH (n)
            RETURN labels(n) as node_type,
                   avg(size((n)-[]-())) as avg_degree,
                   max(size((n)-[]-())) as max_degree
            """
            connectivity_result = self.db.run_direct(connectivity_query)

            return {
                "node_types": node_result.get("data", []),
                "relationship_types": rel_result.get("data", []),
                "connectivity": connectivity_result.get("data", []),
                "total_analysis": "completed"
            }

        except Exception as e:
            logger.error(f"Graph structure analysis failed: {e}")
            return {"error": str(e)}

    def _compute_entity_importance(self) -> Dict[str, float]:
        """Compute importance scores for entities based on centrality measures."""
        try:
            # Use degree centrality as a proxy for importance
            centrality_query = """
            MATCH (n)
            WITH n, size((n)-[]->()) + size((n)<-[]-()) as degree
            WHERE degree > 0
            RETURN n.name as entity, degree,
                   toFloat(degree) / 100.0 as normalized_importance
            ORDER BY degree DESC
            LIMIT 1000
            """

            result = self.db.run_direct(centrality_query)

            importance_scores = {}
            if result["success"] and result["data"]:
                for row in result["data"]:
                    entity = row.get("entity")
                    importance = row.get("normalized_importance", 0.0)
                    if entity:
                        importance_scores[entity] = min(importance, 1.0)

            return importance_scores

        except Exception as e:
            logger.error(f"Entity importance computation failed: {e}")
            return {}

    def _analyze_relationship_patterns(self) -> Dict[str, float]:
        """Analyze patterns in relationships to understand their importance."""
        try:
            pattern_query = """
            MATCH ()-[r]->()
            WITH type(r) as rel_type, count(r) as frequency
            ORDER BY frequency DESC
            RETURN rel_type, frequency,
                   toFloat(frequency) / toFloat(sum(frequency)) OVER () as weight
            """

            result = self.db.run_direct(pattern_query)

            weights = {}
            if result["success"] and result["data"]:
                for row in result["data"]:
                    rel_type = row.get("rel_type")
                    weight = row.get("weight", 0.0)
                    if rel_type:
                        weights[rel_type] = weight

            return weights

        except Exception as e:
            logger.error(f"Relationship pattern analysis failed: {e}")
            return {}

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Extract potential entity names from the question."""
        # Simple keyword extraction - could be enhanced with NER
        words = question.split()
        entities = []

        # Look for capitalized words that might be entity names
        for word in words:
            if word.isalpha() and word[0].isupper() and len(word) > 2:
                entities.append(word)

        # Look for known patterns (e.g., region names, scientific terms)
        scientific_terms = [
            "morphological", "molecular", "neurotransmitter", "axonal", "dendritic",
            "cortical", "subcortical", "hippocampal", "thalamic"
        ]

        concepts = []
        question_lower = question.lower()
        for term in scientific_terms:
            if term in question_lower:
                concepts.append(term)

        return entities + concepts

    def _extract_concepts_from_question(self, question: str) -> List[str]:
        """Extract conceptual terms from the question."""
        concept_keywords = {
            "similarity": ["similar", "alike", "comparable", "resembling"],
            "difference": ["different", "distinct", "contrasting", "divergent"],
            "morphology": ["morphological", "structural", "anatomical", "shape"],
            "molecular": ["molecular", "genetic", "biochemical", "cellular"],
            "relationship": ["relationship", "connection", "association", "correlation"],
            "comparison": ["compare", "contrast", "analyze", "examine"]
        }

        question_lower = question.lower()
        extracted_concepts = []

        for concept, keywords in concept_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                extracted_concepts.append(concept)

        return extracted_concepts

    def _extract_explored_entities(self, state: ReasoningState) -> Set[str]:
        """Extract entities that have been explored in the current reasoning state."""
        explored = set()

        # Extract from observations
        for obs in state.observations:
            if "data" in obs.result:
                data = obs.result["data"]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, str) and len(value) < 50:
                                    explored.add(value)

        # Extract from thoughts and reflections
        for thought in state.thoughts:
            words = thought.content.split()
            for word in words:
                if word.isalpha() and word[0].isupper():
                    explored.add(word)

        return explored

    def _find_relevant_central_entities(self, mentioned_entities: List[str],
                                      concepts: List[str]) -> List[str]:
        """Find central entities relevant to the question."""
        relevant_entities = []

        # Start with mentioned entities that are also central
        for entity in mentioned_entities:
            if entity in self.entity_importance:
                importance = self.entity_importance[entity]
                if importance > 0.5:  # Threshold for centrality
                    relevant_entities.append(entity)

        # If no mentioned entities are central, get top central entities
        if not relevant_entities:
            sorted_entities = sorted(
                self.entity_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            relevant_entities = [entity for entity, _ in sorted_entities[:10]]

        return relevant_entities

    def _identify_exploration_paths(self, mentioned_entities: List[str],
                                  explored_entities: Set[str],
                                  central_entities: List[str]) -> List[KGExplorationPath]:
        """Identify promising paths for exploration."""
        paths = []

        # Create paths between mentioned entities and central entities
        for start in mentioned_entities[:5]:
            for end in central_entities[:5]:
                if start != end and start not in explored_entities:
                    path = KGExplorationPath(
                        path_id=f"{start}->{end}",
                        start_node=start,
                        end_node=end,
                        relationships=["unknown"],  # Would be filled by path finding
                        node_types=["unknown"],
                        path_length=2,  # Estimate
                        exploration_value=self.entity_importance.get(end, 0.5)
                    )
                    paths.append(path)

        # Sort by exploration value
        paths.sort(key=lambda p: p.exploration_value, reverse=True)
        return paths[:5]

    def _identify_information_gaps(self, question: str, mentioned_entities: List[str],
                                 explored_entities: Set[str]) -> List[str]:
        """Identify information gaps based on question and exploration state."""
        gaps = []

        # Entities mentioned but not explored
        for entity in mentioned_entities:
            if entity not in explored_entities:
                gaps.append(f"Detailed information about {entity}")

        # Conceptual gaps based on question type
        question_lower = question.lower()
        if "similar" in question_lower and "different" in question_lower:
            gaps.append("Comprehensive similarity metrics")
            gaps.append("Molecular profile comparisons")

        if "morphological" in question_lower:
            gaps.append("Detailed morphological feature analysis")

        if "relationship" in question_lower:
            gaps.append("Network connectivity analysis")

        return gaps

    def _analyze_relevant_schema(self, entities: List[str], concepts: List[str]) -> Dict[str, Any]:
        """Analyze schema elements relevant to the current reasoning."""
        relevant_schema = {
            "node_types": [],
            "relationship_types": [],
            "properties": []
        }

        # Based on concepts, identify relevant schema elements
        if "morphological" in concepts:
            relevant_schema["properties"].extend([
                "axonal_length", "dendritic_length", "axonal_branches", "dendritic_branches"
            ])

        if "molecular" in concepts:
            relevant_schema["properties"].extend([
                "markers", "transcription_factor_markers", "dominant_neurotransmitter_type"
            ])
            relevant_schema["relationship_types"].append("HAS_SUBCLASS")

        if "similarity" in concepts:
            relevant_schema["node_types"].append("Region")

        return relevant_schema

    def _recommend_reasoning_depth(self, question: str) -> int:
        """Recommend reasoning depth based on question complexity."""
        complexity_indicators = [
            "compare", "analyze", "relationship", "pattern", "significant",
            "comprehensive", "detailed", "complex", "multiple"
        ]

        question_lower = question.lower()
        complexity_score = sum(1 for indicator in complexity_indicators
                             if indicator in question_lower)

        # Map complexity to depth
        if complexity_score >= 4:
            return 4  # Deep reasoning
        elif complexity_score >= 2:
            return 3  # Moderate reasoning
        else:
            return 2  # Basic reasoning

    def _generate_path_exploration_query(self, path: KGExplorationPath) -> Dict[str, Any]:
        """Generate query to explore a specific path."""
        return {
            "query": f"""
            MATCH path = (start)-[*1..3]-(end)
            WHERE start.name CONTAINS '{path.start_node}'
              AND end.name CONTAINS '{path.end_node}'
            RETURN start.name as start_entity,
                   end.name as end_entity,
                   relationships(path) as relationships,
                   length(path) as path_length
            LIMIT 10
            """
        }

    def _suggest_tool_for_gap(self, gap: str) -> str:
        """Suggest appropriate tool for filling a specific information gap."""
        gap_lower = gap.lower()

        if "morphological" in gap_lower:
            return "find_morphologically_similar_regions"
        elif "molecular" in gap_lower:
            return "get_neurotransmitter_profiles"
        elif "similarity" in gap_lower:
            return "compare_molecular_markers"
        elif "network" in gap_lower or "connectivity" in gap_lower:
            return "compute_graph_metrics"
        else:
            return "enhanced_neo4j_query"

    def _generate_gap_filling_query(self, gap: str) -> Dict[str, Any]:
        """Generate query parameters to fill a specific information gap."""
        gap_lower = gap.lower()

        if "morphological" in gap_lower:
            return {"similarity_threshold": 0.1, "limit": 20}
        elif "molecular" in gap_lower:
            return {"region_names": ["unknown"]}  # Would be filled based on context
        else:
            return {"query": "MATCH (n) RETURN n LIMIT 10"}

    def _extract_concepts_from_thoughts(self, thoughts: List[Thought]) -> List[str]:
        """Extract conceptual terms from reasoning thoughts."""
        concepts = set()
        for thought in thoughts:
            words = thought.content.lower().split()
            concept_terms = [
                "similar", "different", "morphological", "molecular",
                "relationship", "pattern", "structure", "function"
            ]
            for term in concept_terms:
                if term in words:
                    concepts.add(term)
        return list(concepts)

    def _analyze_exploration_depth(self, state: ReasoningState) -> int:
        """Analyze how deep the exploration has gone."""
        max_depth = 1

        # Analyze action sequences for depth indicators
        for action in state.actions:
            if "neighborhoods" in action.tool_name:
                depth = action.parameters.get("max_depth", 1)
                max_depth = max(max_depth, depth)
            elif "shortest_paths" in action.tool_name:
                max_depth = max(max_depth, 3)  # Path finding indicates depth

        return max_depth

    def _analyze_reasoning_coherence(self, state: ReasoningState) -> float:
        """Analyze the coherence of the reasoning process."""
        if len(state.thoughts) < 2:
            return 0.5

        # Simple coherence measure based on thought progression
        coherent_transitions = 0
        total_transitions = len(state.thoughts) - 1

        for i in range(len(state.thoughts) - 1):
            current_thought = state.thoughts[i].content.lower()
            next_thought = state.thoughts[i + 1].content.lower()

            # Check for logical progression keywords
            if any(keyword in next_thought for keyword in
                   ["therefore", "thus", "however", "furthermore", "because", "since"]):
                coherent_transitions += 1

        return coherent_transitions / max(total_transitions, 1)

    def _analyze_reasoning_completeness(self, state: ReasoningState) -> float:
        """Analyze completeness of the reasoning process."""
        completeness_factors = {
            "has_hypothesis": any("hypothesis" in t.content.lower() for t in state.thoughts),
            "has_evidence": len(state.observations) > 0,
            "has_reflection": len(state.reflections) > 0,
            "has_conclusion": any("conclusion" in t.content.lower() for t in state.thoughts[-2:]),
            "multiple_perspectives": len(set(t.type for t in state.thoughts)) > 1
        }

        return sum(completeness_factors.values()) / len(completeness_factors)

    def _generate_reasoning_guidance(self, kg_reasoning: KGReasoning,
                                   current_state: ReasoningState) -> Dict[str, Any]:
        """Generate guidance for the reasoning process."""
        return {
            "focus_areas": kg_reasoning.central_entities[:3],
            "recommended_depth": kg_reasoning.reasoning_depth,
            "key_relationships_to_explore": kg_reasoning.key_relationships[:3],
            "information_priorities": kg_reasoning.information_gaps[:3],
            "schema_guidance": {
                "relevant_node_types": kg_reasoning.schema_constraints.get("node_types", []),
                "relevant_relationships": kg_reasoning.schema_constraints.get("relationship_types", []),
                "key_properties": kg_reasoning.schema_constraints.get("properties", [])
            }
        }

    def _generate_quality_recommendations(self, coverage: float, depth: int,
                                        coherence: float, completeness: float) -> List[str]:
        """Generate recommendations for improving reasoning quality."""
        recommendations = []

        if coverage < 0.3:
            recommendations.append("Explore more entities relevant to the question")

        if depth < 2:
            recommendations.append("Deepen the analysis with more detailed exploration")

        if coherence < 0.4:
            recommendations.append("Improve logical flow between reasoning steps")

        if completeness < 0.6:
            recommendations.append("Include more comprehensive analysis covering hypotheses, evidence, and conclusions")

        if not recommendations:
            recommendations.append("Reasoning quality is good - consider synthesis of findings")

        return recommendations