"""
AIPOM V5: Dynamic Scientific Reasoning System with Deep Semantic Understanding
No hardcoded patterns - pure dynamic reasoning and learning

Author: NeuroXiv Team
Date: 2025-12-19
Version: 5.0
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Core Abstractions ====================

@dataclass
class SemanticContext:
    """Maintains semantic understanding throughout reasoning"""
    question_intent: str  # What the user really wants
    key_concepts: Dict[str, str]  # concept -> interpretation
    constraints: Dict[str, Any]  # Constraints to maintain
    focus_trajectory: List[str]  # How focus evolved
    validity_criteria: List[str]  # What makes an answer valid


@dataclass
class LogicalReasoning:
    """Represents a single logical reasoning step"""
    premise: str  # What we assume/know
    inference: str  # What we deduce
    operation: str  # How we test this
    expected_outcome: str  # What success looks like
    actual_outcome: Optional[str] = None
    validity: Optional[bool] = None
    critique: Optional[str] = None


@dataclass
class QueryIntention:
    """What a query is trying to achieve"""
    scientific_goal: str  # High-level goal
    data_need: str  # What data we need
    success_metric: str  # How to measure success
    fallback_strategy: str  # What to do if it fails


# ==================== Semantic Understanding Engine ====================

class SemanticUnderstanding:
    """Deep semantic understanding of scientific questions"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def understand_question(self, question: str) -> SemanticContext:
        """Truly understand what the question is asking"""

        prompt = f"""You are a neuroscientist analyzing a research question. 
        Provide DEEP SEMANTIC UNDERSTANDING, not surface-level keyword extraction.

Question: {question}

Think step by step:
1. What is the TRUE INTENT behind this question? What scientific insight is sought?
2. What are the KEY CONCEPTS and how should they be interpreted IN THIS CONTEXT?
3. What CONSTRAINTS are implied (not just stated)?
4. What would constitute a VALID, MEANINGFUL answer?
5. What LOGICAL STEPS are needed to answer this?

For example, "find region pairs with similar morphology but different molecular features":
- TRUE INTENT: Identify regions that look structurally alike but function differently
- KEY CONCEPT "similar morphology": Not identical, but comparable within reasonable threshold
- KEY CONCEPT "different molecular": Distinct functional signatures, not minor variations
- IMPLIED CONSTRAINT: Both regions must have sufficient data for comparison
- VALID ANSWER: Specific pairs with quantified similarity/difference metrics

Return JSON with your SEMANTIC UNDERSTANDING:
{{
    "question_intent": "The deeper scientific goal",
    "key_concepts": {{
        "concept1": "How to interpret this concept",
        "concept2": "What this really means"
    }},
    "constraints": {{
        "explicit": ["stated constraints"],
        "implicit": ["unstated but necessary constraints"]
    }},
    "validity_criteria": [
        "What makes an answer scientifically valid",
        "What evidence is needed"
    ],
    "reasoning_approach": "How to logically approach this"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in neuroscience and scientific reasoning."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            understanding = json.loads(response.choices[0].message.content)

            return SemanticContext(
                question_intent=understanding.get('question_intent', ''),
                key_concepts=understanding.get('key_concepts', {}),
                constraints=understanding.get('constraints', {}),
                focus_trajectory=[],
                validity_criteria=understanding.get('validity_criteria', [])
            )

        except Exception as e:
            logger.error(f"Failed to understand question semantically: {e}")
            # Fallback to basic understanding
            return SemanticContext(
                question_intent="Explore patterns in the data",
                key_concepts={"main": question},
                constraints={},
                focus_trajectory=[],
                validity_criteria=["Find relevant patterns"]
            )

    def understand_query_failure(self, query: str, error: Optional[str],
                                 empty_result: bool, context: SemanticContext) -> Dict:
        """Understand WHY a query failed semantically"""

        prompt = f"""A database query failed. Analyze WHY it failed SEMANTICALLY, not just technically.

Query: {query}

Error: {error if error else "No error, but returned empty results"}
Context: We're trying to understand: {context.question_intent}
Key concepts: {json.dumps(context.key_concepts)}

Think deeply:
1. What ASSUMPTION did the query make that might be wrong?
2. What SEMANTIC MISMATCH exists between intent and implementation?
3. What CONCEPTUAL ERROR might have occurred?
4. What ALTERNATIVE INTERPRETATION could work better?

Don't just say "property doesn't exist" - explain WHY the query structure doesn't match the data model.

Return JSON:
{{
    "semantic_failure_reason": "Deep reason for failure",
    "flawed_assumptions": ["assumptions that were wrong"],
    "conceptual_corrections": {{
        "wrong": "what was conceptually wrong",
        "correct": "what would be conceptually correct"
    }},
    "alternative_approach": "A fundamentally different way to approach this"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a database and scientific reasoning expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Failed to analyze query failure: {e}")
            return {
                "semantic_failure_reason": "Unknown",
                "flawed_assumptions": [],
                "conceptual_corrections": {},
                "alternative_approach": "Try simpler exploration"
            }


# ==================== Logical Reasoning Engine ====================

class LogicalReasoningEngine:
    """Maintains logical coherence throughout reasoning"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.reasoning_chain: List[LogicalReasoning] = []

    def create_reasoning_step(self, context: SemanticContext,
                              previous_steps: List[LogicalReasoning],
                              discoveries: List[str]) -> LogicalReasoning:
        """Create next logical reasoning step"""

        # Build context of what we know
        known_facts = "\n".join([f"- {d}" for d in discoveries]) if discoveries else "None yet"
        prev_reasoning = "\n".join([f"- Tried: {s.inference}, Result: {s.actual_outcome}"
                                    for s in previous_steps[-3:]]) if previous_steps else "None"

        prompt = f"""Create the NEXT LOGICAL STEP in answering this scientific question.

Goal: {context.question_intent}
Key Concepts: {json.dumps(context.key_concepts)}

What we know so far:
{known_facts}

Previous reasoning:
{prev_reasoning}

Create the next LOGICAL step that:
1. Builds on what we know
2. Stays focused on the goal
3. Doesn't repeat failed approaches
4. Is scientifically sound

Return JSON:
{{
    "premise": "What we assume or know to be true",
    "inference": "What we can deduce from this",
    "operation": "How to test this inference", 
    "expected_outcome": "What successful result looks like"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a scientific reasoning expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            step_data = json.loads(response.choices[0].message.content)

            return LogicalReasoning(
                premise=step_data.get('premise', ''),
                inference=step_data.get('inference', ''),
                operation=step_data.get('operation', ''),
                expected_outcome=step_data.get('expected_outcome', '')
            )

        except Exception as e:
            logger.error(f"Failed to create reasoning step: {e}")
            return LogicalReasoning(
                premise="Need to explore available data",
                inference="Understanding data structure will guide analysis",
                operation="Query schema and sample data",
                expected_outcome="Discover available properties and relationships"
            )

    def critique_reasoning(self, step: LogicalReasoning,
                           context: SemanticContext) -> str:
        """Self-critique the reasoning step"""

        prompt = f"""Critically evaluate this reasoning step for LOGICAL VALIDITY and RELEVANCE.

Goal: {context.question_intent}
Reasoning:
- Premise: {step.premise}
- Inference: {step.inference}
- Operation: {step.operation}
- Expected: {step.expected_outcome}
- Actual: {step.actual_outcome}

Critique:
1. Is the inference logically valid from the premise?
2. Does this step advance toward the goal?
3. Is the operation appropriate for testing the inference?
4. What logical flaws exist?

Provide a brief, critical assessment (max 100 words)."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a critical scientific reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to critique reasoning: {e}")
            return "Unable to critique"


# ==================== Dynamic Query Generator ====================

class DynamicQueryGenerator:
    """Generates queries based on semantic understanding and available schema"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate_query(self, reasoning: LogicalReasoning,
                       schema: Dict, context: SemanticContext,
                       failed_queries: List[str]) -> Tuple[str, QueryIntention]:
        """Generate query that implements the logical operation"""

        # Show what failed before
        failures = "\n".join([f"Failed: {q[:100]}..." for q in failed_queries[-3:]]) if failed_queries else "None"

        prompt = f"""Generate a Neo4j Cypher query to implement this scientific operation.

OPERATION: {reasoning.operation}
GOAL: {reasoning.expected_outcome}
CONTEXT: {context.question_intent}

AVAILABLE SCHEMA:
Nodes: {', '.join(schema.get('nodes', {}).keys())}
Relationships: {', '.join(schema.get('relationships', {}).keys())}
Sample Region properties: {', '.join(list(schema.get('nodes', {}).get('Region', {}).get('properties', []))[:10])}

PREVIOUSLY FAILED QUERIES:
{failures}

Generate a query that:
1. Correctly uses the available schema
2. Implements the operation accurately
3. Avoids patterns that failed before
4. Returns meaningful data for the scientific goal

Return JSON:
{{
    "query": "Complete Cypher query",
    "scientific_goal": "What this query discovers",
    "data_need": "What data it retrieves",
    "success_metric": "How to measure if it worked",
    "fallback_strategy": "What to try if this fails"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Neo4j and neuroscience expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)

            intention = QueryIntention(
                scientific_goal=result.get('scientific_goal', ''),
                data_need=result.get('data_need', ''),
                success_metric=result.get('success_metric', ''),
                fallback_strategy=result.get('fallback_strategy', '')
            )

            return result.get('query', ''), intention

        except Exception as e:
            logger.error(f"Failed to generate query: {e}")
            # Simple fallback
            return "MATCH (n) RETURN n LIMIT 5", QueryIntention(
                scientific_goal="Explore data",
                data_need="Any data",
                success_metric="Non-empty results",
                fallback_strategy="Try different node type"
            )


# ==================== Scientific Insight Extractor ====================

class ScientificInsightExtractor:
    """Extracts meaningful scientific insights from data"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def extract_insight(self, data: List[Dict], reasoning: LogicalReasoning,
                        context: SemanticContext) -> Tuple[str, float]:
        """Extract scientific insight and confidence"""

        if not data:
            return "No data to analyze", 0.0

        # Prepare data sample
        data_sample = json.dumps(data[:5], default=str, indent=2) if len(data) > 5 else json.dumps(data, default=str,
                                                                                                   indent=2)

        prompt = f"""Extract SCIENTIFIC INSIGHT from this data.

SCIENTIFIC GOAL: {context.question_intent}
HYPOTHESIS: {reasoning.inference}
EXPECTED: {reasoning.expected_outcome}

DATA ({len(data)} total results):
{data_sample}

Provide:
1. KEY SCIENTIFIC FINDING (what this data reveals)
2. RELEVANCE to the goal (how this helps answer the question)
3. CONFIDENCE (0-1, based on data quality and relevance)
4. LIMITATIONS (what this doesn't tell us)

Return JSON:
{{
    "finding": "The key scientific discovery",
    "relevance": "How this advances toward the goal",
    "confidence": 0.0,
    "limitations": ["what we still don't know"],
    "next_question": "What to investigate next"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a neuroscience data analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)

            # Combine finding and relevance for insight
            insight = f"{result.get('finding', '')} - {result.get('relevance', '')}"
            confidence = float(result.get('confidence', 0.5))

            return insight, confidence

        except Exception as e:
            logger.error(f"Failed to extract insight: {e}")
            return f"Found {len(data)} results", 0.3


# ==================== Enhanced Knowledge Graph Interface ====================

class AdaptiveKGInterface:
    """KG interface that learns and adapts from interactions"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.schema_cache = {}
        self.query_patterns = {}
        self.discover_schema()

    def discover_schema(self):
        """Discover available schema"""
        logger.info("Discovering schema...")

        # Get node labels
        with self.driver.session() as session:
            result = session.run("CALL db.labels()")
            labels = [r['label'] for r in result]

        # Get relationships
        with self.driver.session() as session:
            result = session.run("CALL db.relationshipTypes()")
            rels = [r['relationshipType'] for r in result]

        # Get sample properties for each label
        schema = {'nodes': {}, 'relationships': {}}

        for label in labels:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (n:{label})
                    WITH n LIMIT 5
                    RETURN keys(n) as props
                """)
                all_props = set()
                for record in result:
                    all_props.update(record['props'])
                schema['nodes'][label] = {'properties': list(all_props)}

        for rel in rels:
            schema['relationships'][rel] = {}

        self.schema_cache = schema
        logger.info(f"Schema discovered: {len(labels)} nodes, {len(rels)} relationships")

    def execute_with_learning(self, query: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute query and learn from result"""

        try:
            with self.driver.session() as session:
                result = session.run(query)
                data = [dict(r) for r in result]

                # Learn successful pattern
                if data:
                    self._record_success(query, len(data))

                return data, None

        except Exception as e:
            error_msg = str(e)
            self._record_failure(query, error_msg)
            return [], error_msg

    def _record_success(self, query: str, result_count: int):
        """Record successful query pattern"""
        # Simplified pattern extraction
        pattern = query[:50]
        self.query_patterns[pattern] = {
            'success': True,
            'count': result_count
        }

    def _record_failure(self, query: str, error: str):
        """Record failed query pattern"""
        pattern = query[:50]
        self.query_patterns[pattern] = {
            'success': False,
            'error': error[:100]
        }

    def get_schema(self) -> Dict:
        """Get discovered schema"""
        return self.schema_cache

    def close(self):
        """Close connection"""
        self.driver.close()


# ==================== Main Scientific Reasoning Agent ====================

class ScientificReasoningAgent:
    """Main agent that orchestrates scientific reasoning"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str):

        # Initialize components
        self.llm = OpenAI(api_key=openai_api_key)
        self.kg = AdaptiveKGInterface(neo4j_uri, neo4j_user, neo4j_password)

        # Initialize reasoning components
        self.semantic_engine = SemanticUnderstanding(self.llm)
        self.logic_engine = LogicalReasoningEngine(self.llm)
        self.query_generator = DynamicQueryGenerator(self.llm)
        self.insight_extractor = ScientificInsightExtractor(self.llm)

        # State
        self.max_steps = 10
        self.failed_queries: List[str] = []
        self.discoveries: List[str] = []

    def answer(self, question: str) -> Dict:
        """Answer scientific question with deep reasoning"""

        logger.info(f"Processing question: {question}")

        # 1. Semantic understanding
        context = self.semantic_engine.understand_question(question)
        logger.info(f"Understood intent: {context.question_intent}")

        # 2. Execute reasoning chain
        reasoning_steps = []
        total_confidence = 0.0

        for step_num in range(self.max_steps):
            # Create logical reasoning step
            reasoning = self.logic_engine.create_reasoning_step(
                context, reasoning_steps, self.discoveries
            )

            # Generate query
            query, intention = self.query_generator.generate_query(
                reasoning, self.kg.get_schema(), context, self.failed_queries
            )

            # Execute query
            data, error = self.kg.execute_with_learning(query)

            if not data and error:
                # Understand failure
                failure_analysis = self.semantic_engine.understand_query_failure(
                    query, error, True, context
                )
                reasoning.actual_outcome = f"Failed: {failure_analysis['semantic_failure_reason']}"
                reasoning.validity = False
                self.failed_queries.append(query)

                # Learn from failure
                logger.info(f"Step {step_num + 1} failed: {failure_analysis['semantic_failure_reason']}")

            else:
                # Extract insight
                insight, confidence = self.insight_extractor.extract_insight(
                    data, reasoning, context
                )
                reasoning.actual_outcome = insight
                reasoning.validity = confidence > 0.3
                total_confidence += confidence

                if reasoning.validity:
                    self.discoveries.append(insight)
                    # Update focus
                    context.focus_trajectory.append(f"Step {step_num + 1}: {insight[:50]}")

                logger.info(f"Step {step_num + 1}: {insight[:100]} (confidence: {confidence:.2f})")

            # Self-critique
            reasoning.critique = self.logic_engine.critique_reasoning(reasoning, context)
            reasoning_steps.append(reasoning)

            # Check if we've answered the question
            if self._is_answer_complete(context, self.discoveries):
                logger.info("Answer complete")
                break

            # Check if we're stuck
            if len(self.failed_queries) > 5 and not self.discoveries:
                logger.warning("Too many failures, concluding")
                break

        # 3. Synthesize answer
        answer = self._synthesize_answer(context, reasoning_steps, self.discoveries)
        avg_confidence = total_confidence / max(len(reasoning_steps), 1)

        return {
            'question': question,
            'semantic_understanding': {
                'intent': context.question_intent,
                'key_concepts': context.key_concepts,
                'validity_criteria': context.validity_criteria
            },
            'answer': answer,
            'reasoning_steps': [
                {
                    'step': i + 1,
                    'premise': r.premise,
                    'inference': r.inference,
                    'outcome': r.actual_outcome,
                    'valid': r.validity,
                    'critique': r.critique
                }
                for i, r in enumerate(reasoning_steps)
            ],
            'discoveries': self.discoveries,
            'confidence': min(avg_confidence, 1.0),
            'focus_trajectory': context.focus_trajectory
        }

    def _is_answer_complete(self, context: SemanticContext,
                            discoveries: List[str]) -> bool:
        """Check if we have sufficient answer"""

        if not discoveries:
            return False

        # Check against validity criteria
        for criterion in context.validity_criteria:
            criterion_met = any(
                criterion.lower()[:20] in discovery.lower()
                for discovery in discoveries
            )
            if not criterion_met:
                return False

        # Need at least some discoveries
        return len(discoveries) >= 2

    def _synthesize_answer(self, context: SemanticContext,
                           steps: List[LogicalReasoning],
                           discoveries: List[str]) -> str:
        """Synthesize final answer"""

        if not discoveries:
            return (f"Unable to fully address the question: {context.question_intent}\n\n"
                    f"The system attempted {len(steps)} reasoning steps but could not find sufficient data. "
                    f"This may indicate that the required data relationships don't exist in the current graph structure.")

        # Build answer from discoveries
        answer_parts = [
            f"Regarding: {context.question_intent}\n",
            "Key findings:"
        ]

        for i, discovery in enumerate(discoveries, 1):
            answer_parts.append(f"{i}. {discovery}")

        # Add synthesis
        answer_parts.append(f"\nBased on {len(steps)} logical reasoning steps, "
                            f"the analysis focused on: {' → '.join(context.focus_trajectory[:3])}")

        return "\n".join(answer_parts)

    def close(self):
        """Clean up"""
        self.kg.close()


# ==================== Main Execution ====================

def main():
    """Demonstrate the system"""

    config = {
        'neo4j_uri': "bolt://100.88.72.32:7687",
        'neo4j_user': "neo4j",
        'neo4j_password': "neuroxiv",
        'openai_api_key': ""
    }

    if not config['openai_api_key']:
        print("Please set OpenAI API key")
        return

    agent = ScientificReasoningAgent(**config)

    try:
        # Test with the challenging question
        # question = "find some region pairs with enough data and their morphological features are similar but molecular features are very different"
        question = "Compare morphological and transcriptomic differences between motor and visual cortices"
        # question = "Analyze the projection pattern of the brain region with the highest proportion of Car3 transcriptome subclass neurons"
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        result = agent.answer(question)

        print("\n[Semantic Understanding]")
        print(f"Intent: {result['semantic_understanding']['intent']}")
        print(f"Key Concepts: {result['semantic_understanding']['key_concepts']}")

        print("\n[Answer]")
        print(result['answer'])

        print(f"\n[Overall Confidence: {result['confidence']:.1%}]")

        print("\n[Logical Reasoning Chain]")
        for step in result['reasoning_steps']:
            status = "✓" if step['valid'] else "✗"
            print(f"{status} Step {step['step']}: {step['inference']}")
            print(f"   Outcome: {step['outcome'][:100]}")
            if step['critique']:
                print(f"   Critique: {step['critique'][:100]}")

        print("\n[Focus Trajectory]")
        for focus in result['focus_trajectory']:
            print(f"  → {focus}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()