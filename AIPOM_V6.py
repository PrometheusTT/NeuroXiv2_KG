"""
Universal TAOR Agent with GPT-4o
Uses latest OpenAI models with strict JSON parsing

Key Features:
- GPT-4o for fast and capable reasoning
- GPT-4-turbo for deep analysis when needed
- Complete Think-Act-Observe-Reflect loop
- Strict JSON response format with validation
"""

import json
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import traceback

from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class ModelConfig:
    """Model configuration for different tasks"""
    # Use GPT-4o as primary model (fast and capable)
    PRIMARY_MODEL = "gpt-4o"
    # Use GPT-4-turbo for complex analysis
    DEEP_MODEL = "gpt-5"
    # Temperature settings
    CREATIVE_TEMP = 0.7
    ANALYTICAL_TEMP = 0.3
    PRECISE_TEMP = 0.1


# ==================== Core Data Models ====================

class ThoughtType(Enum):
    """Types of thoughts"""
    UNDERSTANDING = "understanding"
    HYPOTHESIS = "hypothesis"
    EXPLORATION = "exploration"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"

class ActionType(Enum):
    """Types of actions"""
    QUERY = "query"
    EXPLORE = "explore"
    CALCULATE = "calculate"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"

@dataclass
class Thought:
    """A thought in the reasoning process"""
    thought_type: ThoughtType
    content: str
    reasoning: str
    confidence: float
    next_action_suggestion: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "thought_type": self.thought_type.value,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "next_action_suggestion": self.next_action_suggestion,
            "timestamp": self.timestamp
        }

@dataclass
class Action:
    """An action to take"""
    action_type: ActionType
    query: Optional[str]
    parameters: Dict[str, Any]
    expected_outcome: str
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type.value,
            "query": self.query,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "reasoning": self.reasoning
        }

@dataclass
class Observation:
    """Observation from an action"""
    success: bool
    data: Any
    row_count: int
    relevance: float
    surprise: float
    insights: List[str]

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "row_count": self.row_count,
            "relevance": self.relevance,
            "surprise": self.surprise,
            "insights": self.insights
        }

@dataclass
class Reflection:
    """Reflection on the experience"""
    what_worked: str
    what_failed: str
    lesson_learned: str
    confidence_change: float
    should_continue: bool
    new_hypothesis: Optional[str]

    def to_dict(self) -> Dict:
        return {
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "lesson_learned": self.lesson_learned,
            "confidence_change": self.confidence_change,
            "should_continue": self.should_continue,
            "new_hypothesis": self.new_hypothesis
        }


# ==================== JSON Parser with Validation ====================

class StrictJSONParser:
    """Strict JSON parser with validation and error recovery"""

    @staticmethod
    def _clean_json(text: str) -> str:
        """Clean JSON text for parsing"""
        # Remove markdown code blocks
        text = text.replace('```json', '').replace('```', '')
        # Remove any leading/trailing whitespace
        text = text.strip()
        # Handle common JSON issues
        if not text.startswith('{'):
            # Find first {
            idx = text.find('{')
            if idx != -1:
                text = text[idx:]
        if not text.endswith('}'):
            # Find last }
            idx = text.rfind('}')
            if idx != -1:
                text = text[:idx+1]
        return text

    @staticmethod
    def parse_thought(response: str) -> Thought:
        """Parse thought response with validation"""
        try:
            cleaned = StrictJSONParser._clean_json(response)
            data = json.loads(cleaned)

            # Validate and convert thought_type
            thought_type_str = data.get("thought_type", "exploration")
            try:
                thought_type = ThoughtType(thought_type_str)
            except ValueError:
                thought_type = ThoughtType.EXPLORATION

            return Thought(
                thought_type=thought_type,
                content=str(data.get("content", "Exploring...")),
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5)),
                next_action_suggestion=str(data.get("next_action_suggestion", "Continue exploring"))
            )

        except Exception as e:
            logger.error(f"Failed to parse thought: {e}\nResponse: {response[:200]}")
            return Thought(
                thought_type=ThoughtType.EXPLORATION,
                content="Need to explore available data",
                reasoning="Parsing failed, defaulting to exploration",
                confidence=0.1,
                next_action_suggestion="Explore database schema"
            )

    @staticmethod
    def parse_action(response: str) -> Action:
        """Parse action response with validation"""
        try:
            cleaned = StrictJSONParser._clean_json(response)
            data = json.loads(cleaned)

            # Validate action_type
            action_type_str = data.get("action_type", "explore")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                action_type = ActionType.EXPLORE

            return Action(
                action_type=action_type,
                query=data.get("query"),
                parameters=data.get("parameters", {}),
                expected_outcome=str(data.get("expected_outcome", "Gather information")),
                reasoning=str(data.get("reasoning", ""))
            )

        except Exception as e:
            logger.error(f"Failed to parse action: {e}\nResponse: {response[:200]}")
            return Action(
                action_type=ActionType.EXPLORE,
                query="MATCH (n) RETURN DISTINCT labels(n)[0] as label, count(n) as count LIMIT 10",
                parameters={},
                expected_outcome="Understand data structure",
                reasoning="Parsing failed, exploring schema"
            )

    @staticmethod
    def parse_observation_analysis(response: str) -> Dict:
        """Parse observation analysis"""
        try:
            cleaned = StrictJSONParser._clean_json(response)
            data = json.loads(cleaned)

            return {
                "relevance": float(data.get("relevance", 0.5)),
                "surprise": float(data.get("surprise", 0.5)),
                "insights": data.get("insights", [])
            }

        except Exception as e:
            logger.error(f"Failed to parse observation: {e}")
            return {"relevance": 0.5, "surprise": 0.5, "insights": []}

    @staticmethod
    def parse_reflection(response: str) -> Reflection:
        """Parse reflection response"""
        try:
            cleaned = StrictJSONParser._clean_json(response)
            data = json.loads(cleaned)

            return Reflection(
                what_worked=str(data.get("what_worked", "")),
                what_failed=str(data.get("what_failed", "")),
                lesson_learned=str(data.get("lesson_learned", "")),
                confidence_change=float(data.get("confidence_change", 0)),
                should_continue=bool(data.get("should_continue", True)),
                new_hypothesis=data.get("new_hypothesis")
            )

        except Exception as e:
            logger.error(f"Failed to parse reflection: {e}")
            return Reflection(
                what_worked="",
                what_failed="Parsing failed",
                lesson_learned="Need better error handling",
                confidence_change=0,
                should_continue=True,
                new_hypothesis=None
            )


# ==================== Memory Systems ====================

class WorkingMemory:
    """Agent's working memory for current problem"""

    def __init__(self):
        self.original_question = None
        self.current_understanding = {}
        self.discovered_facts = []
        self.open_questions = deque()
        self.attempted_queries = []
        self.current_hypothesis = None
        self.confidence_evolution = []
        self.discovered_schema = {}

    def add_fact(self, fact: str, evidence: Any, confidence: float):
        """Add a discovered fact"""
        self.discovered_facts.append({
            'fact': fact,
            'evidence': evidence,
            'confidence': confidence,
            'timestamp': time.time()
        })

    def add_attempted_query(self, query: str, success: bool, row_count: int):
        """Record an attempted query"""
        self.attempted_queries.append({
            'query': query,
            'success': success,
            'row_count': row_count,
            'timestamp': time.time()
        })

    def update_schema(self, schema_info: Dict):
        """Update discovered schema"""
        self.discovered_schema.update(schema_info)

    def get_context_summary(self) -> Dict:
        """Get summary of current context"""
        return {
            'question': self.original_question,
            'understanding': self.current_understanding,
            'facts_discovered': len(self.discovered_facts),
            'queries_attempted': len(self.attempted_queries),
            'current_hypothesis': self.current_hypothesis,
            'confidence': self.confidence_evolution[-1] if self.confidence_evolution else 0.0,
            'schema_elements': list(self.discovered_schema.keys())
        }


# ==================== Universal TAOR Agent ====================

class UniversalTAORAgent:
    """Universal Think-Act-Observe-Reflect Agent using GPT-4o"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str, database: str = "neo4j"):

        # Initialize connections
        self.db = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self.llm = OpenAI(api_key=openai_api_key)

        # Memory
        self.working_memory = WorkingMemory()

        # Parameters
        self.max_iterations = 15
        self.confidence_threshold = 0.75

        # Parser
        self.parser = StrictJSONParser()

    def solve(self, question: str) -> Dict:
        """Main entry point - solve a question using TAOR loop"""

        logger.info(f"Starting to solve: {question}")

        # Initialize
        self.working_memory = WorkingMemory()
        self.working_memory.original_question = question
        iteration_count = 0

        # TAOR Loop
        while iteration_count < self.max_iterations:
            iteration_count += 1
            logger.info(f"\n{'='*50}\nIteration {iteration_count}\n{'='*50}")

            try:
                # THINK
                thought = self._think()
                logger.info(f"THOUGHT: {thought.content[:200]}")

                # ACT
                action = self._act(thought)
                logger.info(f"ACTION: {action.action_type.value} - {action.reasoning}")

                # OBSERVE
                observation = self._observe(action)
                logger.info(f"OBSERVATION: Success={observation.success}, Rows={observation.row_count}")

                # REFLECT
                reflection = self._reflect(thought, action, observation)
                logger.info(f"REFLECTION: Continue={reflection.should_continue}, Confidence change={reflection.confidence_change:.2f}")

                # Update confidence
                current_confidence = (self.working_memory.confidence_evolution[-1] if self.working_memory.confidence_evolution else 0.5) + reflection.confidence_change
                current_confidence = max(0, min(1, current_confidence))
                self.working_memory.confidence_evolution.append(current_confidence)

                # Update hypothesis if needed
                if reflection.new_hypothesis:
                    self.working_memory.current_hypothesis = reflection.new_hypothesis

                # Check if we should stop
                if not reflection.should_continue or current_confidence >= self.confidence_threshold:
                    logger.info(f"Stopping: Continue={reflection.should_continue}, Confidence={current_confidence:.2f}")
                    break

            except Exception as e:
                logger.error(f"Error in iteration {iteration_count}: {e}")
                logger.error(traceback.format_exc())

        # Synthesize final answer
        answer = self._synthesize_answer()

        return {
            'question': question,
            'answer': answer,
            'iterations': iteration_count,
            'final_confidence': self.working_memory.confidence_evolution[-1] if self.working_memory.confidence_evolution else 0.0,
            'facts_discovered': len(self.working_memory.discovered_facts),
            'hypothesis': self.working_memory.current_hypothesis
        }

    def _think(self) -> Thought:
        """Think phase - generate thoughts using GPT-4o"""

        context = self.working_memory.get_context_summary()

        prompt = f"""You are in the THINK phase of problem-solving. Generate a thought about how to proceed.

Question: {self.working_memory.original_question}

Current Context:
- Understanding: {json.dumps(context['understanding'], indent=2) if context['understanding'] else 'None yet'}
- Facts discovered: {context['facts_discovered']}
- Queries attempted: {context['queries_attempted']}
- Current hypothesis: {context['current_hypothesis'] or 'None yet'}
- Current confidence: {context['confidence']:.2f}
- Known schema elements: {context['schema_elements'][:10] if context['schema_elements'] else 'None discovered'}

Recent discoveries:
{json.dumps(self.working_memory.discovered_facts[-3:], indent=2) if self.working_memory.discovered_facts else 'None yet'}

Generate a thought about what to do next. Consider:
1. What do we still need to know?
2. What assumptions can we test?
3. What patterns might exist in the data?

Return JSON:
{{
    "thought_type": "understanding|hypothesis|exploration|validation|synthesis",
    "content": "The main thought",
    "reasoning": "Why this thought is relevant",
    "confidence": 0.0-1.0,
    "next_action_suggestion": "What action this thought suggests"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model=ModelConfig.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are a scientific reasoning system. Generate insightful thoughts that advance understanding. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=ModelConfig.CREATIVE_TEMP,
                max_tokens=2000
            )

            return self.parser.parse_thought(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error in _think: {e}")
            return Thought(
                thought_type=ThoughtType.EXPLORATION,
                content="Need to explore the data structure",
                reasoning="Error occurred, falling back to exploration",
                confidence=0.3,
                next_action_suggestion="Explore schema"
            )

    def _act(self, thought: Thought) -> Action:
        """Act phase - convert thought to action using GPT-4o"""

        # Build context of what we know
        schema_summary = json.dumps(self.working_memory.discovered_schema, indent=2) if self.working_memory.discovered_schema else "Not discovered yet"

        # Get recent failed queries to avoid repeating
        recent_failures = [q['query'] for q in self.working_memory.attempted_queries[-3:] if not q['success']]

        prompt = f"""You are in the ACT phase. Convert the thought into a concrete action.

Current Thought: {thought.content}
Thought Reasoning: {thought.reasoning}
Suggested Action: {thought.next_action_suggestion}

Question we're trying to answer: {self.working_memory.original_question}

Known Schema:
{schema_summary}

Recent failed queries to avoid:
{json.dumps(recent_failures, indent=2) if recent_failures else 'None'}

Generate an action. If it involves a database query, write the Cypher query.
Think from first principles - don't use templates or patterns.

Return JSON:
{{
    "action_type": "query|explore|calculate|validate|synthesize",
    "query": "Cypher query if action_type is query/explore/validate, otherwise null",
    "parameters": {{"any": "additional parameters"}},
    "expected_outcome": "What we expect to learn",
    "reasoning": "Why this action will help"
}}"""

        try:
            # Use GPT-4o for query generation
            response = self.llm.chat.completions.create(
                model=ModelConfig.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are a Neo4j expert. Generate actions that will gather relevant information. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=ModelConfig.ANALYTICAL_TEMP,
                max_tokens=2000
            )

            return self.parser.parse_action(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error in _act: {e}")
            return Action(
                action_type=ActionType.EXPLORE,
                query="MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC LIMIT 20",
                parameters={},
                expected_outcome="Discover schema",
                reasoning="Error occurred, exploring schema"
            )

    def _observe(self, action: Action) -> Observation:
        """Observe phase - execute action and analyze results"""

        if action.query:
            # Execute the query
            try:
                with self.db.session(database=self.database) as session:
                    result = session.run(action.query)
                    data = [dict(record) for record in result]

                success = True
                row_count = len(data)

                # Record the attempt
                self.working_memory.add_attempted_query(action.query, success, row_count)

                # Update schema if this is an exploration
                if action.action_type == ActionType.EXPLORE and data:
                    self._update_schema_from_data(data)

            except Exception as e:
                logger.error(f"Query execution error: {e}")
                success = False
                data = []
                row_count = 0
                self.working_memory.add_attempted_query(action.query, success, row_count)
        else:
            # Non-query action
            success = True
            data = []
            row_count = 0

        # Analyze the observation using GPT-4o
        if success and data:
            analysis = self._analyze_observation(data, action)
        else:
            analysis = {
                "relevance": 0.1 if not success else 0.3,
                "surprise": 0.8 if not success else 0.2,
                "insights": ["Query failed" if not success else "No data returned"]
            }

        return Observation(
            success=success,
            data=data[:10],  # Keep sample for memory
            row_count=row_count,
            relevance=analysis["relevance"],
            surprise=analysis["surprise"],
            insights=analysis["insights"]
        )

    def _analyze_observation(self, data: List[Dict], action: Action) -> Dict:
        """Analyze observation using GPT-4o"""

        # Prepare data sample
        data_sample = data[:5] if len(data) > 5 else data

        prompt = f"""Analyze this query result in context of our goal.

Goal: {self.working_memory.original_question}
Action taken: {action.reasoning}
Expected outcome: {action.expected_outcome}

Data received ({len(data)} rows total, showing first {len(data_sample)}):
{json.dumps(data_sample, indent=2, default=str)}

Analyze:
1. How relevant is this data to answering the question? (0-1)
2. How surprising/unexpected is this result? (0-1)
3. What insights can we extract?

Return JSON:
{{
    "relevance": 0.0-1.0,
    "surprise": 0.0-1.0,
    "insights": ["insight 1", "insight 2", ...]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model=ModelConfig.PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are analyzing database query results. Extract meaningful insights. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=ModelConfig.ANALYTICAL_TEMP,
                max_tokens=2000
            )

            return self.parser.parse_observation_analysis(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error analyzing observation: {e}")
            return {"relevance": 0.5, "surprise": 0.5, "insights": ["Analysis failed"]}

    def _reflect(self, thought: Thought, action: Action, observation: Observation) -> Reflection:
        """Reflect phase - learn from the experience using GPT-4-turbo for deep analysis"""

        context = self.working_memory.get_context_summary()

        prompt = f"""You are in the REFLECT phase. Analyze what just happened and decide next steps.

Original Question: {self.working_memory.original_question}

What just happened:
- Thought: {thought.content}
- Action: {action.action_type.value} - {action.reasoning}
- Result: Success={observation.success}, Rows={observation.row_count}, Relevance={observation.relevance:.2f}
- Insights: {observation.insights}

Current Status:
- Facts discovered: {context['facts_discovered']}
- Current confidence: {context['confidence']:.2f}
- Current hypothesis: {context['current_hypothesis'] or 'None'}

Reflect on:
1. What worked well?
2. What didn't work?
3. What lesson can we learn?
4. How should our confidence change? (-1.0 to 1.0)
5. Should we continue or do we have enough to answer?
6. Do we need a new hypothesis?

Return JSON:
{{
    "what_worked": "What was successful",
    "what_failed": "What didn't work",
    "lesson_learned": "Key lesson from this iteration",
    "confidence_change": -1.0 to 1.0,
    "should_continue": true/false,
    "new_hypothesis": "New hypothesis if needed, otherwise null"
}}"""

        try:
            # Use GPT-4-turbo for deeper reflection
            response = self.llm.chat.completions.create(
                model=ModelConfig.DEEP_MODEL,
                messages=[
                    {"role": "system", "content": "You are reflecting on problem-solving progress. Be critical and honest. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                # temperature=ModelConfig.ANALYTICAL_TEMP,
                max_completion_tokens=2000
            )

            reflection = self.parser.parse_reflection(response.choices[0].message.content)

            # Add insights as facts if highly relevant
            if observation.relevance > 0.7:
                for insight in observation.insights:
                    self.working_memory.add_fact(
                        insight,
                        observation.data[:3] if observation.data else None,
                        observation.relevance
                    )

            return reflection

        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return Reflection(
                what_worked="",
                what_failed="Reflection failed",
                lesson_learned="Need to continue exploring",
                confidence_change=0.1 if observation.success else -0.1,
                should_continue=True,
                new_hypothesis=None
            )

    def _update_schema_from_data(self, data: List[Dict]):
        """Update discovered schema from query results"""

        if not data:
            return

        # Extract schema information
        for row in data[:10]:  # Sample first 10 rows
            for key, value in row.items():
                if 'label' in key.lower() and value:
                    if 'nodes' not in self.working_memory.discovered_schema:
                        self.working_memory.discovered_schema['nodes'] = set()
                    self.working_memory.discovered_schema['nodes'].add(value)

                elif 'type' in key.lower() and value:
                    if 'relationships' not in self.working_memory.discovered_schema:
                        self.working_memory.discovered_schema['relationships'] = set()
                    self.working_memory.discovered_schema['relationships'].add(value)

        # Convert sets to lists for JSON serialization
        for key in self.working_memory.discovered_schema:
            if isinstance(self.working_memory.discovered_schema[key], set):
                self.working_memory.discovered_schema[key] = list(self.working_memory.discovered_schema[key])

    def _synthesize_answer(self) -> str:
        """Synthesize final answer using GPT-4-turbo for comprehensive analysis"""

        # Prepare all discovered information
        facts = self.working_memory.discovered_facts
        hypothesis = self.working_memory.current_hypothesis
        confidence = self.working_memory.confidence_evolution[-1] if self.working_memory.confidence_evolution else 0.0

        prompt = f"""Synthesize a final answer to the question based on all discoveries.

Question: {self.working_memory.original_question}

Discovered Facts ({len(facts)} total):
{json.dumps(facts[-10:], indent=2, default=str)}

Final Hypothesis: {hypothesis or 'No formal hypothesis formed'}
Final Confidence: {confidence:.2f}

Queries Attempted: {len(self.working_memory.attempted_queries)}
Successful Queries: {sum(1 for q in self.working_memory.attempted_queries if q['success'])}

Provide a comprehensive answer that:
1. Directly addresses the question
2. Cites the evidence found
3. Acknowledges any limitations or uncertainties
4. Is clear and well-structured

Write the answer in natural language, not JSON."""

        try:
            # Use GPT-4-turbo for comprehensive synthesis
            response = self.llm.chat.completions.create(
                model=ModelConfig.DEEP_MODEL,
                messages=[
                    {"role": "system", "content": "You are synthesizing research findings into a clear answer. Be comprehensive but concise."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=ModelConfig.ANALYTICAL_TEMP,
                max_completion_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in synthesis: {e}")

            # Fallback answer
            if facts:
                answer = f"Based on {len(facts)} discovered facts:\n\n"
                for fact in facts[-5:]:
                    answer += f"- {fact['fact']}\n"
                answer += f"\nConfidence: {confidence:.1%}"
            else:
                answer = "Unable to find sufficient information to answer the question."

            return answer

    def close(self):
        """Clean up resources"""
        self.db.close()


# ==================== Main Usage ====================

def main():
    """Example usage of the Universal TAOR Agent"""
    config = {
        'neo4j_uri': "bolt://100.88.72.32:7687",  # Update with actual
        'neo4j_user': "neo4j",
        'neo4j_password': "neuroxiv",  # Update with actual
        'openai_api_key': "",
        'database': "neo4j"
    }

    # Initialize agent
    agent = UniversalTAORAgent(**config)

    try:
        # Example questions
        questions = [
            "Find region pairs with similar morphological features but different molecular features",
            "Which regions have the highest proportion of Car3 neurons?",
            "What are the main projection patterns in the brain?"
        ]

        for question in questions:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print('='*80)

            # Solve the question
            result = agent.solve(question)

            print(f"\nAnswer:")
            print(result['answer'])
            print(f"\nMetadata:")
            print(f"- Iterations: {result['iterations']}")
            print(f"- Final confidence: {result['final_confidence']:.1%}")
            print(f"- Facts discovered: {result['facts_discovered']}")
            if result['hypothesis']:
                print(f"- Final hypothesis: {result['hypothesis']}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()