"""
Universal NeuroXiv Agent System
A fully autonomous, adaptive agent for exploring and reasoning over knowledge graphs
No hardcoded optimizations - learns and adapts to any query type
"""

import json
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict, deque
import numpy as np
from datetime import datetime
import hashlib

from neo4j import GraphDatabase
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Core Data Structures ====================

@dataclass
class QueryResult:
    """Encapsulates query execution results"""
    query: str
    success: bool
    data: List[Dict]
    execution_time: float
    error: Optional[str] = None
    row_count: int = 0

    def __hash__(self):
        return hash(self.query)


@dataclass
class Hypothesis:
    """Represents a hypothesis about the data"""
    statement: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)

    def update_confidence(self):
        support = len(self.supporting_evidence)
        contradict = len(self.contradicting_evidence)
        if support + contradict > 0:
            self.confidence = support / (support + contradict)


@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_type: str  # 'explore', 'hypothesis', 'verify', 'refine', 'synthesize'
    thought: str
    action: str
    observation: Any
    reflection: str
    timestamp: datetime = field(default_factory=datetime.now)


# ==================== Schema Discovery ====================

class SchemaDiscovery:
    """Dynamically discovers and maintains KG schema"""

    def __init__(self, db_session):
        self.session = db_session
        self.node_types = {}
        self.relationship_types = {}
        self.property_stats = defaultdict(lambda: defaultdict(int))
        self.discovered = False

    def discover_schema(self) -> Dict:
        """Fully discover the KG schema"""
        logger.info("Discovering knowledge graph schema...")

        # Discover node types and counts
        self._discover_nodes()

        # Discover relationships
        self._discover_relationships()

        # Discover properties
        self._discover_properties()

        self.discovered = True
        return self.get_schema_summary()

    def _discover_nodes(self):
        """Discover all node types"""
        query = """
        CALL db.labels() YIELD label
        WITH label
        MATCH (n)
        WHERE label IN labels(n)
        RETURN label, count(n) as count
        """

        result = self._execute(query)
        if result:
            for row in result:
                self.node_types[row['label']] = {
                    'count': row['count'],
                    'properties': set()
                }

    def _discover_relationships(self):
        """Discover all relationship types"""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WITH relationshipType
        MATCH ()-[r]->()
        WHERE type(r) = relationshipType
        WITH relationshipType, r
        LIMIT 100
        RETURN relationshipType, 
               count(r) as count,
               collect(DISTINCT [labels(startNode(r)), labels(endNode(r))]) as patterns
        """

        result = self._execute(query)
        if result:
            for row in result:
                self.relationship_types[row['relationshipType']] = {
                    'count': row['count'],
                    'patterns': row['patterns']
                }

    def _discover_properties(self):
        """Discover properties for each node type"""
        for node_type in self.node_types:
            query = f"""
            MATCH (n:{node_type})
            WITH n LIMIT 100
            UNWIND keys(n) as key
            WITH key, count(n) as frequency
            RETURN key, frequency
            ORDER BY frequency DESC
            """

            result = self._execute(query)
            if result:
                properties = set()
                for row in result:
                    properties.add(row['key'])
                    self.property_stats[node_type][row['key']] = row['frequency']
                self.node_types[node_type]['properties'] = properties

    def _execute(self, query: str) -> Optional[List[Dict]]:
        """Execute query with error handling"""
        try:
            result = self.session.run(query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Schema discovery query failed: {e}")
            return None

    def get_schema_summary(self) -> Dict:
        """Get comprehensive schema summary"""
        return {
            'node_types': self.node_types,
            'relationship_types': self.relationship_types,
            'property_stats': dict(self.property_stats)
        }


# ==================== Query Generation Engine ====================

class AdaptiveQueryGenerator:
    """Generates queries based on discovered schema and learned patterns"""

    def __init__(self, schema: SchemaDiscovery):
        self.schema = schema
        self.successful_patterns = []
        self.failed_patterns = []
        self.query_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict:
        """Initialize basic query templates"""
        return {
            'explore': [
                "MATCH (n:{node_type}) RETURN n LIMIT 10",
                "MATCH (n:{node_type}) WHERE n.{property} IS NOT NULL RETURN n.{property}, count(n) ORDER BY count(n) DESC LIMIT 20",
                "MATCH (n:{node_type}) RETURN keys(n) as properties, count(n) as count LIMIT 1"
            ],
            'relationship': [
                "MATCH (a:{type1})-[r:{rel_type}]->(b:{type2}) RETURN a, r, b LIMIT 20",
                "MATCH (a)-[r:{rel_type}]->(b) RETURN labels(a)[0] as source, labels(b)[0] as target, count(r) as count",
                "MATCH p=(a:{type1})-[*..3]-(b:{type2}) RETURN p LIMIT 10"
            ],
            'aggregate': [
                "MATCH (n:{node_type}) RETURN avg(n.{property}) as average, min(n.{property}) as minimum, max(n.{property}) as maximum",
                "MATCH (n:{node_type}) WITH n.{property} as value, count(n) as frequency RETURN value, frequency ORDER BY frequency DESC LIMIT 20"
            ],
            'compare': [
                "MATCH (a:{type1}), (b:{type2}) WHERE a.{prop1} = b.{prop2} RETURN a, b LIMIT 20",
                "MATCH (a:{node_type}) WHERE a.{prop1} > {value1} AND a.{prop2} < {value2} RETURN a LIMIT 20"
            ]
        }

    def generate_query(self, intent: str, context: Dict) -> str:
        """Generate query based on intent and context"""

        # Clean any markdown formatting from LLM
        intent = self._clean_llm_response(intent)

        # Try to understand query intent
        query_type = self._classify_intent(intent)

        # Generate based on type
        if query_type == 'custom':
            return self._generate_custom_query(intent, context)
        else:
            return self._generate_template_query(query_type, intent, context)

    def _clean_llm_response(self, text: str) -> str:
        """Remove markdown code blocks and clean LLM response"""
        # Remove markdown code blocks
        text = re.sub(r'```(?:cypher|sql|neo4j)?\n?', '', text)
        text = re.sub(r'```\n?', '', text)

        # Remove any leading/trailing whitespace
        text = text.strip()

        # If it starts with common prefixes, remove them
        prefixes = ['Query:', 'Cypher:', 'cypher:', 'CYPHER:']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text

    def _classify_intent(self, intent: str) -> str:
        """Classify the query intent"""
        intent_lower = intent.lower()

        if any(word in intent_lower for word in ['explore', 'show', 'list', 'find all']):
            return 'explore'
        elif any(word in intent_lower for word in ['connect', 'relation', 'link', 'path']):
            return 'relationship'
        elif any(word in intent_lower for word in ['average', 'sum', 'count', 'statistics']):
            return 'aggregate'
        elif any(word in intent_lower for word in ['compare', 'similar', 'different']):
            return 'compare'
        else:
            return 'custom'

    def _generate_template_query(self, query_type: str, intent: str, context: Dict) -> str:
        """Generate query from template"""
        templates = self.query_templates.get(query_type, [])
        if not templates:
            return self._generate_fallback_query()

        # Select template based on context
        template = templates[0]  # Simple selection for now

        # Fill in template with actual schema elements
        node_types = list(self.schema.node_types.keys())
        if node_types:
            template = template.replace('{node_type}', node_types[0])
            template = template.replace('{type1}', node_types[0])
            if len(node_types) > 1:
                template = template.replace('{type2}', node_types[1])

        # Add properties if needed
        if '{property}' in template or '{prop1}' in template:
            for node_type in node_types:
                props = self.schema.node_types[node_type].get('properties', set())
                if props:
                    prop = list(props)[0]
                    template = template.replace('{property}', prop)
                    template = template.replace('{prop1}', prop)
                    if len(props) > 1:
                        template = template.replace('{prop2}', list(props)[1])

        # Add relationships
        rel_types = list(self.schema.relationship_types.keys())
        if rel_types and '{rel_type}' in template:
            template = template.replace('{rel_type}', rel_types[0])

        return template

    def _generate_custom_query(self, intent: str, context: Dict) -> str:
        """Generate custom query based on intent"""

        # Check if intent already looks like a query
        if any(keyword in intent.upper() for keyword in ['MATCH', 'WHERE', 'RETURN']):
            # It's already a query, just clean it
            return self._clean_llm_response(intent)

        # Otherwise, build a simple exploration query
        return self._generate_fallback_query()

    def _generate_fallback_query(self) -> str:
        """Generate a safe fallback query"""
        if self.schema.node_types:
            node_type = list(self.schema.node_types.keys())[0]
            return f"MATCH (n:{node_type}) RETURN n LIMIT 10"
        else:
            return "MATCH (n) RETURN n LIMIT 10"

    def learn_from_result(self, query: str, result: QueryResult):
        """Learn from query execution results"""
        if result.success and result.row_count > 0:
            self.successful_patterns.append(query)
        elif not result.success:
            self.failed_patterns.append(query)


# ==================== Reasoning Engine ====================

class ReasoningEngine:
    """Manages the reasoning process"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.reasoning_chain = []
        self.hypotheses = []
        self.current_confidence = 0.0

    def reason_about_question(self, question: str, context: Dict) -> Dict:
        """Generate reasoning about how to answer the question"""

        prompt = f"""
        Analyze this neuroscience question and plan how to explore the knowledge graph:
        
        Question: {question}
        
        Available node types: {context.get('node_types', [])}
        Available relationships: {context.get('relationship_types', [])}
        
        Provide:
        1. What information we need to find
        2. What queries might help
        3. How to validate findings
        
        Be specific and concise.
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are analyzing how to answer neuroscience questions using a knowledge graph."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        reasoning = response.choices[0].message.content

        step = ReasoningStep(
            step_type='explore',
            thought=reasoning,
            action="Planning exploration",
            observation=None,
            reflection="Initial analysis"
        )
        self.reasoning_chain.append(step)

        return {
            'reasoning': reasoning,
            'next_action': self._extract_next_action(reasoning)
        }

    def _extract_next_action(self, reasoning: str) -> str:
        """Extract actionable next step from reasoning"""
        # Simple extraction - can be enhanced
        lines = reasoning.split('\n')
        for line in lines:
            if 'query' in line.lower() or 'find' in line.lower():
                return line
        return "Explore available data"

    def generate_hypothesis(self, observations: List[Dict]) -> Hypothesis:
        """Generate hypothesis from observations"""

        if not observations:
            return Hypothesis(
                statement="Insufficient data for hypothesis",
                confidence=0.0
            )

        # Format observations for LLM
        obs_text = "\n".join([str(obs) for obs in observations[:5]])

        prompt = f"""
        Based on these observations from the knowledge graph:
        
        {obs_text}
        
        Generate a hypothesis about the pattern or relationship you see.
        Be specific and testable.
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate scientific hypotheses from data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=200
        )

        hypothesis = Hypothesis(
            statement=response.choices[0].message.content,
            confidence=0.5
        )

        self.hypotheses.append(hypothesis)
        return hypothesis

    def verify_hypothesis(self, hypothesis: Hypothesis, new_evidence: Dict) -> Hypothesis:
        """Update hypothesis based on new evidence"""

        prompt = f"""
        Hypothesis: {hypothesis.statement}
        Current confidence: {hypothesis.confidence}
        
        New evidence: {new_evidence}
        
        Does this evidence support or contradict the hypothesis?
        Respond with: SUPPORT, CONTRADICT, or NEUTRAL
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Evaluate scientific evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )

        evaluation = response.choices[0].message.content.strip().upper()

        if 'SUPPORT' in evaluation:
            hypothesis.supporting_evidence.append(str(new_evidence))
        elif 'CONTRADICT' in evaluation:
            hypothesis.contradicting_evidence.append(str(new_evidence))

        hypothesis.update_confidence()
        return hypothesis

    def synthesize_answer(self, question: str, hypotheses: List[Hypothesis],
                         all_observations: List[Dict]) -> str:
        """Synthesize final answer from all findings"""

        # Select best hypotheses
        valid_hypotheses = [h for h in hypotheses if h.confidence > 0.3]

        if not valid_hypotheses and not all_observations:
            return "Unable to find sufficient data to answer this question."

        # Format findings
        findings = []
        if valid_hypotheses:
            findings.append("Key findings:")
            for h in valid_hypotheses:
                findings.append(f"- {h.statement} (confidence: {h.confidence:.1%})")

        if all_observations:
            findings.append("\nSupporting data points:")
            for obs in all_observations[:5]:
                findings.append(f"- {obs}")

        findings_text = "\n".join(findings)

        prompt = f"""
        Question: {question}
        
        Based on the exploration of the knowledge graph:
        
        {findings_text}
        
        Provide a comprehensive answer that:
        1. Directly addresses the question
        2. Cites specific findings
        3. Notes any limitations or uncertainties
        
        Be clear and scientific in your response.
        """

        response = self.llm.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Synthesize neuroscience findings into clear answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        return response.choices[0].message.content


# ==================== Query Execution Manager ====================

class QueryExecutionManager:
    """Manages query execution with caching, retries, and optimization"""

    def __init__(self, db_driver, database: str):
        self.driver = db_driver
        self.database = database
        self.cache = {}
        self.execution_stats = defaultdict(list)

    def execute(self, query: str, retry_on_fail: bool = True) -> QueryResult:
        """Execute query with full error handling"""

        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.cache:
            logger.info("Using cached result")
            return self.cache[query_hash]

        # Clean query
        query = self._clean_query(query)

        # Execute
        start_time = time.time()

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                data = [dict(record) for record in result]

            execution_time = time.time() - start_time

            query_result = QueryResult(
                query=query,
                success=True,
                data=data,
                execution_time=execution_time,
                row_count=len(data)
            )

            # Cache successful results
            self.cache[query_hash] = query_result
            self.execution_stats[query_hash].append(execution_time)

            logger.info(f"Query executed successfully: {len(data)} rows in {execution_time:.2f}s")

            return query_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Query failed: {error_msg}")

            # Try to fix and retry
            if retry_on_fail and 'SyntaxError' in error_msg:
                fixed_query = self._attempt_fix(query, error_msg)
                if fixed_query != query:
                    logger.info("Attempting with fixed query...")
                    return self.execute(fixed_query, retry_on_fail=False)

            return QueryResult(
                query=query,
                success=False,
                data=[],
                execution_time=execution_time,
                error=error_msg
            )

    def _clean_query(self, query: str) -> str:
        """Clean query from LLM artifacts"""
        # Remove markdown code blocks
        query = re.sub(r'```(?:cypher|sql|neo4j)?\n?', '', query)
        query = re.sub(r'```\n?', '', query)

        # Remove common prefixes
        prefixes = ['Query:', 'Cypher:', 'cypher:', 'CYPHER:', 'query:']
        for prefix in prefixes:
            if query.strip().startswith(prefix):
                query = query[len(prefix):].strip()

        # Ensure query ends properly
        query = query.strip()
        if not query:
            query = "MATCH (n) RETURN n LIMIT 1"

        return query

    def _attempt_fix(self, query: str, error: str) -> str:
        """Attempt to fix common query errors"""

        # Fix backticks
        if '`' in error:
            query = query.replace('`', '')

        # Fix quotes
        query = query.replace('"', "'")

        # Ensure LIMIT
        if 'LIMIT' not in query.upper():
            query += ' LIMIT 20'

        return query

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        stats = {}
        for query_hash, times in self.execution_stats.items():
            stats[query_hash] = {
                'executions': len(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return stats


# ==================== Main Agent System ====================

class UniversalNeuroXivAgent:
    """Universal agent for autonomous KG exploration and reasoning"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str, database: str = "neo4j"):

        # Core components
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self.llm = OpenAI(api_key=openai_api_key)

        # Initialize subsystems
        with self.driver.session(database=database) as session:
            self.schema = SchemaDiscovery(session)
            self.schema.discover_schema()

        self.query_gen = AdaptiveQueryGenerator(self.schema)
        self.executor = QueryExecutionManager(self.driver, database)
        self.reasoner = ReasoningEngine(self.llm)

        # Agent state
        self.max_iterations = 20
        self.min_confidence = 0.8
        self.conversation_history = []
        self.working_memory = {
            'question': None,
            'observations': [],
            'hypotheses': [],
            'failed_attempts': [],
            'successful_queries': []
        }

    def solve(self, question: str) -> Dict:
        """Main solving method - fully autonomous"""

        logger.info(f"\n{'='*60}")
        logger.info(f"Solving: {question}")
        logger.info(f"{'='*60}\n")

        self.working_memory['question'] = question

        # Initial reasoning
        schema_context = {
            'node_types': list(self.schema.node_types.keys()),
            'relationship_types': list(self.schema.relationship_types.keys())
        }

        initial_reasoning = self.reasoner.reason_about_question(question, schema_context)

        # Main solving loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")

            # Think: What should we explore next?
            thought = self._think(iteration)

            # Act: Generate and execute query
            query = self._act(thought)

            # Observe: Execute and get results
            observation = self.executor.execute(query)

            # Reflect: Update understanding
            self._reflect(thought, observation)

            # Check if we have enough confidence
            if self._should_stop():
                logger.info("Sufficient confidence reached")
                break

            # Generate/update hypotheses
            if iteration % 3 == 0 and self.working_memory['observations']:
                hypothesis = self.reasoner.generate_hypothesis(
                    self.working_memory['observations']
                )
                self.working_memory['hypotheses'].append(hypothesis)

        # Synthesize final answer
        answer = self._synthesize_answer()

        return {
            'question': question,
            'answer': answer,
            'iterations': iteration,
            'confidence': self.reasoner.current_confidence,
            'queries_executed': len(self.working_memory['successful_queries']),
            'hypotheses_generated': len(self.working_memory['hypotheses'])
        }

    def _think(self, iteration: int) -> str:
        """Generate thought about what to explore next"""

        # Build context
        context = f"""
        Iteration: {iteration}
        Question: {self.working_memory['question']}
        Observations so far: {len(self.working_memory['observations'])}
        Failed attempts: {len(self.working_memory['failed_attempts'])}
        Current hypotheses: {len(self.working_memory['hypotheses'])}
        
        Schema summary:
        - Node types: {list(self.schema.node_types.keys())[:10]}
        - Relationships: {list(self.schema.relationship_types.keys())[:10]}
        """

        if self.working_memory['observations']:
            context += f"\nRecent observations: {self.working_memory['observations'][-2:]}"

        prompt = f"""
        {context}
        
        What specific aspect should we explore next to answer the question?
        Provide a clear, actionable exploration strategy.
        
        If you want to write a specific Cypher query, write it clearly.
        Otherwise, describe what type of data to look for.
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Guide knowledge graph exploration. Be specific and actionable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        thought = response.choices[0].message.content
        logger.info(f"Thought: {thought[:100]}...")

        return thought

    def _act(self, thought: str) -> str:
        """Generate query based on thought"""

        # Check if thought contains a query
        if 'MATCH' in thought.upper():
            # Extract and clean the query
            query = self.query_gen._clean_llm_response(thought)
        else:
            # Generate query based on thought
            query = self.query_gen.generate_query(thought, self.working_memory)

        logger.info(f"Generated query: {query[:100]}...")

        return query

    def _reflect(self, thought: str, observation: QueryResult):
        """Reflect on observation and update state"""

        if observation.success:
            self.working_memory['successful_queries'].append(observation.query)

            if observation.row_count > 0:
                # Store meaningful observations
                summary = self._summarize_results(observation.data)
                self.working_memory['observations'].append(summary)

                # Learn from success
                self.query_gen.learn_from_result(observation.query, observation)

                # Update confidence
                self.reasoner.current_confidence += 0.1

                logger.info(f"Found {observation.row_count} results. Confidence: {self.reasoner.current_confidence:.1%}")
            else:
                logger.info("Query returned no results")
        else:
            self.working_memory['failed_attempts'].append({
                'thought': thought,
                'query': observation.query,
                'error': observation.error
            })

            # Learn from failure
            self.query_gen.learn_from_result(observation.query, observation)

            logger.warning(f"Query failed: {observation.error[:100]}")

    def _summarize_results(self, data: List[Dict]) -> Dict:
        """Summarize query results"""

        if not data:
            return {}

        summary = {
            'row_count': len(data),
            'sample': data[:3] if len(data) > 3 else data
        }

        # Add statistics if numerical data
        for key in data[0].keys():
            values = [row.get(key) for row in data if row.get(key) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                summary[f'{key}_stats'] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values)
                }

        return summary

    def _should_stop(self) -> bool:
        """Determine if we should stop exploring"""

        # Stop if high confidence
        if self.reasoner.current_confidence >= self.min_confidence:
            return True

        # Stop if we have substantial observations
        if len(self.working_memory['observations']) > 15:
            return True

        # Stop if too many failures
        if len(self.working_memory['failed_attempts']) > 10:
            logger.warning("Too many failed attempts, stopping")
            return True

        return False

    def _synthesize_answer(self) -> str:
        """Synthesize final answer from all findings"""

        return self.reasoner.synthesize_answer(
            self.working_memory['question'],
            self.working_memory['hypotheses'],
            self.working_memory['observations']
        )

    def close(self):
        """Clean up resources"""
        self.driver.close()


# ==================== Advanced Features ====================

class LearningModule:
    """Learns from past queries to improve future performance"""

    def __init__(self):
        self.query_patterns = defaultdict(list)
        self.success_patterns = defaultdict(float)
        self.concept_mappings = {}

    def learn_from_session(self, question: str, queries: List[str], success_rate: float):
        """Learn patterns from a Q&A session"""

        # Extract concepts from question
        concepts = self._extract_concepts(question)

        for concept in concepts:
            self.query_patterns[concept].extend(queries)
            self.success_patterns[concept] = (
                self.success_patterns[concept] + success_rate
            ) / 2

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""

        # Simple keyword extraction
        keywords = []
        important_terms = [
            'morpholog', 'molecular', 'projection', 'connect',
            'neurotransmitter', 'dendrit', 'axon', 'region',
            'pathway', 'similar', 'different', 'compare'
        ]

        text_lower = text.lower()
        for term in important_terms:
            if term in text_lower:
                keywords.append(term)

        return keywords

    def suggest_queries(self, question: str) -> List[str]:
        """Suggest queries based on learned patterns"""

        concepts = self._extract_concepts(question)
        suggestions = []

        for concept in concepts:
            if concept in self.query_patterns:
                # Get successful queries for this concept
                patterns = self.query_patterns[concept]
                if patterns:
                    suggestions.extend(patterns[:3])

        return suggestions


class MetaReasoningModule:
    """Reasons about the reasoning process itself"""

    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.current_strategy = 'exploratory'

    def select_strategy(self, context: Dict) -> str:
        """Select reasoning strategy based on context"""

        observations = context.get('observations', [])
        failed_attempts = context.get('failed_attempts', [])

        # Adaptive strategy selection
        if len(failed_attempts) > len(observations):
            return 'conservative'  # Stick to simple queries
        elif len(observations) < 3:
            return 'exploratory'  # Broad exploration
        elif len(observations) < 10:
            return 'focused'  # Focus on promising areas
        else:
            return 'synthesis'  # Focus on synthesis

    def evaluate_progress(self, context: Dict) -> Dict:
        """Evaluate reasoning progress"""

        return {
            'exploration_coverage': self._calculate_coverage(context),
            'hypothesis_quality': self._evaluate_hypotheses(context),
            'efficiency': self._calculate_efficiency(context)
        }

    def _calculate_coverage(self, context: Dict) -> float:
        """Calculate how well we've explored the space"""

        # Simple metric based on variety of queries
        queries = context.get('successful_queries', [])
        if not queries:
            return 0.0

        unique_patterns = set()
        for query in queries:
            # Extract query pattern
            pattern = re.sub(r'\b\w+\b', 'X', query)
            unique_patterns.add(pattern)

        return min(len(unique_patterns) / 10, 1.0)

    def _evaluate_hypotheses(self, context: Dict) -> float:
        """Evaluate hypothesis quality"""

        hypotheses = context.get('hypotheses', [])
        if not hypotheses:
            return 0.0

        # Average confidence of hypotheses
        confidences = [h.confidence for h in hypotheses if hasattr(h, 'confidence')]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_efficiency(self, context: Dict) -> float:
        """Calculate query efficiency"""

        successful = len(context.get('successful_queries', []))
        failed = len(context.get('failed_attempts', []))

        total = successful + failed
        if total == 0:
            return 1.0

        return successful / total


# ==================== Enhanced Main Agent ====================

class EnhancedUniversalAgent(UniversalNeuroXivAgent):
    """Enhanced version with learning and meta-reasoning"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add advanced modules
        self.learner = LearningModule()
        self.meta_reasoner = MetaReasoningModule()

        # Enhanced state
        self.session_history = []

    def solve(self, question: str) -> Dict:
        """Enhanced solving with learning"""

        # Check if we have learned patterns for this question
        suggested_queries = self.learner.suggest_queries(question)
        if suggested_queries:
            logger.info(f"Found {len(suggested_queries)} learned patterns")
            # Try learned patterns first
            for query in suggested_queries[:2]:
                result = self.executor.execute(query)
                if result.success and result.row_count > 0:
                    self.working_memory['observations'].append(
                        self._summarize_results(result.data)
                    )

        # Select strategy
        strategy = self.meta_reasoner.select_strategy(self.working_memory)
        logger.info(f"Using {strategy} strategy")

        # Execute main solving loop
        result = super().solve(question)

        # Learn from this session
        success_rate = len(self.working_memory['successful_queries']) / max(
            len(self.working_memory['successful_queries']) +
            len(self.working_memory['failed_attempts']), 1
        )

        self.learner.learn_from_session(
            question,
            self.working_memory['successful_queries'],
            success_rate
        )

        # Evaluate performance
        performance = self.meta_reasoner.evaluate_progress(self.working_memory)
        result['performance_metrics'] = performance

        # Store in history
        self.session_history.append({
            'question': question,
            'result': result,
            'performance': performance
        })

        return result

    def get_learning_summary(self) -> Dict:
        """Get summary of what the agent has learned"""

        return {
            'sessions_completed': len(self.session_history),
            'concepts_learned': list(self.learner.concept_mappings.keys()),
            'avg_success_rate': np.mean([
                s['performance']['efficiency']
                for s in self.session_history
            ]) if self.session_history else 0.0,
            'best_strategies': dict(self.meta_reasoner.strategy_performance)
        }


# ==================== Usage Example ====================

def main():
    """Example usage of the universal agent"""

    # Configuration
    config = {
        'neo4j_uri': "bolt://100.88.72.32:7687",  # Update with your URI
        'neo4j_user': "neo4j",
        'neo4j_password': "neuroxiv",  # Update with your password
        'openai_api_key': "",
        'database': "neo4j"  # or "neuroxiv" depending on your setup
    }

    # Create enhanced agent
    agent = EnhancedUniversalAgent(**config)

    try:
        # Example questions - agent will adapt to any type
        questions = [
            "Find region pairs with similar morphological features but different molecular features",
            "What are the main projection pathways from visual cortex?",
            "Which regions have the highest dendritic complexity?",
            "Identify neurotransmitter distribution patterns across brain regions"
        ]

        for question in questions:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print('='*80)

            result = agent.solve(question)

            print(f"\nAnswer: {result['answer']}")
            print(f"\nMetrics:")
            print(f"  - Iterations: {result['iterations']}")
            print(f"  - Queries executed: {result['queries_executed']}")
            print(f"  - Confidence: {result['confidence']:.1%}")

            if 'performance_metrics' in result:
                print(f"  - Exploration coverage: {result['performance_metrics']['exploration_coverage']:.1%}")
                print(f"  - Hypothesis quality: {result['performance_metrics']['hypothesis_quality']:.1%}")
                print(f"  - Query efficiency: {result['performance_metrics']['efficiency']:.1%}")

        # Show learning summary
        print(f"\n{'='*80}")
        print("Learning Summary")
        print('='*80)
        summary = agent.get_learning_summary()
        print(f"Sessions completed: {summary['sessions_completed']}")
        print(f"Average success rate: {summary['avg_success_rate']:.1%}")
        print(f"Concepts learned: {summary['concepts_learned']}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()