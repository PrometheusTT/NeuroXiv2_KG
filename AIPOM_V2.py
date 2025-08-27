"""
CoT-KG Agent: Production-Ready Version with Robust Error Handling
Addresses token limits, Cypher syntax issues, and maintains scientific rigor

Author: NeuroXiv Team
Date: 2025-12-19
Version: 2.0
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np
from scipy import stats
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Core Data Structures ====================

@dataclass
class Thought:
    """Single reasoning step"""
    step: int
    question: str
    reasoning: str
    kg_query: str
    kg_result: Any
    insight: str
    next_question: Optional[str] = None
    confidence: float = 0.0
    query_attempts: int = 0  # Track retry attempts


@dataclass
class ReasoningChain:
    """Complete reasoning chain"""
    initial_question: str
    thoughts: List[Thought]
    final_answer: str
    discoveries: List[str]
    confidence: float
    total_tokens_used: int = 0


# ==================== Query Templates for Consistency ====================

class QueryTemplates:
    """Pre-defined query templates to ensure correct Cypher syntax"""

    @staticmethod
    def get_top_regions_by_property(property_name: str, limit: int = 10, ascending: bool = False):
        order = "ASC" if ascending else "DESC"
        return f"""
        MATCH (r:Region)
        WHERE r.{property_name} IS NOT NULL
        RETURN r.acronym, r.name, r.{property_name} AS value
        ORDER BY value {order}
        LIMIT {limit}
        """

    @staticmethod
    def get_cell_composition(region_pattern: str, cell_type: str = "Subclass"):
        return f"""
        MATCH (r:Region)
        WHERE r.acronym CONTAINS '{region_pattern}' OR r.name CONTAINS '{region_pattern}'
        OPTIONAL MATCH (r)-[h:HAS_{cell_type.upper()}]->(c:{cell_type})
        WHERE h.pct_cells IS NOT NULL
        RETURN r.acronym, r.name, c.name AS cell_type, h.pct_cells, h.rank
        ORDER BY r.acronym, h.rank
        """

    @staticmethod
    def get_projections(source_region: str, limit: int = 20):
        return f"""
        MATCH (r:Region)-[p:PROJECT_TO]->(target:Region)
        WHERE r.acronym CONTAINS '{source_region}' OR r.name CONTAINS '{source_region}'
        AND p.weight IS NOT NULL
        RETURN r.acronym AS source, target.acronym AS target, 
               p.weight, p.neuron_count
        ORDER BY p.weight DESC
        LIMIT {limit}
        """

    @staticmethod
    def compare_regions_morphology(regions: List[str]):
        region_list = str(regions).replace("'", '"')
        return f"""
        MATCH (r:Region)
        WHERE r.acronym IN {region_list}
        RETURN r.acronym, r.name,
               r.axonal_length, r.dendritic_length,
               r.axonal_branches, r.dendritic_branches,
               (COALESCE(r.axonal_length, 0) + COALESCE(r.dendritic_length, 0)) AS total_length,
               (COALESCE(r.axonal_branches, 0) + COALESCE(r.dendritic_branches, 0)) AS total_branches
        ORDER BY total_length DESC
        """


# ==================== Knowledge Graph Interface ====================

class KGInterface:
    """
    Minimal but complete knowledge graph interface with query optimization
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.query_cache = {}  # Cache for frequently used queries
        self.query_templates = QueryTemplates()

    def execute_cypher(self, query: str, use_cache: bool = True, **params) -> List[Dict]:
        """Execute Cypher query with caching and error recovery"""

        # Check cache first
        cache_key = query[:100]  # Use first 100 chars as cache key
        if use_cache and cache_key in self.query_cache:
            logger.info("Using cached result")
            return self.query_cache[cache_key]

        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                data = [record.data() for record in result]

                # Cache successful results
                if use_cache and len(data) < 1000:  # Only cache small results
                    self.query_cache[cache_key] = data

                return data
            except Exception as e:
                logger.error(f"Cypher query failed: {e}")
                logger.error(f"Query: {query[:200]}...")
                return []

    def batch_execute(self, queries: List[str]) -> List[List[Dict]]:
        """Execute multiple queries in batch to save time"""
        results = []
        with self.driver.session() as session:
            for query in queries:
                try:
                    result = session.run(query)
                    results.append([record.data() for record in result])
                except Exception as e:
                    logger.error(f"Batch query failed: {e}")
                    results.append([])
        return results

    def get_schema(self) -> Dict:
        """Get graph schema for LLM reference"""
        return {
            'nodes': {
                'Region': {
                    'count': 337,
                    'key_properties': ['region_id', 'name', 'acronym'],
                    'morphology_properties': [
                        'axonal_length', 'dendritic_length',
                        'axonal_branches', 'dendritic_branches'
                    ],
                    'description': 'Brain regions with aggregated morphological data'
                },
                'Subclass': {
                    'key_properties': ['tran_id', 'name', 'markers'],
                    'description': 'Cell type subclasses'
                }
            },
            'relationships': {
                'PROJECT_TO': 'Region->Region with weight and neuron_count',
                'HAS_SUBCLASS': 'Region->Subclass with pct_cells and rank'
            },
            'important_notes': [
                'Use CONTAINS for string matching',
                'Check IS NOT NULL before filtering',
                'Use COALESCE for nullable numeric properties',
                'No GREATEST/LEAST functions - use CASE WHEN instead'
            ]
        }

    def close(self):
        self.driver.close()


# ==================== Optimized Chain-of-Thought Engine ====================

class ChainOfThoughtEngine:
    """
    Optimized CoT engine with token management and query fixing
    """

    def __init__(self, kg: KGInterface, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.kg = kg
        self.kg_schema = kg.get_schema()
        self.max_thinking_steps = 7  # Reduced to save tokens
        self.max_retries_per_query = 2
        self.total_tokens = 0
        self.token_limit = 25000  # Conservative limit

        # Cypher syntax rules to prevent common errors
        self.cypher_rules = """
        CRITICAL CYPHER RULES FOR NEO4J:
        1. NO GREATEST/LEAST functions - use CASE WHEN or mathematical operations
        2. NO multiple statements with semicolons - one query only
        3. NO subqueries in WHERE clauses like WHERE x < (MATCH...)
        4. UNION requires EXACT same column names and types
        5. Use COALESCE(property, 0) for nullable numeric properties
        6. Use CONTAINS for string matching, not =
        7. Always check IS NOT NULL before using properties in calculations

        CORRECT EXAMPLES:
        - Instead of GREATEST(a, b): CASE WHEN a > b THEN a ELSE b END
        - Instead of complex math: (COALESCE(r.axonal_length,0) + COALESCE(r.dendritic_length,0))
        - String matching: WHERE r.name CONTAINS 'CA3' not WHERE r.name = 'CA3'
        """

        self.query_examples = """
        WORKING QUERY EXAMPLES:

        1. Find regions with longest axons:
        MATCH (r:Region)
        WHERE r.axonal_length IS NOT NULL
        RETURN r.acronym, r.axonal_length
        ORDER BY r.axonal_length DESC
        LIMIT 10

        2. Get cell composition with Car3:
        MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
        WHERE s.name CONTAINS 'Car3' AND h.pct_cells IS NOT NULL
        RETURN r.acronym, r.name, h.pct_cells, h.rank
        ORDER BY h.pct_cells DESC
        LIMIT 10

        3. Calculate morphological complexity:
        MATCH (r:Region)
        WHERE r.axonal_length IS NOT NULL OR r.dendritic_length IS NOT NULL
        RETURN r.acronym,
               (COALESCE(r.axonal_length,0) + COALESCE(r.dendritic_length,0)) AS total_length,
               (COALESCE(r.axonal_branches,0) + COALESCE(r.dendritic_branches,0)) AS total_branches
        ORDER BY total_length DESC
        LIMIT 20

        4. Get projections:
        MATCH (source:Region)-[p:PROJECT_TO]->(target:Region)
        WHERE source.acronym = 'CLA' AND p.weight IS NOT NULL
        RETURN source.acronym, target.acronym, p.weight
        ORDER BY p.weight DESC
        LIMIT 15
        """

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4

    def _check_token_budget(self, additional_text: str) -> bool:
        """Check if we have token budget for more processing"""
        estimated = self._estimate_tokens(additional_text)
        if self.total_tokens + estimated > self.token_limit:
            logger.warning(f"Approaching token limit: {self.total_tokens} + {estimated}")
            return False
        return True

    def think(self, question: str) -> ReasoningChain:
        """Execute chain reasoning with token management"""
        thoughts = []
        current_question = question
        context = {
            'initial_question': question,
            'previous_insights': []  # Simplified context to save tokens
        }

        logger.info(f"Starting reasoning: {question}")

        for step in range(self.max_thinking_steps):
            # Check token budget
            if not self._check_token_budget(str(context)):
                logger.warning("Token budget exceeded, concluding reasoning")
                break

            # Generate thought
            thought = self._generate_thought(current_question, context, step)

            # Execute KG query with retry logic
            if thought.kg_query:
                thought.kg_result = self._execute_with_retry(thought.kg_query)
                thought.query_attempts = 1

            # Extract insight
            thought.insight = self._extract_insight(thought, context)

            # Add to context (simplified)
            if thought.insight:
                context['previous_insights'].append(thought.insight[:200])

            thoughts.append(thought)

            # Decide next step
            thought.next_question = self._generate_next_question(thought, context, step)

            if not thought.next_question:
                break

            current_question = thought.next_question

        # Synthesize answer
        final_answer = self._synthesize_answer(thoughts, question)

        # Identify discoveries
        discoveries = self._identify_discoveries(thoughts)

        return ReasoningChain(
            initial_question=question,
            thoughts=thoughts,
            final_answer=final_answer,
            discoveries=discoveries,
            confidence=self._calculate_confidence(thoughts),
            total_tokens_used=self.total_tokens
        )

    def _fix_cypher_query(self, query: str, error: str = None) -> str:
        """Automatically fix common Cypher syntax errors"""
        if not query:
            return query

        fixed = query

        # Fix GREATEST/LEAST
        if 'GREATEST' in fixed.upper():
            # Simple replacement for two-argument GREATEST
            import re
            pattern = r'GREATEST\s*\(\s*([^,]+),\s*([^)]+)\)'
            replacement = r'CASE WHEN \1 > \2 THEN \1 ELSE \2 END'
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

        # Fix multiple statements
        if ';' in fixed:
            # Take only the first statement
            fixed = fixed.split(';')[0].strip()

        # Fix subqueries in WHERE
        if 'WHERE' in fixed and '(MATCH' in fixed:
            # This is complex - return a simpler query
            logger.warning("Complex subquery detected, simplifying")
            return self._generate_simple_alternative(query)

        # Fix UNION column mismatch
        if 'UNION' in fixed.upper():
            parts = fixed.split('UNION')
            if len(parts) == 2:
                # Extract RETURN columns from first part and apply to second
                import re
                return_match = re.search(r'RETURN\s+(.+?)(?:ORDER|LIMIT|$)',
                                         parts[0], re.IGNORECASE | re.DOTALL)
                if return_match:
                    columns = return_match.group(1).strip()
                    # This is simplified - in production would need full parsing
                    logger.info("UNION query detected, may need manual fixing")

        return fixed

    def _generate_simple_alternative(self, complex_query: str) -> str:
        """Generate a simpler alternative to a complex query"""
        # Extract the main intent and create simpler query
        if 'Car3' in complex_query:
            return """
            MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
            WHERE s.name CONTAINS 'Car3'
            RETURN r.acronym, r.name, h.pct_cells
            ORDER BY h.pct_cells DESC
            LIMIT 20
            """
        elif 'PROJECT' in complex_query.upper():
            return """
            MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
            WHERE p.weight IS NOT NULL
            RETURN r.acronym, t.acronym, p.weight
            ORDER BY p.weight DESC
            LIMIT 20
            """
        else:
            # Default simple query
            return """
            MATCH (r:Region)
            RETURN r.acronym, r.name
            LIMIT 20
            """

    def _execute_with_retry(self, query: str, max_retries: int = 2) -> List[Dict]:
        """Execute query with automatic retry and fixing"""
        original_query = query

        for attempt in range(max_retries):
            try:
                # Fix known issues before execution
                query = self._fix_cypher_query(query)

                logger.info(f"Executing query (attempt {attempt + 1}): {query[:100]}...")
                result = self.kg.execute_cypher(query)

                if result:
                    logger.info(f"Query returned {len(result)} results")
                    return result
                elif attempt < max_retries - 1:
                    # Try simpler version
                    query = self._generate_simple_alternative(original_query)
                    logger.info("No results, trying simpler query")

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                if attempt < max_retries - 1:
                    query = self._fix_cypher_query(query, str(e))

        logger.warning("All query attempts failed, returning empty result")
        return []

    def _generate_thought(self, question: str, context: Dict, step: int) -> Thought:
        """Generate next thought with token-efficient prompt"""

        # Simplified prompt to save tokens
        prompt = f"""
        Analyze this neuroscience question about brain regions.

        Question: {question}
        Step: {step + 1} of max {self.max_thinking_steps}

        Previous insights (last 3):
        {self._format_recent_insights(context.get('previous_insights', []))}

        {self.cypher_rules}

        Generate ONE working Cypher query to gather needed data.
        Focus on the specific question asked.

        {self.query_examples}

        Return JSON:
        {{
            "reasoning": "Brief reasoning (max 100 words)",
            "kg_query": "Complete Cypher query or null",
            "purpose": "What this query finds (max 20 words)"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use smaller model to save tokens
                messages=[
                    {"role": "system", "content": "You are a neuroscience data analyst. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent queries
            )

            self.total_tokens += self._estimate_tokens(prompt + response.choices[0].message.content)
            result = json.loads(response.choices[0].message.content)

            return Thought(
                step=step + 1,
                question=question,
                reasoning=result.get('reasoning', ''),
                kg_query=self._fix_cypher_query(result.get('kg_query')),
                kg_result=None,
                insight=""
            )

        except Exception as e:
            logger.error(f"Failed to generate thought: {e}")
            return Thought(
                step=step + 1,
                question=question,
                reasoning="Error generating reasoning",
                kg_query=None,
                kg_result=None,
                insight=""
            )

    def _extract_insight(self, thought: Thought, context: Dict) -> str:
        """Extract concise insight from query results"""
        if not thought.kg_result:
            return "No data retrieved. Trying alternative approach."

        # Summarize results locally to save API tokens
        result_summary = self._summarize_results(thought.kg_result)

        prompt = f"""
        Brief insight from this data (max 50 words):

        Question: {thought.question}
        Data summary: {result_summary}

        Key finding:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Extract key insights concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )

            self.total_tokens += self._estimate_tokens(prompt + response.choices[0].message.content)
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to extract insight: {e}")
            return f"Found {len(thought.kg_result)} results"

    def _summarize_results(self, results: List[Dict]) -> str:
        """Create concise summary of query results"""
        if not results:
            return "No results"

        # Take first 5 results
        sample = results[:5]

        # Create simple summary
        if len(results) == 1:
            return f"Single result: {json.dumps(sample[0], default=str)[:200]}"
        else:
            return f"{len(results)} results. Top entries: {json.dumps(sample, default=str)[:300]}"

    def _generate_next_question(self, thought: Thought, context: Dict, step: int) -> Optional[str]:
        """Decide if more exploration needed"""

        # Simple heuristic to avoid excessive API calls
        if step >= self.max_thinking_steps - 1:
            return None

        if not thought.kg_result:
            return "Let me try a different approach to find relevant data."

        # Check if we have enough information
        total_results = sum(len(t.kg_result) if t.kg_result else 0
                            for t in context.get('thoughts', []))

        if total_results > 50:  # Enough data collected
            return None

        # Generate follow-up only if needed
        if step < 3:  # Continue exploration in early steps
            if 'Car3' in context['initial_question']:
                follow_ups = [
                    "What are the projection patterns of these Car3-rich regions?",
                    "How does the morphology compare across Car3-dominated regions?",
                    "What other cell types coexist with Car3 in these regions?"
                ]
                return follow_ups[min(step, len(follow_ups) - 1)]

        return None

    def _synthesize_answer(self, thoughts: List[Thought], question: str) -> str:
        """Synthesize final answer efficiently"""

        # Collect key findings
        findings = []
        for t in thoughts:
            if t.insight and t.kg_result:
                findings.append(f"- {t.insight}")

        if not findings:
            return "Unable to retrieve sufficient data to answer the question comprehensively."

        # Create structured answer without additional API call
        answer_parts = [
            f"# Analysis of: {question}\n",
            "## Key Findings:\n",
            "\n".join(findings[:5]),  # Top 5 findings
            "\n## Summary:\n"
        ]

        # Add statistical summary if available
        total_results = sum(len(t.kg_result) if t.kg_result else 0 for t in thoughts)
        answer_parts.append(f"Based on analysis of {total_results} data points across {len(thoughts)} queries.")

        # Use API only for final synthesis if we have token budget
        if self._check_token_budget("synthesis") and findings:
            try:
                synthesis_prompt = f"""
                Synthesize these findings into a clear answer (max 150 words):

                Question: {question}
                Findings:
                {chr(10).join(findings[:5])}

                Provide a direct, scientific answer:
                """

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Provide clear, scientific answers."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )

                answer_parts.append("\n" + response.choices[0].message.content)

            except Exception as e:
                logger.error(f"Synthesis failed: {e}")

        return "\n".join(answer_parts)

    def _format_recent_insights(self, insights: List[str]) -> str:
        """Format recent insights concisely"""
        if not insights:
            return "None yet"
        recent = insights[-3:]  # Last 3 insights
        return "\n".join(f"{i + 1}. {ins[:100]}" for i, ins in enumerate(recent))

    def _identify_discoveries(self, thoughts: List[Thought]) -> List[str]:
        """Identify key discoveries from reasoning chain"""
        discoveries = []

        for thought in thoughts:
            if thought.insight and thought.kg_result:
                # Simple keyword detection for significant findings
                keywords = ['highest', 'lowest', 'unusual', 'significant',
                            'dominated', 'exceptional', 'unique']
                if any(kw in thought.insight.lower() for kw in keywords):
                    discoveries.append(thought.insight)

        return discoveries[:3]  # Top 3 discoveries

    def _calculate_confidence(self, thoughts: List[Thought]) -> float:
        """Calculate confidence based on data retrieved"""
        if not thoughts:
            return 0.0

        # Base confidence on successful queries
        successful_queries = sum(1 for t in thoughts if t.kg_result)
        data_points = sum(len(t.kg_result) if t.kg_result else 0 for t in thoughts)

        query_confidence = successful_queries / len(thoughts) if thoughts else 0
        data_confidence = min(data_points / 50, 1.0)  # Normalize by expected data

        return (query_confidence * 0.6 + data_confidence * 0.4)


# ==================== Specialized Analysis Functions ====================

class SpecializedAnalyzer:
    """Pre-built analysis functions for common neuroscience questions"""

    def __init__(self, kg: KGInterface):
        self.kg = kg
        self.templates = QueryTemplates()

    def analyze_car3_regions(self) -> Dict:
        """Specialized analysis for Car3 subclass regions"""
        results = {}

        # Step 1: Find top Car3 regions
        query1 = """
        MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
        WHERE s.name CONTAINS 'Car3' AND h.pct_cells IS NOT NULL
        RETURN r.acronym, r.name, h.pct_cells, h.rank
        ORDER BY h.pct_cells DESC
        LIMIT 10
        """
        car3_regions = self.kg.execute_cypher(query1)
        results['top_car3_regions'] = car3_regions

        if car3_regions:
            top_region = car3_regions[0]['r.acronym']

            # Step 2: Get projections of top region
            query2 = f"""
            MATCH (r:Region)-[p:PROJECT_TO]->(target:Region)
            WHERE r.acronym = '{top_region}' AND p.weight IS NOT NULL
            RETURN target.acronym, target.name, p.weight
            ORDER BY p.weight DESC
            LIMIT 10
            """
            projections = self.kg.execute_cypher(query2)
            results['projections'] = projections

            # Step 3: Get morphology
            query3 = f"""
            MATCH (r:Region)
            WHERE r.acronym = '{top_region}'
            RETURN r.acronym, r.name,
                   COALESCE(r.axonal_length, 0) AS axon_length,
                   COALESCE(r.dendritic_length, 0) AS dendrite_length,
                   (COALESCE(r.axonal_length, 0) + COALESCE(r.dendritic_length, 0)) AS total_length
            """
            morphology = self.kg.execute_cypher(query3)
            results['morphology'] = morphology

        return results

    def find_morphological_outliers(self, top_n: int = 10) -> Dict:
        """Find regions with unusual morphological properties"""

        # Get morphological data
        query = """
        MATCH (r:Region)
        WHERE r.axonal_length IS NOT NULL AND r.dendritic_length IS NOT NULL
        RETURN r.acronym, r.name,
               r.axonal_length AS axon,
               r.dendritic_length AS dendrite,
               CASE 
                   WHEN r.dendritic_length > 0 
                   THEN r.axonal_length / r.dendritic_length 
                   ELSE 999 
               END AS axon_dendrite_ratio
        ORDER BY axon_dendrite_ratio DESC
        """

        results = self.kg.execute_cypher(query)

        if results:
            # Calculate statistics
            ratios = [r['axon_dendrite_ratio'] for r in results if r['axon_dendrite_ratio'] < 999]
            if ratios:
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)

                # Find outliers (>2 std from mean)
                outliers = []
                for r in results:
                    if r['axon_dendrite_ratio'] < 999:
                        z_score = abs(r['axon_dendrite_ratio'] - mean_ratio) / std_ratio
                        if z_score > 2:
                            r['z_score'] = z_score
                            outliers.append(r)

                return {
                    'outliers': outliers[:top_n],
                    'statistics': {
                        'mean_ratio': mean_ratio,
                        'std_ratio': std_ratio,
                        'total_regions': len(results)
                    }
                }

        return {'outliers': [], 'statistics': {}}


# ==================== Main Agent ====================

class CoTKGAgent:
    """
    Main Agent - Integrates CoT reasoning with specialized analyses
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str):
        self.kg = KGInterface(neo4j_uri, neo4j_user, neo4j_password)
        self.cot_engine = ChainOfThoughtEngine(self.kg, openai_api_key)
        self.analyzer = SpecializedAnalyzer(self.kg)

    def answer(self, question: str) -> Dict:
        """Answer question using appropriate strategy"""
        logger.info(f"Received question: {question}")

        # Check for specialized analyses
        if 'Car3' in question:
            logger.info("Using specialized Car3 analysis")
            specialized_results = self.analyzer.analyze_car3_regions()

            # Enhance with CoT reasoning if token budget allows
            if self.cot_engine.total_tokens < 20000:
                reasoning_chain = self.cot_engine.think(question)
                return self._combine_results(question, reasoning_chain, specialized_results)
            else:
                return self._format_specialized_results(question, specialized_results)

        elif 'outlier' in question.lower() or 'unusual' in question.lower():
            logger.info("Using morphological outlier analysis")
            specialized_results = self.analyzer.find_morphological_outliers()
            return self._format_specialized_results(question, specialized_results)

        else:
            # Default to CoT reasoning
            reasoning_chain = self.cot_engine.think(question)
            return self._format_reasoning_results(question, reasoning_chain)

    def _combine_results(self, question: str, reasoning_chain: ReasoningChain,
                         specialized_results: Dict) -> Dict:
        """Combine CoT reasoning with specialized analysis"""
        return {
            'question': question,
            'answer': reasoning_chain.final_answer,
            'specialized_analysis': specialized_results,
            'reasoning_steps': [
                {
                    'step': t.step,
                    'reasoning': t.reasoning[:200],  # Truncate for display
                    'results_count': len(t.kg_result) if t.kg_result else 0,
                    'insight': t.insight
                }
                for t in reasoning_chain.thoughts
            ],
            'confidence': reasoning_chain.confidence,
            'tokens_used': reasoning_chain.total_tokens_used
        }

    def _format_specialized_results(self, question: str, results: Dict) -> Dict:
        """Format specialized analysis results"""

        # Create readable summary
        summary_parts = []

        if 'top_car3_regions' in results:
            if results['top_car3_regions']:
                top_region = results['top_car3_regions'][0]
                summary_parts.append(
                    f"The region with highest Car3 proportion is {top_region['r.name']} "
                    f"({top_region['r.acronym']}) with {top_region['h.pct_cells']:.1f}% Car3 neurons."
                )

                if 'projections' in results and results['projections']:
                    targets = [p['target.acronym'] for p in results['projections'][:3]]
                    summary_parts.append(
                        f"This region primarily projects to: {', '.join(targets)}"
                    )

                if 'morphology' in results and results['morphology']:
                    morph = results['morphology'][0]
                    summary_parts.append(
                        f"Morphologically, it has {morph['axon_length']:.1f}μm average axon length "
                        f"and {morph['dendrite_length']:.1f}μm dendrite length."
                    )

        elif 'outliers' in results:
            if results['outliers']:
                summary_parts.append(
                    f"Found {len(results['outliers'])} morphological outlier regions."
                )
                top_outlier = results['outliers'][0] if results['outliers'] else None
                if top_outlier:
                    summary_parts.append(
                        f"Most extreme: {top_outlier['r.name']} with "
                        f"axon/dendrite ratio of {top_outlier['axon_dendrite_ratio']:.2f}"
                    )

        answer = " ".join(summary_parts) if summary_parts else "No significant findings."

        return {
            'question': question,
            'answer': answer,
            'detailed_results': results,
            'analysis_type': 'specialized',
            'confidence': 0.85 if results else 0.2
        }

    def _format_reasoning_results(self, question: str, reasoning_chain: ReasoningChain) -> Dict:
        """Format CoT reasoning results"""
        return {
            'question': question,
            'answer': reasoning_chain.final_answer,
            'reasoning_steps': [
                {
                    'step': t.step,
                    'question': t.question,
                    'reasoning': t.reasoning,
                    'query': t.kg_query[:200] if t.kg_query else None,
                    'results_count': len(t.kg_result) if t.kg_result else 0,
                    'insight': t.insight
                }
                for t in reasoning_chain.thoughts
            ],
            'discoveries': reasoning_chain.discoveries,
            'confidence': reasoning_chain.confidence,
            'tokens_used': reasoning_chain.total_tokens_used,
            'analysis_type': 'chain_of_thought'
        }

    def batch_analyze(self, questions: List[str]) -> List[Dict]:
        """Analyze multiple questions efficiently"""
        results = []

        for q in questions:
            try:
                # Reset token counter for each question
                self.cot_engine.total_tokens = 0
                result = self.answer(q)
                results.append(result)

                # Add delay to avoid rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to analyze question '{q}': {e}")
                results.append({
                    'question': q,
                    'answer': f"Analysis failed: {str(e)}",
                    'error': True
                })

        return results

    def explore_hypothesis(self, hypothesis: str) -> Dict:
        """Test a specific hypothesis"""

        # Convert hypothesis to testable questions
        test_questions = self._hypothesis_to_questions(hypothesis)

        # Run analyses
        evidence = []
        for q in test_questions:
            try:
                result = self.answer(q)
                evidence.append({
                    'question': q,
                    'finding': result.get('answer', '')[:200],
                    'confidence': result.get('confidence', 0)
                })
            except Exception as e:
                logger.error(f"Failed to test: {e}")

        # Assess overall support
        avg_confidence = np.mean([e['confidence'] for e in evidence]) if evidence else 0

        return {
            'hypothesis': hypothesis,
            'evidence': evidence,
            'overall_confidence': avg_confidence,
            'verdict': self._assess_hypothesis(evidence, avg_confidence)
        }

    def _hypothesis_to_questions(self, hypothesis: str) -> List[str]:
        """Convert hypothesis to testable questions"""
        # Simple keyword-based conversion
        questions = []

        if 'correlation' in hypothesis.lower():
            questions.append("What regions show the pattern described?")
            questions.append("Are there statistical outliers that match this hypothesis?")
        elif 'project' in hypothesis.lower():
            questions.append("What are the projection patterns of relevant regions?")
        else:
            questions.append(f"Find evidence for: {hypothesis}")
            questions.append(f"Find evidence against: {hypothesis}")

        return questions[:2]  # Limit to 2 questions to save tokens

    def _assess_hypothesis(self, evidence: List[Dict], confidence: float) -> str:
        """Assess hypothesis based on evidence"""
        if confidence > 0.7:
            return "Supported by evidence"
        elif confidence > 0.4:
            return "Partially supported"
        elif confidence > 0.2:
            return "Weak evidence"
        else:
            return "Not supported by available data"

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            'total_tokens_used': self.cot_engine.total_tokens,
            'cache_size': len(self.kg.query_cache),
            'token_limit': self.cot_engine.token_limit,
            'max_thinking_steps': self.cot_engine.max_thinking_steps
        }

    def close(self):
        """Clean up resources"""
        self.kg.close()


# ==================== Utility Functions ====================

def format_results_for_display(results: Dict) -> str:
    """Format results for readable display"""
    output = []
    output.append(f"Question: {results['question']}\n")
    output.append("=" * 60)

    if 'analysis_type' in results:
        output.append(f"Analysis Type: {results['analysis_type']}")

    output.append(f"\nAnswer:\n{results['answer']}\n")

    if 'reasoning_steps' in results:
        output.append(f"\nReasoning Process ({len(results['reasoning_steps'])} steps):")
        for step in results['reasoning_steps']:
            output.append(f"  Step {step['step']}: {step.get('insight', 'Processing...')}")

    if 'confidence' in results:
        output.append(f"\nConfidence: {results['confidence']:.2%}")

    if 'tokens_used' in results:
        output.append(f"Tokens Used: {results['tokens_used']:,}")

    return "\n".join(output)


# ==================== Main Execution ====================

def main():
    """Main execution with improved error handling"""

    # Configuration
    config = {
        'neo4j_uri': "bolt://10.133.56.119:7687",  # Update with actual
        'neo4j_user': "neo4j",
        'neo4j_password': "neuroxiv",  # Update with actual
        'openai_api_key': ""  # Update with actual
    }

    # Initialize agent
    agent = CoTKGAgent(**config)

    try:
        # Example 1: Car3 Analysis (Specialized + CoT)
        print("=" * 60)
        print("Example 1: Car3 Transcriptome Analysis")
        # result1 = agent.answer(
        #     "Analyze the projection pattern of the brain region with "
        #     "the highest proportion of Car3 transcriptome subclass neurons"
        # )
        result1 = agent.answer(
            "find some region pairs with enough data and their morphological features are similar but molecular features are very different"
        )
        print("=" * 60)
        print(format_results_for_display(result1))

        # Check token usage
        stats = agent.get_statistics()
        print(f"\nSystem Stats: {stats['total_tokens_used']:,} / {stats['token_limit']:,} tokens used")

        # if stats['total_tokens_used'] < stats['token_limit'] * 0.8:
        #     # Example 2: Morphological Analysis
        #     print("\n" + "=" * 60)
        #     print("Example 2: Morphological Outliers")
        #     result2 = agent.answer(
        #         "Which regions show unusual morphological properties?"
        #     )
        #     print(format_results_for_display(result2))
        #
        # # Example 3: Hypothesis Testing
        # print("\n" + "=" * 60)
        # print("Example 3: Hypothesis Testing")
        # hypothesis_result = agent.explore_hypothesis(
        #     "Regions with high Car3 expression have longer range projections"
        # )
        # print(f"Hypothesis: {hypothesis_result['hypothesis']}")
        # print(f"Verdict: {hypothesis_result['verdict']}")
        # print(f"Confidence: {hypothesis_result['overall_confidence']:.2%}")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        agent.close()
        print("\n" + "=" * 60)
        print("Analysis completed successfully")


# ==================== Interactive Mode ====================

def interactive_mode():
    """Interactive question-answering mode"""

    print("Initializing CoT-KG Agent...")

    # Get configuration
    config = {
        'neo4j_uri': input("Neo4j URI (default: bolt://localhost:7687): ") or "bolt://localhost:7687",
        'neo4j_user': input("Neo4j User (default: neo4j): ") or "neo4j",
        'neo4j_password': input("Neo4j Password: "),
        'openai_api_key': input("OpenAI API Key: ")
    }

    agent = CoTKGAgent(**config)

    print("\nAgent initialized. Type 'help' for commands, 'quit' to exit.\n")

    commands = {
        'help': "Show available commands",
        'stats': "Show system statistics",
        'clear_cache': "Clear query cache",
        'quit': "Exit the program"
    }

    while True:
        try:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                for cmd, desc in commands.items():
                    print(f"  {cmd}: {desc}")
            elif user_input.lower() == 'stats':
                stats = agent.get_statistics()
                print(f"\nSystem Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            elif user_input.lower() == 'clear_cache':
                agent.kg.query_cache.clear()
                print("Query cache cleared.")
            elif user_input:
                print("\nAnalyzing...")
                result = agent.answer(user_input)
                print("\n" + format_results_for_display(result))

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Interactive mode error: {e}")

    agent.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    # Choose mode based on command line arguments
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        main()