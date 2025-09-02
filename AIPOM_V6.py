"""
NeuroXiv-Aware Universal Agent
Based on actual KG structure from KG_ConstructorV3_Neo4j.py

Key insight: The KG has specific structure that MUST be respected:
- Region nodes with morphological properties
- Class/Subclass/Supertype/Cluster nodes for molecular features
- Specific relationship types: PROJECT_TO, HAS_CLASS, etc.
"""

import json
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import numpy as np

from neo4j import GraphDatabase
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== NeuroXiv KG Structure ====================

class NeuroXivSchema:
    """Exact schema from KG_Constructor"""

    # Node types and their ACTUAL properties
    NODE_SCHEMAS = {
        'Region': {
            'id_field': 'region_id',
            'properties': {
                # Morphological attributes
                'morphological': [
                    'axonal_bifurcation_remote_angle',
                    'axonal_branches',
                    'axonal_length',
                    'axonal_maximum_branch_order',
                    'dendritic_bifurcation_remote_angle',
                    'dendritic_branches',
                    'dendritic_length',
                    'dendritic_maximum_branch_order'
                ],
                # Statistical attributes
                'statistical': [
                    'number_of_apical_dendritic_morphologies',
                    'number_of_axonal_morphologies',
                    'number_of_dendritic_morphologies',
                    'number_of_neuron_morphologies',
                    'number_of_transcriptomic_neurons'
                ],
                # Basic attributes
                'basic': ['name', 'full_name', 'acronym', 'parent_id']
            }
        },
        'Class': {
            'id_field': 'tran_id',
            'properties': ['name', 'neighborhood', 'number_of_child_types',
                          'number_of_neurons', 'dominant_neurotransmitter_type', 'markers']
        },
        'Subclass': {
            'id_field': 'tran_id',
            'properties': ['name', 'neighborhood', 'dominant_neurotransmitter_type',
                          'number_of_child_types', 'number_of_neurons', 'markers',
                          'transcription_factor_markers']
        },
        'Supertype': {
            'id_field': 'tran_id',
            'properties': ['name', 'number_of_child_types', 'number_of_neurons',
                          'markers', 'within_subclass_markers']
        },
        'Cluster': {
            'id_field': 'tran_id',
            'properties': ['name', 'anatomical_annotation', 'broad_region_distribution',
                          'dominant_neurotransmitter_type', 'number_of_neurons', 'markers',
                          'neuropeptide_mark_genes', 'neurotransmitter_mark_genes',
                          'transcription_factor_markers', 'within_subclass_markers']
        }
    }

    # Relationships and their properties
    RELATIONSHIPS = {
        'PROJECT_TO': {
            'from': 'Region', 'to': 'Region',
            'properties': ['weight', 'total', 'neuron_count', 'source_acronym', 'target_acronym']
        },
        'HAS_CLASS': {
            'from': 'Region', 'to': 'Class',
            'properties': ['pct_cells', 'rank']
        },
        'HAS_SUBCLASS': {
            'from': 'Region', 'to': 'Subclass',
            'properties': ['pct_cells', 'rank']
        },
        'HAS_SUPERTYPE': {
            'from': 'Region', 'to': 'Supertype',
            'properties': ['pct_cells', 'rank']
        },
        'HAS_CLUSTER': {
            'from': 'Region', 'to': 'Cluster',
            'properties': ['pct_cells', 'rank']
        },
        'BELONGS_TO': {
            'patterns': [
                ('Subclass', 'Class'),
                ('Supertype', 'Subclass'),
                ('Cluster', 'Supertype')
            ]
        }
    }


# ==================== Guaranteed Working Query Generator ====================

class GuaranteedQueryGenerator:
    """Generates queries that WILL return data based on actual KG structure"""

    def __init__(self):
        self.schema = NeuroXivSchema()
        self.verified_patterns = []  # Patterns we know work

    def generate_exploration_sequence(self) -> List[Tuple[str, str]]:
        """Generate a sequence of queries guaranteed to work"""

        queries = []

        # 1. Count all nodes (WILL work)
        queries.append((
            "MATCH (n) RETURN count(n) as total_nodes",
            "Count total nodes"
        ))

        # 2. Count by label (WILL work)
        queries.append((
            """
            MATCH (n)
            WITH labels(n)[0] as label, count(n) as count
            RETURN label, count
            ORDER BY count DESC
            """,
            "Count nodes by type"
        ))

        # 3. Sample Region nodes with morphology (WILL work if data exists)
        queries.append((
            """
            MATCH (r:Region)
            WHERE r.dendritic_length IS NOT NULL
            RETURN r.region_id, r.acronym, r.dendritic_length, r.axonal_length
            LIMIT 10
            """,
            "Sample regions with morphology"
        ))

        # 4. Sample molecular features (WILL work if data exists)
        queries.append((
            """
            MATCH (r:Region)-[h:HAS_CLASS]->(c:Class)
            RETURN r.acronym as region, c.name as class_name, 
                   c.dominant_neurotransmitter_type as neurotransmitter,
                   h.pct_cells as percentage
            ORDER BY h.pct_cells DESC
            LIMIT 10
            """,
            "Sample molecular features"
        ))

        # 5. Sample projections (WILL work if data exists)
        queries.append((
            """
            MATCH (r1:Region)-[p:PROJECT_TO]->(r2:Region)
            RETURN r1.acronym as source, r2.acronym as target, p.weight
            ORDER BY p.weight DESC
            LIMIT 10
            """,
            "Sample projections"
        ))

        return queries

    def build_query_for_concept(self, concept: str, context: Dict) -> str:
        """Build query based on concept understanding"""

        concept_lower = concept.lower()

        # Morphological queries
        if any(term in concept_lower for term in ['morpholog', 'dendritic', 'axonal', 'branch']):
            return self._build_morphology_query(context)

        # Molecular queries
        if any(term in concept_lower for term in ['molecular', 'neurotransmitter', 'class', 'cell type']):
            return self._build_molecular_query(context)

        # Projection queries
        if any(term in concept_lower for term in ['project', 'connect', 'pathway']):
            return self._build_projection_query(context)

        # Comparison queries
        if 'similar' in concept_lower and 'different' in concept_lower:
            return self._build_comparison_query(context)

        # Default: explore what exists
        return """
        MATCH (r:Region)
        RETURN r.region_id, r.acronym, r.name
        LIMIT 20
        """

    def _build_morphology_query(self, context: Dict) -> str:
        """Build morphology-focused query"""

        # Start simple - just get regions with morphology data
        if not context.get('has_morphology_data'):
            return """
            MATCH (r:Region)
            WHERE r.dendritic_length IS NOT NULL OR r.axonal_length IS NOT NULL
            RETURN count(r) as regions_with_morphology,
                   avg(r.dendritic_length) as avg_dendritic,
                   avg(r.axonal_length) as avg_axonal
            """

        # Get actual morphology data
        return """
        MATCH (r:Region)
        WHERE r.dendritic_length IS NOT NULL AND r.axonal_length IS NOT NULL
        RETURN r.region_id, r.acronym,
               r.dendritic_length, r.axonal_length,
               r.dendritic_branches, r.axonal_branches
        ORDER BY r.dendritic_length DESC
        LIMIT 50
        """

    def _build_molecular_query(self, context: Dict) -> str:
        """Build molecular-focused query"""

        # Check what molecular data exists
        if not context.get('has_molecular_data'):
            return """
            MATCH (r:Region)-[:HAS_CLASS|HAS_SUBCLASS|HAS_SUPERTYPE|HAS_CLUSTER]->()
            RETURN count(DISTINCT r) as regions_with_molecular_data
            """

        # Get actual molecular data
        return """
        MATCH (r:Region)-[h:HAS_CLASS]->(c:Class)
        WHERE c.dominant_neurotransmitter_type IS NOT NULL
        RETURN r.region_id, r.acronym,
               c.name as class_name,
               c.dominant_neurotransmitter_type as neurotransmitter,
               h.pct_cells as percentage
        ORDER BY h.pct_cells DESC
        LIMIT 50
        """

    def _build_projection_query(self, context: Dict) -> str:
        """Build projection-focused query"""

        return """
        MATCH (r1:Region)-[p:PROJECT_TO]->(r2:Region)
        WHERE p.weight > 0
        RETURN r1.acronym as source, r2.acronym as target,
               p.weight, p.neuron_count
        ORDER BY p.weight DESC
        LIMIT 50
        """

    def _build_comparison_query(self, context: Dict) -> str:
        """Build comparison query (e.g., similar morphology, different molecular)"""

        # This is complex - build it step by step
        if not context.get('comparison_attempted'):
            # First, find regions with both types of data
            return """
            MATCH (r:Region)
            WHERE r.dendritic_length IS NOT NULL
            OPTIONAL MATCH (r)-[:HAS_CLASS]->(c:Class)
            WHERE c.dominant_neurotransmitter_type IS NOT NULL
            WITH r, count(c) as has_molecular
            WHERE has_molecular > 0
            RETURN count(r) as regions_with_both_data_types
            """

        # Actual comparison
        return """
        MATCH (r1:Region), (r2:Region)
        WHERE r1.region_id < r2.region_id
        AND r1.dendritic_length IS NOT NULL AND r2.dendritic_length IS NOT NULL
        AND abs(r1.dendritic_length - r2.dendritic_length) < 100
        WITH r1, r2, abs(r1.dendritic_length - r2.dendritic_length) as morph_diff
        OPTIONAL MATCH (r1)-[:HAS_CLASS]->(c1:Class)
        OPTIONAL MATCH (r2)-[:HAS_CLASS]->(c2:Class)
        WHERE c1.dominant_neurotransmitter_type IS NOT NULL 
        AND c2.dominant_neurotransmitter_type IS NOT NULL
        AND c1.dominant_neurotransmitter_type <> c2.dominant_neurotransmitter_type
        RETURN r1.acronym as region1, r2.acronym as region2,
               morph_diff,
               c1.dominant_neurotransmitter_type as r1_neurotrans,
               c2.dominant_neurotransmitter_type as r2_neurotrans
        LIMIT 20
        """

    def fix_generated_query(self, query: str) -> str:
        """Fix common issues in LLM-generated queries"""

        # Replace non-existent node types
        wrong_mappings = {
            ':Morphology': ':Region',
            ':Molecular': ':Class',
            ':Cell': ':Class',
            ':Neuron': ':Region'
        }

        for wrong, correct in wrong_mappings.items():
            query = query.replace(wrong, correct)

        # Fix property names
        property_fixes = {
            '.morphology': '.dendritic_length',
            '.molecular': '.dominant_neurotransmitter_type',
            '.type': '.dominant_neurotransmitter_type',
            '.feature': '.markers'
        }

        for wrong, correct in property_fixes.items():
            query = query.replace(wrong, correct)

        # Fix relationship names
        rel_fixes = {
            ':HAS_MORPHOLOGY': ':HAS_CLASS',
            ':HAS_MOLECULAR': ':HAS_CLASS',
            ':CONNECTS_TO': ':PROJECT_TO',
            ':HAS_CELL': ':HAS_CLASS'
        }

        for wrong, correct in rel_fixes.items():
            query = query.replace(wrong, correct)

        # Ensure LIMIT
        if 'LIMIT' not in query.upper():
            query += '\nLIMIT 20'

        return query


# ==================== Enhanced TAOR Agent ====================

class NeuroXivTAORAgent:
    """TAOR Agent that understands NeuroXiv KG structure"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str, database: str = "neuroxiv"):

        self.db = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self.llm = OpenAI(api_key=openai_api_key)

        self.query_gen = GuaranteedQueryGenerator()
        self.context = {
            'has_morphology_data': None,
            'has_molecular_data': None,
            'has_projection_data': None,
            'comparison_attempted': False,
            'successful_patterns': []
        }

        self.max_iterations = 15
        self.working_memory = {
            'question': None,
            'facts': [],
            'hypothesis': None,
            'confidence': 0.3
        }

    def solve(self, question: str) -> Dict:
        """Solve question with guaranteed queries"""

        logger.info(f"Solving: {question}")
        self.working_memory['question'] = question

        # Phase 1: Verify what data exists
        self._verify_data_availability()

        # Phase 2: TAOR loop with guaranteed queries
        iteration = 0
        while iteration < self.max_iterations and self.working_memory['confidence'] < 0.8:
            iteration += 1
            logger.info(f"\nIteration {iteration}")

            # Think
            thought = self._think()

            # Act - with guaranteed query generation
            query = self._generate_guaranteed_query(thought)

            # Observe
            observation = self._execute_query(query)

            # Reflect
            self._reflect(thought, observation)

            if self.working_memory['confidence'] >= 0.8:
                break

        # Synthesize
        answer = self._synthesize()

        return {
            'question': question,
            'answer': answer,
            'confidence': self.working_memory['confidence'],
            'iterations': iteration
        }

    def _verify_data_availability(self):
        """Verify what types of data actually exist"""

        logger.info("Verifying data availability...")

        # Check morphology data
        query = """
        MATCH (r:Region)
        WHERE r.dendritic_length IS NOT NULL OR r.axonal_length IS NOT NULL
        RETURN count(r) as count
        """
        result = self._execute_query(query)
        self.context['has_morphology_data'] = result['data'][0]['count'] > 0 if result['success'] else False

        # Check molecular data
        query = """
        MATCH (r:Region)-[:HAS_CLASS|HAS_SUBCLASS]->()
        RETURN count(DISTINCT r) as count
        """
        result = self._execute_query(query)
        self.context['has_molecular_data'] = result['data'][0]['count'] > 0 if result['success'] else False

        # Check projection data
        query = """
        MATCH ()-[:PROJECT_TO]->()
        RETURN count(*) as count
        """
        result = self._execute_query(query)
        self.context['has_projection_data'] = result['data'][0]['count'] > 0 if result['success'] else False

        logger.info(f"Data availability: Morphology={self.context['has_morphology_data']}, "
                   f"Molecular={self.context['has_molecular_data']}, "
                   f"Projection={self.context['has_projection_data']}")

    def _think(self) -> str:
        """Generate thought about what to explore"""

        prompt = f"""Think about how to answer this neuroscience question.

Question: {self.working_memory['question']}

Available data:
- Morphological data (dendritic/axonal properties): {self.context['has_morphology_data']}
- Molecular data (cell types, neurotransmitters): {self.context['has_molecular_data']}
- Projection data (connections between regions): {self.context['has_projection_data']}

Facts discovered: {len(self.working_memory['facts'])}
Current hypothesis: {self.working_memory['hypothesis']}

What aspect should we explore next? Be specific about what data to query."""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are exploring a neuroscience knowledge graph."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )

        return response.choices[0].message.content

    def _generate_guaranteed_query(self, thought: str) -> str:
        """Generate a query that WILL work"""

        # First try using the query generator based on concept
        query = self.query_gen.build_query_for_concept(thought, self.context)

        # If LLM wants to generate custom query, fix it
        if 'specific' in thought.lower() or 'custom' in thought.lower():
            prompt = f"""Generate a Cypher query for: {thought}

IMPORTANT - Use ONLY these node types and properties:

Region node properties:
- dendritic_length, axonal_length, dendritic_branches, axonal_branches
- region_id, acronym, name

Class/Subclass nodes:
- name, dominant_neurotransmitter_type, markers

Relationships:
- (Region)-[:HAS_CLASS]->(Class) with properties: pct_cells, rank
- (Region)-[:PROJECT_TO]->(Region) with properties: weight
- (Region)-[:HAS_SUBCLASS]->(Subclass)

Write a SIMPLE query that uses ONLY these elements:"""

            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Generate only valid Cypher using the exact schema provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            custom_query = response.choices[0].message.content
            # Fix any issues
            query = self.query_gen.fix_generated_query(custom_query)

        return query

    def _execute_query(self, query: str) -> Dict:
        """Execute query and return results"""

        try:
            with self.db.session(database=self.database) as session:
                result = session.run(query)
                data = [dict(record) for record in result]

            return {
                'success': True,
                'data': data,
                'count': len(data)
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'data': [],
                'count': 0,
                'error': str(e)
            }

    def _reflect(self, thought: str, observation: Dict):
        """Reflect on observation"""

        if observation['success'] and observation['count'] > 0:
            # Extract insights
            insight = f"Found {observation['count']} results related to: {thought[:50]}"
            self.working_memory['facts'].append(insight)
            self.working_memory['confidence'] += 0.1

            # Record successful pattern
            self.context['successful_patterns'].append(thought[:50])
        else:
            self.working_memory['confidence'] -= 0.05

    def _synthesize(self) -> str:
        """Synthesize final answer"""

        if not self.working_memory['facts']:
            return "Unable to find relevant data to answer the question."

        facts_text = "\n".join([f"- {fact}" for fact in self.working_memory['facts'][-10:]])

        prompt = f"""Synthesize an answer based on these findings:

Question: {self.working_memory['question']}

Findings:
{facts_text}

Provide a clear, comprehensive answer:"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Synthesize findings into a clear answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content

    def close(self):
        self.db.close()


# ==================== Usage ====================

def main():
    config = {
        'neo4j_uri': "bolt://100.88.72.32:7687",  # Update with actual
        'neo4j_user': "neo4j",
        'neo4j_password': "neuroxiv",  # Update with actual
        'openai_api_key': "",
        'database': "neo4j"
        # Update with actual
    }

    agent = NeuroXivTAORAgent(**config)

    try:
        result = agent.solve("Find region pairs with similar morphological features but different molecular features")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.1%}")
    finally:
        agent.close()

if __name__ == "__main__":
    main()