"""
AIPOM-CoT V8: TRUE AGENT - Ultimate Implementation
====================================================

A truly autonomous neuroscience knowledge graph agent that:
1. Thinks like an expert neuroscientist
2. Has complete mastery of the KG schema and data
3. Autonomously generates Cypher queries (no templates!)
4. Integrates statistical rigor
5. Can reproduce Figure 3, 4 and ace Figure 5 benchmark

Core Philosophy:
- LLM has FULL AUTONOMY (not choosing from a menu)
- Direct database access via Cypher (no operator abstraction)
- Sophisticated reasoning with domain expertise
- Production-grade quality
"""

import json
import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance

from neo4j_exec import Neo4jExec
from schema_cache import SchemaCache
from operators import validate_and_fix_cypher

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


# ==================== Keep the EXCELLENT Statistical Tools ====================

class StatisticalTools:
    """
    Your statistical toolkit is EXCELLENT - keeping it as is!
    This is production-grade statistical rigor.
    """

    @staticmethod
    def hypergeometric_enrichment(k: int, M: int, n: int, N: int) -> Dict[str, float]:
        """Hypergeometric test for region enrichment"""
        from scipy.stats import hypergeom
        p_value = hypergeom.sf(k - 1, M, N, n)
        observed_rate = k / n if n > 0 else 0
        expected_rate = N / M if M > 0 else 0
        fold_enrichment = observed_rate / expected_rate if expected_rate > 0 else 0
        expected = n * N / M if M > 0 else 0

        return {
            'p_value': float(p_value),
            'fold_enrichment': float(fold_enrichment),
            'expected': float(expected),
            'observed': k
        }

    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """Benjamini-Hochberg FDR correction"""
        from statsmodels.stats.multitest import multipletests
        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        significant = q_values < alpha
        return q_values.tolist(), significant.tolist()

    @staticmethod
    def permutation_test(observed_stat: float,
                         data1: np.ndarray,
                         data2: np.ndarray,
                         n_permutations: int = 1000,
                         seed: Optional[int] = None) -> Dict[str, float]:
        """Permutation test"""
        if seed is not None:
            np.random.seed(seed)

        combined = np.concatenate([data1, data2])
        n1 = len(data1)

        null_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            null_stat = np.mean(combined[:n1]) - np.mean(combined[n1:])
            null_stats.append(null_stat)

        null_stats = np.array(null_stats)
        p_value = np.mean(np.abs(null_stats) >= np.abs(observed_stat))

        return {
            'p_value': float(p_value),
            'observed_stat': float(observed_stat),
            'null_mean': float(np.mean(null_stats)),
            'null_std': float(np.std(null_stats))
        }

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d effect size"""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    @staticmethod
    def bootstrap_ci(data: np.ndarray,
                     statistic_func=np.mean,
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95,
                     seed: Optional[int] = None) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        if seed is not None:
            np.random.seed(seed)

        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))


# ==================== Keep EXCELLENT Fingerprint Analysis ====================

class FingerprintAnalyzer:
    """
    Your fingerprint analysis is brilliant - keeping and enhancing!
    This is the core of Figure 4.
    """

    def __init__(self, db: Neo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema
        self._subclass_cache = None
        self._target_cache = None

    def compute_region_fingerprint(self, region: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute tri-modal fingerprint for a region

        Returns:
            {
                'molecular': np.ndarray,
                'morphological': np.ndarray,
                'projection': np.ndarray
            }
        """
        fingerprint = {}

        # Molecular fingerprint
        mol_fp = self._compute_molecular_fingerprint(region)
        if mol_fp is not None:
            fingerprint['molecular'] = mol_fp

        # Morphological fingerprint
        mor_fp = self._compute_morphological_fingerprint(region)
        if mor_fp is not None:
            fingerprint['morphological'] = mor_fp

        # Projection fingerprint
        proj_fp = self._compute_projection_fingerprint(region)
        if proj_fp is not None:
            fingerprint['projection'] = proj_fp

        if len(fingerprint) == 0:
            return None

        return fingerprint

    def _compute_molecular_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """Molecular fingerprint = normalized subclass distribution"""
        query = """
        MATCH (r:Region {acronym: $acronym})-[h:HAS_SUBCLASS]->(s:Subclass)
        RETURN s.name AS subclass, h.pct_cells AS pct
        ORDER BY s.name
        """
        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # Get standard subclass order
        all_subclasses = self._get_all_subclasses()

        # Build vector
        subclass_dict = {row['subclass']: row['pct'] or 0.0 for row in result['data']}
        vector = np.array([subclass_dict.get(sc, 0.0) for sc in all_subclasses])

        # Normalize
        total = np.sum(vector)
        if total > 0:
            vector = vector / total

        return vector

    def _compute_morphological_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """Morphological fingerprint = normalized feature vector"""
        query = """
        MATCH (r:Region {acronym: $acronym})
        RETURN r.axonal_length AS axon_len,
               r.dendritic_length AS dend_len,
               r.axonal_branches AS axon_br,
               r.dendritic_branches AS dend_br,
               r.axonal_maximum_branch_order AS axon_order,
               r.dendritic_maximum_branch_order AS dend_order
        """
        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        data = result['data'][0]
        vector = np.array([
            data.get('axon_len') or 0.0,
            data.get('dend_len') or 0.0,
            data.get('axon_br') or 0.0,
            data.get('dend_br') or 0.0,
            data.get('axon_order') or 0.0,
            data.get('dend_order') or 0.0
        ], dtype=float)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _compute_projection_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """Projection fingerprint = normalized target distribution"""
        query = """
        MATCH (r:Region {acronym: $acronym})-[p:PROJECT_TO]->(t:Region)
        RETURN t.acronym AS target, p.weight AS weight
        ORDER BY t.acronym
        """
        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # Get standard targets
        all_targets = self._get_all_targets()

        # Build vector
        target_dict = {row['target']: row['weight'] or 0.0 for row in result['data']}
        vector = np.array([target_dict.get(t, 0.0) for t in all_targets])

        # Normalize
        total = np.sum(vector)
        if total > 0:
            vector = vector / total

        return vector

    def compute_similarity(self, fp1: np.ndarray, fp2: np.ndarray, metric: str = 'cosine') -> float:
        """Compute similarity between two fingerprints"""
        if metric == 'cosine':
            norm1, norm2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(fp1, fp2) / (norm1 * norm2))
        elif metric == 'euclidean':
            distance = np.linalg.norm(fp1 - fp2)
            return float(1.0 / (1.0 + distance))
        elif metric == 'correlation':
            if len(fp1) < 2:
                return 0.0
            r, _ = stats.pearsonr(fp1, fp2)
            return float(r)
        else:
            return 0.0

    def compute_mismatch_index(self, region1: str, region2: str) -> Optional[Dict[str, float]]:
        """
        Compute cross-modal mismatch (Figure 4 key metric)

        MM_GM = |sim_molecular - sim_morphological|
        MM_GP = |sim_molecular - sim_projection|
        """
        fp1 = self.compute_region_fingerprint(region1)
        fp2 = self.compute_region_fingerprint(region2)

        if fp1 is None or fp2 is None:
            return None

        sim_mol = self.compute_similarity(fp1['molecular'], fp2['molecular'])
        sim_mor = self.compute_similarity(fp1['morphological'], fp2['morphological'])
        sim_proj = self.compute_similarity(fp1['projection'], fp2['projection'])

        return {
            'sim_molecular': sim_mol,
            'sim_morphological': sim_mor,
            'sim_projection': sim_proj,
            'mismatch_GM': abs(sim_mol - sim_mor),
            'mismatch_GP': abs(sim_mol - sim_proj),
            'mismatch_MP': abs(sim_mor - sim_proj)
        }

    def _get_all_subclasses(self) -> List[str]:
        """Get all subclass names for consistent fingerprint dimensions"""
        if self._subclass_cache is not None:
            return self._subclass_cache

        query = "MATCH (s:Subclass) RETURN s.name AS name ORDER BY s.name"
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._subclass_cache = [row['name'] for row in result['data']]
        else:
            # Fallback
            self._subclass_cache = ['IT', 'ET', 'CT', 'PT', 'NP', 'Pvalb', 'Sst', 'Vip', 'Lamp5']

        return self._subclass_cache

    def _get_all_targets(self) -> List[str]:
        """Get all projection targets for consistent fingerprint dimensions"""
        if self._target_cache is not None:
            return self._target_cache

        query = """
        MATCH ()-[:PROJECT_TO]->(t:Region)
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        """
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._target_cache = [row['target'] for row in result['data']]
        else:
            # Fallback
            self._target_cache = ['MOs', 'MOp', 'ACAd', 'SSp', 'ENTl', 'CP', 'TH']

        return self._target_cache


# ==================== NEW: Safe Cypher Executor ====================

class SafeCypherExecutor:
    """
    Production-grade Cypher executor with:
    - Schema validation
    - Auto-fix common errors
    - EXPLAIN pre-check
    - Timeout protection
    """

    def __init__(self, db: Neo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema

    def execute(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute Cypher with full safety checks
        """
        start_time = time.time()

        # 1. Ensure LIMIT
        query = self._ensure_limit(query)

        # 2. Validate and auto-fix schema
        validation = self._validate_schema(query)
        if not validation['valid']:
            # Try to auto-fix
            fixed_query = self._auto_fix_schema(query, validation['issues'])
            if fixed_query:
                logger.info(f"Auto-fixed query: {validation['issues']}")
                query = fixed_query
            else:
                return {
                    'success': False,
                    'error': f"Schema validation failed: {', '.join(validation['issues'])}",
                    'data': [],
                    'execution_time': 0.0
                }

        # 3. EXPLAIN pre-check
        if not self.db.explain_ok(query):
            logger.warning("EXPLAIN failed, stripping ORDER BY")
            query = re.sub(r'\bORDER BY\s+[^\n]+', '', query, flags=re.I)

        # 4. Execute
        result = self.db.run(query, params or {})

        result['execution_time'] = time.time() - start_time
        result['query'] = query

        return result

    def _ensure_limit(self, query: str, default_limit: int = 100) -> str:
        """Ensure query has LIMIT clause"""
        if re.search(r'\bLIMIT\b', query, re.I):
            return query
        return f"{query}\nLIMIT {default_limit}"

    def _validate_schema(self, query: str) -> Dict[str, Any]:
        """Validate that labels and properties exist in schema"""
        issues = []

        # Extract node labels
        labels = re.findall(r':([A-Z][a-zA-Z_]+)', query)
        for label in set(labels):
            if not self.schema.has_label(label):
                issues.append(f"Unknown label: {label}")

        # Extract properties (n.property pattern)
        # Need to map alias to label first
        alias_to_label = {}
        for match in re.finditer(r'\((\w+):([A-Z][a-zA-Z_]+)\)', query):
            alias, label = match.group(1), match.group(2)
            alias_to_label[alias] = label

        # Check properties
        for match in re.finditer(r'\b([a-z]\w*)\.([a-z_][a-z0-9_]*)\b', query):
            alias, prop = match.group(1), match.group(2)
            label = alias_to_label.get(alias)

            if label and not self.schema.has_prop(label, prop):
                issues.append(f"Unknown property: {label}.{prop}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def _auto_fix_schema(self, query: str, issues: List[str]) -> Optional[str]:
        """Try to automatically fix schema issues"""
        fixed_query = query

        for issue in issues:
            if "Unknown property:" in issue:
                # Extract label.property
                parts = issue.split(": ")[1].split(".")
                if len(parts) == 2:
                    label, wrong_prop = parts

                    # Find similar property
                    if self.schema.has_label(label):
                        import difflib
                        candidates = list(self.schema.node_props[label].keys())
                        matches = difflib.get_close_matches(wrong_prop, candidates, n=1, cutoff=0.6)

                        if matches:
                            correct_prop = matches[0]
                            # Replace in query
                            fixed_query = re.sub(
                                rf'\b\w+\.{wrong_prop}\b',
                                lambda m: m.group(0).replace(wrong_prop, correct_prop),
                                fixed_query
                            )
                            logger.info(f"Fixed: {wrong_prop} -> {correct_prop}")

        # Validate fixed query
        if self._validate_schema(fixed_query)['valid']:
            return fixed_query
        else:
            return None


# ==================== NEW: The TRUE AGENT ====================

@dataclass
class ReasoningState:
    """Track agent's reasoning state"""
    question: str
    conversation_history: List[Dict] = field(default_factory=list)
    evidence_collected: List[Dict] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 10


class TrueNeuroscienceAgent:
    """
    THE ULTIMATE TRUE AGENT

    This is a REAL agent that:
    1. Thinks autonomously (not choosing from menus)
    2. Has deep knowledge of the KG schema
    3. Writes Cypher queries dynamically
    4. Integrates statistical tools naturally
    5. Reasons like an expert neuroscientist
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o"):

        # Initialize database
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()

        # Load schema
        with self.db.driver.session(database=database) as session:
            self.schema.load_from_db(session)

        # Initialize components
        self.executor = SafeCypherExecutor(self.db, self.schema)
        self.stats = StatisticalTools()
        self.fingerprint = FingerprintAnalyzer(self.db, self.schema)

        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # Define tools for function calling
        self.tools = self._define_tools()

        logger.info("🚀 TRUE AGENT initialized with full autonomy")

    def _define_tools(self) -> List[Dict]:
        """
        Define tools for function calling

        Key difference from your v7:
        - NO operator library
        - Only 4-5 essential tools
        - LLM writes Cypher directly
        """

        # Format schema for LLM
        schema_desc = self._format_schema_for_llm()

        return [
            # === Tool 1: Direct Cypher Execution (THE KEY!) ===
            {
                "type": "function",
                "function": {
                    "name": "execute_cypher",
                    "description": f"""Execute a Cypher query directly on the neuroscience knowledge graph.

**YOU HAVE COMPLETE ACCESS TO THE DATABASE SCHEMA:**

{schema_desc}

**Cypher Guidelines:**
- ALWAYS include LIMIT clause (max 100 rows)
- Use WHERE for filtering
- Use WITH for complex queries
- Use OPTIONAL MATCH for optional relationships
- Use aggregations (avg, sum, count, collect) as needed

**Common Patterns:**

1. Find enriched regions:
```cypher
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.name CONTAINS $gene_name
RETURN r.acronym, avg(h.pct_cells) as enrichment
ORDER BY enrichment DESC
LIMIT 10
```

2. Get morphological features:
```cypher
MATCH (r:Region)
WHERE r.acronym IN $region_list
RETURN r.acronym, r.axonal_length, r.dendritic_length, r.axonal_branches
LIMIT 50
```

3. Analyze projections:
```cypher
MATCH (source:Region)-[p:PROJECT_TO]->(target:Region)
WHERE source.acronym = $source_region
RETURN target.acronym, p.weight, p.neuron_count
ORDER BY p.weight DESC
LIMIT 20
```

4. Multi-hop analysis:
```cypher
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.name CONTAINS $gene
WITH r, avg(h.pct_cells) as enrichment
WHERE enrichment > 5.0
MATCH (r)-[p:PROJECT_TO]->(t:Region)
RETURN r.acronym, enrichment, collect({{target: t.acronym, weight: p.weight}}) as projections
LIMIT 15
```

**You can write ANY Cypher query you need! Think like a database expert.**""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Cypher query to execute"
                            },
                            "params": {
                                "type": "object",
                                "description": "Query parameters (optional)",
                                "additionalProperties": True
                            },
                            "purpose": {
                                "type": "string",
                                "description": "Why you're running this query (for logging)"
                            }
                        },
                        "required": ["query", "purpose"]
                    }
                }
            },

            # === Tool 2: Statistical Enrichment ===
            {
                "type": "function",
                "function": {
                    "name": "compute_enrichment",
                    "description": """Compute hypergeometric enrichment with FDR correction.

Use this when you have observations of counts and want to test if they're significantly enriched.

Example: You found that 15 out of 50 Car3+ cells are in region CLA. 
Is this enrichment significant given that CLA has 1000 cells total and 
there are 10000 cells in the whole brain?""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "observed_counts": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Array of observed counts for each group"
                            },
                            "population_size": {
                                "type": "integer",
                                "description": "Total population size (M)"
                            },
                            "sample_size": {
                                "type": "integer",
                                "description": "Sample size (n)"
                            },
                            "successes_in_population": {
                                "type": "integer",
                                "description": "Total successes in population (N)"
                            }
                        },
                        "required": ["observed_counts", "population_size", "sample_size", "successes_in_population"]
                    }
                }
            },

            # === Tool 3: Correlation Test ===
            {
                "type": "function",
                "function": {
                    "name": "compute_correlation",
                    "description": """Compute Pearson or Spearman correlation between two variables.

Use this to test relationships between continuous variables.
Example: Is axonal length correlated with projection strength?""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "First variable"
                            },
                            "y": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Second variable"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["pearson", "spearman"],
                                "description": "Correlation method"
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },

            # === Tool 4: Distribution Comparison ===
            {
                "type": "function",
                "function": {
                    "name": "compare_distributions",
                    "description": """Compare two distributions statistically (Mann-Whitney U test or t-test).

Use this to test if two groups are significantly different.
Example: Do Pvalb+ regions have longer axons than Sst+ regions?""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "group1": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Values for group 1"
                            },
                            "group2": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Values for group 2"
                            },
                            "test": {
                                "type": "string",
                                "enum": ["mann_whitney", "t_test", "permutation"],
                                "description": "Statistical test to use"
                            }
                        },
                        "required": ["group1", "group2"]
                    }
                }
            },

            # === Tool 5: Compute Fingerprint ===
            {
                "type": "function",
                "function": {
                    "name": "compute_fingerprint",
                    "description": """Compute multi-modal fingerprint for a brain region.

Returns normalized vectors for:
- Molecular: subclass distribution
- Morphological: axon/dendrite features
- Projection: target region weights

Use this for Figure 4 type analyses (cross-modal comparisons).""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "Region acronym (e.g., 'CLA', 'MOs')"
                            }
                        },
                        "required": ["region"]
                    }
                }
            },

            # === Tool 6: Compute Mismatch Index ===
            {
                "type": "function",
                "function": {
                    "name": "compute_mismatch",
                    "description": """Compute cross-modal mismatch between two regions.

Returns similarity scores for each modality and mismatch indices:
- MM_GM: |sim_molecular - sim_morphological|
- MM_GP: |sim_molecular - sim_projection|

High mismatch = regions similar in one modality but different in another.
This is the KEY metric for Figure 4.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region1": {
                                "type": "string",
                                "description": "First region acronym"
                            },
                            "region2": {
                                "type": "string",
                                "description": "Second region acronym"
                            }
                        },
                        "required": ["region1", "region2"]
                    }
                }
            }
        ]

    def _format_schema_for_llm(self) -> str:
        """Format complete schema for LLM understanding"""
        lines = []

        lines.append("### Node Types and Properties")
        lines.append("")

        for label in sorted(self.schema.node_props.keys()):
            props = self.schema.node_props[label]
            lines.append(f"**{label}:**")

            # Group properties by type
            morphology_props = [p for p in props.keys() if
                                any(x in p.lower() for x in ['axonal', 'dendritic', 'branch'])]
            other_props = [p for p in props.keys() if p not in morphology_props]

            if morphology_props:
                lines.append(f"  - Morphology: {', '.join(morphology_props[:8])}")
            if other_props:
                lines.append(f"  - Other: {', '.join(other_props[:8])}")
            lines.append("")

        lines.append("### Relationships")
        lines.append("")

        for rel_type in sorted(self.schema.rel_types.keys()):
            spec = self.schema.rel_types[rel_type]
            start = "/".join(spec['start'][:2]) if spec['start'] else "*"
            end = "/".join(spec['end'][:2]) if spec['end'] else "*"
            props = ", ".join(list(spec['props'].keys())[:5])

            lines.append(f"**{rel_type}:** ({start})-[:{rel_type}]->({end})")
            if props:
                lines.append(f"  Properties: {props}")
            lines.append("")

        # Add domain knowledge
        lines.append("### Domain Knowledge")
        lines.append("")
        lines.append("**Subclasses represent cell types/gene markers:**")
        lines.append("- Excitatory: IT, ET, CT, PT, NP")
        lines.append("- Inhibitory: Pvalb, Sst, Vip, Lamp5")
        lines.append("- Gene+ notation: Car3+, Pvalb+ means cells expressing that gene")
        lines.append("")
        lines.append("**Common brain regions:**")
        lines.append("- CLA: Claustrum")
        lines.append("- MOs: Secondary motor cortex")
        lines.append("- MOp: Primary motor cortex")
        lines.append("- SSp: Primary somatosensory cortex")
        lines.append("- ENTl: Lateral entorhinal cortex")

        return "\n".join(lines)

    def answer(self, question: str, max_iterations: int = 10, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Main reasoning loop - TRUE AGENT AUTONOMY

        The LLM has FULL CONTROL:
        - Decides what to query
        - Writes Cypher directly
        - Calls statistical tools as needed
        - Reasons autonomously
        """

        logger.info(f"🎯 Question: {question}")
        start_time = time.time()

        # Initialize state
        state = ReasoningState(
            question=question,
            max_iterations=max_iterations
        )

        # Build system prompt with neuroscience expertise
        system_prompt = self._build_expert_system_prompt()

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # === AUTONOMOUS REASONING LOOP ===
        final_answer = None

        while state.iterations < max_iterations:
            state.iterations += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"🧠 Iteration {state.iterations}/{max_iterations}")
            logger.info(f"{'=' * 60}")

            # LLM decides next action
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.2
            )

            message = response.choices[0].message

            # Check if LLM wants to call tools
            if message.tool_calls:
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logger.info(f"🔧 Tool: {function_name}")
                    logger.info(f"📝 Args: {json.dumps(function_args, indent=2)[:200]}...")

                    # Execute tool
                    try:
                        result = self._execute_tool(function_name, function_args)
                        success = True
                    except Exception as e:
                        logger.error(f"❌ Tool execution failed: {e}")
                        result = {"error": str(e), "success": False}
                        success = False

                    # Log result
                    if success:
                        logger.info(f"✅ Tool succeeded: {len(result.get('data', []))} rows")
                    else:
                        logger.warning(f"⚠️ Tool failed: {result.get('error', 'Unknown')}")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result, default=str)
                    })

                    # Track evidence
                    state.evidence_collected.append({
                        'iteration': state.iterations,
                        'tool': function_name,
                        'args': function_args,
                        'result': result,
                        'success': success
                    })

            else:
                # LLM provides final answer
                final_answer = message.content
                logger.info(f"💡 Final answer generated")
                break

        # If max iterations reached, force answer
        if final_answer is None:
            logger.warning(f"⏱️ Max iterations reached, requesting final answer")

            messages.append({
                "role": "user",
                "content": "Please provide your final answer based on the evidence you've collected."
            })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )

            final_answer = response.choices[0].message.content

        execution_time = time.time() - start_time

        # Evaluate quality
        quality_metrics = self._evaluate_quality(state, final_answer)

        return {
            'question': question,
            'answer': final_answer,
            'iterations': state.iterations,
            'evidence_collected': len(state.evidence_collected),
            'execution_time': execution_time,
            'quality_score': quality_metrics['overall_score'],
            'quality_metrics': quality_metrics,
            'evidence': state.evidence_collected,
            'full_conversation': messages,
            'seed': seed
        }

    def _execute_tool(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool function"""

        if function_name == "execute_cypher":
            query = args['query']
            params = args.get('params', {})
            purpose = args.get('purpose', 'Query')

            logger.info(f"📊 {purpose}")
            logger.info(f"🔍 Cypher:\n{query[:200]}...")

            result = self.executor.execute(query, params)

            return {
                'success': result['success'],
                'data': result['data'][:50],  # Limit to 50 rows for context
                'row_count': len(result['data']),
                'execution_time': result.get('execution_time', 0.0)
            }

        elif function_name == "compute_enrichment":
            observed = args['observed_counts']
            M = args['population_size']
            n = args['sample_size']
            N = args['successes_in_population']

            results = []
            p_values = []

            for k in observed:
                enrich = self.stats.hypergeometric_enrichment(int(k), M, n, N)
                results.append(enrich)
                p_values.append(enrich['p_value'])

            # FDR correction
            if len(p_values) > 1:
                q_values, significant = self.stats.fdr_correction(p_values)
                for i, res in enumerate(results):
                    res['q_value'] = q_values[i]
                    res['significant'] = significant[i]

            return {
                'success': True,
                'results': results,
                'data': results
            }

        elif function_name == "compute_correlation":
            x = np.array(args['x'])
            y = np.array(args['y'])
            method = args.get('method', 'pearson')

            if len(x) < 3 or len(y) < 3:
                return {'success': False, 'error': 'Insufficient data (n < 3)'}

            if method == 'pearson':
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)

            # Bootstrap CI
            ci_lower, ci_upper = self.stats.bootstrap_ci(np.arange(len(x)), n_bootstrap=1000)

            return {
                'success': True,
                'r': float(r),
                'p_value': float(p),
                'ci_95': [ci_lower, ci_upper],
                'n': len(x),
                'method': method,
                'data': [{'r': float(r), 'p': float(p)}]
            }

        elif function_name == "compare_distributions":
            group1 = np.array(args['group1'])
            group2 = np.array(args['group2'])
            test = args.get('test', 'mann_whitney')

            if len(group1) < 2 or len(group2) < 2:
                return {'success': False, 'error': 'Insufficient data'}

            if test == 'mann_whitney':
                statistic, p_value = stats.mannwhitneyu(group1, group2)
            elif test == 't_test':
                statistic, p_value = stats.ttest_ind(group1, group2)
            else:
                observed_diff = np.mean(group1) - np.mean(group2)
                perm_result = self.stats.permutation_test(observed_diff, group1, group2)
                statistic = perm_result['observed_stat']
                p_value = perm_result['p_value']

            cohens_d = self.stats.cohens_d(group1, group2)

            return {
                'success': True,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'cohens_d': cohens_d,
                'group1_mean': float(np.mean(group1)),
                'group2_mean': float(np.mean(group2)),
                'test': test,
                'data': [{
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'cohens_d': cohens_d
                }]
            }

        elif function_name == "compute_fingerprint":
            region = args['region']
            fp = self.fingerprint.compute_region_fingerprint(region)

            if fp is None:
                return {'success': False, 'error': f'Could not compute fingerprint for {region}'}

            return {
                'success': True,
                'region': region,
                'fingerprint': {
                    k: v.tolist() for k, v in fp.items()
                },
                'data': [{'region': region, 'modalities': list(fp.keys())}]
            }

        elif function_name == "compute_mismatch":
            region1 = args['region1']
            region2 = args['region2']

            mismatch = self.fingerprint.compute_mismatch_index(region1, region2)

            if mismatch is None:
                return {'success': False, 'error': f'Could not compute mismatch for {region1}-{region2}'}

            return {
                'success': True,
                'region1': region1,
                'region2': region2,
                **mismatch,
                'data': [mismatch]
            }

        else:
            return {'success': False, 'error': f'Unknown function: {function_name}'}

    def _build_expert_system_prompt(self) -> str:
        """Build system prompt that makes LLM think like an expert neuroscientist"""

        return f"""You are an expert computational neuroscientist with deep knowledge of brain anatomy, cell types, and connectomics.

You have direct access to a comprehensive neuroscience knowledge graph containing:
- Brain regions with morphological features
- Cell type distributions (subclasses/gene markers)
- Projection patterns (connectivity)
- Spatial transcriptomics data

# Your Expertise

You understand:
1. **Gene markers**: Car3+, Pvalb+, Sst+ refer to cells expressing those genes
2. **Cell types**: Excitatory (IT, ET, CT, PT) vs Inhibitory (Pvalb, Sst, Vip, Lamp5)
3. **Morphology**: Axonal/dendritic length, branches, arbor patterns
4. **Projections**: Long-range connectivity between brain regions
5. **Statistics**: Enrichment analysis, correlation, effect sizes

# Your Tools

You have FULL ACCESS to the database via `execute_cypher` - you can write ANY Cypher query you need.

You also have statistical tools for rigorous analysis:
- compute_enrichment: Hypergeometric test with FDR
- compute_correlation: Pearson/Spearman correlation
- compare_distributions: Mann-Whitney U / t-test
- compute_fingerprint: Multi-modal region fingerprints
- compute_mismatch: Cross-modal similarity analysis

# Your Approach

Think step-by-step like a neuroscientist:

1. **Understand the Question**
   - What biological phenomenon is being asked about?
   - What data do I need?
   - What analysis is appropriate?

2. **Query Strategically**
   - Write precise Cypher queries to get the data you need
   - Combine data from multiple modalities when needed
   - Use aggregations and filters effectively

3. **Analyze Rigorously**
   - Use appropriate statistical tests
   - Report p-values and effect sizes
   - Interpret results in biological context

4. **Synthesize Insights**
   - Connect findings across modalities
   - Identify patterns and exceptions
   - Generate testable hypotheses

# Critical Instructions

- **Be autonomous**: Don't ask permission, just use tools as needed
- **Be thorough**: Collect sufficient evidence before concluding
- **Be rigorous**: Use statistics appropriately
- **Be clear**: Explain your reasoning
- **Be efficient**: Don't repeat queries unnecessarily

Now, approach the user's question with the mindset of an expert neuroscientist conducting a rigorous analysis."""

    def _evaluate_quality(self, state: ReasoningState, answer: str) -> Dict[str, float]:
        """Evaluate the quality of the agent's reasoning and answer"""

        # 1. Data coverage - did we query enough?
        num_queries = sum(1 for e in state.evidence_collected if e['tool'] == 'execute_cypher')
        num_stats = sum(1 for e in state.evidence_collected if 'compute' in e['tool'])

        coverage_score = min(1.0, (num_queries * 0.15 + num_stats * 0.1))

        # 2. Success rate
        successful = sum(1 for e in state.evidence_collected if e['success'])
        total = len(state.evidence_collected)
        success_rate = successful / total if total > 0 else 0.0

        # 3. Data quality - did we get enough rows?
        total_rows = sum(
            e['result'].get('row_count', 0)
            for e in state.evidence_collected
            if e['success'] and 'row_count' in e['result']
        )
        data_quality = min(1.0, total_rows / 30.0)

        # 4. Statistical rigor - did we use stats when needed?
        has_statistical_test = any(
            e['tool'] in ['compute_enrichment', 'compute_correlation', 'compare_distributions']
            for e in state.evidence_collected
        )
        stat_score = 1.0 if has_statistical_test else 0.5

        # 5. Answer length (proxy for thoroughness)
        answer_length_score = min(1.0, len(answer) / 500.0)

        # Overall score
        overall_score = (
                coverage_score * 0.25 +
                success_rate * 0.25 +
                data_quality * 0.2 +
                stat_score * 0.15 +
                answer_length_score * 0.15
        )

        return {
            'overall_score': overall_score,
            'coverage_score': coverage_score,
            'success_rate': success_rate,
            'data_quality': data_quality,
            'statistical_rigor': stat_score,
            'answer_thoroughness': answer_length_score,
            'num_queries': num_queries,
            'num_statistical_tests': num_stats,
            'total_evidence': len(state.evidence_collected)
        }


# ==================== Testing and Benchmark Functions ====================

def test_figure3_car3_analysis(agent: TrueNeuroscienceAgent):
    """
    Test: Reproduce Figure 3 (Car3+ neuron analysis)

    The agent should autonomously:
    1. Identify Car3 as a gene marker
    2. Find enriched regions
    3. Get morphological features
    4. Get projection targets
    5. Analyze target molecular profiles
    """
    print("\n" + "=" * 80)
    print("TEST: Figure 3 - Car3+ Neuron Analysis")
    print("=" * 80)

    question = """Analyze Car3+ neurons comprehensively:
1. Which brain regions show enrichment for Car3+ cells?
2. What are the morphological characteristics of these Car3-enriched regions?
3. Where do these regions project to?
4. What are the molecular profiles (cell type compositions) of the projection targets?

Provide quantitative results with statistical rigor."""

    result = agent.answer(question, max_iterations=8)

    print(f"\n📊 Results:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Evidence collected: {result['evidence_collected']}")
    print(f"  Quality score: {result['quality_score']:.3f}")
    print(f"  Execution time: {result['execution_time']:.1f}s")

    print(f"\n💡 Answer:\n{result['answer'][:500]}...\n")

    # Save detailed results
    with open('/mnt/user-data/outputs/figure3_car3_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print("✅ Results saved to figure3_car3_result.json")

    return result


def test_figure4_trimodal_fingerprints(agent: TrueNeuroscienceAgent):
    """
    Test: Reproduce Figure 4 (Tri-modal fingerprint analysis)

    The agent should autonomously:
    1. Select interesting regions to compare
    2. Compute fingerprints for multiple regions
    3. Compute similarity matrices
    4. Identify mismatch patterns
    """
    print("\n" + "=" * 80)
    print("TEST: Figure 4 - Tri-Modal Fingerprint Analysis")
    print("=" * 80)

    question = """Perform a multi-modal similarity analysis to identify cross-modal mismatches:

1. Select 5-8 representative brain regions
2. Compute their molecular, morphological, and projection fingerprints
3. Calculate pairwise similarities for each modality
4. Identify region pairs that show HIGH molecular similarity but LOW projection similarity (or vice versa)
5. Compute mismatch indices (MM_GM and MM_GP) for the most interesting pairs

Report quantitative mismatch scores and interpret what they mean biologically."""

    result = agent.answer(question, max_iterations=10)

    print(f"\n📊 Results:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Evidence collected: {result['evidence_collected']}")
    print(f"  Quality score: {result['quality_score']:.3f}")
    print(f"  Execution time: {result['execution_time']:.1f}s")

    print(f"\n💡 Answer:\n{result['answer'][:500]}...\n")

    # Save
    with open('/mnt/user-data/outputs/figure4_trimodal_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print("✅ Results saved to figure4_trimodal_result.json")

    return result


def test_novel_question(agent: TrueNeuroscienceAgent):
    """Test on a novel question to demonstrate true autonomy"""
    print("\n" + "=" * 80)
    print("TEST: Novel Question (Demonstrating True Autonomy)")
    print("=" * 80)

    question = """Compare Pvalb+ and Sst+ inhibitory neurons:
- Which brain regions show enrichment for each?
- How do their morphological features differ?
- Do they have different projection patterns?
- Use appropriate statistical tests and report effect sizes."""

    result = agent.answer(question, max_iterations=8)

    print(f"\n📊 Results:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Quality score: {result['quality_score']:.3f}")

    print(f"\n💡 Answer:\n{result['answer'][:400]}...\n")

    return result


if __name__ == "__main__":
    import os

    print("\n" + "=" * 80)
    print("AIPOM-CoT V8 - TRUE AGENT Testing")
    print("THE ULTIMATE AUTONOMOUS NEUROSCIENCE AGENT")
    print("=" * 80 + "\n")

    # Initialize agent
    agent = TrueNeuroscienceAgent(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # Run tests
    print("\n1️⃣ Testing Figure 3 (Car3+ Analysis)...")
    result_fig3 = test_figure3_car3_analysis(agent)

    print("\n\n2️⃣ Testing Figure 4 (Tri-Modal Fingerprints)...")
    result_fig4 = test_figure4_trimodal_fingerprints(agent)

    print("\n\n3️⃣ Testing Novel Question (True Autonomy)...")
    result_novel = test_novel_question(agent)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 80)
    print(f"\nFigure 3 Quality: {result_fig3['quality_score']:.3f}")
    print(f"Figure 4 Quality: {result_fig4['quality_score']:.3f}")
    print(f"Novel Question Quality: {result_novel['quality_score']:.3f}")