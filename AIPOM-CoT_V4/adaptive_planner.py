"""
Dynamic Adaptive Planner
=========================
Schema感知的动态规划器，根据分析状态智能生成下一步

核心设计：
1. 状态驱动 - 根据当前状态动态决定下一步
2. LLM评估 - LLM评估候选步骤价值
3. 闭环支持 - 支持source→target→target_composition分析
4. 多策略 - 支持Focus-Driven/Comparative/Adaptive

Author: Lijun
Date: 2025-01
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from core_structures import (
    Entity, EntityCluster, CandidateStep, SchemaPath,
    Modality, AnalysisDepth, PlannerType, QuestionIntent,
    AnalysisState
)
from llm_intelligence import LLMClient

logger = logging.getLogger(__name__)


# ==================== Schema Graph ====================

class SchemaGraph:
    """
    Schema图结构 - 用于路径发现
    """

    def __init__(self, schema_data: Dict = None):
        """
        初始化Schema图

        Args:
            schema_data: 从schema.json加载的数据
        """
        # 默认NeuroXiv-KG schema
        self.node_types = {
            'Region': {'properties': ['acronym', 'name', 'region_id']},
            'Subregion': {'properties': ['acronym', 'name']},
            'Cluster': {'properties': ['name', 'markers', 'number_of_neurons']},
            'Subclass': {'properties': ['name', 'markers', 'description']},
            'Neuron': {'properties': ['neuron_id', 'axonal_length', 'dendritic_length']},
        }

        self.relationships = {
            'HAS_CLUSTER': {'source': 'Region', 'target': 'Cluster'},
            'HAS_SUBCLASS': {'source': 'Region', 'target': 'Subclass'},
            'PROJECT_TO': {'source': 'Region', 'target': 'Region'},
            'LOCATE_AT': {'source': 'Neuron', 'target': 'Region'},
            'BELONGS_TO': {'source': 'Region', 'target': 'Region'},
        }

        if schema_data:
            self._load_from_data(schema_data)

    def _load_from_data(self, data: Dict):
        """从数据加载schema"""
        if 'node_types' in data:
            self.node_types.update(data['node_types'])
        if 'rel_types' in data:
            for rel_type, rel_info in data['rel_types'].items():
                patterns = rel_info.get('patterns', [])
                if patterns:
                    self.relationships[rel_type] = {
                        'source': patterns[0][0],
                        'target': patterns[0][1]
                    }

    def find_paths(self, start: str, end: str, max_hops: int = 2) -> List[SchemaPath]:
        """
        查找两个节点类型之间的路径
        """
        paths = []

        # 直接关系
        for rel_type, rel_info in self.relationships.items():
            if rel_info['source'] == start and rel_info['target'] == end:
                paths.append(SchemaPath(
                    path_id=f"{start}_{rel_type}_{end}",
                    start_label=start,
                    end_label=end,
                    hops=[(start, rel_type, end)],
                    score=1.0,
                    description=f"{start} -[{rel_type}]-> {end}"
                ))
            # 反向
            elif rel_info['target'] == start and rel_info['source'] == end:
                paths.append(SchemaPath(
                    path_id=f"{end}_{rel_type}_{start}_rev",
                    start_label=start,
                    end_label=end,
                    hops=[(end, rel_type, start)],
                    score=0.9,
                    description=f"{end} -[{rel_type}]-> {start} (reverse)"
                ))

        return paths


# ==================== Candidate Step Generator ====================

class CandidateStepGenerator:
    """
    候选步骤生成器

    根据当前状态和目标，生成所有可能的下一步
    """

    def __init__(self, schema: SchemaGraph):
        self.schema = schema

    def generate_for_gene_marker(self,
                                 gene: str,
                                 state: AnalysisState,
                                 target_depth: AnalysisDepth) -> List[CandidateStep]:
        """为基因marker分析生成候选步骤"""
        candidates = []
        executed_ids = {s.get('step_id', '') for s in state.executed_steps}

        # Step: Gene -> Subclass
        if 'gene_to_subclass' not in executed_ids:
            candidates.append(CandidateStep(
                step_id='gene_to_subclass',
                step_type='molecular',
                purpose=f'Find Subclass cell types expressing {gene}',
                rationale='Subclass provides high-level cell type taxonomy',
                priority=9.5,
                cypher_template="""
                MATCH (s:Subclass)
                WHERE s.markers CONTAINS $gene
                RETURN s.name AS subclass,
                       s.markers AS markers,
                       s.description AS description
                ORDER BY s.name
                """,
                parameters={'gene': gene},
            ))

        # Step: Gene -> Cluster
        if 'gene_to_cluster' not in executed_ids:
            candidates.append(CandidateStep(
                step_id='gene_to_cluster',
                step_type='molecular',
                purpose=f'Find cell clusters expressing {gene}',
                rationale='Clusters provide fine-grained cell type resolution',
                priority=9.0,
                cypher_template="""
                MATCH (c:Cluster)
                WHERE c.markers CONTAINS $gene
                RETURN c.name AS cluster,
                       c.markers AS markers,
                       c.number_of_neurons AS neuron_count,
                       c.broad_region_distribution AS region_dist
                ORDER BY c.number_of_neurons DESC
                """,
                parameters={'gene': gene},
            ))

        # Step: Gene -> Region enrichment
        if 'gene_to_regions' not in executed_ids and 'Region' not in state.discovered_entities:
            candidates.append(CandidateStep(
                step_id='gene_to_regions',
                step_type='molecular',
                purpose=f'Identify brain regions enriched for {gene}+ cell types',
                rationale='Spatial localization reveals functional specialization',
                priority=8.5,
                cypher_template="""
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE c.markers CONTAINS $gene
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       count(c) AS cluster_count,
                       sum(c.number_of_neurons) AS total_neurons,
                       collect(c.name)[0..5] AS sample_clusters
                ORDER BY total_neurons DESC
                """,
                parameters={'gene': gene},
            ))

        # Step: Morphology (if regions discovered)
        if 'Region' in state.discovered_entities and Modality.MORPHOLOGICAL not in state.modalities_covered:
            regions = state.discovered_entities['Region'][:10]
            if 'morphology_analysis' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='morphology_analysis',
                    step_type='morphological',
                    purpose=f'Analyze morphological features of {gene}+ enriched regions',
                    rationale='Morphology reveals structural specialization',
                    priority=8.0,
                    cypher_template="""
                    MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                    WHERE r.acronym IN $regions
                    RETURN r.acronym AS region,
                           count(n) AS neuron_count,
                           avg(n.axonal_length) AS avg_axon_length,
                           avg(n.dendritic_length) AS avg_dendrite_length,
                           avg(n.axonal_branches) AS avg_axon_branches,
                           avg(n.dendritic_branches) AS avg_dendrite_branches,
                           stdev(n.axonal_length) AS std_axon,
                           stdev(n.dendritic_length) AS std_dendrite
                    ORDER BY neuron_count DESC
                    """,
                    parameters={'regions': regions},
                ))

        # Step: Projections (if regions discovered)
        if 'Region' in state.discovered_entities and Modality.PROJECTION not in state.modalities_covered:
            regions = state.discovered_entities['Region'][:10]
            if 'projection_analysis' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='projection_analysis',
                    step_type='projection',
                    purpose=f'Identify projection targets of {gene}+ regions',
                    rationale='Connectivity reveals functional integration',
                    priority=8.5,
                    cypher_template="""
                    MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                    WHERE r.acronym IN $regions
                    RETURN r.acronym AS source,
                           t.acronym AS target,
                           t.name AS target_name,
                           p.weight AS projection_weight,
                           p.neuron_count AS neuron_count
                    ORDER BY p.weight DESC
                    """,
                    parameters={'regions': regions},
                ))

        # Step: Closed-loop - Target composition (DEEP only)
        if target_depth == AnalysisDepth.DEEP:
            if Modality.PROJECTION in state.modalities_covered:
                targets = state.discovered_entities.get('ProjectionTarget', [])[:5]
                if targets and 'target_composition' not in executed_ids:
                    candidates.append(CandidateStep(
                        step_id='target_composition',
                        step_type='molecular',
                        purpose='Analyze molecular composition of projection targets (CLOSED LOOP)',
                        rationale='Complete circuit analysis by characterizing downstream targets',
                        priority=9.8,  # 高优先级 - 闭环分析
                        cypher_template="""
                        MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                        WHERE r.acronym IN $targets
                        RETURN r.acronym AS target_region,
                               r.name AS target_name,
                               c.name AS cluster,
                               c.markers AS markers,
                               c.number_of_neurons AS neurons
                        ORDER BY r.acronym, c.number_of_neurons DESC
                        """,
                        parameters={'targets': targets},
                    ))

        return candidates

    def generate_for_region(self,
                            region: str,
                            state: AnalysisState,
                            target_depth: AnalysisDepth) -> List[CandidateStep]:
        """为区域分析生成候选步骤"""
        candidates = []
        executed_ids = {s.get('step_id', '') for s in state.executed_steps}

        # Step: Region info
        if 'region_info' not in executed_ids:
            candidates.append(CandidateStep(
                step_id='region_info',
                step_type='spatial',
                purpose=f'Get basic information about {region}',
                rationale='Establish region identity and hierarchy',
                priority=9.0,
                cypher_template="""
                MATCH (r:Region {acronym: $region})
                OPTIONAL MATCH (r)-[:BELONGS_TO]->(parent:Region)
                RETURN r.acronym AS acronym,
                       r.name AS full_name,
                       r.region_id AS region_id,
                       parent.acronym AS parent_region,
                       parent.name AS parent_name
                """,
                parameters={'region': region},
            ))

        # Step: Cell composition
        if 'region_composition' not in executed_ids:
            candidates.append(CandidateStep(
                step_id='region_composition',
                step_type='molecular',
                purpose=f'Get cell type composition of {region}',
                rationale='Understand cellular diversity',
                priority=9.0,
                cypher_template="""
                MATCH (r:Region {acronym: $region})-[:HAS_CLUSTER]->(c:Cluster)
                RETURN c.name AS cluster,
                       c.markers AS markers,
                       c.number_of_neurons AS neurons
                ORDER BY c.number_of_neurons DESC
                """,
                parameters={'region': region},
            ))

        # Step: Subclass distribution
        if 'region_subclass' not in executed_ids:
            candidates.append(CandidateStep(
                step_id='region_subclass',
                step_type='molecular',
                purpose=f'Get Subclass distribution in {region}',
                rationale='Higher-level cell type taxonomy',
                priority=8.5,
                cypher_template="""
                MATCH (r:Region {acronym: $region})-[h:HAS_SUBCLASS]->(s:Subclass)
                RETURN s.name AS subclass,
                       h.pct_cells AS percentage,
                       s.markers AS markers
                ORDER BY h.pct_cells DESC
                """,
                parameters={'region': region},
            ))

        # Step: Morphology
        if 'region_morphology' not in executed_ids and Modality.MORPHOLOGICAL not in state.modalities_covered:
            candidates.append(CandidateStep(
                step_id='region_morphology',
                step_type='morphological',
                purpose=f'Analyze morphological features of neurons in {region}',
                rationale='Morphology reflects functional roles',
                priority=8.0,
                cypher_template="""
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $region})
                RETURN count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches,
                       stdev(n.axonal_length) AS std_axon,
                       stdev(n.dendritic_length) AS std_dendrite
                """,
                parameters={'region': region},
            ))

        # Step: Projections
        if 'region_projections' not in executed_ids and Modality.PROJECTION not in state.modalities_covered:
            candidates.append(CandidateStep(
                step_id='region_projections',
                step_type='projection',
                purpose=f'Identify projection targets of {region}',
                rationale='Connectivity patterns reveal functional integration',
                priority=8.5,
                cypher_template="""
                MATCH (r:Region {acronym: $region})-[p:PROJECT_TO]->(t:Region)
                RETURN t.acronym AS target,
                       t.name AS target_name,
                       p.weight AS weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                """,
                parameters={'region': region},
            ))

        return candidates

    def generate_for_comparison(self,
                                entities: List[Entity],
                                state: AnalysisState) -> List[CandidateStep]:
        """为比较分析生成候选步骤"""
        candidates = []
        executed_ids = {s.get('step_id', '') for s in state.executed_steps}

        # 获取要比较的实体
        regions = [e.name for e in entities if e.entity_type == 'Region'][:2]
        genes = [e.name for e in entities if e.entity_type == 'GeneMarker'][:2]

        if len(regions) >= 2:
            r1, r2 = regions[0], regions[1]

            # Molecular comparison
            if 'compare_molecular' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='compare_molecular',
                    step_type='molecular',
                    purpose=f'Compare molecular profiles: {r1} vs {r2}',
                    rationale='Cell type composition differences reveal regional specialization',
                    priority=9.0,
                    cypher_template="""
                    MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                    WHERE r.acronym IN $regions
                    RETURN r.acronym AS region,
                           c.name AS cluster,
                           c.markers AS markers,
                           c.number_of_neurons AS neurons
                    ORDER BY r.acronym, c.number_of_neurons DESC
                    """,
                    parameters={'regions': [r1, r2]},
                ))

            # Morphological comparison
            if 'compare_morphology' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='compare_morphology',
                    step_type='morphological',
                    purpose=f'Compare morphological features: {r1} vs {r2}',
                    rationale='Morphological differences indicate functional specialization',
                    priority=8.5,
                    cypher_template="""
                    MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                    WHERE r.acronym IN $regions
                    RETURN r.acronym AS region,
                           count(n) AS neuron_count,
                           avg(n.axonal_length) AS avg_axon,
                           avg(n.dendritic_length) AS avg_dendrite,
                           stdev(n.axonal_length) AS std_axon,
                           stdev(n.dendritic_length) AS std_dendrite
                    """,
                    parameters={'regions': [r1, r2]},
                ))

            # Projection comparison
            if 'compare_projection' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='compare_projection',
                    step_type='projection',
                    purpose=f'Compare connectivity patterns: {r1} vs {r2}',
                    rationale='Projection differences reveal functional roles',
                    priority=8.5,
                    cypher_template="""
                    MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                    WHERE r.acronym IN $regions
                    RETURN r.acronym AS source,
                           t.acronym AS target,
                           p.weight AS weight
                    ORDER BY r.acronym, p.weight DESC
                    """,
                    parameters={'regions': [r1, r2]},
                ))

        return candidates

    def generate_for_screening(self,
                               state: AnalysisState,
                               screening_type: str = 'mismatch') -> List[CandidateStep]:
        """为筛选分析生成候选步骤"""
        candidates = []
        executed_ids = {s.get('step_id', '') for s in state.executed_steps}

        if screening_type == 'mismatch':
            # Step 1: Get top regions
            if 'screen_get_regions' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='screen_get_regions',
                    step_type='spatial',
                    purpose='Identify top brain regions by neuron count',
                    rationale='Select regions with most data for screening',
                    priority=9.5,
                    cypher_template="""
                    MATCH (r:Region)
                    OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
                    WITH r, COUNT(DISTINCT n) AS neuron_count
                    WHERE neuron_count > 0
                    RETURN r.acronym AS region,
                           r.name AS region_name,
                           neuron_count
                    ORDER BY neuron_count DESC
                    LIMIT 30
                    """,
                    parameters={},
                ))

            # Step 2: Compute mismatch
            if 'screen_get_regions' in executed_ids and 'screen_compute_mismatch' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='screen_compute_mismatch',
                    step_type='multi-modal',
                    purpose='Compute cross-modal mismatch index for region pairs',
                    rationale='Mismatch quantifies molecular-morphological-projection discordance',
                    priority=10.0,
                    cypher_template='',  # Python计算
                    parameters={
                        'analysis_type': 'cross_modal_mismatch',
                        'modalities': ['molecular', 'morphological', 'projection']
                    },
                ))

            # Step 3: FDR correction
            if 'screen_compute_mismatch' in executed_ids and 'screen_fdr' not in executed_ids:
                candidates.append(CandidateStep(
                    step_id='screen_fdr',
                    step_type='statistical',
                    purpose='FDR correction for multiple testing',
                    rationale='Control false discovery rate',
                    priority=9.5,
                    cypher_template='',
                    parameters={'test_type': 'fdr', 'alpha': 0.05},
                ))

        return candidates


# ==================== LLM Step Ranker ====================

class LLMStepRanker:
    """
    LLM驱动的步骤排序器

    评估候选步骤的价值，选择最优步骤
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rank_steps(self,
                   candidates: List[CandidateStep],
                   question: str,
                   state: AnalysisState) -> List[CandidateStep]:
        """使用LLM对候选步骤排序"""

        if not candidates:
            return []

        # 如果候选步骤少，直接按priority排序
        if len(candidates) <= 2:
            return sorted(candidates, key=lambda x: x.priority, reverse=True)

        # 准备候选摘要
        candidates_summary = []
        for i, c in enumerate(candidates):
            candidates_summary.append({
                'index': i,
                'purpose': c.purpose,
                'rationale': c.rationale,
                'type': c.step_type,
                'priority': c.priority
            })

        progress = state.get_progress_summary()

        system_prompt = """You are planning neuroscience analysis steps.
Rank the candidate steps by SCIENTIFIC VALUE for answering the original question.

Consider:
1. Direct relevance to the question
2. Logical progression of analysis
3. Completing multi-modal coverage (molecular → morphological → projection)
4. Closed-loop analysis importance (if projections found, analyze targets)

Return JSON with ranked indices and scores."""

        user_prompt = f"""Rank these analysis steps:

**Question:** {question}

**Current State:**
- Modalities covered: {progress['modalities_covered']}
- Entities found: {progress['entities_found']}
- Steps executed: {progress['steps_executed']}
- Target depth: {progress['target_depth']}

**Candidates:**
{json.dumps(candidates_summary, indent=2)}

Return JSON:
{{
    "ranked_steps": [
        {{"index": 0, "score": 0.95, "reasoning": "Why this step is valuable"}}
    ]
}}"""

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)

            ranked = []
            for item in result.get('ranked_steps', []):
                idx = item.get('index', 0)
                if 0 <= idx < len(candidates):
                    candidates[idx].llm_score = item.get('score', 0.5)
                    candidates[idx].llm_reasoning = item.get('reasoning', '')
                    ranked.append(candidates[idx])

            return ranked if ranked else sorted(candidates, key=lambda x: x.priority, reverse=True)

        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}")
            return sorted(candidates, key=lambda x: x.priority, reverse=True)


# ==================== Adaptive Planner ====================

class AdaptivePlanner:
    """
    自适应规划器

    根据问题类型和当前状态，动态生成最优分析计划
    """

    def __init__(self, llm: LLMClient, schema: SchemaGraph = None):
        self.llm = llm
        self.schema = schema or SchemaGraph()
        self.step_generator = CandidateStepGenerator(self.schema)
        self.step_ranker = LLMStepRanker(llm)

    def plan_next_steps(self,
                        state: AnalysisState,
                        question: str,
                        entities: List[Entity],
                        classification: 'IntentClassification',
                        max_steps: int = 2) -> List[CandidateStep]:
        """
        规划下一批步骤

        Args:
            state: 当前分析状态
            question: 原始问题
            entities: 识别的实体
            classification: 意图分类
            max_steps: 最多返回几步

        Returns:
            排序后的候选步骤
        """
        logger.info(f"📋 Planning next steps (depth: {classification.recommended_depth.value})...")

        # 根据意图类型生成候选
        intent = classification.intent
        target_depth = classification.recommended_depth

        candidates = []

        # 基因marker分析
        genes = [e for e in entities if e.entity_type == 'GeneMarker']
        if genes:
            gene = genes[0].name
            candidates.extend(self.step_generator.generate_for_gene_marker(gene, state, target_depth))

        # 区域分析
        regions = [e for e in entities if e.entity_type == 'Region']
        if regions:
            region = regions[0].name
            candidates.extend(self.step_generator.generate_for_region(region, state, target_depth))

        # 比较分析
        if intent == QuestionIntent.COMPARISON:
            candidates.extend(self.step_generator.generate_for_comparison(entities, state))

        # 筛选分析
        if intent == QuestionIntent.SCREENING:
            candidates.extend(self.step_generator.generate_for_screening(state))

        if not candidates:
            logger.info("   No candidates generated")
            return []

        # 去重
        seen_ids = set()
        unique_candidates = []
        for c in candidates:
            if c.step_id not in seen_ids:
                seen_ids.add(c.step_id)
                unique_candidates.append(c)

        logger.info(f"   Generated {len(unique_candidates)} unique candidates")

        # LLM排序
        ranked = self.step_ranker.rank_steps(unique_candidates, question, state)

        # 返回top-N
        selected = ranked[:max_steps]

        for i, step in enumerate(selected, 1):
            logger.info(f"   {i}. {step.purpose[:50]}... (score: {step.llm_score:.2f})")

        return selected

    def should_continue(self,
                        state: AnalysisState,
                        classification: 'IntentClassification') -> bool:
        """
        判断是否应该继续分析
        """
        # 检查预算
        budget = state.check_budget()
        if not budget['can_continue']:
            logger.info("📌 Budget exhausted")
            return False

        # 检查步数限制
        target_depth = classification.recommended_depth
        max_steps_map = {
            AnalysisDepth.SHALLOW: 2,
            AnalysisDepth.MEDIUM: 4,
            AnalysisDepth.DEEP: 8
        }

        max_allowed = max_steps_map[target_depth]
        if len(state.executed_steps) >= max_allowed:
            logger.info(f"📌 Reached max steps for {target_depth.value} depth")
            return False

        # 检查模态覆盖
        expected_modalities = set(classification.expected_modalities)
        covered = state.modalities_covered

        if expected_modalities and not expected_modalities.issubset(covered):
            logger.info(f"   Missing modalities: {expected_modalities - covered}")
            return True

        # 检查闭环
        if target_depth == AnalysisDepth.DEEP:
            has_projection = Modality.PROJECTION in covered
            has_target = 'ProjectionTarget' in state.discovered_entities
            targets_analyzed = any('target' in s.get('step_id', '').lower()
                                   and 'composition' in s.get('step_id', '').lower()
                                   for s in state.executed_steps)

            if has_projection and has_target and not targets_analyzed:
                logger.info("   Continue: Need to close the loop")
                return True

        return len(state.executed_steps) < 2  # 至少执行2步


# ==================== Export ====================

__all__ = [
    'SchemaGraph',
    'CandidateStepGenerator',
    'LLMStepRanker',
    'AdaptivePlanner',
]