"""
Focus-Driven Analysis Planner
==============================
实现Figure 3的"聚焦式深度分析"策略

核心思想:
1. Broad search: 找到所有相关entities
2. Identify focus: 识别PRIMARY entity (最重要的)
3. Deep dive: 对primary entity做多模态深度分析
4. Close loop: 分析projection targets的molecular composition

Author: Claude & PrometheusTT
Date: 2025-11-13
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== Focus Entity ====================

@dataclass
class FocusEntity:
    """聚焦实体"""
    entity_id: str
    entity_type: str
    focus_score: float  # 0-1: 重要性分数
    supporting_data: Dict  # 支持数据


# ==================== Focus Identifier ====================

class FocusIdentifier:
    """识别PRIMARY FOCUS"""

    def __init__(self, db):
        self.db = db

    def identify_primary_focus(self,
                               analysis_state,
                               question: str) -> Optional[FocusEntity]:
        """
        识别PRIMARY FOCUS

        策略:
        - 如果有regions: 找enrichment最高的
        - 如果只有gene: gene本身是focus
        """

        # 如果有regions,找最富集的
        if 'Region' in analysis_state.discovered_entities:
            regions = analysis_state.discovered_entities['Region']
            if len(regions) > 0:
                return self._find_most_enriched_region(analysis_state)

        # 如果没有regions但有gene
        if 'GeneMarker' in analysis_state.discovered_entities:
            gene = analysis_state.discovered_entities['GeneMarker'][0]
            return FocusEntity(
                entity_id=gene,
                entity_type='GeneMarker',
                focus_score=1.0,
                supporting_data={'source': 'gene_query'}
            )

        return None

    def _find_most_enriched_region(self, analysis_state) -> Optional[FocusEntity]:
        """找到最富集的region"""

        regions = analysis_state.discovered_entities.get('Region', [])
        gene = analysis_state.discovered_entities.get('GeneMarker', ['unknown'])[0]

        if not regions:
            return None

        # 查询enrichment
        query = """
        MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
        WHERE r.acronym IN $regions
          AND c.markers CONTAINS $gene
        WITH r, 
             count(c) AS cluster_count,
             sum(c.number_of_neurons) AS total_neurons
        RETURN r.acronym AS region,
               r.name AS region_name,
               cluster_count,
               total_neurons
        ORDER BY total_neurons DESC, cluster_count DESC
        LIMIT 1
        """

        result = self.db.run(query, {'regions': regions[:20], 'gene': gene})

        if result['success'] and result['data']:
            top = result['data'][0]

            logger.info(f"🎯 PRIMARY FOCUS: {top['region']}")
            logger.info(f"   Enrichment: {top['total_neurons']:,} neurons, {top['cluster_count']} clusters")

            return FocusEntity(
                entity_id=top['region'],
                entity_type='Region',
                focus_score=1.0,
                supporting_data={
                    'region_name': top['region_name'],
                    'cluster_count': top['cluster_count'],
                    'total_neurons': top['total_neurons']
                }
            )

        # Fallback
        return FocusEntity(
            entity_id=regions[0],
            entity_type='Region',
            focus_score=0.7,
            supporting_data={}
        )


# ==================== Focus-Driven Planner ====================

class FocusDrivenPlanner:
    """聚焦式规划器"""

    def __init__(self, schema, db):
        self.schema = schema
        self.db = db
        self.focus_identifier = FocusIdentifier(db)

    def generate_focus_driven_plan(self,
                                   analysis_state,
                                   question: str) -> List:
        """
        生成聚焦式计划 (修复版)

        Phase 1: Broad search (找所有相关entities)
        Phase 2 + 3: Identify focus + Deep dive (合并执行)

        🔧 修复: Phase 2识别focus后立即生成Phase 3步骤
        """

        candidates = []

        # 判断当前phase
        if not analysis_state.discovered_entities.get('Region'):
            # Phase 1: Broad search
            logger.info("   Focus-Driven Phase 1: Broad search for regions")
            candidates.extend(self._phase1_broad_search(analysis_state))

        elif not hasattr(analysis_state, 'primary_focus') or analysis_state.primary_focus is None:
            # Phase 2: Identify focus (在这里直接执行，不作为单独步骤)
            logger.info("   Focus-Driven Phase 2: Identifying primary focus")

            primary_focus = self.focus_identifier.identify_primary_focus(
                analysis_state,
                question
            )

            if primary_focus:
                analysis_state.primary_focus = primary_focus

                supporting_info = primary_focus.supporting_data
                logger.info(f"🎯 PRIMARY FOCUS: {primary_focus.entity_id}")
                logger.info(
                    f"   Enrichment: {supporting_info.get('total_neurons', 'N/A')} neurons, {supporting_info.get('cluster_count', 'N/A')} clusters")

            # 🔧 关键修复: 立即生成Phase 3的步骤
            logger.info("   Focus-Driven Phase 3: Deep analysis of primary focus")
            candidates.extend(self._phase3_deep_analysis(analysis_state))

        else:
            # Phase 3: Deep dive (后续迭代，继续分析)
            logger.info("   Focus-Driven Phase 3 (continued): Deep analysis")
            candidates.extend(self._phase3_deep_analysis(analysis_state))

        return candidates

    def _phase1_broad_search(self, analysis_state) -> List:
        """Phase 1: 广泛搜索"""
        from adaptive_planner import CandidateStep

        gene = analysis_state.discovered_entities.get('GeneMarker', ['unknown'])[0]

        return [
            CandidateStep(
                step_id='phase1_find_regions',
                step_type='molecular',
                purpose=f'Identify ALL brain regions expressing {gene} (Broad Search)',
                rationale='Comprehensive survey to find candidate regions for focused analysis',
                priority=10.0,
                schema_path='Region -[HAS_CLUSTER]-> Cluster',
                expected_data='Ranked list of regions with enrichment metrics',
                cypher_template="""
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE c.markers CONTAINS $gene
                WITH r,
                     count(c) AS cluster_count,
                     sum(c.number_of_neurons) AS total_neurons,
                     collect(c.name)[0..5] AS sample_clusters,
                     collect(c.number_of_neurons) AS neuron_counts
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       cluster_count,
                       total_neurons,
                       sample_clusters,
                       total_neurons * 100.0 / (
                           CASE 
                               WHEN reduce(s = 0, x IN neuron_counts | s + x) > 0 
                               THEN reduce(s = 0, x IN neuron_counts | s + x)
                               ELSE 1 
                           END
                       ) AS enrichment_percentage
                ORDER BY total_neurons DESC, cluster_count DESC
                LIMIT 20
                """,
                parameters={'gene': gene},
                depends_on=[]
            )
        ]

    def _phase2_identify_focus(self, analysis_state, question) -> List:
        """Phase 2: 识别PRIMARY FOCUS"""
        from adaptive_planner import CandidateStep

        # 识别focus
        primary_focus = self.focus_identifier.identify_primary_focus(
            analysis_state,
            question
        )

        if primary_focus:
            # 记录到state
            analysis_state.primary_focus = primary_focus

            supporting_info = primary_focus.supporting_data
            neuron_count = supporting_info.get('total_neurons', 'N/A')
            cluster_count = supporting_info.get('cluster_count', 'N/A')

            logger.info(f"   Focus entity: {primary_focus.entity_id}")
            logger.info(f"   Focus score: {primary_focus.focus_score:.2f}")

        # 返回meta-step (标记focus已确定)
        return []  # 不需要实际查询,直接进入phase 3

    def _phase3_deep_analysis(self, analysis_state) -> List:
        """Phase 3: 深入分析PRIMARY FOCUS"""
        from adaptive_planner import CandidateStep

        primary = analysis_state.primary_focus
        primary_region = primary.entity_id

        candidates = []

        # 🔹 Step 3.1: Detailed cell composition
        if not self._has_step(analysis_state, 'phase3_primary_composition'):
            candidates.append(CandidateStep(
                step_id='phase3_primary_composition',
                step_type='molecular',
                purpose=f'Detailed molecular characterization of {primary_region} (PRIMARY FOCUS)',
                rationale=f'{primary_region} shows highest enrichment, warranting comprehensive cell type profiling',
                priority=9.5,
                schema_path=f'{primary_region} -[HAS_CLUSTER]-> Cluster',
                expected_data='Complete cluster profile with quantitative metrics',
                cypher_template="""
                MATCH (r:Region {acronym: $primary_region})-[:HAS_CLUSTER]->(c:Cluster)
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       c.name AS cluster,
                       c.markers AS markers,
                       c.number_of_neurons AS neurons,
                       c.broad_region_distribution AS distribution
                ORDER BY c.number_of_neurons DESC
                LIMIT 50
                """,
                parameters={'primary_region': primary_region},
                depends_on=['phase1_find_regions']
            ))

        # 🔹 Step 3.2: Morphology
        if not self._has_step(analysis_state, 'phase3_primary_morphology'):
            candidates.append(CandidateStep(
                step_id='phase3_primary_morphology',
                step_type='morphological',
                purpose=f'Morphological profiling of {primary_region} neurons',
                rationale='Morphology reveals structural specializations related to function',
                priority=9.0,
                schema_path=f'Neuron -[LOCATE_AT]-> {primary_region}',
                expected_data='Morphological statistics with mean and standard deviation',
                cypher_template="""
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $primary_region})
                RETURN r.acronym AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches,
                       stdev(n.axonal_length) AS std_axon_length,
                       stdev(n.dendritic_length) AS std_dendrite_length
                """,
                parameters={'primary_region': primary_region},
                depends_on=['phase1_find_regions']
            ))

        # 🔹 Step 3.3: Projections
        if not self._has_step(analysis_state, 'phase3_primary_projections'):
            candidates.append(CandidateStep(
                step_id='phase3_primary_projections',
                step_type='projection',
                purpose=f'Connectivity mapping: projection targets of {primary_region}',
                rationale='Projection patterns reveal functional integration and information flow pathways',
                priority=9.5,
                schema_path=f'{primary_region} -[PROJECT_TO]-> Target',
                expected_data='Ranked projection targets with quantitative weights',
                cypher_template="""
                MATCH (r:Region {acronym: $primary_region})-[p:PROJECT_TO]->(t:Region)
                RETURN r.acronym AS source,
                       t.acronym AS target,
                       t.name AS target_name,
                       p.weight AS projection_weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                LIMIT 30
                """,
                parameters={'primary_region': primary_region},
                depends_on=['phase1_find_regions']
            ))

        # 🔹 Step 3.4: Target molecular composition (CLOSED LOOP!)
        if (self._has_step(analysis_state, 'phase3_primary_projections') and
                not self._has_step(analysis_state, 'phase3_target_composition')):

            # 获取targets
            targets = analysis_state.discovered_entities.get('ProjectionTarget', [])[:5]

            if targets:
                candidates.append(CandidateStep(
                    step_id='phase3_target_composition',
                    step_type='molecular',
                    purpose=f'Molecular composition of TOP projection targets (CLOSING THE LOOP)',
                    rationale='Complete circuit analysis by characterizing cell types in downstream targets',
                    priority=10.0,  # 最高优先级!
                    schema_path='Target -[HAS_CLUSTER]-> Cluster',
                    expected_data='Cell type profiles of major downstream regions',
                    cypher_template="""
                    MATCH (t:Region)-[:HAS_CLUSTER]->(c:Cluster)
                    WHERE t.acronym IN $targets
                    RETURN t.acronym AS target_region,
                           t.name AS target_name,
                           c.name AS cluster,
                           c.markers AS markers,
                           c.number_of_neurons AS neurons
                    ORDER BY t.acronym, c.number_of_neurons DESC
                    LIMIT 50
                    """,
                    parameters={'targets': targets},
                    depends_on=['phase3_primary_projections']
                ))

        return candidates

    def _has_step(self, state, step_id: str) -> bool:
        """检查step是否已执行"""
        for step in state.executed_steps:
            if step.get('step_id') == step_id:
                return True
        return False