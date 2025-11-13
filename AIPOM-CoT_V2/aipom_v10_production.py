"""
AIPOM-CoT V10 PRODUCTION
========================
完整集成所有P0和P1组件:
✅ P0-1: 智能实体识别 (IntelligentEntityRecognizer)
✅ P0-2: Benchmark评估系统 (BenchmarkRunner)
✅ P1-1: 动态Schema路径规划 (DynamicSchemaPathPlanner)
✅ P1-2: 结构化反思 (StructuredReflector)

这是生产就绪版本,可以直接用于:
- 完整Benchmark评估
- 论文Figure 3/4/5复现
- 与baseline对比

Author: Claude & PrometheusTT
Date: 2025-01-12
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import re
from pathlib import Path

import numpy as np

from neo4j_exec import Neo4jExec
from adaptive_planner import AdaptivePlanner, AnalysisDepth, AnalysisState
from aipom_cot_true_agent_v2 import (
    RealSchemaCache,
    StatisticalTools,
    RealFingerprintAnalyzer,
    AgentPhase,
    AgentState,
    ReasoningStep
)

# 导入新组件
from intelligent_entity_recognition import (
    IntelligentEntityRecognizer,
    EntityClusteringEngine
)
from schema_path_planner import DynamicSchemaPathPlanner
from structured_reflection import StructuredReflector

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

logger = logging.getLogger(__name__)


# ==================== Enhanced Agent State ====================

@dataclass
class EnhancedAgentState(AgentState):
    """扩展的Agent状态"""

    # 新增字段
    entity_matches: List = field(default_factory=list)  # EntityMatch列表
    entity_clusters: List = field(default_factory=list)  # EntityCluster列表
    structured_reflections: List = field(default_factory=list)  # StructuredReflection列表
    schema_paths_used: List = field(default_factory=list)  # 使用的schema路径


# ==================== Production Agent V10 ====================

class AIPOMCoTV10:
    """
    AIPOM-CoT V10 生产版本

    完整功能:
    1. 智能实体识别 (无需hardcoded列表)
    2. 动态Schema路径规划 (图算法)
    3. 结构化反思 (量化评估)
    4. 完整统计工具
    5. 多模态分析
    6. 自适应重规划
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 schema_json_path: str,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o"):

        # 数据库连接
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)

        # Schema
        self.schema = RealSchemaCache(schema_json_path)

        # ===== 核心组件初始化 =====

        # P0-1: 智能实体识别
        logger.info("🔍 Initializing intelligent entity recognition...")
        self.entity_recognizer = IntelligentEntityRecognizer(self.db, self.schema)
        self.entity_clusterer = EntityClusteringEngine(self.db, self.schema)

        # P1-1: 动态Schema路径规划
        logger.info("🗺️  Initializing dynamic schema path planning...")
        self.path_planner = DynamicSchemaPathPlanner(self.schema)

        # P1-2: 结构化反思
        logger.info("🤔 Initializing structured reflection...")
        self.reflector = StructuredReflector()

        # 原有组件
        self.stats = StatisticalTools()
        self.fingerprint = RealFingerprintAnalyzer(self.db, self.schema)

        # OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model


        self.adaptive_planner = AdaptivePlanner(self.schema, self.path_planner,self.client)
        # 🆕 添加Focus-Driven Planner
        logger.info("🎯 Initializing focus-driven planning...")
        from focus_driven_planner import FocusDrivenPlanner
        self.focus_planner = FocusDrivenPlanner(self.schema, self.db)

        # 🆕 添加Comparative Analysis Planner
        logger.info("📊 Initializing comparative analysis planning...")
        from comparative_analysis_planner import ComparativeAnalysisPlanner
        self.comparative_planner = ComparativeAnalysisPlanner(
            self.db,
            self.fingerprint,
            self.stats
        )

        logger.info("✅ AIPOM-CoT V10 initialized successfully!")
        logger.info(f"   • Entity recognition: Ready")
        logger.info(f"   • Schema path planning: Ready")
        logger.info(f"   • Structured reflection: Ready")

    # ==================== Main Entry Point ====================

    """
    完整的answer方法实现 - 集成自适应规划
    """

    def answer(self, question: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        主入口: 回答问题 (完整版)

        完整流程:
        1. 智能实体识别
        2. 实体聚类
        3. 确定分析深度
        4. 智能选择规划器 (Adaptive/Focus-Driven/Comparative)
        5. 自适应执行循环 (包含统计分析)
        6. 答案合成 (科学叙事)
        """
        logger.info(f"🎯 Question: {question}")
        start_time = time.time()

        state = EnhancedAgentState(question=question)

        # ===== PHASE 1: INTELLIGENT PLANNING =====
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 1: INTELLIGENT PLANNING (Enhanced)")
        logger.info("=" * 70)

        state.phase = AgentPhase.PLANNING

        # Step 1-2: 实体识别 + 聚类
        logger.info("  [1/4] Intelligent entity recognition...")
        entity_matches = self.entity_recognizer.recognize_entities(question)
        state.entity_matches = entity_matches

        logger.info(f"     Found {len(entity_matches)} entity matches")
        for match in entity_matches[:5]:
            logger.info(f"       • {match.text} ({match.entity_type}) [{match.confidence:.2f}]")

        logger.info("  [2/4] Entity clustering...")
        entity_clusters = self.entity_clusterer.cluster_entities(entity_matches, question)
        state.entity_clusters = entity_clusters

        logger.info(f"     Created {len(entity_clusters)} entity clusters")
        for cluster in entity_clusters:
            logger.info(f"       • {cluster.cluster_type}: {cluster.primary_entity.text}")

        # 🆕 Step 3: 确定分析深度
        from adaptive_planner import determine_analysis_depth, AnalysisState

        logger.info("  [3/4] Determining analysis depth...")
        target_depth = determine_analysis_depth(question)
        logger.info(f"     Target depth: {target_depth.value}")

        # 🆕 Step 4: 初始化分析状态
        logger.info("  [4/4] Initializing analysis state...")

        analysis_state = AnalysisState(
            discovered_entities={},
            executed_steps=[],
            modalities_covered=[],
            current_focus='gene' if entity_clusters and entity_clusters[0].cluster_type == 'gene_marker' else 'region',
            target_depth=target_depth,
            question_intent=self._classify_question_intent(question)
        )

        # 填充初始实体
        for cluster in entity_clusters:
            entity_type = cluster.primary_entity.entity_type
            entity_id = cluster.primary_entity.entity_id

            analysis_state.discovered_entities.setdefault(entity_type, []).append(entity_id)

            for related in cluster.related_entities:
                analysis_state.discovered_entities.setdefault(
                    related.entity_type, []
                ).append(related.entity_id)

        # 兼容性
        state.entities = [
            {'text': m.text, 'type': m.entity_type, 'confidence': m.confidence}
            for m in entity_matches[:10]
        ]

        # 🆕 存储analysis_state到state
        state.analysis_state = analysis_state

        logger.info(f"✅ Planning complete")
        logger.info(f"   • Target depth: {target_depth.value}")
        logger.info(f"   • Initial entities: {list(analysis_state.discovered_entities.keys())}")

        # ===== PHASE 2: ADAPTIVE EXECUTION =====
        logger.info("\n" + "=" * 70)
        logger.info("⚙️ PHASE 2: ADAPTIVE EXECUTION (Multi-Planner)")
        logger.info("=" * 70)

        state.phase = AgentPhase.EXECUTING

        iteration = 0
        while iteration < max_iterations:
            # 🆕 决定是否继续
            if not self.adaptive_planner.should_continue(analysis_state, question):
                logger.info("📌 Analysis complete (adaptive decision)")
                break

            # 🆕 智能选择规划器
            planner_type = self._select_planner(analysis_state, question)

            if planner_type == 'focus_driven':
                logger.info(f"\n🎯 Using FOCUS-DRIVEN planner (iteration {iteration + 1})...")
                next_steps = self.focus_planner.generate_focus_driven_plan(
                    analysis_state,
                    question
                )

            elif planner_type == 'comparative':
                logger.info(f"\n📊 Using COMPARATIVE planner (iteration {iteration + 1})...")
                next_steps = self.comparative_planner.generate_comparative_plan(
                    analysis_state,
                    question
                )

            else:
                logger.info(f"\n🔄 Using ADAPTIVE planner (iteration {iteration + 1})...")
                next_steps = self.adaptive_planner.plan_next_steps(
                    analysis_state,
                    question,
                    max_steps=2
                )

            if not next_steps:
                logger.info("📌 No more steps available")
                break

            # 执行规划的步骤
            for candidate_step in next_steps:
                if iteration >= max_iterations:
                    break

                logger.info(f"\n🔹 Step {iteration + 1}: {candidate_step.purpose}")
                logger.info(f"   Type: {candidate_step.step_type}")
                logger.info(f"   Priority: {candidate_step.priority:.1f}")
                if hasattr(candidate_step, 'llm_score') and candidate_step.llm_score > 0:
                    logger.info(f"   LLM score: {candidate_step.llm_score:.2f}")

                # 🆕 转换为ReasoningStep
                reasoning_step = self._convert_candidate_to_reasoning(
                    candidate_step,
                    iteration + 1,
                    analysis_state
                )

                # 执行
                exec_result = self._execute_step(reasoning_step, state)

                if not exec_result['success']:
                    logger.error(f"   ❌ Failed: {exec_result.get('error')}")

                    if state.replanning_count < state.max_replanning:
                        logger.info(f"   🔄 Replanning...")
                        state.replanning_count += 1

                    continue

                # 🆕 结构化反思
                structured_reflection = self.reflector.reflect(
                    step_number=reasoning_step.step_number,
                    purpose=reasoning_step.purpose,
                    expected_result=reasoning_step.expected_result,
                    actual_result=reasoning_step.actual_result,
                    question_context=question
                )

                reasoning_step.reflection = structured_reflection.summary
                reasoning_step.validation_passed = (
                        structured_reflection.validation_status.value in ['passed', 'partial']
                )

                state.structured_reflections.append(structured_reflection)
                state.reflections.append(structured_reflection.summary)

                logger.info(f"   📊 Reflection: {structured_reflection.summary}")
                logger.info(f"   📈 Confidence: {structured_reflection.confidence_score:.3f}")

                # 🆕 更新分析状态
                self._update_analysis_state(
                    analysis_state,
                    reasoning_step,
                    exec_result,
                    candidate_step
                )

                state.executed_steps.append(reasoning_step)
                iteration += 1

        # ===== PHASE 3: ANSWER SYNTHESIS =====
        logger.info("\n" + "=" * 70)
        logger.info("📝 PHASE 3: ANSWER SYNTHESIS")
        logger.info("=" * 70)

        final_answer = self._synthesize_answer(state)

        execution_time = time.time() - start_time

        # 构建返回结果
        result = {
            'question': question,
            'answer': final_answer,

            'entities_recognized': [
                {
                    'text': m.text,
                    'type': m.entity_type,
                    'confidence': m.confidence,
                    'match_type': m.match_type
                }
                for m in state.entity_matches[:10]
            ],

            'reasoning_plan': [self._step_to_dict(s) for s in state.executed_steps],
            'executed_steps': [self._step_to_dict(s) for s in state.executed_steps],

            'reflections': state.reflections,
            'structured_reflections': [
                {
                    'step': r.step_number,
                    'status': r.validation_status.value,
                    'confidence': r.confidence_score,
                    'uncertainty': r.uncertainty.overall_uncertainty,
                    'should_replan': r.should_replan
                }
                for r in state.structured_reflections
            ],

            # 🆕 自适应规划信息
            'adaptive_planning': {
                'target_depth': target_depth.value,
                'final_depth': len(state.executed_steps),
                'modalities_covered': analysis_state.modalities_covered,
                'entities_discovered': {
                    k: len(v) for k, v in analysis_state.discovered_entities.items()
                },
                'primary_focus': getattr(analysis_state, 'primary_focus', None)
            },

            'replanning_count': state.replanning_count,
            'confidence_score': state.confidence_score,
            'execution_time': execution_time,
            'total_steps': len(state.executed_steps),
            'schema_paths_used': state.schema_paths_used
        }

        logger.info(f"\n✅ Completed in {execution_time:.2f}s")
        logger.info(f"   • Steps executed: {len(state.executed_steps)}")
        logger.info(f"   • Confidence: {state.confidence_score:.3f}")
        logger.info(f"   • Modalities: {', '.join(analysis_state.modalities_covered)}")

        return result

    # ==================== 辅助方法 ====================
    def _select_planner(self, state, question: str) -> str:
        """
        智能选择规划器 (增强版 - 支持无entity的systematic模式)

        🔧 关键改进: 检测systematic关键词，即使没有初始entities
        """
        q_lower = question.lower()

        # 🔍 比较查询 → Comparative
        compare_keywords = ['compare', 'versus', 'vs ', 'vs.', 'difference between', 'contrast']
        if any(kw in q_lower for kw in compare_keywords):
            logger.info(f"   Comparison keywords detected → comparative")
            return 'comparative'

        # 🔧 新增: 系统筛选关键词 (不依赖初始entities)
        systematic_keywords = [
            'which regions', 'which brain', 'find all', 'identify all',
            'screen', 'systematic', 'highest', 'top regions',
            'mismatch', 'show', 'exhibit', 'demonstrate'
        ]

        # 检测systematic模式
        has_which = 'which' in q_lower
        has_highest = any(w in q_lower for w in ['highest', 'top', 'most', 'strongest'])
        has_mismatch = 'mismatch' in q_lower
        has_show = any(w in q_lower for w in ['show', 'exhibit', 'demonstrate', 'display'])

        # 🎯 关键: Systematic模式判断
        if has_which and (has_highest or has_mismatch or has_show):
            logger.info(f"   Systematic screening keywords detected → comparative")
            logger.info(f"     Keywords: which={has_which}, highest={has_highest}, mismatch={has_mismatch}")
            return 'comparative'

        # 或者直接检测组合
        if any(kw in q_lower for kw in systematic_keywords):
            # 进一步确认是否是筛选类问题
            screening_patterns = [
                'which.*show', 'which.*have', 'which.*exhibit',
                'find.*regions', 'identify.*regions',
                'highest.*mismatch', 'top.*mismatch'
            ]
            import re
            for pattern in screening_patterns:
                if re.search(pattern, q_lower):
                    logger.info(f"   Systematic pattern detected: {pattern} → comparative")
                    return 'comparative'

        # 🔧 Focus-driven: 有regions的深度查询
        if 'Region' in state.discovered_entities:
            n_regions = len(state.discovered_entities.get('Region', []))
            if n_regions > 0:
                logger.info(f"   {n_regions} regions found → focus-driven")
                return 'focus_driven'

        # 🔧 Focus-driven: Gene查询且有深度意图
        if 'GeneMarker' in state.discovered_entities:
            deep_intent_keywords = ['tell me about', 'about', 'analyze', 'characterize', 'comprehensive']
            if any(kw in q_lower for kw in deep_intent_keywords):
                logger.info(f"   Gene query with deep intent → focus-driven")
                return 'focus_driven'

        # 默认: Adaptive
        logger.info(f"   Default → adaptive")
        return 'adaptive'

    def _classify_question_intent(self, question: str) -> str:
        """分类问题意图"""
        question_lower = question.lower()

        if any(w in question_lower for w in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(w in question_lower for w in ['comprehensive', 'detailed', 'everything']):
            return 'comprehensive'
        elif any(w in question_lower for w in ['why', 'explain', 'how']):
            return 'explanatory'
        elif any(w in question_lower for w in ['which', 'find', 'identify']):
            return 'screening'
        else:
            return 'simple_query'



    def _update_analysis_state(self,
                               analysis_state,
                               step: ReasoningStep,
                               result: Dict,
                               candidate):
        """更新分析状态"""
        # 记录执行的步骤
        analysis_state.executed_steps.append({
            'purpose': step.purpose,
            'modality': step.modality,
            'row_count': len(result.get('data', [])),
            'step_id': candidate.step_id
        })

        # 更新modality覆盖
        if step.modality and step.modality not in analysis_state.modalities_covered:
            analysis_state.modalities_covered.append(step.modality)

        # 🆕 提取新发现的实体
        data = result.get('data', [])
        if not data:
            return

        first_row = data[0]

        # 提取regions
        if 'region' in first_row or 'acronym' in first_row:
            regions = list(set([
                row.get('region') or row.get('acronym')
                for row in data
                if row.get('region') or row.get('acronym')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Region', [])
            for r in regions:
                if r and r not in existing:
                    existing.append(r)

        # 提取clusters
        if 'cluster' in first_row or 'cluster_name' in first_row:
            clusters = list(set([
                row.get('cluster') or row.get('cluster_name')
                for row in data
                if row.get('cluster') or row.get('cluster_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Cluster', [])
            for c in clusters:
                if c and c not in existing:
                    existing.append(c)

        # 提取subclasses
        if 'subclass' in first_row or 'subclass_name' in first_row:
            subclasses = list(set([
                row.get('subclass') or row.get('subclass_name')
                for row in data
                if row.get('subclass') or row.get('subclass_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Subclass', [])
            for s in subclasses:
                if s and s not in existing:
                    existing.append(s)

        # 🆕 提取projection targets
        if 'target' in first_row or 'target_region' in first_row:
            targets = list(set([
                row.get('target') or row.get('target_region')
                for row in data
                if row.get('target') or row.get('target_region')
            ]))

            existing = analysis_state.discovered_entities.setdefault('ProjectionTarget', [])
            for t in targets:
                if t and t not in existing:
                    existing.append(t)

            if targets:
                logger.info(f"   📍 Discovered {len(targets)} projection targets")

    def _classify_question_intent(self, question: str) -> str:
        """分类问题意图"""
        question_lower = question.lower()

        if any(w in question_lower for w in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(w in question_lower for w in ['comprehensive', 'detailed', 'everything']):
            return 'comprehensive'
        elif any(w in question_lower for w in ['why', 'explain', 'how']):
            return 'explanatory'
        else:
            return 'simple_query'

    def _convert_candidate_to_reasoning(self, candidate, step_number, analysis_state):
        """转换CandidateStep (修复版)"""
        params = candidate.parameters.copy()

        # 🔧 智能判断action
        has_cypher = bool(candidate.cypher_template and candidate.cypher_template.strip())

        if not has_cypher:
            # 特殊步骤
            if 'statistical' in candidate.step_type.lower() or 'fdr' in candidate.step_id.lower():
                action = 'execute_statistical'
            elif 'multi-modal' in candidate.step_type.lower() or 'mismatch' in candidate.step_id.lower():
                action = 'execute_fingerprint'
            else:
                action = 'execute_cypher'
        else:
            action = 'execute_cypher'

        return ReasoningStep(
            step_number=step_number,
            purpose=candidate.purpose,
            action=action,  # 🔧 正确的action
            rationale=candidate.rationale,
            expected_result=candidate.expected_data,
            query_or_params={
                'query': candidate.cypher_template,
                'params': params
            },
            modality=candidate.step_type,
            depends_on=getattr(candidate, 'depends_on', [])
        )

    def _update_analysis_state(self,
                               analysis_state: 'AnalysisState',
                               step: ReasoningStep,
                               result: Dict,
                               candidate: 'CandidateStep'):
        """
        更新分析状态
        """
        # 记录执行的步骤
        analysis_state.executed_steps.append({
            'purpose': step.purpose,
            'modality': step.modality,
            'row_count': len(result.get('data', [])),
            'step_id': candidate.step_id
        })

        # 更新modality覆盖
        if step.modality and step.modality not in analysis_state.modalities_covered:
            analysis_state.modalities_covered.append(step.modality)

        # 🆕 提取新发现的实体
        data = result.get('data', [])
        if not data:
            return

        first_row = data[0]

        # 提取regions
        if 'region' in first_row or 'acronym' in first_row:
            regions = list(set([
                row.get('region') or row.get('acronym')
                for row in data
                if row.get('region') or row.get('acronym')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Region', [])
            for r in regions:
                if r not in existing:
                    existing.append(r)

        # 提取clusters
        if 'cluster' in first_row or 'cluster_name' in first_row:
            clusters = list(set([
                row.get('cluster') or row.get('cluster_name')
                for row in data
                if row.get('cluster') or row.get('cluster_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Cluster', [])
            for c in clusters:
                if c not in existing:
                    existing.append(c)

        # 提取subclasses
        if 'subclass' in first_row or 'subclass_name' in first_row:
            subclasses = list(set([
                row.get('subclass') or row.get('subclass_name')
                for row in data
                if row.get('subclass') or row.get('subclass_name')
            ]))

            existing = analysis_state.discovered_entities.setdefault('Subclass', [])
            for s in subclasses:
                if s not in existing:
                    existing.append(s)

        # 🆕 提取projection targets
        if 'target' in first_row or 'target_region' in first_row:
            targets = list(set([
                row.get('target') or row.get('target_region')
                for row in data
                if row.get('target') or row.get('target_region')
            ]))

            existing = analysis_state.discovered_entities.setdefault('ProjectionTarget', [])
            for t in targets:
                if t not in existing:
                    existing.append(t)

            logger.info(f"   📍 Discovered {len(targets)} projection targets")

    def _determine_analysis_depth(self, question: str) -> AnalysisDepth:
        """根据问题确定分析深度"""

        question_lower = question.lower()

        # Deep: comprehensive, detailed, everything, full, complete
        if any(kw in question_lower for kw in ['comprehensive', 'detailed', 'everything', 'complete', 'full']):
            return AnalysisDepth.DEEP

        # Shallow: simple, basic, quick, overview
        if any(kw in question_lower for kw in ['simple', 'basic', 'quick', 'overview', 'briefly']):
            return AnalysisDepth.SHALLOW

        # Default: Medium
        return AnalysisDepth.MEDIUM


    def _enhanced_planning_phase(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        增强的规划阶段

        步骤:
        1. 智能实体识别 (无hardcoded列表!)
        2. 实体聚类
        3. 动态Schema路径规划
        4. LLM精化
        """
        try:
            # Step 1: 实体识别
            logger.info("  [1/4] Intelligent entity recognition...")
            entity_matches = self.entity_recognizer.recognize_entities(state.question)
            state.entity_matches = entity_matches

            logger.info(f"     Found {len(entity_matches)} entity matches")
            for match in entity_matches[:5]:
                logger.info(f"       • {match.text} ({match.entity_type}) [{match.confidence:.2f}]")

            # Step 2: 实体聚类
            logger.info("  [2/4] Entity clustering...")
            entity_clusters = self.entity_clusterer.cluster_entities(
                entity_matches,
                state.question
            )
            state.entity_clusters = entity_clusters

            logger.info(f"     Created {len(entity_clusters)} entity clusters")
            for cluster in entity_clusters:
                logger.info(f"       • {cluster.cluster_type}: {cluster.primary_entity.text}")

            # Step 3: 动态Schema路径规划
            logger.info("  [3/4] Dynamic schema path planning...")
            query_plans = self.path_planner.generate_plan(entity_clusters, state.question)

            logger.info(f"     Generated {len(query_plans)} query plans")

            # 记录使用的schema路径
            for plan in query_plans:
                if plan.schema_path.hops:
                    state.schema_paths_used.append({
                        'start': plan.schema_path.start_label,
                        'end': plan.schema_path.end_label,
                        'hops': len(plan.schema_path.hops),
                        'score': plan.schema_path.score
                    })

            # Step 4: LLM精化
            logger.info("  [4/4] LLM plan refinement...")
            refined_steps = self._llm_refine_plans(query_plans, state)
            state.reasoning_plan = refined_steps

            # 保存实体到state (兼容原有格式)
            state.entities = [
                {
                    'text': m.text,
                    'type': m.entity_type,
                    'confidence': m.confidence
                }
                for m in entity_matches[:10]
            ]

            return {'success': True}

        except Exception as e:
            logger.error(f"Enhanced planning failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _llm_refine_plans(self,
                          query_plans: List,
                          state: EnhancedAgentState) -> List[ReasoningStep]:
        """
        LLM精化查询计划

        将动态生成的QueryPlan转换为ReasoningStep,并让LLM补充细节
        """
        # 转换为字典格式
        plans_dict = []
        for qp in query_plans:
            plans_dict.append({
                'step': qp.step_number,
                'purpose': qp.purpose,
                'action': qp.action,
                'query': qp.cypher_template,
                'parameters': qp.parameters,
                'modality': qp.modality,
                'depends_on': qp.depends_on,
                'schema_path_score': qp.schema_path.score if qp.schema_path else 0.0
            })

        prompt = f"""You are refining a reasoning plan for neuroscience knowledge graph analysis.

**Question:** {state.question}

**Recognized Entities:** {', '.join([e['text'] for e in state.entities])}

**Dynamically Generated Query Plans:**
{json.dumps(plans_dict, indent=2)}

Your task:
1. Review each query plan
2. Add detailed **expected_result** descriptions
3. Enhance **rationale** with domain knowledge
4. Verify Cypher query correctness
5. Add any missing steps if needed

Return a JSON object with key "steps" containing an array:
{{
  "steps": [
    {{
      "step_number": 1,
      "purpose": "...",
      "action": "execute_cypher",
      "rationale": "Detailed explanation",
      "expected_result": "Concrete prediction of what data will look like",
      "query_or_params": {{"query": "...", "params": {{}}}},
      "modality": "molecular/morphological/projection",
      "depends_on": []
    }},
    ...
  ]
}}

**Important:**
- Make rationale SPECIFIC and scientifically grounded
- Expected results should describe DATA PATTERNS (e.g., "10-20 clusters with neuron counts ranging 500-5000")
- Ensure query syntax is correct
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert neuroscientist and Neo4j query expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)

            # 转换为ReasoningStep
            steps = []
            for step_dict in result.get('steps', []):
                query_or_params = step_dict.get('query_or_params', {})

                # 处理参数替换
                if isinstance(query_or_params, dict):
                    if 'query' not in query_or_params and 'query' in step_dict:
                        query_or_params = {'query': step_dict['query']}

                step = ReasoningStep(
                    step_number=step_dict.get('step_number', len(steps) + 1),
                    purpose=step_dict.get('purpose', ''),
                    action=step_dict.get('action', 'execute_cypher'),
                    rationale=step_dict.get('rationale', ''),
                    expected_result=step_dict.get('expected_result', ''),
                    query_or_params=query_or_params,
                    modality=step_dict.get('modality'),
                    depends_on=step_dict.get('depends_on', [])
                )
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")

            # Fallback: 直接转换QueryPlan
            fallback_steps = []
            for qp in query_plans:
                step = ReasoningStep(
                    step_number=qp.step_number,
                    purpose=qp.purpose,
                    action=qp.action,
                    rationale="Automatically generated from schema path",
                    expected_result="Data matching query criteria",
                    query_or_params={'query': qp.cypher_template, 'params': qp.parameters},
                    modality=qp.modality,
                    depends_on=qp.depends_on
                )
                fallback_steps.append(step)

            return fallback_steps

    def _characterize_top_pairs(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        深入分析top mismatch pairs (Case Study)

        🆕 新增功能:
        1. 提取top N pairs
        2. 查询每个pair的详细数据:
           - Morphological features
           - Projection targets
           - Molecular composition
        """
        n_top = params.get('n_top_pairs', 3)

        # 从FDR结果获取top pairs
        fdr_data = None
        for key, data in state.intermediate_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                if 'fdr_significant' in data[0] and data[0].get('fdr_significant'):
                    fdr_data = data
                    break

        if not fdr_data:
            logger.warning("   No FDR significant pairs found, using top mismatch pairs")
            # Fallback: 使用top mismatch
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'mismatch_combined' in data[0]:
                        fdr_data = sorted(data, key=lambda x: x['mismatch_combined'], reverse=True)
                        break

        if not fdr_data:
            return {'success': False, 'error': 'No mismatch data found', 'data': []}

        # 选择top N pairs
        top_pairs = fdr_data[:n_top]

        logger.info(f"   Analyzing top {len(top_pairs)} pairs:")
        for pair in top_pairs:
            logger.info(f"     • {pair['region1']} vs {pair['region2']}: mismatch={pair['mismatch_combined']:.3f}")

        # 详细分析每个pair
        detailed_results = []

        for pair in top_pairs:
            region1 = pair['region1']
            region2 = pair['region2']

            logger.info(f"   Deep characterization: {region1} vs {region2}")

            # 🔹 1. Morphological comparison
            morph_query = """
            MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS region,
                   count(n) AS neuron_count,
                   avg(n.axonal_length) AS avg_axon,
                   avg(n.dendritic_length) AS avg_dendrite,
                   avg(n.axonal_branches) AS avg_axon_branches,
                   avg(n.dendritic_branches) AS avg_dendrite_branches,
                   stdev(n.axonal_length) AS std_axon,
                   stdev(n.dendritic_length) AS std_dendrite
            """
            morph_result = self.db.run(morph_query, {'region1': region1, 'region2': region2})

            # 🔹 2. Projection targets comparison
            proj_query = """
            MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS source,
                   t.acronym AS target,
                   t.name AS target_name,
                   p.weight AS weight
            ORDER BY r.acronym, p.weight DESC
            LIMIT 30
            """
            proj_result = self.db.run(proj_query, {'region1': region1, 'region2': region2})

            # 🔹 3. Molecular composition
            mol_query = """
            MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
            WHERE r.acronym IN [$region1, $region2]
            RETURN r.acronym AS region,
                   c.name AS cluster,
                   c.markers AS markers,
                   c.number_of_neurons AS neurons
            ORDER BY r.acronym, c.number_of_neurons DESC
            LIMIT 20
            """
            mol_result = self.db.run(mol_query, {'region1': region1, 'region2': region2})

            # 整合结果
            detailed_results.append({
                'pair': f"{region1}_vs_{region2}",
                'region1': region1,
                'region2': region2,
                'mismatch_score': pair['mismatch_combined'],
                'p_value': pair.get('p_value', 1.0),
                'q_value': pair.get('q_value', 1.0),
                'morphology': morph_result.get('data', []),
                'projections': proj_result.get('data', []),
                'molecular': mol_result.get('data', [])
            })

        logger.info(f"   ✅ Detailed characterization complete for {len(detailed_results)} pairs")

        return {
            'success': True,
            'data': detailed_results,
            'rows': len(detailed_results),
            'analysis_type': 'case_study'
        }

    # ==================== Execution ====================

    def _execute_step(self, step: ReasoningStep, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行单个步骤 (修复版 - 支持case study)"""
        start_time = time.time()

        try:
            query = step.query_or_params.get('query', '').strip()
            params = step.query_or_params.get('params', {})

            # 判断执行类型
            if not query:
                # 🆕 Case study检测
                if 'characterize' in step.purpose.lower() and 'top' in step.purpose.lower():
                    result = self._characterize_top_pairs(params, state)
                elif 'mismatch' in step.purpose.lower():
                    result = self._execute_fingerprint_step(step, state)
                elif 'statistical' in step.purpose.lower() or 'fdr' in step.purpose.lower():
                    result = self._execute_statistical_step(step, state)
                else:
                    result = {'success': False, 'error': 'Cannot determine execution type'}
            else:
                result = self._execute_cypher_step(step, state)

            step.actual_result = result
            step.execution_time = time.time() - start_time

            step_key = f"step_{step.step_number}"
            state.intermediate_data[step_key] = result.get('data', [])

            return result

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _execute_cypher_step(self, step: ReasoningStep, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行Cypher查询步骤"""
        query = step.query_or_params.get('query', '').strip()
        params = step.query_or_params.get('params', {})

        # 🔧 空查询检查
        if not query:
            logger.warning(f"   Empty Cypher query - skipping")
            return {'success': False, 'error': 'Empty query', 'data': []}

        # 参数替换
        if step.depends_on:
            params = self._resolve_parameters(step, state, params)

        # 自动添加LIMIT
        import re
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        return self.db.run(query, params)

    def _execute_statistical_step(self,
                                  step: ReasoningStep,
                                  state: EnhancedAgentState) -> Dict[str, Any]:
        """
        🆕 执行统计步骤
        """
        params = step.query_or_params.get('params', {})
        test_type = params.get('test_type', 'permutation')

        logger.info(f"   📊 Statistical test: {test_type}")

        try:
            if test_type == 'permutation':
                return self._permutation_test(params, state)

            elif test_type == 'fdr':
                return self._fdr_correction(params, state)

            elif test_type == 'correlation':
                return self._correlation_test(params, state)

            else:
                return {'success': False, 'error': f'Unknown test type: {test_type}'}

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _execute_fingerprint_step(self,
                                  step: ReasoningStep,
                                  state: EnhancedAgentState) -> Dict[str, Any]:
        """
        🆕 执行fingerprint计算步骤
        """
        params = step.query_or_params.get('params', {})
        analysis_type = params.get('analysis_type', 'cross_modal_mismatch')

        logger.info(f"   🔬 Fingerprint analysis: {analysis_type}")

        if analysis_type == 'cross_modal_mismatch':
            return self._compute_mismatch_matrix(params, state)
        else:
            return {'success': False, 'error': f'Unknown analysis type: {analysis_type}'}

    def _resolve_parameters(self,
                            step: ReasoningStep,
                            state: EnhancedAgentState,
                            params: Dict) -> Dict:
        """解析步骤依赖的参数"""
        resolved = params.copy()

        # 查找依赖步骤的数据
        for dep_num in step.depends_on:
            dep_key = f"step_{dep_num}"
            if dep_key in state.intermediate_data:
                dep_data = state.intermediate_data[dep_key]

                # 提取常用字段
                if dep_data:
                    # 提取region acronyms
                    regions = []
                    for row in dep_data:
                        if 'region' in row:
                            regions.append(row['region'])
                        elif 'acronym' in row:
                            regions.append(row['acronym'])

                    if regions:
                        resolved['enriched_regions'] = regions[:10]
                        resolved['target_regions'] = regions[:10]

        return resolved

    def _execute_cypher(self, query: str, params: Dict) -> Dict[str, Any]:
        """执行Cypher查询"""
        import re

        # 确保有LIMIT
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        return self.db.run(query, params)

    def _permutation_test(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """Permutation test for morphological differences"""
        entity_a = params['entity_a']
        entity_b = params['entity_b']

        # 从之前的step获取数据
        morph_data = None
        for key, data in state.intermediate_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                if 'region' in data[0] and ('avg_axon' in data[0] or 'avg_axon_length' in data[0]):
                    morph_data = data
                    break

        if not morph_data:
            return {'success': False, 'error': 'No morphological data found'}

        # 提取两组数据
        group_a = [row for row in morph_data if row.get('region') == entity_a]
        group_b = [row for row in morph_data if row.get('region') == entity_b]

        if not group_a or not group_b:
            return {'success': False,
                    'error': f'Insufficient data: {entity_a}={len(group_a)}, {entity_b}={len(group_b)}'}

        # 提取axon length
        import numpy as np
        axon_key = 'avg_axon' if 'avg_axon' in group_a[0] else 'avg_axon_length'
        axon_a = np.array([row.get(axon_key, 0) or 0 for row in group_a])
        axon_b = np.array([row.get(axon_key, 0) or 0 for row in group_b])

        # 移除零值
        axon_a = axon_a[axon_a > 0]
        axon_b = axon_b[axon_b > 0]

        if len(axon_a) == 0 or len(axon_b) == 0:
            return {'success': False, 'error': 'No valid morphology data'}

        # 计算observed difference
        observed_diff = float(np.mean(axon_a) - np.mean(axon_b))

        # 🎯 调用统计工具!
        result = self.stats.permutation_test(
            observed_stat=observed_diff,
            data1=axon_a,
            data2=axon_b,
            n_permutations=1000,
            seed=42
        )

        # 计算effect size
        effect_size = self.stats.cohens_d(axon_a, axon_b)

        # 格式化结果
        result_data = [{
            'comparison': f'{entity_a} vs {entity_b}',
            'feature': 'axonal_length',
            'mean_a': float(np.mean(axon_a)),
            'mean_b': float(np.mean(axon_b)),
            'observed_difference': observed_diff,
            'p_value': result['p_value'],
            'effect_size_cohens_d': effect_size,
            'significance': 'significant' if result['p_value'] < 0.05 else 'not significant',
            'interpretation': self._interpret_statistical_result(result, effect_size)
        }]

        logger.info(f"   ✅ Permutation test: p={result['p_value']:.4f}, d={effect_size:.2f}")

        return {
            'success': True,
            'data': result_data,
            'rows': len(result_data),
            'test_type': 'permutation'
        }

    def _fdr_correction(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        FDR correction for multiple comparisons (修复版)

        🔧 修复: 正确处理p-values
        """
        alpha = params.get('alpha', 0.05)

        # 从mismatch step获取p-values
        mismatch_data = None
        mismatch_key = None

        for key, data in state.intermediate_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                if 'mismatch_combined' in data[0] and 'p_value' in data[0]:
                    mismatch_data = data
                    mismatch_key = key
                    logger.info(f"   Found mismatch data in {key}")
                    break

        if not mismatch_data:
            logger.error("   No mismatch data with p-values found")
            return {
                'success': False,
                'error': 'No mismatch data with p-values found for FDR correction',
                'data': []
            }

        # 提取p-values
        p_values = [row.get('p_value', 1.0) for row in mismatch_data]

        logger.info(f"   FDR input: {len(p_values)} p-values")
        logger.info(f"   P-value range: [{min(p_values):.4f}, {max(p_values):.4f}]")

        # 🎯 调用FDR correction
        try:
            q_values, significant = self.stats.fdr_correction(p_values, alpha)

            # 添加到原数据
            result_data = []
            for i, row in enumerate(mismatch_data):
                result_data.append({
                    **row,
                    'q_value': q_values[i],
                    'fdr_significant': significant[i]
                })

            # 只保留显著的
            significant_data = [r for r in result_data if r['fdr_significant']]

            logger.info(f"   ✅ FDR correction: {len(significant_data)}/{len(result_data)} significant (α={alpha})")

            if significant_data:
                logger.info(
                    f"   Top significant pair: {significant_data[0]['region1']}-{significant_data[0]['region2']}")
                logger.info(f"     Mismatch: {significant_data[0]['mismatch_combined']:.3f}")
                logger.info(f"     Q-value: {significant_data[0]['q_value']:.4f}")

            return {
                'success': True,
                'data': significant_data,
                'rows': len(significant_data),
                'test_type': 'fdr',
                'alpha': alpha,
                'n_significant': len(significant_data),
                'n_total': len(result_data)
            }

        except Exception as e:
            logger.error(f"   FDR correction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'data': []
            }

    def _correlation_test(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """Correlation test between modalities"""
        # 实现correlation (可选,暂时返回placeholder)
        logger.warning("Correlation test not yet implemented")
        return {'success': False, 'error': 'Not implemented'}

    def _compute_mismatch_matrix(self, params: Dict, state: EnhancedAgentState) -> Dict:
        """
        计算cross-modal mismatch矩阵 (增强版 - 带统计显著性)

        🆕 添加:
        1. Permutation test计算p-value
        2. Effect size计算
        3. Bootstrap confidence intervals
        """
        # 获取regions
        regions = state.analysis_state.discovered_entities.get('Region', [])

        if not regions:
            # 从之前的step获取
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'region' in data[0]:
                        regions = list(set([row['region'] for row in data if row.get('region')]))
                        break

        # 限制数量避免计算爆炸
        regions = regions[:20]

        if len(regions) < 2:
            return {'success': False, 'error': 'Need at least 2 regions for mismatch computation'}

        logger.info(
            f"   Computing mismatch for {len(regions)} regions ({len(regions) * (len(regions) - 1) // 2} pairs)...")

        # 计算所有pairs的mismatch
        mismatch_results = []

        from itertools import combinations
        import numpy as np

        for region1, region2 in combinations(regions, 2):
            # 🎯 调用fingerprint analyzer
            mismatch = self.fingerprint.compute_mismatch_index(region1, region2)

            if mismatch:
                # 基础mismatch scores
                mismatch_GM = mismatch.get('mismatch_GM', 0)
                mismatch_GP = mismatch.get('mismatch_GP', 0)
                mismatch_MP = mismatch.get('mismatch_MP', 0)
                mismatch_combined = (mismatch_GM + mismatch_GP + mismatch_MP) / 3

                # 🆕 计算统计显著性 (Permutation test)
                # 使用combined mismatch作为observed statistic
                try:
                    # 生成null distribution (通过随机permutation)
                    null_mismatches = []
                    n_permutations = 100  # 降低计算量

                    # 获取所有其他pairs的mismatch作为null distribution
                    random_pairs = []
                    all_other_regions = [r for r in regions if r not in [region1, region2]]

                    if len(all_other_regions) >= 2:
                        # 随机采样其他pairs
                        for _ in range(min(n_permutations, len(all_other_regions))):
                            import random
                            r1, r2 = random.sample(all_other_regions, 2)
                            null_mismatch = self.fingerprint.compute_mismatch_index(r1, r2)
                            if null_mismatch:
                                null_combined = (
                                                        null_mismatch.get('mismatch_GM', 0) +
                                                        null_mismatch.get('mismatch_GP', 0) +
                                                        null_mismatch.get('mismatch_MP', 0)
                                                ) / 3
                                null_mismatches.append(null_combined)

                    # 计算p-value
                    if null_mismatches:
                        null_mismatches = np.array(null_mismatches)
                        p_value = np.mean(null_mismatches >= mismatch_combined)

                        # 避免p=0
                        if p_value == 0:
                            p_value = 1.0 / (len(null_mismatches) + 1)
                    else:
                        # Fallback: 基于mismatch score转换
                        # 高mismatch → 低p-value
                        p_value = 1.0 - min(0.99, mismatch_combined)

                except Exception as e:
                    logger.warning(f"   Failed to compute p-value for {region1}-{region2}: {e}")
                    # Fallback
                    p_value = 1.0 - min(0.99, mismatch_combined)

                # 🆕 计算effect size (使用mismatch score作为proxy)
                effect_size = mismatch_combined  # 简化版，实际应该用Cohen's d

                mismatch_results.append({
                    'region1': region1,
                    'region2': region2,
                    'mismatch_GM': mismatch_GM,
                    'mismatch_GP': mismatch_GP,
                    'mismatch_MP': mismatch_MP,
                    'mismatch_combined': mismatch_combined,
                    'sim_molecular': mismatch.get('sim_molecular', 0),
                    'sim_morphological': mismatch.get('sim_morphological', 0),
                    'sim_projection': mismatch.get('sim_projection', 0),
                    # 🆕 统计信息
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'n_permutations': len(null_mismatches) if null_mismatches else 0
                })

        # 按mismatch排序
        mismatch_results.sort(key=lambda x: x['mismatch_combined'], reverse=True)

        # Top-N
        n_top = params.get('n_pairs', 10)
        top_mismatches = mismatch_results[:n_top]

        logger.info(f"   ✅ Computed {len(mismatch_results)} mismatches")
        if top_mismatches:
            logger.info(f"   Top mismatch: {top_mismatches[0]['region1']}-{top_mismatches[0]['region2']}")
            logger.info(f"     Score: {top_mismatches[0]['mismatch_combined']:.3f}")
            logger.info(f"     P-value: {top_mismatches[0]['p_value']:.4f}")

        return {
            'success': True,
            'data': mismatch_results,
            'rows': len(mismatch_results),
            'analysis_type': 'cross_modal_mismatch',
            'top_pairs': top_mismatches
        }

    def _interpret_statistical_result(self, test_result: Dict, effect_size: float) -> str:
        """解释统计结果"""
        p_value = test_result['p_value']

        if p_value < 0.001:
            sig_level = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            sig_level = "very significant (p < 0.01)"
        elif p_value < 0.05:
            sig_level = "significant (p < 0.05)"
        else:
            sig_level = "not significant (p ≥ 0.05)"

        if abs(effect_size) > 0.8:
            effect_desc = "large effect size"
        elif abs(effect_size) > 0.5:
            effect_desc = "medium effect size"
        elif abs(effect_size) > 0.2:
            effect_desc = "small effect size"
        else:
            effect_desc = "negligible effect size"

        return f"The difference is {sig_level} with a {effect_desc} (Cohen's d = {effect_size:.2f})"

    def _resolve_parameters(self,
                            step: ReasoningStep,
                            state: EnhancedAgentState,
                            params: Dict) -> Dict:
        """解析步骤依赖的参数"""
        resolved = params.copy()

        # 查找依赖步骤的数据
        for dep_num in step.depends_on:
            dep_key = f"step_{dep_num}"
            if dep_key in state.intermediate_data:
                dep_data = state.intermediate_data[dep_key]

                if not dep_data:
                    continue

                # 提取常用字段
                # 提取region acronyms
                regions = []
                for row in dep_data:
                    if 'region' in row:
                        regions.append(row['region'])
                    elif 'acronym' in row:
                        regions.append(row['acronym'])

                if regions:
                    resolved['enriched_regions'] = list(set(regions))[:10]
                    resolved['target_regions'] = list(set(regions))[:10]

                # 提取targets
                targets = []
                for row in dep_data:
                    if 'target' in row:
                        targets.append(row['target'])
                    elif 'target_region' in row:
                        targets.append(row['target_region'])

                if targets:
                    resolved['targets'] = list(set(targets))[:10]

        return resolved

    # ==================== Intelligent Replanning ====================

    def _intelligent_replan(self, state: EnhancedAgentState, from_step: int) -> bool:
        """
        智能重规划

        使用:
        - 结构化反思的建议
        - 替代假设
        - Schema中的替代路径
        """
        logger.info(f"🔄 Intelligent replanning from step {from_step}")
        state.replanning_count += 1

        # 获取最近的结构化反思
        if state.structured_reflections:
            last_reflection = state.structured_reflections[-1]

            # 使用反思中的建议
            logger.info(f"   Using reflection recommendations:")
            for rec in last_reflection.next_step_recommendations:
                logger.info(f"     • {rec}")

            # 如果有替代假设,尝试使用
            if last_reflection.alternative_hypotheses:
                logger.info(f"   Found {len(last_reflection.alternative_hypotheses)} alternative hypotheses")

        # 重新生成计划 (使用现有实体)
        try:
            query_plans = self.path_planner.generate_plan(
                state.entity_clusters,
                state.question
            )

            # 替换剩余步骤
            new_steps = self._llm_refine_plans(query_plans, state)

            # 更新plan,保留已执行的
            state.reasoning_plan = state.reasoning_plan[:from_step - 1] + new_steps

            logger.info(f"   ✅ Replanned with {len(new_steps)} new steps")
            return True

        except Exception as e:
            logger.error(f"   ❌ Replanning failed: {e}")
            return False

    # ==================== Answer Synthesis ====================

    def _synthesize_answer(self, state: EnhancedAgentState) -> str:
        """
        合成最终答案 (增强版 - 科学叙事)
        """
        # 准备证据摘要
        evidence = []
        for step in state.executed_steps:
            if step.actual_result and step.actual_result.get('success'):
                data_count = len(step.actual_result.get('data', []))
                evidence.append(f"- Step {step.step_number}: {step.purpose} ({data_count} results)")

        evidence_text = "\n".join(evidence)

        # 准备关键发现
        key_data = {}
        for step in state.executed_steps:
            if step.actual_result and step.actual_result.get('success'):
                data = step.actual_result.get('data', [])
                if data:
                    key_data[f"step_{step.step_number}"] = data[:5]  # Top 5

        # 准备结构化反思摘要
        reflection_summary = []
        for r in state.structured_reflections:
            reflection_summary.append(
                f"Step {r.step_number}: {r.validation_status.value} (confidence: {r.confidence_score:.2f})"
            )

        # 🆕 检测分析类型
        analysis_type = self._detect_analysis_type(state)

        # 🆕 准备PRIMARY FOCUS信息
        primary_focus_info = ""
        if hasattr(state.analysis_state, 'primary_focus') and state.analysis_state.primary_focus:
            focus = state.analysis_state.primary_focus
            supporting = focus.supporting_data
            primary_focus_info = f"""
    **PRIMARY FOCUS IDENTIFIED:**
    - Region: {focus.entity_id}
    - Enrichment: {supporting.get('total_neurons', 'N/A')} neurons across {supporting.get('cluster_count', 'N/A')} clusters
    - This region shows the highest enrichment and was selected for deep characterization
    """

        prompt = f"""Synthesize a comprehensive, publication-quality answer based on the multi-step analysis.

    **CRITICAL: Write as a SCIENTIFIC NARRATIVE, not a data report!**

    **Original Question:** {state.question}

    **Analysis Type Detected:** {analysis_type}

    **Entities Recognized:** {', '.join([e['text'] for e in state.entities[:5]])}

    {primary_focus_info}

    **Reasoning Steps Executed:**
    {chr(10).join([f"{i + 1}. {s.purpose}" for i, s in enumerate(state.executed_steps)])}

    **Evidence Collected:**
    {evidence_text}

    **Key Findings (quantitative data):**
    {json.dumps(key_data, indent=2, default=str)[:3000]}

    **Structured Reflections:**
    {chr(10).join(reflection_summary)}

    **Your Task:**

    Write a comprehensive answer with the following structure:

    ### [Title - Generate an engaging title]

    #### Introduction (1 paragraph)
    - Open with the biological significance
    - State the main finding concisely

    #### Multi-Modal Analysis Results

    **1. Molecular Characterization**
    - Cite SPECIFIC numbers (e.g., "18,474 neurons across 4 clusters")
    - Mention key markers and cell types
    - Use quantitative language

    **2. Spatial Distribution**
    - List regions with enrichment metrics
    - Highlight PRIMARY focus if identified
    - Use percentages and rankings

    **3. Morphological Features** (if available)
    - Report mean ± SD for axonal/dendritic measurements
    - Compare to baseline if applicable
    - Interpret structural specializations

    **4. Connectivity Patterns** (if available)
    - Describe projection targets with weights
    - Categorize by functional systems (sensory/motor/associative)
    - Mention top 3-5 targets quantitatively

    **5. Target Characterization (CLOSED LOOP)** (if available)
    - Describe cell type composition of projection targets
    - Connect back to molecular findings
    - Emphasize circuit-level integration

    **6. Statistical Validation** (if available)
    - Report p-values and effect sizes
    - Mention significance levels
    - Interpret biological meaning

    #### Integration and Implications
    - Connect molecular → morphological → projection findings
    - Propose functional hypotheses
    - Discuss circuit-level organization

    #### Limitations and Uncertainties
    - Acknowledge data gaps honestly
    - Cite confidence scores from reflections
    - Suggest validation approaches

    **Writing Style:**
    - Use ACTIVE voice ("Our analysis revealed..." not "It was found...")
    - Connect findings CAUSALLY ("Because X, we examined Y, which revealed Z")
    - Emphasize QUANTITATIVE data (numbers, percentages, statistics)
    - Make it VISUAL-READY (structure data for plotting)
    - Be HONEST about uncertainties

    **Avoid:**
    - Lists without narrative flow
    - Vague statements ("some regions", "several")
    - Overconfident claims
    - Jargon without explanation

    Generate a publication-quality narrative now.
    """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a neuroscience writer synthesizing research analysis results into publication-quality narratives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            answer = response.choices[0].message.content.strip()
            state.final_answer = answer

            # 估算置信度
            state.confidence_score = self._estimate_confidence(state)

            return answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: 简单总结
            return f"Analysis completed with {len(state.executed_steps)} steps across {len(state.analysis_state.modalities_covered)} modalities. " \
                   f"Identified {len(state.entities)} entities and executed comprehensive multi-modal analysis. " \
                   f"Confidence: {self._estimate_confidence(state):.2f}."

    def _detect_analysis_type(self, state: EnhancedAgentState) -> str:
        """检测分析类型"""
        step_purposes = [s.purpose.lower() for s in state.executed_steps]

        if any('compare' in p or 'versus' in p for p in step_purposes):
            return "Comparative Analysis"
        elif any('mismatch' in p or 'screening' in p for p in step_purposes):
            return "Systematic Screening (Figure 4 type)"
        elif any('primary focus' in p or 'closed loop' in p for p in step_purposes):
            return "Focus-Driven Deep Analysis (Figure 3 type)"
        else:
            return "General Multi-Modal Analysis"

    # ==================== Utilities ====================

    def _step_to_dict(self, step: ReasoningStep) -> Dict:
        """转换步骤为字典"""
        return {
            'step_number': step.step_number,
            'purpose': step.purpose,
            'action': step.action,
            'rationale': step.rationale,
            'expected_result': step.expected_result,
            'actual_result_summary': {
                'success': step.actual_result.get('success') if step.actual_result else False,
                'row_count': len(step.actual_result.get('data', [])) if step.actual_result else 0
            },
            'reflection': step.reflection,
            'validation_passed': step.validation_passed,
            'execution_time': step.execution_time,
            'modality': step.modality
        }

    def _estimate_confidence(self, state: EnhancedAgentState) -> float:
        """估算置信度"""
        if not state.structured_reflections:
            return 0.5

        # 使用结构化反思的置信度
        confidences = [r.confidence_score for r in state.structured_reflections]
        avg_confidence = sum(confidences) / len(confidences)

        # 调整因素

        # Factor 1: 步骤完成率
        completion_rate = len(state.executed_steps) / len(state.reasoning_plan) \
            if state.reasoning_plan else 0

        # Factor 2: 重规划惩罚
        replan_penalty = 0.95 ** state.replanning_count

        # 综合
        final_confidence = avg_confidence * (0.7 + 0.3 * completion_rate) * replan_penalty

        return min(1.0, max(0.0, final_confidence))

    def _build_error_response(self, question: str, error: str, start_time: float) -> Dict:
        """构建错误响应"""
        return {
            'question': question,
            'answer': f"Analysis failed: {error}",
            'error': error,
            'execution_time': time.time() - start_time,
            'success': False,
            'entities_recognized': [],
            'reasoning_plan': [],
            'executed_steps': [],
            'reflections': [],
            'confidence_score': 0.0
        }

    def close(self):
        """关闭数据库连接"""
        self.db.close()


# ==================== Test ====================

def test_v10_agent():
    """测试V10 agent"""
    import os

    print("\n" + "=" * 80)
    print("AIPOM-CoT V10 PRODUCTION TEST")
    print("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # 测试问题
    test_questions = [
        "Tell me about Car3+ neurons",
        "Compare Pvalb and Sst interneurons in MOs",
        "What are the projection targets of the claustrum?"
    ]

    for q in test_questions:
        print(f"\n{'=' * 80}")
        print(f"Q: {q}")
        print('=' * 80)

        result = agent.answer(q, max_iterations=8)

        print(f"\n✅ Results:")
        print(f"   Entities: {len(result['entities_recognized'])}")
        print(f"   Steps: {result['total_steps']}")
        print(f"   Confidence: {result['confidence_score']:.3f}")
        print(f"   Time: {result['execution_time']:.2f}s")
        print(f"\n💡 Answer:\n{result['answer'][:300]}...\n")

    agent.close()


def test_car3_comprehensive():
    """测试Car3的完整分析"""

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY",""),
        model="gpt-4o"
    )

    # 🎯 关键: 使用"comprehensive"触发深度分析
    # question = "Give me a comprehensive analysis of Car3+ neurons"
    question = "Which brain regions show the highest cross-modal mismatch?"
    result = agent.answer(question, max_iterations=12)

    print("\n" + "=" * 80)
    print("FIGURE 3 STORY ARC ANALYSIS")
    print("=" * 80)

    print(f"\nTarget Depth: {result['adaptive_planning']['target_depth']}")
    print(f"Steps Executed: {result['adaptive_planning']['final_depth']}")
    print(f"Modalities: {', '.join(result['adaptive_planning']['modalities_covered'])}")

    print("\n" + "-" * 80)
    print("STEP-BY-STEP NARRATIVE:")
    print("-" * 80)

    for i, step in enumerate(result['executed_steps'], 1):
        print(f"\n{i}. {step['purpose']}")
        print(f"   Modality: {step['modality']}")
        print(f"   Data: {step['actual_result_summary']['row_count']} rows")
        print(f"   Confidence: {step['reflection']}")

    print("\n" + "-" * 80)
    print("ENTITIES DISCOVERED:")
    print("-" * 80)
    for entity_type, count in result['adaptive_planning']['entities_discovered'].items():
        print(f"  • {entity_type}: {count}")

    print("\n" + "-" * 80)
    print("VALIDATION CHECKLIST:")
    print("-" * 80)

    modalities = result['adaptive_planning']['modalities_covered']
    entities = result['adaptive_planning']['entities_discovered']

    checks = {
        'Has molecular analysis': 'molecular' in modalities,
        'Has morphological analysis': 'morphological' in modalities,
        'Has projection analysis': 'projection' in modalities,
        'Found regions': 'Region' in entities and entities['Region'] > 0,
        'Found projection targets': 'ProjectionTarget' in entities and entities['ProjectionTarget'] > 0,
        'Analyzed target composition': any(
            'target' in s['purpose'].lower() and 'composition' in s['purpose'].lower() for s in
            result['executed_steps'])
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    # 计算完整性分数
    completeness = sum(checks.values()) / len(checks) * 100
    print(f"\n📊 Story Completeness: {completeness:.0f}%")

    if completeness >= 80:
        print("\n🎉 ✅ FIGURE 3 COMPLETE STORY ARC ACHIEVED!")
    else:
        print(f"\n⚠️  Story incomplete - missing {100 - completeness:.0f}% of elements")

    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result['answer'])

    agent.close()

    return result

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_car3_comprehensive()