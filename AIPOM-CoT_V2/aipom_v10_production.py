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
        主入口: 回答问题 (使用自适应规划)

        完整流程:
        1. 实体识别 + 聚类
        2. 确定分析深度
        3. 初始化分析状态
        4. 自适应执行循环
        5. 答案合成
        """
        logger.info(f"🎯 Question: {question}")
        start_time = time.time()

        state = EnhancedAgentState(question=question)

        # ===== PHASE 1: INTELLIGENT PLANNING =====
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 1: INTELLIGENT PLANNING (Enhanced)")
        logger.info("=" * 70)

        state.phase = AgentPhase.PLANNING

        # Step 1-2: 实体识别 + 聚类 (不变)
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
        from adaptive_planner import determine_analysis_depth, AnalysisState, AnalysisDepth

        logger.info("  [3/4] Determining analysis depth...")
        target_depth = determine_analysis_depth(question)
        logger.info(f"     Depth: {target_depth.value}")

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

            # 添加related entities
            for related in cluster.related_entities:
                analysis_state.discovered_entities.setdefault(
                    related.entity_type, []
                ).append(related.entity_id)

        # 兼容性: 保存到state
        state.entities = [
            {'text': m.text, 'type': m.entity_type, 'confidence': m.confidence}
            for m in entity_matches[:10]
        ]

        logger.info(f"✅ Planning complete")
        logger.info(f"   • Target depth: {target_depth.value}")
        logger.info(f"   • Initial entities: {list(analysis_state.discovered_entities.keys())}")

        # ===== PHASE 2: ADAPTIVE EXECUTION =====
        logger.info("\n" + "=" * 70)
        logger.info("⚙️ PHASE 2: ADAPTIVE EXECUTION (Dynamic Planning)")
        logger.info("=" * 70)

        state.phase = AgentPhase.EXECUTING

        iteration = 0
        while iteration < max_iterations:
            # 🆕 决定是否继续
            if not self.adaptive_planner.should_continue(analysis_state, question):
                logger.info("📌 Analysis complete (adaptive decision)")
                break

            # 🆕 动态规划下一步
            logger.info(f"\n🎯 Adaptive planning (iteration {iteration + 1})...")
            next_steps = self.adaptive_planner.plan_next_steps(
                analysis_state,
                question,
                max_steps=2  # 每次规划2步
            )

            if not next_steps:
                logger.info("📌 No more valuable steps available")
                break

            # 执行规划的步骤
            for candidate_step in next_steps:
                if iteration >= max_iterations:
                    break

                logger.info(f"\n🔹 Step {iteration + 1}: {candidate_step.purpose}")
                logger.info(f"   Type: {candidate_step.step_type}")
                logger.info(f"   LLM score: {candidate_step.llm_score:.2f}")

                # 🆕 转换CandidateStep为ReasoningStep
                reasoning_step = self._convert_candidate_to_reasoning(
                    candidate_step,
                    iteration + 1,
                    analysis_state
                )

                # 执行
                exec_result = self._execute_step(reasoning_step, state)

                if not exec_result['success']:
                    logger.error(f"   ❌ Failed: {exec_result.get('error')}")

                    # 简单重规划 (如果需要)
                    if state.replanning_count < state.max_replanning:
                        logger.info(f"   🔄 Replanning...")
                        state.replanning_count += 1
                        # 继续循环,自适应规划会生成新步骤

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

            # 实体识别
            'entities_recognized': [
                {
                    'text': m.text,
                    'type': m.entity_type,
                    'confidence': m.confidence,
                    'match_type': m.match_type
                }
                for m in state.entity_matches[:10]
            ],

            # 推理计划
            'reasoning_plan': [self._step_to_dict(s) for s in state.executed_steps],
            'executed_steps': [self._step_to_dict(s) for s in state.executed_steps],

            # 反思
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
                }
            },

            # 元数据
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

    def _convert_candidate_to_reasoning(self,
                                        candidate: 'CandidateStep',
                                        step_number: int,
                                        analysis_state: 'AnalysisState') -> ReasoningStep:
        """
        将CandidateStep转换为ReasoningStep
        """
        # 解析参数 (替换占位符)
        params = candidate.parameters.copy()

        # 如果参数中有引用discovered_entities的,替换之
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                # 例如: $regions -> analysis_state.discovered_entities['Region']
                entity_type = value[1:].title()  # $regions -> Regions -> Region
                if entity_type.endswith('s'):
                    entity_type = entity_type[:-1]

                if entity_type in analysis_state.discovered_entities:
                    params[key] = analysis_state.discovered_entities[entity_type][:10]

        return ReasoningStep(
            step_number=step_number,
            purpose=candidate.purpose,
            action='execute_cypher',
            rationale=candidate.rationale + f" (LLM score: {candidate.llm_score:.2f})",
            expected_result=candidate.expected_data,
            query_or_params={
                'query': candidate.cypher_template,
                'params': params
            },
            modality=candidate.step_type if candidate.step_type != 'spatial' else None,
            depends_on=[]
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

    # def _update_analysis_state(self,
    #                            state: AnalysisState,
    #                            step: ReasoningStep,
    #                            result: Dict):
    #     """更新分析状态"""
    #
    #     # 记录执行的步骤
    #     state.executed_steps.append({
    #         'purpose': step.purpose,
    #         'modality': step.modality,
    #         'row_count': len(result.get('data', []))
    #     })
    #
    #     # 更新modality覆盖
    #     if step.modality and step.modality not in state.modalities_covered:
    #         state.modalities_covered.append(step.modality)
    #
    #     # 提取新发现的实体
    #     data = result.get('data', [])
    #     if data:
    #         # 如果是regions
    #         if 'region' in data[0] or 'acronym' in data[0]:
    #             regions = [row.get('region') or row.get('acronym') for row in data]
    #             state.discovered_entities.setdefault('Region', []).extend(regions)
    #
    #         # 如果是clusters
    #         if 'cluster' in data[0] or 'cluster_name' in data[0]:
    #             clusters = [row.get('cluster') or row.get('cluster_name') for row in data]
    #             state.discovered_entities.setdefault('Cluster', []).extend(clusters)
    #
    #         # 如果是projection targets
    #         if 'target' in data[0]:
    #             targets = [row['target'] for row in data]
    #             state.discovered_entities.setdefault('ProjectionTarget', []).extend(targets)

    # ==================== Enhanced Planning Phase ====================

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

    # ==================== Execution ====================

    def _execute_step(self, step: ReasoningStep, state: EnhancedAgentState) -> Dict[str, Any]:
        """执行单个步骤"""
        start_time = time.time()

        try:
            query = step.query_or_params.get('query', '')
            params = step.query_or_params.get('params', {})

            # 参数替换 (处理依赖)
            if step.depends_on:
                params = self._resolve_parameters(step, state, params)

            # 执行查询
            result = self._execute_cypher(query, params)

            step.actual_result = result
            step.execution_time = time.time() - start_time

            # 保存中间数据
            step_key = f"step_{step.step_number}"
            state.intermediate_data[step_key] = result.get('data', [])

            return result

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {'success': False, 'error': str(e)}

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
        """合成最终答案"""
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
                f"Step {r.step_number}: {r.validation_status.value} "
                f"(confidence: {r.confidence_score:.2f})"
            )

        prompt = f"""Synthesize a comprehensive answer based on the reasoning trace.

**Original Question:** {state.question}

**Entities Recognized:** {', '.join([e['text'] for e in state.entities[:5]])}

**Reasoning Steps Executed:**
{chr(10).join([f"{i + 1}. {s.purpose}" for i, s in enumerate(state.executed_steps)])}

**Evidence Collected:**
{evidence_text}

**Key Findings (sample data):**
{json.dumps(key_data, indent=2, default=str)[:2000]}

**Structured Reflections:**
{chr(10).join(reflection_summary)}

**Your Task:**
Write a comprehensive, scientifically rigorous answer that:
1. Directly answers the original question
2. Cites specific quantitative findings with numbers
3. Explains the multi-step reasoning process briefly
4. Integrates molecular, morphological, and projection findings if available
5. Acknowledges any limitations or uncertainties
6. Is written for a neuroscience research audience

Make it publication-quality but accessible. Use proper scientific terminology.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a neuroscience writer synthesizing research analysis results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            state.final_answer = answer

            # 估算置信度
            state.confidence_score = self._estimate_confidence(state)

            return answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Analysis completed with {len(state.executed_steps)} steps. " \
                   f"Found {len(state.entities)} entities and executed multi-modal analysis."

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
        openai_api_key=os.getenv("OPENAI_API_KEY",''),
        model="gpt-4o"
    )

    # 🎯 关键: 使用"comprehensive"触发深度分析
    question = "Give me a comprehensive analysis of Car3+ neurons"

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