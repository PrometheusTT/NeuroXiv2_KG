"""
TPAR Engine - Think-Plan-Act-Reflect Core Loop
===============================================
NeuroXiv-KG Agent的核心推理引擎

完整实现Figure 2C的TPAR循环：
1. Think: 解析问题，识别实体和意图
2. Plan: 选择规划器，生成步骤
3. Act: 执行步骤，收集证据
4. Reflect: 评估结果，决定下一步
5. Synthesize: 综合生成答案

Author: Lijun
Date: 2025-01
"""

import json
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from core_structures import (
    AnalysisState, AnalysisDepth, Modality, QuestionIntent,
    PlannerType, ReflectionDecision, EvidenceRecord, EvidenceBuffer,
    Entity, EntityCluster, CandidateStep, StructuredReflection,
    SessionMemory, AgentConfig
)

from llm_intelligence import (
    LLMClient, LLMIntentClassifier, LLMEntityRecognizer,
    IntentClassification
)

from adaptive_planner import AdaptivePlanner, SchemaGraph

from llm_reflector import LLMReflector, ReflectionAggregator

logger = logging.getLogger(__name__)


class TPAREngine:
    """
    TPAR推理引擎 - NeuroXiv-KG Agent的核心

    关键特性：
    1. LLM深度参与每个阶段
    2. 统一的证据缓冲
    3. 预算控制
    4. 智能终止
    5. 闭环分析支持
    """

    def __init__(self,
                 db_executor,
                 llm_client: LLMClient,
                 schema: SchemaGraph = None,
                 config: AgentConfig = None):
        """
        初始化TPAR引擎

        Args:
            db_executor: Neo4j数据库执行器
            llm_client: LLM客户端
            schema: Schema图
            config: Agent配置
        """
        self.db = db_executor
        self.llm = llm_client
        self.schema = schema or SchemaGraph()
        self.config = config or AgentConfig()

        # 核心组件
        self.intent_classifier = LLMIntentClassifier(llm_client)
        self.entity_recognizer = LLMEntityRecognizer(llm_client, db_executor)
        self.planner = AdaptivePlanner(llm_client, self.schema)
        self.reflector = LLMReflector(llm_client)
        self.reflection_aggregator = ReflectionAggregator()

        # 会话记忆
        self.session_memory = SessionMemory()

        logger.info("🚀 TPAR Engine initialized")

    # ==================== Main Entry ====================

    def answer(self,
               question: str,
               max_iterations: int = None) -> Dict[str, Any]:
        """
        主入口：回答问题

        完整的TPAR循环实现
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🎯 Question: {question}")
        logger.info(f"{'=' * 70}\n")

        start_time = time.time()
        max_iter = max_iterations or self.config.max_iterations

        # 初始化状态
        state = AnalysisState(question=question)
        state.budget['max_steps'] = max_iter

        # 获取会话上下文
        session_context = self.session_memory.get_relevant_context(question)

        try:
            # ===== PHASE 1: THINK =====
            entities, classification = self._think_phase(state, session_context)

            # ===== PHASE 2-4: PLAN-ACT-REFLECT LOOP =====
            iteration = 0
            while iteration < max_iter:
                # 检查预算
                if not state.check_budget()['can_continue']:
                    logger.info("📌 Budget exhausted")
                    break

                # PLAN
                next_steps = self._plan_phase(state, entities, classification)

                if not next_steps:
                    logger.info("📌 No more steps")
                    break

                # ACT + REFLECT
                should_continue = True
                for step in next_steps:
                    if iteration >= max_iter:
                        break

                    # ACT
                    evidence = self._act_phase(state, step, iteration + 1)

                    # REFLECT
                    reflection = self._reflect_phase(state, step, evidence)

                    # 处理决策
                    should_continue = self._handle_decision(state, reflection, classification)

                    if not should_continue:
                        break

                    iteration += 1

                if not should_continue:
                    break

                # 检查是否应该继续
                if not self.planner.should_continue(state, classification):
                    break

            # ===== PHASE 5: SYNTHESIZE =====
            final_answer = self._synthesize_phase(state)

            # 记录到会话
            self.session_memory.add_qa(question, final_answer, state.discovered_entities)

            return self._build_result(state, final_answer, start_time)

        except Exception as e:
            logger.error(f"TPAR loop failed: {e}")
            import traceback
            traceback.print_exc()
            return self._build_error_result(question, str(e), start_time)

    # ==================== THINK Phase ====================

    def _think_phase(self,
                     state: AnalysisState,
                     context: str = "") -> Tuple[List[Entity], IntentClassification]:
        """
        Think阶段

        1. 实体识别
        2. 意图分类
        3. 初始化状态
        """
        logger.info("🧠 THINK PHASE")
        logger.info("-" * 50)

        # 1. 实体识别
        logger.info("  [1/2] Entity recognition...")
        state.increment_budget('llm')
        entities = self.entity_recognizer.recognize(state.question)

        # 填充state
        for entity in entities:
            if entity.entity_type not in state.discovered_entities:
                state.discovered_entities[entity.entity_type] = []
            if entity.name not in state.discovered_entities[entity.entity_type]:
                state.discovered_entities[entity.entity_type].append(entity.name)

        logger.info(f"       Found {len(entities)} entities")
        for e in entities[:5]:
            logger.info(f"         • {e.name} ({e.entity_type})")

        # 2. 意图分类
        logger.info("  [2/2] Intent classification...")
        state.increment_budget('llm')
        classification = self.intent_classifier.classify(state.question, context)

        # 更新状态
        state.question_intent = classification.intent
        state.target_depth = classification.recommended_depth
        state._classification = classification

        logger.info(f"       Intent: {classification.intent.value}")
        logger.info(f"       Depth: {classification.recommended_depth.value}")
        logger.info(f"       Planner: {classification.recommended_planner.value}")

        logger.info("  ✅ Think phase complete\n")

        return entities, classification

    # ==================== PLAN Phase ====================

    def _plan_phase(self,
                    state: AnalysisState,
                    entities: List[Entity],
                    classification: IntentClassification) -> List[CandidateStep]:
        """
        Plan阶段

        1. 生成候选步骤
        2. LLM评估排序
        3. 返回最优步骤
        """
        logger.info("📋 PLAN PHASE")
        logger.info("-" * 50)

        state.increment_budget('llm')

        candidates = self.planner.plan_next_steps(
            state=state,
            question=state.question,
            entities=entities,
            classification=classification,
            max_steps=2
        )

        logger.info(f"  Selected {len(candidates)} steps")
        logger.info("  ✅ Plan phase complete\n")

        return candidates

    # ==================== ACT Phase ====================

    def _act_phase(self,
                   state: AnalysisState,
                   step: CandidateStep,
                   step_number: int) -> EvidenceRecord:
        """
        Act阶段

        1. 执行步骤
        2. 收集证据
        3. 更新状态
        """
        logger.info(f"⚙️ ACT PHASE - Step {step_number}")
        logger.info("-" * 50)
        logger.info(f"  Purpose: {step.purpose}")

        start_time = time.time()

        # 执行
        result = self._execute_step(step, state)

        execution_time = time.time() - start_time

        # 创建证据记录
        evidence = self._create_evidence(step, result, step_number, execution_time)

        # 更新状态
        self._update_state(state, step, result, step_number)

        # 记录步骤
        state.executed_steps.append({
            'step_number': step_number,
            'step_id': step.step_id,
            'purpose': step.purpose,
            'modality': step.step_type,
            'success': result.get('success', False),
            'row_count': len(result.get('data', [])),
            'execution_time': execution_time
        })

        state.evidence_buffer.add(evidence)

        logger.info(f"  Time: {execution_time:.2f}s")
        logger.info(f"  Rows: {len(result.get('data', []))}")
        logger.info("  ✅ Act phase complete\n")

        return evidence

    def _execute_step(self, step: CandidateStep, state: AnalysisState) -> Dict:
        """执行步骤"""
        query = step.cypher_template.strip()
        params = step.parameters.copy()

        # 解析参数依赖
        params = self._resolve_params(params, state)

        # 特殊处理：非Cypher步骤
        if not query:
            if step.step_type == 'multi-modal':
                return self._execute_multimodal(step, state)
            elif step.step_type == 'statistical':
                return self._execute_statistical(step, state)
            else:
                return {'success': False, 'error': 'No query template', 'data': []}

        # 确保LIMIT
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        state.increment_budget('cypher')
        return self.db.run(query, params)

    def _execute_multimodal(self, step: CandidateStep, state: AnalysisState) -> Dict:
        """执行多模态分析"""
        # TODO: 实现指纹分析
        logger.info("  Executing multi-modal analysis...")
        return {'success': True, 'data': [], 'note': 'Multi-modal analysis placeholder'}

    def _execute_statistical(self, step: CandidateStep, state: AnalysisState) -> Dict:
        """执行统计分析"""
        # TODO: 实现统计检验
        logger.info("  Executing statistical analysis...")
        return {'success': True, 'data': [], 'note': 'Statistical analysis placeholder'}

    def _resolve_params(self, params: Dict, state: AnalysisState) -> Dict:
        """解析参数依赖"""
        resolved = params.copy()

        # 从中间数据中提取
        for key, data in state.intermediate_data.items():
            if not data or not isinstance(data, list):
                continue

            if not data:
                continue

            first_row = data[0]

            # 提取regions
            if 'regions' not in resolved or not resolved['regions']:
                if 'region' in first_row or 'acronym' in first_row:
                    regions = [row.get('region') or row.get('acronym')
                              for row in data
                              if row.get('region') or row.get('acronym')]
                    if regions:
                        resolved['regions'] = list(set(regions))[:10]

            # 提取targets
            if 'targets' not in resolved or not resolved['targets']:
                if 'target' in first_row or 'target_region' in first_row:
                    targets = [row.get('target') or row.get('target_region')
                              for row in data
                              if row.get('target') or row.get('target_region')]
                    if targets:
                        resolved['targets'] = list(set(targets))[:10]

        return resolved

    def _create_evidence(self,
                         step: CandidateStep,
                         result: Dict,
                         step_number: int,
                         execution_time: float) -> EvidenceRecord:
        """创建证据记录"""
        data = result.get('data', [])

        # 计算完整性
        if data:
            total = sum(len(row) for row in data if isinstance(row, dict))
            non_null = sum(1 for row in data if isinstance(row, dict)
                          for v in row.values() if v is not None)
            completeness = non_null / total if total > 0 else 0.0
        else:
            completeness = 0.0

        # 模态
        modality_map = {
            'molecular': Modality.MOLECULAR,
            'morphological': Modality.MORPHOLOGICAL,
            'projection': Modality.PROJECTION,
            'spatial': Modality.SPATIAL,
            'statistical': Modality.STATISTICAL,
        }
        modality = modality_map.get(step.step_type)

        return EvidenceRecord(
            step_number=step_number,
            query_hash=EvidenceRecord.compute_query_hash(step.cypher_template, step.parameters),
            execution_time=execution_time,
            data_completeness=completeness,
            row_count=len(data),
            column_count=len(data[0]) if data and isinstance(data[0], dict) else 0,
            modality=modality,
            raw_data_key=f"step_{step_number}",
            confidence_score=0.8 if result.get('success') else 0.2
        )

    def _update_state(self,
                      state: AnalysisState,
                      step: CandidateStep,
                      result: Dict,
                      step_number: int):
        """更新状态"""
        data = result.get('data', [])

        # 存储中间数据
        state.intermediate_data[f"step_{step_number}"] = data

        # 更新模态覆盖
        modality_map = {
            'molecular': Modality.MOLECULAR,
            'morphological': Modality.MORPHOLOGICAL,
            'projection': Modality.PROJECTION,
            'spatial': Modality.SPATIAL,
        }
        if step.step_type in modality_map:
            state.add_modality(modality_map[step.step_type])

        if not data:
            return

        first_row = data[0] if data else {}

        # 提取发现的实体

        # Regions
        if 'region' in first_row or 'acronym' in first_row:
            regions = list(set([
                row.get('region') or row.get('acronym')
                for row in data
                if row.get('region') or row.get('acronym')
            ]))
            existing = state.discovered_entities.setdefault('Region', [])
            for r in regions:
                if r and r not in existing:
                    existing.append(r)

        # Projection targets
        if 'target' in first_row or 'target_region' in first_row:
            targets = list(set([
                row.get('target') or row.get('target_region')
                for row in data
                if row.get('target') or row.get('target_region')
            ]))
            existing = state.discovered_entities.setdefault('ProjectionTarget', [])
            for t in targets:
                if t and t not in existing:
                    existing.append(t)

        # 记录路径
        state.add_path({
            'step_id': step.step_id,
            'step_type': step.step_type,
            'result_count': len(data)
        })

    # ==================== REFLECT Phase ====================

    def _reflect_phase(self,
                       state: AnalysisState,
                       step: CandidateStep,
                       evidence: EvidenceRecord) -> StructuredReflection:
        """
        Reflect阶段

        LLM驱动的深度反思
        """
        logger.info("🤔 REFLECT PHASE")
        logger.info("-" * 50)

        state.increment_budget('llm')

        # 获取实际结果
        actual_result = {
            'success': evidence.row_count > 0 or evidence.confidence_score > 0.5,
            'data': state.intermediate_data.get(evidence.raw_data_key, [])
        }

        reflection = self.reflector.reflect(
            step_number=evidence.step_number,
            purpose=step.purpose,
            expected_result=step.rationale,
            actual_result=actual_result,
            state=state,
            use_llm=True
        )

        # 记录反思
        state.reflections.append({
            'step_number': evidence.step_number,
            'decision': reflection.decision.value,
            'confidence': reflection.confidence_score,
            'summary': reflection.summary
        })

        logger.info(f"  Decision: {reflection.decision.value}")
        logger.info(f"  Confidence: {reflection.confidence_score:.2f}")
        logger.info("  ✅ Reflect phase complete\n")

        return reflection

    def _handle_decision(self,
                         state: AnalysisState,
                         reflection: StructuredReflection,
                         classification: IntentClassification) -> bool:
        """处理反思决策"""
        decision = reflection.decision

        if decision == ReflectionDecision.CONTINUE:
            return True

        elif decision == ReflectionDecision.TERMINATE:
            logger.info("📌 Analysis complete (TERMINATE)")
            return False

        elif decision == ReflectionDecision.REPLAN:
            if state.replanning_count < state.max_replanning:
                logger.info("🔄 Replanning...")
                state.replanning_count += 1
                # 重新分类
                state._classification = self.intent_classifier.classify(state.question)
                return True
            else:
                logger.info("📌 Max replanning reached")
                return False

        elif decision == ReflectionDecision.DEEPEN:
            logger.info("🔍 Deepening analysis")
            if state.target_depth == AnalysisDepth.SHALLOW:
                state.target_depth = AnalysisDepth.MEDIUM
            elif state.target_depth == AnalysisDepth.MEDIUM:
                state.target_depth = AnalysisDepth.DEEP
            return True

        elif decision == ReflectionDecision.PIVOT:
            logger.info("↪️ Pivoting direction")
            state.replanning_count += 1
            return True

        return True

    # ==================== SYNTHESIZE Phase ====================

    def _synthesize_phase(self, state: AnalysisState) -> str:
        """
        Synthesize阶段 - 生成最终答案
        """
        logger.info("📝 SYNTHESIZE PHASE")
        logger.info("-" * 50)

        state.increment_budget('llm')

        # 准备上下文
        evidence_summary = state.evidence_buffer.summarize()

        # 准备关键数据
        key_data = {}
        for step in state.executed_steps:
            step_key = f"step_{step['step_number']}"
            if step_key in state.intermediate_data:
                data = state.intermediate_data[step_key]
                if data:
                    key_data[step['purpose'][:50]] = data[:5]

        system_prompt = self._get_synthesis_system_prompt()
        user_prompt = self._build_synthesis_prompt(state, evidence_summary, key_data)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            answer = self.llm.chat(messages, temperature=0.3, max_tokens=2000)

            logger.info("  ✅ Synthesis complete\n")

            return answer.strip()

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(state)

    def _get_synthesis_system_prompt(self) -> str:
        return """You are a neuroscience writer synthesizing research analysis results.

Write a comprehensive, publication-quality answer that:
1. Directly answers the original question
2. Cites specific quantitative findings
3. Connects findings across modalities (molecular → morphological → projection)
4. Acknowledges limitations

Write in active voice. Be precise with numbers. Be concise but thorough."""

    def _build_synthesis_prompt(self,
                                state: AnalysisState,
                                evidence_summary: Dict,
                                key_data: Dict) -> str:

        steps_summary = "\n".join([
            f"- Step {s['step_number']}: {s['purpose']} ({s['row_count']} results)"
            for s in state.executed_steps
        ])

        reflections_summary = "\n".join([
            f"- Step {r['step_number']}: {r['summary'][:80]}..."
            for r in state.reflections[-5:]
        ])

        return f"""Synthesize a comprehensive answer:

**Original Question:** {state.question}

**Analysis Steps:**
{steps_summary}

**Modalities Covered:** {[m.value for m in state.modalities_covered]}

**Evidence Summary:**
- Records: {evidence_summary.get('total_records', 0)}
- Completeness: {evidence_summary.get('data_completeness', 0):.2f}
- Confidence: {evidence_summary.get('confidence_score', 0):.2f}

**Reflections:**
{reflections_summary}

**Key Data:**
{json.dumps(key_data, indent=2, default=str)[:3000]}

Write a structured answer:
1. **Main Finding** - Direct answer
2. **Supporting Evidence** - Key quantitative results
3. **Multi-Modal Integration** - How modalities connect
4. **Limitations** - What we don't know

Be concise but comprehensive."""

    def _fallback_synthesis(self, state: AnalysisState) -> str:
        """Fallback答案生成"""
        parts = [f"Analysis of '{state.question}':"]

        for step in state.executed_steps:
            parts.append(f"- {step['purpose']}: {step['row_count']} results")

        parts.append(f"\nModalities covered: {[m.value for m in state.modalities_covered]}")
        parts.append(f"Confidence: {state.evidence_buffer.get_overall_confidence():.2f}")

        return "\n".join(parts)

    # ==================== Result Building ====================

    def _build_result(self,
                      state: AnalysisState,
                      answer: str,
                      start_time: float) -> Dict:
        """构建返回结果"""
        return {
            'question': state.question,
            'answer': answer,

            'entities_recognized': state.discovered_entities,
            'executed_steps': state.executed_steps,
            'evidence_summary': state.evidence_buffer.summarize(),
            'reflections': state.reflections,

            'analysis_info': {
                'intent': state.question_intent.value,
                'target_depth': state.target_depth.value,
                'modalities_covered': [m.value for m in state.modalities_covered],
                'paths_used': state.paths_used,
                'replanning_count': state.replanning_count,
                'budget_used': {
                    'cypher_calls': state.budget['current_cypher_calls'],
                    'llm_calls': state.budget['current_llm_calls']
                }
            },

            'confidence_score': state.evidence_buffer.get_overall_confidence(),
            'execution_time': time.time() - start_time,
            'total_steps': len(state.executed_steps),

            'intermediate_data': state.intermediate_data,
            'success': True
        }

    def _build_error_result(self, question: str, error: str, start_time: float) -> Dict:
        """构建错误结果"""
        return {
            'question': question,
            'answer': f"Analysis failed: {error}",
            'error': error,
            'execution_time': time.time() - start_time,
            'success': False
        }


# ==================== Export ====================

__all__ = ['TPAREngine']