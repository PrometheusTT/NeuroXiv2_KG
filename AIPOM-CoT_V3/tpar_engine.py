"""
TPAR Engine - Think-Plan-Act-Reflect Core Loop
===============================================
核心推理引擎，实现Figure 2C的完整TPAR循环

关键改进：
1. 统一的循环架构
2. LLM参与每个阶段
3. 预算控制
4. 终止条件推理

Author: Claude & Lijun
Date: 2025-01-15
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from core_structures import (
    AnalysisState,
    AnalysisDepth,
    Modality,
    QuestionIntent,
    PlannerType,
    ReflectionDecision,
    EvidenceRecord,
    EvidenceBuffer,
    StatisticalEvidence,
    CandidateStep,
    ReasoningStep,
    StructuredReflection,
    SessionMemory
)

from intent_classifier import LLMIntentClassifier, PlannerRouter, IntentClassification
from llm_reflector import LLMReflector, ReflectionAggregator
from multimodal_analyzer import UnifiedFingerprintAnalyzer, StatisticalToolkit

logger = logging.getLogger(__name__)


# ==================== TPAR Engine ====================

class TPAREngine:
    """
    TPAR推理引擎 - 对齐Figure 2C

    核心循环:
    1. Think: 解析问题，识别实体和意图
    2. Plan: 选择规划器，生成步骤
    3. Act: 执行步骤，收集证据
    4. Reflect: 评估结果，决定下一步

    关键改进:
    - LLM参与所有阶段的决策
    - 统一的证据缓冲
    - 预算控制
    - 智能终止
    """

    def __init__(self,
                 db,
                 schema,
                 llm_client,
                 entity_recognizer,
                 focus_planner,
                 comparative_planner,
                 adaptive_planner,
                 model: str = "gpt-4o"):

        self.db = db
        self.schema = schema
        self.llm = llm_client
        self.model = model

        # 核心组件
        self.entity_recognizer = entity_recognizer
        self.intent_classifier = LLMIntentClassifier(llm_client, model)
        self.reflector = LLMReflector(llm_client, model)
        self.reflection_aggregator = ReflectionAggregator()

        # 规划器
        self.planner_router = PlannerRouter(
            focus_planner,
            comparative_planner,
            adaptive_planner
        )
        self.planners = {
            PlannerType.FOCUS_DRIVEN: focus_planner,
            PlannerType.COMPARATIVE: comparative_planner,
            PlannerType.ADAPTIVE: adaptive_planner
        }

        # 分析工具
        self.fingerprint_analyzer = UnifiedFingerprintAnalyzer(db)
        self.stats = StatisticalToolkit()

        # 会话记忆
        self.session_memory = SessionMemory()

        logger.info("🚀 TPAR Engine initialized")

    # ==================== Main Entry Point ====================

    def answer(self,
               question: str,
               max_iterations: int = 15) -> Dict[str, Any]:
        """
        主入口：回答问题

        完整的TPAR循环实现
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🎯 Question: {question}")
        logger.info(f"{'=' * 70}\n")

        start_time = time.time()

        # 初始化状态
        state = AnalysisState(question=question)

        # 获取会话上下文
        session_context = self.session_memory.get_relevant_context(question)

        try:
            # ===== PHASE 1: THINK =====
            self._think_phase(state, session_context)

            # ===== PHASE 2-4: PLAN-ACT-REFLECT LOOP =====
            iteration = 0
            while iteration < max_iterations:

                # 检查预算
                budget_status = state.check_budget()
                if not budget_status['can_continue']:
                    logger.info("📌 Budget exhausted, stopping")
                    break

                # PLAN
                next_steps = self._plan_phase(state)

                if not next_steps:
                    logger.info("📌 No more steps, stopping")
                    break

                # ACT + REFLECT for each step
                should_continue = True
                for step in next_steps:
                    if iteration >= max_iterations:
                        break

                    # ACT
                    reasoning_step, evidence = self._act_phase(state, step, iteration + 1)

                    # REFLECT
                    reflection = self._reflect_phase(state, reasoning_step)

                    # 处理反思决策
                    should_continue = self._handle_reflection_decision(
                        state, reflection, reasoning_step
                    )

                    if not should_continue:
                        break

                    iteration += 1

                if not should_continue:
                    break

            # ===== PHASE 5: SYNTHESIZE =====
            final_answer = self._synthesize_phase(state)

            # 记录到会话
            self.session_memory.add_qa(
                question,
                final_answer,
                state.discovered_entities
            )

            # 构建返回结果
            return self._build_result(state, final_answer, start_time)

        except Exception as e:
            logger.error(f"TPAR loop failed: {e}")
            import traceback
            traceback.print_exc()
            return self._build_error_result(question, str(e), start_time)

    # ==================== THINK Phase ====================

    def _think_phase(self, state: AnalysisState, context: str = ""):
        """
        Think阶段 - 对齐Figure 2C

        1. 实体识别
        2. 意图分类
        3. 初始化状态
        """
        logger.info("🧠 THINK PHASE")
        logger.info("-" * 50)

        # 1. 实体识别
        logger.info("  [1/2] Entity recognition...")
        entity_matches = self.entity_recognizer.recognize_entities(state.question)

        # 填充discovered_entities
        for match in entity_matches:
            entity_type = match.entity_type
            entity_id = match.entity_id

            if entity_type not in state.discovered_entities:
                state.discovered_entities[entity_type] = []

            if entity_id not in state.discovered_entities[entity_type]:
                state.discovered_entities[entity_type].append(entity_id)

        logger.info(f"       Found {len(entity_matches)} entities")
        for match in entity_matches[:5]:
            logger.info(f"         • {match.text} ({match.entity_type})")

        # 2. 意图分类
        logger.info("  [2/2] Intent classification...")
        state.increment_budget('llm')

        classification = self.intent_classifier.classify(state.question, context)

        # 更新状态
        state.question_intent = classification.intent
        state.target_depth = classification.recommended_depth

        # 存储分类结果供后续使用
        state._classification = classification

        logger.info(f"       Intent: {classification.intent.value}")
        logger.info(f"       Depth: {classification.recommended_depth.value}")
        logger.info(f"       Planner: {classification.recommended_planner.value}")

        logger.info("  ✅ Think phase complete\n")

    # ==================== PLAN Phase ====================

    def _plan_phase(self, state: AnalysisState) -> List[CandidateStep]:
        """
        Plan阶段 - 对齐Figure 2C

        1. 选择规划器
        2. 生成候选步骤
        3. LLM评分排序
        """
        logger.info("📋 PLAN PHASE")
        logger.info("-" * 50)

        classification = getattr(state, '_classification', None)

        if classification is None:
            logger.warning("  No classification found, using adaptive planner")
            planner = self.planners[PlannerType.ADAPTIVE]
            config = {}
        else:
            planner, config = self.planner_router.route(classification, state)

        # 构建planner需要的analysis_state
        from adaptive_planner import AnalysisState as AdaptiveState, AnalysisDepth

        adaptive_state = AdaptiveState(
            discovered_entities=state.discovered_entities,
            executed_steps=[
                {'purpose': s['purpose'], 'modality': s.get('modality')}
                for s in state.executed_steps
            ],
            modalities_covered=[m.value for m in state.modalities_covered],
            target_depth=AnalysisDepth(state.target_depth.value),
            question_intent=state.question_intent.value
        )

        # 获取候选步骤
        if hasattr(planner, 'generate_focus_driven_plan'):
            candidates = planner.generate_focus_driven_plan(adaptive_state, state.question)
        elif hasattr(planner, 'generate_comparative_plan'):
            candidates = planner.generate_comparative_plan(adaptive_state, state.question)
        elif hasattr(planner, 'plan_next_steps'):
            candidates = planner.plan_next_steps(adaptive_state, state.question, max_steps=2)
        else:
            candidates = []

        logger.info(f"  Generated {len(candidates)} candidate steps")

        for i, step in enumerate(candidates[:3], 1):
            logger.info(f"    {i}. {step.purpose[:50]}... (priority: {step.priority:.1f})")

        logger.info("  ✅ Plan phase complete\n")

        return candidates

    # ==================== ACT Phase ====================

    def _act_phase(self,
                   state: AnalysisState,
                   candidate: CandidateStep,
                   step_number: int) -> Tuple[ReasoningStep, EvidenceRecord]:
        """
        Act阶段 - 对齐Figure 2C

        1. 执行步骤
        2. 收集证据
        3. 更新状态
        """
        logger.info(f"⚙️ ACT PHASE - Step {step_number}")
        logger.info("-" * 50)
        logger.info(f"  Purpose: {candidate.purpose}")

        start_time = time.time()

        # 构建ReasoningStep
        reasoning_step = ReasoningStep(
            step_number=step_number,
            purpose=candidate.purpose,
            action=self._determine_action(candidate),
            rationale=candidate.rationale,
            expected_result=candidate.expected_data,
            query_or_params={
                'query': candidate.cypher_template,
                'params': candidate.parameters
            },
            modality=self._convert_modality(candidate.step_type),
            depends_on=[]
        )

        # 执行
        result = self._execute_step(reasoning_step, state)

        reasoning_step.actual_result = result
        reasoning_step.execution_time = time.time() - start_time

        # 构建证据记录
        evidence = self._create_evidence_record(reasoning_step, result, state)
        reasoning_step.evidence_record = evidence

        # 更新状态
        self._update_state_from_result(state, reasoning_step, result, candidate)

        # 记录到状态
        state.executed_steps.append({
            'step_number': step_number,
            'purpose': candidate.purpose,
            'step_id': candidate.step_id,
            'modality': candidate.step_type if isinstance(candidate.step_type, str) else candidate.step_type.value,
            'success': result.get('success', False),
            'row_count': len(result.get('data', []))
        })

        state.evidence_buffer.add(evidence)

        logger.info(f"  Execution time: {reasoning_step.execution_time:.2f}s")
        logger.info(f"  Result: {len(result.get('data', []))} rows")
        logger.info("  ✅ Act phase complete\n")

        return reasoning_step, evidence

    def _execute_step(self, step: ReasoningStep, state: AnalysisState) -> Dict:
        """执行步骤"""
        action = step.action
        query = step.query_or_params.get('query', '').strip()
        params = step.query_or_params.get('params', {})

        if action == 'execute_cypher' and query:
            # 解析参数依赖
            params = self._resolve_parameters(step, state, params)

            # 确保LIMIT
            import re
            if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
                query = f"{query}\nLIMIT 100"

            state.increment_budget('cypher')
            return self.db.run(query, params)

        elif action == 'execute_statistical':
            return self._execute_statistical(step, state)

        elif action == 'execute_fingerprint':
            return self._execute_fingerprint(step, state)

        else:
            return {'success': False, 'error': f'Unknown action: {action}', 'data': []}

    def _execute_statistical(self, step: ReasoningStep, state: AnalysisState) -> Dict:
        """执行统计分析"""
        params = step.query_or_params.get('params', {})
        test_type = params.get('test_type', 'fdr')

        if test_type == 'fdr':
            # 从之前的步骤获取p-values
            mismatch_data = None
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'p_value' in data[0]:
                        mismatch_data = data
                        break

            if not mismatch_data:
                return {'success': False, 'error': 'No p-values found', 'data': []}

            p_values = [row.get('p_value', 1.0) for row in mismatch_data]
            q_values, significant = StatisticalToolkit.fdr_correction(p_values, params.get('alpha', 0.05))

            # 整合结果
            result_data = []
            for i, row in enumerate(mismatch_data):
                result_data.append({
                    **row,
                    'q_value': q_values[i],
                    'fdr_significant': significant[i]
                })

            significant_data = [r for r in result_data if r['fdr_significant']]

            return {
                'success': True,
                'data': significant_data,
                'test_type': 'fdr'
            }

        return {'success': False, 'error': f'Unknown test: {test_type}', 'data': []}

    def _execute_fingerprint(self, step: ReasoningStep, state: AnalysisState) -> Dict:
        """执行指纹分析"""
        params = step.query_or_params.get('params', {})

        # 获取regions
        regions = state.discovered_entities.get('Region', [])
        if not regions:
            # 尝试从intermediate_data获取
            for key, data in state.intermediate_data.items():
                if data and isinstance(data, list) and len(data) > 0:
                    if 'region' in data[0]:
                        regions = list(set([row['region'] for row in data if row.get('region')]))
                        break

        max_regions = params.get('max_regions', 30)
        regions = regions[:max_regions]

        if len(regions) < 2:
            return {'success': False, 'error': 'Need at least 2 regions', 'data': []}

        # 计算mismatch矩阵
        mismatch_results = self.fingerprint_analyzer.compute_mismatch_matrix(regions)

        # 转换为dict格式
        result_data = []
        for mr in mismatch_results:
            result_data.append({
                'region1': mr.region1,
                'region2': mr.region2,
                'sim_molecular': mr.sim_molecular,
                'sim_morphological': mr.sim_morphological,
                'sim_projection': mr.sim_projection,
                'mismatch_GM': mr.mismatch_GM,
                'mismatch_GP': mr.mismatch_GP,
                'mismatch_MP': mr.mismatch_MP,
                'mismatch_combined': mr.mismatch_combined,
                'p_value': mr.p_value,
                'z_score': mr.z_score
            })

        return {
            'success': True,
            'data': result_data,
            'analysis_type': 'cross_modal_mismatch'
        }

    # ==================== REFLECT Phase ====================

    def _reflect_phase(self,
                       state: AnalysisState,
                       step: ReasoningStep) -> StructuredReflection:
        """
        Reflect阶段 - 对齐Figure 2C

        LLM驱动的深度反思
        """
        logger.info("🤔 REFLECT PHASE")
        logger.info("-" * 50)

        # 使用LLM反思
        state.increment_budget('llm')

        reflection = self.reflector.reflect(
            step_number=step.step_number,
            purpose=step.purpose,
            expected_result=step.expected_result,
            actual_result=step.actual_result or {},
            state=state,
            use_llm=True
        )

        # 记录反思
        state.reflections.append({
            'step_number': step.step_number,
            'decision': reflection.decision.value,
            'confidence': reflection.confidence_score,
            'summary': reflection.summary
        })

        step.reflection = {
            'decision': reflection.decision.value,
            'summary': reflection.summary
        }
        step.validation_passed = reflection.validation_status.value in ['passed', 'partial']

        logger.info(f"  Decision: {reflection.decision.value}")
        logger.info(f"  Confidence: {reflection.confidence_score:.2f}")
        logger.info("  ✅ Reflect phase complete\n")

        return reflection

    def _handle_reflection_decision(self,
                                    state: AnalysisState,
                                    reflection: StructuredReflection,
                                    step: ReasoningStep) -> bool:
        """
        处理反思决策

        Returns:
            是否继续执行
        """
        decision = reflection.decision

        if decision == ReflectionDecision.CONTINUE:
            return True

        elif decision == ReflectionDecision.TERMINATE:
            logger.info("📌 Analysis complete (TERMINATE decision)")
            return False

        elif decision == ReflectionDecision.REPLAN:
            if state.replanning_count < state.max_replanning:
                logger.info("🔄 Replanning triggered")
                state.replanning_count += 1
                # 重新分类意图
                state._classification = self.intent_classifier.classify(state.question)
                return True
            else:
                logger.info("📌 Max replanning reached")
                return False

        elif decision == ReflectionDecision.DEEPEN:
            logger.info("🔍 Deepening analysis")
            # 提升目标深度
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
        reflection_agg = self.reflection_aggregator.aggregate(
            [StructuredReflection(**r) if isinstance(r, dict) else r
             for r in state.reflections if isinstance(r, (dict, StructuredReflection))]
        ) if state.reflections else {}

        # 准备关键数据
        key_data = {}
        for step in state.executed_steps:
            step_key = f"step_{step['step_number']}"
            if step_key in state.intermediate_data:
                data = state.intermediate_data[step_key]
                if data:
                    key_data[step_key] = data[:5]  # Top 5

        prompt = self._build_synthesis_prompt(state, evidence_summary, key_data)

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_synthesis_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            answer = response.choices[0].message.content.strip()

            logger.info("  ✅ Synthesis complete\n")

            return answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Analysis completed with {len(state.executed_steps)} steps. " \
                   f"Covered modalities: {[m.value for m in state.modalities_covered]}."

    def _get_synthesis_system_prompt(self) -> str:
        return """You are a neuroscience writer synthesizing research analysis results.

Your task is to write a comprehensive, publication-quality answer that:
1. Directly answers the original question
2. Cites specific quantitative findings
3. Connects findings across modalities (molecular → morphological → projection)
4. Acknowledges limitations and uncertainties
5. Is written for a neuroscience audience

Write in active voice. Be precise with numbers. Connect findings causally."""

    def _build_synthesis_prompt(self,
                                state: AnalysisState,
                                evidence_summary: Dict,
                                key_data: Dict) -> str:

        steps_summary = "\n".join([
            f"- Step {s['step_number']}: {s['purpose']} ({s.get('row_count', 0)} results)"
            for s in state.executed_steps
        ])

        reflections_summary = "\n".join([
            f"- Step {r['step_number']}: {r['summary'][:100]}..."
            for r in state.reflections[-5:]
        ])

        return f"""Synthesize a comprehensive answer based on the analysis.

**Original Question:** {state.question}

**Analysis Steps Executed:**
{steps_summary}

**Modalities Covered:** {[m.value for m in state.modalities_covered]}

**Evidence Summary:**
- Total records: {evidence_summary.get('total_records', 0)}
- Data completeness: {evidence_summary.get('data_completeness', 0):.2f}
- Overall confidence: {evidence_summary.get('confidence_score', 0):.2f}

**Key Reflections:**
{reflections_summary}

**Key Data Samples:**
{json.dumps(key_data, indent=2, default=str)[:2000]}

Write a structured answer with:
1. **Main Finding** - Direct answer to the question
2. **Supporting Evidence** - Key quantitative results
3. **Multi-Modal Integration** - How different modalities connect
4. **Limitations** - What we don't know or couldn't determine

Be concise but comprehensive."""

    # ==================== Helper Methods ====================

    def _determine_action(self, candidate: CandidateStep) -> str:
        """确定步骤的action类型"""
        if candidate.cypher_template and candidate.cypher_template.strip():
            return 'execute_cypher'

        step_type = candidate.step_type
        if isinstance(step_type, str):
            if 'statistical' in step_type.lower():
                return 'execute_statistical'
            elif 'multi-modal' in step_type.lower() or 'mismatch' in step_type.lower():
                return 'execute_fingerprint'

        return 'execute_cypher'

    def _convert_modality(self, step_type) -> Optional[Modality]:
        """转换step_type到Modality"""
        if isinstance(step_type, Modality):
            return step_type

        if isinstance(step_type, str):
            mapping = {
                'molecular': Modality.MOLECULAR,
                'morphological': Modality.MORPHOLOGICAL,
                'projection': Modality.PROJECTION,
                'spatial': Modality.SPATIAL,
                'statistical': Modality.STATISTICAL
            }
            return mapping.get(step_type.lower())

        return None

    def _resolve_parameters(self,
                            step: ReasoningStep,
                            state: AnalysisState,
                            params: Dict) -> Dict:
        """解析参数依赖"""
        resolved = params.copy()

        # 查找可能需要的数据
        for key, data in state.intermediate_data.items():
            if not data or not isinstance(data, list):
                continue

            if not data:
                continue

            first_row = data[0]

            # 提取regions
            if 'enriched_regions' not in resolved:
                if isinstance(first_row, dict) and ('region' in first_row or 'acronym' in first_row):
                    regions = [row.get('region') or row.get('acronym') for row in data if
                               row.get('region') or row.get('acronym')]
                    if regions:
                        resolved['enriched_regions'] = list(set(regions))[:10]

            # 提取targets
            if 'targets' not in resolved:
                if isinstance(first_row, dict) and ('target' in first_row or 'target_region' in first_row):
                    targets = [row.get('target') or row.get('target_region') for row in data if
                               row.get('target') or row.get('target_region')]
                    if targets:
                        resolved['targets'] = list(set(targets))[:10]

        return resolved

    def _create_evidence_record(self,
                                step: ReasoningStep,
                                result: Dict,
                                state: AnalysisState) -> EvidenceRecord:
        """创建证据记录"""
        data = result.get('data', [])

        # 计算数据完整性
        if data:
            total_values = sum(len(row) for row in data if isinstance(row, dict))
            non_null = sum(1 for row in data if isinstance(row, dict) for v in row.values() if v is not None)
            completeness = non_null / total_values if total_values > 0 else 0.0
        else:
            completeness = 0.0

        # 构建query hash
        query = step.query_or_params.get('query', '')
        params = step.query_or_params.get('params', {})
        query_hash = EvidenceRecord.compute_query_hash(query, params)

        return EvidenceRecord(
            step_number=step.step_number,
            query_hash=query_hash,
            snapshot_id=f"snap_{int(time.time())}",
            execution_time=step.execution_time,
            data_completeness=completeness,
            row_count=len(data),
            column_count=len(data[0]) if data and isinstance(data[0], dict) else 0,
            modality=step.modality,
            raw_data_key=f"step_{step.step_number}",
            validation_status=step.validation_passed and 'passed' or 'partial',
            confidence_score=0.8 if result.get('success') else 0.2
        )

    def _update_state_from_result(self,
                                  state: AnalysisState,
                                  step: ReasoningStep,
                                  result: Dict,
                                  candidate: CandidateStep):
        """从结果更新状态"""
        data = result.get('data', [])

        # 存储intermediate data
        state.intermediate_data[f"step_{step.step_number}"] = data

        # 更新modality覆盖
        if step.modality:
            state.add_modality(step.modality)

        # 提取新发现的实体
        if not data:
            return

        first_row = data[0] if data else {}

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

        # 记录使用的schema路径
        state.add_path({
            'step_id': candidate.step_id,
            'schema_path': candidate.schema_path,
            'result_count': len(data)
        })

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

            'intermediate_data': state.intermediate_data
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