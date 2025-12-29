"""
TPAR Engine - Think-Plan-Act-Reflect循环引擎
=============================================

实现手稿Figure 2描述的完整TPAR循环:
1. Think: 问题理解、实体识别、意图分类
2. Plan: 路径规划、步骤生成
3. Act: 查询执行、数据获取
4. Reflect: 结果评估、决策制定

集成修复:
- 分析深度与手稿一致
- 统计验证真正实现
- 三模态指纹集成

Author: Lijun
Date: 2025-01
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .core_structures import (
    AnalysisState, AnalysisDepth, QuestionIntent,
    Modality, ReflectionDecision, ValidationStatus,
    Entity, EntityCluster, SchemaPath, CandidateStep,
    EvidenceRecord, StatisticalEvidence, StructuredReflection,
    AgentConfig, SessionMemory, PlannerType,
    ThinkResult, PlanResult, ActResult, ReflectResult,
    TPARIteration, AgentOutput
)
from .scientific_operators import OperatorRegistry, OperatorResult

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    意图分类器 - 与手稿Figure 2A对齐

    5种主要意图:
    - simple_query: 简单查询
    - profiling: 深度剖析
    - comparison: 比较分析
    - screening: 筛选排序
    - explanation: 解释说明
    """

    INTENT_PATTERNS = {
        QuestionIntent.SIMPLE_QUERY: [
            r'what is', r'what are', r'define', r'meaning of',
            r'什么是', r'是什么'
        ],
        QuestionIntent.DEEP_PROFILING: [
            r'tell me about', r'describe', r'profile', r'characterize',
            r'what can you tell', r'详细介绍', r'描述'
        ],
        QuestionIntent.COMPARISON: [
            r'compare', r'difference between', r'vs', r'versus',
            r'similar to', r'比较', r'区别'
        ],
        QuestionIntent.SCREENING: [
            r'which', r'find all', r'list', r'rank', r'top',
            r'highest', r'lowest', r'筛选', r'排序'
        ],
        QuestionIntent.EXPLANATION: [
            r'why', r'how does', r'explain', r'reason',
            r'mechanism', r'为什么', r'如何'
        ],
    }

    INTENT_TO_DEPTH = {
        QuestionIntent.SIMPLE_QUERY: AnalysisDepth.SHALLOW,
        QuestionIntent.DEFINITION: AnalysisDepth.SHALLOW,
        QuestionIntent.DEEP_PROFILING: AnalysisDepth.DEEP,
        QuestionIntent.COMPARISON: AnalysisDepth.MEDIUM,
        QuestionIntent.SCREENING: AnalysisDepth.MEDIUM,
        QuestionIntent.EXPLANATION: AnalysisDepth.DEEP,
        QuestionIntent.CONNECTIVITY: AnalysisDepth.MEDIUM,
        QuestionIntent.COMPOSITION: AnalysisDepth.MEDIUM,
        QuestionIntent.QUANTIFICATION: AnalysisDepth.SHALLOW,
    }

    def classify(self, question: str) -> Tuple[QuestionIntent, AnalysisDepth]:
        """分类问题意图并确定分析深度"""
        question_lower = question.lower()

        import re
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    depth = self.INTENT_TO_DEPTH.get(intent, AnalysisDepth.MEDIUM)
                    return intent, depth

        return QuestionIntent.UNKNOWN, AnalysisDepth.MEDIUM


class EntityRecognizer:
    """实体识别器"""

    ENTITY_PATTERNS = {
        'GeneMarker': [
            r'\b(Car3|Slc17a7|Gad1|Gad2|Vip|Sst|Pvalb|Lamp5)\b',
            r'\b[A-Z][a-z]+\d*\b',  # 基因命名模式
        ],
        'Region': [
            r'\b(CLA|claustrum|cortex|thalamus|hippocampus|striatum)\b',
            r'\b(VIS|ACA|MOp|SSp|RSP|AUD|TEa|PERI|ECT)\b',
        ],
        'CellType': [
            r'\b(neuron|interneuron|pyramidal|stellate|granule)\b',
            r'\b(excitatory|inhibitory|glutamatergic|GABAergic)\b',
        ],
    }

    def recognize(self, text: str) -> Dict[str, List[str]]:
        """识别文本中的实体"""
        import re

        entities = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            if matches:
                entities[entity_type] = list(set(matches))

        return entities


class SchemaGraph:
    """Schema图谱 - 用于路径规划"""

    def __init__(self, schema_path: str = None):
        self.schema_path = schema_path
        self.node_labels = []
        self.relationship_types = []
        self._load_schema()

    def _load_schema(self):
        """加载schema定义"""
        # 默认schema（如果没有文件）
        self.node_labels = [
            'Region', 'Cell', 'Cluster', 'Gene', 'Neuron',
            'Morphology', 'Projection'
        ]
        self.relationship_types = [
            'LOCATED_IN', 'EXPRESSES', 'BELONGS_TO',
            'PROJECTS_TO', 'HAS_MORPHOLOGY', 'SIMILAR_TO'
        ]

    def find_paths(self, start_label: str, end_label: str,
                  max_hops: int = 3) -> List[SchemaPath]:
        """查找两节点类型之间的路径"""
        # 简化实现 - 实际应该从schema推导
        paths = []

        # 常见路径模式
        common_paths = [
            ('Region', 'Gene', [('Region', 'LOCATED_IN', 'Cell'),
                               ('Cell', 'EXPRESSES', 'Gene')]),
            ('Region', 'Neuron', [('Region', 'LOCATED_IN', 'Neuron')]),
            ('Region', 'Projection', [('Region', 'PROJECTS_TO', 'Region')]),
            ('Cluster', 'Gene', [('Cluster', 'CONTAINS', 'Cell'),
                                ('Cell', 'EXPRESSES', 'Gene')]),
        ]

        for s, e, hops in common_paths:
            if s == start_label and e == end_label:
                paths.append(SchemaPath(
                    path_id=f"{s}_to_{e}",
                    start_label=s,
                    end_label=e,
                    hops=hops,
                    score=1.0
                ))

        return paths


class AdaptivePlanner:
    """
    自适应规划器 - 与手稿Figure 2B对齐

    三种规划策略:
    - Focus-Driven: 深度剖析单一实体
    - Comparative: 系统对比/筛选
    - Adaptive: 自适应探索
    """

    def __init__(self, schema_graph: SchemaGraph):
        self.schema = schema_graph

    def select_planner_type(self, intent: QuestionIntent,
                           entities: Dict[str, List[str]]) -> PlannerType:
        """选择规划器类型"""
        if intent in [QuestionIntent.DEEP_PROFILING, QuestionIntent.EXPLANATION]:
            return PlannerType.FOCUS_DRIVEN
        elif intent in [QuestionIntent.COMPARISON, QuestionIntent.SCREENING]:
            return PlannerType.COMPARATIVE
        else:
            return PlannerType.ADAPTIVE

    def generate_plan(self, state: AnalysisState) -> List[CandidateStep]:
        """生成执行计划"""
        steps = []
        entities = state.discovered_entities
        intent = state.question_intent

        # 确定需要覆盖的模态
        target_modalities = self._determine_target_modalities(intent)

        # 为每个模态生成步骤
        for i, modality in enumerate(target_modalities):
            step = CandidateStep(
                step_id=f"step_{len(state.executed_steps) + i}",
                step_type=modality,
                purpose=f"Gather {modality} data",
                rationale=f"Required for {intent.value} analysis",
                priority=1.0 - i * 0.1,
                parameters={'entities': entities}
            )
            steps.append(step)

        # 添加统计验证步骤
        if len(steps) >= 2:
            steps.append(CandidateStep(
                step_id=f"step_{len(state.executed_steps) + len(steps)}",
                step_type='statistical',
                purpose='Statistical validation',
                rationale='Validate findings with statistical tests',
                priority=0.5,
                parameters={'analysis_type': 'comprehensive'}
            ))

        # 添加多模态整合步骤（如果是深度分析）
        if state.target_depth == AnalysisDepth.DEEP:
            steps.append(CandidateStep(
                step_id=f"step_{len(state.executed_steps) + len(steps)}",
                step_type='multimodal',
                purpose='Tri-modal fingerprint analysis',
                rationale='Integrate molecular, morphological, projection data',
                priority=0.4,
                parameters={}
            ))

        return steps

    def _determine_target_modalities(self, intent: QuestionIntent) -> List[str]:
        """确定目标模态"""
        if intent == QuestionIntent.DEEP_PROFILING:
            return ['molecular', 'morphological', 'projection']
        elif intent == QuestionIntent.COMPARISON:
            return ['molecular', 'projection']
        elif intent == QuestionIntent.SCREENING:
            return ['molecular']
        else:
            return ['molecular']


class ReflectionEngine:
    """
    反思引擎 - 与手稿Figure 2C对齐

    5种战略决策:
    - continue: 继续执行
    - replan: 重新规划
    - deepen: 加深分析
    - pivot: 转向替代
    - terminate: 证据充足，终止
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def reflect(self, state: AnalysisState,
               last_result: OperatorResult) -> StructuredReflection:
        """执行反思"""
        step_number = len(state.executed_steps)

        # 1. 验证结果
        validation_status, validation_reason = self._validate_result(last_result)

        # 2. 评估不确定性
        uncertainty, uncertainty_sources = self._assess_uncertainty(state, last_result)

        # 3. 提取发现
        findings, surprises = self._extract_findings(state, last_result)

        # 4. 计算置信度
        confidence, confidence_factors = self._compute_confidence(state)

        # 5. 做出决策
        decision, decision_reason = self._make_decision(
            state, validation_status, confidence, uncertainty
        )

        # 6. 生成建议
        suggestions = self._generate_suggestions(state, decision)

        return StructuredReflection(
            step_number=step_number,
            validation_status=validation_status,
            validation_reasoning=validation_reason,
            uncertainty_level=uncertainty,
            uncertainty_sources=uncertainty_sources,
            key_findings=findings,
            surprising_results=surprises,
            decision=decision,
            decision_reasoning=decision_reason,
            next_step_suggestions=suggestions,
            alternative_approaches=[],
            confidence_score=confidence,
            confidence_factors=confidence_factors,
            summary=self._generate_summary(state, decision, confidence)
        )

    def _validate_result(self, result: OperatorResult) -> Tuple[ValidationStatus, str]:
        """验证执行结果"""
        if not result.success:
            return ValidationStatus.FAILED, f"Execution failed: {result.error}"

        if result.row_count == 0:
            return ValidationStatus.EMPTY, "No data returned"

        if result.row_count < 5:
            return ValidationStatus.PARTIAL, f"Limited data: {result.row_count} rows"

        return ValidationStatus.PASSED, f"Success with {result.row_count} rows"

    def _assess_uncertainty(self, state: AnalysisState,
                           result: OperatorResult) -> Tuple[float, List[str]]:
        """评估不确定性"""
        sources = []
        uncertainty = 0.0

        # 数据量不足
        if result.row_count < 10:
            uncertainty += 0.2
            sources.append("Limited data")

        # 模态覆盖不足
        if len(state.modalities_covered) < 2:
            uncertainty += 0.2
            sources.append("Limited modality coverage")

        # 执行步骤少
        if len(state.executed_steps) < state.target_depth.get_min_steps():
            uncertainty += 0.1
            sources.append("Insufficient exploration")

        return min(1.0, uncertainty), sources

    def _extract_findings(self, state: AnalysisState,
                         result: OperatorResult) -> Tuple[List[str], List[str]]:
        """提取关键发现"""
        findings = []
        surprises = []

        if result.success and result.row_count > 0:
            findings.append(f"Retrieved {result.row_count} data points")
            findings.append(f"Operator: {result.operator_name}")

        # 检查是否有意外结果
        if result.row_count > 100:
            surprises.append("Unexpectedly large result set")

        return findings, surprises

    def _compute_confidence(self, state: AnalysisState) -> Tuple[float, Dict[str, float]]:
        """计算置信度"""
        factors = {}

        # 1. 证据置信度
        evidence_conf = state.evidence_buffer.get_overall_confidence()
        factors['evidence_strength'] = evidence_conf

        # 2. 数据完整度
        completeness = state.evidence_buffer.get_data_completeness()
        factors['data_completeness'] = completeness

        # 3. 模态覆盖度
        modality_coverage = len(state.modalities_covered) / 3.0
        factors['modality_coverage'] = modality_coverage

        # 4. 步骤进度
        progress = len(state.executed_steps) / state.target_depth.get_max_steps()
        factors['progress'] = min(1.0, progress)

        # 加权平均
        total_conf = (
            0.35 * evidence_conf +
            0.25 * completeness +
            0.25 * modality_coverage +
            0.15 * progress
        )

        return total_conf, factors

    def _make_decision(self, state: AnalysisState,
                      validation: ValidationStatus,
                      confidence: float,
                      uncertainty: float) -> Tuple[ReflectionDecision, str]:
        """做出战略决策"""
        budget = state.check_budget()

        # 预算耗尽
        if not budget['can_continue']:
            return ReflectionDecision.TERMINATE, "Budget exhausted"

        # 置信度足够高
        if confidence >= self.confidence_threshold:
            return ReflectionDecision.TERMINATE, f"Confidence {confidence:.2f} >= threshold"

        # 验证失败需要重规划
        if validation == ValidationStatus.FAILED:
            if state.replanning_count < state.max_replanning:
                return ReflectionDecision.REPLAN, "Execution failed, replanning"
            else:
                return ReflectionDecision.TERMINATE, "Max replanning reached"

        # 空结果尝试转向
        if validation == ValidationStatus.EMPTY:
            return ReflectionDecision.PIVOT, "Empty results, trying alternative"

        # 不确定性高需要加深
        if uncertainty > 0.5:
            return ReflectionDecision.DEEPEN, "High uncertainty, deepening analysis"

        # 默认继续
        return ReflectionDecision.CONTINUE, "Continuing analysis"

    def _generate_suggestions(self, state: AnalysisState,
                             decision: ReflectionDecision) -> List[str]:
        """生成下一步建议"""
        suggestions = []

        if decision == ReflectionDecision.CONTINUE:
            # 建议覆盖未探索的模态
            covered = state.modalities_covered
            if Modality.MOLECULAR not in covered:
                suggestions.append("Query molecular/gene expression data")
            if Modality.MORPHOLOGICAL not in covered:
                suggestions.append("Query morphological features")
            if Modality.PROJECTION not in covered:
                suggestions.append("Query projection patterns")

        elif decision == ReflectionDecision.DEEPEN:
            suggestions.append("Perform statistical validation")
            suggestions.append("Build tri-modal fingerprint")

        elif decision == ReflectionDecision.PIVOT:
            suggestions.append("Try alternative query approach")
            suggestions.append("Broaden search criteria")

        return suggestions

    def _generate_summary(self, state: AnalysisState,
                         decision: ReflectionDecision,
                         confidence: float) -> str:
        """生成反思摘要"""
        return (
            f"Step {len(state.executed_steps)}: "
            f"Decision={decision.value}, "
            f"Confidence={confidence:.2f}, "
            f"Modalities={len(state.modalities_covered)}/3"
        )


class TPAREngine:
    """
    TPAR循环引擎 - 主控制器

    实现完整的Think-Plan-Act-Reflect循环
    """

    def __init__(self, config: AgentConfig = None, neo4j_driver=None):
        self.config = config or AgentConfig()
        self.driver = neo4j_driver

        # 组件初始化
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.schema_graph = SchemaGraph(self.config.schema_json_path)
        self.planner = AdaptivePlanner(self.schema_graph)
        self.reflection_engine = ReflectionEngine(self.config.confidence_threshold)
        self.operator_registry = OperatorRegistry(neo4j_driver, {})

        # 会话记忆
        self.session_memory = SessionMemory()

    def run(self, question: str) -> AgentOutput:
        """
        运行TPAR循环

        Args:
            question: 用户问题

        Returns:
            完整的Agent输出
        """
        start_time = time.time()
        iterations = []

        # 初始化状态
        state = self._initialize_state(question)

        logger.info(f"Starting TPAR loop for: {question[:50]}...")
        logger.info(f"Intent: {state.question_intent.value}, Depth: {state.target_depth.value}")

        # TPAR循环
        while state.check_budget()['can_continue']:
            iteration = TPARIteration(iteration_number=len(iterations))

            # Think
            think_result = self._think(state)
            iteration.think = think_result

            # Plan
            plan_result = self._plan(state)
            iteration.plan = plan_result

            # Act
            act_result = self._act(state, plan_result)
            iteration.act = act_result

            # Reflect
            reflect_result = self._reflect(state, act_result)
            iteration.reflect = reflect_result

            iterations.append(iteration)

            # 检查是否终止
            if reflect_result.decision == ReflectionDecision.TERMINATE:
                logger.info(f"Terminating: {reflect_result.reasoning}")
                break

            # 处理重规划
            if reflect_result.decision == ReflectionDecision.REPLAN:
                state.replanning_count += 1
                logger.info(f"Replanning (count: {state.replanning_count})")

        # 生成最终答案
        answer = self._generate_answer(state)

        return AgentOutput(
            question=question,
            answer=answer,
            iterations=iterations,
            final_state=state,
            total_time=time.time() - start_time,
            task_status="completed"
        )

    def _initialize_state(self, question: str) -> AnalysisState:
        """初始化分析状态"""
        # 意图分类
        intent, depth = self.intent_classifier.classify(question)

        # 实体识别
        entities = self.entity_recognizer.recognize(question)

        # 创建状态
        state = AnalysisState(
            question=question,
            question_intent=intent,
            target_depth=depth,
            discovered_entities=entities
        )

        # 设置预算
        state.budget['max_steps'] = depth.get_max_steps()

        # 如果有主要实体，设置焦点
        if 'Region' in entities and entities['Region']:
            state.primary_focus = Entity(
                name=entities['Region'][0],
                entity_type='Region'
            )
        elif 'GeneMarker' in entities and entities['GeneMarker']:
            state.primary_focus = Entity(
                name=entities['GeneMarker'][0],
                entity_type='GeneMarker'
            )

        return state

    def _think(self, state: AnalysisState) -> ThinkResult:
        """Think阶段"""
        # 提取实体
        entities = [
            Entity(name=name, entity_type=etype)
            for etype, names in state.discovered_entities.items()
            for name in names
        ]

        # 确定关注模态
        modalities = []
        if state.question_intent in [QuestionIntent.DEEP_PROFILING]:
            modalities = [Modality.MOLECULAR, Modality.MORPHOLOGICAL, Modality.PROJECTION]
        elif state.question_intent == QuestionIntent.COMPARISON:
            modalities = [Modality.MOLECULAR, Modality.PROJECTION]
        else:
            modalities = [Modality.MOLECULAR]

        return ThinkResult(
            entities=entities,
            intent=state.question_intent,
            focus_modalities=modalities,
            key_constraints={},
            reasoning=f"Identified {len(entities)} entities with intent {state.question_intent.value}"
        )

    def _plan(self, state: AnalysisState) -> PlanResult:
        """Plan阶段"""
        # 选择规划器类型
        planner_type = self.planner.select_planner_type(
            state.question_intent,
            state.discovered_entities
        )

        # 生成计划
        steps = self.planner.generate_plan(state)

        return PlanResult(
            selected_paths=[],
            planner_type=planner_type,
            steps=steps,
            reasoning=f"Generated {len(steps)} steps using {planner_type.value} planner"
        )

    def _act(self, state: AnalysisState, plan: PlanResult) -> ActResult:
        """Act阶段"""
        if not plan.steps:
            return ActResult(
                step_id="no_steps",
                success=False,
                data=[],
                row_count=0,
                execution_time=0,
                operator="none"
            )

        # 执行第一个待执行步骤
        step = plan.steps[0]

        start_time = time.time()
        result = self.operator_registry.execute(step, state)
        execution_time = time.time() - start_time

        # 更新状态
        state.executed_steps.append({
            'step_id': step.step_id,
            'step_type': step.step_type,
            'success': result.success,
            'row_count': result.row_count
        })

        if result.success and step.step_type in ['molecular', 'morphological', 'projection']:
            modality = Modality(step.step_type) if step.step_type in [m.value for m in Modality] else None
            if modality:
                state.add_modality(modality)

        state.increment_budget('cypher')

        return ActResult(
            step_id=step.step_id,
            success=result.success,
            data=result.data if isinstance(result.data, list) else [],
            row_count=result.row_count,
            execution_time=execution_time,
            operator=result.operator_name,
            modality=result.modality
        )

    def _reflect(self, state: AnalysisState, act_result: ActResult) -> ReflectResult:
        """Reflect阶段"""
        # 将ActResult转换为OperatorResult
        op_result = OperatorResult(
            success=act_result.success,
            data=act_result.data,
            row_count=act_result.row_count,
            execution_time=act_result.execution_time,
            operator_name=act_result.operator,
            modality=act_result.modality
        )

        reflection = self.reflection_engine.reflect(state, op_result)

        # 记录反思
        state.reflections.append(reflection.to_dict())

        return ReflectResult(
            reflection=reflection,
            decision=reflection.decision,
            reasoning=reflection.decision_reasoning
        )

    def _generate_answer(self, state: AnalysisState) -> str:
        """生成最终答案"""
        # 收集关键信息
        parts = []

        # 开头
        if state.primary_focus:
            parts.append(f"Analysis of {state.primary_focus.name}:")
        else:
            parts.append("Analysis results:")

        # 模态覆盖
        modalities = [m.value for m in state.modalities_covered]
        if modalities:
            parts.append(f"Modalities analyzed: {', '.join(modalities)}")

        # 证据摘要
        evidence = state.evidence_buffer.summarize()
        parts.append(f"Evidence confidence: {evidence['confidence_score']:.2f}")
        parts.append(f"Data completeness: {evidence['data_completeness']:.2f}")

        # 统计结果（如果有）
        if 'statistical_results' in state.intermediate_data:
            stats = state.intermediate_data['statistical_results']
            if 'p_value' in stats:
                parts.append(f"Statistical validation: p={stats['p_value']:.4f}")

        # 指纹分析（如果有）
        if state.fingerprints:
            parts.append(f"Fingerprint analysis: {len(state.fingerprints)} regions profiled")

        return "\n".join(parts)


# ==================== Export ====================

__all__ = [
    'IntentClassifier',
    'EntityRecognizer',
    'SchemaGraph',
    'AdaptivePlanner',
    'ReflectionEngine',
    'TPAREngine'
]