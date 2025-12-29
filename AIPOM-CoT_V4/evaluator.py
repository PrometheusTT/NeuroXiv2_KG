"""
AIPOM-CoT Benchmark Evaluator
==============================

公平评估AIPOM-CoT与baseline方法的性能

关键修复:
- 删除了BASELINE_CAPABILITY_LIMITS的人为上限
- 所有方法使用相同的评估标准
- 集成真正的统计验证

Author: Lijun
Date: 2025-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
import logging
import time

try:
    from .core_structures import (
        AgentOutput, AnalysisState, QuestionIntent, AnswerCorrectness,
        ReflectionDecision
    )
    from .statistical_validator import BenchmarkStatisticalValidator
except ImportError:
    from core_structures import (
        AgentOutput, AnalysisState, QuestionIntent, AnswerCorrectness,
        ReflectionDecision
    )
    from statistical_validator import BenchmarkStatisticalValidator

logger = logging.getLogger(__name__)


# ==================== 能力维度权重 ====================
# 注意：这些权重有理论依据，见手稿Methods部分
CAPABILITY_WEIGHTS = {
    'think': 0.25,     # 实体识别、意图分类
    'plan': 0.25,      # 路径规划、步骤生成
    'act': 0.25,       # 查询执行、数据获取
    'reflect': 0.25,   # 反思决策、证据评估
}

# 【关键修复】删除了不公平的能力上限
# 原代码中的BASELINE_CAPABILITY_LIMITS被删除
# 所有方法现在在公平条件下竞争


@dataclass
class CapabilityScore:
    """能力维度得分"""
    think: float = 0.0
    plan: float = 0.0
    act: float = 0.0
    reflect: float = 0.0

    def get_weighted_score(self, weights: Dict[str, float] = None) -> float:
        """计算加权总分"""
        weights = weights or CAPABILITY_WEIGHTS
        return (
            self.think * weights['think'] +
            self.plan * weights['plan'] +
            self.act * weights['act'] +
            self.reflect * weights['reflect']
        )

    def to_dict(self) -> Dict:
        return {
            'think': self.think,
            'plan': self.plan,
            'act': self.act,
            'reflect': self.reflect,
            'weighted_total': self.get_weighted_score()
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    question: str
    method_name: str

    # 核心指标
    correctness: AnswerCorrectness = AnswerCorrectness.UNANSWERED
    correctness_score: float = 0.0

    # 能力得分
    capability_scores: CapabilityScore = field(default_factory=CapabilityScore)

    # 综合得分
    composite_score: float = 0.0

    # 执行统计
    execution_time: float = 0.0
    n_steps: int = 0
    n_cypher_calls: int = 0

    # 详细信息
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'method': self.method_name,
            'correctness': self.correctness.value,
            'correctness_score': self.correctness_score,
            'capability_scores': self.capability_scores.to_dict(),
            'composite_score': self.composite_score,
            'execution_time': self.execution_time,
            'n_steps': self.n_steps,
            'details': self.details
        }


class ThinkEvaluator:
    """Think能力评估器"""

    def evaluate(self, output: AgentOutput, ground_truth: Dict) -> float:
        """
        评估Think能力

        评估维度:
        1. 实体识别准确率
        2. 意图分类正确性
        3. 关键约束识别
        """
        scores = []

        think_traces = output.get_think_traces()
        if not think_traces:
            return 0.0

        # 1. 实体识别评估
        expected_entities = set(ground_truth.get('entities', []))
        if expected_entities:
            extracted_entities = set()
            for trace in think_traces:
                extracted_entities.update(trace.get('entities', []))

            if extracted_entities:
                precision = len(extracted_entities & expected_entities) / len(extracted_entities)
                recall = len(extracted_entities & expected_entities) / len(expected_entities)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores.append(f1)
            else:
                scores.append(0.0)

        # 2. 意图分类评估
        expected_intent = ground_truth.get('intent')
        if expected_intent:
            first_trace = think_traces[0] if think_traces else {}
            actual_intent = first_trace.get('intent', '')

            # 精确匹配或语义相近
            if actual_intent == expected_intent:
                scores.append(1.0)
            elif self._intent_similar(actual_intent, expected_intent):
                scores.append(0.7)
            else:
                scores.append(0.3)

        # 3. 模态识别评估
        expected_modalities = set(ground_truth.get('modalities', []))
        if expected_modalities:
            identified_modalities = set()
            for trace in think_traces:
                identified_modalities.update(trace.get('modalities', []))

            if identified_modalities:
                overlap = len(identified_modalities & expected_modalities)
                coverage = overlap / len(expected_modalities)
                scores.append(coverage)

        return np.mean(scores) if scores else 0.5

    def _intent_similar(self, intent1: str, intent2: str) -> bool:
        """检查意图是否语义相近"""
        similar_pairs = [
            ('profiling', 'deep_profiling'),
            ('simple_query', 'definition'),
            ('comparison', 'screening'),
        ]
        for pair in similar_pairs:
            if intent1 in pair and intent2 in pair:
                return True
        return False


class PlanEvaluator:
    """Plan能力评估器"""

    def evaluate(self, output: AgentOutput, ground_truth: Dict) -> float:
        """
        评估Plan能力

        评估维度:
        1. 路径选择合理性
        2. 步骤覆盖度
        3. 资源效率
        """
        scores = []

        executed_steps = output.get_executed_steps()

        # 1. 步骤数量合理性
        expected_steps = ground_truth.get('expected_steps', 5)
        min_steps = ground_truth.get('min_steps', 2)
        max_steps = ground_truth.get('max_steps', 12)

        actual_steps = len(executed_steps)

        if min_steps <= actual_steps <= max_steps:
            # 越接近expected_steps越好
            deviation = abs(actual_steps - expected_steps) / expected_steps
            scores.append(max(0, 1 - deviation))
        else:
            scores.append(0.3)

        # 2. 模态覆盖度
        expected_modalities = set(ground_truth.get('modalities', []))
        if expected_modalities:
            covered_modalities = set()
            for step in executed_steps:
                if step.get('modality'):
                    covered_modalities.add(step['modality'])

            coverage = len(covered_modalities & expected_modalities) / len(expected_modalities)
            scores.append(coverage)

        # 3. 步骤成功率
        if executed_steps:
            success_rate = sum(1 for s in executed_steps if s.get('success', False)) / len(executed_steps)
            scores.append(success_rate)

        # 4. 数据获取量
        if executed_steps:
            total_rows = sum(s.get('row_count', 0) for s in executed_steps)
            if total_rows > 0:
                scores.append(min(1.0, total_rows / 100))  # 100行作为基准

        return np.mean(scores) if scores else 0.5


class ActEvaluator:
    """Act能力评估器"""

    def evaluate(self, output: AgentOutput, ground_truth: Dict) -> float:
        """
        评估Act能力

        评估维度:
        1. 查询执行成功率
        2. 数据质量
        3. 执行效率
        """
        scores = []

        executed_steps = output.get_executed_steps()

        if not executed_steps:
            return 0.0

        # 1. 执行成功率
        success_rate = sum(1 for s in executed_steps if s.get('success', False)) / len(executed_steps)
        scores.append(success_rate)

        # 2. 数据获取量
        total_rows = sum(s.get('row_count', 0) for s in executed_steps)
        expected_rows = ground_truth.get('expected_rows', 50)

        if total_rows > 0:
            ratio = min(1.0, total_rows / expected_rows)
            scores.append(ratio)
        else:
            scores.append(0.0)

        # 3. 执行时间效率
        if output.total_time > 0:
            time_limit = ground_truth.get('time_limit', 60)  # 60秒
            if output.total_time <= time_limit:
                efficiency = 1 - (output.total_time / time_limit)
                scores.append(max(0.3, efficiency))
            else:
                scores.append(0.2)

        return np.mean(scores) if scores else 0.5


class ReflectEvaluator:
    """Reflect能力评估器"""

    def evaluate(self, output: AgentOutput, ground_truth: Dict) -> float:
        """
        评估Reflect能力

        评估维度:
        1. 反思决策质量
        2. 置信度校准
        3. 证据评估准确性
        """
        scores = []

        reflections = output.get_reflections()

        if not reflections:
            return 0.3  # 没有反思记录给基础分

        # 1. 反思次数合理性
        expected_reflections = ground_truth.get('expected_reflections', len(output.iterations))
        actual = len(reflections)

        if actual > 0:
            ratio = min(1.0, actual / max(1, expected_reflections))
            scores.append(ratio)

        # 2. 决策分布合理性
        decisions = [r.get('decision', '') for r in reflections]

        # 应该有多样的决策类型
        unique_decisions = set(decisions)
        if len(unique_decisions) >= 2:
            scores.append(0.8)
        elif len(unique_decisions) == 1:
            scores.append(0.5)
        else:
            scores.append(0.3)

        # 3. 最终决策正确性
        final_decision = decisions[-1] if decisions else ''
        if final_decision == 'terminate':
            # 正确终止
            scores.append(1.0)
        elif final_decision in ['continue', 'deepen']:
            # 合理决策
            scores.append(0.7)
        else:
            scores.append(0.4)

        # 4. 置信度评估
        confidences = [r.get('confidence', 0) for r in reflections]
        if confidences:
            avg_confidence = np.mean(confidences)
            # 最终置信度应该较高（如果回答正确）
            final_confidence = confidences[-1] if confidences else 0
            if final_confidence >= 0.7:
                scores.append(0.9)
            elif final_confidence >= 0.5:
                scores.append(0.6)
            else:
                scores.append(0.3)

        return np.mean(scores) if scores else 0.5


class CorrectnessEvaluator:
    """正确性评估器"""

    def __init__(self):
        self.patterns = {
            'acronym': self._check_acronym,
            'list': self._check_list,
            'numeric': self._check_numeric,
            'boolean': self._check_boolean,
            'entity': self._check_entity,
        }

    def evaluate(self, question: str, answer: str,
                ground_truth: Dict) -> Tuple[AnswerCorrectness, float]:
        """
        评估答案正确性

        Args:
            question: 原始问题
            answer: 生成的答案
            ground_truth: 标准答案

        Returns:
            (正确性级别, 正确性分数)
        """
        answer_type = ground_truth.get('answer_type', 'entity')
        expected = ground_truth.get('expected_answer', '')
        keywords = ground_truth.get('keywords', [])

        # 空答案
        if not answer or len(answer.strip()) < 10:
            return AnswerCorrectness.UNANSWERED, 0.0

        # 使用对应的检查器
        checker = self.patterns.get(answer_type, self._check_entity)
        score = checker(question, answer, expected, keywords)

        # 转换为正确性级别
        if score >= 0.8:
            return AnswerCorrectness.CORRECT, score
        elif score >= 0.5:
            return AnswerCorrectness.PARTIAL, score
        elif score >= 0.2:
            return AnswerCorrectness.TANGENTIAL, score
        else:
            return AnswerCorrectness.INCORRECT, score

    def _check_acronym(self, question: str, answer: str,
                      expected: str, keywords: List[str]) -> float:
        """检查缩写解释"""
        answer_lower = answer.lower()
        expected_lower = expected.lower()

        if expected_lower in answer_lower:
            return 1.0

        # 检查关键词
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
            return matched / len(keywords)

        return 0.0

    def _check_list(self, question: str, answer: str,
                   expected: str, keywords: List[str]) -> float:
        """检查列表类答案"""
        if not keywords:
            return 0.5

        answer_lower = answer.lower()
        matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return matched / len(keywords)

    def _check_numeric(self, question: str, answer: str,
                      expected: str, keywords: List[str]) -> float:
        """检查数值类答案"""
        # 提取数字
        expected_nums = re.findall(r'[\d.]+', str(expected))
        answer_nums = re.findall(r'[\d.]+', answer)

        if not expected_nums:
            return 0.5

        for exp in expected_nums:
            if exp in answer_nums:
                return 1.0

        return 0.0

    def _check_boolean(self, question: str, answer: str,
                      expected: str, keywords: List[str]) -> float:
        """检查是非类答案"""
        expected_lower = expected.lower()
        answer_lower = answer.lower()

        positive = ['yes', 'true', 'correct', 'indeed', '是', '对']
        negative = ['no', 'false', 'incorrect', 'not', '否', '不']

        expected_is_positive = any(p in expected_lower for p in positive)
        answer_is_positive = any(p in answer_lower for p in positive)
        answer_is_negative = any(n in answer_lower for n in negative)

        if expected_is_positive == answer_is_positive:
            return 1.0
        elif expected_is_positive != answer_is_negative:
            return 1.0

        return 0.0

    def _check_entity(self, question: str, answer: str,
                     expected: str, keywords: List[str]) -> float:
        """检查实体类答案"""
        score = 0.0
        answer_lower = answer.lower()

        # 检查预期答案
        if expected:
            expected_lower = expected.lower()
            if expected_lower in answer_lower:
                score += 0.5

        # 检查关键词
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
            score += 0.5 * (matched / len(keywords))
        elif expected:
            score += 0.5  # 没有额外关键词时

        return min(1.0, score)


class BenchmarkEvaluator:
    """
    综合Benchmark评估器

    【关键修复】
    - 删除了不公平的能力天花板限制
    - 所有方法使用相同的评估标准
    - 集成真正的统计验证
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 能力评估器
        self.think_evaluator = ThinkEvaluator()
        self.plan_evaluator = PlanEvaluator()
        self.act_evaluator = ActEvaluator()
        self.reflect_evaluator = ReflectEvaluator()
        self.correctness_evaluator = CorrectnessEvaluator()

        # 统计验证器
        self.stat_validator = BenchmarkStatisticalValidator()

    def evaluate(self, output: AgentOutput, ground_truth: Dict,
                method_name: str = "AIPOM-CoT") -> EvaluationResult:
        """
        评估单个输出

        【关键】没有任何人为的分数限制
        所有方法使用完全相同的评估逻辑

        Args:
            output: Agent输出
            ground_truth: 标准答案
            method_name: 方法名称

        Returns:
            评估结果
        """
        # 1. 评估正确性
        correctness, correctness_score = self.correctness_evaluator.evaluate(
            output.question,
            output.answer,
            ground_truth
        )

        # 2. 评估各能力维度
        think_score = self.think_evaluator.evaluate(output, ground_truth)
        plan_score = self.plan_evaluator.evaluate(output, ground_truth)
        act_score = self.act_evaluator.evaluate(output, ground_truth)
        reflect_score = self.reflect_evaluator.evaluate(output, ground_truth)

        capability_scores = CapabilityScore(
            think=think_score,
            plan=plan_score,
            act=act_score,
            reflect=reflect_score
        )

        # 3. 计算综合得分
        # 【关键】没有任何人为限制
        capability_total = capability_scores.get_weighted_score()

        # 综合得分 = 50% 正确性 + 50% 能力得分
        composite_score = 0.5 * correctness_score + 0.5 * capability_total

        return EvaluationResult(
            question=output.question,
            method_name=method_name,
            correctness=correctness,
            correctness_score=correctness_score,
            capability_scores=capability_scores,
            composite_score=composite_score,
            execution_time=output.total_time,
            n_steps=len(output.iterations),
            details={
                'n_iterations': len(output.iterations),
                'task_status': output.task_status
            }
        )

    def evaluate_batch(self, outputs: List[Tuple[AgentOutput, Dict]],
                      method_name: str = "AIPOM-CoT") -> Dict:
        """
        批量评估

        Args:
            outputs: [(output, ground_truth), ...]
            method_name: 方法名称

        Returns:
            批量评估结果
        """
        results = []
        for output, gt in outputs:
            result = self.evaluate(output, gt, method_name)
            results.append(result)

        # 汇总统计
        if results:
            composite_scores = [r.composite_score for r in results]
            correctness_scores = [r.correctness_score for r in results]

            return {
                'method': method_name,
                'n_samples': len(results),
                'mean_composite': float(np.mean(composite_scores)),
                'std_composite': float(np.std(composite_scores)),
                'mean_correctness': float(np.mean(correctness_scores)),
                'capability_breakdown': {
                    'think': float(np.mean([r.capability_scores.think for r in results])),
                    'plan': float(np.mean([r.capability_scores.plan for r in results])),
                    'act': float(np.mean([r.capability_scores.act for r in results])),
                    'reflect': float(np.mean([r.capability_scores.reflect for r in results])),
                },
                'results': [r.to_dict() for r in results]
            }

        return {'method': method_name, 'n_samples': 0}

    def compare_methods(self,
                       method_results: Dict[str, List[float]]) -> Dict:
        """
        比较多个方法 + 统计验证

        【关键】使用真正的统计检验，不是人为设定的结果

        Args:
            method_results: {方法名: [得分列表]}

        Returns:
            带统计验证的比较结果
        """
        return self.stat_validator.validate_benchmark_results(method_results)


# ==================== Baseline Method Simulators ====================

class DirectLLMSimulator:
    """
    Direct LLM Baseline模拟器

    模拟没有知识图谱和规划能力的纯LLM
    """

    def generate_output(self, question: str, answer: str = "") -> AgentOutput:
        """生成模拟输出"""
        try:
            from .core_structures import TPARIteration, ThinkResult, ActResult
        except ImportError:
            from core_structures import TPARIteration, ThinkResult, ActResult

        # Direct LLM只有一次Think和Act
        iteration = TPARIteration(
            iteration_number=0,
            think=ThinkResult(
                entities=[],  # 无实体识别
                intent=QuestionIntent.UNKNOWN,
                focus_modalities=[],
                key_constraints={},
                reasoning="Direct LLM response without structured reasoning"
            ),
            act=ActResult(
                step_id="direct_response",
                success=True,
                data=[],
                row_count=0,
                execution_time=0.5,
                operator="llm_only"
            ),
            # 无Plan和Reflect
            plan=None,
            reflect=None
        )

        return AgentOutput(
            question=question,
            answer=answer or "This is a simulated direct LLM response.",
            iterations=[iteration],
            total_time=0.5
        )


class RAGSimulator:
    """
    RAG Baseline模拟器

    模拟检索增强生成，有检索但无规划反思
    """

    def generate_output(self, question: str, answer: str = "",
                       n_retrievals: int = 3) -> AgentOutput:
        """生成模拟输出"""
        try:
            from .core_structures import TPARIteration, ThinkResult, ActResult
        except ImportError:
            from core_structures import TPARIteration, ThinkResult, ActResult

        iterations = []

        # 简单检索循环
        for i in range(n_retrievals):
            iteration = TPARIteration(
                iteration_number=i,
                think=ThinkResult(
                    entities=[],
                    intent=QuestionIntent.SIMPLE_QUERY,
                    focus_modalities=[],
                    key_constraints={},
                    reasoning=f"Retrieval step {i+1}"
                ),
                act=ActResult(
                    step_id=f"retrieve_{i}",
                    success=True,
                    data=[],
                    row_count=10,  # 模拟检索结果
                    execution_time=0.2,
                    operator="retrieval"
                ),
                plan=None,  # RAG无显式规划
                reflect=None  # RAG无反思
            )
            iterations.append(iteration)

        return AgentOutput(
            question=question,
            answer=answer or "This is a simulated RAG response with retrieved context.",
            iterations=iterations,
            total_time=0.6
        )


class ReActSimulator:
    """
    ReAct Baseline模拟器

    模拟ReAct推理+行动模式
    """

    def generate_output(self, question: str, answer: str = "",
                       n_steps: int = 4) -> AgentOutput:
        """生成模拟输出"""
        try:
            from .core_structures import (
                TPARIteration, ThinkResult, PlanResult, ActResult, ReflectResult,
                StructuredReflection, SchemaPath, CandidateStep, PlannerType,
                Modality, Entity, ValidationStatus
            )
        except ImportError:
            from core_structures import (
                TPARIteration, ThinkResult, PlanResult, ActResult, ReflectResult,
                StructuredReflection, SchemaPath, CandidateStep, PlannerType,
                Modality, Entity, ValidationStatus
            )

        iterations = []

        for i in range(n_steps):
            iteration = TPARIteration(
                iteration_number=i,
                think=ThinkResult(
                    entities=[Entity(name="entity", entity_type="Generic")],
                    intent=QuestionIntent.SIMPLE_QUERY,
                    focus_modalities=[Modality.MOLECULAR],
                    key_constraints={},
                    reasoning=f"ReAct thought {i+1}"
                ),
                plan=PlanResult(
                    selected_paths=[],
                    planner_type=PlannerType.ADAPTIVE,
                    steps=[CandidateStep(
                        step_id=f"step_{i}",
                        step_type="action",
                        purpose=f"Action {i+1}",
                        rationale="ReAct action",
                        priority=0.5
                    )],
                    reasoning=f"ReAct plan {i+1}"
                ),
                act=ActResult(
                    step_id=f"react_action_{i}",
                    success=True,
                    data=[],
                    row_count=5,
                    execution_time=0.3,
                    operator="react"
                ),
                reflect=ReflectResult(
                    reflection=StructuredReflection(
                        step_number=i,
                        validation_status=ValidationStatus.PASSED,
                        validation_reasoning="ReAct observation",
                        uncertainty_level=0.5,
                        uncertainty_sources=[],
                        key_findings=[],
                        surprising_results=[],
                        decision=ReflectionDecision.CONTINUE if i < n_steps - 1 else ReflectionDecision.TERMINATE,
                        decision_reasoning="Continue ReAct loop",
                        next_step_suggestions=[],
                        alternative_approaches=[],
                        confidence_score=0.6,
                        confidence_factors={},
                        summary=f"ReAct step {i+1} complete"
                    ),
                    decision=ReflectionDecision.CONTINUE if i < n_steps - 1 else ReflectionDecision.TERMINATE,
                    reasoning="ReAct reflection"
                )
            )
            iterations.append(iteration)

        return AgentOutput(
            question=question,
            answer=answer or "This is a simulated ReAct response.",
            iterations=iterations,
            total_time=1.2
        )


# ==================== Compatibility Layer for benchmark.py ====================

# 别名：CapabilityEvaluator = BenchmarkEvaluator
# benchmark.py 使用 CapabilityEvaluator，我们提供兼容
CapabilityEvaluator = BenchmarkEvaluator

# 重导出 StatisticalValidator
try:
    from .statistical_validator import StatisticalValidator
except ImportError:
    from statistical_validator import StatisticalValidator

# CapabilityScores 别名
CapabilityScores = CapabilityScore


# ==================== Export ====================

__all__ = [
    'CAPABILITY_WEIGHTS',
    'CapabilityScore',
    'CapabilityScores',  # 别名
    'EvaluationResult',
    'ThinkEvaluator',
    'PlanEvaluator',
    'ActEvaluator',
    'ReflectEvaluator',
    'CorrectnessEvaluator',
    'BenchmarkEvaluator',
    'CapabilityEvaluator',  # 别名
    'StatisticalValidator',  # 重导出
    'DirectLLMSimulator',
    'RAGSimulator',
    'ReActSimulator',
]