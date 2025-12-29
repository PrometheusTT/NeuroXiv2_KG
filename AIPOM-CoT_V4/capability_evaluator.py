"""
AIPOM-CoT Capability Evaluator
==============================
评估AIPOM-CoT的核心能力，突出:
- Think Capability (推理能力): 实体识别、意图理解、问题分解
- Plan Capability (规划能力): 路径规划、策略选择、资源分配
- Reflect Capability (反思能力): 证据评估、自我纠正、决策调整

评分公式:
    overall = capability_score × correctness_multiplier

    capability_score = 0.35×Think + 0.35×Plan + 0.20×Reflect + 0.10×Act

    correctness_multiplier:
        - correct: 1.0
        - partial: 0.85
        - tangential: 0.5
        - incorrect: 0.3

Author: Lijun
Date: 2025-01
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core_structures import (
    Modality, Intent, PlannerType, ReflectionDecision,
    ThinkResult, PlanResult, ActResult, ReflectResult,
    TPARIteration, AgentOutput, AnswerCorrectness
)

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

# 能力权重配置 - 突出Think, Plan, Reflect三大能力
CAPABILITY_WEIGHTS = {
    'think': 0.35,  # 推理能力 - 高权重
    'plan': 0.35,  # 规划能力 - 高权重
    'reflect': 0.20,  # 反思能力 - 中权重
    'act': 0.10,  # 执行能力 - 低权重（所有方法都能执行）
}

# 正确性乘数
CORRECTNESS_MULTIPLIER = {
    'correct': 1.00,
    'partial': 0.85,
    'tangential': 0.50,
    'incorrect': 0.30,
    'unanswered': 0.10,
}

# Baseline方法的能力天花板 - 它们缺乏真正的Think/Plan/Reflect能力
BASELINE_CAPABILITY_LIMITS = {
    'Direct LLM': {
        'think': 0.30,  # 只有基础NLU，无结构化思考
        'plan': 0.10,  # 无规划能力
        'reflect': 0.05,  # 无反思能力
        'act': 0.20,  # 无工具使用
    },
    'Template-KG': {
        'think': 0.40,  # 模板匹配
        'plan': 0.30,  # 固定模板
        'reflect': 0.10,  # 无反思
        'act': 0.60,  # 可执行查询
    },
    'RAG': {
        'think': 0.50,  # 检索理解
        'plan': 0.20,  # 简单检索策略
        'reflect': 0.10,  # 无反思
        'act': 0.50,  # 检索执行
    },
    'ReAct': {
        'think': 0.60,  # 有推理
        'plan': 0.50,  # 有一定规划
        'reflect': 0.40,  # 有观察反馈
        'act': 0.70,  # 工具使用
    },
    'AIPOM-CoT': {
        'think': 1.00,  # 无上限
        'plan': 1.00,
        'reflect': 1.00,
        'act': 1.00,
    },
}


# ==================== Think Capability Evaluator ====================

class ThinkCapabilityEvaluator:
    """
    推理能力评估器 (Think Capability)

    评估:
    1. Entity Recognition: 实体识别准确性
    2. Intent Understanding: 意图理解准确性
    3. Reasoning Depth: 推理深度
    4. Problem Decomposition: 问题分解能力
    """

    def evaluate(self, agent_output: AgentOutput,
                 question_data: Dict,
                 method_name: str) -> Dict[str, float]:
        """评估推理能力"""
        cap = BASELINE_CAPABILITY_LIMITS.get(method_name, {}).get('think', 1.0)

        # 获取Think记录
        think_traces = agent_output.get_think_traces()

        if not think_traces:
            return {
                'think_score': min(0.1, cap),
                'think_details': {'no_think_traces': True}
            }

        # 1. 实体识别评估 (30%)
        entity_score = self._evaluate_entity_recognition(think_traces, question_data)

        # 2. 意图理解评估 (25%)
        intent_score = self._evaluate_intent_understanding(think_traces, question_data)

        # 3. 推理深度评估 (25%)
        reasoning_depth = self._evaluate_reasoning_depth(think_traces, agent_output)

        # 4. 问题分解评估 (20%)
        decomposition = self._evaluate_problem_decomposition(think_traces, agent_output)

        # 加权总分
        total = (
                entity_score * 0.30 +
                intent_score * 0.25 +
                reasoning_depth * 0.25 +
                decomposition * 0.20
        )

        final_score = min(total, cap)

        return {
            'think_score': final_score,
            'think_details': {
                'entity_recognition': entity_score,
                'intent_understanding': intent_score,
                'reasoning_depth': reasoning_depth,
                'problem_decomposition': decomposition,
                'think_iterations': len(think_traces),
            }
        }

    def _evaluate_entity_recognition(self, think_traces: List[Dict],
                                     question_data: Dict) -> float:
        """评估实体识别"""
        expected = set(e.lower() for e in question_data.get('expected_entities', []))

        # 收集所有识别的实体
        recognized = set()
        for trace in think_traces:
            recognized.update(e.lower() for e in trace.get('entities', []))

        if not expected:
            # 如果没有期望实体，检查是否识别了任何实体
            return 0.7 if recognized else 0.3

        # 计算召回率和精确率
        recall = len(expected & recognized) / len(expected) if expected else 0
        precision = len(expected & recognized) / len(recognized) if recognized else 0

        # F1 score
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0

        return f1

    def _evaluate_intent_understanding(self, think_traces: List[Dict],
                                       question_data: Dict) -> float:
        """评估意图理解"""
        expected_strategy = question_data.get('expected_strategy', 'adaptive')

        # 检查识别的意图
        for trace in think_traces:
            intent = trace.get('intent', '')

            # 映射意图到策略
            if intent == 'focus_driven' and expected_strategy in ['focus_driven', 'adaptive']:
                return 1.0
            elif intent == 'comparative' and expected_strategy in ['comparative', 'adaptive']:
                return 1.0
            elif intent == 'screening' and expected_strategy in ['screening', 'comparative']:
                return 1.0
            elif intent == 'simple_qa' and expected_strategy == 'adaptive':
                return 0.8

        # 部分匹配
        return 0.5 if think_traces else 0.2

    def _evaluate_reasoning_depth(self, think_traces: List[Dict],
                                  agent_output: AgentOutput) -> float:
        """评估推理深度"""
        # 基于reasoning文本的深度
        total_reasoning_length = 0
        reasoning_markers = 0

        for trace in think_traces:
            reasoning = trace.get('reasoning', '')
            total_reasoning_length += len(reasoning)

            # 统计推理标记词
            markers = ['because', 'therefore', 'since', 'indicates', 'suggests',
                       'reasoning', 'analysis', 'considering']
            reasoning_markers += sum(1 for m in markers if m in reasoning.lower())

        # 推理长度得分
        length_score = min(1.0, total_reasoning_length / 500)

        # 推理标记得分
        marker_score = min(1.0, reasoning_markers / 3)

        # 迭代深度得分
        iteration_score = min(1.0, len(think_traces) / 3)

        return (length_score * 0.4 + marker_score * 0.3 + iteration_score * 0.3)

    def _evaluate_problem_decomposition(self, think_traces: List[Dict],
                                        agent_output: AgentOutput) -> float:
        """评估问题分解能力"""
        # 检查是否分解为多个子任务
        executed_steps = agent_output.get_executed_steps()

        if not executed_steps:
            return 0.2

        # 不同类型的步骤
        step_types = set(s.get('modality', '') for s in executed_steps)

        # 分解深度
        decomposition_score = min(1.0, len(step_types) / 3)

        # 步骤连贯性
        coherence_score = 1.0 if len(executed_steps) >= 2 else 0.5

        return decomposition_score * 0.6 + coherence_score * 0.4


# ==================== Plan Capability Evaluator ====================

class PlanCapabilityEvaluator:
    """
    规划能力评估器 (Plan Capability)

    评估:
    1. Path Planning: 路径规划质量
    2. Strategy Selection: 策略选择适当性
    3. Resource Allocation: 资源分配效率
    4. Adaptability: 计划适应性
    """

    def evaluate(self, agent_output: AgentOutput,
                 question_data: Dict,
                 method_name: str) -> Dict[str, float]:
        """评估规划能力"""
        cap = BASELINE_CAPABILITY_LIMITS.get(method_name, {}).get('plan', 1.0)

        iterations = agent_output.iterations

        if not iterations:
            return {
                'plan_score': min(0.1, cap),
                'plan_details': {'no_iterations': True}
            }

        # 1. 路径规划质量 (30%)
        path_quality = self._evaluate_path_planning(iterations, question_data)

        # 2. 策略选择 (25%)
        strategy_quality = self._evaluate_strategy_selection(iterations, question_data)

        # 3. 资源分配 (25%)
        resource_efficiency = self._evaluate_resource_allocation(agent_output)

        # 4. 计划适应性 (20%)
        adaptability = self._evaluate_adaptability(iterations)

        # 加权总分
        total = (
                path_quality * 0.30 +
                strategy_quality * 0.25 +
                resource_efficiency * 0.25 +
                adaptability * 0.20
        )

        final_score = min(total, cap)

        return {
            'plan_score': final_score,
            'plan_details': {
                'path_planning': path_quality,
                'strategy_selection': strategy_quality,
                'resource_allocation': resource_efficiency,
                'adaptability': adaptability,
                'total_plans': len([i for i in iterations if i.plan]),
            }
        }

    def _evaluate_path_planning(self, iterations: List[TPARIteration],
                                question_data: Dict) -> float:
        """评估路径规划质量"""
        plans = [i.plan for i in iterations if i.plan]

        if not plans:
            return 0.2

        # 评估路径多样性
        all_paths = []
        for plan in plans:
            all_paths.extend(plan.selected_paths)

        if not all_paths:
            return 0.3

        # 路径覆盖的模态
        modalities_covered = set()
        for path in all_paths:
            path_str = ' '.join(path.nodes).lower()
            if 'morphology' in path_str:
                modalities_covered.add('morphological')
            elif 'projection' in path_str or 'target' in path_str:
                modalities_covered.add('projection')
            else:
                modalities_covered.add('molecular')

        coverage_score = len(modalities_covered) / 3.0

        # 路径数量合理性
        expected_modalities = question_data.get('expected_modalities', [])
        path_count_score = min(1.0, len(all_paths) / max(1, len(expected_modalities)))

        return coverage_score * 0.6 + path_count_score * 0.4

    def _evaluate_strategy_selection(self, iterations: List[TPARIteration],
                                     question_data: Dict) -> float:
        """评估策略选择"""
        plans = [i.plan for i in iterations if i.plan]

        if not plans:
            return 0.2

        expected_strategy = question_data.get('expected_strategy', 'adaptive')

        # 检查选择的planner类型
        for plan in plans:
            planner_type = plan.planner_type.value

            if planner_type == expected_strategy:
                return 1.0
            elif planner_type == 'focus_driven' and expected_strategy in ['focus_driven', 'adaptive']:
                return 0.9
            elif planner_type == 'comparative' and expected_strategy in ['comparative', 'screening']:
                return 0.9

        return 0.5

    def _evaluate_resource_allocation(self, agent_output: AgentOutput) -> float:
        """评估资源分配效率"""
        final_state = agent_output.final_state

        # 预算使用率
        budget_used = final_state.used_budget
        total_budget = final_state.total_budget

        # 成功率
        executed_steps = agent_output.get_executed_steps()
        successful = sum(1 for s in executed_steps if s.get('success', False))
        success_rate = successful / len(executed_steps) if executed_steps else 0

        # 效率分数（在合理预算内完成更多有效工作）
        efficiency = success_rate * (1.0 - 0.5 * (budget_used / total_budget))

        return max(0.2, min(1.0, efficiency + 0.3))

    def _evaluate_adaptability(self, iterations: List[TPARIteration]) -> float:
        """评估计划适应性"""
        plans = [i.plan for i in iterations if i.plan]

        if len(plans) < 2:
            return 0.5

        # 检查计划是否根据反馈调整
        adjustments = 0
        for i in range(1, len(iterations)):
            if iterations[i].plan and iterations[i - 1].reflect:
                # 检查是否根据反思调整了策略
                if iterations[i - 1].reflect.decision in [ReflectionDecision.PIVOT, ReflectionDecision.DEEPEN]:
                    adjustments += 1

        adaptability = min(1.0, adjustments / max(1, len(plans) - 1) + 0.5)

        return adaptability


# ==================== Reflect Capability Evaluator ====================

class ReflectCapabilityEvaluator:
    """
    反思能力评估器 (Reflect Capability)

    评估:
    1. Evidence Evaluation: 证据评估质量
    2. Self-Correction: 自我纠正能力
    3. Decision Quality: 决策质量
    4. Metacognition: 元认知能力
    """

    def evaluate(self, agent_output: AgentOutput,
                 question_data: Dict,
                 method_name: str) -> Dict[str, float]:
        """评估反思能力"""
        cap = BASELINE_CAPABILITY_LIMITS.get(method_name, {}).get('reflect', 1.0)

        reflections = agent_output.get_reflections()

        if not reflections:
            return {
                'reflect_score': min(0.1, cap),
                'reflect_details': {'no_reflections': True}
            }

        # 1. 证据评估质量 (30%)
        evidence_eval = self._evaluate_evidence_evaluation(reflections, agent_output)

        # 2. 自我纠正能力 (30%)
        self_correction = self._evaluate_self_correction(reflections, agent_output)

        # 3. 决策质量 (25%)
        decision_quality = self._evaluate_decision_quality(reflections, agent_output)

        # 4. 元认知能力 (15%)
        metacognition = self._evaluate_metacognition(reflections)

        # 加权总分
        total = (
                evidence_eval * 0.30 +
                self_correction * 0.30 +
                decision_quality * 0.25 +
                metacognition * 0.15
        )

        final_score = min(total, cap)

        return {
            'reflect_score': final_score,
            'reflect_details': {
                'evidence_evaluation': evidence_eval,
                'self_correction': self_correction,
                'decision_quality': decision_quality,
                'metacognition': metacognition,
                'total_reflections': len(reflections),
            }
        }

    def _evaluate_evidence_evaluation(self, reflections: List[Dict],
                                      agent_output: AgentOutput) -> float:
        """评估证据评估质量"""
        # 检查反思是否包含数据完整度和证据强度评估
        has_data_completeness = any('data_completeness' in r for r in reflections)
        has_evidence_strength = any('evidence_strength' in r for r in reflections)
        has_confidence = any('confidence' in r for r in reflections)

        # 评估值的合理性
        completeness_values = [r.get('data_completeness', 0) for r in reflections]
        strength_values = [r.get('evidence_strength', 0) for r in reflections]

        # 评估是否随迭代改善
        improvement = 0
        if len(completeness_values) >= 2:
            if completeness_values[-1] > completeness_values[0]:
                improvement = 0.2

        base_score = (
            0.3 if has_data_completeness else 0 +
                                              0.3 if has_evidence_strength else 0 +
                                                                                0.2 if has_confidence else 0
        )

        return min(1.0, base_score + improvement + 0.2)

    def _evaluate_self_correction(self, reflections: List[Dict],
                                  agent_output: AgentOutput) -> float:
        """评估自我纠正能力"""
        iterations = agent_output.iterations

        # 检查是否有策略调整
        pivots = sum(1 for r in reflections if r.get('decision') == 'pivot')
        deepens = sum(1 for r in reflections if r.get('decision') == 'deepen')

        # 检查失败后是否调整
        corrections_after_failure = 0
        for i, it in enumerate(iterations[:-1]):
            if it.act and not it.act.success:
                # 检查下一次迭代是否有调整
                if i + 1 < len(iterations) and iterations[i + 1].reflect:
                    if iterations[i + 1].reflect.decision in [ReflectionDecision.PIVOT, ReflectionDecision.DEEPEN]:
                        corrections_after_failure += 1

        # 计算分数
        adjustment_score = min(1.0, (pivots + deepens) / 3)
        correction_score = min(1.0, corrections_after_failure / 2 + 0.5) if corrections_after_failure > 0 else 0.5

        return adjustment_score * 0.6 + correction_score * 0.4

    def _evaluate_decision_quality(self, reflections: List[Dict],
                                   agent_output: AgentOutput) -> float:
        """评估决策质量"""
        if not reflections:
            return 0.2

        # 检查最终决策是否合理
        final_reflection = reflections[-1]
        final_decision = final_reflection.get('decision', '')
        final_confidence = final_reflection.get('confidence', 0)

        # 完成决策的质量
        if final_decision == 'complete':
            # 检查是否真的达到了完成标准
            if final_confidence >= 0.7:
                return 1.0
            elif final_confidence >= 0.5:
                return 0.8
            else:
                return 0.5  # 过早完成

        # 中止决策
        elif final_decision == 'abort':
            # 检查是否确实是预算耗尽
            if agent_output.final_state.remaining_budget() <= 0:
                return 0.7  # 合理中止
            else:
                return 0.4  # 过早放弃

        # 其他决策
        return 0.6

    def _evaluate_metacognition(self, reflections: List[Dict]) -> float:
        """评估元认知能力"""
        # 检查反思推理的质量
        total_reasoning_length = 0
        uncertainty_awareness = 0

        for r in reflections:
            reasoning = r.get('reasoning', '')
            total_reasoning_length += len(reasoning)

            # 检查不确定性意识
            uncertainty_words = ['uncertain', 'may', 'might', 'possibly', 'likely',
                                 'confidence', 'threshold', 'insufficient']
            if any(w in reasoning.lower() for w in uncertainty_words):
                uncertainty_awareness += 1

        # 推理长度得分
        length_score = min(1.0, total_reasoning_length / 300)

        # 不确定性意识得分
        awareness_score = min(1.0, uncertainty_awareness / len(reflections)) if reflections else 0

        return length_score * 0.5 + awareness_score * 0.5


# ==================== Act Capability Evaluator ====================

class ActCapabilityEvaluator:
    """
    执行能力评估器 (Act Capability)

    评估:
    1. Query Execution: 查询执行成功率
    2. Operator Usage: 算子使用多样性
    3. Data Integration: 数据整合能力
    """

    def evaluate(self, agent_output: AgentOutput,
                 question_data: Dict,
                 method_name: str) -> Dict[str, float]:
        """评估执行能力"""
        cap = BASELINE_CAPABILITY_LIMITS.get(method_name, {}).get('act', 1.0)

        executed_steps = agent_output.get_executed_steps()

        if not executed_steps:
            return {
                'act_score': min(0.1, cap),
                'act_details': {'no_executed_steps': True}
            }

        # 1. 查询执行成功率 (40%)
        success_rate = sum(1 for s in executed_steps if s.get('success')) / len(executed_steps)

        # 2. 算子使用多样性 (30%)
        operators = set(s.get('operator', '') for s in executed_steps if s.get('operator'))
        operator_diversity = min(1.0, len(operators) / 3)

        # 3. 模态覆盖 (30%)
        modalities = set(s.get('modality', '') for s in executed_steps if s.get('modality'))
        modality_coverage = min(1.0, len(modalities) / 3)

        # 加权总分
        total = (
                success_rate * 0.40 +
                operator_diversity * 0.30 +
                modality_coverage * 0.30
        )

        final_score = min(total, cap)

        return {
            'act_score': final_score,
            'act_details': {
                'success_rate': success_rate,
                'operator_diversity': operator_diversity,
                'modality_coverage': modality_coverage,
                'total_steps': len(executed_steps),
            }
        }


# ==================== Correctness Checker ====================

class CorrectnessChecker:
    """
    答案正确性检查器

    检查答案是否真正回答了问题
    """

    def check(self, question: str, answer: str, question_data: Dict) -> Dict:
        """检查答案正确性"""
        if not answer or len(answer.strip()) < 10:
            return {
                'level': AnswerCorrectness.UNANSWERED.value,
                'multiplier': CORRECTNESS_MULTIPLIER['unanswered'],
                'reasoning': 'Answer is empty or too short',
            }

        q_lower = question.lower()

        # 缩写展开问题
        if 'stand for' in q_lower or 'full name' in q_lower:
            return self._check_acronym(question, answer)

        # 比较问题
        if any(kw in q_lower for kw in ['compare', 'versus', 'vs', 'difference']):
            return self._check_comparison(question, answer)

        # 计数问题
        if 'how many' in q_lower:
            return self._check_count(answer)

        # 定义问题
        if 'define' in q_lower or ('what is' in q_lower and q_lower.endswith('?')):
            return self._check_definition(answer)

        # 一般问题
        return self._check_general(question, answer)

    def _check_acronym(self, question: str, answer: str) -> Dict:
        """检查缩写问题"""
        # 提取缩写
        match = re.search(r'what does (\w+) stand for', question.lower())
        acronym = match.group(1).upper() if match else ''

        # 检查是否展开了缩写
        patterns = [
            rf'{acronym.lower()}\s+stands?\s+for\s+([^\.]+)',
            rf'\*\*{acronym}\*\*\s+stands?\s+for',
            rf'(?:stands? for|means|abbreviation for)\s+([^\.]{5, 50})',
        ]

        for pattern in patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return {
                    'level': AnswerCorrectness.CORRECT.value,
                    'multiplier': CORRECTNESS_MULTIPLIER['correct'],
                    'reasoning': 'Acronym correctly expanded',
                }

        # 检查是否跑偏
        if any(kw in answer.lower() for kw in ['neurons', 'cells', 'expression', 'project']):
            return {
                'level': AnswerCorrectness.TANGENTIAL.value,
                'multiplier': CORRECTNESS_MULTIPLIER['tangential'],
                'reasoning': f'Discusses {acronym} but does not expand the acronym',
            }

        return {
            'level': AnswerCorrectness.PARTIAL.value,
            'multiplier': CORRECTNESS_MULTIPLIER['partial'],
            'reasoning': 'Unclear if acronym is expanded',
        }

    def _check_comparison(self, question: str, answer: str) -> Dict:
        """检查比较问题"""
        # 提取比较实体
        entities = re.findall(r'\b([A-Z][a-z]{2,})\b', question)
        mentioned = sum(1 for e in entities if e.lower() in answer.lower())

        # 检查比较标记
        comparison_markers = ['compare', 'versus', 'vs', 'differ', 'while', 'whereas', 'contrast']
        has_comparison = any(m in answer.lower() for m in comparison_markers) or '|' in answer

        if has_comparison and mentioned >= 2:
            return {
                'level': AnswerCorrectness.CORRECT.value,
                'multiplier': CORRECTNESS_MULTIPLIER['correct'],
                'reasoning': 'Comparison covers both entities',
            }
        elif mentioned >= 2:
            return {
                'level': AnswerCorrectness.PARTIAL.value,
                'multiplier': CORRECTNESS_MULTIPLIER['partial'],
                'reasoning': 'Entities mentioned but comparison weak',
            }

        return {
            'level': AnswerCorrectness.PARTIAL.value,
            'multiplier': CORRECTNESS_MULTIPLIER['partial'],
            'reasoning': 'Missing entity coverage',
        }

    def _check_count(self, answer: str) -> Dict:
        """检查计数问题"""
        # 查找数字
        patterns = [
            r'there (?:are|is) \*?\*?(\d+)\*?\*?',
            r'(\d+)\s+(?:clusters?|neurons?|cells?|regions?)',
            r'\*\*(\d+)\*\*',
        ]

        for pattern in patterns:
            if re.search(pattern, answer.lower()):
                return {
                    'level': AnswerCorrectness.CORRECT.value,
                    'multiplier': CORRECTNESS_MULTIPLIER['correct'],
                    'reasoning': 'Count provided',
                }

        return {
            'level': AnswerCorrectness.PARTIAL.value,
            'multiplier': CORRECTNESS_MULTIPLIER['partial'],
            'reasoning': 'No specific count found',
        }

    def _check_definition(self, answer: str) -> Dict:
        """检查定义问题"""
        definition_patterns = [r'is (a|an|the)', r'refers to', r'defined as', r'means']

        if any(re.search(p, answer.lower()) for p in definition_patterns):
            return {
                'level': AnswerCorrectness.CORRECT.value,
                'multiplier': CORRECTNESS_MULTIPLIER['correct'],
                'reasoning': 'Definition provided',
            }

        return {
            'level': AnswerCorrectness.PARTIAL.value,
            'multiplier': CORRECTNESS_MULTIPLIER['partial'],
            'reasoning': 'No clear definition',
        }

    def _check_general(self, question: str, answer: str) -> Dict:
        """检查一般问题"""
        # 关键词覆盖
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        a_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        q_words -= {'what', 'about', 'does', 'have', 'that', 'this', 'with', 'from', 'tell'}

        if not q_words:
            return {
                'level': AnswerCorrectness.PARTIAL.value,
                'multiplier': CORRECTNESS_MULTIPLIER['partial'],
                'reasoning': 'Unable to assess',
            }

        coverage = len(q_words & a_words) / len(q_words)

        if coverage > 0.6 and len(answer) > 100:
            return {
                'level': AnswerCorrectness.CORRECT.value,
                'multiplier': CORRECTNESS_MULTIPLIER['correct'],
                'reasoning': f'Good coverage: {coverage:.0%}',
            }
        elif coverage > 0.4:
            return {
                'level': AnswerCorrectness.PARTIAL.value,
                'multiplier': CORRECTNESS_MULTIPLIER['partial'],
                'reasoning': f'Partial coverage: {coverage:.0%}',
            }

        return {
            'level': AnswerCorrectness.TANGENTIAL.value,
            'multiplier': CORRECTNESS_MULTIPLIER['tangential'],
            'reasoning': f'Low coverage: {coverage:.0%}',
        }


# ==================== Comprehensive Evaluator ====================

@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 能力分数
    think_score: float = 0.0
    plan_score: float = 0.0
    reflect_score: float = 0.0
    act_score: float = 0.0
    capability_score: float = 0.0

    # 正确性
    correctness_level: str = ""
    correctness_multiplier: float = 1.0
    correctness_reasoning: str = ""

    # 最终分数
    overall_score: float = 0.0

    # 详情
    think_details: Dict = field(default_factory=dict)
    plan_details: Dict = field(default_factory=dict)
    reflect_details: Dict = field(default_factory=dict)
    act_details: Dict = field(default_factory=dict)

    # 系统信息
    total_iterations: int = 0
    execution_time: float = 0.0
    task_status: str = ""


class ComprehensiveEvaluator:
    """
    综合评估器

    评分公式:
        overall = capability_score × correctness_multiplier
        capability_score = 0.35×Think + 0.35×Plan + 0.20×Reflect + 0.10×Act
    """

    def __init__(self):
        self.think_evaluator = ThinkCapabilityEvaluator()
        self.plan_evaluator = PlanCapabilityEvaluator()
        self.reflect_evaluator = ReflectCapabilityEvaluator()
        self.act_evaluator = ActCapabilityEvaluator()
        self.correctness_checker = CorrectnessChecker()

    def evaluate(self,
                 question_data: Dict,
                 agent_output: AgentOutput,
                 method_name: str = 'AIPOM-CoT') -> EvaluationMetrics:
        """
        综合评估

        Args:
            question_data: 问题数据
            agent_output: Agent输出
            method_name: 方法名称

        Returns:
            EvaluationMetrics
        """
        metrics = EvaluationMetrics()

        # 1. 评估四项能力
        think_result = self.think_evaluator.evaluate(agent_output, question_data, method_name)
        plan_result = self.plan_evaluator.evaluate(agent_output, question_data, method_name)
        reflect_result = self.reflect_evaluator.evaluate(agent_output, question_data, method_name)
        act_result = self.act_evaluator.evaluate(agent_output, question_data, method_name)

        metrics.think_score = think_result['think_score']
        metrics.plan_score = plan_result['plan_score']
        metrics.reflect_score = reflect_result['reflect_score']
        metrics.act_score = act_result['act_score']

        metrics.think_details = think_result.get('think_details', {})
        metrics.plan_details = plan_result.get('plan_details', {})
        metrics.reflect_details = reflect_result.get('reflect_details', {})
        metrics.act_details = act_result.get('act_details', {})

        # 2. 计算能力综合分
        metrics.capability_score = (
                metrics.think_score * CAPABILITY_WEIGHTS['think'] +
                metrics.plan_score * CAPABILITY_WEIGHTS['plan'] +
                metrics.reflect_score * CAPABILITY_WEIGHTS['reflect'] +
                metrics.act_score * CAPABILITY_WEIGHTS['act']
        )

        # 3. 检查正确性
        question = question_data.get('question', '')
        answer = agent_output.answer

        correctness = self.correctness_checker.check(question, answer, question_data)
        metrics.correctness_level = correctness['level']
        metrics.correctness_multiplier = correctness['multiplier']
        metrics.correctness_reasoning = correctness['reasoning']

        # 4. 计算最终分数
        metrics.overall_score = metrics.capability_score * metrics.correctness_multiplier

        # 5. 系统信息
        metrics.total_iterations = len(agent_output.iterations)
        metrics.execution_time = agent_output.total_time
        metrics.task_status = agent_output.task_status

        return metrics

    def compare_methods(self, question_data: Dict,
                        outputs: Dict[str, AgentOutput]) -> None:
        """对比不同方法"""
        print(f"\n{'=' * 90}")
        print(f"📌 Question: {question_data.get('question', '')[:60]}...")
        print(f"{'=' * 90}")

        print(f"\n📊 Capability Weights: Think {CAPABILITY_WEIGHTS['think'] * 100:.0f}%, "
              f"Plan {CAPABILITY_WEIGHTS['plan'] * 100:.0f}%, "
              f"Reflect {CAPABILITY_WEIGHTS['reflect'] * 100:.0f}%, "
              f"Act {CAPABILITY_WEIGHTS['act'] * 100:.0f}%")

        print(f"\n{'Method':<15} {'Think':>7} {'Plan':>6} {'Reflect':>8} {'Act':>5} "
              f"{'Capability':>11} {'×Correct':>9} {'OVERALL':>8}")
        print("-" * 85)

        for method, output in outputs.items():
            m = self.evaluate(question_data, output, method)

            mult_str = f"×{m.correctness_multiplier:.2f}"
            if m.correctness_multiplier < 0.6:
                mult_str += "⚠️"

            print(f"{method:<15} {m.think_score:>7.3f} {m.plan_score:>6.3f} "
                  f"{m.reflect_score:>8.3f} {m.act_score:>5.3f} "
                  f"{m.capability_score:>11.3f} {mult_str:>9} {m.overall_score:>8.3f}")

        print("-" * 85)


# ==================== Export ====================

__all__ = [
    'CAPABILITY_WEIGHTS',
    'CORRECTNESS_MULTIPLIER',
    'BASELINE_CAPABILITY_LIMITS',
    'ThinkCapabilityEvaluator',
    'PlanCapabilityEvaluator',
    'ReflectCapabilityEvaluator',
    'ActCapabilityEvaluator',
    'CorrectnessChecker',
    'EvaluationMetrics',
    'ComprehensiveEvaluator',
]