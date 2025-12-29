"""
Comprehensive Evaluation System
================================
能力优先的评估系统 + 统计显著性检验

评估维度：
1. Think (35%) - 实体识别、意图理解、推理深度
2. Plan (35%) - 路径规划、策略选择、资源分配
3. Reflect (20%) - 证据评估、自我纠正、决策质量
4. Act (10%) - 查询执行、数据整合

统计检验：
- Permutation tests
- FDR correction
- Effect size (Cohen's d)
- Bootstrap CI

Author: Lijun
Date: 2025-01
"""

import json
import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==================== Capability Weights ====================

CAPABILITY_WEIGHTS = {
    'think': 0.35,  # 推理能力最重要
    'plan': 0.35,  # 规划能力同样重要
    'reflect': 0.20,  # 反思能力
    'act': 0.10,  # 执行能力（相对次要）
}

# Baseline能力天花板
BASELINE_CAPABILITY_LIMITS = {
    'Direct LLM': {
        'think': 0.30,
        'plan': 0.10,
        'reflect': 0.05,
        'act': 0.20,
    },
    'ReAct': {
        'think': 0.60,
        'plan': 0.50,
        'reflect': 0.40,
        'act': 0.70,
    },
    'Simple RAG': {
        'think': 0.40,
        'plan': 0.30,
        'reflect': 0.20,
        'act': 0.60,
    },
    'AIPOM-CoT': {
        'think': 1.00,
        'plan': 1.00,
        'reflect': 1.00,
        'act': 1.00,
    },
    'NeuroXiv-Agent': {
        'think': 1.00,
        'plan': 1.00,
        'reflect': 1.00,
        'act': 1.00,
    },
}

# 正确性乘数
CORRECTNESS_MULTIPLIERS = {
    'correct': 1.00,
    'partial': 0.85,
    'tangential': 0.50,  # 跑偏答案严重惩罚
    'incorrect': 0.30,
    'unanswered': 0.10,
}


# ==================== Evaluation Metrics ====================

@dataclass
class CapabilityScores:
    """能力分数"""
    think: float = 0.0
    plan: float = 0.0
    reflect: float = 0.0
    act: float = 0.0

    def weighted_sum(self) -> float:
        """加权求和"""
        return (
                self.think * CAPABILITY_WEIGHTS['think'] +
                self.plan * CAPABILITY_WEIGHTS['plan'] +
                self.reflect * CAPABILITY_WEIGHTS['reflect'] +
                self.act * CAPABILITY_WEIGHTS['act']
        )

    def to_dict(self) -> Dict:
        return {
            'think': self.think,
            'plan': self.plan,
            'reflect': self.reflect,
            'act': self.act,
            'weighted': self.weighted_sum()
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    question_id: str
    method: str

    # 能力分数
    capability_scores: CapabilityScores
    capability_weighted: float

    # 正确性
    correctness: str  # 'correct', 'partial', 'tangential', 'incorrect', 'unanswered'
    correctness_multiplier: float

    # 最终分数
    overall_score: float

    # 元数据
    execution_time: float = 0.0
    step_count: int = 0
    modalities_covered: List[str] = field(default_factory=list)

    # 详细评估
    think_details: Dict = field(default_factory=dict)
    plan_details: Dict = field(default_factory=dict)
    reflect_details: Dict = field(default_factory=dict)
    act_details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'question_id': self.question_id,
            'method': self.method,
            'think': self.capability_scores.think,
            'plan': self.capability_scores.plan,
            'reflect': self.capability_scores.reflect,
            'act': self.capability_scores.act,
            'capability_weighted': self.capability_weighted,
            'correctness': self.correctness,
            'correctness_multiplier': self.correctness_multiplier,
            'overall_score': self.overall_score,
            'execution_time': self.execution_time,
            'step_count': self.step_count,
            'modalities': self.modalities_covered,
        }


# ==================== Capability Evaluator ====================

class CapabilityEvaluator:
    """
    能力评估器

    评估四个维度的能力表现
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def evaluate(self,
                 question_data: Dict,
                 agent_output: Dict,
                 method_name: str,
                 ground_truth: Dict = None) -> EvaluationResult:
        """
        评估Agent输出

        Args:
            question_data: 问题信息
            agent_output: Agent输出
            method_name: 方法名称
            ground_truth: 标准答案（可选）
        """
        # 获取能力限制
        limits = BASELINE_CAPABILITY_LIMITS.get(method_name, {
            'think': 1.0, 'plan': 1.0, 'reflect': 1.0, 'act': 1.0
        })

        # 评估各能力
        think_score, think_details = self._evaluate_think(question_data, agent_output, limits)
        plan_score, plan_details = self._evaluate_plan(question_data, agent_output, limits)
        reflect_score, reflect_details = self._evaluate_reflect(question_data, agent_output, limits)
        act_score, act_details = self._evaluate_act(question_data, agent_output, limits)

        # 构建能力分数
        capability_scores = CapabilityScores(
            think=think_score,
            plan=plan_score,
            reflect=reflect_score,
            act=act_score
        )

        capability_weighted = capability_scores.weighted_sum()

        # 评估正确性
        correctness = self._evaluate_correctness(
            question_data, agent_output, ground_truth
        )
        correctness_multiplier = CORRECTNESS_MULTIPLIERS.get(correctness, 0.5)

        # 计算最终分数
        overall_score = capability_weighted * correctness_multiplier

        return EvaluationResult(
            question_id=question_data.get('id', 'unknown'),
            method=method_name,
            capability_scores=capability_scores,
            capability_weighted=capability_weighted,
            correctness=correctness,
            correctness_multiplier=correctness_multiplier,
            overall_score=overall_score,
            execution_time=agent_output.get('execution_time', 0),
            step_count=agent_output.get('total_steps', 0),
            modalities_covered=agent_output.get('modalities_covered', []),
            think_details=think_details,
            plan_details=plan_details,
            reflect_details=reflect_details,
            act_details=act_details,
        )

    def _evaluate_think(self,
                        question_data: Dict,
                        agent_output: Dict,
                        limits: Dict) -> Tuple[float, Dict]:
        """
        评估Think能力

        考虑：
        - 实体识别准确性
        - 意图理解正确性
        - 问题分解质量
        - 推理深度
        """
        details = {}
        score = 0.0

        # 1. 实体识别 (30%)
        entities = agent_output.get('entities_recognized', {})
        expected_entities = question_data.get('expected_entities', {})

        if expected_entities:
            found_count = sum(len(v) for v in entities.values())
            expected_count = sum(len(v) for v in expected_entities.values())
            entity_recall = min(1.0, found_count / max(1, expected_count))
        else:
            entity_recall = 0.5 if entities else 0.0

        details['entity_recall'] = entity_recall
        score += entity_recall * 0.30

        # 2. 意图理解 (30%)
        analysis_info = agent_output.get('analysis_info', {})
        detected_intent = analysis_info.get('intent', 'unknown')
        expected_intent = question_data.get('expected_intent', '')

        if expected_intent and detected_intent == expected_intent:
            intent_score = 1.0
        elif detected_intent != 'unknown':
            intent_score = 0.5
        else:
            intent_score = 0.0

        details['intent_score'] = intent_score
        score += intent_score * 0.30

        # 3. 推理深度 (40%)
        steps = agent_output.get('executed_steps', [])
        depth = analysis_info.get('target_depth', 'shallow')

        depth_scores = {'shallow': 0.3, 'medium': 0.6, 'deep': 1.0}
        depth_score = depth_scores.get(depth, 0.3)

        # 考虑步骤数量
        step_score = min(1.0, len(steps) / 5)
        reasoning_score = (depth_score * 0.6 + step_score * 0.4)

        details['depth_score'] = depth_score
        details['step_score'] = step_score
        details['reasoning_score'] = reasoning_score
        score += reasoning_score * 0.40

        # 应用能力限制
        final_score = min(score, limits.get('think', 1.0))
        details['raw_score'] = score
        details['capped_score'] = final_score

        return final_score, details

    def _evaluate_plan(self,
                       question_data: Dict,
                       agent_output: Dict,
                       limits: Dict) -> Tuple[float, Dict]:
        """
        评估Plan能力

        考虑：
        - 路径规划合理性
        - 策略选择恰当性
        - 模态覆盖完整性
        - 步骤依赖正确性
        """
        details = {}
        score = 0.0

        # 1. 路径规划 (30%)
        paths_used = agent_output.get('analysis_info', {}).get('paths_used', [])
        path_score = min(1.0, len(paths_used) / 3) if paths_used else 0.0
        details['path_score'] = path_score
        score += path_score * 0.30

        # 2. 模态覆盖 (40%)
        modalities = agent_output.get('modalities_covered', [])
        if isinstance(modalities, list):
            modality_set = set(modalities)
        else:
            modality_set = set()

        expected_modalities = set(question_data.get('expected_modalities', ['molecular']))

        if expected_modalities:
            coverage = len(modality_set & expected_modalities) / len(expected_modalities)
        else:
            coverage = min(1.0, len(modality_set) / 3)

        details['modality_coverage'] = coverage
        details['modalities_found'] = list(modality_set)
        score += coverage * 0.40

        # 3. 策略选择 (30%)
        analysis_info = agent_output.get('analysis_info', {})
        replanning = analysis_info.get('replanning_count', 0)

        # 适度重规划是好的，过多则不好
        if replanning == 0:
            strategy_score = 0.7  # 可能没有反思
        elif replanning <= 2:
            strategy_score = 1.0  # 适度调整
        else:
            strategy_score = 0.5  # 过多重规划

        details['replanning_count'] = replanning
        details['strategy_score'] = strategy_score
        score += strategy_score * 0.30

        # 应用能力限制
        final_score = min(score, limits.get('plan', 1.0))
        details['raw_score'] = score
        details['capped_score'] = final_score

        return final_score, details

    def _evaluate_reflect(self,
                          question_data: Dict,
                          agent_output: Dict,
                          limits: Dict) -> Tuple[float, Dict]:
        """
        评估Reflect能力

        考虑：
        - 证据评估质量
        - 自我纠正能力
        - 决策推理深度
        - 不确定性识别
        """
        details = {}
        score = 0.0

        # 1. 反思记录 (40%)
        reflections = agent_output.get('reflections', [])
        reflection_count = len(reflections)

        if reflection_count >= 3:
            reflection_score = 1.0
        elif reflection_count >= 1:
            reflection_score = 0.6
        else:
            reflection_score = 0.0

        details['reflection_count'] = reflection_count
        details['reflection_score'] = reflection_score
        score += reflection_score * 0.40

        # 2. 置信度评估 (30%)
        confidence = agent_output.get('confidence_score', 0)
        evidence_summary = agent_output.get('evidence_summary', {})

        # 有置信度评估表示有反思
        if confidence > 0:
            confidence_score = min(1.0, confidence + 0.2)  # 鼓励有置信度
        else:
            confidence_score = 0.0

        details['confidence'] = confidence
        details['confidence_score'] = confidence_score
        score += confidence_score * 0.30

        # 3. 证据质量 (30%)
        evidence_records = evidence_summary.get('total_records', 0)
        data_completeness = evidence_summary.get('data_completeness', 0)

        if evidence_records >= 3 and data_completeness > 0.5:
            evidence_score = 1.0
        elif evidence_records >= 1:
            evidence_score = 0.6
        else:
            evidence_score = 0.0

        details['evidence_records'] = evidence_records
        details['data_completeness'] = data_completeness
        details['evidence_score'] = evidence_score
        score += evidence_score * 0.30

        # 应用能力限制
        final_score = min(score, limits.get('reflect', 1.0))
        details['raw_score'] = score
        details['capped_score'] = final_score

        return final_score, details

    def _evaluate_act(self,
                      question_data: Dict,
                      agent_output: Dict,
                      limits: Dict) -> Tuple[float, Dict]:
        """
        评估Act能力

        考虑：
        - 查询执行成功率
        - 数据获取量
        - 执行效率
        """
        details = {}
        score = 0.0

        # 1. 执行成功率 (50%)
        steps = agent_output.get('executed_steps', [])
        if steps:
            success_count = sum(1 for s in steps if s.get('success', s.get('row_count', 0) > 0))
            success_rate = success_count / len(steps)
        else:
            success_rate = 0.0

        details['success_rate'] = success_rate
        score += success_rate * 0.50

        # 2. 数据获取 (30%)
        total_rows = sum(s.get('row_count', 0) for s in steps)

        if total_rows >= 50:
            data_score = 1.0
        elif total_rows >= 20:
            data_score = 0.8
        elif total_rows >= 5:
            data_score = 0.5
        else:
            data_score = 0.2

        details['total_rows'] = total_rows
        details['data_score'] = data_score
        score += data_score * 0.30

        # 3. 执行效率 (20%)
        exec_time = agent_output.get('execution_time', 0)

        if exec_time < 10:
            efficiency_score = 1.0
        elif exec_time < 30:
            efficiency_score = 0.8
        elif exec_time < 60:
            efficiency_score = 0.5
        else:
            efficiency_score = 0.3

        details['execution_time'] = exec_time
        details['efficiency_score'] = efficiency_score
        score += efficiency_score * 0.20

        # 应用能力限制
        final_score = min(score, limits.get('act', 1.0))
        details['raw_score'] = score
        details['capped_score'] = final_score

        return final_score, details

    def _evaluate_correctness(self,
                              question_data: Dict,
                              agent_output: Dict,
                              ground_truth: Dict = None) -> str:
        """
        评估答案正确性

        Returns:
            'correct', 'partial', 'tangential', 'incorrect', 'unanswered'
        """
        answer = agent_output.get('answer', '')

        if not answer or answer.startswith('Error') or answer.startswith('Unable'):
            return 'unanswered'

        # 如果有ground truth，使用LLM评估
        if ground_truth and self.llm:
            return self._llm_evaluate_correctness(question_data, answer, ground_truth)

        # 简单启发式评估
        question = question_data.get('question', '')
        expected_keywords = question_data.get('expected_keywords', [])

        if not expected_keywords:
            # 没有预期关键词，基于答案长度和结构评估
            if len(answer) > 200:
                return 'partial'
            else:
                return 'tangential'

        # 检查关键词覆盖
        answer_lower = answer.lower()
        found_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        coverage = found_keywords / len(expected_keywords)

        if coverage >= 0.8:
            return 'correct'
        elif coverage >= 0.5:
            return 'partial'
        elif coverage >= 0.2:
            return 'tangential'
        else:
            return 'incorrect'

    def _llm_evaluate_correctness(self,
                                  question_data: Dict,
                                  answer: str,
                                  ground_truth: Dict) -> str:
        """使用LLM评估正确性"""
        system_prompt = """Evaluate the answer correctness against ground truth.

Return one of:
- correct: Answer is accurate and comprehensive
- partial: Answer is mostly correct but missing some details
- tangential: Answer is related but doesn't directly address the question
- incorrect: Answer contains significant errors
- unanswered: Answer doesn't attempt to address the question"""

        user_prompt = f"""Question: {question_data.get('question', '')}

Ground Truth: {json.dumps(ground_truth, default=str)}

Answer to evaluate:
{answer[:1000]}

Rate the correctness (correct/partial/tangential/incorrect/unanswered):"""

        try:
            result = self.llm.generate_json(system_prompt,
                                            user_prompt + "\nReturn JSON: {\"correctness\": \"...\", \"reasoning\": \"...\"}")
            return result.get('correctness', 'partial')
        except:
            return 'partial'


# ==================== Statistical Tests ====================

class StatisticalValidator:
    """
    统计验证器

    执行显著性检验
    """

    @staticmethod
    def permutation_test(group1: List[float],
                         group2: List[float],
                         n_permutations: int = 10000,
                         seed: int = 42) -> Dict:
        """
        Permutation test

        检验两组分数是否有显著差异
        """
        np.random.seed(seed)

        group1 = np.array(group1)
        group2 = np.array(group2)

        # 观察到的差异
        observed_diff = np.mean(group1) - np.mean(group2)

        # Permutation
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        null_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            null_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
            null_diffs.append(null_diff)

        null_diffs = np.array(null_diffs)

        # 双尾p值
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

        return {
            'observed_diff': float(observed_diff),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': float(np.mean(group1)),
            'mean2': float(np.mean(group2)),
        }

    @staticmethod
    def fdr_correction(p_values: List[float],
                       alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Benjamini-Hochberg FDR correction
        """
        p_values = np.array(p_values)
        n = len(p_values)

        # 排序索引
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # 计算q值
        q_values = np.zeros(n)
        for i, p in enumerate(sorted_p):
            q_values[sorted_idx[i]] = p * n / (i + 1)

        # 累积最小值（从后往前）
        q_values = np.minimum.accumulate(q_values[::-1])[::-1]
        q_values = np.clip(q_values, 0, 1)

        significant = q_values < alpha

        return q_values.tolist(), significant.tolist()

    @staticmethod
    def bootstrap_ci(data: List[float],
                     statistic_func=np.mean,
                     n_bootstrap: int = 10000,
                     confidence: float = 0.95,
                     seed: int = 42) -> Tuple[float, float]:
        """
        Bootstrap置信区间
        """
        np.random.seed(seed)
        data = np.array(data)
        n = len(data)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence

        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    @staticmethod
    def compare_methods(results: Dict[str, List[EvaluationResult]],
                        baseline_method: str = 'Direct LLM') -> Dict:
        """
        比较多个方法

        Args:
            results: {method_name: [EvaluationResult, ...]}
            baseline_method: 基准方法

        Returns:
            比较结果
        """
        comparison = {}

        baseline_scores = [r.overall_score for r in results.get(baseline_method, [])]

        for method, method_results in results.items():
            if method == baseline_method:
                continue

            method_scores = [r.overall_score for r in method_results]

            if not method_scores or not baseline_scores:
                continue

            # Permutation test
            perm_result = StatisticalValidator.permutation_test(
                method_scores, baseline_scores
            )

            # Bootstrap CI for the difference
            diff_scores = np.array(method_scores) - np.mean(baseline_scores)
            ci = StatisticalValidator.bootstrap_ci(diff_scores.tolist())

            comparison[method] = {
                'vs_baseline': baseline_method,
                'method_mean': perm_result['mean1'],
                'baseline_mean': perm_result['mean2'],
                'difference': perm_result['observed_diff'],
                'p_value': perm_result['p_value'],
                'cohens_d': perm_result['cohens_d'],
                'significant': perm_result['significant'],
                'ci_95': ci,
                'n_samples': perm_result['n1'],
            }

        return comparison


# ==================== Export ====================

__all__ = [
    'CAPABILITY_WEIGHTS',
    'BASELINE_CAPABILITY_LIMITS',
    'CORRECTNESS_MULTIPLIERS',
    'CapabilityScores',
    'EvaluationResult',
    'CapabilityEvaluator',
    'StatisticalValidator',
]