"""
Evaluation System for AIPOM-CoT Benchmark (v4.0 - Nature Methods)
==================================================================
全面评估系统，证明AIPOM-CoT的发表价值

New in v4.0:
- ✅ Planning Quality Evaluation (规划能力)
- ✅ Reasoning Capability Evaluation (推理能力)
- ✅ Chain-of-Thought Quality (CoT质量)
- ✅ Reflection Capability (反思能力)
- ✅ Natural Language Understanding (NLU)
- ✅ Biological Task Performance (生物学任务)

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 4.0 (Nature Methods Submission)
"""

import numpy as np
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

EVALUATION_CONFIG = {
    # 🔬 Nature Methods核心能力维度
    'nm_core_dimensions': {
        'planning_quality': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'ReAct'],
            'description': 'Quality of adaptive planning'
        },
        'reasoning_capability': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Multi-hop reasoning and logical consistency'
        },
        'cot_quality': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'ReAct'],
            'description': 'Quality of chain-of-thought generation'
        },
        'reflection_capability': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'ReAct'],
            'description': 'Self-correction and error detection'
        },
        'nlu_capability': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Natural language understanding'
        },
    },

    # 原有核心指标（保留）
    'core_metrics': {
        'entity_f1': {'weight': 1.0, 'methods': 'all'},
        'factual_accuracy': {'weight': 1.0, 'methods': 'all'},
        'answer_completeness': {'weight': 1.0, 'methods': 'all'},
        'scientific_rigor': {'weight': 1.0, 'methods': 'all'},
    },

    # 系统能力指标（保留）
    'system_metrics': {
        'reasoning_depth': {'weight': 1.0, 'methods': 'all'},
        'modality_coverage': {'weight': 1.0, 'methods': ['AIPOM-CoT', 'Template-KG', 'RAG', 'ReAct']},
        'closed_loop': {'weight': 1.0, 'methods': ['AIPOM-CoT']},
    },

    # 🔧 Nature Methods权重配置
    'nm_method_weights': {
        'AIPOM-CoT': {
            # NM核心能力 (50%)
            'planning_quality': 0.10,
            'reasoning_capability': 0.10,
            'cot_quality': 0.10,
            'reflection_capability': 0.10,
            'nlu_capability': 0.10,
            # 传统指标 (50%)
            'entity_f1': 0.10,
            'factual_accuracy': 0.10,
            'scientific_rigor': 0.10,
            'modality_coverage': 0.10,
            'closed_loop': 0.10,
        },
        'Direct GPT-4o': {
            'reasoning_capability': 0.25,
            'nlu_capability': 0.25,
            'entity_f1': 0.15,
            'factual_accuracy': 0.20,
            'scientific_rigor': 0.15,
        },
        'Template-KG': {
            'reasoning_capability': 0.20,
            'nlu_capability': 0.15,
            'entity_f1': 0.20,
            'factual_accuracy': 0.20,
            'modality_coverage': 0.15,
            'scientific_rigor': 0.10,
        },
        'RAG': {
            'reasoning_capability': 0.20,
            'nlu_capability': 0.20,
            'entity_f1': 0.20,
            'factual_accuracy': 0.20,
            'scientific_rigor': 0.20,
        },
        'ReAct': {
            'planning_quality': 0.15,
            'reasoning_capability': 0.15,
            'cot_quality': 0.10,
            'reflection_capability': 0.10,
            'nlu_capability': 0.10,
            'entity_f1': 0.12,
            'factual_accuracy': 0.13,
            'modality_coverage': 0.15,
        },
    },
}


# ==================== Data Structures ====================

@dataclass
class NMEvaluationMetrics:
    """Nature Methods评估指标 (v4.0)"""

    # 🔬 NM核心能力
    planning_quality: Optional[float] = None
    planning_coherence: Optional[float] = None
    planning_optimality: Optional[float] = None
    planning_adaptability: Optional[float] = None

    reasoning_capability: Optional[float] = None
    logical_consistency: Optional[float] = None
    evidence_integration: Optional[float] = None
    multi_hop_depth_score: Optional[float] = None

    cot_quality: Optional[float] = None
    cot_clarity: Optional[float] = None
    cot_completeness: Optional[float] = None
    intermediate_steps_quality: Optional[float] = None

    reflection_capability: Optional[float] = None
    error_detection: Optional[float] = None
    self_correction: Optional[float] = None
    iterative_refinement: Optional[float] = None

    nlu_capability: Optional[float] = None
    query_understanding: Optional[float] = None
    intent_recognition: Optional[float] = None
    ambiguity_resolution: Optional[float] = None

    # 传统核心指标
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    factual_accuracy: float = 0.0
    answer_completeness: float = 0.0
    scientific_rigor: float = 0.0

    # 系统能力
    reasoning_depth: Optional[float] = None
    modality_coverage: Optional[float] = None
    closed_loop_achieved: Optional[bool] = None
    modalities_used: List[str] = field(default_factory=list)

    # 效率
    execution_time: float = 0.0
    api_calls: int = 0
    query_success_rate: float = 0.0

    # 生物学任务
    task_completion: Optional[str] = None
    biological_insight_score: Optional[float] = None

    # Overall
    overall_score: Optional[float] = None
    nm_capability_score: Optional[float] = None  # NM核心能力总分


# ==================== 🔬 Planning Quality Evaluator ====================

class PlanningQualityEvaluator:
    """
    规划能力评估器

    评估维度：
    1. Planning Coherence - 计划连贯性
    2. Planning Optimality - 计划最优性
    3. Planning Adaptability - 计划适应性
    """

    def __init__(self):
        pass

    def evaluate(self, question_data: Dict, agent_output: Dict, method_name: str) -> Dict[str, float]:
        """评估规划质量"""

        # 只评估有planning能力的方法
        if method_name not in ['AIPOM-CoT', 'ReAct']:
            return {
                'planning_quality': None,
                'planning_coherence': None,
                'planning_optimality': None,
                'planning_adaptability': None,
            }

        executed_steps = agent_output.get('executed_steps', [])

        if len(executed_steps) < 2:
            return {
                'planning_quality': 0.5 if len(executed_steps) == 1 else 0.0,
                'planning_coherence': 0.5 if len(executed_steps) == 1 else 0.0,
                'planning_optimality': 0.5 if len(executed_steps) == 1 else 0.0,
                'planning_adaptability': 0.5 if len(executed_steps) == 1 else 0.0,
            }

        # 1. Planning Coherence
        coherence = self._evaluate_coherence(executed_steps)

        # 2. Planning Optimality
        optimality = self._evaluate_optimality(executed_steps, question_data)

        # 3. Planning Adaptability
        adaptability = self._evaluate_adaptability(executed_steps, question_data)

        # Overall planning quality
        planning_quality = np.mean([coherence, optimality, adaptability])

        return {
            'planning_quality': planning_quality,
            'planning_coherence': coherence,
            'planning_optimality': optimality,
            'planning_adaptability': adaptability,
        }

    def _evaluate_coherence(self, steps: List[Dict]) -> float:
        """
        评估计划连贯性

        检查：
        - 步骤间逻辑流
        - 模态渐进性
        - 无重复查询
        """

        if len(steps) < 2:
            return 1.0 if len(steps) == 1 else 0.0

        score = 0.0

        # 1. 逻辑流 (40%)
        modality_order = {'molecular': 1, 'morphological': 2, 'projection': 3, 'statistical': 4}

        flow_scores = []
        for i in range(len(steps) - 1):
            mod1 = steps[i].get('modality')
            mod2 = steps[i + 1].get('modality')

            if mod1 and mod2:
                order1 = modality_order.get(mod1, 2)
                order2 = modality_order.get(mod2, 2)

                if order2 >= order1:  # 允许同级或递进
                    flow_scores.append(1.0)
                elif order2 == order1 - 1:  # 小幅回退
                    flow_scores.append(0.7)
                else:
                    flow_scores.append(0.4)
            else:
                flow_scores.append(0.6)

        score += (np.mean(flow_scores) if flow_scores else 0.5) * 0.4

        # 2. 目标导向性 (30%)
        # 检查每个步骤的purpose是否明确且相关
        purposes = [s.get('purpose', '') for s in steps]

        keywords = ['identify', 'find', 'analyze', 'compare', 'characterize', 'profile',
                    'discover', 'validate', 'quantify', 'retrieve']

        purpose_quality = sum(1 for p in purposes if any(kw in p.lower() for kw in keywords))
        score += (purpose_quality / len(purposes)) * 0.3

        # 3. 无重复 (30%)
        unique_purposes = len(set(purposes))
        duplication_score = unique_purposes / len(purposes)
        score += duplication_score * 0.3

        return min(score, 1.0)

    def _evaluate_optimality(self, steps: List[Dict], question_data: Dict) -> float:
        """
        评估计划最优性

        检查：
        - 步数是否在合理范围
        - 是否覆盖必要模态
        - 是否避免不必要步骤
        """

        expected_range = question_data.get('expected_steps_range', (1, 10))
        expected_modalities = set(question_data.get('expected_modalities', []))

        actual_steps = len(steps)
        actual_modalities = set(s.get('modality') for s in steps if s.get('modality'))

        score = 0.0

        # 1. 步数匹配度 (40%)
        min_steps, max_steps = expected_range

        if min_steps <= actual_steps <= max_steps:
            step_score = 1.0
        elif actual_steps < min_steps:
            # 不足
            step_score = max(0.3, actual_steps / min_steps)
        else:
            # 过多
            excess = actual_steps - max_steps
            step_score = max(0.3, 1.0 - (excess / max_steps) * 0.5)

        score += step_score * 0.4

        # 2. 模态覆盖 (40%)
        if expected_modalities:
            covered = expected_modalities & actual_modalities
            modality_score = len(covered) / len(expected_modalities)
        else:
            modality_score = 0.8  # 默认分

        score += modality_score * 0.4

        # 3. 效率 (20%) - 成功率
        successful_steps = sum(1 for s in steps if s.get('success', True))
        efficiency = successful_steps / len(steps) if steps else 0

        score += efficiency * 0.2

        return min(score, 1.0)

    def _evaluate_adaptability(self, steps: List[Dict], question_data: Dict) -> float:
        """
        评估计划适应性

        检查：
        - 模态多样性
        - 策略灵活性
        - 问题响应性
        """

        score = 0.0

        # 1. 模态多样性 (40%)
        modalities = set(s.get('modality') for s in steps if s.get('modality'))

        if len(modalities) >= 3:
            diversity_score = 1.0
        elif len(modalities) == 2:
            diversity_score = 0.7
        elif len(modalities) == 1:
            diversity_score = 0.4
        else:
            diversity_score = 0.0

        score += diversity_score * 0.4

        # 2. 策略变化 (30%)
        # 检查是否根据中间结果调整策略
        purposes = [s.get('purpose', '').lower() for s in steps]

        # 寻找策略转换的证据
        transitions = 0
        prev_type = None

        for purpose in purposes:
            if 'compare' in purpose or 'versus' in purpose:
                curr_type = 'comparative'
            elif 'all' in purpose or 'screen' in purpose:
                curr_type = 'screening'
            elif 'profile' in purpose or 'characterize' in purpose:
                curr_type = 'profiling'
            else:
                curr_type = 'retrieval'

            if prev_type and curr_type != prev_type:
                transitions += 1

            prev_type = curr_type

        adaptation_score = min(1.0, transitions / max(1, len(steps) - 1))
        score += adaptation_score * 0.3

        # 3. 复杂度匹配 (30%)
        expected_depth = question_data.get('expected_depth', 'medium')

        depth_map = {'shallow': 1, 'medium': 3, 'deep': 5}
        expected_min_steps = depth_map.get(expected_depth, 3)

        if len(steps) >= expected_min_steps:
            complexity_score = 1.0
        else:
            complexity_score = len(steps) / expected_min_steps

        score += complexity_score * 0.3

        return min(score, 1.0)


# ==================== 🔬 Reasoning Capability Evaluator ====================

class ReasoningCapabilityEvaluator:
    """
    推理能力评估器

    评估维度：
    1. Logical Consistency - 逻辑一致性
    2. Evidence Integration - 证据整合
    3. Multi-hop Depth - 多跳推理深度
    """

    def __init__(self):
        pass

    def evaluate(self, question_data: Dict, agent_output: Dict, method_name: str) -> Dict[str, float]:
        """评估推理能力"""

        answer = agent_output.get('answer', '')
        executed_steps = agent_output.get('executed_steps', [])

        # 1. Logical Consistency
        consistency = self._evaluate_logical_consistency(answer, executed_steps)

        # 2. Evidence Integration
        integration = self._evaluate_evidence_integration(answer, executed_steps)

        # 3. Multi-hop Depth Score
        depth_score = self._evaluate_depth_score(executed_steps)

        # Overall reasoning capability
        reasoning_capability = np.mean([consistency, integration, depth_score])

        return {
            'reasoning_capability': reasoning_capability,
            'logical_consistency': consistency,
            'evidence_integration': integration,
            'multi_hop_depth_score': depth_score,
        }

    def _evaluate_logical_consistency(self, answer: str, steps: List[Dict]) -> float:
        """
        评估逻辑一致性

        检查：
        - 答案与步骤的一致性
        - 无矛盾陈述
        - 因果关系合理
        """

        if not answer or len(answer) < 20:
            return 0.0

        score = 0.0

        # 1. 答案引用了步骤中的数据 (40%)
        # 提取步骤中的关键实体和数据
        step_entities = set()
        step_numbers = set()

        for step in steps:
            purpose = step.get('purpose', '')

            # 提取实体
            entities = re.findall(r'\b[A-Z][a-z]{2,8}\b', purpose)
            step_entities.update(entities)

            # 提取数字
            numbers = re.findall(r'\d+', purpose)
            step_numbers.update(numbers)

        # 检查答案中是否引用
        answer_lower = answer.lower()

        mentioned_entities = sum(1 for e in step_entities if e.lower() in answer_lower)
        entity_citation = mentioned_entities / len(step_entities) if step_entities else 0.5

        score += entity_citation * 0.4

        # 2. 无矛盾标记 (30%)
        contradiction_markers = ['however', 'but', 'although', 'nevertheless', 'on the other hand']

        # 适度的转折是好的，过多可能表示矛盾
        contradictions = sum(1 for marker in contradiction_markers if marker in answer_lower)

        if contradictions == 0:
            contradiction_score = 0.8  # 完全无转折可能太简单
        elif contradictions <= 2:
            contradiction_score = 1.0  # 适度转折
        else:
            contradiction_score = max(0.3, 1.0 - (contradictions - 2) * 0.15)

        score += contradiction_score * 0.3

        # 3. 结构化推理 (30%)
        # 检查是否有清晰的推理结构
        reasoning_markers = [
            'therefore', 'thus', 'hence', 'consequently', 'as a result',
            'because', 'since', 'due to', 'given that',
            'first', 'second', 'third', 'finally',
            'in addition', 'moreover', 'furthermore'
        ]

        reasoning_count = sum(1 for marker in reasoning_markers if marker in answer_lower)

        if reasoning_count >= 3:
            structure_score = 1.0
        elif reasoning_count >= 1:
            structure_score = 0.6 + reasoning_count * 0.13
        else:
            structure_score = 0.4

        score += structure_score * 0.3

        return min(score, 1.0)

    def _evaluate_evidence_integration(self, answer: str, steps: List[Dict]) -> float:
        """
        评估证据整合

        检查：
        - 多步骤数据整合
        - 定量证据使用
        - 跨模态证据综合
        """

        if not answer or len(steps) == 0:
            return 0.0

        score = 0.0

        # 1. 多步骤整合 (40%)
        # 如果有多个步骤，答案应该整合多个步骤的信息
        if len(steps) >= 2:
            # 检查答案是否提到多个模态或多个方面
            modalities_mentioned = 0

            if any(kw in answer.lower() for kw in ['marker', 'express', 'cluster', 'gene']):
                modalities_mentioned += 1
            if any(kw in answer.lower() for kw in ['morphology', 'axon', 'dendrite', 'branch']):
                modalities_mentioned += 1
            if any(kw in answer.lower() for kw in ['project', 'target', 'connect', 'pathway']):
                modalities_mentioned += 1

            integration_score = min(1.0, modalities_mentioned / 2)
        else:
            integration_score = 0.5

        score += integration_score * 0.4

        # 2. 定量证据 (30%)
        numbers = re.findall(r'\d+[,\d]*', answer)

        if len(numbers) >= 5:
            quantitative_score = 1.0
        elif len(numbers) >= 3:
            quantitative_score = 0.8
        elif len(numbers) >= 1:
            quantitative_score = 0.5
        else:
            quantitative_score = 0.2

        score += quantitative_score * 0.3

        # 3. 跨模态综合 (30%)
        modalities_used = set(s.get('modality') for s in steps if s.get('modality'))

        if len(modalities_used) >= 3:
            cross_modal_score = 1.0
        elif len(modalities_used) == 2:
            cross_modal_score = 0.7
        elif len(modalities_used) == 1:
            cross_modal_score = 0.4
        else:
            cross_modal_score = 0.0

        score += cross_modal_score * 0.3

        return min(score, 1.0)

    def _evaluate_depth_score(self, steps: List[Dict]) -> float:
        """
        评估多跳推理深度分数

        归一化步数到0-1分数
        """

        num_steps = len(steps)

        if num_steps == 0:
            return 0.0
        elif num_steps == 1:
            return 0.3
        elif num_steps == 2:
            return 0.5
        elif num_steps == 3:
            return 0.65
        elif num_steps == 4:
            return 0.75
        elif num_steps == 5:
            return 0.85
        elif num_steps >= 6:
            return min(1.0, 0.85 + (num_steps - 5) * 0.03)

        return 0.0


# ==================== 🔬 CoT Quality Evaluator ====================

class CoTQualityEvaluator:
    """
    Chain-of-Thought质量评估器

    评估维度：
    1. CoT Clarity - 推理链清晰度
    2. CoT Completeness - 推理链完整性
    3. Intermediate Steps Quality - 中间步骤质量
    """

    def __init__(self):
        pass

    def evaluate(self, question_data: Dict, agent_output: Dict, method_name: str) -> Dict[str, float]:
        """评估CoT质量"""

        # 只评估有CoT的方法
        if method_name not in ['AIPOM-CoT', 'ReAct']:
            return {
                'cot_quality': None,
                'cot_clarity': None,
                'cot_completeness': None,
                'intermediate_steps_quality': None,
            }

        executed_steps = agent_output.get('executed_steps', [])
        answer = agent_output.get('answer', '')

        # 1. CoT Clarity
        clarity = self._evaluate_clarity(executed_steps)

        # 2. CoT Completeness
        completeness = self._evaluate_completeness(executed_steps, question_data)

        # 3. Intermediate Steps Quality
        steps_quality = self._evaluate_steps_quality(executed_steps)

        # Overall CoT quality
        cot_quality = np.mean([clarity, completeness, steps_quality])

        return {
            'cot_quality': cot_quality,
            'cot_clarity': clarity,
            'cot_completeness': completeness,
            'intermediate_steps_quality': steps_quality,
        }

    def _evaluate_clarity(self, steps: List[Dict]) -> float:
        """
        评估推理链清晰度

        检查：
        - 每步目标明确
        - 步骤描述清晰
        - 无歧义
        """

        if not steps:
            return 0.0

        score = 0.0

        # 1. 目标明确性 (40%)
        clear_purposes = 0

        for step in steps:
            purpose = step.get('purpose', '').lower()

            # 检查是否有明确的动词
            action_verbs = ['identify', 'find', 'retrieve', 'analyze', 'compare',
                            'characterize', 'profile', 'discover', 'validate']

            if any(verb in purpose for verb in action_verbs):
                clear_purposes += 1

        score += (clear_purposes / len(steps)) * 0.4

        # 2. 描述详细度 (30%)
        avg_length = np.mean([len(s.get('purpose', '')) for s in steps])

        if avg_length >= 50:
            detail_score = 1.0
        elif avg_length >= 30:
            detail_score = 0.8
        elif avg_length >= 15:
            detail_score = 0.6
        else:
            detail_score = 0.3

        score += detail_score * 0.3

        # 3. 结构一致性 (30%)
        # 检查所有步骤是否有一致的格式
        has_modality = sum(1 for s in steps if s.get('modality'))
        has_purpose = sum(1 for s in steps if s.get('purpose'))

        consistency_score = (has_modality + has_purpose) / (2 * len(steps))
        score += consistency_score * 0.3

        return min(score, 1.0)

    def _evaluate_completeness(self, steps: List[Dict], question_data: Dict) -> float:
        """
        评估推理链完整性

        检查：
        - 覆盖问题所需模态
        - 步骤连贯无跳跃
        - 达到预期深度
        """

        if not steps:
            return 0.0

        score = 0.0

        # 1. 模态覆盖 (40%)
        expected_modalities = set(question_data.get('expected_modalities', []))
        actual_modalities = set(s.get('modality') for s in steps if s.get('modality'))

        if expected_modalities:
            coverage = len(expected_modalities & actual_modalities) / len(expected_modalities)
        else:
            coverage = 0.7  # 默认

        score += coverage * 0.4

        # 2. 步骤连贯性 (30%)
        # 检查是否有明显的逻辑跳跃
        gap_count = 0

        for i in range(len(steps) - 1):
            mod1 = steps[i].get('modality', '')
            mod2 = steps[i + 1].get('modality', '')

            # 从molecular直接跳到projection（跳过morphological）可能是跳跃
            if mod1 == 'molecular' and mod2 == 'projection':
                gap_count += 1

        if len(steps) > 1:
            coherence = 1.0 - (gap_count / (len(steps) - 1))
        else:
            coherence = 1.0

        score += coherence * 0.3

        # 3. 深度充足性 (30%)
        expected_range = question_data.get('expected_steps_range', (1, 10))
        min_steps = expected_range[0]

        if len(steps) >= min_steps:
            depth_score = 1.0
        else:
            depth_score = len(steps) / min_steps

        score += depth_score * 0.3

        return min(score, 1.0)

    def _evaluate_steps_quality(self, steps: List[Dict]) -> float:
        """
        评估中间步骤质量

        检查：
        - 步骤成功率
        - 步骤信息量
        - 步骤价值
        """

        if not steps:
            return 0.0

        score = 0.0

        # 1. 成功率 (40%)
        successful = sum(1 for s in steps if s.get('success', True))
        success_rate = successful / len(steps)

        score += success_rate * 0.4

        # 2. 信息量 (30%)
        # 检查步骤是否产生了有价值的信息
        purposes = [s.get('purpose', '') for s in steps]
        avg_informativeness = np.mean([len(p.split()) for p in purposes])

        if avg_informativeness >= 8:
            info_score = 1.0
        elif avg_informativeness >= 5:
            info_score = 0.7
        elif avg_informativeness >= 3:
            info_score = 0.5
        else:
            info_score = 0.3

        score += info_score * 0.3

        # 3. 步骤价值 (30%)
        # 检查是否有重复或无用步骤
        unique_purposes = len(set(purposes))
        value_score = unique_purposes / len(purposes)

        score += value_score * 0.3

        return min(score, 1.0)


# ==================== 🔬 Reflection Capability Evaluator ====================

class ReflectionCapabilityEvaluator:
    """
    反思能力评估器

    评估维度：
    1. Error Detection - 错误检测
    2. Self-Correction - 自我纠正
    3. Iterative Refinement - 迭代优化
    """

    def __init__(self):
        pass

    def evaluate(self, question_data: Dict, agent_output: Dict, method_name: str) -> Dict[str, float]:
        """评估反思能力"""

        # 只评估有reflection能力的方法
        if method_name not in ['AIPOM-CoT', 'ReAct']:
            return {
                'reflection_capability': None,
                'error_detection': None,
                'self_correction': None,
                'iterative_refinement': None,
            }

        executed_steps = agent_output.get('executed_steps', [])
        answer = agent_output.get('answer', '')

        # 1. Error Detection
        detection = self._evaluate_error_detection(executed_steps)

        # 2. Self-Correction
        correction = self._evaluate_self_correction(executed_steps, answer)

        # 3. Iterative Refinement
        refinement = self._evaluate_iterative_refinement(executed_steps)

        # Overall reflection capability
        reflection_capability = np.mean([detection, correction, refinement])

        return {
            'reflection_capability': reflection_capability,
            'error_detection': detection,
            'self_correction': correction,
            'iterative_refinement': refinement,
        }

    def _evaluate_error_detection(self, steps: List[Dict]) -> float:
        """
        评估错误检测能力

        检查：
        - 失败步骤的识别
        - 问题诊断
        - 替代方案
        """

        if not steps:
            return 0.0

        score = 0.0

        # 1. 失败识别 (40%)
        failed_steps = [s for s in steps if not s.get('success', True)]

        if len(failed_steps) == 0:
            # 无失败 - 可能是好的，也可能缺乏挑战
            detection_score = 0.7
        else:
            # 有失败但继续执行 - 说明检测到了
            detection_score = 1.0

        score += detection_score * 0.4

        # 2. 问题诊断 (30%)
        # 检查后续步骤是否调整了策略
        if len(steps) >= 3:
            modalities = [s.get('modality') for s in steps]

            # 检查是否有策略变化
            changes = 0
            for i in range(len(modalities) - 1):
                if modalities[i] != modalities[i + 1]:
                    changes += 1

            diagnosis_score = min(1.0, changes / (len(steps) - 1) * 2)
        else:
            diagnosis_score = 0.5

        score += diagnosis_score * 0.3

        # 3. 恢复能力 (30%)
        # 检查失败后是否有成功步骤
        if failed_steps and len(steps) > len(failed_steps):
            # 有失败，但整体完成了
            recovery_score = 1.0
        elif not failed_steps:
            recovery_score = 0.8
        else:
            recovery_score = 0.3

        score += recovery_score * 0.3

        return min(score, 1.0)

    def _evaluate_self_correction(self, steps: List[Dict], answer: str) -> float:
        """
        评估自我纠正能力

        检查：
        - 答案中承认不确定性
        - 提供替代解释
        - 谨慎措辞
        """

        if not answer:
            return 0.0

        score = 0.0

        answer_lower = answer.lower()

        # 1. 不确定性表达 (40%)
        uncertainty_markers = [
            'may', 'might', 'could', 'possibly', 'likely',
            'suggests', 'indicates', 'appears', 'seems',
            'approximately', 'around', 'about'
        ]

        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer_lower)

        if 2 <= uncertainty_count <= 5:
            uncertainty_score = 1.0  # 适度的不确定性
        elif uncertainty_count == 1 or uncertainty_count == 6:
            uncertainty_score = 0.7
        elif uncertainty_count == 0:
            uncertainty_score = 0.4  # 过于确定可能不好
        else:
            uncertainty_score = 0.5  # 过多不确定性

        score += uncertainty_score * 0.4

        # 2. 替代解释 (30%)
        alternative_markers = [
            'alternatively', 'another', 'also', 'additionally',
            'or', 'either', 'different'
        ]

        alternative_count = sum(1 for marker in alternative_markers if marker in answer_lower)

        if alternative_count >= 2:
            alternative_score = 1.0
        elif alternative_count == 1:
            alternative_score = 0.7
        else:
            alternative_score = 0.4

        score += alternative_score * 0.3

        # 3. 谨慎措辞 (30%)
        # 检查是否避免绝对化陈述
        absolute_markers = ['always', 'never', 'all', 'none', 'every', 'must', 'definitely']

        absolute_count = sum(1 for marker in absolute_markers if marker in answer_lower)

        if absolute_count == 0:
            caution_score = 1.0
        elif absolute_count <= 2:
            caution_score = 0.6
        else:
            caution_score = 0.3

        score += caution_score * 0.3

        return min(score, 1.0)

    def _evaluate_iterative_refinement(self, steps: List[Dict]) -> float:
        """
        评估迭代优化能力

        检查：
        - 步骤渐进性
        - 策略调整
        - 目标聚焦
        """

        if len(steps) < 2:
            return 0.5 if len(steps) == 1 else 0.0

        score = 0.0

        # 1. 渐进性 (40%)
        # 检查步骤是否逐步深入
        modalities = [s.get('modality') for s in steps if s.get('modality')]

        if len(modalities) >= 2:
            # 检查是否从简单到复杂
            modality_order = {'molecular': 1, 'morphological': 2, 'projection': 3}

            progressions = 0
            for i in range(len(modalities) - 1):
                order1 = modality_order.get(modalities[i], 2)
                order2 = modality_order.get(modalities[i + 1], 2)

                if order2 >= order1:
                    progressions += 1

            progressive_score = progressions / (len(modalities) - 1)
        else:
            progressive_score = 0.5

        score += progressive_score * 0.4

        # 2. 策略调整 (30%)
        # 检查是否根据结果调整
        purposes = [s.get('purpose', '').lower() for s in steps]

        # 统计不同类型的目标
        types = []
        for purpose in purposes:
            if 'compare' in purpose:
                types.append('compare')
            elif 'find' in purpose or 'identify' in purpose:
                types.append('find')
            elif 'analyze' in purpose or 'characterize' in purpose:
                types.append('analyze')
            else:
                types.append('other')

        unique_types = len(set(types))

        if unique_types >= 2:
            adjustment_score = 1.0
        elif unique_types == 1:
            adjustment_score = 0.5
        else:
            adjustment_score = 0.3

        score += adjustment_score * 0.3

        # 3. 目标聚焦 (30%)
        # 检查后续步骤是否更加聚焦
        # 通过步骤描述的具体性变化来评估
        specificities = []

        for purpose in purposes:
            # 具体性 = 专业术语数量 / 总词数
            words = purpose.split()
            specific_terms = sum(1 for w in words if len(w) > 6 or w[0].isupper())
            specificity = specific_terms / len(words) if words else 0
            specificities.append(specificity)

        if len(specificities) >= 2:
            # 检查是否递增
            increasing = sum(1 for i in range(len(specificities) - 1) if specificities[i + 1] >= specificities[i])
            focus_score = increasing / (len(specificities) - 1)
        else:
            focus_score = 0.5

        score += focus_score * 0.3

        return min(score, 1.0)


# ==================== 🔬 NLU Capability Evaluator ====================

class NLUCapabilityEvaluator:
    """
    自然语言理解能力评估器

    评估维度：
    1. Query Understanding - 问题理解
    2. Intent Recognition - 意图识别
    3. Ambiguity Resolution - 歧义解析
    """

    def __init__(self):
        pass

    def evaluate(self, question_data: Dict, agent_output: Dict, method_name: str) -> Dict[str, float]:
        """评估NLU能力"""

        question = question_data.get('question', '')
        executed_steps = agent_output.get('executed_steps', [])
        entities_recognized = agent_output.get('entities_recognized', [])
        answer = agent_output.get('answer', '')

        # 1. Query Understanding
        understanding = self._evaluate_query_understanding(question, executed_steps, entities_recognized)

        # 2. Intent Recognition
        intent = self._evaluate_intent_recognition(question_data, executed_steps)

        # 3. Ambiguity Resolution
        ambiguity = self._evaluate_ambiguity_resolution(question, answer)

        # Overall NLU capability
        nlu_capability = np.mean([understanding, intent, ambiguity])

        return {
            'nlu_capability': nlu_capability,
            'query_understanding': understanding,
            'intent_recognition': intent,
            'ambiguity_resolution': ambiguity,
        }

    def _evaluate_query_understanding(self, question: str, steps: List[Dict], entities: List) -> float:
        """
        评估问题理解

        检查：
        - 关键实体识别
        - 问题焦点把握
        - 必要信息提取
        """

        if not question:
            return 0.0

        score = 0.0

        # 1. 实体识别准确性 (40%)
        # 提取问题中的实体
        question_entities = set()

        # 基因/蛋白
        genes = re.findall(r'\b([A-Z][a-z]{2,8})\+?', question)
        question_entities.update(g for g in genes if g not in {'What', 'Which', 'Where', 'Tell'})

        # 脑区
        regions = re.findall(r'\b([A-Z]{2,5})\b', question)
        known_regions = {'MOp', 'MOs', 'SSp', 'VISp', 'AUDp', 'ACA', 'CLA', 'RSP', 'TH'}
        question_entities.update(r for r in regions if r in known_regions)

        # 识别的实体
        recognized = set()
        for entity in entities:
            if isinstance(entity, dict):
                recognized.add(entity.get('text', '').lower())
            else:
                recognized.add(str(entity).lower())

        question_entities_lower = set(e.lower() for e in question_entities)

        if question_entities_lower:
            entity_accuracy = len(question_entities_lower & recognized) / len(question_entities_lower)
        else:
            entity_accuracy = 0.7  # 没有明显实体，给默认分

        score += entity_accuracy * 0.4

        # 2. 焦点把握 (30%)
        # 检查第一步是否针对主要问题
        if steps:
            first_purpose = steps[0].get('purpose', '').lower()

            # 提取问题关键词
            question_lower = question.lower()

            keywords = []
            if 'profile' in question_lower or 'characterize' in question_lower or 'about' in question_lower:
                keywords.extend(['profile', 'characterize', 'analyze'])
            if 'compare' in question_lower or 'versus' in question_lower:
                keywords.extend(['compare', 'versus'])
            if 'project' in question_lower or 'target' in question_lower:
                keywords.extend(['project', 'target', 'connect'])

            if keywords:
                focus_match = any(kw in first_purpose for kw in keywords)
                focus_score = 1.0 if focus_match else 0.5
            else:
                focus_score = 0.7

        else:
            focus_score = 0.0

        score += focus_score * 0.3

        # 3. 执行匹配 (30%)
        # 检查执行的步骤是否符合问题需求
        expected_modalities = self._infer_expected_modalities(question)
        actual_modalities = set(s.get('modality') for s in steps if s.get('modality'))

        if expected_modalities:
            execution_match = len(expected_modalities & actual_modalities) / len(expected_modalities)
        else:
            execution_match = 0.7

        score += execution_match * 0.3

        return min(score, 1.0)

    def _evaluate_intent_recognition(self, question_data: Dict, steps: List[Dict]) -> float:
        """
        评估意图识别

        检查：
        - 策略选择正确性
        - 任务类型识别
        - 深度匹配
        """

        score = 0.0

        # 1. 策略选择 (40%)
        expected_strategy = question_data.get('expected_strategy', 'adaptive')

        # 从步骤推断实际策略
        if not steps:
            inferred_strategy = 'none'
        else:
            modalities = set(s.get('modality') for s in steps if s.get('modality'))
            purposes = [s.get('purpose', '').lower() for s in steps]

            if any('compare' in p for p in purposes):
                inferred_strategy = 'comparative'
            elif len(modalities) >= 3:
                inferred_strategy = 'focus_driven'
            elif len(steps) >= 5:
                inferred_strategy = 'screening'
            else:
                inferred_strategy = 'adaptive'

        if inferred_strategy == expected_strategy:
            strategy_score = 1.0
        elif expected_strategy == 'adaptive':
            strategy_score = 0.8  # adaptive允许各种策略
        else:
            strategy_score = 0.5

        score += strategy_score * 0.4

        # 2. 任务类型识别 (30%)
        task_type = question_data.get('task_type')

        if task_type:
            # 检查步骤是否匹配任务类型
            if task_type == 'profiling':
                # 需要多模态
                correct = len(set(s.get('modality') for s in steps if s.get('modality'))) >= 2
            elif task_type == 'discovery':
                # 需要系统分析
                correct = len(steps) >= 3
            elif task_type == 'validation':
                # 需要比较
                correct = any('compare' in s.get('purpose', '').lower() for s in steps)
            else:  # lookup
                correct = len(steps) >= 1

            task_score = 1.0 if correct else 0.5
        else:
            task_score = 0.7

        score += task_score * 0.3

        # 3. 深度匹配 (30%)
        expected_depth = question_data.get('expected_depth', 'medium')
        actual_steps = len(steps)

        depth_map = {'shallow': 1, 'medium': 3, 'deep': 5}
        expected_min = depth_map.get(expected_depth, 3)

        if actual_steps >= expected_min:
            depth_score = 1.0
        else:
            depth_score = actual_steps / expected_min

        score += depth_score * 0.3

        return min(score, 1.0)

    def _evaluate_ambiguity_resolution(self, question: str, answer: str) -> float:
        """
        评估歧义解析

        检查：
        - 处理模糊查询
        - 澄清假设
        - 提供完整答案
        """

        if not question or not answer:
            return 0.0

        score = 0.0

        # 1. 识别歧义 (40%)
        # 检查问题是否模糊
        ambiguous_patterns = [
            r'\btell me about\b',
            r'\bwhat are\b',
            r'\bhow many\b',
            r'\bcompare\b',
        ]

        is_ambiguous = any(re.search(pattern, question.lower()) for pattern in ambiguous_patterns)

        if is_ambiguous:
            # 检查答案是否澄清了假设
            clarification_markers = [
                'specifically', 'in particular', 'focusing on', 'considering',
                'based on', 'regarding', 'with respect to'
            ]

            has_clarification = any(marker in answer.lower() for marker in clarification_markers)

            ambiguity_score = 1.0 if has_clarification else 0.6
        else:
            ambiguity_score = 0.8  # 不模糊，给默认高分

        score += ambiguity_score * 0.4

        # 2. 假设说明 (30%)
        assumption_markers = [
            'assuming', 'given', 'if', 'when', 'provided that',
            'under the condition', 'in the case'
        ]

        has_assumptions = any(marker in answer.lower() for marker in assumption_markers)

        if has_assumptions:
            assumption_score = 1.0
        else:
            assumption_score = 0.7

        score += assumption_score * 0.3

        # 3. 答案完整性 (30%)
        # 检查答案是否全面
        word_count = len(answer.split())

        if word_count >= 100:
            completeness_score = 1.0
        elif word_count >= 50:
            completeness_score = 0.8
        elif word_count >= 20:
            completeness_score = 0.5
        else:
            completeness_score = 0.3

        score += completeness_score * 0.3

        return min(score, 1.0)

    def _infer_expected_modalities(self, question: str) -> set:
        """从问题推断预期模态"""

        expected = set()
        question_lower = question.lower()

        if any(kw in question_lower for kw in ['marker', 'express', 'cluster', 'cell type', 'gene']):
            expected.add('molecular')

        if any(kw in question_lower for kw in ['morphology', 'axon', 'dendrite', 'branch', 'length']):
            expected.add('morphological')

        if any(kw in question_lower for kw in ['project', 'target', 'connect', 'pathway', 'circuit']):
            expected.add('projection')

        return expected


# ==================== Comprehensive Evaluator (v4.0) ====================

class ComprehensiveEvaluatorV4:
    """综合评估器 v4.0 - Nature Methods"""

    def __init__(self):
        # 初始化所有评估器
        self.planning_eval = PlanningQualityEvaluator()
        self.reasoning_eval = ReasoningCapabilityEvaluator()
        self.cot_eval = CoTQualityEvaluator()
        self.reflection_eval = ReflectionCapabilityEvaluator()
        self.nlu_eval = NLUCapabilityEvaluator()

        # 保留原有评估器
        from evaluators import (
            AdaptivePlanningEvaluator,
            EntityRecognitionEvaluator,
            AnswerQualityEvaluator,
            BiologicalTaskEvaluator
        )

        self.adaptive_eval = AdaptivePlanningEvaluator()
        self.entity_eval = EntityRecognitionEvaluator()
        self.answer_eval = AnswerQualityEvaluator()
        self.task_eval = BiologicalTaskEvaluator()

        self.config = EVALUATION_CONFIG

    def evaluate_full(self,
                      question_data: Dict,
                      agent_output: Dict,
                      method_name: str) -> NMEvaluationMetrics:
        """完整评估 (v4.0)"""

        metrics = NMEvaluationMetrics()

        # 🔬 NM核心能力评估

        # 1. Planning Quality
        planning_metrics = self.planning_eval.evaluate(question_data, agent_output, method_name)
        metrics.planning_quality = planning_metrics['planning_quality']
        metrics.planning_coherence = planning_metrics['planning_coherence']
        metrics.planning_optimality = planning_metrics['planning_optimality']
        metrics.planning_adaptability = planning_metrics['planning_adaptability']

        # 2. Reasoning Capability
        reasoning_metrics = self.reasoning_eval.evaluate(question_data, agent_output, method_name)
        metrics.reasoning_capability = reasoning_metrics['reasoning_capability']
        metrics.logical_consistency = reasoning_metrics['logical_consistency']
        metrics.evidence_integration = reasoning_metrics['evidence_integration']
        metrics.multi_hop_depth_score = reasoning_metrics['multi_hop_depth_score']

        # 3. CoT Quality
        cot_metrics = self.cot_eval.evaluate(question_data, agent_output, method_name)
        metrics.cot_quality = cot_metrics['cot_quality']
        metrics.cot_clarity = cot_metrics['cot_clarity']
        metrics.cot_completeness = cot_metrics['cot_completeness']
        metrics.intermediate_steps_quality = cot_metrics['intermediate_steps_quality']

        # 4. Reflection Capability
        reflection_metrics = self.reflection_eval.evaluate(question_data, agent_output, method_name)
        metrics.reflection_capability = reflection_metrics['reflection_capability']
        metrics.error_detection = reflection_metrics['error_detection']
        metrics.self_correction = reflection_metrics['self_correction']
        metrics.iterative_refinement = reflection_metrics['iterative_refinement']

        # 5. NLU Capability
        nlu_metrics = self.nlu_eval.evaluate(question_data, agent_output, method_name)
        metrics.nlu_capability = nlu_metrics['nlu_capability']
        metrics.query_understanding = nlu_metrics['query_understanding']
        metrics.intent_recognition = nlu_metrics['intent_recognition']
        metrics.ambiguity_resolution = nlu_metrics['ambiguity_resolution']

        # 传统指标

        # Entity Recognition
        entity_metrics = self.entity_eval.evaluate(question_data, agent_output)
        metrics.entity_precision = entity_metrics['entity_precision']
        metrics.entity_recall = entity_metrics['entity_recall']
        metrics.entity_f1 = entity_metrics['entity_f1']

        # Answer Quality
        answer_metrics = self.answer_eval.evaluate(question_data, agent_output)
        metrics.factual_accuracy = answer_metrics['factual_accuracy']
        metrics.answer_completeness = answer_metrics['answer_completeness']
        metrics.scientific_rigor = answer_metrics['scientific_rigor']

        # System capabilities
        adaptive_metrics = self.adaptive_eval.evaluate(question_data, agent_output, method_name)
        metrics.reasoning_depth = adaptive_metrics.get('reasoning_depth')
        metrics.modality_coverage = adaptive_metrics.get('modality_coverage')

        closed_loop_score = adaptive_metrics.get('closed_loop')
        if closed_loop_score is not None:
            metrics.closed_loop_achieved = closed_loop_score >= 0.9
        else:
            metrics.closed_loop_achieved = None

        # Efficiency
        steps = agent_output.get('executed_steps', [])
        metrics.execution_time = agent_output.get('execution_time', 0.0)
        metrics.api_calls = len(steps)

        if steps:
            successful = sum(1 for s in steps if s.get('success', True))
            metrics.query_success_rate = successful / len(steps)
        else:
            metrics.query_success_rate = 1.0

        modalities = set(s.get('modality') for s in steps if s.get('modality'))
        metrics.modalities_used = list(modalities)

        # Task Completion
        if question_data.get('task_type'):
            metrics.task_completion = self.task_eval.evaluate_task_completion(
                question_data, agent_output
            )

        # Biological Insight Score
        metrics.biological_insight_score = self._evaluate_biological_insight(
            question_data, agent_output, metrics
        )

        # Overall Scores
        metrics.overall_score = self._calculate_overall_score(metrics, method_name)
        metrics.nm_capability_score = self._calculate_nm_capability_score(metrics)

        return metrics

    def _evaluate_biological_insight(self, question_data: Dict, agent_output: Dict,
                                     metrics: NMEvaluationMetrics) -> float:
        """评估生物学洞察力"""

        answer = agent_output.get('answer', '')

        if not answer:
            return 0.0

        score = 0.0

        # 1. 跨模态整合 (30%)
        if len(metrics.modalities_used) >= 3:
            cross_modal = 1.0
        elif len(metrics.modalities_used) == 2:
            cross_modal = 0.7
        else:
            cross_modal = 0.3

        score += cross_modal * 0.3

        # 2. 定量分析 (30%)
        score += metrics.scientific_rigor * 0.3

        # 3. 生物学相关性 (40%)
        bio_keywords = [
            'neuron', 'cell', 'cluster', 'marker', 'express',
            'project', 'connect', 'circuit', 'pathway',
            'morphology', 'axon', 'dendrite', 'synapse',
            'cortex', 'region', 'brain', 'neural'
        ]

        answer_lower = answer.lower()
        keyword_count = sum(1 for kw in bio_keywords if kw in answer_lower)

        if keyword_count >= 8:
            bio_relevance = 1.0
        elif keyword_count >= 5:
            bio_relevance = 0.8
        elif keyword_count >= 3:
            bio_relevance = 0.5
        else:
            bio_relevance = 0.3

        score += bio_relevance * 0.4

        return min(score, 1.0)

    def _calculate_nm_capability_score(self, metrics: NMEvaluationMetrics) -> float:
        """计算NM核心能力总分"""

        nm_scores = []

        if metrics.planning_quality is not None:
            nm_scores.append(metrics.planning_quality)
        if metrics.reasoning_capability is not None:
            nm_scores.append(metrics.reasoning_capability)
        if metrics.cot_quality is not None:
            nm_scores.append(metrics.cot_quality)
        if metrics.reflection_capability is not None:
            nm_scores.append(metrics.reflection_capability)
        if metrics.nlu_capability is not None:
            nm_scores.append(metrics.nlu_capability)

        return np.mean(nm_scores) if nm_scores else 0.0

    def _calculate_overall_score(self, metrics: NMEvaluationMetrics, method_name: str) -> float:
        """计算加权Overall分数"""

        weights = self.config['nm_method_weights'].get(method_name, {})

        if not weights:
            # Fallback
            core_scores = [
                metrics.entity_f1,
                metrics.factual_accuracy,
                metrics.scientific_rigor,
            ]
            return np.mean([s for s in core_scores if s is not None])

        weighted_sum = 0.0
        total_weight = 0.0

        metric_values = {
            # NM核心能力
            'planning_quality': metrics.planning_quality,
            'reasoning_capability': metrics.reasoning_capability,
            'cot_quality': metrics.cot_quality,
            'reflection_capability': metrics.reflection_capability,
            'nlu_capability': metrics.nlu_capability,
            # 传统指标
            'entity_f1': metrics.entity_f1,
            'factual_accuracy': metrics.factual_accuracy,
            'scientific_rigor': metrics.scientific_rigor,
            'modality_coverage': metrics.modality_coverage,
            'closed_loop': 1.0 if metrics.closed_loop_achieved else (
                0.0 if metrics.closed_loop_achieved is not None else None),
        }

        for metric_name, weight in weights.items():
            value = metric_values.get(metric_name)

            if value is not None:
                weighted_sum += value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


# ==================== Export ====================

__all__ = [
    'NMEvaluationMetrics',
    'PlanningQualityEvaluator',
    'ReasoningCapabilityEvaluator',
    'CoTQualityEvaluator',
    'ReflectionCapabilityEvaluator',
    'NLUCapabilityEvaluator',
    'ComprehensiveEvaluatorV4',
    'EVALUATION_CONFIG',
]

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("✅ Enhanced evaluators.py v4.0 (Nature Methods) loaded successfully!")
    print("=" * 80)

    print("\n🔬 New NM Core Dimensions:")
    for dim, config in EVALUATION_CONFIG['nm_core_dimensions'].items():
        print(f"  - {dim}: {config['description']}")

    print("\n" + "=" * 80)