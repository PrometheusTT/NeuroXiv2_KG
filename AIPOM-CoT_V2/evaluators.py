"""
Evaluation System for AIPOM-CoT Benchmark (v3.0 - Fair Evaluation)
===================================================================
包含完整的评估体系：
- 分层评估（核心指标 vs 系统指标）
- 方法特定权重
- 生物学任务评估
- 公平的Overall分数计算

Changes in v3.0 (公平性修复):
- ✅ 分层评估：区分核心能力和系统能力
- ✅ 方法特定权重：LLM和Agent使用不同评估标准
- ✅ None-able指标：不强制所有方法在所有指标上评分
- ✅ 生物学任务评估：明确的成功标准

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 3.0
"""

import numpy as np
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== 🔧 新增：评估配置 ====================

EVALUATION_CONFIG = {
    # 核心指标：所有方法都必须评估
    'core_metrics': {
        'entity_f1': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Accuracy of entity recognition'
        },
        'factual_accuracy': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Factual correctness of answer'
        },
        'answer_completeness': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Completeness of answer'
        },
        'scientific_rigor': {
            'weight': 1.0,
            'methods': 'all',
            'description': 'Scientific rigor and quantitative support'
        },
    },

    # 系统能力指标：只评估有该能力的方法
    'system_metrics': {
        'depth_matching': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'ReAct', 'Template-KG'],  # 有planning/步骤的方法
            'description': 'Adaptive depth matching'
        },
        'plan_coherence': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'ReAct'],  # 只有动态planning的
            'description': 'Coherence of execution plan'
        },
        'closed_loop': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT'],  # 只有AIPOM设计了闭环
            'description': 'Closed-loop circuit analysis'
        },
        'modality_coverage': {
            'weight': 1.0,
            'methods': ['AIPOM-CoT', 'Template-KG', 'RAG', 'ReAct'],  # 有KG访问的
            'description': 'Multi-modal data coverage'
        },
    },

    # 🔧 方法特定权重（用于计算Overall分数）
    'method_weights': {
        'AIPOM-CoT': {
            # 全面评估
            'entity_f1': 0.15,
            'factual_accuracy': 0.15,
            'answer_completeness': 0.12,
            'scientific_rigor': 0.13,
            'depth_matching': 0.15,
            'plan_coherence': 0.10,
            'closed_loop': 0.10,
            'modality_coverage': 0.10,
        },
        'Direct GPT-4o': {
            # 重点评估答案质量（无planning指标）
            'entity_f1': 0.20,
            'factual_accuracy': 0.30,
            'answer_completeness': 0.25,
            'scientific_rigor': 0.25,
        },
        'Template-KG': {
            # 有KG访问和固定步骤
            'entity_f1': 0.20,
            'factual_accuracy': 0.20,
            'answer_completeness': 0.15,
            'scientific_rigor': 0.15,
            'depth_matching': 0.15,  # 评估步骤匹配
            'modality_coverage': 0.15,
        },
        'RAG': {
            # 重点评估检索和答案质量
            'entity_f1': 0.20,
            'factual_accuracy': 0.25,
            'answer_completeness': 0.20,
            'scientific_rigor': 0.20,
            'modality_coverage': 0.15,
        },
        'ReAct': {
            # 评估推理和planning
            'entity_f1': 0.15,
            'factual_accuracy': 0.20,
            'answer_completeness': 0.15,
            'scientific_rigor': 0.15,
            'depth_matching': 0.15,
            'plan_coherence': 0.10,
            'modality_coverage': 0.10,
        },
    },
}


# ==================== Data Structures ====================

@dataclass
class EvaluationMetrics:
    """
    评估指标（更新版 - 支持None值）

    None值表示该指标不适用于当前方法
    """

    # D1: Adaptive Planning (系统能力)
    depth_matching_accuracy: Optional[float] = None
    plan_coherence: Optional[float] = None
    strategy_selection_accuracy: Optional[float] = None

    # D2: Entity Recognition (核心能力)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0

    # D3: Multi-hop Reasoning (核心能力)
    multi_hop_depth: int = 0
    query_success_rate: float = 0.0

    # D4: Multi-Modal Integration (系统能力)
    modality_coverage: Optional[float] = None
    modalities_used: List[str] = field(default_factory=list)
    closed_loop_achieved: Optional[bool] = None

    # D5: Answer Quality (核心能力)
    factual_accuracy: float = 0.0
    answer_completeness: float = 0.0
    scientific_rigor: float = 0.0

    # D6: Efficiency (所有方法)
    execution_time: float = 0.0
    api_calls: int = 0

    # 🔧 新增：方法特定Overall分数
    overall_score: Optional[float] = None

    # 🔧 新增：任务完成度
    task_completion: Optional[str] = None  # 'completed', 'partial', 'failed', None


# ==================== D1: Adaptive Planning Evaluator ====================

class AdaptivePlanningEvaluator:
    """
    自适应规划评估器

    🔧 修复：只对有planning能力的方法评估
    """

    def __init__(self):
        self.depth_map = {
            'shallow': 2,
            'medium': 4,
            'deep': 6,
        }

    def evaluate(self,
                question_data: Dict,
                agent_output: Dict,
                method_name: str) -> Dict[str, float]:
        """
        评估adaptive planning

        🔧 修复：对于无planning的方法，返回None而非0
        """

        metrics = {}

        # 检查是否应该评估planning
        should_evaluate_planning = method_name in EVALUATION_CONFIG['system_metrics']['plan_coherence']['methods']

        # D1.1: Depth Matching
        if method_name in EVALUATION_CONFIG['system_metrics']['depth_matching']['methods']:
            metrics['depth_matching'] = self._evaluate_depth_matching(
                question_data, agent_output
            )
        else:
            metrics['depth_matching'] = None

        # D1.2: Plan Coherence
        if should_evaluate_planning:
            metrics['plan_coherence'] = self._evaluate_plan_coherence(
                agent_output
            )
        else:
            metrics['plan_coherence'] = None

        # D1.3: Strategy Selection
        if should_evaluate_planning:
            metrics['strategy_selection'] = self._evaluate_strategy_selection(
                question_data, agent_output
            )
        else:
            metrics['strategy_selection'] = None

        # D1.4: Modality Coverage (有KG访问的方法都评估)
        if method_name in EVALUATION_CONFIG['system_metrics']['modality_coverage']['methods']:
            metrics['modality_coverage'] = self._evaluate_modality_coverage(
                question_data, agent_output
            )
        else:
            metrics['modality_coverage'] = None

        # D1.5: Closed-Loop (只有AIPOM评估)
        if method_name in EVALUATION_CONFIG['system_metrics']['closed_loop']['methods']:
            metrics['closed_loop'] = self._evaluate_closed_loop(
                question_data, agent_output
            )
        else:
            metrics['closed_loop'] = None

        return metrics

    def _evaluate_depth_matching(self, question_data: Dict, agent_output: Dict) -> float:
        """评估深度匹配"""

        expected_depth = question_data.get('expected_depth', 'medium')
        expected_steps = self.depth_map.get(expected_depth, 4)

        executed_steps = agent_output.get('executed_steps', [])
        actual_steps = len(executed_steps)

        if actual_steps == 0:
            return 0.0

        # 计算匹配度（允许±2步的误差）
        diff = abs(actual_steps - expected_steps)

        if diff == 0:
            score = 1.0
        elif diff == 1:
            score = 0.9
        elif diff == 2:
            score = 0.75
        elif diff == 3:
            score = 0.5
        else:
            score = max(0.0, 1.0 - (diff - 3) * 0.15)

        return score

    def _evaluate_plan_coherence(self, agent_output: Dict) -> float:
        """评估计划连贯性"""

        executed_steps = agent_output.get('executed_steps', [])

        if len(executed_steps) < 2:
            return 1.0 if len(executed_steps) == 1 else 0.0

        coherence_score = 0.0

        # 1. 步骤间逻辑连贯性 (40%)
        logical_coherence = self._check_logical_flow(executed_steps)
        coherence_score += logical_coherence * 0.4

        # 2. 模态多样性 (30%)
        modality_diversity = self._check_modality_diversity(executed_steps)
        coherence_score += modality_diversity * 0.3

        # 3. 无重复查询 (30%)
        no_duplication = self._check_no_duplication(executed_steps)
        coherence_score += no_duplication * 0.3

        return coherence_score

    def _check_logical_flow(self, steps: List[Dict]) -> float:
        """检查逻辑流"""

        if len(steps) < 2:
            return 1.0

        # 检查是否有合理的progression
        # molecular → morphological → projection是好的流程
        modality_order = {
            'molecular': 1,
            'morphological': 2,
            'projection': 3,
            'statistical': 4,
        }

        scores = []
        for i in range(len(steps) - 1):
            mod1 = steps[i].get('modality')
            mod2 = steps[i+1].get('modality')

            if mod1 and mod2:
                order1 = modality_order.get(mod1, 0)
                order2 = modality_order.get(mod2, 0)

                # 允许平级或递进
                if order2 >= order1:
                    scores.append(1.0)
                elif order2 == order1 - 1:  # 允许小幅回退
                    scores.append(0.8)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.7)  # 未知模态

        return np.mean(scores) if scores else 0.5

    def _check_modality_diversity(self, steps: List[Dict]) -> float:
        """检查模态多样性"""

        modalities = set(s.get('modality') for s in steps if s.get('modality'))

        num_modalities = len(modalities)

        if num_modalities >= 3:
            return 1.0
        elif num_modalities == 2:
            return 0.7
        elif num_modalities == 1:
            return 0.4
        else:
            return 0.0

    def _check_no_duplication(self, steps: List[Dict]) -> float:
        """检查是否有重复查询"""

        purposes = [s.get('purpose', '') for s in steps]

        if len(purposes) == 0:
            return 1.0

        unique_purposes = len(set(purposes))
        total_purposes = len(purposes)

        return unique_purposes / total_purposes

    def _evaluate_strategy_selection(self, question_data: Dict, agent_output: Dict) -> float:
        """评估策略选择"""

        expected_strategy = question_data.get('expected_strategy', 'adaptive')

        # 从agent_output推断实际策略
        steps = agent_output.get('executed_steps', [])

        if not steps:
            return 0.0

        # 启发式判断策略
        modalities = set(s.get('modality') for s in steps if s.get('modality'))

        if len(modalities) >= 3:
            inferred_strategy = 'focus_driven'
        elif len(steps) > 5:
            inferred_strategy = 'comparative'
        else:
            inferred_strategy = 'adaptive'

        # 匹配度
        if inferred_strategy == expected_strategy:
            return 1.0
        else:
            return 0.6  # 部分匹配

    def _evaluate_modality_coverage(self, question_data: Dict, agent_output: Dict) -> float:
        """评估模态覆盖"""

        expected_modalities = set(question_data.get('expected_modalities', []))

        if not expected_modalities:
            return 1.0

        executed_steps = agent_output.get('executed_steps', [])
        covered_modalities = set(s.get('modality') for s in executed_steps if s.get('modality'))

        if not covered_modalities:
            return 0.0

        intersection = expected_modalities & covered_modalities

        recall = len(intersection) / len(expected_modalities)
        precision = len(intersection) / len(covered_modalities) if covered_modalities else 0

        # F1 score
        if recall + precision == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def _evaluate_closed_loop(self, question_data: Dict, agent_output: Dict) -> float:
        """评估闭环完成度"""

        expected_closed_loop = question_data.get('expected_closed_loop', False)

        if not expected_closed_loop:
            return 1.0  # 不需要闭环

        # 检查是否完成闭环
        executed_steps = agent_output.get('executed_steps', [])

        # 闭环需要：projection步骤 + target composition步骤
        has_projection = False
        has_target_composition = False

        for step in executed_steps:
            purpose = step.get('purpose', '').lower()
            modality = step.get('modality', '')

            if 'projection' in purpose or modality == 'projection':
                has_projection = True

            if ('target' in purpose and 'composition' in purpose) or \
               ('target' in purpose and modality == 'molecular'):
                has_target_composition = True

        if has_projection and has_target_composition:
            return 1.0
        elif has_projection:
            return 0.5  # 部分完成
        else:
            return 0.0


# ==================== D2: Entity Recognition Evaluator ====================

class EntityRecognitionEvaluator:
    """实体识别评估器（所有方法都评估）"""

    def evaluate(self, question_data: Dict, agent_output: Dict) -> Dict[str, float]:
        """评估实体识别"""

        expected_entities = set(question_data.get('expected_entities', []))
        recognized_entities = agent_output.get('entities_recognized', [])

        if not expected_entities:
            # 没有预期实体，检查是否识别了任何实体
            if recognized_entities:
                return {
                    'entity_precision': 0.5,
                    'entity_recall': 0.5,
                    'entity_f1': 0.5,
                }
            else:
                return {
                    'entity_precision': 1.0,
                    'entity_recall': 1.0,
                    'entity_f1': 1.0,
                }

        # 提取识别的实体文本
        recognized_texts = set()
        for entity in recognized_entities:
            if isinstance(entity, dict):
                text = entity.get('text', '')
            else:
                text = str(entity)

            recognized_texts.add(text.lower().strip())

        # 标准化预期实体
        expected_texts = set(e.lower().strip() for e in expected_entities)

        # 计算precision, recall, F1
        if not recognized_texts:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            true_positives = len(expected_texts & recognized_texts)

            precision = true_positives / len(recognized_texts) if recognized_texts else 0.0
            recall = true_positives / len(expected_texts) if expected_texts else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
        }


# ==================== D5: Answer Quality Evaluator ====================

class AnswerQualityEvaluator:
    """答案质量评估器（所有方法都评估）"""

    def evaluate(self, question_data: Dict, agent_output: Dict) -> Dict[str, float]:
        """评估答案质量"""

        answer = agent_output.get('answer', '')

        if not answer or len(answer) < 20:
            return {
                'factual_accuracy': 0.0,
                'answer_completeness': 0.0,
                'scientific_rigor': 0.0,
            }

        metrics = {}

        # D5.1: Factual Accuracy (基于关键实体和数据的存在)
        metrics['factual_accuracy'] = self._evaluate_factual_accuracy(
            question_data, answer, agent_output
        )

        # D5.2: Answer Completeness
        metrics['answer_completeness'] = self._evaluate_completeness(
            question_data, answer
        )

        # D5.3: Scientific Rigor
        metrics['scientific_rigor'] = self._evaluate_scientific_rigor(
            answer
        )

        return metrics

    def _evaluate_factual_accuracy(self, question_data: Dict, answer: str, agent_output: Dict) -> float:
        """评估事实准确性"""

        score = 0.0

        # 1. 检查预期实体是否在答案中 (40%)
        expected_entities = question_data.get('expected_entities', [])
        if expected_entities:
            mentioned = sum(1 for entity in expected_entities if entity.lower() in answer.lower())
            entity_score = mentioned / len(expected_entities)
            score += entity_score * 0.4
        else:
            score += 0.4

        # 2. 检查是否有定量数据 (30%)
        has_numbers = bool(re.search(r'\d+', answer))
        has_specific_data = bool(re.search(r'\d+[,\d]*\s*(neurons?|cells?|clusters?|%)', answer, re.IGNORECASE))

        if has_specific_data:
            score += 0.3
        elif has_numbers:
            score += 0.15

        # 3. 检查是否成功执行 (30%)
        if agent_output.get('success', False):
            steps = agent_output.get('executed_steps', [])
            if steps:
                successful_steps = sum(1 for s in steps if s.get('success', True))
                success_rate = successful_steps / len(steps)
                score += success_rate * 0.3
            else:
                score += 0.3

        return min(score, 1.0)

    def _evaluate_completeness(self, question_data: Dict, answer: str) -> float:
        """评估答案完整性"""

        score = 0.0

        # 1. 答案长度 (20%)
        word_count = len(answer.split())
        if word_count >= 100:
            score += 0.2
        elif word_count >= 50:
            score += 0.15
        elif word_count >= 20:
            score += 0.1

        # 2. 覆盖预期模态 (40%)
        expected_modalities = set(question_data.get('expected_modalities', []))
        if expected_modalities:
            answer_lower = answer.lower()

            modality_keywords = {
                'molecular': ['marker', 'gene', 'express', 'cluster', 'cell type'],
                'morphological': ['morphology', 'axon', 'dendrite', 'branch', 'length'],
                'projection': ['project', 'target', 'connect', 'pathway'],
            }

            covered = 0
            for modality in expected_modalities:
                keywords = modality_keywords.get(modality, [])
                if any(kw in answer_lower for kw in keywords):
                    covered += 1

            modality_score = covered / len(expected_modalities)
            score += modality_score * 0.4
        else:
            score += 0.4

        # 3. 结构化程度 (20%)
        has_list = bool(re.search(r'(\d+\.|•|-)(\s+\w+)', answer))
        has_sections = answer.count('\n') >= 2

        if has_list and has_sections:
            score += 0.2
        elif has_list or has_sections:
            score += 0.1

        # 4. 无明显错误标记 (20%)
        error_markers = ['error', 'failed', 'unable', 'cannot', 'no data', 'not found']
        has_errors = any(marker in answer.lower() for marker in error_markers)

        if not has_errors:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_scientific_rigor(self, answer: str) -> float:
        """评估科学严谨性"""

        score = 0.0

        # 1. 定量数据 (40%)
        numbers = re.findall(r'\d+[,\d]*', answer)
        num_count = len(numbers)

        if num_count >= 5:
            score += 0.4
        elif num_count >= 3:
            score += 0.3
        elif num_count >= 1:
            score += 0.2

        # 2. 科学术语 (30%)
        scientific_terms = [
            'neuron', 'cluster', 'marker', 'express', 'project',
            'morphology', 'axon', 'dendrite', 'synapse', 'circuit',
            'region', 'cortex', 'connectivity', 'distribution'
        ]

        answer_lower = answer.lower()
        term_count = sum(1 for term in scientific_terms if term in answer_lower)

        if term_count >= 8:
            score += 0.3
        elif term_count >= 5:
            score += 0.2
        elif term_count >= 3:
            score += 0.1

        # 3. 引用KG数据 (30%)
        kg_citations = [
            'according to', 'based on', 'data shows', 'found',
            'identified', 'observed', 'recorded', 'measured'
        ]

        has_citation = any(cite in answer_lower for cite in kg_citations)

        if has_citation:
            score += 0.3

        return min(score, 1.0)


# ==================== 🔧 新增：Biological Task Evaluator ====================

class BiologicalTaskEvaluator:
    """
    生物学任务评估器

    评估任务完成度：'completed', 'partial', 'failed'
    """

    def __init__(self):
        pass

    def evaluate_task_completion(self,
                                 question_data: Dict,
                                 agent_output: Dict) -> str:
        """
        评估任务完成度

        Returns:
            'completed' | 'partial' | 'failed'
        """

        # 获取success和partial criteria
        success_criteria = question_data.get('success_criteria', {})
        partial_criteria = question_data.get('partial_criteria', {})

        if not success_criteria:
            # 如果没有定义criteria，使用默认评估
            return self._default_evaluation(question_data, agent_output)

        # 检查success criteria
        success_checks = self._check_criteria(
            success_criteria,
            agent_output,
            question_data
        )

        if all(success_checks.values()):
            return 'completed'

        # 检查partial criteria
        if partial_criteria:
            partial_checks = self._check_criteria(
                partial_criteria,
                agent_output,
                question_data
            )

            if all(partial_checks.values()):
                return 'partial'

        return 'failed'

    def _check_criteria(self,
                       criteria: Dict,
                       agent_output: Dict,
                       question_data: Dict) -> Dict[str, bool]:
        """检查标准是否满足"""

        checks = {}

        answer = agent_output.get('answer', '')
        steps = agent_output.get('executed_steps', [])

        for criterion, requirement in criteria.items():

            if criterion == 'modalities_covered':
                # 检查模态覆盖
                checks[criterion] = self._check_modalities(requirement, steps)

            elif criterion == 'min_steps':
                # 检查最小步数
                checks[criterion] = len(steps) >= requirement

            elif criterion == 'closed_loop_required':
                # 检查闭环
                if requirement:
                    checks[criterion] = self._check_closed_loop(steps)
                else:
                    checks[criterion] = True

            elif criterion == 'systematic_analysis':
                # 检查系统分析
                if requirement:
                    checks[criterion] = self._check_systematic(steps, answer)
                else:
                    checks[criterion] = True

            elif criterion == 'min_regions_compared':
                # 检查比较的脑区数量
                checks[criterion] = self._check_regions_compared(answer, requirement)

            elif criterion == 'statistical_testing':
                # 检查统计检验
                if requirement:
                    checks[criterion] = self._check_statistical_test(steps, answer)
                else:
                    checks[criterion] = True

            elif criterion == 'regions_identified':
                # 检查是否识别了脑区
                checks[criterion] = self._check_regions_identified(answer, requirement)

            elif criterion == 'quantitative_data':
                # 检查是否有定量数据
                checks[criterion] = self._check_quantitative_data(answer, requirement)

            elif criterion == 'factual_correct':
                # 检查事实正确性
                checks[criterion] = agent_output.get('success', False)

            else:
                # 未知criterion，默认通过
                checks[criterion] = True

        return checks

    def _check_modalities(self, required_modalities: List[str], steps: List[Dict]) -> bool:
        """检查模态覆盖"""
        modalities_used = set()

        for step in steps:
            modality = step.get('modality')
            if modality:
                modalities_used.add(modality)

        return set(required_modalities).issubset(modalities_used)

    def _check_closed_loop(self, steps: List[Dict]) -> bool:
        """检查是否完成闭环"""
        has_projection = False
        has_target_composition = False

        for step in steps:
            purpose = step.get('purpose', '').lower()
            modality = step.get('modality', '')

            if 'projection' in purpose or modality == 'projection':
                has_projection = True

            if ('target' in purpose and 'composition' in purpose) or \
               ('target' in purpose and modality == 'molecular'):
                has_target_composition = True

        return has_projection and has_target_composition

    def _check_systematic(self, steps: List[Dict], answer: str) -> bool:
        """检查是否进行了系统分析"""
        # 检查步骤中是否有comparative/systematic关键词
        for step in steps:
            purpose = step.get('purpose', '').lower()
            if any(kw in purpose for kw in ['compare', 'systematic', 'all', 'multiple', 'screen']):
                return True

        # 检查答案中是否有systematic分析的迹象
        answer_lower = answer.lower()
        if any(kw in answer_lower for kw in ['compared', 'across regions', 'systematic', 'all regions']):
            return True

        return False

    def _check_regions_compared(self, answer: str, min_count: int) -> bool:
        """检查比较了多少个脑区"""
        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)

        unique_regions = set(regions)

        stopwords = {'DNA', 'RNA', 'ATP', 'GABA', 'LLM', 'ALL', 'THE'}
        unique_regions -= stopwords

        return len(unique_regions) >= min_count

    def _check_statistical_test(self, steps: List[Dict], answer: str) -> bool:
        """检查是否进行了统计检验"""
        for step in steps:
            step_type = step.get('step_type', '')
            purpose = step.get('purpose', '').lower()

            if step_type == 'statistical' or \
               any(kw in purpose for kw in ['statistic', 'test', 'fdr', 'p-value', 'significance']):
                return True

        answer_lower = answer.lower()
        stat_terms = ['p-value', 'p value', 'statistical', 'significance', 'fdr', 't-test', 'anova']

        return any(term in answer_lower for term in stat_terms)

    def _check_regions_identified(self, answer: str, required: bool) -> bool:
        """检查是否识别了脑区"""
        if not required:
            return True

        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)
        return len(regions) > 0

    def _check_quantitative_data(self, answer: str, required: bool) -> bool:
        """检查是否有定量数据"""
        if not required:
            return True

        has_numbers = bool(re.search(r'\d+', answer))
        return has_numbers

    def _default_evaluation(self, question_data: Dict, agent_output: Dict) -> str:
        """默认评估方法（当没有定义criteria时）"""

        expected_depth = question_data.get('expected_depth', 'medium')
        steps = agent_output.get('executed_steps', [])
        answer = agent_output.get('answer', '')

        if not agent_output.get('success', False):
            return 'failed'

        if len(answer) < 50:
            return 'failed'

        # 简单的启发式
        if expected_depth == 'shallow':
            return 'completed' if len(steps) >= 1 else 'failed'

        elif expected_depth == 'medium':
            if len(steps) >= 2:
                return 'completed'
            elif len(steps) >= 1:
                return 'partial'
            else:
                return 'failed'

        else:  # deep
            if len(steps) >= 4:
                return 'completed'
            elif len(steps) >= 2:
                return 'partial'
            else:
                return 'failed'


# ==================== Comprehensive Evaluator (Updated) ====================

class ComprehensiveEvaluator:
    """
    综合评估器（v3.0 - 公平的分层评估）

    🔧 关键改进：
    - 分层评估：区分核心能力和系统能力
    - None-able指标：不强制所有方法在所有指标上评分
    - 方法特定权重：计算Overall分数
    """

    def __init__(self):
        self.planning_eval = AdaptivePlanningEvaluator()
        self.entity_eval = EntityRecognitionEvaluator()
        self.answer_eval = AnswerQualityEvaluator()
        self.task_eval = BiologicalTaskEvaluator()

        self.config = EVALUATION_CONFIG

    def evaluate_full(self,
                     question_data: Dict,
                     agent_output: Dict,
                     method_name: str) -> EvaluationMetrics:
        """
        完整评估（v3.0 - 公平版）

        🔧 修复：
        - 只评估适用的指标
        - 使用方法特定权重计算Overall
        """

        metrics = EvaluationMetrics()

        # D1: Adaptive Planning (系统能力 - 分层评估)
        planning_metrics = self.planning_eval.evaluate(
            question_data, agent_output, method_name
        )

        metrics.depth_matching_accuracy = planning_metrics.get('depth_matching')
        metrics.plan_coherence = planning_metrics.get('plan_coherence')
        metrics.strategy_selection_accuracy = planning_metrics.get('strategy_selection')
        metrics.modality_coverage = planning_metrics.get('modality_coverage')

        closed_loop_score = planning_metrics.get('closed_loop')
        if closed_loop_score is not None:
            metrics.closed_loop_achieved = closed_loop_score >= 0.9
        else:
            metrics.closed_loop_achieved = None

        # D2: Entity Recognition (核心能力 - 所有方法)
        entity_metrics = self.entity_eval.evaluate(question_data, agent_output)
        metrics.entity_precision = entity_metrics['entity_precision']
        metrics.entity_recall = entity_metrics['entity_recall']
        metrics.entity_f1 = entity_metrics['entity_f1']

        # D3: Multi-hop (所有有KG访问的方法)
        steps = agent_output.get('executed_steps', [])
        metrics.multi_hop_depth = len(steps)

        if steps:
            successful = sum(1 for s in steps if s.get('success', True))
            metrics.query_success_rate = successful / len(steps)
        else:
            metrics.query_success_rate = 1.0

        # D4: Multi-Modal (已在planning中评估)
        modalities = set(s.get('modality') for s in steps if s.get('modality'))
        metrics.modalities_used = list(modalities)

        # D5: Answer Quality (核心能力 - 所有方法)
        answer_metrics = self.answer_eval.evaluate(question_data, agent_output)
        metrics.factual_accuracy = answer_metrics['factual_accuracy']
        metrics.answer_completeness = answer_metrics['answer_completeness']
        metrics.scientific_rigor = answer_metrics['scientific_rigor']

        # D6: Efficiency (所有方法)
        metrics.execution_time = agent_output.get('execution_time', 0.0)
        metrics.api_calls = len(steps)

        # 🔧 Task Completion (如果有定义)
        if question_data.get('task_type'):
            metrics.task_completion = self.task_eval.evaluate_task_completion(
                question_data, agent_output
            )

        # 🔧 计算方法特定的Overall分数
        metrics.overall_score = self._calculate_weighted_overall(metrics, method_name)

        return metrics

    def _calculate_weighted_overall(self, metrics: EvaluationMetrics, method_name: str) -> float:
        """
        🔧 计算方法特定的加权Overall分数

        关键：只对non-None的指标加权
        """

        weights = self.config['method_weights'].get(method_name, {})

        if not weights:
            # Fallback：核心指标简单平均
            core_scores = [
                metrics.entity_f1,
                metrics.factual_accuracy,
                metrics.answer_completeness,
                metrics.scientific_rigor,
            ]
            valid_scores = [s for s in core_scores if s is not None]
            return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # 加权平均（只对non-None的指标）
        weighted_sum = 0.0
        total_weight = 0.0

        metric_values = {
            'entity_f1': metrics.entity_f1,
            'factual_accuracy': metrics.factual_accuracy,
            'answer_completeness': metrics.answer_completeness,
            'scientific_rigor': metrics.scientific_rigor,
            'depth_matching': metrics.depth_matching_accuracy,
            'plan_coherence': metrics.plan_coherence,
            'modality_coverage': metrics.modality_coverage,
            'closed_loop': 1.0 if metrics.closed_loop_achieved else (0.0 if metrics.closed_loop_achieved is not None else None),
        }

        for metric_name, weight in weights.items():
            value = metric_values.get(metric_name)

            if value is not None:  # 🔧 只计算non-None的指标
                weighted_sum += value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


# ==================== Export ====================

__all__ = [
    'EvaluationMetrics',
    'AdaptivePlanningEvaluator',
    'EntityRecognitionEvaluator',
    'AnswerQualityEvaluator',
    'BiologicalTaskEvaluator',
    'ComprehensiveEvaluator',
    'EVALUATION_CONFIG',
]


# ==================== Test ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("✅ Enhanced evaluators.py v3.0 (Fair Evaluation) loaded successfully!")
    print("="*80)

    print("\n📊 Evaluation Configuration:")
    print("\nCore Metrics (all methods):")
    for metric, config in EVALUATION_CONFIG['core_metrics'].items():
        print(f"  - {metric}: {config['description']}")

    print("\nSystem Metrics (method-specific):")
    for metric, config in EVALUATION_CONFIG['system_metrics'].items():
        print(f"  - {metric}: {config['description']}")
        print(f"    → Applicable to: {', '.join(config['methods'])}")

    print("\n🔧 Method-Specific Weights:")
    for method, weights in EVALUATION_CONFIG['method_weights'].items():
        print(f"\n{method}:")
        total = sum(weights.values())
        for metric, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  - {metric:25s}: {weight:.2f} ({weight/total*100:.1f}%)")

    print("\n" + "="*80)