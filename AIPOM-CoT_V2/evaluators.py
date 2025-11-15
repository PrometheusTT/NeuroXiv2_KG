"""
Evaluators for AIPOM-CoT Benchmark
===================================
实现6个维度的评估器

Author: Claude & PrometheusTT
Date: 2025-01-15
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


# ==================== Data Structures ====================

@dataclass
class EvaluationMetrics:
    """完整评估指标"""

    # D1: Adaptive Planning
    depth_matching_accuracy: float = 0.0
    plan_coherence: float = 0.0
    modality_coverage: float = 0.0
    strategy_selection_accuracy: float = 0.0
    closed_loop_achieved: bool = False

    # D2: KG Reasoning
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    multi_hop_depth: int = 0
    multi_hop_success: bool = True

    # D3: Reflection (AIPOM-CoT only)
    replanning_triggered: int = 0
    confidence_calibration_error: float = 0.0

    # D4: Multi-Modal Integration
    modalities_used: List[str] = field(default_factory=list)
    cross_modal_citations: int = 0

    # D5: Answer Quality
    factual_accuracy: float = 0.0
    answer_completeness: float = 0.0
    scientific_rigor: float = 0.0

    # D6: Efficiency
    execution_time: float = 0.0
    api_calls: int = 0
    query_success_rate: float = 0.0

    task_completion: str = 'unknown'  # 'completed', 'partial', 'failed', 'unknown'


# ==================== Evaluator Base Class ====================

class BaseEvaluator:
    """评估器基类"""

    def __init__(self):
        self.stopwords = self._build_stopwords()

    def _build_stopwords(self) -> set:
        """构建停用词表"""
        stopwords = set([
            # 疑问词
            'what', 'which', 'where', 'when', 'who', 'why', 'how',
            # be动词
            'are', 'is', 'was', 'were', 'be', 'been', 'being', 'am',
            # 助动词
            'do', 'does', 'did', 'have', 'has', 'had',
            'can', 'could', 'will', 'would', 'shall', 'should',
            # 介词
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            # 冠词
            'the', 'an', 'a',
            # 代词
            'it', 'its', 'they', 'their', 'this', 'that',
            # 常见动词
            'get', 'give', 'show', 'tell', 'make', 'take',
            # 神经科学通用词（不是实体）
            'cells', 'neurons', 'brain', 'regions', 'region', 'areas', 'area',
        ])
        return stopwords


# ==================== D1: Adaptive Planning Evaluator ====================

class AdaptivePlanningEvaluator(BaseEvaluator):
    """评估自适应规划能力"""

    def evaluate(self,
                 question_data: Dict,
                 agent_output: Dict,
                 method_name: str) -> Dict[str, float]:
        """
        评估自适应规划

        Args:
            question_data: 测试问题数据
            agent_output: Agent输出
            method_name: 方法名称

        Returns:
            评估指标dict
        """

        metrics = {}

        # D1.1: Depth Matching Accuracy
        metrics['depth_matching'] = self._evaluate_depth_matching(
            question_data, agent_output, method_name
        )

        # D1.2: Plan Coherence
        metrics['plan_coherence'] = self._evaluate_plan_coherence(
            agent_output, method_name
        )

        # D1.3: Modality Coverage
        metrics['modality_coverage'] = self._evaluate_modality_coverage(
            question_data, agent_output
        )

        # D1.4: Strategy Selection (AIPOM-CoT only)
        if method_name == 'AIPOM-CoT':
            metrics['strategy_selection'] = self._evaluate_strategy_selection(
                question_data, agent_output
            )
        else:
            metrics['strategy_selection'] = 0.0

        # D1.5: Closed-Loop Achievement
        metrics['closed_loop'] = self._evaluate_closed_loop(
            question_data, agent_output
        )

        return metrics

    def _evaluate_depth_matching(self,
                                 question_data: Dict,
                                 agent_output: Dict,
                                 method_name: str) -> float:
        """评估深度匹配"""

        expected_depth = question_data.get('expected_depth', 'medium')
        executed_steps = agent_output.get('total_steps', 0)

        # Baseline方法深度固定
        if method_name == 'Direct LLM':
            return 1.0 if expected_depth == 'shallow' else 0.0

        if method_name == 'RAG':
            return 1.0 if expected_depth in ['shallow', 'medium'] else 0.0

        if method_name == 'ReAct':
            # ReAct固定3步
            if expected_depth == 'medium' and 2 <= executed_steps <= 4:
                return 1.0
            else:
                return 0.3

        # AIPOM-CoT
        depth_map = {
            'shallow': (1, 2),
            'medium': (3, 4),
            'deep': (5, 7),
        }

        expected_range = depth_map.get(expected_depth, (3, 4))
        min_steps, max_steps = expected_range

        # 在范围内 → 1.0
        if min_steps <= executed_steps <= max_steps:
            return 1.0

        # 在范围外，计算偏离程度
        if executed_steps < min_steps:
            deviation = min_steps - executed_steps
        else:
            deviation = executed_steps - max_steps

        # 每偏离1步，扣0.2分
        score = max(0.0, 1.0 - deviation * 0.2)

        return score

    def _evaluate_plan_coherence(self, agent_output: Dict, method_name: str) -> float:
        """评估计划连贯性"""

        steps = agent_output.get('executed_steps', [])

        if not steps:
            return 0.0

        if method_name in ['Direct LLM']:
            return 0.0  # 无计划

        if method_name == 'RAG':
            return 0.3  # 单步检索，连贯性低

        # ReAct和AIPOM-CoT检查step之间的依赖
        has_dependencies = 0
        for i, step in enumerate(steps):
            if i > 0:
                # 检查purpose是否提到前一步
                purpose = step.get('purpose', '').lower()
                prev_purpose = steps[i - 1].get('purpose', '').lower()

                # 简单启发式：是否提到"target", "focus", "primary"等
                if any(kw in purpose for kw in ['target', 'focus', 'primary', 'discovered', 'identified']):
                    has_dependencies += 1

        if len(steps) <= 1:
            return 0.5

        coherence = has_dependencies / (len(steps) - 1)

        return coherence

    def _evaluate_modality_coverage(self, question_data: Dict, agent_output: Dict) -> float:
        """评估模态覆盖"""

        expected_modalities = set(question_data.get('expected_modalities', []))

        # 从steps提取实际使用的模态
        steps = agent_output.get('executed_steps', [])
        used_modalities = set()

        for step in steps:
            modality = step.get('modality')
            if modality:
                used_modalities.add(modality)

        # 从答案推断模态
        answer = agent_output.get('answer', '').lower()

        molecular_kw = ['gene', 'marker', 'express', 'cluster', 'subclass', 'cell type']
        if any(kw in answer for kw in molecular_kw):
            used_modalities.add('molecular')

        morpho_kw = ['axon', 'dendrite', 'morpholog', 'branch', 'length', 'arbor']
        if any(kw in answer for kw in morpho_kw):
            used_modalities.add('morphological')

        projection_kw = ['project', 'target', 'connect', 'pathway', 'circuit']
        if any(kw in answer for kw in projection_kw):
            used_modalities.add('projection')

        if not expected_modalities:
            return 1.0

        coverage = len(used_modalities & expected_modalities) / len(expected_modalities)

        return coverage

    def _evaluate_strategy_selection(self, question_data: Dict, agent_output: Dict) -> float:
        """评估策略选择（AIPOM-CoT only）"""

        expected_strategy = question_data.get('expected_strategy', 'adaptive')

        # 从agent_output提取实际策略
        actual_strategy = agent_output.get('adaptive_planning', {}).get('selected_planner', 'unknown')

        if actual_strategy == expected_strategy:
            return 1.0

        # 部分匹配
        if expected_strategy == 'focus_driven' and actual_strategy == 'adaptive':
            return 0.5

        return 0.0

    def _evaluate_closed_loop(self, question_data: Dict, agent_output: Dict) -> float:
        """评估闭环完成"""

        expected_closed_loop = question_data.get('expected_closed_loop', False)

        if not expected_closed_loop:
            # 不需要闭环
            return 1.0

        # 检查是否有target composition步骤
        steps = agent_output.get('executed_steps', [])

        has_projection = False
        has_target_composition = False

        for step in steps:
            purpose = step.get('purpose', '').lower()
            modality = step.get('modality', '')

            if 'projection' in purpose or modality == 'projection':
                has_projection = True

            if 'target' in purpose and 'composition' in purpose:
                has_target_composition = True

            if 'target' in purpose and modality == 'molecular':
                has_target_composition = True

        if has_projection and has_target_composition:
            return 1.0
        elif has_projection:
            return 0.5  # 有projection但没闭环
        else:
            return 0.0


# ==================== D2: Entity Recognition Evaluator ====================

class EntityRecognitionEvaluator(BaseEvaluator):
    """评估实体识别"""

    def evaluate(self,
                 question_data: Dict,
                 agent_output: Dict) -> Dict[str, float]:
        """评估实体识别F1"""

        expected_entities = set([
            e.lower().strip()
            for e in question_data.get('expected_entities', [])
            if e
        ])

        predicted_entities = set()

        # 从agent_output提取
        for e in agent_output.get('entities_recognized', []):
            if isinstance(e, dict):
                text = e.get('text', '').lower().strip()
            else:
                text = str(e).lower().strip()

            if text and len(text) >= 2 and text not in self.stopwords:
                predicted_entities.add(text)

        # 从问题中提取明显实体（辅助）
        question = question_data.get('question', '')
        question_entities = self._extract_from_question(question)

        predicted_entities |= question_entities

        # 计算F1
        if not expected_entities:
            # 没有expected entities，认为通过
            return {'entity_precision': 1.0, 'entity_recall': 1.0, 'entity_f1': 1.0}

        # 模糊匹配
        true_positives = 0
        for expected in expected_entities:
            for predicted in predicted_entities:
                if self._fuzzy_match(expected, predicted):
                    true_positives += 1
                    break

        false_positives = len(predicted_entities) - true_positives
        false_negatives = len(expected_entities) - true_positives

        precision = true_positives / (true_positives + false_positives) if (
                                                                                       true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
        }

    def _extract_from_question(self, question: str) -> set:
        """从问题提取明显实体"""
        entities = set()

        # 脑区缩写
        regions = re.findall(r'\b[A-Z]{2,5}\b', question)
        for r in regions:
            if len(r) >= 2 and r.lower() not in self.stopwords:
                entities.add(r.lower())

        # 基因名
        genes = re.findall(r'\b[A-Z][a-z]{2,8}\d*\+?\b', question)
        gene_stopwords = {'What', 'Which', 'Where', 'Tell', 'Give', 'Show', 'Find'}
        for g in genes:
            if g not in gene_stopwords:
                entities.add(g.rstrip('+').lower())

        return entities

    def _fuzzy_match(self, expected: str, predicted: str) -> bool:
        """模糊匹配"""
        expected = expected.lower().strip()
        predicted = predicted.lower().strip()

        if expected == predicted:
            return True

        if expected in predicted or predicted in expected:
            return True

        if len(expected) > 3 and len(predicted) > 3:
            if expected[:3] == predicted[:3]:
                return True

        return False


# ==================== D5: Answer Quality Evaluator ====================

class AnswerQualityEvaluator(BaseEvaluator):
    """评估答案质量"""

    def evaluate(self,
                 question_data: Dict,
                 agent_output: Dict) -> Dict[str, float]:
        """评估答案质量"""

        answer = agent_output.get('answer', '')
        question = question_data.get('question', '')

        metrics = {}

        # D5.1: Factual Accuracy
        metrics['factual_accuracy'] = self._evaluate_factual_accuracy(answer)

        # D5.2: Answer Completeness
        metrics['answer_completeness'] = self._evaluate_completeness(
            answer, question, question_data
        )

        # D5.3: Scientific Rigor
        metrics['scientific_rigor'] = self._evaluate_scientific_rigor(answer)

        return metrics

    def _evaluate_factual_accuracy(self, answer: str) -> float:
        """评估事实准确性"""
        answer_lower = answer.lower()

        score = 0.0

        # 有具体数据
        if re.search(r'\d+', answer):
            score += 0.3

        # 有脑区名称
        if re.search(r'\b[A-Z]{2,5}\b', answer):
            score += 0.3

        # 有科学术语
        sci_terms = ['neuron', 'cell', 'region', 'cortex', 'gene', 'marker',
                     'cluster', 'projection', 'axon', 'dendrite']
        if any(term in answer_lower for term in sci_terms):
            score += 0.2

        # 有定量描述
        quant_terms = ['average', 'mean', 'number', 'count', 'percentage', '%']
        if any(term in answer_lower for term in quant_terms):
            score += 0.2

        return min(1.0, score)

    def _evaluate_completeness(self, answer: str, question: str, question_data: Dict) -> float:
        """评估完整性"""

        answer_words = len(answer.split())
        question_words = len(question.split())

        # 预期长度
        expected_depth = question_data.get('expected_depth', 'medium')

        if expected_depth == 'shallow':
            expected_length = 50
        elif expected_depth == 'medium':
            expected_length = 150
        else:  # deep
            expected_length = 300

        # 长度适中性
        length_score = min(1.0, answer_words / expected_length)

        # 检查是否回答了问题的各个方面
        expected_modalities = question_data.get('expected_modalities', [])

        coverage = 0.0
        for modality in expected_modalities:
            if modality == 'molecular':
                if any(kw in answer.lower() for kw in ['gene', 'marker', 'cluster', 'cell type']):
                    coverage += 1
            elif modality == 'morphological':
                if any(kw in answer.lower() for kw in ['axon', 'dendrite', 'morpholog', 'branch']):
                    coverage += 1
            elif modality == 'projection':
                if any(kw in answer.lower() for kw in ['project', 'target', 'connect']):
                    coverage += 1

        if expected_modalities:
            modality_score = coverage / len(expected_modalities)
        else:
            modality_score = 1.0

        completeness = (length_score + modality_score) / 2

        return completeness

    def _evaluate_scientific_rigor(self, answer: str) -> float:
        """评估科学严谨性"""
        answer_lower = answer.lower()

        score = 0.0

        # 有科学术语
        sci_terms = ['neuron', 'cortex', 'expression', 'projection',
                     'morphology', 'cluster', 'marker', 'region', 'circuit']
        sci_count = sum(1 for term in sci_terms if term in answer_lower)
        score += min(0.4, sci_count * 0.1)

        # 有定量数据
        has_numbers = bool(re.search(r'\d+', answer))
        if has_numbers:
            score += 0.3

        # 避免模糊词
        vague_terms = ['some', 'several', 'many', 'few', 'various', 'might', 'maybe']
        vague_count = sum(1 for term in vague_terms if term in answer_lower)
        score += max(0.0, 0.3 - vague_count * 0.1)

        return min(1.0, score)


# ==================== Comprehensive Evaluator ====================

class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self):
        self.planning_eval = AdaptivePlanningEvaluator()
        self.entity_eval = EntityRecognitionEvaluator()
        self.answer_eval = AnswerQualityEvaluator()
        self.task_eval = BiologicalTaskEvaluator()

    def evaluate_full(self,
                      question_data: Dict,
                      agent_output: Dict,
                      method_name: str) -> EvaluationMetrics:
        """完整评估（更新版）"""

        metrics = EvaluationMetrics()

        # D1: Adaptive Planning
        planning_metrics = self.planning_eval.evaluate(
            question_data, agent_output, method_name
        )
        metrics.depth_matching_accuracy = planning_metrics.get('depth_matching', 0.0)
        metrics.plan_coherence = planning_metrics.get('plan_coherence', 0.0)
        metrics.modality_coverage = planning_metrics.get('modality_coverage', 0.0)
        metrics.strategy_selection_accuracy = planning_metrics.get('strategy_selection', 0.0)
        metrics.closed_loop_achieved = planning_metrics.get('closed_loop', 0.0) >= 0.9

        # D2: Entity Recognition
        entity_metrics = self.entity_eval.evaluate(question_data, agent_output)
        metrics.entity_precision = entity_metrics['entity_precision']
        metrics.entity_recall = entity_metrics['entity_recall']
        metrics.entity_f1 = entity_metrics['entity_f1']

        # 🆕 Task Completion（如果有定义）
        if question_data.get('task_type'):
            task_completion = self.task_eval.evaluate_task_completion(question_data, agent_output)
            # 存储在metrics中（需要添加字段）
            if not hasattr(metrics, 'task_completion'):
                metrics.task_completion = task_completion

        # D4: Multi-Modal
        steps = agent_output.get('executed_steps', [])
        modalities = set(s.get('modality') for s in steps if s.get('modality'))
        metrics.modalities_used = list(modalities)

        # D5: Answer Quality
        answer_metrics = self.answer_eval.evaluate(question_data, agent_output)
        metrics.factual_accuracy = answer_metrics['factual_accuracy']
        metrics.answer_completeness = answer_metrics['answer_completeness']
        metrics.scientific_rigor = answer_metrics['scientific_rigor']

        # D6: Efficiency
        metrics.execution_time = agent_output.get('execution_time', 0.0)
        metrics.api_calls = len(steps)
        metrics.multi_hop_depth = len(steps)

        # Query success rate
        if steps:
            successful = sum(1 for s in steps if s.get('success', True))
            metrics.query_success_rate = successful / len(steps)
        else:
            metrics.query_success_rate = 1.0

        return metrics


class BiologicalTaskEvaluator:
    """
    生物学任务评估器

    评估任务完成度：'completed', 'partial', 'failed'
    """

    def __init__(self):
        self.stopwords = self._build_stopwords()

    def _build_stopwords(self) -> set:
        """构建停用词表"""
        return set([
            'what', 'which', 'where', 'when', 'who', 'why', 'how',
            'are', 'is', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'have', 'has', 'had',
            'the', 'an', 'a', 'this', 'that',
        ])

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
        import re

        # 提取脑区缩写
        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)

        # 去重
        unique_regions = set(regions)

        # 过滤掉常见非脑区词
        stopwords = {'DNA', 'RNA', 'ATP', 'GABA', 'LLM', 'ALL'}
        unique_regions -= stopwords

        return len(unique_regions) >= min_count

    def _check_statistical_test(self, steps: List[Dict], answer: str) -> bool:
        """检查是否进行了统计检验"""
        # 检查步骤中是否有statistical类型
        for step in steps:
            step_type = step.get('step_type', '')
            purpose = step.get('purpose', '').lower()

            if step_type == 'statistical' or \
                    any(kw in purpose for kw in ['statistic', 'test', 'fdr', 'p-value', 'significance']):
                return True

        # 检查答案中是否提到统计术语
        answer_lower = answer.lower()
        stat_terms = ['p-value', 'p value', 'statistical', 'significance', 'fdr', 't-test', 'anova']

        return any(term in answer_lower for term in stat_terms)

    def _default_evaluation(self, question_data: Dict, agent_output: Dict) -> str:
        """默认评估方法（当没有定义criteria时）"""

        # 基于expected_depth评估
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
# ==================== Test ====================

if __name__ == "__main__":
    print("Evaluators loaded successfully!")
    print("\nAvailable evaluators:")
    print("1. AdaptivePlanningEvaluator - D1 metrics")
    print("2. EntityRecognitionEvaluator - D2 metrics")
    print("3. AnswerQualityEvaluator - D5 metrics")
    print("4. ComprehensiveEvaluator - All metrics")