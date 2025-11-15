"""
Benchmark Runner for AIPOM-CoT Evaluation
==========================================
主测试运行器，协调所有组件

Author: Claude & PrometheusTT
Date: 2025-01-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict

from test_questions import ALL_QUESTIONS, QuestionTier, TestQuestion
from baselines import (
    DirectGPT5Baseline,    # 🆕 新增
    TemplateKGBaseline, # 🆕 新增
    RAGBaseline,
    ReActBaseline,
)
from evaluators import ComprehensiveEvaluator, EvaluationMetrics

logger = logging.getLogger(__name__)


# ==================== Result Structures ====================

class BenchmarkResult:
    """单个测试结果"""

    def __init__(self,
                 question: TestQuestion,
                 method_name: str,
                 agent_output: Dict,
                 metrics: EvaluationMetrics):
        self.question = question
        self.method_name = method_name
        self.agent_output = agent_output
        self.metrics = metrics

    def to_dict(self) -> Dict:
        """转换为可序列化格式"""
        return {
            'question_id': self.question.id,
            'tier': self.question.tier.value,
            'question': self.question.question,
            'method': self.method_name,
            'answer': self.agent_output.get('answer', ''),
            'success': self.agent_output.get('success', False),
            'metrics': {
                # D1
                'depth_matching': self.metrics.depth_matching_accuracy,
                'plan_coherence': self.metrics.plan_coherence,
                'modality_coverage': self.metrics.modality_coverage,
                'strategy_selection': self.metrics.strategy_selection_accuracy,
                'closed_loop': 1.0 if self.metrics.closed_loop_achieved else 0.0,

                # D2
                'entity_precision': self.metrics.entity_precision,
                'entity_recall': self.metrics.entity_recall,
                'entity_f1': self.metrics.entity_f1,
                'multi_hop_depth': self.metrics.multi_hop_depth,

                # D4
                'modalities_used': self.metrics.modalities_used,

                # D5
                'factual_accuracy': self.metrics.factual_accuracy,
                'answer_completeness': self.metrics.answer_completeness,
                'scientific_rigor': self.metrics.scientific_rigor,

                # D6
                'execution_time': self.metrics.execution_time,
                'api_calls': self.metrics.api_calls,
                'query_success_rate': self.metrics.query_success_rate,
            }
        }


# ==================== Benchmark Runner ====================

class BenchmarkRunner:
    """
    主Benchmark运行器

    协调：
    - AIPOM-CoT agent
    - 3个baseline方法
    - 评估器
    - 结果保存
    """

    def __init__(self,
                 aipom_agent,
                 neo4j_exec,
                 openai_client,
                 output_dir: str = "./benchmark_results"):

        self.aipom_agent = aipom_agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 🔧 更新：使用GPT-5的baselines
        logger.info("Initializing baseline methods (with GPT-5)...")
        self.baselines = {
            'Direct GPT-5': DirectGPT5Baseline(openai_client),  # 🆕
            'Template-KG': TemplateKGBaseline(neo4j_exec, openai_client),
            'RAG': RAGBaseline(neo4j_exec, openai_client),
            'ReAct': ReActBaseline(neo4j_exec, openai_client, max_iterations=5),
        }

        # 初始化evaluator
        self.evaluator = ComprehensiveEvaluator()

        # 结果存储
        self.results = defaultdict(list)

        logger.info("✅ BenchmarkRunner initialized (v2.1, GPT-5)")

    def run_full_benchmark(self,
                           questions: Optional[List[TestQuestion]] = None,
                           methods: Optional[List[str]] = None,
                           max_questions: Optional[int] = None,
                           save_interval: int = 10) -> Dict:
        """运行完整benchmark（更新版）"""

        # 准备问题
        if questions is None:
            questions = ALL_QUESTIONS

        if max_questions:
            questions = questions[:max_questions]

        # 🔧 更新：准备方法（新的默认列表）
        if methods is None:
            methods = ['AIPOM-CoT', 'o1-preview', 'Template-KG', 'RAG', 'ReAct']

        logger.info(f"\n{'=' * 80}")
        logger.info(f"🚀 Starting Benchmark (v2.0)")
        logger.info(f"{'=' * 80}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'=' * 80}\n")

        # 运行测试
        total_tests = len(questions) * len(methods)

        with tqdm(total=total_tests, desc="Benchmark Progress") as pbar:
            for q_idx, question in enumerate(questions, 1):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Question {q_idx}/{len(questions)}: {question.id}")
                logger.info(
                    f"Complexity: {question.complexity_level.value if hasattr(question, 'complexity_level') else 'N/A'}")
                logger.info(f"Q: {question.question[:80]}...")
                logger.info(f"{'=' * 80}")

                for method_name in methods:
                    logger.info(f"\n[{method_name}] Testing...")

                    result = self._run_single_test(question, method_name)
                    self.results[method_name].append(result)

                    # 打印关键指标
                    metrics = result.metrics
                    logger.info(f"  ✓ Entity F1: {metrics.entity_f1:.3f}")
                    logger.info(f"  ✓ Depth Match: {metrics.depth_matching_accuracy:.3f}")
                    logger.info(f"  ✓ Closed Loop: {'Yes' if metrics.closed_loop_achieved else 'No'}")

                    # 🆕 打印task completion（如果有）
                    if hasattr(metrics, 'task_completion') and metrics.task_completion != 'unknown':
                        logger.info(f"  ✓ Task: {metrics.task_completion}")

                    logger.info(f"  ✓ Time: {metrics.execution_time:.2f}s")

                    pbar.update(1)

                # 定期保存
                if q_idx % save_interval == 0:
                    self._save_intermediate_results()
                    logger.info(f"\n💾 Intermediate results saved at Q{q_idx}")

        # 最终保存
        logger.info(f"\n{'=' * 80}")
        logger.info("📊 Generating final report...")
        logger.info(f"{'=' * 80}")

        self._save_final_results()
        summary = self._generate_summary()

        logger.info(f"\n✅ Benchmark Complete!")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary

    def _run_single_test(self,
                         question: TestQuestion,
                         method_name: str) -> BenchmarkResult:
        """运行单个测试"""

        try:
            # 运行agent/baseline
            if method_name == 'AIPOM-CoT':
                agent_output = self._run_aipom(question)
            else:
                agent_output = self._run_baseline(question, method_name)

            # 评估
            question_data = self._question_to_dict(question)
            metrics = self.evaluator.evaluate_full(
                question_data, agent_output, method_name
            )

            return BenchmarkResult(question, method_name, agent_output, metrics)

        except Exception as e:
            logger.error(f"  ✗ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()

            # 返回失败结果
            return self._create_failed_result(question, method_name, str(e))

    def _run_aipom(self, question: TestQuestion) -> Dict:
        """运行AIPOM-CoT"""

        # 根据问题复杂度设置max_iterations
        if question.tier == QuestionTier.SIMPLE:
            max_iter = 4
        elif question.tier == QuestionTier.MEDIUM:
            max_iter = 6
        elif question.tier == QuestionTier.DEEP:
            max_iter = 10
        else:  # SCREENING
            max_iter = 8

        result = self.aipom_agent.answer(
            question.question,
            max_iterations=max_iter
        )

        return result

    def _run_baseline(self, question: TestQuestion, method_name: str) -> Dict:
        """运行baseline方法"""

        baseline = self.baselines.get(method_name)
        if not baseline:
            raise ValueError(f"Unknown method: {method_name}")

        # 设置timeout
        if question.tier == QuestionTier.SIMPLE:
            timeout = 30
        elif question.tier == QuestionTier.MEDIUM:
            timeout = 60
        else:
            timeout = 120

        result = baseline.answer(question.question, timeout=timeout)

        return result

    def _question_to_dict(self, question: TestQuestion) -> Dict:
        """转换TestQuestion为dict"""
        return {
            'id': question.id,
            'tier': question.tier.value,
            'question': question.question,
            'expected_entities': question.expected_entities,
            'expected_depth': question.expected_depth,
            'expected_strategy': question.expected_strategy,
            'expected_modalities': question.expected_modalities,
            'expected_closed_loop': question.expected_closed_loop,
            'expected_steps_range': question.expected_steps_range,
            'domain': question.domain,
            'difficulty_score': question.difficulty_score,
        }

    def _create_failed_result(self,
                              question: TestQuestion,
                              method_name: str,
                              error: str) -> BenchmarkResult:
        """创建失败结果"""

        failed_output = {
            'question': question.question,
            'answer': f"ERROR: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': 0.0,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': method_name,
            'error': error,
        }

        # 创建零指标
        metrics = EvaluationMetrics()

        return BenchmarkResult(question, method_name, failed_output, metrics)

    def _save_intermediate_results(self):
        """保存中间结果"""
        filepath = self.output_dir / "intermediate_results.json"

        data = {}
        for method, results in self.results.items():
            data[method] = [r.to_dict() for r in results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_final_results(self):
        """保存最终结果"""

        # 1. 详细结果
        detailed_file = self.output_dir / "detailed_results.json"
        data = {}
        for method, results in self.results.items():
            data[method] = [r.to_dict() for r in results]

        with open(detailed_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"  ✓ Detailed results: {detailed_file}")

        # 2. 聚合统计
        summary = self._generate_summary()

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  ✓ Summary: {summary_file}")

    def _generate_summary(self) -> Dict:
        """生成聚合统计"""

        summary = {}

        for method, results in self.results.items():
            if not results:
                continue

            # 提取所有指标
            all_metrics = {
                'depth_matching': [],
                'entity_f1': [],
                'closed_loop': [],
                'modality_coverage': [],
                'factual_accuracy': [],
                'scientific_rigor': [],
                'execution_time': [],
            }

            for result in results:
                m = result.metrics
                all_metrics['depth_matching'].append(m.depth_matching_accuracy)
                all_metrics['entity_f1'].append(m.entity_f1)
                all_metrics['closed_loop'].append(1.0 if m.closed_loop_achieved else 0.0)
                all_metrics['modality_coverage'].append(m.modality_coverage)
                all_metrics['factual_accuracy'].append(m.factual_accuracy)
                all_metrics['scientific_rigor'].append(m.scientific_rigor)
                all_metrics['execution_time'].append(m.execution_time)

            # 计算均值和标准差
            import statistics

            summary[method] = {}
            for metric_name, values in all_metrics.items():
                if values:
                    summary[method][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                    }

            # 总体分数
            overall_scores = [
                all_metrics['depth_matching'],
                all_metrics['entity_f1'],
                all_metrics['modality_coverage'],
                all_metrics['scientific_rigor'],
            ]

            # 计算每个问题的平均分
            n = len(results)
            overall_per_question = []
            for i in range(n):
                scores = [overall_scores[j][i] for j in range(len(overall_scores))]
                overall_per_question.append(statistics.mean(scores))

            summary[method]['overall'] = {
                'mean': statistics.mean(overall_per_question),
                'std': statistics.stdev(overall_per_question) if len(overall_per_question) > 1 else 0.0,
            }

            # 按Tier统计
            by_tier = defaultdict(list)
            for result in results:
                tier = result.question.tier.value
                by_tier[tier].append(result)

            summary[method]['by_tier'] = {}
            for tier, tier_results in by_tier.items():
                tier_scores = []
                for result in tier_results:
                    m = result.metrics
                    score = statistics.mean([
                        m.depth_matching_accuracy,
                        m.entity_f1,
                        m.modality_coverage,
                        m.scientific_rigor,
                    ])
                    tier_scores.append(score)

                summary[method]['by_tier'][tier] = {
                    'mean': statistics.mean(tier_scores) if tier_scores else 0.0,
                    'count': len(tier_results),
                }

        return summary

    def print_summary(self):
        """打印汇总统计"""

        summary = self._generate_summary()

        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)

        for method in ['AIPOM-CoT', 'Direct LLM', 'RAG', 'ReAct']:
            if method not in summary:
                continue

            print(f"\n{method}:")
            print("-" * 40)

            overall = summary[method].get('overall', {})
            print(f"Overall Score: {overall.get('mean', 0):.3f} ± {overall.get('std', 0):.3f}")

            print(f"\nKey Metrics:")
            for metric in ['entity_f1', 'depth_matching', 'closed_loop', 'scientific_rigor']:
                if metric in summary[method]:
                    m = summary[method][metric]
                    print(f"  {metric:20s}: {m['mean']:.3f} ± {m['std']:.3f}")

            print(f"\nBy Tier:")
            by_tier = summary[method].get('by_tier', {})
            for tier in ['simple', 'medium', 'deep', 'screening']:
                if tier in by_tier:
                    t = by_tier[tier]
                    print(f"  {tier.capitalize():12s}: {t['mean']:.3f} (n={t['count']})")

        print("\n" + "=" * 80)


# ==================== Quick Test Function ====================

def run_quick_test(aipom_agent, neo4j_exec, openai_client, n_questions: int = 10):
    """运行快速测试（10题）"""

    from test_questions import TIER1_SIMPLE, TIER2_MEDIUM, TIER3_DEEP, TIER4_SCREENING

    # 选择代表性问题
    selected = []
    selected.extend(TIER1_SIMPLE[:2])  # 2个简单
    selected.extend(TIER2_MEDIUM[:3])  # 3个中等
    selected.extend(TIER3_DEEP[:3])  # 3个深度
    selected.extend(TIER4_SCREENING[:2])  # 2个筛选

    selected = selected[:n_questions]

    runner = BenchmarkRunner(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_quick"
    )

    summary = runner.run_full_benchmark(
        questions=selected,
        methods=['AIPOM-CoT', 'Direct LLM', 'RAG', 'ReAct'],
        save_interval=5
    )

    runner.print_summary()

    return summary


# ==================== Test ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("BenchmarkRunner loaded successfully!")