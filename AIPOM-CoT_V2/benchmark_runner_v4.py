"""
Benchmark Runner for AIPOM-CoT Evaluation (v4.0 - Nature Methods)
==================================================================
支持NM核心能力评估的主运行器

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 4.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict

from test_questions import ALL_QUESTIONS, QuestionTier, TestQuestion
from baselines import (
    DirectGPT4oBaseline,
    TemplateKGBaseline,
    RAGBaseline,
    ReActBaseline,
)

# 🔧 导入v4.0评估器
import sys

sys.path.append('/tmp')
from evaluators_v4 import ComprehensiveEvaluatorV4, NMEvaluationMetrics

logger = logging.getLogger(__name__)


# ==================== Result Structures ====================

class BenchmarkResultV4:
    """单个测试结果 (v4.0)"""

    def __init__(self,
                 question: TestQuestion,
                 method_name: str,
                 agent_output: Dict,
                 metrics: NMEvaluationMetrics):
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
                # 🔬 NM核心能力
                'planning_quality': self.metrics.planning_quality,
                'planning_coherence': self.metrics.planning_coherence,
                'planning_optimality': self.metrics.planning_optimality,
                'planning_adaptability': self.metrics.planning_adaptability,

                'reasoning_capability': self.metrics.reasoning_capability,
                'logical_consistency': self.metrics.logical_consistency,
                'evidence_integration': self.metrics.evidence_integration,
                'multi_hop_depth_score': self.metrics.multi_hop_depth_score,

                'cot_quality': self.metrics.cot_quality,
                'cot_clarity': self.metrics.cot_clarity,
                'cot_completeness': self.metrics.cot_completeness,
                'intermediate_steps_quality': self.metrics.intermediate_steps_quality,

                'reflection_capability': self.metrics.reflection_capability,
                'error_detection': self.metrics.error_detection,
                'self_correction': self.metrics.self_correction,
                'iterative_refinement': self.metrics.iterative_refinement,

                'nlu_capability': self.metrics.nlu_capability,
                'query_understanding': self.metrics.query_understanding,
                'intent_recognition': self.metrics.intent_recognition,
                'ambiguity_resolution': self.metrics.ambiguity_resolution,

                # 传统指标
                'entity_f1': self.metrics.entity_f1,
                'factual_accuracy': self.metrics.factual_accuracy,
                'answer_completeness': self.metrics.answer_completeness,
                'scientific_rigor': self.metrics.scientific_rigor,

                'reasoning_depth': self.metrics.reasoning_depth,
                'modality_coverage': self.metrics.modality_coverage,
                'closed_loop': 1.0 if self.metrics.closed_loop_achieved else 0.0 if self.metrics.closed_loop_achieved is not None else None,

                # 效率
                'execution_time': self.metrics.execution_time,
                'api_calls': self.metrics.api_calls,

                # Overall
                'overall_score': self.metrics.overall_score,
                'nm_capability_score': self.metrics.nm_capability_score,
                'biological_insight_score': self.metrics.biological_insight_score,
                'task_completion': self.metrics.task_completion,
            }
        }


# ==================== Benchmark Runner (v4.0) ====================

class BenchmarkRunnerV4:
    """主Benchmark运行器 (v4.0)"""

    def __init__(self,
                 aipom_agent,
                 neo4j_exec,
                 openai_client,
                 output_dir: str = "./benchmark_results_v4"):

        self.aipom_agent = aipom_agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Initializing baseline methods...")
        self.baselines = {
            'Direct GPT-4o': DirectGPT4oBaseline(openai_client),
            'Template-KG': TemplateKGBaseline(neo4j_exec, openai_client),
            'RAG': RAGBaseline(neo4j_exec, openai_client),
            'ReAct': ReActBaseline(neo4j_exec, openai_client, base_max_iterations=10),
        }

        # 🔧 使用v4.0评估器
        self.evaluator = ComprehensiveEvaluatorV4()

        self.results = defaultdict(list)

        logger.info("✅ BenchmarkRunner v4.0 (Nature Methods) initialized")

    def run_full_benchmark(self,
                           questions: Optional[List[TestQuestion]] = None,
                           methods: Optional[List[str]] = None,
                           max_questions: Optional[int] = None,
                           save_interval: int = 10) -> Dict:
        """运行完整benchmark (v4.0)"""

        if questions is None:
            questions = ALL_QUESTIONS

        if max_questions:
            questions = questions[:max_questions]

        if methods is None:
            methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        logger.info(f"\n{'=' * 80}")
        logger.info(f"🚀 Starting Benchmark v4.0 (Nature Methods)")
        logger.info(f"{'=' * 80}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'=' * 80}\n")

        total_tests = len(questions) * len(methods)

        with tqdm(total=total_tests, desc="Benchmark Progress") as pbar:
            for q_idx, question in enumerate(questions, 1):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Question {q_idx}/{len(questions)}: {question.id}")
                logger.info(f"Q: {question.question[:80]}...")
                logger.info(f"{'=' * 80}")

                for method_name in methods:
                    logger.info(f"\n[{method_name}] Testing...")

                    result = self._run_single_test(question, method_name)
                    self.results[method_name].append(result)

                    # 打印关键指标
                    metrics = result.metrics

                    # NM核心能力
                    if metrics.planning_quality is not None:
                        logger.info(f"  ✓ Planning: {metrics.planning_quality:.3f}")
                    if metrics.reasoning_capability is not None:
                        logger.info(f"  ✓ Reasoning: {metrics.reasoning_capability:.3f}")
                    if metrics.cot_quality is not None:
                        logger.info(f"  ✓ CoT Quality: {metrics.cot_quality:.3f}")
                    if metrics.reflection_capability is not None:
                        logger.info(f"  ✓ Reflection: {metrics.reflection_capability:.3f}")
                    if metrics.nlu_capability is not None:
                        logger.info(f"  ✓ NLU: {metrics.nlu_capability:.3f}")

                    # 传统指标
                    logger.info(f"  ✓ Entity F1: {metrics.entity_f1:.3f}")
                    logger.info(f"  ✓ Scientific Rigor: {metrics.scientific_rigor:.3f}")

                    # Overall
                    logger.info(f"  ✓ NM Score: {metrics.nm_capability_score:.3f}")
                    logger.info(f"  ✓ Overall: {metrics.overall_score:.3f}")
                    logger.info(f"  ✓ Time: {metrics.execution_time:.2f}s")

                    pbar.update(1)

                if q_idx % save_interval == 0:
                    self._save_intermediate_results()
                    logger.info(f"\n💾 Intermediate results saved at Q{q_idx}")

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 Generating final report...")
        logger.info(f"{'=' * 80}")

        self._save_final_results()
        summary = self._generate_summary()

        logger.info(f"\n✅ Benchmark Complete!")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary

    def _run_single_test(self, question: TestQuestion, method_name: str) -> BenchmarkResultV4:
        """运行单个测试"""

        try:
            if method_name == 'AIPOM-CoT':
                agent_output = self._run_aipom(question)
            else:
                agent_output = self._run_baseline(question, method_name)

            question_data = self._question_to_dict(question)
            metrics = self.evaluator.evaluate_full(
                question_data, agent_output, method_name
            )

            return BenchmarkResultV4(question, method_name, agent_output, metrics)

        except Exception as e:
            logger.error(f"  ✗ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()

            return self._create_failed_result(question, method_name, str(e))

    def _run_aipom(self, question: TestQuestion) -> Dict:
        """运行AIPOM-CoT"""

        if question.tier == QuestionTier.SIMPLE:
            max_iter = 4
        elif question.tier == QuestionTier.MEDIUM:
            max_iter = 6
        elif question.tier == QuestionTier.DEEP:
            max_iter = 10
        else:
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

        if question.tier == QuestionTier.SIMPLE:
            timeout = 40
        elif question.tier == QuestionTier.MEDIUM:
            timeout = 80
        elif question.tier == QuestionTier.DEEP:
            timeout = 150
        else:
            timeout = 120

        kwargs = {
            'timeout': timeout,
            'question_tier': question.tier.value,
        }

        result = baseline.answer(question.question, **kwargs)

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
            'task_type': question.task_type.value if question.task_type else None,
            'success_criteria': question.success_criteria,
            'partial_criteria': question.partial_criteria,
        }

    def _create_failed_result(self, question: TestQuestion, method_name: str, error: str) -> BenchmarkResultV4:
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

        metrics = NMEvaluationMetrics()

        return BenchmarkResultV4(question, method_name, failed_output, metrics)

    def _save_intermediate_results(self):
        """保存中间结果"""
        filepath = self.output_dir / "intermediate_results_v4.json"

        data = {}
        for method, results in self.results.items():
            data[method] = [r.to_dict() for r in results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_final_results(self):
        """保存最终结果"""

        detailed_file = self.output_dir / "detailed_results_v4.json"
        data = {}
        for method, results in self.results.items():
            data[method] = [r.to_dict() for r in results]

        with open(detailed_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"  ✓ Detailed results: {detailed_file}")

        summary = self._generate_summary()

        summary_file = self.output_dir / "summary_v4.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  ✓ Summary: {summary_file}")

    def _generate_summary(self) -> Dict:
        """生成聚合统计"""

        summary = {}

        # 🔬 NM核心能力指标
        nm_metrics = [
            'planning_quality',
            'reasoning_capability',
            'cot_quality',
            'reflection_capability',
            'nlu_capability',
        ]

        # 传统指标
        traditional_metrics = [
            'entity_f1',
            'factual_accuracy',
            'scientific_rigor',
            'modality_coverage',
        ]

        # 综合指标
        overall_metrics = [
            'nm_capability_score',
            'overall_score',
            'biological_insight_score',
            'execution_time',
        ]

        all_metric_names = nm_metrics + traditional_metrics + overall_metrics

        for method, results in self.results.items():
            if not results:
                continue

            summary[method] = {}

            for metric_name in all_metric_names:
                values = []

                for result in results:
                    value = result.to_dict()['metrics'].get(metric_name)
                    if value is not None:
                        values.append(value)

                if values:
                    import statistics
                    summary[method][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                    }

        return summary

    def print_summary(self):
        """打印汇总统计"""

        summary = self._generate_summary()

        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY (v4.0 - Nature Methods)")
        print("=" * 80)

        for method in ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']:
            if method not in summary:
                continue

            print(f"\n{method}:")
            print("-" * 40)

            # NM能力总分
            nm_score = summary[method].get('nm_capability_score', {})
            print(f"NM Capability: {nm_score.get('mean', 0):.3f} ± {nm_score.get('std', 0):.3f}")

            # Overall
            overall = summary[method].get('overall_score', {})
            print(f"Overall Score: {overall.get('mean', 0):.3f} ± {overall.get('std', 0):.3f}")

            print(f"\n🔬 NM Core Capabilities:")
            for metric in ['planning_quality', 'reasoning_capability', 'cot_quality', 'reflection_capability',
                           'nlu_capability']:
                if metric in summary[method]:
                    m = summary[method][metric]
                    print(f"  {metric:25s}: {m['mean']:.3f} ± {m['std']:.3f}")

            print(f"\n📊 Traditional Metrics:")
            for metric in ['entity_f1', 'factual_accuracy', 'scientific_rigor']:
                if metric in summary[method]:
                    m = summary[method][metric]
                    print(f"  {metric:25s}: {m['mean']:.3f} ± {m['std']:.3f}")

        print("\n" + "=" * 80)


# ==================== Export ====================

__all__ = ['BenchmarkRunnerV4', 'BenchmarkResultV4']

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("BenchmarkRunner v4.0 (Nature Methods) loaded successfully!")