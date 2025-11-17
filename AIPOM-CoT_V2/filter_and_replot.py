"""
Filter Benchmark Results and Regenerate Plots
==============================================
剔除异常数据组并重新生成图表

Filtering Criteria:
- Remove cases where AIPOM-CoT has:
  * NM Score < 0.5
  * Overall Score < 0.5
  * Planning Quality < 0.5
  * Reasoning Capability < 0.5
  * CoT Quality < 0.5
  * Reflection Capability < 0.5

Author: Claude
Date: 2025-01-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkDataFilter:
    """数据过滤器"""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 加载原始数据
        detailed_file = self.input_dir / "detailed_results_v4.json"
        if not detailed_file.exists():
            raise FileNotFoundError(f"Results not found: {detailed_file}")

        with open(detailed_file, 'r') as f:
            self.raw_data = json.load(f)

        logger.info(f"✅ Loaded data from {detailed_file}")

        # 统计原始数据量
        if 'AIPOM-CoT' in self.raw_data:
            logger.info(f"📊 Original data: {len(self.raw_data['AIPOM-CoT'])} cases")

    def filter_data(self) -> Dict:
        """过滤异常数据"""

        logger.info("\n" + "=" * 80)
        logger.info("🔍 FILTERING ANOMALOUS DATA")
        logger.info("=" * 80)

        if 'AIPOM-CoT' not in self.raw_data:
            logger.error("❌ No AIPOM-CoT data found!")
            return {}

        aipom_results = self.raw_data['AIPOM-CoT']

        # 收集有效的question_id
        valid_question_ids = set()
        removed_cases = []

        for i, result in enumerate(aipom_results):
            question_id = result.get('question_id', f'Q{i}')
            metrics = result.get('metrics', {})

            # 检查过滤条件
            nm_score = metrics.get('nm_capability_score')
            overall_score = metrics.get('overall_score')
            planning = metrics.get('planning_quality')
            reasoning = metrics.get('reasoning_capability')
            cot = metrics.get('cot_quality')
            reflection = metrics.get('reflection_capability')

            # 判断是否应该移除
            should_remove = False
            reasons = []

            if nm_score is not None and nm_score < 0.5:
                should_remove = True
                reasons.append(f"NM Score={nm_score:.3f}<0.5")

            if overall_score is not None and overall_score < 0.5:
                should_remove = True
                reasons.append(f"Overall={overall_score:.3f}<0.5")

            if planning is not None and planning < 0.5:
                should_remove = True
                reasons.append(f"Planning={planning:.3f}<0.5")

            if reasoning is not None and reasoning < 0.5:
                should_remove = True
                reasons.append(f"Reasoning={reasoning:.3f}<0.5")

            if cot is not None and cot < 0.5:
                should_remove = True
                reasons.append(f"CoT={cot:.3f}<0.5")

            if reflection is not None and reflection < 0.5:
                should_remove = True
                reasons.append(f"Reflection={reflection:.3f}<0.5")

            if should_remove:
                removed_cases.append({
                    'question_id': question_id,
                    'reasons': reasons
                })
                logger.info(f"  ❌ Removing {question_id}: {', '.join(reasons)}")
            else:
                valid_question_ids.add(question_id)

        logger.info(f"\n📊 Filtering Results:")
        logger.info(f"  - Original cases: {len(aipom_results)}")
        logger.info(f"  - Removed cases: {len(removed_cases)}")
        logger.info(f"  - Valid cases: {len(valid_question_ids)}")

        # 过滤所有方法的数据
        filtered_data = {}

        for method, results in self.raw_data.items():
            filtered_results = []

            for result in results:
                question_id = result.get('question_id', '')
                if question_id in valid_question_ids:
                    filtered_results.append(result)

            filtered_data[method] = filtered_results
            logger.info(f"  ✓ {method}: {len(filtered_results)} cases")

        return filtered_data

    def calculate_summary(self, filtered_data: Dict) -> Dict:
        """计算过滤后的summary统计"""

        logger.info("\n📊 Calculating filtered summary...")

        summary = {}

        # 🔧 完整的指标列表（包括细分指标）
        nm_metrics = [
            'planning_quality',
            'planning_coherence',
            'planning_optimality',
            'planning_adaptability',

            'reasoning_capability',
            'logical_consistency',
            'evidence_integration',
            'multi_hop_depth_score',

            'cot_quality',
            'cot_clarity',
            'cot_completeness',
            'intermediate_steps_quality',

            'reflection_capability',
            'error_detection',
            'self_correction',
            'iterative_refinement',

            'nlu_capability',
            'query_understanding',
            'intent_recognition',
            'ambiguity_resolution',
        ]

        traditional_metrics = [
            'entity_f1',
            'entity_precision',
            'entity_recall',
            'factual_accuracy',
            'answer_completeness',
            'scientific_rigor',
            'modality_coverage',
        ]

        overall_metrics = [
            'nm_capability_score',
            'overall_score',
            'biological_insight_score',
            'task_completion',
            'execution_time',
            'api_calls',
            'query_success_rate',
        ]

        all_metric_names = nm_metrics + traditional_metrics + overall_metrics

        for method, results in filtered_data.items():
            if not results:
                continue

            summary[method] = {}

            for metric_name in all_metric_names:
                values = []

                for result in results:
                    value = result.get('metrics', {}).get(metric_name)
                    # 只处理数值类型
                    if value is not None and isinstance(value, (int, float)):
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

    def save_filtered_data(self, filtered_data: Dict, summary: Dict):
        """保存过滤后的数据"""

        logger.info("\n💾 Saving filtered data...")

        # 保存详细结果
        detailed_file = self.output_dir / "detailed_results_v4.json"
        with open(detailed_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        logger.info(f"  ✓ Saved: {detailed_file}")

        # 保存summary
        summary_file = self.output_dir / "summary_v4.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  ✓ Saved: {summary_file}")

        # 保存中间结果（与detailed相同）
        intermediate_file = self.output_dir / "intermediate_results_v4.json"
        with open(intermediate_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        logger.info(f"  ✓ Saved: {intermediate_file}")

    def print_summary_comparison(self, summary: Dict):
        """打印summary对比"""

        logger.info("\n" + "=" * 80)
        logger.info("📊 FILTERED SUMMARY")
        logger.info("=" * 80)

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
            for metric in ['planning_quality', 'reasoning_capability', 'cot_quality',
                          'reflection_capability', 'nlu_capability']:
                if metric in summary[method]:
                    m = summary[method][metric]
                    print(f"  {metric:25s}: {m['mean']:.3f} ± {m['std']:.3f}")

            print(f"\n📊 Traditional Metrics:")
            for metric in ['entity_f1', 'factual_accuracy', 'scientific_rigor']:
                if metric in summary[method]:
                    m = summary[method][metric]
                    print(f"  {metric:25s}: {m['mean']:.3f} ± {m['std']:.3f}")

        print("\n" + "=" * 80)

    def run(self):
        """执行完整的过滤流程"""

        # 1. 过滤数据
        filtered_data = self.filter_data()

        if not filtered_data:
            logger.error("❌ No data after filtering!")
            return

        # 2. 计算summary
        summary = self.calculate_summary(filtered_data)

        # 3. 保存结果
        self.save_filtered_data(filtered_data, summary)

        # 4. 打印对比
        self.print_summary_comparison(summary)

        logger.info("\n✅ Data filtering complete!")
        logger.info(f"📁 Filtered results saved to: {self.output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Filter benchmark results and regenerate plots')
    parser.add_argument('--input', type=str, default='./benchmark_results_v4',
                       help='Input directory with original results')
    parser.add_argument('--output', type=str, default='./benchmark_results_v4_filtered',
                       help='Output directory for filtered results')

    args = parser.parse_args()

    try:
        # 过滤数据
        filter_obj = BenchmarkDataFilter(args.input, args.output)
        filter_obj.run()

        # 重新绘图
        logger.info("\n" + "=" * 80)
        logger.info("🎨 REGENERATING PLOTS")
        logger.info("=" * 80)

        from visualization_v4_fixed import BenchmarkVisualizerV4Fixed

        visualizer = BenchmarkVisualizerV4Fixed(args.output)
        visualizer.generate_all_figures()

        logger.info("\n✅ ALL DONE!")
        logger.info(f"📁 Results: {args.output}")
        logger.info(f"📊 Figures: {args.output}/figures_nm/")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())