"""
Statistical Analysis for AIPOM-CoT Benchmark
=============================================
统计显著性检验和效应量计算

Author: Claude & PrometheusTT
Date: 2025-01-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)


# ==================== Statistical Analyzer ====================

class StatisticalAnalyzer:
    """统计分析器"""

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "detailed_results.json"

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'r') as f:
            self.raw_results = json.load(f)

        logger.info(f"✅ Loaded results from {self.results_file}")

    def run_full_analysis(self) -> pd.DataFrame:
        """运行完整统计分析"""

        logger.info("\n" + "=" * 80)
        logger.info("📊 STATISTICAL ANALYSIS")
        logger.info("=" * 80)

        # 提取关键指标
        metrics_data = self._extract_metrics()

        # 运行对比检验
        comparisons = []

        aipom_scores = metrics_data.get('AIPOM-CoT', {})

        for method in ['Direct LLM', 'RAG', 'ReAct']:
            if method not in metrics_data:
                continue

            method_scores = metrics_data[method]

            # 对比每个指标
            for metric_name in ['entity_f1', 'depth_matching', 'modality_coverage', 'scientific_rigor']:
                if metric_name in aipom_scores and metric_name in method_scores:
                    comparison = self.compare_methods(
                        aipom_scores[metric_name],
                        method_scores[metric_name],
                        'AIPOM-CoT',
                        method,
                        metric_name
                    )
                    comparisons.append(comparison)

        # 转为DataFrame
        df = pd.DataFrame(comparisons)

        # 保存
        output_file = self.results_dir / "statistical_analysis.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"\n✅ Statistical analysis saved to: {output_file}")

        # 打印关键结果
        self._print_summary(df)

        return df

    def _extract_metrics(self) -> Dict[str, Dict[str, List[float]]]:
        """提取指标数据"""

        metrics_data = {}

        for method, results in self.raw_results.items():
            metrics_data[method] = {
                'entity_f1': [],
                'depth_matching': [],
                'modality_coverage': [],
                'scientific_rigor': [],
                'closed_loop': [],
                'execution_time': [],
            }

            for result in results:
                metrics = result.get('metrics', {})

                metrics_data[method]['entity_f1'].append(metrics.get('entity_f1', 0.0))
                metrics_data[method]['depth_matching'].append(metrics.get('depth_matching', 0.0))
                metrics_data[method]['modality_coverage'].append(metrics.get('modality_coverage', 0.0))
                metrics_data[method]['scientific_rigor'].append(metrics.get('scientific_rigor', 0.0))
                metrics_data[method]['closed_loop'].append(metrics.get('closed_loop', 0.0))
                metrics_data[method]['execution_time'].append(metrics.get('execution_time', 0.0))

        return metrics_data

    def compare_methods(self,
                        scores_a: List[float],
                        scores_b: List[float],
                        method_a: str,
                        method_b: str,
                        metric_name: str) -> Dict:
        """
        比较两个方法

        Returns:
            comparison dict with t-test, Cohen's d, CI, etc.
        """

        # 确保长度一致
        n = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        # 基本统计
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)

        # T-test (paired if same questions)
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # 95% CI for difference
        diff = np.array(scores_a) - np.array(scores_b)
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_95_lower = np.mean(diff) - 1.96 * se
        ci_95_upper = np.mean(diff) + 1.96 * se

        # Significance
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'

        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size_label = 'negligible'
        elif abs_d < 0.5:
            effect_size_label = 'small'
        elif abs_d < 0.8:
            effect_size_label = 'medium'
        else:
            effect_size_label = 'large'

        return {
            'metric': metric_name,
            'method_a': method_a,
            'method_b': method_b,
            'mean_a': mean_a,
            'std_a': std_a,
            'mean_b': mean_b,
            'std_b': std_b,
            'mean_diff': mean_a - mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance,
            'cohens_d': cohens_d,
            'effect_size': effect_size_label,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'n': n,
        }

    def _print_summary(self, df: pd.DataFrame):
        """打印统计摘要"""

        print("\n" + "=" * 80)
        print("Key Findings:")
        print("=" * 80)

        # 按metric分组
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]

            print(f"\n{metric.upper()}:")
            print("-" * 40)

            for _, row in metric_df.iterrows():
                mean_a = row['mean_a']
                mean_b = row['mean_b']
                p_value = row['p_value']
                cohens_d = row['cohens_d']
                sig = row['significance']

                improvement = (mean_a - mean_b) / mean_b * 100 if mean_b > 0 else 0

                print(f"{row['method_b']:15s}: {mean_a:.3f} vs {mean_b:.3f} "
                      f"(+{improvement:+.1f}%, p={p_value:.4f}{sig}, d={cohens_d:.2f})")

        print("\n" + "=" * 80)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")
        print("Effect size: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)")
        print("=" * 80)

    def generate_latex_table(self) -> str:
        """生成LaTeX表格"""

        metrics_data = self._extract_metrics()

        methods = ['AIPOM-CoT', 'Direct LLM', 'RAG', 'ReAct']
        metric_names = ['Entity F1', 'Depth Match', 'Modal Cov', 'Sci Rigor']
        metric_keys = ['entity_f1', 'depth_matching', 'modality_coverage', 'scientific_rigor']

        latex = []
        latex.append(r"\begin{table}[h]")
        latex.append(r"\centering")
        latex.append(r"\caption{Performance Comparison Across Methods}")
        latex.append(r"\begin{tabular}{lcccc}")
        latex.append(r"\hline")
        latex.append(f"Method & {' & '.join(metric_names)} \\\\")
        latex.append(r"\hline")

        for method in methods:
            if method not in metrics_data:
                continue

            row_data = [method]

            for key in metric_keys:
                scores = metrics_data[method].get(key, [])
                if scores:
                    mean = np.mean(scores)
                    std = np.std(scores, ddof=1)

                    if method == 'AIPOM-CoT':
                        cell = f"\\textbf{{{mean:.3f}}} $\\pm$ {std:.3f}"
                    else:
                        cell = f"{mean:.3f} $\\pm$ {std:.3f}"
                else:
                    cell = "N/A"

                row_data.append(cell)

            latex.append(" & ".join(row_data) + " \\\\")

        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        latex_str = "\n".join(latex)

        # 保存
        output_file = self.results_dir / "table.tex"
        output_file.write_text(latex_str)

        logger.info(f"✅ LaTeX table saved to: {output_file}")

        return latex_str


# ==================== Test ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # 假设results已经存在
    try:
        analyzer = StatisticalAnalyzer("./benchmark_results")
        df = analyzer.run_full_analysis()
        latex = analyzer.generate_latex_table()

        print("\n✅ Statistical analysis complete!")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run benchmark first: python run_benchmark.py")