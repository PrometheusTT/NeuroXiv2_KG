"""
Visualization for AIPOM-CoT Benchmark (v4.0 - Nature Methods)
==============================================================
Publication-quality figures for NM submission

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 4.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


class BenchmarkVisualizerV4:
    """Benchmark可视化生成器 (v4.0 - NM)"""

    def __init__(self, results_dir: str = "./benchmark_results_v4"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures_nm"
        self.figures_dir.mkdir(exist_ok=True)

        results_file = self.results_dir / "detailed_results_v4.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")

        with open(results_file, 'r') as f:
            self.results = json.load(f)

        summary_file = self.results_dir / "summary_v4.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}

        logger.info(f"✅ Loaded results from {results_file}")

    def generate_all_figures(self):
        """生成所有图表"""

        logger.info("\n" + "=" * 80)
        logger.info("🎨 GENERATING NATURE METHODS FIGURES")
        logger.info("=" * 80)

        # 主要图表
        self.generate_nm_figure1_capabilities()
        self.generate_nm_figure2_planning_analysis()
        self.generate_nm_figure3_reasoning_quality()
        self.generate_nm_figure4_cot_reflection()
        self.generate_nm_figure5_comprehensive()

        logger.info(f"\n✅ All figures saved to: {self.figures_dir}")

    def generate_nm_figure1_capabilities(self):
        """NM Figure 1: 5个核心能力雷达图"""

        logger.info("\n📊 NM Figure 1: Core Capabilities Radar...")

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']
        colors = {
            'AIPOM-CoT': '#2ecc71',
            'Direct GPT-4o': '#9b59b6',
            'Template-KG': '#e67e22',
            'RAG': '#e74c3c',
            'ReAct': '#3498db',
        }

        dimensions = [
            'Planning\nQuality',
            'Reasoning\nCapability',
            'CoT\nQuality',
            'Reflection\nCapability',
            'NLU\nCapability'
        ]

        scores = {}

        for method in methods:
            if method not in self.summary:
                continue

            s = self.summary[method]

            method_scores = []

            # 对于每个维度，如果有数据则使用，否则使用0
            planning = s.get('planning_quality', {}).get('mean', 0)
            method_scores.append(planning if planning is not None else 0)

            reasoning = s.get('reasoning_capability', {}).get('mean', 0)
            method_scores.append(reasoning if reasoning is not None else 0)

            cot = s.get('cot_quality', {}).get('mean', 0)
            method_scores.append(cot if cot is not None else 0)

            reflection = s.get('reflection_capability', {}).get('mean', 0)
            method_scores.append(reflection if reflection is not None else 0)

            nlu = s.get('nlu_capability', {}).get('mean', 0)
            method_scores.append(nlu if nlu is not None else 0)

            scores[method] = method_scores

        # 绘制
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]

        for method in methods:
            if method not in scores:
                continue

            values = scores[method]
            values += values[:1]

            linewidth = 3.5 if method == 'AIPOM-CoT' else 2
            markersize = 12 if method == 'AIPOM-CoT' else 7

            ax.plot(angles, values, 'o-', linewidth=linewidth,
                    label=method, color=colors.get(method, 'gray'),
                    markersize=markersize)

            alpha = 0.25 if method == 'AIPOM-CoT' else 0.05
            ax.fill(angles, values, alpha=alpha, color=colors.get(method, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
        ax.set_title('AIPOM-CoT: Core AI Capabilities\n(Nature Methods Submission)',
                     size=16, weight='bold', pad=25)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "nm_fig1_capabilities.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "nm_fig1_capabilities.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ NM Figure 1 saved")

    def generate_nm_figure2_planning_analysis(self):
        """NM Figure 2: Planning Quality深度分析"""

        logger.info("\n📊 NM Figure 2: Planning Quality Analysis...")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        methods = ['AIPOM-CoT', 'ReAct']  # 只有这两个方法有planning
        colors = {'AIPOM-CoT': '#2ecc71', 'ReAct': '#3498db'}

        planning_metrics = ['planning_coherence', 'planning_optimality', 'planning_adaptability']
        metric_labels = ['Coherence', 'Optimality', 'Adaptability']

        for idx, (metric, label) in enumerate(zip(planning_metrics, metric_labels)):
            ax = axes[idx]

            values = []
            errors = []

            for method in methods:
                if method not in self.summary:
                    continue

                m = self.summary[method].get(metric, {})
                values.append(m.get('mean', 0))
                errors.append(m.get('std', 0))

            x = np.arange(len(methods))
            bars = ax.bar(x, values, yerr=errors, capsize=5,
                          color=[colors[m] for m in methods], alpha=0.8)

            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{label}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('Planning Quality: Detailed Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "nm_fig2_planning.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "nm_fig2_planning.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ NM Figure 2 saved")

    def generate_nm_figure3_reasoning_quality(self):
        """NM Figure 3: Reasoning & CoT质量对比"""

        logger.info("\n📊 NM Figure 3: Reasoning & CoT Quality...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']
        colors = {
            'AIPOM-CoT': '#2ecc71',
            'Direct GPT-4o': '#9b59b6',
            'Template-KG': '#e67e22',
            'RAG': '#e74c3c',
            'ReAct': '#3498db',
        }

        # (0,0): Reasoning Components
        ax = axes[0, 0]
        reasoning_components = ['logical_consistency', 'evidence_integration', 'multi_hop_depth_score']
        component_labels = ['Logical\nConsistency', 'Evidence\nIntegration', 'Multi-hop\nDepth']

        x = np.arange(len(component_labels))
        width = 0.15

        for i, method in enumerate(methods):
            if method not in self.summary:
                continue

            values = []
            for comp in reasoning_components:
                val = self.summary[method].get(comp, {}).get('mean', 0)
                values.append(val if val is not None else 0)

            offset = (i - 2) * width
            ax.bar(x + offset, values, width, label=method,
                   color=colors[method], alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Reasoning Components', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(component_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # (0,1): CoT Quality (只有AIPOM和ReAct)
        ax = axes[0, 1]
        cot_methods = ['AIPOM-CoT', 'ReAct']
        cot_components = ['cot_clarity', 'cot_completeness', 'intermediate_steps_quality']
        cot_labels = ['Clarity', 'Completeness', 'Steps Quality']

        x = np.arange(len(cot_labels))
        width = 0.3

        for i, method in enumerate(cot_methods):
            if method not in self.summary:
                continue

            values = []
            for comp in cot_components:
                val = self.summary[method].get(comp, {}).get('mean', 0)
                values.append(val if val is not None else 0)

            offset = (i - 0.5) * width
            ax.bar(x + offset, values, width, label=method,
                   color=colors[method], alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Chain-of-Thought Quality', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cot_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # (1,0): Overall Reasoning vs CoT
        ax = axes[1, 0]

        reasoning_vals = []
        cot_vals = []

        for method in cot_methods:
            if method not in self.summary:
                continue

            reasoning_vals.append(self.summary[method].get('reasoning_capability', {}).get('mean', 0))
            cot_vals.append(self.summary[method].get('cot_quality', {}).get('mean', 0))

        x = np.arange(len(cot_methods))
        width = 0.35

        ax.bar(x - width/2, reasoning_vals, width, label='Reasoning',
               color='#3498db', alpha=0.8)
        ax.bar(x + width/2, cot_vals, width, label='CoT Quality',
               color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Reasoning vs CoT Quality', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cot_methods, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, (r, c) in enumerate(zip(reasoning_vals, cot_vals)):
            ax.text(i - width/2, r + 0.02, f'{r:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, c + 0.02, f'{c:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # (1,1): Reasoning Capability Overall Comparison
        ax = axes[1, 1]

        # 🔧 修复：只使用有数据的方法
        available_methods = []
        values = []
        errors = []

        for method in methods:
            if method not in self.summary:
                continue

            m = self.summary[method].get('reasoning_capability', {})
            if m.get('mean') is not None:
                available_methods.append(method)
                values.append(m.get('mean', 0))
                errors.append(m.get('std', 0))

        if not available_methods:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            x = np.arange(len(available_methods))
            bars = ax.bar(x, values, yerr=errors, capsize=5,
                          color=[colors[m] for m in available_methods], alpha=0.8)

        if not available_methods:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            x = np.arange(len(available_methods))
            bars = ax.bar(x, values, yerr=errors, capsize=5,
                          color=[colors[m] for m in available_methods], alpha=0.8)

            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Overall Reasoning Capability', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(available_methods, fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('Reasoning & Chain-of-Thought Analysis', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "nm_fig3_reasoning_cot.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "nm_fig3_reasoning_cot.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ NM Figure 3 saved")

    def generate_nm_figure4_cot_reflection(self):
        """NM Figure 4: Reflection能力分析"""

        logger.info("\n📊 NM Figure 4: Reflection Capability...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        methods = ['AIPOM-CoT', 'ReAct']
        colors = {'AIPOM-CoT': '#2ecc71', 'ReAct': '#3498db'}

        # Left: Reflection Components
        ax = axes[0]

        reflection_components = ['error_detection', 'self_correction', 'iterative_refinement']
        component_labels = ['Error\nDetection', 'Self\nCorrection', 'Iterative\nRefinement']

        x = np.arange(len(component_labels))
        width = 0.35

        for i, method in enumerate(methods):
            if method not in self.summary:
                continue

            values = []
            for comp in reflection_components:
                val = self.summary[method].get(comp, {}).get('mean', 0)
                values.append(val if val is not None else 0)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method,
                          color=colors[method], alpha=0.8)

            # 添加数值
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Reflection Components', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(component_labels, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Right: Overall Reflection Capability
        ax = axes[1]

        values = []
        errors = []

        for method in methods:
            if method not in self.summary:
                continue

            m = self.summary[method].get('reflection_capability', {})
            values.append(m.get('mean', 0))
            errors.append(m.get('std', 0))

        x = np.arange(len(methods))
        bars = ax.bar(x, values, yerr=errors, capsize=5,
                      color=[colors[m] for m in methods], alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Reflection Capability', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.suptitle('Reflection & Self-Correction Analysis', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "nm_fig4_reflection.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "nm_fig4_reflection.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ NM Figure 4 saved")

    def generate_nm_figure5_comprehensive(self):
        """NM Figure 5: 综合对比（Overall + NM Score + Bio Insight）"""

        logger.info("\n📊 NM Figure 5: Comprehensive Comparison...")

        fig, ax = plt.subplots(figsize=(14, 7))

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']
        colors = {
            'AIPOM-CoT': '#2ecc71',
            'Direct GPT-4o': '#9b59b6',
            'Template-KG': '#e67e22',
            'RAG': '#e74c3c',
            'ReAct': '#3498db',
        }

        # 3个综合指标
        composite_metrics = ['nm_capability_score', 'overall_score', 'biological_insight_score']
        metric_labels = ['NM Capability\nScore', 'Overall\nScore', 'Biological\nInsight']

        x = np.arange(len(metric_labels))
        width = 0.15

        # 🔧 修复：只绘制有数据的方法
        available_methods = [m for m in methods if m in self.summary]

        for i, method in enumerate(available_methods):
            values = []
            for metric in composite_metrics:
                val = self.summary[method].get(metric, {}).get('mean', 0)
                values.append(val if val is not None else 0)

            offset = (i - len(available_methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method,
                          color=colors[method], alpha=0.85)

            # 添加数值标签
            for j, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.2f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Comprehensive Performance Comparison\n(Nature Methods Benchmark)',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加水平参考线
        ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(len(metric_labels) - 0.3, 0.82, 'Excellent (0.8)', fontsize=10, color='green')

        ax.axhline(y=0.6, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(len(metric_labels) - 0.3, 0.62, 'Good (0.6)', fontsize=10, color='orange')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "nm_fig5_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "nm_fig5_comprehensive.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ NM Figure 5 saved")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        visualizer = BenchmarkVisualizerV4("./benchmark_results_v4")
        visualizer.generate_all_figures()

        print("\n✅ All NM figures generated successfully!")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run benchmark first: python run_benchmark_v4.py")