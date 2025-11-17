"""
Enhanced Visualization for AIPOM-CoT Benchmark
==============================================
Publication-quality figures with diverse chart types

改进点:
1. 多样化的图表类型（不只是柱状图）
2. 更专业的配色方案
3. 去除NM/Nature Methods相关表述
4. 重命名核心指标

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 5.0 (Enhanced)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# 设置专业配色
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


class BenchmarkVisualizerEnhanced:
    """增强版可视化生成器"""

    def __init__(self, results_dir: str = "./benchmark_results_v5"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures_enhanced"
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

        # 专业配色方案
        self.colors = {
            'AIPOM-CoT': '#2E86AB',      # Professional blue
            'Direct GPT-4o': '#A23B72',   # Deep purple
            'Template-KG': '#F18F01',     # Warm orange
            'RAG': '#C73E1D',             # Red-orange
            'ReAct': '#6A994E',           # Nature green
        }

        # 辅助色
        self.accent_colors = {
            'excellent': '#2E7D32',
            'good': '#F57C00',
            'moderate': '#FBC02D',
        }

        logger.info(f"✅ Loaded results from {results_file}")

    def generate_all_figures(self):
        """生成所有增强版图表"""

        logger.info("\n" + "=" * 80)
        logger.info("🎨 GENERATING ENHANCED FIGURES")
        logger.info("=" * 80)

        self.generate_fig1_capabilities_radar()
        self.generate_fig1b_baseline_comparison()
        self.generate_fig2_planning_violin()
        self.generate_fig3_reasoning_heatmap()
        self.generate_fig4_reflection_polar()
        self.generate_fig5_comprehensive_scatter()

        logger.info(f"\n✅ All enhanced figures saved to: {self.figures_dir}")

    def generate_fig1_capabilities_radar(self):
        """Figure 1: 核心能力雷达图（只显示有完整数据的方法）"""

        logger.info("\n📊 Figure 1: Core Capabilities Radar...")

        # 只显示有完整5维度数据的方法
        methods_to_show = ['AIPOM-CoT', 'ReAct']

        dimensions = [
            'Planning\nQuality',
            'Reasoning\nCapability',
            'Chain-of-Thought\nQuality',
            'Reflection\nCapability',
            'Language\nUnderstanding'
        ]

        scores = {}

        for method in methods_to_show:
            if method not in self.summary:
                continue

            s = self.summary[method]

            # 检查是否所有5个维度都有数据
            planning = s.get('planning_quality', {}).get('mean')
            reasoning = s.get('reasoning_capability', {}).get('mean')
            cot = s.get('cot_quality', {}).get('mean')
            reflection = s.get('reflection_capability', {}).get('mean')
            nlu = s.get('nlu_capability', {}).get('mean')

            # 只有当所有维度都有数据时才显示
            if all(v is not None for v in [planning, reasoning, cot, reflection, nlu]):
                method_scores = [planning, reasoning, cot, reflection, nlu]
                scores[method] = method_scores

        # 绘制
        fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]

        # 添加参考圆圈
        for radius in [0.6, 0.8]:
            circle_angles = np.linspace(0, 2 * np.pi, 100)
            circle_r = [radius] * 100
            ax.plot(circle_angles, circle_r,
                   linestyle=':', color='gray', alpha=0.3, linewidth=1.5)

        for method in methods_to_show:
            if method not in scores:
                continue

            values = scores[method]
            values += values[:1]

            linewidth = 4.0 if method == 'AIPOM-CoT' else 3.0
            markersize = 14 if method == 'AIPOM-CoT' else 10
            alpha_fill = 0.25 if method == 'AIPOM-CoT' else 0.15

            ax.plot(angles, values, 'o-', linewidth=linewidth,
                    label=method, color=self.colors.get(method, 'gray'),
                    markersize=markersize, alpha=0.95, zorder=3)

            ax.fill(angles, values, alpha=alpha_fill,
                   color=self.colors.get(method, 'gray'), zorder=2)

            # 添加数值标签
            for angle, value in zip(angles[:-1], values[:-1]):
                ax.text(angle, value + 0.08, f'{value:.2f}',
                       ha='center', va='center', fontsize=9,
                       fontweight='bold', color=self.colors.get(method, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)
        ax.set_title('Core Agentic Capabilities: AIPOM-CoT vs ReAct',
                     size=17, weight='bold', pad=30)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加说明文字
        note_text = ("Note: Other baseline methods (Direct GPT-4o, Template-KG, RAG)\n"
                    "lack planning and reflection capabilities, thus excluded from this comparison")
        plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10,
                   style='italic', color='gray', wrap=True)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig1_capabilities_radar.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig1_capabilities_radar.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 1 saved (with complete data only)")

    def generate_fig1b_baseline_comparison(self):
        """Figure 1b: Baseline方法对比（棒棒糖图）"""

        logger.info("\n📊 Figure 1b: Baseline Methods Comparison...")

        fig, ax = plt.subplots(figsize=(12, 8))

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        # 只选择所有方法都有的维度
        common_dimensions = ['reasoning_capability', 'nlu_capability']
        dimension_labels = ['Reasoning\nCapability', 'Language\nUnderstanding']

        y_positions = np.arange(len(methods))
        bar_height = 0.35

        for i, (dim, label) in enumerate(zip(common_dimensions, dimension_labels)):
            values = []
            errors = []

            for method in methods:
                if method not in self.summary:
                    values.append(0)
                    errors.append(0)
                    continue

                m = self.summary[method].get(dim, {})
                mean_val = m.get('mean')
                if mean_val is not None:
                    values.append(mean_val)
                    errors.append(m.get('std', 0))
                else:
                    values.append(0)
                    errors.append(0)

            offset = (i - 0.5) * bar_height

            # 绘制水平棒棒糖图
            for j, (method, val, err) in enumerate(zip(methods, values, errors)):
                y_pos = y_positions[j] + offset

                # 绘制线条
                ax.plot([0, val], [y_pos, y_pos],
                       color=self.colors[method], linewidth=3, alpha=0.7)

                # 绘制圆点
                ax.scatter(val, y_pos, s=150,
                          color=self.colors[method],
                          edgecolors='white', linewidth=2, zorder=3)

                # 添加数值标签
                if val > 0.05:
                    ax.text(val + 0.03, y_pos, f'{val:.2f}',
                           va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(methods, fontsize=12, fontweight='bold')
        ax.set_xlabel('Score', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.set_title('Baseline Methods: Reasoning & Language Understanding',
                     fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # 参考区域
        ax.axvspan(0.8, 1.0, alpha=0.1, color=self.accent_colors['excellent'])
        ax.text(0.9, len(methods)-0.5, 'Excellent',
               fontsize=10, color=self.accent_colors['excellent'],
               ha='center', fontweight='bold')

        # 图例
        legend_elements = [
            mpatches.Patch(color=self.colors['AIPOM-CoT'], label='Reasoning'),
            mpatches.Patch(color=self.colors['AIPOM-CoT'], alpha=0.5, label='Language Understanding')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig1b_baseline_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig1b_baseline_comparison.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 1b saved")

    def generate_fig2_planning_violin(self):
        """Figure 2: Planning质量分析 - 小提琴图+散点"""

        logger.info("\n📊 Figure 2: Planning Quality Analysis (Violin Plot)...")

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        methods = ['AIPOM-CoT', 'ReAct']
        planning_metrics = ['planning_coherence', 'planning_optimality', 'planning_adaptability']
        metric_labels = ['Coherence', 'Optimality', 'Adaptability']

        for idx, (metric, label) in enumerate(zip(planning_metrics, metric_labels)):
            ax = axes[idx]

            # 收集所有数据点
            data_points = {m: [] for m in methods}

            for method in methods:
                if method not in self.results:
                    continue

                for result in self.results[method]:
                    val = result.get('metrics', {}).get(metric)
                    if val is not None and isinstance(val, (int, float)):
                        data_points[method].append(val)

            # 绘制小提琴图
            positions = []
            data_list = []
            colors_list = []

            for i, method in enumerate(methods):
                if data_points[method]:
                    positions.append(i)
                    data_list.append(data_points[method])
                    colors_list.append(self.colors[method])

            if data_list:
                parts = ax.violinplot(data_list, positions=positions,
                                     widths=0.7, showmeans=True, showmedians=True)

                # 设置颜色
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_list[i])
                    pc.set_alpha(0.6)
                    pc.set_edgecolor(colors_list[i])
                    pc.set_linewidth(1.5)

                # 美化均值和中位数线
                parts['cmeans'].set_color('black')
                parts['cmeans'].set_linewidth(2)
                parts['cmedians'].set_color('white')
                parts['cmedians'].set_linewidth(1.5)

                # 添加散点
                for i, method in enumerate(methods):
                    if data_points[method]:
                        y_data = data_points[method]
                        x_data = np.random.normal(i, 0.04, size=len(y_data))
                        ax.scatter(x_data, y_data, alpha=0.4, s=30,
                                 color=self.colors[method], edgecolors='white', linewidth=0.5)

                ax.set_xticks(positions)
                ax.set_xticklabels(methods, fontsize=11)
                ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                ax.set_title(f'{label}', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                # 添加参考线
                ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                          linestyle=':', alpha=0.5, linewidth=1.5)
                ax.text(len(methods)-0.5, 0.82, 'Excellent',
                       fontsize=9, color=self.accent_colors['excellent'])

            else:
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes)

        plt.suptitle('Planning Quality: Multi-dimensional Analysis',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig2_planning_violin.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig2_planning_violin.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 2 saved")

    def generate_fig3_reasoning_heatmap(self):
        """Figure 3: Reasoning & CoT - 热力图+分组点图"""

        logger.info("\n📊 Figure 3: Reasoning & CoT Analysis (Heatmap + Dot Plot)...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        # (0,0): Reasoning Components - 热力图
        ax = fig.add_subplot(gs[0, 0])

        reasoning_components = ['logical_consistency', 'evidence_integration', 'multi_hop_depth_score']
        component_labels = ['Logical\nConsistency', 'Evidence\nIntegration', 'Multi-hop\nDepth']

        heatmap_data = []
        available_methods = []

        for method in methods:
            if method not in self.summary:
                continue

            row_data = []
            has_data = False
            for comp in reasoning_components:
                val = self.summary[method].get(comp, {}).get('mean')
                if val is not None:
                    row_data.append(val)
                    has_data = True
                else:
                    row_data.append(0)

            if has_data:
                heatmap_data.append(row_data)
                available_methods.append(method)

        if heatmap_data:
            heatmap_data = np.array(heatmap_data)

            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(component_labels)))
            ax.set_yticks(np.arange(len(available_methods)))
            ax.set_xticklabels(component_labels, fontsize=10)
            ax.set_yticklabels(available_methods, fontsize=10)

            # 添加数值标注
            for i in range(len(available_methods)):
                for j in range(len(component_labels)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                 ha="center", va="center", color="black" if heatmap_data[i, j] < 0.6 else "white",
                                 fontsize=10, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Score', rotation=270, labelpad=15, fontweight='bold')

        ax.set_title('Reasoning Components', fontsize=13, fontweight='bold')

        # (0,1): CoT Quality - 分组点图
        ax = fig.add_subplot(gs[0, 1])

        cot_methods = ['AIPOM-CoT', 'ReAct']
        cot_components = ['cot_clarity', 'cot_completeness', 'intermediate_steps_quality']
        cot_labels = ['Clarity', 'Completeness', 'Steps\nQuality']

        available_cot_methods = [m for m in cot_methods if m in self.summary]

        if available_cot_methods:
            x_positions = np.arange(len(cot_labels))
            width = 0.35

            for i, method in enumerate(available_cot_methods):
                values = []
                errors = []
                for comp in cot_components:
                    val = self.summary[method].get(comp, {}).get('mean')
                    std = self.summary[method].get(comp, {}).get('std', 0)
                    values.append(val if val is not None else 0)
                    errors.append(std)

                offset = (i - len(available_cot_methods)/2 + 0.5) * width

                # 绘制点和误差棒
                ax.errorbar(x_positions + offset, values, yerr=errors,
                           fmt='o', markersize=12, linewidth=2.5,
                           color=self.colors[method], label=method,
                           capsize=5, capthick=2, alpha=0.8)

            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(cot_labels, fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # 参考线
            ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                      linestyle=':', alpha=0.5, linewidth=1.5)

        ax.set_title('Chain-of-Thought Quality', fontsize=13, fontweight='bold')

        # (1,0): Reasoning vs CoT - 散点对比
        ax = fig.add_subplot(gs[1, 0])

        for method in cot_methods:
            if method not in self.summary:
                continue

            reasoning_val = self.summary[method].get('reasoning_capability', {}).get('mean')
            cot_val = self.summary[method].get('cot_quality', {}).get('mean')

            if reasoning_val is not None and cot_val is not None:
                ax.scatter(reasoning_val, cot_val, s=250,
                          color=self.colors[method], alpha=0.7,
                          edgecolors='white', linewidth=2,
                          label=method, marker='o')

                # 添加标签
                ax.annotate(method, (reasoning_val, cot_val),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Reasoning Capability', fontsize=12, fontweight='bold')
        ax.set_ylabel('CoT Quality', fontsize=12, fontweight='bold')
        ax.set_title('Reasoning vs CoT Quality', fontsize=13, fontweight='bold')
        ax.set_xlim(0.4, 0.9)
        ax.set_ylim(0.4, 0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        # (1,1): Overall Reasoning - 棒棒糖图
        ax = fig.add_subplot(gs[1, 1])

        available_reasoning_methods = []
        values = []
        errors = []

        for method in methods:
            if method not in self.summary:
                continue

            m = self.summary[method].get('reasoning_capability', {})
            mean_val = m.get('mean')

            if mean_val is not None:
                available_reasoning_methods.append(method)
                values.append(mean_val)
                errors.append(m.get('std', 0))

        if available_reasoning_methods:
            y_positions = np.arange(len(available_reasoning_methods))

            # 绘制棒棒糖图
            for i, (method, val, err) in enumerate(zip(available_reasoning_methods, values, errors)):
                ax.plot([0, val], [i, i], color=self.colors[method],
                       linewidth=3, alpha=0.7)
                ax.scatter(val, i, s=200, color=self.colors[method],
                          edgecolors='white', linewidth=2, zorder=3)

                # 添加数值标签
                ax.text(val + 0.02, i, f'{val:.3f}',
                       va='center', fontsize=10, fontweight='bold')

            ax.set_yticks(y_positions)
            ax.set_yticklabels(available_reasoning_methods, fontsize=11)
            ax.set_xlabel('Score', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # 参考区域
            ax.axvspan(0.8, 1.0, alpha=0.1, color=self.accent_colors['excellent'])
            ax.text(0.9, len(available_reasoning_methods)-0.5, 'Excellent',
                   fontsize=9, color=self.accent_colors['excellent'], ha='center')

        ax.set_title('Overall Reasoning Capability', fontsize=13, fontweight='bold')

        plt.suptitle('Reasoning & Chain-of-Thought Analysis',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.figures_dir / "fig3_reasoning_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig3_reasoning_heatmap.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 3 saved")

    def generate_fig4_reflection_polar(self):
        """Figure 4: Reflection - 极坐标图"""

        logger.info("\n📊 Figure 4: Reflection Capability (Polar Bar Chart)...")

        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, wspace=0.3)

        methods = ['AIPOM-CoT', 'ReAct']

        # Left: Reflection Components - 极坐标柱状图
        ax = fig.add_subplot(gs[0, 0], projection='polar')

        reflection_components = ['error_detection', 'self_correction', 'iterative_refinement']
        component_labels = ['Error\nDetection', 'Self\nCorrection', 'Iterative\nRefinement']

        available_methods = [m for m in methods if m in self.summary]

        if available_methods:
            N = len(component_labels)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

            width = 2 * np.pi / N / len(available_methods)

            for i, method in enumerate(available_methods):
                values = []
                for comp in reflection_components:
                    val = self.summary[method].get(comp, {}).get('mean')
                    values.append(val if val is not None else 0)

                offset = width * (i - len(available_methods)/2 + 0.5)
                bars = ax.bar([a + offset for a in angles], values, width=width,
                             label=method, alpha=0.8, color=self.colors[method],
                             edgecolor='white', linewidth=2)

                # 添加数值标签
                for angle, value, bar in zip(angles, values, bars):
                    rotation = np.rad2deg(angle + offset)
                    alignment = "left" if angle < np.pi else "right"
                    ax.text(angle + offset, value + 0.05, f'{value:.2f}',
                           ha=alignment, va='center', fontsize=9,
                           rotation=rotation, rotation_mode='anchor')

            ax.set_xticks(angles)
            ax.set_xticklabels(component_labels, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
            ax.grid(True, alpha=0.3)

        ax.set_title('Reflection Components', fontsize=14, fontweight='bold', pad=20)

        # Right: Overall Reflection - 水平误差棒图
        ax = fig.add_subplot(gs[0, 1])

        values = []
        errors = []
        available_methods_overall = []

        for method in methods:
            if method not in self.summary:
                continue

            m = self.summary[method].get('reflection_capability', {})
            mean_val = m.get('mean')

            if mean_val is not None:
                available_methods_overall.append(method)
                values.append(mean_val)
                errors.append(m.get('std', 0))

        if available_methods_overall:
            y_positions = np.arange(len(available_methods_overall))

            # 绘制水平误差棒
            for i, (method, val, err) in enumerate(zip(available_methods_overall, values, errors)):
                ax.barh(i, val, xerr=err, height=0.6,
                       color=self.colors[method], alpha=0.7,
                       error_kw={'linewidth': 2, 'elinewidth': 2, 'capsize': 5})

                # 添加数值标签
                ax.text(val + 0.05, i, f'{val:.3f}',
                       va='center', fontsize=11, fontweight='bold')

            ax.set_yticks(y_positions)
            ax.set_yticklabels(available_methods_overall, fontsize=12)
            ax.set_xlabel('Score', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # 参考区域
            ax.axvspan(0.8, 1.1, alpha=0.1, color=self.accent_colors['excellent'])
            ax.text(0.95, len(available_methods_overall)-0.5, 'Excellent',
                   fontsize=10, color=self.accent_colors['excellent'],
                   ha='center', fontweight='bold')

        ax.set_title('Overall Reflection Capability', fontsize=14, fontweight='bold')

        plt.suptitle('Reflection & Self-Correction Analysis',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.figures_dir / "fig4_reflection_polar.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig4_reflection_polar.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 4 saved")

    def generate_fig5_comprehensive_scatter(self):
        """Figure 5: 综合对比 - 分组散点图"""

        logger.info("\n📊 Figure 5: Comprehensive Comparison (Grouped Scatter)...")

        fig, ax = plt.subplots(figsize=(14, 8))

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        # 重命名指标
        composite_metrics_orig = ['nm_capability_score', 'overall_score', 'biological_insight_score']
        metric_labels = [
            'Agentic Capability\nIndex (ACI)',
            'Composite Performance\nScore (CPS)',
            'Domain Task\nScore (DTS)'
        ]

        x_positions = np.arange(len(metric_labels))

        # 为每个方法绘制散点
        for method in methods:
            if method not in self.summary:
                continue

            values = []
            errors = []

            for orig_metric in composite_metrics_orig:
                m = self.summary[method].get(orig_metric, {})
                mean_val = m.get('mean')
                std_val = m.get('std', 0)

                values.append(mean_val if mean_val is not None else 0)
                errors.append(std_val)

            # 添加一些抖动以避免重叠
            jitter = (methods.index(method) - 2) * 0.06
            x_jittered = x_positions + jitter

            # 绘制误差棒和散点
            ax.errorbar(x_jittered, values, yerr=errors,
                       fmt='o', markersize=14, linewidth=2.5,
                       color=self.colors[method], label=method,
                       capsize=5, capthick=2.5, alpha=0.85,
                       elinewidth=2)

            # 连线
            if method == 'AIPOM-CoT':
                ax.plot(x_jittered, values, color=self.colors[method],
                       linewidth=2, alpha=0.3, linestyle='--')

        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Comprehensive Performance Comparison Across Metrics',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加水平参考线和区域
        ax.axhspan(0.8, 1.0, alpha=0.1, color=self.accent_colors['excellent'])
        ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                  linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text(len(metric_labels) - 0.3, 0.82, 'Excellent (≥0.8)',
               fontsize=10, color=self.accent_colors['excellent'], fontweight='bold')

        ax.axhspan(0.6, 0.8, alpha=0.05, color=self.accent_colors['good'])
        ax.axhline(y=0.6, color=self.accent_colors['good'],
                  linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text(len(metric_labels) - 0.3, 0.62, 'Good (0.6-0.8)',
               fontsize=10, color=self.accent_colors['good'], fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 5 saved")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        visualizer = BenchmarkVisualizerEnhanced("./results_filtered")
        visualizer.generate_all_figures()

        print("\n✅ All enhanced figures generated successfully!")
        print("📁 Location: ./benchmark_results_final/figures_enhanced/")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure you have run the filtering script first")