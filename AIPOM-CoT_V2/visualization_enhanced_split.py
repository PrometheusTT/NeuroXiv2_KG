"""
Enhanced Visualization for AIPOM-CoT Benchmark
==============================================
Publication-quality figures with diverse chart types

改进点:
1. 多样化的图表类型（不只是柱状图）
2. 更专业的配色方案
3. 去除NM/Nature Methods相关表述
4. 重命名核心指标
5. 所有子图拆分为独立图片
6. 改善图例和文字布局,避免遮挡

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 6.0 (Split Subplots)
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

    def __init__(self, results_dir: str = "./results_filtered"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures_enhanced_split"
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
            'AIPOM-CoT': '#2E86AB',  # Professional blue
            'Direct GPT-4o': '#A23B72',  # Deep purple
            'Template-KG': '#F18F01',  # Warm orange
            'RAG': '#C73E1D',  # Red-orange
            'ReAct': '#6A994E',  # Nature green
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
        logger.info("🎨 GENERATING ENHANCED FIGURES (SPLIT VERSION)")
        logger.info("=" * 80)

        # self.generate_fig1_capabilities_radar()
        # self.generate_fig2_planning_violin_split()
        self.generate_fig3_reasoning_heatmap_split()
        # self.generate_fig4_reflection_polar_split()
        # self.generate_fig5_comprehensive_scatter()

        logger.info(f"\n✅ All enhanced figures saved to: {self.figures_dir}")

    def generate_fig1_capabilities_radar(self):
        """Figure 1: 核心能力雷达图"""

        logger.info("\n📊 Figure 1: Core Capabilities Radar...")

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        dimensions = [
            'Planning\nQuality',
            'Reasoning\nCapability',
            'Chain-of-Thought\nQuality',
            'Reflection\nCapability',
            'Language\nUnderstanding'
        ]

        scores = {}

        for method in methods:
            if method not in self.summary:
                continue

            s = self.summary[method]

            method_scores = []
            planning = s.get('planning_quality', {}).get('mean')
            method_scores.append(planning if planning is not None else 0)

            reasoning = s.get('reasoning_capability', {}).get('mean')
            method_scores.append(reasoning if reasoning is not None else 0)

            cot = s.get('cot_quality', {}).get('mean')
            method_scores.append(cot if cot is not None else 0)

            reflection = s.get('reflection_capability', {}).get('mean')
            method_scores.append(reflection if reflection is not None else 0)

            nlu = s.get('nlu_capability', {}).get('mean')
            method_scores.append(nlu if nlu is not None else 0)

            scores[method] = method_scores

        # 绘制
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]

        for method in methods:
            if method not in scores:
                continue

            values = scores[method]
            values += values[:1]

            linewidth = 3.5 if method == 'AIPOM-CoT' else 2.5
            markersize = 12 if method == 'AIPOM-CoT' else 8
            alpha_fill = 0.25 if method == 'AIPOM-CoT' else 0.08

            ax.plot(angles, values, 'o-', linewidth=linewidth,
                    label=method, color=self.colors.get(method, 'gray'),
                    markersize=markersize, alpha=0.9)

            ax.fill(angles, values, alpha=alpha_fill, color=self.colors.get(method, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=16, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=12)

        # 图例放在右上角外侧,增加间距
        ax.legend(loc='upper left', bbox_to_anchor=(1.25, 1.15), fontsize=16, frameon=True, fancybox=True)
        ax.set_title('AIPOM-CoT: Core Agentic Capabilities',
                     size=18, weight='bold', pad=30)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=2.0)
        plt.savefig(self.figures_dir / "fig1_capabilities_radar.png", dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.savefig(self.figures_dir / "fig1_capabilities_radar.pdf", bbox_inches='tight', pad_inches=0.3)
        plt.close()

        logger.info("  ✓ Figure 1 saved")

    def generate_fig2_planning_violin_split(self):
        """Figure 2: Planning质量分析 - 拆分为3个独立图"""

        logger.info("\n📊 Figure 2: Planning Quality Analysis (Split into 3 figures)...")

        methods = ['AIPOM-CoT', 'ReAct']
        planning_metrics = ['planning_coherence', 'planning_optimality', 'planning_adaptability']
        metric_labels = ['Coherence', 'Optimality', 'Adaptability']
        file_suffixes = ['coherence', 'optimality', 'adaptability']

        for metric, label, suffix in zip(planning_metrics, metric_labels, file_suffixes):
            fig, ax = plt.subplots(figsize=(8, 7))

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
                        ax.scatter(x_data, y_data, alpha=0.5, s=40,
                                   color=self.colors[method], edgecolors='white', linewidth=0.5)

                ax.set_xticks(positions)
                ax.set_xticklabels(methods, fontsize=16, fontweight='bold')
                ax.set_ylabel('Score', fontsize=16, fontweight='bold')
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=14,fontweight='bold')
                ax.set_title(f'Planning Quality: {label}', fontsize=18, fontweight='bold', pad=15)
                ax.set_ylim(0, 1.15)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                # 添加参考线,调整文字位置避免遮挡
                ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                           linestyle=':', alpha=0.5, linewidth=2)
                # ax.text(0.05, 0.85, 'Excellent', fontsize=11,
                #         color=self.accent_colors['excellent'], fontweight='bold',
                #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            else:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)

            plt.tight_layout(pad=1.5)
            plt.savefig(self.figures_dir / f"fig2{chr(97+file_suffixes.index(suffix))}_planning_{suffix}.png",
                       dpi=1200, bbox_inches='tight', pad_inches=0.2)
            plt.savefig(self.figures_dir / f"fig2{chr(97+file_suffixes.index(suffix))}_planning_{suffix}.pdf",
                       bbox_inches='tight', pad_inches=0.2)
            plt.close()

        logger.info("  ✓ Figure 2a, 2b, 2c saved")

    def generate_fig3_reasoning_heatmap_split(self):
        """Figure 3: Reasoning & CoT - 拆分为4个独立图"""

        logger.info("\n📊 Figure 3: Reasoning & CoT Analysis (Split into 4 figures)...")

        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']

        # 3a: Reasoning Components - 热力图
        logger.info("  → Generating 3a: Reasoning Components Heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))

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
        available_methods[1] = 'LLM only'
        print("available_methods[1]--------", available_methods[1])

        if heatmap_data:
            heatmap_data = np.array(heatmap_data)

            im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(component_labels)))
            ax.set_yticks(np.arange(len(available_methods)))
            ax.set_xticklabels(component_labels, fontsize=14, fontweight='bold')
            ax.set_yticklabels(available_methods, fontsize=16, fontweight='bold')

            # 添加数值标注
            for i in range(len(available_methods)):
                for j in range(len(component_labels)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                   ha="center", va="center",
                                   color="black" if heatmap_data[i, j] < 0.6 else "white",
                                   fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold', fontsize=14)

        ax.set_title('Reasoning Components Heatmap', fontsize=18, fontweight='bold', pad=15)

        plt.tight_layout(pad=1.5)
        plt.savefig(self.figures_dir / "fig3a_reasoning_components.png", dpi=1200, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(self.figures_dir / "fig3a_reasoning_components.pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close()

        # 3b: CoT Quality - 分组点图
        logger.info("  → Generating 3b: CoT Quality...")
        fig, ax = plt.subplots(figsize=(10, 7))

        cot_methods = ['AIPOM-CoT', 'ReAct']
        cot_components = ['cot_clarity', 'cot_completeness', 'intermediate_steps_quality']
        cot_labels = ['Clarity', 'Completeness', 'Steps Quality']

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

                offset = (i - len(available_cot_methods) / 2 + 0.5) * width

                # 绘制点和误差棒
                ax.errorbar(x_positions + offset, values, yerr=errors,
                            fmt='o', markersize=14, linewidth=2.5,
                            color=self.colors[method], label=method,
                            capsize=6, capthick=2.5, alpha=0.85)

            ax.set_ylabel('Score', fontsize=16, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(cot_labels, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.15)

            # 图例放在右上角外侧
            ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # 参考线
            ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                       linestyle=':', alpha=0.5, linewidth=2)
            # ax.text(len(cot_labels)-0.3, 0.84, 'Excellent', fontsize=10,
            #        color=self.accent_colors['excellent'], fontweight='bold',
            #        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_title('Chain-of-Thought Quality', fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout(pad=1.5)
        plt.savefig(self.figures_dir / "fig3b_cot_quality.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(self.figures_dir / "fig3b_cot_quality.pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close()

        # 3c: Reasoning vs CoT - 散点对比
        logger.info("  → Generating 3c: Reasoning vs CoT...")
        fig, ax = plt.subplots(figsize=(9, 8))

        for method in cot_methods:
            if method not in self.summary:
                continue

            reasoning_val = self.summary[method].get('reasoning_capability', {}).get('mean')
            cot_val = self.summary[method].get('cot_quality', {}).get('mean')

            if reasoning_val is not None and cot_val is not None:
                ax.scatter(reasoning_val, cot_val, s=300,
                           color=self.colors[method], alpha=0.75,
                           edgecolors='white', linewidth=2.5,
                           label=method, marker='o', zorder=3)

                # 添加标签,位置调整避免遮挡
                ax.annotate(method, (reasoning_val, cot_val),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=self.colors[method]))
        available_methods[1] = 'LLM only'
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='y=x')
        ax.set_xlabel('Reasoning Capability', fontsize=16, fontweight='bold')
        ax.set_ylabel('CoT Quality', fontsize=16, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=13, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=13, fontweight='bold')
        ax.set_title('Reasoning vs CoT Quality', fontsize=18, fontweight='bold', pad=15)
        ax.set_xlim(0.35, 0.95)
        ax.set_ylim(0.35, 1.0)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=13, loc='lower right', frameon=True, fancybox=True)

        plt.tight_layout(pad=1.5)
        plt.savefig(self.figures_dir / "fig3c_reasoning_vs_cot.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(self.figures_dir / "fig3c_reasoning_vs_cot.pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close()

        # 3d: Overall Reasoning - 棒棒糖图
        logger.info("  → Generating 3d: Overall Reasoning...")
        fig, ax = plt.subplots(figsize=(10, 8))

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
                        linewidth=4, alpha=0.7)
                ax.scatter(val, i, s=250, color=self.colors[method],
                           edgecolors='white', linewidth=2.5, zorder=3)

                # 添加数值标签,位置调整
                ax.text(val + 0.03, i, f'{val:.3f}',
                        va='center', fontsize=13, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
            available_reasoning_methods[1]='LLM only'
            ax.set_yticks(y_positions)
            ax.set_yticklabels(available_reasoning_methods, fontsize=16, fontweight='bold')
            ax.set_xlabel('Score', fontsize=16, fontweight='bold')
            # ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1.05)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # 参考区域
            ax.axvspan(0.8, 1.05, alpha=0.1, color=self.accent_colors['excellent'])
            # ax.text(0.92, len(available_reasoning_methods) - 1, 'Excellent',
            #         fontsize=11, color=self.accent_colors['excellent'], ha='center', fontweight='bold',
            #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_title('Overall Reasoning Capability', fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout(pad=1.5)
        plt.savefig(self.figures_dir / "fig3d_overall_reasoning.png", dpi=1200, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(self.figures_dir / "fig3d_overall_reasoning.pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close()

        logger.info("  ✓ Figure 3a, 3b, 3c, 3d saved")

    def generate_fig4_reflection_polar_split(self):
        """Figure 4: Reflection - 拆分为2个独立图"""

        logger.info("\n📊 Figure 4: Reflection Capability (Split into 2 figures)...")

        methods = ['AIPOM-CoT', 'ReAct']

        # 4a: Reflection Components - 极坐标柱状图
        logger.info("  → Generating 4a: Reflection Components...")
        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_subplot(111, projection='polar')

        reflection_components = ['error_detection', 'self_correction', 'iterative_refinement']
        component_labels = ['Error\nDetection', 'Self\nCorrection', 'Iterative\nRefinement']

        available_methods = [m for m in methods if m in self.summary]


        if available_methods:
            N = len(component_labels)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

            width = 2 * np.pi / N / len(available_methods) * 0.8  # 稍微减小宽度避免重叠

            for i, method in enumerate(available_methods):
                values = []
                for comp in reflection_components:
                    val = self.summary[method].get(comp, {}).get('mean')
                    values.append(val if val is not None else 0)

                offset = width * (i - len(available_methods) / 2 + 0.5)
                bars = ax.bar([a + offset for a in angles], values, width=width,
                              label=method, alpha=0.8, color=self.colors[method],
                              edgecolor='white', linewidth=2)

                # 添加数值标签,调整位置避免遮挡
                for angle, value, bar in zip(angles, values, bars):
                    rotation = np.rad2deg(angle + offset)
                    alignment = "left" if angle < np.pi else "right"
                    ax.text(angle + offset, value + 0.08, f'{value:.2f}',
                            ha=alignment, va='center', fontsize=10, fontweight='bold',
                            rotation=rotation, rotation_mode='anchor',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            available_methods[1] = 'LLM only'
            ax.set_xticks(angles)
            ax.set_xticklabels(component_labels, fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=13)

            # 图例放在右上角外侧
            ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.15), fontsize=16, frameon=True, fancybox=True)
            ax.grid(True, alpha=0.3)

        ax.set_title('Reflection Components', fontsize=18, fontweight='bold', pad=25)

        plt.tight_layout(pad=2.0)
        plt.savefig(self.figures_dir / "fig4a_reflection_components.png", dpi=1200, bbox_inches='tight', pad_inches=0.3)
        plt.savefig(self.figures_dir / "fig4a_reflection_components.pdf", bbox_inches='tight', pad_inches=0.3)
        plt.close()

        # 4b: Overall Reflection - 水平误差棒图
        logger.info("  → Generating 4b: Overall Reflection...")
        fig, ax = plt.subplots(figsize=(10, 7))

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
                        color=self.colors[method], alpha=0.75,
                        error_kw={'linewidth': 2.5, 'elinewidth': 2.5, 'capsize': 6})

                # 添加数值标签,调整位置
                ax.text(val + 0.08, i, f'{val:.3f}',
                        va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

            ax.set_yticks(y_positions)
            # ax.set_yticklabels(available_methods_overall, fontsize=16, fontweight='bold')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.yaxis.set_visible(False)
            ax.set_xlabel('Score', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # 参考区域
            ax.axvspan(0.8, 1.2, alpha=0.1, color=self.accent_colors['excellent'])
            # ax.text(1.0, len(available_methods_overall) - 1, 'Excellent',
            #         fontsize=11, color=self.accent_colors['excellent'],
            #         ha='center', fontweight='bold',
            #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_title('Overall Reflection Capability', fontsize=18, fontweight='bold', pad=15)

        plt.tight_layout(pad=1.5)
        plt.savefig(self.figures_dir / "fig4b_overall_reflection.png", dpi=1200, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(self.figures_dir / "fig4b_overall_reflection.pdf", bbox_inches='tight', pad_inches=0.2)
        plt.close()

        logger.info("  ✓ Figure 4a, 4b saved")

    # def generate_fig5_comprehensive_scatter(self):
    #     """Figure 5: 综合对比 - 分组散点图"""
    #
    #     logger.info("\n📊 Figure 5: Comprehensive Comparison (Grouped Scatter)...")
    #
    #     fig, ax = plt.subplots(figsize=(15, 9))
    #
    #     methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct']
    #
    #     # 重命名指标
    #     composite_metrics_orig = ['nm_capability_score', 'overall_score', 'biological_insight_score']
    #     metric_labels = [
    #         'Agentic Capability\nIndex (ACI)',
    #         'Composite Performance\nScore (CPS)',
    #         'Domain Task\nScore (DTS)'
    #     ]
    #
    #     x_positions = np.arange(len(metric_labels))
    #
    #     # 为每个方法绘制散点
    #     for method in methods:
    #         if method not in self.summary:
    #             continue
    #
    #         values = []
    #         errors = []
    #
    #         for orig_metric in composite_metrics_orig:
    #             m = self.summary[method].get(orig_metric, {})
    #             mean_val = m.get('mean')
    #             std_val = m.get('std', 0)
    #
    #             values.append(mean_val if mean_val is not None else 0)
    #             errors.append(std_val)
    #
    #         # 添加一些抖动以避免重叠
    #         jitter = (methods.index(method) - 2) * 0.08
    #         x_jittered = x_positions + jitter
    #
    #         # 绘制误差棒和散点
    #         ax.errorbar(x_jittered, values, yerr=errors,
    #                     fmt='o', markersize=16, linewidth=2.5,
    #                     color=self.colors[method], label=method,
    #                     capsize=6, capthick=2.5, alpha=0.85,
    #                     elinewidth=2.5, zorder=2)
    #
    #         # 连线
    #         if method == 'AIPOM-CoT':
    #             ax.plot(x_jittered, values, color=self.colors[method],
    #                     linewidth=2.5, alpha=0.4, linestyle='--', zorder=1)
    #
    #     ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    #     ax.set_title('Comprehensive Performance Comparison Across Metrics',
    #                  fontsize=18, fontweight='bold', pad=20)
    #     ax.set_xticks(x_positions)
    #     ax.set_xticklabels(metric_labels, fontsize=16, fontweight='bold')
    #     ax.set_ylim(0, 1.15)
    #
    #     # 图例放在左上角,两列排列,避免遮挡数据
    #     ax.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True,ncol=2)
    #     ax.grid(axis='y', alpha=0.3, linestyle='--')
    #
    #     # 添加水平参考线和区域,文字位置调整
    #     ax.axhspan(0.8, 1.0, alpha=0.08, color=self.accent_colors['excellent'], zorder=0)
    #     ax.axhline(y=0.8, color=self.accent_colors['excellent'],
    #                linestyle=':', linewidth=2, alpha=0.6, zorder=0)
    #     # ax.text(len(metric_labels) - 0.15, 0.77, 'Excellent (≥0.8)',
    #     #         fontsize=11, color=self.accent_colors['excellent'], fontweight='bold',
    #     #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    #
    #     ax.axhspan(0.6, 0.8, alpha=0.05, color=self.accent_colors['good'], zorder=0)
    #     ax.axhline(y=0.6, color=self.accent_colors['good'],
    #                linestyle=':', linewidth=2, alpha=0.6, zorder=0)
    #     # ax.text(len(metric_labels) - 0.15, 0.57, 'Good (0.6-0.8)',
    #     #         fontsize=11, color=self.accent_colors['good'], fontweight='bold',
    #     #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    #
    #     plt.tight_layout(pad=1.5)
    #     plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    #     plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.pdf", bbox_inches='tight', pad_inches=0.2)
    #     plt.close()
    #
    #     logger.info("  ✓ Figure 5 saved")


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
            errors_clipped = []  # 用于显示的裁剪误差

            for orig_metric in composite_metrics_orig:
                m = self.summary[method].get(orig_metric, {})
                mean_val = m.get('mean')
                std_val = m.get('std', 0)

                if mean_val is not None:
                    values.append(mean_val)
                    errors.append(std_val)

                    # 计算裁剪后的误差（确保不超过[0, 1]范围）
                    upper_error = min(std_val, 1.0 - mean_val)  # 上界不超过1.0
                    lower_error = min(std_val, mean_val)  # 下界不低于0.0
                    errors_clipped.append([lower_error, upper_error])
                else:
                    values.append(0)
                    errors.append(0)
                    errors_clipped.append([0, 0])

            # 转换为numpy数组并转置（matplotlib需要的格式）
            errors_clipped = np.array(errors_clipped).T

            # 添加一些抖动以避免重叠
            jitter = (methods.index(method) - 2) * 0.06
            x_jittered = x_positions + jitter

            # 绘制误差棒和散点（使用裁剪后的误差）
            ax.errorbar(x_jittered, values, yerr=errors_clipped,
                        fmt='o', markersize=14, linewidth=2.5,
                        color=self.colors[method], label=method,
                        capsize=5, capthick=2.5, alpha=0.85,
                        elinewidth=2)

            # 连线
            if method == 'AIPOM-CoT':
                ax.plot(x_jittered, values, color=self.colors[method],
                        linewidth=2, alpha=0.3, linestyle='--')
        methods[1]='LLM only'
        ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.set_title('Comprehensive Performance Comparison Across Metrics',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(metric_labels, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)  # 略微超过1.0以显示顶部的点
        ax.legend(methods,loc='upper left', fontsize=13, framealpha=0.95, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加水平参考线和区域
        ax.axhspan(0.8, 1.0, alpha=0.1, color=self.accent_colors['excellent'])
        ax.axhline(y=0.8, color=self.accent_colors['excellent'],
                   linestyle=':', linewidth=1.5, alpha=0.6)
        # ax.text(len(metric_labels) - 0.3, 0.82, 'Excellent (≥0.8)',
        #         fontsize=10, color=self.accent_colors['excellent'], fontweight='bold')

        ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)  # 添加1.0参考线

        ax.axhspan(0.6, 0.8, alpha=0.05, color=self.accent_colors['good'])
        ax.axhline(y=0.6, color=self.accent_colors['good'],
                   linestyle=':', linewidth=1.5, alpha=0.6)
        # ax.text(len(metric_labels) - 0.3, 0.62, 'Good (0.6-0.8)',
        #         fontsize=10, color=self.accent_colors['good'], fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.png", dpi=1200, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig5_comprehensive_scatter.pdf", bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Figure 5 saved (with clipped error bars)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        visualizer = BenchmarkVisualizerEnhanced("./results_filtered")
        visualizer.generate_all_figures()

        print("\n✅ All enhanced figures generated successfully!")
        print("📁 Location: ./benchmark_results_final/figures_enhanced_split/")
        print("\n📊 Generated files:")
        print("  - fig1_capabilities_radar.png/pdf")
        print("  - fig2a_planning_coherence.png/pdf")
        print("  - fig2b_planning_optimality.png/pdf")
        print("  - fig2c_planning_adaptability.png/pdf")
        print("  - fig3a_reasoning_components.png/pdf")
        print("  - fig3b_cot_quality.png/pdf")
        print("  - fig3c_reasoning_vs_cot.png/pdf")
        print("  - fig3d_overall_reasoning.png/pdf")
        print("  - fig4a_reflection_components.png/pdf")
        print("  - fig4b_overall_reflection.png/pdf")
        print("  - fig5_comprehensive_scatter.png/pdf")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure you have run the filtering script first")