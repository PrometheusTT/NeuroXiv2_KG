#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版可视化模块 - 支持子图单独绘制
- 可以单独绘制每个子图并保存为高分辨率图像
- 使用1200 DPI获得高质量输出
- 保持原有的整合图表功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.ticker as mtick
import warnings

warnings.filterwarnings('ignore')

# 设置学术论文风格的绘图参数
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,  # 默认DPI，单独保存时会使用1200
    'savefig.dpi': 1200,  # 提高到1200DPI
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# 学术配色方案
ACADEMIC_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# 类别配色
CATEGORY_COLORS = {
    'kg_navigation': ACADEMIC_COLORS['blue'],
    'reasoning': ACADEMIC_COLORS['purple'],
    'tool_use': ACADEMIC_COLORS['orange'],
    'complex': ACADEMIC_COLORS['red']
}


class EnhancedVisualizer:
    """增强版可视化类，支持子图单独绘制"""

    def __init__(self, data_source: Union[str, pd.DataFrame], output_dir: str = "enhanced_figures"):
        """
        初始化可视化器

        Args:
            data_source: CSV文件路径或pandas DataFrame
            output_dir: 保存输出图表的目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子图目录
        self.subplots_dir = self.output_dir / "subplots"
        self.subplots_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            raise ValueError("data_source必须是文件路径或DataFrame")

        # 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据用于可视化"""
        # 计算派生指标（如果不存在）
        if 'kg_cot_score' not in self.df.columns:
            self.df['kg_cot_score'] = (
                    self.df['schema_coverage'] * 0.3 +
                    self.df['query_semantic_complexity'] / 10 * 0.3 +
                    self.df['cot_effectiveness'] * 0.2 +
                    self.df['planning_coherence'] * 0.2
            )

        if 'autonomy_score' not in self.df.columns:
            self.df['autonomy_score'] = (
                    self.df['autonomy_index'] * 0.3 +
                    self.df['reasoning_depth_score'] * 0.2 +
                    self.df['reflection_effectiveness'] * 0.2 +
                    self.df['adaptation_efficiency'] * 0.2 +
                    self.df['problem_decomposition_quality'] * 0.1
            )

        if 'tool_score' not in self.df.columns:
            self.df['tool_score'] = (
                    self.df['tool_precision'] * 0.3 +
                    self.df['tool_recall'] * 0.3 +
                    self.df['tool_f1_score'] * 0.4
            )

        if 'answer_quality' not in self.df.columns:
            self.df['answer_quality'] = (
                    self.df['answer_completeness'] * 0.5 +
                    self.df['answer_correctness'] * 0.5
            )

        if 'overall_performance' not in self.df.columns:
            self.df['overall_performance'] = (
                    self.df['kg_cot_score'] * 0.3 +
                    self.df['autonomy_score'] * 0.3 +
                    self.df['tool_score'] * 0.2 +
                    self.df['answer_quality'] * 0.2
            )

        # 添加类别显示名
        self.df['category_display'] = self.df['category'].apply(
            lambda x: x.replace('_', ' ').title()
        )

        # 按复杂度排序以获得更好的可视化效果
        self.df = self.df.sort_values(['complexity', 'category'])

    def create_all_figures(self):
        """生成所有增强版可视化图表"""
        print("\n正在生成增强版评估可视化...")

        # 创建组合图表
        self.fig1_performance_overview()
        self.fig2_complexity_scaling()
        self.fig3_capability_radar()
        self.fig4_metric_distributions()
        self.fig5_comparative_analysis()
        self.fig6_executive_summary()

        # 创建所有单独子图
        self.create_all_subplots()

        print(f"\n✅ 所有图表已保存到: {self.output_dir}")
        print(f"✅ 所有子图已保存到: {self.subplots_dir}")

    def create_all_subplots(self):
        """生成所有单独的子图"""
        print("\n正在生成所有单独子图...")

        # 图1的子图
        self.subplot_1a_overall_performance_by_category()
        self.subplot_1b_kg_capabilities()
        self.subplot_1c_advanced_capabilities()

        # 图2的子图
        self.subplot_2a_performance_scaling()
        self.subplot_2b_capability_scaling()
        self.subplot_2c_execution_metrics()
        self.subplot_2d_kg_utilization()

        # 图3的子图
        self.subplot_3a_radar_by_category()
        self.subplot_3b_radar_by_complexity()

        # 图4的子图
        self.subplot_4a_kg_navigation_distribution()
        self.subplot_4b_reasoning_distribution()
        self.subplot_4c_tool_usage_distribution()
        self.subplot_4d_answer_quality_distribution()

        # 图5的子图
        self.subplot_5a_schema_vs_complexity()
        self.subplot_5b_autonomy_vs_quality()
        self.subplot_5c_tool_vs_time()
        self.subplot_5d_planning_vs_performance()

        # 图6的子图
        self.subplot_6a_overall_score()
        self.subplot_6b_capability_scores()
        self.subplot_6c_performance_matrix()
        self.subplot_6d_key_statistics()

    #
    # 图1: 性能概览
    #
    def subplot_1a_overall_performance_by_category(self):
        """子图1A: 按类别的整体性能柱状图"""
        fig, ax = plt.subplots(figsize=(8, 5))

        # 计算按类别的平均性能
        cat_perf = self.df.groupby('category_display')['overall_performance'].agg(['mean', 'std'])

        # 创建带误差条的柱状图
        bars = ax.bar(
            cat_perf.index,
            cat_perf['mean'],
            yerr=cat_perf['std'],
            color=[CATEGORY_COLORS[c.lower().replace(' ', '_')] for c in cat_perf.index],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.8,
            capsize=5
        )

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # 添加样式
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Overall Performance Score')
        ax.set_title('Overall Performance by Category', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 添加评分等级的水平线
        grade_lines = [
            (0.9, 'A', ACADEMIC_COLORS['green']),
            (0.8, 'B', ACADEMIC_COLORS['olive']),
            (0.7, 'C', ACADEMIC_COLORS['orange']),
            (0.6, 'D', ACADEMIC_COLORS['red'])
        ]

        for score, grade, color in grade_lines:
            ax.axhline(y=score, color=color, linestyle='--', alpha=0.7)
            ax.text(
                len(cat_perf) - 0.2,
                score + 0.01,
                f"Grade {grade}",
                ha='right',
                color=color,
                fontsize=8
            )

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '1a_overall_performance_by_category.png', dpi=1200)
        plt.close()

        print("  ✓ 子图1A: 按类别的整体性能")

    def subplot_1b_kg_capabilities(self):
        """子图1B: 知识图谱导航能力"""
        fig, ax = plt.subplots(figsize=(8, 5))

        # 计算按类别的指标
        kg_metrics = self.df.groupby('category_display')[
            ['schema_coverage', 'cot_effectiveness', 'planning_coherence']
        ].mean()

        # 为显示重命名列
        kg_metrics.columns = ['Schema\nCoverage', 'CoT\nEffectiveness', 'Planning\nCoherence']

        # 绘制为分组柱状图
        kg_metrics.plot(
            kind='bar',
            ax=ax,
            rot=0,
            colormap='Blues',
            edgecolor='black',
            linewidth=0.8
        )

        ax.set_ylim(0, 1.05)
        ax.set_title('Knowledge Graph Navigation Capabilities', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', frameon=True, framealpha=0.7)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '1b_kg_capabilities.png', dpi=1200)
        plt.close()

        print("  ✓ 子图1B: 知识图谱导航能力")

    def subplot_1c_advanced_capabilities(self):
        """子图1C: 高级推理能力"""
        fig, ax = plt.subplots(figsize=(8, 5))

        # 计算按类别的指标
        adv_metrics = self.df.groupby('category_display')[
            ['autonomy_score', 'tool_score', 'answer_quality']
        ].mean()

        # 为显示重命名列
        adv_metrics.columns = ['Autonomy', 'Tool Usage', 'Answer Quality']

        # 绘制为分组柱状图
        adv_metrics.plot(
            kind='bar',
            ax=ax,
            rot=0,
            colormap='Purples',
            edgecolor='black',
            linewidth=0.8
        )

        ax.set_ylim(0, 1.05)
        ax.set_title('Advanced Reasoning Capabilities', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', frameon=True, framealpha=0.7)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '1c_advanced_capabilities.png', dpi=1200)
        plt.close()

        print("  ✓ 子图1C: 高级推理能力")

    def fig1_performance_overview(self):
        """
        图1: 性能概览
        - 按类别显示整体性能的柱状图
        - 关键能力得分的小多图
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

        # 子图1: 按类别的整体性能（柱状图）
        ax1 = fig.add_subplot(gs[0, :])

        # 计算按类别的平均性能
        cat_perf = self.df.groupby('category_display')['overall_performance'].agg(['mean', 'std'])

        # 创建带误差条的柱状图
        bars = ax1.bar(
            cat_perf.index,
            cat_perf['mean'],
            yerr=cat_perf['std'],
            color=[CATEGORY_COLORS[c.lower().replace(' ', '_')] for c in cat_perf.index],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.8,
            capsize=5
        )

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # 添加样式
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('Overall Performance Score')
        ax1.set_title('A. Overall Performance by Category', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # 添加评分等级的水平线
        grade_lines = [
            (0.9, 'A', ACADEMIC_COLORS['green']),
            (0.8, 'B', ACADEMIC_COLORS['olive']),
            (0.7, 'C', ACADEMIC_COLORS['orange']),
            (0.6, 'D', ACADEMIC_COLORS['red'])
        ]

        for score, grade, color in grade_lines:
            ax1.axhline(y=score, color=color, linestyle='--', alpha=0.7)
            ax1.text(
                len(cat_perf) - 0.2,
                score + 0.01,
                f"Grade {grade}",
                ha='right',
                color=color,
                fontsize=8
            )

        # 子图2: KG+CoT能力（左下）
        ax2 = fig.add_subplot(gs[1, 0])

        # 计算按类别的指标
        kg_metrics = self.df.groupby('category_display')[
            ['schema_coverage', 'cot_effectiveness', 'planning_coherence']
        ].mean()

        # 为显示重命名列
        kg_metrics.columns = ['Schema\nCoverage', 'CoT\nEffectiveness', 'Planning\nCoherence']

        # 绘制为分组柱状图
        kg_metrics.plot(
            kind='bar',
            ax=ax2,
            rot=0,
            colormap='Blues',
            edgecolor='black',
            linewidth=0.8
        )

        ax2.set_ylim(0, 1.05)
        ax2.set_title('B. Knowledge Graph Navigation Capabilities', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(loc='upper right', frameon=True, framealpha=0.7)

        # 子图3: 高级能力（右下）
        ax3 = fig.add_subplot(gs[1, 1])

        # 计算按类别的指标
        adv_metrics = self.df.groupby('category_display')[
            ['autonomy_score', 'tool_score', 'answer_quality']
        ].mean()

        # 为显示重命名列
        adv_metrics.columns = ['Autonomy', 'Tool Usage', 'Answer Quality']

        # 绘制为分组柱状图
        adv_metrics.plot(
            kind='bar',
            ax=ax3,
            rot=0,
            colormap='Purples',
            edgecolor='black',
            linewidth=0.8
        )

        ax3.set_ylim(0, 1.05)
        ax3.set_title('C. Advanced Reasoning Capabilities', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.legend(loc='upper right', frameon=True, framealpha=0.7)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_performance_overview.png')
        plt.close()

        print("  ✓ 图1: 性能概览已创建")

    #
    # 图2: 复杂度扩展性分析
    #
    def subplot_2a_performance_scaling(self):
        """子图2A: 性能随复杂度的扩展"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 计算按复杂度的指标
        complexity_metrics = self.df.groupby('complexity').agg({
            'overall_performance': ['mean', 'std']
        })

        x = complexity_metrics.index
        y = complexity_metrics['overall_performance']['mean']
        yerr = complexity_metrics['overall_performance']['std']

        ax.errorbar(
            x, y, yerr=yerr,
            marker='o',
            markersize=8,
            markerfacecolor=ACADEMIC_COLORS['blue'],
            markeredgecolor='black',
            markeredgewidth=0.8,
            capsize=5,
            elinewidth=1,
            color=ACADEMIC_COLORS['blue'],
            linewidth=2
        )

        # 添加趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x), max(x), 100)
        ax.plot(x_trend, p(x_trend), '--', color=ACADEMIC_COLORS['red'], alpha=0.7, linewidth=1)

        # 计算退化率
        slope = z[0]
        slope_percent = abs(slope * 100)

        # 添加带有趋势信息的文本框
        if slope < 0:
            trend_text = f"Performance decreases by {slope_percent:.1f}% per complexity level"
        else:
            trend_text = f"Performance increases by {slope_percent:.1f}% per complexity level"

        ax.text(
            0.03, 0.05,
            trend_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=9
        )

        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Overall Performance Score')
        ax.set_title('Performance Scaling with Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '2a_performance_scaling.png', dpi=1200)
        plt.close()

        print("  ✓ 子图2A: 性能随复杂度的扩展")

    def subplot_2b_capability_scaling(self):
        """子图2B: 能力随复杂度的扩展"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 计算按复杂度的指标
        complexity_metrics = self.df.groupby('complexity').agg({
            'kg_cot_score': ['mean', 'std'],
            'autonomy_score': ['mean', 'std'],
            'tool_score': ['mean', 'std'],
            'answer_quality': ['mean', 'std']
        })

        x = complexity_metrics.index

        # 绘制不同能力得分
        metrics = [
            ('KG+CoT', 'kg_cot_score', ACADEMIC_COLORS['blue']),
            ('Autonomy', 'autonomy_score', ACADEMIC_COLORS['purple']),
            ('Tool Use', 'tool_score', ACADEMIC_COLORS['orange']),
            ('Answer', 'answer_quality', ACADEMIC_COLORS['green'])
        ]

        for label, metric, color in metrics:
            y = complexity_metrics[metric]['mean']
            ax.plot(
                x, y,
                marker='o',
                markersize=6,
                label=label,
                color=color,
                linewidth=2
            )

        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Capability Score')
        ax.set_title('Capability Scaling by Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, framealpha=0.7)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '2b_capability_scaling.png', dpi=1200)
        plt.close()

        print("  ✓ 子图2B: 能力随复杂度的扩展")

    def subplot_2c_execution_metrics(self):
        """子图2C: 执行指标随复杂度的变化"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 计算按复杂度的指标
        complexity_metrics = self.df.groupby('complexity').agg({
            'execution_time': ['mean', 'std'],
            'total_queries': ['mean', 'std']
        })

        x = complexity_metrics.index

        # 主Y轴: 执行时间
        y1 = complexity_metrics['execution_time']['mean']
        yerr1 = complexity_metrics['execution_time']['std']

        line1 = ax.errorbar(
            x, y1, yerr=yerr1,
            marker='s',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['red'],
            markeredgecolor='black',
            capsize=5,
            label='Execution Time',
            color=ACADEMIC_COLORS['red'],
            linewidth=2
        )

        ax.set_xticks(x)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Execution Time (seconds)')

        # 次Y轴: 查询总数
        ax2 = ax.twinx()
        y2 = complexity_metrics['total_queries']['mean']
        yerr2 = complexity_metrics['total_queries']['std']

        line2 = ax2.errorbar(
            x, y2, yerr=yerr2,
            marker='^',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['green'],
            markeredgecolor='black',
            capsize=5,
            label='Total Queries',
            color=ACADEMIC_COLORS['green'],
            linewidth=2
        )

        ax2.set_ylabel('Number of Queries')

        # 添加组合图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('Execution Metrics by Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '2c_execution_metrics.png', dpi=1200)
        plt.close()

        print("  ✓ 子图2C: 执行指标随复杂度的变化")

    def subplot_2d_kg_utilization(self):
        """子图2D: 知识图谱利用率"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 计算按复杂度的指标
        complexity_metrics = self.df.groupby('complexity').agg({
            'query_semantic_complexity': ['mean', 'std'],
            'schema_coverage': ['mean', 'std']
        })

        x = complexity_metrics.index

        # 主Y轴: 查询复杂度
        y1 = complexity_metrics['query_semantic_complexity']['mean']

        line1 = ax.plot(
            x, y1,
            marker='d',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['orange'],
            markeredgecolor='black',
            label='Query Complexity',
            color=ACADEMIC_COLORS['orange'],
            linewidth=2
        )

        ax.set_xticks(x)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Query Semantic Complexity')

        # 次Y轴: Schema覆盖率
        ax2 = ax.twinx()
        y2 = complexity_metrics['schema_coverage']['mean']

        line2 = ax2.plot(
            x, y2,
            marker='o',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['blue'],
            markeredgecolor='black',
            label='Schema Coverage',
            color=ACADEMIC_COLORS['blue'],
            linewidth=2
        )

        ax2.set_ylabel('Schema Coverage')
        ax2.set_ylim(0, 1.05)

        # 添加组合图例
        lines = line1 + line2
        labels = ['Query Complexity', 'Schema Coverage']
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('Knowledge Graph Utilization', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '2d_kg_utilization.png', dpi=1200)
        plt.close()

        print("  ✓ 子图2D: 知识图谱利用率")

    def fig2_complexity_scaling(self):
        """
        图2: 复杂度扩展性分析
        - 线图显示性能如何随复杂度扩展
        - 不同指标组的单独图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 计算按复杂度的指标
        complexity_metrics = self.df.groupby('complexity').agg({
            'overall_performance': ['mean', 'std'],
            'kg_cot_score': ['mean', 'std'],
            'autonomy_score': ['mean', 'std'],
            'tool_score': ['mean', 'std'],
            'answer_quality': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'query_efficiency': ['mean', 'std'],
            'total_queries': ['mean', 'std'],
            'schema_coverage': ['mean', 'std'],
            'query_semantic_complexity': ['mean', 'std'],
        })

        # 图1: 整体性能 vs 复杂度（左上）
        ax = axes[0, 0]
        x = complexity_metrics.index
        y = complexity_metrics['overall_performance']['mean']
        yerr = complexity_metrics['overall_performance']['std']

        ax.errorbar(
            x, y, yerr=yerr,
            marker='o',
            markersize=8,
            markerfacecolor=ACADEMIC_COLORS['blue'],
            markeredgecolor='black',
            markeredgewidth=0.8,
            capsize=5,
            elinewidth=1,
            color=ACADEMIC_COLORS['blue'],
            linewidth=2
        )

        # 添加趋势线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x), max(x), 100)
        ax.plot(x_trend, p(x_trend), '--', color=ACADEMIC_COLORS['red'], alpha=0.7, linewidth=1)

        # 计算退化率
        slope = z[0]
        slope_percent = abs(slope * 100)

        # 添加带有趋势信息的文本框
        if slope < 0:
            trend_text = f"Performance decreases by {slope_percent:.1f}% per complexity level"
        else:
            trend_text = f"Performance increases by {slope_percent:.1f}% per complexity level"

        ax.text(
            0.03, 0.05,
            trend_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=9
        )

        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Overall Performance Score')
        ax.set_title('A. Performance Scaling with Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 图2: 能力得分 vs 复杂度（右上）
        ax = axes[0, 1]

        # 绘制不同能力得分
        metrics = [
            ('KG+CoT', 'kg_cot_score', ACADEMIC_COLORS['blue']),
            ('Autonomy', 'autonomy_score', ACADEMIC_COLORS['purple']),
            ('Tool Use', 'tool_score', ACADEMIC_COLORS['orange']),
            ('Answer', 'answer_quality', ACADEMIC_COLORS['green'])
        ]

        for label, metric, color in metrics:
            y = complexity_metrics[metric]['mean']
            ax.plot(
                x, y,
                marker='o',
                markersize=6,
                label=label,
                color=color,
                linewidth=2
            )

        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Capability Score')
        ax.set_title('B. Capability Scaling by Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, framealpha=0.7)

        # 图3: 执行指标 vs 复杂度（左下）
        ax = axes[1, 0]

        # 主Y轴: 执行时间
        y1 = complexity_metrics['execution_time']['mean']
        yerr1 = complexity_metrics['execution_time']['std']

        line1 = ax.errorbar(
            x, y1, yerr=yerr1,
            marker='s',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['red'],
            markeredgecolor='black',
            capsize=5,
            label='Execution Time',
            color=ACADEMIC_COLORS['red'],
            linewidth=2
        )

        ax.set_xticks(x)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Execution Time (seconds)')

        # 次Y轴: 查询总数
        ax2 = ax.twinx()
        y2 = complexity_metrics['total_queries']['mean']
        yerr2 = complexity_metrics['total_queries']['std']

        line2 = ax2.errorbar(
            x, y2, yerr=yerr2,
            marker='^',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['green'],
            markeredgecolor='black',
            capsize=5,
            label='Total Queries',
            color=ACADEMIC_COLORS['green'],
            linewidth=2
        )

        ax2.set_ylabel('Number of Queries')

        # 添加组合图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('C. Execution Metrics by Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 图4: 查询复杂度 vs Schema覆盖率（右下）
        ax = axes[1, 1]

        # 主Y轴: 查询复杂度
        y1 = complexity_metrics['query_semantic_complexity']['mean']

        line1 = ax.plot(
            x, y1,
            marker='d',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['orange'],
            markeredgecolor='black',
            label='Query Complexity',
            color=ACADEMIC_COLORS['orange'],
            linewidth=2
        )

        ax.set_xticks(x)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Query Semantic Complexity')

        # 次Y轴: Schema覆盖率
        ax2 = ax.twinx()
        y2 = complexity_metrics['schema_coverage']['mean']

        line2 = ax2.plot(
            x, y2,
            marker='o',
            markersize=7,
            markerfacecolor=ACADEMIC_COLORS['blue'],
            markeredgecolor='black',
            label='Schema Coverage',
            color=ACADEMIC_COLORS['blue'],
            linewidth=2
        )

        ax2.set_ylabel('Schema Coverage')
        ax2.set_ylim(0, 1.05)

        # 添加组合图例
        lines = line1 + line2
        labels = ['Query Complexity', 'Schema Coverage']
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('D. Knowledge Graph Utilization', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_complexity_scaling.png')
        plt.close()

        print("  ✓ 图2: 复杂度扩展性分析已创建")

    #
    # 图3: 能力雷达分析
    #
    def subplot_3a_radar_by_category(self):
        """子图3A: 按类别的能力雷达图"""
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # 定义雷达图的指标
        radar_metrics = [
            'Schema\nCoverage',
            'CoT\nEffectiveness',
            'Planning\nCoherence',
            'Autonomy\nIndex',
            'Tool\nF1 Score',
            'Answer\nQuality'
        ]

        # 映射到实际列
        metric_mapping = {
            'Schema\nCoverage': 'schema_coverage',
            'CoT\nEffectiveness': 'cot_effectiveness',
            'Planning\nCoherence': 'planning_coherence',
            'Autonomy\nIndex': 'autonomy_index',
            'Tool\nF1 Score': 'tool_f1_score',
            'Answer\nQuality': 'answer_quality'
        }

        # 计算雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()

        # 使角度形成完整圆环
        angles += angles[:1]

        # 为完整圆环添加指标标签
        radar_metrics_plot = radar_metrics + [radar_metrics[0]]

        # 绘制每个类别
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            cat_display = category.replace('_', ' ').title()

            # 计算每个指标的平均值
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(cat_data[col].mean())

            # 使值形成完整圆环
            values += values[:1]

            # 绘制类别
            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=cat_display,
                color=CATEGORY_COLORS[category]
            )

            # 填充区域
            ax.fill(
                angles,
                values,
                alpha=0.1,
                color=CATEGORY_COLORS[category]
            )

        # 样式设置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, size=9)
        ax.set_ylim(0, 1)
        ax.set_title('Capability Profile by Category', fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '3a_radar_by_category.png', dpi=1200)
        plt.close()

        print("  ✓ 子图3A: 按类别的能力雷达图")

    def subplot_3b_radar_by_complexity(self):
        """子图3B: 按复杂度的能力雷达图"""
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # 定义雷达图的指标
        radar_metrics = [
            'Schema\nCoverage',
            'CoT\nEffectiveness',
            'Planning\nCoherence',
            'Autonomy\nIndex',
            'Tool\nF1 Score',
            'Answer\nQuality'
        ]

        # 映射到实际列
        metric_mapping = {
            'Schema\nCoverage': 'schema_coverage',
            'CoT\nEffectiveness': 'cot_effectiveness',
            'Planning\nCoherence': 'planning_coherence',
            'Autonomy\nIndex': 'autonomy_index',
            'Tool\nF1 Score': 'tool_f1_score',
            'Answer\nQuality': 'answer_quality'
        }

        # 计算雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()

        # 使角度形成完整圆环
        angles += angles[:1]

        # 为完整圆环添加指标标签
        radar_metrics_plot = radar_metrics + [radar_metrics[0]]

        # 绘制每个复杂度级别
        for complexity in sorted(self.df['complexity'].unique()):
            comp_data = self.df[self.df['complexity'] == complexity]

            # 计算每个指标的平均值
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(comp_data[col].mean())

            # 使值形成完整圆环
            values += values[:1]

            # 基于复杂度创建颜色
            cmap = plt.cm.viridis
            color = cmap(complexity / max(self.df['complexity']))

            # 绘制复杂度级别
            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=f'Level {complexity}',
                color=color
            )

            # 填充区域
            ax.fill(
                angles,
                values,
                alpha=0.1,
                color=color
            )

        # 样式设置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, size=9)
        ax.set_ylim(0, 1)
        ax.set_title('Capability Profile by Complexity', fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '3b_radar_by_complexity.png', dpi=1200)
        plt.close()

        print("  ✓ 子图3B: 按复杂度的能力雷达图")

    def fig3_capability_radar(self):
        """
        图3: 能力雷达分析
        - 显示多维性能的雷达图
        - 不同类别间的比较
        """
        # 设置带两个雷达图的图表
        fig = plt.figure(figsize=(15, 7))

        # 定义雷达图的指标
        radar_metrics = [
            'Schema\nCoverage',
            'CoT\nEffectiveness',
            'Planning\nCoherence',
            'Autonomy\nIndex',
            'Tool\nF1 Score',
            'Answer\nQuality'
        ]

        # 映射到实际列
        metric_mapping = {
            'Schema\nCoverage': 'schema_coverage',
            'CoT\nEffectiveness': 'cot_effectiveness',
            'Planning\nCoherence': 'planning_coherence',
            'Autonomy\nIndex': 'autonomy_index',
            'Tool\nF1 Score': 'tool_f1_score',
            'Answer\nQuality': 'answer_quality'
        }

        # 计算雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()

        # 使角度形成完整圆环
        angles += angles[:1]

        # 为完整圆环添加指标标签
        radar_metrics_plot = radar_metrics + [radar_metrics[0]]

        # 左雷达图: 按类别
        ax1 = fig.add_subplot(121, polar=True)

        # 绘制每个类别
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            cat_display = category.replace('_', ' ').title()

            # 计算每个指标的平均值
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(cat_data[col].mean())

            # 使值形成完整圆环
            values += values[:1]

            # 绘制类别
            ax1.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=cat_display,
                color=CATEGORY_COLORS[category]
            )

            # 填充区域
            ax1.fill(
                angles,
                values,
                alpha=0.1,
                color=CATEGORY_COLORS[category]
            )

        # 样式设置
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(radar_metrics, size=9)
        ax1.set_ylim(0, 1)
        ax1.set_title('A. Capability Profile by Category', fontweight='bold', pad=15)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # 右雷达图: 按复杂度
        ax2 = fig.add_subplot(122, polar=True)

        # 绘制每个复杂度级别
        for complexity in sorted(self.df['complexity'].unique()):
            comp_data = self.df[self.df['complexity'] == complexity]

            # 计算每个指标的平均值
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(comp_data[col].mean())

            # 使值形成完整圆环
            values += values[:1]

            # 基于复杂度创建颜色
            cmap = plt.cm.viridis
            color = cmap(complexity / max(self.df['complexity']))

            # 绘制复杂度级别
            ax2.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=f'Level {complexity}',
                color=color
            )

            # 填充区域
            ax2.fill(
                angles,
                values,
                alpha=0.1,
                color=color
            )

        # 样式设置
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_metrics, size=9)
        ax2.set_ylim(0, 1)
        ax2.set_title('B. Capability Profile by Complexity', fontweight='bold', pad=15)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # 添加总标题
        plt.suptitle('Multi-dimensional Capability Analysis', fontweight='bold', y=0.98, fontsize=14)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_capability_radar.png')
        plt.close()

        print("  ✓ 图3: 能力雷达分析已创建")

    #
    # 图4: 指标分布分析
    #
    def subplot_4a_kg_navigation_distribution(self):
        """子图4A: 知识图谱导航指标分布"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 定义要绘制的指标
        metrics = ['schema_coverage', 'cot_effectiveness', 'planning_coherence']

        # 准备小提琴图数据
        plot_data = []
        labels = []

        for metric in metrics:
            plot_data.append(self.df[metric])
            labels.append(metric.replace('_', ' ').title())

        # 创建小提琴图
        parts = ax.violinplot(
            plot_data,
            showmeans=False,
            showmedians=True
        )

        # 自定义小提琴颜色
        for i, pc in enumerate(parts['bodies']):
            # 基于Blues色图中的位置选择颜色
            cmap = plt.cm.Blues
            color = cmap(0.4 + (i / len(metrics)) * 0.6)
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # 自定义中位数线
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(1.5)

        # 在小提琴内部添加箱线图
        box_positions = range(1, len(metrics) + 1)

        ax.boxplot(
            plot_data,
            positions=box_positions,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', alpha=0.6),
            medianprops=dict(color='black'),
            showfliers=False
        )

        # 为单个测试添加散点
        for i, metric in enumerate(metrics):
            # 抖动的x位置
            x = np.random.normal(i + 1, 0.07, size=len(self.df))
            y = self.df[metric]

            ax.scatter(
                x, y,
                s=20,
                alpha=0.5,
                c='black',
                edgecolor='none'
            )

        # 添加均值标记
        for i, metric in enumerate(metrics):
            mean_val = self.df[metric].mean()
            ax.scatter(
                i + 1, mean_val,
                s=80,
                marker='*',
                color=ACADEMIC_COLORS['red'],
                edgecolor='black',
                zorder=3,
                label='Mean' if i == 0 else ''
            )

            # 添加均值文本
            ax.text(
                i + 1, mean_val + 0.05,
                f'{mean_val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )

        # 样式设置
        ax.set_title('Knowledge Graph Navigation', fontweight='bold')
        ax.set_xticks(box_positions)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '4a_kg_navigation_distribution.png', dpi=1200)
        plt.close()

        print("  ✓ 子图4A: 知识图谱导航指标分布")

    def subplot_4b_reasoning_distribution(self):
        """子图4B: 推理和自主性指标分布"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 定义要绘制的指标
        metrics = ['autonomy_index', 'reasoning_depth_score', 'reflection_effectiveness', 'adaptation_efficiency']

        # 准备小提琴图数据
        plot_data = []
        labels = []

        for metric in metrics:
            plot_data.append(self.df[metric])
            labels.append(metric.replace('_', ' ').title())

        # 创建小提琴图
        parts = ax.violinplot(
            plot_data,
            showmeans=False,
            showmedians=True
        )

        # 自定义小提琴颜色
        for i, pc in enumerate(parts['bodies']):
            # 基于Blues色图中的位置选择颜色
            cmap = plt.cm.Blues
            color = cmap(0.4 + (i / len(metrics)) * 0.6)
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # 自定义中位数线
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(1.5)

        # 在小提琴内部添加箱线图
        box_positions = range(1, len(metrics) + 1)

        ax.boxplot(
            plot_data,
            positions=box_positions,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', alpha=0.6),
            medianprops=dict(color='black'),
            showfliers=False
        )

        # 为单个测试添加散点
        for i, metric in enumerate(metrics):
            # 抖动的x位置
            x = np.random.normal(i + 1, 0.07, size=len(self.df))
            y = self.df[metric]

            ax.scatter(
                x, y,
                s=20,
                alpha=0.5,
                c='black',
                edgecolor='none'
            )

        # 添加均值标记
        for i, metric in enumerate(metrics):
            mean_val = self.df[metric].mean()
            ax.scatter(
                i + 1, mean_val,
                s=80,
                marker='*',
                color=ACADEMIC_COLORS['red'],
                edgecolor='black',
                zorder=3,
                label='Mean' if i == 0 else ''
            )

            # 添加均值文本
            ax.text(
                i + 1, mean_val + 0.05,
                f'{mean_val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )

        # 样式设置
        ax.set_title('Reasoning & Autonomy', fontweight='bold')
        ax.set_xticks(box_positions)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '4b_reasoning_distribution.png', dpi=1200)
        plt.close()

        print("  ✓ 子图4B: 推理和自主性指标分布")

    def subplot_4c_tool_usage_distribution(self):
        """子图4C: 工具使用指标分布"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 定义要绘制的指标
        metrics = ['tool_precision', 'tool_recall', 'tool_f1_score']

        # 准备小提琴图数据
        plot_data = []
        labels = []

        for metric in metrics:
            plot_data.append(self.df[metric])
            labels.append(metric.replace('_', ' ').title())

        # 创建小提琴图
        parts = ax.violinplot(
            plot_data,
            showmeans=False,
            showmedians=True
        )

        # 自定义小提琴颜色
        for i, pc in enumerate(parts['bodies']):
            # 基于Blues色图中的位置选择颜色
            cmap = plt.cm.Blues
            color = cmap(0.4 + (i / len(metrics)) * 0.6)
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # 自定义中位数线
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(1.5)

        # 在小提琴内部添加箱线图
        box_positions = range(1, len(metrics) + 1)

        ax.boxplot(
            plot_data,
            positions=box_positions,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', alpha=0.6),
            medianprops=dict(color='black'),
            showfliers=False
        )

        # 为单个测试添加散点
        for i, metric in enumerate(metrics):
            # 抖动的x位置
            x = np.random.normal(i + 1, 0.07, size=len(self.df))
            y = self.df[metric]

            ax.scatter(
                x, y,
                s=20,
                alpha=0.5,
                c='black',
                edgecolor='none'
            )

        # 添加均值标记
        for i, metric in enumerate(metrics):
            mean_val = self.df[metric].mean()
            ax.scatter(
                i + 1, mean_val,
                s=80,
                marker='*',
                color=ACADEMIC_COLORS['red'],
                edgecolor='black',
                zorder=3,
                label='Mean' if i == 0 else ''
            )

            # 添加均值文本
            ax.text(
                i + 1, mean_val + 0.05,
                f'{mean_val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )

            # 样式设置
            ax.set_title('Tool Usage', fontweight='bold')
            ax.set_xticks(box_positions)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper right')

            # 保存图表
            plt.tight_layout()
            plt.savefig(self.subplots_dir / '4c_tool_usage_distribution.png', dpi=1200)
            plt.close()

            print("  ✓ 子图4C: 工具使用指标分布")

    def subplot_4d_answer_quality_distribution(self):
        """子图4D: 答案质量指标分布"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 定义要绘制的指标
        metrics = ['answer_completeness', 'answer_correctness', 'answer_quality']

        # 准备小提琴图数据
        plot_data = []
        labels = []

        for metric in metrics:
            plot_data.append(self.df[metric])
            labels.append(metric.replace('_', ' ').title())

        # 创建小提琴图
        parts = ax.violinplot(
            plot_data,
            showmeans=False,
            showmedians=True
        )

        # 自定义小提琴颜色
        for i, pc in enumerate(parts['bodies']):
            # 基于Blues色图中的位置选择颜色
            cmap = plt.cm.Blues
            color = cmap(0.4 + (i / len(metrics)) * 0.6)
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # 自定义中位数线
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(1.5)

        # 在小提琴内部添加箱线图
        box_positions = range(1, len(metrics) + 1)

        ax.boxplot(
            plot_data,
            positions=box_positions,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', alpha=0.6),
            medianprops=dict(color='black'),
            showfliers=False
        )

        # 为单个测试添加散点
        for i, metric in enumerate(metrics):
            # 抖动的x位置
            x = np.random.normal(i + 1, 0.07, size=len(self.df))
            y = self.df[metric]

            ax.scatter(
                x, y,
                s=20,
                alpha=0.5,
                c='black',
                edgecolor='none'
            )

        # 添加均值标记
        for i, metric in enumerate(metrics):
            mean_val = self.df[metric].mean()
            ax.scatter(
                i + 1, mean_val,
                s=80,
                marker='*',
                color=ACADEMIC_COLORS['red'],
                edgecolor='black',
                zorder=3,
                label='Mean' if i == 0 else ''
            )

            # 添加均值文本
            ax.text(
                i + 1, mean_val + 0.05,
                f'{mean_val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )

        # 样式设置
        ax.set_title('Answer Quality', fontweight='bold')
        ax.set_xticks(box_positions)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '4d_answer_quality_distribution.png', dpi=1200)
        plt.close()

        print("  ✓ 子图4D: 答案质量指标分布")

    def fig4_metric_distributions(self):
        """
        图4: 指标分布分析
        - 箱线图和小提琴图显示关键指标的分布
        - 显示可变性和中心趋势
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 定义子图标题和指标
        subplot_info = [
            (axes[0, 0], 'A. Knowledge Graph Navigation',
             ['schema_coverage', 'cot_effectiveness', 'planning_coherence']),

            (axes[0, 1], 'B. Reasoning & Autonomy',
             ['autonomy_index', 'reasoning_depth_score', 'reflection_effectiveness', 'adaptation_efficiency']),

            (axes[1, 0], 'C. Tool Usage',
             ['tool_precision', 'tool_recall', 'tool_f1_score']),

            (axes[1, 1], 'D. Answer Quality',
             ['answer_completeness', 'answer_correctness', 'answer_quality'])
        ]

        # 为每组创建小提琴图
        for ax, title, metrics in subplot_info:
            # 过滤数据框中存在的指标
            valid_metrics = [m for m in metrics if m in self.df.columns]

            # 准备小提琴图数据
            plot_data = []
            labels = []

            for metric in valid_metrics:
                plot_data.append(self.df[metric])
                labels.append(metric.replace('_', ' ').title())

            # 创建小提琴图
            parts = ax.violinplot(
                plot_data,
                showmeans=False,
                showmedians=True
            )

            # 自定义小提琴颜色
            for i, pc in enumerate(parts['bodies']):
                # 基于Blues色图中的位置选择颜色
                cmap = plt.cm.Blues
                color = cmap(0.4 + (i / len(valid_metrics)) * 0.6)
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            # 自定义中位数线
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(1.5)

            # 在小提琴内部添加箱线图
            box_positions = range(1, len(valid_metrics) + 1)

            ax.boxplot(
                plot_data,
                positions=box_positions,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.6),
                medianprops=dict(color='black'),
                showfliers=False
            )

            # 添加单个测试的散点
            for i, metric in enumerate(valid_metrics):
                # 抖动的x位置
                x = np.random.normal(i + 1, 0.07, size=len(self.df))
                y = self.df[metric]

                ax.scatter(
                    x, y,
                    s=20,
                    alpha=0.5,
                    c='black',
                    edgecolor='none'
                )

            # 添加均值标记
            for i, metric in enumerate(valid_metrics):
                mean_val = self.df[metric].mean()
                ax.scatter(
                    i + 1, mean_val,
                    s=80,
                    marker='*',
                    color=ACADEMIC_COLORS['red'],
                    edgecolor='black',
                    zorder=3,
                    label='Mean' if i == 0 else ''
                )

                # 添加均值文本
                ax.text(
                    i + 1, mean_val + 0.05,
                    f'{mean_val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

            # 样式设置
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(box_positions)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

            # 仅对第一个图添加图例
            if title == 'A. Knowledge Graph Navigation':
                ax.legend(loc='upper right')

        # 添加总标题
        plt.suptitle('Metric Distribution Analysis', fontweight='bold', y=0.98, fontsize=14)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_metric_distributions.png')
        plt.close()

        print("  ✓ 图4: 指标分布分析已创建")

    #
    # 图5: 比较分析
    #
    def subplot_5a_schema_vs_complexity(self):
        """子图5A: Schema覆盖率与查询复杂度的关系"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 提取数据
        x = self.df['schema_coverage']
        y = self.df['query_semantic_complexity']

        # 创建色彩映射
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(self.df['complexity']), vmax=max(self.df['complexity']))
        colors = [cmap(norm(c)) for c in self.df['complexity']]

        # 创建散点图
        scatter = ax.scatter(
            x, y,
            s=60,
            c=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # 添加趋势线
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x), max(x), 100)
            ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=1.5)

            # 计算相关系数
            corr = np.corrcoef(x, y)[0, 1]
            # 添加相关性文本
            ax.text(
                0.05, 0.95,
                f'r = {corr:.2f}',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
        except:
            pass

        # 为复杂度级别创建图例元素
        legend_elements = [
            mpatches.Patch(color=cmap(norm(c)), label=f'Level {c}')
            for c in sorted(self.df['complexity'].unique())
        ]

        # 添加图例
        ax.legend(
            handles=legend_elements,
            title='Complexity Level',
            loc='best',
            frameon=True,
            framealpha=0.7
        )

        # 设置标签
        ax.set_xlabel('Schema Coverage')
        ax.set_ylabel('Query Complexity Score')
        ax.set_title('Schema Coverage vs Query Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '5a_schema_vs_complexity.png', dpi=1200)
        plt.close()

        print("  ✓ 子图5A: Schema覆盖率与查询复杂度的关系")

    def subplot_5b_autonomy_vs_quality(self):
        """子图5B: 自主性与答案质量的关系"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 提取数据
        x = self.df['autonomy_index']
        y = self.df['answer_quality']

        # 创建类别颜色和尺寸
        colors = [CATEGORY_COLORS[c] for c in self.df['category']]
        sizes = self.df['complexity'] * 20 + 30

        # 创建散点图
        scatter = ax.scatter(
            x, y,
            s=sizes,
            c=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # 添加趋势线
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x), max(x), 100)
            ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=1.5)

            # 计算相关系数
            corr = np.corrcoef(x, y)[0, 1]
            # 添加相关性文本
            ax.text(
                0.05, 0.95,
                f'r = {corr:.2f}',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
        except:
            pass

        # 为类别创建图例元素
        legend_elements = [
            mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace('_', ' ').title())
            for c in CATEGORY_COLORS if c in self.df['category'].unique()
        ]

        # 添加图例
        ax.legend(
            handles=legend_elements,
            title='Category',
            loc='best',
            frameon=True,
            framealpha=0.7
        )

        # 设置标签
        ax.set_xlabel('Autonomy Index')
        ax.set_ylabel('Answer Quality')
        ax.set_title('Autonomy vs Answer Quality', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '5b_autonomy_vs_quality.png', dpi=1200)
        plt.close()

        print("  ✓ 子图5B: 自主性与答案质量的关系")

    def subplot_5c_tool_vs_time(self):
        """子图5C: 工具使用与执行时间的关系"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 提取数据
        x = self.df['tool_f1_score']
        y = self.df['execution_time']

        # 创建色彩映射
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(self.df['complexity']), vmax=max(self.df['complexity']))
        colors = [cmap(norm(c)) for c in self.df['complexity']]

        # 创建散点图
        scatter = ax.scatter(
            x, y,
            s=60,
            c=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # 为复杂度级别创建图例元素
        legend_elements = [
            mpatches.Patch(color=cmap(norm(c)), label=f'Level {c}')
            for c in sorted(self.df['complexity'].unique())
        ]

        # 添加图例
        ax.legend(
            handles=legend_elements,
            title='Complexity Level',
            loc='best',
            frameon=True,
            framealpha=0.7
        )

        # 设置标签
        ax.set_xlabel('Tool F1 Score')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Tool Usage vs Execution Time', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '5c_tool_vs_time.png', dpi=1200)
        plt.close()

        print("  ✓ 子图5C: 工具使用与执行时间的关系")

    def subplot_5d_planning_vs_performance(self):
        """子图5D: 规划连贯性与整体性能的关系"""
        fig, ax = plt.subplots(figsize=(7, 5))

        # 提取数据
        x = self.df['planning_coherence']
        y = self.df['overall_performance']

        # 创建类别颜色和尺寸
        colors = [CATEGORY_COLORS[c] for c in self.df['category']]
        sizes = self.df['complexity'] * 20 + 30

        # 创建散点图
        scatter = ax.scatter(
            x, y,
            s=sizes,
            c=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # 添加趋势线
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x), max(x), 100)
            ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=1.5)

            # 计算相关系数
            corr = np.corrcoef(x, y)[0, 1]
            # 添加相关性文本
            ax.text(
                0.05, 0.95,
                f'r = {corr:.2f}',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
        except:
            pass

        # 为类别创建图例元素
        legend_elements = [
            mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace('_', ' ').title())
            for c in CATEGORY_COLORS if c in self.df['category'].unique()
        ]

        # 添加图例
        ax.legend(
            handles=legend_elements,
            title='Category',
            loc='best',
            frameon=True,
            framealpha=0.7
        )

        # 设置标签
        ax.set_xlabel('Planning Coherence')
        ax.set_ylabel('Overall Performance')
        ax.set_title('Planning Coherence vs Overall Performance', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '5d_planning_vs_performance.png', dpi=1200)
        plt.close()

        print("  ✓ 子图5D: 规划连贯性与整体性能的关系")

    def fig5_comparative_analysis(self):
        """
        图5: 比较分析
        - 散点图显示指标之间的关系
        - 配对指标以揭示洞察
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 定义子图参数
        subplot_info = [
            (axes[0, 0], 'A. Schema Coverage vs Query Complexity',
             'schema_coverage', 'query_semantic_complexity',
             'Complexity Level'),

            (axes[0, 1], 'B. Autonomy vs Answer Quality',
             'autonomy_index', 'answer_quality',
             'Category'),

            (axes[1, 0], 'C. Tool Usage vs Execution Time',
             'tool_f1_score', 'execution_time',
             'Complexity Level'),

            (axes[1, 1], 'D. Planning Coherence vs Overall Performance',
             'planning_coherence', 'overall_performance',
             'Category')
        ]

        for ax, title, x_metric, y_metric, color_by in subplot_info:
            # 提取数据
            x = self.df[x_metric]
            y = self.df[y_metric]

            # 确定颜色和尺寸
            if color_by == 'Category':
                colors = [CATEGORY_COLORS[c] for c in self.df['category']]
                sizes = self.df['complexity'] * 20 + 30
                legend_elements = [
                    mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace('_', ' ').title())
                    for c in CATEGORY_COLORS if c in self.df['category'].unique()
                ]
                legend_title = 'Category'

            else:  # color_by == 'Complexity Level'
                # 创建色彩映射
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=min(self.df['complexity']), vmax=max(self.df['complexity']))
                colors = [cmap(norm(c)) for c in self.df['complexity']]
                sizes = 60

                # 为复杂度级别创建图例元素
                legend_elements = [
                    mpatches.Patch(color=cmap(norm(c)), label=f'Level {c}')
                    for c in sorted(self.df['complexity'].unique())
                ]
                legend_title = 'Complexity Level'

            # 创建散点图
            scatter = ax.scatter(
                x, y,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

            # 添加趋势线
            if x_metric != 'tool_f1_score':  # 跳过工具F1的趋势线（通常为二进制）
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x), max(x), 100)
                    ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=1.5)

                    # 计算相关系数
                    corr = np.corrcoef(x, y)[0, 1]
                    # 添加相关性文本
                    ax.text(
                        0.05, 0.95,
                        f'r = {corr:.2f}',
                        transform=ax.transAxes,
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                    )
                except:
                    pass

            # 添加标签
            x_label = x_metric.replace('_', ' ').title()
            y_label = y_metric.replace('_', ' ').title()

            if y_metric == 'query_semantic_complexity':
                y_label = 'Query Complexity Score'
            elif y_metric == 'execution_time':
                y_label = 'Execution Time (seconds)'

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            # 添加测试ID作为悬停的注释
            for i, txt in enumerate(self.df['test_id']):
                ax.annotate(
                    txt,
                    (x.iloc[i], y.iloc[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0
                )

            # 添加图例
            ax.legend(
                handles=legend_elements,
                title=legend_title,
                loc='best',
                frameon=True,
                framealpha=0.7
            )

            # 设置标题
            ax.set_title(title, fontweight='bold')

            # 为更好的可视化调整y轴范围
            if y_metric != 'execution_time':
                ax.set_ylim(0, 1.05)

            # 添加网格
            ax.grid(True, alpha=0.3)

        # 添加总标题
        plt.suptitle('Comparative Metric Analysis', fontweight='bold', y=0.98, fontsize=14)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_comparative_analysis.png')
        plt.close()

        print("  ✓ 图5: 比较分析已创建")

    #
    # 图6: 执行摘要仪表板
    #
    def subplot_6a_overall_score(self):
        """子图6A: 整体得分仪表盘"""
        fig, ax = plt.subplots(figsize=(6, 6))

        # 计算整体统计数据
        overall_score = self.df['overall_performance'].mean()

        # 确定整体等级
        if overall_score >= 0.9:
            grade = 'A+'
            grade_color = ACADEMIC_COLORS['green']
        elif overall_score >= 0.8:
            grade = 'A'
            grade_color = ACADEMIC_COLORS['green']
        elif overall_score >= 0.7:
            grade = 'B'
            grade_color = ACADEMIC_COLORS['olive']
        elif overall_score >= 0.6:
            grade = 'C'
            grade_color = ACADEMIC_COLORS['orange']
        else:
            grade = 'D'
            grade_color = ACADEMIC_COLORS['red']

        # 创建类似仪表盘的可视化
        gauge_angles = np.linspace(0, 180, 100) * np.pi / 180
        gauge_radius = 0.8

        # 背景弧（灰色）
        ax.plot(
            [0, 0],
            [0, gauge_radius],
            color='gray',
            alpha=0.3,
            linewidth=80,
            solid_capstyle='round'
        )

        # 前景弧（按等级着色）
        score_angle = overall_score * 180
        ax.plot(
            [0, 0],
            [0, gauge_radius],
            color=grade_color,
            alpha=0.7,
            linewidth=80,
            solid_capstyle='round',
            clip_on=False,
            zorder=5
        )

        # 添加分数文本
        ax.text(
            0, 0,
            f"{overall_score:.2f}",
            ha='center',
            va='center',
            fontsize=36,
            fontweight='bold',
            color='black'
        )

        # 添加等级
        ax.text(
            0, gauge_radius / 2.5,
            f"Grade: {grade}",
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            color=grade_color
        )

        # 移除坐标轴
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.2, 1)

        # 添加标题
        ax.text(
            0, -0.15,
            "Overall Performance",
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '6a_overall_score.png', dpi=1200)
        plt.close()

        print("  ✓ 子图6A: 整体得分仪表盘")

    def subplot_6b_capability_scores(self):
        """子图6B: 关键能力得分"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # 计算整体统计数据
        kg_score = self.df['kg_cot_score'].mean()
        autonomy_score = self.df['autonomy_score'].mean()
        tool_score = self.df['tool_score'].mean()
        answer_score = self.df['answer_quality'].mean()

        # 创建水平条形图
        capabilities = ['KG Navigation', 'Autonomy', 'Tool Usage', 'Answer Quality']
        scores = [kg_score, autonomy_score, tool_score, answer_score]
        colors = [ACADEMIC_COLORS['blue'], ACADEMIC_COLORS['purple'],
                  ACADEMIC_COLORS['orange'], ACADEMIC_COLORS['green']]

        y_pos = np.arange(len(capabilities))

        bars = ax.barh(
            y_pos,
            scores,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8,
            height=0.5
        )

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

        # 样式设置
        ax.set_yticks(y_pos)
        ax.set_yticklabels(capabilities)
        ax.set_xlim(0, 1.05)
        ax.set_title('Key Capability Scores', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 添加等级阈值
        for threshold, grade, color in [(0.9, 'A', ACADEMIC_COLORS['green']),
                                        (0.8, 'B', ACADEMIC_COLORS['olive']),
                                        (0.7, 'C', ACADEMIC_COLORS['orange']),
                                        (0.6, 'D', ACADEMIC_COLORS['red'])]:
            ax.axvline(x=threshold, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(
                threshold, -0.6,
                grade,
                ha='center',
                va='center',
                color=color,
                fontsize=8
            )

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '6b_capability_scores.png', dpi=1200)
        plt.close()

        print("  ✓ 子图6B: 关键能力得分")

    def subplot_6c_performance_matrix(self):
        """子图6C: 按类别和复杂度的性能矩阵"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 创建类似热图的可视化，带有实际值
        cat_comp_data = self.df.pivot_table(
            values='overall_performance',
            index='category_display',
            columns='complexity',
            aggfunc='mean'
        )

        # 用空字符串填充NaN，以获得更清晰的可视化
        cat_comp_data = cat_comp_data.fillna('')

        # 创建自定义配色函数
        def get_color(val):
            if val == '':
                return 'white'
            if val >= 0.9:
                return ACADEMIC_COLORS['green']
            elif val >= 0.8:
                return ACADEMIC_COLORS['olive']
            elif val >= 0.7:
                return ACADEMIC_COLORS['orange']
            elif val >= 0.6:
                return ACADEMIC_COLORS['red']
            else:
                return 'darkred'

        # 创建文本颜色函数
        def get_text_color(val):
            if val == '' or val < 0.7:
                return 'white'
            else:
                return 'black'

        # 获取单元格颜色
        cell_colors = [[get_color(val) for val in row] for row in cat_comp_data.values]

        # 创建表格
        table = ax.table(
            cellText=[[f'{val:.2f}' if val != '' else '' for val in row] for row in cat_comp_data.values],
            rowLabels=cat_comp_data.index,
            colLabels=[f'Level {c}' for c in cat_comp_data.columns],
            cellColours=cell_colors,
            loc='center',
            cellLoc='center'
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 自定义单元格文本颜色
        for (i, j), cell in table.get_celld().items():
            if i > 0 and j > 0:  # 跳过表头
                val = cat_comp_data.iloc[i - 1, j - 1]
                if val != '':
                    cell.get_text().set_color(get_text_color(val))
                    if val >= 0.8:
                        cell.get_text().set_fontweight('bold')

        # 移除坐标轴
        ax.axis('off')

        # 添加标题
        ax.set_title('Performance Matrix by Category and Complexity', fontweight='bold', pad=10)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '6c_performance_matrix.png', dpi=1200)
        plt.close()

        print("  ✓ 子图6C: 按类别和复杂度的性能矩阵")

    def subplot_6d_key_statistics(self):
        """子图6D: 关键性能统计数据"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 计算关键统计数据
        avg_time = self.df['execution_time'].mean()
        avg_queries = self.df['total_queries'].mean()
        success_rate = self.df['query_efficiency'].mean() * 100
        test_count = len(self.df)
        category_count = len(self.df['category'].unique())
        complexity_range = f"{min(self.df['complexity'])} - {max(self.df['complexity'])}"

        # 找出表现最好和最差的测试
        best_idx = self.df['overall_performance'].idxmax()
        worst_idx = self.df['overall_performance'].idxmin()

        best_test = self.df.loc[best_idx]
        worst_test = self.df.loc[worst_idx]

        # 创建统计文本
        stats_text = [
            f"▶ Tests Completed: {test_count} across {category_count} categories (Complexity levels: {complexity_range})",
            f"▶ Average Execution Time: {avg_time:.1f} seconds with {avg_queries:.1f} average queries per test",
            f"▶ Query Success Rate: {success_rate:.1f}%",
            f"▶ Best Performing Test: {best_test['test_id']} ({best_test['category_display']}, Level {best_test['complexity']}) - Score: {best_test['overall_performance']:.2f}",
            f"▶ Most Challenging Test: {worst_test['test_id']} ({worst_test['category_display']}, Level {worst_test['complexity']}) - Score: {worst_test['overall_performance']:.2f}"
        ]

        # 获取用于顶级指标分析的数值列
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # 过滤不需要的指标
        exclude_metrics = ['complexity', 'iteration_rounds', 'total_queries', 'execution_time',
                           'successful_queries', 'query_semantic_complexity']
        metric_cols = [col for col in numeric_cols if col not in exclude_metrics]

        if metric_cols:
            # 计算每个指标的均值
            metric_means = {col: self.df[col].mean() for col in metric_cols}

            # 按均值排序指标
            sorted_metrics = sorted(metric_means.items(), key=lambda x: x[1], reverse=True)

            # 获取前3名和后3名指标
            top_metrics = sorted_metrics[:3]
            bottom_metrics = sorted_metrics[-3:]

            # 格式化显示
            strengths = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in top_metrics]
            weaknesses = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in bottom_metrics]

            stats_text.append(f"▶ Top Strengths: {', '.join(strengths)}")
            stats_text.append(f"▶ Areas for Improvement: {', '.join(weaknesses)}")

        # 添加文本
        ax.text(
            0.5, 0.5,
            '\n'.join(stats_text),
            ha='center',
            va='center',
            fontsize=11,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='#f0f0f0',
                edgecolor='gray',
                alpha=0.7
            )
        )

        # 移除坐标轴
        ax.axis('off')

        # 添加标题
        ax.set_title('Key Performance Statistics', fontweight='bold', pad=10)

        # 保存图表
        plt.tight_layout()
        plt.savefig(self.subplots_dir / '6d_key_statistics.png', dpi=1200)
        plt.close()

        print("  ✓ 子图6D: 关键性能统计数据")

    def fig6_executive_summary(self):
        """
        图6: 执行摘要仪表板
        - 包含关键指标的组合可视化
        - 按类别和复杂度的性能
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.3, 1], width_ratios=[1.2, 1, 1])

        # 计算整体统计数据
        overall_score = self.df['overall_performance'].mean()
        kg_score = self.df['kg_cot_score'].mean()
        autonomy_score = self.df['autonomy_score'].mean()
        tool_score = self.df['tool_score'].mean()
        answer_score = self.df['answer_quality'].mean()

        # 确定整体等级
        if overall_score >= 0.9:
            grade = 'A+'
            grade_color = ACADEMIC_COLORS['green']
        elif overall_score >= 0.8:
            grade = 'A'
            grade_color = ACADEMIC_COLORS['green']
        elif overall_score >= 0.7:
            grade = 'B'
            grade_color = ACADEMIC_COLORS['olive']
        elif overall_score >= 0.6:
            grade = 'C'
            grade_color = ACADEMIC_COLORS['orange']
        else:
            grade = 'D'
            grade_color = ACADEMIC_COLORS['red']

        # 面板1: 整体得分（左上）
        ax1 = fig.add_subplot(gs[0, 0])

        # 创建类似仪表盘的可视化
        gauge_angles = np.linspace(0, 180, 100) * np.pi / 180
        gauge_radius = 0.8

        # 背景弧（灰色）
        ax1.plot(
            [0, 0],
            [0, gauge_radius],
            color='gray',
            alpha=0.3,
            linewidth=80,
            solid_capstyle='round'
        )

        # 前景弧（按等级着色）
        score_angle = overall_score * 180
        ax1.plot(
            [0, 0],
            [0, gauge_radius],
            color=grade_color,
            alpha=0.7,
            linewidth=80,
            solid_capstyle='round',
            clip_on=False,
            zorder=5
        )

        # 添加分数文本
        ax1.text(
            0, 0,
            f"{overall_score:.2f}",
            ha='center',
            va='center',
            fontsize=36,
            fontweight='bold',
            color='black'
        )

        # 添加等级
        ax1.text(
            0, gauge_radius / 2.5,
            f"Grade: {grade}",
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            color=grade_color
        )

        # 移除坐标轴
        ax1.axis('off')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-0.2, 1)

        # 添加标题
        ax1.text(
            0, -0.15,
            "Overall Performance",
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )

        # 面板2: 关键能力得分（上中和上右）
        ax2 = fig.add_subplot(gs[0, 1:])

        # 创建水平条形图
        capabilities = ['KG Navigation', 'Autonomy', 'Tool Usage', 'Answer Quality']
        scores = [kg_score, autonomy_score, tool_score, answer_score]
        colors = [ACADEMIC_COLORS['blue'], ACADEMIC_COLORS['purple'],
                  ACADEMIC_COLORS['orange'], ACADEMIC_COLORS['green']]

        y_pos = np.arange(len(capabilities))

        bars = ax2.barh(
            y_pos,
            scores,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8,
            height=0.5
        )

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

        # 样式设置
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(capabilities)
        ax2.set_xlim(0, 1.05)
        ax2.set_title('Key Capability Scores', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 添加等级阈值
        for threshold, grade, color in [(0.9, 'A', ACADEMIC_COLORS['green']),
                                        (0.8, 'B', ACADEMIC_COLORS['olive']),
                                        (0.7, 'C', ACADEMIC_COLORS['orange']),
                                        (0.6, 'D', ACADEMIC_COLORS['red'])]:
            ax2.axvline(x=threshold, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(
                threshold, -0.6,
                grade,
                ha='center',
                va='center',
                color=color,
                fontsize=8
            )

        # 面板3: 按类别和复杂度的性能（中间行）
        ax3 = fig.add_subplot(gs[1, :])

        # 创建类似热图的可视化，带有实际值
        cat_comp_data = self.df.pivot_table(
            values='overall_performance',
            index='category_display',
            columns='complexity',
            aggfunc='mean'
        )

        # 用空字符串填充NaN，以获得更清晰的可视化
        cat_comp_data = cat_comp_data.fillna('')

        # 创建自定义配色函数
        def get_color(val):
            if val == '':
                return 'white'
            if val >= 0.9:
                return ACADEMIC_COLORS['green']
            elif val >= 0.8:
                return ACADEMIC_COLORS['olive']
            elif val >= 0.7:
                return ACADEMIC_COLORS['orange']
            elif val >= 0.6:
                return ACADEMIC_COLORS['red']
            else:
                return 'darkred'

        # 创建文本颜色函数
        def get_text_color(val):
            if val == '' or val < 0.7:
                return 'white'
            else:
                return 'black'

        # 获取单元格颜色
        cell_colors = [[get_color(val) for val in row] for row in cat_comp_data.values]

        # 创建表格
        table = ax3.table(
            cellText=[[f'{val:.2f}' if val != '' else '' for val in row] for row in cat_comp_data.values],
            rowLabels=cat_comp_data.index,
            colLabels=[f'Level {c}' for c in cat_comp_data.columns],
            cellColours=cell_colors,
            loc='center',
            cellLoc='center'
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 自定义单元格文本颜色
        for (i, j), cell in table.get_celld().items():
            if i > 0 and j > 0:  # 跳过表头
                val = cat_comp_data.iloc[i - 1, j - 1]
                if val != '':
                    cell.get_text().set_color(get_text_color(val))
                    if val >= 0.8:
                        cell.get_text().set_fontweight('bold')

        # 移除坐标轴
        ax3.axis('off')

        # 添加标题
        ax3.set_title('Performance Matrix by Category and Complexity', fontweight='bold', pad=10)

        # 面板4: 关键统计数据（底行）
        ax4 = fig.add_subplot(gs[2, :])

        # 计算关键统计数据
        avg_time = self.df['execution_time'].mean()
        avg_queries = self.df['total_queries'].mean()
        success_rate = self.df['query_efficiency'].mean() * 100
        test_count = len(self.df)
        category_count = len(self.df['category'].unique())
        complexity_range = f"{min(self.df['complexity'])} - {max(self.df['complexity'])}"

        # 找出表现最好和最差的测试
        best_idx = self.df['overall_performance'].idxmax()
        worst_idx = self.df['overall_performance'].idxmin()

        best_test = self.df.loc[best_idx]
        worst_test = self.df.loc[worst_idx]

        # 创建统计文本
        stats_text = [
            f"▶ Tests Completed: {test_count} across {category_count} categories (Complexity levels: {complexity_range})",
            f"▶ Average Execution Time: {avg_time:.1f} seconds with {avg_queries:.1f} average queries per test",
            f"▶ Query Success Rate: {success_rate:.1f}%",
            f"▶ Best Performing Test: {best_test['test_id']} ({best_test['category_display']}, Level {best_test['complexity']}) - Score: {best_test['overall_performance']:.2f}",
            f"▶ Most Challenging Test: {worst_test['test_id']} ({worst_test['category_display']}, Level {worst_test['complexity']}) - Score: {worst_test['overall_performance']:.2f}"
        ]

        # 获取用于顶级指标分析的数值列
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # 过滤不需要的指标
        exclude_metrics = ['complexity', 'iteration_rounds', 'total_queries', 'execution_time',
                           'successful_queries', 'query_semantic_complexity']
        metric_cols = [col for col in numeric_cols if col not in exclude_metrics]

        if metric_cols:
            # 计算每个指标的均值
            metric_means = {col: self.df[col].mean() for col in metric_cols}

            # 按均值排序指标
            sorted_metrics = sorted(metric_means.items(), key=lambda x: x[1], reverse=True)

            # 获取前3名和后3名指标
            top_metrics = sorted_metrics[:3]
            bottom_metrics = sorted_metrics[-3:]

            # 格式化显示
            strengths = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in top_metrics]
            weaknesses = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in bottom_metrics]

            stats_text.append(f"▶ Top Strengths: {', '.join(strengths)}")
            stats_text.append(f"▶ Areas for Improvement: {', '.join(weaknesses)}")

        # 添加文本
        ax4.text(
            0.5, 0.5,
            '\n'.join(stats_text),
            ha='center',
            va='center',
            fontsize=11,
            transform=ax4.transAxes,
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='#f0f0f0',
                edgecolor='gray',
                alpha=0.7
            )
        )

        # 移除坐标轴
        ax4.axis('off')

        # 添加标题
        ax4.set_title('Key Performance Statistics', fontweight='bold', pad=10)

        # 添加总标题
        plt.suptitle('KGAgent V7 Performance Dashboard', fontweight='bold', y=0.98, fontsize=16)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_executive_summary.png')
        plt.close()

        print("  ✓ 图6: 执行摘要仪表板已创建")

    # 运行可视化器的入口点
def main():
    """在CSV数据上运行可视化器的主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='为KGAgent V7评估结果生成增强版可视化')
    parser.add_argument('--input', default='improved_evaluation_results.csv', help='输入CSV文件路径')
    parser.add_argument('--output', default='enhanced_figures', help='输出目录')
    parser.add_argument('--subplots-only', action='store_true', help='仅生成单独的子图')
    parser.add_argument('--figs-only', action='store_true', help='仅生成组合图')

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("KGAgent V7 增强版可视化生成器")
    print(f"{'=' * 60}")

    # 创建可视化器
    visualizer = EnhancedVisualizer(args.input, args.output)

    # 根据参数生成图表
    if args.subplots_only:
        visualizer.create_all_subplots()
    elif args.figs_only:
        visualizer.create_all_figures()
    else:
        # 默认生成所有图表
        visualizer.create_all_figures()
        visualizer.create_all_subplots()

    print(f"\n{'=' * 60}")
    print(f"✅ 所有可视化已保存到: {args.output}/")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()