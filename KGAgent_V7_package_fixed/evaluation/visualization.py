#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 评估结果可视化模块
生成7个独立的高质量图表用于学术论文展示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置全局绘图参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 定义配色方案
COLOR_PALETTE = {
    'primary': '#2E86AB',  # 深蓝
    'secondary': '#A23B72',  # 紫红
    'tertiary': '#F18F01',  # 橙色
    'quaternary': '#C73E1D',  # 红色
    'success': '#4CAF50',  # 绿色
    'neutral': '#757575',  # 灰色
    'light': '#E8E8E8',  # 浅灰
    'dark': '#263238'  # 深灰
}

CATEGORY_COLORS = {
    'kg_navigation': '#2E86AB',
    'reasoning': '#A23B72',
    'tool_use': '#F18F01',
    'complex': '#C73E1D'
}


def generate_sample_data(n_tests: int = 20, seed: int = 42) -> pd.DataFrame:
    """生成示例评估数据用于可视化演示"""
    np.random.seed(seed)

    # 设计测试分布
    test_distribution = []
    categories = ['kg_navigation', 'reasoning', 'tool_use', 'complex']

    # 确保每个类别和复杂度都有覆盖
    for complexity in range(1, 6):
        for category in categories:
            if (complexity <= 2 and category == 'kg_navigation') or \
                    (complexity == 3 and category == 'reasoning') or \
                    (complexity >= 4 and category in ['tool_use', 'complex']):
                test_distribution.append((category, complexity))

    # 补充到n_tests个测试
    while len(test_distribution) < n_tests:
        test_distribution.append((
            np.random.choice(categories),
            np.random.choice(range(1, 6))
        ))

    test_distribution = test_distribution[:n_tests]

    data = []
    for idx, (category, complexity) in enumerate(test_distribution):
        # 基于复杂度和类别生成合理的指标值
        base_performance = 0.9 - (complexity - 1) * 0.12
        category_modifier = {
            'kg_navigation': 0.05,
            'reasoning': 0,
            'tool_use': -0.05,
            'complex': -0.1
        }[category]

        base_score = np.clip(base_performance + category_modifier, 0.3, 0.95)

        record = {
            'test_id': f'test_{idx + 1:02d}',
            'category': category,
            'complexity': complexity,

            # KG+CoT驱动指标
            'schema_utilization_rate': np.clip(
                base_score + np.random.normal(0, 0.1), 0.2, 1.0
            ),
            'query_complexity_score': complexity * 1.8 + np.random.uniform(-0.5, 0.5),
            'cot_depth': max(1, complexity + np.random.randint(-1, 2)),
            'planning_quality': np.clip(
                base_score + np.random.normal(0.05, 0.1), 0.3, 1.0
            ),

            # 自主分析推理指标
            'autonomy_score': np.clip(
                base_score + np.random.normal(0, 0.08), 0.3, 1.0
            ),
            'reasoning_steps': max(1, complexity * 2 + np.random.randint(-1, 3)),
            'reflection_quality': np.clip(
                base_score + np.random.normal(0, 0.12), 0.2, 1.0
            ),
            'adaptation_rate': np.clip(
                base_score + np.random.normal(0.05, 0.1), 0.4, 1.0
            ),
            'problem_decomposition': np.clip(
                base_score + np.random.normal(0, 0.1), 0.3, 1.0
            ),

            # 工具调用指标
            'tool_usage_rate': np.clip(
                (0.2 if complexity <= 2 else 0.8) + np.random.normal(0, 0.15),
                0, 1.0
            ),
            'tool_selection_accuracy': np.clip(
                base_score + np.random.normal(0, 0.1), 0.5, 1.0
            ) if category in ['tool_use', 'complex'] else np.clip(
                0.7 + np.random.normal(0, 0.1), 0.5, 1.0
            ),
            'mismatch_computation': (
                    (category in ['tool_use', 'complex'] or complexity >= 4)
                    and np.random.random() > 0.4
            ),
            'stats_computation': (
                    category in ['reasoning', 'complex']
                    and np.random.random() > 0.5
            ),

            # 性能指标
            'execution_time': max(0.5, complexity * 1.2 + np.random.exponential(0.8)),
            'total_queries': max(1, complexity * 3 + np.random.randint(-2, 4)),
            'successful_queries': 0,  # 将在后面计算
            'iteration_rounds': min(5, max(1, complexity - 1 + np.random.randint(0, 2))),
            'final_answer_quality': np.clip(
                base_score + np.random.normal(0, 0.1), 0.3, 1.0
            )
        }

        # 计算成功查询数（基于base_score）
        success_rate = np.clip(base_score + np.random.normal(0, 0.05), 0.6, 0.98)
        record['successful_queries'] = int(record['total_queries'] * success_rate)

        data.append(record)

    return pd.DataFrame(data)


class AdvancedVisualizer:
    """高级可视化生成器"""

    def __init__(self, df: pd.DataFrame = None, output_dir: str = "evaluation_figures"):
        """
        初始化可视化器

        Args:
            df: 评估数据DataFrame，如果为None则生成示例数据
            output_dir: 输出目录路径
        """
        self.df = df if df is not None else generate_sample_data(20)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 预计算一些常用指标
        self._precompute_metrics()

    def _precompute_metrics(self):
        """预计算常用指标"""
        # 计算成功率
        self.df['query_success_rate'] = (
                self.df['successful_queries'] / self.df['total_queries'].clip(lower=1)
        )

        # 计算综合得分
        self.df['overall_score'] = (
                self.df['autonomy_score'] * 0.3 +
                self.df['planning_quality'] * 0.2 +
                self.df['tool_selection_accuracy'] * 0.2 +
                self.df['final_answer_quality'] * 0.3
        )

        # 计算效率得分（反向指标，时间越短越好）
        max_time = self.df['execution_time'].max()
        self.df['efficiency_score'] = 1 - (self.df['execution_time'] / max_time)

    def generate_all_figures(self):
        """生成所有7个评估图表"""
        print("\n" + "=" * 60)
        print("Generating KGAgent V7 Evaluation Figures")
        print("=" * 60)

        self.fig1_kg_cot_capabilities()
        self.fig2_autonomous_reasoning()
        self.fig3_tool_utilization()
        self.fig4_performance_scaling()
        self.fig5_category_analysis()
        self.fig6_comprehensive_heatmap()
        self.fig7_summary_dashboard()

        print("\n" + "=" * 60)
        print(f"✅ All 7 figures successfully generated in: {self.output_dir}")
        print("=" * 60)

    def fig1_kg_cot_capabilities(self):
        """Figure 1: Knowledge Graph + Chain-of-Thought Capabilities"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Figure 1: Knowledge Graph + Chain-of-Thought Capability Assessment',
                     fontsize=15, fontweight='bold', y=1.02)

        # Subplot 1: Schema Utilization by Category
        ax1 = plt.subplot(2, 2, 1)
        category_schema = self.df.groupby('category')['schema_utilization_rate'].agg(['mean', 'std'])
        categories = [c.replace('_', ' ').title() for c in category_schema.index]
        colors = [CATEGORY_COLORS[cat] for cat in category_schema.index]

        bars = ax1.bar(categories, category_schema['mean'],
                       yerr=category_schema['std'], capsize=5,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_ylabel('Schema Utilization Rate', fontweight='bold')
        ax1.set_title('(a) Schema Utilization by Test Category', fontweight='bold', pad=10)
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, category_schema['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        # Subplot 2: Query Complexity vs CoT Depth
        ax2 = plt.subplot(2, 2, 2)
        scatter = ax2.scatter(self.df['query_complexity_score'],
                              self.df['cot_depth'],
                              c=self.df['complexity'],
                              s=150, alpha=0.7,
                              cmap='viridis',
                              edgecolors='black',
                              linewidth=1)

        # Add trend line
        z = np.polyfit(self.df['query_complexity_score'], self.df['cot_depth'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df['query_complexity_score'].min(),
                              self.df['query_complexity_score'].max(), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.6, linewidth=2,
                 label=f'Trend (R²={np.corrcoef(self.df["query_complexity_score"], self.df["cot_depth"])[0, 1] ** 2:.3f})')

        ax2.set_xlabel('Query Complexity Score', fontweight='bold')
        ax2.set_ylabel('Chain-of-Thought Depth', fontweight='bold')
        ax2.set_title('(b) Query Complexity vs Reasoning Depth', fontweight='bold', pad=10)
        ax2.legend(loc='upper left')

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Test Complexity Level', rotation=270, labelpad=20)

        # Subplot 3: Planning Quality Heatmap
        ax3 = plt.subplot(2, 2, 3)
        pivot_planning = self.df.pivot_table(
            values='planning_quality',
            index='category',
            columns='complexity',
            aggfunc='mean',
            fill_value=0
        )

        im = ax3.imshow(pivot_planning.values, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=1, interpolation='nearest')

        ax3.set_xticks(np.arange(len(pivot_planning.columns)))
        ax3.set_yticks(np.arange(len(pivot_planning.index)))
        ax3.set_xticklabels([f'L{c}' for c in pivot_planning.columns])
        ax3.set_yticklabels([c.replace('_', ' ').title() for c in pivot_planning.index])
        ax3.set_xlabel('Complexity Level', fontweight='bold')
        ax3.set_ylabel('Test Category', fontweight='bold')
        ax3.set_title('(c) Planning Quality Matrix', fontweight='bold', pad=10)

        # Add text annotations
        for i in range(len(pivot_planning.index)):
            for j in range(len(pivot_planning.columns)):
                value = pivot_planning.values[i, j]
                if value > 0:
                    color = 'white' if value < 0.5 else 'black'
                    ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                             color=color, fontweight='bold')

        plt.colorbar(im, ax=ax3, label='Planning Quality Score')

        # Subplot 4: KG+CoT Capability Radar
        ax4 = plt.subplot(2, 2, 4, projection='polar')

        metrics = ['Schema\nUtilization', 'Query\nComplexity', 'Planning\nQuality',
                   'CoT Depth', 'Adaptation\nRate']

        # Normalize metrics to 0-1 scale
        values = [
            self.df['schema_utilization_rate'].mean(),
            min(self.df['query_complexity_score'].mean() / 10, 1),
            self.df['planning_quality'].mean(),
            min(self.df['cot_depth'].mean() / 5, 1),
            self.df['adaptation_rate'].mean()
        ]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]

        ax4.plot(angles_plot, values_plot, 'o-', linewidth=2.5,
                 color=COLOR_PALETTE['primary'], markersize=10)
        ax4.fill(angles_plot, values_plot, alpha=0.25, color=COLOR_PALETTE['primary'])

        ax4.set_xticks(angles)
        ax4.set_xticklabels(metrics, fontweight='bold', size=10)
        ax4.set_ylim([0, 1])
        ax4.set_title('(d) KG+CoT Capability Overview', fontweight='bold', pad=20)
        ax4.grid(True, linestyle='--', alpha=0.5)

        # Add value labels
        for angle, value, metric in zip(angles, values, metrics):
            ax4.text(angle, value + 0.08, f'{value:.2f}',
                     ha='center', va='center', fontweight='bold', size=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_kg_cot_capabilities.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 1: KG+CoT capabilities assessment")

    def fig2_autonomous_reasoning(self):
        """Figure 2: Autonomous Analysis and Reasoning Capabilities"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Figure 2: Autonomous Analysis and Reasoning Assessment',
                     fontsize=15, fontweight='bold', y=1.02)

        # Subplot 1: Autonomy Score Distribution
        ax1 = plt.subplot(2, 2, 1)

        # Create histogram with KDE overlay
        n, bins, patches = ax1.hist(self.df['autonomy_score'], bins=12,
                                    color=COLOR_PALETTE['secondary'],
                                    alpha=0.7, edgecolor='black', linewidth=1)

        # Add KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(self.df['autonomy_score'])
        x_range = np.linspace(0, 1, 100)
        kde_values = kde(x_range) * len(self.df) * (bins[1] - bins[0])
        ax1.plot(x_range, kde_values, 'k-', linewidth=2.5, label='KDE')

        # Add mean and std lines
        mean_val = self.df['autonomy_score'].mean()
        std_val = self.df['autonomy_score'].std()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.3f}')
        ax1.axvspan(mean_val - std_val, mean_val + std_val,
                    alpha=0.2, color='red', label=f'±1σ: {std_val:.3f}')

        ax1.set_xlabel('Autonomy Score', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('(a) Distribution of Autonomy Scores', fontweight='bold', pad=10)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)

        # Subplot 2: Reasoning Steps vs Adaptation Rate
        ax2 = plt.subplot(2, 2, 2)

        for complexity in sorted(self.df['complexity'].unique()):
            mask = self.df['complexity'] == complexity
            data = self.df[mask]
            ax2.scatter(data['reasoning_steps'], data['adaptation_rate'],
                        s=120, alpha=0.7, label=f'Level {complexity}',
                        edgecolors='black', linewidth=1)

        # Add regression line
        x = self.df['reasoning_steps']
        y = self.df['adaptation_rate']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)

        ax2.set_xlabel('Reasoning Steps', fontweight='bold')
        ax2.set_ylabel('Adaptation Rate', fontweight='bold')
        ax2.set_title('(b) Reasoning Depth vs Adaptation Capability', fontweight='bold', pad=10)
        ax2.legend(title='Complexity', loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Reflection Quality by Category
        ax3 = plt.subplot(2, 2, 3)

        categories = self.df['category'].unique()
        data_boxplot = [self.df[self.df['category'] == cat]['reflection_quality'].values
                        for cat in categories]

        bp = ax3.boxplot(data_boxplot,
                         labels=[c.replace('_', ' ').title() for c in categories],
                         patch_artist=True, notch=True, showmeans=True,
                         boxprops=dict(linewidth=1.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='red'),
                         meanprops=dict(marker='D', markerfacecolor='blue', markersize=8))

        # Color the boxes
        for patch, cat in zip(bp['boxes'], categories):
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.7)

        ax3.set_ylabel('Reflection Quality Score', fontweight='bold')
        ax3.set_title('(c) Reflection Quality Distribution by Category', fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Subplot 4: Iteration Analysis
        ax4 = plt.subplot(2, 2, 4)

        iteration_stats = self.df['iteration_rounds'].value_counts().sort_index()
        colors_iter = plt.cm.viridis(np.linspace(0.3, 0.9, len(iteration_stats)))

        bars = ax4.bar(iteration_stats.index, iteration_stats.values,
                       color=colors_iter, edgecolor='black', linewidth=1.5)

        # Add percentage labels
        total = len(self.df)
        for bar, count in zip(bars, iteration_stats.values):
            height = bar.get_height()
            percentage = count / total * 100
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{count}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontweight='bold')

        ax4.set_xlabel('Number of Iteration Rounds', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('(d) Distribution of Iteration Rounds', fontweight='bold', pad=10)
        ax4.set_xticks(iteration_stats.index)
        ax4.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_iter = self.df['iteration_rounds'].mean()
        ax4.axvline(mean_iter, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_iter:.1f}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_autonomous_reasoning.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 2: Autonomous reasoning assessment")

    def fig3_tool_utilization(self):
        """Figure 3: Tool Utilization and Selection Capabilities"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Figure 3: Tool Utilization and Selection Assessment',
                     fontsize=15, fontweight='bold', y=1.02)

        # Subplot 1: Tool Usage Metrics Comparison
        ax1 = plt.subplot(2, 2, 1)

        tool_metrics = self.df.groupby('category')[
            ['tool_usage_rate', 'tool_selection_accuracy']
        ].mean()

        x = np.arange(len(tool_metrics))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, tool_metrics['tool_usage_rate'], width,
                        label='Usage Rate', color=COLOR_PALETTE['tertiary'],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width / 2, tool_metrics['tool_selection_accuracy'], width,
                        label='Selection Accuracy', color=COLOR_PALETTE['quaternary'],
                        alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('Test Category', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(a) Tool Usage Metrics by Category', fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace('_', ' ').title() for c in tool_metrics.index])
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        # Subplot 2: Mismatch Index Usage
        ax2 = plt.subplot(2, 2, 2)

        mismatch_data = self.df['mismatch_computation'].value_counts()
        colors_pie = [COLOR_PALETTE['light'], COLOR_PALETTE['success']]
        explode = (0.05, 0.1)

        wedges, texts, autotexts = ax2.pie(
            mismatch_data.values,
            labels=['Not Used', 'Used'] if False in mismatch_data.index else ['Used', 'Not Used'],
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={'fontweight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)

        ax2.set_title('(b) Mismatch Index Computation Usage', fontweight='bold', pad=10)

        # Subplot 3: Tool Usage vs Complexity
        ax3 = plt.subplot(2, 2, 3)

        complexity_tools = self.df.groupby('complexity').agg({
            'tool_usage_rate': 'mean',
            'tool_selection_accuracy': 'mean',
            'mismatch_computation': 'mean'
        })

        ax3.plot(complexity_tools.index, complexity_tools['tool_usage_rate'],
                 'o-', label='Usage Rate', linewidth=2.5, markersize=10,
                 color=COLOR_PALETTE['tertiary'])
        ax3.plot(complexity_tools.index, complexity_tools['tool_selection_accuracy'],
                 's-', label='Selection Accuracy', linewidth=2.5, markersize=10,
                 color=COLOR_PALETTE['quaternary'])
        ax3.plot(complexity_tools.index, complexity_tools['mismatch_computation'],
                 '^--', label='Mismatch Usage', linewidth=2, markersize=9,
                 color=COLOR_PALETTE['success'], alpha=0.7)

        ax3.set_xlabel('Test Complexity Level', fontweight='bold')
        ax3.set_ylabel('Rate / Score', fontweight='bold')
        ax3.set_title('(c) Tool Metrics vs Complexity', fontweight='bold', pad=10)
        ax3.set_xticks(complexity_tools.index)
        ax3.set_xticklabels([f'L{c}' for c in complexity_tools.index])
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Subplot 4: Tool Selection Accuracy Heatmap
        ax4 = plt.subplot(2, 2, 4)

        accuracy_matrix = self.df.pivot_table(
            values='tool_selection_accuracy',
            index='category',
            columns='complexity',
            aggfunc='mean',
            fill_value=0
        )

        im = sns.heatmap(accuracy_matrix, annot=True, fmt='.2f',
                         cmap='RdYlGn', vmin=0, vmax=1,
                         cbar_kws={'label': 'Accuracy Score'},
                         linewidths=1, linecolor='white',
                         square=True, ax=ax4)

        ax4.set_xlabel('Complexity Level', fontweight='bold')
        ax4.set_ylabel('Test Category', fontweight='bold')
        ax4.set_title('(d) Tool Selection Accuracy Matrix', fontweight='bold', pad=10)
        ax4.set_xticklabels([f'L{c}' for c in accuracy_matrix.columns], rotation=0)
        ax4.set_yticklabels([c.replace('_', ' ').title() for c in accuracy_matrix.index],
                            rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_tool_utilization.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 3: Tool utilization assessment")

    def fig4_performance_scaling(self):
        """Figure 4: Performance Scaling with Complexity"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Figure 4: Performance Scaling Analysis',
                     fontsize=15, fontweight='bold', y=1.02)

        # Subplot 1: Execution Time Scaling
        ax1 = plt.subplot(2, 2, 1)

        time_stats = self.df.groupby('complexity')['execution_time'].agg(['mean', 'std', 'min', 'max'])

        ax1.errorbar(time_stats.index, time_stats['mean'],
                     yerr=time_stats['std'],
                     fmt='o-', linewidth=2.5, markersize=10,
                     capsize=5, capthick=2,
                     color=COLOR_PALETTE['primary'],
                     label='Mean ± Std')

        # Add min-max range
        ax1.fill_between(time_stats.index, time_stats['min'], time_stats['max'],
                         alpha=0.2, color=COLOR_PALETTE['primary'])

        ax1.set_xlabel('Complexity Level', fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax1.set_title('(a) Execution Time Scaling', fontweight='bold', pad=10)
        ax1.set_xticks(time_stats.index)
        ax1.set_xticklabels([f'L{c}' for c in time_stats.index])
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add trend annotation
        slope = np.polyfit(time_stats.index, time_stats['mean'], 1)[0]
        ax1.text(0.95, 0.95, f'Avg. increase: {slope:.2f}s/level',
                 transform=ax1.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Subplot 2: Success Rate Scaling
        ax2 = plt.subplot(2, 2, 2)

        success_stats = self.df.groupby('complexity')['query_success_rate'].agg(['mean', 'std'])

        bars = ax2.bar(success_stats.index, success_stats['mean'],
                       yerr=success_stats['std'],
                       color=COLOR_PALETTE['success'],
                       alpha=0.8, edgecolor='black',
                       linewidth=1.5, capsize=5)

        # Add trend line
        z = np.polyfit(success_stats.index, success_stats['mean'], 1)
        p = np.poly1d(z)
        ax2.plot(success_stats.index, p(success_stats.index),
                 'r--', alpha=0.6, linewidth=2, label='Trend')

        ax2.set_xlabel('Complexity Level', fontweight='bold')
        ax2.set_ylabel('Query Success Rate', fontweight='bold')
        ax2.set_title('(b) Success Rate vs Complexity', fontweight='bold', pad=10)
        ax2.set_ylim([0, 1.05])
        ax2.set_xticks(success_stats.index)
        ax2.set_xticklabels([f'L{c}' for c in success_stats.index])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, success_stats['mean']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

        # Subplot 3: Answer Quality by Category and Complexity
        ax3 = plt.subplot(2, 2, 3)

        for cat in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == cat].groupby('complexity')['final_answer_quality'].mean()
            ax3.plot(cat_data.index, cat_data.values, 'o-',
                     label=cat.replace('_', ' ').title(),
                     linewidth=2, markersize=8,
                     color=CATEGORY_COLORS[cat], alpha=0.8)

        ax3.set_xlabel('Complexity Level', fontweight='bold')
        ax3.set_ylabel('Answer Quality Score', fontweight='bold')
        ax3.set_title('(c) Answer Quality Trends', fontweight='bold', pad=10)
        ax3.set_xticks(sorted(self.df['complexity'].unique()))
        ax3.set_xticklabels([f'L{c}' for c in sorted(self.df['complexity'].unique())])
        ax3.legend(loc='best', ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Subplot 4: Overall Performance Score
        ax4 = plt.subplot(2, 2, 4)

        overall_stats = self.df.groupby('complexity')['overall_score'].agg(['mean', 'std'])

        bars = ax4.bar(overall_stats.index, overall_stats['mean'],
                       yerr=overall_stats['std'],
                       color=plt.cm.plasma(np.linspace(0.3, 0.8, len(overall_stats))),
                       alpha=0.8, edgecolor='black',
                       linewidth=1.5, capsize=5)

        ax4.set_xlabel('Complexity Level', fontweight='bold')
        ax4.set_ylabel('Overall Performance Score', fontweight='bold')
        ax4.set_title('(d) Comprehensive Performance Score', fontweight='bold', pad=10)
        ax4.set_ylim([0, 1.05])
        ax4.set_xticks(overall_stats.index)
        ax4.set_xticklabels([f'L{c}' for c in overall_stats.index])
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels and performance grade
        for bar, val in zip(bars, overall_stats['mean']):
            height = bar.get_height()
            # Determine grade
            if val >= 0.8:
                grade = 'A'
                color = 'green'
            elif val >= 0.7:
                grade = 'B'
                color = 'blue'
            elif val >= 0.6:
                grade = 'C'
                color = 'orange'
            else:
                grade = 'D'
                color = 'red'

            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{val:.2f}\n({grade})', ha='center', va='bottom',
                     fontweight='bold', color=color)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_performance_scaling.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 4: Performance scaling analysis")

    def fig5_category_analysis(self):
        """Figure 5: Comprehensive Category Analysis"""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Figure 5: Comprehensive Category Analysis',
                     fontsize=15, fontweight='bold', y=1.02)

        # Subplot 1: Core Metrics Comparison
        ax1 = plt.subplot(2, 3, 1)

        core_metrics = self.df.groupby('category').agg({
            'autonomy_score': 'mean',
            'tool_usage_rate': 'mean',
            'final_answer_quality': 'mean'
        })

        core_metrics.plot(kind='bar', ax=ax1, width=0.8,
                          color=[COLOR_PALETTE['primary'],
                                 COLOR_PALETTE['tertiary'],
                                 COLOR_PALETTE['success']])

        ax1.set_xlabel('Category', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(a) Core Metrics Comparison', fontweight='bold', pad=10)
        ax1.set_xticklabels([c.replace('_', ' ').title() for c in core_metrics.index],
                            rotation=45, ha='right')
        ax1.legend(['Autonomy', 'Tool Usage', 'Answer Quality'], loc='upper right')
        ax1.set_ylim([0, 1.05])
        ax1.grid(axis='y', alpha=0.3)

        # Subplot 2: Efficiency Analysis
        ax2 = plt.subplot(2, 3, 2)

        efficiency = self.df.groupby('category').agg({
            'execution_time': 'mean',
            'iteration_rounds': 'mean'
        })

        ax2_twin = ax2.twinx()

        x = np.arange(len(efficiency))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, efficiency['execution_time'], width,
                        label='Exec Time', color=COLOR_PALETTE['primary'], alpha=0.8)
        bars2 = ax2_twin.bar(x + width / 2, efficiency['iteration_rounds'], width,
                             label='Iterations', color=COLOR_PALETTE['secondary'], alpha=0.8)

        ax2.set_xlabel('Category', fontweight='bold')
        ax2.set_ylabel('Execution Time (s)', color=COLOR_PALETTE['primary'], fontweight='bold')
        ax2_twin.set_ylabel('Iteration Rounds', color=COLOR_PALETTE['secondary'], fontweight='bold')
        ax2.set_title('(b) Execution Efficiency', fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.replace('_', ' ').title() for c in efficiency.index],
                            rotation=45, ha='right')
        ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE['primary'])
        ax2_twin.tick_params(axis='y', labelcolor=COLOR_PALETTE['secondary'])

        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Subplot 3: Success Rate Comparison
        ax3 = plt.subplot(2, 3, 3)

        success_by_cat = self.df.groupby('category')['query_success_rate'].agg(['mean', 'std'])

        bars = ax3.bar(success_by_cat.index, success_by_cat['mean'],
                       yerr=success_by_cat['std'],
                       color=[CATEGORY_COLORS[cat] for cat in success_by_cat.index],
                       alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)

        ax3.set_xlabel('Category', fontweight='bold')
        ax3.set_ylabel('Query Success Rate', fontweight='bold')
        ax3.set_title('(c) Success Rate by Category', fontweight='bold', pad=10)
        ax3.set_ylim([0, 1.05])
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in success_by_cat.index],
                            rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, success_by_cat['mean']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

        # Subplot 4: Complexity Distribution
        ax4 = plt.subplot(2, 3, 4)

        complexity_dist = self.df.pivot_table(
            index='category',
            columns='complexity',
            values='test_id',
            aggfunc='count',
            fill_value=0
        )

        complexity_dist.plot(kind='bar', stacked=True, ax=ax4,
                             colormap='viridis', width=0.8)

        ax4.set_xlabel('Category', fontweight='bold')
        ax4.set_ylabel('Number of Tests', fontweight='bold')
        ax4.set_title('(d) Test Complexity Distribution', fontweight='bold', pad=10)
        ax4.set_xticklabels([c.replace('_', ' ').title() for c in complexity_dist.index],
                            rotation=45, ha='right')
        ax4.legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(axis='y', alpha=0.3)

        # Subplot 5: Reasoning Depth Analysis
        ax5 = plt.subplot(2, 3, 5)

        reasoning = self.df.groupby('category').agg({
            'cot_depth': 'mean',
            'reasoning_steps': 'mean'
        })

        x = np.arange(len(reasoning))
        width = 0.35

        bars1 = ax5.bar(x - width / 2, reasoning['cot_depth'], width,
                        label='CoT Depth', color=COLOR_PALETTE['primary'], alpha=0.8)
        bars2 = ax5.bar(x + width / 2, reasoning['reasoning_steps'], width,
                        label='Reasoning Steps', color=COLOR_PALETTE['secondary'], alpha=0.8)

        ax5.set_xlabel('Category', fontweight='bold')
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('(e) Reasoning Depth Comparison', fontweight='bold', pad=10)
        ax5.set_xticks(x)
        ax5.set_xticklabels([c.replace('_', ' ').title() for c in reasoning.index],
                            rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # Subplot 6: Performance Radar
        ax6 = plt.subplot(2, 3, 6, projection='polar')

        metrics_radar = ['Autonomy', 'Planning', 'Tool Use', 'Adaptation', 'Quality']
        metric_cols = ['autonomy_score', 'planning_quality', 'tool_usage_rate',
                       'adaptation_rate', 'final_answer_quality']

        angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]

        for cat in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == cat]
            values = [cat_data[col].mean() for col in metric_cols]
            values += values[:1]

            ax6.plot(angles, values, 'o-', linewidth=2,
                     label=cat.replace('_', ' ').title(),
                     color=CATEGORY_COLORS[cat], markersize=8)
            ax6.fill(angles, values, alpha=0.15, color=CATEGORY_COLORS[cat])

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics_radar, fontweight='bold')
        ax6.set_ylim([0, 1])
        ax6.set_title('(f) Multi-dimensional Performance', fontweight='bold', y=1.08, pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax6.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_category_analysis.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 5: Category comparison analysis")

    def fig6_comprehensive_heatmap(self):
        """Figure 6: Comprehensive Performance Heatmap"""
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Figure 6: Comprehensive Performance Heatmap',
                     fontsize=15, fontweight='bold', y=0.98)

        # Prepare metrics for heatmap
        metrics_list = [
            'schema_utilization_rate', 'query_complexity_score', 'planning_quality',
            'autonomy_score', 'reasoning_steps', 'reflection_quality',
            'adaptation_rate', 'tool_usage_rate', 'tool_selection_accuracy',
            'final_answer_quality', 'efficiency_score', 'query_success_rate'
        ]

        # Normalize metrics to 0-1 scale
        heatmap_data = []
        test_labels = []

        # Sort by category and complexity for better visualization
        df_sorted = self.df.sort_values(['category', 'complexity'])

        for _, row in df_sorted.iterrows():
            test_labels.append(f"{row['test_id']} (L{row['complexity']})")
            row_values = []
            for metric in metrics_list:
                val = row[metric]
                # Normalize specific metrics
                if metric == 'query_complexity_score':
                    val = min(val / 10, 1)
                elif metric == 'reasoning_steps':
                    val = min(val / 10, 1)
                row_values.append(val)
            heatmap_data.append(row_values)

        # Convert to numpy array and transpose
        heatmap_array = np.array(heatmap_data).T

        # Create main heatmap
        ax = plt.subplot(111)

        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(test_labels)))
        ax.set_yticks(np.arange(len(metrics_list)))
        ax.set_xticklabels(test_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics_list], fontsize=10)

        # Add grid lines
        for i in range(len(metrics_list)):
            ax.axhline(i + 0.5, color='white', linewidth=0.5)

        # Add category separators
        category_changes = []
        last_cat = df_sorted.iloc[0]['category']
        for i in range(1, len(df_sorted)):
            if df_sorted.iloc[i]['category'] != last_cat:
                category_changes.append(i - 0.5)
                last_cat = df_sorted.iloc[i]['category']

        for x in category_changes:
            ax.axvline(x, color='blue', linewidth=3, alpha=0.7)

        # Add category labels
        cat_positions = {}
        start = 0
        for cat in df_sorted['category'].unique():
            cat_data = df_sorted[df_sorted['category'] == cat]
            end = start + len(cat_data)
            cat_positions[cat] = (start + end - 1) / 2
            start = end

        for cat, pos in cat_positions.items():
            ax.text(pos, -1.5, cat.replace('_', ' ').title(),
                    ha='center', fontweight='bold', color='blue', fontsize=11)

        # Add performance indicators
        for i in range(len(test_labels)):
            avg_score = np.mean(heatmap_array[:, i])
            if avg_score >= 0.8:
                marker = '★★★'
                color = 'gold'
            elif avg_score >= 0.7:
                marker = '★★'
                color = 'silver'
            elif avg_score >= 0.6:
                marker = '★'
                color = '#CD7F32'  # Bronze
            else:
                marker = '○'
                color = 'gray'

            ax.text(i, len(metrics_list) + 0.5, marker,
                    ha='center', va='center', fontsize=8, color=color)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Performance Score (0=Poor, 1=Excellent)',
                       rotation=270, labelpad=20, fontweight='bold')

        # Add title for performance indicators
        ax.text(len(test_labels) / 2, len(metrics_list) + 0.5,
                'Performance Rating:', ha='right', va='center',
                fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_comprehensive_heatmap.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 6: Comprehensive performance heatmap")

    def fig7_summary_dashboard(self):
        """Figure 7: Executive Summary Dashboard"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Figure 7: Executive Summary Dashboard',
                     fontsize=15, fontweight='bold', y=0.98)

        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Summary Statistics Table
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.axis('tight')
        ax1.axis('off')

        # Calculate summary statistics
        summary_metrics = [
            ('KG Schema Utilization', 'schema_utilization_rate'),
            ('Query Complexity', 'query_complexity_score'),
            ('Planning Quality', 'planning_quality'),
            ('Autonomy Score', 'autonomy_score'),
            ('Tool Selection Accuracy', 'tool_selection_accuracy'),
            ('Answer Quality', 'final_answer_quality'),
            ('Execution Time (s)', 'execution_time'),
            ('Success Rate', 'query_success_rate'),
            ('Overall Score', 'overall_score')
        ]

        table_data = []
        for name, metric in summary_metrics:
            if metric in self.df.columns:
                mean_val = self.df[metric].mean()
                std_val = self.df[metric].std()
                min_val = self.df[metric].min()
                max_val = self.df[metric].max()

                # Find best performing category
                best_cat = self.df.groupby('category')[metric].mean().idxmax()
                best_cat = best_cat.replace('_', ' ').title()

                # Format values
                if metric == 'execution_time':
                    row = [name, f'{mean_val:.2f}', f'{std_val:.2f}',
                           f'{min_val:.2f}', f'{max_val:.2f}', best_cat]
                elif metric == 'query_complexity_score':
                    row = [name, f'{mean_val:.1f}', f'{std_val:.1f}',
                           f'{min_val:.1f}', f'{max_val:.1f}', best_cat]
                else:
                    row = [name, f'{mean_val:.3f}', f'{std_val:.3f}',
                           f'{min_val:.3f}', f'{max_val:.3f}', best_cat]

                table_data.append(row)

        # Create table
        col_labels = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Best Category']
        table = ax1.table(cellText=table_data,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(len(col_labels)):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor(COLOR_PALETTE['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    else:
                        cell.set_facecolor('white')

                    # Highlight good values
                    if j == 1 and i > 0:  # Mean column
                        val_str = table_data[i - 1][1]
                        try:
                            val = float(val_str)
                            if table_data[i - 1][0] == 'Overall Score' and val > 0.7:
                                cell.set_text_props(weight='bold', color='green')
                            elif table_data[i - 1][0] != 'Execution Time (s)' and val > 0.8:
                                cell.set_text_props(weight='bold', color='green')
                        except:
                            pass

        ax1.set_title('Summary Statistics Table', fontweight='bold', pad=20, fontsize=13)

        # Panel 2: Key Findings
        ax2 = fig.add_subplot(gs[2, :])
        ax2.axis('off')

        # Calculate key findings
        overall_mean = self.df['overall_score'].mean()
        best_category = self.df.groupby('category')['overall_score'].mean().idxmax()
        most_efficient = self.df.groupby('category')['execution_time'].mean().idxmin()
        highest_tool_acc = self.df.groupby('category')['tool_selection_accuracy'].mean().idxmax()

        # Determine system grade
        if overall_mean >= 0.8:
            grade = 'A (Excellent)'
            grade_color = 'green'
        elif overall_mean >= 0.7:
            grade = 'B (Good)'
            grade_color = 'blue'
        elif overall_mean >= 0.6:
            grade = 'C (Satisfactory)'
            grade_color = 'orange'
        else:
            grade = 'D (Needs Improvement)'
            grade_color = 'red'

        findings = [
            f"• Overall System Performance: {overall_mean:.3f}/1.000 - Grade: {grade}",
            f"• Best Performing Category: {best_category.replace('_', ' ').title()}",
            f"• Most Efficient Category: {most_efficient.replace('_', ' ').title()}",
            f"• Highest Tool Accuracy: {highest_tool_acc.replace('_', ' ').title()}",
            f"• Average Query Success Rate: {self.df['query_success_rate'].mean():.1%}",
            f"• Mismatch Index Usage: {self.df['mismatch_computation'].mean() * 100:.1f}% of applicable tests",
            f"• Average Iteration Rounds: {self.df['iteration_rounds'].mean():.1f}",
            f"• Tests Completed: {len(self.df)} across {len(self.df['category'].unique())} categories"
        ]

        findings_text = "Key Findings:\n\n" + "\n".join(findings)

        # Add colored background box
        bbox = FancyBboxPatch((0.02, 0.1), 0.96, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor='lightblue',
                              edgecolor=COLOR_PALETTE['primary'],
                              alpha=0.3, linewidth=2)
        ax2.add_patch(bbox)

        ax2.text(0.05, 0.5, findings_text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment='center', fontweight='bold')

        # Add grade indicator
        ax2.text(0.95, 0.5, grade, transform=ax2.transAxes,
                 fontsize=20, verticalalignment='center',
                 horizontalalignment='right',
                 color=grade_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig7_summary_dashboard.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✅ Figure 7: Executive summary dashboard")


def main():
    """Main function to generate all evaluation visualizations"""
    print("\n" + "=" * 60)
    print("KGAgent V7 Evaluation Visualization System")
    print("=" * 60)

    # Generate sample data
    print("\n📊 Generating sample evaluation data...")
    df = generate_sample_data(n_tests=25)

    # Save sample data
    output_dir = Path("evaluation_figures")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "sample_evaluation_data.csv", index=False)
    print(f"✅ Sample data saved to: {output_dir}/sample_evaluation_data.csv")

    # Create visualizer and generate all figures
    print("\n🎨 Generating visualization figures...")
    visualizer = AdvancedVisualizer(df, str(output_dir))
    visualizer.generate_all_figures()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n📈 Dataset Overview:")
    print(f"  • Total Tests: {len(df)}")
    print(f"  • Categories: {', '.join(df['category'].unique())}")
    print(f"  • Complexity Levels: {sorted(df['complexity'].unique())}")

    print(f"\n🎯 Performance Metrics:")
    print(f"  • Mean Autonomy Score: {df['autonomy_score'].mean():.3f} ± {df['autonomy_score'].std():.3f}")
    print(f"  • Mean Planning Quality: {df['planning_quality'].mean():.3f} ± {df['planning_quality'].std():.3f}")
    print(
        f"  • Mean Tool Accuracy: {df['tool_selection_accuracy'].mean():.3f} ± {df['tool_selection_accuracy'].std():.3f}")
    print(f"  • Mean Answer Quality: {df['final_answer_quality'].mean():.3f} ± {df['final_answer_quality'].std():.3f}")

    print(f"\n⚡ Efficiency Metrics:")
    print(f"  • Mean Execution Time: {df['execution_time'].mean():.2f}s ± {df['execution_time'].std():.2f}s")
    print(f"  • Mean Iteration Rounds: {df['iteration_rounds'].mean():.2f} ± {df['iteration_rounds'].std():.2f}")
    print(f"  • Query Success Rate: {df['query_success_rate'].mean():.1%}")

    # Calculate overall system score
    overall_score = df['overall_score'].mean()

    print(f"\n🏆 Overall System Assessment:")
    print(f"  • Overall Score: {overall_score:.3f}/1.000")

    # Determine grade
    if overall_score >= 0.8:
        grade = "A (Excellent)"
        emoji = "🌟"
    elif overall_score >= 0.7:
        grade = "B (Good)"
        emoji = "✨"
    elif overall_score >= 0.6:
        grade = "C (Satisfactory)"
        emoji = "👍"
    else:
        grade = "D (Needs Improvement)"
        emoji = "📈"

    print(f"  • System Grade: {grade} {emoji}")

    # Best and worst performing tests
    best_test = df.loc[df['overall_score'].idxmax()]
    worst_test = df.loc[df['overall_score'].idxmin()]

    print(f"\n🎖️ Performance Extremes:")
    print(
        f"  • Best Test: {best_test['test_id']} (Category: {best_test['category']}, Score: {best_test['overall_score']:.3f})")
    print(
        f"  • Worst Test: {worst_test['test_id']} (Category: {worst_test['category']}, Score: {worst_test['overall_score']:.3f})")

    # Category rankings
    cat_rankings = df.groupby('category')['overall_score'].mean().sort_values(ascending=False)

    print(f"\n📊 Category Rankings:")
    for i, (cat, score) in enumerate(cat_rankings.items(), 1):
        print(f"  {i}. {cat.replace('_', ' ').title()}: {score:.3f}")

    print("\n" + "=" * 60)
    print("✅ Visualization generation complete!")
    print(f"📁 All figures saved in: {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()