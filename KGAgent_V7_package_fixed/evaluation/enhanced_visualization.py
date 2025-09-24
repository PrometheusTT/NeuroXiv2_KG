#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced visualization module for KGAgent V7 evaluation results
- Creates diverse visualization types optimized for academic papers
- Directly works with evaluation results CSV
- Highlights agent strengths more effectively
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

# Set academic paper-style plotting parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 100,
    'savefig.dpi': 300,
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

# Academic color palette
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

# Category colors
CATEGORY_COLORS = {
    'kg_navigation': ACADEMIC_COLORS['blue'],
    'reasoning': ACADEMIC_COLORS['purple'],
    'tool_use': ACADEMIC_COLORS['orange'],
    'complex': ACADEMIC_COLORS['red']
}


class EnhancedVisualizer:
    """Enhanced visualization class for KGAgent V7 evaluation results"""

    def __init__(self, data_source: Union[str, pd.DataFrame], output_dir: str = "enhanced_figures"):
        """
        Initialize visualizer with data source

        Args:
            data_source: Either path to CSV file or pandas DataFrame
            output_dir: Directory to save output figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            raise ValueError("data_source must be a file path or DataFrame")

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess data for visualization"""
        # Calculate derived metrics if not present
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

        # Add category display names
        self.df['category_display'] = self.df['category'].apply(
            lambda x: x.replace('_', ' ').title()
        )

        # Sort by complexity for better visualization
        self.df = self.df.sort_values(['complexity', 'category'])

    def create_all_figures(self):
        """Generate all enhanced visualization figures"""
        print("\nGenerating enhanced evaluation visualizations...")

        # Create individual figures
        self.fig1_performance_overview()
        self.fig2_complexity_scaling()
        self.fig3_capability_radar()
        self.fig4_metric_distributions()
        self.fig5_comparative_analysis()
        self.fig6_executive_summary()

        print(f"\n✅ All enhanced figures saved to: {self.output_dir}")

    def fig1_performance_overview(self):
        """
        Figure 1: Performance Overview
        - Bar chart showing overall performance by category
        - Small multiples for key capability scores
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

        # Plot 1: Overall performance by category (bar chart)
        ax1 = fig.add_subplot(gs[0, :])

        # Calculate mean performance by category
        cat_perf = self.df.groupby('category_display')['overall_performance'].agg(['mean', 'std'])

        # Create bar chart with error bars
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

        # Add value labels
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

        # Add styling
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('Overall Performance Score')
        ax1.set_title('A. Overall Performance by Category', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add horizontal lines for grading
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

        # Plot 2: KG+CoT Capability (lower left)
        ax2 = fig.add_subplot(gs[1, 0])

        # Calculate metrics by category
        kg_metrics = self.df.groupby('category_display')[
            ['schema_coverage', 'cot_effectiveness', 'planning_coherence']
        ].mean()

        # Rename columns for display
        kg_metrics.columns = ['Schema\nCoverage', 'CoT\nEffectiveness', 'Planning\nCoherence']

        # Plot as grouped bar chart
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

        # Plot 3: Advanced Capabilities (lower right)
        ax3 = fig.add_subplot(gs[1, 1])

        # Calculate metrics by category
        adv_metrics = self.df.groupby('category_display')[
            ['autonomy_score', 'tool_score', 'answer_quality']
        ].mean()

        # Rename columns for display
        adv_metrics.columns = ['Autonomy', 'Tool Usage', 'Answer Quality']

        # Plot as grouped bar chart
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

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_performance_overview.png')
        plt.close()

        print("  ✓ Figure 1: Performance overview created")

    def fig2_complexity_scaling(self):
        """
        Figure 2: Complexity Scaling Analysis
        - Line charts showing how performance scales with complexity
        - Separate charts for different metric groups
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Calculate metrics by complexity
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

        # Plot 1: Overall Performance vs Complexity (upper left)
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

        # Add trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x), max(x), 100)
        ax.plot(x_trend, p(x_trend), '--', color=ACADEMIC_COLORS['red'], alpha=0.7, linewidth=1)

        # Calculate degradation rate
        slope = z[0]
        slope_percent = abs(slope * 100)

        # Add text box with trend info
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

        # Plot 2: Capability Scores vs Complexity (upper right)
        ax = axes[0, 1]

        # Plot different capability scores
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

        # Plot 3: Execution Metrics vs Complexity (lower left)
        ax = axes[1, 0]

        # Primary y-axis: Execution Time
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

        # Secondary y-axis: Total Queries
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

        # Add combined legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('C. Execution Metrics by Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 4: Query Complexity vs Schema Coverage (lower right)
        ax = axes[1, 1]

        # Primary y-axis: Query Complexity
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

        # Secondary y-axis: Schema Coverage
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

        # Add combined legend
        lines = line1 + line2
        labels = ['Query Complexity', 'Schema Coverage']
        ax.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.7)

        ax.set_title('D. Knowledge Graph Utilization', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_complexity_scaling.png')
        plt.close()

        print("  ✓ Figure 2: Complexity scaling analysis created")

    def fig3_capability_radar(self):
        """
        Figure 3: Capability Radar Analysis
        - Radar charts showing multi-dimensional performance
        - Comparisons across categories
        """
        # Setup figure with two radar charts
        fig = plt.figure(figsize=(15, 7))

        # Define metrics for radar chart
        radar_metrics = [
            'Schema\nCoverage',
            'CoT\nEffectiveness',
            'Planning\nCoherence',
            'Autonomy\nIndex',
            'Tool\nF1 Score',
            'Answer\nQuality'
        ]

        # Map to actual columns
        metric_mapping = {
            'Schema\nCoverage': 'schema_coverage',
            'CoT\nEffectiveness': 'cot_effectiveness',
            'Planning\nCoherence': 'planning_coherence',
            'Autonomy\nIndex': 'autonomy_index',
            'Tool\nF1 Score': 'tool_f1_score',
            'Answer\nQuality': 'answer_quality'
        }

        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()

        # Make angles go full circle
        angles += angles[:1]

        # Add metric labels for full circle
        radar_metrics_plot = radar_metrics + [radar_metrics[0]]

        # Left radar: By Category
        ax1 = fig.add_subplot(121, polar=True)

        # Plot each category
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            cat_display = category.replace('_', ' ').title()

            # Calculate mean for each metric
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(cat_data[col].mean())

            # Make values go full circle
            values += values[:1]

            # Plot the category
            ax1.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=cat_display,
                color=CATEGORY_COLORS[category]
            )

            # Fill area
            ax1.fill(
                angles,
                values,
                alpha=0.1,
                color=CATEGORY_COLORS[category]
            )

        # Styling
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(radar_metrics, size=9)
        ax1.set_ylim(0, 1)
        ax1.set_title('A. Capability Profile by Category', fontweight='bold', pad=15)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # Right radar: By Complexity
        ax2 = fig.add_subplot(122, polar=True)

        # Plot each complexity level
        for complexity in sorted(self.df['complexity'].unique()):
            comp_data = self.df[self.df['complexity'] == complexity]

            # Calculate mean for each metric
            values = []
            for metric in radar_metrics:
                col = metric_mapping[metric]
                values.append(comp_data[col].mean())

            # Make values go full circle
            values += values[:1]

            # Create a color based on complexity
            cmap = plt.cm.viridis
            color = cmap(complexity / max(self.df['complexity']))

            # Plot the complexity level
            ax2.plot(
                angles,
                values,
                'o-',
                linewidth=2.5,
                label=f'Level {complexity}',
                color=color
            )

            # Fill area
            ax2.fill(
                angles,
                values,
                alpha=0.1,
                color=color
            )

        # Styling
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_metrics, size=9)
        ax2.set_ylim(0, 1)
        ax2.set_title('B. Capability Profile by Complexity', fontweight='bold', pad=15)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True)

        # Add overall title
        plt.suptitle('Multi-dimensional Capability Analysis', fontweight='bold', y=0.98, fontsize=14)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_capability_radar.png')
        plt.close()

        print("  ✓ Figure 3: Capability radar analysis created")

    def fig4_metric_distributions(self):
        """
        Figure 4: Metric Distribution Analysis
        - Box and violin plots showing distributions of key metrics
        - Allows seeing variability and central tendency
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Define subplot titles and metrics
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

        # Create violin plots for each group
        for ax, title, metrics in subplot_info:
            # Filter metrics that exist in the dataframe
            valid_metrics = [m for m in metrics if m in self.df.columns]

            # Prepare data for violin plot
            plot_data = []
            labels = []

            for metric in valid_metrics:
                plot_data.append(self.df[metric])
                labels.append(metric.replace('_', ' ').title())

            # Create violin plot
            parts = ax.violinplot(
                plot_data,
                showmeans=False,
                showmedians=True
            )

            # Customize violin colors
            for i, pc in enumerate(parts['bodies']):
                # Choose color based on position in the Blues colormap
                cmap = plt.cm.Blues
                color = cmap(0.4 + (i / len(valid_metrics)) * 0.6)
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            # Customize medians
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(1.5)

            # Add box plots inside violins
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

            # Add scatter points for individual tests
            for i, metric in enumerate(valid_metrics):
                # Jittered x positions
                x = np.random.normal(i + 1, 0.07, size=len(self.df))
                y = self.df[metric]

                ax.scatter(
                    x, y,
                    s=20,
                    alpha=0.5,
                    c='black',
                    edgecolor='none'
                )

            # Add mean markers
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

                # Add mean value text
                ax.text(
                    i + 1, mean_val + 0.05,
                    f'{mean_val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

            # Styling
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(box_positions)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

            # Only add legend to first plot
            if title == 'A. Knowledge Graph Navigation':
                ax.legend(loc='upper right')

        # Add overall title
        plt.suptitle('Metric Distribution Analysis', fontweight='bold', y=0.98, fontsize=14)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_metric_distributions.png')
        plt.close()

        print("  ✓ Figure 4: Metric distributions created")

    def fig5_comparative_analysis(self):
        """
        Figure 5: Comparative Analysis
        - Scatter plots showing relationships between metrics
        - Paired metrics to reveal insights
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Define subplot parameters
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
            # Extract data
            x = self.df[x_metric]
            y = self.df[y_metric]

            # Determine colors and sizes
            if color_by == 'Category':
                colors = [CATEGORY_COLORS[c] for c in self.df['category']]
                sizes = self.df['complexity'] * 20 + 30
                legend_elements = [
                    mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace('_', ' ').title())
                    for c in CATEGORY_COLORS if c in self.df['category'].unique()
                ]
                legend_title = 'Category'

            else:  # color_by == 'Complexity Level'
                # Create a colormap
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=min(self.df['complexity']), vmax=max(self.df['complexity']))
                colors = [cmap(norm(c)) for c in self.df['complexity']]
                sizes = 60

                # Create legend elements for complexity levels
                legend_elements = [
                    mpatches.Patch(color=cmap(norm(c)), label=f'Level {c}')
                    for c in sorted(self.df['complexity'].unique())
                ]
                legend_title = 'Complexity Level'

            # Create scatter plot
            scatter = ax.scatter(
                x, y,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

            # Add trend line
            if x_metric != 'tool_f1_score':  # Skip trendline for tool F1 (often binary)
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x), max(x), 100)
                    ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=1.5)

                    # Calculate correlation coefficient
                    corr = np.corrcoef(x, y)[0, 1]
                    # Add correlation text
                    ax.text(
                        0.05, 0.95,
                        f'r = {corr:.2f}',
                        transform=ax.transAxes,
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                    )
                except:
                    pass

            # Add labels
            x_label = x_metric.replace('_', ' ').title()
            y_label = y_metric.replace('_', ' ').title()

            if y_metric == 'query_semantic_complexity':
                y_label = 'Query Complexity Score'
            elif y_metric == 'execution_time':
                y_label = 'Execution Time (seconds)'

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            # Add test IDs as annotations for hover
            for i, txt in enumerate(self.df['test_id']):
                ax.annotate(
                    txt,
                    (x.iloc[i], y.iloc[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0
                )

            # Add legend
            ax.legend(
                handles=legend_elements,
                title=legend_title,
                loc='best',
                frameon=True,
                framealpha=0.7
            )

            # Set title
            ax.set_title(title, fontweight='bold')

            # Adjust y-limits for better visualization
            if y_metric != 'execution_time':
                ax.set_ylim(0, 1.05)

            # Add grid
            ax.grid(True, alpha=0.3)

        # Add overall title
        plt.suptitle('Comparative Metric Analysis', fontweight='bold', y=0.98, fontsize=14)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_comparative_analysis.png')
        plt.close()

        print("  ✓ Figure 5: Comparative analysis created")

    def fig6_executive_summary(self):
        """
        Figure 6: Executive Summary Dashboard
        - Combined visualization with key metrics
        - Performance by category and complexity
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.3, 1], width_ratios=[1.2, 1, 1])

        # Calculate overall statistics
        overall_score = self.df['overall_performance'].mean()
        kg_score = self.df['kg_cot_score'].mean()
        autonomy_score = self.df['autonomy_score'].mean()
        tool_score = self.df['tool_score'].mean()
        answer_score = self.df['answer_quality'].mean()

        # Determine overall grade
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

        # Panel 1: Overall Score (top left)
        ax1 = fig.add_subplot(gs[0, 0])

        # Create a gauge-like visualization
        gauge_angles = np.linspace(0, 180, 100) * np.pi / 180
        gauge_radius = 0.8

        # Background arc (gray)
        ax1.plot(
            [0, 0],
            [0, gauge_radius],
            color='gray',
            alpha=0.3,
            linewidth=80,
            solid_capstyle='round'
        )

        # Foreground arc (colored by grade)
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

        # Add score text
        ax1.text(
            0, 0,
            f"{overall_score:.2f}",
            ha='center',
            va='center',
            fontsize=36,
            fontweight='bold',
            color='black'
        )

        # Add grade
        ax1.text(
            0, gauge_radius / 2.5,
            f"Grade: {grade}",
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            color=grade_color
        )

        # Remove axes
        ax1.axis('off')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-0.2, 1)

        # Add title
        ax1.text(
            0, -0.15,
            "Overall Performance",
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )

        # Panel 2: Key Capability Scores (top center)
        ax2 = fig.add_subplot(gs[0, 1:])

        # Create horizontal bar chart
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

        # Add value labels
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

        # Styling
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(capabilities)
        ax2.set_xlim(0, 1.05)
        ax2.set_title('Key Capability Scores', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add grade thresholds
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

        # Panel 3: Performance by Category and Complexity (middle row)
        ax3 = fig.add_subplot(gs[1, :])

        # Create heatmap-like visualization with actual values
        cat_comp_data = self.df.pivot_table(
            values='overall_performance',
            index='category_display',
            columns='complexity',
            aggfunc='mean'
        )

        # Fill NaN with empty string for cleaner visualization
        cat_comp_data = cat_comp_data.fillna('')

        # Create a custom colormap function
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

        # Create text color function
        def get_text_color(val):
            if val == '' or val < 0.7:
                return 'white'
            else:
                return 'black'

        # Get cell colors
        cell_colors = [[get_color(val) for val in row] for row in cat_comp_data.values]

        # Create table
        table = ax3.table(
            cellText=[[f'{val:.2f}' if val != '' else '' for val in row] for row in cat_comp_data.values],
            rowLabels=cat_comp_data.index,
            colLabels=[f'Level {c}' for c in cat_comp_data.columns],
            cellColours=cell_colors,
            loc='center',
            cellLoc='center'
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Customize cell text color
        for (i, j), cell in table.get_celld().items():
            if i > 0 and j > 0:  # Skip headers
                val = cat_comp_data.iloc[i - 1, j - 1]
                if val != '':
                    cell.get_text().set_color(get_text_color(val))
                    if val >= 0.8:
                        cell.get_text().set_fontweight('bold')

        # Remove axes
        ax3.axis('off')

        # Add title
        ax3.set_title('Performance Matrix by Category and Complexity', fontweight='bold', pad=10)

        # Panel 4: Key Statistics (bottom row)
        ax4 = fig.add_subplot(gs[2, :])

        # Calculate key statistics
        avg_time = self.df['execution_time'].mean()
        avg_queries = self.df['total_queries'].mean()
        success_rate = self.df['query_efficiency'].mean() * 100
        test_count = len(self.df)
        category_count = len(self.df['category'].unique())
        complexity_range = f"{min(self.df['complexity'])} - {max(self.df['complexity'])}"

        # Best and worst performance
        best_test = self.df.loc[self.df['overall_performance'].idxmax()]
        worst_test = self.df.loc[self.df['overall_performance'].idxmin()]

        # Create text for statistics
        stats_text = [
            f"▶ Tests Completed: {test_count} across {category_count} categories (Complexity levels: {complexity_range})",
            f"▶ Average Execution Time: {avg_time:.1f} seconds with {avg_queries:.1f} average queries per test",
            f"▶ Query Success Rate: {success_rate:.1f}%",
            f"▶ Best Performing Test: {best_test['test_id']} ({best_test['category_display']}, Level {best_test['complexity']}) - Score: {best_test['overall_performance']:.2f}",
            f"▶ Most Challenging Test: {worst_test['test_id']} ({worst_test['category_display']}, Level {worst_test['complexity']}) - Score: {worst_test['overall_performance']:.2f}"
        ]

        # Strengths and weaknesses based on metric averages
        top_metrics = self.df.mean().sort_values(ascending=False).head(3)
        bottom_metrics = self.df.mean().sort_values().head(3)

        strengths = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in top_metrics.items()
                     if m not in ['complexity', 'iteration_rounds', 'total_queries', 'execution_time']]

        weaknesses = [f"{m.replace('_', ' ').title()}: {v:.2f}" for m, v in bottom_metrics.items()
                      if m not in ['complexity', 'iteration_rounds', 'total_queries', 'execution_time']]

        if strengths:
            stats_text.append(f"▶ Top Strengths: {', '.join(strengths[:3])}")

        if weaknesses:
            stats_text.append(f"▶ Areas for Improvement: {', '.join(weaknesses[:3])}")

        # Add text
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

        # Remove axes
        ax4.axis('off')

        # Add title
        ax4.set_title('Key Performance Statistics', fontweight='bold', pad=10)

        # Add overall title
        plt.suptitle('KGAgent V7 Performance Dashboard', fontweight='bold', y=0.98, fontsize=16)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_executive_summary.png')
        plt.close()

        print("  ✓ Figure 6: Executive summary dashboard created")


# Entry point for running the visualizer
def main():
    """Main function to run visualizer on CSV data"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate enhanced visualizations for KGAgent V7 evaluation results')
    parser.add_argument('--input', default='improved_evaluation_results.csv', help='Input CSV file path')
    parser.add_argument('--output', default='enhanced_figures', help='Output directory')

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("KGAgent V7 Enhanced Visualization Generator")
    print(f"{'=' * 60}")

    # Create visualizer and generate figures
    visualizer = EnhancedVisualizer(args.input, args.output)
    visualizer.create_all_figures()

    print(f"\n{'=' * 60}")
    print(f"✅ All visualizations saved to: {args.output}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()