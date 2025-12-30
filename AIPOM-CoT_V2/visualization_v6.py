"""
AIPOM-CoT Benchmark Visualization - Publication Quality
========================================================
Designed for high-impact journal submission (Nature Methods style)

Key improvements:
1. Clear, intuitive metric naming
2. Professional color scheme
3. Emphasis on AIPOM-CoT advantages
4. Statistical annotations

Author: Claude
Date: 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROFESSIONAL STYLING
# ============================================================================

# Nature-inspired color palette
COLORS = {
    'AIPOM-CoT': '#1f77b4',      # Strong blue - main method
    'Direct GPT-4o': '#7f7f7f',  # Gray
    'Template-KG': '#bcbd22',    # Olive
    'RAG': '#17becf',            # Cyan
    'ReAct': '#ff7f0e',          # Orange - main competitor
}

# Display names (shorter, clearer)
METHOD_DISPLAY = {
    'AIPOM-CoT': 'AIPOM-CoT',
    'Direct GPT-4o': 'GPT-4o',
    'Template-KG': 'Template',
    'RAG': 'RAG',
    'ReAct': 'ReAct',
}

# Metric renaming - from technical jargon to intuitive names
METRIC_RENAME = {
    # Core capabilities
    'planning_quality': 'Task Planning',
    'reasoning_capability': 'Reasoning',
    'reflection_capability': 'Self-Correction',
    'nlu_capability': 'Query Understanding',

    # Planning sub-metrics
    'planning_coherence': 'Plan Coherence',
    'planning_optimality': 'Plan Efficiency',
    'planning_adaptability': 'Adaptive Planning',

    # Reasoning sub-metrics
    'logical_consistency': 'Logical Consistency',
    'evidence_integration': 'Evidence Usage',
    'multi_hop_depth_score': 'Multi-step Depth',

    # Reflection sub-metrics
    'error_detection': 'Error Detection',
    'self_correction': 'Self-Correction',
    'iterative_refinement': 'Iterative Improvement',

    # NLU sub-metrics
    'query_understanding': 'Query Understanding',
    'intent_recognition': 'Intent Recognition',
    'ambiguity_resolution': 'Ambiguity Handling',

    # Output quality
    'factual_accuracy': 'Factual Accuracy',
    'answer_completeness': 'Answer Completeness',
    'scientific_rigor': 'Scientific Rigor',

    # Composite scores
    'overall_score': 'Overall Performance',
    'nm_capability_score': 'Agent Capability Index',
    'biological_insight_score': 'Domain Task Score',
}

# Set global styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def load_and_process_data(filepath):
    """Load data and compute summary statistics"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Compute summary for each method
    summary = {}
    for method, results in data.items():
        metrics_agg = defaultdict(list)

        for result in results:
            for key, value in result.get('metrics', {}).items():
                if value is not None and isinstance(value, (int, float)):
                    metrics_agg[key].append(value)

        summary[method] = {}
        for metric, values in metrics_agg.items():
            if values:
                summary[method][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n': len(values),
                    'median': np.median(values),
                }

    return data, summary


def get_improvement(summary, metric, base='AIPOM-CoT', compare='ReAct'):
    """Calculate improvement percentage"""
    if base not in summary or compare not in summary:
        return None
    base_val = summary[base].get(metric, {}).get('mean')
    comp_val = summary[compare].get(metric, {}).get('mean')
    if base_val and comp_val and comp_val > 0:
        return ((base_val - comp_val) / comp_val) * 100
    return None


# ============================================================================
# FIGURE 1: Main Performance Overview (Hero Figure)
# ============================================================================

def fig1_main_comparison(summary, output_dir):
    """
    Main comparison figure - Bar chart with clear advantage highlighting
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods_order = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']

    # Panel A: Core Composite Metrics
    ax = axes[0]
    metrics = ['overall_score', 'nm_capability_score', 'biological_insight_score']
    metric_labels = ['Overall\nPerformance', 'Agent Capability\nIndex', 'Domain Task\nScore']

    x = np.arange(len(metrics))
    width = 0.15

    for i, method in enumerate(methods_order):
        if method not in summary:
            continue
        values = [summary[method].get(m, {}).get('mean', 0) for m in metrics]
        errors = [summary[method].get(m, {}).get('std', 0) for m in metrics]

        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width,
                     label=METHOD_DISPLAY[method],
                     color=COLORS[method],
                     edgecolor='white',
                     linewidth=1.5,
                     alpha=0.9,
                     yerr=errors,
                     capsize=3,
                     error_kw={'linewidth': 1.5})

        # Add value labels for AIPOM-CoT
        if method == 'AIPOM-CoT':
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.set_title('A. Composite Performance Metrics', fontsize=14, fontweight='bold', loc='left')

    # Add excellence threshold
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(2.5, 0.82, 'Excellence threshold', fontsize=9, color='#2ca02c', style='italic')

    # Panel B: Core Capability Comparison (only methods with full data)
    ax = axes[1]

    # Only AIPOM-CoT and ReAct have all dimensions
    agentic_methods = ['AIPOM-CoT', 'ReAct']
    capabilities = ['planning_quality', 'reasoning_capability',
                   'reflection_capability', 'nlu_capability']
    cap_labels = ['Planning', 'Reasoning', 'Self-\nCorrection', 'Query\nUnderstanding']

    x = np.arange(len(capabilities))
    width = 0.35

    for i, method in enumerate(agentic_methods):
        if method not in summary:
            continue
        values = []
        errors = []
        for cap in capabilities:
            m = summary[method].get(cap, {})
            values.append(m.get('mean', 0))
            errors.append(m.get('std', 0))

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width,
                     label=METHOD_DISPLAY[method],
                     color=COLORS[method],
                     edgecolor='white',
                     linewidth=1.5,
                     alpha=0.9,
                     yerr=errors,
                     capsize=3)

        # Add improvement annotations for AIPOM-CoT
        if method == 'AIPOM-CoT':
            react_values = [summary['ReAct'].get(cap, {}).get('mean', 0) for cap in capabilities]
            for j, (bar, val, react_val) in enumerate(zip(bars, values, react_values)):
                if react_val > 0:
                    improvement = ((val - react_val) / react_val) * 100
                    if improvement > 5:
                        ax.annotate(f'+{improvement:.0f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   xytext=(0, 5), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=8,
                                   color='#2ca02c', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cap_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('B. Agentic Capabilities (vs ReAct)', fontsize=14, fontweight='bold', loc='left')
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_main_comparison.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_main_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Main comparison saved")


# ============================================================================
# FIGURE 2: Radar Chart - Capability Profile
# ============================================================================

def fig2_capability_radar(summary, output_dir):
    """
    Radar chart showing capability profiles for agentic methods
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    dimensions = ['Planning', 'Reasoning', 'Self-Correction', 'Query Understanding']
    metrics = ['planning_quality', 'reasoning_capability',
               'reflection_capability', 'nlu_capability']

    # Only show methods with complete data
    methods_to_show = ['AIPOM-CoT', 'ReAct']

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    # Add reference circles
    for r in [0.4, 0.6, 0.8]:
        circle = plt.Circle((0, 0), r, transform=ax.transData._b,
                           fill=False, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    for method in methods_to_show:
        if method not in summary:
            continue

        values = [summary[method].get(m, {}).get('mean', 0) for m in metrics]
        values += values[:1]

        linewidth = 3.5 if method == 'AIPOM-CoT' else 2.5
        alpha = 0.3 if method == 'AIPOM-CoT' else 0.15

        ax.plot(angles, values, 'o-', linewidth=linewidth,
               label=METHOD_DISPLAY[method], color=COLORS[method],
               markersize=10)
        ax.fill(angles, values, alpha=alpha, color=COLORS[method])

        # Add value labels
        for angle, value, dim in zip(angles[:-1], values[:-1], dimensions):
            ha = 'left' if angle < np.pi else 'right'
            offset = 0.08
            ax.text(angle, value + offset, f'{value:.2f}',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color=COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=12)
    ax.set_title('Agentic Capability Profile\n', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_capability_radar.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_capability_radar.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Capability radar saved")


# ============================================================================
# FIGURE 3: Detailed Breakdown - Sub-metrics
# ============================================================================

def fig3_detailed_breakdown(summary, output_dir):
    """
    Detailed breakdown of each capability dimension
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Define sub-metrics for each dimension
    dimensions = {
        'Planning': {
            'metrics': ['planning_coherence', 'planning_optimality', 'planning_adaptability'],
            'labels': ['Coherence', 'Efficiency', 'Adaptability'],
            'methods': ['AIPOM-CoT', 'ReAct']
        },
        'Reasoning': {
            'metrics': ['logical_consistency', 'evidence_integration', 'multi_hop_depth_score'],
            'labels': ['Logic', 'Evidence', 'Multi-step'],
            'methods': ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
        },
        'Self-Correction': {
            'metrics': ['error_detection', 'self_correction', 'iterative_refinement'],
            'labels': ['Error Detection', 'Correction', 'Refinement'],
            'methods': ['AIPOM-CoT', 'ReAct']
        },
        'Output Quality': {
            'metrics': ['answer_completeness', 'scientific_rigor'],
            'labels': ['Completeness', 'Rigor'],
            'methods': ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
        },
    }

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (dim_name, dim_info) in enumerate(dimensions.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        metrics = dim_info['metrics']
        labels = dim_info['labels']
        methods = dim_info['methods']

        x = np.arange(len(metrics))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            if method not in summary:
                continue

            values = [summary[method].get(m, {}).get('mean', 0) for m in metrics]
            errors = [summary[method].get(m, {}).get('std', 0) for m in metrics]

            offset = (i - len(methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width,
                         label=METHOD_DISPLAY[method] if idx == 0 else '',
                         color=COLORS[method],
                         edgecolor='white' if method == 'AIPOM-CoT' else 'none',
                         linewidth=2 if method == 'AIPOM-CoT' else 0,
                         alpha=0.9 if method == 'AIPOM-CoT' else 0.7,
                         yerr=errors,
                         capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(dim_name, fontsize=13, fontweight='bold')
        ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add summary panel (Advantages)
    ax = fig.add_subplot(gs[0:2, 2])  # Take the right column

    # Calculate improvements
    improvements = []
    imp_labels = []
    for dim_name, dim_info in dimensions.items():
        for metric in dim_info['metrics']:
            imp = get_improvement(summary, metric, 'AIPOM-CoT', 'ReAct')
            if imp is not None and abs(imp) > 1:
                improvements.append(imp)
                imp_labels.append(METRIC_RENAME.get(metric, metric))

    # Sort by improvement
    sorted_idx = np.argsort(improvements)[::-1][:8]  # Top 8
    sorted_imp = [improvements[i] for i in sorted_idx]
    sorted_labels = [imp_labels[i] for i in sorted_idx]

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in sorted_imp]
    y_pos = np.arange(len(sorted_imp))

    bars = ax.barh(y_pos, sorted_imp, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel('Improvement vs ReAct (%)', fontsize=10)
    ax.set_title('AIPOM-CoT Advantages', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, sorted_imp):
        x_pos = val + 2 if val > 0 else val - 2
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
               va='center', ha=ha, fontsize=9, fontweight='bold')

    # Add legend
    handles = [mpatches.Patch(color=COLORS[m], label=METHOD_DISPLAY[m])
               for m in ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
               if m in summary]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=11,
              bbox_to_anchor=(0.5, 0.98), frameon=True, fancybox=True)

    plt.savefig(output_dir / 'fig3_detailed_breakdown.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_detailed_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Detailed breakdown saved")


# ============================================================================
# FIGURE 4: All Methods Horizontal Bar Comparison
# ============================================================================

def fig4_method_ranking(summary, output_dir):
    """
    Horizontal bar chart showing method ranking across key metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    methods_order = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']

    key_metrics = [
        ('overall_score', 'Overall Performance'),
        ('reasoning_capability', 'Reasoning Capability'),
        ('nlu_capability', 'Query Understanding'),
    ]

    for idx, (metric, title) in enumerate(key_metrics):
        ax = axes[idx]

        values = []
        errors = []
        valid_methods = []

        for method in methods_order:
            if method not in summary:
                continue
            m = summary[method].get(metric, {})
            val = m.get('mean')
            if val is not None and val > 0:
                values.append(val)
                errors.append(m.get('std', 0))
                valid_methods.append(method)

        # Sort by value
        sorted_idx = np.argsort(values)[::-1]
        values = [values[i] for i in sorted_idx]
        errors = [errors[i] for i in sorted_idx]
        valid_methods = [valid_methods[i] for i in sorted_idx]

        y_pos = np.arange(len(valid_methods))
        colors = [COLORS[m] for m in valid_methods]

        bars = ax.barh(y_pos, values, xerr=errors, color=colors,
                      alpha=0.9, edgecolor='white', linewidth=2, capsize=4)

        # Highlight AIPOM-CoT bar
        for i, (bar, method) in enumerate(zip(bars, valid_methods)):
            if method == 'AIPOM-CoT':
                bar.set_edgecolor('#1f77b4')
                bar.set_linewidth(3)

        # Add value labels
        for i, (bar, val, method) in enumerate(zip(bars, values, valid_methods)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

            # Add rank badge
            if i == 0:
                ax.text(0.02, bar.get_y() + bar.get_height()/2,
                       '🥇', va='center', fontsize=14)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in valid_methods], fontsize=11)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axvline(x=0.8, color='#2ca02c', linestyle=':', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_method_ranking.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_method_ranking.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Method ranking saved")


# ============================================================================
# FIGURE 5: Statistical Summary Table
# ============================================================================

def fig5_summary_table(summary, output_dir):
    """
    Summary table with key metrics
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    methods = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
    metrics = [
        ('overall_score', 'Overall'),
        ('planning_quality', 'Planning'),
        ('reasoning_capability', 'Reasoning'),
        ('reflection_capability', 'Self-Correction'),
        ('nlu_capability', 'Query Understanding'),
        ('scientific_rigor', 'Rigor'),
    ]

    # Prepare table data
    col_labels = [METHOD_DISPLAY[m] for m in methods]
    row_labels = [m[1] for m in metrics]

    cell_data = []
    cell_colors = []

    for metric, label in metrics:
        row = []
        row_colors = []

        # Find best value for highlighting
        values_for_max = []
        for method in methods:
            val = summary.get(method, {}).get(metric, {}).get('mean')
            values_for_max.append(val if val is not None else 0)
        max_val = max(values_for_max)

        for i, method in enumerate(methods):
            m = summary.get(method, {}).get(metric, {})
            val = m.get('mean')
            std = m.get('std')

            if val is not None:
                text = f'{val:.3f}'
                if std:
                    text += f' ±{std:.2f}'
                row.append(text)

                # Color based on performance
                if val == max_val and val > 0:
                    row_colors.append('#c8e6c9')  # Light green for best
                elif val >= 0.8:
                    row_colors.append('#e3f2fd')  # Light blue for excellent
                elif val >= 0.6:
                    row_colors.append('#fff3e0')  # Light orange for good
                else:
                    row_colors.append('#ffebee')  # Light red for below average
            else:
                row.append('—')
                row_colors.append('#f5f5f5')

        cell_data.append(row)
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_data,
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    cellColours=cell_colors,
                    loc='center',
                    cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for i, method in enumerate(methods):
        cell = table[(0, i)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor(COLORS[method])

    # Style row labels
    for i in range(len(metrics)):
        cell = table[(i+1, -1)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#e0e0e0')

    ax.set_title('Performance Summary (Mean ± Std)\n', fontsize=16, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='black', label='Best'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='black', label='Excellent (≥0.8)'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='black', label='Good (0.6-0.8)'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='black', label='Below Average (<0.6)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=4,
             bbox_to_anchor=(0.5, 0.08), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_summary_table.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Summary table saved")


# ============================================================================
# FIGURE 6: Improvement Waterfall Chart
# ============================================================================

def fig6_improvement_waterfall(summary, output_dir):
    """
    Waterfall chart showing AIPOM-CoT improvement over ReAct
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = [
        ('planning_quality', 'Planning'),
        ('reasoning_capability', 'Reasoning'),
        ('reflection_capability', 'Self-Correction'),
        ('nlu_capability', 'Understanding'),
        ('scientific_rigor', 'Rigor'),
    ]

    improvements = []
    labels = []

    for metric, label in metrics:
        imp = get_improvement(summary, metric, 'AIPOM-CoT', 'ReAct')
        if imp is not None:
            improvements.append(imp)
            labels.append(label)

    x = np.arange(len(labels))
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in improvements]

    bars = ax.bar(x, improvements, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, improvements):
        y_pos = val + 2 if val > 0 else val - 3
        va = 'bottom' if val > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%',
               ha='center', va=va, fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement over ReAct (%)', fontsize=12)
    ax.set_title('AIPOM-CoT Performance Gains vs ReAct Baseline\n', fontsize=15, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.axhline(y=10, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.5, label='10% improvement')
    ax.axhline(y=20, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5, label='20% improvement')

    # Add average improvement line
    avg_imp = np.mean([v for v in improvements if v > 0])
    ax.axhline(y=avg_imp, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.7)
    ax.text(len(labels)-0.5, avg_imp+2, f'Avg: +{avg_imp:.1f}%', fontsize=10,
           color='#1f77b4', fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(min(improvements) - 15, max(improvements) + 15)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_improvement_waterfall.png', dpi=1200, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_improvement_waterfall.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Improvement waterfall saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  AIPOM-CoT Benchmark Visualization - Enhanced Edition")
    print("="*70 + "\n")

    # Setup
    data_file = Path('./results_filtered/detailed_results_v4.json')
    output_dir = Path('./figure6_v6')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("📂 Loading data...")
    data, summary = load_and_process_data(data_file)
    print(f"   Found {len(data)} methods: {list(data.keys())}")

    # Generate figures
    print("\n🎨 Generating figures...\n")

    fig1_main_comparison(summary, output_dir)
    fig2_capability_radar(summary, output_dir)
    fig3_detailed_breakdown(summary, output_dir)
    fig4_method_ranking(summary, output_dir)
    fig5_summary_table(summary, output_dir)
    fig6_improvement_waterfall(summary, output_dir)

    print("\n" + "="*70)
    print(f"  ✅ All figures saved to: {output_dir}")
    print("="*70 + "\n")

    # Print key findings
    print("📊 KEY FINDINGS:")
    print("-" * 40)

    aipom = summary.get('AIPOM-CoT', {})
    react = summary.get('ReAct', {})

    for metric, label in [('overall_score', 'Overall'),
                          ('reasoning_capability', 'Reasoning'),
                          ('planning_quality', 'Planning')]:
        aipom_val = aipom.get(metric, {}).get('mean', 0)
        react_val = react.get(metric, {}).get('mean', 0)
        if react_val > 0:
            imp = ((aipom_val - react_val) / react_val) * 100
            print(f"   {label}: AIPOM-CoT={aipom_val:.3f}, ReAct={react_val:.3f} (+{imp:.1f}%)")


if __name__ == '__main__':
    main()