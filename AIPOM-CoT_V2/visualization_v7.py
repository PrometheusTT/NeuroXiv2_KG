"""
AIPOM-CoT Benchmark Visualization - Publication Quality
========================================================
Designed for high-impact journal submission (Nature Methods style)

Key improvements:
1. Clear, intuitive metric naming
2. Professional color scheme
3. Emphasis on AIPOM-CoT advantages
4. Statistical annotations
5. Each panel saved as individual file with original proportions

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

    # Composite scores
    'overall_score': 'Overall Performance',
    'nm_capability_score': 'Agent Capability Index',
    'biological_insight_score': 'Domain Task Score',
}

# ============================================================================
# UNIFIED PANEL SIZE - All bar chart panels use same aspect ratio
# ============================================================================
PANEL_WIDTH = 6
PANEL_HEIGHT = 5
PANEL_SIZE = (PANEL_WIDTH, PANEL_HEIGHT)  # 6:5 aspect ratio for all panels

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


def save_panel(fig, output_dir, name):
    """Helper function to save panel in both PNG and PDF formats"""
    fig.savefig(output_dir / f'{name}.png', dpi=1200, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {name}")


# ============================================================================
# FIGURE 1: Main Performance Overview
# Panel A: All methods composite scores (unique: full method comparison)
# Panel B: AIPOM-CoT vs ReAct capability comparison with improvement %
# ============================================================================

def fig1_main_comparison(summary, output_dir):
    """
    Main comparison figure - save individual panels with unified size
    """
    methods_order = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']

    # ========== Panel A: All Methods Composite Scores ==========
    # Unique info: Complete comparison of all 5 methods on 2 composite metrics
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    # Removed: biological_insight_score (Domain Task Score)
    metrics = ['overall_score', 'nm_capability_score']
    metric_labels = ['Overall\nPerformance', 'Agent Capability\nIndex']

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
                     capsize=2,
                     error_kw={'linewidth': 1.2})

        if method == 'AIPOM-CoT':
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.set_title('A. Composite Performance (All Methods)', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig1a_composite_metrics')

    # ========== Panel B: AIPOM-CoT vs ReAct Core Capabilities ==========
    # Unique info: Head-to-head comparison with improvement percentages
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

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
                     capsize=2)

        if method == 'AIPOM-CoT':
            react_values = [summary['ReAct'].get(cap, {}).get('mean', 0) for cap in capabilities]
            for j, (bar, val, react_val) in enumerate(zip(bars, values, react_values)):
                if react_val > 0:
                    improvement = ((val - react_val) / react_val) * 100
                    if improvement > 5:
                        ax.annotate(f'+{improvement:.0f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   xytext=(0, 4), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=8,
                                   color='#2ca02c', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cap_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('B. Agentic Capabilities (AIPOM-CoT vs ReAct)', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig1b_agentic_capabilities')

    # ========== Combined figure (2 panels side by side) ==========
    fig, axes = plt.subplots(1, 2, figsize=(PANEL_WIDTH * 2, PANEL_HEIGHT))

    # Panel A (left)
    ax = axes[0]
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
                     capsize=2,
                     error_kw={'linewidth': 1.2})

        if method == 'AIPOM-CoT':
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.set_title('A. Composite Performance', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.7)

    # Panel B (right)
    ax = axes[1]
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
                     capsize=2)

        if method == 'AIPOM-CoT':
            react_values = [summary['ReAct'].get(cap, {}).get('mean', 0) for cap in capabilities]
            for j, (bar, val, react_val) in enumerate(zip(bars, values, react_values)):
                if react_val > 0:
                    improvement = ((val - react_val) / react_val) * 100
                    if improvement > 5:
                        ax.annotate(f'+{improvement:.0f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   xytext=(0, 4), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=8,
                                   color='#2ca02c', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cap_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('B. Agentic Capabilities', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig1_combined')

    print("✓ Figure 1: Main comparison saved (combined + 2 panels)")


# ============================================================================
# FIGURE 2: Radar Chart - Original size (10, 10)
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
    save_panel(fig, output_dir, 'fig2_capability_radar')
    print("✓ Figure 2: Capability radar saved")


# ============================================================================
# FIGURE 3: Detailed Sub-metrics Breakdown
# Each panel shows different capability dimension with all available methods
# Unique info per panel: specific sub-metrics not shown elsewhere
# Sub-panels use same proportions as in the combined figure
# ============================================================================

def fig3_detailed_breakdown(summary, output_dir):
    """
    Detailed breakdown of each capability dimension
    Save individual panels with unified size, using shared legend
    Sub-panels match the proportions in combined figure (GridSpec 2x3)
    """
    # Combined figure dimensions
    combined_width = PANEL_WIDTH * 3   # 18
    combined_height = PANEL_HEIGHT * 2  # 10
    hspace, wspace = 0.35, 0.3

    # Calculate actual panel sizes based on GridSpec 2x3 layout
    # Small panels (left 2x2 grid): each takes 1/3 width, 1/2 height
    small_panel_width = combined_width / 3 * 0.85   # ~5.1 (with some margin)
    small_panel_height = combined_height / 2 * 0.85  # ~4.25
    small_panel_size = (small_panel_width, small_panel_height)

    # Right panel (spans 2 rows): 1/3 width, full height
    right_panel_width = combined_width / 3 * 0.85   # ~5.1
    right_panel_height = combined_height * 0.85     # ~8.5
    right_panel_size = (right_panel_width, right_panel_height)

    # Define sub-metrics for each dimension - each panel has UNIQUE sub-metrics
    # Removed: multi_hop_depth_score, scientific_rigor
    dimensions = {
        'Planning Sub-metrics': {
            'metrics': ['planning_coherence', 'planning_optimality', 'planning_adaptability'],
            'labels': ['Coherence', 'Efficiency', 'Adaptability'],
            'methods': ['AIPOM-CoT', 'ReAct'],
            'panel_name': 'fig3a_planning'
        },
        'Reasoning Sub-metrics': {
            'metrics': ['logical_consistency', 'evidence_integration'],
            'labels': ['Logic', 'Evidence'],
            'methods': ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG'],
            'panel_name': 'fig3b_reasoning'
        },
        'Self-Correction Sub-metrics': {
            'metrics': ['error_detection', 'self_correction', 'iterative_refinement'],
            'labels': ['Error\nDetection', 'Correction', 'Refinement'],
            'methods': ['AIPOM-CoT', 'ReAct'],
            'panel_name': 'fig3c_self_correction'
        },
        'Output Quality': {
            'metrics': ['answer_completeness'],
            'labels': ['Completeness'],
            'methods': ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG'],
            'panel_name': 'fig3d_output_quality'
        },
    }

    # ========== Save individual panels with proportions matching combined figure ==========
    for dim_name, dim_info in dimensions.items():
        fig, ax = plt.subplots(figsize=small_panel_size)

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
                         label=METHOD_DISPLAY[method],
                         color=COLORS[method],
                         edgecolor='white' if method == 'AIPOM-CoT' else 'none',
                         linewidth=2 if method == 'AIPOM-CoT' else 0,
                         alpha=0.9 if method == 'AIPOM-CoT' else 0.7,
                         yerr=errors,
                         capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(dim_name, fontsize=11, fontweight='bold', loc='left')
        ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5)
        # NO legend on individual panels - use shared legend

        plt.tight_layout()
        save_panel(fig, output_dir, dim_info['panel_name'])

    # ========== Panel E: Improvement Summary (horizontal bar) ==========
    # Uses right panel proportions (taller, spans 2 rows in combined)
    fig, ax = plt.subplots(figsize=right_panel_size)

    # Calculate improvements for ALL sub-metrics
    all_improvements = []
    for dim_name, dim_info in dimensions.items():
        for metric in dim_info['metrics']:
            imp = get_improvement(summary, metric, 'AIPOM-CoT', 'ReAct')
            if imp is not None:
                all_improvements.append({
                    'metric': metric,
                    'label': METRIC_RENAME.get(metric, metric),
                    'improvement': imp
                })

    # Sort by improvement (descending)
    all_improvements.sort(key=lambda x: x['improvement'], reverse=True)

    labels = [item['label'] for item in all_improvements]
    improvements = [item['improvement'] for item in all_improvements]

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, improvements, color=colors, alpha=0.8, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Improvement vs ReAct (%)', fontsize=10)
    ax.set_title('E. All Sub-metric Improvements', fontsize=12, fontweight='bold', loc='left')
    ax.axvline(x=0, color='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, improvements):
        x_pos = val + 1 if val > 0 else val - 1
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
               va='center', ha=ha, fontsize=8, fontweight='bold')

    # Add average line
    avg_imp = np.mean(improvements)
    ax.axvline(x=avg_imp, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(avg_imp + 1, len(labels) - 0.5, f'Avg: {avg_imp:+.1f}%', fontsize=8,
           color='#1f77b4', fontweight='bold')

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig3e_improvements')

    # ========== Save shared legend separately ==========
    fig_legend = plt.figure(figsize=(8, 0.5))
    handles = [mpatches.Patch(color=COLORS[m], label=METHOD_DISPLAY[m])
               for m in ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
               if m in summary]
    fig_legend.legend(handles=handles, loc='center', ncol=5, fontsize=10,
                     frameon=True, fancybox=True)
    fig_legend.tight_layout()
    save_panel(fig_legend, output_dir, 'fig3_legend')

    # ========== Combined figure (2x3 grid) ==========
    fig = plt.figure(figsize=(combined_width, combined_height))
    gs = GridSpec(2, 3, figure=fig, hspace=hspace, wspace=wspace)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    dim_list = list(dimensions.items())

    for idx, (dim_name, dim_info) in enumerate(dim_list):
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
            ax.bar(x + offset, values, width,
                   label=METHOD_DISPLAY[method] if idx == 0 else '',
                   color=COLORS[method],
                   edgecolor='white' if method == 'AIPOM-CoT' else 'none',
                   linewidth=2 if method == 'AIPOM-CoT' else 0,
                   alpha=0.9 if method == 'AIPOM-CoT' else 0.7,
                   yerr=errors,
                   capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(dim_name, fontsize=11, fontweight='bold')
        ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add improvements panel (right column spanning 2 rows)
    ax = fig.add_subplot(gs[0:2, 2])

    labels_imp = [item['label'] for item in all_improvements]
    improvements_imp = [item['improvement'] for item in all_improvements]
    colors_imp = ['#2ca02c' if x > 0 else '#d62728' for x in improvements_imp]
    y_pos = np.arange(len(labels_imp))

    bars = ax.barh(y_pos, improvements_imp, color=colors_imp, alpha=0.8, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_imp, fontsize=8)
    ax.set_xlabel('Improvement vs ReAct (%)', fontsize=10)
    ax.set_title('Sub-metric Improvements', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)

    for bar, val in zip(bars, improvements_imp):
        x_pos = val + 1 if val > 0 else val - 1
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
               va='center', ha=ha, fontsize=7, fontweight='bold')

    # Add shared legend at top
    handles = [mpatches.Patch(color=COLORS[m], label=METHOD_DISPLAY[m])
               for m in ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
               if m in summary]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, 0.98), frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_panel(fig, output_dir, 'fig3_combined')

    print(f"✓ Figure 3: Detailed breakdown saved")
    print(f"  Small panels: {small_panel_size[0]:.1f}x{small_panel_size[1]:.1f}")
    print(f"  Right panel (fig3e): {right_panel_size[0]:.1f}x{right_panel_size[1]:.1f}")
    print(f"  Combined: {combined_width}x{combined_height}")


# ============================================================================
# FIGURE 4: Cross-Method Comparison on Different Metrics
# Unique info: Shows how each method performs across different metric types
# Different from Fig1: focuses on single metrics with all methods side-by-side
# ============================================================================

def fig4_method_ranking(summary, output_dir):
    """
    Cross-method comparison showing performance distribution
    Using unified panel size
    """
    methods_order = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']

    # Different metrics to show unique information (removed scientific_rigor)
    key_metrics = [
        ('overall_score', 'Overall Score'),
        ('reasoning_capability', 'Reasoning'),
        ('answer_completeness', 'Completeness'),
    ]

    panel_names = ['fig4a_overall', 'fig4b_reasoning', 'fig4c_completeness']

    # ========== Save individual panels with unified size ==========
    for idx, (metric, title) in enumerate(key_metrics):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)

        # Get valid methods and their values
        valid_methods = []
        values = []
        errors = []

        for method in methods_order:
            if method not in summary:
                continue
            m = summary[method].get(metric, {})
            val = m.get('mean')
            if val is not None:
                valid_methods.append(method)
                values.append(val)
                errors.append(m.get('std', 0))

        x = np.arange(len(valid_methods))
        width = 0.6

        # Draw bars
        for i, (method, val, err) in enumerate(zip(valid_methods, values, errors)):
            bar = ax.bar(x[i], val, width,
                        color=COLORS[method],
                        edgecolor='white' if method == 'AIPOM-CoT' else 'none',
                        linewidth=2 if method == 'AIPOM-CoT' else 0,
                        alpha=0.9 if method == 'AIPOM-CoT' else 0.7,
                        yerr=err,
                        capsize=3)

            # Add value on top of bar
            ax.text(x[i], val + err + 0.02, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in valid_methods], fontsize=9, rotation=15, ha='right')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5)

        plt.tight_layout()
        save_panel(fig, output_dir, panel_names[idx])

    # ========== Combined figure ==========
    fig, axes = plt.subplots(1, 3, figsize=(PANEL_WIDTH * 3, PANEL_HEIGHT))

    for idx, (metric, title) in enumerate(key_metrics):
        ax = axes[idx]

        valid_methods = []
        values = []
        errors = []

        for method in methods_order:
            if method not in summary:
                continue
            m = summary[method].get(metric, {})
            val = m.get('mean')
            if val is not None:
                valid_methods.append(method)
                values.append(val)
                errors.append(m.get('std', 0))

        x = np.arange(len(valid_methods))
        width = 0.6

        for i, (method, val, err) in enumerate(zip(valid_methods, values, errors)):
            bar = ax.bar(x[i], val, width,
                        color=COLORS[method],
                        edgecolor='white' if method == 'AIPOM-CoT' else 'none',
                        linewidth=2 if method == 'AIPOM-CoT' else 0,
                        alpha=0.9 if method == 'AIPOM-CoT' else 0.7,
                        yerr=err,
                        capsize=3)

            ax.text(x[i], val + err + 0.02, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in valid_methods], fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(y=0.8, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.5)

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig4_combined')
    print("✓ Figure 4: Method ranking saved (combined + 3 panels)")


# ============================================================================
# FIGURE 5: Statistical Summary Table - Original size (14, 8)
# Including all metrics from fig3e (improvement percentages)
# ============================================================================

def fig5_summary_table(summary, output_dir):
    """
    Summary table with key metrics and improvement percentages from fig3e
    """
    # Define all metrics used in fig3e (removed multi_hop_depth_score, scientific_rigor)
    all_metrics_for_improvement = [
        # Planning sub-metrics
        ('planning_coherence', 'Plan Coherence'),
        ('planning_optimality', 'Plan Efficiency'),
        ('planning_adaptability', 'Adaptive Planning'),
        # Reasoning sub-metrics
        ('logical_consistency', 'Logical Consistency'),
        ('evidence_integration', 'Evidence Usage'),
        # Reflection sub-metrics
        ('error_detection', 'Error Detection'),
        ('self_correction', 'Self-Correction'),
        ('iterative_refinement', 'Iterative Improvement'),
        # Output quality
        ('answer_completeness', 'Answer Completeness'),
    ]

    # Calculate all improvements for fig3e
    improvements_data = {}
    for metric, label in all_metrics_for_improvement:
        imp = get_improvement(summary, metric, 'AIPOM-CoT', 'ReAct')
        if imp is not None:
            improvements_data[metric] = {
                'label': label,
                'improvement': imp,
                'aipom_mean': summary.get('AIPOM-CoT', {}).get(metric, {}).get('mean'),
                'aipom_std': summary.get('AIPOM-CoT', {}).get(metric, {}).get('std'),
                'react_mean': summary.get('ReAct', {}).get(metric, {}).get('mean'),
                'react_std': summary.get('ReAct', {}).get(metric, {}).get('std'),
            }

    # ========== Table 1: Main Performance Summary ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    methods = ['AIPOM-CoT', 'ReAct', 'Direct GPT-4o', 'RAG', 'Template-KG']
    metrics = [
        ('overall_score', 'Overall'),
        ('planning_quality', 'Planning'),
        ('reasoning_capability', 'Reasoning'),
        ('reflection_capability', 'Self-Correction'),
        ('nlu_capability', 'Query Understanding'),
        ('answer_completeness', 'Completeness'),
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
    save_panel(fig, output_dir, 'fig5_summary_table')

    # ========== Save legend separately ==========
    fig_legend = plt.figure(figsize=(10, 0.8))
    fig_legend.legend(handles=legend_elements, loc='center', ncol=4, fontsize=10)
    fig_legend.tight_layout()
    save_panel(fig_legend, output_dir, 'fig5_legend')

    # ========== Table 2: Detailed Improvement Table (fig3e data) ==========
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    ax2.axis('off')

    # Prepare data for improvement table
    col_labels2 = ['Metric', 'AIPOM-CoT', 'ReAct', 'Improvement (%)']

    cell_data2 = []
    cell_colors2 = []

    # Sort by improvement (descending)
    sorted_metrics = sorted(improvements_data.items(), key=lambda x: x[1]['improvement'], reverse=True)

    for metric, data in sorted_metrics:
        aipom_text = f"{data['aipom_mean']:.3f}" if data['aipom_mean'] is not None else '—'
        if data['aipom_std'] is not None:
            aipom_text += f" ±{data['aipom_std']:.2f}"

        react_text = f"{data['react_mean']:.3f}" if data['react_mean'] is not None else '—'
        if data['react_std'] is not None:
            react_text += f" ±{data['react_std']:.2f}"

        imp_text = f"{data['improvement']:+.2f}%"

        row = [data['label'], aipom_text, react_text, imp_text]
        cell_data2.append(row)

        # Color based on improvement
        if data['improvement'] >= 20:
            row_color = ['#c8e6c9'] * 4  # Strong improvement - green
        elif data['improvement'] >= 10:
            row_color = ['#e8f5e9'] * 4  # Moderate improvement - light green
        elif data['improvement'] >= 0:
            row_color = ['#fff3e0'] * 4  # Slight improvement - light orange
        else:
            row_color = ['#ffebee'] * 4  # Negative - light red
        cell_colors2.append(row_color)

    table2 = ax2.table(cellText=cell_data2,
                      colLabels=col_labels2,
                      cellColours=cell_colors2,
                      loc='center',
                      cellLoc='center')

    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 1.8)

    # Style header
    for i in range(len(col_labels2)):
        cell = table2[(0, i)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#1f77b4')

    ax2.set_title('AIPOM-CoT vs ReAct: Detailed Metrics Comparison\n(All metrics from Figure 3e)\n',
                  fontsize=16, fontweight='bold')

    # Add legend for improvement colors
    legend_elements2 = [
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='black', label='Strong (≥20%)'),
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='black', label='Moderate (10-20%)'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='black', label='Slight (0-10%)'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='black', label='Negative (<0%)'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper center', ncol=4,
              bbox_to_anchor=(0.5, 0.05), fontsize=10)

    plt.tight_layout()
    save_panel(fig2, output_dir, 'fig5_detailed_improvement_table')

    # ========== Table 3: Compact version with all values ==========
    fig3, ax3 = plt.subplots(figsize=(18, 12))
    ax3.axis('off')

    # Group metrics by category (removed multi_hop_depth_score, scientific_rigor)
    categories = {
        'Planning': ['planning_coherence', 'planning_optimality', 'planning_adaptability'],
        'Reasoning': ['logical_consistency', 'evidence_integration'],
        'Self-Correction': ['error_detection', 'self_correction', 'iterative_refinement'],
        'Output Quality': ['answer_completeness'],
    }

    col_labels3 = ['Category', 'Metric', 'AIPOM-CoT', 'ReAct', 'Δ (%)']

    cell_data3 = []
    cell_colors3 = []

    for cat_name, cat_metrics in categories.items():
        for i, metric in enumerate(cat_metrics):
            if metric in improvements_data:
                data = improvements_data[metric]

                # Show category name only for first row
                cat_display = cat_name if i == 0 else ''

                aipom_val = data['aipom_mean']
                react_val = data['react_mean']

                aipom_text = f"{aipom_val:.3f}" if aipom_val is not None else '—'
                react_text = f"{react_val:.3f}" if react_val is not None else '—'
                imp_text = f"{data['improvement']:+.1f}%"

                row = [cat_display, data['label'], aipom_text, react_text, imp_text]
                cell_data3.append(row)

                # Color for improvement column
                if data['improvement'] >= 20:
                    imp_color = '#c8e6c9'
                elif data['improvement'] >= 10:
                    imp_color = '#e8f5e9'
                elif data['improvement'] >= 0:
                    imp_color = '#fff3e0'
                else:
                    imp_color = '#ffebee'

                row_color = ['#f5f5f5', '#ffffff', '#e3f2fd', '#fff3e0', imp_color]
                cell_colors3.append(row_color)

    table3 = ax3.table(cellText=cell_data3,
                      colLabels=col_labels3,
                      cellColours=cell_colors3,
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.15, 0.25, 0.2, 0.2, 0.15])

    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1.0, 2.0)

    # Style header
    header_colors = ['#424242', '#616161', '#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(len(col_labels3)):
        cell = table3[(0, i)]
        cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor(header_colors[i])

    ax3.set_title('Complete Metrics Summary: AIPOM-CoT vs ReAct\n', fontsize=16, fontweight='bold')

    # Add summary statistics at bottom
    all_improvements = [d['improvement'] for d in improvements_data.values()]
    avg_imp = np.mean(all_improvements)
    max_imp = max(all_improvements)
    min_imp = min(all_improvements)

    summary_text = f"Summary: Avg Improvement = {avg_imp:+.1f}%  |  Max = {max_imp:+.1f}%  |  Min = {min_imp:+.1f}%"
    ax3.text(0.5, 0.02, summary_text, transform=ax3.transAxes,
             fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

    plt.tight_layout()
    save_panel(fig3, output_dir, 'fig5_complete_metrics_table')

    print("✓ Figure 5: Summary tables saved (3 tables + legend)")


# ============================================================================
# FIGURE 6: Core Capability Improvement Waterfall
# Unique info: Shows the magnitude of improvement for main capability dimensions
# Different from Fig3e: focuses on high-level capabilities, not sub-metrics
# ============================================================================

def fig6_improvement_waterfall(summary, output_dir):
    """
    Waterfall chart showing AIPOM-CoT improvement over ReAct
    Using unified panel size
    """
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    # Core capability metrics (different from sub-metrics in Fig3e)
    metrics = [
        ('planning_quality', 'Planning'),
        ('reasoning_capability', 'Reasoning'),
        ('reflection_capability', 'Self-Correction'),
        ('nlu_capability', 'Understanding'),
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

    bars = ax.bar(x, improvements, color=colors, alpha=0.85, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels
    for bar, val in zip(bars, improvements):
        y_pos = val + 1.5 if val > 0 else val - 2
        va = 'bottom' if val > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%',
               ha='center', va=va, fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement vs ReAct (%)', fontsize=11)
    ax.set_title('Core Capability Improvements', fontsize=12, fontweight='bold', loc='left')
    ax.axhline(y=0, color='black', linewidth=1.5)

    # Add reference lines
    ax.axhline(y=10, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.5)
    ax.axhline(y=20, color='#2ca02c', linestyle=':', linewidth=1.2, alpha=0.5)

    # Add average improvement line
    avg_imp = np.mean([v for v in improvements if v > 0])
    ax.axhline(y=avg_imp, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.7)
    ax.text(len(labels)-0.3, avg_imp+1.5, f'Avg: +{avg_imp:.1f}%', fontsize=9,
           color='#1f77b4', fontweight='bold')

    # Set y-axis range
    y_min = min(min(improvements) - 10, -5)
    y_max = max(max(improvements) + 10, 30)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_panel(fig, output_dir, 'fig6_improvement_waterfall')
    print("✓ Figure 6: Improvement waterfall saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  AIPOM-CoT Benchmark Visualization - Unified Panel Size Edition")
    print(f"  Panel Size: {PANEL_WIDTH}x{PANEL_HEIGHT} (aspect ratio 6:5)")
    print("="*70 + "\n")

    # Setup
    data_file = Path('./results_filtered/detailed_results_v4.json')
    output_dir = Path('./figure6_v8')
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

    # Summary of generated files
    print("📁 Generated files (unified 6:5 aspect ratio for bar charts):")
    print("-" * 60)
    files = {
        "Figure 1 - Composite Performance": [
            "fig1a_composite_metrics  (All methods, 2 composite scores)",
            "fig1b_agentic_capabilities  (AIPOM-CoT vs ReAct, 4 capabilities)",
            "fig1_combined",
        ],
        "Figure 2 - Radar Chart (10x10)": [
            "fig2_capability_radar  (Visual capability profile)",
        ],
        "Figure 3 - Sub-metric Breakdown (proportions match combined)": [
            "fig3a_planning  (3 planning sub-metrics) ~5.1x4.25",
            "fig3b_reasoning  (2 reasoning sub-metrics) ~5.1x4.25",
            "fig3c_self_correction  (3 self-correction sub-metrics) ~5.1x4.25",
            "fig3d_output_quality  (1 output quality metric) ~5.1x4.25",
            "fig3e_improvements  (All improvements, taller) ~5.1x8.5",
            "fig3_legend  (SHARED legend for all panels)",
            "fig3_combined  (18x10)",
        ],
        "Figure 4 - Single Metric Comparison": [
            "fig4a_overall  (Overall score, all methods)",
            "fig4b_reasoning  (Reasoning, all methods)",
            "fig4c_completeness  (Completeness, all methods)",
            "fig4_combined",
        ],
        "Figure 5 - Summary Tables": [
            "fig5_summary_table  (Main metrics, all methods)",
            "fig5_legend",
            "fig5_detailed_improvement_table  (All sub-metric comparisons)",
            "fig5_complete_metrics_table  (Grouped by category)",
        ],
        "Figure 6 - Improvement Waterfall": [
            "fig6_improvement_waterfall  (Core capability improvements)",
        ],
    }

    for group, fnames in files.items():
        print(f"\n{group}:")
        for f in fnames:
            print(f"   • {f}")

    # Key insights about unique information per panel
    print("\n" + "="*60)
    print("📊 UNIQUE INFORMATION PER PANEL:")
    print("-" * 60)
    print("""
    Fig1a: All 5 methods on composite scores (full comparison)
    Fig1b: AIPOM-CoT vs ReAct with improvement % annotations
    
    Fig2:  Visual radar profile (different visualization type)
    
    Fig3a-d: Sub-metric details (NO individual legend, use fig3_legend)
             - 3a: Planning (Coherence, Efficiency, Adaptability)
             - 3b: Reasoning (Logic, Evidence)
             - 3c: Self-Correction (Error Detection, Correction, Refinement)
             - 3d: Output Quality (Completeness)
    Fig3e: All 9 sub-metric improvements in one sorted view
    
    Fig4a-c: Single metric view with exact values on bars
    
    Fig5: Complete numerical data in table format
    
    Fig6: Core capability improvements (high-level, not sub-metrics)
    """)

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