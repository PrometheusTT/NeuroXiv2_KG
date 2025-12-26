"""
Plot Classification Results from CSV
=====================================
读取classification_results.csv，绘制单模态vs多模态对比图

Author: Claude
Date: 2025-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_results(csv_path:  str) -> pd.DataFrame:
    """加载分类结果CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    print("\nResults summary:")
    print(df[['task', 'input', 'target', 'n_clusters', 'train_accuracy', 'test_accuracy']].to_string())
    return df


def plot_classification_comparison(df: pd.DataFrame, output_path:  str = None,
                                   use_late_fusion: bool = True):
    """
    绘制单模态vs多模态分类准确率对比图

    Parameters:
    -----------
    df : pd.DataFrame
        classification_results.csv的内容
    output_path : str
        输出文件路径
    use_late_fusion : bool
        是否使用late fusion结果（否则使用feature concat）
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 定义三组任务
    task_groups = [
        {
            'title': 'Predict Projection',
            'target': 'Proj',
            'single_tasks': [
                ('morph_to_proj', 'Morph'),
                ('gene_to_proj', 'Gene'),
            ],
            'fusion_task': 'morph_gene_to_proj_fusion' if use_late_fusion else 'morph_gene_to_proj',
            'fusion_label': 'Morph+Gene',
            'color_light': '#a8d5e5',  # 浅蓝
            'color_dark': '#3498db',   # 深蓝
        },
        {
            'title':  'Predict Morphology',
            'target': 'Morph',
            'single_tasks': [
                ('gene_to_morph', 'Gene'),
                ('proj_to_morph', 'Proj'),
            ],
            'fusion_task': 'gene_proj_to_morph_fusion' if use_late_fusion else 'gene_proj_to_morph',
            'fusion_label': 'Gene+Proj',
            'color_light': '#a8e6cf',  # 浅绿
            'color_dark':  '#27ae60',   # 深绿
        },
        {
            'title':  'Predict Molecular',
            'target': 'Gene',
            'single_tasks': [
                ('morph_to_gene', 'Morph'),
                ('proj_to_gene', 'Proj'),
            ],
            'fusion_task':  'morph_proj_to_gene_fusion' if use_late_fusion else 'morph_proj_to_gene',
            'fusion_label': 'Morph+Proj',
            'color_light': '#f5b7b1',  # 浅红
            'color_dark': '#e74c3c',   # 深红
        },
    ]

    for ax, group in zip(axes, task_groups):
        labels = []
        train_accs = []
        test_accs = []
        n_clusters = None

        # 获取单模态结果
        for task_name, label in group['single_tasks']:
            row = df[df['task'] == task_name]
            if len(row) > 0:
                labels.append(label)
                train_accs.append(row['train_accuracy'].values[0])
                test_accs.append(row['test_accuracy'].values[0])
                # 获取n_clusters（从第一个匹配的任务）
                if n_clusters is None:
                    n_clusters = int(row['n_clusters'].values[0])

        # 获取多模态结果
        fusion_row = df[df['task'] == group['fusion_task']]
        if len(fusion_row) > 0:
            labels.append(group['fusion_label'])
            train_accs.append(fusion_row['train_accuracy'].values[0])
            test_accs.append(fusion_row['test_accuracy'].values[0])
            # 也可以从fusion结果获取n_clusters
            if n_clusters is None:
                n_clusters = int(fusion_row['n_clusters'].values[0])

        # 如果还是没有获取到n_clusters，设置默认值
        if n_clusters is None or n_clusters <= 0:
            print(f"Warning:  Could not get n_clusters for {group['title']}, using default")
            n_clusters = 10

        # 计算随机基线
        baseline = 1.0 / n_clusters

        print(f"\n{group['title']}:  K={n_clusters}, baseline={baseline:.4f}")
        for label, train_acc, test_acc in zip(labels, train_accs, test_accs):
            print(f"  {label}: train={train_acc:.4f}, test={test_acc:.4f}")

        # 绑图
        x = np.arange(len(labels))
        width = 0.35

        # Train bars (浅色)
        bars_train = ax.bar(x - width/2, train_accs, width,
                           label='Train', color=group['color_light'],
                           edgecolor='white', linewidth=0.5)

        # Test bars (深色)
        bars_test = ax.bar(x + width/2, test_accs, width,
                          label='Test', color=group['color_dark'],
                          edgecolor='white', linewidth=0.5)

        # 添加数值标注（只标注Test准确率）
        for i, (bar, acc) in enumerate(zip(bars_test, test_accs)):
            ax.annotate(f'{acc:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, acc),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='medium')

        # 添加随机基线
        ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # 设置
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{group["title"]} (K={n_clusters})', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        # 图例
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

        # 在图例旁边添加Random baseline说明
        ax.text(0.02, baseline + 0.02, f'Random ({baseline:.2f})',
               transform=ax.get_yaxis_transform(),
               fontsize=9, color='gray', va='bottom')

        # 网格
        ax.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)

        # 移除上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 总标题
    fusion_type = 'Late Fusion' if use_late_fusion else 'Feature Concat'
    plt.suptitle(f'Classification Accuracy ({fusion_type})',
                fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Figure saved to: {output_path}")

    plt.show()
    return fig


def plot_both_fusion_methods(df: pd.DataFrame, output_dir: str = "."):
    """同时绘制Late Fusion和Feature Concat的对比图"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Late Fusion版本
    print("\n" + "="*60)
    print("Plotting Late Fusion version...")
    print("="*60)
    fig1 = plot_classification_comparison(
        df,
        output_path=f"{output_dir}/classification_accuracy_late_fusion.png",
        use_late_fusion=True
    )

    # Feature Concat版本
    print("\n" + "="*60)
    print("Plotting Feature Concat version...")
    print("="*60)
    fig2 = plot_classification_comparison(
        df,
        output_path=f"{output_dir}/classification_accuracy_feature_concat.png",
        use_late_fusion=False
    )

    return fig1, fig2


def main():
    parser = argparse.ArgumentParser(description='Plot classification results from CSV')
    parser.add_argument('--csv', type=str, default='./classification_results_v3/classification_results.csv',
                       help='Path to classification_results.csv')
    parser.add_argument('--output', type=str, default='./classification_results_v3',
                       help='Output directory for figures')
    parser.add_argument('--fusion-type', type=str, choices=['late', 'concat', 'both'], default='both',
                       help='Which fusion type to plot')
    args = parser.parse_args()

    # 加载数据
    df = load_results(args.csv)

    # 检查n_clusters列是否存在
    if 'n_clusters' not in df.columns:
        print("\nError: 'n_clusters' column not found in CSV!")
        print("Available columns:", df.columns.tolist())
        return

    # 打印n_clusters信息
    print("\nCluster info:")
    for target in ['Proj', 'Morph', 'Gene']:
        rows = df[df['target'] == target]
        if len(rows) > 0:
            k = rows['n_clusters'].iloc[0]
            print(f"  {target}: K={k}, baseline={1/k:.4f}")

    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 绑图
    if args.fusion_type == 'late':
        plot_classification_comparison(
            df,
            output_path=f"{args.output}/classification_accuracy_late_fusion.png",
            use_late_fusion=True
        )
    elif args.fusion_type == 'concat':
        plot_classification_comparison(
            df,
            output_path=f"{args.output}/classification_accuracy_feature_concat.png",
            use_late_fusion=False
        )
    else:
        plot_both_fusion_methods(df, args.output)


if __name__ == "__main__":
    main()