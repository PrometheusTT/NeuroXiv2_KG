#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 改进版评估系统
- 更科学的指标计算
- 独立的高质量子图
- 学术论文风格可视化
"""

import json
import time
import logging
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# 学术配色方案（参考您的图片）
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


@dataclass
class ImprovedEvaluationMetrics:
    """改进的评估指标，包含更科学的计算方法"""

    # KG+CoT驱动指标（改进计算方法）
    schema_coverage: float = 0.0  # 实际使用的schema元素 / 总schema元素
    query_semantic_complexity: float = 0.0  # 基于语义特征的复杂度
    cot_effectiveness: float = 0.0  # CoT有效性：成功查询数/总步骤数
    planning_coherence: float = 0.0  # 规划连贯性：步骤间的逻辑关联度

    # 自主分析推理指标（改进计算方法）
    autonomy_index: float = 0.0  # 综合自主性指数
    reasoning_depth_score: float = 0.0  # 推理深度得分
    reflection_effectiveness: float = 0.0  # 反思有效性：改进率
    adaptation_efficiency: float = 0.0  # 适应效率：成功适应/总尝试
    problem_decomposition_quality: float = 0.0  # 问题分解质量

    # 工具调用指标（改进计算方法）
    tool_precision: float = 0.0  # 工具精确率：正确使用/总使用
    tool_recall: float = 0.0  # 工具召回率：使用的/应该使用的
    tool_f1_score: float = 0.0  # F1分数
    computational_accuracy: float = 0.0  # 计算准确性

    # 性能指标（改进计算方法）
    time_efficiency: float = 0.0  # 时间效率：1/(执行时间/复杂度)
    query_efficiency: float = 0.0  # 查询效率：成功查询/总查询
    iteration_efficiency: float = 0.0  # 迭代效率：结果改进/迭代次数
    answer_completeness: float = 0.0  # 答案完整性
    answer_correctness: float = 0.0  # 答案正确性

    # 原始数据
    execution_time: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    iteration_rounds: int = 0


class ImprovedMetricsCalculator:
    """改进的指标计算器"""

    def __init__(self, schema_info: Dict[str, Any] = None):
        """
        初始化计算器
        Args:
            schema_info: 数据库schema信息，用于更准确的计算
        """
        self.schema_info = schema_info or self._get_default_schema()

    def _get_default_schema(self) -> Dict[str, Any]:
        """获取默认的schema信息"""
        return {
            'labels': ['Region', 'Subclass', 'Neuron', 'Morphology'],
            'relationships': ['PROJECT_TO', 'HAS_SUBCLASS', 'HAS_MORPHOLOGY'],
            'properties': {
                'Region': ['acronym', 'name', 'axonal_length', 'dendritic_length',
                           'axonal_branches', 'dendritic_branches'],
                'Subclass': ['name', 'type', 'percentage'],
            }
        }

    def calculate_schema_coverage(self, queries: List[str]) -> float:
        """
        计算schema覆盖率
        基于实际使用的schema元素占总元素的比例
        """
        if not queries:
            return 0.0

        used_elements = set()
        total_elements = set()

        # 收集所有schema元素
        for label in self.schema_info['labels']:
            total_elements.add(f'label:{label}')
        for rel in self.schema_info['relationships']:
            total_elements.add(f'rel:{rel}')
        for label, props in self.schema_info.get('properties', {}).items():
            for prop in props:
                total_elements.add(f'prop:{label}.{prop}')

        # 分析查询中使用的元素
        for query in queries:
            # 提取labels
            labels = re.findall(r':([A-Z][a-zA-Z]+)', query)
            for label in labels:
                if label in self.schema_info['labels']:
                    used_elements.add(f'label:{label}')

            # 提取relationships
            rels = re.findall(r'\[[\w:]*:([A-Z_]+)', query)
            for rel in rels:
                if rel in self.schema_info['relationships']:
                    used_elements.add(f'rel:{rel}')

            # 提取properties
            props = re.findall(r'(\w+)\.(\w+)', query)
            for alias, prop in props:
                # 尝试匹配到对应的label
                for label in self.schema_info['labels']:
                    if prop in self.schema_info.get('properties', {}).get(label, []):
                        used_elements.add(f'prop:{label}.{prop}')
                        break

        coverage = len(used_elements) / len(total_elements) if total_elements else 0
        return min(coverage, 1.0)

    def calculate_query_semantic_complexity(self, queries: List[str]) -> float:
        """
        计算查询的语义复杂度
        基于查询特征的加权组合
        """
        if not queries:
            return 0.0

        complexity_features = {
            # 基础操作
            'MATCH': 1.0,
            'RETURN': 1.0,

            # 过滤和条件
            'WHERE': 2.0,
            'AND': 1.5,
            'OR': 1.5,
            'NOT': 2.0,

            # 高级操作
            'WITH': 3.0,
            'UNWIND': 3.0,
            'OPTIONAL MATCH': 3.5,
            'CASE': 4.0,
            'FOREACH': 4.0,

            # 聚合函数
            'count': 2.0,
            'sum': 2.5,
            'avg': 2.5,
            'min': 2.0,
            'max': 2.0,
            'collect': 3.0,
            'DISTINCT': 2.5,

            # 排序和限制
            'ORDER BY': 2.0,
            'LIMIT': 1.0,
            'SKIP': 1.5,

            # 字符串和数学操作
            'contains': 2.0,
            'starts with': 2.0,
            'coalesce': 2.5,
            'toFloat': 2.0,
            'toString': 2.0,

            # 路径操作
            'shortestPath': 5.0,
            'allShortestPaths': 5.5,
        }

        total_complexity = 0
        for query in queries:
            query_complexity = 0
            query_upper = query.upper()

            for feature, weight in complexity_features.items():
                count = query_upper.count(feature.upper())
                query_complexity += count * weight

            # 额外因素
            # 查询长度因素
            query_complexity *= (1 + len(query) / 1000)

            # 嵌套深度因素
            nested_depth = query.count('(') - query.count('MATCH')
            query_complexity *= (1 + nested_depth * 0.1)

            total_complexity += query_complexity

        # 归一化到0-10
        avg_complexity = total_complexity / len(queries)
        normalized = min(avg_complexity / 20, 10.0)

        return normalized

    def calculate_cot_effectiveness(self, results: List[Dict], plan: Dict) -> float:
        """
        计算思维链有效性
        基于计划执行的成功率和结果质量
        """
        if not results:
            return 0.0

        planned_steps = len(plan.get('cypher_attempts', []))
        executed_steps = len(results)
        successful_steps = sum(1 for r in results if r.get('success'))

        # 执行率
        execution_rate = executed_steps / max(planned_steps, 1)

        # 成功率
        success_rate = successful_steps / max(executed_steps, 1)

        # 数据获取率
        total_rows = sum(r.get('rows', 0) for r in results)
        data_rate = min(total_rows / (executed_steps * 10), 1.0)  # 假设每步期望10行数据

        # 综合有效性
        effectiveness = (execution_rate * 0.3 + success_rate * 0.5 + data_rate * 0.2)

        return effectiveness

    def calculate_autonomy_index(self, result: Dict) -> float:
        """
        计算综合自主性指数
        基于多个自主性特征的加权组合
        """
        score = 0.0
        weights = {
            'planning': 0.25,
            'iteration': 0.2,
            'adaptation': 0.25,
            'tool_use': 0.15,
            'reflection': 0.15
        }

        # 规划能力
        if result.get('plan'):
            plan_quality = len(result['plan'].get('cypher_attempts', [])) / 5  # 假设5个是好的规划
            score += min(plan_quality, 1.0) * weights['planning']

        # 迭代能力
        rounds = result.get('rounds', 0)
        iteration_score = min(rounds / 3, 1.0)  # 3轮是理想的
        score += iteration_score * weights['iteration']

        # 适应能力
        results_list = result.get('results', [])
        if len(results_list) > rounds:
            adaptation_score = 1.0
        else:
            adaptation_score = 0.5
        score += adaptation_score * weights['adaptation']

        # 工具使用
        if result.get('metrics'):
            tool_score = min(len(result['metrics']) / 2, 1.0)
            score += tool_score * weights['tool_use']

        # 反思能力
        if rounds > 1 and len(results_list) > len(result.get('plan', {}).get('cypher_attempts', [])):
            reflection_score = 1.0
        else:
            reflection_score = 0.3
        score += reflection_score * weights['reflection']

        return score

    def calculate_tool_metrics(self, metrics_computed: List[Dict],
                               tool_requirements: List[str]) -> Tuple[float, float, float]:
        """
        计算工具使用的精确率、召回率和F1分数
        """
        if not tool_requirements:
            # 如果不需要工具，检查是否误用
            if not metrics_computed:
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 1.0, 0.0

        # 计算实际使用和应该使用的工具
        required_tools = set(tool_requirements)
        used_tools = set()

        for metric in metrics_computed:
            if metric.get('type') == 'mismatch':
                used_tools.add('compute_mismatch_index')
            elif metric.get('type') == 'stats':
                used_tools.add('basic_stats')

        # 计算精确率和召回率
        correct_uses = used_tools & required_tools

        precision = len(correct_uses) / len(used_tools) if used_tools else 0.0
        recall = len(correct_uses) / len(required_tools) if required_tools else 0.0

        # 计算F1分数
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1


class ScientificVisualizer:
    """学术风格的可视化生成器"""

    def __init__(self, df: pd.DataFrame, output_dir: str = "evaluation_figures"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """数据预处理"""
        # 计算综合指标
        self.df['kg_cot_score'] = (
                self.df['schema_coverage'] * 0.3 +
                self.df['query_semantic_complexity'] / 10 * 0.3 +
                self.df['cot_effectiveness'] * 0.2 +
                self.df['planning_coherence'] * 0.2
        )

        self.df['autonomy_score'] = (
                self.df['autonomy_index'] * 0.3 +
                self.df['reasoning_depth_score'] * 0.2 +
                self.df['reflection_effectiveness'] * 0.2 +
                self.df['adaptation_efficiency'] * 0.2 +
                self.df['problem_decomposition_quality'] * 0.1
        )

        self.df['tool_score'] = (
                self.df['tool_precision'] * 0.3 +
                self.df['tool_recall'] * 0.3 +
                self.df['tool_f1_score'] * 0.4
        )

        self.df['overall_performance'] = (
                self.df['kg_cot_score'] * 0.3 +
                self.df['autonomy_score'] * 0.3 +
                self.df['tool_score'] * 0.2 +
                self.df['answer_completeness'] * 0.1 +
                self.df['answer_correctness'] * 0.1
        )

    def create_figure_a_projection_patterns(self):
        """创建图A：投射模式分析（类似您的图A）"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 准备数据：按类别和复杂度分组
        categories = self.df['category'].unique()
        complexities = sorted(self.df['complexity'].unique())

        # 创建矩阵数据
        matrix_data = np.zeros((len(categories), len(complexities)))
        for i, cat in enumerate(categories):
            for j, comp in enumerate(complexities):
                mask = (self.df['category'] == cat) & (self.df['complexity'] == comp)
                if mask.any():
                    matrix_data[i, j] = self.df[mask]['overall_performance'].mean()

        # 绘制热力图
        im = ax.imshow(matrix_data, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

        # 设置标签
        ax.set_xticks(np.arange(len(complexities)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels([f'Level {c}' for c in complexities])
        ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])

        # 添加数值标注
        for i in range(len(categories)):
            for j in range(len(complexities)):
                if matrix_data[i, j] > 0:
                    text = ax.text(j, i, f'{matrix_data[i, j]:.2f}',
                                   ha="center", va="center",
                                   color="white" if matrix_data[i, j] < 0.5 else "black",
                                   fontsize=8)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Performance Score', rotation=270, labelpad=15)

        ax.set_title('A. Performance Matrix by Category and Complexity',
                     fontweight='bold', pad=20)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Test Category')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_a_performance_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_a_performance_matrix.png'

    def create_figure_b_mismatch_analysis(self):
        """创建图B：不匹配分析（类似您的图B）"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # 创建相关性矩阵
        metrics = ['schema_coverage', 'query_semantic_complexity', 'cot_effectiveness',
                   'autonomy_index', 'reasoning_depth_score', 'tool_f1_score',
                   'answer_completeness', 'answer_correctness']

        corr_matrix = self.df[metrics].corr()

        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot=True, fmt='.2f', annot_kws={'size': 8},
                    vmin=-1, vmax=1, ax=ax)

        ax.set_title('B. Metric Correlation Matrix', fontweight='bold', pad=20)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics], rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_b_correlation_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_b_correlation_matrix.png'

    def create_figure_c_radar_comparisons(self):
        """创建图C：雷达图比较（类似您的图C）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))

        # 选择三个代表性的测试进行比较
        test_groups = [
            self.df[self.df['complexity'] <= 2],  # 简单测试
            self.df[(self.df['complexity'] > 2) & (self.df['complexity'] <= 4)],  # 中等测试
            self.df[self.df['complexity'] > 4]  # 复杂测试
        ]

        labels = ['Simple Tests', 'Medium Tests', 'Complex Tests']
        metrics = ['Schema\nCoverage', 'Query\nComplexity', 'CoT\nEffectiveness',
                   'Autonomy', 'Tool Use', 'Answer\nQuality']

        for idx, (ax, group, label) in enumerate(zip(axes, test_groups, labels)):
            if group.empty:
                continue

            # 计算平均值
            values = [
                group['schema_coverage'].mean(),
                group['query_semantic_complexity'].mean() / 10,
                group['cot_effectiveness'].mean(),
                group['autonomy_index'].mean(),
                group['tool_f1_score'].mean(),
                group['answer_correctness'].mean()
            ]

            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            ax.plot(angles, values, 'o-', linewidth=2, color=ACADEMIC_COLORS['blue'])
            ax.fill(angles, values, alpha=0.25, color=ACADEMIC_COLORS['blue'])

            # 添加对比（如果有baseline）
            baseline = [0.5] * len(metrics)
            baseline += baseline[:1]
            ax.plot(angles, baseline, '--', linewidth=1, color=ACADEMIC_COLORS['red'],
                    alpha=0.5, label='Baseline')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, size=8)
            ax.set_ylim([0, 1])
            ax.set_title(label, fontweight='bold', pad=20)
            ax.grid(True, linestyle='--', alpha=0.3)

            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        fig.suptitle('C. Performance Radar Analysis by Complexity',
                     fontweight='bold', y=1.05)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_c_radar_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_c_radar_analysis.png'

    def create_figure_d_distance_analysis(self):
        """创建图D：距离分析（下方的距离矩阵）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 为每个类别创建距离分析
        categories = self.df['category'].unique()[:3]  # 取前3个类别

        for idx, (ax_top, ax_bottom, cat) in enumerate(zip(axes[0], axes[1], categories)):
            cat_data = self.df[self.df['category'] == cat]

            if cat_data.empty:
                continue

            # 上图：形态学距离
            metrics_morph = ['schema_coverage', 'query_semantic_complexity', 'cot_effectiveness']
            morph_data = cat_data[metrics_morph].values

            # 计算距离矩阵
            n_samples = min(len(morph_data), 5)  # 最多5个样本
            morph_dist = np.zeros((n_samples, n_samples))

            for i in range(n_samples):
                for j in range(n_samples):
                    morph_dist[i, j] = euclidean(morph_data[i], morph_data[j])

            im1 = ax_top.imshow(morph_dist, cmap='Blues', aspect='auto')
            ax_top.set_title(f'Morphological Distance\n{cat.replace("_", " ").title()}',
                             fontsize=10)
            ax_top.set_xticks(range(n_samples))
            ax_top.set_yticks(range(n_samples))
            ax_top.set_xticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)
            ax_top.set_yticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)

            # 添加数值
            for i in range(n_samples):
                for j in range(n_samples):
                    ax_top.text(j, i, f'{morph_dist[i, j]:.2f}',
                                ha='center', va='center', fontsize=7,
                                color='white' if morph_dist[i, j] > morph_dist.max() / 2 else 'black')

            # 下图：功能距离
            metrics_func = ['autonomy_index', 'tool_f1_score', 'answer_correctness']
            func_data = cat_data[metrics_func].values[:n_samples]

            func_dist = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    func_dist[i, j] = euclidean(func_data[i], func_data[j])

            im2 = ax_bottom.imshow(func_dist, cmap='Reds', aspect='auto')
            ax_bottom.set_title('Functional Distance', fontsize=10)
            ax_bottom.set_xticks(range(n_samples))
            ax_bottom.set_yticks(range(n_samples))
            ax_bottom.set_xticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)
            ax_bottom.set_yticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)

            # 添加数值
            for i in range(n_samples):
                for j in range(n_samples):
                    ax_bottom.text(j, i, f'{func_dist[i, j]:.2f}',
                                   ha='center', va='center', fontsize=7,
                                   color='white' if func_dist[i, j] > func_dist.max() / 2 else 'black')

            # 添加颜色条
            if idx == 2:
                plt.colorbar(im1, ax=ax_top, fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=ax_bottom, fraction=0.046, pad=0.04)

        fig.suptitle('D. Morphological and Functional Distance Analysis',
                     fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_d_distance_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_d_distance_analysis.png'

    def create_all_figures(self):
        """生成所有图表"""
        print("\n生成学术风格的评估图表...")

        figures = []

        # 生成各个子图
        fig_a = self.create_figure_a_projection_patterns()
        print(f"  ✓ Figure A saved to: {fig_a}")
        figures.append(fig_a)

        fig_b = self.create_figure_b_mismatch_analysis()
        print(f"  ✓ Figure B saved to: {fig_b}")
        figures.append(fig_b)

        fig_c = self.create_figure_c_radar_comparisons()
        print(f"  ✓ Figure C saved to: {fig_c}")
        figures.append(fig_c)

        fig_d = self.create_figure_d_distance_analysis()
        print(f"  ✓ Figure D saved to: {fig_d}")
        figures.append(fig_d)

        print(f"\n✅ 所有图表已生成到: {self.output_dir}")
        return figures


class ImprovedEvaluator:
    """改进的评估器主类"""

    def __init__(self, agent):
        self.agent = agent
        self.metrics_calculator = ImprovedMetricsCalculator()
        self.results = []

    def evaluate_single(self, test_case: 'TestCase', timeout: int = 60) -> Tuple[Dict, ImprovedEvaluationMetrics]:
        """评估单个测试用例"""
        print(f"评估: {test_case.id}")

        start_time = time.time()

        # 执行测试
        try:
            result = self.agent.answer(test_case.question, max_rounds=3)
        except Exception as e:
            print(f"  错误: {e}")
            result = {'error': str(e), 'results': [], 'plan': {}, 'metrics': [], 'final': ''}

        execution_time = time.time() - start_time

        # 计算改进的指标
        metrics = self._calculate_improved_metrics(result, test_case, execution_time)

        return result, metrics

    def _calculate_improved_metrics(self, result: Dict, test_case: 'TestCase',
                                    exec_time: float) -> ImprovedEvaluationMetrics:
        """计算改进的评估指标"""
        metrics = ImprovedEvaluationMetrics()

        # 基础指标
        metrics.execution_time = exec_time
        metrics.iteration_rounds = result.get('rounds', 0)

        # 获取查询和结果
        all_queries = []
        results_list = result.get('results', [])
        for r in results_list:
            if 'query' in r:
                all_queries.append(r['query'])

        plan = result.get('plan', {})

        # KG+CoT指标
        metrics.schema_coverage = self.metrics_calculator.calculate_schema_coverage(all_queries)
        metrics.query_semantic_complexity = self.metrics_calculator.calculate_query_semantic_complexity(all_queries)
        metrics.cot_effectiveness = self.metrics_calculator.calculate_cot_effectiveness(results_list, plan)
        metrics.planning_coherence = self._calculate_planning_coherence(plan, results_list)

        # 自主性指标
        metrics.autonomy_index = self.metrics_calculator.calculate_autonomy_index(result)
        metrics.reasoning_depth_score = self._calculate_reasoning_depth(result)
        metrics.reflection_effectiveness = self._calculate_reflection_effectiveness(result)
        metrics.adaptation_efficiency = self._calculate_adaptation_efficiency(results_list)
        metrics.problem_decomposition_quality = self._calculate_decomposition_quality(plan)

        # 工具指标
        metrics_computed = result.get('metrics', [])
        precision, recall, f1 = self.metrics_calculator.calculate_tool_metrics(
            metrics_computed, test_case.tool_requirements
        )
        metrics.tool_precision = precision
        metrics.tool_recall = recall
        metrics.tool_f1_score = f1
        metrics.computational_accuracy = self._calculate_computational_accuracy(metrics_computed)

        # 性能指标
        metrics.total_queries = len(results_list)
        metrics.successful_queries = sum(1 for r in results_list if r.get('success'))
        metrics.query_efficiency = metrics.successful_queries / max(metrics.total_queries, 1)
        metrics.time_efficiency = self._calculate_time_efficiency(exec_time, test_case.complexity)
        metrics.iteration_efficiency = self._calculate_iteration_efficiency(result)

        # 答案质量
        metrics.answer_completeness = self._calculate_answer_completeness(result, test_case)
        metrics.answer_correctness = self._calculate_answer_correctness(result, test_case)

        return metrics

    def _calculate_planning_coherence(self, plan: Dict, results: List[Dict]) -> float:
        """计算规划连贯性"""
        if not plan or not results:
            return 0.0

        attempts = plan.get('cypher_attempts', [])
        if not attempts:
            return 0.0

        # 检查计划的逻辑连贯性
        coherence_score = 0.0

        # 1. 检查步骤是否有明确目的
        purposes = [a.get('purpose', '') for a in attempts]
        if all(p and len(p) > 10 for p in purposes):
            coherence_score += 0.3

        # 2. 检查步骤间的逻辑关系
        if len(attempts) > 1:
            # 检查是否有递进关系
            queries = [a.get('query', '') for a in attempts]
            has_progression = any(
                q2.count('WITH') > q1.count('WITH') or
                len(q2) > len(q1) * 1.2
                for q1, q2 in zip(queries[:-1], queries[1:])
            )
            if has_progression:
                coherence_score += 0.3

        # 3. 检查执行是否遵循计划
        execution_rate = min(len(results) / len(attempts), 1.0)
        coherence_score += execution_rate * 0.4

        return coherence_score

    def _calculate_reasoning_depth(self, result: Dict) -> float:
        """计算推理深度得分"""
        depth = 0.0

        # 基于迭代轮数
        rounds = result.get('rounds', 0)
        depth += min(rounds / 3, 1.0) * 0.3

        # 基于查询复杂度递增
        results = result.get('results', [])
        if len(results) > 1:
            complexities = []
            for r in results:
                query = r.get('query', '')
                complexity = query.count('WITH') + query.count('WHERE') + query.count('MATCH')
                complexities.append(complexity)

            # 检查是否有递增趋势
            if len(complexities) > 1:
                increasing = sum(c2 > c1 for c1, c2 in zip(complexities[:-1], complexities[1:]))
                depth += (increasing / (len(complexities) - 1)) * 0.3

        # 基于结果数据量
        total_rows = sum(r.get('rows', 0) for r in results)
        depth += min(total_rows / 100, 1.0) * 0.2

        # 基于工具使用
        if result.get('metrics'):
            depth += 0.2

        return depth

    def _calculate_reflection_effectiveness(self, result: Dict) -> float:
        """计算反思有效性"""
        rounds = result.get('rounds', 0)
        results = result.get('results', [])

        if rounds <= 1:
            return 0.0

        # 检查后续轮次是否有改进
        if len(results) > rounds:
            # 有额外尝试，说明在反思
            effectiveness = 0.5

            # 检查成功率是否提升
            early_results = results[:len(results) // 2]
            late_results = results[len(results) // 2:]

            early_success = sum(1 for r in early_results if r.get('success')) / max(len(early_results), 1)
            late_success = sum(1 for r in late_results if r.get('success')) / max(len(late_results), 1)

            if late_success > early_success:
                effectiveness += 0.5

            return effectiveness

        return 0.3

    def _calculate_adaptation_efficiency(self, results: List[Dict]) -> float:
        """计算适应效率"""
        if not results:
            return 0.0

        # 检查失败后的恢复能力
        adaptation_score = 0.0
        failed_indices = [i for i, r in enumerate(results) if not r.get('success')]

        if not failed_indices:
            # 全部成功
            return 1.0

        # 检查失败后是否有成功的尝试
        for fail_idx in failed_indices:
            if fail_idx < len(results) - 1:
                # 检查后续是否成功
                if any(r.get('success') for r in results[fail_idx + 1:]):
                    adaptation_score += 1.0

        if failed_indices:
            adaptation_score = adaptation_score / len(failed_indices)

        return min(adaptation_score, 1.0)

    def _calculate_decomposition_quality(self, plan: Dict) -> float:
        """计算问题分解质量"""
        if not plan:
            return 0.0

        attempts = plan.get('cypher_attempts', [])
        if not attempts:
            return 0.0

        quality = 0.0

        # 检查分解的合理性
        if len(attempts) >= 2:
            quality += 0.3

        # 检查每个子任务是否有明确目标
        purposes = [a.get('purpose', '') for a in attempts]
        if all(p for p in purposes):
            quality += 0.3

            # 检查目标是否不重复
            unique_purposes = len(set(purposes))
            if unique_purposes == len(purposes):
                quality += 0.2

        # 检查分析计划
        if plan.get('analysis_plan'):
            quality += 0.2

        return quality

    def _calculate_computational_accuracy(self, metrics_computed: List[Dict]) -> float:
        """计算计算准确性"""
        if not metrics_computed:
            return 1.0  # 没有计算就没有错误

        accuracy = 1.0

        for metric in metrics_computed:
            # 检查计算结果的合理性
            if metric.get('type') == 'mismatch':
                value = metric.get('value', 0)
                # Mismatch应该在0-1之间
                if not (0 <= value <= 1):
                    accuracy -= 0.5

            elif metric.get('type') == 'stats':
                data = metric.get('data', {})
                # 检查统计数据的合理性
                if 'mean' in data and 'std' in data:
                    if data['std'] < 0:  # 标准差不能为负
                        accuracy -= 0.5

        return max(accuracy, 0.0)

    def _calculate_time_efficiency(self, exec_time: float, complexity: int) -> float:
        """计算时间效率"""
        # 期望时间：复杂度 * 2秒
        expected_time = complexity * 2

        if exec_time <= expected_time:
            return 1.0
        elif exec_time <= expected_time * 1.5:
            return 0.8
        elif exec_time <= expected_time * 2:
            return 0.6
        else:
            return max(0.3, expected_time / exec_time)

    def _calculate_iteration_efficiency(self, result: Dict) -> float:
        """计算迭代效率"""
        rounds = result.get('rounds', 0)
        if rounds <= 1:
            return 0.5  # 没有迭代

        # 检查迭代是否带来改进
        results = result.get('results', [])
        if not results:
            return 0.0

        # 计算每轮的平均成功率
        results_per_round = len(results) / rounds
        success_rates = []

        for i in range(rounds):
            start_idx = int(i * results_per_round)
            end_idx = int((i + 1) * results_per_round)
            round_results = results[start_idx:end_idx]

            if round_results:
                success_rate = sum(1 for r in round_results if r.get('success')) / len(round_results)
                success_rates.append(success_rate)

        # 检查是否有提升趋势
        if len(success_rates) > 1:
            improvements = sum(sr2 > sr1 for sr1, sr2 in zip(success_rates[:-1], success_rates[1:]))
            efficiency = improvements / (len(success_rates) - 1)
            return efficiency

        return 0.5

    def _calculate_answer_completeness(self, result: Dict, test_case: 'TestCase') -> float:
        """计算答案完整性"""
        final_answer = result.get('final', '')
        if not final_answer:
            return 0.0

        completeness = 0.0

        # 检查答案长度
        if len(final_answer) > 100:
            completeness += 0.2

        # 检查是否包含数据支撑
        if any(r.get('rows', 0) > 0 for r in result.get('results', [])):
            completeness += 0.3

        # 检查是否包含预期的关键词
        for pattern in test_case.expected_patterns:
            if pattern.lower() in final_answer.lower():
                completeness += 0.1

        # 检查是否有结构化内容
        if final_answer.count('\n') > 2:
            completeness += 0.1

        # 检查是否有数值
        if re.search(r'\d+\.?\d*', final_answer):
            completeness += 0.1

        return min(completeness, 1.0)

    def _calculate_answer_correctness(self, result: Dict, test_case: 'TestCase') -> float:
        """计算答案正确性"""
        final_answer = result.get('final', '')
        if not final_answer:
            return 0.0

        correctness = 0.0

        # 基于查询成功率
        results = result.get('results', [])
        if results:
            success_rate = sum(1 for r in results if r.get('success')) / len(results)
            correctness += success_rate * 0.4

        # 基于关键概念覆盖
        covered_patterns = 0
        for pattern in test_case.expected_patterns:
            if pattern.lower() in final_answer.lower():
                covered_patterns += 1

        if test_case.expected_patterns:
            pattern_coverage = covered_patterns / len(test_case.expected_patterns)
            correctness += pattern_coverage * 0.3

        # 基于工具使用正确性
        if test_case.tool_requirements:
            tool_score = result.get('metrics', [])
            if tool_score:
                correctness += 0.3
        else:
            # 不需要工具的情况
            correctness += 0.3

        return min(correctness, 1.0)


def run_improved_evaluation(agent, test_cases: List['TestCase'], output_dir: str = "evaluation_results"):
    """运行改进的评估"""
    print("\n" + "=" * 60)
    print("运行改进的KGAgent V7评估系统")
    print("=" * 60)

    evaluator = ImprovedEvaluator(agent)
    all_metrics = []

    # 评估每个测试用例
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 评估测试: {test_case.id}")
        print(f"  问题: {test_case.question}")

        try:
            result, metrics = evaluator.evaluate_single(test_case)

            # 转换为字典
            metric_dict = {
                'test_id': test_case.id,
                'category': test_case.category,
                'complexity': test_case.complexity,
                **metrics.__dict__
            }
            all_metrics.append(metric_dict)

            print(f"  ✓ 完成 - 综合得分: {metrics.answer_correctness:.2f}")

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            all_metrics.append({
                'test_id': test_case.id,
                'category': test_case.category,
                'complexity': test_case.complexity,
                'error': str(e)
            })

    # 创建DataFrame
    df = pd.DataFrame(all_metrics)

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / "improved_evaluation_results.csv", index=False)
    print(f"\n✅ 结果已保存到: {output_path / 'improved_evaluation_results.csv'}")

    # 生成可视化
    print("\n生成学术风格可视化...")
    visualizer = ScientificVisualizer(df, str(output_path))
    visualizer.create_all_figures()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("评估统计摘要")
    print("=" * 60)

    valid_df = df[~df.get('error', pd.Series()).notna()]
    if not valid_df.empty:
        print(f"\n完成测试: {len(valid_df)}/{len(df)}")
        print(f"\n核心指标:")
        print(f"  • Schema覆盖率: {valid_df['schema_coverage'].mean():.3f} ± {valid_df['schema_coverage'].std():.3f}")
        print(
            f"  • 查询语义复杂度: {valid_df['query_semantic_complexity'].mean():.2f} ± {valid_df['query_semantic_complexity'].std():.2f}")
        print(f"  • CoT有效性: {valid_df['cot_effectiveness'].mean():.3f} ± {valid_df['cot_effectiveness'].std():.3f}")
        print(f"  • 自主性指数: {valid_df['autonomy_index'].mean():.3f} ± {valid_df['autonomy_index'].std():.3f}")
        print(f"  • 工具F1分数: {valid_df['tool_f1_score'].mean():.3f} ± {valid_df['tool_f1_score'].std():.3f}")
        print(
            f"  • 答案正确性: {valid_df['answer_correctness'].mean():.3f} ± {valid_df['answer_correctness'].std():.3f}")

        print(f"\n性能指标:")
        print(f"  • 平均执行时间: {valid_df['execution_time'].mean():.2f}s")
        print(f"  • 查询效率: {valid_df['query_efficiency'].mean():.1%}")
        print(f"  • 时间效率: {valid_df['time_efficiency'].mean():.3f}")

    return df


# 主函数
if __name__ == "__main__":
    # 这里可以导入您的测试用例和Agent
    print("改进的评估系统已准备就绪")
    print("使用方法:")
    print("  from improved_evaluation import run_improved_evaluation")
    print("  results = run_improved_evaluation(agent, test_cases)")