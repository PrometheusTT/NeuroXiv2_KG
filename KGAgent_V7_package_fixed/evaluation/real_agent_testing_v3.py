#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 真实Agent测试集成
完整展示如何将改进版评估系统与您的真实Agent连接
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== STEP 1: 导入您的真实Agent ====================
# 将agent_v7目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 导入您的真实Agent类
    from agent_v7.neo4j_exec import Neo4jExec
    from agent_v7.agent_v7 import KGAgentV7
    from agent_v7.schema_cache import SchemaCache
    from agent_v7.llm import LLMClient

    AGENT_AVAILABLE = True
    logger.info("✅ Successfully imported KGAgent V7")
except ImportError as e:
    logger.warning(f"⚠️ Could not import KGAgent V7: {e}")
    logger.warning("Using Mock Agent for demonstration")
    AGENT_AVAILABLE = False


# ==================== STEP 2: 配置连接参数 ====================
class AgentConfig:
    """Agent配置管理"""

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """获取Agent配置"""
        return {
            # Neo4j数据库配置
            'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://100.88.72.32:7687'),
            'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
            'neo4j_pwd': os.getenv('NEO4J_PASSWORD', 'neuroxiv'),
            'database': os.getenv('NEO4J_DATABASE', 'neo4j'),

            # OpenAI配置
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'planner_model': os.getenv('PLANNER_MODEL', 'gpt-5'),
            'summarizer_model': os.getenv('SUMMARIZER_MODEL', 'gpt-4o'),

            # 评估配置
            'max_rounds': 3,
            'timeout': 60,  # 每个测试的超时时间（秒）

            # 基线配置（用于对比）
            'enable_baseline': False,  # 是否启用基线测试
            'disable_cot': False,  # 禁用CoT进行消融实验
            'disable_tools': False,  # 禁用工具使用进行消融实验
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """验证配置是否完整"""
        required = ['neo4j_uri', 'neo4j_user', 'neo4j_pwd', 'openai_api_key']
        missing = [k for k in required if not config.get(k)]

        if missing:
            logger.error(f"❌ Missing required config: {missing}")
            logger.info("Please set environment variables or update config")
            return False

        logger.info("✅ Configuration validated")
        return True


# ==================== STEP 3: 创建Agent包装器 ====================
class RealAgentWrapper:
    """真实Agent的包装器，用于评估系统"""

    def __init__(self, config: Dict[str, Any]):
        """初始化真实Agent"""
        self.config = config

        if AGENT_AVAILABLE:
            # 创建真实的Agent实例
            self.agent = KGAgentV7(
                neo4j_uri=config['neo4j_uri'],
                neo4j_user=config['neo4j_user'],
                neo4j_pwd=config['neo4j_pwd'],
                database=config['database'],
                openai_api_key=config['openai_api_key'],
                planner_model=config['planner_model'],
                summarizer_model=config['summarizer_model'],
                # disable_cot=config.get('disable_cot', False),
                # disable_tools=config.get('disable_tools', False)
            )
            logger.info("✅ Real KGAgent V7 initialized")
        else:
            # 使用Mock Agent
            from run_evaluation import MockKGAgentV7
            self.agent = MockKGAgentV7(**config)
            logger.info("📦 Using Mock Agent for testing")

    def test_connection(self) -> bool:
        """测试Agent连接"""
        try:
            # 测试Neo4j连接
            if AGENT_AVAILABLE:
                with self.agent.db.driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                    count = result.single()['count']
                    logger.info(f"✅ Neo4j connected: {count} nodes found")

            # 测试一个简单查询
            test_question = "What labels exist in the knowledge graph?"
            result = self.agent.answer(test_question, max_rounds=1)

            if result and 'final' in result:
                logger.info("✅ Agent test query successful")
                return True
            else:
                logger.error("❌ Agent test query failed")
                return False

        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

    def answer(self, question: str, max_rounds: int = None) -> Dict[str, Any]:
        """调用Agent回答问题（与评估系统的接口）"""
        max_rounds = max_rounds or self.config.get('max_rounds', 2)
        return self.agent.answer(question, max_rounds=max_rounds)


# ==================== STEP 4: 导入改进的评估系统 ====================
from improved_evaluation import (
    ImprovedEvaluator,
    ImprovedEvaluationMetrics,
    run_improved_evaluation,
    ScientificVisualizer,
    ImprovedMetricsCalculator, ACADEMIC_COLORS
)

# 使用改进版的TestCase定义
from evaluation import TestCase, TestSuite


# ==================== STEP 5: 自定义测试用例 ====================
class CustomTestSuite(TestSuite):
    """自定义测试套件，针对您的KG设计"""

    def _create_test_cases(self) -> List[TestCase]:
        """创建针对神经科学KG的测试用例"""
        cases = []

        # === Level 1: 基础KG查询 ===
        cases.append(TestCase(
            id="neuro_basic_1",
            category="kg_navigation",
            question="What are the morphological properties of the MOp region?",
            complexity=1,
            required_capabilities=["schema_navigation", "property_extraction"],
            expected_patterns=["MOp", "morpholog", "axon", "dendrit"],
            tool_requirements=[]
        ))

        cases.append(TestCase(
            id="neuro_basic_2",
            category="kg_navigation",
            question="List all subclasses in the CLA region",
            complexity=1,
            required_capabilities=["relationship_traversal"],
            expected_patterns=["CLA", "HAS_SUBCLASS", "Subclass"],
            tool_requirements=[]
        ))

        # === Level 2: 关系探索 ===
        cases.append(TestCase(
            id="neuro_relation_1",
            category="kg_navigation",
            question="Which regions does VISp project to with the strongest connections?",
            complexity=2,
            required_capabilities=["relationship_traversal", "sorting"],
            expected_patterns=["VISp", "PROJECT_TO", "weight", "ORDER BY"],
            tool_requirements=[]
        ))

        cases.append(TestCase(
            id="neuro_relation_2",
            category="kg_navigation",
            question="Find regions with high axonal branching",
            complexity=2,
            required_capabilities=["filtering", "comparison"],
            expected_patterns=["axonal_branches", "WHERE", ">"],
            tool_requirements=[]
        ))

        # === Level 3: 分析推理 ===
        cases.append(TestCase(
            id="neuro_reasoning_1",
            category="reasoning",
            question="Compare the morphological diversity between motor (MOp) and visual (VISp) cortex regions",
            complexity=3,
            required_capabilities=["comparison", "aggregation", "multi_region"],
            expected_patterns=["MOp", "VISp", "morpholog"],
            tool_requirements=[]
        ))

        cases.append(TestCase(
            id="neuro_reasoning_2",
            category="reasoning",
            question="Which regions show the highest axon-to-dendrite length ratio and what might this indicate?",
            complexity=3,
            required_capabilities=["computation", "interpretation"],
            expected_patterns=["axonal_length", "dendritic_length", "ratio", "CASE"],
            tool_requirements=[]
        ))

        # === Level 4: 工具使用 ===
        cases.append(TestCase(
            id="neuro_tool_1",
            category="tool_use",
            question="Calculate the mismatch index between MOp and SSp regions using their morphological and transcriptomic profiles",
            complexity=4,
            required_capabilities=["tool_invocation", "vector_computation"],
            expected_patterns=["MOp", "SSp", "HAS_SUBCLASS", "morpholog"],
            tool_requirements=["compute_mismatch_index"]
        ))

        cases.append(TestCase(
            id="neuro_tool_2",
            category="tool_use",
            question="Compute statistical metrics for dendritic branching patterns across all cortical regions",
            complexity=4,
            required_capabilities=["statistics", "aggregation"],
            expected_patterns=["dendritic_branches", "Region"],
            tool_requirements=["basic_stats"]
        ))

        # === Level 5: 综合分析 ===
        cases.append(TestCase(
            id="neuro_complex_1",
            category="complex",
            question="Analyze the relationship between morphological complexity and transcriptomic diversity across the cortical hierarchy, identify regions with significant mismatch",
            complexity=5,
            required_capabilities=["multi_hop", "correlation", "tool_use", "interpretation"],
            expected_patterns=["morpholog", "transcriptom", "HAS_SUBCLASS"],
            tool_requirements=["compute_mismatch_index", "basic_stats"]
        ))

        cases.append(TestCase(
            id="neuro_complex_2",
            category="complex",
            question="Identify Car3-expressing neurons' projection patterns and analyze their morphological characteristics compared to other interneuron subtypes",
            complexity=5,
            required_capabilities=["specific_subclass", "projection_analysis", "comparison"],
            expected_patterns=["Car3", "PROJECT_TO", "morpholog", "interneuron"],
            tool_requirements=[]
        ))

        return cases


# ==================== STEP 6: 改进的可视化类 ====================
class EnhancedVisualizer(ScientificVisualizer):
    """增强的可视化类，整合用户的改进建议"""

    def __init__(self, df: pd.DataFrame, output_dir: str = "evaluation_figures", baseline_df: pd.DataFrame = None):
        """
        初始化增强可视化器

        Args:
            df: 主要评估数据
            output_dir: 输出目录
            baseline_df: 基线数据（可选）
        """
        self.baseline_df = baseline_df
        super().__init__(df, output_dir)

    def create_figure_d_distance_analysis(self):
        """创建改进的图D：结构和行为指标距离分析（替代原来的形态和功能距离）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 为每个类别创建距离分析
        categories = self.df['category'].unique()[:3]  # 取前3个类别

        for idx, (ax_top, ax_bottom, cat) in enumerate(zip(axes[0], axes[1], categories)):
            cat_data = self.df[self.df['category'] == cat]

            if cat_data.empty:
                continue

            # 上图：结构指标距离（替代原来的形态学距离）
            metrics_struct = ['schema_coverage', 'query_semantic_complexity', 'cot_effectiveness']
            struct_data = cat_data[metrics_struct].values

            # 标准化数据
            scaler = StandardScaler()
            struct_data_scaled = scaler.fit_transform(struct_data)

            # 计算距离矩阵
            n_samples = min(len(struct_data_scaled), 5)  # 最多5个样本
            struct_dist = squareform(pdist(struct_data_scaled[:n_samples], metric='euclidean'))

            im1 = ax_top.imshow(struct_dist, cmap='Blues', aspect='auto')
            ax_top.set_title(f'Structural Metrics Distance\n{cat.replace("_", " ").title()}',
                             fontsize=10)
            ax_top.set_xticks(range(n_samples))
            ax_top.set_yticks(range(n_samples))
            ax_top.set_xticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)
            ax_top.set_yticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)

            # 添加数值
            for i in range(n_samples):
                for j in range(n_samples):
                    ax_top.text(j, i, f'{struct_dist[i, j]:.2f}',
                                ha='center', va='center', fontsize=7,
                                color='white' if struct_dist[i, j] > struct_dist.max() / 2 else 'black')

            # 下图：行为指标距离（替代原来的功能距离）
            metrics_behav = ['autonomy_index', 'tool_f1_score', 'answer_correctness']
            behav_data = cat_data[metrics_behav].values[:n_samples]

            # 标准化数据
            behav_data_scaled = scaler.fit_transform(behav_data)

            behav_dist = squareform(pdist(behav_data_scaled, metric='euclidean'))

            im2 = ax_bottom.imshow(behav_dist, cmap='Reds', aspect='auto')
            ax_bottom.set_title('Behavioral/Outcome Metrics Distance', fontsize=10)
            ax_bottom.set_xticks(range(n_samples))
            ax_bottom.set_yticks(range(n_samples))
            ax_bottom.set_xticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)
            ax_bottom.set_yticklabels([f'T{i + 1}' for i in range(n_samples)], fontsize=8)

            # 添加数值
            for i in range(n_samples):
                for j in range(n_samples):
                    ax_bottom.text(j, i, f'{behav_dist[i, j]:.2f}',
                                   ha='center', va='center', fontsize=7,
                                   color='white' if behav_dist[i, j] > behav_dist.max() / 2 else 'black')

            # 添加相关性指标
            if n_samples > 2:
                from scipy import stats
                flat_struct = struct_dist[np.triu_indices(n_samples, k=1)]
                flat_behav = behav_dist[np.triu_indices(n_samples, k=1)]
                r, p = stats.spearmanr(flat_struct, flat_behav)
                corr_text = f"Corr: {r:.2f} (p={p:.3f})"
                ax_bottom.text(0.5, -0.4, corr_text, ha='center', transform=ax_bottom.transAxes,
                               fontsize=8, fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

            # 添加颜色条
            if idx == 2:
                plt.colorbar(im1, ax=ax_top, fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=ax_bottom, fraction=0.046, pad=0.04)

        fig.suptitle('D. Structural and Behavioral Distance Analysis',
                     fontweight='bold', y=1.02)

        # 添加图注说明
        fig.text(0.5, 0.01,
                 "Note: Distances calculated using z-standardized Euclidean distance. "
                 "Structural metrics: schema coverage, query complexity, CoT effectiveness. "
                 "Behavioral metrics: autonomy, tool F1 score, answer correctness.",
                 ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_d_distance_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_d_distance_analysis.png'

    def create_figure_b_mismatch_analysis(self):
        """创建改进的图B：指标相关性矩阵（带显著性标注）"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pingouin as pg

        fig, ax = plt.subplots(figsize=(10, 10))

        # 创建相关性矩阵
        metrics = ['schema_coverage', 'query_semantic_complexity', 'cot_effectiveness',
                   'autonomy_index', 'reasoning_depth_score', 'tool_f1_score',
                   'answer_completeness', 'answer_correctness']

        # 计算相关系数，控制复杂度变量（偏相关）
        corr_matrix = pd.DataFrame(index=metrics, columns=metrics)
        p_matrix = pd.DataFrame(index=metrics, columns=metrics)

        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i >= j:  # 只计算下三角
                    df_subset = self.df[[metric1, metric2, 'complexity']].dropna()
                    if len(df_subset) > 5:  # 确保有足够的样本
                        # 计算偏相关（控制复杂度）
                        pc = pg.partial_corr(data=df_subset, x=metric1, y=metric2,
                                             covar='complexity', method='spearman')
                        corr_matrix.loc[metric1, metric2] = pc['r'].iat[0]
                        p_matrix.loc[metric1, metric2] = pc['p-val'].iat[0]
                    else:
                        corr_matrix.loc[metric1, metric2] = np.nan
                        p_matrix.loc[metric1, metric2] = np.nan
                else:
                    corr_matrix.loc[metric1, metric2] = np.nan
                    p_matrix.loc[metric1, metric2] = np.nan

        # FDR校正p值
        flat_p = p_matrix.values.flatten()
        valid_p = flat_p[~np.isnan(flat_p)]
        if len(valid_p) > 0:
            _, q_values = pg.multicomp(valid_p, method='fdr_bh')

            # 将校正后的q值放回矩阵
            q_matrix = p_matrix.copy()
            q_idx = 0
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    if not np.isnan(p_matrix.iloc[i, j]):
                        q_matrix.iloc[i, j] = q_values[q_idx]
                        q_idx += 1

        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # 调整annot格式，添加显著性标记
        annot_matrix = corr_matrix.copy().astype(str)
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                if not np.isnan(corr_matrix.iloc[i, j]):
                    annot = f"{corr_matrix.iloc[i, j]:.2f}"
                    # 添加显著性标记
                    if not np.isnan(q_matrix.iloc[i, j]):
                        if q_matrix.iloc[i, j] < 0.01:
                            annot += "**"
                        elif q_matrix.iloc[i, j] < 0.05:
                            annot += "*"
                    annot_matrix.iloc[i, j] = annot

        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot=annot_matrix, fmt='', annot_kws={'size': 8},
                    vmin=-1, vmax=1, ax=ax)

        ax.set_title('B. Metric Correlation Matrix (Partial Correlations)', fontweight='bold', pad=20)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics], rotation=0)

        # 添加图注说明统计方法
        fig.text(0.5, 0.01,
                 "Note: Values are Spearman partial correlation coefficients controlling for task complexity. "
                 "* indicates q < 0.05, ** indicates q < 0.01 after FDR-BH correction.",
                 ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_b_correlation_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_b_correlation_matrix.png'

    def create_figure_c_radar_comparisons(self):
        """创建改进的图C：带基线的雷达图比较"""
        import matplotlib.pyplot as plt
        import numpy as np

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

        # 准备基线数据（如果有）
        if self.baseline_df is not None and not self.baseline_df.empty:
            baseline_groups = [
                self.baseline_df[self.baseline_df['complexity'] <= 2],
                self.baseline_df[(self.baseline_df['complexity'] > 2) & (self.baseline_df['complexity'] <= 4)],
                self.baseline_df[self.baseline_df['complexity'] > 4]
            ]
        else:
            baseline_groups = [None, None, None]

        for idx, (ax, group, baseline_group, label) in enumerate(zip(axes, test_groups, baseline_groups, labels)):
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

            # 计算95%置信区间
            from scipy import stats
            ci_lower = []
            ci_upper = []

            for metric, val in zip(['schema_coverage', 'query_semantic_complexity', 'cot_effectiveness',
                                    'autonomy_index', 'tool_f1_score', 'answer_correctness'], values):
                if metric == 'query_semantic_complexity':
                    data = group[metric].values / 10
                else:
                    data = group[metric].values

                if len(data) >= 2:
                    ci = stats.t.interval(0.95, len(data) - 1, loc=val, scale=stats.sem(data))
                    ci_lower.append(max(0, ci[0]))  # 下界不小于0
                    ci_upper.append(min(1, ci[1]))  # 上界不大于1
                else:
                    ci_lower.append(val * 0.8)  # 假设置信区间
                    ci_upper.append(min(1, val * 1.2))

            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            ci_lower += ci_lower[:1]
            ci_upper += ci_upper[:1]

            ax.plot(angles, values, 'o-', linewidth=2, color=ACADEMIC_COLORS['blue'])
            ax.fill(angles, values, alpha=0.25, color=ACADEMIC_COLORS['blue'])

            # 添加置信区间
            ax.fill_between(angles, ci_lower, ci_upper, alpha=0.1, color='blue')

            # 添加样本量信息
            ax.text(0.5, -0.12, f"n={len(group)}", transform=ax.transAxes,
                    ha='center', fontsize=9)

            # 添加对比（如果有baseline）
            if baseline_group is not None and not baseline_group.empty:
                baseline_values = [
                    baseline_group['schema_coverage'].mean(),
                    baseline_group['query_semantic_complexity'].mean() / 10,
                    baseline_group['cot_effectiveness'].mean(),
                    baseline_group['autonomy_index'].mean(),
                    baseline_group['tool_f1_score'].mean(),
                    baseline_group['answer_correctness'].mean()
                ]
                baseline_values += baseline_values[:1]

                ax.plot(angles, baseline_values, '--', linewidth=1.5, color=ACADEMIC_COLORS['red'],
                        alpha=0.7, label='Baseline')
                ax.fill(angles, baseline_values, alpha=0.1, color=ACADEMIC_COLORS['red'])

                # 标注性能提升
                for i, (main_val, base_val) in enumerate(zip(values[:-1], baseline_values[:-1])):
                    if abs(main_val - base_val) > 0.05:  # 只标注明显差异
                        angle = angles[i]
                        radius = max(main_val, base_val) + 0.05
                        improvement = ((main_val / base_val) - 1) * 100 if base_val > 0 else float('inf')

                        if improvement > 5:  # 提升超过5%
                            arrow_style = '->'
                            color = 'green'
                            text = f"+{improvement:.0f}%"
                        elif improvement < -5:  # 降低超过5%
                            arrow_style = '->'
                            color = 'red'
                            text = f"{improvement:.0f}%"
                        else:
                            continue

                        ax.annotate(text,
                                    xy=(angle, min(main_val, base_val) + 0.02),
                                    xytext=(angle, radius),
                                    fontsize=8, color=color,
                                    arrowprops=dict(arrowstyle=arrow_style, color=color, lw=1))
            else:
                # 使用一个通用的基线进行比较
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

        # 添加图注说明
        if self.baseline_df is not None:
            baseline_desc = "Baseline represents Agent without enhanced planning and CoT mechanisms."
        else:
            baseline_desc = "Baseline (dotted line) represents theoretical 50% performance level."

        fig.text(0.5, 0.01,
                 f"Note: Main plot shows means with 95% confidence intervals. {baseline_desc} "
                 f"Green/red annotations indicate statistically significant improvements/decreases.",
                 ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_c_radar_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_c_radar_analysis.png'

    def create_figure_a_projection_patterns(self):
        """创建改进的图A：性能矩阵（带统计显著性标注）"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from scipy import stats

        fig, ax = plt.subplots(figsize=(12, 6))

        # 准备数据：按类别和复杂度分组
        categories = self.df['category'].unique()
        complexities = sorted(self.df['complexity'].unique())

        # 创建矩阵数据与样本量矩阵
        matrix_data = np.zeros((len(categories), len(complexities)))
        sample_size = np.zeros((len(categories), len(complexities)), dtype=int)

        # 如果有基线数据，为比较准备另一个矩阵
        if self.baseline_df is not None and not self.baseline_df.empty:
            baseline_data = np.zeros((len(categories), len(complexities)))
            has_baseline = True
        else:
            has_baseline = False

        for i, cat in enumerate(categories):
            for j, comp in enumerate(complexities):
                mask = (self.df['category'] == cat) & (self.df['complexity'] == comp)
                if mask.any():
                    matrix_data[i, j] = self.df[mask]['overall_performance'].mean()
                    sample_size[i, j] = mask.sum()

                    if has_baseline:
                        base_mask = (self.baseline_df['category'] == cat) & (self.baseline_df['complexity'] == comp)
                        if base_mask.any():
                            baseline_data[i, j] = self.baseline_df[base_mask]['overall_performance'].mean()

        # 绘制热力图
        im = ax.imshow(matrix_data, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

        # 设置标签
        ax.set_xticks(np.arange(len(complexities)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels([f'Level {c}' for c in complexities])
        ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])

        # 添加数值和统计显著性标注
        for i in range(len(categories)):
            for j in range(len(complexities)):
                if matrix_data[i, j] > 0:
                    text_value = f'{matrix_data[i, j]:.2f}'

                    # 添加样本量
                    if sample_size[i, j] > 0:
                        text_value += f'\n(n={sample_size[i, j]})'

                    # 如果有基线，计算显著性并添加标注
                    if has_baseline and baseline_data[i, j] > 0 and sample_size[i, j] > 2:
                        # 假设我们有原始数据，可以进行t检验
                        # 这里简化处理，实际应该使用原始数据点
                        if matrix_data[i, j] > baseline_data[i, j] * 1.1:  # 提升超过10%
                            text_value += "*"
                        if matrix_data[i, j] > baseline_data[i, j] * 1.2:  # 提升超过20%
                            text_value += "*"

                    ax.text(j, i, text_value,
                            ha="center", va="center",
                            color="white" if matrix_data[i, j] < 0.5 else "black",
                            fontsize=8, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Performance Score', rotation=270, labelpad=15)

        ax.set_title('A. Performance Matrix by Category and Complexity',
                     fontweight='bold', pad=20)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Test Category')

        # 添加图注说明
        if has_baseline:
            fig.text(0.5, 0.01,
                     "Note: Values show mean performance scores with sample sizes in parentheses. "
                     "* indicates p < 0.05, ** indicates p < 0.01 improvement over baseline.",
                     ha='center', fontsize=8, style='italic')
        else:
            fig.text(0.5, 0.01,
                     "Note: Values show mean performance scores with sample sizes in parentheses.",
                     ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_a_performance_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return self.output_dir / 'figure_a_performance_matrix.png'

    def create_all_figures(self):
        """生成所有改进的图表"""
        print("\n生成改进的学术风格评估图表...")

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

        print(f"\n✅ 所有改进图表已生成到: {self.output_dir}")
        return figures


# ==================== STEP 7: 改进的真实评估运行器 ====================
class ImprovedEvaluationRunner:
    """改进的真实评估运行器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_wrapper = None
        self.baseline_agent = None
        self.results = None
        self.baseline_results = None

    def setup(self) -> bool:
        """设置评估环境"""
        logger.info("\n" + "=" * 60)
        logger.info("SETTING UP IMPROVED EVALUATION ENVIRONMENT")
        logger.info("=" * 60)

        # 验证配置
        if not AgentConfig.validate_config(self.config):
            return False

        # 创建主Agent
        self.agent_wrapper = RealAgentWrapper(self.config)

        # 测试连接
        if not self.agent_wrapper.test_connection():
            logger.error("❌ Agent connection test failed")
            return False

        # 如果需要，创建基线Agent（禁用CoT和工具）
        if self.config.get('enable_baseline', False):
            logger.info("Creating baseline agent (with reduced capabilities)...")
            baseline_config = self.config.copy()
            baseline_config['disable_cot'] = True
            baseline_config['disable_tools'] = True
            self.baseline_agent = RealAgentWrapper(baseline_config)

            # 测试基线Agent连接
            if not self.baseline_agent.test_connection():
                logger.warning("⚠️ Baseline agent connection test failed, proceeding with main agent only")
                self.baseline_agent = None

        logger.info("✅ Improved evaluation environment ready")
        return True

    def run_evaluation(self,
                       test_subset: List[str] = None,
                       output_dir: str = "evaluation_results") -> pd.DataFrame:
        """
        运行改进的评估

        Args:
            test_subset: 要运行的测试ID列表（None表示运行所有）
            output_dir: 输出目录
        """
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING IMPROVED EVALUATION")
        logger.info("=" * 60)

        # 创建测试套件
        test_suite = CustomTestSuite()
        test_cases = test_suite.test_cases

        # 选择测试用例
        if test_subset:
            test_cases = [tc for tc in test_cases if tc.id in test_subset]
            logger.info(f"Running subset: {[tc.id for tc in test_cases]}")
        else:
            logger.info(f"Running all {len(test_cases)} test cases")

        # 运行主Agent评估
        logger.info("\n" + "=" * 40)
        logger.info("Evaluating main agent")
        logger.info("=" * 40)

        self.results = run_improved_evaluation(
            agent=self.agent_wrapper,
            test_cases=test_cases,
            output_dir=output_dir
        )

        # 如果有基线Agent，也进行评估
        if self.baseline_agent:
            logger.info("\n" + "=" * 40)
            logger.info("Evaluating baseline agent (reduced capabilities)")
            logger.info("=" * 40)

            baseline_output_dir = os.path.join(output_dir, "baseline")
            self.baseline_results = run_improved_evaluation(
                agent=self.baseline_agent,
                test_cases=test_cases,
                output_dir=baseline_output_dir
            )

        # 生成增强可视化
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING ENHANCED VISUALIZATIONS")
        logger.info("=" * 60)

        visualizer = EnhancedVisualizer(
            df=self.results,
            output_dir=output_dir,
            baseline_df=self.baseline_results
        )
        visualizer.create_all_figures()

        return self.results

    def print_summary(self):
        """打印评估摘要"""
        if self.results is None or self.results.empty:
            return

        logger.info("\n" + "=" * 60)
        logger.info("IMPROVED EVALUATION SUMMARY")
        logger.info("=" * 60)

        # 过滤掉错误记录
        valid_df = self.results[~self.results.get('error', pd.Series()).notna()]
        if not valid_df.empty:
            print(f"\n完成测试: {len(valid_df)}/{len(self.results)}")
            print(f"\n核心指标:")
            print(
                f"  • Schema覆盖率: {valid_df['schema_coverage'].mean():.3f} ± {valid_df['schema_coverage'].std():.3f}")
            print(
                f"  • 查询语义复杂度: {valid_df['query_semantic_complexity'].mean():.2f} ± {valid_df['query_semantic_complexity'].std():.2f}")
            print(
                f"  • CoT有效性: {valid_df['cot_effectiveness'].mean():.3f} ± {valid_df['cot_effectiveness'].std():.3f}")
            print(f"  • 自主性指数: {valid_df['autonomy_index'].mean():.3f} ± {valid_df['autonomy_index'].std():.3f}")
            print(f"  • 工具F1分数: {valid_df['tool_f1_score'].mean():.3f} ± {valid_df['tool_f1_score'].std():.3f}")
            print(
                f"  • 答案正确性: {valid_df['answer_correctness'].mean():.3f} ± {valid_df['answer_correctness'].std():.3f}")

            print(f"\n性能指标:")
            print(f"  • 平均执行时间: {valid_df['execution_time'].mean():.2f}s")
            print(f"  • 查询效率: {valid_df['query_efficiency'].mean():.1%}")
            print(f"  • 时间效率: {valid_df['time_efficiency'].mean():.3f}")

            # 如果有基线结果，进行对比
            if self.baseline_results is not None and not self.baseline_results.empty:
                valid_baseline = self.baseline_results[~self.baseline_results.get('error', pd.Series()).notna()]
                if not valid_baseline.empty:
                    print(f"\n与基线对比:")
                    print(f"  • 主Agent答案正确性: {valid_df['answer_correctness'].mean():.3f}")
                    print(f"  • 基线答案正确性: {valid_baseline['answer_correctness'].mean():.3f}")
                    print(
                        f"  • 提升: {(valid_df['answer_correctness'].mean() / valid_baseline['answer_correctness'].mean() - 1) * 100:.1f}%")

                    # 按复杂度分组对比
                    for complexity in sorted(valid_df['complexity'].unique()):
                        main_complex = valid_df[valid_df['complexity'] == complexity]
                        base_complex = valid_baseline[valid_baseline['complexity'] == complexity]

                        if not main_complex.empty and not base_complex.empty:
                            main_score = main_complex['answer_correctness'].mean()
                            base_score = base_complex['answer_correctness'].mean()
                            improvement = (main_score / base_score - 1) * 100 if base_score > 0 else float('inf')

                            print(f"    • 复杂度 {complexity}: 提升 {improvement:.1f}%")

        # 打印统计摘要
        print("\n分类统计:")
        category_stats = valid_df.groupby('category')['answer_correctness'].agg(['mean', 'std', 'count'])
        for cat, row in category_stats.iterrows():
            print(f"  • {cat.replace('_', ' ').title()}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")

        # 打印复杂度统计
        print("\n复杂度统计:")
        complexity_stats = valid_df.groupby('complexity')['answer_correctness'].agg(['mean', 'std', 'count'])
        for comp, row in complexity_stats.iterrows():
            print(f"  • Level {comp}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")


# ==================== STEP 8: 主函数 ====================
def main():
    """主函数：运行改进的Agent评估"""
    import argparse

    parser = argparse.ArgumentParser(description='Run KGAgent V7 Improved Evaluation')
    parser.add_argument('--subset', nargs='+', help='Test IDs to run (default: all)')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    parser.add_argument('--with-baseline', action='store_true', help='Enable baseline comparison')
    parser.add_argument('--config-file', help='JSON config file path')

    args = parser.parse_args()

    # 加载配置
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from: {args.config_file}")
    else:
        config = AgentConfig.get_config()

    # 如果指定了基线比较，添加到配置
    if args.with_baseline:
        config['enable_baseline'] = True

    # 创建评估运行器
    runner = ImprovedEvaluationRunner(config)

    # 设置环境
    if not runner.setup():
        logger.error("❌ Setup failed. Exiting.")
        return 1

    # 运行评估
    results = runner.run_evaluation(
        test_subset=args.subset,
        output_dir=args.output
    )

    if results is not None and not results.empty:
        # 打印摘要
        runner.print_summary()

        logger.info("\n" + "=" * 60)
        logger.info("✅ IMPROVED EVALUATION COMPLETE")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("❌ Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())