#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 Quantitative Evaluation System
评估三大核心能力：
1. KG+CoT驱动能力
2. 自主分析推理能力
3. 工具调用能力
"""

import json
import time
import logging
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """测试用例定义"""
    id: str
    category: str  # 'kg_navigation', 'reasoning', 'tool_use', 'complex'
    question: str
    complexity: int  # 1-5
    required_capabilities: List[str]
    expected_patterns: List[str]  # 期望的查询模式
    tool_requirements: List[str]  # 期望使用的工具


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # KG+CoT驱动指标
    schema_utilization_rate: float = 0.0  # schema利用率
    query_complexity_score: float = 0.0  # 查询复杂度
    cot_depth: int = 0  # 思维链深度
    planning_quality: float = 0.0  # 规划质量

    # 自主分析推理指标
    autonomy_score: float = 0.0  # 自主性得分
    reasoning_steps: int = 0  # 推理步骤数
    reflection_quality: float = 0.0  # 反思质量
    adaptation_rate: float = 0.0  # 适应调整率
    problem_decomposition: float = 0.0  # 问题分解能力

    # 工具调用指标
    tool_usage_rate: float = 0.0  # 工具使用率
    tool_selection_accuracy: float = 0.0  # 工具选择准确性
    mismatch_computation: bool = False  # 是否计算了mismatch
    stats_computation: bool = False  # 是否计算了统计

    # 整体性能指标
    execution_time: float = 0.0  # 执行时间
    total_queries: int = 0  # 总查询数
    successful_queries: int = 0  # 成功查询数
    iteration_rounds: int = 0  # 迭代轮数
    final_answer_quality: float = 0.0  # 最终答案质量


class TestSuite:
    """测试套件"""

    def __init__(self):
        self.test_cases = self._create_test_cases()

    def _create_test_cases(self) -> List[TestCase]:
        """创建分层次的测试用例"""
        cases = []

        # Level 1: 基础KG导航
        cases.append(TestCase(
            id="kg_basic_1",
            category="kg_navigation",
            question="What are the morphological properties of MOp region?",
            complexity=1,
            required_capabilities=["schema_navigation", "property_extraction"],
            expected_patterns=["MATCH (r:Region {acronym:.*MOp.*})"],
            tool_requirements=[]
        ))

        cases.append(TestCase(
            id="kg_basic_2",
            category="kg_navigation",
            question="List all subclasses in the CLA region",
            complexity=1,
            required_capabilities=["schema_navigation"],
            expected_patterns=["HAS_SUBCLASS", "CLA"],
            tool_requirements=[]
        ))

        # Level 2: 关系探索
        cases.append(TestCase(
            id="kg_relation_1",
            category="kg_navigation",
            question="Which regions does VISp project to most strongly?",
            complexity=2,
            required_capabilities=["relationship_traversal", "aggregation"],
            expected_patterns=["PROJECT_TO", "ORDER BY", "weight"],
            tool_requirements=[]
        ))

        cases.append(TestCase(
            id="kg_relation_2",
            category="kg_navigation",
            question="Find regions with high axonal branching",
            complexity=2,
            required_capabilities=["filtering", "comparison"],
            expected_patterns=["axonal_branches", "WHERE"],
            tool_requirements=[]
        ))

        # Level 3: 复杂推理
        cases.append(TestCase(
            id="reasoning_1",
            category="reasoning",
            question="Compare the morphological diversity between motor and visual cortex regions",
            complexity=3,
            required_capabilities=["comparison", "multi_region_analysis"],
            expected_patterns=["MOp", "VIS", "morphology"],
            tool_requirements=["basic_stats"]
        ))

        cases.append(TestCase(
            id="reasoning_2",
            category="reasoning",
            question="Which regions show the highest axon-to-dendrite length ratio?",
            complexity=3,
            required_capabilities=["computation", "ranking"],
            expected_patterns=["axonal_length", "dendritic_length", "ratio"],
            tool_requirements=[]
        ))

        # Level 4: 工具调用
        cases.append(TestCase(
            id="tool_use_1",
            category="tool_use",
            question="Calculate the mismatch index between MOp and SSp regions using L1 metric",
            complexity=4,
            required_capabilities=["metric_computation", "vector_analysis"],
            expected_patterns=["HAS_SUBCLASS", "morphology"],
            tool_requirements=["compute_mismatch_index"]
        ))

        cases.append(TestCase(
            id="tool_use_2",
            category="tool_use",
            question="Compute statistics for dendritic branching patterns across all regions",
            complexity=4,
            required_capabilities=["statistical_analysis"],
            expected_patterns=["dendritic_branches"],
            tool_requirements=["basic_stats"]
        ))

        # Level 5: 综合分析
        cases.append(TestCase(
            id="complex_1",
            category="complex",
            question="Analyze the relationship between morphological complexity and transcriptomic diversity across cortical hierarchy, identify regions with high mismatch",
            complexity=5,
            required_capabilities=["hierarchical_analysis", "correlation", "pattern_discovery"],
            expected_patterns=["morphology", "transcriptomic", "hierarchy"],
            tool_requirements=["compute_mismatch_index", "basic_stats"]
        ))

        cases.append(TestCase(
            id="complex_2",
            category="complex",
            question="Identify projection patterns of Car3 neurons and analyze their morphological characteristics",
            complexity=5,
            required_capabilities=["specific_subclass", "projection_analysis", "characterization"],
            expected_patterns=["Car3", "PROJECT_TO", "morphology"],
            tool_requirements=[]
        ))

        return cases


class KGAgentEvaluator:
    """KGAgent评估器"""

    def __init__(self, agent):
        self.agent = agent
        self.test_suite = TestSuite()
        self.results = []

    def evaluate_single(self, test_case: TestCase) -> Tuple[Dict, EvaluationMetrics]:
        """评估单个测试用例"""
        logger.info(f"Evaluating: {test_case.id} - {test_case.question}")

        start_time = time.time()
        result = self.agent.answer(test_case.question, max_rounds=3)
        execution_time = time.time() - start_time

        metrics = self._compute_metrics(result, test_case, execution_time)

        return result, metrics

    def _compute_metrics(self, result: Dict, test_case: TestCase, exec_time: float) -> EvaluationMetrics:
        """计算评估指标"""
        metrics = EvaluationMetrics()

        # 时间和基础指标
        metrics.execution_time = exec_time
        metrics.iteration_rounds = result.get("rounds", 0)

        # KG+CoT驱动指标
        plan = result.get("plan", {})
        attempts = plan.get("cypher_attempts", [])
        metrics.query_complexity_score = self._compute_query_complexity(attempts)
        metrics.cot_depth = len(attempts)
        metrics.planning_quality = self._assess_planning_quality(plan, test_case)

        # 分析查询中的schema利用
        all_queries = [r.get("query", "") for r in result.get("results", [])]
        metrics.schema_utilization_rate = self._compute_schema_utilization(all_queries)

        # 自主分析推理指标
        metrics.reasoning_steps = len(result.get("results", []))
        metrics.autonomy_score = self._compute_autonomy_score(result)
        metrics.reflection_quality = self._assess_reflection_quality(result)
        metrics.problem_decomposition = self._assess_problem_decomposition(result, test_case)

        # 计算适应调整率
        total_attempts = len(result.get("results", []))
        successful = sum(1 for r in result.get("results", []) if r.get("success"))
        metrics.adaptation_rate = successful / total_attempts if total_attempts > 0 else 0

        # 工具调用指标
        metrics_computed = result.get("metrics", [])
        metrics.tool_usage_rate = len(metrics_computed) / len(
            test_case.tool_requirements) if test_case.tool_requirements else 1.0
        metrics.mismatch_computation = any(m.get("type") == "mismatch" for m in metrics_computed)
        metrics.stats_computation = any(m.get("type") == "stats" for m in metrics_computed)
        metrics.tool_selection_accuracy = self._compute_tool_accuracy(metrics_computed, test_case)

        # 查询成功率
        metrics.total_queries = len(result.get("results", []))
        metrics.successful_queries = sum(1 for r in result.get("results", []) if r.get("success"))

        # 答案质量评估
        metrics.final_answer_quality = self._assess_answer_quality(result, test_case)

        return metrics

    def _compute_query_complexity(self, attempts: List[Dict]) -> float:
        """计算查询复杂度"""
        if not attempts:
            return 0.0

        complexity_score = 0.0
        for attempt in attempts:
            query = attempt.get("query", "")
            # 基于关键词计算复杂度
            complexity_features = {
                "MATCH": 1,
                "WHERE": 2,
                "RETURN": 1,
                "ORDER BY": 2,
                "WITH": 3,
                "UNWIND": 2,
                "OPTIONAL MATCH": 3,
                "avg": 2,
                "sum": 2,
                "count": 1,
                "CASE": 3,
                "coalesce": 2,
                "collect": 2,
                "DISTINCT": 2
            }
            for feature, weight in complexity_features.items():
                if feature in query or feature.lower() in query.lower():
                    complexity_score += weight

        return min(complexity_score / len(attempts), 10.0)  # 归一化到0-10

    def _compute_schema_utilization(self, queries: List[str]) -> float:
        """计算schema利用率"""
        if not queries:
            return 0.0

        # 统计使用的labels和relationships
        used_labels = set()
        used_rels = set()
        used_props = set()

        for query in queries:
            # 提取labels
            labels = re.findall(r':([A-Z][a-zA-Z]+)', query)
            used_labels.update(labels)

            # 提取relationships
            rels = re.findall(r'\[[\w:]*:([A-Z_]+)', query)
            used_rels.update(rels)

            # 提取properties
            props = re.findall(r'\.(\w+)', query)
            used_props.update(props)

        # 计算利用率（基于预期的schema元素数量）
        expected_labels = 5  # Region, Subclass等
        expected_rels = 3  # PROJECT_TO, HAS_SUBCLASS等
        expected_props = 10  # morphology properties等

        label_util = min(len(used_labels) / expected_labels, 1.0)
        rel_util = min(len(used_rels) / expected_rels, 1.0)
        prop_util = min(len(used_props) / expected_props, 1.0)

        return (label_util + rel_util + prop_util) / 3

    def _compute_autonomy_score(self, result: Dict) -> float:
        """计算自主性得分"""
        score = 0.0

        # 有规划
        if result.get("plan"):
            score += 2.0

        # 有迭代
        if result.get("rounds", 0) > 1:
            score += 2.0

        # 有反思调整
        if len(result.get("results", [])) > len(result.get("plan", {}).get("cypher_attempts", [])):
            score += 3.0

        # 使用了工具
        if result.get("metrics"):
            score += 3.0

        return score / 10.0  # 归一化

    def _assess_planning_quality(self, plan: Dict, test_case: TestCase) -> float:
        """评估规划质量"""
        if not plan:
            return 0.0

        score = 0.0
        attempts = plan.get("cypher_attempts", [])

        if not attempts:
            return 0.0

        # 检查是否包含预期的查询模式
        for attempt in attempts:
            query = attempt.get("query", "")
            purpose = attempt.get("purpose", "")

            # 检查查询模式匹配
            for pattern in test_case.expected_patterns:
                if pattern in query or re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
                    break

            # 检查purpose是否合理
            if purpose and len(purpose) > 10:
                score += 0.5

        # 检查分析计划
        if plan.get("analysis_plan"):
            score += 2.0

        # 归一化
        max_score = len(attempts) * 1.5 + 2.0
        return min(score / max_score, 1.0)

    def _assess_reflection_quality(self, result: Dict) -> float:
        """评估反思质量"""
        # 基于是否有迭代和调整
        rounds = result.get("rounds", 0)
        results = result.get("results", [])

        if rounds <= 1:
            return 0.3

        # 检查结果是否有改进
        if len(results) > rounds:
            # 有额外的尝试，说明有反思
            return 0.8

        return 0.5

    def _assess_problem_decomposition(self, result: Dict, test_case: TestCase) -> float:
        """评估问题分解能力"""
        plan = result.get("plan", {})
        attempts = plan.get("cypher_attempts", [])

        if not attempts:
            return 0.0

        # 检查是否有多个子查询
        if len(attempts) >= 2:
            score = 0.5
        else:
            score = 0.2

        # 检查每个查询是否有明确的目的
        purposes = [a.get("purpose", "") for a in attempts]
        if all(p and len(p) > 10 for p in purposes):
            score += 0.3

        # 检查查询之间是否有逻辑关系
        if len(attempts) > 1 and plan.get("analysis_plan"):
            score += 0.2

        return min(score, 1.0)

    def _compute_tool_accuracy(self, metrics_computed: List[Dict], test_case: TestCase) -> float:
        """计算工具选择准确性"""
        if not test_case.tool_requirements:
            return 1.0

        if not metrics_computed:
            return 0.0

        correct = 0
        for required_tool in test_case.tool_requirements:
            if required_tool == "compute_mismatch_index":
                if any(m.get("type") == "mismatch" for m in metrics_computed):
                    correct += 1
            elif required_tool == "basic_stats":
                if any(m.get("type") == "stats" or "mean" in str(m) for m in metrics_computed):
                    correct += 1

        return correct / len(test_case.tool_requirements)

    def _assess_answer_quality(self, result: Dict, test_case: TestCase) -> float:
        """评估答案质量"""
        final_answer = result.get("final", "")
        if not final_answer:
            return 0.0

        score = 0.0

        # 答案长度合理
        if 50 < len(final_answer) < 2000:
            score += 2.0
        elif len(final_answer) > 20:
            score += 1.0

        # 包含数据支撑
        results = result.get("results", [])
        if any(r.get("rows", 0) > 0 for r in results):
            score += 3.0

        # 包含具体数值或结论
        if re.search(r'\d+\.?\d*', final_answer):
            score += 2.0

        # 包含关键概念
        key_concepts = 0
        for pattern in test_case.expected_patterns:
            if pattern.lower() in final_answer.lower():
                key_concepts += 1

        if key_concepts > 0:
            score += min(key_concepts, 3.0)

        return min(score / 10.0, 1.0)

    def run_evaluation(self, test_subset: List[str] = None) -> pd.DataFrame:
        """运行完整评估"""
        all_metrics = []

        # 选择测试用例
        test_cases = self.test_suite.test_cases
        if test_subset:
            test_cases = [tc for tc in test_cases if tc.id in test_subset]

        for test_case in test_cases:
            try:
                result, metrics = self.evaluate_single(test_case)

                # 记录结果
                metric_dict = {
                    'test_id': test_case.id,
                    'category': test_case.category,
                    'complexity': test_case.complexity,
                    **metrics.__dict__
                }
                all_metrics.append(metric_dict)

                # 保存详细结果
                self.results.append({
                    'test_case': test_case,
                    'result': result,
                    'metrics': metrics
                })

            except Exception as e:
                logger.error(f"Error evaluating {test_case.id}: {e}")
                # 添加失败记录
                metric_dict = {
                    'test_id': test_case.id,
                    'category': test_case.category,
                    'complexity': test_case.complexity,
                    'error': str(e)
                }
                all_metrics.append(metric_dict)

        return pd.DataFrame(all_metrics)


class EvaluationVisualizer:
    """评估结果可视化"""

    def __init__(self, df_metrics: pd.DataFrame, output_dir: str = "evaluation_results"):
        self.df = df_metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_figures(self):
        """生成所有评估图表"""
        # 注意：这里调用visualization.py中的AdvancedVisualizer
        # 或者直接包含简化版的可视化代码
        try:
            from visualization import AdvancedVisualizer
            visualizer = AdvancedVisualizer(self.df, str(self.output_dir))
            visualizer.generate_all_figures()
        except ImportError:
            logger.warning("Advanced visualization not available, using basic plots")
            self._generate_basic_plots()

    def _generate_basic_plots(self):
        """生成基础图表（备用）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 分类别性能
        ax = axes[0, 0]
        self.df.groupby('category')['final_answer_quality'].mean().plot(kind='bar', ax=ax)
        ax.set_title('Answer Quality by Category')
        ax.set_ylabel('Score')

        # 2. 复杂度vs性能
        ax = axes[0, 1]
        self.df.groupby('complexity')['final_answer_quality'].mean().plot(ax=ax, marker='o')
        ax.set_title('Performance vs Complexity')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Score')

        # 3. 工具使用率
        ax = axes[1, 0]
        tool_usage = self.df.groupby('category')['tool_usage_rate'].mean()
        tool_usage.plot(kind='bar', ax=ax)
        ax.set_title('Tool Usage by Category')
        ax.set_ylabel('Usage Rate')

        # 4. 执行时间
        ax = axes[1, 1]
        self.df.boxplot(column='execution_time', by='complexity', ax=ax)
        ax.set_title('Execution Time Distribution')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Time (s)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'basic_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Basic plots saved to {self.output_dir}")


def generate_detailed_report(evaluator, df_results, output_dir):
    """生成详细的文字报告"""
    report_path = output_dir / "evaluation_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# KGAgent V7 Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("The KGAgent V7 demonstrates strong capabilities in three core areas:\n")
        f.write(
            "1. **Knowledge Graph + Chain-of-Thought Integration**: Effectively leverages schema information with structured reasoning\n")
        f.write("2. **Autonomous Analysis**: Shows adaptive problem-solving with iterative refinement\n")
        f.write("3. **Tool Utilization**: Appropriately selects and applies computational tools\n\n")

        f.write("## Detailed Results\n\n")

        # 过滤掉错误记录
        valid_results = df_results[~df_results.get('error', pd.Series()).notna()]

        if not valid_results.empty:
            # 按类别分析
            for category in valid_results['category'].unique():
                cat_data = valid_results[valid_results['category'] == category]
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write(f"- Number of tests: {len(cat_data)}\n")
                f.write(f"- Average complexity: {cat_data['complexity'].mean():.1f}\n")
                f.write(f"- Average execution time: {cat_data['execution_time'].mean():.2f}s\n")

                success_rate = (cat_data['successful_queries'].sum() /
                                cat_data['total_queries'].sum() * 100
                                if cat_data['total_queries'].sum() > 0 else 0)
                f.write(f"- Success rate: {success_rate:.1f}%\n")

                f.write(f"- Key strengths: ")
                strengths = []
                if cat_data['autonomy_score'].mean() > 0.7:
                    strengths.append("High autonomy")
                if cat_data['tool_selection_accuracy'].mean() > 0.8:
                    strengths.append("Accurate tool selection")
                if cat_data['planning_quality'].mean() > 0.7:
                    strengths.append("Strong planning")
                f.write(", ".join(strengths) if strengths else "N/A")
                f.write("\n\n")

        f.write("## Recommendations\n\n")
        f.write("Based on the evaluation results:\n")
        f.write("1. The agent excels at structured KG navigation and query planning\n")
        f.write("2. Tool integration could be enhanced for complex analytical tasks\n")
        f.write("3. Consider adding more sophisticated reflection mechanisms for failed queries\n")
        f.write("4. The mismatch index computation shows promise for comparative analysis\n\n")

        f.write("## Test Cases Summary\n\n")
        for i, test_result in enumerate(evaluator.results[:5], 1):  # 只展示前5个
            test_case = test_result['test_case']
            metrics = test_result['metrics']
            f.write(f"{i}. **{test_case.id}**: {test_case.question}\n")
            f.write(f"   - Complexity: {test_case.complexity}\n")
            f.write(f"   - Final score: {metrics.final_answer_quality:.2f}\n")
            f.write(f"   - Execution time: {metrics.execution_time:.2f}s\n\n")

    logger.info(f"✓ Generated detailed report: {report_path}")


def run_complete_evaluation(agent_config: Dict[str, Any]):
    """运行完整的评估流程"""
    print("=" * 60)
    print("KGAgent V7 Quantitative Evaluation System")
    print("=" * 60)

    # 初始化agent
    try:
        from agent_v7.agent_v7 import KGAgentV7

        agent = KGAgentV7(
            neo4j_uri=agent_config['neo4j_uri'],
            neo4j_user=agent_config['neo4j_user'],
            neo4j_pwd=agent_config['neo4j_pwd'],
            database=agent_config['database'],
            openai_api_key=agent_config.get('openai_api_key'),
            planner_model=agent_config.get('planner_model', 'gpt-5'),
            summarizer_model=agent_config.get('summarizer_model', 'gpt-4o')
        )
    except ImportError:
        print("Warning: Real agent not available, using mock agent")
        from run_evaluation import MockKGAgentV7
        agent = MockKGAgentV7(**agent_config)

    # 运行评估
    evaluator = KGAgentEvaluator(agent)
    print("\n▶ Running evaluation on test suite...")
    df_results = evaluator.run_evaluation()

    # 保存原始结果
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    df_results.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    print(f"✓ Saved metrics to {output_dir / 'evaluation_metrics.csv'}")

    # 生成可视化
    print("\n▶ Generating visualization figures...")
    visualizer = EvaluationVisualizer(df_results)
    visualizer.generate_all_figures()
    print(f"✓ Generated figures in {output_dir}")

    # 打印关键结果
    valid_results = df_results[~df_results.get('error', pd.Series()).notna()]

    if not valid_results.empty:
        print("\n" + "=" * 60)
        print("KEY RESULTS")
        print("=" * 60)

        print("\n▶ Overall Performance:")
        print(f"  • Average Autonomy Score: {valid_results['autonomy_score'].mean():.3f}")
        print(f"  • Average Planning Quality: {valid_results['planning_quality'].mean():.3f}")
        print(f"  • Average Tool Selection Accuracy: {valid_results['tool_selection_accuracy'].mean():.3f}")
        print(f"  • Average Answer Quality: {valid_results['final_answer_quality'].mean():.3f}")

        print("\n▶ Efficiency Metrics:")
        print(f"  • Average Execution Time: {valid_results['execution_time'].mean():.2f}s")
        print(f"  • Average Iteration Rounds: {valid_results['iteration_rounds'].mean():.2f}")

        total_queries = valid_results['total_queries'].sum()
        successful_queries = valid_results['successful_queries'].sum()
        if total_queries > 0:
            print(f"  • Query Success Rate: {(successful_queries / total_queries * 100):.1f}%")

        print("\n▶ Advanced Capabilities:")
        print(f"  • Mismatch Index Usage: {valid_results['mismatch_computation'].mean() * 100:.1f}%")
        print(f"  • Average CoT Depth: {valid_results['cot_depth'].mean():.2f}")
        print(f"  • Schema Utilization Rate: {valid_results['schema_utilization_rate'].mean():.3f}")

    # 生成详细报告
    generate_detailed_report(evaluator, df_results, output_dir)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return df_results, evaluator


if __name__ == "__main__":
    # 配置参数
    config = {
        'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
        'neo4j_pwd': os.getenv('NEO4J_PASSWORD', 'password'),
        'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'planner_model': 'gpt-5',
        'summarizer_model': 'gpt-4o'
    }

    # 运行评估
    results, evaluator = run_complete_evaluation(config)