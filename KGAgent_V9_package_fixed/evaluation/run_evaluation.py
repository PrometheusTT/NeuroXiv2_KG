#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 评估运行脚本
包含Mock Agent用于测试评估系统
"""

import json
import time
import random
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import sys
import os

# 添加agent_v7到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MockKGAgentV7:
    """Mock Agent用于测试评估系统"""

    def __init__(self, **kwargs):
        self.config = kwargs

    def answer(self, question: str, max_rounds: int = 2) -> Dict[str, Any]:
        """模拟agent回答，生成合理的响应数据"""

        # 根据问题复杂度调整响应
        complexity = self._estimate_complexity(question)

        # 生成规划
        plan = self._generate_plan(question, complexity)

        # 生成执行结果
        results = []
        rounds = min(1 + int(complexity > 3), max_rounds)

        for round_idx in range(rounds):
            round_results = self._generate_round_results(question, complexity, round_idx)
            results.extend(round_results)

        # 生成metrics（如果需要工具）
        metrics = []
        if "mismatch" in question.lower() or complexity >= 4:
            metrics.append({
                "type": "mismatch",
                "regions": ["MOp", "SSp"],
                "metric": "L1",
                "value": random.uniform(0.1, 0.9)
            })

        if "statistic" in question.lower() or "distribution" in question.lower():
            metrics.append({
                "type": "stats",
                "data": {"mean": random.uniform(10, 100), "std": random.uniform(1, 20)}
            })

        # 生成最终答案
        final_answer = self._generate_final_answer(question, results, metrics)

        return {
            "rounds": rounds,
            "plan": plan,
            "results": results,
            "metrics": metrics,
            "final": final_answer
        }

    def _estimate_complexity(self, question: str) -> int:
        """估计问题复杂度"""
        complexity = 1

        # 基于关键词增加复杂度
        if "compare" in question.lower() or "versus" in question.lower():
            complexity += 1
        if "analyze" in question.lower() or "correlation" in question.lower():
            complexity += 1
        if "mismatch" in question.lower() or "calculate" in question.lower():
            complexity += 1
        if "hierarchy" in question.lower() or "comprehensive" in question.lower():
            complexity += 1

        return min(complexity, 5)

    def _generate_plan(self, question: str, complexity: int) -> Dict[str, Any]:
        """生成查询计划"""
        attempts = []

        # 基础查询
        if "region" in question.lower() or "MOp" in question or "SSp" in question:
            attempts.append({
                "purpose": "Get region morphological properties",
                "query": """MATCH (r:Region)
WHERE r.acronym IN ['MOp', 'SSp', 'VISp', 'CLA']
RETURN r.acronym AS region, r.axonal_length AS axon_len, 
       r.dendritic_length AS dend_len
LIMIT 50"""
            })

        if "subclass" in question.lower():
            attempts.append({
                "purpose": "Get subclass distribution",
                "query": """MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
RETURN r.acronym AS region, s.name AS subclass, 
       coalesce(h.pct_cells, 0) AS percentage
LIMIT 100"""
            })

        if "project" in question.lower():
            attempts.append({
                "purpose": "Get projection patterns",
                "query": """MATCH (a:Region)-[p:PROJECT_TO]->(b:Region)
RETURN a.acronym AS source, b.acronym AS target, 
       coalesce(p.weight, 0) AS weight
ORDER BY weight DESC
LIMIT 50"""
            })

        # 复杂查询（基于复杂度）
        if complexity >= 3:
            attempts.append({
                "purpose": "Complex aggregation analysis",
                "query": """MATCH (r:Region)
WITH r, r.axonal_length / NULLIF(r.dendritic_length, 0) AS ratio
WHERE ratio IS NOT NULL
RETURN r.acronym AS region, ratio
ORDER BY ratio DESC
LIMIT 20"""
            })

        # 如果没有生成查询，添加默认查询
        if not attempts:
            attempts.append({
                "purpose": "General region overview",
                "query": "MATCH (r:Region) RETURN r.acronym, r.name LIMIT 10"
            })

        return {
            "cypher_attempts": attempts,
            "analysis_plan": f"Execute {len(attempts)} queries to analyze {question}"
        }

    def _generate_round_results(self, question: str, complexity: int, round_idx: int) -> List[Dict]:
        """生成一轮执行结果"""
        results = []

        # 每轮生成1-3个查询结果
        num_queries = min(2 + round_idx, 3)

        for i in range(num_queries):
            # 模拟查询执行
            success = random.random() > 0.1  # 90%成功率
            rows = random.randint(5, 50) if success else 0

            # 生成模拟数据
            data = []
            if success and rows > 0:
                for j in range(min(rows, 5)):  # 只保留前5行
                    if "region" in question.lower():
                        data.append({
                            "region": random.choice(["MOp", "SSp", "VISp", "CLA"]),
                            "value": random.uniform(0, 100)
                        })
                    elif "subclass" in question.lower():
                        data.append({
                            "region": random.choice(["MOp", "SSp"]),
                            "subclass": f"Subclass_{j}",
                            "percentage": random.uniform(0, 30)
                        })
                    else:
                        data.append({"id": j, "value": random.uniform(0, 100)})

            results.append({
                "idx": len(results) + 1,
                "purpose": f"Query {i + 1} in round {round_idx + 1}",
                "query": "MATCH ... RETURN ... LIMIT 50",
                "success": success,
                "rows": rows,
                "data": data,
                "t": random.uniform(0.1, 2.0)
            })

        return results

    def _generate_final_answer(self, question: str, results: List[Dict], metrics: List[Dict]) -> str:
        """生成最终答案"""
        successful_queries = sum(1 for r in results if r["success"])
        total_rows = sum(r["rows"] for r in results)

        answer = f"Based on the analysis of {successful_queries} successful queries retrieving {total_rows} total rows:\n\n"

        if "mismatch" in question.lower() and metrics:
            mismatch_val = metrics[0].get("value", 0.5)
            answer += f"The mismatch index between the specified regions is {mismatch_val:.3f}, "
            answer += "indicating a moderate discrepancy between morphological and transcriptomic distances.\n\n"

        if "morpholog" in question.lower():
            answer += "The morphological analysis reveals significant variation across regions:\n"
            answer += "- MOp shows high axonal branching (mean: 45.2)\n"
            answer += "- SSp exhibits balanced dendritic patterns\n"
            answer += "- VISp has the highest axon-to-dendrite ratio (2.3)\n\n"

        if "subclass" in question.lower():
            answer += "Subclass distribution analysis shows:\n"
            answer += "- L2/3 IT neurons comprise 25-30% across cortical regions\n"
            answer += "- L5 ET neurons are enriched in motor areas (18% in MOp vs 8% in VISp)\n"
            answer += "- Interneuron diversity is highest in sensory regions\n\n"

        if not answer.strip().endswith(":"):
            answer += f"The analysis successfully addressed the query with {len(results)} analytical steps."

        return answer


def run_mock_evaluation():
    """运行Mock评估以测试评估系统"""
    print("=" * 60)
    print("Running Mock Evaluation for Testing")
    print("=" * 60)

    # 导入评估系统
    from evaluation import (
        TestSuite, KGAgentEvaluator, EvaluationVisualizer,
        EvaluationMetrics, TestCase, generate_detailed_report
    )

    # 创建mock agent
    mock_agent = MockKGAgentV7(
        neo4j_uri="mock://localhost",
        neo4j_user="mock_user",
        neo4j_pwd="mock_pwd",
        database="mock_db"
    )

    # 创建评估器
    evaluator = KGAgentEvaluator(mock_agent)

    # 运行评估
    print("\n▶ Running evaluation on test suite...")
    df_results = evaluator.run_evaluation()

    # 保存结果
    output_dir = Path("evaluation_results_mock")
    output_dir.mkdir(exist_ok=True)
    df_results.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    print(f"✓ Saved metrics to {output_dir / 'evaluation_metrics.csv'}")

    # 生成可视化
    print("\n▶ Generating visualization figures...")
    visualizer = EvaluationVisualizer(df_results, str(output_dir))
    visualizer.generate_all_figures()
    print(f"✓ Generated 7 figures in {output_dir}")

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("MOCK EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n▶ Tests Completed: {len(df_results)}")
    print(f"▶ Average Execution Time: {df_results['execution_time'].mean():.2f}s")
    print(
        f"▶ Overall Success Rate: {(df_results['successful_queries'].sum() / df_results['total_queries'].sum() * 100):.1f}%")

    # 按类别统计
    print("\n▶ Performance by Category:")
    for category in df_results['category'].unique():
        cat_data = df_results[df_results['category'] == category]
        print(f"  • {category}:")
        print(f"    - Tests: {len(cat_data)}")
        print(f"    - Avg Autonomy: {cat_data['autonomy_score'].mean():.3f}")
        print(f"    - Avg Quality: {cat_data['final_answer_quality'].mean():.3f}")

    # 生成报告
    generate_detailed_report(evaluator, df_results, output_dir)

    return df_results


def run_real_evaluation(config: Dict[str, Any]):
    """运行真实的Agent评估"""
    print("=" * 60)
    print("KGAgent V7 Real Evaluation")
    print("=" * 60)

    try:
        # 导入真实的agent
        from agent_v7.agent_v7 import KGAgentV7

        # 导入评估系统
        from evaluation import (
            TestSuite, KGAgentEvaluator, EvaluationVisualizer,
            generate_detailed_report
        )

        # 创建真实agent
        agent = KGAgentV7(
            neo4j_uri=config['neo4j_uri'],
            neo4j_user=config['neo4j_user'],
            neo4j_pwd=config['neo4j_pwd'],
            database=config['database'],
            openai_api_key=config.get('openai_api_key'),
            planner_model=config.get('planner_model', 'gpt-5'),
            summarizer_model=config.get('summarizer_model', 'gpt-4o')
        )

        # 创建评估器
        evaluator = KGAgentEvaluator(agent)

        # 运行评估
        print("\n▶ Running evaluation on test suite...")
        df_results = evaluator.run_evaluation()

        # 保存结果
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        df_results.to_csv(output_dir / "evaluation_metrics.csv", index=False)
        print(f"✓ Saved metrics to {output_dir / 'evaluation_metrics.csv'}")

        # 生成可视化
        print("\n▶ Generating visualization figures...")
        visualizer = EvaluationVisualizer(df_results, str(output_dir))
        visualizer.generate_all_figures()
        print(f"✓ Generated 7 figures in {output_dir}")

        # 打印关键结果
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print("\n▶ Overall Performance:")
        print(f"  • Average Autonomy Score: {df_results['autonomy_score'].mean():.3f}")
        print(f"  • Average Planning Quality: {df_results['planning_quality'].mean():.3f}")
        print(f"  • Average Tool Selection Accuracy: {df_results['tool_selection_accuracy'].mean():.3f}")
        print(f"  • Average Answer Quality: {df_results['final_answer_quality'].mean():.3f}")

        print("\n▶ Efficiency Metrics:")
        print(f"  • Average Execution Time: {df_results['execution_time'].mean():.2f}s")
        print(f"  • Average Iteration Rounds: {df_results['iteration_rounds'].mean():.2f}")
        total_q = df_results['total_queries'].sum()
        success_q = df_results['successful_queries'].sum()
        print(f"  • Query Success Rate: {(success_q / total_q * 100 if total_q > 0 else 0):.1f}%")

        print("\n▶ Advanced Capabilities:")
        print(f"  • Mismatch Index Usage: {df_results['mismatch_computation'].mean() * 100:.1f}%")
        print(f"  • Average CoT Depth: {df_results['cot_depth'].mean():.2f}")
        print(f"  • Schema Utilization Rate: {df_results['schema_utilization_rate'].mean():.3f}")

        # 生成详细报告
        generate_detailed_report(evaluator, df_results, output_dir)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        return df_results, evaluator

    except ImportError as e:
        print(f"\n⚠ Cannot import real agent: {e}")
        print("▶ Running mock evaluation instead...")
        return run_mock_evaluation()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='KGAgent V7 Evaluation System')
    parser.add_argument('--mode', choices=['mock', 'real'], default='mock',
                        help='Run mock or real evaluation')
    parser.add_argument('--neo4j-uri', default='bolt://100.88.72.32:7687',
                        help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--neo4j-password', default='neuroxiv',
                        help='Neo4j password')
    parser.add_argument('--database', default='neo4j',
                        help='Neo4j database')
    parser.add_argument('--openai-key', default=None,
                        help='OpenAI API key')
    parser.add_argument('--planner-model', default='gpt-5',
                        help='Planner model')
    parser.add_argument('--summarizer-model', default='gpt-4o',
                        help='Summarizer model')

    args = parser.parse_args()

    if args.mode == 'mock':
        print("Running MOCK evaluation for testing...")
        results = run_mock_evaluation()
    else:
        print("Running REAL evaluation...")
        config = {
            'neo4j_uri': args.neo4j_uri,
            'neo4j_user': args.neo4j_user,
            'neo4j_pwd': args.neo4j_password,
            'database': args.database,
            'openai_api_key': args.openai_key or os.getenv('OPENAI_API_KEY'),
            'planner_model': args.planner_model,
            'summarizer_model': args.summarizer_model
        }
        results, evaluator = run_real_evaluation(config)

    print("\n✓ All evaluations completed successfully!")
    print(f"✓ Results saved in: evaluation_results{'_mock' if args.mode == 'mock' else ''}/")

    # 显示结果统计
    if results is not None and not results.empty:
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        # 计算综合得分
        overall_score = (
                results['autonomy_score'].mean() * 0.3 +
                results['planning_quality'].mean() * 0.2 +
                results['tool_selection_accuracy'].mean() * 0.2 +
                results['final_answer_quality'].mean() * 0.3
        )

        print(f"\n▶ Overall System Score: {overall_score:.3f}/1.000")

        # 评级
        if overall_score >= 0.8:
            grade = "A (Excellent)"
        elif overall_score >= 0.7:
            grade = "B (Good)"
        elif overall_score >= 0.6:
            grade = "C (Satisfactory)"
        else:
            grade = "D (Needs Improvement)"

        print(f"▶ System Grade: {grade}")

        # 找出最佳和最差的测试
        results['overall_test_score'] = (
                results['autonomy_score'] * 0.3 +
                results['planning_quality'] * 0.2 +
                results['tool_selection_accuracy'] * 0.2 +
                results['final_answer_quality'] * 0.3
        )

        best_test = results.loc[results['overall_test_score'].idxmax()]
        worst_test = results.loc[results['overall_test_score'].idxmin()]

        print(f"\n▶ Best Performing Test:")
        print(f"  • ID: {best_test['test_id']}")
        print(f"  • Category: {best_test['category']}")
        print(f"  • Score: {best_test['overall_test_score']:.3f}")

        print(f"\n▶ Worst Performing Test:")
        print(f"  • ID: {worst_test['test_id']}")
        print(f"  • Category: {worst_test['category']}")
        print(f"  • Score: {worst_test['overall_test_score']:.3f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()