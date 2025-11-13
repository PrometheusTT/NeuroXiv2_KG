"""
Complete System Test Script
===========================
测试所有新增功能:
1. Figure 3 (Focus-Driven)
2. Figure 4 (Comparative)
3. 统计分析自动调用

Author: Claude & PrometheusTT
Date: 2025-11-13
"""

import os
import logging
import json
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from aipom_v10_production import AIPOMCoTV10


def test_figure3_focus_driven():
    """
    测试Figure 3: Focus-Driven深度分析

    期望:
    - 找到所有Car3+ regions
    - 识别PRIMARY FOCUS (CLA)
    - 深入分析CLA的molecular/morphological/projection
    - 闭环: 分析projection targets的molecular composition
    """
    print("\n" + "=" * 80)
    print("TEST 1: Figure 3 Focus-Driven Analysis")
    print("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # 🎯 不需要"comprehensive"关键词
    question = "Tell me about Car3+ neurons"

    result = agent.answer(question, max_iterations=12)

    # 验证结果
    print("\n" + "-" * 80)
    print("VALIDATION:")
    print("-" * 80)

    checks = {
        '✓ Has regions': 'Region' in result['adaptive_planning']['entities_discovered'],
        '✓ Identified primary focus': result['adaptive_planning'].get('primary_focus') is not None,
        '✓ Has molecular analysis': 'molecular' in result['adaptive_planning']['modalities_covered'],
        '✓ Has morphological analysis': 'morphological' in result['adaptive_planning']['modalities_covered'],
        '✓ Has projection analysis': 'projection' in result['adaptive_planning']['modalities_covered'],
        '✓ Has projection targets': 'ProjectionTarget' in result['adaptive_planning']['entities_discovered'],
        '✓ Closed loop (target composition)': any(
            'target' in s['purpose'].lower() and 'composition' in s['purpose'].lower()
            for s in result['executed_steps'])
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    # 打印关键指标
    print("\n" + "-" * 80)
    print("KEY METRICS:")
    print("-" * 80)
    print(f"Steps executed: {result['total_steps']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"Modalities: {', '.join(result['adaptive_planning']['modalities_covered'])}")

    primary_focus = result['adaptive_planning'].get('primary_focus')
    if primary_focus:
        print(f"Primary focus: {primary_focus.entity_id}")

    # 打印答案摘要
    print("\n" + "-" * 80)
    print("ANSWER (first 500 chars):")
    print("-" * 80)
    print(result['answer'][:500] + "...")

    # 保存结果
    output_dir = Path("./test_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "figure3_result.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ Full result saved to: {output_dir / 'figure3_result.json'}")

    agent.close()

    # 评分
    score = sum(checks.values()) / len(checks) * 100
    print(f"\n📊 Figure 3 Completeness: {score:.0f}%")

    return score >= 80  # 80%以上算成功


def test_figure4_comparative():
    """
    测试Figure 4: Comparative Analysis

    期望:
    - Pairwise模式: 对比两个regions
    - 自动调用统计检验
    - 计算effect size
    """
    print("\n" + "=" * 80)
    print("TEST 2: Figure 4 Comparative Analysis (Pairwise)")
    print("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    question = "Compare MOs and SSp"

    result = agent.answer(question, max_iterations=10)

    # 验证结果
    print("\n" + "-" * 80)
    print("VALIDATION:")
    print("-" * 80)

    checks = {
        '✓ Detected comparison intent': 'comparison' in result.get('adaptive_planning', {}).get('question_intent',
                                                                                                '').lower(),
        '✓ Has molecular comparison': any('molecular' in s['purpose'].lower() and 'compare' in s['purpose'].lower()
                                          for s in result['executed_steps']),
        '✓ Has morphological comparison': any('morphological' in s['purpose'].lower()
                                              for s in result['executed_steps']),
        '✓ Has statistical test': any('statistical' in s['purpose'].lower()
                                      for s in result['executed_steps']),
        '✓ Answer mentions both entities': 'MOs' in result['answer'] and 'SSp' in result['answer']
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    print("\n" + "-" * 80)
    print("ANSWER (first 500 chars):")
    print("-" * 80)
    print(result['answer'][:500] + "...")

    # 保存结果
    output_dir = Path("./test_results")
    with open(output_dir / "figure4_pairwise_result.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ Full result saved to: {output_dir / 'figure4_pairwise_result.json'}")

    agent.close()

    score = sum(checks.values()) / len(checks) * 100
    print(f"\n📊 Figure 4 Pairwise Completeness: {score:.0f}%")

    return score >= 60  # 60%以上算成功


def test_statistical_tools():
    """
    测试统计工具自动调用
    """
    print("\n" + "=" * 80)
    print("TEST 3: Statistical Tools Auto-Invocation")
    print("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="./schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # 测试StatisticalTools.correlation_test是否添加成功
    print("\n✓ Testing StatisticalTools.correlation_test...")

    import numpy as np
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    result = agent.stats.correlation_test(x, y, method='pearson')

    print(f"  Correlation: {result['correlation']:.3f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")

    assert abs(result['correlation'] - 1.0) < 0.01, "Correlation should be ~1.0"
    assert result['p_value'] < 0.05, "Should be significant"

    print("✅ StatisticalTools.correlation_test works!")

    agent.close()

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🚀 AIPOM-CoT V10 COMPLETE SYSTEM TEST")
    print("=" * 80)

    results = {}

    try:
        results['figure3'] = test_figure3_focus_driven()
    except Exception as e:
        logger.error(f"Figure 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        results['figure3'] = False

    try:
        results['figure4'] = test_figure4_comparative()
    except Exception as e:
        logger.error(f"Figure 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        results['figure4'] = False

    try:
        results['stats'] = test_statistical_tools()
    except Exception as e:
        logger.error(f"Statistical tools test failed: {e}")
        import traceback
        traceback.print_exc()
        results['stats'] = False

    # 总结
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n{'=' * 80}")
    print(f"Overall: {total_passed}/{total_tests} tests passed ({total_passed / total_tests * 100:.0f}%)")
    print(f"{'=' * 80}")

    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is ready for production!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    run_all_tests()