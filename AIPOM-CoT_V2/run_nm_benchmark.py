"""
Main Entry Point for Nature Methods Benchmark
==============================================
运行完整的NM benchmark评估

Usage:
    python run_nm_benchmark.py --mode quick     # 10 questions
    python run_nm_benchmark.py --mode extended  # 30 questions
    python run_nm_benchmark.py --mode full      # 100 questions

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 4.0 (Nature Methods)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append('/tmp')

from test_questions import (
    ALL_QUESTIONS,
    TIER1_SIMPLE,
    TIER2_MEDIUM,
    TIER3_DEEP,
    TIER4_SCREENING,
    ComplexityLevel,
    get_questions_by_complexity
)
from benchmark_runner_v4 import BenchmarkRunnerV4
from visualization_v4 import BenchmarkVisualizerV4

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nm_benchmark.log', mode='w')
        ]
    )


def initialize_systems():
    """初始化所有系统组件"""

    logger.info("Initializing systems for NM benchmark...")

    # Neo4j
    from neo4j_exec import Neo4jExec

    neo4j_exec = Neo4jExec(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )

    logger.info("  ✓ Neo4j connected")

    # OpenAI
    from openai import OpenAI

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logger.info("  ✓ OpenAI client initialized")

    # AIPOM-CoT Agent
    from aipom_v10_production import AIPOMCoTV10

    aipom_agent = AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path="schema_output/schema.json",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    logger.info("  ✓ AIPOM-CoT agent initialized")

    return neo4j_exec, openai_client, aipom_agent


def run_quick(neo4j_exec, openai_client, aipom_agent):
    """Quick test (10 questions) - v4.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 QUICK NM BENCHMARK (10 questions)")
    logger.info("=" * 80)

    # 选择代表性问题
    selected = []
    selected.extend(TIER1_SIMPLE[:2])
    selected.extend(TIER2_MEDIUM[:3])
    selected.extend(TIER3_DEEP[:3])
    selected.extend(TIER4_SCREENING[:2])

    runner = BenchmarkRunnerV4(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_nm_quick"
    )

    summary = runner.run_full_benchmark(
        questions=selected,
        methods=['AIPOM-CoT', 'Direct GPT-4o', 'RAG', 'ReAct'],
        save_interval=5
    )

    runner.print_summary()

    # 生成可视化
    logger.info("\n📊 Generating visualizations...")
    visualizer = BenchmarkVisualizerV4("./benchmark_results_nm_quick")
    visualizer.generate_all_figures()

    return summary


def run_extended(neo4j_exec, openai_client, aipom_agent):
    """Extended test (30 questions) - v4.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 EXTENDED NM BENCHMARK (30 questions)")
    logger.info("=" * 80)

    # 按complexity level选择
    selected = []

    level1 = get_questions_by_complexity(ComplexityLevel.LEVEL_1)
    selected.extend(level1[:8])

    level2 = get_questions_by_complexity(ComplexityLevel.LEVEL_2)
    selected.extend(level2[:15])

    level3 = get_questions_by_complexity(ComplexityLevel.LEVEL_3)
    selected.extend(level3[:7])

    runner = BenchmarkRunnerV4(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_nm_extended"
    )

    summary = runner.run_full_benchmark(
        questions=selected,
        methods=['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct'],
        save_interval=5
    )

    runner.print_summary()

    logger.info("\n📊 Generating visualizations...")
    visualizer = BenchmarkVisualizerV4("./benchmark_results_nm_extended")
    visualizer.generate_all_figures()

    return summary


def run_full(neo4j_exec, openai_client, aipom_agent):
    """Full benchmark (100 questions) - v4.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 FULL NM BENCHMARK (100 questions)")
    logger.info("=" * 80)

    runner = BenchmarkRunnerV4(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_nm_full"
    )

    summary = runner.run_full_benchmark(
        questions=ALL_QUESTIONS,
        methods=['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct'],
        save_interval=10
    )

    runner.print_summary()

    logger.info("\n📊 Generating visualizations...")
    visualizer = BenchmarkVisualizerV4("./benchmark_results_nm_full")
    visualizer.generate_all_figures()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Nature Methods Benchmark Suite for AIPOM-CoT'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'extended', 'full'],
        default='quick',
        help='Benchmark mode'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("\n" + "=" * 80)
    logger.info("🎯 AIPOM-CoT Nature Methods Benchmark Suite v4.0")
    logger.info("=" * 80)
    logger.info("\n🔬 Evaluation Dimensions:")
    logger.info("  1. Planning Quality")
    logger.info("  2. Reasoning Capability")
    logger.info("  3. Chain-of-Thought Quality")
    logger.info("  4. Reflection Capability")
    logger.info("  5. Natural Language Understanding")
    logger.info("\n" + "=" * 80)

    try:
        # 初始化系统
        neo4j_exec, openai_client, aipom_agent = initialize_systems()

        # 运行benchmark
        if args.mode == 'quick':
            summary = run_quick(neo4j_exec, openai_client, aipom_agent)

        elif args.mode == 'extended':
            summary = run_extended(neo4j_exec, openai_client, aipom_agent)

        elif args.mode == 'full':
            summary = run_full(neo4j_exec, openai_client, aipom_agent)

        logger.info("\n" + "=" * 80)
        logger.info("✅ NATURE METHODS BENCHMARK COMPLETE!")
        logger.info("=" * 80)

        # 打印关键发现
        logger.info("\n🔬 KEY FINDINGS:")

        if 'AIPOM-CoT' in summary:
            aipom_summary = summary['AIPOM-CoT']

            logger.info("\nAIPOM-CoT Performance:")
            logger.info(f"  - NM Capability Score: {aipom_summary.get('nm_capability_score', {}).get('mean', 0):.3f}")
            logger.info(f"  - Planning Quality: {aipom_summary.get('planning_quality', {}).get('mean', 0):.3f}")
            logger.info(f"  - Reasoning Capability: {aipom_summary.get('reasoning_capability', {}).get('mean', 0):.3f}")
            logger.info(f"  - CoT Quality: {aipom_summary.get('cot_quality', {}).get('mean', 0):.3f}")
            logger.info(
                f"  - Reflection Capability: {aipom_summary.get('reflection_capability', {}).get('mean', 0):.3f}")
            logger.info(f"  - NLU Capability: {aipom_summary.get('nlu_capability', {}).get('mean', 0):.3f}")

        logger.info("\n" + "=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Benchmark interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()