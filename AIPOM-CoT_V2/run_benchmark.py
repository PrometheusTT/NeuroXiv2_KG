"""
Main Entry Point for AIPOM-CoT Benchmark
=========================================
运行完整benchmark的主入口

Usage:
    # Quick test (10 questions)
    python run_benchmark.py --mode quick

    # Extended test (30 questions)
    python run_benchmark.py --mode extended

    # Full benchmark (100 questions)
    python run_benchmark.py --mode full

    # Analyze existing results
    python run_benchmark.py --mode analyze

Author: Claude & PrometheusTT
Date: 2025-01-15
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from test_questions import ALL_QUESTIONS, TIER1_SIMPLE, TIER2_MEDIUM, TIER3_DEEP, TIER4_SCREENING
from benchmark_runner import BenchmarkRunner, run_quick_test
from statistical_analysis import StatisticalAnalyzer
from visualization import BenchmarkVisualizer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log', mode='w')
        ]
    )


def initialize_systems():
    """初始化所有系统组件"""

    logger.info("Initializing systems...")

    # 1. Neo4j
    from neo4j_exec import Neo4jExec

    neo4j_exec = Neo4jExec(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )

    logger.info("  ✓ Neo4j connected")

    # 2. OpenAI
    from openai import OpenAI

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",''))

    logger.info("  ✓ OpenAI client initialized")

    # 3. AIPOM-CoT Agent
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


# 修改main()函数中的方法列表

def run_quick(neo4j_exec, openai_client, aipom_agent):
    """Quick test (10 questions) - v2.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 QUICK TEST MODE (10 questions, v2.0)")
    logger.info("=" * 80)

    from test_questions import TIER1_SIMPLE, TIER2_MEDIUM, TIER3_DEEP, TIER4_SCREENING

    # 选择代表性问题
    selected = []
    selected.extend(TIER1_SIMPLE[:2])  # 2个简单
    selected.extend(TIER2_MEDIUM[:2])  # 3个中等
    selected.extend(TIER3_DEEP[:2])  # 3个深度
    selected.extend(TIER4_SCREENING[:2])  # 2个筛选

    runner = BenchmarkRunner(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_quick_v2"
    )

    # 🔧 更新：新的方法列表
    summary = runner.run_full_benchmark(
        questions=selected,
        methods = ['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct'],  # 🔧
        save_interval=5
    )

    runner.print_summary()

    return summary


def run_extended(neo4j_exec, openai_client, aipom_agent):
    """Extended test (30 questions) - v2.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 EXTENDED TEST MODE (30 questions, v2.0)")
    logger.info("=" * 80)

    from test_questions import ComplexityLevel, get_questions_by_complexity

    # 🔧 按complexity level选择
    selected = []

    level1_questions = get_questions_by_complexity(ComplexityLevel.LEVEL_1)
    selected.extend(level1_questions[:8])  # 8个Level 1

    level2_questions = get_questions_by_complexity(ComplexityLevel.LEVEL_2)
    selected.extend(level2_questions[:15])  # 15个Level 2

    level3_questions = get_questions_by_complexity(ComplexityLevel.LEVEL_3)
    selected.extend(level3_questions[:7])  # 7个Level 3

    runner = BenchmarkRunner(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_extended_v2"
    )

    summary = runner.run_full_benchmark(
        questions=selected,
        methods=['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct'],  # 🔧
        save_interval=5
    )

    runner.print_summary()

    return summary


def run_full(neo4j_exec, openai_client, aipom_agent):
    """Full benchmark (100 questions) - v2.0"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 FULL BENCHMARK MODE (100 questions, v2.0)")
    logger.info("=" * 80)

    from test_questions import ALL_QUESTIONS

    runner = BenchmarkRunner(
        aipom_agent,
        neo4j_exec,
        openai_client,
        output_dir="./benchmark_results_full_v2"
    )

    summary = runner.run_full_benchmark(
        questions=ALL_QUESTIONS,
        methods=['AIPOM-CoT', 'Direct GPT-4o', 'Template-KG', 'RAG', 'ReAct'],
        save_interval=10
    )

    runner.print_summary()

    return summary


def run_analyze(results_dir: str = "./benchmark_results"):
    """Analyze existing results"""

    logger.info("\n" + "="*80)
    logger.info("📊 ANALYSIS MODE")
    logger.info("="*80)

    # Statistical analysis
    analyzer = StatisticalAnalyzer(results_dir)
    df = analyzer.run_full_analysis()
    latex = analyzer.generate_latex_table()

    # Visualization
    visualizer = BenchmarkVisualizer(results_dir)
    visualizer.generate_all_figures()

    logger.info("\n✅ Analysis complete!")


def main():
    parser = argparse.ArgumentParser(description='AIPOM-CoT Benchmark Suite')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'extended', 'full', 'analyze'],
        default='quick',
        help='Benchmark mode'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./benchmark_results',
        help='Results directory for analysis mode'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("\n" + "="*80)
    logger.info("🎯 AIPOM-CoT Benchmark Suite")
    logger.info("="*80)

    try:
        if args.mode == 'analyze':
            # 只分析，不运行测试
            run_analyze(args.results_dir)

        else:
            # 初始化系统
            neo4j_exec, openai_client, aipom_agent = initialize_systems()

            # 运行benchmark
            if args.mode == 'quick':
                summary = run_quick(neo4j_exec, openai_client, aipom_agent)

            elif args.mode == 'extended':
                summary = run_extended(neo4j_exec, openai_client, aipom_agent)

            elif args.mode == 'full':
                summary = run_full(neo4j_exec, openai_client, aipom_agent)

            # 自动分析
            logger.info("\n📊 Running analysis...")

            if args.mode == 'quick':
                results_dir = "./benchmark_results_quick"
            elif args.mode == 'extended':
                results_dir = "./benchmark_results_extended"
            else:
                results_dir = "./benchmark_results_full"

            run_analyze(results_dir)

        logger.info("\n" + "="*80)
        logger.info("✅ BENCHMARK COMPLETE!")
        logger.info("="*80)

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