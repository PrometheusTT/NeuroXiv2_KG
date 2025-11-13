"""
Complete Benchmark Evaluation Script
====================================
运行完整的benchmark评估并生成报告

使用:
    python run_benchmark.py --max-questions 50

Author: Claude & PrometheusTT
Date: 2025-01-12
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from aipom_v10_production import AIPOMCoTV10
from benchmark_system import (
    BenchmarkQuestionBank,
    BenchmarkRunner,
    generate_test_questions_file
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run AIPOM-CoT benchmark evaluation')

    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str, required=True,
                        help='Neo4j password')
    parser.add_argument('--database', type=str, default='neo4j',
                        help='Neo4j database name')
    parser.add_argument('--schema-json', type=str, default='./schema_output/schema.json',
                        help='Path to schema.json')
    parser.add_argument('--openai-api-key', type=str, default="",
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model name')

    parser.add_argument('--questions-file', type=str, default='test_questions.json',
                        help='Path to test questions JSON')
    parser.add_argument('--generate-questions', action='store_true',
                        help='Generate test questions file')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Maximum number of questions to test (for quick testing)')

    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # 获取API key
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not provided. Set --openai-api-key or OPENAI_API_KEY env var")
        sys.exit(1)

    # 生成测试问题 (如果需要)
    if args.generate_questions:
        logger.info("Generating test questions...")
        generate_test_questions_file()
        logger.info(f"✅ Test questions saved to {args.questions_file}")

    # 检查问题文件
    if not Path(args.questions_file).exists():
        logger.error(f"Questions file not found: {args.questions_file}")
        logger.info("Run with --generate-questions to create it")
        sys.exit(1)

    # 加载测试问题
    logger.info(f"Loading test questions from {args.questions_file}...")
    questions = BenchmarkQuestionBank.load_from_json(args.questions_file)

    logger.info(f"Loaded {len(questions)} test questions")

    # 初始化Agent
    logger.info("\n" + "=" * 80)
    logger.info("Initializing AIPOM-CoT V10 Agent")
    logger.info("=" * 80)

    agent = AIPOMCoTV10(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pwd=args.neo4j_password,
        database=args.database,
        schema_json_path=args.schema_json,
        openai_api_key=api_key,
        model=args.model
    )

    # 初始化Benchmark Runner
    logger.info("\n" + "=" * 80)
    logger.info("Starting Benchmark Evaluation")
    logger.info("=" * 80)

    runner = BenchmarkRunner(agent, output_dir=args.output_dir)

    # 运行benchmark
    try:
        results = runner.run_benchmark(questions, max_questions=args.max_questions)

        logger.info("\n" + "=" * 80)
        logger.info("✅ Benchmark Complete!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")

        # 打印快速摘要
        successful = sum(1 for r in results if r.success)
        avg_accuracy = sum(r.accuracy_score for r in results if r.success) / successful if successful > 0 else 0

        print(f"\n📊 Quick Summary:")
        print(f"   Total: {len(results)}")
        print(f"   Successful: {successful} ({successful / len(results) * 100:.1f}%)")
        print(f"   Average Accuracy: {avg_accuracy:.3f}")

    except KeyboardInterrupt:
        logger.info("\n❌ Benchmark interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        agent.close()


if __name__ == "__main__":
    main()