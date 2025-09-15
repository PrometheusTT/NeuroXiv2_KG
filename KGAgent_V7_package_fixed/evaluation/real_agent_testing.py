#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 真实Agent测试集成
完整展示如何将评估系统与您的真实Agent连接
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

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
                summarizer_model=config['summarizer_model']
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


# ==================== STEP 4: 导入评估系统 ====================
USE_IMPROVED_EVALUATION = True  # 切换到改进版

if USE_IMPROVED_EVALUATION:
    from improved_evaluation import (
        ImprovedEvaluator,
        ImprovedEvaluationMetrics,
        run_improved_evaluation,
        ScientificVisualizer
    )
    # 使用改进版的TestCase定义
    from evaluation import TestCase, TestSuite
else:
    from evaluation import (
        TestCase, TestSuite, EvaluationMetrics,
        KGAgentEvaluator, EvaluationVisualizer
    )


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


# ==================== STEP 6: 运行真实评估 ====================
class RealEvaluationRunner:
    """真实评估运行器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_wrapper = None
        self.evaluator = None
        self.results = None

    def setup(self) -> bool:
        """设置评估环境"""
        logger.info("\n" + "=" * 60)
        logger.info("SETTING UP EVALUATION ENVIRONMENT")
        logger.info("=" * 60)

        # 验证配置
        if not AgentConfig.validate_config(self.config):
            return False

        # 创建Agent
        self.agent_wrapper = RealAgentWrapper(self.config)

        # 测试连接
        if not self.agent_wrapper.test_connection():
            logger.error("❌ Agent connection test failed")
            return False

        # 创建评估器（使用自定义测试套件）
        self.evaluator = KGAgentEvaluator(self.agent_wrapper)
        self.evaluator.test_suite = CustomTestSuite()

        logger.info("✅ Evaluation environment ready")
        return True

    def run_evaluation(self,
                       test_subset: List[str] = None,
                       output_dir: str = "evaluation_results") -> pd.DataFrame:
        """
        运行评估

        Args:
            test_subset: 要运行的测试ID列表（None表示运行所有）
            output_dir: 输出目录
        """
        if not self.evaluator:
            logger.error("❌ Evaluator not initialized. Run setup() first.")
            return None

        logger.info("\n" + "=" * 60)
        logger.info("RUNNING EVALUATION")
        logger.info("=" * 60)

        # 选择测试用例
        test_cases = self.evaluator.test_suite.test_cases
        if test_subset:
            test_cases = [tc for tc in test_cases if tc.id in test_subset]
            logger.info(f"Running subset: {[tc.id for tc in test_cases]}")
        else:
            logger.info(f"Running all {len(test_cases)} test cases")

        # 运行评估
        all_metrics = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n[{i}/{len(test_cases)}] Evaluating: {test_case.id}")
            logger.info(f"  Question: {test_case.question}")

            try:
                result, metrics = self.evaluator.evaluate_single(test_case)

                # 记录结果
                metric_dict = {
                    'test_id': test_case.id,
                    'category': test_case.category,
                    'complexity': test_case.complexity,
                    **metrics.__dict__
                }
                all_metrics.append(metric_dict)

                logger.info(f"  ✅ Success - Score: {metrics.final_answer_quality:.2f}")

            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                # 添加失败记录
                all_metrics.append({
                    'test_id': test_case.id,
                    'category': test_case.category,
                    'complexity': test_case.complexity,
                    'error': str(e)
                })

        # 创建DataFrame
        self.results = pd.DataFrame(all_metrics)

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        self.results.to_csv(output_path / "evaluation_results.csv", index=False)
        logger.info(f"\n✅ Results saved to: {output_path / 'evaluation_results.csv'}")

        return self.results

    def generate_visualizations(self, output_dir: str = "evaluation_results"):
        """生成可视化"""
        if self.results is None or self.results.empty:
            logger.error("❌ No results to visualize")
            return

        logger.info("\n" + "=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        # 导入可视化模块
        from visualization import AdvancedVisualizer

        # 生成图表
        visualizer = AdvancedVisualizer(self.results, output_dir)
        visualizer.generate_all_figures()

        logger.info(f"✅ Visualizations saved to: {output_dir}/")

    def print_summary(self):
        """打印评估摘要"""
        if self.results is None or self.results.empty:
            return

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        # 过滤掉错误记录
        valid_results = self.results[~self.results.get('error', pd.Series()).notna()]

        if not valid_results.empty:
            # 计算统计
            logger.info(f"\n📊 Tests Completed: {len(valid_results)}/{len(self.results)}")

            if 'overall_score' not in valid_results.columns:
                valid_results['overall_score'] = (
                        valid_results.get('autonomy_score', 0) * 0.3 +
                        valid_results.get('planning_quality', 0) * 0.2 +
                        valid_results.get('tool_selection_accuracy', 0) * 0.2 +
                        valid_results.get('final_answer_quality', 0) * 0.3
                )

            logger.info(f"\n🎯 Performance Metrics:")
            for metric in ['autonomy_score', 'planning_quality', 'tool_selection_accuracy', 'final_answer_quality']:
                if metric in valid_results.columns:
                    mean_val = valid_results[metric].mean()
                    std_val = valid_results[metric].std()
                    logger.info(f"  • {metric}: {mean_val:.3f} ± {std_val:.3f}")

            logger.info(f"\n🏆 Overall Score: {valid_results['overall_score'].mean():.3f}")

            # 按类别统计
            logger.info(f"\n📈 By Category:")
            for cat in valid_results['category'].unique():
                cat_data = valid_results[valid_results['category'] == cat]
                logger.info(f"  • {cat}: {len(cat_data)} tests, avg score: {cat_data['overall_score'].mean():.3f}")


# ==================== STEP 7: 主函数 ====================
def main():
    """主函数：运行真实Agent评估"""
    import argparse

    parser = argparse.ArgumentParser(description='Run KGAgent V7 Real Evaluation')
    parser.add_argument('--subset', nargs='+', help='Test IDs to run (default: all)')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--config-file', help='JSON config file path')

    args = parser.parse_args()

    # 加载配置
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from: {args.config_file}")
    else:
        config = AgentConfig.get_config()

    # 创建运行器
    runner = RealEvaluationRunner(config)

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
        # 生成可视化
        if not args.skip_viz:
            runner.generate_visualizations(args.output)

        # 打印摘要
        runner.print_summary()

        logger.info("\n" + "=" * 60)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("❌ Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())