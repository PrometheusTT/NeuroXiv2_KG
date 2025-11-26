"""
AIPOM-CoT V11 Production Agent
===============================
完整集成所有V11组件的生产就绪Agent

解决的问题：
1. ✅ 代码重复 - 统一模块
2. ✅ 图代码一致性 - 完整实现Figure 2所有组件
3. ✅ 自主推理 - LLM参与所有决策
4. ✅ Evidence Buffer - 完整实现
5. ✅ 预算控制 - 统一管理

Author: Claude & Lijun
Date: 2025-01-15
"""

import os
import logging
from typing import Dict, Any, Optional

# 导入核心模块
from core_structures import (
    AnalysisState,
    AnalysisDepth,
    Modality,
    QuestionIntent,
    PlannerType,
    EvidenceBuffer,
    SessionMemory
)

from intent_classifier import LLMIntentClassifier, PlannerRouter
from llm_reflector import LLMReflector
from multimodal_analyzer import UnifiedFingerprintAnalyzer, StatisticalToolkit
from tpar_engine import TPAREngine

# 导入外部依赖（假设存在）
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

logger = logging.getLogger(__name__)


class AIPOMCoTV11:
    """
    AIPOM-CoT V11 生产版Agent

    完整实现Figure 2的所有组件：
    - A: Intelligent Intent Routing
    - B: Schema-aware Path Planning
    - C: TPAR Loop
    - D: Scientific Operator Library

    关键改进：
    1. LLM驱动的意图分类
    2. LLM驱动的反思
    3. 统一的证据缓冲
    4. 预算控制
    5. 会话记忆
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 schema_json_path: str,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o"):
        """
        初始化Agent

        Args:
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_pwd: Neo4j密码
            database: 数据库名
            schema_json_path: Schema JSON文件路径
            openai_api_key: OpenAI API Key
            model: 使用的模型
        """
        logger.info("🚀 Initializing AIPOM-CoT V11...")

        # 1. 数据库连接
        from neo4j_exec import Neo4jExec
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)

        # 2. Schema
        from aipom_cot_true_agent_v2 import RealSchemaCache
        self.schema = RealSchemaCache(schema_json_path)

        # 3. LLM客户端
        self.llm = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

        # 4. 实体识别器
        from intelligent_entity_recognition import IntelligentEntityRecognizer
        self.entity_recognizer = IntelligentEntityRecognizer(self.db, self.schema)

        # 5. 规划器
        self._init_planners()

        # 6. TPAR引擎
        self.tpar_engine = TPAREngine(
            db=self.db,
            schema=self.schema,
            llm_client=self.llm,
            entity_recognizer=self.entity_recognizer,
            focus_planner=self.focus_planner,
            comparative_planner=self.comparative_planner,
            adaptive_planner=self.adaptive_planner,
            model=model
        )

        # 7. 多模态分析器
        self.fingerprint_analyzer = UnifiedFingerprintAnalyzer(self.db)

        # 8. 会话记忆
        self.session_memory = SessionMemory()

        logger.info("✅ AIPOM-CoT V11 initialized successfully!")
        logger.info(f"   • Model: {model}")
        logger.info(f"   • Schema: {len(self.schema.node_types)} node types")
        logger.info(f"   • Database: {database}")

    def _init_planners(self):
        """初始化所有规划器"""
        from schema_path_planner import DynamicSchemaPathPlanner
        from focus_driven_planner import FocusDrivenPlanner
        from comparative_analysis_planner import ComparativeAnalysisPlanner
        from adaptive_planner import AdaptivePlanner

        # Schema路径规划器
        self.path_planner = DynamicSchemaPathPlanner(self.schema)

        # Focus-Driven Planner
        self.focus_planner = FocusDrivenPlanner(self.schema, self.db)

        # Comparative Planner
        fingerprint = UnifiedFingerprintAnalyzer(self.db)
        stats = StatisticalToolkit()
        self.comparative_planner = ComparativeAnalysisPlanner(
            self.db, fingerprint, stats
        )

        # Adaptive Planner
        self.adaptive_planner = AdaptivePlanner(
            self.schema, self.path_planner, self.llm
        )

    # ==================== Main API ====================

    def answer(self,
               question: str,
               max_iterations: int = 15) -> Dict[str, Any]:
        """
        回答问题 - 主入口

        Args:
            question: 用户问题
            max_iterations: 最大迭代次数

        Returns:
            包含答案和分析详情的字典
        """
        return self.tpar_engine.answer(question, max_iterations)

    def answer_with_visualization(self,
                                  question: str,
                                  max_iterations: int = 15,
                                  generate_plots: bool = True,
                                  output_dir: str = "./figure_output") -> Dict[str, Any]:
        """
        回答问题并生成可视化

        Args:
            question: 问题
            max_iterations: 最大迭代次数
            generate_plots: 是否生成图表
            output_dir: 输出目录

        Returns:
            包含答案和可视化文件路径的字典
        """
        # 执行分析
        result = self.answer(question, max_iterations)

        # 检测分析类型并生成图表
        if generate_plots:
            analysis_type = self._detect_analysis_type(result)

            if analysis_type == 'figure4_mismatch':
                try:
                    from aipom_v10_production import generate_figure4_from_agent_result
                    viz_files = generate_figure4_from_agent_result(result, output_dir)
                    result['visualization_files'] = viz_files
                except Exception as e:
                    logger.error(f"Visualization failed: {e}")
                    result['visualization_error'] = str(e)

        return result

    def _detect_analysis_type(self, result: Dict) -> str:
        """检测分析类型"""
        steps = result.get('executed_steps', [])

        has_mismatch = any('mismatch' in s.get('purpose', '').lower() for s in steps)
        has_screening = any('top' in s.get('purpose', '').lower() for s in steps)

        if has_mismatch and has_screening:
            return 'figure4_mismatch'

        return 'other'

    # ==================== Utility Methods ====================

    def get_session_summary(self) -> Dict:
        """获取会话摘要"""
        return {
            'session_id': self.session_memory.session_id,
            'qa_count': len(self.session_memory.qa_history),
            'known_entities': {
                k: len(v) for k, v in self.session_memory.known_entities.items()
            }
        }

    def clear_session(self):
        """清除会话记忆"""
        self.session_memory = SessionMemory()
        logger.info("Session cleared")

    def clear_cache(self):
        """清除缓存"""
        self.fingerprint_analyzer.clear_cache()
        logger.info("Cache cleared")

    def close(self):
        """关闭连接"""
        self.db.close()
        logger.info("Database connection closed")


# ==================== Factory Function ====================

def create_agent(
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_pwd: str = None,
        database: str = "neo4j",
        schema_json_path: str = "./schema_output/schema.json",
        openai_api_key: str = None,
        model: str = "gpt-4o"
) -> AIPOMCoTV11:
    """
    工厂函数：创建Agent实例

    支持从环境变量读取配置
    """
    return AIPOMCoTV11(
        neo4j_uri=neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=neo4j_user or os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=neo4j_pwd or os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=database or os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path=schema_json_path,
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        model=model
    )


# ==================== Test ====================

def test_v11_agent():
    """测试V11 Agent"""
    print("\n" + "=" * 80)
    print("AIPOM-CoT V11 TEST")
    print("=" * 80)

    agent = create_agent()

    test_questions = [
        # Simple query
        "Tell me about Car3+ neurons",

        # Screening query
        "Which brain region pairs show the highest cross-modal mismatch?",

        # Comparison query
        # "Compare Pvalb and Sst interneurons in MOs"
    ]

    for question in test_questions:
        print(f"\n{'=' * 80}")
        print(f"Q: {question}")
        print('=' * 80)

        result = agent.answer(question, max_iterations=10)

        print(f"\n✅ Results:")
        print(f"   Steps: {result.get('total_steps', 0)}")
        print(f"   Confidence: {result.get('confidence_score', 0):.3f}")
        print(f"   Time: {result.get('execution_time', 0):.2f}s")

        # 分析信息
        if 'analysis_info' in result:
            info = result['analysis_info']
            print(f"   Intent: {info.get('intent', 'unknown')}")
            print(f"   Depth: {info.get('target_depth', 'unknown')}")
            print(f"   Modalities: {', '.join(info.get('modalities_covered', []))}")

        print(f"\n💡 Answer:\n{result.get('answer', 'No answer')[:500]}...\n")

    agent.close()
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_v11_agent()