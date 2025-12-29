"""
NeuroXiv-KG Agent
==================
无比强大的自动神经数据分析Agent

核心能力：
1. LLM深度参与 - 真正的推理而非模式匹配
2. 高度灵活 - 动态适应不同问题类型
3. 多模态整合 - 分子/形态/投射三模态分析
4. 闭环分析 - 完整的circuit分析
5. 自我反思 - 智能决策和纠错

使用方式：
    agent = NeuroXivAgent.create(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="sk-..."
    )

    result = agent.answer("Tell me about Car3+ neurons")
    print(result['answer'])

Author: Lijun
Date: 2025-01
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

from core_structures import (
    AgentConfig, SessionMemory, AnalysisState,
    Modality, AnalysisDepth, QuestionIntent
)

from llm_intelligence import LLMClient, OpenAIClient

from tpar_engine import TPAREngine

from adaptive_planner import SchemaGraph

logger = logging.getLogger(__name__)


# ==================== Database Executor ====================

class Neo4jExecutor:
    """
    Neo4j数据库执行器

    特性：
    - 自动LIMIT
    - 重试机制
    - 超时控制
    """

    def __init__(self,
                 uri: str,
                 user: str,
                 password: str,
                 database: str = "neo4j",
                 timeout: int = 30):

        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.timeout = timeout

    def run(self, query: str, params: Dict = None) -> Dict:
        """执行查询"""
        import re
        import time

        params = params or {}

        # 确保LIMIT
        if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            query = f"{query}\nLIMIT 100"

        try:
            start = time.time()

            with self.driver.session(database=self.database) as session:
                result = session.run(query, params, timeout=self.timeout)
                data = [dict(record) for record in result]

            elapsed = time.time() - start

            return {
                'success': True,
                'data': data,
                'rows': len(data),
                'time': elapsed,
                'query': query
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'data': [],
                'error': str(e),
                'query': query
            }

    def close(self):
        """关闭连接"""
        try:
            self.driver.close()
        except:
            pass


class MockExecutor:
    """模拟执行器 - 用于测试"""

    def __init__(self):
        self._mock_data = self._load_mock_data()

    def _load_mock_data(self) -> Dict:
        """加载模拟数据"""
        return {
            'regions': [
                {'acronym': 'MOp', 'name': 'Primary motor area'},
                {'acronym': 'MOs', 'name': 'Secondary motor area'},
                {'acronym': 'SSp', 'name': 'Primary somatosensory area'},
                {'acronym': 'VISp', 'name': 'Primary visual area'},
            ],
            'clusters': [
                {'name': 'L5 IT CTX', 'markers': 'Car3,Satb2', 'number_of_neurons': 12345},
                {'name': 'L6 CT CTX', 'markers': 'Car3,Fezf2', 'number_of_neurons': 9876},
                {'name': 'L4 IT CTX', 'markers': 'Car3,Rorb', 'number_of_neurons': 5432},
            ],
            'subclasses': [
                {'name': 'L5 IT', 'markers': 'Car3,Slc17a7', 'description': 'Layer 5 intratelencephalic'},
                {'name': 'L6 CT', 'markers': 'Car3,Fezf2', 'description': 'Layer 6 corticothalamic'},
            ],
            'gene_abbrevs': {
                'VIP': 'vasoactive intestinal peptide',
                'SST': 'somatostatin',
                'Pvalb': 'parvalbumin',
                'Car3': 'carbonic anhydrase 3',
                'Lamp5': 'lysosomal associated membrane protein 5',
            },
            'region_abbrevs': {
                'MOp': 'Primary motor area',
                'MOs': 'Secondary motor area',
                'SSp': 'Primary somatosensory area',
                'VISp': 'Primary visual area',
                'HIP': 'Hippocampus',
                'TH': 'Thalamus',
            }
        }

    def run(self, query: str, params: Dict = None) -> Dict:
        """模拟执行查询"""
        params = params or {}
        query_lower = query.lower()

        # 解析查询类型
        if 'subclass' in query_lower:
            if 'markers contains' in query_lower:
                gene = params.get('gene', '')
                data = [s for s in self._mock_data['subclasses']
                       if gene.lower() in s['markers'].lower()]
            else:
                data = self._mock_data['subclasses']

        elif 'cluster' in query_lower:
            if 'markers contains' in query_lower:
                gene = params.get('gene', '')
                data = [c for c in self._mock_data['clusters']
                       if gene.lower() in c['markers'].lower()]
            else:
                data = self._mock_data['clusters']

        elif 'region' in query_lower:
            if 'acronym' in query_lower and params.get('region'):
                region = params.get('region', '')
                data = [r for r in self._mock_data['regions']
                       if r['acronym'] == region]
            elif 'has_cluster' in query_lower:
                gene = params.get('gene', '')
                # 模拟region enrichment
                data = [
                    {'region': 'MOp', 'region_name': 'Primary motor area',
                     'cluster_count': 15, 'total_neurons': 45000},
                    {'region': 'MOs', 'region_name': 'Secondary motor area',
                     'cluster_count': 12, 'total_neurons': 32000},
                    {'region': 'SSp', 'region_name': 'Primary somatosensory area',
                     'cluster_count': 18, 'total_neurons': 52000},
                ]
            else:
                data = self._mock_data['regions']

        elif 'neuron' in query_lower and 'locate_at' in query_lower:
            # 形态学查询
            data = [{
                'region': params.get('region', 'MOp'),
                'neuron_count': 1234,
                'avg_axon_length': 4567.89,
                'avg_dendrite_length': 1234.56,
                'avg_axon_branches': 45.6,
                'avg_dendrite_branches': 23.4,
            }]

        elif 'project_to' in query_lower:
            # 投射查询
            data = [
                {'source': 'MOp', 'target': 'TH', 'target_name': 'Thalamus',
                 'projection_weight': 0.85, 'neuron_count': 234},
                {'source': 'MOp', 'target': 'CP', 'target_name': 'Caudoputamen',
                 'projection_weight': 0.72, 'neuron_count': 189},
                {'source': 'MOp', 'target': 'SC', 'target_name': 'Superior colliculus',
                 'projection_weight': 0.56, 'neuron_count': 145},
            ]

        else:
            data = []

        return {
            'success': True,
            'data': data,
            'rows': len(data),
            'query': query
        }

    def close(self):
        pass


# ==================== Mock LLM Client ====================

class MockLLMClient(LLMClient):
    """
    模拟LLM客户端 - 用于测试

    提供智能的模拟响应
    """

    def __init__(self):
        self.call_count = 0

        # 知识库
        self.knowledge = {
            'VIP': 'vasoactive intestinal peptide',
            'SST': 'somatostatin',
            'Pvalb': 'parvalbumin',
            'Car3': 'carbonic anhydrase 3',
            'Lamp5': 'lysosomal associated membrane protein 5',
            'MOp': 'Primary motor area',
            'MOs': 'Secondary motor area',
            'SSp': 'Primary somatosensory area',
            'VISp': 'Primary visual area',
        }

    def chat(self,
             messages: List[Dict],
             temperature: float = 0.2,
             max_tokens: int = 2000,
             json_mode: bool = False) -> str:

        self.call_count += 1

        # 获取用户消息
        user_msg = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
                break

        user_lower = user_msg.lower()

        # 意图分类
        if 'analyze this neuroscience question' in user_lower:
            return self._mock_intent_response(user_msg)

        # 实体提取
        elif 'extract entities' in user_lower:
            return self._mock_entity_response(user_msg)

        # 步骤排序
        elif 'rank' in user_lower and 'steps' in user_lower:
            return self._mock_ranking_response(user_msg)

        # 反思
        elif 'reflect' in user_lower:
            return self._mock_reflection_response(user_msg)

        # 综合答案
        elif 'synthesize' in user_lower:
            return self._mock_synthesis_response(user_msg)

        else:
            return "I understand the request."

    def _mock_intent_response(self, msg: str) -> str:
        """模拟意图分类响应"""
        msg_lower = msg.lower()

        # 判断意图
        if any(w in msg_lower for w in ['stand for', 'full name', 'abbreviation', 'what is']):
            intent = 'definition'
            depth = 'shallow'
            planner = 'adaptive'
        elif any(w in msg_lower for w in ['compare', 'versus', 'vs']):
            intent = 'comparison'
            depth = 'medium'
            planner = 'comparative'
        elif any(w in msg_lower for w in ['tell me about', 'comprehensive', 'analyze', 'profile']):
            intent = 'profiling'
            depth = 'deep'
            planner = 'focus_driven'
        elif any(w in msg_lower for w in ['which', 'top', 'highest', 'screen']):
            intent = 'screening'
            depth = 'medium'
            planner = 'comparative'
        else:
            intent = 'profiling'
            depth = 'medium'
            planner = 'adaptive'

        return json.dumps({
            'intent': intent,
            'intent_confidence': 0.9,
            'intent_reasoning': f'Detected {intent} intent',
            'recommended_depth': depth,
            'depth_reasoning': f'{depth} depth appropriate',
            'recommended_planner': planner,
            'planner_reasoning': f'{planner} is optimal',
            'expected_modalities': ['molecular', 'morphological', 'projection'],
            'modality_reasoning': 'Multi-modal analysis needed',
            'expected_entity_types': ['GeneMarker', 'Region'],
            'key_concepts': ['gene expression', 'cell types'],
            'sub_questions': ['What cell types express this gene?', 'Where are they located?'],
            'analysis_goals': ['Identify cell types', 'Map spatial distribution']
        })

    def _mock_entity_response(self, msg: str) -> str:
        """模拟实体提取响应"""
        entities = []

        for gene, full_name in self.knowledge.items():
            if gene.lower() in msg.lower():
                if gene in ['VIP', 'SST', 'Pvalb', 'Car3', 'Lamp5']:
                    entities.append({
                        'text': gene,
                        'type': 'GeneMarker',
                        'confidence': 0.95
                    })
                else:
                    entities.append({
                        'text': gene,
                        'type': 'Region',
                        'confidence': 0.95
                    })

        return json.dumps({'entities': entities})

    def _mock_ranking_response(self, msg: str) -> str:
        """模拟步骤排序响应"""
        # 简单返回按顺序排序
        return json.dumps({
            'ranked_steps': [
                {'index': 0, 'score': 0.95, 'reasoning': 'High priority step'},
                {'index': 1, 'score': 0.85, 'reasoning': 'Secondary step'},
            ]
        })

    def _mock_reflection_response(self, msg: str) -> str:
        """模拟反思响应"""
        return json.dumps({
            'validation_status': 'passed',
            'validation_reasoning': 'Results match expectations',
            'key_findings': ['Found relevant data', 'Cell types identified'],
            'surprising_results': [],
            'uncertainty_level': 0.2,
            'uncertainty_sources': ['Limited sample size'],
            'decision': 'continue',
            'decision_reasoning': 'Analysis progressing well',
            'next_step_suggestions': ['Proceed with morphology analysis'],
            'alternative_approaches': [],
            'confidence_score': 0.85,
            'confidence_factors': {'data_quality': 0.9, 'expectation_match': 0.8},
            'summary': 'Step completed successfully with meaningful results.'
        })

    def _mock_synthesis_response(self, msg: str) -> str:
        """模拟综合答案"""
        # 从消息中提取问题
        question_match = msg.split('**Original Question:**')
        if len(question_match) > 1:
            question = question_match[1].split('\n')[0].strip()
        else:
            question = "the query"

        return f"""## Analysis Results

Based on the comprehensive multi-modal analysis, here are the key findings:

### Main Finding
The analysis successfully identified relevant cell populations and their characteristics across molecular, morphological, and projection modalities.

### Supporting Evidence
1. **Molecular**: Multiple cell clusters were identified expressing the target markers
2. **Morphological**: Neurons show characteristic axonal and dendritic patterns
3. **Projection**: Clear connectivity patterns to subcortical targets were mapped

### Multi-Modal Integration
The molecular identity correlates with distinct morphological features and projection patterns, suggesting functional specialization.

### Limitations
- Analysis based on available data in the knowledge graph
- Some regions may have limited morphological data

*Analysis completed with high confidence.*"""


# ==================== Main Agent Class ====================

class NeuroXivAgent:
    """
    NeuroXiv-KG Agent - 无比强大的自动神经数据分析Agent

    核心能力：
    1. LLM深度参与每个决策阶段
    2. 动态自适应的分析策略
    3. 三模态整合分析
    4. 完整的TPAR循环
    5. 智能闭环分析
    """

    def __init__(self,
                 db_executor,
                 llm_client: LLMClient,
                 schema: SchemaGraph = None,
                 config: AgentConfig = None):
        """
        初始化Agent

        Args:
            db_executor: 数据库执行器
            llm_client: LLM客户端
            schema: Schema图
            config: 配置
        """
        self.db = db_executor
        self.llm = llm_client
        self.schema = schema or SchemaGraph()
        self.config = config or AgentConfig()

        # 初始化TPAR引擎
        self.tpar_engine = TPAREngine(
            db_executor=db_executor,
            llm_client=llm_client,
            schema=schema,
            config=config
        )

        # 会话记忆
        self.session_memory = SessionMemory()

        logger.info("🚀 NeuroXiv-KG Agent initialized")
        logger.info(f"   • LLM: {type(llm_client).__name__}")
        logger.info(f"   • DB: {type(db_executor).__name__}")

    @classmethod
    def create(cls,
               neo4j_uri: str = None,
               neo4j_user: str = None,
               neo4j_password: str = None,
               neo4j_database: str = "neo4j",
               openai_api_key: str = None,
               model: str = "gpt-4o",
               schema_path: str = None,
               use_mock: bool = False) -> 'NeuroXivAgent':
        """
        工厂方法：创建Agent实例

        Args:
            neo4j_uri: Neo4j URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: 数据库名
            openai_api_key: OpenAI API Key
            model: LLM模型
            schema_path: Schema JSON路径
            use_mock: 是否使用模拟模式

        Returns:
            NeuroXivAgent实例
        """
        # 数据库
        if use_mock:
            db_executor = MockExecutor()
            llm_client = MockLLMClient()
        else:
            # Neo4j
            uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
            password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

            db_executor = Neo4jExecutor(uri, user, password, neo4j_database)

            # LLM
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                llm_client = OpenAIClient(client, model)
            else:
                logger.warning("No OpenAI API key, using mock LLM")
                llm_client = MockLLMClient()

        # Schema
        schema = SchemaGraph()
        if schema_path and os.path.exists(schema_path):
            with open(schema_path) as f:
                schema_data = json.load(f)
            schema = SchemaGraph(schema_data)

        # Config
        config = AgentConfig(
            neo4j_uri=neo4j_uri or "",
            neo4j_user=neo4j_user or "",
            neo4j_password=neo4j_password or "",
            neo4j_database=neo4j_database,
            llm_model=model,
        )

        return cls(db_executor, llm_client, schema, config)

    def answer(self,
               question: str,
               max_iterations: int = None) -> Dict[str, Any]:
        """
        回答问题 - 主入口

        Args:
            question: 用户问题
            max_iterations: 最大迭代次数

        Returns:
            包含答案和分析详情的字典
        """
        return self.tpar_engine.answer(question, max_iterations)

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
        """清除会话"""
        self.session_memory = SessionMemory()
        self.tpar_engine.session_memory = SessionMemory()
        logger.info("Session cleared")

    def close(self):
        """关闭连接"""
        self.db.close()
        logger.info("Connections closed")


# ==================== Quick Test ====================

def quick_test():
    """快速测试"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 70)
    print("🧪 NeuroXiv-KG Agent Quick Test")
    print("=" * 70)

    # 创建模拟Agent
    agent = NeuroXivAgent.create(use_mock=True)

    # 测试问题
    test_questions = [
        "What does VIP stand for?",
        "Tell me about Car3+ neurons",
        # "Compare MOp and MOs regions",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print('=' * 60)

        result = agent.answer(question, max_iterations=5)

        print(f"\n✅ Results:")
        print(f"   Steps: {result.get('total_steps', 0)}")
        print(f"   Confidence: {result.get('confidence_score', 0):.3f}")
        print(f"   Time: {result.get('execution_time', 0):.2f}s")

        if 'analysis_info' in result:
            info = result['analysis_info']
            print(f"   Intent: {info.get('intent', 'unknown')}")
            print(f"   Depth: {info.get('target_depth', 'unknown')}")
            print(f"   Modalities: {info.get('modalities_covered', [])}")

        print(f"\n💡 Answer:\n{result.get('answer', 'No answer')[:500]}...")

    agent.close()
    print("\n✅ Quick test complete!")


# ==================== Export ====================

__all__ = [
    'NeuroXivAgent',
    'Neo4jExecutor',
    'MockExecutor',
    'MockLLMClient',
    'quick_test',
]


if __name__ == "__main__":
    quick_test()