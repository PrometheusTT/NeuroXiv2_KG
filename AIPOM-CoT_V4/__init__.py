"""
NeuroXiv-KG Agent
==================
无比强大的自动神经数据分析Agent

特性：
1. LLM深度参与 - 真正的推理而非模式匹配
2. 高度灵活 - 动态适应不同问题类型
3. 多模态整合 - 分子/形态/投射三模态分析
4. 闭环分析 - 完整的circuit分析
5. TPAR循环 - Think-Plan-Act-Reflect结构化推理

使用方式：

    from neuroxiv_agent import NeuroXivAgent

    # 创建Agent
    agent = NeuroXivAgent.create(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="sk-..."
    )

    # 回答问题
    result = agent.answer("Tell me about Car3+ neurons")
    print(result['answer'])

    # 快速测试（使用Mock）
    from neuroxiv_agent import quick_test
    quick_test()

Author: Lijun
Date: 2025-01
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Lijun'

# Core structures
from .core_structures import (
    # Enums
    Modality,
    AnalysisDepth,
    QuestionIntent,
    PlannerType,
    ReflectionDecision,
    ValidationStatus,

    # Entity
    Entity,
    EntityCluster,

    # Schema
    SchemaPath,
    CandidateStep,

    # Evidence
    StatisticalEvidence,
    EvidenceRecord,
    EvidenceBuffer,

    # State
    AnalysisState,

    # Reflection
    StructuredReflection,

    # Memory
    SessionMemory,

    # Config
    AgentConfig,
)

# LLM Intelligence
from .llm_intelligence import (
    LLMClient,
    OpenAIClient,
    IntentClassification,
    LLMIntentClassifier,
    LLMEntityRecognizer,
    LLMStrategyRecommender,
)

# Planner
from .adaptive_planner import (
    SchemaGraph,
    CandidateStepGenerator,
    LLMStepRanker,
    AdaptivePlanner,
)

# Reflector
from .llm_reflector import (
    LLMReflector,
    RuleBasedReflector,
    ReflectionAggregator,
)

# TPAR Engine
from .tpar_engine import TPAREngine

# Agent
from .agent import (
    NeuroXivAgent,
    Neo4jExecutor,
    MockExecutor,
    MockLLMClient,
    quick_test,
)

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Enums
    'Modality',
    'AnalysisDepth',
    'QuestionIntent',
    'PlannerType',
    'ReflectionDecision',
    'ValidationStatus',

    # Core structures
    'Entity',
    'EntityCluster',
    'SchemaPath',
    'CandidateStep',
    'StatisticalEvidence',
    'EvidenceRecord',
    'EvidenceBuffer',
    'AnalysisState',
    'StructuredReflection',
    'SessionMemory',
    'AgentConfig',

    # LLM
    'LLMClient',
    'OpenAIClient',
    'IntentClassification',
    'LLMIntentClassifier',
    'LLMEntityRecognizer',
    'LLMStrategyRecommender',

    # Planner
    'SchemaGraph',
    'CandidateStepGenerator',
    'LLMStepRanker',
    'AdaptivePlanner',

    # Reflector
    'LLMReflector',
    'RuleBasedReflector',
    'ReflectionAggregator',

    # Engine
    'TPAREngine',

    # Agent
    'NeuroXivAgent',
    'Neo4jExecutor',
    'MockExecutor',
    'MockLLMClient',
    'quick_test',
]