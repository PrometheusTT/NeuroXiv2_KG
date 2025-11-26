"""
AIPOM-CoT V11
=============
Schema-Adaptive, Evidence-Seeking Agent for Neuroscience Knowledge Discovery

Modules:
- core_structures: 统一的数据结构
- intent_classifier: LLM驱动的意图分类
- llm_reflector: LLM驱动的结构化反思
- multimodal_analyzer: 统一的多模态分析器
- tpar_engine: 核心TPAR推理引擎
- termination_decider: 智能终止决策器
- aipom_v11_agent: 主Agent类

Usage:
    from aipom_v11 import create_agent

    agent = create_agent()
    result = agent.answer("Tell me about Car3+ neurons")
    print(result['answer'])

Author: Claude & Lijun
Date: 2025-01-15
"""

from .core_structures import (
    # Enums
    AnalysisDepth,
    Modality,
    PlannerType,
    QuestionIntent,
    ValidationStatus,
    ReflectionDecision,

    # Data classes
    StatisticalEvidence,
    EvidenceRecord,
    EvidenceBuffer,
    AnalysisState,
    CandidateStep,
    ReasoningStep,
    StructuredReflection,
    SessionMemory
)

from .intent_classifier import (
    LLMIntentClassifier,
    PlannerRouter,
    IntentClassification
)

from .llm_reflector import (
    LLMReflector,
    RuleBasedReflector,
    ReflectionAggregator
)

from .multimodal_analyzer import (
    UnifiedFingerprintAnalyzer,
    StatisticalToolkit,
    RegionFingerprint,
    MismatchResult
)

from .tpar_engine import TPAREngine

from .termination_decider import (
    IntelligentTerminator,
    GoalTracker,
    TerminationDecision,
    TerminationReason
)

from .aipom_v11_agent import (
    AIPOMCoTV11,
    create_agent
)

__version__ = "11.0.0"
__author__ = "Claude & Lijun"

__all__ = [
    # Main
    'AIPOMCoTV11',
    'create_agent',

    # Enums
    'AnalysisDepth',
    'Modality',
    'PlannerType',
    'QuestionIntent',
    'ValidationStatus',
    'ReflectionDecision',

    # Core structures
    'AnalysisState',
    'EvidenceBuffer',
    'EvidenceRecord',
    'StatisticalEvidence',
    'ReasoningStep',
    'CandidateStep',
    'StructuredReflection',
    'SessionMemory',

    # Intent
    'LLMIntentClassifier',
    'IntentClassification',
    'PlannerRouter',

    # Reflection
    'LLMReflector',
    'ReflectionAggregator',

    # Analysis
    'UnifiedFingerprintAnalyzer',
    'StatisticalToolkit',
    'RegionFingerprint',
    'MismatchResult',

    # Engine
    'TPAREngine',

    # Termination
    'IntelligentTerminator',
    'GoalTracker',
    'TerminationDecision',
]