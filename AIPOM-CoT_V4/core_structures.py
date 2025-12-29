"""
NeuroXiv-KG Agent Core Structures
==================================
统一的数据结构定义，支持完整的TPAR循环

核心设计原则：
1. 类型安全 - 所有状态有明确类型
2. 可追溯 - 每个决策都有reasoning
3. 证据驱动 - Evidence Buffer管理所有发现
4. 预算控制 - 资源使用有上限

Author: Lijun
Date: 2025-01
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np


# ==================== Enums ====================

class Modality(Enum):
    """数据模态"""
    MOLECULAR = "molecular"          # 分子/转录组
    MORPHOLOGICAL = "morphological"  # 形态学
    PROJECTION = "projection"        # 投射连接
    SPATIAL = "spatial"              # 空间分布
    STATISTICAL = "statistical"      # 统计分析


class AnalysisDepth(Enum):
    """分析深度"""
    SHALLOW = "shallow"   # 1-2步: 简单查询
    MEDIUM = "medium"     # 3-4步: 标准多模态
    DEEP = "deep"         # 5-8步: 完整闭环


class QuestionIntent(Enum):
    """问题意图"""
    DEFINITION = "definition"           # "What is X?" / "What does X stand for?"
    PROFILING = "profiling"             # "Tell me about X"
    COMPARISON = "comparison"           # "Compare A and B"
    SCREENING = "screening"             # "Which regions show..."
    CONNECTIVITY = "connectivity"       # "Where does X project to?"
    COMPOSITION = "composition"         # "What cell types are in X?"
    QUANTIFICATION = "quantification"   # "How many..."
    MECHANISM = "mechanism"             # "Why does X..."
    UNKNOWN = "unknown"


class PlannerType(Enum):
    """规划器类型"""
    FOCUS_DRIVEN = "focus_driven"    # 深度剖析单一实体
    COMPARATIVE = "comparative"       # 系统对比/筛选
    ADAPTIVE = "adaptive"            # 自适应探索


class ReflectionDecision(Enum):
    """反思决策"""
    CONTINUE = "continue"      # 继续执行
    DEEPEN = "deepen"          # 加深分析
    PIVOT = "pivot"            # 转向新方向
    REPLAN = "replan"          # 重新规划
    TERMINATE = "terminate"    # 完成终止


class ValidationStatus(Enum):
    """验证状态"""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"
    UNEXPECTED = "unexpected"


# ==================== Entity Structures ====================

@dataclass
class Entity:
    """识别的实体"""
    name: str
    entity_type: str  # 'GeneMarker', 'Region', 'CellType', 'Cluster'
    canonical_name: str = ""
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.name


@dataclass
class EntityCluster:
    """实体聚类"""
    primary: Entity
    related: List[Entity] = field(default_factory=list)
    cluster_type: str = "unknown"
    relevance_score: float = 0.0


# ==================== Schema Structures ====================

@dataclass
class SchemaPath:
    """Schema中的查询路径"""
    path_id: str
    start_label: str
    end_label: str
    hops: List[Tuple[str, str, str]]  # [(source, rel_type, target), ...]
    score: float = 0.0
    description: str = ""


@dataclass
class CandidateStep:
    """候选执行步骤"""
    step_id: str
    step_type: str  # modality string
    purpose: str
    rationale: str
    priority: float
    schema_path: Optional[SchemaPath] = None
    cypher_template: str = ""
    parameters: Dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    # LLM评估结果
    llm_score: float = 0.0
    llm_reasoning: str = ""


# ==================== Evidence Structures ====================

@dataclass
class StatisticalEvidence:
    """统计证据"""
    test_type: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    fdr_q: Optional[float] = None
    sample_size: int = 0
    is_significant: bool = False

    def to_dict(self) -> Dict:
        return {
            'test_type': self.test_type,
            'effect_size': self.effect_size,
            'ci': self.confidence_interval,
            'p_value': self.p_value,
            'fdr_q': self.fdr_q,
            'n': self.sample_size,
            'significant': self.is_significant
        }


@dataclass
class EvidenceRecord:
    """单条证据记录"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    query_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0

    # 数据质量
    data_completeness: float = 0.0
    row_count: int = 0
    column_count: int = 0

    # 模态和统计
    modality: Optional[Modality] = None
    statistical_evidence: Optional[StatisticalEvidence] = None

    # 验证
    validation_status: ValidationStatus = ValidationStatus.PASSED
    confidence_score: float = 0.0

    # 原始数据引用
    raw_data_key: str = ""

    @staticmethod
    def compute_query_hash(query: str, params: Dict) -> str:
        content = query + str(sorted(params.items()))
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'step': self.step_number,
            'query_hash': self.query_hash,
            'time': self.execution_time,
            'rows': self.row_count,
            'completeness': self.data_completeness,
            'modality': self.modality.value if self.modality else None,
            'confidence': self.confidence_score,
            'validation': self.validation_status.value
        }


class EvidenceBuffer:
    """证据缓冲区 - 管理所有收集的证据"""

    def __init__(self):
        self.records: List[EvidenceRecord] = []
        self._by_modality: Dict[Modality, List[EvidenceRecord]] = {}
        self._by_step: Dict[int, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord):
        self.records.append(record)

        if record.modality:
            if record.modality not in self._by_modality:
                self._by_modality[record.modality] = []
            self._by_modality[record.modality].append(record)

        self._by_step[record.step_number] = record

    def get_by_modality(self, modality: Modality) -> List[EvidenceRecord]:
        return self._by_modality.get(modality, [])

    def get_by_step(self, step_number: int) -> Optional[EvidenceRecord]:
        return self._by_step.get(step_number)

    def get_overall_confidence(self) -> float:
        if not self.records:
            return 0.0

        # 平均置信度
        avg_conf = np.mean([r.confidence_score for r in self.records])

        # 模态覆盖度
        modality_coverage = min(1.0, len(self._by_modality) / 3.0)

        # 统计验证比例
        with_stats = [r for r in self.records if r.statistical_evidence]
        if with_stats:
            sig_ratio = sum(1 for r in with_stats
                           if r.statistical_evidence.is_significant) / len(with_stats)
        else:
            sig_ratio = 0.5

        return min(1.0, 0.5 * avg_conf + 0.3 * modality_coverage + 0.2 * sig_ratio)

    def get_data_completeness(self) -> float:
        if not self.records:
            return 0.0
        return np.mean([r.data_completeness for r in self.records])

    def summarize(self) -> Dict:
        return {
            'total_records': len(self.records),
            'modalities_covered': [m.value for m in self._by_modality.keys()],
            'data_completeness': self.get_data_completeness(),
            'confidence_score': self.get_overall_confidence(),
            'records': [r.to_dict() for r in self.records]
        }


# ==================== Analysis State ====================

@dataclass
class AnalysisState:
    """
    分析状态 - TPAR循环的核心状态容器

    追踪：
    - 问题理解
    - 实体发现
    - 步骤执行
    - 证据积累
    - 预算使用
    """
    # 问题
    question: str = ""
    question_intent: QuestionIntent = QuestionIntent.UNKNOWN
    target_depth: AnalysisDepth = AnalysisDepth.MEDIUM

    # 实体发现
    discovered_entities: Dict[str, List[str]] = field(default_factory=dict)
    primary_focus: Optional[Entity] = None

    # 执行追踪
    executed_steps: List[Dict] = field(default_factory=list)
    current_step: int = 0

    # 模态覆盖
    modalities_covered: Set[Modality] = field(default_factory=set)

    # Schema路径使用记录
    paths_used: List[Dict] = field(default_factory=list)

    # 预算控制
    budget: Dict[str, Any] = field(default_factory=lambda: {
        'max_steps': 10,
        'max_cypher_calls': 25,
        'max_llm_calls': 20,
        'current_cypher_calls': 0,
        'current_llm_calls': 0,
        'time_limit_seconds': 300,
        'start_time': time.time()
    })

    # 证据缓冲
    evidence_buffer: EvidenceBuffer = field(default_factory=EvidenceBuffer)

    # 中间数据存储
    intermediate_data: Dict[str, Any] = field(default_factory=dict)

    # 反思记录
    reflections: List[Dict] = field(default_factory=list)

    # 重规划计数
    replanning_count: int = 0
    max_replanning: int = 3

    # 意图分类结果缓存
    _classification: Any = None

    def add_modality(self, modality: Modality):
        self.modalities_covered.add(modality)

    def add_path(self, path_info: Dict):
        self.paths_used.append(path_info)

    def increment_budget(self, resource: str):
        if resource == 'cypher':
            self.budget['current_cypher_calls'] += 1
        elif resource == 'llm':
            self.budget['current_llm_calls'] += 1

    def check_budget(self) -> Dict[str, bool]:
        elapsed = time.time() - self.budget['start_time']
        return {
            'steps_ok': len(self.executed_steps) < self.budget['max_steps'],
            'cypher_ok': self.budget['current_cypher_calls'] < self.budget['max_cypher_calls'],
            'llm_ok': self.budget['current_llm_calls'] < self.budget['max_llm_calls'],
            'time_ok': elapsed < self.budget['time_limit_seconds'],
            'can_continue': all([
                len(self.executed_steps) < self.budget['max_steps'],
                self.budget['current_cypher_calls'] < self.budget['max_cypher_calls'],
                elapsed < self.budget['time_limit_seconds']
            ])
        }

    def get_progress_summary(self) -> Dict:
        return {
            'question': self.question,
            'intent': self.question_intent.value,
            'target_depth': self.target_depth.value,
            'steps_executed': len(self.executed_steps),
            'modalities_covered': [m.value for m in self.modalities_covered],
            'entities_found': {k: len(v) for k, v in self.discovered_entities.items()},
            'has_primary_focus': self.primary_focus is not None,
            'evidence_confidence': self.evidence_buffer.get_overall_confidence(),
            'budget_status': self.check_budget(),
            'replanning_count': self.replanning_count
        }


# ==================== Reflection Structures ====================

@dataclass
class StructuredReflection:
    """结构化反思结果"""
    step_number: int

    # 验证
    validation_status: ValidationStatus
    validation_reasoning: str

    # 不确定性
    uncertainty_level: float
    uncertainty_sources: List[str]

    # 发现
    key_findings: List[str]
    surprising_results: List[str]

    # 决策
    decision: ReflectionDecision
    decision_reasoning: str

    # 建议
    next_step_suggestions: List[str]
    alternative_approaches: List[str]

    # 置信度
    confidence_score: float
    confidence_factors: Dict[str, float]

    # 摘要
    summary: str


# ==================== Session Memory ====================

@dataclass
class SessionMemory:
    """会话记忆 - 跨问题的知识积累"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    qa_history: List[Dict] = field(default_factory=list)
    known_entities: Dict[str, List[str]] = field(default_factory=dict)

    def add_qa(self, question: str, answer: str, entities: Dict):
        self.qa_history.append({
            'question': question,
            'answer': answer[:500],
            'entities': entities,
            'timestamp': time.time()
        })

        for entity_type, entity_list in entities.items():
            if entity_type not in self.known_entities:
                self.known_entities[entity_type] = []
            for e in entity_list:
                if e not in self.known_entities[entity_type]:
                    self.known_entities[entity_type].append(e)

    def get_relevant_context(self, question: str) -> str:
        if not self.qa_history:
            return ""

        recent = self.qa_history[-3:]
        parts = []
        for qa in recent:
            parts.append(f"Q: {qa['question']}\nEntities: {list(qa['entities'].keys())}")
        return "\n\n".join(parts)


# ==================== Agent Configuration ====================

@dataclass
class AgentConfig:
    """Agent配置"""
    # 数据库
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # LLM
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2

    # Schema
    schema_json_path: str = "./schema.json"

    # 预算
    max_iterations: int = 10
    max_cypher_calls: int = 25
    max_llm_calls: int = 20
    timeout_seconds: int = 300

    # 阈值
    confidence_threshold: float = 0.7
    min_evidence_count: int = 3


# ==================== Export ====================

__all__ = [
    # Enums
    'Modality', 'AnalysisDepth', 'QuestionIntent',
    'PlannerType', 'ReflectionDecision', 'ValidationStatus',

    # Entity
    'Entity', 'EntityCluster',

    # Schema
    'SchemaPath', 'CandidateStep',

    # Evidence
    'StatisticalEvidence', 'EvidenceRecord', 'EvidenceBuffer',

    # State
    'AnalysisState',

    # Reflection
    'StructuredReflection',

    # Memory
    'SessionMemory',

    # Config
    'AgentConfig',
]