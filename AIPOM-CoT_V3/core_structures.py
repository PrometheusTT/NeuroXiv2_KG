"""
AIPOM-CoT V11 Core Data Structures
===================================
统一的数据结构定义，解决代码重复和类型不一致问题

对齐Figure 2的所有组件：
- AnalysisState (包含budget, paths_used等)
- EvidenceRecord (effect size, CI, FDR q, snapshot ID, query hash)
- EvidenceBuffer (完整的证据管理)
- ReasoningStep (增强版)

Author: Claude & Lijun
Date: 2025-01-15
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ==================== Enums ====================

class AnalysisDepth(Enum):
    """分析深度 - 对齐Figure 2"""
    SHALLOW = "shallow"  # 1-2步: 简单查询
    MEDIUM = "medium"  # 3-4步: 标准多模态
    DEEP = "deep"  # 5-8步: 完整闭环分析


class Modality(Enum):
    """数据模态 - 对齐Figure 2D"""
    MOLECULAR = "molecular"
    MORPHOLOGICAL = "morphological"
    PROJECTION = "projection"
    SPATIAL = "spatial"
    STATISTICAL = "statistical"


class PlannerType(Enum):
    """规划器类型 - 对齐Figure 2A"""
    FOCUS_DRIVEN = "focus_driven"  # 深度profiling
    COMPARATIVE = "comparative"  # 比较/筛选
    ADAPTIVE = "adaptive"  # 自适应/简单QA


class QuestionIntent(Enum):
    """问题意图分类 - 对齐Figure 2A IntentClassifier"""
    SIMPLE_QUERY = "simple_query"  # "What is X?"
    DEEP_PROFILING = "deep_profiling"  # "Tell me about X"
    COMPARISON = "comparison"  # "Compare A and B"
    SCREENING = "screening"  # "Which regions show..."
    EXPLANATION = "explanation"  # "Why does X..."
    UNKNOWN = "unknown"


class ValidationStatus(Enum):
    """验证状态"""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"
    UNEXPECTED = "unexpected"


class ReflectionDecision(Enum):
    """反思决策 - LLM反思后的决定"""
    CONTINUE = "continue"  # 继续执行计划
    REPLAN = "replan"  # 需要重新规划
    DEEPEN = "deepen"  # 加深当前分析
    PIVOT = "pivot"  # 转向新方向
    TERMINATE = "terminate"  # 达到目标，终止


# ==================== Evidence Records ====================

@dataclass
class StatisticalEvidence:
    """
    统计证据 - 对齐Figure 2C Evidence Buffer

    包含Figure 2中显示的所有字段：
    - effect size
    - 95% CI
    - perm p
    - FDR q
    - n (sample size)
    """
    test_type: str  # "permutation", "t_test", "fdr", etc.
    effect_size: Optional[float] = None  # Cohen's d or similar
    confidence_interval: Optional[Tuple[float, float]] = None  # 95% CI
    p_value: Optional[float] = None  # raw p-value
    fdr_q: Optional[float] = None  # FDR-corrected q-value
    sample_size: int = 0  # n
    is_significant: bool = False  # significance decision

    def to_dict(self) -> Dict:
        return {
            'test_type': self.test_type,
            'effect_size': self.effect_size,
            'ci_lower': self.confidence_interval[0] if self.confidence_interval else None,
            'ci_upper': self.confidence_interval[1] if self.confidence_interval else None,
            'p_value': self.p_value,
            'fdr_q': self.fdr_q,
            'n': self.sample_size,
            'significant': self.is_significant
        }


@dataclass
class EvidenceRecord:
    """
    单条证据记录 - 对齐Figure 2C Evidence Buffer的完整结构

    每个执行步骤产生一条EvidenceRecord
    """
    # === 标识 ===
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    query_hash: str = ""  # Cypher查询的hash
    snapshot_id: str = ""  # 数据快照ID (用于复现)

    # === 时间 ===
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0

    # === 数据质量 ===
    data_completeness: float = 0.0  # 0-1: 非null比例
    row_count: int = 0
    column_count: int = 0

    # === 统计证据 ===
    statistical_evidence: Optional[StatisticalEvidence] = None

    # === 模态信息 ===
    modality: Optional[Modality] = None

    # === 原始数据引用 ===
    raw_data_key: str = ""  # 指向intermediate_data的key

    # === 验证 ===
    validation_status: ValidationStatus = ValidationStatus.PASSED
    confidence_score: float = 0.0  # 0-1

    @staticmethod
    def compute_query_hash(query: str, params: Dict) -> str:
        """计算查询的hash用于去重和复现"""
        content = query + str(sorted(params.items()))
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'step_number': self.step_number,
            'query_hash': self.query_hash,
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'execution_time': self.execution_time,
            'data_completeness': self.data_completeness,
            'row_count': self.row_count,
            'modality': self.modality.value if self.modality else None,
            'validation_status': self.validation_status.value,
            'confidence_score': self.confidence_score,
            'statistical_evidence': self.statistical_evidence.to_dict() if self.statistical_evidence else None
        }


class EvidenceBuffer:
    """
    证据缓冲区 - 对齐Figure 2C

    管理所有收集的证据，提供：
    - 按模态查询
    - 按步骤查询
    - 统计汇总
    - 置信度计算
    """

    def __init__(self):
        self.records: List[EvidenceRecord] = []
        self._by_modality: Dict[Modality, List[EvidenceRecord]] = {}
        self._by_step: Dict[int, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord):
        """添加证据记录"""
        self.records.append(record)

        # 索引by modality
        if record.modality:
            if record.modality not in self._by_modality:
                self._by_modality[record.modality] = []
            self._by_modality[record.modality].append(record)

        # 索引by step
        self._by_step[record.step_number] = record

    def get_by_modality(self, modality: Modality) -> List[EvidenceRecord]:
        """获取特定模态的所有证据"""
        return self._by_modality.get(modality, [])

    def get_by_step(self, step_number: int) -> Optional[EvidenceRecord]:
        """获取特定步骤的证据"""
        return self._by_step.get(step_number)

    def get_overall_confidence(self) -> float:
        """
        计算整体置信度 - 对齐Figure 2C显示的95%

        综合考虑：
        - 各步骤置信度
        - 模态覆盖度
        - 统计显著性
        """
        if not self.records:
            return 0.0

        # Factor 1: 平均步骤置信度
        avg_confidence = np.mean([r.confidence_score for r in self.records])

        # Factor 2: 模态覆盖度 (3种模态全覆盖 = 1.0)
        covered_modalities = len(self._by_modality)
        modality_coverage = min(1.0, covered_modalities / 3.0)

        # Factor 3: 统计验证比例
        records_with_stats = [r for r in self.records if r.statistical_evidence]
        if records_with_stats:
            significant_ratio = sum(
                1 for r in records_with_stats
                if r.statistical_evidence.is_significant
            ) / len(records_with_stats)
        else:
            significant_ratio = 0.5  # 默认中等

        # 加权组合
        overall = (
                0.5 * avg_confidence +
                0.3 * modality_coverage +
                0.2 * significant_ratio
        )

        return min(1.0, max(0.0, overall))

    def get_data_completeness(self) -> float:
        """获取整体数据完整性"""
        if not self.records:
            return 0.0
        return np.mean([r.data_completeness for r in self.records])

    def get_evidence_strength(self) -> float:
        """获取证据强度（基于统计显著性）"""
        records_with_stats = [r for r in self.records if r.statistical_evidence]
        if not records_with_stats:
            return 0.5

        significant = sum(
            1 for r in records_with_stats
            if r.statistical_evidence.is_significant
        )
        return significant / len(records_with_stats)

    def summarize(self) -> Dict:
        """生成摘要 - 用于Figure 2C的可视化"""
        return {
            'total_records': len(self.records),
            'modalities_covered': list(self._by_modality.keys()),
            'data_completeness': self.get_data_completeness(),
            'evidence_strength': self.get_evidence_strength(),
            'confidence_score': self.get_overall_confidence(),
            'records': [r.to_dict() for r in self.records]
        }


# ==================== Analysis State ====================

@dataclass
class AnalysisState:
    """
    分析状态 - 对齐Figure 2C AnalysisState

    增强版包含Figure 2中显示的所有字段：
    - modalities
    - regions
    - paths_used
    - budget
    """
    # === 基础信息 ===
    question: str = ""
    question_intent: QuestionIntent = QuestionIntent.UNKNOWN
    target_depth: AnalysisDepth = AnalysisDepth.MEDIUM

    # === 实体发现 ===
    discovered_entities: Dict[str, List[str]] = field(default_factory=dict)
    primary_focus: Optional[Any] = None  # FocusEntity

    # === 执行追踪 ===
    executed_steps: List[Dict] = field(default_factory=list)
    current_step: int = 0

    # === 模态覆盖 - 对齐Figure 2C ===
    modalities_covered: List[Modality] = field(default_factory=list)

    # === Schema路径 - 对齐Figure 2C ===
    paths_used: List[Dict] = field(default_factory=list)

    # === 预算控制 - 对齐Figure 2C ===
    budget: Dict[str, Any] = field(default_factory=lambda: {
        'max_steps': 8,
        'max_cypher_calls': 20,
        'max_llm_calls': 15,
        'current_cypher_calls': 0,
        'current_llm_calls': 0,
        'time_limit_seconds': 300,
        'start_time': time.time()
    })

    # === 证据缓冲 ===
    evidence_buffer: EvidenceBuffer = field(default_factory=EvidenceBuffer)

    # === 中间数据 ===
    intermediate_data: Dict[str, Any] = field(default_factory=dict)

    # === 反思记录 ===
    reflections: List[Dict] = field(default_factory=list)

    # === 重规划 ===
    replanning_count: int = 0
    max_replanning: int = 2

    def add_modality(self, modality: Modality):
        """添加已覆盖的模态"""
        if modality not in self.modalities_covered:
            self.modalities_covered.append(modality)

    def add_path(self, path_info: Dict):
        """记录使用的schema路径"""
        self.paths_used.append(path_info)

    def increment_budget(self, resource: str):
        """增加资源使用计数"""
        if resource == 'cypher':
            self.budget['current_cypher_calls'] += 1
        elif resource == 'llm':
            self.budget['current_llm_calls'] += 1

    def check_budget(self) -> Dict[str, bool]:
        """检查预算状态"""
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
        """获取进度摘要 - 用于LLM决策"""
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


# ==================== Reasoning Steps ====================

@dataclass
class CandidateStep:
    """候选步骤 - 对齐Figure 2A的recipe输出"""
    step_id: str
    step_type: Modality
    purpose: str
    rationale: str
    priority: float  # 基础优先级 0-10
    schema_path: str
    expected_data: str
    cypher_template: str
    parameters: Dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    # LLM评估填充
    llm_score: float = 0.0
    llm_reasoning: str = ""


@dataclass
class ReasoningStep:
    """执行步骤"""
    step_number: int
    purpose: str
    action: str  # "execute_cypher", "execute_statistical", etc.
    rationale: str
    expected_result: str
    query_or_params: Dict = field(default_factory=dict)
    modality: Optional[Modality] = None
    depends_on: List[int] = field(default_factory=list)

    # 执行结果
    actual_result: Optional[Dict] = None
    evidence_record: Optional[EvidenceRecord] = None
    reflection: Optional[Dict] = None
    validation_passed: bool = False
    execution_time: float = 0.0


# ==================== Reflection Structures ====================

@dataclass
class StructuredReflection:
    """
    结构化反思结果 - 对齐Figure 2C Reflect phase

    由LLM生成，不是纯规则
    """
    step_number: int

    # 验证结果
    validation_status: ValidationStatus
    validation_reasoning: str

    # 不确定性评估
    uncertainty_level: float  # 0-1
    uncertainty_sources: List[str]  # 不确定性来源

    # 发现与洞察
    key_findings: List[str]
    surprising_results: List[str]

    # 决策
    decision: ReflectionDecision
    decision_reasoning: str

    # 下一步建议
    next_step_suggestions: List[str]
    alternative_approaches: List[str]

    # 置信度
    confidence_score: float
    confidence_factors: Dict[str, float]

    # 生成的摘要
    summary: str


# ==================== Session Memory ====================

@dataclass
class SessionMemory:
    """
    会话记忆 - 支持跨问题的知识积累

    这是对Figure 2的扩展，支持更复杂的对话
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # 之前的问题和答案
    qa_history: List[Dict] = field(default_factory=list)

    # 累积发现的实体
    known_entities: Dict[str, List[str]] = field(default_factory=dict)

    # 用户偏好
    preferences: Dict[str, Any] = field(default_factory=dict)

    def add_qa(self, question: str, answer: str, entities: Dict):
        """记录问答"""
        self.qa_history.append({
            'question': question,
            'answer': answer[:500],  # 截断
            'entities': entities,
            'timestamp': time.time()
        })

        # 更新已知实体
        for entity_type, entity_list in entities.items():
            if entity_type not in self.known_entities:
                self.known_entities[entity_type] = []
            for e in entity_list:
                if e not in self.known_entities[entity_type]:
                    self.known_entities[entity_type].append(e)

    def get_relevant_context(self, question: str) -> str:
        """获取与当前问题相关的历史上下文"""
        if not self.qa_history:
            return ""

        # 简单实现：返回最近3个QA
        recent = self.qa_history[-3:]
        context_parts = []
        for qa in recent:
            context_parts.append(
                f"Previous Q: {qa['question']}\n"
                f"Previous entities found: {list(qa['entities'].keys())}"
            )

        return "\n\n".join(context_parts)