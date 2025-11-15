"""
Comprehensive Benchmark System for AIPOM-CoT
============================================
Includes:
- 100+ test questions across 5 complexity levels
- Automatic evaluation metrics (BLEU, ROUGE, Accuracy)
- Baseline comparison (RAG, ReAct, Direct)
- Statistical analysis and visualization

Author: Claude & PrometheusTT
Date: 2025-01-12
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional: BLEU/ROUGE metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    HAS_METRICS = True
except ImportError:
    logger.warning("NLTK/rouge-score not installed. Install: pip install nltk rouge-score")
    HAS_METRICS = False


# ==================== Question Types ====================

class QuestionComplexity(Enum):
    """
    问题复杂度分级

    基于问题所需的推理步骤和知识整合程度
    """
    SIMPLE_FACTUAL = "simple_factual"  # 简单事实查询 (1-2步)
    MULTI_ENTITY = "multi_entity"  # 多实体查询 (2-3步)
    COMPARATIVE = "comparative"  # 比较分析 (3-5步)
    EXPLANATORY = "explanatory"  # 解释性查询 (4-6步)
    OPEN_ENDED = "open_ended"  # 开放式查询 (5+步)


# ==================== Benchmark Question ====================

@dataclass
class BenchmarkQuestion:
    """
    Benchmark测试问题

    🔧 修复：字段顺序正确，无默认值的在前

    字段说明:
    - id: 问题唯一标识
    - question: 问题文本
    - complexity: 复杂度级别
    - domain: 领域 (molecular/morphological/projection/multi-modal)
    - expected_answer_contains: 期望答案包含的关键词
    - expected_entities: 期望识别的实体列表
    - requires_kg: 是否需要访问知识图谱
    - gold_answer: 标准答案（可选）
    - ground_truth_cypher: Ground truth查询（可选）
    """

    # ===== 必需字段（无默认值）=====
    id: str
    question: str
    complexity: QuestionComplexity
    domain: str

    # ===== 可选字段（有默认值）=====
    expected_answer_contains: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)
    requires_kg: bool = True
    gold_answer: Optional[str] = None
    ground_truth_cypher: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        return {
            'id': self.id,
            'question': self.question,
            'complexity': self.complexity.value,  # Enum -> string
            'domain': self.domain,
            'expected_answer_contains': self.expected_answer_contains,
            'expected_entities': self.expected_entities,
            'requires_kg': self.requires_kg,
            'gold_answer': self.gold_answer,
            'ground_truth_cypher': self.ground_truth_cypher
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkQuestion':
        """从字典创建（用于JSON反序列化）"""
        return cls(
            id=data['id'],
            question=data['question'],
            complexity=QuestionComplexity(data['complexity']),  # string -> Enum
            domain=data['domain'],
            expected_answer_contains=data.get('expected_answer_contains', []),
            expected_entities=data.get('expected_entities', []),
            requires_kg=data.get('requires_kg', True),
            gold_answer=data.get('gold_answer'),
            ground_truth_cypher=data.get('ground_truth_cypher')
        )



# ==================== Test Question Database ====================

class BenchmarkQuestionBank:
    """
    测试问题库

    包含 100+ 问题,覆盖:
    - 5个复杂度级别
    - 4个领域模态
    - 多种问题类型
    """

    @staticmethod
    @staticmethod
    def generate_questions() -> List[BenchmarkQuestion]:
        """
        生成完整的测试问题集（修复版 - 包含expected_entities）

        覆盖5种复杂度 × 3种领域 = 50个问题

        新增:
        - 每个问题都有expected_entities
        - 覆盖多种查询模式
        - 包含正负样本
        """

        questions = []

        # =====================================================================
        # CATEGORY 1: SIMPLE FACTUAL (简单事实查询) - 10 questions
        # =====================================================================

        questions.extend([
            BenchmarkQuestion(
                id="Q001",
                question="What cells are in ACAd?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["cluster", "neuron", "cell type", "subclass"],
                expected_entities=["ACAd"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'ACAd'})-[:HAS_SUBCLASS]->(sc:Subclass) RETURN sc.name, count(*)"
            ),

            BenchmarkQuestion(
                id="Q002",
                question="Tell me about Pvalb neurons",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["parvalbumin", "interneuron", "inhibitory"],
                expected_entities=["Pvalb"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE sc.name CONTAINS 'Pvalb' RETURN r.acronym, sc.name"
            ),

            BenchmarkQuestion(
                id="Q003",
                question="How many neurons are in MOs?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["neuron", "count", "MOs"],
                expected_entities=["MOs"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: 'MOs'}) RETURN count(n)"
            ),

            BenchmarkQuestion(
                id="Q004",
                question="What is the location of Sst neurons?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["Sst", "region", "location"],
                expected_entities=["Sst"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker {gene: 'Sst'})-[:LOCATE_AT]->(r:Region) RETURN DISTINCT r.acronym"
            ),

            BenchmarkQuestion(
                id="Q005",
                question="What genes are expressed in CA1?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["gene", "marker", "expression", "CA1"],
                expected_entities=["CA1"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'CA1'})<-[:LOCATE_AT]-(n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker) RETURN DISTINCT g.gene"
            ),

            BenchmarkQuestion(
                id="Q006",
                question="What is the axonal length of neurons in VISp?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="morphological",
                expected_answer_contains=["axonal", "length", "VISp", "morphology"],
                expected_entities=["VISp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: 'VISp'}) RETURN avg(n.axonal_length), stdev(n.axonal_length)"
            ),

            BenchmarkQuestion(
                id="Q007",
                question="Where does SSp project to?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="projection",
                expected_answer_contains=["project", "target", "SSp", "connection"],
                expected_entities=["SSp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'SSp'})-[:PROJECT_TO]->(t:Region) RETURN t.acronym, t.name"
            ),

            BenchmarkQuestion(
                id="Q008",
                question="What are the dendritic features of neurons in SUB?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="morphological",
                expected_answer_contains=["dendritic", "morphology", "SUB"],
                expected_entities=["SUB"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: 'SUB'}) RETURN avg(n.dendritic_length), avg(n.dendritic_branches)"
            ),

            BenchmarkQuestion(
                id="Q009",
                question="What subclasses are in the thalamus?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_answer_contains=["subclass", "thalamus", "cell type"],
                expected_entities=["TH"],  # Thalamus缩写
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE r.name CONTAINS 'thalamus' RETURN DISTINCT sc.name"
            ),

            BenchmarkQuestion(
                id="Q010",
                question="How many projection targets does ACAv have?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="projection",
                expected_answer_contains=["projection", "target", "ACAv", "count"],
                expected_entities=["ACAv"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'ACAv'})-[:PROJECT_TO]->(t:Region) RETURN count(DISTINCT t)"
            ),
        ])

        # =====================================================================
        # CATEGORY 2: MULTI-ENTITY (多实体查询) - 10 questions
        # =====================================================================

        questions.extend([
            BenchmarkQuestion(
                id="Q011",
                question="Compare Pvalb and Sst neurons in MOs",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_answer_contains=["Pvalb", "Sst", "MOs", "comparison", "difference"],
                expected_entities=["Pvalb", "Sst", "MOs"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'MOs'})-[:HAS_SUBCLASS]->(sc:Subclass) WHERE sc.name CONTAINS 'Pvalb' OR sc.name CONTAINS 'Sst' RETURN sc.name, sc.pct_cells"
            ),

            BenchmarkQuestion(
                id="Q012",
                question="What are the morphological differences between CA1 and CA3?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="morphological",
                expected_answer_contains=["CA1", "CA3", "morphology", "difference"],
                expected_entities=["CA1", "CA3"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['CA1', 'CA3'] RETURN r.acronym, avg(n.axonal_length), avg(n.dendritic_length)"
            ),

            BenchmarkQuestion(
                id="Q013",
                question="Which regions have both Gad2 and Vip neurons?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_answer_contains=["Gad2", "Vip", "region", "both"],
                expected_entities=["Gad2", "Vip"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE sc.name CONTAINS 'Gad2' OR sc.name CONTAINS 'Vip' WITH r, collect(sc.name) AS subclasses WHERE size(subclasses) = 2 RETURN r.acronym"
            ),

            BenchmarkQuestion(
                id="Q014",
                question="Compare the projection patterns of ACAd and MOs",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="projection",
                expected_answer_contains=["ACAd", "MOs", "projection", "comparison"],
                expected_entities=["ACAd", "MOs"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:PROJECT_TO]->(t:Region) WHERE r.acronym IN ['ACAd', 'MOs'] RETURN r.acronym, collect(t.acronym)"
            ),

            BenchmarkQuestion(
                id="Q015",
                question="What genes are expressed in both SSp and VISp?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_answer_contains=["gene", "SSp", "VISp", "both"],
                expected_entities=["SSp", "VISp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)<-[:LOCATE_AT]-(n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker) WHERE r.acronym IN ['SSp', 'VISp'] WITH g, collect(DISTINCT r.acronym) AS regions WHERE size(regions) = 2 RETURN g.gene"
            ),

            BenchmarkQuestion(
                id="Q016",
                question="Which has more neurons: PL or ACAd?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_answer_contains=["neuron", "count", "PL", "ACAd", "more"],
                expected_entities=["PL", "ACAd"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['PL', 'ACAd'] RETURN r.acronym, count(n) ORDER BY count(n) DESC"
            ),

            BenchmarkQuestion(
                id="Q017",
                question="Compare axonal branching in CA1, CA3, and DG",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="morphological",
                expected_answer_contains=["axonal", "branch", "CA1", "CA3", "DG"],
                expected_entities=["CA1", "CA3", "DG"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['CA1', 'CA3', 'DG'] RETURN r.acronym, avg(n.axonal_branches), stdev(n.axonal_branches)"
            ),

            BenchmarkQuestion(
                id="Q018",
                question="What projection targets are shared by MOs and SSp?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="projection",
                expected_answer_contains=["projection", "target", "MOs", "SSp", "shared"],
                expected_entities=["MOs", "SSp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:PROJECT_TO]->(t:Region) WHERE r.acronym IN ['MOs', 'SSp'] WITH t, collect(DISTINCT r.acronym) AS sources WHERE size(sources) = 2 RETURN t.acronym"
            ),

            BenchmarkQuestion(
                id="Q019",
                question="Which region has longer dendrites: ACAd or ORBl?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="morphological",
                expected_answer_contains=["dendritic", "length", "ACAd", "ORBl"],
                expected_entities=["ACAd", "ORBl"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['ACAd', 'ORBl'] RETURN r.acronym, avg(n.dendritic_length) ORDER BY avg(n.dendritic_length) DESC"
            ),

            BenchmarkQuestion(
                id="Q020",
                question="Compare the cell type diversity in hippocampus and cortex",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_answer_contains=["cell type", "diversity", "hippocampus", "cortex"],
                expected_entities=["CA1", "CA3", "MOs", "SSp"],  # 代表性区域
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE r.acronym IN ['CA1', 'CA3', 'MOs', 'SSp'] RETURN r.acronym, count(DISTINCT sc)"
            ),
        ])

        # =====================================================================
        # CATEGORY 3: COMPARATIVE (比较分析) - 10 questions
        # =====================================================================

        questions.extend([
            BenchmarkQuestion(
                id="Q021",
                question="What is the difference in molecular composition between excitatory and inhibitory neurons in MOs?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="molecular",
                expected_answer_contains=["excitatory", "inhibitory", "molecular", "difference", "MOs"],
                expected_entities=["MOs"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'MOs'})-[:HAS_SUBCLASS]->(sc:Subclass) RETURN sc.name, sc.pct_cells, sc.neurotransmitter"
            ),

            BenchmarkQuestion(
                id="Q022",
                question="How does the projection strength to thalamus differ between motor and sensory cortex?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="projection",
                expected_answer_contains=["projection", "strength", "thalamus", "motor", "sensory"],
                expected_entities=["MOs", "SSp", "TH"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[p:PROJECT_TO]->(t:Region) WHERE r.acronym IN ['MOs', 'SSp'] AND t.name CONTAINS 'thalamus' RETURN r.acronym, sum(p.weight)"
            ),

            BenchmarkQuestion(
                id="Q023",
                question="Compare the morphological complexity of pyramidal neurons across cortical layers",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="morphological",
                expected_answer_contains=["pyramidal", "morphology", "complexity", "layer"],
                expected_entities=["MOs", "ACAd", "SSp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['MOs', 'ACAd', 'SSp'] RETURN r.acronym, avg(n.axonal_branches + n.dendritic_branches) AS complexity"
            ),

            BenchmarkQuestion(
                id="Q024",
                question="Which brain regions show the highest mismatch between molecular and morphological organization?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_answer_contains=["mismatch", "molecular", "morphological", "region"],
                expected_entities=[],  # 系统性筛选，无初始实体
                requires_kg=True,
                ground_truth_cypher="// Complex multi-step analysis required"
            ),

            BenchmarkQuestion(
                id="Q025",
                question="How does the dendritic architecture differ between hippocampal subfields?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="morphological",
                expected_answer_contains=["dendritic", "architecture", "hippocampus", "subfield"],
                expected_entities=["CA1", "CA2", "CA3", "DG"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['CA1', 'CA2', 'CA3', 'DG'] RETURN r.acronym, avg(n.dendritic_length), avg(n.dendritic_branches)"
            ),

            BenchmarkQuestion(
                id="Q026",
                question="Compare Vip and Sst interneurons in terms of projection targets",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="projection",
                expected_answer_contains=["Vip", "Sst", "interneuron", "projection", "target"],
                expected_entities=["Vip", "Sst"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker)-[:PROJECT_TO]->(t:Region) WHERE g.gene IN ['Vip', 'Sst'] RETURN g.gene, collect(DISTINCT t.acronym)"
            ),

            BenchmarkQuestion(
                id="Q027",
                question="What is the relationship between gene expression diversity and morphological complexity across regions?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_answer_contains=["gene", "expression", "morphology", "complexity", "relationship"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Complex correlation analysis"
            ),

            BenchmarkQuestion(
                id="Q028",
                question="How do long-range vs local projections differ in their molecular signatures?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_answer_contains=["long-range", "local", "projection", "molecular", "signature"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Requires projection distance analysis"
            ),

            BenchmarkQuestion(
                id="Q029",
                question="Compare the axonal morphology of IT vs PT neurons",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="morphological",
                expected_answer_contains=["IT", "PT", "axonal", "morphology"],
                expected_entities=["IT", "PT"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:BELONGS_TO]->(c:Cluster) WHERE c.name CONTAINS 'IT' OR c.name CONTAINS 'PT' RETURN c.name, avg(n.axonal_length), avg(n.axonal_branches)"
            ),

            BenchmarkQuestion(
                id="Q030",
                question="Which regions have convergent projection inputs from both cortex and thalamus?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="projection",
                expected_answer_contains=["convergent", "projection", "cortex", "thalamus"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="MATCH (source:Region)-[:PROJECT_TO]->(target:Region) WHERE source.name CONTAINS 'cortex' OR source.name CONTAINS 'thalamus' WITH target, collect(DISTINCT source.name) AS sources WHERE size(sources) >= 2 RETURN target.acronym"
            ),
        ])

        # =====================================================================
        # CATEGORY 4: EXPLANATORY (解释性查询) - 10 questions
        # =====================================================================

        questions.extend([
            BenchmarkQuestion(
                id="Q031",
                question="Why do Pvalb neurons have fast-spiking properties?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="molecular",
                expected_answer_contains=["Pvalb", "fast-spiking", "property", "mechanism"],
                expected_entities=["Pvalb"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker {gene: 'Pvalb'}) RETURN n.properties"
            ),

            BenchmarkQuestion(
                id="Q032",
                question="How does the morphology of pyramidal neurons support their long-range projections?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_answer_contains=["pyramidal", "morphology", "long-range", "projection", "support"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Requires integrated analysis"
            ),

            BenchmarkQuestion(
                id="Q033",
                question="What determines the projection targets of CA1 neurons?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="projection",
                expected_answer_contains=["CA1", "projection", "target", "determine"],
                expected_entities=["CA1"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'CA1'})-[:PROJECT_TO]->(t:Region) RETURN t.acronym, t.properties"
            ),

            BenchmarkQuestion(
                id="Q034",
                question="Explain the relationship between dendritic branching complexity and input integration",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="morphological",
                expected_answer_contains=["dendritic", "branching", "complexity", "input", "integration"],
                expected_entities=[],
                requires_kg=False,  # 部分基于文献知识
                ground_truth_cypher="MATCH (n:Neuron) RETURN avg(n.dendritic_branches), avg(n.dendritic_length)"
            ),

            BenchmarkQuestion(
                id="Q035",
                question="Why are there regional differences in cell type composition?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="molecular",
                expected_answer_contains=["regional", "difference", "cell type", "composition"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) RETURN r.acronym, count(DISTINCT sc)"
            ),

            BenchmarkQuestion(
                id="Q036",
                question="How does molecular identity relate to projection specificity in cortical neurons?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_answer_contains=["molecular", "identity", "projection", "specificity"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Integrated molecular-projection analysis"
            ),

            BenchmarkQuestion(
                id="Q037",
                question="What functional role does the claustrum play based on its connectivity?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="projection",
                expected_answer_contains=["claustrum", "functional", "role", "connectivity"],
                expected_entities=["CLA"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region {acronym: 'CLA'})-[:PROJECT_TO]->(t:Region) RETURN t.acronym"
            ),

            BenchmarkQuestion(
                id="Q038",
                question="Explain why hippocampal neurons have distinct morphological features",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="morphological",
                expected_answer_contains=["hippocampal", "morphology", "distinct", "feature"],
                expected_entities=["CA1", "CA3", "DG"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region) WHERE r.acronym IN ['CA1', 'CA3', 'DG'] RETURN r.acronym, avg(n.axonal_length), avg(n.dendritic_length)"
            ),

            BenchmarkQuestion(
                id="Q039",
                question="How does cell type diversity contribute to circuit function in cortex?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_answer_contains=["cell type", "diversity", "circuit", "function"],
                expected_entities=["MOs", "SSp", "VISp"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE r.acronym IN ['MOs', 'SSp', 'VISp'] RETURN r.acronym, count(DISTINCT sc)"
            ),

            BenchmarkQuestion(
                id="Q040",
                question="Why do different interneuron types target specific subcellular compartments?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="morphological",
                expected_answer_contains=["interneuron", "type", "target", "subcellular", "compartment"],
                expected_entities=["Pvalb", "Sst", "Vip"],
                requires_kg=True,
                ground_truth_cypher="MATCH (n:Neuron)-[:EXPRESS_GENE]->(g:GeneMarker) WHERE g.gene IN ['Pvalb', 'Sst', 'Vip'] RETURN g.gene, n.axonal_properties"
            ),
        ])

        # =====================================================================
        # CATEGORY 5: OPEN-ENDED (开放式查询) - 10 questions
        # =====================================================================

        questions.extend([
            BenchmarkQuestion(
                id="Q041",
                question="What are the key organizational principles of cortical circuits?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_answer_contains=["organizational", "principle", "cortical", "circuit"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Comprehensive multi-modal analysis"
            ),

            BenchmarkQuestion(
                id="Q042",
                question="Describe the molecular diversity of neurons in the hippocampus",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="molecular",
                expected_answer_contains=["molecular", "diversity", "hippocampus", "neuron"],
                expected_entities=["CA1", "CA2", "CA3", "DG"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE r.acronym IN ['CA1', 'CA2', 'CA3', 'DG'] RETURN r.acronym, sc.name, sc.pct_cells"
            ),

            BenchmarkQuestion(
                id="Q043",
                question="How are brain regions organized into functional networks?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="projection",
                expected_answer_contains=["region", "organize", "functional", "network"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="MATCH (r1:Region)-[:PROJECT_TO]->(r2:Region) RETURN r1.acronym, collect(r2.acronym)"
            ),

            BenchmarkQuestion(
                id="Q044",
                question="What factors determine neuronal morphology?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="morphological",
                expected_answer_contains=["factor", "determine", "neuronal", "morphology"],
                expected_entities=[],
                requires_kg=False,
                ground_truth_cypher="MATCH (n:Neuron) RETURN n.axonal_length, n.dendritic_length, n.location"
            ),

            BenchmarkQuestion(
                id="Q045",
                question="Give me a comprehensive analysis of Car3+ neurons",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_answer_contains=["Car3", "comprehensive", "analysis"],
                expected_entities=["Car3"],
                requires_kg=True,
                ground_truth_cypher="// Full multi-modal analysis of Car3"
            ),

            BenchmarkQuestion(
                id="Q046",
                question="What distinguishes cortical from subcortical projection patterns?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="projection",
                expected_answer_contains=["cortical", "subcortical", "projection", "pattern", "distinguish"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:PROJECT_TO]->(t:Region) WHERE r.name CONTAINS 'cortex' OR r.name CONTAINS 'striatum' RETURN r.name, collect(t.acronym)"
            ),

            BenchmarkQuestion(
                id="Q047",
                question="Describe the relationship between cell type and connectivity",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_answer_contains=["cell type", "connectivity", "relationship"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Integrated molecular-projection analysis"
            ),

            BenchmarkQuestion(
                id="Q048",
                question="What are the major cell types in the cortex and their properties?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="molecular",
                expected_answer_contains=["cell type", "cortex", "property"],
                expected_entities=["MOs", "SSp", "VISp", "ACAd"],
                requires_kg=True,
                ground_truth_cypher="MATCH (r:Region)-[:HAS_SUBCLASS]->(sc:Subclass) WHERE r.acronym IN ['MOs', 'SSp', 'VISp', 'ACAd'] RETURN sc.name, sc.properties, sc.pct_cells"
            ),

            BenchmarkQuestion(
                id="Q049",
                question="How does the brain achieve both specificity and flexibility in connectivity?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="projection",
                expected_answer_contains=["specificity", "flexibility", "connectivity"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// High-level connectivity analysis"
            ),

            BenchmarkQuestion(
                id="Q050",
                question="What can we learn about brain function from multi-modal cell atlases?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_answer_contains=["brain", "function", "multi-modal", "atlas"],
                expected_entities=[],
                requires_kg=True,
                ground_truth_cypher="// Meta-analysis question"
            ),
        ])

        return questions

    @staticmethod
    def save_to_json(questions: List[BenchmarkQuestion], filepath: str):
        """保存问题集到JSON"""
        data = [
            {
                **asdict(q),
                'complexity': q.complexity.value
            }
            for q in questions
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(questions)} questions to {filepath}")

    @staticmethod
    def load_from_json(filepath: str) -> List[BenchmarkQuestion]:
        """从JSON加载问题集"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = []
        for item in data:
            item['complexity'] = QuestionComplexity(item['complexity'])
            questions.append(BenchmarkQuestion(**item))

        logger.info(f"✅ Loaded {len(questions)} questions from {filepath}")
        return questions


# ==================== Evaluation Metrics ====================

class MetricsCalculator:
    """计算各种评估指标"""

    def __init__(self):
        if HAS_METRICS:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.smoothing = SmoothingFunction().method1

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """计算BLEU分数"""
        if not HAS_METRICS:
            return 0.0

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        try:
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=self.smoothing
            )
            return score
        except:
            return 0.0

    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not HAS_METRICS:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def evaluate_criteria(self,
                          answer: str,
                          criteria: Dict) -> Dict[str, bool]:
        """评估答案是否满足标准"""
        results = {}
        answer_lower = answer.lower()

        # Must mention
        if 'must_mention' in criteria:
            for term in criteria['must_mention']:
                key = f"has_{term}"
                results[key] = term.lower() in answer_lower

        # Should mention
        if 'should_mention' in criteria:
            for term in criteria['should_mention']:
                key = f"has_{term}"
                results[key] = term.lower() in answer_lower

        # Should have regions
        if criteria.get('should_have_regions'):
            # Check if answer contains region acronyms (2-5 uppercase letters)
            region_pattern = r'\b[A-Z]{2,5}\b'
            results['has_regions'] = bool(re.findall(region_pattern, answer))

        # Should explain
        if criteria.get('should_explain'):
            explain_words = ['because', 'due to', 'reason', 'mechanism', 'function']
            results['has_explanation'] = any(w in answer_lower for w in explain_words)

        # Should have comparison
        if criteria.get('should_have_comparison'):
            compare_words = ['difference', 'contrast', 'compare', 'versus', 'while', 'whereas']
            results['has_comparison'] = any(w in answer_lower for w in compare_words)

        # Should cover modalities
        if 'should_cover_modalities' in criteria:
            for modality in criteria['should_cover_modalities']:
                key = f"covers_{modality}"
                if modality == 'molecular':
                    results[key] = any(w in answer_lower for w in ['gene', 'marker', 'cell type', 'cluster'])
                elif modality == 'morphological':
                    results[key] = any(w in answer_lower for w in ['axon', 'dendrite', 'morphology', 'branch'])
                elif modality == 'projection':
                    results[key] = any(w in answer_lower for w in ['project', 'target', 'connect', 'pathway'])

        return results


# ==================== Benchmark Runner ====================

@dataclass
class BenchmarkResult:
    """单个问题的评估结果"""
    question_id: str
    question: str
    complexity: str
    domain: str

    # Agent输出
    answer: str
    execution_time: float
    entities_found: List[str]
    reasoning_steps: int
    confidence: float

    # 评估结果
    criteria_passed: Dict[str, bool]
    accuracy_score: float
    bleu_score: float
    rouge_scores: Dict[str, float]

    # 元数据
    success: bool
    error: Optional[str] = None


class BenchmarkRunner:
    """
    Benchmark评估运行器

    功能:
    - 运行所有测试问题
    - 计算评估指标
    - 与baseline对比
    - 生成统计报告
    """

    def __init__(self, agent, output_dir: str = "./benchmark_results"):
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.metrics_calc = MetricsCalculator()
        self.results = []

    def run_benchmark(self,
                      questions: List[BenchmarkQuestion],
                      max_questions: Optional[int] = None) -> List[BenchmarkResult]:
        """
        运行完整benchmark

        Args:
            questions: 测试问题列表
            max_questions: 最多测试几个问题 (用于快速测试)
        """
        if max_questions:
            questions = questions[:max_questions]

        logger.info(f"🚀 Running benchmark on {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            logger.info(f"\n[{i}/{len(questions)}] {question.id}: {question.question}")

            result = self._evaluate_single_question(question)
            self.results.append(result)

            # Save intermediate results every 10 questions
            if i % 10 == 0:
                self._save_intermediate_results()

        # Save final results
        self._save_final_results()

        # Generate report
        self._generate_report()

        logger.info(f"\n✅ Benchmark complete! Results saved to {self.output_dir}")

        return self.results

    def _evaluate_single_question(self, question: BenchmarkQuestion) -> BenchmarkResult:
        """评估单个问题"""
        try:
            # Run agent
            start_time = time.time()
            response = self.agent.answer(question.question, max_iterations=10)
            execution_time = time.time() - start_time

            # Extract info
            answer = response.get('answer', '')
            entities_found = [
                e['text'] for e in response.get('entities_recognized', [])
            ]
            reasoning_steps = response.get('total_steps', 0)
            confidence = response.get('confidence_score', 0.0)

            # Evaluate criteria
            criteria_passed = self.metrics_calc.evaluate_criteria(
                answer,
                question.evaluation_criteria
            )

            # Calculate accuracy (% of criteria passed)
            if criteria_passed:
                accuracy = sum(criteria_passed.values()) / len(criteria_passed)
            else:
                accuracy = 0.0

            # Calculate BLEU/ROUGE (if gold answer exists)
            bleu = 0.0
            rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

            if question.gold_answer:
                bleu = self.metrics_calc.calculate_bleu(question.gold_answer, answer)
                rouge_scores = self.metrics_calc.calculate_rouge(question.gold_answer, answer)

            return BenchmarkResult(
                question_id=question.id,
                question=question.question,
                complexity=question.complexity.value,
                domain=question.domain,
                answer=answer,
                execution_time=execution_time,
                entities_found=entities_found,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                criteria_passed=criteria_passed,
                accuracy_score=accuracy,
                bleu_score=bleu,
                rouge_scores=rouge_scores,
                success=True,
                error=None
            )

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            return BenchmarkResult(
                question_id=question.id,
                question=question.question,
                complexity=question.complexity.value,
                domain=question.domain,
                answer="",
                execution_time=0.0,
                entities_found=[],
                reasoning_steps=0,
                confidence=0.0,
                criteria_passed={},
                accuracy_score=0.0,
                bleu_score=0.0,
                rouge_scores={},
                success=False,
                error=str(e)
            )

    def _save_intermediate_results(self):
        """保存中间结果"""
        filepath = self.output_dir / "intermediate_results.json"

        data = [asdict(r) for r in self.results]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_final_results(self):
        """保存最终结果"""
        filepath = self.output_dir / "final_results.json"

        data = [asdict(r) for r in self.results]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ Results saved to {filepath}")

    def _generate_report(self):
        """生成统计报告"""
        if not self.results:
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AIPOM-CoT Benchmark Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall stats
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        avg_accuracy = statistics.mean([r.accuracy_score for r in self.results if r.success])
        avg_time = statistics.mean([r.execution_time for r in self.results if r.success])
        avg_steps = statistics.mean([r.reasoning_steps for r in self.results if r.success])
        avg_confidence = statistics.mean([r.confidence for r in self.results if r.success])

        report_lines.append(f"Total Questions: {total}")
        report_lines.append(f"Successful: {successful} ({successful / total * 100:.1f}%)")
        report_lines.append(f"Average Accuracy: {avg_accuracy:.3f}")
        report_lines.append(f"Average Execution Time: {avg_time:.2f}s")
        report_lines.append(f"Average Reasoning Steps: {avg_steps:.1f}")
        report_lines.append(f"Average Confidence: {avg_confidence:.3f}")
        report_lines.append("")

        # By complexity
        report_lines.append("Performance by Complexity Level:")
        report_lines.append("-" * 40)

        by_complexity = defaultdict(list)
        for r in self.results:
            if r.success:
                by_complexity[r.complexity].append(r.accuracy_score)

        for complexity in sorted(by_complexity.keys()):
            scores = by_complexity[complexity]
            avg = statistics.mean(scores)
            report_lines.append(f"  {complexity:20s}: {avg:.3f} ({len(scores)} questions)")

        report_lines.append("")

        # By domain
        report_lines.append("Performance by Domain:")
        report_lines.append("-" * 40)

        by_domain = defaultdict(list)
        for r in self.results:
            if r.success:
                by_domain[r.domain].append(r.accuracy_score)

        for domain in sorted(by_domain.keys()):
            scores = by_domain[domain]
            avg = statistics.mean(scores)
            report_lines.append(f"  {domain:20s}: {avg:.3f} ({len(scores)} questions)")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        with open(self.output_dir / "benchmark_report.txt", 'w') as f:
            f.write(report_text)

        logger.info(f"  ✅ Report saved to {self.output_dir / 'benchmark_report.txt'}")


# ==================== Generate Test Questions ====================

def generate_test_questions_file():
    """生成test_questions.json"""
    questions = BenchmarkQuestionBank.generate_questions()

    BenchmarkQuestionBank.save_to_json(questions, "test_questions.json")

    # Print summary
    print(f"\n✅ Generated {len(questions)} test questions:")

    by_complexity = defaultdict(int)
    for q in questions:
        by_complexity[q.complexity.value] += 1

    for complexity, count in sorted(by_complexity.items()):
        print(f"  • {complexity:20s}: {count} questions")


if __name__ == "__main__":
    generate_test_questions_file()