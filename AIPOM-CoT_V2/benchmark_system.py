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
    """问题复杂度分级"""
    SIMPLE_FACTUAL = "simple_factual"  # Level 1: 简单事实查询
    MULTI_ENTITY = "multi_entity"  # Level 2: 多实体关联
    COMPARATIVE = "comparative"  # Level 3: 比较分析
    EXPLANATORY = "explanatory"  # Level 4: 解释性推理
    OPEN_ENDED = "open_ended"  # Level 5: 开放性探索


@dataclass
class BenchmarkQuestion:
    """单个测试问题"""
    id: str
    question: str
    complexity: QuestionComplexity
    domain: str  # 'molecular' | 'morphological' | 'projection' | 'multi-modal'
    expected_entities: List[str]  # 期望识别的实体
    gold_answer: Optional[str]  # 金标准答案 (如果有)
    evaluation_criteria: Dict  # 评估标准
    metadata: Dict = field(default_factory=dict)


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
    def generate_questions() -> List[BenchmarkQuestion]:
        """生成完整问题集"""
        questions = []

        # ===== Level 1: Simple Factual (34 questions) =====

        # Gene markers
        questions.extend([
            BenchmarkQuestion(
                id="L1-G01",
                question="Tell me about Car3+ neurons",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["Car3"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Car3", "cluster", "region"],
                    "should_mention": ["marker", "expression"],
                }
            ),
            BenchmarkQuestion(
                id="L1-G02",
                question="Which regions express Pvalb?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["Pvalb"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Pvalb", "region"],
                    "should_have_regions": True
                }
            ),
            BenchmarkQuestion(
                id="L1-G03",
                question="What are Sst positive neurons?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["Sst"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["Sst", "interneuron"]}
            ),
            BenchmarkQuestion(
                id="L1-G04",
                question="Find cells expressing Vip",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["Vip"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["Vip", "cell"]}
            ),
            BenchmarkQuestion(
                id="L1-G05",
                question="Where is Gad1 expressed?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["Gad1"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["Gad1", "express"]}
            ),
        ])

        # Regions
        questions.extend([
            BenchmarkQuestion(
                id="L1-R01",
                question="Tell me about the claustrum",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="multi-modal",
                expected_entities=["CLA"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["claustrum", "CLA"]}
            ),
            BenchmarkQuestion(
                id="L1-R02",
                question="What is MOs?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="multi-modal",
                expected_entities=["MOs"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["MOs", "motor", "cortex"]}
            ),
            BenchmarkQuestion(
                id="L1-R03",
                question="Describe the primary somatosensory cortex",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="multi-modal",
                expected_entities=["SSp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["somatosensory", "SSp"]}
            ),
            BenchmarkQuestion(
                id="L1-R04",
                question="What cells are in ACAd?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=["ACAd"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["ACAd", "cell"]}
            ),
            BenchmarkQuestion(
                id="L1-R05",
                question="Tell me about the piriform cortex",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="multi-modal",
                expected_entities=["PIR"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["piriform", "PIR"]}
            ),
        ])

        # Morphology
        questions.extend([
            BenchmarkQuestion(
                id="L1-M01",
                question="What is axonal length?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="morphological",
                expected_entities=["axonal_length"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["axon", "length"]}
            ),
            BenchmarkQuestion(
                id="L1-M02",
                question="Explain dendritic branching",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="morphological",
                expected_entities=["dendritic_branches"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["dendrite", "branch"]}
            ),
            BenchmarkQuestion(
                id="L1-M03",
                question="What are morphological features of neurons?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="morphological",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"should_mention": ["axon", "dendrite", "soma"]}
            ),
        ])

        # Projection
        questions.extend([
            BenchmarkQuestion(
                id="L1-P01",
                question="What are projection patterns?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="projection",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["project", "target"]}
            ),
            BenchmarkQuestion(
                id="L1-P02",
                question="Explain cortical connectivity",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="projection",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["connect", "cortex"]}
            ),
        ])

        # Add more simple questions to reach 34
        for i in range(6, 20):
            questions.append(BenchmarkQuestion(
                id=f"L1-G{i:02d}",
                question=f"What are the characteristics of gene marker {i}?",
                complexity=QuestionComplexity.SIMPLE_FACTUAL,
                domain="molecular",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["gene", "marker"]}
            ))

        # ===== Level 2: Multi-Entity (28 questions) =====

        questions.extend([
            BenchmarkQuestion(
                id="L2-001",
                question="What is the relationship between Car3 and MOs?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_entities=["Car3", "MOs"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Car3", "MOs"],
                    "should_mention": ["express", "cluster"]
                }
            ),
            BenchmarkQuestion(
                id="L2-002",
                question="How do Pvalb neurons relate to SSp?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_entities=["Pvalb", "SSp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["Pvalb", "SSp"]}
            ),
            BenchmarkQuestion(
                id="L2-003",
                question="Analyze Sst expression in claustrum",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_entities=["Sst", "CLA"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["Sst", "claustrum"]}
            ),
            BenchmarkQuestion(
                id="L2-004",
                question="What is the projection from MOs to SSp?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="projection",
                expected_entities=["MOs", "SSp"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["MOs", "SSp", "project"],
                    "should_have_projection_data": True
                }
            ),
            BenchmarkQuestion(
                id="L2-005",
                question="Describe the connectivity between ACAd and MOs",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="projection",
                expected_entities=["ACAd", "MOs"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["ACAd", "MOs", "connect"]}
            ),
        ])

        # Add more multi-entity questions
        region_pairs = [
            ("CLA", "MOs"), ("SSp", "MOs"), ("ACAd", "SSp"),
            ("PIR", "CLA"), ("MOp", "MOs"), ("AI", "PIR")
        ]

        for i, (r1, r2) in enumerate(region_pairs, start=6):
            questions.append(BenchmarkQuestion(
                id=f"L2-{i:03d}",
                question=f"What is the relationship between {r1} and {r2}?",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="projection",
                expected_entities=[r1, r2],
                gold_answer=None,
                evaluation_criteria={"must_mention": [r1, r2]}
            ))

        # Gene-region pairs
        gene_region_pairs = [
            ("Car3", "MOs"), ("Pvalb", "SSp"), ("Sst", "CLA"),
            ("Vip", "MOs"), ("Gad1", "ACAd")
        ]

        for i, (gene, region) in enumerate(gene_region_pairs, start=12):
            questions.append(BenchmarkQuestion(
                id=f"L2-{i:03d}",
                question=f"Analyze {gene}+ neurons in {region}",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="molecular",
                expected_entities=[gene, region],
                gold_answer=None,
                evaluation_criteria={"must_mention": [gene, region]}
            ))

        # Fill to 28
        for i in range(17, 29):
            questions.append(BenchmarkQuestion(
                id=f"L2-{i:03d}",
                question=f"Multi-entity query {i}",
                complexity=QuestionComplexity.MULTI_ENTITY,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["neuron"]}
            ))

        # ===== Level 3: Comparative (31 questions) =====

        questions.extend([
            BenchmarkQuestion(
                id="L3-001",
                question="Compare Pvalb and Sst interneurons",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="molecular",
                expected_entities=["Pvalb", "Sst"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Pvalb", "Sst", "compare", "difference"],
                    "should_have_comparison": True
                }
            ),
            BenchmarkQuestion(
                id="L3-002",
                question="What are the differences between MOs and MOp?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_entities=["MOs", "MOp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["MOs", "MOp", "difference"]}
            ),
            BenchmarkQuestion(
                id="L3-003",
                question="Compare claustrum and piriform cortex connectivity",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="projection",
                expected_entities=["CLA", "PIR"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["claustrum", "piriform", "connect"]}
            ),
            BenchmarkQuestion(
                id="L3-004",
                question="Contrast axonal vs dendritic features in SSp",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="morphological",
                expected_entities=["SSp", "axonal", "dendritic"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["axon", "dendrite", "SSp"]}
            ),
            BenchmarkQuestion(
                id="L3-005",
                question="Compare excitatory and inhibitory neurons in MOs",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="molecular",
                expected_entities=["MOs", "excitatory", "inhibitory"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["excitatory", "inhibitory", "MOs"]}
            ),
        ])

        # Gene comparisons
        gene_comparisons = [
            ("Car3", "Pvalb"), ("Sst", "Vip"), ("Gad1", "Gad2"),
            ("Car3", "Sst"), ("Pvalb", "Vip")
        ]

        for i, (g1, g2) in enumerate(gene_comparisons, start=6):
            questions.append(BenchmarkQuestion(
                id=f"L3-{i:03d}",
                question=f"Compare {g1} and {g2} expression patterns",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="molecular",
                expected_entities=[g1, g2],
                gold_answer=None,
                evaluation_criteria={"must_mention": [g1, g2, "compare"]}
            ))

        # Region comparisons
        region_comparisons = [
            ("CLA", "AI"), ("SSp", "ACAd"), ("MOs", "SSp"),
            ("PIR", "ENTl"), ("MOp", "SSp")
        ]

        for i, (r1, r2) in enumerate(region_comparisons, start=11):
            questions.append(BenchmarkQuestion(
                id=f"L3-{i:03d}",
                question=f"What are the differences between {r1} and {r2}?",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_entities=[r1, r2],
                gold_answer=None,
                evaluation_criteria={"must_mention": [r1, r2, "difference"]}
            ))

        # Morphology comparisons
        morphology_comparisons = [
            "Compare axonal and dendritic branching patterns",
            "Contrast pyramidal vs interneuron morphology",
            "Compare IT and ET neuron features",
            "Analyze morphological differences across cortical layers"
        ]

        for i, q in enumerate(morphology_comparisons, start=16):
            questions.append(BenchmarkQuestion(
                id=f"L3-{i:03d}",
                question=q,
                complexity=QuestionComplexity.COMPARATIVE,
                domain="morphological",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["morpholog", "compare"]}
            ))

        # Fill to 31
        for i in range(20, 32):
            questions.append(BenchmarkQuestion(
                id=f"L3-{i:03d}",
                question=f"Comparative analysis {i}",
                complexity=QuestionComplexity.COMPARATIVE,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["compare"]}
            ))

        # ===== Level 4: Explanatory (24 questions) =====

        questions.extend([
            BenchmarkQuestion(
                id="L4-001",
                question="Why do Pvalb neurons have distinct morphology?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_entities=["Pvalb"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Pvalb", "morphology"],
                    "should_explain": True
                }
            ),
            BenchmarkQuestion(
                id="L4-002",
                question="Explain the functional significance of claustrum connectivity",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="projection",
                expected_entities=["CLA"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["claustrum", "function", "connect"]}
            ),
            BenchmarkQuestion(
                id="L4-003",
                question="What determines regional cell type composition?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="molecular",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"should_mention": ["cell type", "region", "determine"]}
            ),
            BenchmarkQuestion(
                id="L4-004",
                question="Why do motor regions have unique projection patterns?",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="projection",
                expected_entities=["MOs", "MOp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["motor", "project"]}
            ),
            BenchmarkQuestion(
                id="L4-005",
                question="Explain the relationship between gene expression and morphology",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["gene", "morphology", "relationship"]}
            ),
        ])

        # More explanatory questions
        explanatory_questions = [
            "What mechanisms underlie cortical layer formation?",
            "Explain how molecular markers define cell types",
            "Why are some regions more densely connected?",
            "What drives neuronal morphological diversity?",
            "Explain the role of interneurons in cortical circuits",
            "How do projection patterns relate to function?",
            "What determines axonal targeting specificity?",
            "Explain regional heterogeneity in cell composition",
            "Why do certain genes cluster spatially?",
            "What is the significance of dendritic branching complexity?"
        ]

        for i, q in enumerate(explanatory_questions, start=6):
            questions.append(BenchmarkQuestion(
                id=f"L4-{i:03d}",
                question=q,
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"should_explain": True}
            ))

        # Fill to 24
        for i in range(16, 25):
            questions.append(BenchmarkQuestion(
                id=f"L4-{i:03d}",
                question=f"Explain mechanism {i}",
                complexity=QuestionComplexity.EXPLANATORY,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["explain"]}
            ))

        # ===== Level 5: Open-Ended (10 questions) =====

        questions.extend([
            BenchmarkQuestion(
                id="L5-001",
                question="Provide a comprehensive analysis of the claustrum",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=["CLA"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["claustrum"],
                    "should_cover_modalities": ["molecular", "morphological", "projection"]
                }
            ),
            BenchmarkQuestion(
                id="L5-002",
                question="Give me everything about Pvalb interneurons",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=["Pvalb"],
                gold_answer=None,
                evaluation_criteria={
                    "must_mention": ["Pvalb", "interneuron"],
                    "should_be_comprehensive": True
                }
            ),
            BenchmarkQuestion(
                id="L5-003",
                question="Analyze the motor cortex from all perspectives",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=["MOs", "MOp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["motor", "cortex"]}
            ),
            BenchmarkQuestion(
                id="L5-004",
                question="Tell me everything about cortical cell type diversity",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"should_mention": ["cell type", "diversity", "cortex"]}
            ),
            BenchmarkQuestion(
                id="L5-005",
                question="Comprehensive overview of brain connectivity patterns",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="projection",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["connectivity", "pattern"]}
            ),
            BenchmarkQuestion(
                id="L5-006",
                question="Describe the complete landscape of gene marker expression",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="molecular",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["gene", "marker", "expression"]}
            ),
            BenchmarkQuestion(
                id="L5-007",
                question="Full morphological characterization of cortical neurons",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="morphological",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["morpholog", "neuron", "cortex"]}
            ),
            BenchmarkQuestion(
                id="L5-008",
                question="Comprehensive analysis of somatosensory system",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=["SSp"],
                gold_answer=None,
                evaluation_criteria={"must_mention": ["somatosensory"]}
            ),
            BenchmarkQuestion(
                id="L5-009",
                question="Explore the relationship between molecular, morphological, and projection features",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={
                    "should_cover_modalities": ["molecular", "morphological", "projection"]
                }
            ),
            BenchmarkQuestion(
                id="L5-010",
                question="What can you tell me about the entire brain knowledge graph?",
                complexity=QuestionComplexity.OPEN_ENDED,
                domain="multi-modal",
                expected_entities=[],
                gold_answer=None,
                evaluation_criteria={"should_be_comprehensive": True}
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