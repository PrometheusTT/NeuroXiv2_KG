"""
Scientific Operator Library
===========================
对应设计图D: Scientific operator library

包含:
- Tri-modal fingerprint builder: 三模态向量构建
- Concordance & Divergence: 一致性/差异性分析
- Statistical Validation: 统计验证 (permutation tests + FDR correction)

Author: Lijun
Date: 2025-01
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from core_structures import (
    Modality, Evidence, EvidenceBuffer
)

logger = logging.getLogger(__name__)


# ==================== Tri-modal Fingerprint Builder ====================

@dataclass
class ModalityFingerprint:
    """单模态指纹"""
    modality: Modality
    features: Dict[str, float]
    raw_data: Any = None
    confidence: float = 1.0


@dataclass
class TriModalFingerprint:
    """
    三模态指纹 - 对应设计图D的Tri-modal fingerprint builder

    整合molecular, morphological, projection三个模态的特征向量
    """
    entity_name: str
    molecular: Optional[ModalityFingerprint] = None
    morphological: Optional[ModalityFingerprint] = None
    projection: Optional[ModalityFingerprint] = None

    def get_vector(self) -> np.ndarray:
        """获取合并的特征向量"""
        vectors = []

        if self.molecular:
            vectors.extend(list(self.molecular.features.values()))
        if self.morphological:
            vectors.extend(list(self.morphological.features.values()))
        if self.projection:
            vectors.extend(list(self.projection.features.values()))

        return np.array(vectors) if vectors else np.array([])

    def completeness(self) -> float:
        """计算完整度"""
        count = sum([
            1 if self.molecular else 0,
            1 if self.morphological else 0,
            1 if self.projection else 0,
        ])
        return count / 3.0


class TriModalFingerprintBuilder:
    """
    三模态指纹构建器

    从知识图谱查询结果构建三模态特征向量
    """

    def __init__(self):
        self.fingerprints: Dict[str, TriModalFingerprint] = {}

    def build_molecular_fingerprint(self, entity_name: str,
                                    data: Dict[str, Any]) -> ModalityFingerprint:
        """
        构建分子指纹

        特征包括:
        - marker expression levels
        - cluster distribution
        - enrichment scores
        """
        features = {}

        # 从数据提取特征
        if 'clusters' in data:
            features['cluster_count'] = len(data['clusters'])
            features['total_cells'] = sum(c.get('cell_count', 0) for c in data['clusters'])

        if 'markers' in data:
            for i, marker in enumerate(data['markers'][:5]):
                features[f'marker_{i}'] = 1.0

        if 'enrichment' in data:
            for region, score in list(data['enrichment'].items())[:5]:
                features[f'enrich_{region}'] = score

        fingerprint = ModalityFingerprint(
            modality=Modality.MOLECULAR,
            features=features,
            raw_data=data,
            confidence=0.9 if features else 0.5,
        )

        # 更新实体指纹
        if entity_name not in self.fingerprints:
            self.fingerprints[entity_name] = TriModalFingerprint(entity_name=entity_name)
        self.fingerprints[entity_name].molecular = fingerprint

        return fingerprint

    def build_morphological_fingerprint(self, entity_name: str,
                                        data: Dict[str, Any]) -> ModalityFingerprint:
        """
        构建形态指纹

        特征包括:
        - axon/dendrite length statistics
        - branch complexity
        - soma properties
        """
        features = {}

        if 'morphologies' in data:
            morphs = data['morphologies']

            # 轴突长度统计
            axon_lengths = [m.get('axon_length', 0) for m in morphs if m.get('axon_length')]
            if axon_lengths:
                features['axon_mean'] = np.mean(axon_lengths)
                features['axon_std'] = np.std(axon_lengths)

            # 树突长度统计
            dendrite_lengths = [m.get('dendrite_length', 0) for m in morphs if m.get('dendrite_length')]
            if dendrite_lengths:
                features['dendrite_mean'] = np.mean(dendrite_lengths)
                features['dendrite_std'] = np.std(dendrite_lengths)

            # 分支复杂度
            branch_counts = [m.get('branch_count', 0) for m in morphs if m.get('branch_count')]
            if branch_counts:
                features['branch_mean'] = np.mean(branch_counts)

            features['reconstruction_count'] = len(morphs)

        fingerprint = ModalityFingerprint(
            modality=Modality.MORPHOLOGICAL,
            features=features,
            raw_data=data,
            confidence=0.85 if features else 0.4,
        )

        if entity_name not in self.fingerprints:
            self.fingerprints[entity_name] = TriModalFingerprint(entity_name=entity_name)
        self.fingerprints[entity_name].morphological = fingerprint

        return fingerprint

    def build_projection_fingerprint(self, entity_name: str,
                                     data: Dict[str, Any]) -> ModalityFingerprint:
        """
        构建投射指纹

        特征包括:
        - target region distribution
        - projection weights
        - connectivity patterns
        """
        features = {}

        if 'projections' in data:
            projections = data['projections']

            # 目标区域数量
            features['target_count'] = len(projections)

            # 投射权重统计
            weights = [p.get('weight', 0) for p in projections if p.get('weight')]
            if weights:
                features['weight_total'] = sum(weights)
                features['weight_max'] = max(weights)
                features['weight_mean'] = np.mean(weights)

            # 主要目标区域
            sorted_proj = sorted(projections, key=lambda x: x.get('weight', 0), reverse=True)
            for i, proj in enumerate(sorted_proj[:3]):
                features[f'top_target_{i}'] = proj.get('weight', 0) / (features.get('weight_total', 1) or 1)

        fingerprint = ModalityFingerprint(
            modality=Modality.PROJECTION,
            features=features,
            raw_data=data,
            confidence=0.85 if features else 0.4,
        )

        if entity_name not in self.fingerprints:
            self.fingerprints[entity_name] = TriModalFingerprint(entity_name=entity_name)
        self.fingerprints[entity_name].projection = fingerprint

        return fingerprint

    def get_fingerprint(self, entity_name: str) -> Optional[TriModalFingerprint]:
        """获取实体的三模态指纹"""
        return self.fingerprints.get(entity_name)


# ==================== Concordance & Divergence Analysis ====================

@dataclass
class ConcordanceResult:
    """一致性/差异性分析结果 - 对应设计图D的concordance & divergence"""
    entity_a: str
    entity_b: str

    # 相似度矩阵
    similarity_matrix: Dict[str, float] = field(default_factory=dict)

    # Mismatch index |D_mol - D_morph|
    mismatch_index: float = 0.0

    # 各模态距离
    molecular_distance: float = 0.0
    morphological_distance: float = 0.0
    projection_distance: float = 0.0

    # 综合相似度
    overall_similarity: float = 0.0


class ConcordanceDivergenceAnalyzer:
    """
    一致性/差异性分析器

    计算:
    - 相似度矩阵
    - Mismatch index: |D_mol - D_morph|
    - 跨模态一致性
    """

    def __init__(self, fingerprint_builder: TriModalFingerprintBuilder):
        self.fingerprint_builder = fingerprint_builder

    def analyze(self, entity_a: str, entity_b: str) -> ConcordanceResult:
        """
        分析两个实体的一致性和差异性
        """
        fp_a = self.fingerprint_builder.get_fingerprint(entity_a)
        fp_b = self.fingerprint_builder.get_fingerprint(entity_b)

        result = ConcordanceResult(entity_a=entity_a, entity_b=entity_b)

        if not fp_a or not fp_b:
            return result

        # 计算各模态距离
        if fp_a.molecular and fp_b.molecular:
            result.molecular_distance = self._compute_distance(
                fp_a.molecular.features, fp_b.molecular.features
            )
            result.similarity_matrix['molecular'] = 1.0 - result.molecular_distance

        if fp_a.morphological and fp_b.morphological:
            result.morphological_distance = self._compute_distance(
                fp_a.morphological.features, fp_b.morphological.features
            )
            result.similarity_matrix['morphological'] = 1.0 - result.morphological_distance

        if fp_a.projection and fp_b.projection:
            result.projection_distance = self._compute_distance(
                fp_a.projection.features, fp_b.projection.features
            )
            result.similarity_matrix['projection'] = 1.0 - result.projection_distance

        # 计算Mismatch Index: |D_mol - D_morph|
        result.mismatch_index = abs(result.molecular_distance - result.morphological_distance)

        # 综合相似度
        if result.similarity_matrix:
            result.overall_similarity = np.mean(list(result.similarity_matrix.values()))

        return result

    def _compute_distance(self, features_a: Dict[str, float],
                          features_b: Dict[str, float]) -> float:
        """计算特征距离（归一化欧氏距离）"""
        # 获取共同特征
        common_keys = set(features_a.keys()) & set(features_b.keys())

        if not common_keys:
            return 1.0  # 无共同特征，最大距离

        # 计算欧氏距离
        squared_diff = sum(
            (features_a[k] - features_b[k]) ** 2
            for k in common_keys
        )
        distance = np.sqrt(squared_diff / len(common_keys))

        # 归一化到[0, 1]
        return min(1.0, distance)

    def compute_similarity_matrix(self, entities: List[str]) -> np.ndarray:
        """计算多实体相似度矩阵"""
        n = len(entities)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    result = self.analyze(entities[i], entities[j])
                    matrix[i, j] = result.overall_similarity
                    matrix[j, i] = result.overall_similarity

        return matrix


# ==================== Statistical Validation ====================

@dataclass
class StatisticalResult:
    """统计验证结果 - 对应设计图D的Statistical validation"""
    test_name: str
    statistic: float
    p_value: float
    fdr_q: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    significant: bool = False


class StatisticalValidator:
    """
    统计验证器 - 对应设计图D

    执行:
    - Permutation tests
    - FDR correction
    - Effect size calculation
    - Confidence interval estimation
    """

    def __init__(self, fdr_threshold: float = 0.05, n_permutations: int = 1000):
        self.fdr_threshold = fdr_threshold
        self.n_permutations = n_permutations

    def permutation_test(self, group_a: List[float], group_b: List[float],
                         test_name: str = "permutation") -> StatisticalResult:
        """
        Permutation test for comparing two groups
        """
        if not group_a or not group_b:
            return StatisticalResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                sample_size=0,
            )

        # 观察统计量（均值差）
        observed_diff = np.mean(group_a) - np.mean(group_b)

        # 合并数据
        combined = group_a + group_b
        n_a = len(group_a)

        # Permutation
        count_extreme = 0
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1

        p_value = (count_extreme + 1) / (self.n_permutations + 1)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(group_a) * (n_a - 1) + np.var(group_b) * (len(group_b) - 1)) /
            (n_a + len(group_b) - 2)
        ) if len(group_a) > 1 and len(group_b) > 1 else 1.0

        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0

        # 95% CI (bootstrap)
        ci = self._bootstrap_ci(group_a, group_b)

        return StatisticalResult(
            test_name=test_name,
            statistic=observed_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size=len(group_a) + len(group_b),
            significant=p_value < self.fdr_threshold,
        )

    def _bootstrap_ci(self, group_a: List[float], group_b: List[float],
                      n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        diffs = []

        for _ in range(n_bootstrap):
            sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
            sample_b = np.random.choice(group_b, size=len(group_b), replace=True)
            diffs.append(np.mean(sample_a) - np.mean(sample_b))

        lower = np.percentile(diffs, alpha / 2 * 100)
        upper = np.percentile(diffs, (1 - alpha / 2) * 100)

        return (lower, upper)

    def fdr_correction(self, p_values: List[float]) -> List[float]:
        """
        Benjamini-Hochberg FDR correction
        """
        n = len(p_values)
        if n == 0:
            return []

        # 排序
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # 计算FDR校正后的q值
        q_values = np.zeros(n)
        for i in range(n):
            q_values[sorted_indices[i]] = sorted_p[i] * n / (i + 1)

        # 确保单调性
        for i in range(n - 2, -1, -1):
            if q_values[i] > q_values[i + 1]:
                q_values[i] = q_values[i + 1]

        return list(np.minimum(q_values, 1.0))

    def validate_evidence(self, evidence: Evidence,
                          reference_data: Optional[List[float]] = None) -> Evidence:
        """
        为证据添加统计验证
        """
        if evidence.content and 'values' in evidence.content:
            values = evidence.content['values']

            if reference_data:
                result = self.permutation_test(values, reference_data)
                evidence.effect_size = result.effect_size
                evidence.p_value = result.p_value
                evidence.confidence_interval = result.confidence_interval
                evidence.sample_size = result.sample_size
            else:
                # 单样本统计
                evidence.sample_size = len(values)
                if len(values) > 1:
                    mean = np.mean(values)
                    std = np.std(values)
                    se = std / np.sqrt(len(values))
                    evidence.confidence_interval = (mean - 1.96 * se, mean + 1.96 * se)

        return evidence


# ==================== Multi-Modality Analyzer ====================

class MultiModalityAnalyzer:
    """
    多模态分析器 - 整合设计图D的所有算子

    提供统一接口调用:
    - Tri-modal fingerprint builder
    - Concordance & divergence analysis
    - Statistical validation
    """

    def __init__(self, fdr_threshold: float = 0.05):
        self.fingerprint_builder = TriModalFingerprintBuilder()
        self.concordance_analyzer = ConcordanceDivergenceAnalyzer(self.fingerprint_builder)
        self.statistical_validator = StatisticalValidator(fdr_threshold=fdr_threshold)

    def process_molecular_data(self, entity_name: str, data: Dict[str, Any]) -> Evidence:
        """处理分子数据并创建证据"""
        fingerprint = self.fingerprint_builder.build_molecular_fingerprint(entity_name, data)

        evidence = Evidence(
            evidence_id=f"mol_{entity_name}_{hash(str(data)) % 10000}",
            modality=Modality.MOLECULAR,
            content={
                'entity': entity_name,
                'features': fingerprint.features,
                'raw_count': len(data.get('clusters', [])),
            },
            source_query=f"Molecular analysis for {entity_name}",
            confidence=fingerprint.confidence,
        )

        return evidence

    def process_morphological_data(self, entity_name: str, data: Dict[str, Any]) -> Evidence:
        """处理形态数据并创建证据"""
        fingerprint = self.fingerprint_builder.build_morphological_fingerprint(entity_name, data)

        evidence = Evidence(
            evidence_id=f"morph_{entity_name}_{hash(str(data)) % 10000}",
            modality=Modality.MORPHOLOGICAL,
            content={
                'entity': entity_name,
                'features': fingerprint.features,
                'reconstruction_count': len(data.get('morphologies', [])),
            },
            source_query=f"Morphological analysis for {entity_name}",
            confidence=fingerprint.confidence,
        )

        return evidence

    def process_projection_data(self, entity_name: str, data: Dict[str, Any]) -> Evidence:
        """处理投射数据并创建证据"""
        fingerprint = self.fingerprint_builder.build_projection_fingerprint(entity_name, data)

        evidence = Evidence(
            evidence_id=f"proj_{entity_name}_{hash(str(data)) % 10000}",
            modality=Modality.PROJECTION,
            content={
                'entity': entity_name,
                'features': fingerprint.features,
                'target_count': len(data.get('projections', [])),
            },
            source_query=f"Projection analysis for {entity_name}",
            confidence=fingerprint.confidence,
        )

        return evidence

    def compare_entities(self, entity_a: str, entity_b: str) -> Dict[str, Any]:
        """比较两个实体"""
        concordance = self.concordance_analyzer.analyze(entity_a, entity_b)

        return {
            'similarity_matrix': concordance.similarity_matrix,
            'mismatch_index': concordance.mismatch_index,
            'overall_similarity': concordance.overall_similarity,
            'molecular_distance': concordance.molecular_distance,
            'morphological_distance': concordance.morphological_distance,
            'projection_distance': concordance.projection_distance,
        }

    def validate_and_enhance_evidence(self, evidence: Evidence,
                                      reference_data: Optional[List[float]] = None) -> Evidence:
        """验证并增强证据"""
        return self.statistical_validator.validate_evidence(evidence, reference_data)


# ==================== Export ====================

__all__ = [
    'ModalityFingerprint',
    'TriModalFingerprint',
    'TriModalFingerprintBuilder',
    'ConcordanceResult',
    'ConcordanceDivergenceAnalyzer',
    'StatisticalResult',
    'StatisticalValidator',
    'MultiModalityAnalyzer',
]