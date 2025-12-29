"""
Tri-Modal Fingerprint Builder
=============================

实现手稿Result 4的全脑三模态指纹分析:
- Molecular fingerprint (MERFISH基因表达)
- Morphological fingerprint (神经元形态特征)
- Projection fingerprint (投射连接模式)

功能:
1. 构建单区域的三模态指纹
2. 计算区域间相似性矩阵
3. 检测跨模态不匹配
4. 生成综合fingerprint报告

Author: Lijun
Date: 2025-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """相似性度量方法"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    JS_DIVERGENCE = "js_divergence"  # Jensen-Shannon divergence


@dataclass
class ModalityFingerprint:
    """单模态指纹"""
    modality: str  # 'molecular', 'morphological', 'projection'
    region_name: str
    feature_vector: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.feature_vector)

    def normalize(self) -> 'ModalityFingerprint':
        """L2归一化"""
        norm = np.linalg.norm(self.feature_vector)
        if norm > 0:
            normalized_vec = self.feature_vector / norm
        else:
            normalized_vec = self.feature_vector
        return ModalityFingerprint(
            modality=self.modality,
            region_name=self.region_name,
            feature_vector=normalized_vec,
            feature_names=self.feature_names,
            metadata=self.metadata
        )

    def to_dict(self) -> Dict:
        return {
            'modality': self.modality,
            'region': self.region_name,
            'dimension': self.dimension,
            'top_features': self._get_top_features(5),
            'metadata': self.metadata
        }

    def _get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """获取top-N特征"""
        if not self.feature_names:
            return []
        indices = np.argsort(self.feature_vector)[-n:][::-1]
        return [(self.feature_names[i], float(self.feature_vector[i]))
                for i in indices if i < len(self.feature_names)]


@dataclass
class TriModalFingerprint:
    """三模态综合指纹"""
    region_name: str
    molecular: Optional[ModalityFingerprint] = None
    morphological: Optional[ModalityFingerprint] = None
    projection: Optional[ModalityFingerprint] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def completeness(self) -> float:
        """模态完整度 (0-1)"""
        count = sum([
            self.molecular is not None,
            self.morphological is not None,
            self.projection is not None
        ])
        return count / 3.0

    @property
    def available_modalities(self) -> List[str]:
        """可用模态列表"""
        modalities = []
        if self.molecular:
            modalities.append('molecular')
        if self.morphological:
            modalities.append('morphological')
        if self.projection:
            modalities.append('projection')
        return modalities

    def get_modality(self, modality: str) -> Optional[ModalityFingerprint]:
        """获取指定模态"""
        return getattr(self, modality, None)

    def to_dict(self) -> Dict:
        return {
            'region': self.region_name,
            'completeness': self.completeness,
            'available_modalities': self.available_modalities,
            'molecular': self.molecular.to_dict() if self.molecular else None,
            'morphological': self.morphological.to_dict() if self.morphological else None,
            'projection': self.projection.to_dict() if self.projection else None
        }


@dataclass
class SimilarityMatrix:
    """相似性矩阵"""
    modality: str
    region_names: List[str]
    matrix: np.ndarray
    metric: SimilarityMetric = SimilarityMetric.COSINE

    @property
    def n_regions(self) -> int:
        return len(self.region_names)

    def get_similarity(self, region1: str, region2: str) -> float:
        """获取两区域相似度"""
        try:
            i = self.region_names.index(region1)
            j = self.region_names.index(region2)
            return float(self.matrix[i, j])
        except ValueError:
            return np.nan

    def get_most_similar(self, region: str, n: int = 5) -> List[Tuple[str, float]]:
        """获取最相似的n个区域"""
        try:
            i = self.region_names.index(region)
            similarities = self.matrix[i]
            # 排除自身
            indices = np.argsort(similarities)[::-1]
            results = []
            for idx in indices:
                if self.region_names[idx] != region:
                    results.append((self.region_names[idx], float(similarities[idx])))
                    if len(results) >= n:
                        break
            return results
        except ValueError:
            return []

    def to_dict(self) -> Dict:
        return {
            'modality': self.modality,
            'n_regions': self.n_regions,
            'metric': self.metric.value,
            'mean_similarity': float(np.mean(self.matrix)),
            'std_similarity': float(np.std(self.matrix))
        }


class TriModalFingerprintBuilder:
    """
    三模态指纹构建器

    核心功能:
    1. 从原始数据构建各模态指纹
    2. 计算区域间相似性
    3. 整合三模态信息
    4. 检测跨模态不匹配
    """

    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.similarity_metric = similarity_metric
        self._fingerprints: Dict[str, TriModalFingerprint] = {}
        self._similarity_matrices: Dict[str, SimilarityMatrix] = {}

    # ==================== Fingerprint Construction ====================

    def build_molecular_fingerprint(self,
                                   region_name: str,
                                   gene_expression: Dict[str, float],
                                   normalize: bool = True) -> ModalityFingerprint:
        """
        构建分子指纹

        基于MERFISH基因表达数据

        Args:
            region_name: 区域名称
            gene_expression: 基因名 -> 表达值
            normalize: 是否归一化
        """
        genes = sorted(gene_expression.keys())
        values = np.array([gene_expression[g] for g in genes])

        fp = ModalityFingerprint(
            modality='molecular',
            region_name=region_name,
            feature_vector=values,
            feature_names=genes,
            metadata={'n_genes': len(genes)}
        )

        return fp.normalize() if normalize else fp

    def build_morphological_fingerprint(self,
                                       region_name: str,
                                       morphology_features: Dict[str, float],
                                       normalize: bool = True) -> ModalityFingerprint:
        """
        构建形态学指纹

        基于神经元形态特征（树突/轴突长度、分支复杂度等）

        Args:
            region_name: 区域名称
            morphology_features: 特征名 -> 值
            normalize: 是否归一化
        """
        features = sorted(morphology_features.keys())
        values = np.array([morphology_features[f] for f in features])

        fp = ModalityFingerprint(
            modality='morphological',
            region_name=region_name,
            feature_vector=values,
            feature_names=features,
            metadata={'n_features': len(features)}
        )

        return fp.normalize() if normalize else fp

    def build_projection_fingerprint(self,
                                    region_name: str,
                                    projection_targets: Dict[str, float],
                                    normalize: bool = True) -> ModalityFingerprint:
        """
        构建投射指纹

        基于脑区投射连接强度

        Args:
            region_name: 区域名称
            projection_targets: 目标区域 -> 投射强度
            normalize: 是否归一化
        """
        targets = sorted(projection_targets.keys())
        values = np.array([projection_targets[t] for t in targets])

        fp = ModalityFingerprint(
            modality='projection',
            region_name=region_name,
            feature_vector=values,
            feature_names=targets,
            metadata={'n_targets': len(targets)}
        )

        return fp.normalize() if normalize else fp

    def build_trimodal_fingerprint(self,
                                   region_name: str,
                                   molecular_data: Optional[Dict[str, float]] = None,
                                   morphological_data: Optional[Dict[str, float]] = None,
                                   projection_data: Optional[Dict[str, float]] = None) -> TriModalFingerprint:
        """
        构建完整三模态指纹

        Args:
            region_name: 区域名称
            molecular_data: 分子数据
            morphological_data: 形态数据
            projection_data: 投射数据

        Returns:
            三模态指纹对象
        """
        molecular = None
        morphological = None
        projection = None

        if molecular_data:
            molecular = self.build_molecular_fingerprint(region_name, molecular_data)

        if morphological_data:
            morphological = self.build_morphological_fingerprint(region_name, morphological_data)

        if projection_data:
            projection = self.build_projection_fingerprint(region_name, projection_data)

        fp = TriModalFingerprint(
            region_name=region_name,
            molecular=molecular,
            morphological=morphological,
            projection=projection
        )

        self._fingerprints[region_name] = fp
        return fp

    # ==================== Similarity Computation ====================

    def compute_similarity(self, fp1: ModalityFingerprint,
                          fp2: ModalityFingerprint,
                          metric: SimilarityMetric = None) -> float:
        """
        计算两个指纹的相似度

        Args:
            fp1, fp2: 两个模态指纹
            metric: 相似性度量（默认使用构建器设置）

        Returns:
            相似度值 (0-1)
        """
        metric = metric or self.similarity_metric

        v1 = fp1.feature_vector
        v2 = fp2.feature_vector

        # 确保维度一致
        if len(v1) != len(v2):
            logger.warning(f"Dimension mismatch: {len(v1)} vs {len(v2)}")
            # 取交集或填充
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]

        if metric == SimilarityMetric.COSINE:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        elif metric == SimilarityMetric.EUCLIDEAN:
            # 转换为相似度 (0-1)
            dist = np.linalg.norm(v1 - v2)
            return float(1 / (1 + dist))

        elif metric == SimilarityMetric.PEARSON:
            if np.std(v1) == 0 or np.std(v2) == 0:
                return 0.0
            corr = np.corrcoef(v1, v2)[0, 1]
            return float((corr + 1) / 2)  # 转换到0-1

        elif metric == SimilarityMetric.JACCARD:
            # 二值化后计算Jaccard
            b1 = (v1 > 0).astype(float)
            b2 = (v2 > 0).astype(float)
            intersection = np.sum(b1 * b2)
            union = np.sum((b1 + b2) > 0)
            return float(intersection / union) if union > 0 else 0.0

        elif metric == SimilarityMetric.JS_DIVERGENCE:
            # Jensen-Shannon divergence转相似度
            # 确保是概率分布
            p = v1 / (np.sum(v1) + 1e-10)
            q = v2 / (np.sum(v2) + 1e-10)
            m = (p + q) / 2

            # 避免log(0)
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            m = np.clip(m, 1e-10, 1)

            js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            return float(1 - np.sqrt(js))  # 转换为相似度

        return 0.0

    def compute_similarity_matrix(self,
                                  fingerprints: List[TriModalFingerprint],
                                  modality: str) -> SimilarityMatrix:
        """
        计算指定模态的区域间相似性矩阵

        Args:
            fingerprints: 区域指纹列表
            modality: 模态名称

        Returns:
            相似性矩阵
        """
        n = len(fingerprints)
        region_names = [fp.region_name for fp in fingerprints]
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    fp_i = fingerprints[i].get_modality(modality)
                    fp_j = fingerprints[j].get_modality(modality)

                    if fp_i is not None and fp_j is not None:
                        matrix[i, j] = self.compute_similarity(fp_i, fp_j)
                    else:
                        matrix[i, j] = np.nan

        sim_matrix = SimilarityMatrix(
            modality=modality,
            region_names=region_names,
            matrix=matrix,
            metric=self.similarity_metric
        )

        self._similarity_matrices[modality] = sim_matrix
        return sim_matrix

    def compute_all_similarity_matrices(self,
                                        fingerprints: List[TriModalFingerprint]) -> Dict[str, SimilarityMatrix]:
        """
        计算所有模态的相似性矩阵

        Args:
            fingerprints: 区域指纹列表

        Returns:
            模态 -> 相似性矩阵的映射
        """
        results = {}
        for modality in ['molecular', 'morphological', 'projection']:
            results[modality] = self.compute_similarity_matrix(fingerprints, modality)
        return results

    # ==================== Cross-Modal Analysis ====================

    def compute_cross_modal_mismatch(self,
                                    sim_matrices: Dict[str, SimilarityMatrix]) -> Dict:
        """
        计算跨模态不匹配指数

        识别在不同模态下表现差异大的区域对

        Args:
            sim_matrices: 模态 -> 相似性矩阵

        Returns:
            不匹配分析结果
        """
        modalities = list(sim_matrices.keys())
        if len(modalities) < 2:
            return {'error': 'Need at least 2 modalities'}

        n_regions = sim_matrices[modalities[0]].n_regions
        region_names = sim_matrices[modalities[0]].region_names

        # 计算模态间相关性
        modality_correlations = {}
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                mat1 = sim_matrices[mod1].matrix.flatten()
                mat2 = sim_matrices[mod2].matrix.flatten()

                # 处理NaN
                valid = ~(np.isnan(mat1) | np.isnan(mat2))
                if np.sum(valid) > 0:
                    corr = np.corrcoef(mat1[valid], mat2[valid])[0, 1]
                else:
                    corr = 0.0

                modality_correlations[f"{mod1}_vs_{mod2}"] = float(corr)

        # 计算每对区域的跨模态差异
        mismatch_indices = []
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                sims = []
                for mod in modalities:
                    sim = sim_matrices[mod].matrix[i, j]
                    if not np.isnan(sim):
                        sims.append(sim)

                if len(sims) >= 2:
                    # 不匹配指数 = 模态间相似度的标准差
                    mismatch_idx = float(np.std(sims))
                    mismatch_indices.append({
                        'region_1': region_names[i],
                        'region_2': region_names[j],
                        'mismatch_index': mismatch_idx,
                        'modality_similarities': {mod: float(sim_matrices[mod].matrix[i, j])
                                                 for mod in modalities}
                    })

        # 按不匹配程度排序
        mismatch_indices.sort(key=lambda x: x['mismatch_index'], reverse=True)

        return {
            'modality_correlations': modality_correlations,
            'top_mismatches': mismatch_indices[:20],  # 前20个
            'mean_mismatch': float(np.mean([m['mismatch_index'] for m in mismatch_indices])) if mismatch_indices else 0,
            'n_significant_mismatches': sum(1 for m in mismatch_indices if m['mismatch_index'] > 0.2)
        }

    def compute_integrated_similarity(self,
                                      sim_matrices: Dict[str, SimilarityMatrix],
                                      weights: Dict[str, float] = None) -> SimilarityMatrix:
        """
        计算整合相似度（加权平均三模态）

        Args:
            sim_matrices: 模态 -> 相似性矩阵
            weights: 模态权重（默认等权）

        Returns:
            整合后的相似性矩阵
        """
        modalities = list(sim_matrices.keys())

        if weights is None:
            weights = {mod: 1.0 / len(modalities) for mod in modalities}

        # 归一化权重
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # 加权平均
        first_mod = modalities[0]
        n = sim_matrices[first_mod].n_regions
        region_names = sim_matrices[first_mod].region_names

        integrated = np.zeros((n, n))
        valid_counts = np.zeros((n, n))

        for mod in modalities:
            mat = sim_matrices[mod].matrix
            mask = ~np.isnan(mat)
            integrated += np.where(mask, mat * weights[mod], 0)
            valid_counts += mask.astype(float) * weights[mod]

        # 归一化
        integrated = np.where(valid_counts > 0, integrated / valid_counts, np.nan)

        return SimilarityMatrix(
            modality='integrated',
            region_names=region_names,
            matrix=integrated,
            metric=self.similarity_metric
        )

    # ==================== Report Generation ====================

    def generate_fingerprint_report(self,
                                    fingerprints: List[TriModalFingerprint],
                                    include_statistics: bool = True) -> Dict:
        """
        生成综合指纹分析报告

        对应手稿Figure 4的分析

        Args:
            fingerprints: 区域指纹列表
            include_statistics: 是否包含统计验证

        Returns:
            完整分析报告
        """
        # 1. 基础统计
        n_regions = len(fingerprints)
        completeness_scores = [fp.completeness for fp in fingerprints]

        # 2. 计算相似性矩阵
        sim_matrices = self.compute_all_similarity_matrices(fingerprints)

        # 3. 跨模态分析
        mismatch_analysis = self.compute_cross_modal_mismatch(sim_matrices)

        # 4. 整合相似性
        integrated_sim = self.compute_integrated_similarity(sim_matrices)

        # 5. 统计验证（如果需要）
        statistical_validation = None
        if include_statistics:
            try:
                try:
                    from .statistical_validator import StatisticalValidator
                except ImportError:
                    from statistical_validator import StatisticalValidator

                validator = StatisticalValidator()

                # 验证相似性模式
                statistical_validation = {}
                for mod, sim_mat in sim_matrices.items():
                    result = validator.validate_fingerprint_similarity(sim_mat.matrix)
                    statistical_validation[mod] = result
            except ImportError:
                logger.warning("StatisticalValidator not available")

        report = {
            'summary': {
                'n_regions': n_regions,
                'mean_completeness': float(np.mean(completeness_scores)),
                'fully_complete_regions': sum(1 for c in completeness_scores if c == 1.0)
            },
            'similarity_matrices': {mod: sm.to_dict() for mod, sm in sim_matrices.items()},
            'integrated_similarity': integrated_sim.to_dict(),
            'cross_modal_analysis': mismatch_analysis,
            'region_fingerprints': [fp.to_dict() for fp in fingerprints]
        }

        if statistical_validation:
            report['statistical_validation'] = statistical_validation

        return report

    # ==================== Data Loading Helpers ====================

    def build_fingerprints_from_query_results(self,
                                              molecular_results: List[Dict],
                                              morphological_results: List[Dict],
                                              projection_results: List[Dict]) -> List[TriModalFingerprint]:
        """
        从Neo4j查询结果构建指纹

        这是与主流程集成的接口

        Args:
            molecular_results: 分子查询结果 [{region, gene, expression}, ...]
            morphological_results: 形态查询结果 [{region, feature, value}, ...]
            projection_results: 投射查询结果 [{source, target, strength}, ...]

        Returns:
            区域指纹列表
        """
        # 按区域聚合数据
        regions = set()

        molecular_by_region = {}
        for r in molecular_results:
            region = r.get('region', r.get('region_name', ''))
            if region:
                regions.add(region)
                if region not in molecular_by_region:
                    molecular_by_region[region] = {}
                gene = r.get('gene', r.get('gene_name', ''))
                expr = r.get('expression', r.get('value', 0))
                molecular_by_region[region][gene] = float(expr)

        morphological_by_region = {}
        for r in morphological_results:
            region = r.get('region', r.get('region_name', ''))
            if region:
                regions.add(region)
                if region not in morphological_by_region:
                    morphological_by_region[region] = {}
                feature = r.get('feature', r.get('feature_name', ''))
                value = r.get('value', 0)
                morphological_by_region[region][feature] = float(value)

        projection_by_region = {}
        for r in projection_results:
            source = r.get('source', r.get('source_region', ''))
            target = r.get('target', r.get('target_region', ''))
            strength = r.get('strength', r.get('weight', 0))

            if source:
                regions.add(source)
                if source not in projection_by_region:
                    projection_by_region[source] = {}
                projection_by_region[source][target] = float(strength)

        # 构建指纹
        fingerprints = []
        for region in sorted(regions):
            fp = self.build_trimodal_fingerprint(
                region_name=region,
                molecular_data=molecular_by_region.get(region),
                morphological_data=morphological_by_region.get(region),
                projection_data=projection_by_region.get(region)
            )
            fingerprints.append(fp)

        return fingerprints


# ==================== Export ====================

__all__ = [
    'SimilarityMetric',
    'ModalityFingerprint',
    'TriModalFingerprint',
    'SimilarityMatrix',
    'TriModalFingerprintBuilder'
]