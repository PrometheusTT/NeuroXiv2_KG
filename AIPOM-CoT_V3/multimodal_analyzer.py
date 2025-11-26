"""
Unified Multi-Modal Analyzer
=============================
统一的多模态分析器，解决代码重复问题

对齐Figure 2D: Scientific Operator Library
- Tri-modal fingerprint builder
- Similarity & mismatch metrics
- Statistical validation

Author: Claude & Lijun
Date: 2025-01-15
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, euclidean

from core_structures import (
    Modality,
    StatisticalEvidence,
    EvidenceRecord
)

logger = logging.getLogger(__name__)


# ==================== Fingerprint Structures ====================

@dataclass
class RegionFingerprint:
    """
    脑区指纹 - 对齐Figure 2D Tri-modal fingerprint
    """
    region: str
    molecular: Optional[np.ndarray] = None  # Subclass composition
    morphological: Optional[np.ndarray] = None  # Aggregated neuron features
    projection: Optional[np.ndarray] = None  # Target distribution

    # 元数据
    molecular_dim: int = 0
    morphological_dim: int = 0
    projection_dim: int = 0

    # 质量指标
    completeness: float = 0.0

    def is_valid(self) -> bool:
        """检查指纹是否有效"""
        return (self.molecular is not None and
                self.morphological is not None and
                self.projection is not None)

    def compute_completeness(self):
        """计算完整性"""
        valid_count = sum([
            self.molecular is not None,
            self.morphological is not None,
            self.projection is not None
        ])
        self.completeness = valid_count / 3.0


@dataclass
class MismatchResult:
    """
    跨模态Mismatch结果 - 对齐Figure 2D
    """
    region1: str
    region2: str

    # 相似度
    sim_molecular: float
    sim_morphological: float
    sim_projection: float

    # Mismatch指数
    mismatch_GM: float  # |sim_mol - sim_morph|
    mismatch_GP: float  # |sim_mol - sim_proj|
    mismatch_MP: float  # |sim_morph - sim_proj|
    mismatch_combined: float

    # 统计检验
    p_value: float = 1.0
    z_score: float = 0.0
    fdr_q: Optional[float] = None
    is_significant: bool = False


# ==================== Unified Fingerprint Analyzer ====================

class UnifiedFingerprintAnalyzer:
    """
    统一的多模态指纹分析器

    解决问题：
    1. 代码重复 - 统一到单一实现
    2. 一致性 - 统一的数据结构
    3. 缓存 - 避免重复计算
    """

    def __init__(self, db):
        self.db = db

        # 缓存
        self._fingerprint_cache: Dict[str, RegionFingerprint] = {}
        self._subclass_list: Optional[List[str]] = None
        self._target_list: Optional[List[str]] = None
        self._morph_standardized: bool = False
        self._morph_cache: Dict[str, np.ndarray] = {}

    # ==================== Main API ====================

    def get_fingerprint(self, region: str) -> Optional[RegionFingerprint]:
        """
        获取脑区的完整指纹

        Returns:
            RegionFingerprint对象
        """
        # 检查缓存
        if region in self._fingerprint_cache:
            return self._fingerprint_cache[region]

        # 计算各模态
        molecular = self._compute_molecular(region)
        morphological = self._compute_morphological(region)
        projection = self._compute_projection(region)

        # 构建指纹
        fp = RegionFingerprint(
            region=region,
            molecular=molecular,
            morphological=morphological,
            projection=projection,
            molecular_dim=len(molecular) if molecular is not None else 0,
            morphological_dim=len(morphological) if morphological is not None else 0,
            projection_dim=len(projection) if projection is not None else 0
        )
        fp.compute_completeness()

        # 缓存
        self._fingerprint_cache[region] = fp

        return fp

    def compute_similarity(self,
                           fp1: np.ndarray,
                           fp2: np.ndarray,
                           metric: str = 'cosine') -> float:
        """
        计算两个指纹向量的相似度

        Args:
            metric: 'cosine' 或 'correlation'
        """
        if fp1 is None or fp2 is None:
            return 0.0

        # 处理NaN
        mask = ~(np.isnan(fp1) | np.isnan(fp2))
        if mask.sum() < 2:
            return 0.0

        v1, v2 = fp1[mask], fp2[mask]

        if metric == 'cosine':
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        elif metric == 'correlation':
            if len(v1) < 2:
                return 0.0
            r, _ = stats.pearsonr(v1, v2)
            return float(r) if not np.isnan(r) else 0.0

        return 0.0

    def compute_mismatch(self,
                         region1: str,
                         region2: str) -> Optional[MismatchResult]:
        """
        计算两个脑区的跨模态Mismatch

        对齐Figure 4方法
        """
        fp1 = self.get_fingerprint(region1)
        fp2 = self.get_fingerprint(region2)

        if not fp1 or not fp2:
            return None

        if not fp1.is_valid() or not fp2.is_valid():
            return None

        # 计算各模态相似度
        sim_mol = self.compute_similarity(fp1.molecular, fp2.molecular)
        sim_morph = self.compute_similarity(fp1.morphological, fp2.morphological)
        sim_proj = self.compute_similarity(fp1.projection, fp2.projection)

        # 计算Mismatch
        mismatch_GM = abs(sim_mol - sim_morph)
        mismatch_GP = abs(sim_mol - sim_proj)
        mismatch_MP = abs(sim_morph - sim_proj)
        mismatch_combined = (mismatch_GM + mismatch_GP + mismatch_MP) / 3

        return MismatchResult(
            region1=region1,
            region2=region2,
            sim_molecular=sim_mol,
            sim_morphological=sim_morph,
            sim_projection=sim_proj,
            mismatch_GM=mismatch_GM,
            mismatch_GP=mismatch_GP,
            mismatch_MP=mismatch_MP,
            mismatch_combined=mismatch_combined
        )

    def compute_mismatch_matrix(self,
                                regions: List[str],
                                standardize_morphology: bool = True) -> List[MismatchResult]:
        """
        计算多个脑区的Mismatch矩阵

        Args:
            regions: 脑区列表
            standardize_morphology: 是否全局标准化形态数据

        Returns:
            MismatchResult列表（上三角）
        """
        logger.info(f"Computing mismatch matrix for {len(regions)} regions...")

        # Step 1: 预计算所有指纹
        fingerprints = {}
        valid_regions = []

        for region in regions:
            fp = self.get_fingerprint(region)
            if fp and fp.is_valid():
                fingerprints[region] = fp
                valid_regions.append(region)

        logger.info(f"  Valid fingerprints: {len(valid_regions)}/{len(regions)}")

        if len(valid_regions) < 2:
            return []

        # Step 2: 全局形态标准化
        if standardize_morphology:
            self._global_morphology_standardization(valid_regions, fingerprints)

        # Step 3: 构建距离矩阵
        n = len(valid_regions)
        mol_dist = np.zeros((n, n))
        morph_dist = np.zeros((n, n))
        proj_dist = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                fp1 = fingerprints[valid_regions[i]]
                fp2 = fingerprints[valid_regions[j]]

                # 分子距离 (cosine)
                try:
                    mol_dist[i, j] = cosine(fp1.molecular, fp2.molecular)
                except:
                    mol_dist[i, j] = np.nan

                # 形态距离 (Euclidean on z-scored)
                try:
                    mask = ~(np.isnan(fp1.morphological) | np.isnan(fp2.morphological))
                    if mask.sum() >= 4:
                        morph_dist[i, j] = euclidean(
                            fp1.morphological[mask],
                            fp2.morphological[mask]
                        )
                    else:
                        morph_dist[i, j] = np.nan
                except:
                    morph_dist[i, j] = np.nan

                # 投射距离 (cosine)
                try:
                    proj_dist[i, j] = cosine(fp1.projection, fp2.projection)
                except:
                    proj_dist[i, j] = np.nan

        # Step 4: Min-Max归一化
        mol_norm = self._minmax_normalize(mol_dist)
        morph_norm = self._minmax_normalize(morph_dist)
        proj_norm = self._minmax_normalize(proj_dist)

        # Step 5: 计算Mismatch
        results = []
        for i in range(n):
            for j in range(i + 1, n):
                mismatch_GM = abs(mol_norm[i, j] - morph_norm[i, j])
                mismatch_GP = abs(mol_norm[i, j] - proj_norm[i, j])
                mismatch_MP = abs(morph_norm[i, j] - proj_norm[i, j])
                mismatch_combined = (mismatch_GM + mismatch_GP + mismatch_MP) / 3

                results.append(MismatchResult(
                    region1=valid_regions[i],
                    region2=valid_regions[j],
                    sim_molecular=1 - mol_norm[i, j],
                    sim_morphological=1 - morph_norm[i, j],
                    sim_projection=1 - proj_norm[i, j],
                    mismatch_GM=mismatch_GM,
                    mismatch_GP=mismatch_GP,
                    mismatch_MP=mismatch_MP,
                    mismatch_combined=mismatch_combined
                ))

        # Step 6: 统计检验
        self._add_statistics(results)

        # 排序
        results.sort(key=lambda r: r.mismatch_combined, reverse=True)

        logger.info(f"  Computed {len(results)} mismatch pairs")

        return results

    # ==================== Internal: Fingerprint Computation ====================

    def _compute_molecular(self, region: str) -> Optional[np.ndarray]:
        """
        计算分子指纹 = Subclass组成

        使用: Region -[HAS_SUBCLASS]-> Subclass
        """
        query = """
        MATCH (r:Region {acronym: $acronym})-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
        ORDER BY sc.name
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # 构建字典
        data = {}
        for row in result['data']:
            name = row.get('subclass_name')
            pct = row.get('pct_cells')
            if name and pct is not None:
                data[name] = float(pct)

        if not data:
            return None

        # 获取全局subclass列表
        all_subclasses = self._get_all_subclasses()
        if not all_subclasses:
            return None

        # 构建向量
        vector = np.zeros(len(all_subclasses))
        for i, sc in enumerate(all_subclasses):
            if sc in data:
                vector[i] = data[sc]

        return vector

    def _compute_morphological(self, region: str) -> Optional[np.ndarray]:
        """
        计算形态指纹 = 8维特征向量

        从Region节点的聚合属性读取
        """
        query = """
        MATCH (r:Region {acronym: $acronym})
        RETURN
          r.axonal_bifurcation_remote_angle AS f1,
          r.axonal_length AS f2,
          r.axonal_branches AS f3,
          r.axonal_maximum_branch_order AS f4,
          r.dendritic_bifurcation_remote_angle AS f5,
          r.dendritic_length AS f6,
          r.dendritic_branches AS f7,
          r.dendritic_maximum_branch_order AS f8
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        record = result['data'][0]

        # 构建8维向量
        vector = np.array([
            record.get('f1') if record.get('f1') is not None else np.nan,
            record.get('f2') if record.get('f2') is not None else np.nan,
            record.get('f3') if record.get('f3') is not None else np.nan,
            record.get('f4') if record.get('f4') is not None else np.nan,
            record.get('f5') if record.get('f5') is not None else np.nan,
            record.get('f6') if record.get('f6') is not None else np.nan,
            record.get('f7') if record.get('f7') is not None else np.nan,
            record.get('f8') if record.get('f8') is not None else np.nan
        ], dtype=float)

        return vector

    def _compute_projection(self, region: str) -> Optional[np.ndarray]:
        """
        计算投射指纹 = 目标分布

        从Neuron级别聚合
        """
        query = """
        MATCH (r:Region {acronym: $acronym})
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        WITH DISTINCT n WHERE n IS NOT NULL
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        WITH t.acronym AS target, SUM(p.weight) AS total_weight
        RETURN target, total_weight
        ORDER BY total_weight DESC
        """

        result = self.db.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # 获取全局target列表
        all_targets = self._get_all_targets()
        if not all_targets:
            return None

        # 构建向量
        target_dict = {row['target']: row['total_weight'] for row in result['data']}
        raw_values = np.array([target_dict.get(t, 0.0) for t in all_targets])

        # Log稳定化 + 归一化
        log_values = np.log10(1 + raw_values)
        total = log_values.sum()
        if total > 0:
            vector = log_values / (total + 1e-9)
        else:
            vector = log_values

        return vector

    # ==================== Internal: Helper Methods ====================

    def _get_all_subclasses(self) -> List[str]:
        """获取所有subclass名称"""
        if self._subclass_list is not None:
            return self._subclass_list

        query = "MATCH (sc:Subclass) RETURN DISTINCT sc.name AS name ORDER BY name"
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._subclass_list = [row['name'] for row in result['data']]
        else:
            self._subclass_list = []

        return self._subclass_list

    def _get_all_targets(self) -> List[str]:
        """获取所有投射目标"""
        if self._target_list is not None:
            return self._target_list

        query = """
        MATCH ()-[:PROJECT_TO]->(t:Subregion)
        WHERE t.acronym IS NOT NULL
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        LIMIT 500
        """
        result = self.db.run(query)

        if result['success'] and result['data']:
            self._target_list = [row['target'] for row in result['data']]
        else:
            self._target_list = []

        return self._target_list

    def _global_morphology_standardization(self,
                                           regions: List[str],
                                           fingerprints: Dict[str, RegionFingerprint]):
        """
        全局Z-score标准化形态指纹

        对齐Ground Truth方法
        """
        logger.info("  Performing global morphology standardization...")

        # 收集所有形态数据
        all_morph = []
        valid_regions = []

        for region in regions:
            morph = fingerprints[region].morphological
            if morph is not None:
                all_morph.append(morph)
                valid_regions.append(region)

        if len(all_morph) < 2:
            return

        all_morph = np.array(all_morph)  # (N, 8)

        # 处理dendritic特征的0值 (索引4-7)
        for i in [4, 5, 6, 7]:
            col = all_morph[:, i].copy()
            zero_mask = np.abs(col) < 1e-6
            if zero_mask.sum() > 0:
                col[zero_mask] = np.nan
                all_morph[:, i] = col

        # Z-score标准化
        from scipy.stats import zscore
        for i in range(all_morph.shape[1]):
            col = all_morph[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                col[valid] = zscore(col[valid])
                all_morph[:, i] = col

        # 更新fingerprints
        for idx, region in enumerate(valid_regions):
            fingerprints[region].morphological = all_morph[idx]

        self._morph_standardized = True

    def _minmax_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Min-Max归一化到[0,1]"""
        valid = ~np.isnan(matrix)
        if valid.sum() == 0:
            return matrix

        vmin = matrix[valid].min()
        vmax = matrix[valid].max()

        if vmax - vmin < 1e-9:
            return np.zeros_like(matrix)

        normalized = (matrix - vmin) / (vmax - vmin)
        return normalized

    def _add_statistics(self, results: List[MismatchResult]):
        """为Mismatch结果添加统计检验"""
        if not results:
            return

        all_mismatches = [r.mismatch_combined for r in results]
        mean_m = np.mean(all_mismatches)
        std_m = np.std(all_mismatches)

        for result in results:
            m = result.mismatch_combined

            if std_m > 0:
                z_score = (m - mean_m) / std_m
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0

            result.z_score = z_score
            result.p_value = p_value

    def clear_cache(self):
        """清除缓存"""
        self._fingerprint_cache.clear()
        self._morph_cache.clear()
        self._morph_standardized = False


# ==================== Statistical Tools ====================

class StatisticalToolkit:
    """
    统计工具包 - 对齐Figure 2D Statistical Validation

    包含：
    - Permutation tests
    - FDR correction
    - Effect size (Cohen's d)
    - Bootstrap CI
    - Correlation tests
    """

    @staticmethod
    def permutation_test(observed_stat: float,
                         data1: np.ndarray,
                         data2: np.ndarray,
                         n_permutations: int = 1000,
                         seed: int = 42) -> StatisticalEvidence:
        """Permutation检验"""
        np.random.seed(seed)

        combined = np.concatenate([data1, data2])
        n1 = len(data1)

        null_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            null_stat = np.mean(combined[:n1]) - np.mean(combined[n1:])
            null_stats.append(null_stat)

        null_stats = np.array(null_stats)
        p_value = np.mean(np.abs(null_stats) >= np.abs(observed_stat))

        # Effect size
        effect_size = StatisticalToolkit.cohens_d(data1, data2)

        # Bootstrap CI for the difference
        ci = StatisticalToolkit.bootstrap_ci(
            np.concatenate([data1, data2]),
            lambda x: np.mean(x[:n1]) - np.mean(x[n1:])
        )

        return StatisticalEvidence(
            test_type='permutation',
            effect_size=effect_size,
            confidence_interval=ci,
            p_value=p_value,
            sample_size=len(combined),
            is_significant=p_value < 0.05
        )

    @staticmethod
    def fdr_correction(p_values: List[float],
                       alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """Benjamini-Hochberg FDR校正"""
        from statsmodels.stats.multitest import multipletests

        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        significant = q_values < alpha

        return q_values.tolist(), significant.tolist()

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d效应量"""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    @staticmethod
    def bootstrap_ci(data: np.ndarray,
                     statistic_func=np.mean,
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95,
                     seed: int = 42) -> Tuple[float, float]:
        """Bootstrap置信区间"""
        np.random.seed(seed)

        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            try:
                stat = statistic_func(sample)
                bootstrap_stats.append(stat)
            except:
                pass

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    @staticmethod
    def correlation_test(x: np.ndarray,
                         y: np.ndarray,
                         method: str = 'pearson') -> StatisticalEvidence:
        """相关性检验"""
        # 移除NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < 3:
            return StatisticalEvidence(
                test_type=method,
                effect_size=np.nan,
                p_value=1.0,
                sample_size=len(x),
                is_significant=False
            )

        if method == 'pearson':
            r, p = stats.pearsonr(x, y)
        else:
            r, p = stats.spearmanr(x, y)

        # Fisher's z CI
        n = len(x)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))

        return StatisticalEvidence(
            test_type=method,
            effect_size=r,
            confidence_interval=ci,
            p_value=p,
            sample_size=n,
            is_significant=p < 0.05
        )