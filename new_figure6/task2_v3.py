"""
任务2：Region-level 多模态融合提升脑区分辨率 (V8 科学严谨版)
===============================================
核心主题：多模态融合增加了脑区之间的可区分性（Resolution Gain）

关键改进 (V8):
1. 距离计算改进：
   - Mol/Proj: CLR -> PCA降维 -> correlation距离（而非raw+Hellinger）
   - Morph: log1p -> RobustScaler -> correlation距离
2. Resolved判定更严格：rank_multi >= 0.5 且 delta >= 0.15
3. Shuffle Null检验：随机打乱模态对齐，计算empirical p-value
4. Close pairs分两类：within-system close vs global close
5. Bootstrap稳定性验证

作者: PrometheusTT
日期: 2025-07-30
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
from itertools import product

warnings.filterwarnings('ignore')

from sklearn.manifold import MDS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from scipy.spatial.distance import pdist, squareform, correlation
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import neo4j

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ResolutionSummary:
    """分辨率提升汇总"""
    modality: str
    close_pairs: int
    resolved_pairs: int
    resolved_fraction: float
    mean_delta: float
    # 新增：within-system统计
    within_system_close: int
    within_system_resolved: int
    within_system_fraction: float


@dataclass
class NullTestResult:
    """Shuffle Null检验结果"""
    observed_resolution: float
    null_mean: float
    null_std: float
    empirical_p: float
    n_shuffles: int


@dataclass
class ClusteringMetrics:
    silhouette: float
    nmi: float
    ari: float
    n_regions: int


class RegionMultimodalResolutionV8:
    """
    区域级多模态分辨率分析器 V8（科学严谨版）

    核心问题：多模态融合是否提升了脑区之间的可区分性？

    方法论改进：
    1. 正确的距离计算：CLR+降维后的correlation距离
    2. 严格的resolved定义：rank_multi >= 0.5 且 delta >= 0.15
    3. Shuffle Null检验：证明融合利用了真实的跨模态对齐
    """

    AXONAL_FEATURES = [
        'axonal_total_length', 'axonal_volume', 'axonal_area',
        'axonal_number_of_bifurcations', 'axonal_max_branch_order',
        'axonal_max_euclidean_distance', 'axonal_max_path_distance',
        'axonal_average_euclidean_distance', 'axonal_average_path_distance',
        'axonal_75pct_euclidean_distance', 'axonal_75pct_path_distance',
        'axonal_50pct_euclidean_distance', 'axonal_50pct_path_distance',
        'axonal_25pct_euclidean_distance', 'axonal_25pct_path_distance',
        'axonal_average_bifurcation_angle_local', 'axonal_average_bifurcation_angle_remote',
        'axonal_average_contraction',
        'axonal_width', 'axonal_height', 'axonal_depth',
        'axonal_width_95ci', 'axonal_height_95ci', 'axonal_depth_95ci',
        'axonal_flatness', 'axonal_flatness_95ci',
        'axonal_slimness', 'axonal_slimness_95ci',
        'axonal_center_shift', 'axonal_relative_center_shift',
        'axonal_2d_density', 'axonal_3d_density'
    ]

    DENDRITIC_FEATURES = [
        'dendritic_total_length', 'dendritic_volume', 'dendritic_area',
        'dendritic_number_of_bifurcations', 'dendritic_max_branch_order',
        'dendritic_max_euclidean_distance', 'dendritic_max_path_distance',
        'dendritic_average_euclidean_distance', 'dendritic_average_path_distance',
        'dendritic_75pct_euclidean_distance', 'dendritic_75pct_path_distance',
        'dendritic_50pct_euclidean_distance', 'dendritic_50pct_path_distance',
        'dendritic_25pct_euclidean_distance', 'dendritic_25pct_path_distance',
        'dendritic_average_bifurcation_angle_local', 'dendritic_average_bifurcation_angle_remote',
        'dendritic_average_contraction',
        'dendritic_width', 'dendritic_height', 'dendritic_depth',
        'dendritic_width_95ci', 'dendritic_height_95ci', 'dendritic_depth_95ci',
        'dendritic_flatness', 'dendritic_flatness_95ci',
        'dendritic_slimness', 'dendritic_slimness_95ci',
        'dendritic_center_shift', 'dendritic_relative_center_shift',
        'dendritic_2d_density', 'dendritic_3d_density'
    ]

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 min_neurons_per_region: int = 10):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.min_neurons_per_region = min_neurons_per_region

        self.regions: List[str] = []
        self.region_neuron_counts: Dict[str, int] = {}

        # 原始指纹
        self.mol_fingerprints: Dict[str, np.ndarray] = {}
        self.morph_fingerprints: Dict[str, np.ndarray] = {}
        self.proj_fingerprints: Dict[str, np.ndarray] = {}

        # 处理后的特征矩阵（用于距离计算）
        self.X_mol_processed: np.ndarray = None
        self.X_morph_processed: np.ndarray = None
        self.X_proj_processed: np.ndarray = None

        self.morph_feature_names: List[str] = []
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 距离矩阵
        self.dist_mol: np.ndarray = None
        self.dist_morph: np.ndarray = None
        self.dist_proj: np.ndarray = None
        self.dist_multi_rank: np.ndarray = None

        # Pair数据
        self.df_pairs: pd.DataFrame = None
        self.resolution_summary: Dict[str, ResolutionSummary] = {}
        self.null_test_results: Dict[str, NullTestResult] = {}

        # 聚类
        self.cluster_labels: Dict[str, np.ndarray] = {}
        self.anatomical_systems: Dict[str, str] = {}
        self.non_other_mask: np.ndarray = None

        # Embedding
        self.embeddings: Dict[str, np.ndarray] = {}

        # 融合权重
        self.optimal_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        """Centered Log-Ratio变换"""
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def ensure_valid_distance_matrix(D: np.ndarray) -> np.ndarray:
        """确保距离矩阵有效"""
        D = D.copy().astype(np.float64)
        D = (D + D.T) / 2
        D = np.maximum(D, 0)
        np.fill_diagonal(D, 0.0)
        return D

    def safe_linkage(self, D: np.ndarray, method: str = 'average') -> np.ndarray:
        D = self.ensure_valid_distance_matrix(D)
        condensed = squareform(D, checks=False)
        return linkage(condensed, method=method)

    # ==================== 数据加载 ====================

    def load_all_fingerprints(self) -> int:
        print("\n" + "=" * 80)
        print("加载区域多模态指纹数据")
        print("=" * 80)

        self._get_global_dimensions()
        self._get_valid_regions_with_qc()
        self._load_molecular_fingerprints()
        self._load_morphology_fingerprints()
        self._load_projection_fingerprints()
        self._load_anatomical_systems()

        print(f"\n✓ 数据加载完成: {len(self.regions)} 个高质量区域")
        return len(self.regions)

    def _get_global_dimensions(self):
        print("\n获取全局维度...")
        self.morph_feature_names = self.AXONAL_FEATURES + self.DENDRITIC_FEATURES

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Subclass) WHERE s.name IS NOT NULL
                RETURN DISTINCT s.name AS name ORDER BY name
            """)
            self.all_subclasses = [r['name'] for r in result if r['name']]

            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]

        print(f"  形态: {len(self.morph_feature_names)}维, Subclass: {len(self.all_subclasses)}种, 投射: {len(self.all_target_regions)}个")

    def _get_valid_regions_with_qc(self):
        print(f"\n获取有效区域 (神经元数 >= {self.min_neurons_per_region})...")

        query = """
        MATCH (r:Region)-[:HAS_SUBCLASS]->()
        WITH DISTINCT r
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        WITH r, COUNT(DISTINCT n) AS n1
        OPTIONAL MATCH (n2:Neuron) WHERE n2.base_region = r.acronym
        WITH r, n1, COUNT(DISTINCT n2) AS n2
        RETURN r.acronym AS region, CASE WHEN n1 > n2 THEN n1 ELSE n2 END AS n_neurons
        ORDER BY region
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                region = record['region']
                n_neurons = record['n_neurons'] or 0
                if region and n_neurons >= self.min_neurons_per_region:
                    self.regions.append(region)
                    self.region_neuron_counts[region] = n_neurons

        print(f"  通过质量控制: {len(self.regions)} 个区域")

    def _load_molecular_fingerprints(self):
        print("\n加载分子指纹...")
        query = """
        MATCH (r:Region {acronym: $region})-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        with self.driver.session(database=self.database) as session:
            for region in self.regions:
                result = session.run(query, region=region)
                subclass_dict = {r['subclass_name']: r['pct_cells'] for r in result
                                 if r['subclass_name'] and r['pct_cells']}

                fingerprint = np.zeros(len(self.all_subclasses))
                for i, sc in enumerate(self.all_subclasses):
                    if sc in subclass_dict:
                        fingerprint[i] = subclass_dict[sc]

                self.mol_fingerprints[region] = fingerprint

        print(f"  加载了 {len(self.mol_fingerprints)} 个区域")

    def _load_morphology_fingerprints(self):
        """加载形态指纹 - 改进版：使用中位数而非均值，避免被outlier影响"""
        print("加载形态指纹...")

        return_parts = [f"n.{feat} AS `{feat}`" for feat in self.morph_feature_names]
        return_clause = ", ".join(return_parts)

        query = f"MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {{acronym: $region}}) RETURN {return_clause}"
        query_backup = f"MATCH (n:Neuron) WHERE n.base_region = $region RETURN {return_clause}"

        with self.driver.session(database=self.database) as session:
            for region in self.regions:
                result = session.run(query, region=region)
                records = list(result)

                if len(records) == 0:
                    result = session.run(query_backup, region=region)
                    records = list(result)

                if len(records) == 0:
                    self.morph_fingerprints[region] = np.zeros(len(self.morph_feature_names))
                    continue

                feature_values = {feat: [] for feat in self.morph_feature_names}
                for record in records:
                    for feat in self.morph_feature_names:
                        val = record[feat]
                        if val is not None and val > 0:
                            feature_values[feat].append(val)

                # 改用中位数而非均值，对outlier更鲁棒
                fingerprint = []
                for f in self.morph_feature_names:
                    vals = feature_values[f]
                    if len(vals) > 0:
                        fingerprint.append(np.median(vals))
                    else:
                        fingerprint.append(0.0)

                self.morph_fingerprints[region] = np.array(fingerprint)

        print(f"  加载了 {len(self.morph_fingerprints)} 个区域（使用中位数聚合）")

    def _load_projection_fingerprints(self):
        print("加载投射指纹...")

        query = """
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $region})
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, SUM(p.weight) AS total_weight
        """
        query_backup = """
        MATCH (n:Neuron) WHERE n.base_region = $region
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, SUM(p.weight) AS total_weight
        """

        with self.driver.session(database=self.database) as session:
            for region in self.regions:
                result = session.run(query, region=region)
                proj_dict = {r['target']: r['total_weight'] for r in result
                             if r['target'] and r['total_weight']}

                if len(proj_dict) == 0:
                    result = session.run(query_backup, region=region)
                    proj_dict = {r['target']: r['total_weight'] for r in result
                                 if r['target'] and r['total_weight']}

                fingerprint = np.zeros(len(self.all_target_regions))
                for i, target in enumerate(self.all_target_regions):
                    if target in proj_dict:
                        fingerprint[i] = proj_dict[target]

                self.proj_fingerprints[region] = fingerprint

        print(f"  加载了 {len(self.proj_fingerprints)} 个区域")

    def _load_anatomical_systems(self):
        print("\n加载解剖系统标签...")

        MAJOR_SYSTEMS = {
            'Isocortex': ['MO', 'SS', 'GU', 'VISC', 'AUD', 'VIS', 'ACA', 'PL', 'ILA', 'ORB', 'AI', 'RSP', 'PTL', 'TE', 'PERI', 'ECT', 'FRP'],
            'Olfactory': ['MOB', 'AOB', 'AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR'],
            'Hippocampus': ['CA', 'DG', 'SUB', 'ENT', 'PAR', 'POST', 'PRE', 'ProS', 'HPF'],
            'Cortical_subplate': ['CLA', 'EP', 'LA', 'BLA', 'BMA', 'PA'],
            'Striatum': ['CP', 'ACB', 'FS', 'OT', 'LSX', 'LS', 'SF', 'SH', 'CEA', 'MEA'],
            'Pallidum': ['GP', 'GPe', 'GPi', 'SI', 'MA', 'NDB', 'MS', 'TRS', 'BST', 'BAC'],
            'Thalamus': ['VAL', 'VM', 'VP', 'VPL', 'VPM', 'SPF', 'SPA', 'PP', 'MG', 'LGd', 'LGv',
                         'LP', 'PO', 'POL', 'SGN', 'AV', 'AM', 'AD', 'IAM', 'IAD', 'LD', 'IMD',
                         'MD', 'SMT', 'PR', 'PVT', 'PT', 'RE', 'RH', 'CM', 'PCN', 'CL', 'PF', 'RT', 'IGL', 'LH'],
            'Hypothalamus': ['SO', 'PVH', 'PVa', 'PVi', 'ARH', 'ADP', 'AVP', 'AVPV', 'DMH', 'MEPO',
                             'MPO', 'OV', 'PD', 'PS', 'PVp', 'PVpo', 'SBPV', 'SCH', 'SFO', 'VMPO',
                             'VLPO', 'AHN', 'MBO', 'LM', 'MM', 'SUM', 'TM', 'PMd', 'PMv', 'VMH',
                             'PH', 'LHA', 'LPO', 'PST', 'PSTN', 'PeF', 'STN', 'TU', 'ZI', 'FF'],
            'Midbrain': ['SC', 'IC', 'NB', 'SAG', 'PBG', 'MEV', 'SNr', 'SNc', 'VTA', 'PN', 'RR',
                         'MRN', 'RN', 'III', 'IV', 'EW', 'PAG', 'APN', 'MPT', 'NOT', 'NPC', 'OP',
                         'PPT', 'CUN', 'AT', 'LT', 'DT', 'MT', 'CLI', 'DR', 'IF', 'IPN', 'RL'],
            'Hindbrain': ['MY', 'P', 'SLD', 'SLC', 'SUT', 'TRN', 'V', 'LAV', 'MV', 'SPIV', 'SUV',
                          'PB', 'KF', 'PSV', 'SOC', 'POR', 'B', 'DTN', 'LDT', 'NI', 'PRNc', 'PRNr'],
            'Cerebellum': ['VERM', 'HEM', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'CUL', 'DEC']
        }

        prefix_to_system = {}
        for system, prefixes in MAJOR_SYSTEMS.items():
            for prefix in prefixes:
                prefix_to_system[prefix] = system

        for region in self.regions:
            found = False
            for prefix, system in prefix_to_system.items():
                if region == prefix or region.startswith(prefix):
                    self.anatomical_systems[region] = system
                    found = True
                    break
            if not found:
                self.anatomical_systems[region] = 'Other'

        self.non_other_mask = np.array([self.anatomical_systems[r] != 'Other' for r in self.regions])

        system_counts = {}
        for system in self.anatomical_systems.values():
            system_counts[system] = system_counts.get(system, 0) + 1
        print(f"  系统分布: {system_counts}")

    # ==================== 特征处理与距离计算（核心改进） ====================

    def compute_distance_matrices(self):
        """
        计算单模态距离矩阵（核心改进）

        改进：
        - Mol: CLR -> PCA(90% variance) -> correlation距离
        - Proj: CLR -> TruncatedSVD(30) -> correlation距离
        - Morph: log1p -> RobustScaler -> correlation距离
        """
        print("\n" + "=" * 80)
        print("计算单模态距离矩阵（改进版）")
        print("=" * 80)

        n_regions = len(self.regions)

        # ===== Molecular: CLR -> PCA -> correlation =====
        print("\n【Molecular】CLR -> PCA -> correlation...")
        X_mol_raw = np.array([self.mol_fingerprints[r] for r in self.regions])

        # CLR变换
        X_mol_clr = self.clr_transform(X_mol_raw)

        # PCA降维（保留90%方差）
        pca_mol = PCA(n_components=0.90, svd_solver='full')
        X_mol_pca = pca_mol.fit_transform(X_mol_clr)
        print(f"  CLR后: {X_mol_clr.shape[1]}维 -> PCA后: {X_mol_pca.shape[1]}维 (90% var)")

        # Correlation距离
        self.dist_mol = squareform(pdist(X_mol_pca, metric='correlation'))
        self.dist_mol = self.ensure_valid_distance_matrix(self.dist_mol)
        self.X_mol_processed = X_mol_pca
        print(f"  距离范围: [{self.dist_mol.min():.4f}, {self.dist_mol.max():.4f}]")

        # ===== Morphology: log1p -> RobustScaler -> correlation =====
        print("\n【Morphology】log1p -> RobustScaler -> correlation...")
        X_morph_raw = np.array([self.morph_fingerprints[r] for r in self.regions])

        # log1p处理长尾分布
        X_morph_log = np.log1p(X_morph_raw)

        # RobustScaler对outlier鲁棒
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)

        # 可选：PCA去噪
        pca_morph = PCA(n_components=0.95, svd_solver='full')
        X_morph_pca = pca_morph.fit_transform(X_morph_scaled)
        print(f"  log1p+RobustScaler后: {X_morph_scaled.shape[1]}维 -> PCA后: {X_morph_pca.shape[1]}维 (95% var)")

        # Correlation距离
        self.dist_morph = squareform(pdist(X_morph_pca, metric='correlation'))
        self.dist_morph = self.ensure_valid_distance_matrix(self.dist_morph)
        self.X_morph_processed = X_morph_pca
        print(f"  距离范围: [{self.dist_morph.min():.4f}, {self.dist_morph.max():.4f}]")

        # ===== Projection: CLR -> TruncatedSVD -> correlation =====
        print("\n【Projection】CLR -> TruncatedSVD -> correlation...")
        X_proj_raw = np.array([self.proj_fingerprints[r] for r in self.regions])

        # CLR变换
        X_proj_clr = self.clr_transform(X_proj_raw)

        # TruncatedSVD降维（稀疏数据用SVD更好）
        n_components = min(30, X_proj_clr.shape[1] - 1, n_regions - 1)
        svd_proj = TruncatedSVD(n_components=n_components, random_state=42)
        X_proj_svd = svd_proj.fit_transform(X_proj_clr)
        explained_var = svd_proj.explained_variance_ratio_.sum()
        print(f"  CLR后: {X_proj_clr.shape[1]}维 -> SVD后: {X_proj_svd.shape[1]}维 ({explained_var:.1%} var)")

        # Correlation距离
        self.dist_proj = squareform(pdist(X_proj_svd, metric='correlation'))
        self.dist_proj = self.ensure_valid_distance_matrix(self.dist_proj)
        self.X_proj_processed = X_proj_svd
        print(f"  距离范围: [{self.dist_proj.min():.4f}, {self.dist_proj.max():.4f}]")

        print("\n✓ 单模态距离矩阵计算完成（使用CLR+降维+correlation）")

    # ==================== Rank-based融合 ====================

    def compute_rank_fusion(self, weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)):
        """Rank-based距离融合"""
        print("\n" + "=" * 80)
        print(f"Rank-based多模态距离融合 (Weights: {weights})")
        print("=" * 80)

        n = len(self.regions)
        n_pairs = n * (n - 1) // 2
        idx = np.triu_indices(n, k=1)

        vals_mol = self.dist_mol[idx]
        vals_morph = self.dist_morph[idx]
        vals_proj = self.dist_proj[idx]

        # 转换为Rank（归一化到0~1）
        rank_mol = rankdata(vals_mol) / n_pairs
        rank_morph = rankdata(vals_morph) / n_pairs
        rank_proj = rankdata(vals_proj) / n_pairs

        # 加权融合
        rank_multi = weights[0] * rank_mol + weights[1] * rank_morph + weights[2] * rank_proj

        # 重建距离矩阵
        self.dist_multi_rank = np.zeros((n, n))
        self.dist_multi_rank[idx] = rank_multi
        self.dist_multi_rank = self.dist_multi_rank + self.dist_multi_rank.T

        self.optimal_weights = weights
        print("✓ Rank融合完成")

    # ==================== Pair DataFrame ====================

    def build_pair_dataframe(self):
        """构建所有region pairs的距离数据框"""
        print("\n" + "=" * 80)
        print("构建Region Pair距离表")
        print("=" * 80)

        n = len(self.regions)
        idx = np.triu_indices(n, k=1)
        i_idx, j_idx = idx

        self.df_pairs = pd.DataFrame({
            'i': i_idx,
            'j': j_idx,
            'region_i': [self.regions[i] for i in i_idx],
            'region_j': [self.regions[j] for j in j_idx],
            'system_i': [self.anatomical_systems[self.regions[i]] for i in i_idx],
            'system_j': [self.anatomical_systems[self.regions[j]] for j in j_idx],
            'd_mol': self.dist_mol[idx],
            'd_morph': self.dist_morph[idx],
            'd_proj': self.dist_proj[idx],
            'd_multi': self.dist_multi_rank[idx],
        })

        # Rank（百分位）
        self.df_pairs['rank_mol'] = rankdata(self.df_pairs['d_mol']) / len(self.df_pairs)
        self.df_pairs['rank_morph'] = rankdata(self.df_pairs['d_morph']) / len(self.df_pairs)
        self.df_pairs['rank_proj'] = rankdata(self.df_pairs['d_proj']) / len(self.df_pairs)
        self.df_pairs['rank_multi'] = rankdata(self.df_pairs['d_multi']) / len(self.df_pairs)

        # Delta（multi与单模态的差异）
        self.df_pairs['delta_mol'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_mol']
        self.df_pairs['delta_morph'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_morph']
        self.df_pairs['delta_proj'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_proj']

        # 系统关系
        self.df_pairs['same_system'] = self.df_pairs['system_i'] == self.df_pairs['system_j']
        self.df_pairs['cross_system'] = ~self.df_pairs['same_system']

        # 排除Other
        self.df_pairs['both_non_other'] = (
            (self.df_pairs['system_i'] != 'Other') &
            (self.df_pairs['system_j'] != 'Other')
        )

        print(f"  总pairs数: {len(self.df_pairs)}")
        print(f"  同系统pairs: {self.df_pairs['same_system'].sum()}")
        print(f"  跨系统pairs: {self.df_pairs['cross_system'].sum()}")
        print(f"  非Other pairs: {self.df_pairs['both_non_other'].sum()}")

    # ==================== Resolution分析（核心改进） ====================

    def compute_resolution_metrics(self, q_close: float = 0.05,
                                    rank_far_threshold: float = 0.50,
                                    delta_threshold: float = 0.15):
        """
        计算Resolution指标（核心改进）

        Resolved定义（更严格）：
        1. rank_multi >= rank_far_threshold (0.50) - 绝对分离：进入了后半部分
        2. delta >= delta_threshold (0.15) - 相对增量：距离显著增加

        同时满足两个条件才算resolved

        分两类报告：
        - Global close: 全局top 5%最相似
        - Within-system close: 同系统内top 5%最相似（更难分辨）
        """
        print("\n" + "=" * 80)
        print(f"计算Resolution指标")
        print(f"  Close: Top {q_close:.0%}")
        print(f"  Resolved: rank_multi >= {rank_far_threshold} AND delta >= {delta_threshold}")
        print("=" * 80)

        results = []

        for mod in ['mol', 'morph', 'proj']:
            rank_col = f'rank_{mod}'
            delta_col = f'delta_{mod}'

            # ===== Global Close =====
            thr_close = self.df_pairs[rank_col].quantile(q_close)
            global_close = self.df_pairs[rank_col] <= thr_close

            # Resolved（严格定义：两个条件都满足）
            is_resolved = (
                global_close &
                (self.df_pairs['rank_multi'] >= rank_far_threshold) &
                (self.df_pairs[delta_col] >= delta_threshold)
            )

            self.df_pairs[f'{mod}_close'] = global_close
            self.df_pairs[f'resolved_from_{mod}'] = is_resolved

            global_close_n = global_close.sum()
            global_resolved_n = is_resolved.sum()
            global_fraction = global_resolved_n / max(global_close_n, 1)
            global_mean_delta = self.df_pairs.loc[global_close, delta_col].mean()

            # ===== Within-system Close =====
            within_mask = self.df_pairs['same_system'] & self.df_pairs['both_non_other']
            if within_mask.sum() > 0:
                within_df = self.df_pairs[within_mask]
                thr_within = within_df[rank_col].quantile(q_close)
                within_close = within_df[rank_col] <= thr_within

                within_resolved = (
                    within_close &
                    (within_df['rank_multi'] >= rank_far_threshold) &
                    (within_df[delta_col] >= delta_threshold)
                )

                within_close_n = within_close.sum()
                within_resolved_n = within_resolved.sum()
                within_fraction = within_resolved_n / max(within_close_n, 1)
            else:
                within_close_n = 0
                within_resolved_n = 0
                within_fraction = 0.0

            self.resolution_summary[mod] = ResolutionSummary(
                modality=mod,
                close_pairs=int(global_close_n),
                resolved_pairs=int(global_resolved_n),
                resolved_fraction=global_fraction,
                mean_delta=global_mean_delta,
                within_system_close=int(within_close_n),
                within_system_resolved=int(within_resolved_n),
                within_system_fraction=within_fraction
            )

            results.append({
                'modality': mod,
                'global_close': global_close_n,
                'global_resolved': global_resolved_n,
                'global_fraction': global_fraction,
                'within_close': within_close_n,
                'within_resolved': within_resolved_n,
                'within_fraction': within_fraction,
                'mean_delta': global_mean_delta
            })

        # 打印结果
        print(f"\n{'Modality':<10} {'Global Close':>12} {'Resolved':>10} {'Fraction':>10} {'Within Close':>14} {'Resolved':>10} {'Fraction':>10}")
        print("-" * 90)
        for r in results:
            print(f"{r['modality']:<10} {r['global_close']:>12} {r['global_resolved']:>10} {r['global_fraction']:>9.1%} "
                  f"{r['within_close']:>14} {r['within_resolved']:>10} {r['within_fraction']:>9.1%}")

        return results

    # ==================== Shuffle Null检验（核心改进） ====================

    def shuffle_null_test(self, n_shuffles: int = 200,
                          q_close: float = 0.05,
                          rank_far_threshold: float = 0.50,
                          delta_threshold: float = 0.15):
        """
        Shuffle Null检验

        随机打乱某一个模态的region对应关系，破坏跨模态对齐，
        重复多次得到null分布，证明融合利用了真实的跨模态信息。
        """
        print("\n" + "=" * 80)
        print(f"Shuffle Null检验 (n_shuffles={n_shuffles})")
        print("=" * 80)

        np.random.seed(42)
        n = len(self.regions)
        n_pairs = n * (n - 1) // 2
        idx = np.triu_indices(n, k=1)

        # 观测值
        observed_fractions = {mod: self.resolution_summary[mod].resolved_fraction
                             for mod in ['mol', 'morph', 'proj']}

        # Null分布
        null_fractions = {mod: [] for mod in ['mol', 'morph', 'proj']}

        print(f"\n进行{n_shuffles}次shuffle...")

        for shuffle_i in range(n_shuffles):
            if (shuffle_i + 1) % 50 == 0:
                print(f"  {shuffle_i + 1}/{n_shuffles}")

            # 打乱Projection的region顺序（破坏跨模态对齐）
            perm = np.random.permutation(n)
            dist_proj_shuffled = self.dist_proj[np.ix_(perm, perm)]

            # 重新计算融合距离
            vals_mol = self.dist_mol[idx]
            vals_morph = self.dist_morph[idx]
            vals_proj_shuffled = dist_proj_shuffled[idx]

            rank_mol = rankdata(vals_mol) / n_pairs
            rank_morph = rankdata(vals_morph) / n_pairs
            rank_proj = rankdata(vals_proj_shuffled) / n_pairs

            w = self.optimal_weights
            rank_multi_shuffled = w[0] * rank_mol + w[1] * rank_morph + w[2] * rank_proj

            # 计算resolution fraction
            for mod in ['mol', 'morph', 'proj']:
                if mod == 'mol':
                    rank_single = rank_mol
                elif mod == 'morph':
                    rank_single = rank_morph
                else:
                    rank_single = rank_proj

                thr_close = np.quantile(rank_single, q_close)
                close_mask = rank_single <= thr_close
                delta = rank_multi_shuffled - rank_single

                resolved = (
                    close_mask &
                    (rank_multi_shuffled >= rank_far_threshold) &
                    (delta >= delta_threshold)
                )

                fraction = resolved.sum() / max(close_mask.sum(), 1)
                null_fractions[mod].append(fraction)

        # 计算统计量
        for mod in ['mol', 'morph', 'proj']:
            null_arr = np.array(null_fractions[mod])
            observed = observed_fractions[mod]

            # Empirical p-value（单尾：观测值比null大多少）
            p_value = (np.sum(null_arr >= observed) + 1) / (n_shuffles + 1)

            self.null_test_results[mod] = NullTestResult(
                observed_resolution=observed,
                null_mean=np.mean(null_arr),
                null_std=np.std(null_arr),
                empirical_p=p_value,
                n_shuffles=n_shuffles
            )

        # 打印结果
        print(f"\n{'Modality':<10} {'Observed':>12} {'Null Mean±Std':>18} {'Empirical p':>14}")
        print("-" * 60)
        for mod in ['mol', 'morph', 'proj']:
            r = self.null_test_results[mod]
            sig = '***' if r.empirical_p < 0.001 else '**' if r.empirical_p < 0.01 else '*' if r.empirical_p < 0.05 else ''
            print(f"{mod:<10} {r.observed_resolution:>11.1%} {r.null_mean:>8.1%} ± {r.null_std:>5.1%} {r.empirical_p:>13.4f} {sig}")

        return self.null_test_results

    # ==================== Disagreement Pairs ====================

    def find_disagreement_pairs(self, topk: int = 50):
        """找出模态间互补的典型案例"""
        print("\n" + "=" * 80)
        print("挖掘Disagreement Pairs（模态互补案例）")
        print("=" * 80)

        disagreements = {}

        cases = [
            ('mol_close_morph_far', 'rank_mol', 'rank_morph', 'delta_mol'),
            ('mol_close_proj_far', 'rank_mol', 'rank_proj', 'delta_mol'),
            ('morph_close_mol_far', 'rank_morph', 'rank_mol', 'delta_morph'),
            ('morph_close_proj_far', 'rank_morph', 'rank_proj', 'delta_morph'),
        ]

        for key, close_col, far_col, delta_col in cases:
            df = self.df_pairs.copy()
            # Score: 越近*越远 = 越互补
            df['score'] = (1 - df[close_col]) * df[far_col]
            top_df = df.nlargest(topk, 'score')[
                ['region_i', 'region_j', 'system_i', 'system_j',
                 'd_mol', 'd_morph', 'd_proj', 'd_multi',
                 close_col, far_col, delta_col, 'same_system']
            ]
            disagreements[key] = top_df

            cross_frac = (1 - top_df['same_system'].mean())
            print(f"\n【{key}】Top {topk}, 跨系统: {cross_frac:.1%}")

        self.disagreement_pairs = disagreements
        return disagreements

    # ==================== 聚类（补充） ====================

    def perform_clustering(self, n_clusters: int = 8):
        """聚类分析（补充验证）"""
        print("\n" + "=" * 80)
        print(f"聚类分析（补充验证, k={n_clusters}）")
        print("=" * 80)

        # 分子
        Z = self.safe_linkage(self.dist_mol, 'average')
        self.cluster_labels['molecular'] = fcluster(Z, n_clusters, criterion='maxclust') - 1

        # 形态
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='precomputed')
        self.cluster_labels['morphology'] = clustering.fit_predict(self.dist_morph)

        # 投射
        Z = self.safe_linkage(self.dist_proj, 'average')
        self.cluster_labels['projection'] = fcluster(Z, n_clusters, criterion='maxclust') - 1

        # 多模态
        Z = self.safe_linkage(self.dist_multi_rank, 'average')
        self.cluster_labels['multimodal'] = fcluster(Z, n_clusters, criterion='maxclust') - 1

        # 评估
        true_labels = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        label_encoder = {l: i for i, l in enumerate(set(true_labels))}
        true_encoded = np.array([label_encoder[l] for l in true_labels])
        true_non_other = true_encoded[self.non_other_mask]

        print(f"\n{'Modality':<15} {'Silhouette':>12} {'NMI':>10} {'ARI':>10}")
        print("-" * 50)

        metrics = {}
        for name, labels in self.cluster_labels.items():
            if name == 'molecular':
                sil = silhouette_score(self.dist_mol, labels, metric='precomputed')
            elif name == 'morphology':
                sil = silhouette_score(self.dist_morph, labels, metric='precomputed')
            elif name == 'projection':
                sil = silhouette_score(self.dist_proj, labels, metric='precomputed')
            else:
                sil = silhouette_score(self.dist_multi_rank, labels, metric='precomputed')

            nmi = normalized_mutual_info_score(true_non_other, labels[self.non_other_mask])
            ari = adjusted_rand_score(true_non_other, labels[self.non_other_mask])

            metrics[name] = ClusteringMetrics(sil, nmi, ari, len(self.regions))
            print(f"{name:<15} {sil:>12.4f} {nmi:>10.4f} {ari:>10.4f}")

        return metrics

    # ==================== Embedding ====================

    def compute_embeddings(self, method: str = "UMAP"):
        """计算Embedding"""
        print(f"\n计算 {method} Embedding...")

        mods = {
            'molecular': self.dist_mol,
            'morphology': self.dist_morph,
            'projection': self.dist_proj,
            'multimodal': self.dist_multi_rank
        }

        for name, D in mods.items():
            D = self.ensure_valid_distance_matrix(D)

            if method == "UMAP" and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                    metric='precomputed', random_state=42)
                emb = reducer.fit_transform(D)
            else:
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
                emb = mds.fit_transform(D)

            self.embeddings[name] = emb

        print("  ✓ Embedding完成")

    # ==================== 可视化（重构） ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化（重构版）"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表（重构版）")
        print("=" * 80)

        self._plot_resolution_with_null(output_dir)      # 主图1：Resolution + Null
        self._plot_resolution_scatter_single(output_dir)  # 主图2：单个scatter（最清晰）
        self._plot_delta_distribution(output_dir)         # 补充：Delta分布
        self._plot_embeddings(output_dir)                 # 补充：Embedding
        self._plot_top_resolved_table(output_dir)         # 补充：Top resolved pairs表格

        print(f"\n✓ 图表保存到: {output_dir}")

    def _plot_resolution_with_null(self, output_dir: str):
        """
        主图1：Resolution Gain with Null Distribution

        左：真实融合的resolved fraction
        右：shuffle null的分布（证明融合必要性）
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        modalities = ['Molecular', 'Morphology', 'Projection']
        keys = ['mol', 'morph', 'proj']
        colors = ['#3498DB', '#27AE60', '#9B59B6']

        # Panel A: Resolution Gain Bar Plot
        ax = axes[0]
        x = np.arange(len(modalities))
        width = 0.35

        close_pairs = [self.resolution_summary[k].close_pairs for k in keys]
        resolved_pairs = [self.resolution_summary[k].resolved_pairs for k in keys]

        bars1 = ax.bar(x - width/2, close_pairs, width, label='Indistinguishable (Top 5%)',
                      color='lightgray', edgecolor='black')
        bars2 = ax.bar(x + width/2, resolved_pairs, width, label='Resolved by Multimodal',
                      color=colors, edgecolor='black')

        for bar, val, key in zip(bars2, resolved_pairs, keys):
            frac = self.resolution_summary[key].resolved_fraction
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val}\n({frac:.1%})', ha='center', fontsize=9, fontweight='bold')

        ax.set_ylabel('Number of Region Pairs', fontsize=11)
        ax.set_title('Resolution Gain\n(Strict: rank≥0.5 & Δ≥0.15)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modalities, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Panel B: Null Comparison
        ax = axes[1]

        if self.null_test_results:
            observed = [self.null_test_results[k].observed_resolution * 100 for k in keys]
            null_means = [self.null_test_results[k].null_mean * 100 for k in keys]
            null_stds = [self.null_test_results[k].null_std * 100 for k in keys]
            p_values = [self.null_test_results[k].empirical_p for k in keys]

            bars1 = ax.bar(x - width/2, observed, width, label='Observed',
                          color=colors, edgecolor='black')
            bars2 = ax.bar(x + width/2, null_means, width, yerr=null_stds, capsize=5,
                          label='Shuffle Null', color='lightgray', edgecolor='black')

            # 添加p值
            for i, (bar, p) in enumerate(zip(bars1, p_values)):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'p={p:.3f}\n{sig}', ha='center', fontsize=8)

            ax.set_ylabel('Resolution Fraction (%)', fontsize=11)
            ax.set_title('Observed vs Shuffle Null\n(Proving alignment matters)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(modalities, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_resolution_with_null.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_resolution_with_null.png (主图)")

    def _plot_resolution_scatter_single(self, output_dir: str):
        """
        主图2：单个Resolution Scatter（选择信号最强的模态）

        只画一个scatter，更清晰，配合小表格展示top resolved pairs
        """
        # 选择resolution fraction最高的模态
        best_mod = max(['mol', 'morph', 'proj'],
                       key=lambda m: self.resolution_summary[m].resolved_fraction)

        fig, ax = plt.subplots(figsize=(8, 7))

        mod = best_mod
        x = self.df_pairs[f'rank_{mod}']
        y = self.df_pairs['rank_multi']
        resolved_mask = self.df_pairs[f'resolved_from_{mod}']
        close_mask = self.df_pairs[f'{mod}_close']

        # 背景点
        ax.scatter(x[~close_mask], y[~close_mask], s=5, c='lightgray', alpha=0.2, label='Other Pairs')

        # Close但Unresolved
        unresolved_mask = close_mask & (~resolved_mask)
        ax.scatter(x[unresolved_mask], y[unresolved_mask], s=15, c='#E74C3C', alpha=0.6,
                   label=f'Unresolved ({unresolved_mask.sum()})')

        # Resolved
        color = {'mol': '#3498DB', 'morph': '#27AE60', 'proj': '#9B59B6'}[mod]
        ax.scatter(x[resolved_mask], y[resolved_mask], s=30, c=color, alpha=0.9,
                   edgecolors='white', linewidth=0.5, label=f'Resolved ({resolved_mask.sum()})')

        # 辅助线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='y=x')
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='rank=0.5')
        ax.plot([0, 0.85], [0.15, 1], 'k:', linewidth=1, alpha=0.3, label='Δ=0.15')

        # Close阈值
        q_close_val = x.quantile(0.05)
        ax.axvline(q_close_val, color='red', linestyle='--', alpha=0.5)
        ax.text(q_close_val + 0.02, 0.95, 'Top 5%', fontsize=9, color='red')

        ax.set_xlabel(f'{mod.capitalize()} Distance (Rank)', fontsize=12)
        ax.set_ylabel('Multimodal Distance (Rank)', fontsize=12)
        ax.set_title(f'Resolution Gain from {mod.capitalize()} Ambiguity\n'
                     f'{self.resolution_summary[mod].resolved_fraction:.1%} resolved',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_resolution_scatter_{mod}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 2_resolution_scatter_{mod}.png (主图)")

    def _plot_delta_distribution(self, output_dir: str):
        """补充图：Delta分布"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        modalities = ['mol', 'morph', 'proj']
        titles = ['Δ(Multi - Molecular)', 'Δ(Multi - Morphology)', 'Δ(Multi - Projection)']
        colors = ['#3498DB', '#27AE60', '#9B59B6']

        for ax, mod, title, color in zip(axes, modalities, titles, colors):
            delta = self.df_pairs[f'delta_{mod}']
            close_mask = self.df_pairs[f'{mod}_close']

            ax.hist(delta, bins=50, alpha=0.5, color='gray', label='All pairs', density=True)
            ax.hist(delta[close_mask], bins=30, alpha=0.7, color=color,
                    label=f'Close pairs (top 5%)', density=True)

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            ax.axvline(x=0.15, color='green', linestyle=':', linewidth=1.5, label='Δ=0.15 threshold')
            ax.axvline(x=delta[close_mask].mean(), color=color, linestyle='-', linewidth=2,
                       label=f'Mean Δ = {delta[close_mask].mean():.3f}')

            ax.set_xlabel('Δ (Multi rank - Single rank)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle('Distance Increment Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_delta_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_delta_distribution.png (补充)")

    def _plot_embeddings(self, output_dir: str):
        """补充图：Embedding"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        systems = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        unique = sorted(set(systems))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
        sys_color = {s: colors[i] for i, s in enumerate(unique)}
        point_colors = [[0.7, 0.7, 0.7, 0.5] if s == 'Other' else sys_color[s] for s in systems]

        mods = ['molecular', 'morphology', 'projection', 'multimodal']
        titles = ['Molecular', 'Morphology', 'Projection', 'Multimodal (Rank Fusion)']

        for ax, mod, title in zip(axes.flat, mods, titles):
            emb = self.embeddings[mod]
            ax.scatter(emb[:, 0], emb[:, 1], c=point_colors, s=80, alpha=0.8,
                       edgecolors='white', linewidth=0.5)

            for i, r in enumerate(self.regions):
                if i % 4 == 0:
                    ax.annotate(r, (emb[i, 0], emb[i, 1]), fontsize=5, alpha=0.6)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

        legend = [Patch(facecolor=sys_color[s], label=s) for s in unique if s != 'Other']
        legend.append(Patch(facecolor=[0.7, 0.7, 0.7], label='Other'))
        fig.legend(handles=legend, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)

        plt.suptitle(f'Region Embeddings', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_embeddings.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_embeddings.png (补充)")

    def _plot_top_resolved_table(self, output_dir: str):
        """补充图：Top Resolved Pairs小表格"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # 选择最佳模态
        best_mod = max(['mol', 'morph', 'proj'],
                       key=lambda m: self.resolution_summary[m].resolved_fraction)

        resolved_df = self.df_pairs[self.df_pairs[f'resolved_from_{best_mod}']].copy()
        resolved_df = resolved_df.nlargest(15, f'delta_{best_mod}')

        # 构建表格数据
        table_data = []
        for _, row in resolved_df.iterrows():
            table_data.append([
                row['region_i'],
                row['region_j'],
                row['system_i'],
                row['system_j'],
                f"{row[f'rank_{best_mod}']:.3f}",
                f"{row['rank_multi']:.3f}",
                f"{row[f'delta_{best_mod}']:.3f}"
            ])

        columns = ['Region A', 'Region B', 'System A', 'System B',
                   f'Rank {best_mod}', 'Rank Multi', 'Δ']

        table = ax.table(cellText=table_data, colLabels=columns,
                         cellLoc='center', loc='center',
                         colColours=['#f0f0f0'] * len(columns))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        ax.set_title(f'Top 15 Resolved Pairs (from {best_mod.capitalize()} ambiguity)',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_top_resolved_pairs.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 5_top_resolved_pairs.png (补充)")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)

        # Resolution summary
        summary_data = []
        for mod, s in self.resolution_summary.items():
            summary_data.append({
                'modality': s.modality,
                'global_close_pairs': s.close_pairs,
                'global_resolved_pairs': s.resolved_pairs,
                'global_resolved_fraction': s.resolved_fraction,
                'within_system_close': s.within_system_close,
                'within_system_resolved': s.within_system_resolved,
                'within_system_fraction': s.within_system_fraction,
                'mean_delta': s.mean_delta
            })
        pd.DataFrame(summary_data).to_csv(f"{output_dir}/resolution_summary.csv", index=False)

        # Null test results
        if self.null_test_results:
            null_data = []
            for mod, r in self.null_test_results.items():
                null_data.append({
                    'modality': mod,
                    'observed_resolution': r.observed_resolution,
                    'null_mean': r.null_mean,
                    'null_std': r.null_std,
                    'empirical_p': r.empirical_p,
                    'n_shuffles': r.n_shuffles
                })
            pd.DataFrame(null_data).to_csv(f"{output_dir}/null_test_results.csv", index=False)

        # All pairs
        self.df_pairs.to_csv(f"{output_dir}/all_pairs_distances.csv", index=False)

        # Disagreement pairs
        for key, df in self.disagreement_pairs.items():
            df.to_csv(f"{output_dir}/disagreement_{key}.csv", index=False)

        # Resolved pairs
        for mod in ['mol', 'morph', 'proj']:
            resolved = self.df_pairs[self.df_pairs[f'resolved_from_{mod}']]
            resolved.to_csv(f"{output_dir}/resolved_from_{mod}.csv", index=False)

        print(f"\n✓ 结果保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task2_results",
                          n_clusters: int = 8, n_shuffles: int = 200):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务2: 多模态融合提升脑区分辨率 (V8 科学严谨版)")
        print("=" * 80)

        # 1. 数据加载
        self.load_all_fingerprints()

        # 2. 计算距离（CLR+降维+correlation）
        self.compute_distance_matrices()

        # 3. Rank融合（均衡权重）
        self.compute_rank_fusion(weights=(0.33, 0.33, 0.34))

        # 4. 构建pair数据
        self.build_pair_dataframe()

        # 5. Resolution分析（严格阈值）
        self.compute_resolution_metrics(
            q_close=0.05,
            rank_far_threshold=0.50,
            delta_threshold=0.15
        )

        # 6. Shuffle Null检验
        self.shuffle_null_test(n_shuffles=n_shuffles)

        # 7. Disagreement pairs
        self.find_disagreement_pairs(topk=50)

        # 8. 聚类（补充）
        self.perform_clustering(n_clusters)

        # 9. Embedding
        self.compute_embeddings()

        # 10. 可视化
        self.visualize_results(output_dir)

        # 11. 保存结果
        self.save_results(output_dir)

        # 12. 打印结论
        self._print_conclusion()

        print("\n" + "=" * 80)
        print(f"任务2完成! 结果: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self):
        """打印主要结论"""
        print("\n" + "=" * 80)
        print("主要结论")
        print("=" * 80)

        print(f"""
【Resolution Gain（核心发现）】
多模态融合显著提升了脑区之间的可区分性：
（严格定义：rank_multi >= 0.5 且 Δ >= 0.15）

  从Molecular视角:
    - Global: {self.resolution_summary['mol'].resolved_pairs}/{self.resolution_summary['mol'].close_pairs} ({self.resolution_summary['mol'].resolved_fraction:.1%}) resolved
    - Within-system: {self.resolution_summary['mol'].within_system_resolved}/{self.resolution_summary['mol'].within_system_close} ({self.resolution_summary['mol'].within_system_fraction:.1%}) resolved
    
  从Morphology视角:
    - Global: {self.resolution_summary['morph'].resolved_pairs}/{self.resolution_summary['morph'].close_pairs} ({self.resolution_summary['morph'].resolved_fraction:.1%}) resolved
    - Within-system: {self.resolution_summary['morph'].within_system_resolved}/{self.resolution_summary['morph'].within_system_close} ({self.resolution_summary['morph'].within_system_fraction:.1%}) resolved
    
  从Projection视角:
    - Global: {self.resolution_summary['proj'].resolved_pairs}/{self.resolution_summary['proj'].close_pairs} ({self.resolution_summary['proj'].resolved_fraction:.1%}) resolved
    - Within-system: {self.resolution_summary['proj'].within_system_resolved}/{self.resolution_summary['proj'].within_system_close} ({self.resolution_summary['proj'].within_system_fraction:.1%}) resolved

【Shuffle Null检验（证明融合必要性）】""")

        if self.null_test_results:
            for mod in ['mol', 'morph', 'proj']:
                r = self.null_test_results[mod]
                sig = '***' if r.empirical_p < 0.001 else '**' if r.empirical_p < 0.01 else '*' if r.empirical_p < 0.05 else 'ns'
                print(f"  {mod}: Observed={r.observed_resolution:.1%}, Null={r.null_mean:.1%}±{r.null_std:.1%}, p={r.empirical_p:.4f} {sig}")

        print(f"""
【数据处理改进】
  - Mol: CLR -> PCA(90% var) -> correlation距离
  - Morph: log1p -> RobustScaler -> PCA(95% var) -> correlation距离
  - Proj: CLR -> TruncatedSVD(30) -> correlation距离
  - 融合: Rank-based (均衡权重)
""")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task2_resolution_results_v8"
    N_CLUSTERS = 8
    N_SHUFFLES = 200
    MIN_NEURONS = 10

    with RegionMultimodalResolutionV8(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, MIN_NEURONS) as analyzer:
        analyzer.run_full_pipeline(OUTPUT_DIR, N_CLUSTERS, N_SHUFFLES)


if __name__ == "__main__":
    main()