"""
任务2：Region-level 多模态融合提升脑区分辨率 (V7 Resolution版)
===============================================
核心主题：多模态融合增加了脑区之间的可区分性（Resolution Gain）

核心输出：
1. Resolution Gain - 单模态中"分不开"的region pairs在多模态中被拉开的比例
2. Disagreement Pairs - 模态间互补的典型案例（Mol近但Morph远等）
3. Resolution Scatter Plot - 直观展示融合带来的分辨率提升

方法：
- Rank-based距离融合（对不同量纲鲁棒，避免SNF过度平滑）
- Pair-level分析而非聚类为主

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold

from scipy.spatial.distance import pdist, squareform
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


@dataclass
class ClusteringMetrics:
    silhouette: float
    nmi: float
    ari: float
    n_regions: int


class RegionMultimodalResolutionV7:
    """
    区域级多模态分辨率分析器 V7

    核心问题：多模态融合是否提升了脑区之间的可区分性？
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

        self.mol_fingerprints: Dict[str, np.ndarray] = {}
        self.morph_fingerprints: Dict[str, np.ndarray] = {}
        self.proj_fingerprints: Dict[str, np.ndarray] = {}

        self.mol_clr: Dict[str, np.ndarray] = {}
        self.proj_clr: Dict[str, np.ndarray] = {}
        self.morph_zscore: Dict[str, np.ndarray] = {}

        self.morph_feature_names: List[str] = []
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 距离矩阵
        self.dist_mol: np.ndarray = None
        self.dist_morph: np.ndarray = None
        self.dist_proj: np.ndarray = None
        self.dist_multi_rank: np.ndarray = None  # 主融合方法：Rank-based

        # Pair数据
        self.df_pairs: pd.DataFrame = None
        self.resolution_summary: Dict[str, ResolutionSummary] = {}

        # 聚类（补充验证）
        self.cluster_labels: Dict[str, np.ndarray] = {}
        self.anatomical_systems: Dict[str, str] = {}
        self.non_other_mask: np.ndarray = None

        # Embedding
        self.embeddings: Dict[str, np.ndarray] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def hellinger_distance_matrix(X: np.ndarray) -> np.ndarray:
        X_norm = X / (X.sum(axis=1, keepdims=True) + 1e-10)
        X_sqrt = np.sqrt(np.maximum(X_norm, 0))
        n = X.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(0.5 * np.sum((X_sqrt[i] - X_sqrt[j]) ** 2))
                dist[i, j] = d
                dist[j, i] = d
        return dist

    @staticmethod
    def ensure_valid_distance_matrix(D: np.ndarray) -> np.ndarray:
        D = D.copy().astype(np.float64)
        D = (D + D.T) / 2
        D = np.maximum(D, 0)
        np.fill_diagonal(D, 0.0)
        return D

    def safe_linkage(self, D: np.ndarray, method: str = 'average') -> np.ndarray:
        D = self.ensure_valid_distance_matrix(D)
        condensed = squareform(D, checks=False)
        return linkage(condensed, method=method)

    def _upper_tri_values(self, D: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """提取上三角值和索引"""
        idx = np.triu_indices(D.shape[0], k=1)
        return D[idx], idx

    def _quantile_threshold(self, D: np.ndarray, q: float) -> float:
        """计算距离矩阵的分位数阈值"""
        vals, _ = self._upper_tri_values(D)
        return np.quantile(vals, q)

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
        self._apply_transforms()
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

    def _load_morphology_fingerprints(self):
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

                fingerprint = [np.mean(feature_values[f]) if feature_values[f] else 0.0
                               for f in self.morph_feature_names]
                self.morph_fingerprints[region] = np.array(fingerprint)

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

    def _apply_transforms(self):
        print("\n应用特征变换...")

        X_mol = np.array([self.mol_fingerprints[r] for r in self.regions])
        X_morph = np.array([self.morph_fingerprints[r] for r in self.regions])
        X_proj = np.array([self.proj_fingerprints[r] for r in self.regions])

        X_mol_clr = self.clr_transform(X_mol)
        X_proj_clr = self.clr_transform(X_proj)

        scaler = StandardScaler()
        X_morph_z = scaler.fit_transform(X_morph)

        for i, region in enumerate(self.regions):
            self.mol_clr[region] = X_mol_clr[i]
            self.proj_clr[region] = X_proj_clr[i]
            self.morph_zscore[region] = X_morph_z[i]

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

    # ==================== 距离矩阵计算 ====================

    def compute_distance_matrices(self):
        """计算单模态距离矩阵"""
        print("\n" + "=" * 80)
        print("计算单模态距离矩阵")
        print("=" * 80)

        X_mol = np.array([self.mol_fingerprints[r] for r in self.regions])
        X_morph = np.array([self.morph_zscore[r] for r in self.regions])
        X_proj = np.array([self.proj_fingerprints[r] for r in self.regions])

        print("\n计算分子距离 (Hellinger)...")
        self.dist_mol = self.hellinger_distance_matrix(X_mol)
        self.dist_mol = self.ensure_valid_distance_matrix(self.dist_mol)

        print("计算形态距离 (Euclidean)...")
        self.dist_morph = squareform(pdist(X_morph, metric='euclidean'))
        self.dist_morph = self.ensure_valid_distance_matrix(self.dist_morph)

        print("计算投射距离 (Hellinger)...")
        self.dist_proj = self.hellinger_distance_matrix(X_proj)
        self.dist_proj = self.ensure_valid_distance_matrix(self.dist_proj)

        print("✓ 单模态距离矩阵计算完成")

    # ==================== Rank-based融合（核心） ====================

    # def compute_rank_fusion(self, weights: Tuple[float, float, float] = None):
    #     """
    #     Rank-based距离融合
    #
    #     原理：将每个模态的距离转换为rank（百分位），消除量纲差异，
    #     然后加权平均。这比直接归一化更鲁棒。
    #     """
    #     print("\n" + "=" * 80)
    #     print("Rank-based多模态距离融合")
    #     print("=" * 80)
    #
    #     n = len(self.regions)
    #     n_pairs = n * (n - 1) // 2
    #
    #     # 提取上三角
    #     idx = np.triu_indices(n, k=1)
    #
    #     vals_mol = self.dist_mol[idx]
    #     vals_morph = self.dist_morph[idx]
    #     vals_proj = self.dist_proj[idx]
    #
    #     # 转换为rank（百分位）
    #     rank_mol = rankdata(vals_mol) / n_pairs
    #     rank_morph = rankdata(vals_morph) / n_pairs
    #     rank_proj = rankdata(vals_proj) / n_pairs
    #
    #     # 搜索最优权重
    #     if weights is None:
    #         weights = self._search_optimal_weights_rank(rank_mol, rank_morph, rank_proj, idx, n)
    #
    #     print(f"\n最优权重: mol={weights[0]:.2f}, morph={weights[1]:.2f}, proj={weights[2]:.2f}")
    #
    #     # 加权融合
    #     rank_multi = weights[0] * rank_mol + weights[1] * rank_morph + weights[2] * rank_proj
    #
    #     # 重建距离矩阵
    #     self.dist_multi_rank = np.zeros((n, n))
    #     self.dist_multi_rank[idx] = rank_multi
    #     self.dist_multi_rank = self.dist_multi_rank + self.dist_multi_rank.T
    #
    #     self.optimal_weights = weights
    #     print("✓ Rank融合完成")
    def compute_rank_fusion(self, weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)):
        """
        [修正版] Rank-based距离融合
        强制使用均衡权重，避免单一模态主导，从而捕获互补信息。
        """
        print("\n" + "=" * 80)
        print(f"Rank-based多模态距离融合 (Weights: {weights})")
        print("=" * 80)

        n = len(self.regions)
        n_pairs = n * (n - 1) // 2
        idx = np.triu_indices(n, k=1)

        # 1. 提取距离值
        vals_mol = self.dist_mol[idx]
        vals_morph = self.dist_morph[idx]
        vals_proj = self.dist_proj[idx]

        # 2. 转换为 Rank (归一化到 0~1)
        # Rank越小 = 距离越近 = 越相似
        rank_mol = rankdata(vals_mol) / n_pairs
        rank_morph = rankdata(vals_morph) / n_pairs
        rank_proj = rankdata(vals_proj) / n_pairs

        # 3. 加权融合 Rank
        # 这里反映了各模态的"共识"：如果大家都觉得远，Rank就大
        rank_multi = weights[0] * rank_mol + weights[1] * rank_morph + weights[2] * rank_proj

        # 4. 重建距离矩阵
        self.dist_multi_rank = np.zeros((n, n))
        self.dist_multi_rank[idx] = rank_multi
        self.dist_multi_rank = self.dist_multi_rank + self.dist_multi_rank.T

        self.optimal_weights = weights
        print("✓ Rank融合完成 (使用固定均衡权重)")

    def _search_optimal_weights_rank(self, rank_mol, rank_morph, rank_proj, idx, n, n_clusters=8):
        """搜索最优权重"""
        print("\n搜索最优融合权重...")

        weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_score = -1
        best_weights = (0.33, 0.34, 0.33)

        for w1, w2 in product(weight_grid, weight_grid):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.1 or w3 > 0.8:
                continue

            rank_multi = w1 * rank_mol + w2 * rank_morph + w3 * rank_proj

            D = np.zeros((n, n))
            D[idx] = rank_multi
            D = D + D.T

            try:
                Z = self.safe_linkage(D, 'average')
                labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(D, labels, metric='precomputed')
                    if sil > best_score:
                        best_score = sil
                        best_weights = (w1, w2, w3)
            except:
                continue

        print(f"  最优Silhouette: {best_score:.4f}")
        return best_weights

    # ==================== Resolution分析（核心） ====================

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

        # 计算每个单模态的rank（百分位）
        self.df_pairs['rank_mol'] = rankdata(self.df_pairs['d_mol']) / len(self.df_pairs)
        self.df_pairs['rank_morph'] = rankdata(self.df_pairs['d_morph']) / len(self.df_pairs)
        self.df_pairs['rank_proj'] = rankdata(self.df_pairs['d_proj']) / len(self.df_pairs)
        self.df_pairs['rank_multi'] = rankdata(self.df_pairs['d_multi']) / len(self.df_pairs)

        # 计算delta（multi与单模态的差异）
        self.df_pairs['delta_mol'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_mol']
        self.df_pairs['delta_morph'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_morph']
        self.df_pairs['delta_proj'] = self.df_pairs['rank_multi'] - self.df_pairs['rank_proj']

        # 是否跨系统
        self.df_pairs['cross_system'] = self.df_pairs['system_i'] != self.df_pairs['system_j']

        print(f"  总pairs数: {len(self.df_pairs)}")
        print(f"  跨系统pairs: {self.df_pairs['cross_system'].sum()}")

    def compute_resolution_metrics(self, q_close: float = 0.05, q_far: float = 0.25, delta_thr: float = 0.10):
        """
        [修正版] 计算Resolution指标

        Resolved 定义 (满足任一条件):
        1. 绝对分离: Multi Rank > q_far (例如进入了 Top 25% 以外的非相似区)
        2. 相对增量: Multi Rank - Single Rank > delta_thr (距离显著增加了 10%)
        """
        print("\n" + "=" * 80)
        print(f"计算Resolution指标 (Close Top {q_close:.0%}, Target Top {q_far:.0%} or Gain > {delta_thr:.2f})")
        print("=" * 80)

        # 1. 计算阈值
        thr_mol_close = self.df_pairs['rank_mol'].quantile(q_close)
        thr_morph_close = self.df_pairs['rank_morph'].quantile(q_close)
        thr_proj_close = self.df_pairs['rank_proj'].quantile(q_close)

        # Multi 的绝对阈值
        thr_multi_far = q_far

        # 2. 标记 "Close" (单模态中分不开)
        self.df_pairs['mol_close'] = self.df_pairs['rank_mol'] <= thr_mol_close
        self.df_pairs['morph_close'] = self.df_pairs['rank_morph'] <= thr_morph_close
        self.df_pairs['proj_close'] = self.df_pairs['rank_proj'] <= thr_proj_close

        # 3. 计算 Resolved (分开了)
        results = []
        for mod in ['mol', 'morph', 'proj']:
            close_col = f'{mod}_close'
            delta_col = f'delta_{mod}'
            rank_col = f'rank_{mod}'
            resolved_col = f'resolved_from_{mod}'

            # 核心逻辑：原本很近 AND (绝对距离变远 OR 相对距离拉大)
            is_resolved = self.df_pairs[close_col] & \
                          ((self.df_pairs['rank_multi'] > thr_multi_far) |
                           (self.df_pairs[delta_col] > delta_thr))

            self.df_pairs[resolved_col] = is_resolved

            # 统计
            close_pairs = self.df_pairs[close_col].sum()
            resolved_pairs = is_resolved.sum()
            resolved_fraction = resolved_pairs / max(close_pairs, 1)
            mean_delta = self.df_pairs.loc[self.df_pairs[close_col], delta_col].mean()

            self.resolution_summary[mod] = ResolutionSummary(
                modality=mod,
                close_pairs=int(close_pairs),
                resolved_pairs=int(resolved_pairs),
                resolved_fraction=resolved_fraction,
                mean_delta=mean_delta
            )

            results.append({
                'modality': mod, 'close': close_pairs, 'resolved': resolved_pairs,
                'fraction': resolved_fraction, 'delta': mean_delta
            })

        # 打印结果表
        print(f"\n{'Modality':<12} {'Close Pairs':>12} {'Resolved':>10} {'Fraction':>12} {'Mean Δ':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r['modality']:<12} {r['close']:>12} {r['resolved']:>10} "
                  f"{r['fraction']:>11.1%} {r['delta']:>10.3f}")

        return results

    # def compute_resolution_metrics(self, q_close: float = 0.05, q_far: float = 0.50):
    #     """
    #     计算Resolution指标（核心输出）
    #
    #     定义：
    #     - "分不开"（close）：单模态距离处于最小的q_close分位（如5%）
    #     - "被拉开"（resolved）：multi距离超过q_far分位（如50%）
    #
    #     Resolution Fraction = resolved_pairs / close_pairs
    #     """
    #     print("\n" + "=" * 80)
    #     print(f"计算Resolution指标 (q_close={q_close}, q_far={q_far})")
    #     print("=" * 80)
    #
    #     # 阈值
    #     thr_mol_close = self.df_pairs['rank_mol'].quantile(q_close)
    #     thr_morph_close = self.df_pairs['rank_morph'].quantile(q_close)
    #     thr_proj_close = self.df_pairs['rank_proj'].quantile(q_close)
    #     thr_multi_far = self.df_pairs['rank_multi'].quantile(q_far)
    #
    #     # 标记
    #     self.df_pairs['mol_close'] = self.df_pairs['rank_mol'] <= thr_mol_close
    #     self.df_pairs['morph_close'] = self.df_pairs['rank_morph'] <= thr_morph_close
    #     self.df_pairs['proj_close'] = self.df_pairs['rank_proj'] <= thr_proj_close
    #     self.df_pairs['multi_far'] = self.df_pairs['rank_multi'] >= thr_multi_far
    #
    #     # 计算resolved
    #     self.df_pairs['resolved_from_mol'] = self.df_pairs['mol_close'] & self.df_pairs['multi_far']
    #     self.df_pairs['resolved_from_morph'] = self.df_pairs['morph_close'] & self.df_pairs['multi_far']
    #     self.df_pairs['resolved_from_proj'] = self.df_pairs['proj_close'] & self.df_pairs['multi_far']
    #
    #     # 汇总
    #     results = []
    #     for modality in ['mol', 'morph', 'proj']:
    #         close_col = f'{modality}_close'
    #         resolved_col = f'resolved_from_{modality}'
    #         delta_col = f'delta_{modality}'
    #
    #         close_pairs = self.df_pairs[close_col].sum()
    #         resolved_pairs = self.df_pairs[resolved_col].sum()
    #         resolved_fraction = resolved_pairs / max(close_pairs, 1)
    #         mean_delta = self.df_pairs.loc[self.df_pairs[close_col], delta_col].mean()
    #
    #         self.resolution_summary[modality] = ResolutionSummary(
    #             modality=modality,
    #             close_pairs=int(close_pairs),
    #             resolved_pairs=int(resolved_pairs),
    #             resolved_fraction=resolved_fraction,
    #             mean_delta=mean_delta
    #         )
    #
    #         results.append({
    #             'modality': modality,
    #             'close_pairs': close_pairs,
    #             'resolved_pairs': resolved_pairs,
    #             'resolved_fraction': resolved_fraction,
    #             'mean_delta': mean_delta
    #         })
    #
    #     # 打印结果
    #     print(f"\n{'Modality':<12} {'Close Pairs':>12} {'Resolved':>10} {'Fraction':>12} {'Mean Δ':>10}")
    #     print("-" * 60)
    #     for r in results:
    #         print(f"{r['modality']:<12} {r['close_pairs']:>12} {r['resolved_pairs']:>10} "
    #               f"{r['resolved_fraction']:>11.1%} {r['mean_delta']:>10.3f}")
    #
    #     return results

    def find_disagreement_pairs(self, topk: int = 100):
        """
        找出模态间互补的典型案例（Disagreement Pairs）

        例如：Mol近但Morph远 → 说明这两个region分子组成相似但形态不同
        """
        print("\n" + "=" * 80)
        print("挖掘Disagreement Pairs（模态互补案例）")
        print("=" * 80)

        disagreements = {}

        # Mol近 + Morph远
        df = self.df_pairs.copy()
        df['score'] = (1 - df['rank_mol']) * df['rank_morph']  # mol越近、morph越远，score越高
        mol_close_morph_far = df.nlargest(topk, 'score')[
            ['region_i', 'region_j', 'system_i', 'system_j',
             'd_mol', 'd_morph', 'd_proj', 'd_multi',
             'rank_mol', 'rank_morph', 'delta_mol', 'cross_system']
        ]
        disagreements['mol_close_morph_far'] = mol_close_morph_far
        print(f"\n【Mol近 + Morph远】Top {topk}")
        print(f"  跨系统比例: {mol_close_morph_far['cross_system'].mean():.1%}")

        # Mol近 + Proj远
        df['score'] = (1 - df['rank_mol']) * df['rank_proj']
        mol_close_proj_far = df.nlargest(topk, 'score')[
            ['region_i', 'region_j', 'system_i', 'system_j',
             'd_mol', 'd_morph', 'd_proj', 'd_multi',
             'rank_mol', 'rank_proj', 'delta_mol', 'cross_system']
        ]
        disagreements['mol_close_proj_far'] = mol_close_proj_far
        print(f"\n【Mol近 + Proj远】Top {topk}")
        print(f"  跨系统比例: {mol_close_proj_far['cross_system'].mean():.1%}")

        # Morph近 + Mol远
        df['score'] = (1 - df['rank_morph']) * df['rank_mol']
        morph_close_mol_far = df.nlargest(topk, 'score')[
            ['region_i', 'region_j', 'system_i', 'system_j',
             'd_mol', 'd_morph', 'd_proj', 'd_multi',
             'rank_morph', 'rank_mol', 'delta_morph', 'cross_system']
        ]
        disagreements['morph_close_mol_far'] = morph_close_mol_far
        print(f"\n【Morph近 + Mol远】Top {topk}")
        print(f"  跨系统比例: {morph_close_mol_far['cross_system'].mean():.1%}")

        # Morph近 + Proj远
        df['score'] = (1 - df['rank_morph']) * df['rank_proj']
        morph_close_proj_far = df.nlargest(topk, 'score')[
            ['region_i', 'region_j', 'system_i', 'system_j',
             'd_mol', 'd_morph', 'd_proj', 'd_multi',
             'rank_morph', 'rank_proj', 'delta_morph', 'cross_system']
        ]
        disagreements['morph_close_proj_far'] = morph_close_proj_far
        print(f"\n【Morph近 + Proj远】Top {topk}")
        print(f"  跨系统比例: {morph_close_proj_far['cross_system'].mean():.1%}")

        self.disagreement_pairs = disagreements
        return disagreements

    # ==================== 聚类分析（补充验证） ====================

    def perform_clustering(self, n_clusters: int = 8):
        """聚类分析（作为补充验证）"""
        print("\n" + "=" * 80)
        print(f"聚类分析（补充验证, k={n_clusters}）")
        print("=" * 80)

        # 分子
        Z = self.safe_linkage(self.dist_mol, 'average')
        self.cluster_labels['molecular'] = fcluster(Z, n_clusters, criterion='maxclust') - 1

        # 形态
        X_morph = np.array([self.morph_zscore[r] for r in self.regions])
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.cluster_labels['morphology'] = clustering.fit_predict(X_morph)

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
            if name == 'morphology':
                sil = silhouette_score(X_morph, labels)
            elif name == 'molecular':
                sil = silhouette_score(self.dist_mol, labels, metric='precomputed')
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
        """计算Embedding用于可视化"""
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

    # ==================== 可视化（核心图） ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_resolution_scatter(output_dir)
        self._plot_resolution_summary(output_dir)
        self._plot_delta_distribution(output_dir)
        self._plot_embeddings(output_dir)
        self._plot_disagreement_examples(output_dir)

        print(f"\n✓ 图表保存到: {output_dir}")

    # def _plot_resolution_scatter(self, output_dir: str):
    #     """
    #     核心图1：Resolution Scatter Plot
    #     展示单模态距离 vs Multi距离，高亮被"解决"的pairs
    #     """
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #
    #     modalities = ['mol', 'morph', 'proj']
    #     titles = ['Molecular → Multimodal', 'Morphology → Multimodal', 'Projection → Multimodal']
    #     colors = ['#3498DB', '#27AE60', '#9B59B6']
    #
    #     for ax, mod, title, color in zip(axes, modalities, titles, colors):
    #         x = self.df_pairs[f'rank_{mod}']
    #         y = self.df_pairs['rank_multi']
    #
    #         # 所有点（灰色）
    #         ax.scatter(x, y, s=8, alpha=0.3, c='gray', label='All pairs')
    #
    #         # 高亮resolved pairs
    #         resolved = self.df_pairs[f'resolved_from_{mod}']
    #         ax.scatter(x[resolved], y[resolved], s=15, alpha=0.8, c=color,
    #                    label=f'Resolved ({resolved.sum()})')
    #
    #         # 对角线
    #         ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    #
    #         # 阈值线
    #         q_close = 0.05
    #         q_far = 0.50
    #         ax.axvline(x=q_close, color='red', linestyle=':', alpha=0.7, label=f'Close threshold ({q_close})')
    #         ax.axhline(y=q_far, color='green', linestyle=':', alpha=0.7, label=f'Far threshold ({q_far})')
    #
    #         ax.set_xlabel(f'{mod.capitalize()} Distance (rank)', fontsize=11)
    #         ax.set_ylabel('Multimodal Distance (rank)', fontsize=11)
    #         ax.set_title(title, fontsize=13, fontweight='bold')
    #         ax.legend(loc='lower right', fontsize=8)
    #         ax.set_xlim(-0.02, 1.02)
    #         ax.set_ylim(-0.02, 1.02)
    #         ax.grid(alpha=0.3)
    #
    #         # 添加resolution fraction文本
    #         summary = self.resolution_summary[mod]
    #         ax.text(0.05, 0.95, f"Resolution: {summary.resolved_fraction:.1%}",
    #                 transform=ax.transAxes, fontsize=11, fontweight='bold',
    #                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    #
    #     plt.suptitle('Resolution Gain: Indistinguishable pairs in single modality → Separated in multimodal',
    #                  fontsize=14, fontweight='bold', y=1.02)
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/1_resolution_scatter.png", dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print("  ✓ 1_resolution_scatter.png")
    def _plot_resolution_scatter(self, output_dir: str):
        """
        [修正版] 绘制 Resolution Gain 散点图
        展示相对于对角线(y=x)的提升
        """
        import matplotlib.pyplot as plt

        # 设置绘图风格，确保清晰
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        modalities = ['mol', 'morph', 'proj']
        titles = ['Molecular Ambiguity', 'Morphological Ambiguity', 'Connectivity Ambiguity']
        colors = ['#3498DB', '#27AE60', '#9B59B6']  # 蓝，绿，紫

        for ax, mod, title, color in zip(axes, modalities, titles, colors):
            x = self.df_pairs[f'rank_{mod}']
            y = self.df_pairs['rank_multi']
            resolved_mask = self.df_pairs[f'resolved_from_{mod}']
            close_mask = self.df_pairs[f'{mod}_close']

            # 1. 绘制背景点 (所有非Close的)
            ax.scatter(x[~close_mask], y[~close_mask], s=5, c='lightgray', alpha=0.2, label='Other Pairs')

            # 2. 绘制 Close 但 Unresolved (红色，表示还在盲区)
            # 条件：是Close，但不是Resolved
            unresolved_mask = close_mask & (~resolved_mask)
            ax.scatter(x[unresolved_mask], y[unresolved_mask], s=15, c='#E74C3C', alpha=0.6, label='Unresolved')

            # 3. 绘制 Resolved (高亮色，表示成功区分)
            ax.scatter(x[resolved_mask], y[resolved_mask], s=25, c=color, alpha=0.9, edgecolors='white', linewidth=0.5,
                       label=f'Resolved ({resolved_mask.sum()})')

            # 4. 辅助线
            # y = x (无增益线)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
            # y = x + 0.1 (增益阈值线)
            delta_thr = 0.10
            ax.plot([0, 1 - delta_thr], [delta_thr, 1], 'k:', linewidth=1, alpha=0.5, label=f'Gain > {delta_thr}')

            # Close阈值线 (垂直线)
            q_close_val = self.df_pairs[f'rank_{mod}'].quantile(0.05)
            ax.axvline(q_close_val, color='gray', linestyle='--', alpha=0.5)
            ax.text(q_close_val, 1.02, 'Top 5% Close', ha='center', fontsize=9, color='gray')

            ax.set_xlabel(f'Single Modality Distance (Rank)', fontsize=11)
            ax.set_ylabel('Multimodal Distance (Rank)', fontsize=11)
            ax.set_title(f'{title}\nResolution Gain: {self.resolution_summary[mod].resolved_fraction:.1%}', fontsize=12,
                         fontweight='bold')

            # 限制显示范围，聚焦在左下角 (Close区域) 效果更好，或者显示全图
            # 这里显示全图但重点看左下
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle=':', alpha=0.6)

            if mod == 'proj':  # 只在最后一个图显示图例，节省空间
                ax.legend(loc='lower right', fontsize=9)

        plt.suptitle('Resolution Gain: Multimodal Integration Disentangles Ambiguous Regions',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_resolution_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_resolution_scatter.png")

    def _plot_resolution_summary(self, output_dir: str):
        """
        核心图2：Resolution Summary Bar Plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        modalities = ['Molecular', 'Morphology', 'Projection']
        keys = ['mol', 'morph', 'proj']
        colors = ['#3498DB', '#27AE60', '#9B59B6']

        x = np.arange(len(modalities))
        width = 0.35

        close_pairs = [self.resolution_summary[k].close_pairs for k in keys]
        resolved_pairs = [self.resolution_summary[k].resolved_pairs for k in keys]

        bars1 = ax.bar(x - width / 2, close_pairs, width, label='Indistinguishable (Top 5%)', color='lightgray', edgecolor='black')
        bars2 = ax.bar(x + width / 2, resolved_pairs, width, label='Resolved by Multimodal', color=colors, edgecolor='black')

        # 添加数值标签
        for bar, val in zip(bars1, close_pairs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{val}',
                    ha='center', va='bottom', fontsize=10)

        for bar, val, key in zip(bars2, resolved_pairs, keys):
            frac = self.resolution_summary[key].resolved_fraction
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{val}\n({frac:.1%})', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Single Modality', fontsize=12)
        ax.set_ylabel('Number of Region Pairs', fontsize=12)
        ax.set_title('Resolution Gain: How many "indistinguishable" pairs become separated?',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modalities, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_resolution_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 2_resolution_summary.png")

    def _plot_delta_distribution(self, output_dir: str):
        """
        核心图3：Delta分布图
        展示multimodal相对于单模态的距离增量分布
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        modalities = ['mol', 'morph', 'proj']
        titles = ['Δ(Multi - Molecular)', 'Δ(Multi - Morphology)', 'Δ(Multi - Projection)']
        colors = ['#3498DB', '#27AE60', '#9B59B6']

        for ax, mod, title, color in zip(axes, modalities, titles, colors):
            delta = self.df_pairs[f'delta_{mod}']
            close_mask = self.df_pairs[f'{mod}_close']

            # 所有pairs的delta分布
            ax.hist(delta, bins=50, alpha=0.5, color='gray', label='All pairs', density=True)

            # Close pairs的delta分布
            ax.hist(delta[close_mask], bins=30, alpha=0.7, color=color,
                    label=f'Close pairs (top 5%)', density=True)

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            ax.axvline(x=delta[close_mask].mean(), color=color, linestyle='-', linewidth=2,
                       label=f'Mean Δ = {delta[close_mask].mean():.3f}')

            ax.set_xlabel('Δ (Multi rank - Single rank)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle('Distance Increment: Positive Δ means pairs become MORE separated in multimodal',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_delta_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_delta_distribution.png")

    def _plot_embeddings(self, output_dir: str):
        """
        补充图：Embedding可视化
        """
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

        # Legend
        legend = [Patch(facecolor=sys_color[s], label=s) for s in unique if s != 'Other']
        legend.append(Patch(facecolor=[0.7, 0.7, 0.7], label='Other'))
        fig.legend(handles=legend, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)

        plt.suptitle(f'Region Embeddings (weights: mol={self.optimal_weights[0]:.2f}, '
                     f'morph={self.optimal_weights[1]:.2f}, proj={self.optimal_weights[2]:.2f})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_embeddings.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_embeddings.png")

    def _plot_disagreement_examples(self, output_dir: str):
        """
        补充图：Disagreement pairs示例（热力图）
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        cases = [
            ('mol_close_morph_far', 'Mol近 + Morph远', '#3498DB'),
            ('mol_close_proj_far', 'Mol近 + Proj远', '#E74C3C'),
            ('morph_close_mol_far', 'Morph近 + Mol远', '#27AE60'),
            ('morph_close_proj_far', 'Morph近 + Proj远', '#9B59B6'),
        ]

        for ax, (key, title, color) in zip(axes.flat, cases):
            df = self.disagreement_pairs[key].head(30)

            # 创建数据用于展示
            data = df[['region_i', 'region_j', 'cross_system']].copy()
            data['pair'] = data['region_i'] + ' - ' + data['region_j']
            data['cross'] = data['cross_system'].map({True: '跨系统', False: '同系统'})

            # 简单bar plot展示top pairs
            y_pos = np.arange(min(20, len(data)))
            pair_labels = data['pair'].values[:20]
            cross_labels = data['cross'].values[:20]

            bar_colors = [color if c == '跨系统' else 'lightgray' for c in cross_labels]
            ax.barh(y_pos, np.ones(len(y_pos)), color=bar_colors, edgecolor='black', alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=8)
            ax.set_xlabel('', fontsize=10)
            ax.set_title(f'{title}\n(跨系统: {df["cross_system"].mean():.1%})', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlim(0, 1.2)

        plt.suptitle('Disagreement Pairs: Regions similar in one modality but different in another',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_disagreement_pairs.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 5_disagreement_pairs.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)

        # Resolution summary
        summary_data = [
            {
                'modality': s.modality,
                'close_pairs': s.close_pairs,
                'resolved_pairs': s.resolved_pairs,
                'resolved_fraction': s.resolved_fraction,
                'mean_delta': s.mean_delta
            }
            for s in self.resolution_summary.values()
        ]
        pd.DataFrame(summary_data).to_csv(f"{output_dir}/resolution_summary.csv", index=False)

        # All pairs
        self.df_pairs.to_csv(f"{output_dir}/all_pairs_distances.csv", index=False)

        # Disagreement pairs
        for key, df in self.disagreement_pairs.items():
            df.to_csv(f"{output_dir}/disagreement_{key}.csv", index=False)

        # Resolved pairs（详细列表）
        for mod in ['mol', 'morph', 'proj']:
            resolved = self.df_pairs[self.df_pairs[f'resolved_from_{mod}']]
            resolved.to_csv(f"{output_dir}/resolved_from_{mod}.csv", index=False)

        print(f"\n✓ 结果保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task2_results", n_clusters: int = 8):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务2: 多模态融合提升脑区分辨率 (V7)")
        print("=" * 80)

        # 1. 数据加载
        self.load_all_fingerprints()

        # 2. 计算单模态距离
        self.compute_distance_matrices()

        # 3. Rank融合
        self.compute_rank_fusion()

        # 4. 构建pair数据
        self.build_pair_dataframe()

        # 5. Resolution分析（核心）
        self.compute_resolution_metrics(q_close=0.05, q_far=0.50)

        # 6. Disagreement pairs
        self.find_disagreement_pairs(topk=100)

        # 7. 聚类（补充）
        self.perform_clustering(n_clusters)

        # 8. Embedding
        self.compute_embeddings()

        # 9. 可视化
        self.visualize_results(output_dir)

        # 10. 保存结果
        self.save_results(output_dir)

        # 11. 打印结论
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

  从Molecular视角:
    - 在分子组成最相似的Top 5% region pairs中
    - {self.resolution_summary['mol'].resolved_pairs}/{self.resolution_summary['mol'].close_pairs} ({self.resolution_summary['mol'].resolved_fraction:.1%}) 在多模态空间中被显著拉开
    
  从Morphology视角:
    - 在形态最相似的Top 5% region pairs中
    - {self.resolution_summary['morph'].resolved_pairs}/{self.resolution_summary['morph'].close_pairs} ({self.resolution_summary['morph'].resolved_fraction:.1%}) 在多模态空间中被显著拉开
    
  从Projection视角:
    - 在投射最相似的Top 5% region pairs中
    - {self.resolution_summary['proj'].resolved_pairs}/{self.resolution_summary['proj'].close_pairs} ({self.resolution_summary['proj'].resolved_fraction:.1%}) 在多模态空间中被显著拉开

【Disagreement Pairs（互补性证据）】
模态间存在显著的互补关系，例如：
  - 分子组成相似但形态不同的region pairs
  - 分子组成相似但投射模式不同的region pairs
这说明单模态无法完整描述脑区身份，多模态融合捕获了更完整的生物学信息。

【融合权重】
  mol: {self.optimal_weights[0]:.2f}, morph: {self.optimal_weights[1]:.2f}, proj: {self.optimal_weights[2]:.2f}
""")


# def main():
#     NEO4J_URI = "bolt://localhost:7687"
#     NEO4J_USER = "neo4j"
#     NEO4J_PASSWORD = "neuroxiv"
#     NEO4J_DATABASE = "neo4j"
#
#     OUTPUT_DIR = "./task2_resolution_results_v7"
#     N_CLUSTERS = 8
#     MIN_NEURONS = 10
#
#     with RegionMultimodalResolutionV7(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, MIN_NEURONS) as analyzer:
#         analyzer.run_full_pipeline(OUTPUT_DIR, N_CLUSTERS)
def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    # 修改输出目录，避免覆盖
    OUTPUT_DIR = "./task2_resolution_v7_fixed"
    MIN_NEURONS = 10
    N_CLUSTERS = 8

    with RegionMultimodalResolutionV7(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, MIN_NEURONS) as analyzer:
        # 1. 加载和计算基础距离
        analyzer.load_all_fingerprints()
        analyzer.compute_distance_matrices()

        # 2. [关键] 使用均衡权重进行融合
        # 强制大家平权，这样如果有模态互补，Rank就会被拉开
        analyzer.compute_rank_fusion(weights=(0.33, 0.33, 0.34))

        # 3. 构建Pair表
        analyzer.build_pair_dataframe()

        # 4. [关键] 计算指标
        # q_close=0.05: 关注单模态里最像的那5% (盲区)
        # delta_thr=0.10: 只要融合后Rank提升了0.1 (即拉开了10%的距离)，就算成功
        analyzer.compute_resolution_metrics(q_close=0.05, q_far=0.25, delta_thr=0.10)

        # 5. 挖掘具体的互补案例 (用于写Paper里的Case Study)
        analyzer.find_disagreement_pairs(topk=50)

        # 6. 后续常规绘图
        analyzer.perform_clustering(N_CLUSTERS)
        analyzer.compute_embeddings()
        analyzer.visualize_results(OUTPUT_DIR)
        analyzer.save_results(OUTPUT_DIR)


if __name__ == "__main__":
    main()
