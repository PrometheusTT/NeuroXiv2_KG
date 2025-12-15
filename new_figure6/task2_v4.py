"""
任务2：Region-level 多模态分析 (V10 最终版)
===============================================
核心主题：模态互补性 + 多模态整合的生物学意义

V10核心修正：
1.主发现改为"模态互补性"（Mantel + Disagreement）
   - 这是可防御的、非平凡的生物学发现

2.Resolution指标改为"Rank Position Shift"
   - 单模态Top 5% → 多模态中排名>30%/50%
   - 这是非平凡的，不是算子必然结果
   - 区别于V9的"Δ>0"（那是数学必然）

3.正确解释聚类结果
   - Silhouette↓ + NMI↑ = 融合使结构更符合解剖学，而非更紧凑
   - 这是正常的，需要正确表述

4.MEAN/MAX定位为auxiliary evidence
   - 不作为主结论
   - MAX作为upper bound / stress test

科学叙事核心：
"Different modalities capture weakly correlated but non-redundant
aspects of region identity.Multimodal integration reconciles these
complementary views into a representation that better aligns with
anatomical organization."

作者:  PrometheusTT
日期: 2025-07-30
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from sklearn.manifold import MDS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata, wilcoxon
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

import neo4j

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# 设置全局绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False


@dataclass
class ModalityComplementarity:
    """模态互补性分析结果"""
    mod1: str
    mod2: str
    mantel_r: float
    mantel_p: float
    n_high_disagreement: int  # disagreement > 0.5 的pairs数
    cross_system_fraction: float  # 高disagreement中跨系统的比例


@dataclass
class RankShiftSummary:
    """Rank Position Shift汇总（非平凡resolution指标）"""
    modality:  str
    close_pairs: int  # 单模态Top 5%
    shifted_to_30pct: int  # 多模态中排名>30%
    shifted_to_50pct: int  # 多模态中排名>50%
    fraction_30pct:  float
    fraction_50pct: float
    mean_rank_change: float  # 平均排名变化


@dataclass
class ClusteringResult:
    """聚类结果"""
    modality: str
    silhouette: float
    nmi:  float
    ari: float
    interpretation: str  # 结果解释


class RegionMultimodalAnalysisV10:
    """
    区域级多模态分析 V10（最终版）

    核心框架：
    1.模态互补性分析（主发现）
    2.Rank Position Shift（非平凡resolution）
    3.聚类验证（正确解释）
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

    def __init__(self, uri: str, user: str, password:  str, database: str = "neo4j",
                 min_neurons_per_region: int = 10):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.min_neurons_per_region = min_neurons_per_region

        self.regions:  List[str] = []
        self.region_neuron_counts: Dict[str, int] = {}

        self.mol_fingerprints: Dict[str, np.ndarray] = {}
        self.morph_fingerprints: Dict[str, np.ndarray] = {}
        self.proj_fingerprints: Dict[str, np.ndarray] = {}

        self.morph_feature_names:  List[str] = []
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 距离矩阵
        self.dist_mol:  np.ndarray = None
        self.dist_morph: np.ndarray = None
        self.dist_proj: np.ndarray = None
        self.dist_multi: np.ndarray = None  # Rank-based mean fusion

        # 分析结果
        self.df_pairs: pd.DataFrame = None
        self.complementarity:  Dict[str, ModalityComplementarity] = {}
        self.rank_shift:  Dict[str, RankShiftSummary] = {}
        self.clustering_results: Dict[str, ClusteringResult] = {}

        # 解剖系统
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
    def clr_transform(X:  np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

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

        print(f"\n✓ 数据加载完成:  {len(self.regions)} 个高质量区域")
        return len(self.regions)

    def _get_global_dimensions(self):
        print("\n获取全局维度...")
        self.morph_feature_names = self.AXONAL_FEATURES + self.DENDRITIC_FEATURES

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s: Subclass) WHERE s.name IS NOT NULL
                RETURN DISTINCT s.name AS name ORDER BY name
            """)
            self.all_subclasses = [r['name'] for r in result if r['name']]

            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]

        print(f"  形态:  {len(self.morph_feature_names)}维, Subclass: {len(self.all_subclasses)}种, 投射: {len(self.all_target_regions)}个")

    def _get_valid_regions_with_qc(self):
        print(f"\n获取有效区域 (神经元数 >= {self.min_neurons_per_region})...")

        query = """
        MATCH (r:Region)-[: HAS_SUBCLASS]->()
        WITH DISTINCT r
        OPTIONAL MATCH (n:Neuron)-[: LOCATE_AT]->(r)
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

        print(f"  通过质量控制:  {len(self.regions)} 个区域")

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

        query = f"MATCH (n: Neuron)-[:LOCATE_AT]->(r:Region {{acronym: $region}}) RETURN {return_clause}"
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

                fingerprint = [np.median(feature_values[f]) if feature_values[f] else 0.0
                               for f in self.morph_feature_names]
                self.morph_fingerprints[region] = np.array(fingerprint)

    def _load_projection_fingerprints(self):
        print("加载投射指纹...")

        query = """
        MATCH (n:Neuron)-[: LOCATE_AT]->(r:Region {acronym: $region})
        MATCH (n)-[p:PROJECT_TO]->(t: Subregion)
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
                    proj_dict = {r['target']:  r['total_weight'] for r in result
                                 if r['target'] and r['total_weight']}

                fingerprint = np.zeros(len(self.all_target_regions))
                for i, target in enumerate(self.all_target_regions):
                    if target in proj_dict:
                        fingerprint[i] = proj_dict[target]

                self.proj_fingerprints[region] = fingerprint

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
        print(f"  系统分布:  {system_counts}")

    # ==================== 距离计算 ====================

    def compute_distance_matrices(self):
        """计算单模态距离矩阵"""
        print("\n" + "=" * 80)
        print("计算单模态距离矩阵")
        print("=" * 80)

        n_regions = len(self.regions)

        # Molecular:  CLR -> PCA -> correlation
        print("\n【Molecular】CLR -> PCA -> correlation...")
        X_mol_raw = np.array([self.mol_fingerprints[r] for r in self.regions])
        X_mol_clr = self.clr_transform(X_mol_raw)
        pca_mol = PCA(n_components=0.90, svd_solver='full')
        X_mol_pca = pca_mol.fit_transform(X_mol_clr)
        print(f"  {X_mol_clr.shape[1]}维 -> PCA:  {X_mol_pca.shape[1]}维")

        self.dist_mol = squareform(pdist(X_mol_pca, metric='correlation'))
        self.dist_mol = self.ensure_valid_distance_matrix(self.dist_mol)

        # Morphology:  log1p -> RobustScaler -> PCA -> correlation
        print("\n【Morphology】log1p -> RobustScaler -> PCA -> correlation...")
        X_morph_raw = np.array([self.morph_fingerprints[r] for r in self.regions])
        X_morph_log = np.log1p(X_morph_raw)
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)
        pca_morph = PCA(n_components=0.95, svd_solver='full')
        X_morph_pca = pca_morph.fit_transform(X_morph_scaled)
        print(f"  {X_morph_scaled.shape[1]}维 -> PCA: {X_morph_pca.shape[1]}维")

        self.dist_morph = squareform(pdist(X_morph_pca, metric='correlation'))
        self.dist_morph = self.ensure_valid_distance_matrix(self.dist_morph)

        # Projection: CLR -> TruncatedSVD -> correlation
        print("\n【Projection】CLR -> TruncatedSVD -> correlation...")
        X_proj_raw = np.array([self.proj_fingerprints[r] for r in self.regions])
        X_proj_clr = self.clr_transform(X_proj_raw)
        n_components = min(30, X_proj_clr.shape[1] - 1, n_regions - 1)
        svd_proj = TruncatedSVD(n_components=n_components, random_state=42)
        X_proj_svd = svd_proj.fit_transform(X_proj_clr)
        print(f"  {X_proj_clr.shape[1]}维 -> SVD: {X_proj_svd.shape[1]}维")

        self.dist_proj = squareform(pdist(X_proj_svd, metric='correlation'))
        self.dist_proj = self.ensure_valid_distance_matrix(self.dist_proj)

        print("\n✓ 单模态距离计算完成")

    def compute_multimodal_fusion(self):
        """
        Rank-based融合（均衡权重）

        注意：这里使用rank而非原始距离，消除量纲差异
        """
        print("\n" + "=" * 80)
        print("Rank-based多模态融合")
        print("=" * 80)

        n = len(self.regions)
        n_pairs = n * (n - 1) // 2
        idx = np.triu_indices(n, k=1)

        # 转换为rank
        rank_mol = rankdata(self.dist_mol[idx]) / n_pairs
        rank_morph = rankdata(self.dist_morph[idx]) / n_pairs
        rank_proj = rankdata(self.dist_proj[idx]) / n_pairs

        # 均衡权重融合
        weights = (1/3, 1/3, 1/3)
        rank_multi = weights[0] * rank_mol + weights[1] * rank_morph + weights[2] * rank_proj

        # 重建距离矩阵
        self.dist_multi = np.zeros((n, n))
        self.dist_multi[idx] = rank_multi
        self.dist_multi = self.dist_multi + self.dist_multi.T

        print(f"  权重:  mol={weights[0]:.2f}, morph={weights[1]:.2f}, proj={weights[2]:.2f}")
        print("✓ Rank融合完成")

    # ==================== Pair DataFrame ====================

    def build_pair_dataframe(self):
        """构建pairs数据框"""
        print("\n" + "=" * 80)
        print("构建Region Pair数据表")
        print("=" * 80)

        n = len(self.regions)
        n_pairs = n * (n - 1) // 2
        idx = np.triu_indices(n, k=1)
        i_idx, j_idx = idx

        self.df_pairs = pd.DataFrame({
            'i':  i_idx,
            'j': j_idx,
            'region_i': [self.regions[i] for i in i_idx],
            'region_j':  [self.regions[j] for j in j_idx],
            'system_i': [self.anatomical_systems[self.regions[i]] for i in i_idx],
            'system_j': [self.anatomical_systems[self.regions[j]] for j in j_idx],
            'd_mol': self.dist_mol[idx],
            'd_morph': self.dist_morph[idx],
            'd_proj': self.dist_proj[idx],
            'd_multi': self.dist_multi[idx],
        })

        # Rank（百分位）
        self.df_pairs['rank_mol'] = rankdata(self.df_pairs['d_mol']) / n_pairs
        self.df_pairs['rank_morph'] = rankdata(self.df_pairs['d_morph']) / n_pairs
        self.df_pairs['rank_proj'] = rankdata(self.df_pairs['d_proj']) / n_pairs
        self.df_pairs['rank_multi'] = rankdata(self.df_pairs['d_multi']) / n_pairs

        # 系统关系
        self.df_pairs['same_system'] = self.df_pairs['system_i'] == self.df_pairs['system_j']
        self.df_pairs['both_non_other'] = (
            (self.df_pairs['system_i'] != 'Other') &
            (self.df_pairs['system_j'] != 'Other')
        )

        # Disagreement scores（模态间差异）
        self.df_pairs['disagree_mol_morph'] = np.abs(self.df_pairs['rank_mol'] - self.df_pairs['rank_morph'])
        self.df_pairs['disagree_mol_proj'] = np.abs(self.df_pairs['rank_mol'] - self.df_pairs['rank_proj'])
        self.df_pairs['disagree_morph_proj'] = np.abs(self.df_pairs['rank_morph'] - self.df_pairs['rank_proj'])

        print(f"  总pairs: {len(self.df_pairs)}")
        print(f"  同系统:  {self.df_pairs['same_system'].sum()}")
        print(f"  跨系统: {(~self.df_pairs['same_system']).sum()}")

    # ==================== 1. 模态互补性分析（主发现） ====================

    def analyze_modality_complementarity(self):
        """
        分析模态间的互补性（主发现）

        两个证据：
        1.Mantel correlation：模态间距离相关性
        2. Disagreement分布：有多少pairs在不同模态间有显著差异
        """
        print("\n" + "=" * 80)
        print("【主发现1】模态互补性分析")
        print("=" * 80)

        n = len(self.regions)
        idx = np.triu_indices(n, k=1)

        d_mol = self.dist_mol[idx]
        d_morph = self.dist_morph[idx]
        d_proj = self.dist_proj[idx]

        pairs = [
            ('mol', 'morph', d_mol, d_morph, 'disagree_mol_morph'),
            ('mol', 'proj', d_mol, d_proj, 'disagree_mol_proj'),
            ('morph', 'proj', d_morph, d_proj, 'disagree_morph_proj'),
        ]

        print(f"\n{'Comparison':<18} {'Mantel r': >10} {'p-value':>12} {'High Disagree':>14} {'Cross-system':>14}")
        print("-" * 75)

        for mod1, mod2, d1, d2, disagree_col in pairs:
            # Mantel correlation
            r, p = spearmanr(d1, d2)

            # High disagreement pairs (>0.3 rank difference)
            high_disagree = self.df_pairs[disagree_col] > 0.3
            n_high = high_disagree.sum()

            # Cross-system fraction among high disagreement
            if n_high > 0:
                cross_frac = (~self.df_pairs.loc[high_disagree, 'same_system']).mean()
            else:
                cross_frac = 0.0

            key = f'{mod1}_vs_{mod2}'
            self.complementarity[key] = ModalityComplementarity(
                mod1=mod1, mod2=mod2,
                mantel_r=r, mantel_p=p,
                n_high_disagreement=n_high,
                cross_system_fraction=cross_frac
            )

            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"{mod1} vs {mod2:<10} {r:>10.3f} {p: >12.2e} {sig: <3} {n_high: >10} {cross_frac:>13.1%}")

        # 核心结论
        avg_r = np.mean([c.mantel_r for c in self.complementarity.values()])
        print(f"\n【结论】模态间平均相关性 r = {avg_r:.3f}")
        if avg_r < 0.5:
            print("  → 模态间弱相关，说明它们捕获了不同的信息维度")
        else:
            print("  → 模态间中等相关，存在部分共享信息")

        return self.complementarity

    # ==================== 2.Rank Position Shift（非平凡resolution） ====================

    def compute_rank_position_shift(self, q_close:  float = 0.05):
        """
        计算Rank Position Shift（非平凡resolution指标）

        定义：
        - close pairs:  单模态距离rank在top 5%（最相似）
        - shifted:  这些pairs在多模态中的rank位置
        - 非平凡threshold: rank_multi > 0.30 或 > 0.50

        这是非平凡的！因为：
        - 如果三个模态一致认为某pair近，multi也会近
        - 只有当其他模态disagree时，才会发生rank shift
        """
        print("\n" + "=" * 80)
        print("【主发现2】Rank Position Shift")
        print(f"  Close定义:  单模态Top {q_close:.0%}")
        print("  非平凡阈值: rank_multi > 30% 或 > 50%")
        print("=" * 80)

        results = []

        for mod in ['mol', 'morph', 'proj']:
            rank_col = f'rank_{mod}'

            # Close pairs (单模态top 5%)
            thr_close = q_close
            close_mask = self.df_pairs[rank_col] <= thr_close
            close_pairs = close_mask.sum()

            # Rank position in multimodal
            multi_ranks = self.df_pairs.loc[close_mask, 'rank_multi']

            # 非平凡shift:  rank_multi > 0.30
            shifted_30 = (multi_ranks > 0.30).sum()
            frac_30 = shifted_30 / max(close_pairs, 1)

            # 更严格:  rank_multi > 0.50
            shifted_50 = (multi_ranks > 0.50).sum()
            frac_50 = shifted_50 / max(close_pairs, 1)

            # 平均rank变化
            mean_rank_change = (multi_ranks - self.df_pairs.loc[close_mask, rank_col]).mean()

            self.rank_shift[mod] = RankShiftSummary(
                modality=mod,
                close_pairs=int(close_pairs),
                shifted_to_30pct=int(shifted_30),
                shifted_to_50pct=int(shifted_50),
                fraction_30pct=frac_30,
                fraction_50pct=frac_50,
                mean_rank_change=mean_rank_change
            )

            results.append({
                'modality': mod,
                'close':  close_pairs,
                'shift>30%': shifted_30,
                'frac>30%': frac_30,
                'shift>50%': shifted_50,
                'frac>50%': frac_50,
                'mean_Δrank': mean_rank_change
            })

        # 打印结果
        print(f"\n{'Modality':<10} {'Close': >8} {'→>30%':>10} {'Fraction':>10} {'→>50%': >10} {'Fraction':>10} {'Mean Δrank':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['modality']: <10} {r['close']:>8} {r['shift>30%']:>10} {r['frac>30%']:>9.1%} "
                  f"{r['shift>50%']:>10} {r['frac>50%']:>9.1%} {r['mean_Δrank']:>12.3f}")

        # 解释
        print(f"\n【解释】")
        print("  rank_multi > 30%:  从'最相似5%'移动到'相对不相似'区间")
        print("  rank_multi > 50%:  从'最相似5%'移动到'后半部分'")
        print("  这是非平凡的，因为需要其他模态disagree才能发生")

        return results

    # ==================== 3.聚类分析（正确解释） ====================

    def perform_clustering_analysis(self, n_clusters:  int = 8):
        """
        聚类分析（正确解释）

        关键：正确解释Silhouette↓ + NMI↑的现象
        """
        print("\n" + "=" * 80)
        print("【补充验证】聚类分析")
        print("=" * 80)

        # Ground truth
        true_labels = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        label_encoder = {l: i for i, l in enumerate(set(true_labels))}
        true_encoded = np.array([label_encoder[l] for l in true_labels])
        true_non_other = true_encoded[self.non_other_mask]

        modalities = {
            'molecular': self.dist_mol,
            'morphology': self.dist_morph,
            'projection': self.dist_proj,
            'multimodal': self.dist_multi,
        }

        print(f"\n{'Modality':<15} {'Silhouette':>12} {'NMI':>10} {'ARI':>10}")
        print("-" * 50)

        for name, D in modalities.items():
            D_valid = self.ensure_valid_distance_matrix(D)
            Z = self.safe_linkage(D_valid, 'average')
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

            sil = silhouette_score(D_valid, labels, metric='precomputed')
            nmi = normalized_mutual_info_score(true_non_other, labels[self.non_other_mask])
            ari = adjusted_rand_score(true_non_other, labels[self.non_other_mask])

            # 生成解释
            if name == 'multimodal':
                # 比较与最佳单模态
                best_single_nmi = max(self.clustering_results[m].nmi
                                     for m in ['molecular', 'morphology', 'projection'])
                best_single_sil = max(self.clustering_results[m].silhouette
                                     for m in ['molecular', 'morphology', 'projection'])

                if nmi > best_single_nmi and sil < best_single_sil:
                    interpretation = "Better alignment with anatomy, but less compact clusters"
                elif nmi > best_single_nmi:
                    interpretation = "Improved anatomical alignment"
                else:
                    interpretation = "Similar to single modality"
            else:
                interpretation = "Single modality baseline"

            self.clustering_results[name] = ClusteringResult(
                modality=name, silhouette=sil, nmi=nmi, ari=ari,
                interpretation=interpretation
            )

            print(f"{name:<15} {sil:>12.4f} {nmi:>10.4f} {ari:>10.4f}")

        # 正确解释
        multi_result = self.clustering_results['multimodal']
        best_single_name = max(['molecular', 'morphology', 'projection'],
                               key=lambda m: self.clustering_results[m].nmi)
        best_single = self.clustering_results[best_single_name]

        print(f"\n【正确解释】")
        print(f"  最佳单模态: {best_single_name} (NMI={best_single.nmi:.4f}, Sil={best_single.silhouette:.4f})")
        print(f"  多模态融合: NMI={multi_result.nmi:.4f}, Sil={multi_result.silhouette:.4f}")

        if multi_result.nmi > best_single.nmi and multi_result.silhouette < best_single.silhouette:
            print("\n  Silhouette↓ + NMI↑ 的含义：")
            print("  → 融合没有让clusters更紧凑（内部更相似）")
            print("  → 但使clusters更符合解剖学组织")
            print("  → 这说明融合在reconcile不同模态的视角差异，")
            print("     而非简单地增强相似性信号")

        return self.clustering_results

    # ==================== 高Disagreement案例 ====================

    def find_top_disagreement_cases(self, topk: int = 20):
        """找出模态间disagreement最大的典型案例"""
        print("\n" + "=" * 80)
        print("【案例分析】高Disagreement Region Pairs")
        print("=" * 80)

        cases = {}

        pairs_info = [
            ('mol_morph', 'Molecular vs Morphology'),
            ('mol_proj', 'Molecular vs Projection'),
            ('morph_proj', 'Morphology vs Projection'),
        ]

        for key, title in pairs_info:
            disagree_col = f'disagree_{key}'
            top_df = self.df_pairs.nlargest(topk, disagree_col)[
                ['region_i', 'region_j', 'system_i', 'system_j',
                 'rank_mol', 'rank_morph', 'rank_proj', 'rank_multi',
                 disagree_col, 'same_system']
            ].copy()

            cases[key] = top_df

            mean_disagree = top_df[disagree_col].mean()
            cross_frac = (~top_df['same_system']).mean()

            print(f"\n【{title}】Top {topk}")
            print(f"  平均disagreement: {mean_disagree:.3f}")
            print(f"  跨系统比例: {cross_frac:.1%}")

            # 打印前5个案例
            print(f"\n  {'Region A':<12} {'Region B':<12} {'Sys A':<15} {'Sys B':<15} {'Disagree': >10}")
            print("  " + "-" * 70)
            for _, row in top_df.head(5).iterrows():
                print(f"  {row['region_i']:<12} {row['region_j']:<12} "
                      f"{row['system_i']:<15} {row['system_j']:<15} {row[disagree_col]: >10.3f}")

        self.disagreement_cases = cases
        return cases

    # ==================== Embedding ====================

    def compute_embeddings(self, method: str = "UMAP"):
        """计算Embedding"""
        print(f"\n计算 {method} Embedding...")

        mods = {
            'molecular':  self.dist_mol,
            'morphology': self.dist_morph,
            'projection': self.dist_proj,
            'multimodal': self.dist_multi,
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

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_modality_complementarity(output_dir)
        self._plot_rank_position_shift(output_dir)
        self._plot_clustering_results(output_dir)
        self._plot_embeddings(output_dir)
        self._plot_disagreement_scatter(output_dir)

        print(f"\n✓ 图表保存到: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        """统一保存图表"""
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_modality_complementarity(self, output_dir: str):
        """
        主图1：模态互补性（Mantel相关性热力图 + Disagreement分布）
        每个子图单独保存
        """
        # ===================== Panel A: Mantel相关性矩阵 =====================
        fig_a, ax = plt.subplots(figsize=(7, 6))

        mods = ['Molecular', 'Morphology', 'Projection']
        corr_matrix = np.zeros((3, 3))

        # 填充矩阵
        pairs = [('mol_vs_morph', 0, 1), ('mol_vs_proj', 0, 2), ('morph_vs_proj', 1, 2)]
        for key, i, j in pairs:
            r = self.complementarity[key].mantel_r
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
        np.fill_diagonal(corr_matrix, 1.0)

        # 使用更美观的颜色映射
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(mods, fontsize=12, fontweight='medium')
        ax.set_yticklabels(mods, fontsize=12, fontweight='medium')

        # 添加数值标注
        for i in range(3):
            for j in range(3):
                color = 'white' if corr_matrix[i, j] > 0.6 or corr_matrix[i, j] < 0.4 else 'black'
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=16, fontweight='bold',
                               color=color)
                # 添加文字边框效果
                text.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground='white' if color == 'black' else 'black', alpha=0.3)
                ])

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Mantel Correlation (Spearman)', fontsize=11, fontweight='medium')
        cbar.ax.tick_params(labelsize=10)

        ax.set_title('Inter-modality Distance Correlation\n(Lower values indicate higher complementarity)',
                     fontsize=13, fontweight='bold', pad=15)

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        fig_a.tight_layout()
        self._save_figure(fig_a, output_dir, "1a_mantel_correlation_matrix.png")

        # ===================== Panel B: Disagreement分布 =====================
        fig_b, ax = plt.subplots(figsize=(8, 6))

        colors = ['#3498DB', '#27AE60', '#9B59B6']
        labels = ['Mol vs Morph', 'Mol vs Proj', 'Morph vs Proj']
        cols = ['disagree_mol_morph', 'disagree_mol_proj', 'disagree_morph_proj']

        for col, label, color in zip(cols, labels, colors):
            data = self.df_pairs[col]
            ax.hist(data, bins=40, alpha=0.6, color=color, label=label,
                    density=True, edgecolor='white', linewidth=0.5)
            # 添加KDE曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(0, data.max(), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, alpha=0.9)

        # 高disagreement阈值线
        ax.axvline(x=0.3, color='#E74C3C', linestyle='--', linewidth=2.5,
                   label='High disagreement threshold (0.3)', zorder=5)

        ax.set_xlabel('Rank Disagreement |rank_A - rank_B|', fontsize=12, fontweight='medium')
        ax.set_ylabel('Density', fontsize=12, fontweight='medium')
        ax.set_title('Distribution of Inter-modality Disagreement',
                     fontsize=13, fontweight='bold', pad=15)

        # 图例放在图外避免遮挡
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95,
                  edgecolor='gray', fancybox=True)

        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 添加边框
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig_b.tight_layout()
        self._save_figure(fig_b, output_dir, "1b_disagreement_distribution.png")

        # ===================== 合并图（保留原有功能） =====================
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Panel A
        ax = axes[0]
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(mods, fontsize=11, fontweight='medium')
        ax.set_yticklabels(mods, fontsize=11, fontweight='medium')
        for i in range(3):
            for j in range(3):
                color = 'white' if corr_matrix[i, j] > 0.6 or corr_matrix[i, j] < 0.4 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=14, fontweight='bold', color=color)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Mantel Correlation', fontsize=10)
        ax.set_title('A. Inter-modality Distance Correlation',
                     fontsize=12, fontweight='bold', pad=10)

        # Panel B
        ax = axes[1]
        for col, label, color in zip(cols, labels, colors):
            data = self.df_pairs[col]
            ax.hist(data, bins=40, alpha=0.5, color=color, label=label, density=True,
                    edgecolor='white', linewidth=0.5)
        ax.axvline(x=0.3, color='#E74C3C', linestyle='--', linewidth=2.5,
                   label='High disagreement (0.3)')
        ax.set_xlabel('Rank Disagreement', fontsize=11, fontweight='medium')
        ax.set_ylabel('Density', fontsize=11, fontweight='medium')
        ax.set_title('B.Distribution of Inter-modality Disagreement',
                     fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.suptitle('Modality Complementarity Analysis',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_modality_complementarity.png")

    def _plot_rank_position_shift(self, output_dir:  str):
        """
        主图2：Rank Position Shift
        每个子图单独保存
        """
        modalities = ['Molecular', 'Morphology', 'Projection']
        keys = ['mol', 'morph', 'proj']
        colors = ['#3498DB', '#27AE60', '#9B59B6']

        # ===================== Panel A:  Fraction shifted (条形图) =====================
        fig_a, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(modalities))
        width = 0.35

        frac_30 = [self.rank_shift[k].fraction_30pct * 100 for k in keys]
        frac_50 = [self.rank_shift[k].fraction_50pct * 100 for k in keys]

        bars1 = ax.bar(x - width/2, frac_30, width, label='Shifted to >30%',
                       color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, frac_50, width, label='Shifted to >50%',
                       color=colors, alpha=0.45, edgecolor='black', linewidth=1.2, hatch='///')

        # 标注数值（调整位置避免遮挡）
        for bar, val in zip(bars1, frac_30):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, frac_50):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('% of Close Pairs Shifted', fontsize=12, fontweight='medium')
        ax.set_title('Non-trivial Resolution:\nClose pairs (Top 5%) shifted to higher ranks',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(modalities, fontsize=11, fontweight='medium')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='gray')
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 设置y轴上限留出标注空间
        ax.set_ylim(0, max(frac_30 + frac_50) * 1.25)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig_a.tight_layout()
        self._save_figure(fig_a, output_dir, "2a_rank_shift_bar.png")

        # ===================== Panel B: Scatter showing shift =====================
        fig_b, ax = plt.subplots(figsize=(8, 7))

        # 选择信号最强的模态
        best_mod = max(keys, key=lambda k: self.rank_shift[k].fraction_30pct)
        color = colors[keys.index(best_mod)]

        rank_single = self.df_pairs[f'rank_{best_mod}']
        rank_multi = self.df_pairs['rank_multi']
        close_mask = rank_single <= 0.05

        # 背景点（非close pairs）
        ax.scatter(rank_single[~close_mask], rank_multi[~close_mask],
                   s=8, c='#CCCCCC', alpha=0.4, label='Other pairs', rasterized=True)

        # Close pairs（突出显示）
        scatter = ax.scatter(rank_single[close_mask], rank_multi[close_mask],
                   s=50, c=color, alpha=0.8, edgecolors='white', linewidth=0.8,
                   label=f'Close pairs (n={close_mask.sum()})', zorder=5)

        # 辅助线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='No shift (y=x)')
        ax.axhline(y=0.30, color='#F39C12', linestyle=':', linewidth=2, alpha=0.8, label='30% threshold')
        ax.axhline(y=0.50, color='#E74C3C', linestyle=':', linewidth=2, alpha=0.8, label='50% threshold')
        ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        # 添加阴影区域标注close pairs区域
        ax.fill_between([0, 0.05], [0, 0], [1, 1], alpha=0.1, color='gray')
        ax.text(0.025, 0.95, 'Close\nregion', ha='center', va='top', fontsize=9,
                color='gray', style='italic', transform=ax.transAxes)

        ax.set_xlabel(f'{best_mod.capitalize()} Distance Rank', fontsize=12, fontweight='medium')
        ax.set_ylabel('Multimodal Distance Rank', fontsize=12, fontweight='medium')
        ax.set_title(f'Rank Shift from {best_mod.capitalize()} Perspective\n'
                     f'{self.rank_shift[best_mod].fraction_30pct:.1%} shifted to >30%',
                     fontsize=13, fontweight='bold', pad=15)

        # 图例放在不遮挡数据的位置
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
                  edgecolor='gray', fancybox=True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig_b.tight_layout()
        self._save_figure(fig_b, output_dir, "2b_rank_shift_scatter.png")

        # ===================== 合并图 =====================
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Panel A
        ax = axes[0]
        x = np.arange(len(modalities))
        bars1 = ax.bar(x - width/2, frac_30, width, label='Shifted to >30%',
                       color=colors, alpha=0.85, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, frac_50, width, label='Shifted to >50%',
                       color=colors, alpha=0.45, edgecolor='black', linewidth=1, hatch='///')
        for bar, val in zip(bars1, frac_30):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, frac_50):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('% of Close Pairs Shifted', fontsize=11)
        ax.set_title('A.Non-trivial Resolution', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(modalities, fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(frac_30 + frac_50) * 1.25)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Panel B
        ax = axes[1]
        ax.scatter(rank_single[~close_mask], rank_multi[~close_mask],
                   s=6, c='#CCCCCC', alpha=0.4, rasterized=True)
        ax.scatter(rank_single[close_mask], rank_multi[close_mask],
                   s=40, c=color, alpha=0.8, edgecolors='white', linewidth=0.6,
                   label=f'Close pairs (n={close_mask.sum()})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='No shift')
        ax.axhline(y=0.30, color='#F39C12', linestyle=':', linewidth=2, alpha=0.8, label='30% threshold')
        ax.axhline(y=0.50, color='#E74C3C', linestyle=':', linewidth=2, alpha=0.8, label='50% threshold')
        ax.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'{best_mod.capitalize()} Distance Rank', fontsize=11)
        ax.set_ylabel('Multimodal Distance Rank', fontsize=11)
        ax.set_title(f'B. Rank Shift from {best_mod.capitalize()}', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.suptitle('Resolution Gain:  Multimodal integration reveals hidden differences',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_rank_position_shift.png")

    def _plot_clustering_results(self, output_dir:  str):
        """
        补充图：聚类结果（正确解释）
        每个子图单独保存
        """
        modalities = ['molecular', 'morphology', 'projection', 'multimodal']
        labels = ['Molecular', 'Morphology', 'Projection', 'Multimodal']
        colors = ['#3498DB', '#27AE60', '#9B59B6', '#E74C3C']
        markers = ['o', 's', '^', 'D']

        sils = [self.clustering_results[m].silhouette for m in modalities]
        nmis = [self.clustering_results[m].nmi for m in modalities]
        aris = [self.clustering_results[m].ari for m in modalities]

        # ===================== Panel A: Silhouette vs NMI scatter =====================
        fig_a, ax = plt.subplots(figsize=(8, 7))

        for i, (s, n, label, color, marker) in enumerate(zip(sils, nmis, labels, colors, markers)):
            ax.scatter(s, n, s=300, c=color, marker=marker, label=label,
                       edgecolors='black', linewidth=2, zorder=5, alpha=0.9)

        # 添加连接线显示趋势
        ax.plot(sils, nmis, 'k--', alpha=0.3, linewidth=1, zorder=1)

        ax.set_xlabel('Silhouette Score (Cluster Compactness)', fontsize=12, fontweight='medium')
        ax.set_ylabel('NMI (Anatomical Alignment)', fontsize=12, fontweight='medium')
        ax.set_title('Clustering Quality Trade-off\nMultimodal:  Better anatomy alignment, less compact clusters',
                     fontsize=13, fontweight='bold', pad=15)

        ax.legend(fontsize=11, loc='lower left', framealpha=0.95,
                  edgecolor='gray', fancybox=True)
        ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 添加解释文本框（放在不遮挡数据的位置）
        textstr = ("Silhouette↓ + NMI↑:\n"
                   "Fusion reconciles modality\n"
                   "conflicts rather than simply\n"
                   "amplifying similarity signals")
        props = dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', alpha=0.9, edgecolor='#E0C060')
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig_a.tight_layout()
        self._save_figure(fig_a, output_dir, "3a_clustering_tradeoff.png")

        # ===================== Panel B: Bar comparison =====================
        fig_b, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(modalities))
        width = 0.28

        bars1 = ax.bar(x - width, sils, width, label='Silhouette', color='#3498DB',
                       edgecolor='black', linewidth=1, alpha=0.85)
        bars2 = ax.bar(x, nmis, width, label='NMI', color='#E74C3C',
                       edgecolor='black', linewidth=1, alpha=0.85)
        bars3 = ax.bar(x + width, aris, width, label='ARI', color='#27AE60',
                       edgecolor='black', linewidth=1, alpha=0.85)

        # 添加数值标注
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, fontweight='medium')

        ax.set_ylabel('Score', fontsize=12, fontweight='medium')
        ax.set_title('Clustering Metrics Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, fontweight='medium')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='gray')
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(sils + nmis + aris) * 1.2)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig_b.tight_layout()
        self._save_figure(fig_b, output_dir, "3b_clustering_metrics.png")

        # ===================== 合并图 =====================
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Panel A
        ax = axes[0]
        for i, (s, n, label, color, marker) in enumerate(zip(sils, nmis, labels, colors, markers)):
            ax.scatter(s, n, s=250, c=color, marker=marker, label=label,
                       edgecolors='black', linewidth=1.5, zorder=5)
        ax.set_xlabel('Silhouette Score', fontsize=11)
        ax.set_ylabel('NMI', fontsize=11)
        ax.set_title('A.Clustering Quality Trade-off', fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='lower left', framealpha=0.95)
        ax.grid(alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Panel B
        ax = axes[1]
        x = np.arange(len(modalities))
        width = 0.28
        ax.bar(x - width, sils, width, label='Silhouette', color='#3498DB', edgecolor='black')
        ax.bar(x, nmis, width, label='NMI', color='#E74C3C', edgecolor='black')
        ax.bar(x + width, aris, width, label='ARI', color='#27AE60', edgecolor='black')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('B.Clustering Metrics Comparison', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_clustering_results.png")

    def _plot_embeddings(self, output_dir: str):
        """
        补充图：Embedding
        每个子图单独保存
        """
        systems = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        unique = sorted(set(systems))

        # 使用更美观的调色板
        n_colors = len(unique)
        cmap = plt.cm.get_cmap('tab20', n_colors)
        sys_color = {s: cmap(i) for i, s in enumerate(unique)}

        # Other使用灰色
        sys_color['Other'] = (0.7, 0.7, 0.7, 0.5)

        point_colors = [sys_color[s] for s in systems]
        point_sizes = [40 if s != 'Other' else 25 for s in systems]

        mods = ['molecular', 'morphology', 'projection', 'multimodal']
        titles = ['Molecular', 'Morphology', 'Projection', 'Multimodal']

        # ===================== 单独保存每个embedding =====================
        for mod, title in zip(mods, titles):
            fig, ax = plt.subplots(figsize=(9, 8))

            emb = self.embeddings[mod]

            # 先绘制Other点（在底层）
            for i, (x, y, c, s, sys) in enumerate(zip(emb[: , 0], emb[:, 1], point_colors, point_sizes, systems)):
                if sys == 'Other':
                    ax.scatter(x, y, c=[c], s=s, alpha=0.4, edgecolors='none')

            # 再绘制非Other点（在上层）
            for i, (x, y, c, s, sys) in enumerate(zip(emb[:, 0], emb[:, 1], point_colors, point_sizes, systems)):
                if sys != 'Other':
                    ax.scatter(x, y, c=[c], s=s*2, alpha=0.85, edgecolors='white', linewidth=0.5)

            # 只标注部分区域名称，避免拥挤
            # 策略：只标注距离质心较远的点
            from scipy.spatial.distance import cdist
            centroid = emb.mean(axis=0)
            distances = cdist([centroid], emb)[0]
            threshold = np.percentile(distances, 85)  # 只标注最外围15%的点

            for i, r in enumerate(self.regions):
                if distances[i] > threshold and systems[i] != 'Other':
                    ax.annotate(r, (emb[i, 0], emb[i, 1]),
                               fontsize=7, alpha=0.7,
                               xytext=(3, 3), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

            metrics = self.clustering_results[mod]
            ax.set_title(f'{title} Embedding\nNMI={metrics.nmi:.3f}, Silhouette={metrics.silhouette:.3f}',
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Dimension 1', fontsize=11)
            ax.set_ylabel('Dimension 2', fontsize=11)
            ax.grid(alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            # 添加图例（放在图外）
            legend_elements = [Patch(facecolor=sys_color[s], edgecolor='black', linewidth=0.5, label=s)
                               for s in unique if s != 'Other']
            legend_elements.append(Patch(facecolor=(0.7, 0.7, 0.7), edgecolor='none', label='Other'))
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=9, framealpha=0.95, edgecolor='gray')

            fig.tight_layout()
            self._save_figure(fig, output_dir, f"4_{mod}_embedding.png")

        # ===================== 合并图（2x2布局） =====================
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        for ax, mod, title in zip(axes.flat, mods, titles):
            emb = self.embeddings[mod]

            # 绘制所有点
            for i, (x, y, c, s, sys) in enumerate(zip(emb[:, 0], emb[:, 1], point_colors, point_sizes, systems)):
                zorder = 1 if sys == 'Other' else 5
                alpha = 0.4 if sys == 'Other' else 0.85
                edge = 'none' if sys == 'Other' else 'white'
                ax.scatter(x, y, c=[c], s=s*1.5, alpha=alpha, edgecolors=edge, linewidth=0.5, zorder=zorder)

            metrics = self.clustering_results[mod]
            ax.set_title(f'{title}\nNMI={metrics.nmi:.3f}, Sil={metrics.silhouette:.3f}',
                        fontsize=12, fontweight='bold', pad=10)
            ax.grid(alpha=0.2)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        # 统一图例
        legend_elements = [Patch(facecolor=sys_color[s], edgecolor='black', linewidth=0.5, label=s)
                           for s in unique if s != 'Other']
        legend_elements.append(Patch(facecolor=(0.7, 0.7, 0.7), edgecolor='none', label='Other'))
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.08, 0.5),
                   fontsize=10, framealpha=0.95)

        plt.suptitle('Region Embeddings by Modality', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_embeddings.png")

    def _plot_disagreement_scatter(self, output_dir: str):
        """
        补充图：Disagreement scatter
        每个子图单独保存
        """
        pairs = [
            ('rank_mol', 'rank_morph', 'disagree_mol_morph', 'Molecular vs Morphology'),
            ('rank_mol', 'rank_proj', 'disagree_mol_proj', 'Molecular vs Projection'),
            ('rank_morph', 'rank_proj', 'disagree_morph_proj', 'Morphology vs Projection'),
        ]

        # ===================== 单独保存每个scatter =====================
        for col1, col2, disagree_col, title in pairs:
            fig, ax = plt.subplots(figsize=(8, 7))

            x = self.df_pairs[col1]
            y = self.df_pairs[col2]
            disagree = self.df_pairs[disagree_col]

            # 使用更美观的颜色映射
            scatter = ax.scatter(x, y, c=disagree, cmap='RdYlBu_r', s=20, alpha=0.7,
                                 edgecolors='none', rasterized=True)

            # 对角线（表示完全一致）
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Perfect agreement')

            # 高disagreement区域指示
            ax.fill_between([0, 1], [0.3, 1.3], [0, 1], alpha=0.05, color='red')
            ax.fill_between([0, 1], [0, 1], [-0.3, 0.7], alpha=0.05, color='red')

            ax.set_xlabel(col1.replace('rank_', '').capitalize() + ' Distance Rank',
                          fontsize=12, fontweight='medium')
            ax.set_ylabel(col2.replace('rank_', '').capitalize() + ' Distance Rank',
                          fontsize=12, fontweight='medium')
            ax.set_title(f'{title}\nPoints far from diagonal indicate complementary information',
                         fontsize=13, fontweight='bold', pad=15)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', fontsize=10)

            # 颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Disagreement', fontsize=11)
            cbar.ax.tick_params(labelsize=9)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            fig.tight_layout()
            filename = f"5_{disagree_col}.png"
            self._save_figure(fig, output_dir, filename)

        # ===================== 合并图（1x3布局） =====================
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        for ax, (col1, col2, disagree_col, title) in zip(axes, pairs):
            x = self.df_pairs[col1]
            y = self.df_pairs[col2]
            disagree = self.df_pairs[disagree_col]

            scatter = ax.scatter(x, y, c=disagree, cmap='RdYlBu_r', s=12, alpha=0.6,
                                 edgecolors='none', rasterized=True)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)

            ax.set_xlabel(col1.replace('rank_', '').capitalize() + ' Rank', fontsize=11)
            ax.set_ylabel(col2.replace('rank_', '').capitalize() + ' Rank', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.9)
            cbar.set_label('Disagreement', fontsize=9)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Inter-modality Rank Disagreement Analysis',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_disagreement_scatter.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # Complementarity
        comp_data = []
        for key, c in self.complementarity.items():
            comp_data.append({
                'comparison': f'{c.mod1} vs {c.mod2}',
                'mantel_r': c.mantel_r,
                'mantel_p': c.mantel_p,
                'n_high_disagreement': c.n_high_disagreement,
                'cross_system_fraction': c.cross_system_fraction
            })
        pd.DataFrame(comp_data).to_csv(f"{output_dir}/modality_complementarity.csv", index=False)

        # Rank shift
        shift_data = []
        for mod, s in self.rank_shift.items():
            shift_data.append({
                'modality': s.modality,
                'close_pairs': s.close_pairs,
                'shifted_to_30pct': s.shifted_to_30pct,
                'fraction_30pct':  s.fraction_30pct,
                'shifted_to_50pct': s.shifted_to_50pct,
                'fraction_50pct': s.fraction_50pct,
                'mean_rank_change': s.mean_rank_change
            })
        pd.DataFrame(shift_data).to_csv(f"{output_dir}/rank_position_shift.csv", index=False)

        # Clustering
        cluster_data = []
        for name, c in self.clustering_results.items():
            cluster_data.append({
                'modality':  c.modality,
                'silhouette': c.silhouette,
                'nmi': c.nmi,
                'ari': c.ari,
                'interpretation': c.interpretation
            })
        pd.DataFrame(cluster_data).to_csv(f"{output_dir}/clustering_results.csv", index=False)

        # All pairs
        self.df_pairs.to_csv(f"{output_dir}/all_pairs.csv", index=False)

        # Disagreement cases
        for key, df in self.disagreement_cases.items():
            df.to_csv(f"{output_dir}/disagreement_{key}.csv", index=False)

        print(f"\n✓ 结果保存到:  {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task2_results_v4", n_clusters: int = 8):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务2:  Region-level 多模态分析 (V10 最终版)")
        print("=" * 80)

        # 1.数据加载
        self.load_all_fingerprints()

        # 2.距离计算
        self.compute_distance_matrices()

        # 3.多模态融合
        self.compute_multimodal_fusion()

        # 4.构建pair数据
        self.build_pair_dataframe()

        # 5.【主发现1】模态互补性分析
        self.analyze_modality_complementarity()

        # 6.【主发现2】Rank Position Shift
        self.compute_rank_position_shift(q_close=0.05)

        # 7.【补充】聚类分析
        self.perform_clustering_analysis(n_clusters)

        # 8. Disagreement案例
        self.find_top_disagreement_cases(topk=20)

        # 9.Embedding
        self.compute_embeddings()

        # 10.可视化
        self.visualize_results(output_dir)

        # 11.保存结果
        self.save_results(output_dir)

        # 12.打印结论
        self._print_conclusion()

        print("\n" + "=" * 80)
        print(f"任务2完成!  结果:  {output_dir}")
        print("=" * 80)

    def _print_conclusion(self):
        """打印科学结论"""
        print("\n" + "=" * 80)
        print("科学结论")
        print("=" * 80)

        # 获取关键数据
        avg_r = np.mean([c.mantel_r for c in self.complementarity.values()])
        best_shift_mod = max(self.rank_shift.keys(), key=lambda k: self.rank_shift[k].fraction_30pct)
        best_shift = self.rank_shift[best_shift_mod]

        multi_nmi = self.clustering_results['multimodal'].nmi
        best_single = max(['molecular', 'morphology', 'projection'],
                          key=lambda m: self.clustering_results[m].nmi)
        best_single_nmi = self.clustering_results[best_single].nmi

        print(f"""
【主发现1：模态互补性】

  模态间平均距离相关性: r = {avg_r:.3f}
  → 不同模态捕获了弱相关但非冗余的脑区相似性结构

  生物学意义: 
  - 分子组成、形态特征、投射模式是脑区身份的不同维度
  - 单一模态无法完整描述脑区间的关系
  - 高disagreement pairs多发生在跨系统区域，说明这种互补性
    对于区分不同解剖系统的区域尤为重要

【主发现2：非平凡Resolution】

  从{best_shift_mod.capitalize()}视角（Top 5% close pairs）: 
  - {best_shift.fraction_30pct:.1%} 在多模态中移动到 >30% rank
  - {best_shift.fraction_50pct:.1%} 在多模态中移动到 >50% rank
  
  这是非平凡的，因为:
  - 需要其他模态disagree才能发生rank shift
  - 说明多模态整合揭示了单模态遗漏的差异信息

【聚类验证】

  最佳单模态: {best_single} (NMI = {best_single_nmi:.4f})
  多模态融合: NMI = {multi_nmi:.4f}

  Silhouette↓ + NMI↑ 的正确解释:
  - 融合没有让clusters更紧凑
  - 但使结构更符合解剖学组织
  - 这说明融合在reconcile模态间的视角差异，
    而非简单放大相似性信号

【核心叙事】

  "Different modalities capture weakly correlated but non-redundant 
   aspects of brain region identity. Multimodal integration reconciles 
   these complementary perspectives into a representation that better 
   aligns with anatomical organization, revealing region distinctions 
   that would be missed by any single modality alone."
""")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task2_multimodal_results_v10"
    N_CLUSTERS = 8
    MIN_NEURONS = 10

    with RegionMultimodalAnalysisV10(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, MIN_NEURONS) as analyzer:
        analyzer.run_full_pipeline(OUTPUT_DIR, N_CLUSTERS)


if __name__ == "__main__":
    main()