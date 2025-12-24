"""
模态互补性与聚类质量分析 (修复版)
==================================
三大部分：
1. 模态间相关性 + 异质性：证明互补性
   - 在 CCA/RSA/Retrieval 空间评估跨模态关系

2. 单模态 vs 多模态聚类：K值扫描 + 稳定性
   - 单模态：各自 PCA 空间
   - 多模态：PCA 拼接后再降维的 joint embedding

3. 同质性 + Confusion Score：结构性结论
   - 同质性：基于外部标签 (subclass) 的 homogeneity_score
   - Confusion Score: within_dist / nearest_between_dist

修复内容：
1. 移除"统一CCA空间"的说法，明确各部分用的空间
2. K选择改用 test silhouette 或组合指标
3. stability 改用 subsample without replacement (80%)
4. 添加基于外部标签的 homogeneity 指标
5. confusion_score 定义明确化
6. 增加 fixed-K 对比分析

Author: Claude
Date: 2025-01
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import neo4j

# ==================== Nature Methods 样式 ====================

def setup_nm_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 0.75,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
    })

NM_COLORS = {
    'blue': '#4477AA',
    'cyan': '#66CCEE',
    'green': '#228833',
    'yellow': '#CCBB44',
    'red': '#EE6677',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
    'dark_blue': '#004488',
    'orange': '#EE7733',
}

MODALITY_COLORS = {
    'Morph': NM_COLORS['blue'],
    'Gene': NM_COLORS['green'],
    'Proj': NM_COLORS['red'],
    'Morph+Gene': NM_COLORS['purple'],
    'Gene+Proj': NM_COLORS['cyan'],
    'Morph+Proj': NM_COLORS['orange'],
    'All': NM_COLORS['dark_blue'],
}

SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4


def save_panel(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        fig.savefig(f"{output_dir}/{name}.{fmt}", format=fmt, dpi=300,
                   bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {name}")


# ==================== 数据结构 ====================

@dataclass
class ModalityData:
    name: str
    vectors: np.ndarray
    raw_dim: int
    pca_dim: int
    variance_explained: float
    effective_dim: int = 0


@dataclass
class CrossModalMetrics:
    modality_pair: Tuple[str, str]
    cca_correlations: np.ndarray
    cca_mean_corr: float
    rv_coefficient: float
    rsa_correlation: float
    rsa_pvalue: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_rank: float
    mrr: float


@dataclass
class ClusteringResult:
    representation: str
    k: int
    silhouette_train: float
    silhouette_test: float
    within_cluster_dist: float
    between_cluster_dist: float
    confusion_score: float  # within / nearest_between
    stability_ari: float
    stability_nmi: float
    # 基于外部标签的同质性指标
    homogeneity: float  # 需要外部标签
    completeness: float
    v_measure: float
    # 其他
    cluster_sizes: np.ndarray
    size_imbalance: float
    labels_train: np.ndarray
    labels_test: np.ndarray


# ==================== 核心分析类 ====================

class ModalityComplementarityAnalysis:
    """
    模态互补性与聚类质量分析 (修复版)

    空间说明：
    - Part1 跨模态分析：CCA / RSA / Retrieval（各自计算跨模态相关性）
    - Part2/3 聚类分析：
      - 单模态：各自 PCA 空间
      - 多模态：PCA 拼接后标准化（joint embedding）
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

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 4.0,
                 pca_variance_threshold: float = 0.95,
                 test_ratio: float = 0.2,
                 cca_n_components: int = 15,
                 k_range: Tuple[int, int] = (2, 25),
                 n_stability_runs: int = 20,
                 stability_subsample_ratio: float = 0.8,
                 k_selection_method: str = 'combined',  # 'train', 'test', 'combined'
                 output_dir: str = "./modality_analysis"):

        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.cca_n_components = cca_n_components
        self.k_range = k_range
        self.n_stability_runs = n_stability_runs
        self.stability_subsample_ratio = stability_subsample_ratio
        self.k_selection_method = k_selection_method
        self.output_dir = output_dir

        setup_nm_style()
        os.makedirs(output_dir, exist_ok=True)

        # 数据容器
        self.valid_neuron_ids: List[str] = []
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

        # 原始数据
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 外部标签（用于homogeneity）
        self.external_labels: np.ndarray = None  # subclass 编码
        self.external_label_names: List[str] = []
        self.neuron_subclass_map: Dict[str, str] = {}

        # 处理后的模态数据
        self.modalities: Dict[str, ModalityData] = {}

        # 聚类空间表示
        self.clustering_representations: Dict[str, np.ndarray] = {}

        # 分析结果
        self.cross_modal_metrics: Dict[str, CrossModalMetrics] = {}
        self.clustering_results: Dict[str, Dict[int, ClusteringResult]] = {}
        self.optimal_k: Dict[str, int] = {}
        self.heterogeneity_results: pd.DataFrame = None
        self.quality_summary: pd.DataFrame = None

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 ====================

    def load_all_data(self) -> int:
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._load_external_labels()
        self._filter_valid_neurons()
        self._split_train_test()
        self._process_all_modalities()
        self._create_clustering_representations()

        print(f"\n✓ Data loading complete: {len(self.valid_neuron_ids)} neurons")
        print(f"  External labels: {len(self.external_label_names)} unique celltypes")
        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.local_gene_features_raw = cache_data['local_environments']
        self.all_subclasses = cache_data['all_subclasses']
        print(f"  Loaded molecular environment for {len(self.local_gene_features_raw)} neurons")

    def _get_global_dimensions(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]
        print(f"  Projection targets: {len(self.all_target_regions)} regions")

    def _load_all_neuron_features(self):
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        RETURN n.neuron_id AS neuron_id, {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id: $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                neuron_id = record['neuron_id']
                axon_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.AXONAL_FEATURES]
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                dend_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.DENDRITIC_FEATURES]
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                proj_result = session.run(proj_query, neuron_id=neuron_id)
                proj_dict = {r['target']: r['weight'] for r in proj_result if r['target']}
                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        proj_vector[i] = proj_dict.get(target, 0)
                    self.projection_vectors_raw[neuron_id] = proj_vector

        print(f"  Loaded {len(self.axon_features_raw)} neurons")

    def _load_external_labels(self):
        """
        加载外部标签用于 homogeneity 计算

        根据知识图谱结构，可用的标签来源：
        1. celltype 属性 (如 "VISp1", "MOp5") - 存储在 Neuron 节点上
        2. base_region 属性 - 从 celltype 提取的基础区域
        3. 通过 LOCATE_AT 关系连接的 Region

        这里使用 celltype 作为外部标签（最细粒度）
        """
        print("  Loading external labels (celltype from Neuron nodes)...")

        # 方案1：直接从 Neuron 节点获取 celltype 属性
        query = """
        MATCH (n:Neuron)
        WHERE n.neuron_id IS NOT NULL AND n.celltype IS NOT NULL
        RETURN n.neuron_id AS neuron_id, 
               n.celltype AS celltype,
               n.base_region AS base_region
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                neuron_id = record['neuron_id']
                celltype = record['celltype']
                if celltype:
                    self.neuron_subclass_map[neuron_id] = celltype

        print(f"    Found celltype for {len(self.neuron_subclass_map)} neurons")

        # 如果 celltype 不可用，尝试通过 LOCATE_AT 关系获取 Region
        if len(self.neuron_subclass_map) == 0:
            print("    Celltype not available, trying LOCATE_AT -> Region...")
            query2 = """
            MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
            WHERE n.neuron_id IS NOT NULL
            RETURN n.neuron_id AS neuron_id, 
                   r.acronym AS region_acronym
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(query2)
                for record in result:
                    neuron_id = record['neuron_id']
                    region = record['region_acronym']
                    if region:
                        self.neuron_subclass_map[neuron_id] = region

            print(f"    Found region for {len(self.neuron_subclass_map)} neurons via LOCATE_AT")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        self.valid_neuron_ids = sorted(list(candidates))

        # 编码外部标签 (celltype)
        celltypes_for_valid = []
        for nid in self.valid_neuron_ids:
            ct = self.neuron_subclass_map.get(nid, 'Unknown')
            celltypes_for_valid.append(ct)

        # 创建标签编码
        unique_celltypes = sorted(set(celltypes_for_valid))
        self.external_label_names = unique_celltypes
        celltype_to_idx = {ct: i for i, ct in enumerate(unique_celltypes)}
        self.external_labels = np.array([celltype_to_idx[ct] for ct in celltypes_for_valid])

        print(f"  Valid neurons: {len(self.valid_neuron_ids)}")
        print(f"  Unique celltypes: {len(unique_celltypes)}")

        # 显示标签分布
        unknown_count = celltypes_for_valid.count('Unknown')
        if unknown_count > 0:
            print(f"    (Note: {unknown_count} neurons have Unknown celltype)")

    def _split_train_test(self):
        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]
        print(f"  Train/Test: {len(self.train_idx)}/{len(self.test_idx)}")

    def _process_all_modalities(self):
        """处理三个模态，计算PCA"""
        print("\nProcessing modalities (PCA space)...")
        neurons = self.valid_neuron_ids

        # Morphology
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.modalities['Morph'] = self._process_modality(morph_raw, 'Morph')

        # Molecular
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        gene_raw = gene_raw[:, col_sums > 0]
        self.modalities['Gene'] = self._process_modality(gene_raw, 'Gene')

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        proj_raw = proj_raw[:, col_sums > 0]
        self.modalities['Proj'] = self._process_modality(proj_raw, 'Proj')

    def _process_modality(self, X_raw: np.ndarray, name: str) -> ModalityData:
        raw_dim = X_raw.shape[1]
        X = np.log1p(X_raw)

        scaler = StandardScaler()
        scaler.fit(X[self.train_idx])
        X_scaled = scaler.transform(X)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(X_scaled[self.train_idx])
        X_pca = pca.transform(X_scaled)

        pca_dim = X_pca.shape[1]
        variance = np.sum(pca.explained_variance_ratio_)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.searchsorted(cumsum, 0.80) + 1

        print(f"  {name}: {raw_dim}D → {pca_dim}D (var={variance:.1%}, eff_dim={effective_dim})")

        return ModalityData(
            name=name,
            vectors=X_pca,
            raw_dim=raw_dim,
            pca_dim=pca_dim,
            variance_explained=variance,
            effective_dim=effective_dim
        )

    def _create_clustering_representations(self):
        """
        创建聚类用的表示空间
        - 单模态：各自 PCA 空间
        - 多模态：PCA 拼接后标准化（joint embedding）
        """
        print("\nCreating clustering representations...")

        # 单模态：直接用 PCA
        for name in ['Morph', 'Gene', 'Proj']:
            self.clustering_representations[name] = self.modalities[name].vectors

        # 多模态：拼接后标准化
        morph = self.modalities['Morph'].vectors
        gene = self.modalities['Gene'].vectors
        proj = self.modalities['Proj'].vectors

        def concat_and_standardize(*arrays):
            """拼接后在训练集上标准化"""
            concat = np.hstack(arrays)
            scaler = StandardScaler()
            scaler.fit(concat[self.train_idx])
            return scaler.transform(concat)

        self.clustering_representations['Morph+Gene'] = concat_and_standardize(morph, gene)
        self.clustering_representations['Gene+Proj'] = concat_and_standardize(gene, proj)
        self.clustering_representations['Morph+Proj'] = concat_and_standardize(morph, proj)
        self.clustering_representations['All'] = concat_and_standardize(morph, gene, proj)

        for name, vec in self.clustering_representations.items():
            print(f"  {name}: {vec.shape[1]}D")

    # ==================== Part 1: 模态间相关性 + 异质性 ====================

    def analyze_cross_modal_correlation(self):
        """
        分析模态间相关性（在训练集上）
        注意：这里的 CCA/RSA/Retrieval 是跨模态分析，不是聚类空间
        """
        print("\n" + "=" * 80)
        print("Part 1: Cross-Modal Correlation Analysis")
        print("(Using CCA/RSA/Retrieval to assess modality relationships)")
        print("=" * 80)

        pairs = [('Morph', 'Gene'), ('Gene', 'Proj'), ('Morph', 'Proj')]

        for m1, m2 in pairs:
            print(f"\n--- {m1} ↔ {m2} ---")

            X1 = self.modalities[m1].vectors[self.train_idx]
            X2 = self.modalities[m2].vectors[self.train_idx]

            # 1. CCA
            cca_corrs, cca_mean = self._compute_cca(X1, X2)
            print(f"  CCA: mean_corr={cca_mean:.4f}, top3={cca_corrs[:3]}")

            # 2. RV coefficient
            rv = self._compute_rv_coefficient(X1, X2)
            print(f"  RV coefficient: {rv:.4f}")

            # 3. RSA
            rsa_corr, rsa_p = self._compute_rsa(X1, X2)
            print(f"  RSA: r={rsa_corr:.4f}, p={rsa_p:.2e}")

            # 4. Cross-modal retrieval
            r1, r5, r10, mr, mrr = self._compute_cross_modal_retrieval(X1, X2)
            print(f"  Retrieval: R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}, MRR={mrr:.3f}")

            self.cross_modal_metrics[f'{m1}-{m2}'] = CrossModalMetrics(
                modality_pair=(m1, m2),
                cca_correlations=cca_corrs,
                cca_mean_corr=cca_mean,
                rv_coefficient=rv,
                rsa_correlation=rsa_corr,
                rsa_pvalue=rsa_p,
                recall_at_1=r1,
                recall_at_5=r5,
                recall_at_10=r10,
                mean_rank=mr,
                mrr=mrr
            )

    def _compute_cca(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, float]:
        n_components = min(self.cca_n_components, X1.shape[1], X2.shape[1])
        cca = CCA(n_components=n_components)
        X1_c, X2_c = cca.fit_transform(X1, X2)

        correlations = []
        for i in range(n_components):
            corr, _ = pearsonr(X1_c[:, i], X2_c[:, i])
            correlations.append(abs(corr))

        correlations = np.array(correlations)
        return correlations, np.mean(correlations)

    def _compute_rv_coefficient(self, X1: np.ndarray, X2: np.ndarray) -> float:
        X1_c = X1 - X1.mean(axis=0)
        X2_c = X2 - X2.mean(axis=0)

        S1 = X1_c @ X1_c.T
        S2 = X2_c @ X2_c.T

        numerator = np.trace(S1 @ S2)
        denominator = np.sqrt(np.trace(S1 @ S1) * np.trace(S2 @ S2))

        return numerator / denominator if denominator > 0 else 0

    def _compute_rsa(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float]:
        """RSA: 距离矩阵的 Spearman 相关"""
        D1 = squareform(pdist(X1, metric='euclidean'))
        D2 = squareform(pdist(X2, metric='euclidean'))

        triu_idx = np.triu_indices(len(D1), k=1)
        d1_vec = D1[triu_idx]
        d2_vec = D2[triu_idx]

        corr, pvalue = spearmanr(d1_vec, d2_vec)
        return corr, pvalue

    def _compute_cross_modal_retrieval(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Cross-modal retrieval: 用 X1 检索 X2 中的同一样本

        由于两个模态维度不同，需要先投影到 CCA 空间
        然后在对齐的空间中计算 cosine 相似度
        """
        n = len(X1)

        # 投影到 CCA 空间（维度对齐）
        n_components = min(self.cca_n_components, X1.shape[1], X2.shape[1])
        cca = CCA(n_components=n_components)
        X1_cca, X2_cca = cca.fit_transform(X1, X2)

        # 在 CCA 空间中计算 cosine 相似度
        X1_norm = X1_cca / (np.linalg.norm(X1_cca, axis=1, keepdims=True) + 1e-8)
        X2_norm = X2_cca / (np.linalg.norm(X2_cca, axis=1, keepdims=True) + 1e-8)

        sim_matrix = X1_norm @ X2_norm.T  # 现在两者都是 n_components 维

        ranks = []
        for i in range(n):
            sorted_indices = np.argsort(-sim_matrix[i])
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)

        ranks = np.array(ranks)

        return (np.mean(ranks == 1), np.mean(ranks <= 5), np.mean(ranks <= 10),
                np.mean(ranks), np.mean(1.0 / ranks))

    def analyze_modality_heterogeneity(self):
        """分析各模态的异质性/内部复杂度"""
        print("\n" + "=" * 80)
        print("Part 1b: Modality Heterogeneity Analysis")
        print("=" * 80)

        results = []

        for name, mod in self.modalities.items():
            X_train = mod.vectors[self.train_idx]

            # 1. 有效维度
            effective_dim = mod.effective_dim

            # 2. 局部密度变异
            nn = NearestNeighbors(n_neighbors=min(10, len(X_train) - 1))
            nn.fit(X_train)
            distances, _ = nn.kneighbors(X_train)
            mean_knn_dist = distances[:, 1:].mean(axis=1)
            density_cv = mean_knn_dist.std() / (mean_knn_dist.mean() + 1e-8)

            # 3. 单模态最优聚类的silhouette上限
            best_sil = -1
            best_k = 2
            for k in range(2, min(15, len(X_train) // 10)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_train)
                sil = silhouette_score(X_train, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k

            results.append({
                'modality': name,
                'raw_dim': mod.raw_dim,
                'pca_dim': mod.pca_dim,
                'effective_dim': effective_dim,
                'density_cv': density_cv,
                'best_silhouette': best_sil,
                'best_k': best_k,
            })

            print(f"\n  {name}:")
            print(f"    Raw dim: {mod.raw_dim}, PCA dim: {mod.pca_dim}")
            print(f"    Effective dim (80%): {effective_dim}")
            print(f"    Density CV: {density_cv:.4f}")
            print(f"    Best silhouette: {best_sil:.4f} at K={best_k}")

        self.heterogeneity_results = pd.DataFrame(results)
        return self.heterogeneity_results

    # ==================== Part 2: 聚类 K 值扫描 ====================

    def analyze_clustering_k_scan(self):
        """
        对每个表示扫描 K 值
        使用修复后的 stability 计算方法（subsample without replacement）
        """
        print("\n" + "=" * 80)
        print("Part 2: Clustering K-Scan Analysis")
        print(f"(Stability: {self.n_stability_runs} runs, {self.stability_subsample_ratio:.0%} subsample)")
        print(f"(K selection: {self.k_selection_method})")
        print("=" * 80)

        k_min, k_max = self.k_range

        for rep_name, vectors in self.clustering_representations.items():
            print(f"\n--- {rep_name} ({vectors.shape[1]}D) ---")

            X_train = vectors[self.train_idx]
            X_test = vectors[self.test_idx]
            y_train = self.external_labels[self.train_idx]
            y_test = self.external_labels[self.test_idx]

            self.clustering_results[rep_name] = {}
            best_score = -np.inf
            best_k = k_min

            for k in range(k_min, k_max + 1):
                result = self._evaluate_clustering(X_train, X_test, y_train, y_test, k, rep_name)
                self.clustering_results[rep_name][k] = result

                # K 选择逻辑
                if self.k_selection_method == 'train':
                    score = result.silhouette_train
                elif self.k_selection_method == 'test':
                    score = result.silhouette_test
                else:  # combined
                    # 组合指标：test_sil + 0.3*stability - 0.2*confusion
                    score = result.silhouette_test + 0.3 * result.stability_ari - 0.2 * result.confusion_score

                if score > best_score:
                    best_score = score
                    best_k = k

                if k % 5 == 0 or k == k_min:
                    print(f"    K={k}: sil_train={result.silhouette_train:.4f}, "
                          f"sil_test={result.silhouette_test:.4f}, "
                          f"stability={result.stability_ari:.4f}, "
                          f"homogeneity={result.homogeneity:.4f}")

            self.optimal_k[rep_name] = best_k
            opt_result = self.clustering_results[rep_name][best_k]
            print(f"  → Optimal K={best_k}, sil_test={opt_result.silhouette_test:.4f}")

    def _evaluate_clustering(self, X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             k: int, rep_name: str) -> ClusteringResult:
        """评估单个 K 值的聚类质量"""
        # 主聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)

        labels_train = kmeans.predict(X_train)
        labels_test = kmeans.predict(X_test)

        # Silhouette
        sil_train = silhouette_score(X_train, labels_train)
        sil_test = silhouette_score(X_test, labels_test)

        # Within-cluster distance
        within_dist = self._compute_within_cluster_distance(X_train, labels_train)

        # Between-cluster distance
        between_dist = self._compute_between_cluster_distance(X_train, labels_train, kmeans.cluster_centers_)

        # Confusion score = within / nearest_between
        confusion = within_dist / between_dist if between_dist > 0 else float('inf')

        # 修复后的 stability 计算
        stability_ari, stability_nmi = self._compute_clustering_stability_fixed(X_train, k)

        # 基于外部标签的同质性指标
        homogeneity = homogeneity_score(y_train, labels_train)
        completeness = completeness_score(y_train, labels_train)
        v_measure = v_measure_score(y_train, labels_train)

        # Cluster sizes
        cluster_sizes = np.bincount(labels_train, minlength=k)
        size_imbalance = cluster_sizes.std() / (cluster_sizes.mean() + 1e-8)

        return ClusteringResult(
            representation=rep_name,
            k=k,
            silhouette_train=sil_train,
            silhouette_test=sil_test,
            within_cluster_dist=within_dist,
            between_cluster_dist=between_dist,
            confusion_score=confusion,
            stability_ari=stability_ari,
            stability_nmi=stability_nmi,
            homogeneity=homogeneity,
            completeness=completeness,
            v_measure=v_measure,
            cluster_sizes=cluster_sizes,
            size_imbalance=size_imbalance,
            labels_train=labels_train,
            labels_test=labels_test
        )

    def _compute_within_cluster_distance(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        计算加权平均簇内距离
        W = Σ_c (n_c / N) * mean_dist(X_c)
        """
        unique_labels = np.unique(labels)
        total_dist = 0
        n_total = len(X)

        for label in unique_labels:
            mask = labels == label
            X_cluster = X[mask]
            n_cluster = len(X_cluster)

            if n_cluster > 1:
                # 为了效率，采样计算
                if n_cluster > 500:
                    idx = np.random.choice(n_cluster, 500, replace=False)
                    X_sample = X_cluster[idx]
                    dists = pdist(X_sample, metric='euclidean')
                else:
                    dists = pdist(X_cluster, metric='euclidean')
                mean_dist = np.mean(dists)
                total_dist += (n_cluster / n_total) * mean_dist

        return total_dist

    def _compute_between_cluster_distance(self, X: np.ndarray, labels: np.ndarray,
                                           centers: np.ndarray) -> float:
        """
        计算加权平均最近邻簇间距离
        B = Σ_c (n_c / N) * min_{c' ≠ c} dist(center_c, center_c')
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_total = len(X)

        if n_clusters < 2:
            return 0

        center_dists = squareform(pdist(centers, metric='euclidean'))
        np.fill_diagonal(center_dists, np.inf)

        total_between = 0
        for i, label in enumerate(unique_labels):
            mask = labels == label
            n_cluster = np.sum(mask)
            nearest_dist = np.min(center_dists[i])
            total_between += (n_cluster / n_total) * nearest_dist

        return total_between

    def _compute_clustering_stability_fixed(self, X: np.ndarray, k: int) -> Tuple[float, float]:
        """
        修复后的聚类稳定性计算
        使用 subsample without replacement（不是 bootstrap with replacement）
        """
        n = len(X)
        n_subsample = int(n * self.stability_subsample_ratio)

        all_labels = []
        all_indices = []

        for run in range(self.n_stability_runs):
            # 不放回采样
            np.random.seed(run)
            idx = np.random.choice(n, n_subsample, replace=False)
            X_sub = X[idx]

            kmeans = KMeans(n_clusters=k, random_state=run, n_init=5)
            labels = kmeans.fit_predict(X_sub)

            all_labels.append(labels)
            all_indices.append(idx)

        # 计算两两之间的 ARI/NMI（在共同样本上）
        aris, nmis = [], []

        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                idx1, labels1 = all_indices[i], all_labels[i]
                idx2, labels2 = all_indices[j], all_labels[j]

                # 找共同样本的原始索引
                common_original = np.intersect1d(idx1, idx2)

                if len(common_original) < max(k * 2, 20):
                    continue

                # 正确对齐：为每个共同样本找到它在两个子集中的位置
                # idx1_to_pos: original_idx -> position in labels1
                idx1_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(idx1)}
                idx2_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(idx2)}

                l1 = np.array([labels1[idx1_to_pos[orig]] for orig in common_original])
                l2 = np.array([labels2[idx2_to_pos[orig]] for orig in common_original])

                aris.append(adjusted_rand_score(l1, l2))
                nmis.append(normalized_mutual_info_score(l1, l2))

        return np.mean(aris) if aris else 0, np.mean(nmis) if nmis else 0

    # ==================== Part 3: 同质性 + Confusion Score ====================

    def analyze_cluster_quality_at_optimal_k(self):
        """在最优 K 处详细分析聚类质量"""
        print("\n" + "=" * 80)
        print("Part 3: Cluster Quality at Optimal K")
        print("(Including homogeneity based on external celltype labels)")
        print("=" * 80)

        summary_data = []

        for rep_name, k in self.optimal_k.items():
            result = self.clustering_results[rep_name][k]

            summary_data.append({
                'representation': rep_name,
                'optimal_k': k,
                'silhouette_train': result.silhouette_train,
                'silhouette_test': result.silhouette_test,
                'within_dist': result.within_cluster_dist,
                'between_dist': result.between_cluster_dist,
                'confusion_score': result.confusion_score,
                'stability_ari': result.stability_ari,
                'stability_nmi': result.stability_nmi,
                'homogeneity': result.homogeneity,
                'completeness': result.completeness,
                'v_measure': result.v_measure,
                'size_imbalance': result.size_imbalance,
            })

            print(f"\n  {rep_name} (K={k}):")
            print(f"    Silhouette (train/test): {result.silhouette_train:.4f} / {result.silhouette_test:.4f}")
            print(f"    Confusion score (W/B): {result.confusion_score:.4f}")
            print(f"    Stability (ARI/NMI): {result.stability_ari:.4f} / {result.stability_nmi:.4f}")
            print(f"    Homogeneity: {result.homogeneity:.4f}")
            print(f"    V-measure: {result.v_measure:.4f}")

        self.quality_summary = pd.DataFrame(summary_data)
        return self.quality_summary

    def compare_at_fixed_k(self, k_values: List[int] = [5, 10, 15]):
        """在固定 K 值下比较不同表示"""
        print("\n" + "=" * 80)
        print("Fixed-K Comparison")
        print("=" * 80)

        all_comparisons = []

        for k in k_values:
            print(f"\n--- K = {k} ---")
            comparison_data = []

            for rep_name in self.clustering_results:
                if k in self.clustering_results[rep_name]:
                    result = self.clustering_results[rep_name][k]
                    comparison_data.append({
                        'representation': rep_name,
                        'k': k,
                        'silhouette_test': result.silhouette_test,
                        'confusion_score': result.confusion_score,
                        'stability_ari': result.stability_ari,
                        'homogeneity': result.homogeneity,
                    })

            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            all_comparisons.append(df)

        self.fixed_k_comparisons = pd.concat(all_comparisons, ignore_index=True)
        return self.fixed_k_comparisons

    # ==================== 可视化 ====================

    def plot_all_panels(self):
        print("\n" + "=" * 80)
        print("Generating Visualization Panels")
        print("=" * 80)

        self.plot_panel_cross_modal_heatmap()
        self.plot_panel_retrieval_performance()
        self.plot_panel_heterogeneity()
        self.plot_panel_silhouette_curves()
        self.plot_panel_stability_curves()
        self.plot_panel_confusion_score_curves()
        self.plot_panel_homogeneity_curves()
        self.plot_panel_optimal_k_comparison()
        self.plot_panel_quality_summary()
        self.plot_panel_fixed_k_comparison()

        print(f"\n✓ All panels saved to: {self.output_dir}")

    def plot_panel_cross_modal_heatmap(self):
        """Panel 1a: 跨模态相关性热力图"""
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 0.9, 2))

        metrics = ['cca_mean_corr', 'rv_coefficient', 'rsa_correlation']
        titles = ['CCA Correlation', 'RV Coefficient', 'RSA (Distance Corr.)']

        modalities = ['Morph', 'Gene', 'Proj']

        for ax, metric, title in zip(axes, metrics, titles):
            matrix = np.eye(3)
            for i, m1 in enumerate(modalities):
                for j, m2 in enumerate(modalities):
                    if i < j:
                        key = f'{m1}-{m2}'
                        if key in self.cross_modal_metrics:
                            val = getattr(self.cross_modal_metrics[key], metric)
                            matrix[i, j] = val
                            matrix[j, i] = val

            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=modalities, yticklabels=modalities,
                       ax=ax, vmin=0, vmax=1, cbar_kws={'shrink': 0.8},
                       annot_kws={'size': 7})
            ax.set_title(title, fontsize=8, fontweight='bold')

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_1a_cross_modal_correlation')

    def plot_panel_retrieval_performance(self):
        """Panel 1b: Cross-modal retrieval 性能"""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.9, 2.2))

        pairs = ['Morph-Gene', 'Gene-Proj', 'Morph-Proj']
        colors = [NM_COLORS['purple'], NM_COLORS['cyan'], NM_COLORS['orange']]

        x = np.arange(3)
        width = 0.25

        for i, (pair, color) in enumerate(zip(pairs, colors)):
            if pair in self.cross_modal_metrics:
                m = self.cross_modal_metrics[pair]
                recalls = [m.recall_at_1, m.recall_at_5, m.recall_at_10]
                ax.bar(x + i * width, recalls, width, label=pair.replace('-', '↔'),
                      color=color, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Recall')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['R@1', 'R@5', 'R@10'])
        ax.legend(loc='upper left', fontsize=5)
        ax.set_title('Cross-Modal Retrieval', fontweight='bold', fontsize=8)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_1b_retrieval_performance')

    def plot_panel_heterogeneity(self):
        """Panel 1c: 模态异质性"""
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 0.7, 2))

        df = self.heterogeneity_results
        modalities = df['modality'].tolist()
        colors = [MODALITY_COLORS[m] for m in modalities]

        axes[0].bar(modalities, df['effective_dim'], color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Effective Dimension')
        axes[0].set_title('Intrinsic Dimensionality', fontsize=8, fontweight='bold')

        axes[1].bar(modalities, df['density_cv'], color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Density CV')
        axes[1].set_title('Local Density Variation', fontsize=8, fontweight='bold')

        axes[2].bar(modalities, df['best_silhouette'], color=colors, edgecolor='black', linewidth=0.5)
        for i, (m, k) in enumerate(zip(modalities, df['best_k'])):
            axes[2].annotate(f'K={k}', xy=(i, df['best_silhouette'].iloc[i]),
                           xytext=(0, 2), textcoords='offset points', ha='center', fontsize=5)
        axes[2].set_ylabel('Best Silhouette')
        axes[2].set_title('Clustering Potential', fontsize=8, fontweight='bold')

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_1c_heterogeneity')

    def plot_panel_silhouette_curves(self):
        """Panel 2a: Silhouette vs K 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.8, 2.5))

        rep_order = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        for ax, metric, title in zip(axes, ['silhouette_train', 'silhouette_test'],
                                     ['Train Silhouette', 'Test Silhouette']):
            for rep_name in rep_order:
                if rep_name not in self.clustering_results:
                    continue

                ks = sorted(self.clustering_results[rep_name].keys())
                vals = [getattr(self.clustering_results[rep_name][k], metric) for k in ks]

                color = MODALITY_COLORS.get(rep_name, NM_COLORS['grey'])
                linestyle = '-' if '+' not in rep_name else '--'
                linewidth = 1.2 if '+' in rep_name else 0.8

                ax.plot(ks, vals, color=color, linestyle=linestyle, label=rep_name, linewidth=linewidth)

                # 标记最优 K
                if metric == 'silhouette_test':
                    opt_k = self.optimal_k.get(rep_name)
                    if opt_k:
                        opt_val = getattr(self.clustering_results[rep_name][opt_k], metric)
                        ax.scatter([opt_k], [opt_val], color=color, s=30, zorder=5, marker='o')

            ax.set_xlabel('K')
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold', fontsize=8)
            ax.grid(alpha=0.3, linewidth=0.5)

        axes[1].legend(loc='upper right', fontsize=5, ncol=2)
        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_2a_silhouette_curves')

    def plot_panel_stability_curves(self):
        """Panel 2b: Stability vs K 曲线"""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, 2.5))

        rep_order = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        for rep_name in rep_order:
            if rep_name not in self.clustering_results:
                continue

            ks = sorted(self.clustering_results[rep_name].keys())
            stabs = [self.clustering_results[rep_name][k].stability_ari for k in ks]

            color = MODALITY_COLORS.get(rep_name, NM_COLORS['grey'])
            linestyle = '-' if '+' not in rep_name else '--'

            ax.plot(ks, stabs, color=color, linestyle=linestyle, label=rep_name,
                   linewidth=1.2 if '+' in rep_name else 0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Stability (ARI)')
        ax.set_title('Clustering Stability vs K', fontweight='bold', fontsize=8)
        ax.legend(loc='upper right', fontsize=5, ncol=2)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_2b_stability_curves')

    def plot_panel_confusion_score_curves(self):
        """Panel 3a: Confusion Score vs K"""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, 2.5))

        rep_order = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        for rep_name in rep_order:
            if rep_name not in self.clustering_results:
                continue

            ks = sorted(self.clustering_results[rep_name].keys())
            confs = [self.clustering_results[rep_name][k].confusion_score for k in ks]

            color = MODALITY_COLORS.get(rep_name, NM_COLORS['grey'])
            linestyle = '-' if '+' not in rep_name else '--'

            ax.plot(ks, confs, color=color, linestyle=linestyle, label=rep_name,
                   linewidth=1.2 if '+' in rep_name else 0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Confusion Score (W/B)')
        ax.set_title('Cluster Separability vs K\n(lower is better)', fontweight='bold', fontsize=8)
        ax.legend(loc='upper right', fontsize=5, ncol=2)
        ax.grid(alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_3a_confusion_score')

    def plot_panel_homogeneity_curves(self):
        """Panel 3b: Homogeneity vs K（基于外部标签 celltype）"""
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, 2.5))

        rep_order = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']

        for rep_name in rep_order:
            if rep_name not in self.clustering_results:
                continue

            ks = sorted(self.clustering_results[rep_name].keys())
            homos = [self.clustering_results[rep_name][k].homogeneity for k in ks]

            color = MODALITY_COLORS.get(rep_name, NM_COLORS['grey'])
            linestyle = '-' if '+' not in rep_name else '--'

            ax.plot(ks, homos, color=color, linestyle=linestyle, label=rep_name,
                   linewidth=1.2 if '+' in rep_name else 0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Homogeneity (vs celltype)')
        ax.set_title('Cluster Homogeneity vs K\n(based on external celltype labels)', fontweight='bold', fontsize=8)
        ax.legend(loc='lower right', fontsize=5, ncol=2)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_3b_homogeneity')

    def plot_panel_optimal_k_comparison(self):
        """Panel 2c: 最优 K 值对比"""
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.7, 2.2))

        df = self.quality_summary.sort_values('silhouette_test', ascending=False)
        reps = df['representation'].tolist()
        colors = [MODALITY_COLORS.get(r, NM_COLORS['grey']) for r in reps]

        x = np.arange(len(reps))

        axes[0].bar(x, df['optimal_k'], color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(reps, rotation=45, ha='right', fontsize=5)
        axes[0].set_ylabel('Optimal K')
        axes[0].set_title('Supported Cluster Number', fontsize=8, fontweight='bold')

        axes[1].bar(x, df['silhouette_test'], color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(reps, rotation=45, ha='right', fontsize=5)
        axes[1].set_ylabel('Test Silhouette')
        axes[1].set_title('Quality at Optimal K', fontsize=8, fontweight='bold')

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_2c_optimal_k_comparison')

    def plot_panel_quality_summary(self):
        """Panel 3c: 聚类质量综合对比"""
        fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, 2.2))

        df = self.quality_summary

        single = df[~df['representation'].str.contains(r'\+')]
        multi = df[df['representation'].str.contains(r'\+')]

        metrics = [
            ('silhouette_test', 'Test Silhouette', False),
            ('confusion_score', 'Confusion (W/B)', True),
            ('stability_ari', 'Stability (ARI)', False),
            ('homogeneity', 'Homogeneity', False),
        ]

        for ax, (metric, title, invert) in zip(axes, metrics):
            # 单模态
            x_single = np.arange(len(single))
            colors_s = [MODALITY_COLORS.get(r, NM_COLORS['grey']) for r in single['representation']]
            bars1 = ax.bar(x_single, single[metric], 0.6, color=colors_s,
                          edgecolor='black', linewidth=0.5)

            # 多模态
            x_multi = np.arange(len(multi)) + len(single) + 0.5
            colors_m = [MODALITY_COLORS.get(r, NM_COLORS['grey']) for r in multi['representation']]
            bars2 = ax.bar(x_multi, multi[metric], 0.6, color=colors_m,
                          edgecolor='black', linewidth=0.5, hatch='//')

            all_labels = list(single['representation']) + [''] + list(multi['representation'])
            all_x = list(x_single) + [len(single)] + list(x_multi)
            ax.set_xticks(all_x)
            ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=4)

            ax.set_title(title, fontsize=7, fontweight='bold')

            if invert:
                ax.invert_yaxis()

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_3c_quality_summary')

    def plot_panel_fixed_k_comparison(self):
        """Panel 3d: 固定 K 值对比"""
        if not hasattr(self, 'fixed_k_comparisons') or self.fixed_k_comparisons is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 0.9, 2.5))

        for ax, k in zip(axes, [5, 10, 15]):
            df_k = self.fixed_k_comparisons[self.fixed_k_comparisons['k'] == k]
            if len(df_k) == 0:
                continue

            x = np.arange(len(df_k))
            colors = [MODALITY_COLORS.get(r, NM_COLORS['grey']) for r in df_k['representation']]

            ax.bar(x, df_k['silhouette_test'], color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(df_k['representation'], rotation=45, ha='right', fontsize=5)
            ax.set_ylabel('Test Silhouette')
            ax.set_title(f'K = {k}', fontsize=8, fontweight='bold')
            ax.set_ylim(0, df_k['silhouette_test'].max() * 1.2)

        plt.tight_layout()
        save_panel(fig, self.output_dir, 'panel_3d_fixed_k_comparison')

    # ==================== 保存结果 ====================

    def save_all_results(self):
        print("\n" + "=" * 80)
        print("Saving Results")
        print("=" * 80)

        # Cross-modal metrics
        cross_modal_data = []
        for key, metrics in self.cross_modal_metrics.items():
            cross_modal_data.append({
                'pair': key,
                'cca_mean_corr': metrics.cca_mean_corr,
                'rv_coefficient': metrics.rv_coefficient,
                'rsa_correlation': metrics.rsa_correlation,
                'rsa_pvalue': metrics.rsa_pvalue,
                'recall_at_1': metrics.recall_at_1,
                'recall_at_5': metrics.recall_at_5,
                'recall_at_10': metrics.recall_at_10,
                'mrr': metrics.mrr,
            })
        pd.DataFrame(cross_modal_data).to_csv(f"{self.output_dir}/cross_modal_metrics.csv", index=False)

        # Heterogeneity
        if self.heterogeneity_results is not None:
            self.heterogeneity_results.to_csv(f"{self.output_dir}/heterogeneity_results.csv", index=False)

        # Clustering results
        clustering_data = []
        for rep_name, k_results in self.clustering_results.items():
            for k, result in k_results.items():
                clustering_data.append({
                    'representation': rep_name,
                    'k': k,
                    'silhouette_train': result.silhouette_train,
                    'silhouette_test': result.silhouette_test,
                    'within_dist': result.within_cluster_dist,
                    'between_dist': result.between_cluster_dist,
                    'confusion_score': result.confusion_score,
                    'stability_ari': result.stability_ari,
                    'stability_nmi': result.stability_nmi,
                    'homogeneity': result.homogeneity,
                    'completeness': result.completeness,
                    'v_measure': result.v_measure,
                    'size_imbalance': result.size_imbalance,
                })
        pd.DataFrame(clustering_data).to_csv(f"{self.output_dir}/clustering_k_scan.csv", index=False)

        # Quality summary
        if self.quality_summary is not None:
            self.quality_summary.to_csv(f"{self.output_dir}/quality_summary.csv", index=False)

        # Fixed-K comparisons
        if hasattr(self, 'fixed_k_comparisons') and self.fixed_k_comparisons is not None:
            self.fixed_k_comparisons.to_csv(f"{self.output_dir}/fixed_k_comparisons.csv", index=False)

        # Optimal K
        pd.DataFrame([
            {'representation': k, 'optimal_k': v} for k, v in self.optimal_k.items()
        ]).to_csv(f"{self.output_dir}/optimal_k.csv", index=False)

        print(f"  ✓ Results saved to: {self.output_dir}")

    # ==================== 主流程 ====================

    def run_full_analysis(self):
        print("\n" + "=" * 80)
        print("MODALITY COMPLEMENTARITY & CLUSTERING QUALITY ANALYSIS")
        print("(Fixed Version)")
        print("=" * 80)

        self.load_all_data()

        # Part 1
        self.analyze_cross_modal_correlation()
        self.analyze_modality_heterogeneity()

        # Part 2
        self.analyze_clustering_k_scan()

        # Part 3
        self.analyze_cluster_quality_at_optimal_k()
        self.compare_at_fixed_k([5, 10, 15])

        # 可视化
        self.plot_all_panels()

        # 保存
        self.save_all_results()

        # 总结
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        print("\n【Part 1: Cross-Modal Correlation】")
        print("  (High RSA = shared structure, Low Retrieval R@1 = modality-specific info)")
        for key, m in self.cross_modal_metrics.items():
            print(f"  {key}: CCA={m.cca_mean_corr:.3f}, RV={m.rv_coefficient:.3f}, "
                  f"RSA={m.rsa_correlation:.3f}, R@1={m.recall_at_1:.3f}")

        print("\n【Part 2: Optimal K (selected by combined score)】")
        for rep, k in sorted(self.optimal_k.items(), key=lambda x: -self.clustering_results[x[0]][x[1]].silhouette_test):
            result = self.clustering_results[rep][k]
            print(f"  {rep}: K={k}, sil_test={result.silhouette_test:.4f}, "
                  f"stability={result.stability_ari:.4f}")

        print("\n【Part 3: Quality Metrics at Optimal K】")
        df = self.quality_summary.sort_values('silhouette_test', ascending=False)
        print(df[['representation', 'optimal_k', 'silhouette_test',
                  'confusion_score', 'stability_ari', 'homogeneity']].to_string(index=False))

        # 关键发现
        print("\n【KEY FINDINGS】")
        single_mask = ~df['representation'].str.contains(r'\+')
        multi_mask = df['representation'].str.contains(r'\+')

        best_single_sil = df[single_mask]['silhouette_test'].max()
        best_multi_sil = df[multi_mask]['silhouette_test'].max()
        best_single_homo = df[single_mask]['homogeneity'].max()
        best_multi_homo = df[multi_mask]['homogeneity'].max()

        print(f"  Silhouette: single best={best_single_sil:.4f}, multi best={best_multi_sil:.4f} "
              f"(Δ={best_multi_sil - best_single_sil:+.4f})")
        print(f"  Homogeneity: single best={best_single_homo:.4f}, multi best={best_multi_homo:.4f} "
              f"(Δ={best_multi_homo - best_single_homo:+.4f})")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./modality_analysis"

    with ModalityComplementarityAnalysis(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            test_ratio=0.2,
            cca_n_components=15,
            k_range=(2, 70),
            n_stability_runs=20,
            stability_subsample_ratio=0.8,  # 80% 不放回采样
            k_selection_method='combined',  # 组合指标选 K
            output_dir=OUTPUT_DIR,
    ) as analyzer:
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main()