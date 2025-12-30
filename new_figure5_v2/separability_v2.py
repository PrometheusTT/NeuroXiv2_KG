"""
Task 2: Region Confusion Score Analysis (V3 - Fisher-style Local Separability)
================================================================================
核心改进：
1. 新的Confusion Score：局部噪声归一化分离度（Fisher-style）
   - s_r^2 = E[||x - μ_r||^2 / d] （区域内噪声，维度归一化）
   - b_ij = ||μ_i - μ_j||^2 / d （中心间距离，维度归一化）
   - C_ij = (s_i^2 + s_j^2) / (b_ij + ε) （pair-level混淆度）

2. 只在"容易混淆的局部近邻"上汇总
   - 用Morph-only在train上选择k个最近邻脑区
   - 在test上计算confusion，避免被远距离pair稀释

3. 可视化改进：只展示单模态下最难分开的10-20个脑区

Author: Claude (V3 - Fisher-style)
Date: 2025-01
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import multiprocessing

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import zscore, sem
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import neo4j

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 1200

N_CORES = multiprocessing.cpu_count()


@dataclass
class ModalityData:
    """存储单个模态的处理后数据"""
    name: str
    vectors_train: np.ndarray
    vectors_test: np.ndarray
    original_dim: int
    reduced_dim: int
    variance_explained: float


@dataclass
class FisherConfusionResult:
    """存储Fisher-style Confusion Score分析结果"""
    modality: str
    # 全局confusion score (基于k近邻)
    global_confusion: float
    global_confusion_sem: float
    # 每个脑区的局部confusion
    region_confusion: Dict[str, float]  # C_i(k)
    # 每个脑区的内部噪声 s_r^2
    region_noise: Dict[str, float]
    # 脑区中心
    region_centroids: Dict[str, np.ndarray]
    # 中心间距离矩阵 b_ij (维度归一化)
    between_distance_matrix: np.ndarray
    # pair-level confusion矩阵 C_ij
    pairwise_confusion_matrix: np.ndarray
    # 近邻集合 (基于参考模态)
    neighbor_sets: Dict[str, List[str]]  # region -> list of k nearest neighbors
    # 元信息
    region_labels: List[str]
    n_neurons_per_region: Dict[str, int]
    dimension: int
    k_neighbors: int


class RegionConfusionAnalysisV3:
    """
    区域混淆度分析 V3 (Fisher-style Local Separability)

    核心公式：
    - s_r^2 = E[||x - μ_r||^2 / d]  (区域内噪声，维度归一化)
    - b_ij = ||μ_i - μ_j||^2 / d    (中心间距离，维度归一化)
    - C_ij = (s_i^2 + s_j^2) / (b_ij + ε)  (pair混淆度)
    - C_i(k) = s_i^2 / mean_{j∈N_k(i)}(b_ij)  (局部混淆度)
    - Confusion(k) = mean_i C_i(k)  (全局混淆度)
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
                 min_neurons_per_region: int = 10,
                 test_ratio: float = 0.2,
                 equal_dim_for_fusion: int = 20,
                 k_neighbors: int = 5,  # 近邻数量
                 reference_modality: str = 'Morph',  # 用于选择近邻的参考模态
                 epsilon: float = 1e-8,  # 防止除零
                 random_seed: int = 42):

        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.min_neurons_per_region = min_neurons_per_region
        self.test_ratio = test_ratio
        self.equal_dim_for_fusion = equal_dim_for_fusion
        self.k_neighbors = k_neighbors
        self.reference_modality = reference_modality
        self.epsilon = epsilon
        self.random_seed = random_seed

        print(f"Configuration (V3 - Fisher-style Local Separability):")
        print(f"  PCA variance threshold: {self.pca_variance_threshold}")
        print(f"  Min neurons per region: {self.min_neurons_per_region}")
        print(f"  Test ratio: {self.test_ratio}")
        print(f"  Equal dim for fusion: {self.equal_dim_for_fusion}")
        print(f"  K neighbors: {self.k_neighbors}")
        print(f"  Reference modality: {self.reference_modality}")
        print(f"  Random seed: {self.random_seed}")

        # Data storage
        self.valid_neuron_ids: List[str] = []
        self.neuron_regions: Dict[str, str] = {}
        self.region_neurons: Dict[str, List[str]] = {}

        # Train/Test split
        self.train_neuron_ids: List[str] = []
        self.test_neuron_ids: List[str] = []

        # Raw features
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}

        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []
        self.valid_regions: List[str] = []

        # Processed vectors
        self.modality_data: Dict[str, ModalityData] = {}

        # 对等降维后的向量
        self.equal_dim_vectors_train: Dict[str, np.ndarray] = {}
        self.equal_dim_vectors_test: Dict[str, np.ndarray] = {}

        # 近邻集合 (基于参考模态在train上确定)
        self.neighbor_sets: Dict[str, List[str]] = {}

        # Results
        self.confusion_results: Dict[str, FisherConfusionResult] = {}

        # 最难分开的脑区对 (用于可视化)
        self.hardest_pairs: List[Tuple[str, str, float]] = []
        self.hardest_regions: List[str] = []

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Data Loading ====================

    def load_all_data(self) -> int:
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._load_neuron_regions()
        self._filter_valid_neurons()
        self._filter_valid_regions()
        self._stratified_train_test_split()
        self._process_all_vectors()
        self._determine_neighbor_sets()

        print(f"\n✓ Data loading complete:")
        print(f"  Total neurons: {len(self.valid_neuron_ids)}")
        print(f"  Train neurons: {len(self.train_neuron_ids)}")
        print(f"  Test neurons: {len(self.test_neuron_ids)}")
        print(f"  Valid regions: {len(self.valid_regions)}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

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
        print(f"  Projection targets: {len(self.all_target_regions)} brain regions")

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
            records = list(result)

            for record in records:
                neuron_id = record['neuron_id']
                axon_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.AXONAL_FEATURES]
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                dend_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.DENDRITIC_FEATURES]
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                proj_result = session.run(proj_query, neuron_id=neuron_id)
                proj_dict = {r['target']: r['weight'] for r in proj_result
                             if r['target'] and r['weight']}
                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]
                    self.projection_vectors_raw[neuron_id] = proj_vector

        print(f"  Loaded {len(self.axon_features_raw)} neurons with morphology")

    def _load_neuron_regions(self):
        query = """
        MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
        RETURN n.neuron_id AS neuron_id, r.acronym AS region
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                neuron_id = record['neuron_id']
                region = record['region']
                if neuron_id and region:
                    self.neuron_regions[neuron_id] = region

        print(f"  Loaded region info for {len(self.neuron_regions)} neurons")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        candidates &= set(self.neuron_regions.keys())
        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  Valid neurons (with all modalities + region): {len(self.valid_neuron_ids)}")

    def _filter_valid_regions(self):
        region_counts = {}
        for nid in self.valid_neuron_ids:
            region = self.neuron_regions[nid]
            if region not in region_counts:
                region_counts[region] = []
            region_counts[region].append(nid)

        self.region_neurons = {}
        for region, neurons in region_counts.items():
            if len(neurons) >= self.min_neurons_per_region:
                self.region_neurons[region] = neurons

        self.valid_regions = sorted(self.region_neurons.keys())

        valid_neuron_set = set()
        for neurons in self.region_neurons.values():
            valid_neuron_set.update(neurons)
        self.valid_neuron_ids = sorted(list(valid_neuron_set))

        print(f"  Valid regions (>={self.min_neurons_per_region} neurons): {len(self.valid_regions)}")
        for region in self.valid_regions:
            print(f"    {region}: {len(self.region_neurons[region])} neurons")

    def _stratified_train_test_split(self):
        np.random.seed(self.random_seed)

        self.train_neuron_ids = []
        self.test_neuron_ids = []

        for region, neurons in self.region_neurons.items():
            neurons = np.array(neurons)
            np.random.shuffle(neurons)
            n_test = max(1, int(len(neurons) * self.test_ratio))
            self.test_neuron_ids.extend(neurons[:n_test].tolist())
            self.train_neuron_ids.extend(neurons[n_test:].tolist())

        print(f"  Stratified split: {len(self.train_neuron_ids)} train, {len(self.test_neuron_ids)} test")

    def _process_all_vectors(self):
        print("\nProcessing vectors (fit on TRAIN, transform on TEST)...")

        # Morphology
        morph_train, morph_test, morph_info = self._process_morph_vector()
        self.modality_data['Morph'] = ModalityData(
            name='Morphology',
            vectors_train=morph_train,
            vectors_test=morph_test,
            original_dim=morph_info['original_dim'],
            reduced_dim=morph_train.shape[1],
            variance_explained=morph_info['variance']
        )

        # Gene/Molecular
        gene_train, gene_test, gene_info = self._process_gene_vector()
        self.modality_data['Gene'] = ModalityData(
            name='Molecular',
            vectors_train=gene_train,
            vectors_test=gene_test,
            original_dim=gene_info['original_dim'],
            reduced_dim=gene_train.shape[1],
            variance_explained=gene_info['variance']
        )

        # Projection
        proj_train, proj_test, proj_info = self._process_proj_vector()
        self.modality_data['Proj'] = ModalityData(
            name='Projection',
            vectors_train=proj_train,
            vectors_test=proj_test,
            original_dim=proj_info['original_dim'],
            reduced_dim=proj_train.shape[1],
            variance_explained=proj_info['variance']
        )

        # 对等降维
        self._create_equal_dim_vectors()

        # 多模态融合
        self._create_multimodal_vectors()

    def _process_morph_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        train_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in self.train_neuron_ids
        ])
        test_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in self.test_neuron_ids
        ])

        original_dim = train_raw.shape[1]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Morph: {original_dim}D → {train_pca.shape[1]}D ({variance:.1%} variance)")

        return train_pca, test_pca, {'variance': variance, 'original_dim': original_dim}

    def _process_gene_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        train_raw = np.array([self.local_gene_features_raw[nid] for nid in self.train_neuron_ids])
        test_raw = np.array([self.local_gene_features_raw[nid] for nid in self.test_neuron_ids])

        original_dim = train_raw.shape[1]

        col_sums = train_raw.sum(axis=0)
        valid_cols = col_sums > 0
        train_pruned = train_raw[:, valid_cols]
        test_pruned = test_raw[:, valid_cols]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_pruned)
        test_scaled = scaler.transform(test_pruned)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Gene: {original_dim}D → {train_pruned.shape[1]}D (pruned) → {train_pca.shape[1]}D ({variance:.1%})")

        return train_pca, test_pca, {'variance': variance, 'original_dim': original_dim}

    def _process_proj_vector(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        train_raw = np.array([self.projection_vectors_raw[nid] for nid in self.train_neuron_ids])
        test_raw = np.array([self.projection_vectors_raw[nid] for nid in self.test_neuron_ids])

        original_dim = train_raw.shape[1]

        col_sums = train_raw.sum(axis=0)
        valid_cols = col_sums > 0
        train_pruned = train_raw[:, valid_cols]
        test_pruned = test_raw[:, valid_cols]

        train_log = np.log1p(train_pruned)
        test_log = np.log1p(test_pruned)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_log)
        test_scaled = scaler.transform(test_log)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  Proj: {original_dim}D → {train_pruned.shape[1]}D → log1p → {train_pca.shape[1]}D ({variance:.1%})")

        return train_pca, test_pca, {'variance': variance, 'original_dim': original_dim}

    def _create_equal_dim_vectors(self):
        target_dim = self.equal_dim_for_fusion
        print(f"\nCreating equal dimension vectors ({target_dim}D each for fusion)...")

        for modality in ['Morph', 'Gene', 'Proj']:
            data = self.modality_data[modality]
            train_vec = data.vectors_train
            test_vec = data.vectors_test

            if train_vec.shape[1] > target_dim:
                pca = PCA(n_components=target_dim)
                train_reduced = pca.fit_transform(train_vec)
                test_reduced = pca.transform(test_vec)
                print(f"  {modality}: {train_vec.shape[1]}D → {target_dim}D")
            else:
                train_reduced = train_vec
                test_reduced = test_vec
                print(f"  {modality}: {train_vec.shape[1]}D (kept as is)")

            self.equal_dim_vectors_train[modality] = train_reduced
            self.equal_dim_vectors_test[modality] = test_reduced

    def _create_multimodal_vectors(self):
        print("\nCreating multimodal fusion vectors...")

        combinations = [
            ('Morph+Gene', ['Morph', 'Gene']),
            ('Gene+Proj', ['Gene', 'Proj']),
            ('Morph+Proj', ['Morph', 'Proj']),
            ('All', ['Morph', 'Gene', 'Proj']),
        ]

        for name, modalities in combinations:
            train_vectors = [self.equal_dim_vectors_train[m] for m in modalities]
            test_vectors = [self.equal_dim_vectors_test[m] for m in modalities]

            train_concat = np.hstack(train_vectors)
            test_concat = np.hstack(test_vectors)

            self.modality_data[name] = ModalityData(
                name=name,
                vectors_train=train_concat,
                vectors_test=test_concat,
                original_dim=sum(v.shape[1] for v in train_vectors),
                reduced_dim=train_concat.shape[1],
                variance_explained=0.0
            )
            print(f"  {name}: {train_concat.shape[1]}D")

    def _determine_neighbor_sets(self):
        """
        在训练集上用参考模态(Morph)确定每个脑区的k个最近邻
        这个近邻集合会被所有模态共享使用
        """
        print(f"\nDetermining k={self.k_neighbors} nearest neighbors using {self.reference_modality} on TRAIN...")

        ref_data = self.modality_data[self.reference_modality]
        vectors = ref_data.vectors_train

        # 建立训练集神经元索引映射
        train_neuron_to_idx = {nid: i for i, nid in enumerate(self.train_neuron_ids)}

        # 计算每个脑区的中心（在训练集上）
        region_centroids = {}
        for region in self.valid_regions:
            neurons = [nid for nid in self.region_neurons[region] if nid in train_neuron_to_idx]
            if neurons:
                idx = [train_neuron_to_idx[nid] for nid in neurons]
                region_centroids[region] = np.mean(vectors[idx], axis=0)

        # 计算脑区中心间的距离矩阵
        n_regions = len(self.valid_regions)
        centroid_matrix = np.array([region_centroids[r] for r in self.valid_regions])
        dist_matrix = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

        # 对每个脑区找k个最近邻（不包括自己）
        for i, region in enumerate(self.valid_regions):
            # 获取到其他脑区的距离，排除自己
            distances = dist_matrix[i].copy()
            distances[i] = np.inf  # 排除自己

            # 找k个最近的
            k = min(self.k_neighbors, n_regions - 1)
            nearest_idx = np.argsort(distances)[:k]
            self.neighbor_sets[region] = [self.valid_regions[j] for j in nearest_idx]

        print(f"  Neighbor sets determined for {len(self.neighbor_sets)} regions")
        # 打印几个例子
        for i, region in enumerate(self.valid_regions[:3]):
            print(f"    {region}: {self.neighbor_sets[region]}")
        if len(self.valid_regions) > 3:
            print(f"    ...")

    # ==================== Fisher-style Confusion Score Calculation ====================

    def compute_confusion_scores(self):
        """计算所有模态的Fisher-style confusion score（在测试集上评估）"""
        print("\n" + "=" * 80)
        print("Computing Fisher-style Confusion Scores (on TEST set)")
        print("=" * 80)
        print("\nFormula:")
        print("  s_r² = E[||x - μ_r||² / d]  (within-region noise, dim-normalized)")
        print("  b_ij = ||μ_i - μ_j||² / d   (between-region distance, dim-normalized)")
        print("  C_ij = (s_i² + s_j²) / (b_ij + ε)  (pair-level confusion)")
        print("  C_i(k) = s_i² / mean_{j∈N_k(i)}(b_ij)  (local confusion)")
        print("  Confusion(k) = mean_i C_i(k)  (global confusion)")
        print("\nLower is better (small within-noise, large between-distance)")

        for modality_key, modality_data in self.modality_data.items():
            result = self._compute_fisher_confusion(modality_key, modality_data)
            self.confusion_results[modality_key] = result

        # 确定最难分开的脑区（基于参考模态）
        self._identify_hardest_regions()

        # 打印结果摘要
        print("\n" + "-" * 70)
        print("Confusion Score Summary (lower is better):")
        print("-" * 70)
        print(f"{'Modality':<15} {'Confusion':>12} {'±SEM':>10} {'Dim':>6}")
        print("-" * 70)
        for key, result in sorted(self.confusion_results.items(), key=lambda x: x[1].global_confusion):
            print(f"{key:<15} {result.global_confusion:>12.4f} {result.global_confusion_sem:>10.4f} {result.dimension:>6}")

    def _compute_fisher_confusion(self, modality_key: str,
                                   modality_data: ModalityData) -> FisherConfusionResult:
        """计算单个模态的Fisher-style confusion score"""
        print(f"\n--- {modality_key} ---")

        vectors = modality_data.vectors_test
        d = vectors.shape[1]  # 维度
        n_regions = len(self.valid_regions)

        # 建立测试集神经元索引映射
        test_neuron_to_idx = {nid: i for i, nid in enumerate(self.test_neuron_ids)}

        # 1. 计算每个脑区的中心 μ_r 和噪声 s_r²
        region_centroids = {}
        region_noise = {}
        n_neurons_per_region = {}

        for region in self.valid_regions:
            neurons = [nid for nid in self.region_neurons[region] if nid in test_neuron_to_idx]
            n_neurons_per_region[region] = len(neurons)

            if len(neurons) >= 2:
                idx = [test_neuron_to_idx[nid] for nid in neurons]
                region_vectors = vectors[idx]

                # 中心
                centroid = np.mean(region_vectors, axis=0)
                region_centroids[region] = centroid

                # 噪声 s_r² = E[||x - μ_r||² / d]
                squared_distances = np.sum((region_vectors - centroid) ** 2, axis=1)
                s_r_squared = np.mean(squared_distances) / d
                region_noise[region] = s_r_squared
            else:
                # 样本太少，使用0或跳过
                region_centroids[region] = np.mean(vectors[idx], axis=0) if neurons else np.zeros(d)
                region_noise[region] = 0.0

        # 2. 计算中心间距离矩阵 b_ij = ||μ_i - μ_j||² / d
        between_distance_matrix = np.zeros((n_regions, n_regions))
        for i, region_i in enumerate(self.valid_regions):
            for j, region_j in enumerate(self.valid_regions):
                if i != j:
                    diff = region_centroids[region_i] - region_centroids[region_j]
                    b_ij = np.sum(diff ** 2) / d
                    between_distance_matrix[i, j] = b_ij

        # 3. 计算pair-level confusion C_ij = (s_i² + s_j²) / (b_ij + ε)
        pairwise_confusion_matrix = np.zeros((n_regions, n_regions))
        for i, region_i in enumerate(self.valid_regions):
            for j, region_j in enumerate(self.valid_regions):
                if i != j:
                    s_i = region_noise[region_i]
                    s_j = region_noise[region_j]
                    b_ij = between_distance_matrix[i, j]
                    C_ij = (s_i + s_j) / (b_ij + self.epsilon)
                    pairwise_confusion_matrix[i, j] = C_ij

        # 4. 计算local confusion C_i(k) = s_i² / mean_{j∈N_k(i)}(b_ij)
        region_confusion = {}
        region_to_idx = {r: i for i, r in enumerate(self.valid_regions)}

        for region in self.valid_regions:
            i = region_to_idx[region]
            s_i = region_noise[region]

            # 获取近邻集合
            neighbors = self.neighbor_sets.get(region, [])
            if neighbors:
                neighbor_distances = [between_distance_matrix[i, region_to_idx[n]] for n in neighbors]
                mean_b = np.mean(neighbor_distances)
                C_i = s_i / (mean_b + self.epsilon)
            else:
                C_i = 0.0

            region_confusion[region] = C_i

        # 5. 计算global confusion = mean_i C_i(k)
        confusion_values = list(region_confusion.values())
        global_confusion = np.mean(confusion_values)
        global_confusion_sem = sem(confusion_values) if len(confusion_values) > 1 else 0.0

        print(f"  Dimensions: {d}")
        print(f"  Mean within-region noise (s²): {np.mean(list(region_noise.values())):.4f}")
        print(f"  Mean between-region distance (b): {np.mean(between_distance_matrix[between_distance_matrix > 0]):.4f}")
        print(f"  Global Confusion (k={self.k_neighbors}): {global_confusion:.4f} ± {global_confusion_sem:.4f}")

        return FisherConfusionResult(
            modality=modality_key,
            global_confusion=global_confusion,
            global_confusion_sem=global_confusion_sem,
            region_confusion=region_confusion,
            region_noise=region_noise,
            region_centroids=region_centroids,
            between_distance_matrix=between_distance_matrix,
            pairwise_confusion_matrix=pairwise_confusion_matrix,
            neighbor_sets=self.neighbor_sets.copy(),
            region_labels=self.valid_regions,
            n_neurons_per_region=n_neurons_per_region,
            dimension=d,
            k_neighbors=self.k_neighbors
        )

    def _identify_hardest_regions(self):
        """识别在参考模态下最难分开的脑区（用于可视化）"""
        ref_result = self.confusion_results[self.reference_modality]

        # 方法1：基于region-level confusion排序
        sorted_regions = sorted(ref_result.region_confusion.items(),
                                key=lambda x: x[1], reverse=True)

        # 取confusion最高的前N个脑区
        n_hardest = min(15, len(sorted_regions))
        self.hardest_regions = [r[0] for r in sorted_regions[:n_hardest]]

        # 方法2：也收集pair-level最难分开的pairs
        n_regions = len(self.valid_regions)
        pairs = []
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                C_ij = ref_result.pairwise_confusion_matrix[i, j]
                if C_ij > 0:
                    pairs.append((self.valid_regions[i], self.valid_regions[j], C_ij))

        pairs.sort(key=lambda x: x[2], reverse=True)
        self.hardest_pairs = pairs[:20]

        print(f"\n  Hardest regions in {self.reference_modality} (highest confusion):")
        for region in self.hardest_regions[:5]:
            c = ref_result.region_confusion[region]
            print(f"    {region}: C_i = {c:.4f}")

    # ==================== Visualization ====================

    def visualize_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_confusion_score_comparison(output_dir)
        self._plot_region_confusion_comparison(output_dir)
        self._plot_hardest_regions_heatmaps(output_dir)
        self._plot_pairwise_confusion_improvement(output_dir)
        self._plot_noise_vs_separation(output_dir)
        self._plot_comprehensive_analysis(output_dir)

        print(f"\n✓ Figures saved to: {output_dir}")

    def _create_custom_colormap(self):
        """创建红色渐变colormap"""
        colors = ['white', '#fee0d2', '#fc9272', '#de2d26', '#a50f15']
        return LinearSegmentedColormap.from_list('confusion_cmap', colors, N=256)

    def _create_improvement_colormap(self):
        """创建改进度colormap（绿色=好，红色=差）"""
        colors = ['#c0392b', '#e74c3c', '#f5f5f5', '#27ae60', '#1e8449']
        return LinearSegmentedColormap.from_list('improvement_cmap', colors, N=256)

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=1200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_confusion_score_comparison(self, output_dir: str):
        """绘制confusion score对比条形图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：所有模态的对比
        ax1 = axes[0]

        modalities = list(self.confusion_results.keys())
        scores = [self.confusion_results[m].global_confusion for m in modalities]
        sems = [self.confusion_results[m].global_confusion_sem for m in modalities]

        sorted_indices = np.argsort(scores)
        modalities_sorted = [modalities[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]
        sems_sorted = [sems[i] for i in sorted_indices]

        colors = []
        for m in modalities_sorted:
            if '+' in m or m == 'All':
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')

        y_pos = np.arange(len(modalities_sorted))
        bars = ax1.barh(y_pos, scores_sorted, xerr=sems_sorted, color=colors,
                        edgecolor='black', linewidth=1, capsize=3)

        for i, (bar, score, sem_val) in enumerate(zip(bars, scores_sorted, sems_sorted)):
            ax1.text(score + max(scores_sorted) * 0.02, i, f'{score:.4f}',
                     va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(modalities_sorted, fontsize=11)
        ax1.set_xlabel('Confusion Score (lower is better)', fontsize=11)
        ax1.set_title(f'Fisher-style Confusion Score\n(k={self.k_neighbors} neighbors)', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, max(scores_sorted) * 1.2)
        ax1.grid(axis='x', alpha=0.3)

        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Single Modality'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Multi-Modal Fusion'),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # 右图：改进百分比
        ax2 = axes[1]

        single_modalities = ['Morph', 'Gene', 'Proj']
        single_scores = {m: self.confusion_results[m].global_confusion for m in single_modalities}
        best_single = min(single_scores.values())
        best_single_name = min(single_scores, key=single_scores.get)

        multi_modalities = ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']
        improvements = []
        for m in multi_modalities:
            if m in self.confusion_results:
                multi_score = self.confusion_results[m].global_confusion
                improvement = (best_single - multi_score) / best_single * 100
                improvements.append(improvement)
            else:
                improvements.append(0)

        colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(range(len(multi_modalities)), improvements, color=colors,
                       edgecolor='black', linewidth=1)

        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            va = 'bottom' if imp >= 0 else 'top'
            offset = 1 if imp >= 0 else -1
            ax2.annotate(f'{imp:+.1f}%', xy=(i, imp), xytext=(0, offset * 5),
                         textcoords='offset points', ha='center', va=va,
                         fontsize=11, fontweight='bold')

        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xticks(range(len(multi_modalities)))
        ax2.set_xticklabels(multi_modalities, rotation=15, ha='right', fontsize=11)
        ax2.set_ylabel(f'Improvement vs Best Single ({best_single_name})', fontsize=11)
        ax2.set_title(f'Multi-Modal Improvement\n(Baseline: {best_single_name}={best_single:.4f})',
                      fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_confusion_score_comparison.png")

    def _plot_region_confusion_comparison(self, output_dir: str):
        """绘制每个脑区的confusion在不同模态下的对比 - 每个子图单独保存"""
        # 按参考模态的confusion排序
        ref_result = self.confusion_results[self.reference_modality]
        sorted_regions = sorted(ref_result.region_confusion.items(),
                                key=lambda x: x[1], reverse=True)
        regions = [r[0] for r in sorted_regions]

        # 选择要显示的模态
        modalities_to_show = ['Morph', 'Gene', 'Proj', 'All']
        x = np.arange(len(regions))
        width = 0.2

        colors = {'Morph': '#3498db', 'Gene': '#2ecc71', 'Proj': '#9b59b6', 'All': '#e74c3c'}

        # 保存完整的对比图
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, modality in enumerate(modalities_to_show):
            result = self.confusion_results[modality]
            values = [result.region_confusion[r] for r in regions]
            ax.bar(x + i * width, values, width, label=modality, color=colors[modality],
                   edgecolor='black', linewidth=0.5, alpha=0.8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Local Confusion C_i(k)', fontsize=11)
        ax.set_xlabel('Brain Regions (sorted by Morph confusion)', fontsize=11)
        ax.set_title(f'Per-Region Confusion Comparison\n(k={self.k_neighbors} neighbors)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_region_confusion_comparison.png")

        # 为每个模态单独保存一个图
        for modality in modalities_to_show:
            fig_single, ax_single = plt.subplots(figsize=(12, 6))
            result = self.confusion_results[modality]
            values = [result.region_confusion[r] for r in regions]

            ax_single.bar(x, values, color=colors[modality], edgecolor='black', linewidth=0.5, alpha=0.8)
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
            ax_single.set_ylabel('Local Confusion C_i(k)', fontsize=11)
            ax_single.set_xlabel('Brain Regions (sorted by Morph confusion)', fontsize=11)
            ax_single.set_title(f'{modality}: Per-Region Confusion\n(C={self.confusion_results[modality].global_confusion:.4f}, k={self.k_neighbors})',
                               fontsize=13, fontweight='bold')
            ax_single.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            self._save_figure(fig_single, output_dir, f"2_{modality}_region_confusion.png")

    def _plot_hardest_regions_heatmaps(self, output_dir: str):
        """只绘制最难分开的脑区子集的热力图 - 每个模态单独保存"""
        if not self.hardest_regions:
            print("  No hardest regions identified, skipping heatmap")
            return

        hardest = self.hardest_regions
        n_hardest = len(hardest)

        # 获取这些脑区在完整列表中的索引
        region_to_idx = {r: i for i, r in enumerate(self.valid_regions)}
        hardest_idx = [region_to_idx[r] for r in hardest]

        modalities_to_show = ['Morph', 'Gene', 'Proj', 'All']
        n_modalities = len(modalities_to_show)
        cmap = self._create_custom_colormap()

        # 找到这些脑区子集中的最大confusion值（用于统一色标）
        max_confusion = 0
        for modality in modalities_to_show:
            result = self.confusion_results[modality]
            submatrix = result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]
            max_confusion = max(max_confusion, np.max(submatrix[submatrix > 0]))

        # 保存完整的组合图
        fig, axes = plt.subplots(1, n_modalities, figsize=(4 * n_modalities, 4))

        for ax, modality in zip(axes, modalities_to_show):
            result = self.confusion_results[modality]
            submatrix = result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]

            im = ax.imshow(submatrix, cmap=cmap, vmin=0, vmax=max_confusion, aspect='equal')

            ax.set_xticks(np.arange(n_hardest))
            ax.set_yticks(np.arange(n_hardest))
            ax.set_xticklabels(hardest, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(hardest, fontsize=8)

            title_color = '#e74c3c' if '+' in modality or modality == 'All' else '#3498db'
            ax.set_title(f'{modality}\nC={result.global_confusion:.4f}',
                         fontsize=11, fontweight='bold', color=title_color)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Pairwise Confusion C_ij', fontsize=10)

        plt.suptitle(f'Pairwise Confusion: Top {n_hardest} Hardest Regions\n'
                     f'(Selected by {self.reference_modality} confusion)',
                     fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        self._save_figure(fig, output_dir, "3_hardest_regions_heatmaps.png")

        # 为每个模态单独保存热力图
        for modality in modalities_to_show:
            fig_single, ax_single = plt.subplots(figsize=(6, 5))

            result = self.confusion_results[modality]
            submatrix = result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]

            im = ax_single.imshow(submatrix, cmap=cmap, vmin=0, vmax=max_confusion, aspect='equal')

            ax_single.set_xticks(np.arange(n_hardest))
            ax_single.set_yticks(np.arange(n_hardest))
            ax_single.set_xticklabels(hardest, rotation=45, ha='right', fontsize=9)
            ax_single.set_yticklabels(hardest, fontsize=9)

            title_color = '#e74c3c' if '+' in modality or modality == 'All' else '#3498db'
            ax_single.set_title(f'{modality}\nGlobal Confusion: {result.global_confusion:.4f}',
                               fontsize=12, fontweight='bold', color=title_color)

            cbar = plt.colorbar(im, ax=ax_single, shrink=0.8)
            cbar.set_label('Pairwise Confusion C_ij', fontsize=10)

            plt.tight_layout()
            self._save_figure(fig_single, output_dir, f"3_{modality}_hardest_heatmap.png")

    def _plot_pairwise_confusion_improvement(self, output_dir: str):
        """绘制多模态相对于单模态的pairwise改进 - 每个子图单独保存"""
        if not self.hardest_regions:
            return

        hardest = self.hardest_regions
        n_hardest = len(hardest)
        region_to_idx = {r: i for i, r in enumerate(self.valid_regions)}
        hardest_idx = [region_to_idx[r] for r in hardest]

        ref_result = self.confusion_results[self.reference_modality]
        all_result = self.confusion_results['All']

        # 计算改进：(ref - all) / ref * 100
        ref_submatrix = ref_result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]
        all_submatrix = all_result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]

        improvement_matrix = np.zeros_like(ref_submatrix)
        for i in range(n_hardest):
            for j in range(n_hardest):
                if i != j and ref_submatrix[i, j] > 0:
                    improvement_matrix[i, j] = (ref_submatrix[i, j] - all_submatrix[i, j]) / ref_submatrix[i, j] * 100

        cmap_confusion = self._create_custom_colormap()
        cmap_improve = self._create_improvement_colormap()
        max_val = max(np.max(ref_submatrix), np.max(all_submatrix))
        max_imp = max(abs(np.max(improvement_matrix)), abs(np.min(improvement_matrix)))

        # 保存完整的组合图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 左：参考模态的confusion
        ax1 = axes[0]
        im1 = ax1.imshow(ref_submatrix, cmap=cmap_confusion, vmin=0, vmax=max_val, aspect='equal')
        ax1.set_xticks(np.arange(n_hardest))
        ax1.set_yticks(np.arange(n_hardest))
        ax1.set_xticklabels(hardest, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(hardest, fontsize=8)
        ax1.set_title(f'{self.reference_modality}\n(Baseline)', fontsize=12, fontweight='bold', color='#3498db')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='C_ij')

        # 中：All模态的confusion
        ax2 = axes[1]
        im2 = ax2.imshow(all_submatrix, cmap=cmap_confusion, vmin=0, vmax=max_val, aspect='equal')
        ax2.set_xticks(np.arange(n_hardest))
        ax2.set_yticks(np.arange(n_hardest))
        ax2.set_xticklabels(hardest, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(hardest, fontsize=8)
        ax2.set_title('All Modalities\n(Multi-Modal)', fontsize=12, fontweight='bold', color='#e74c3c')
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='C_ij')

        # 右：改进百分比
        ax3 = axes[2]
        im3 = ax3.imshow(improvement_matrix, cmap=cmap_improve, vmin=-max_imp, vmax=max_imp, aspect='equal')
        ax3.set_xticks(np.arange(n_hardest))
        ax3.set_yticks(np.arange(n_hardest))
        ax3.set_xticklabels(hardest, rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(hardest, fontsize=8)
        ax3.set_title('Improvement (%)\n(Green=Better)', fontsize=12, fontweight='bold', color='#27ae60')
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='% Reduction')

        plt.suptitle(f'Pairwise Confusion Improvement: {self.reference_modality} → All Modalities',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_pairwise_improvement.png")

        # 单独保存：参考模态confusion
        fig1, ax1_single = plt.subplots(figsize=(6, 5))
        im1_s = ax1_single.imshow(ref_submatrix, cmap=cmap_confusion, vmin=0, vmax=max_val, aspect='equal')
        ax1_single.set_xticks(np.arange(n_hardest))
        ax1_single.set_yticks(np.arange(n_hardest))
        ax1_single.set_xticklabels(hardest, rotation=45, ha='right', fontsize=9)
        ax1_single.set_yticklabels(hardest, fontsize=9)
        ax1_single.set_title(f'{self.reference_modality} Pairwise Confusion\n(Baseline: C={ref_result.global_confusion:.4f})',
                            fontsize=12, fontweight='bold', color='#3498db')
        plt.colorbar(im1_s, ax=ax1_single, shrink=0.8, label='Pairwise Confusion C_ij')
        plt.tight_layout()
        self._save_figure(fig1, output_dir, f"4_{self.reference_modality}_baseline_confusion.png")

        # 单独保存：All模态confusion
        fig2, ax2_single = plt.subplots(figsize=(6, 5))
        im2_s = ax2_single.imshow(all_submatrix, cmap=cmap_confusion, vmin=0, vmax=max_val, aspect='equal')
        ax2_single.set_xticks(np.arange(n_hardest))
        ax2_single.set_yticks(np.arange(n_hardest))
        ax2_single.set_xticklabels(hardest, rotation=45, ha='right', fontsize=9)
        ax2_single.set_yticklabels(hardest, fontsize=9)
        ax2_single.set_title(f'All Modalities Pairwise Confusion\n(Multi-Modal: C={all_result.global_confusion:.4f})',
                            fontsize=12, fontweight='bold', color='#e74c3c')
        plt.colorbar(im2_s, ax=ax2_single, shrink=0.8, label='Pairwise Confusion C_ij')
        plt.tight_layout()
        self._save_figure(fig2, output_dir, "4_All_multimodal_confusion.png")

        # 单独保存：改进百分比
        fig3, ax3_single = plt.subplots(figsize=(6, 5))
        im3_s = ax3_single.imshow(improvement_matrix, cmap=cmap_improve, vmin=-max_imp, vmax=max_imp, aspect='equal')
        ax3_single.set_xticks(np.arange(n_hardest))
        ax3_single.set_yticks(np.arange(n_hardest))
        ax3_single.set_xticklabels(hardest, rotation=45, ha='right', fontsize=9)
        ax3_single.set_yticklabels(hardest, fontsize=9)

        # 计算平均改进
        valid_improvements = improvement_matrix[improvement_matrix != 0]
        mean_improvement = np.mean(valid_improvements) if len(valid_improvements) > 0 else 0

        ax3_single.set_title(f'Confusion Reduction: {self.reference_modality} → All\n(Mean: {mean_improvement:+.1f}%)',
                            fontsize=12, fontweight='bold', color='#27ae60')
        plt.colorbar(im3_s, ax=ax3_single, shrink=0.8, label='% Reduction (positive=better)')
        plt.tight_layout()
        self._save_figure(fig3, output_dir, "4_improvement_percentage.png")

    def _plot_noise_vs_separation(self, output_dir: str):
        """绘制噪声 vs 分离度的散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        modalities = ['Morph', 'Gene', 'Proj', 'All']
        colors = {'Morph': '#3498db', 'Gene': '#2ecc71', 'Proj': '#9b59b6', 'All': '#e74c3c'}
        markers = {'Morph': 'o', 'Gene': 's', 'Proj': '^', 'All': 'D'}

        # 左图：每个脑区的 noise vs mean-neighbor-distance
        ax1 = axes[0]

        for modality in modalities:
            result = self.confusion_results[modality]
            noises = []
            mean_b = []

            region_to_idx = {r: i for i, r in enumerate(self.valid_regions)}

            for region in self.valid_regions:
                i = region_to_idx[region]
                s_i = result.region_noise[region]
                neighbors = self.neighbor_sets.get(region, [])
                if neighbors:
                    neighbor_b = [result.between_distance_matrix[i, region_to_idx[n]] for n in neighbors]
                    mean_b.append(np.mean(neighbor_b))
                    noises.append(s_i)

            ax1.scatter(mean_b, noises, c=colors[modality], marker=markers[modality],
                        s=80, alpha=0.7, label=modality, edgecolors='black', linewidth=0.5)

        # 添加等confusion线
        x_range = np.linspace(0.01, ax1.get_xlim()[1], 100)
        for c_val in [0.5, 1.0, 2.0]:
            ax1.plot(x_range, c_val * x_range, '--', alpha=0.3, color='gray')
            ax1.text(x_range[-1], c_val * x_range[-1], f'C={c_val}', fontsize=8, color='gray')

        ax1.set_xlabel('Mean Neighbor Distance (b̄)', fontsize=11)
        ax1.set_ylabel('Within-Region Noise (s²)', fontsize=11)
        ax1.set_title('Noise vs Separation by Region', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(alpha=0.3)

        # 右图：模态级别的汇总
        ax2 = axes[1]

        for modality in modalities:
            result = self.confusion_results[modality]

            # 计算平均noise和平均between
            mean_noise = np.mean(list(result.region_noise.values()))
            mean_between = np.mean(result.between_distance_matrix[result.between_distance_matrix > 0])

            ax2.scatter(mean_between, mean_noise, c=colors[modality], marker=markers[modality],
                        s=200, edgecolors='black', linewidth=2, zorder=5)
            ax2.annotate(modality, xy=(mean_between, mean_noise), xytext=(5, 5),
                         textcoords='offset points', fontsize=10, fontweight='bold')

        # 添加等confusion线
        x_range = np.linspace(0.01, ax2.get_xlim()[1] * 1.2, 100)
        for c_val in [0.5, 1.0, 2.0]:
            ax2.plot(x_range, c_val * x_range, '--', alpha=0.3, color='gray')

        ax2.set_xlabel('Mean Between-Region Distance (b̄)', fontsize=11)
        ax2.set_ylabel('Mean Within-Region Noise (s²)', fontsize=11)
        ax2.set_title('Modality-Level Summary\n(Lower-right = better separation)', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)

        # 填充区域
        ax2.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1]], [0, 0],
                         alpha=0.1, color='green', label='Good separation')

        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_noise_vs_separation.png")

    def _plot_comprehensive_analysis(self, output_dir: str):
        """综合分析图"""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1.2, 1, 0.8], hspace=0.35, wspace=0.3)

        # ========== 第一行：最难脑区的热力图 ==========
        if self.hardest_regions:
            hardest = self.hardest_regions[:10]  # 只取前10个
            n_hardest = len(hardest)
            region_to_idx = {r: i for i, r in enumerate(self.valid_regions)}
            hardest_idx = [region_to_idx[r] for r in hardest]

            modalities_row1 = ['Morph', 'Gene', 'Proj', 'All']
            cmap = self._create_custom_colormap()

            # 找最大值
            max_c = 0
            for m in modalities_row1:
                submat = self.confusion_results[m].pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]
                max_c = max(max_c, np.max(submat[submat > 0]))

            for idx, modality in enumerate(modalities_row1):
                ax = fig.add_subplot(gs[0, idx])
                result = self.confusion_results[modality]
                submatrix = result.pairwise_confusion_matrix[np.ix_(hardest_idx, hardest_idx)]

                im = ax.imshow(submatrix, cmap=cmap, vmin=0, vmax=max_c, aspect='equal')

                ax.set_xticks(np.arange(n_hardest))
                ax.set_yticks(np.arange(n_hardest))
                ax.set_xticklabels(hardest, rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(hardest, fontsize=7)

                title_color = '#e74c3c' if '+' in modality or modality == 'All' else '#3498db'
                ax.set_title(f'{modality}\nC={result.global_confusion:.4f}',
                             fontsize=11, fontweight='bold', color=title_color)

                if idx == 3:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('C_ij', fontsize=9)

        # ========== 第二行：分数排名 + 散点图 ==========
        ax_rank = fig.add_subplot(gs[1, :2])

        modalities = list(self.confusion_results.keys())
        scores = [self.confusion_results[m].global_confusion for m in modalities]
        sorted_idx = np.argsort(scores)

        colors = ['#e74c3c' if '+' in modalities[i] or modalities[i] == 'All' else '#3498db'
                  for i in sorted_idx]

        y_pos = np.arange(len(modalities))
        bars = ax_rank.barh(y_pos, [scores[i] for i in sorted_idx],
                            color=colors, edgecolor='black', linewidth=1)

        for i, idx_val in enumerate(sorted_idx):
            score = scores[idx_val]
            ax_rank.text(score + max(scores) * 0.02, i, f'{score:.4f}',
                         va='center', fontsize=9, fontweight='bold')

        ax_rank.set_yticks(y_pos)
        ax_rank.set_yticklabels([modalities[i] for i in sorted_idx], fontsize=10)
        ax_rank.set_xlabel('Confusion Score (lower = better)', fontsize=11)
        ax_rank.set_title('Confusion Score Ranking', fontsize=12, fontweight='bold')
        ax_rank.grid(axis='x', alpha=0.3)

        # Noise vs Between scatter
        ax_scatter = fig.add_subplot(gs[1, 2:])

        modality_colors = {'Morph': '#3498db', 'Gene': '#2ecc71', 'Proj': '#9b59b6',
                           'Morph+Gene': '#f39c12', 'Gene+Proj': '#1abc9c',
                           'Morph+Proj': '#8e44ad', 'All': '#e74c3c'}

        for modality in modalities:
            result = self.confusion_results[modality]
            mean_noise = np.mean(list(result.region_noise.values()))
            mean_between = np.mean(result.between_distance_matrix[result.between_distance_matrix > 0])

            ax_scatter.scatter(mean_between, mean_noise, c=modality_colors.get(modality, 'gray'),
                               s=150, edgecolors='black', linewidth=1.5, zorder=5)
            ax_scatter.annotate(modality, xy=(mean_between, mean_noise), xytext=(5, 5),
                                textcoords='offset points', fontsize=8, fontweight='bold')

        ax_scatter.set_xlabel('Mean Between Distance (b̄)', fontsize=11)
        ax_scatter.set_ylabel('Mean Within Noise (s²)', fontsize=11)
        ax_scatter.set_title('Separability Space\n(Lower-right = better)', fontsize=12, fontweight='bold')
        ax_scatter.grid(alpha=0.3)

        # ========== 第三行：统计摘要 ==========
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')

        single_scores = {m: self.confusion_results[m].global_confusion for m in ['Morph', 'Gene', 'Proj']}
        multi_scores = {m: self.confusion_results[m].global_confusion
                        for m in ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All']}

        best_single = min(single_scores.items(), key=lambda x: x[1])
        worst_single = max(single_scores.items(), key=lambda x: x[1])
        best_multi = min(multi_scores.items(), key=lambda x: x[1])

        improvement = (best_single[1] - best_multi[1]) / best_single[1] * 100

        summary_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['─' * 25, '─' * 20, '─' * 35],
            ['Best Single Modality', f'{best_single[0]}: {best_single[1]:.4f}', 'Baseline for comparison'],
            ['Worst Single Modality', f'{worst_single[0]}: {worst_single[1]:.4f}', 'Most confused modality'],
            ['Best Multi-Modal', f'{best_multi[0]}: {best_multi[1]:.4f}', 'Best fusion result'],
            ['Improvement', f'{improvement:+.2f}%', 'Positive = fusion helps' if improvement > 0 else 'Negative = fusion hurts'],
            ['─' * 25, '─' * 20, '─' * 35],
            ['K Neighbors', f'{self.k_neighbors}', 'Used for local confusion'],
            ['Reference Modality', f'{self.reference_modality}', 'Used to select neighbors'],
            ['Number of Regions', f'{len(self.valid_regions)}', ', '.join(self.valid_regions[:5]) + '...'],
            ['Test Set Size', f'{len(self.test_neuron_ids)}', 'Neurons used for evaluation'],
        ]

        table_text = '\n'.join([f'{row[0]:<28} {row[1]:<25} {row[2]}' for row in summary_data])

        ax_summary.text(0.5, 0.5, table_text, transform=ax_summary.transAxes,
                        fontsize=10, fontfamily='monospace', verticalalignment='center',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                                  edgecolor='#dee2e6', alpha=0.9))

        plt.suptitle('Comprehensive Region Confusion Analysis (V3 - Fisher-style)\n'
                     f'C_i(k) = s_i² / mean(b_ij), k={self.k_neighbors} | Lower is better',
                     fontsize=14, fontweight='bold', y=0.98)

        self._save_figure(fig, output_dir, "6_comprehensive_analysis.png")

    # ==================== Save Results ====================

    def save_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)

        # Save confusion scores
        rows = []
        for modality, result in self.confusion_results.items():
            rows.append({
                'modality': modality,
                'global_confusion': result.global_confusion,
                'global_confusion_sem': result.global_confusion_sem,
                'dimension': result.dimension,
                'k_neighbors': result.k_neighbors,
            })
        df_scores = pd.DataFrame(rows)
        df_scores = df_scores.sort_values('global_confusion')
        df_scores.to_csv(f"{output_dir}/confusion_scores.csv", index=False)

        # Save per-region confusion
        region_data = []
        for modality, result in self.confusion_results.items():
            for region in self.valid_regions:
                region_data.append({
                    'modality': modality,
                    'region': region,
                    'local_confusion': result.region_confusion[region],
                    'within_noise': result.region_noise[region],
                    'n_neurons': result.n_neurons_per_region[region],
                })
        pd.DataFrame(region_data).to_csv(f"{output_dir}/region_confusion.csv", index=False)

        # Save pairwise confusion matrices
        for modality, result in self.confusion_results.items():
            df = pd.DataFrame(result.pairwise_confusion_matrix,
                              index=result.region_labels,
                              columns=result.region_labels)
            df.to_csv(f"{output_dir}/pairwise_confusion_{modality}.csv")

        # Save neighbor sets
        neighbor_data = []
        for region, neighbors in self.neighbor_sets.items():
            neighbor_data.append({
                'region': region,
                'neighbors': ', '.join(neighbors),
            })
        pd.DataFrame(neighbor_data).to_csv(f"{output_dir}/neighbor_sets.csv", index=False)

        # Save hardest regions
        if self.hardest_regions:
            pd.DataFrame({'hardest_regions': self.hardest_regions}).to_csv(
                f"{output_dir}/hardest_regions.csv", index=False)

        # Save full results as pickle
        with open(f"{output_dir}/full_results_v3.pkl", 'wb') as f:
            pickle.dump({
                'confusion_results': self.confusion_results,
                'neighbor_sets': self.neighbor_sets,
                'hardest_regions': self.hardest_regions,
                'valid_regions': self.valid_regions,
                'config': {
                    'pca_variance_threshold': self.pca_variance_threshold,
                    'min_neurons_per_region': self.min_neurons_per_region,
                    'test_ratio': self.test_ratio,
                    'equal_dim_for_fusion': self.equal_dim_for_fusion,
                    'k_neighbors': self.k_neighbors,
                    'reference_modality': self.reference_modality,
                    'random_seed': self.random_seed,
                }
            }, f)

        print(f"\n✓ Results saved to: {output_dir}")

    def run_full_pipeline(self, output_dir: str = "./confusion_analysis_v3"):
        print("\n" + "=" * 80)
        print("Region Confusion Score Analysis V3 (Fisher-style Local Separability)")
        print("=" * 80)
        print("\nKey improvements:")
        print("  1. Fisher-style confusion: C = s²/b (noise/separation ratio)")
        print("  2. Dimension-normalized: all quantities divided by d")
        print("  3. Local neighbors: only compare to k nearest regions")
        print(f"  4. Neighbor selection: based on {self.reference_modality} on train set")

        n = self.load_all_data()
        if n == 0:
            print("No valid data found!")
            return

        self.compute_confusion_scores()
        self.visualize_results(output_dir)
        self.save_results(output_dir)

        self._print_final_summary()

    def _print_final_summary(self):
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        print("\n【Configuration】")
        print(f"  PCA variance threshold: {self.pca_variance_threshold}")
        print(f"  Equal dim for fusion: {self.equal_dim_for_fusion}")
        print(f"  K neighbors: {self.k_neighbors}")
        print(f"  Reference modality: {self.reference_modality}")

        print("\n【Data】")
        print(f"  Valid regions: {len(self.valid_regions)}")
        print(f"  Train neurons: {len(self.train_neuron_ids)}")
        print(f"  Test neurons: {len(self.test_neuron_ids)}")

        print("\n【Confusion Scores】(lower = better separation)")
        print(f"{'Modality':<15} {'Confusion':>12} {'±SEM':>10}")
        print("-" * 40)

        sorted_results = sorted(self.confusion_results.items(),
                                key=lambda x: x[1].global_confusion)

        for modality, result in sorted_results:
            marker = "★" if '+' in modality or modality == 'All' else " "
            print(f"{marker} {modality:<13} {result.global_confusion:>12.4f} {result.global_confusion_sem:>10.4f}")

        single_best = min(self.confusion_results[m].global_confusion for m in ['Morph', 'Gene', 'Proj'])
        multi_best = min(self.confusion_results[m].global_confusion
                         for m in ['Morph+Gene', 'Gene+Proj', 'Morph+Proj', 'All'])

        improvement = (single_best - multi_best) / single_best * 100

        print("\n【Conclusion】")
        print(f"  Best single modality: {single_best:.4f}")
        print(f"  Best multi-modal: {multi_best:.4f}")
        print(f"  Improvement from fusion: {improvement:+.2f}%")

        if improvement > 0:
            print(f"\n  ✓ Multi-modal fusion REDUCES confusion by {improvement:.1f}%!")
            print(f"    → On the {self.k_neighbors} nearest neighbors (hardest to separate),")
            print(f"      multi-modal provides significantly better region discrimination.")
        else:
            print(f"\n  ✗ Multi-modal fusion does NOT improve separation.")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = ("./confusion_analysis_v2_Morph")

    with RegionConfusionAnalysisV3(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            min_neurons_per_region=10,
            test_ratio=0.2,
            equal_dim_for_fusion=20,
            k_neighbors=5,  # 只看5个最近邻
            reference_modality='Morph',  # 用Morph选择近邻
            # reference_modality= 'Proj',  # 用Gene选择近邻
            epsilon=1e-8,
            random_seed=42,
    ) as analysis:
        analysis.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()