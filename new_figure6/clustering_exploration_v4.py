"""
Clustering Number Pre-experiment V4 (Fixed Version)
====================================================
关键修复：
1. 与分类任务保持一致的 train/test split
2. 只在 train set 上 fit scaler/PCA/聚类
3. 添加 UMAP 非线性降维选项
4. 添加不同 Scaler 选项（RobustScaler）
5. 使用相同的 pca_variance_threshold=0.95

Author: Claude (Fixed)
Date: 2025-01
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import neo4j

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()
print(f"Detected {N_CORES} CPU cores")

# ==================== Global Variables for Worker ====================
_GLOBAL_X = None
_GLOBAL_N_INIT = None
_GLOBAL_RANDOM_STATE = None
_GLOBAL_SILHOUETTE_SAMPLE_SIZE = None
_GLOBAL_N_SEEDS = None


def _init_worker(X: np.ndarray, n_init: int, random_state: int,
                 sil_sample_size: int, n_seeds: int):
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE
    global _GLOBAL_SILHOUETTE_SAMPLE_SIZE, _GLOBAL_N_SEEDS
    _GLOBAL_X = X
    _GLOBAL_N_INIT = n_init
    _GLOBAL_RANDOM_STATE = random_state
    _GLOBAL_SILHOUETTE_SAMPLE_SIZE = sil_sample_size
    _GLOBAL_N_SEEDS = n_seeds


def _cluster_single_k(k: int) -> dict:
    """Clustering for a single K value with multi-seed stability."""
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE
    global _GLOBAL_SILHOUETTE_SAMPLE_SIZE, _GLOBAL_N_SEEDS

    X = _GLOBAL_X
    n_init = _GLOBAL_N_INIT
    base_seed = _GLOBAL_RANDOM_STATE
    sil_sample_size = _GLOBAL_SILHOUETTE_SAMPLE_SIZE
    n_seeds = _GLOBAL_N_SEEDS

    if X is None or len(X) == 0:
        raise ValueError("Worker received empty data")

    sils, chs, dbs, inertias = [], [], [], []
    start = time.time()

    for i in range(n_seeds):
        rs = base_seed + i
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=rs)
        labels = kmeans.fit_predict(X)

        # 检查标签数量
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            raise ValueError(f"Only {n_unique} unique labels produced")

        if sil_sample_size and X.shape[0] > sil_sample_size:
            sil = silhouette_score(X, labels, sample_size=sil_sample_size, random_state=rs)
        else:
            sil = silhouette_score(X, labels)

        sils.append(sil)
        chs.append(calinski_harabasz_score(X, labels))
        dbs.append(davies_bouldin_score(X, labels))
        inertias.append(kmeans.inertia_)

    elapsed = time.time() - start

    return {
        'k': k,
        'silhouette': float(np.mean(sils)),
        'silhouette_std': float(np.std(sils)),
        'calinski_harabasz': float(np.mean(chs)),
        'calinski_harabasz_std': float(np.std(chs)),
        'davies_bouldin': float(np.mean(dbs)),
        'davies_bouldin_std': float(np.std(dbs)),
        'inertia': float(np.mean(inertias)),
        'inertia_std': float(np.std(inertias)),
        'time_seconds': elapsed,
    }


def find_elbow_point(k_values: np.ndarray, values: np.ndarray,
                     direction: str = 'decreasing',
                     curve: str = 'convex') -> Tuple[int, np.ndarray]:
    """Kneedle algorithm for elbow detection"""
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    v_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)

    if direction == 'decreasing':
        v_norm = 1 - v_norm

    if curve == 'convex':
        diff = v_norm - k_norm
    else:
        diff = k_norm - v_norm

    elbow_idx = np.argmax(diff)
    return k_values[elbow_idx], diff


@dataclass
class ClusterMetrics:
    """Clustering metrics"""
    k: int
    silhouette: float
    silhouette_std: float
    calinski_harabasz: float
    calinski_harabasz_std: float
    davies_bouldin: float
    davies_bouldin_std: float
    inertia: float
    inertia_std: float
    time_seconds: float


class ClusteringExplorerV4:
    """
    修正版聚类探索器

    关键修复：
    1. 使用与分类任务相同的 train/test split
    2. 只在 train set 上 fit scaler/PCA
    3. 只在 train set 上计算聚类指标
    4. 添加 UMAP 降维选项
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

    VOTING_WEIGHTS = {
        'Kneedle_Inertia': 2.0,
        'Max_Silhouette': 1.5,
        'Max_CH': 1.0,
        'Min_DB': 1.0,
    }

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 4.0,
                 pca_variance_threshold: float = 0.95,  # 与分类任务一致
                 test_ratio: float = 0.2,  # 与分类任务一致
                 k_range: Tuple[int, int] = (2, 50),
                 min_k: int = 5,
                 n_init: int = 10,
                 n_seeds: int = 5,
                 n_jobs: int = 8,
                 silhouette_sample_size: int = 3000,
                 use_umap: bool = False,
                 umap_n_components: int = 15,
                 scaler_type: str = 'standard',  # 'standard', 'robust', 'minmax'
                 gene_log_transform: bool = True):

        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.k_min, self.k_max = k_range
        self.min_k = max(min_k, self.k_min)
        self.n_init = n_init
        self.n_seeds = n_seeds
        self.n_jobs = min(n_jobs, N_CORES)
        self.silhouette_sample_size = silhouette_sample_size
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.scaler_type = scaler_type
        self.gene_log_transform = gene_log_transform

        print(f"Configuration (V4 - Fixed Version):")
        print(f"  K range: {self.k_min} - {self.k_max}")
        print(f"  MIN_K: {self.min_k}")
        print(f"  PCA variance threshold: {self.pca_variance_threshold}")
        print(f"  Test ratio: {self.test_ratio}")
        print(f"  Scaler type: {self.scaler_type}")
        print(f"  Use UMAP: {self.use_umap}")
        if self.use_umap:
            print(f"  UMAP components: {self.umap_n_components}")
        print(f"  Gene log transform: {self.gene_log_transform}")

        # Data
        self.valid_neuron_ids: List[str] = []
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # Train/Test split (与分类任务一致)
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

        # Processed vectors
        self.morph_vectors: np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # Preprocessors (保存用于分类任务)
        self.preprocessors: Dict[str, Dict] = {}

        # Results
        self.results: Dict[str, List[ClusterMetrics]] = {}
        self.elbow_points: Dict[str, Dict[str, int]] = {}
        self.recommendations: Dict[str, Dict] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_scaler(self):
        """Get scaler based on configuration"""
        if self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()

    def load_all_data(self) -> int:
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._filter_valid_neurons()
        self._split_train_test()  # 关键：与分类任务一致
        self._process_all_vectors()

        print(f"\n✓ Data loading complete:")
        print(f"  Total neurons: {len(self.valid_neuron_ids)}")
        print(f"  Train set: {len(self.train_idx)}")
        print(f"  Test set: {len(self.test_idx)}")
        print(f"  Morph: {self.morph_vectors.shape}")
        print(f"  Gene: {self.gene_vectors.shape}")
        print(f"  Proj: {self.proj_vectors.shape}")

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

        print(f"  Loaded {len(self.axon_features_raw)} neurons")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  Valid neurons: {len(self.valid_neuron_ids)}")

    def _split_train_test(self):
        """与分类任务完全一致的 train/test split"""
        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)  # 与分类任务相同的 seed
        np.random.shuffle(indices)

        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]

        print(f"  Train/Test split: {len(self.train_idx)}/{len(self.test_idx)} (seed=42)")

    def _process_all_vectors(self):
        """处理向量 - 只在 train set 上 fit"""
        print("\nProcessing vectors (fit on TRAIN SET only)...")
        neurons = self.valid_neuron_ids

        # Morphology
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_prep = self._process_vector(
            morph_raw, 'Morph', do_log=True)
        self.preprocessors['morph'] = morph_prep

        # Molecular - 使用可配置的 log transform
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        # 只根据 train set 过滤列
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        gene_raw = gene_raw[:, valid_cols]
        self.gene_vectors, gene_prep = self._process_vector(
            gene_raw, 'Gene', do_log=self.gene_log_transform)
        self.preprocessors['gene'] = gene_prep
        self.preprocessors['gene']['valid_cols'] = valid_cols

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        proj_raw = proj_raw[:, valid_cols]
        self.proj_vectors, proj_prep = self._process_vector(
            proj_raw, 'Proj', do_log=True)
        self.preprocessors['proj'] = proj_prep
        self.preprocessors['proj']['valid_cols'] = valid_cols

    def _process_vector(self, X_raw: np.ndarray, name: str,
                        do_log: bool = False) -> Tuple[np.ndarray, Dict]:
        """预处理向量 - 只在 train set 上 fit"""
        original_dims = X_raw.shape[1]

        if do_log:
            X = np.log1p(X_raw)
        else:
            X = X_raw

        # 只在 train set 上 fit scaler
        scaler = self._get_scaler()
        scaler.fit(X[self.train_idx])
        X_scaled = scaler.transform(X)

        # 选择降维方法
        if self.use_umap:
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    n_neighbors=30,
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42
                )
                # 只在 train set 上 fit
                reducer.fit(X_scaled[self.train_idx])
                X_reduced = reducer.transform(X_scaled)
                variance = None
                print(f"  {name}: {original_dims}D → {X_reduced.shape[1]}D (UMAP)")
            except ImportError:
                print("  Warning: UMAP not installed, falling back to PCA")
                self.use_umap = False

        if not self.use_umap:
            # PCA - 只在 train set 上 fit
            pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
            pca.fit(X_scaled[self.train_idx])
            X_reduced = pca.transform(X_scaled)
            variance = np.sum(pca.explained_variance_ratio_)
            reducer = pca
            print(f"  {name}: {original_dims}D → {X_reduced.shape[1]}D ({variance:.1%} var)")

        preprocessor = {
            'scaler': scaler,
            'reducer': reducer,
            'do_log': do_log,
            'use_umap': self.use_umap,
        }

        return X_reduced, preprocessor

    def explore_clustering(self, X: np.ndarray, name: str) -> List[ClusterMetrics]:
        """探索聚类效果 - 只使用 TRAIN SET"""
        print(f"\n{'=' * 60}")
        print(f"{name} (K={self.k_min} to {self.k_max})")
        print(f"{'=' * 60}")

        # 关键修复：只使用 train set
        X_train = X[self.train_idx]
        print(f"  Using TRAIN SET only: {X_train.shape}")
        print(f"  Workers: {self.n_jobs}, Seeds: {self.n_seeds}")

        k_values = list(range(self.k_min, self.k_max + 1))
        results = []

        print(f"\n  Computing metrics for {len(k_values)} K values...")
        main_start = time.time()

        with ProcessPoolExecutor(
                max_workers=self.n_jobs,
                initializer=_init_worker,
                initargs=(X_train, self.n_init, 42, self.silhouette_sample_size, self.n_seeds)
        ) as executor:
            futures = {executor.submit(_cluster_single_k, k): k for k in k_values}

            completed = 0
            for future in as_completed(futures):
                k = futures[future]
                try:
                    result_dict = future.result()
                    metric = ClusterMetrics(**result_dict)
                    results.append(metric)
                    completed += 1

                    if completed % 10 == 0 or completed == len(k_values):
                        elapsed = time.time() - main_start
                        print(f"    Progress: {completed}/{len(k_values)} | Elapsed: {elapsed:.1f}s")

                except Exception as e:
                    print(f"    K={k} failed: {e}")

        results.sort(key=lambda x: x.k)
        print(f"  Completed in {time.time() - main_start:.1f}s")

        # 找拐点
        self._find_elbow_points(name, results)
        self._compute_recommendation(name)

        return results

    def _find_elbow_points(self, name: str, results: List[ClusterMetrics]):
        """Find elbow points with MIN_K constraint"""
        k_values = np.array([r.k for r in results])
        inertia = np.array([r.inertia for r in results])
        sil = np.array([r.silhouette for r in results])
        ch = np.array([r.calinski_harabasz for r in results])
        db = np.array([r.davies_bouldin for r in results])

        MIN_K = self.min_k
        self.elbow_points[name] = {}

        # Kneedle (inertia)
        elbow_inertia, _ = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
        elbow_inertia = int(max(elbow_inertia, MIN_K))
        self.elbow_points[name]['Kneedle_Inertia'] = elbow_inertia

        # Silhouette / CH / DB - 只考虑 K >= MIN_K
        valid = k_values >= MIN_K
        vk, vsil, vch, vdb = k_values[valid], sil[valid], ch[valid], db[valid]

        self.elbow_points[name]['Max_Silhouette'] = int(vk[np.argmax(vsil)])
        self.elbow_points[name]['Max_CH'] = int(vk[np.argmax(vch)])
        self.elbow_points[name]['Min_DB'] = int(vk[np.argmin(vdb)])

        print(f"\n  Elbow detection results (MIN_K={MIN_K}):")
        for method, k in self.elbow_points[name].items():
            print(f"    {method}: K={k}")

    def _compute_recommendation(self, name: str):
        """Compute weighted voting recommendation"""
        elbows = self.elbow_points[name]

        k_scores = {}
        for method, k in elbows.items():
            weight = self.VOTING_WEIGHTS.get(method, 1.0)
            k_scores[k] = k_scores.get(k, 0.0) + weight

        recommended_k = max(k_scores.keys(), key=lambda kk: k_scores[kk])
        recommended_score = k_scores[recommended_k]

        # Candidate range
        sorted_candidates = sorted(k_scores.items(), key=lambda x: -x[1])
        candidate_range = sorted([k for k, _ in sorted_candidates[:5]])

        self.recommendations[name] = {
            'recommended_k': recommended_k,
            'weighted_score': recommended_score,
            'k_scores': k_scores,
            'candidate_range': candidate_range,
            'all_methods': elbows.copy(),
        }

        print(f"\n  Recommendation: K={recommended_k} (score={recommended_score:.1f})")
        print(f"  Candidates: {candidate_range}")

    def run_exploration(self):
        """Run exploration for all modalities"""
        print("\n" + "=" * 80)
        print("Clustering Exploration (V4 - Fixed)")
        print(f"Using TRAIN SET only for clustering")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        for name, vectors in modalities:
            self.results[name] = self.explore_clustering(vectors, name)

        return self.results

    def visualize_results(self, output_dir: str = "."):
        """Generate visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_silhouette_curves(output_dir)
        self._plot_comprehensive_view(output_dir)

        print(f"\n✓ Figures saved to: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_silhouette_curves(self, output_dir: str):
        """Plot silhouette curves with recommendations"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph': '#2E86AB', 'Gene': '#A23B72', 'Proj': '#F18F01'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            sil = np.array([r.silhouette for r in results])
            sil_std = np.array([r.silhouette_std for r in results])

            # Error band
            ax.fill_between(k_values, sil - sil_std, sil + sil_std,
                            alpha=0.2, color=colors[name])

            # Main curve
            ax.plot(k_values, sil, color=colors[name], linewidth=2,
                    marker='o', markersize=3, label='Silhouette')

            # Smoothed curve
            sil_smooth = gaussian_filter1d(sil, sigma=1.5)
            ax.plot(k_values, sil_smooth, color='black', linewidth=2,
                    linestyle='--', alpha=0.7, label='Smoothed')

            # Mark recommended K
            rec_k = self.recommendations[name]['recommended_k']
            rec_idx = np.where(k_values == rec_k)[0][0]
            ax.axvline(x=rec_k, color='red', linestyle='-', linewidth=2.5)
            ax.scatter([rec_k], [sil[rec_idx]], color='red', s=200, zorder=10,
                       marker='*', edgecolor='black')
            ax.annotate(f'K={rec_k}\n({sil[rec_idx]:.3f})',
                        xy=(rec_k, sil[rec_idx]),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=11, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='red', alpha=0.9))

            # MIN_K line
            ax.axvline(x=self.min_k, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

            ax.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax.set_ylabel('Silhouette Score', fontsize=12)
            ax.set_title(f'{name}', fontsize=14, fontweight='bold', color=colors[name])
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.suptitle('Silhouette Score Analysis (Computed on TRAIN SET only)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "silhouette_curves.png")

    def _plot_comprehensive_view(self, output_dir: str):
        """Comprehensive metrics view"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        colors = {'Morph': '#2E86AB', 'Gene': '#A23B72', 'Proj': '#F18F01'}
        modalities = list(self.results.keys())

        for row, name in enumerate(modalities):
            results = self.results[name]
            k_values = np.array([r.k for r in results])
            color = colors[name]
            rec_k = self.recommendations[name]['recommended_k']

            # Silhouette
            ax1 = axes[row, 0]
            sil = np.array([r.silhouette for r in results])
            ax1.plot(k_values, sil, color=color, linewidth=2)
            ax1.fill_between(k_values, sil, alpha=0.2, color=color)
            ax1.axvline(x=rec_k, color='red', linestyle='-', linewidth=2)
            ax1.set_title(f'{name} - Silhouette', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # CH
            ax2 = axes[row, 1]
            ch = np.array([r.calinski_harabasz for r in results])
            ax2.plot(k_values, ch, color=color, linewidth=2)
            ax2.fill_between(k_values, ch, alpha=0.2, color=color)
            ax2.axvline(x=rec_k, color='red', linestyle='-', linewidth=2)
            ax2.set_title(f'{name} - Calinski-Harabasz', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # DB
            ax3 = axes[row, 2]
            db = np.array([r.davies_bouldin for r in results])
            ax3.plot(k_values, db, color=color, linewidth=2)
            ax3.fill_between(k_values, db, alpha=0.2, color=color)
            ax3.axvline(x=rec_k, color='red', linestyle='-', linewidth=2)
            ax3.set_title(f'{name} - Davies-Bouldin', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Inertia (Elbow)
            ax4 = axes[row, 3]
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())
            ax4.plot(k_values, inertia_norm, color=color, linewidth=2)
            ax4.fill_between(k_values, inertia_norm, alpha=0.2, color=color)
            ax4.axvline(x=rec_k, color='red', linestyle='-', linewidth=2, label=f'Rec={rec_k}')
            ax4.set_title(f'{name} - Elbow (Inertia)', fontweight='bold')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Clustering Metrics (TRAIN SET only)',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "comprehensive_metrics.png")

    def save_results(self, output_dir: str = "."):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)

        # Metrics CSV
        for name, results in self.results.items():
            rows = [{
                'k': r.k,
                'silhouette': r.silhouette,
                'silhouette_std': r.silhouette_std,
                'calinski_harabasz': r.calinski_harabasz,
                'davies_bouldin': r.davies_bouldin,
                'inertia': r.inertia,
            } for r in results]
            pd.DataFrame(rows).to_csv(f"{output_dir}/metrics_{name.lower()}.csv", index=False)

        # Recommendations
        rec_rows = []
        for name, rec in self.recommendations.items():
            row = {
                'modality': name,
                'recommended_k': rec['recommended_k'],
                'weighted_score': rec['weighted_score'],
                'candidate_range': str(rec['candidate_range']),
            }
            row.update({f'{m}_k': k for m, k in rec['all_methods'].items()})
            rec_rows.append(row)
        pd.DataFrame(rec_rows).to_csv(f"{output_dir}/recommended_k.csv", index=False)

        # Full results pickle
        with open(f"{output_dir}/full_results.pkl", 'wb') as f:
            pickle.dump({
                'results': self.results,
                'recommendations': self.recommendations,
                'elbow_points': self.elbow_points,
                'preprocessors': self.preprocessors,
                'train_idx': self.train_idx,
                'test_idx': self.test_idx,
                'config': {
                    'k_range': (self.k_min, self.k_max),
                    'min_k': self.min_k,
                    'pca_variance_threshold': self.pca_variance_threshold,
                    'test_ratio': self.test_ratio,
                    'use_umap': self.use_umap,
                    'scaler_type': self.scaler_type,
                }
            }, f)

        print(f"\n✓ Results saved to: {output_dir}")

    def run_full_pipeline(self, output_dir: str = "./clustering_exploration_v4"):
        """Run full pipeline"""
        print("\n" + "=" * 80)
        print("Clustering Exploration V4 (Fixed Version)")
        print("Key fix: Train/test split consistent with classification")
        print("=" * 80)

        start_time = time.time()

        n = self.load_all_data()
        if n == 0:
            return

        self.run_exploration()
        self.visualize_results(output_dir)
        self.save_results(output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL RECOMMENDATIONS")
        print("=" * 80)
        for name, rec in self.recommendations.items():
            print(f"  {name}: K = {rec['recommended_k']} (candidates: {rec['candidate_range']})")

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./clustering_exploration_v4"

    with ClusteringExplorerV4(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,  # 与分类任务一致
            test_ratio=0.2,  # 与分类任务一致
            k_range=(2, 50),
            min_k=5,
            n_init=10,
            n_seeds=5,
            n_jobs=8,
            silhouette_sample_size=3000,
            use_umap=False,  # 可选：尝试 UMAP
            umap_n_components=15,
            scaler_type='standard',  # 可选：'robust' for gene data
            gene_log_transform=True,
    ) as explorer:
        explorer.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()