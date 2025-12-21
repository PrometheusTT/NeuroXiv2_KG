"""
Clustering Number Pre-experiment:  Search for Optimal K (Optimized & Robust Version)
==================================================================================
Features:
1.Kneedle algorithm for automatic elbow detection
2.Second derivative/curvature visualization
3.Gap Statistic (with coarse-fine search)
4.Multi-seed stability for robust metrics
5.MIN_K constraint to avoid trivial solutions
6.Weighted voting for final recommendation
7.Candidate range output for sensitivity analysis

Author: PrometheusTT
Date: 2025-01-xx
"""

import os

# ===== Critical: Limit inner threads to avoid oversubscription =====
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

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Neo4j
import neo4j

# Settings
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


def _init_worker(X:  np.ndarray, n_init: int, random_state:  int, 
                 sil_sample_size: int, n_seeds: int):
    """Initialize worker with shared data (called once per worker process)"""
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE
    global _GLOBAL_SILHOUETTE_SAMPLE_SIZE, _GLOBAL_N_SEEDS
    _GLOBAL_X = X
    _GLOBAL_N_INIT = n_init
    _GLOBAL_RANDOM_STATE = random_state
    _GLOBAL_SILHOUETTE_SAMPLE_SIZE = sil_sample_size
    _GLOBAL_N_SEEDS = n_seeds


def _cluster_single_k(k: int) -> dict:
    """
    Clustering for a single K value with multi-seed stability.
    Runs multiple seeds and returns mean metrics for robustness.
    """
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE
    global _GLOBAL_SILHOUETTE_SAMPLE_SIZE, _GLOBAL_N_SEEDS
    
    X = _GLOBAL_X
    n_init = _GLOBAL_N_INIT
    base_seed = _GLOBAL_RANDOM_STATE
    sil_sample_size = _GLOBAL_SILHOUETTE_SAMPLE_SIZE
    n_seeds = _GLOBAL_N_SEEDS
    
    sils, chs, dbs, inertias = [], [], [], []
    start = time.time()
    
    for i in range(n_seeds):
        rs = base_seed + i
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=rs)
        labels = kmeans.fit_predict(X)
        
        # Silhouette with sampling
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
        'silhouette_std':  float(np.std(sils)),
        'calinski_harabasz': float(np.mean(chs)),
        'calinski_harabasz_std': float(np.std(chs)),
        'davies_bouldin': float(np.mean(dbs)),
        'davies_bouldin_std': float(np.std(dbs)),
        'inertia': float(np.mean(inertias)),
        'inertia_std':  float(np.std(inertias)),
        'time_seconds': elapsed,
    }


# ==================== Kneedle Algorithm ====================

def find_elbow_point(k_values: np.ndarray, values: np.ndarray,
                     direction: str = 'decreasing',
                     curve:  str = 'convex') -> Tuple[int, np.ndarray]: 
    """Use Kneedle algorithm to find elbow point"""
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


def compute_curvature(k_values: np.ndarray, values:  np.ndarray,
                      smooth_sigma: float = 1.5) -> np.ndarray:
    """Calculate curve curvature"""
    values_smooth = gaussian_filter1d(values, sigma=smooth_sigma)
    first_deriv = np.gradient(values_smooth, k_values)
    second_deriv = np.gradient(first_deriv, k_values)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5
    return curvature


def find_elbow_by_curvature(k_values: np.ndarray, values: np.ndarray,
                            min_k: int = 5) -> int:
    """Find elbow point by maximum curvature"""
    curvature = compute_curvature(k_values, values)
    valid_mask = k_values >= min_k
    valid_k = k_values[valid_mask]
    valid_curv = curvature[valid_mask]
    max_idx = np.argmax(valid_curv)
    return valid_k[max_idx]


# ==================== Gap Statistic ====================

def compute_gap_statistic_with_progress(X: np.ndarray, k_values: List[int],
                                        n_refs: int = 5,
                                        random_state: int = 42,
                                        modality_name: str = "",
                                        max_gap_dims: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gap Statistic with progress output.
    
    Uses only first max_gap_dims dimensions for Gap calculation
    to avoid high-dimensional sparse reference distribution problem.
    """
    np.random.seed(random_state)
    
    # Use only first max_gap_dims dimensions for Gap (high-dim uniform is unreliable)
    if X.shape[1] > max_gap_dims: 
        X_gap = X[:, : max_gap_dims]
        print(f"    [Gap] Using first {max_gap_dims} dims (original:  {X.shape[1]})")
    else:
        X_gap = X
    
    n_samples, n_features = X_gap.shape
    mins = X_gap.min(axis=0)
    maxs = X_gap.max(axis=0)

    gaps = []
    gap_stds = []
    
    total_k = len(k_values)
    start_time = time.time()

    for idx, k in enumerate(k_values):
        k_start = time.time()
        
        # Original data
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(X_gap)
        log_wk = np.log(kmeans.inertia_ + 1e-10)

        # Reference data
        log_wk_refs = []
        for ref_idx in range(n_refs):
            X_ref = np.random.uniform(mins, maxs, (n_samples, n_features))
            kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=random_state + ref_idx)
            kmeans_ref.fit(X_ref)
            log_wk_refs.append(np.log(kmeans_ref.inertia_ + 1e-10))

        log_wk_refs = np.array(log_wk_refs)
        gap = log_wk_refs.mean() - log_wk
        gap_std = log_wk_refs.std() * np.sqrt(1 + 1/n_refs)

        gaps.append(gap)
        gap_stds.append(gap_std)
        
        # Progress output
        k_elapsed = time.time() - k_start
        total_elapsed = time.time() - start_time
        avg_per_k = total_elapsed / (idx + 1)
        eta = avg_per_k * (total_k - idx - 1)
        
        print(f"    Gap[{modality_name}] K={k: 3d}:  {idx+1}/{total_k} | "
              f"This K: {k_elapsed:.1f}s | Total: {total_elapsed:.1f}s | ETA: {eta:.1f}s")

    return np.array(gaps), np.array(gap_stds)


def find_optimal_k_gap(k_values: np.ndarray, gaps: np.ndarray,
                       gap_stds: np.ndarray) -> int:
    """Find optimal K based on Gap Statistic rule"""
    for i in range(len(k_values) - 1):
        if gaps[i] >= gaps[i+1] - gap_stds[i+1]:
            return k_values[i]
    return k_values[-1]


@dataclass
class ClusterMetrics:
    """Clustering metrics with stability info"""
    k: int
    silhouette: float
    silhouette_std:  float
    calinski_harabasz:  float
    calinski_harabasz_std: float
    davies_bouldin: float
    davies_bouldin_std: float
    inertia:  float
    inertia_std: float
    time_seconds: float


class ClusteringExplorer:
    """Clustering Number Explorer (Optimized & Robust Version)"""

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

    # Voting weights for different methods
    VOTING_WEIGHTS = {
        'Kneedle_Inertia': 2.0,      # Most reliable for elbow
        'Curvature_Inertia': 1.5,
        'Gap_Statistic': 1.5,
        'Max_Silhouette': 1.0,
        'Max_CH': 1.0,
        'Min_DB': 1.0,
    }

    def __init__(self, uri: str, user: str, password:  str,
                 data_dir: str, database:  str = "neo4j",
                 search_radius: float = 8.0,
                 pca_variance_threshold: float = 0.95,
                 k_range: Tuple[int, int] = (2, 70),
                 min_k: int = 5,
                 n_init: int = 10,
                 n_seeds: int = 3,
                 n_jobs: int = 8,
                 compute_gap: bool = True,
                 gap_n_refs: int = 5,
                 gap_k_step: int = 5,
                 gap_max_dims: int = 20,
                 silhouette_sample_size: int = 3000):
        """
        Args:
            k_range: K value search range (min, max)
            min_k: Minimum K for biological relevance (K=2,3 usually meaningless)
            n_init: KMeans initialization count
            n_seeds: Number of random seeds for stability (recommended: 3-5)
            n_jobs: Number of parallel workers
            compute_gap: Whether to compute Gap Statistic
            gap_n_refs: Number of reference datasets for Gap
            gap_k_step: Step size for Gap K sampling
            gap_max_dims: Max dimensions for Gap (high-dim uniform is unreliable)
            silhouette_sample_size: Sample size for silhouette score
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.k_min, self.k_max = k_range
        self.min_k = max(min_k, self.k_min)  # Ensure min_k >= k_min
        self.n_init = n_init
        self.n_seeds = n_seeds
        self.n_jobs = min(n_jobs, N_CORES)
        self.compute_gap = compute_gap
        self.gap_n_refs = gap_n_refs
        self.gap_k_step = gap_k_step
        self.gap_max_dims = gap_max_dims
        self.silhouette_sample_size = silhouette_sample_size

        print(f"Configuration:")
        print(f"  K range: {self.k_min} - {self.k_max}")
        print(f"  MIN_K (biological constraint): {self.min_k}")
        print(f"  n_jobs: {self.n_jobs}")
        print(f"  n_seeds (for stability): {self.n_seeds}")
        print(f"  silhouette_sample_size: {self.silhouette_sample_size}")
        print(f"  compute_gap:  {self.compute_gap}")
        if self.compute_gap:
            print(f"  gap_n_refs: {self.gap_n_refs}")
            print(f"  gap_k_step: {self.gap_k_step}")
            print(f"  gap_max_dims: {self.gap_max_dims}")

        # Data
        self.valid_neuron_ids:  List[str] = []
        self.axon_features_raw:  Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses:  List[str] = []
        self.all_target_regions: List[str] = []

        # PCA vectors
        self.morph_vectors:  np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # Results
        self.results: Dict[str, List[ClusterMetrics]] = {}
        self.gap_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.elbow_points: Dict[str, Dict[str, int]] = {}
        self.recommendations: Dict[str, Dict] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Data Loading ====================

    def load_all_data(self) -> int:
        """Load data"""
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._filter_valid_neurons()
        self._process_all_vectors()

        print(f"\n✓ Data loading complete:")
        print(f"  Neurons: {len(self.valid_neuron_ids)}")
        print(f"  Morph:  {self.morph_vectors.shape}")
        print(f"  Gene:  {self.gene_vectors.shape}")
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
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t: Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]
        print(f"  Projection targets: {len(self.all_target_regions)} brain regions")

    def _load_all_neuron_features(self):
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n: Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        RETURN n.neuron_id AS neuron_id, {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id:  $neuron_id})-[p:PROJECT_TO]->(t: Subregion)
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
                proj_dict = {r['target']:  r['weight'] for r in proj_result
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

    def _process_all_vectors(self):
        """Process vectors"""
        print("\nProcessing vectors...")
        neurons = self.valid_neuron_ids

        # Morphology
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors = self._process_vector(morph_raw, 'Morph', do_log=True)

        # Molecular
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw.sum(axis=0)
        gene_raw = gene_raw[: , col_sums > 0]
        self.gene_vectors = self._process_vector(gene_raw, 'Gene')

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw.sum(axis=0)
        proj_raw = proj_raw[:, col_sums > 0]
        self.proj_vectors = self._process_vector(proj_raw, 'Proj', do_log=True)

    def _process_vector(self, X_raw: np.ndarray, name: str,
                        do_log: bool = False) -> np.ndarray:
        original_dims = X_raw.shape[1]

        if do_log: 
            X = np.log1p(X_raw)
        else: 
            X = X_raw

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  {name}:  {original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        return X_pca

    # ==================== Clustering Exploration ====================

    def explore_clustering(self, X: np.ndarray, name: str) -> List[ClusterMetrics]:
        """Explore clustering effects for different K values"""
        print(f"\n{'='*60}")
        print(f"{name} (K={self.k_min} to {self.k_max})")
        print(f"{'='*60}")
        print(f"  Data shape: {X.shape}")
        print(f"  Workers: {self.n_jobs}, Seeds: {self.n_seeds}, Silhouette sample:  {self.silhouette_sample_size}")

        k_values = list(range(self.k_min, self.k_max + 1))
        results = []

        # ===== Phase 1: Main clustering loop =====
        print(f"\n  [Phase 1] Main clustering loop ({len(k_values)} K values, {self.n_seeds} seeds each)...")
        main_start = time.time()

        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_init_worker,
            initargs=(X, self.n_init, 42, self.silhouette_sample_size, self.n_seeds)
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
                        eta = elapsed / completed * (len(k_values) - completed)
                        print(f"    Progress: {completed}/{len(k_values)} | "
                              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                              
                except Exception as e:
                    print(f"    K={k} failed: {e}")

        results.sort(key=lambda x: x.k)
        main_time = time.time() - main_start
        print(f"  [Phase 1] Completed in {main_time:.1f}s")

        # ===== Phase 2: Gap Statistic =====
        gap_time = 0.0  # Fix: Initialize to avoid NameError when compute_gap=False
        
        if self.compute_gap:
            print(f"\n  [Phase 2] Gap Statistic (n_refs={self.gap_n_refs}, step={self.gap_k_step})...")
            gap_start = time.time()
            
            # Sample K values
            gap_k_values = list(range(self.k_min, self.k_max + 1, self.gap_k_step))
            if self.k_max not in gap_k_values:
                gap_k_values.append(self.k_max)
            
            print(f"    Gap K points: {len(gap_k_values)} (from {gap_k_values[0]} to {gap_k_values[-1]})")
            
            gaps, gap_stds = compute_gap_statistic_with_progress(
                X, gap_k_values,
                n_refs=self.gap_n_refs,
                random_state=42,
                modality_name=name,
                max_gap_dims=self.gap_max_dims
            )
            self.gap_results[name] = (np.array(gap_k_values), gaps, gap_stds)
            
            gap_time = time.time() - gap_start
            print(f"  [Phase 2] Completed in {gap_time:.1f}s")

        # ===== Phase 3: Find elbow points =====
        self._find_elbow_points(name, results)
        
        # ===== Phase 4:  Compute recommendation =====
        self._compute_recommendation(name)

        total_time = main_time + gap_time
        print(f"\n  {name} TOTAL: {total_time:.1f}s ({total_time/60:.1f} min)")

        return results

    def _find_elbow_points(self, name: str, results: List[ClusterMetrics]):
        """Find elbow points using multiple methods with MIN_K constraint and boundary fallback"""
        k_values = np.array([r.k for r in results])
        inertia = np.array([r.inertia for r in results])
        sil = np.array([r.silhouette for r in results])
        ch = np.array([r.calinski_harabasz for r in results])
        db = np.array([r.davies_bouldin for r in results])

        MIN_K = self.min_k
        self.elbow_points[name] = {}

        # 1) Kneedle (inertia elbow) - main reference
        elbow_inertia, _ = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
        elbow_inertia = int(max(elbow_inertia, MIN_K))
        self.elbow_points[name]['Kneedle_Inertia'] = elbow_inertia

        # 2) Curvature (max curvature) - fallback to Kneedle if at boundary
        elbow_curv = int(find_elbow_by_curvature(k_values, inertia, min_k=MIN_K))
        if elbow_curv >= self.k_max - 1:
            elbow_curv = elbow_inertia  # Boundary fallback
        self.elbow_points[name]['Curvature_Inertia'] = elbow_curv

        # 3/4/5) Sil / CH / DB - only consider K >= MIN_K
        valid = k_values >= MIN_K
        vk, vsil, vch, vdb = k_values[valid], sil[valid], ch[valid], db[valid]
        self.elbow_points[name]['Max_Silhouette'] = int(vk[np.argmax(vsil)])
        self.elbow_points[name]['Max_CH'] = int(vk[np.argmax(vch)])
        self.elbow_points[name]['Min_DB'] = int(vk[np.argmin(vdb)])

        # 6) Gap - fallback to Kneedle if at boundary
        if name in self.gap_results:
            gap_k, gaps, gap_stds = self.gap_results[name]
            optimal_gap_k = int(find_optimal_k_gap(gap_k, gaps, gap_stds))
            if optimal_gap_k >= self.k_max - 1:
                optimal_gap_k = elbow_inertia  # Boundary fallback
            optimal_gap_k = max(optimal_gap_k, MIN_K)
            self.elbow_points[name]['Gap_Statistic'] = optimal_gap_k

        print(f"\n  Elbow detection results (MIN_K={MIN_K}):")
        for method, k in self.elbow_points[name].items():
            print(f"    {method}: K={k}")

    def _compute_recommendation(self, name:  str):
        """Compute weighted voting recommendation with candidate range"""
        elbows = self.elbow_points[name]
        
        # Weighted voting
        k_scores = {}
        for method, k in elbows.items():
            weight = self.VOTING_WEIGHTS.get(method, 1.0)
            k_scores[k] = k_scores.get(k, 0.0) + weight
        
        # Primary recommendation (highest weighted score)
        recommended_k = max(k_scores.keys(), key=lambda kk: k_scores[kk])
        recommended_score = k_scores[recommended_k]
        
        # Candidate range (top-3 by weighted score, or Kneedle ± 5)
        sorted_candidates = sorted(k_scores.items(), key=lambda x:  -x[1])
        top_candidates = [k for k, _ in sorted_candidates[: 3]]
        
        # Also include Kneedle ± range as candidates
        kneedle_k = elbows.get('Kneedle_Inertia', recommended_k)
        extended_candidates = set(top_candidates)
        for delta in [-5, -3, 0, 3, 5]:
            candidate = kneedle_k + delta
            if self.min_k <= candidate <= self.k_max:
                extended_candidates.add(candidate)
        
        candidate_range = sorted(extended_candidates)
        
        self.recommendations[name] = {
            'recommended_k': recommended_k,
            'weighted_score': recommended_score,
            'k_scores': k_scores,
            'candidate_range': candidate_range,
            'all_methods': elbows.copy(),
        }
        
        print(f"\n  Recommendation:")
        print(f"    ★ Primary: K={recommended_k} (score={recommended_score:.1f})")
        print(f"    Candidates for sensitivity analysis: {candidate_range}")

    def run_exploration(self):
        """Run exploration for all modalities"""
        print("\n" + "=" * 80)
        print(f"Clustering Number Exploration")
        print(f"K range: {self.k_min}-{self.k_max} | MIN_K: {self.min_k} | Workers: {self.n_jobs}")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        total_start = time.time()
        
        for name, vectors in modalities: 
            self.results[name] = self.explore_clustering(vectors, name)

        total_time = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"ALL MODALITIES COMPLETED:  {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}")

        return self.results

    # ==================== Visualization ====================

    def visualize_results(self, output_dir: str = "."):
        """Generate visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_elbow_with_detection(output_dir)
        self._plot_curvature_analysis(output_dir)
        self._plot_silhouette_with_peaks(output_dir)
        self._plot_stability_analysis(output_dir)
        if self.compute_gap:
            self._plot_gap_statistic(output_dir)
        self._plot_comprehensive_view(output_dir)
        self._plot_voting_summary(output_dir)

        print(f"\n✓ Figures saved to:  {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_elbow_with_detection(self, output_dir: str):
        """Elbow curve + automatic elbow detection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph': '#3498DB', 'Gene':  '#27AE60', 'Proj': '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())

            ax.plot(k_values, inertia_norm, color=colors[name], linewidth=2, label='Inertia (normalized)')

            # Mark recommended K
            rec_k = self.recommendations[name]['recommended_k']
            rec_idx = np.where(k_values == rec_k)[0][0]
            ax.axvline(x=rec_k, color='red', linestyle='-', linewidth=2.5, alpha=0.9, label=f'Recommended:  K={rec_k}')
            ax.scatter(rec_k, inertia_norm[rec_idx], color='red', s=200, zorder=5, edgecolor='black', linewidth=2, marker='*')

            # Mark Kneedle
            elbow_k = self.elbow_points[name]['Kneedle_Inertia']
            if elbow_k != rec_k:
                elbow_idx = np.where(k_values == elbow_k)[0][0]
                ax.axvline(x=elbow_k, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Kneedle: K={elbow_k}')
                ax.scatter(elbow_k, inertia_norm[elbow_idx], color='orange', s=100, zorder=5, edgecolor='black')

            # Kneedle difference curve
            _, diff = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
            ax2 = ax.twinx()
            ax2.fill_between(k_values, diff, alpha=0.15, color='gray')
            ax2.set_ylabel('Kneedle Distance', color='gray', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='gray')

            # MIN_K line
            ax.axvline(x=self.min_k, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'MIN_K={self.min_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Normalized Inertia', fontsize=11)
            ax.set_title(f'{name} - Elbow Detection (Rec: K={rec_k})', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_elbow_with_detection.png")

    def _plot_curvature_analysis(self, output_dir:  str):
        """Curvature analysis plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            k_values = np.array([r.k for r in results])
            inertia = np.array([r.inertia for r in results])

            ax1 = axes[0, idx]
            inertia_smooth = gaussian_filter1d(inertia, sigma=1.5)
            first_deriv = np.gradient(inertia_smooth, k_values)
            ax1.plot(k_values, first_deriv, color=colors[name], linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.fill_between(k_values, first_deriv, alpha=0.2, color=colors[name])
            ax1.set_xlabel('K')
            ax1.set_ylabel("First Derivative")
            ax1.set_title(f'{name} - First Derivative', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1, idx]
            curvature = compute_curvature(k_values, inertia)
            ax2.plot(k_values, curvature, color=colors[name], linewidth=2)
            ax2.fill_between(k_values, curvature, alpha=0.2, color=colors[name])
            max_curv_k = self.elbow_points[name]['Curvature_Inertia']
            max_curv_idx = np.where(k_values == max_curv_k)[0][0]
            ax2.axvline(x=max_curv_k, color='red', linestyle='--', alpha=0.8)
            ax2.scatter(max_curv_k, curvature[max_curv_idx], color='red', s=100, zorder=5, edgecolor='black')
            ax2.annotate(f'K={max_curv_k}', xy=(max_curv_k, curvature[max_curv_idx]),
                         xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold', color='red')
            ax2.set_xlabel('K')
            ax2.set_ylabel('Curvature')
            ax2.set_title(f'{name} - Curvature (Max at K={max_curv_k})', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Derivative and Curvature Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_curvature_analysis.png")

    def _plot_silhouette_with_peaks(self, output_dir: str):
        """Silhouette curve + peak detection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph': '#3498DB', 'Gene':  '#27AE60', 'Proj': '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            sil = np.array([r.silhouette for r in results])
            sil_smooth = gaussian_filter1d(sil, sigma=1.0)

            ax.plot(k_values, sil, color=colors[name], linewidth=1.5, alpha=0.5, label='Raw')
            ax.plot(k_values, sil_smooth, color=colors[name], linewidth=2.5, label='Smoothed')
            ax.fill_between(k_values, sil, alpha=0.15, color=colors[name])

            # Only show peaks in valid range (K >= MIN_K)
            valid_mask = k_values >= self.min_k
            peaks, _ = find_peaks(sil_smooth[valid_mask], prominence=0.01, distance=3)
            valid_k = k_values[valid_mask]
            valid_sil = sil[valid_mask]
            
            for peak_idx in peaks[: 5]: 
                ax.scatter(valid_k[peak_idx], valid_sil[peak_idx], color='red', s=80, zorder=5, marker='^', edgecolor='black')
                ax.annotate(f'K={valid_k[peak_idx]}', xy=(valid_k[peak_idx], valid_sil[peak_idx]),
                            xytext=(5, 8), textcoords='offset points', fontsize=9, color='red')

            max_k = self.elbow_points[name]['Max_Silhouette']
            ax.axvline(x=max_k, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label=f'Best: K={max_k}')
            ax.axvline(x=self.min_k, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'MIN_K={self.min_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Silhouette Score', fontsize=11)
            ax.set_title(f'{name} - Silhouette (Best K={max_k} for K≥{self.min_k})', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_silhouette_with_peaks.png")

    def _plot_stability_analysis(self, output_dir: str):
        """Stability analysis showing std across seeds"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            k_values = np.array([r.k for r in results])
            sil = np.array([r.silhouette for r in results])
            sil_std = np.array([r.silhouette_std for r in results])
            inertia = np.array([r.inertia for r in results])
            inertia_std = np.array([r.inertia_std for r in results])

            # Silhouette with error band
            ax1 = axes[0, idx]
            ax1.fill_between(k_values, sil - sil_std, sil + sil_std, alpha=0.3, color=colors[name])
            ax1.plot(k_values, sil, color=colors[name], linewidth=2)
            rec_k = self.recommendations[name]['recommended_k']
            ax1.axvline(x=rec_k, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Rec: K={rec_k}')
            ax1.set_xlabel('K')
            ax1.set_ylabel('Silhouette ± std')
            ax1.set_title(f'{name} - Silhouette Stability ({self.n_seeds} seeds)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Coefficient of variation for inertia
            ax2 = axes[1, idx]
            cv = inertia_std / (inertia + 1e-10)  # Coefficient of variation
            ax2.plot(k_values, cv * 100, color=colors[name], linewidth=2)
            ax2.fill_between(k_values, 0, cv * 100, alpha=0.2, color=colors[name])
            ax2.axvline(x=rec_k, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Rec:  K={rec_k}')
            ax2.set_xlabel('K')
            ax2.set_ylabel('Inertia CV (%)')
            ax2.set_title(f'{name} - Inertia Coefficient of Variation', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Stability Analysis (n_seeds={self.n_seeds})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_stability_analysis.png")

    def _plot_gap_statistic(self, output_dir: str):
        """Gap Statistic plot"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph': '#3498DB', 'Gene': '#27AE60', 'Proj': '#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]
            if name not in self.gap_results:
                ax.text(0.5, 0.5, 'Gap Statistic not computed', ha='center', va='center', transform=ax.transAxes)
                continue

            gap_k, gaps, gap_stds = self.gap_results[name]
            ax.errorbar(gap_k, gaps, yerr=gap_stds, color=colors[name], linewidth=2, capsize=4, marker='o', markersize=6)
            ax.fill_between(gap_k, gaps - gap_stds, gaps + gap_stds, alpha=0.2, color=colors[name])

            optimal_k = self.elbow_points[name].get('Gap_Statistic', gap_k[np.argmax(gaps)])
            ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Gap optimal: K={optimal_k}')
            
            rec_k = self.recommendations[name]['recommended_k']
            if rec_k != optimal_k: 
                ax.axvline(x=rec_k, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Final rec: K={rec_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Gap Statistic', fontsize=11)
            ax.set_title(f'{name} - Gap Statistic', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_gap_statistic.png")

    def _plot_comprehensive_view(self, output_dir: str):
        """Comprehensive view"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}
        modalities = list(self.results.keys())

        for row, name in enumerate(modalities):
            results = self.results[name]
            k_values = np.array([r.k for r in results])
            color = colors[name]
            elbows = self.elbow_points[name]
            rec_k = self.recommendations[name]['recommended_k']

            # Silhouette
            ax1 = axes[row, 0]
            sil = np.array([r.silhouette for r in results])
            ax1.plot(k_values, sil, color=color, linewidth=2)
            ax1.fill_between(k_values, sil, alpha=0.2, color=color)
            ax1.axvline(x=elbows['Max_Silhouette'], color='orange', linestyle='--', linewidth=1.5)
            ax1.axvline(x=rec_k, color='red', linestyle='-', linewidth=2, label=f'Rec={rec_k}')
            ax1.set_title(f'{name} - Silhouette (Best={elbows["Max_Silhouette"]})', fontweight='bold', fontsize=10)
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)

            # CH
            ax2 = axes[row, 1]
            ch = np.array([r.calinski_harabasz for r in results])
            ax2.plot(k_values, ch, color=color, linewidth=2)
            ax2.fill_between(k_values, ch, alpha=0.2, color=color)
            ax2.axvline(x=elbows['Max_CH'], color='orange', linestyle='--', linewidth=1.5)
            ax2.axvline(x=rec_k, color='red', linestyle='-', linewidth=2)
            ax2.set_title(f'{name} - CH (Best={elbows["Max_CH"]})', fontweight='bold', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # DB
            ax3 = axes[row, 2]
            db = np.array([r.davies_bouldin for r in results])
            ax3.plot(k_values, db, color=color, linewidth=2)
            ax3.fill_between(k_values, db, alpha=0.2, color=color)
            ax3.axvline(x=elbows['Min_DB'], color='orange', linestyle='--', linewidth=1.5)
            ax3.axvline(x=rec_k, color='red', linestyle='-', linewidth=2)
            ax3.set_title(f'{name} - DB (Best={elbows["Min_DB"]})', fontweight='bold', fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Inertia
            ax4 = axes[row, 3]
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())
            ax4.plot(k_values, inertia_norm, color=color, linewidth=2)
            ax4.fill_between(k_values, inertia_norm, alpha=0.2, color=color)
            ax4.axvline(x=elbows['Kneedle_Inertia'], color='orange', linestyle='--', linewidth=1.5,
                        label=f'Kneedle={elbows["Kneedle_Inertia"]}')
            ax4.axvline(x=rec_k, color='red', linestyle='-', linewidth=2, label=f'Rec={rec_k}')
            ax4.set_title(f'{name} - Elbow', fontweight='bold', fontsize=10)
            ax4.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Clustering Metrics Analysis', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "6_comprehensive_view.png")

    def _plot_voting_summary(self, output_dir: str):
        """Voting summary plot with weighted scores"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]
            rec = self.recommendations[name]
            k_scores = rec['k_scores']
            
            sorted_ks = sorted(k_scores.keys())
            scores = [k_scores[k] for k in sorted_ks]

            bars = ax.bar(range(len(sorted_ks)), scores, color=colors[name], alpha=0.7)
            
            # Highlight recommended K
            rec_k = rec['recommended_k']
            rec_idx = sorted_ks.index(rec_k)
            bars[rec_idx].set_color('red')
            bars[rec_idx].set_alpha(0.9)

            ax.set_xticks(range(len(sorted_ks)))
            ax.set_xticklabels([str(k) for k in sorted_ks], fontsize=9)
            ax.set_xlabel('Optimal K (by different methods)', fontsize=11)
            ax.set_ylabel('Weighted Score', fontsize=11)

            ax.set_title(f'{name} - Weighted Voting\n★ Recommended: K={rec_k} (score={rec["weighted_score"]:.1f})',
                         fontsize=12, fontweight='bold')

            for i, (k, s) in enumerate(zip(sorted_ks, scores)):
                ax.annotate(f'{s:.1f}', xy=(i, s), ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y')

            # Add method breakdown
            elbows = rec['all_methods']
            method_text = '\n'.join([f'{m}: K={k} (w={self.VOTING_WEIGHTS.get(m, 1.0)})' 
                                     for m, k in elbows.items()])
            ax.text(1.02, 0.5, method_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        self._save_figure(fig, output_dir, "7_voting_summary.png")

    # ==================== Save Results ====================

    def save_results(self, output_dir: str = "."):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics for each modality (with stability info)
        for name, results in self.results.items():
            rows = [{
                'k': r.k,
                'silhouette': r.silhouette,
                                'silhouette_std': r.silhouette_std,
                'calinski_harabasz': r.calinski_harabasz,
                'calinski_harabasz_std': r.calinski_harabasz_std,
                'davies_bouldin':  r.davies_bouldin,
                'davies_bouldin_std': r.davies_bouldin_std,
                'inertia': r.inertia,
                'inertia_std': r.inertia_std,
                'time_seconds': r.time_seconds,
            } for r in results]
            pd.DataFrame(rows).to_csv(f"{output_dir}/clustering_metrics_{name.lower()}.csv", index=False)

        # Save elbow points summary
        elbow_rows = [{'modality': name, **elbows} for name, elbows in self.elbow_points.items()]
        pd.DataFrame(elbow_rows).to_csv(f"{output_dir}/elbow_points_summary.csv", index=False)

        # Save recommendations with weighted voting
        rec_rows = []
        for name, rec in self.recommendations.items():
            row = {
                'modality': name,
                'recommended_k': rec['recommended_k'],
                'weighted_score': rec['weighted_score'],
                'candidate_range': str(rec['candidate_range']),
                'kneedle_k': rec['all_methods'].get('Kneedle_Inertia'),
                'curvature_k': rec['all_methods'].get('Curvature_Inertia'),
                'silhouette_k':  rec['all_methods'].get('Max_Silhouette'),
                'ch_k': rec['all_methods'].get('Max_CH'),
                'db_k': rec['all_methods'].get('Min_DB'),
                'gap_k': rec['all_methods'].get('Gap_Statistic'),
            }
            rec_rows.append(row)
        pd.DataFrame(rec_rows).to_csv(f"{output_dir}/recommended_k.csv", index=False)

        # Save full recommendations as pickle for programmatic access
        with open(f"{output_dir}/recommendations_full.pkl", 'wb') as f:
            pickle.dump({
                'recommendations':  self.recommendations,
                'elbow_points': self.elbow_points,
                'voting_weights': self.VOTING_WEIGHTS,
                'config': {
                    'k_range': (self.k_min, self.k_max),
                    'min_k': self.min_k,
                    'n_seeds': self.n_seeds,
                    'n_jobs': self.n_jobs,
                    'compute_gap': self.compute_gap,
                    'gap_n_refs': self.gap_n_refs,
                    'gap_k_step': self.gap_k_step,
                    'silhouette_sample_size': self.silhouette_sample_size,
                }
            }, f)

        print(f"\n✓ Results saved to:  {output_dir}")

    # ==================== Main Pipeline ====================

    def run_full_pipeline(self, output_dir: str = "./clustering_exploration"):
        """Run full pipeline"""
        print("\n" + "=" * 80)
        print("Clustering Number Pre-experiment (Optimized & Robust Version)")
        print("=" * 80)

        start_time = time.time()

        n = self.load_all_data()
        if n == 0:
            return

        self.run_exploration()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"TOTAL PIPELINE TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}")

    def _print_summary(self):
        """Print summary"""
        print("\n" + "=" * 80)
        print("FINAL RECOMMENDATIONS SUMMARY")
        print("=" * 80)

        print(f"\nVoting Weights Used:")
        for method, weight in self.VOTING_WEIGHTS.items():
            print(f"  {method}: {weight}")

        for name, rec in self.recommendations.items():
            print(f"\n{'─'*60}")
            print(f"【{name}】")
            print(f"{'─'*60}")
            
            print(f"\n  Method Results (MIN_K={self.min_k}):")
            for method, k in rec['all_methods'].items():
                weight = self.VOTING_WEIGHTS.get(method, 1.0)
                print(f"    {method: 25s}: K={k: 3d} (weight={weight})")
            
            print(f"\n  Weighted Scores:")
            for k in sorted(rec['k_scores'].keys()):
                score = rec['k_scores'][k]
                marker = " ★" if k == rec['recommended_k'] else ""
                print(f"    K={k:3d}:  {score:.1f}{marker}")
            
            print(f"\n  ★ PRIMARY RECOMMENDATION: K = {rec['recommended_k']}")
            print(f"  Candidates for sensitivity analysis: {rec['candidate_range']}")

        print("\n" + "=" * 80)
        print("HOW TO USE THESE RESULTS:")
        print("-" * 80)
        print("1.Use 'recommended_k.csv' as input to classification experiments")
        print("2.For sensitivity analysis, try K values in 'candidate_range'")
        print("3.If results are inconsistent, prioritize Kneedle_Inertia")
        print("=" * 80)


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./clustering_exploration"

    with ClusteringExplorer(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.99,
            k_range=(2, 80),
            min_k=5,                       # Biological minimum (K=2,3,4 usually meaningless)
            n_init=10,
            n_seeds=3,                     # Multi-seed for stability
            n_jobs=8,                      # Limit workers
            compute_gap=True,
            gap_n_refs=5,                  # Reduced for speed
            gap_k_step=5,                  # Coarse sampling
            gap_max_dims=20,               # Avoid high-dim uniform problem
            silhouette_sample_size=3000    # Avoid O(n²)
    ) as explorer:
        explorer.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
