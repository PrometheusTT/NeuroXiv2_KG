"""
Clustering Number Pre-experiment: Search for Optimal K (Optimized Version)
===============================================
Performance optimizations:
- Use initializer to avoid repeated X pickle
- Silhouette sampling for large datasets
- Control thread parallelism to avoid oversubscription
- Gap Statistic with progress output

Author:PrometheusTT
Date:2025-01-xx
"""

import os

# ===== 关键：限制内层线程，避免过度并行 =====
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


def _init_worker(X: np.ndarray, n_init:int, random_state: int, sil_sample_size:int):
    """Initialize worker with shared data (called once per worker process)"""
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE, _GLOBAL_SILHOUETTE_SAMPLE_SIZE
    _GLOBAL_X = X
    _GLOBAL_N_INIT = n_init
    _GLOBAL_RANDOM_STATE = random_state
    _GLOBAL_SILHOUETTE_SAMPLE_SIZE = sil_sample_size


def _cluster_single_k(k:int) -> dict:
    """Clustering for a single K value using global variables."""
    global _GLOBAL_X, _GLOBAL_N_INIT, _GLOBAL_RANDOM_STATE, _GLOBAL_SILHOUETTE_SAMPLE_SIZE
    
    X = _GLOBAL_X
    n_init = _GLOBAL_N_INIT
    random_state = _GLOBAL_RANDOM_STATE
    sil_sample_size = _GLOBAL_SILHOUETTE_SAMPLE_SIZE
    
    start = time.time()
    
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    # Silhouette with sampling
    n_samples = X.shape[0]
    if sil_sample_size and n_samples > sil_sample_size:
        sil = silhouette_score(X, labels, sample_size=sil_sample_size, random_state=random_state)
    else:
        sil = silhouette_score(X, labels)
    
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    inertia = kmeans.inertia_
    
    elapsed = time.time() - start
    
    return {
        'k':k,
        'silhouette':sil,
        'calinski_harabasz':ch,
        'davies_bouldin':db,
        'inertia':inertia,
        'time_seconds':elapsed
    }


# ==================== Kneedle Algorithm ====================

def find_elbow_point(k_values:np.ndarray, values:np.ndarray,
                     direction:str = 'decreasing',
                     curve: str = 'convex') -> Tuple[int, np.ndarray]:
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


def compute_curvature(k_values:np.ndarray, values:np.ndarray,
                      smooth_sigma:float = 1.5) -> np.ndarray:
    """Calculate curve curvature"""
    values_smooth = gaussian_filter1d(values, sigma=smooth_sigma)
    first_deriv = np.gradient(values_smooth, k_values)
    second_deriv = np.gradient(first_deriv, k_values)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5
    return curvature


def find_elbow_by_curvature(k_values:np.ndarray, values:np.ndarray,
                            min_k:int = 5) -> int:
    """Find elbow point by maximum curvature"""
    curvature = compute_curvature(k_values, values)
    valid_mask = k_values >= min_k
    valid_k = k_values[valid_mask]
    valid_curv = curvature[valid_mask]
    max_idx = np.argmax(valid_curv)
    return valid_k[max_idx]


# ==================== Gap Statistic (with progress) ====================

def compute_gap_statistic_with_progress(X: np.ndarray, k_values:List[int],
                                         n_refs:int = 5, 
                                         random_state:int = 42,
                                         modality_name:str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gap Statistic with progress output.
    
    Gap(k) = E[log(W_k^*)] - log(W_k)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    gaps = []
    gap_stds = []
    
    total_k = len(k_values)
    start_time = time.time()

    for idx, k in enumerate(k_values):
        k_start = time.time()
        
        # Original data
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(X)
        log_wk = np.log(kmeans.inertia_ + 1e-10)

        # Reference data
        log_wk_refs = []
        for ref_idx in range(n_refs):
            X_ref = np.random.uniform(mins, maxs, (n_samples, n_features))
            kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=random_state)
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
        
        print(f"    Gap[{modality_name}] K={k:3d}:{idx+1}/{total_k} | "
              f"This K:{k_elapsed:.1f}s | Total:{total_elapsed:.1f}s | ETA:{eta:.1f}s")

    return np.array(gaps), np.array(gap_stds)


def find_optimal_k_gap(k_values:np.ndarray, gaps:np.ndarray,
                       gap_stds:np.ndarray) -> int:
    """Find optimal K based on Gap Statistic"""
    for i in range(len(k_values) - 1):
        if gaps[i] >= gaps[i+1] - gap_stds[i+1]:
            return k_values[i]
    return k_values[-1]


@dataclass
class ClusterMetrics:
    """Clustering metrics"""
    k:int
    silhouette:float
    calinski_harabasz:float
    davies_bouldin:float
    inertia:float
    time_seconds: float


class ClusteringExplorer:
    """Clustering Number Explorer (Optimized Version)"""

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

    def __init__(self, uri:str, user: str, password:str,
                 data_dir:str, database:str = "neo4j",
                 search_radius:float = 8.0,
                 pca_variance_threshold:float = 0.95,
                 k_range: Tuple[int, int] = (2, 70),
                 n_init:int = 10,
                 n_jobs:int = 8,
                 compute_gap:bool = True,
                 gap_n_refs: int = 5,
                 gap_k_step:int = 5,
                 silhouette_sample_size:int = 3000):
        """
        Args:
            k_range:K value search range (min, max)
            n_init: KMeans initialization count
            n_jobs:Number of parallel workers (recommend: 8)
            compute_gap:Whether to compute Gap Statistic
            gap_n_refs:Number of reference datasets for Gap (recommend:3-5)
            gap_k_step: Step size for Gap K sampling (recommend:5)
            silhouette_sample_size: Sample size for silhouette score
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.k_min, self.k_max = k_range
        self.n_init = n_init
        self.n_jobs = min(n_jobs, N_CORES)
        self.compute_gap = compute_gap
        self.gap_n_refs = gap_n_refs
        self.gap_k_step = gap_k_step
        self.silhouette_sample_size = silhouette_sample_size

        print(f"Configuration:")
        print(f"  K range: {self.k_min} - {self.k_max}")
        print(f"  n_jobs:{self.n_jobs}")
        print(f"  silhouette_sample_size:{self.silhouette_sample_size}")
        print(f"  compute_gap:{self.compute_gap}")
        if self.compute_gap:
            print(f"  gap_n_refs:{self.gap_n_refs}")
            print(f"  gap_k_step:{self.gap_k_step}")

        # Data
        self.valid_neuron_ids: List[str] = []
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw:Dict[str, np.ndarray] = {}
        self.local_gene_features_raw:Dict[str, np.ndarray] = {}
        self.projection_vectors_raw:Dict[str, np.ndarray] = {}
        self.all_subclasses:List[str] = []
        self.all_target_regions:List[str] = []

        # PCA vectors
        self.morph_vectors: np.ndarray = None
        self.gene_vectors:np.ndarray = None
        self.proj_vectors:np.ndarray = None

        # Results
        self.results:Dict[str, List[ClusterMetrics]] = {}
        self.gap_results:Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.elbow_points:Dict[str, Dict[str, int]] = {}

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
        print(f"  Neurons:{len(self.valid_neuron_ids)}")
        print(f"  Morph:{self.morph_vectors.shape}")
        print(f"  Gene:{self.gene_vectors.shape}")
        print(f"  Proj:{self.proj_vectors.shape}")

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
        print(f"  Projection targets:{len(self.all_target_regions)} brain regions")

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
        print(f"  Valid neurons:{len(self.valid_neuron_ids)}")

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
        gene_raw = gene_raw[:, col_sums > 0]
        self.gene_vectors = self._process_vector(gene_raw, 'Gene')

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw.sum(axis=0)
        proj_raw = proj_raw[:, col_sums > 0]
        self.proj_vectors = self._process_vector(proj_raw, 'Proj', do_log=True)

    def _process_vector(self, X_raw:np.ndarray, name:str,
                        do_log:bool = False) -> np.ndarray:
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
        print(f"  {name}: {original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        return X_pca

    # ==================== Clustering Exploration ====================

    def explore_clustering(self, X:np.ndarray, name:str) -> List[ClusterMetrics]:
        """Explore clustering effects for different K values"""
        print(f"\n{'='*60}")
        print(f"{name} (K={self.k_min} to {self.k_max})")
        print(f"{'='*60}")
        print(f"  Data shape:{X.shape}")
        print(f"  Workers:{self.n_jobs}, Silhouette sample: {self.silhouette_sample_size}")

        k_values = list(range(self.k_min, self.k_max + 1))
        results = []

        # ===== Main clustering loop =====
        print(f"\n  [Phase 1] Main clustering loop ({len(k_values)} K values)...")
        main_start = time.time()

        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_init_worker,
            initargs=(X, self.n_init, 42, self.silhouette_sample_size)
        ) as executor:
            futures = {executor.submit(_cluster_single_k, k):k for k in k_values}

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
                        print(f"    Progress:{completed}/{len(k_values)} | "
                              f"Elapsed:{elapsed:.1f}s | ETA:{eta:.1f}s")
                              
                except Exception as e:
                    print(f"    K={k} failed:{e}")

        results.sort(key=lambda x:x.k)
        main_time = time.time() - main_start
        print(f"  [Phase 1] Completed in {main_time:.1f}s")

        # ===== Gap Statistic =====
        if self.compute_gap:
            print(f"\n  [Phase 2] Gap Statistic (n_refs={self.gap_n_refs}, step={self.gap_k_step})...")
            gap_start = time.time()
            
            # Sample K values
            gap_k_values = list(range(self.k_min, self.k_max + 1, self.gap_k_step))
            if self.k_max not in gap_k_values:
                gap_k_values.append(self.k_max)
            
            print(f"    Gap K points:{len(gap_k_values)} (from {gap_k_values[0]} to {gap_k_values[-1]})")
            
            gaps, gap_stds = compute_gap_statistic_with_progress(
                X, gap_k_values, 
                n_refs=self.gap_n_refs, 
                random_state=42,
                modality_name=name
            )
            self.gap_results[name] = (np.array(gap_k_values), gaps, gap_stds)
            
            gap_time = time.time() - gap_start
            print(f"  [Phase 2] Completed in {gap_time:.1f}s")

        # ===== Find elbow points =====
        self._find_elbow_points(name, results)

        total_time = main_time + (gap_time if self.compute_gap else 0)
        print(f"\n  {name} TOTAL:{total_time:.1f}s ({total_time/60:.1f} min)")

        return results

    def _find_elbow_points(self, name:str, results:List[ClusterMetrics]):
        """Find elbow points using multiple methods"""
        k_values = np.array([r.k for r in results])
        inertia = np.array([r.inertia for r in results])
        sil = np.array([r.silhouette for r in results])
        ch = np.array([r.calinski_harabasz for r in results])
        db = np.array([r.davies_bouldin for r in results])

        self.elbow_points[name] = {}

        # 1.Kneedle - Inertia
        elbow_inertia, _ = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
        self.elbow_points[name]['Kneedle_Inertia'] = elbow_inertia

        # 2.Curvature - Inertia
        elbow_curv = find_elbow_by_curvature(k_values, inertia, min_k=5)
        self.elbow_points[name]['Curvature_Inertia'] = elbow_curv

        # 3.Max Silhouette
        self.elbow_points[name]['Max_Silhouette'] = k_values[np.argmax(sil)]

        # 4.Max CH
        self.elbow_points[name]['Max_CH'] = k_values[np.argmax(ch)]

        # 5.Min DB
        self.elbow_points[name]['Min_DB'] = k_values[np.argmin(db)]

        # 6.Gap Statistic
        if name in self.gap_results:
            gap_k, gaps, gap_stds = self.gap_results[name]
            optimal_gap_k = find_optimal_k_gap(gap_k, gaps, gap_stds)
            self.elbow_points[name]['Gap_Statistic'] = optimal_gap_k

        print(f"\n  Elbow detection results:")
        for method, k in self.elbow_points[name].items():
            print(f"    {method}:K={k}")

    def run_exploration(self):
        """Run exploration for all modalities"""
        print("\n" + "=" * 80)
        print(f"Clustering Number Exploration")
        print(f"K range:{self.k_min}-{self.k_max} | Workers:{self.n_jobs}")
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
        print(f"ALL MODALITIES COMPLETED: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}")

        return self.results

    # ==================== Visualization ====================

    def visualize_results(self, output_dir:str = "."):
        """Generate visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_elbow_with_detection(output_dir)
        self._plot_curvature_analysis(output_dir)
        self._plot_silhouette_with_peaks(output_dir)
        if self.compute_gap:
            self._plot_gap_statistic(output_dir)
        self._plot_comprehensive_view(output_dir)
        self._plot_voting_summary(output_dir)

        print(f"\n✓ Figures saved to: {output_dir}")

    def _save_figure(self, fig, output_dir:str, filename:str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_elbow_with_detection(self, output_dir:str):
        """Elbow curve + automatic elbow detection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph':'#3498DB', 'Gene': '#27AE60', 'Proj':'#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())

            ax.plot(k_values, inertia_norm, color=colors[name], linewidth=2, label='Inertia (normalized)')

            elbow_k = self.elbow_points[name]['Kneedle_Inertia']
            elbow_idx = np.where(k_values == elbow_k)[0][0]
            ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Kneedle: K={elbow_k}')
            ax.scatter(elbow_k, inertia_norm[elbow_idx], color='red', s=150, zorder=5, edgecolor='black', linewidth=2)

            curv_k = self.elbow_points[name]['Curvature_Inertia']
            if curv_k != elbow_k:
                curv_idx = np.where(k_values == curv_k)[0][0]
                ax.axvline(x=curv_k, color='orange', linestyle=':', linewidth=2, alpha=0.8, label=f'Curvature: K={curv_k}')
                ax.scatter(curv_k, inertia_norm[curv_idx], color='orange', s=100, zorder=5, marker='s', edgecolor='black')

            _, diff = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
            ax2 = ax.twinx()
            ax2.fill_between(k_values, diff, alpha=0.2, color='gray')
            ax2.set_ylabel('Kneedle Distance', color='gray', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='gray')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Normalized Inertia', fontsize=11)
            ax.set_title(f'{name} - Elbow Detection', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_elbow_with_detection.png")

    def _plot_curvature_analysis(self, output_dir: str):
        """Curvature analysis plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = {'Morph': '#3498DB', 'Gene':'#27AE60', 'Proj': '#E74C3C'}

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

    def _plot_silhouette_with_peaks(self, output_dir:str):
        """Silhouette curve + peak detection"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph':'#3498DB', 'Gene': '#27AE60', 'Proj':'#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            sil = np.array([r.silhouette for r in results])
            sil_smooth = gaussian_filter1d(sil, sigma=1.0)

            ax.plot(k_values, sil, color=colors[name], linewidth=1.5, alpha=0.5, label='Raw')
            ax.plot(k_values, sil_smooth, color=colors[name], linewidth=2.5, label='Smoothed')
            ax.fill_between(k_values, sil, alpha=0.15, color=colors[name])

            peaks, _ = find_peaks(sil_smooth, prominence=0.01, distance=3)
            for peak_idx in peaks[:5]:
                ax.scatter(k_values[peak_idx], sil[peak_idx], color='red', s=80, zorder=5, marker='^', edgecolor='black')
                ax.annotate(f'K={k_values[peak_idx]}', xy=(k_values[peak_idx], sil[peak_idx]),
                            xytext=(5, 8), textcoords='offset points', fontsize=9, color='red')

            max_k = self.elbow_points[name]['Max_Silhouette']
            ax.axvline(x=max_k, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label=f'Best: K={max_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Silhouette Score', fontsize=11)
            ax.set_title(f'{name} - Silhouette (Best K={max_k})', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_silhouette_with_peaks.png")

    def _plot_gap_statistic(self, output_dir:str):
        """Gap Statistic plot"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {'Morph':'#3498DB', 'Gene': '#27AE60', 'Proj':'#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]
            if name not in self.gap_results:
                ax.text(0.5, 0.5, 'Gap Statistic not computed', ha='center', va='center', transform=ax.transAxes)
                continue

            gap_k, gaps, gap_stds = self.gap_results[name]
            ax.errorbar(gap_k, gaps, yerr=gap_stds, color=colors[name], linewidth=2, capsize=4, marker='o', markersize=6)
            ax.fill_between(gap_k, gaps - gap_stds, gaps + gap_stds, alpha=0.2, color=colors[name])

            optimal_k = self.elbow_points[name].get('Gap_Statistic', gap_k[np.argmax(gaps)])
            ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Optimal:K={optimal_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Gap Statistic', fontsize=11)
            ax.set_title(f'{name} - Gap Statistic (Optimal K={optimal_k})', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_gap_statistic.png")

    def _plot_comprehensive_view(self, output_dir:str):
        """Comprehensive view"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        colors = {'Morph': '#3498DB', 'Gene':'#27AE60', 'Proj': '#E74C3C'}
        modalities = list(self.results.keys())

        for row, name in enumerate(modalities):
            results = self.results[name]
            k_values = np.array([r.k for r in results])
            color = colors[name]
            elbows = self.elbow_points[name]

            # Silhouette
            ax1 = axes[row, 0]
            sil = np.array([r.silhouette for r in results])
            ax1.plot(k_values, sil, color=color, linewidth=2)
            ax1.fill_between(k_values, sil, alpha=0.2, color=color)
            ax1.axvline(x=elbows['Max_Silhouette'], color='red', linestyle='--', linewidth=1.5)
            ax1.set_title(f'{name} - Silhouette (K={elbows["Max_Silhouette"]})', fontweight='bold', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # CH
            ax2 = axes[row, 1]
            ch = np.array([r.calinski_harabasz for r in results])
            ax2.plot(k_values, ch, color=color, linewidth=2)
            ax2.fill_between(k_values, ch, alpha=0.2, color=color)
            ax2.axvline(x=elbows['Max_CH'], color='red', linestyle='--', linewidth=1.5)
            ax2.set_title(f'{name} - CH (K={elbows["Max_CH"]})', fontweight='bold', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # DB
            ax3 = axes[row, 2]
            db = np.array([r.davies_bouldin for r in results])
            ax3.plot(k_values, db, color=color, linewidth=2)
            ax3.fill_between(k_values, db, alpha=0.2, color=color)
            ax3.axvline(x=elbows['Min_DB'], color='red', linestyle='--', linewidth=1.5)
            ax3.set_title(f'{name} - DB (K={elbows["Min_DB"]})', fontweight='bold', fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Inertia
            ax4 = axes[row, 3]
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())
            ax4.plot(k_values, inertia_norm, color=color, linewidth=2)
            ax4.fill_between(k_values, inertia_norm, alpha=0.2, color=color)
            ax4.axvline(x=elbows['Kneedle_Inertia'], color='red', linestyle='--', linewidth=1.5, 
                        label=f'Kneedle={elbows["Kneedle_Inertia"]}')
            ax4.axvline(x=elbows['Curvature_Inertia'], color='orange', linestyle=':', linewidth=1.5, 
                        label=f'Curvature={elbows["Curvature_Inertia"]}')
            ax4.set_title(f'{name} - Elbow', fontweight='bold', fontsize=10)
            ax4.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Clustering Metrics Analysis', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_comprehensive_view.png")

    def _plot_voting_summary(self, output_dir:str):
        """Voting summary plot"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = {'Morph': '#3498DB', 'Gene':'#27AE60', 'Proj': '#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]
            elbows = self.elbow_points[name]

            k_votes = {}
            method_names = []
            method_ks = []

            for method, k in elbows.items():
                method_names.append(method)
                method_ks.append(k)
                k_votes[k] = k_votes.get(k, 0) + 1

            sorted_ks = sorted(k_votes.keys())
            votes = [k_votes[k] for k in sorted_ks]

            ax.bar(range(len(sorted_ks)), votes, color=colors[name], alpha=0.7)
            ax.set_xticks(range(len(sorted_ks)))
            ax.set_xticklabels([str(k) for k in sorted_ks], fontsize=9)
            ax.set_xlabel('Optimal K (by different methods)', fontsize=11)
            ax.set_ylabel('Number of Votes', fontsize=11)

            best_k = sorted_ks[np.argmax(votes)]
            max_votes = max(votes)
            ax.set_title(f'{name} - Voting (Recommended: K={best_k}, {max_votes} votes)', fontsize=12, fontweight='bold')

            for i, v in enumerate(votes):
                ax.annotate(str(v), xy=(i, v), ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y')

            method_text = '\n'.join([f'{m}:K={k}' for m, k in zip(method_names, method_ks)])
            ax.text(1.02, 0.5, method_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        self._save_figure(fig, output_dir, "6_voting_summary.png")

    # ==================== Save Results ====================

    def save_results(self, output_dir:str = "."):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)

        for name, results in self.results.items():
            rows = [{'k':r.k, 'silhouette':r.silhouette, 'calinski_harabasz':r.calinski_harabasz,
                     'davies_bouldin':r.davies_bouldin, 'inertia':r.inertia, 'time_seconds':r.time_seconds}
                    for r in results]
            pd.DataFrame(rows).to_csv(f"{output_dir}/clustering_metrics_{name.lower()}.csv", index=False)

        elbow_rows = [{'modality':name, **elbows} for name, elbows in self.elbow_points.items()]
        pd.DataFrame(elbow_rows).to_csv(f"{output_dir}/elbow_points_summary.csv", index=False)

        recommendations = []
        for name, elbows in self.elbow_points.items():
            k_votes = {}
            for method, k in elbows.items():
                k_votes[k] = k_votes.get(k, 0) + 1
            recommended_k = max(k_votes.keys(), key=lambda x:k_votes[x])
            recommendations.append({
                'modality':name, 'recommended_k':recommended_k, 'vote_count':k_votes[recommended_k],
                'total_methods':len(elbows), 'silhouette_k':elbows.get('Max_Silhouette'),
                'kneedle_k': elbows.get('Kneedle_Inertia'), 'curvature_k':elbows.get('Curvature_Inertia'),
            })
        pd.DataFrame(recommendations).to_csv(f"{output_dir}/recommended_k.csv", index=False)

        print(f"\n✓ Results saved to: {output_dir}")

    # ==================== Main Pipeline ====================

    def run_full_pipeline(self, output_dir:str = "./clustering_exploration"):
        """Run full pipeline"""
        print("\n" + "=" * 80)
        print("Clustering Number Pre-experiment (Optimized Version)")
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
        print(f"TOTAL PIPELINE TIME:{total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}")

    def _print_summary(self):
        """Print summary"""
        print("\n" + "=" * 80)
        print("Optimal K Recommendation Summary")
        print("=" * 80)

        for name, elbows in self.elbow_points.items():
            print(f"\n【{name}】")
            k_votes = {}
            for method, k in elbows.items():
                k_votes[k] = k_votes.get(k, 0) + 1
                print(f"  {method:25s}:K={k}")

            print(f"\n  Voting results:")
            for k in sorted(k_votes.keys()):
                print(f"    K={k:3d}:{k_votes[k]} votes")

            recommended_k = max(k_votes.keys(), key=lambda x:k_votes[x])
            print(f"\n  ★ Recommended K: {recommended_k}")


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
            search_radius=8.0,
            pca_variance_threshold=0.99,
            k_range=(2, 70),
            n_init=10,
            n_jobs=8,                      # 限制 worker 数
            compute_gap=True,
            gap_n_refs=5,                  # 减少 Gap 参考次数 (原来10)
            gap_k_step=5,                  # 增大 Gap K采样步长 (原来自动~3)
            silhouette_sample_size=3000    # Silhouette 采样
    ) as explorer:
        explorer.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
