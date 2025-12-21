"""
Task 1:Cluster Label Prediction Experiment
===============================================
Core idea:
1.Cluster each target modality independently → get labels
2.Use other modalities to predict the labels (classification task)
3.Evaluate accuracy

Experiment flow:
1.Use fixed K values from clustering exploration OR search for optimal K
2. Tune RF classifier for each prediction task
3.Compare single-modal vs multi-modal prediction accuracy

Author:PrometheusTT
Date:2025-01-xx
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import time
import multiprocessing
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    silhouette_score, confusion_matrix
)
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Neo4j
import neo4j

# Settings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()
print(f"Detected {N_CORES} CPU cores")


# Brain region mapping (target region -> major region)
MAJOR_BRAIN_REGIONS = {
    'Cortex':['MOp', 'MOs', 'SSp', 'SSs', 'ACA', 'PL', 'ILA', 'ORB', 'AI', 'RSP',
               'PTLp', 'VIS', 'AUD', 'TEa', 'PERI', 'ECT', 'FRP', 'GU', 'VISC',
               'MOp-bfd', 'MOp-ll', 'MOp-m', 'MOp-oro', 'MOp-ul', 'MOs-fef',
               'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un',
               'VISa', 'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor',
               'ACAd', 'ACAv', 'ORBl', 'ORBm', 'ORBvl', 'RSPagl', 'RSPd', 'RSPv'],
    'Thalamus':['VAL', 'VM', 'VPL', 'VPM', 'VPMpc', 'VPLpc', 'PoT', 'Po',
                 'LP', 'LD', 'MD', 'CM', 'PCN', 'CL', 'PF', 'RT', 'ZI',
                 'LGd', 'LGv', 'MG', 'MGd', 'MGm', 'MGv', 'PP', 'PIL',
                 'AD', 'AV', 'AM', 'IAM', 'IAD', 'RE', 'Xi', 'PVT', 'PT',
                 'SMT', 'IMD', 'RH', 'SPF', 'SPFm', 'SPFp', 'SPA', 'SubG',
                 'TH', 'DORpm', 'DORsm', 'EPI', 'ATN', 'MED', 'MTN', 'ILM', 'LAT', 'VENT'],
    'Striatum':['CP', 'ACB', 'FS', 'LSX', 'sAMY', 'CEA', 'MEA', 'AAA', 'BA', 'LA', 'BLA', 'BMA',
                 'OT', 'LSc', 'LSr', 'LSv', 'SF', 'SH', 'GPe', 'GPi', 'MS', 'NDB', 'TRS', 'BST',
                 'BAC', 'SI', 'MA', 'STR', 'STRd', 'STRv', 'PAL', 'PALc', 'PALd', 'PALm', 'PALv'],
    'Hippocampus':['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'ProS', 'PRE', 'POST', 'PAR', 'APr',
                    'ENT', 'ENTl', 'ENTm', 'ENTmv', 'HIP', 'HPF', 'FC', 'IG', 'RHP'],
    'Midbrain':['SC', 'IC', 'SCs', 'SCm', 'SCig', 'SCiw', 'SCdg', 'SCdw', 'SCop', 'SCsg', 'SCzo',
                 'SNr', 'SNc', 'VTA', 'RR', 'MRN', 'PAG', 'PRT', 'APN', 'NOT', 'NPC', 'OP', 'PPT',
                 'CUN', 'RN', 'III', 'EW', 'MA3', 'IV', 'VTN', 'AT', 'LT', 'DT', 'MT', 'PPN',
                 'RAmb', 'IF', 'IPN', 'RL', 'CLI', 'DR', 'NB', 'PBG', 'SAG', 'MEV', 'SUT',
                 'PRC', 'INC', 'ND', 'Su3', 'PrEW', 'InCo', 'MB', 'MBsen', 'MBmot', 'MBsta'],
    'Hypothalamus':['LHA', 'PVH', 'VMH', 'DMH', 'ARH', 'AHN', 'PH', 'TU', 'ZI',
                     'LPO', 'MPO', 'VLPO', 'MPN', 'PS', 'PVp', 'PVi', 'PVa', 'AVP', 'AVPV',
                     'SCH', 'SFO', 'MEPO', 'SBPV', 'OVLT', 'PeF', 'PMd', 'PMv', 'MM', 'MMp',
                     'SUM', 'TMd', 'TMv', 'LM', 'ME', 'STN', 'HY', 'PVZ', 'PeVN', 'MBO'],
    'Cerebellum':['CB', 'CBX', 'CBN', 'VERM', 'HEM', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL',
                   'CUL', 'DEC', 'FOTU', 'PYR', 'UVU', 'NOD', 'CENT', 'IP', 'FN', 'DN', 'VeCB'],
    'Brainstem':['MY', 'P', 'NTS', 'AP', 'DMX', 'XII', 'NI', 'NR', 'GRN', 'LRN', 'MARN',
                  'MDRNd', 'MDRNv', 'PARN', 'PGRNd', 'PGRNl', 'PRNr', 'PRNc', 'IO', 'LIN',
                  'ROB', 'RM', 'RMg', 'RPa', 'RO', 'B', 'VNC', 'CN', 'DCN', 'VCO', 'DCO',
                  'PB', 'KF', 'SOC', 'POR', 'TRN', 'V', 'VI', 'VII', 'AMB', 'MV', 'LAV',
                  'NLL', 'PSV', 'Acs5', 'PC5', 'I5', 'SPV', 'SPVI', 'SPVO', 'SPVC', 'PG',
                  'PRNc', 'SG', 'SLC', 'SLD', 'LC', 'LDT', 'NI', 'PBl', 'PBm', 'PBlc', 'PBld'],
    'Olfactory':['MOB', 'AOB', 'AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR',
                  'OLF', 'AONd', 'AONe', 'AONl', 'AONm', 'AONpv'],
}


def load_recommended_k(clustering_output_dir:str) -> Dict[str, int]:
    """
    Load recommended K values from clustering exploration experiment results.
    
    Args:
        clustering_output_dir:Directory containing clustering exploration results
        
    Returns:
        Dict mapping modality names to recommended K values
    """
    csv_path = os.path.join(clustering_output_dir, "recommended_k.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Recommended K file not found:{csv_path}")
    
    df = pd.read_csv(csv_path)
    
    fixed_k = {}
    for _, row in df.iterrows():
        fixed_k[row['modality']] = int(row['recommended_k'])
    
    print(f"Loaded recommended K values from {csv_path}:")
    for modality, k in fixed_k.items():
        print(f"  {modality}:K={k}")
    
    return fixed_k


@dataclass
class ClusteringInfo:
    """Clustering information"""
    modality:str
    n_clusters:int
    silhouette:float
    labels:np.ndarray
    cluster_sizes:np.ndarray
    kmeans_model:KMeans = None


@dataclass
class ClassificationResult:
    """Classification result"""
    task_name:str
    input_modality:str
    target_modality:str
    n_clusters:int
    # Tuning results
    best_params:Dict
    best_cv_score:float
    tuning_time:float
    # Train set evaluation
    train_accuracy:float
    train_f1_macro:float
    train_f1_weighted:float
    # Test set evaluation
    test_accuracy:float
    test_f1_macro:float
    test_f1_weighted:float
    # Overfitting
    overfit_gap:float
    # Confusion matrix (test set)
    confusion_matrix:np.ndarray
    # Per-sample predictions for distribution analysis
    train_predictions:np.ndarray = None
    train_true_labels:np.ndarray = None
    test_predictions:np.ndarray = None
    test_true_labels:np.ndarray = None
    # Per-sample correctness (for boxplot)
    train_correct:np.ndarray = None
    test_correct:np.ndarray = None


@dataclass
class VectorInfo:
    name:str
    original_dims:int
    pca_dims:int
    variance_explained:float


class ClusterClassificationExperiment:
    """Cluster Label Classification Experiment"""

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

    @staticmethod
    def get_param_distributions():
        """Get parameter distribution (continuous range)"""
        from scipy.stats import randint, uniform

        common_params = {
            'n_estimators':randint(50, 501),
            'max_depth':randint(2, 7),
            'min_samples_split':randint(2, 51),
            'min_samples_leaf':randint(1, 21),
            'max_features':uniform(0.1, 0.9),
            'class_weight':['balanced', 'balanced_subsample', None],
        }

        return [
            {
                'bootstrap':[True],
                'max_samples':uniform(0.5, 0.5),
                **common_params
            },
            {
                'bootstrap':[False],
                **common_params
            }
        ]

    def __init__(self, uri:str, user:str, password:str,
                 data_dir:str, database:str = "neo4j",
                 search_radius:float = 8.0,
                 pca_variance_threshold:float = 0.95,
                 test_ratio:float = 0.2,
                 k_candidates:List[int] = None,
                 fixed_k:Dict[str, int] = None,
                 clustering_results_dir:str = None,
                 coarse_sample_ratio:float = 0.1,
                 coarse_n_iter:int = 500,
                 fine_n_iter:int = 100,
                 cv_folds:int = 5,
                 n_jobs:int = -1):
        """
        Parameters:
            uri:Neo4j connection URI
            user:Neo4j username
            password:Neo4j password
            data_dir:Data directory path
            database:Neo4j database name
            search_radius:Search radius for local gene environment
            pca_variance_threshold:PCA variance threshold
            test_ratio:Test set ratio
            k_candidates:Candidate cluster numbers for search (ignored if fixed_k provided)
            fixed_k:Dict specifying fixed K for each modality, e.g.,
                     {'Morph':15, 'Gene':20, 'Proj':10}
                     If provided, will skip K search and use these values directly
            clustering_results_dir:Path to clustering exploration results directory
                                   If provided, will load recommended K from there
            coarse_sample_ratio:Data ratio for coarse search
            coarse_n_iter:Coarse search iterations
            fine_n_iter:Fine search iterations
            cv_folds:Cross-validation folds
            n_jobs:Number of parallel jobs (-1 for all cores)
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.k_candidates = k_candidates or [5, 10, 15, 20, 30, 50]
        
        # Handle fixed_k:either from parameter or from clustering results
        if fixed_k is not None:
            self.fixed_k = fixed_k
        elif clustering_results_dir is not None:
            self.fixed_k = load_recommended_k(clustering_results_dir)
        else:
            self.fixed_k = {}
        
        self.coarse_sample_ratio = coarse_sample_ratio
        self.coarse_n_iter = coarse_n_iter
        self.fine_n_iter = fine_n_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs if n_jobs > 0 else N_CORES

        # Parallel strategy
        self.coarse_cv_jobs = max(4, self.n_jobs // 16)
        self.coarse_rf_jobs = max(8, self.n_jobs // self.coarse_cv_jobs)
        self.fine_cv_jobs = 2
        self.fine_rf_jobs = max(8, self.n_jobs // 2)

        print(f"Parallel strategy:")
        print(f"  Coarse search:CV parallel={self.coarse_cv_jobs}, RF parallel={self.coarse_rf_jobs}")
        print(f"  Fine search:CV parallel={self.fine_cv_jobs}, RF parallel={self.fine_rf_jobs}")

        # Data
        self.valid_neuron_ids:List[str] = []
        self.axon_features_raw:Dict[str, np.ndarray] = {}
        self.dendrite_features_raw:Dict[str, np.ndarray] = {}
        self.local_gene_features_raw:Dict[str, np.ndarray] = {}
        self.projection_vectors_raw:Dict[str, np.ndarray] = {}
        self.all_subclasses:List[str] = []
        self.all_target_regions:List[str] = []

        # Neuron soma locations for brain region analysis
        self.neuron_soma_regions:Dict[str, str] = {}

        # PCA vectors
        self.morph_vectors:np.ndarray = None
        self.gene_vectors:np.ndarray = None
        self.proj_vectors:np.ndarray = None

        # Train/Test indices
        self.train_idx:np.ndarray = None
        self.test_idx:np.ndarray = None

        # Clustering results
        self.clustering_info:Dict[str, ClusteringInfo] = {}

        # Classification results
        self.classification_results:Dict[str, ClassificationResult] = {}

        # Trained models storage
        self.trained_models:Dict[str, RandomForestClassifier] = {}

        # Preprocessing objects for inference
        self.preprocessors:Dict[str, Dict] = {}

        self.vector_info:Dict[str, VectorInfo] = {}

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
        self._split_train_test()
        self._process_all_vectors()

        print(f"\n✓ Data loading complete:")
        print(f"  Total neurons:{len(self.valid_neuron_ids)}")
        print(f"  Train set:{len(self.train_idx)}, Test set:{len(self.test_idx)}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found:{cache_file}")

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

    def _get_major_brain_region(self, region:str) -> str:
        """Map a brain region to its major category"""
        for major, regions in MAJOR_BRAIN_REGIONS.items():
            if region in regions:
                return major
        # Check if region starts with any known prefix
        for major, regions in MAJOR_BRAIN_REGIONS.items():
            for r in regions:
                if region.startswith(r) or r.startswith(region):
                    return major
        return 'Other'

    def _load_all_neuron_features(self):
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n:Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        OPTIONAL MATCH (n)-[:SOMA_IN]->(r:Subregion)
        RETURN n.neuron_id AS neuron_id, r.acronym AS soma_region, {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id:$neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            records = list(result)

            for record in records:
                neuron_id = record['neuron_id']

                # Store soma region
                soma_region = record.get('soma_region', '')
                if soma_region:
                    self.neuron_soma_regions[neuron_id] = soma_region

                axon_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.AXONAL_FEATURES]
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                dend_feats = [float(record[f]) if record[f] is not None else 0.0
                              for f in self.DENDRITIC_FEATURES]
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                proj_result = session.run(proj_query, neuron_id=neuron_id)
                proj_dict = {r['target']:r['weight'] for r in proj_result
                             if r['target'] and r['weight']}
                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]
                    self.projection_vectors_raw[neuron_id] = proj_vector

        print(f"  Loaded {len(self.axon_features_raw)} neurons")
        print(f"  Neurons with soma region info:{len(self.neuron_soma_regions)}")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  Valid neurons:{len(self.valid_neuron_ids)}")

    def _split_train_test(self):
        """Split train/test sets"""
        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]

        print(f"  Train set:{len(self.train_idx)}, Test set:{len(self.test_idx)}")

    def _process_all_vectors(self):
        """Process vectors"""
        print("\nProcessing vectors...")
        neurons = self.valid_neuron_ids

        # Morphology
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_info, morph_preprocessor = self._process_vector(
            morph_raw, 'Morph', do_log=True, return_preprocessor=True)
        self.vector_info['morph'] = morph_info
        self.preprocessors['morph'] = morph_preprocessor

        # Molecular
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        gene_raw = gene_raw[:, valid_cols]
        self.gene_vectors, gene_info, gene_preprocessor = self._process_vector(
            gene_raw, 'Gene', return_preprocessor=True)
        self.vector_info['gene'] = gene_info
        self.preprocessors['gene'] = gene_preprocessor
        self.preprocessors['gene']['valid_cols'] = valid_cols

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        proj_raw = proj_raw[:, valid_cols]
        self.proj_vectors, proj_info, proj_preprocessor = self._process_vector(
            proj_raw, 'Proj', do_log=True, return_preprocessor=True)
        self.vector_info['proj'] = proj_info
        self.preprocessors['proj'] = proj_preprocessor
        self.preprocessors['proj']['valid_cols'] = valid_cols

    def _process_vector(self, X_raw:np.ndarray, name:str,
                        do_log:bool = False,
                        return_preprocessor:bool = False) -> Tuple[np.ndarray, VectorInfo, Optional[Dict]]:
        original_dims = X_raw.shape[1]

        if do_log:
            X = np.log1p(X_raw)
        else:
            X = X_raw

        # Fit only on train set
        scaler = StandardScaler()
        scaler.fit(X[self.train_idx])
        X_scaled = scaler.transform(X)

        # PCA fit only on train set
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(X_scaled[self.train_idx])
        X_pca = pca.transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  {name}:{original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        if return_preprocessor:
            preprocessor = {
                'scaler':scaler,
                'pca':pca,
                'do_log':do_log
            }
            return X_pca, VectorInfo(name, original_dims, X_pca.shape[1], variance), preprocessor

        return X_pca, VectorInfo(name, original_dims, X_pca.shape[1], variance)

    # ==================== Clustering ====================

    def find_optimal_clustering(self):
        """Find optimal cluster number for each modality (or use fixed K)"""
        print("\n" + "=" * 80)
        
        # Check if using fixed K values
        if self.fixed_k:
            print("Using Fixed K Values")
            print(f"  Specified K values:{self.fixed_k}")
        else:
            print("Searching for Optimal Cluster Number K")
            print(f"  Candidate K values:{self.k_candidates}")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        for name, vectors in modalities:
            print(f"\n--- {name} ---")

            # Check if fixed K is specified for this modality
            if name in self.fixed_k:
                k = self.fixed_k[name]
                print(f"  Using fixed K={k}")
                
                # Fit KMeans with fixed K
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                
                # Calculate silhouette on train set only
                sil = silhouette_score(vectors[self.train_idx], labels[self.train_idx])
                print(f"  Silhouette Score:{sil:.4f}")
                
            else:
                # Search for optimal K
                print(f"  Searching among K candidates:{self.k_candidates}")
                k, sil, labels, kmeans = self._search_best_k(vectors, name)
                print(f"  Found optimal K={k}, Silhouette={sil:.4f}")

            cluster_sizes = np.bincount(labels)

            self.clustering_info[name] = ClusteringInfo(
                modality=name,
                n_clusters=k,
                silhouette=sil,
                labels=labels,
                cluster_sizes=cluster_sizes,
                kmeans_model=kmeans
            )

            print(f"  Cluster sizes:min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
                  f"mean={cluster_sizes.mean():.1f}")

    def _search_best_k(self, X:np.ndarray, name:str) -> Tuple[int, float, np.ndarray, KMeans]:
        """Search for optimal K value"""
        X_train = X[self.train_idx]

        best_k = self.k_candidates[0]
        best_sil = -1
        best_kmeans = None

        for k in self.k_candidates:
            sils = []
            for seed in range(5):
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                labels = kmeans.fit_predict(X_train)
                sil = silhouette_score(X_train, labels)
                sils.append(sil)

            avg_sil = np.mean(sils)
            print(f"    K={k}:Silhouette={avg_sil:.4f}")

            if avg_sil > best_sil:
                best_sil = avg_sil
                best_k = k

        # Use optimal K on all data
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        all_labels = kmeans.fit_predict(X)

        return best_k, best_sil, all_labels, kmeans

    # ==================== Classification ====================

    def _create_fine_param_dist(self, coarse_best:Dict) -> List[Dict]:
        """Create fine search parameter space based on coarse search results"""
        from scipy.stats import randint, uniform

        best_n = coarse_best.get('n_estimators', 200)
        n_low = max(50, best_n - 100)
        n_high = min(600, best_n + 100)

        best_depth = coarse_best.get('max_depth', 4)
        depth_low = max(2, best_depth - 1)
        depth_high = min(7, best_depth + 2)

        best_split = coarse_best.get('min_samples_split', 10)
        split_low = max(2, int(best_split * 0.5))
        split_high = min(100, int(best_split * 2) + 1)

        best_leaf = coarse_best.get('min_samples_leaf', 4)
        leaf_low = max(1, int(best_leaf * 0.5))
        leaf_high = min(50, int(best_leaf * 2) + 1)

        best_feat = coarse_best.get('max_features', 0.5)
        feat_low = max(0.1, best_feat - 0.2)
        feat_high = min(1.0, best_feat + 0.2)

        common_params = {
            'n_estimators':randint(n_low, n_high + 1),
            'max_depth':randint(depth_low, depth_high),
            'min_samples_split':randint(split_low, split_high),
            'min_samples_leaf':randint(leaf_low, leaf_high),
            'max_features':uniform(feat_low, feat_high - feat_low),
            'class_weight':[coarse_best.get('class_weight', 'balanced')],
        }

        best_bootstrap = coarse_best.get('bootstrap', True)

        if best_bootstrap:
            best_samples = coarse_best.get('max_samples', 0.8)
            if best_samples is None:
                best_samples = 1.0
            samples_low = max(0.5, best_samples - 0.15)
            samples_high = min(1.0, best_samples + 0.15)

            return [{
                'bootstrap':[True],
                'max_samples':uniform(samples_low, samples_high - samples_low),
                **common_params
            }]
        else:
            return [{
                'bootstrap':[False],
                **common_params
            }]

    def run_classification_task(self, X:np.ndarray, y:np.ndarray,
                                task_name:str, input_name:str,
                                target_name:str, n_clusters:int) -> ClassificationResult:
        """Run a single classification task (two-stage tuning)"""
        print(f"\n{'='*70}")
        print(f"Classification Task:{task_name}")
        print(f"  Input:{input_name} ({X.shape[1]}D) → Target:{target_name} ({n_clusters} classes)")
        print(f"{'='*70}")

        X_train = X[self.train_idx]
        y_train = y[self.train_idx]
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        print(f"  Train set:{len(X_train)}, Test set:{len(X_test)}")
        print(f"  Class distribution (train):{np.bincount(y_train)}")

        # ========== Stage 1:Coarse Search ==========
        print(f"\n--- Stage 1:Coarse Search ({self.coarse_n_iter} iter, {self.coarse_sample_ratio:.0%} data) ---")

        n_coarse = int(len(X_train) * self.coarse_sample_ratio)
        np.random.seed(42)
        coarse_idx = np.random.choice(len(X_train), n_coarse, replace=False)
        X_coarse = X_train[coarse_idx]
        y_coarse = y_train[coarse_idx]

        print(f"  Coarse search data size:{n_coarse}")
        print(f"  Parallel:CV={self.coarse_cv_jobs}, RF={self.coarse_rf_jobs}")

        start_time = time.time()
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        coarse_rf = RandomForestClassifier(random_state=42, n_jobs=self.coarse_rf_jobs)
        coarse_search = RandomizedSearchCV(
            coarse_rf,
            param_distributions=self.get_param_distributions(),
            n_iter=self.coarse_n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=self.coarse_cv_jobs,
            random_state=42,
            verbose=1
        )
        coarse_search.fit(X_coarse, y_coarse)

        coarse_time = time.time() - start_time
        coarse_best_params = coarse_search.best_params_
        coarse_best_score = coarse_search.best_score_

        print(f"\n  Coarse search complete!  Time:{coarse_time/60:.1f} min")
        print(f"  Best CV accuracy:{coarse_best_score:.4f}")
        print(f"  Best params:{coarse_best_params}")

        # ========== Stage 2:Fine Search ==========
        print(f"\n--- Stage 2:Fine Search ({self.fine_n_iter} iter, 100% train data) ---")

        fine_param_dist = self._create_fine_param_dist(coarse_best_params)
        print(f"  Fine search param range:narrowed based on coarse search")
        print(f"  Parallel:CV={self.fine_cv_jobs}, RF={self.fine_rf_jobs}")

        start_time = time.time()

        fine_rf = RandomForestClassifier(random_state=42, n_jobs=self.fine_rf_jobs)
        fine_search = RandomizedSearchCV(
            fine_rf,
            param_distributions=fine_param_dist,
            n_iter=self.fine_n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=self.fine_cv_jobs,
            random_state=42,
            verbose=1
        )
        fine_search.fit(X_train, y_train)

        fine_time = time.time() - start_time
        best_params = fine_search.best_params_
        best_cv_score = fine_search.best_score_

        print(f"\n  Fine search complete! Time:{fine_time/60:.1f} min")
        print(f"  Best CV accuracy:{best_cv_score:.4f}")
        print(f"  Best params:{best_params}")

        total_tuning_time = coarse_time + fine_time

        # ========== Final Evaluation ==========
        print(f"\n--- Final Evaluation ---")

        final_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=self.n_jobs)
        final_rf.fit(X_train, y_train)

        # Save trained model
        self.trained_models[task_name] = final_rf

        # Train set
        y_train_pred = final_rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
        train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
        train_correct = (y_train == y_train_pred).astype(int)

        # Test set
        y_test_pred = final_rf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        test_correct = (y_test == y_test_pred).astype(int)

        conf_matrix = confusion_matrix(y_test, y_test_pred)

        overfit_gap = train_acc - test_acc

        print(f"  Train set:Acc={train_acc:.4f}, F1_macro={train_f1_macro:.4f}")
        print(f"  Test set:Acc={test_acc:.4f}, F1_macro={test_f1_macro:.4f}")
        print(f"  Overfit gap:{overfit_gap:.4f}")

        if overfit_gap > 0.1:
            print(f"  ⚠️ Warning:Severe overfitting detected!")

        random_baseline = 1.0 / n_clusters
        print(f"  Random baseline:{random_baseline:.4f}")
        print(f"  Above baseline:{test_acc - random_baseline:+.4f}")

        return ClassificationResult(
            task_name=task_name,
            input_modality=input_name,
            target_modality=target_name,
            n_clusters=n_clusters,
            best_params=best_params,
            best_cv_score=best_cv_score,
            tuning_time=total_tuning_time,
            train_accuracy=train_acc,
            train_f1_macro=train_f1_macro,
            train_f1_weighted=train_f1_weighted,
            test_accuracy=test_acc,
            test_f1_macro=test_f1_macro,
            test_f1_weighted=test_f1_weighted,
            overfit_gap=overfit_gap,
            confusion_matrix=conf_matrix,
            train_predictions=y_train_pred,
            train_true_labels=y_train,
            test_predictions=y_test_pred,
            test_true_labels=y_test,
            train_correct=train_correct,
            test_correct=test_correct
        )

    def run_all_classification(self):
        """Run all classification tasks"""
        print("\n" + "=" * 80)
        print("Running Classification Tasks")
        print("=" * 80)

        results = {}

        tasks = [
            # Predict Projection class
            ('morph_to_proj', 'Morph', 'Proj', self.morph_vectors),
            ('gene_to_proj', 'Gene', 'Proj', self.gene_vectors),
            ('morph_gene_to_proj', 'Morph+Gene', 'Proj',
             np.hstack([self.morph_vectors, self.gene_vectors])),

            # Predict Morphology class
            ('gene_to_morph', 'Gene', 'Morph', self.gene_vectors),
            ('proj_to_morph', 'Proj', 'Morph', self.proj_vectors),
            ('gene_proj_to_morph', 'Gene+Proj', 'Morph',
             np.hstack([self.gene_vectors, self.proj_vectors])),

            # Predict Molecular class
            ('morph_to_gene', 'Morph', 'Gene', self.morph_vectors),
            ('proj_to_gene', 'Proj', 'Gene', self.proj_vectors),
            ('morph_proj_to_gene', 'Morph+Proj', 'Gene',
             np.hstack([self.morph_vectors, self.proj_vectors])),
        ]

        for task_name, input_name, target_name, X in tasks:
            target_info = self.clustering_info[target_name]
            y = target_info.labels
            n_clusters = target_info.n_clusters

            result = self.run_classification_task(
                X, y, task_name, input_name, target_name, n_clusters
            )
            results[task_name] = result

            # Intermediate save
            self.classification_results = results
            self._save_intermediate_results()

        self.classification_results = results
        return results

    def _save_intermediate_results(self):
        """Save intermediate results"""
        rows = []
        for name, result in self.classification_results.items():
            row = {
                'task':result.task_name,
                'input':result.input_modality,
                'target':result.target_modality,
                'n_clusters':result.n_clusters,
                'cv_accuracy':result.best_cv_score,
                'train_accuracy':result.train_accuracy,
                'test_accuracy':result.test_accuracy,
                'test_f1_macro':result.test_f1_macro,
                'overfit_gap':result.overfit_gap,
            }
            rows.append(row)

        pd.DataFrame(rows).to_csv('classification_intermediate.csv', index=False)

    # ==================== Visualization ====================

    def visualize_results(self, output_dir:str = "."):
        """Generate visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_clustering_info(output_dir)
        self._plot_accuracy_comparison(output_dir)
        self._plot_multimodal_gain(output_dir)
        self._plot_confusion_matrices(output_dir)
        self._plot_accuracy_boxplots(output_dir)
        self._plot_brain_region_distribution(output_dir)

        print(f"✓ Figures saved to:{output_dir}")

    def _save_figure(self, fig, output_dir:str, filename:str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_clustering_info(self, output_dir:str):
        """Clustering information plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (name, info) in zip(axes, self.clustering_info.items()):
            sizes = info.cluster_sizes
            ax.bar(range(len(sizes)), sorted(sizes, reverse=True), color='#3498DB', alpha=0.7)
            ax.set_xlabel('Cluster (sorted by size)')
            ax.set_ylabel('Number of neurons')
            ax.set_title(f'{name}:K={info.n_clusters}, Sil={info.silhouette:.3f}',
                         fontweight='bold')
            ax.axhline(y=np.mean(sizes), color='red', linestyle='--',
                       label=f'Mean={np.mean(sizes):.1f}')
            ax.legend()

        plt.suptitle('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_clustering_info.png")

    def _plot_accuracy_comparison(self, output_dir:str):
        """Accuracy comparison plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        groups = [
            ('Predict Projection Class', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('Predict Morphology Class', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('Predict Molecular Class', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        colors = ['#3498DB', '#27AE60', '#E74C3C']

        for ax, (title, tasks), color in zip(axes, groups, colors):
            labels = [self.classification_results[t].input_modality for t in tasks]
            train_acc = [self.classification_results[t].train_accuracy for t in tasks]
            test_acc = [self.classification_results[t].test_accuracy for t in tasks]

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, train_acc, width, label='Train', color=color, alpha=0.5)
            ax.bar(x + width / 2, test_acc, width, label='Test', color=color, alpha=0.9)

            # Random baseline
            n_clusters = self.classification_results[tasks[0]].n_clusters
            baseline = 1.0 / n_clusters
            ax.axhline(y=baseline, color='gray', linestyle='--',
                       label=f'Random ({baseline:.2f})')

            for i, (tr, te) in enumerate(zip(train_acc, test_acc)):
                ax.annotate(f'{te:.2f}', xy=(i + width / 2, te), xytext=(0, 3),
                            textcoords='offset points', ha='center', fontsize=9)

            ax.set_ylabel('Accuracy')
            ax.set_title(f'{title} (K={n_clusters})', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.legend(loc='upper left', fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Classification Accuracy:Train vs Test', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_accuracy_comparison.png")

    def _plot_multimodal_gain(self, output_dir:str):
        """Multi-modal gain plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        comparisons = [
            ('Proj', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('Morph', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('Gene', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]

        x_positions = []
        gains = []
        labels = []
        colors = []

        for i, (target, multi_key, single_keys) in enumerate(comparisons):
            multi_acc = self.classification_results[multi_key].test_accuracy
            single_accs = [self.classification_results[k].test_accuracy for k in single_keys]
            best_single = max(single_accs)
            gain = multi_acc - best_single

            x_positions.append(i)
            gains.append(gain)
            labels.append(f'→{target}')
            colors.append('#27AE60' if gain > 0 else '#E74C3C')

        bars = ax.bar(x_positions, gains, color=colors, alpha=0.8, edgecolor='black')

        for bar, gain in zip(bars, gains):
            va = 'bottom' if gain >= 0 else 'top'
            offset = 0.005 if gain >= 0 else -0.005
            ax.annotate(f'{gain:+.3f}', xy=(bar.get_x() + bar.get_width() / 2, gain + offset),
                        ha='center', va=va, fontweight='bold')

        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_ylabel('Accuracy Gain (Multi - Best Single)')
        ax.set_title('Multi-modal Classification Gain (Test Set)', fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_multimodal_gain.png")

    def _plot_confusion_matrices(self, output_dir:str):
        """Confusion matrices plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        multi_tasks = ['morph_gene_to_proj', 'gene_proj_to_morph', 'morph_proj_to_gene']
        titles = ['Morph+Gene → Proj', 'Gene+Proj → Morph', 'Morph+Proj → Gene']

        for ax, task, title in zip(axes, multi_tasks, titles):
            result = self.classification_results[task]
            cm = result.confusion_matrix

            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm_norm, ax=ax, cmap='Blues', annot=False,
                        cbar_kws={'shrink':0.5})
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{title}\nAcc={result.test_accuracy:.3f}', fontweight='bold')

        plt.suptitle('Confusion Matrices (Normalized, Test Set)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_confusion_matrices.png")

    def _plot_accuracy_boxplots(self, output_dir:str):
        """Plot accuracy distribution as boxplots using per-class accuracy"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        groups = [
            ('Predict Projection Class', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('Predict Morphology Class', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('Predict Molecular Class', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        for ax, (title, tasks) in zip(axes, groups):
            # Calculate per-class accuracy for boxplot
            box_data_train = []
            box_data_test = []
            labels = []

            for task in tasks:
                result = self.classification_results[task]

                # Per-class accuracy for train
                train_per_class = []
                for c in range(result.n_clusters):
                    mask = result.train_true_labels == c
                    if mask.sum() > 0:
                        class_acc = (result.train_predictions[mask] == c).mean()
                        train_per_class.append(class_acc)
                box_data_train.append(train_per_class)

                # Per-class accuracy for test
                test_per_class = []
                for c in range(result.n_clusters):
                    mask = result.test_true_labels == c
                    if mask.sum() > 0:
                        class_acc = (result.test_predictions[mask] == c).mean()
                        test_per_class.append(class_acc)
                box_data_test.append(test_per_class)

                labels.append(result.input_modality)

            # Create boxplot
            positions_train = np.arange(len(labels)) * 2.5
            positions_test = positions_train + 0.8

            bp_train = ax.boxplot(box_data_train, positions=positions_train,
                                  widths=0.6, patch_artist=True,
                                  boxprops=dict(facecolor='#3498DB', alpha=0.7),
                                  medianprops=dict(color='darkblue', linewidth=2),
                                  whiskerprops=dict(color='#3498DB'),
                                  capprops=dict(color='#3498DB'),
                                  flierprops=dict(marker='o', markersize=3, alpha=0.5))

            bp_test = ax.boxplot(box_data_test, positions=positions_test,
                                 widths=0.6, patch_artist=True,
                                 boxprops=dict(facecolor='#E74C3C', alpha=0.7),
                                 medianprops=dict(color='darkred', linewidth=2),
                                 whiskerprops=dict(color='#E74C3C'),
                                 capprops=dict(color='#E74C3C'),
                                 flierprops=dict(marker='o', markersize=3, alpha=0.5))

            # Random baseline
            n_clusters = self.classification_results[tasks[0]].n_clusters
            baseline = 1.0 / n_clusters
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
                       label=f'Random baseline ({baseline:.2f})')

            ax.set_xticks(positions_train + 0.4)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.set_ylabel('Per-class Accuracy')
            ax.set_title(f'{title} (K={n_clusters})', fontweight='bold')
            ax.legend([bp_train["boxes"][0], bp_test["boxes"][0]],
                      ['Train', 'Test'], loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.suptitle('Per-class Accuracy Distribution (Boxplot)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_accuracy_boxplots.png")

    def _plot_brain_region_distribution(self, output_dir:str):
        """Plot accuracy distribution by major brain region"""

        # Get neuron indices and their major brain regions
        neuron_major_regions = []
        for nid in self.valid_neuron_ids:
            if nid in self.neuron_soma_regions:
                region = self.neuron_soma_regions[nid]
                major = self._get_major_brain_region(region)
                neuron_major_regions.append(major)
            else:
                neuron_major_regions.append('Unknown')

        neuron_major_regions = np.array(neuron_major_regions)

        # Get unique major regions with sufficient samples
        unique_regions = []
        for region in ['Cortex', 'Thalamus', 'Striatum', 'Hippocampus',
                       'Midbrain', 'Hypothalamus', 'Brainstem', 'Olfactory', 'Other']:
            test_mask = neuron_major_regions[self.test_idx] == region
            if test_mask.sum() >= 10:
                unique_regions.append(region)

                if len(unique_regions) == 0:
                    print("  Warning:No brain regions with sufficient samples for region-wise analysis")
                    return

        # Create figure for each prediction task group
        task_groups = [
            ('Predict Projection', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('Predict Morphology', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('Predict Molecular', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        for group_name, tasks in task_groups:
            fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6))
            if len(tasks) == 1:
                axes = [axes]

            for ax, task in zip(axes, tasks):
                result = self.classification_results[task]
                if result.test_correct is None:
                    continue

                # Collect data by region (accuracy = mean of correct predictions)
                region_data = {}
                test_regions = neuron_major_regions[self.test_idx]

                for region in unique_regions:
                    mask = test_regions == region
                    if mask.sum() > 0:
                        region_correct = result.test_correct[mask]
                        region_data[region] = region_correct

                # Create bar plot
                if region_data:
                    data = [region_data[r] for r in unique_regions if r in region_data]
                    labels_r = [r for r in unique_regions if r in region_data]
                    counts = [len(region_data[r]) for r in unique_regions if r in region_data]
                    accuracies = [np.mean(region_data[r]) for r in unique_regions if r in region_data]

                    # Bar plot for accuracy by region
                    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_r)))
                    bars = ax.bar(range(len(labels_r)), accuracies, color=colors, alpha=0.8,
                                  edgecolor='black')

                    # Add sample counts and accuracy values
                    for i, (label, count, acc) in enumerate(zip(labels_r, counts, accuracies)):
                        ax.annotate(f'{acc:.2f}\n(n={count})', xy=(i, acc),
                                    xytext=(0, 5), textcoords='offset points',
                                    ha='center', fontsize=8)

                    # Random baseline
                    baseline = 1.0 / result.n_clusters
                    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7,
                               label=f'Random baseline ({baseline:.2f})')

                    ax.set_xticks(range(len(labels_r)))
                    ax.set_xticklabels(labels_r, rotation=45, ha='right')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'{result.input_modality} → {result.target_modality}', fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    ax.set_ylim(0, 1.05)
                    ax.legend(loc='lower right', fontsize=8)

            plt.suptitle(f'{group_name} - Accuracy by Brain Region (Test Set)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            filename = f"6_brain_region_{group_name.lower().replace(' ', '_')}.png"
            self._save_figure(fig, output_dir, filename)

        # Also create a summary heatmap
        self._plot_brain_region_heatmap(output_dir, neuron_major_regions, unique_regions)

    def _plot_brain_region_heatmap(self, output_dir:str,
                                    neuron_major_regions:np.ndarray,
                                    unique_regions:List[str]):
        """Create a heatmap of accuracy by brain region and task"""

        tasks = list(self.classification_results.keys())

        # Create matrix of accuracies
        data = np.zeros((len(tasks), len(unique_regions)))
        counts = np.zeros((len(tasks), len(unique_regions)), dtype=int)

        test_regions = neuron_major_regions[self.test_idx]

        for i, task in enumerate(tasks):
            result = self.classification_results[task]
            if result.test_correct is None:
                continue

            for j, region in enumerate(unique_regions):
                mask = test_regions == region
                if mask.sum() > 0:
                    data[i, j] = result.test_correct[mask].mean()
                    counts[i, j] = mask.sum()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create annotation with accuracy and count
        annot = np.empty_like(data, dtype=object)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if counts[i, j] > 0:
                    annot[i, j] = f'{data[i, j]:.2f}\n(n={counts[i, j]})'
                else:
                    annot[i, j] = 'N/A'

        sns.heatmap(data, annot=annot, fmt='', cmap='RdYlGn',
                    xticklabels=unique_regions,
                    yticklabels=[self.classification_results[t].input_modality + ' → ' +
                                 self.classification_results[t].target_modality for t in tasks],
                    ax=ax, vmin=0, vmax=1,
                    cbar_kws={'label':'Accuracy'})

        ax.set_title('Accuracy by Brain Region and Task (Test Set)',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Major Brain Region')
        ax.set_ylabel('Prediction Task')

        plt.tight_layout()
        self._save_figure(fig, output_dir, "7_brain_region_heatmap.png")

    # ==================== Save Results ====================

    def save_results(self, output_dir:str = "."):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        models_dir = os.path.join(output_dir, "models")
        data_dir = os.path.join(output_dir, "intermediate_data")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Clustering info
        cluster_rows = []
        for name, info in self.clustering_info.items():
            cluster_rows.append({
                'modality':info.modality,
                'n_clusters':info.n_clusters,
                'silhouette':info.silhouette,
                'min_size':info.cluster_sizes.min(),
                'max_size':info.cluster_sizes.max(),
                'mean_size':info.cluster_sizes.mean(),
            })
        pd.DataFrame(cluster_rows).to_csv(f"{output_dir}/clustering_info.csv", index=False)

        # Classification results
        rows = []
        for name, result in self.classification_results.items():
            row = {
                'task':result.task_name,
                'input':result.input_modality,
                'target':result.target_modality,
                'n_clusters':result.n_clusters,
                'cv_accuracy':result.best_cv_score,
                'train_accuracy':result.train_accuracy,
                'train_f1_macro':result.train_f1_macro,
                'test_accuracy':result.test_accuracy,
                'test_f1_macro':result.test_f1_macro,
                'test_f1_weighted':result.test_f1_weighted,
                'overfit_gap':result.overfit_gap,
                'tuning_time_min':result.tuning_time / 60,
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{output_dir}/classification_results.csv", index=False)

        # Best parameters
        param_rows = []
        for name, result in self.classification_results.items():
            row = {'task':name}
            row.update(result.best_params)
            param_rows.append(row)
        pd.DataFrame(param_rows).to_csv(f"{output_dir}/best_params.csv", index=False)

        # Save detailed results (pickle)
        with open(f"{output_dir}/full_results.pkl", 'wb') as f:
            pickle.dump({
                'clustering_info':self.clustering_info,
                'classification_results':self.classification_results,
            }, f)

        # ========== Save trained models ==========
        print("\nSaving trained models...")
        for task_name, model in self.trained_models.items():
            model_path = os.path.join(models_dir, f"{task_name}_rf_model.joblib")
            joblib.dump(model, model_path)
            print(f"  Saved:{model_path}")

        # Save KMeans models for clustering
        for modality, info in self.clustering_info.items():
            if info.kmeans_model is not None:
                kmeans_path = os.path.join(models_dir, f"{modality}_kmeans_model.joblib")
                joblib.dump(info.kmeans_model, kmeans_path)
                print(f"  Saved:{kmeans_path}")

        # ========== Save preprocessors ==========
        preprocessors_path = os.path.join(models_dir, "preprocessors.pkl")
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(self.preprocessors, f)
        print(f"  Saved:{preprocessors_path}")

        # ========== Save intermediate training data ==========
        print("\nSaving intermediate training data...")

        # Save processed vectors
        np.savez(os.path.join(data_dir, "processed_vectors.npz"),
                 morph_vectors=self.morph_vectors,
                 gene_vectors=self.gene_vectors,
                 proj_vectors=self.proj_vectors,
                 train_idx=self.train_idx,
                 test_idx=self.test_idx)
        print(f"  Saved processed vectors")

        # Save cluster labels
        cluster_labels = {}
        for name, info in self.clustering_info.items():
            cluster_labels[name] = info.labels
        np.savez(os.path.join(data_dir, "cluster_labels.npz"), **cluster_labels)
        print(f"  Saved cluster labels")

        # Save neuron IDs and mappings
        with open(os.path.join(data_dir, "neuron_info.pkl"), 'wb') as f:
            pickle.dump({
                'valid_neuron_ids':self.valid_neuron_ids,
                'neuron_soma_regions':self.neuron_soma_regions,
                'all_target_regions':self.all_target_regions,
                'all_subclasses':self.all_subclasses,
                'vector_info':self.vector_info,
            }, f)
        print(f"  Saved neuron info")

        # Save per-sample predictions for each task
        predictions_data = {}
        for task_name, result in self.classification_results.items():
            predictions_data[task_name] = {
                'train_predictions':result.train_predictions,
                'train_true_labels':result.train_true_labels,
                'test_predictions':result.test_predictions,
                'test_true_labels':result.test_true_labels,
                'train_correct':result.train_correct,
                'test_correct':result.test_correct,
                'confusion_matrix':result.confusion_matrix,
            }

        with open(os.path.join(data_dir, "predictions.pkl"), 'wb') as f:
            pickle.dump(predictions_data, f)
        print(f"  Saved per-sample predictions")

        # Save raw features
        with open(os.path.join(data_dir, "raw_features.pkl"), 'wb') as f:
            pickle.dump({
                'axon_features_raw':self.axon_features_raw,
                'dendrite_features_raw':self.dendrite_features_raw,
                'local_gene_features_raw':self.local_gene_features_raw,
                'projection_vectors_raw':self.projection_vectors_raw,
            }, f)
        print(f"  Saved raw features")

        # Save configuration
        config = {
            'search_radius':self.search_radius,
            'pca_variance_threshold':self.pca_variance_threshold,
            'test_ratio':self.test_ratio,
            'k_candidates':self.k_candidates,
            'fixed_k':self.fixed_k,
            'coarse_sample_ratio':self.coarse_sample_ratio,
            'coarse_n_iter':self.coarse_n_iter,
            'fine_n_iter':self.fine_n_iter,
            'cv_folds':self.cv_folds,
            'n_jobs':self.n_jobs,
            'timestamp':datetime.now().isoformat(),
        }

        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved configuration")

        print(f"\n✓ All results saved to:{output_dir}")

    # ==================== Inference Helper ====================

    @staticmethod
    def load_model_for_inference(model_dir:str, task_name:str):
        """
        Load trained model and preprocessors for inference.

        Args:
            model_dir:Directory containing saved models
            task_name:Name of the task (e.g., 'morph_to_proj')

        Returns:
            model:Trained RandomForestClassifier
            preprocessors:Dict of preprocessing objects
            kmeans_models:Dict of KMeans models for each modality
        """
        models_dir = os.path.join(model_dir, "models")

        # Load model
        model_path = os.path.join(models_dir, f"{task_name}_rf_model.joblib")
        model = joblib.load(model_path)

        # Load preprocessors
        preprocessors_path = os.path.join(models_dir, "preprocessors.pkl")
        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)

        # Load KMeans models
        kmeans_models = {}
        for modality in ['Morph', 'Gene', 'Proj']:
            kmeans_path = os.path.join(models_dir, f"{modality}_kmeans_model.joblib")
            if os.path.exists(kmeans_path):
                kmeans_models[modality] = joblib.load(kmeans_path)

        return model, preprocessors, kmeans_models

    @staticmethod
    def preprocess_for_inference(raw_data:np.ndarray,
                                  preprocessor:Dict,
                                  modality:str) -> np.ndarray:
        """
        Preprocess raw data for inference using saved preprocessors.

        Args:
            raw_data:Raw feature array (n_samples, n_features)
            preprocessor:Dict containing scaler and PCA
            modality:'morph', 'gene', or 'proj'

        Returns:
            Preprocessed data ready for model input
        """
        # Apply log transform if needed
        if preprocessor.get('do_log', False):
            data = np.log1p(raw_data)
        else:
            data = raw_data

        # Apply column filtering if exists (for gene and proj)
        if 'valid_cols' in preprocessor:
            data = data[:, preprocessor['valid_cols']]

        # Scale
        data_scaled = preprocessor['scaler'].transform(data)

        # PCA
        data_pca = preprocessor['pca'].transform(data_scaled)

        return data_pca

    # ==================== Main Pipeline ====================

    def run_full_pipeline(self, output_dir:str = "./classification_results"):
        """Run complete pipeline"""
        print("\n" + "=" * 80)
        print("Cluster Label Classification Experiment")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  CPU cores:{self.n_jobs}")
        
        if self.fixed_k:
            print(f"  Using fixed K values:{self.fixed_k}")
        else:
            print(f"  Candidate K values:{self.k_candidates}")
            
        print(f"  Coarse search:{self.coarse_sample_ratio:.0%} data, {self.coarse_n_iter} iterations")
        print(f"  Fine search:100% data, {self.fine_n_iter} iterations")
        print(f"  max_depth range:[2, 6]")
        print(f"  Parameter search:Continuous distributions")

        n = self.load_all_data()
        if n == 0:
            return

        self.find_optimal_clustering()
        self.run_all_classification()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

    def _print_summary(self):
        """Print summary"""
        print("\n" + "=" * 80)
        print("Experiment Results Summary")
        print("=" * 80)

        # Clustering info
        print(f"\n【Clustering Info】")
        for name, info in self.clustering_info.items():
            fixed_str = "(fixed)" if name in self.fixed_k else "(searched)"
            print(f"  {name}:K={info.n_clusters} {fixed_str}, Silhouette={info.silhouette:.4f}")

        # Classification results
        print(f"\n【Classification Results (Test Set)】")
        print(f"{'Task':<25} {'Input':<12} {'Target':<8} {'K':>4} {'Accuracy':>10} {'F1':>10} {'Baseline':>8}")
        print("-" * 80)

        for name, result in self.classification_results.items():
            baseline = 1.0 / result.n_clusters
            print(f"{name:<25} {result.input_modality:<12} {result.target_modality:<8} "
                  f"{result.n_clusters:>4} {result.test_accuracy:>10.4f} "
                  f"{result.test_f1_macro:>10.4f} {baseline:>8.4f}")

        # Multi-modal gain
        print(f"\n【Multi-modal Gain (Test Set Accuracy)】")
        for target, multi_key, single_keys in [
            ('Projection', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('Morphology', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('Molecular', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]:
            multi = self.classification_results[multi_key].test_accuracy
            best_single = max(self.classification_results[k].test_accuracy for k in single_keys)
            gain = multi - best_single
            status = "✓" if gain > 0 else "✗"
            print(f"  Predict {target}:{multi:.4f} vs {best_single:.4f} (gain:{gain:+.4f} {status})")


# ==================== Inference Example ====================

def inference_example():
    """
    Example of how to load and use trained models for inference.
    """
    model_dir = "./classification_results_v1"
    task_name = "morph_to_proj"

    # Load model and preprocessors
    model, preprocessors, kmeans_models = ClusterClassificationExperiment.load_model_for_inference(
        model_dir, task_name
    )

    # Load some test data
    data_dir = os.path.join(model_dir, "intermediate_data")
    with open(os.path.join(data_dir, "raw_features.pkl"), 'rb') as f:
        raw_features = pickle.load(f)

    # Example:get first 10 neurons
    neuron_ids = list(raw_features['axon_features_raw'].keys())[:10]

    # Prepare morphology input
    morph_raw = np.array([
        np.concatenate([
            raw_features['axon_features_raw'][nid],
            raw_features['dendrite_features_raw'][nid]
        ])
        for nid in neuron_ids
    ])

    # Preprocess
    morph_processed = ClusterClassificationExperiment.preprocess_for_inference(
        morph_raw, preprocessors['morph'], 'morph'
    )

    # Predict class labels
    predictions = model.predict(morph_processed)

    # Get prediction probabilities
    probabilities = model.predict_proba(morph_processed)

    print(f"Input shape:{morph_processed.shape}")
    print(f"Predicted labels:{predictions}")
    print(f"Prediction probabilities shape:{probabilities.shape}")

    return predictions, probabilities


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./classification_results_v1"
    
    # ============================================================
    # Option 1:Manually specify fixed K values
    # ============================================================
    FIXED_K = {
        'Morph':7,  # Set based on your clustering exploration results
        'Gene':25,   # Set based on your clustering exploration results
        'Proj':13,   # Set based on your clustering exploration results
    }

    # ============================================================
    # Option 2:Load from clustering exploration results
    # ============================================================
    # CLUSTERING_RESULTS_DIR = "./clustering_exploration"
    # Uncomment the line below and comment out FIXED_K above to use this option
    # FIXED_K = None

    with ClusterClassificationExperiment(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            test_ratio=0.2,
            # Use one of the following options:
            fixed_k=FIXED_K,  # Option 1:manually specify
            # clustering_results_dir=CLUSTERING_RESULTS_DIR,  # Option 2:load from file
            coarse_sample_ratio=0.1,
            coarse_n_iter=200,
            fine_n_iter=15,
            cv_folds=5,
            n_jobs=-1
    ) as experiment:
        experiment.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
