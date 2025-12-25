"""
Task 1: Cluster Label Classification V3 (Fixed Version)
========================================================
关键修复：
1.聚类只在训练集上 fit，测试集用 kmeans.predict() 分配标签
2.添加后期融合 (Late Fusion) 选项
3.添加对等降维 (Equal Dimension) 选项
4.增加粗搜索最小样本数检查
5.添加分层采样选项

Author: Claude (Fixed)
Date: 2025-01
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from pathlib import Path
import time
import multiprocessing
import json
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, silhouette_score, confusion_matrix
)
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import neo4j

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()
print(f"Detected {N_CORES} CPU cores")


@dataclass
class ClusteringInfo:
    """Clustering information"""
    modality: str
    n_clusters: int
    silhouette_train: float  # 只在训练集上计算
    silhouette_test: float  # 测试集上的 silhouette (用于验证)
    labels_train: np.ndarray
    labels_test: np.ndarray
    cluster_sizes_train: np.ndarray
    kmeans_model: KMeans = None


@dataclass
class ClassificationResult:
    """Classification result"""
    task_name: str
    input_modality: str
    target_modality: str
    n_clusters: int
    best_params: Dict
    best_cv_score: float
    tuning_time: float
    train_accuracy: float
    train_f1_macro: float
    test_accuracy: float
    test_f1_macro: float
    overfit_gap: float
    confusion_matrix: np.ndarray
    train_predictions: np.ndarray = None
    train_true_labels: np.ndarray = None
    test_predictions: np.ndarray = None
    test_true_labels: np.ndarray = None


class ClusterClassificationV3:
    """
    修正版聚类分类实验

    关键修复：
    1.聚类只在训练集上 fit，测试集用 predict
    2.支持后期融合 (Late Fusion)
    3.支持对等降维
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

    @staticmethod
    def get_param_distributions():
        from scipy.stats import randint, uniform
        common_params = {
            'n_estimators': randint(100, 401),
            'max_depth': randint(3, 8),
            'min_samples_split': randint(5, 30),
            'min_samples_leaf': randint(2, 15),
            'max_features': uniform(0.2, 0.6),
            'class_weight': ['balanced', 'balanced_subsample'],
        }
        return [
            {'bootstrap': [True], 'max_samples': uniform(0.6, 0.4), **common_params},
            {'bootstrap': [False], **common_params}
        ]

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 4.0,
                 pca_variance_threshold: float = 0.95,
                 test_ratio: float = 0.2,
                 fixed_k: Dict[str, int] = None,
                 coarse_sample_ratio: float = 0.2,  # 提高到 20%
                 coarse_n_iter: int = 150,
                 fine_n_iter: int = 50,
                 cv_folds: int = 5,
                 n_jobs: int = -1,
                 use_late_fusion: bool = True,  # 后期融合
                 equal_dim_for_fusion: int = None,  # 对等降维维度
                 min_samples_per_class: int = 5):  # 粗搜索最小样本数

        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.fixed_k = fixed_k or {}
        self.coarse_sample_ratio = coarse_sample_ratio
        self.coarse_n_iter = coarse_n_iter
        self.fine_n_iter = fine_n_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs if n_jobs > 0 else N_CORES
        self.use_late_fusion = use_late_fusion
        self.equal_dim_for_fusion = equal_dim_for_fusion
        self.min_samples_per_class = min_samples_per_class

        print(f"Configuration (V3 - Fixed):")
        print(f"  PCA variance: {self.pca_variance_threshold}")
        print(f"  Test ratio: {self.test_ratio}")
        print(f"  Fixed K: {self.fixed_k}")
        print(f"  Coarse sample ratio: {self.coarse_sample_ratio}")
        print(f"  Use late fusion: {self.use_late_fusion}")
        print(f"  Equal dim for fusion: {self.equal_dim_for_fusion}")
        print(f"  Min samples per class: {self.min_samples_per_class}")

        # Data
        self.valid_neuron_ids: List[str] = []
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # Split indices
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

        # Vectors
        self.morph_vectors: np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # 对等降维后的向量 (用于融合)
        self.morph_vectors_equal: np.ndarray = None
        self.gene_vectors_equal: np.ndarray = None
        self.proj_vectors_equal: np.ndarray = None

        self.preprocessors: Dict[str, Dict] = {}
        self.clustering_info: Dict[str, ClusteringInfo] = {}
        self.classification_results: Dict[str, ClassificationResult] = {}
        self.trained_models: Dict[str, object] = {}

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
        self._filter_valid_neurons()
        self._split_train_test()
        self._process_all_vectors()

        print(f"\n✓ Data loading complete:")
        print(f"  Total neurons: {len(self.valid_neuron_ids)}")
        print(f"  Train: {len(self.train_idx)}, Test: {len(self.test_idx)}")

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
        """与聚类探索一致的 split (seed=42)"""
        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]

        print(f"  Train/Test split: {len(self.train_idx)}/{len(self.test_idx)} (seed=42)")

    def _process_all_vectors(self):
        print("\nProcessing vectors (fit on TRAIN SET only)...")
        neurons = self.valid_neuron_ids

        # Morphology
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_prep = self._process_vector(morph_raw, 'Morph', do_log=True)
        self.preprocessors['morph'] = morph_prep

        # Molecular
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        gene_raw = gene_raw[:, valid_cols]
        self.gene_vectors, gene_prep = self._process_vector(gene_raw, 'Gene', do_log=True)
        self.preprocessors['gene'] = gene_prep
        self.preprocessors['gene']['valid_cols'] = valid_cols

        # Projection
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        proj_raw = proj_raw[:, valid_cols]
        self.proj_vectors, proj_prep = self._process_vector(proj_raw, 'Proj', do_log=True)
        self.preprocessors['proj'] = proj_prep
        self.preprocessors['proj']['valid_cols'] = valid_cols

        # 对等降维 (如果启用)
        if self.equal_dim_for_fusion:
            self._create_equal_dim_vectors()

    def _process_vector(self, X_raw: np.ndarray, name: str,
                        do_log: bool = False) -> Tuple[np.ndarray, Dict]:
        original_dims = X_raw.shape[1]

        if do_log:
            X = np.log1p(X_raw)
        else:
            X = X_raw

        scaler = StandardScaler()
        scaler.fit(X[self.train_idx])
        X_scaled = scaler.transform(X)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(X_scaled[self.train_idx])
        X_pca = pca.transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  {name}: {original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        preprocessor = {'scaler': scaler, 'pca': pca, 'do_log': do_log}
        return X_pca, preprocessor

    def _create_equal_dim_vectors(self):
        """创建对等维度的向量用于融合"""
        target_dim = self.equal_dim_for_fusion
        print(f"\nCreating equal dimension vectors ({target_dim}D each)...")

        for name, vectors in [('morph', self.morph_vectors),
                              ('gene', self.gene_vectors),
                              ('proj', self.proj_vectors)]:
            if vectors.shape[1] > target_dim:
                pca = PCA(n_components=target_dim)
                pca.fit(vectors[self.train_idx])
                reduced = pca.transform(vectors)
            else:
                # 如果维度已经小于目标，保持不变或填充
                reduced = vectors

            if name == 'morph':
                self.morph_vectors_equal = reduced
            elif name == 'gene':
                self.gene_vectors_equal = reduced
            else:
                self.proj_vectors_equal = reduced

            print(f"  {name}: {vectors.shape[1]}D → {reduced.shape[1]}D")

    # ==================== Clustering ====================

    def find_optimal_clustering(self):
        """
        关键修复：聚类只在训练集上 fit，测试集用 predict
        """
        print("\n" + "=" * 80)
        print("Clustering (FIT on TRAIN only, PREDICT on TEST)")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        for name, vectors in modalities:
            print(f"\n--- {name} ---")

            k = self.fixed_k.get(name, 10)
            print(f"  Using K={k}")

            # 关键：只在训练集上 fit
            X_train = vectors[self.train_idx]
            X_test = vectors[self.test_idx]

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train)  # 只 fit 训练集

            # 训练集标签
            labels_train = kmeans.predict(X_train)
            # 测试集标签 - 用 predict 而非 fit_predict
            labels_test = kmeans.predict(X_test)

            # 计算 silhouette
            sil_train = silhouette_score(X_train, labels_train)
            sil_test = silhouette_score(X_test, labels_test)

            print(f"  Silhouette (train): {sil_train:.4f}")
            print(f"  Silhouette (test):  {sil_test:.4f}")

            cluster_sizes_train = np.bincount(labels_train, minlength=k)

            self.clustering_info[name] = ClusteringInfo(
                modality=name,
                n_clusters=k,
                silhouette_train=sil_train,
                silhouette_test=sil_test,
                labels_train=labels_train,
                labels_test=labels_test,
                cluster_sizes_train=cluster_sizes_train,
                kmeans_model=kmeans
            )

            print(f"  Cluster sizes (train): min={cluster_sizes_train.min()}, "
                  f"max={cluster_sizes_train.max()}, mean={cluster_sizes_train.mean():.1f}")

    # ==================== Classification ====================

    def _get_stratified_coarse_sample(self, y_train: np.ndarray) -> np.ndarray:
        """获取分层采样的粗搜索样本，确保每个类别有足够样本"""
        n_classes = len(np.unique(y_train))
        n_coarse = int(len(y_train) * self.coarse_sample_ratio)

        # 确保每个类别至少有 min_samples_per_class 个样本
        min_total = n_classes * self.min_samples_per_class
        n_coarse = max(n_coarse, min_total)
        n_coarse = min(n_coarse, len(y_train))

        # 分层采样
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - n_coarse / len(y_train),
                                     random_state=42)

        try:
            train_idx, _ = next(sss.split(np.zeros(len(y_train)), y_train))
            return train_idx
        except ValueError:
            # 如果分层采样失败，使用随机采样
            np.random.seed(42)
            return np.random.choice(len(y_train), n_coarse, replace=False)

    def run_classification_task(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                task_name: str, input_name: str,
                                target_name: str) -> ClassificationResult:
        """Run classification with fixed labels"""
        n_clusters = len(np.unique(y_train))

        print(f"\n{'=' * 70}")
        print(f"Task: {task_name}")
        print(f"  Input: {input_name} ({X_train.shape[1]}D)")
        print(f"  Target: {target_name} ({n_clusters} classes)")
        print(f"{'=' * 70}")

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Class distribution (train): {np.bincount(y_train, minlength=n_clusters)}")
        print(f"  Class distribution (test):  {np.bincount(y_test, minlength=n_clusters)}")

        # ========== Coarse Search ==========
        print(f"\n--- Coarse Search ({self.coarse_n_iter} iter) ---")

        coarse_idx = self._get_stratified_coarse_sample(y_train)
        X_coarse = X_train[coarse_idx]
        y_coarse = y_train[coarse_idx]

        print(f"  Coarse sample size: {len(coarse_idx)}")
        print(f"  Coarse class dist: {np.bincount(y_coarse, minlength=n_clusters)}")

        start_time = time.time()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        coarse_rf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs // 2)
        coarse_search = RandomizedSearchCV(
            coarse_rf,
            param_distributions=self.get_param_distributions(),
            n_iter=self.coarse_n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=2,
            random_state=42,
            verbose=0
        )
        coarse_search.fit(X_coarse, y_coarse)

        coarse_time = time.time() - start_time
        print(f"  Coarse done: {coarse_time:.1f}s, Best CV: {coarse_search.best_score_:.4f}")

        # ========== Fine Search ==========
        print(f"\n--- Fine Search ({self.fine_n_iter} iter, full train) ---")

        fine_param_dist = self._create_fine_param_dist(coarse_search.best_params_)

        start_time = time.time()
        fine_rf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs // 2)
        fine_search = RandomizedSearchCV(
            fine_rf,
            param_distributions=fine_param_dist,
            n_iter=self.fine_n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=2,
            random_state=42,
            verbose=0
        )
        fine_search.fit(X_train, y_train)

        fine_time = time.time() - start_time
        best_params = fine_search.best_params_
        best_cv_score = fine_search.best_score_

        print(f"  Fine done: {fine_time:.1f}s, Best CV: {best_cv_score:.4f}")

        # ========== Final Evaluation ==========
        print(f"\n--- Final Evaluation ---")

        final_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=self.n_jobs)
        final_rf.fit(X_train, y_train)

        self.trained_models[task_name] = final_rf

        y_train_pred = final_rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')

        y_test_pred = final_rf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        overfit_gap = train_acc - test_acc
        baseline = 1.0 / n_clusters

        print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
        print(f"  Overfit gap: {overfit_gap:.4f}")
        print(f"  Baseline: {baseline:.4f}, Above: {test_acc - baseline:+.4f}")

        if overfit_gap > 0.15:
            print(f"  ⚠️  High overfitting!")
        if test_acc < train_acc * 0.8:
            print(f"  ⚠️  Test << Train, possible data issue")

        return ClassificationResult(
            task_name=task_name,
            input_modality=input_name,
            target_modality=target_name,
            n_clusters=n_clusters,
            best_params=best_params,
            best_cv_score=best_cv_score,
            tuning_time=coarse_time + fine_time,
            train_accuracy=train_acc,
            train_f1_macro=train_f1,
            test_accuracy=test_acc,
            test_f1_macro=test_f1,
            overfit_gap=overfit_gap,
            confusion_matrix=confusion_matrix(y_test, y_test_pred),
            train_predictions=y_train_pred,
            train_true_labels=y_train,
            test_predictions=y_test_pred,
            test_true_labels=y_test,
        )

    def _create_fine_param_dist(self, coarse_best: Dict) -> List[Dict]:
        from scipy.stats import randint, uniform

        best_n = coarse_best.get('n_estimators', 200)
        best_depth = coarse_best.get('max_depth', 5)
        best_split = coarse_best.get('min_samples_split', 10)
        best_leaf = coarse_best.get('min_samples_leaf', 5)
        best_feat = coarse_best.get('max_features', 0.4)

        common_params = {
            'n_estimators': randint(max(50, best_n - 100), min(500, best_n + 100)),
            'max_depth': randint(max(2, best_depth - 2), min(15, best_depth + 3)),
            'min_samples_split': randint(max(2, best_split - 10), min(50, best_split + 15)),
            'min_samples_leaf': randint(max(1, best_leaf - 3), min(20, best_leaf + 5)),
            'max_features': uniform(max(0.1, best_feat - 0.15), 0.3),
            'class_weight': [coarse_best.get('class_weight', 'balanced')],
        }

        if coarse_best.get('bootstrap', True):
            best_samples = coarse_best.get('max_samples', 0.8) or 0.8
            return [{'bootstrap': [True],
                     'max_samples': uniform(max(0.5, best_samples - 0.1), 0.2),
                     **common_params}]
        else:
            return [{'bootstrap': [False], **common_params}]

    def run_late_fusion_task(self, X1_train: np.ndarray, X1_test: np.ndarray,
                             X2_train: np.ndarray, X2_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray,
                             task_name: str, input_name: str,
                             target_name: str) -> ClassificationResult:
        """
        后期融合：分别训练两个模型，然后用 Meta-classifier 融合
        """
        n_clusters = len(np.unique(y_train))

        print(f"\n{'=' * 70}")
        print(f"Late Fusion Task: {task_name}")
        print(f"  Input: {input_name} → Target: {target_name} ({n_clusters} classes)")
        print(f"{'=' * 70}")

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # 训练两个基础模型
        print("\n  Training base models...")

        rf1 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=self.n_jobs)
        rf1.fit(X1_train, y_train)

        rf2 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=self.n_jobs)
        rf2.fit(X2_train, y_train)

        # 获取概率预测
        prob1_train = rf1.predict_proba(X1_train)
        prob2_train = rf2.predict_proba(X2_train)
        prob1_test = rf1.predict_proba(X1_test)
        prob2_test = rf2.predict_proba(X2_test)

        # 合并概率作为 Meta 特征
        meta_train = np.hstack([prob1_train, prob2_train])
        meta_test = np.hstack([prob1_test, prob2_test])

        print(f"  Meta features: {meta_train.shape[1]}D")

        # 训练 Meta-classifier
        print("  Training meta-classifier...")
        start_time = time.time()

        meta_clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=self.n_jobs)
        meta_clf.fit(meta_train, y_train)

        tuning_time = time.time() - start_time

        # 评估
        y_train_pred = meta_clf.predict(meta_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')

        y_test_pred = meta_clf.predict(meta_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        overfit_gap = train_acc - test_acc

        print(f"\n  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
        print(f"  Overfit gap: {overfit_gap:.4f}")

        # 保存模型
        self.trained_models[task_name] = {'rf1': rf1, 'rf2': rf2, 'meta': meta_clf}

        return ClassificationResult(
            task_name=task_name,
            input_modality=input_name,
            target_modality=target_name,
            n_clusters=n_clusters,
            best_params={'method': 'late_fusion'},
            best_cv_score=0.0,
            tuning_time=tuning_time,
            train_accuracy=train_acc,
            train_f1_macro=train_f1,
            test_accuracy=test_acc,
            test_f1_macro=test_f1,
            overfit_gap=overfit_gap,
            confusion_matrix=confusion_matrix(y_test, y_test_pred),
            train_predictions=y_train_pred,
            train_true_labels=y_train,
            test_predictions=y_test_pred,
            test_true_labels=y_test,
        )

    def run_all_classification(self):
        """Run all classification tasks"""
        print("\n" + "=" * 80)
        print("Running Classification Tasks")
        print("=" * 80)

        results = {}

        # 获取每个模态的向量
        morph_train = self.morph_vectors[self.train_idx]
        morph_test = self.morph_vectors[self.test_idx]
        gene_train = self.gene_vectors[self.train_idx]
        gene_test = self.gene_vectors[self.test_idx]
        proj_train = self.proj_vectors[self.train_idx]
        proj_test = self.proj_vectors[self.test_idx]

        # 单模态任务
        single_tasks = [
            # Predict Projection
            ('morph_to_proj', 'Morph', 'Proj', morph_train, morph_test),
            ('gene_to_proj', 'Gene', 'Proj', gene_train, gene_test),
            # Predict Morphology
            ('gene_to_morph', 'Gene', 'Morph', gene_train, gene_test),
            ('proj_to_morph', 'Proj', 'Morph', proj_train, proj_test),
            # Predict Molecular
            ('morph_to_gene', 'Morph', 'Gene', morph_train, morph_test),
            ('proj_to_gene', 'Proj', 'Gene', proj_train, proj_test),
        ]

        for task_name, input_name, target_name, X_train, X_test in single_tasks:
            target_info = self.clustering_info[target_name]
            y_train = target_info.labels_train
            y_test = target_info.labels_test

            result = self.run_classification_task(
                X_train, X_test, y_train, y_test,
                task_name, input_name, target_name
            )
            results[task_name] = result

        # 多模态任务
        if self.use_late_fusion:
            print("\n" + "=" * 80)
            print("Late Fusion Tasks")
            print("=" * 80)

            fusion_tasks = [
                ('morph_gene_to_proj_fusion', 'Morph+Gene', 'Proj',
                 morph_train, morph_test, gene_train, gene_test),
                ('gene_proj_to_morph_fusion', 'Gene+Proj', 'Morph',
                 gene_train, gene_test, proj_train, proj_test),
                ('morph_proj_to_gene_fusion', 'Morph+Proj', 'Gene',
                 morph_train, morph_test, proj_train, proj_test),
            ]

            for task_name, input_name, target_name, X1_train, X1_test, X2_train, X2_test in fusion_tasks:
                target_info = self.clustering_info[target_name]
                y_train = target_info.labels_train
                y_test = target_info.labels_test

                result = self.run_late_fusion_task(
                    X1_train, X1_test, X2_train, X2_test,
                    y_train, y_test, task_name, input_name, target_name
                )
                results[task_name] = result

        # 也运行传统的特征拼接方法进行对比
        print("\n" + "=" * 80)
        print("Feature Concatenation Tasks (for comparison)")
        print("=" * 80)

        concat_tasks = [
            ('morph_gene_to_proj', 'Morph+Gene', 'Proj',
             np.hstack([morph_train, gene_train]), np.hstack([morph_test, gene_test])),
            ('gene_proj_to_morph', 'Gene+Proj', 'Morph',
             np.hstack([gene_train, proj_train]), np.hstack([gene_test, proj_test])),
            ('morph_proj_to_gene', 'Morph+Proj', 'Gene',
             np.hstack([morph_train, proj_train]), np.hstack([morph_test, proj_test])),
        ]

        for task_name, input_name, target_name, X_train, X_test in concat_tasks:
            target_info = self.clustering_info[target_name]
            y_train = target_info.labels_train
            y_test = target_info.labels_test

            result = self.run_classification_task(
                X_train, X_test, y_train, y_test,
                task_name, input_name, target_name
            )
            results[task_name] = result

        self.classification_results = results
        return results

    # ==================== Visualization ====================

    def visualize_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        self._plot_clustering_info(output_dir)
        self._plot_accuracy_comparison(output_dir)
        self._plot_fusion_comparison(output_dir)

        print(f"✓ Figures saved to: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_clustering_info(self, output_dir: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (name, info) in zip(axes, self.clustering_info.items()):
            sizes = info.cluster_sizes_train
            ax.bar(range(len(sizes)), sorted(sizes, reverse=True), color='#3498DB', alpha=0.7)
            ax.set_xlabel('Cluster (sorted)')
            ax.set_ylabel('Neurons (train)')
            ax.set_title(f'{name}: K={info.n_clusters}\n'
                         f'Sil(train)={info.silhouette_train:.3f}, Sil(test)={info.silhouette_test:.3f}',
                         fontweight='bold')
            ax.axhline(y=np.mean(sizes), color='red', linestyle='--')

        plt.suptitle('Cluster Size Distribution (Train Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_clustering_info.png")

    def _plot_accuracy_comparison(self, output_dir: str):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        groups = [
            ('Predict Projection', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('Predict Morphology', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('Predict Molecular', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        colors = ['#3498DB', '#27AE60', '#E74C3C']

        for ax, (title, tasks), color in zip(axes, groups, colors):
            available_tasks = [t for t in tasks if t in self.classification_results]
            if not available_tasks:
                continue

            labels = [self.classification_results[t].input_modality for t in available_tasks]
            train_acc = [self.classification_results[t].train_accuracy for t in available_tasks]
            test_acc = [self.classification_results[t].test_accuracy for t in available_tasks]

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, train_acc, width, label='Train', color=color, alpha=0.5)
            ax.bar(x + width / 2, test_acc, width, label='Test', color=color, alpha=0.9)

            n_clusters = self.classification_results[available_tasks[0]].n_clusters
            baseline = 1.0 / n_clusters
            ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Random ({baseline:.2f})')

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

        plt.suptitle('Classification Accuracy (Feature Concat)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_accuracy_comparison.png")

    def _plot_fusion_comparison(self, output_dir: str):
        """Compare late fusion vs feature concatenation"""
        if not self.use_late_fusion:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        comparisons = [
            ('→Proj', 'morph_gene_to_proj', 'morph_gene_to_proj_fusion'),
            ('→Morph', 'gene_proj_to_morph', 'gene_proj_to_morph_fusion'),
            ('→Gene', 'morph_proj_to_gene', 'morph_proj_to_gene_fusion'),
        ]

        x = np.arange(len(comparisons))
        width = 0.35

        concat_acc = []
        fusion_acc = []

        for _, concat_key, fusion_key in comparisons:
            if concat_key in self.classification_results:
                concat_acc.append(self.classification_results[concat_key].test_accuracy)
            else:
                concat_acc.append(0)
            if fusion_key in self.classification_results:
                fusion_acc.append(self.classification_results[fusion_key].test_accuracy)
            else:
                fusion_acc.append(0)

        bars1 = ax.bar(x - width / 2, concat_acc, width, label='Feature Concat', color='#3498DB')
        bars2 = ax.bar(x + width / 2, fusion_acc, width, label='Late Fusion', color='#E74C3C')

        for i, (c, f) in enumerate(zip(concat_acc, fusion_acc)):
            ax.annotate(f'{c:.3f}', xy=(i - width / 2, c), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=10)
            ax.annotate(f'{f:.3f}', xy=(i + width / 2, f), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=10)

        ax.set_ylabel('Test Accuracy')
        ax.set_title('Feature Concatenation vs Late Fusion', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in comparisons])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(concat_acc), max(fusion_acc)) * 1.15)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_fusion_comparison.png")

    # ==================== Save ====================

    def save_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)

        # Clustering info
        cluster_rows = []
        for name, info in self.clustering_info.items():
            cluster_rows.append({
                'modality': info.modality,
                'n_clusters': info.n_clusters,
                'silhouette_train': info.silhouette_train,
                'silhouette_test': info.silhouette_test,
            })
        pd.DataFrame(cluster_rows).to_csv(f"{output_dir}/clustering_info.csv", index=False)

        # Classification results
        rows = []
        for name, result in self.classification_results.items():
            rows.append({
                'task': result.task_name,
                'input': result.input_modality,
                'target': result.target_modality,
                'n_clusters': result.n_clusters,
                'cv_accuracy': result.best_cv_score,
                'train_accuracy': result.train_accuracy,
                'test_accuracy': result.test_accuracy,
                'test_f1_macro': result.test_f1_macro,
                'overfit_gap': result.overfit_gap,
            })
        pd.DataFrame(rows).to_csv(f"{output_dir}/classification_results.csv", index=False)

        # Full results
        with open(f"{output_dir}/full_results.pkl", 'wb') as f:
            pickle.dump({
                'clustering_info': self.clustering_info,
                'classification_results': self.classification_results,
            }, f)

        print(f"\n✓ Results saved to: {output_dir}")

    def run_full_pipeline(self, output_dir: str = "./classification_results_v3"):
        print("\n" + "=" * 80)
        print("Cluster Classification V3 (Fixed Version)")
        print("Key fix: Clustering fit on TRAIN only, predict on TEST")
        print("=" * 80)

        n = self.load_all_data()
        if n == 0:
            return

        self.find_optimal_clustering()
        self.run_all_classification()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print("\n【Clustering】")
        for name, info in self.clustering_info.items():
            print(f"  {name}: K={info.n_clusters}, "
                  f"Sil(train)={info.silhouette_train:.4f}, Sil(test)={info.silhouette_test:.4f}")

        print("\n【Classification Results】")
        print(f"{'Task':<30} {'Train':>8} {'Test':>8} {'Gap':>8} {'Baseline':>8}")
        print("-" * 70)

        for name, result in self.classification_results.items():
            baseline = 1.0 / result.n_clusters
            print(f"{name:<30} {result.train_accuracy:>8.4f} {result.test_accuracy:>8.4f} "
                  f"{result.overfit_gap:>8.4f} {baseline:>8.4f}")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./classification_results_v3"

    FIXED_K = {
        'Morph': 11,
        'Gene': 7,  # 降低 K 值，因为原来 K=25 时 silhouette 为负
        'Proj': 12,
    }

    with ClusterClassificationV3(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            test_ratio=0.2,
            fixed_k=FIXED_K,
            coarse_sample_ratio=0.2,  # 提高到 20%
            coarse_n_iter=150,
            fine_n_iter=50,
            cv_folds=5,
            n_jobs=-1,
            use_late_fusion=True,  # 启用后期融合
            equal_dim_for_fusion=None,  # 可设为 20 启用对等降维
            min_samples_per_class=5,
    ) as experiment:
        experiment.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()