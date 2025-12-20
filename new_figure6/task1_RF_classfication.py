"""
任务1：聚类标签预测实验
===============================================
核心思路：
1. 对每个目标模态独立聚类 → 得到标签
2. 用其他模态预测该标签（分类任务）
3. 评估准确率

实验流程：
1. 搜索最优聚类数K（用silhouette score）
2. 对每个预测任务进行RF分类器调参
3. 对比单模态 vs 多模态的预测准确率

作者: PrometheusTT
日期: 2025-01-xx
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

warnings.filterwarnings('ignore')

# 机器学习
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    silhouette_score, confusion_matrix
)

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# Neo4j
import neo4j

# 设置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()
print(f"检测到 {N_CORES} 个CPU核心")


@dataclass
class ClusteringInfo:
    """聚类信息"""
    modality: str
    n_clusters: int
    silhouette: float
    labels: np.ndarray
    cluster_sizes: np.ndarray


@dataclass
class ClassificationResult:
    """分类结果"""
    task_name: str
    input_modality: str
    target_modality: str
    n_clusters: int
    # 调参结果
    best_params: Dict
    best_cv_score: float
    tuning_time: float
    # 训练集评估
    train_accuracy: float
    train_f1_macro: float
    train_f1_weighted: float
    # 测试集评估
    test_accuracy: float
    test_f1_macro: float
    test_f1_weighted: float
    # 过拟合
    overfit_gap: float
    # 混淆矩阵（测试集）
    confusion_matrix: np.ndarray


@dataclass
class VectorInfo:
    name: str
    original_dims: int
    pca_dims: int
    variance_explained: float


class ClusterClassificationExperiment:
    """聚类标签分类实验"""

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

    # RF分类器参数空间（使用连续分布）
    # 注意：使用scipy.stats分布实现连续范围搜索
    @staticmethod
    def get_param_distributions():
        """获取参数分布（连续范围）"""
        from scipy.stats import randint, uniform

        # 共享参数（连续分布）
        common_params = {
            'n_estimators': randint(50, 501),        # [50, 500] 整数
            'max_depth': randint(2, 7),              # [2, 6] 整数
            'min_samples_split': randint(2, 51),     # [2, 50] 整数
            'min_samples_leaf': randint(1, 21),      # [1, 20] 整数
            'max_features': uniform(0.1, 0.9),       # [0.1, 1.0] 连续
            'class_weight': ['balanced', 'balanced_subsample', None],
        }

        return [
            {  # bootstrap=True: 可以用max_samples
                'bootstrap': [True],
                'max_samples': uniform(0.5, 0.5),    # [0.5, 1.0] 连续
                **common_params
            },
            {  # bootstrap=False: 不能用max_samples
                'bootstrap': [False],
                **common_params
            }
        ]

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 8.0,
                 pca_variance_threshold: float = 0.95,
                 test_ratio: float = 0.2,
                 k_candidates: List[int] = None,
                 coarse_sample_ratio: float = 0.1,
                 coarse_n_iter: int = 500,
                 fine_n_iter: int = 100,
                 cv_folds: int = 5,
                 n_jobs: int = -1):
        """
        参数:
            k_candidates: 候选聚类数目列表
            coarse_sample_ratio: 粗搜索数据比例
            coarse_n_iter: 粗搜索迭代次数
            fine_n_iter: 精搜索迭代次数
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.k_candidates = k_candidates or [5, 10, 15, 20, 30, 50]
        self.coarse_sample_ratio = coarse_sample_ratio
        self.coarse_n_iter = coarse_n_iter
        self.fine_n_iter = fine_n_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs if n_jobs > 0 else N_CORES

        # 并行策略：根据总核心数分配
        # 粗搜索：小数据，多任务并行
        # 精搜索：大数据，单任务内并行
        self.coarse_cv_jobs = max(4, self.n_jobs // 16)  # CV并行数
        self.coarse_rf_jobs = max(8, self.n_jobs // self.coarse_cv_jobs)  # RF并行数
        self.fine_cv_jobs = 2  # 精搜索CV并行少
        self.fine_rf_jobs = max(8, self.n_jobs // 2)  # RF并行多

        print(f"并行策略:")
        print(f"  粗搜索: CV并行={self.coarse_cv_jobs}, RF并行={self.coarse_rf_jobs}")
        print(f"  精搜索: CV并行={self.fine_cv_jobs}, RF并行={self.fine_rf_jobs}")

        # 数据
        self.valid_neuron_ids: List[str] = []
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # PCA向量
        self.morph_vectors: np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # 训练/测试集索引
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

        # 聚类结果
        self.clustering_info: Dict[str, ClusteringInfo] = {}

        # 分类结果
        self.classification_results: Dict[str, ClassificationResult] = {}

        self.vector_info: Dict[str, VectorInfo] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 ====================

    def load_all_data(self) -> int:
        """加载数据"""
        print("\n" + "=" * 80)
        print("加载数据")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._filter_valid_neurons()
        self._split_train_test()
        self._process_all_vectors()

        print(f"\n✓ 数据加载完成:")
        print(f"  神经元总数: {len(self.valid_neuron_ids)}")
        print(f"  训练集: {len(self.train_idx)}, 测试集: {len(self.test_idx)}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        self.local_gene_features_raw = cache_data['local_environments']
        self.all_subclasses = cache_data['all_subclasses']
        print(f"  加载了 {len(self.local_gene_features_raw)} 个神经元的分子环境")

    def _get_global_dimensions(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]
        print(f"  投射目标: {len(self.all_target_regions)} 个脑区")

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

        print(f"  加载了 {len(self.axon_features_raw)} 个神经元")

    def _filter_valid_neurons(self):
        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())
        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  有效神经元: {len(self.valid_neuron_ids)}")

    def _split_train_test(self):
        """划分训练/测试集"""
        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]

        print(f"  训练集: {len(self.train_idx)}, 测试集: {len(self.test_idx)}")

    def _process_all_vectors(self):
        """处理向量"""
        print("\n处理向量...")
        neurons = self.valid_neuron_ids

        # 形态
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_info = self._process_vector(morph_raw, 'Morph', do_log=True)
        self.vector_info['morph'] = morph_info

        # 分子
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        gene_raw = gene_raw[:, col_sums > 0]
        self.gene_vectors, gene_info = self._process_vector(gene_raw, 'Gene')
        self.vector_info['gene'] = gene_info

        # 投射
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        proj_raw = proj_raw[:, col_sums > 0]
        self.proj_vectors, proj_info = self._process_vector(proj_raw, 'Proj', do_log=True)
        self.vector_info['proj'] = proj_info

    def _process_vector(self, X_raw: np.ndarray, name: str,
                        do_log: bool = False) -> Tuple[np.ndarray, VectorInfo]:
        original_dims = X_raw.shape[1]

        if do_log:
            X = np.log1p(X_raw)
        else:
            X = X_raw

        # 只在训练集上fit
        scaler = StandardScaler()
        scaler.fit(X[self.train_idx])
        X_scaled = scaler.transform(X)

        # PCA只在训练集上fit
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(X_scaled[self.train_idx])
        X_pca = pca.transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  {name}: {original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        return X_pca, VectorInfo(name, original_dims, X_pca.shape[1], variance)

    # ==================== 聚类 ====================

    def find_optimal_clustering(self):
        """为每个模态找到最优聚类数"""
        print("\n" + "=" * 80)
        print("搜索最优聚类数K")
        print(f"候选K值: {self.k_candidates}")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        for name, vectors in modalities:
            print(f"\n--- {name} ---")
            best_k, best_sil, best_labels = self._search_best_k(vectors, name)

            cluster_sizes = np.bincount(best_labels)

            self.clustering_info[name] = ClusteringInfo(
                modality=name,
                n_clusters=best_k,
                silhouette=best_sil,
                labels=best_labels,
                cluster_sizes=cluster_sizes
            )

            print(f"  最优K={best_k}, Silhouette={best_sil:.4f}")
            print(f"  类别大小: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
                  f"mean={cluster_sizes.mean():.1f}")

    def _search_best_k(self, X: np.ndarray, name: str) -> Tuple[int, float, np.ndarray]:
        """搜索最优K值"""
        # 只在训练集上确定K
        X_train = X[self.train_idx]

        best_k = self.k_candidates[0]
        best_sil = -1
        best_labels_train = None

        results = []

        for k in self.k_candidates:
            # 多次运行取平均
            sils = []
            for seed in range(5):
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                labels = kmeans.fit_predict(X_train)
                sil = silhouette_score(X_train, labels)
                sils.append(sil)

            avg_sil = np.mean(sils)
            results.append((k, avg_sil))
            print(f"    K={k}: Silhouette={avg_sil:.4f}")

            if avg_sil > best_sil:
                best_sil = avg_sil
                best_k = k

        # 用最优K在全部数据上聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        all_labels = kmeans.fit_predict(X)

        return best_k, best_sil, all_labels

    # ==================== 分类 ====================

    def _create_fine_param_dist(self, coarse_best: Dict) -> List[Dict]:
        """基于粗搜索结果创建精搜索参数空间（缩小范围的连续分布）"""
        from scipy.stats import randint, uniform

        # n_estimators: 在最优值±100范围内
        best_n = coarse_best.get('n_estimators', 200)
        n_low = max(50, best_n - 100)
        n_high = min(600, best_n + 100)

        # max_depth: 在最优值±1范围内
        best_depth = coarse_best.get('max_depth', 4)
        depth_low = max(2, best_depth - 1)
        depth_high = min(7, best_depth + 2)  # randint是[low, high)

        # min_samples_split: 在最优值附近
        best_split = coarse_best.get('min_samples_split', 10)
        split_low = max(2, int(best_split * 0.5))
        split_high = min(100, int(best_split * 2) + 1)

        # min_samples_leaf: 在最优值附近
        best_leaf = coarse_best.get('min_samples_leaf', 4)
        leaf_low = max(1, int(best_leaf * 0.5))
        leaf_high = min(50, int(best_leaf * 2) + 1)

        # max_features: 在最优值±0.2范围内
        best_feat = coarse_best.get('max_features', 0.5)
        feat_low = max(0.1, best_feat - 0.2)
        feat_high = min(1.0, best_feat + 0.2)

        common_params = {
            'n_estimators': randint(n_low, n_high + 1),
            'max_depth': randint(depth_low, depth_high),
            'min_samples_split': randint(split_low, split_high),
            'min_samples_leaf': randint(leaf_low, leaf_high),
            'max_features': uniform(feat_low, feat_high - feat_low),
            'class_weight': [coarse_best.get('class_weight', 'balanced')],  # 固定
        }

        best_bootstrap = coarse_best.get('bootstrap', True)

        if best_bootstrap:
            best_samples = coarse_best.get('max_samples', 0.8)
            if best_samples is None:
                best_samples = 1.0
            samples_low = max(0.5, best_samples - 0.15)
            samples_high = min(1.0, best_samples + 0.15)

            return [{
                'bootstrap': [True],
                'max_samples': uniform(samples_low, samples_high - samples_low),
                **common_params
            }]
        else:
            return [{
                'bootstrap': [False],
                **common_params
            }]

    def run_classification_task(self, X: np.ndarray, y: np.ndarray,
                                 task_name: str, input_name: str,
                                 target_name: str, n_clusters: int) -> ClassificationResult:
        """运行单个分类任务（两阶段调参）"""
        print(f"\n{'='*70}")
        print(f"分类任务: {task_name}")
        print(f"  输入: {input_name} ({X.shape[1]}D) → 目标: {target_name} ({n_clusters}类)")
        print(f"{'='*70}")

        X_train = X[self.train_idx]
        y_train = y[self.train_idx]
        X_test = X[self.test_idx]
        y_test = y[self.test_idx]

        print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
        print(f"  类别分布(训练): {np.bincount(y_train)}")

        # ========== 阶段1: 粗搜索 ==========
        print(f"\n--- 阶段1: 粗搜索 ({self.coarse_n_iter}次, {self.coarse_sample_ratio:.0%}数据) ---")

        # 采样子集
        n_coarse = int(len(X_train) * self.coarse_sample_ratio)
        np.random.seed(42)
        coarse_idx = np.random.choice(len(X_train), n_coarse, replace=False)
        X_coarse = X_train[coarse_idx]
        y_coarse = y_train[coarse_idx]

        print(f"  粗搜索数据量: {n_coarse}")
        print(f"  并行: CV={self.coarse_cv_jobs}, RF={self.coarse_rf_jobs}")

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

        print(f"\n  粗搜索完成! 用时: {coarse_time/60:.1f}分钟")
        print(f"  CV最佳准确率: {coarse_best_score:.4f}")
        print(f"  最佳参数: {coarse_best_params}")

        # ========== 阶段2: 精搜索 ==========
        print(f"\n--- 阶段2: 精搜索 ({self.fine_n_iter}次, 100%训练数据) ---")

        fine_param_dist = self._create_fine_param_dist(coarse_best_params)
        print(f"  精搜索参数范围: 基于粗搜索结果缩小")
        print(f"  并行: CV={self.fine_cv_jobs}, RF={self.fine_rf_jobs}")

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

        print(f"\n  精搜索完成! 用时: {fine_time/60:.1f}分钟")
        print(f"  CV最佳准确率: {best_cv_score:.4f}")
        print(f"  最佳参数: {best_params}")

        total_tuning_time = coarse_time + fine_time

        # ========== 最终评估 ==========
        print(f"\n--- 最终评估 ---")

        final_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=self.n_jobs)
        final_rf.fit(X_train, y_train)

        # 训练集
        y_train_pred = final_rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
        train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

        # 测试集
        y_test_pred = final_rf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

        conf_matrix = confusion_matrix(y_test, y_test_pred)

        overfit_gap = train_acc - test_acc

        print(f"  训练集: Acc={train_acc:.4f}, F1_macro={train_f1_macro:.4f}")
        print(f"  测试集: Acc={test_acc:.4f}, F1_macro={test_f1_macro:.4f}")
        print(f"  过拟合差距: {overfit_gap:.4f}")

        if overfit_gap > 0.1:
            print(f"  ⚠️ 警告: 过拟合较严重!")

        # 随机基线
        random_baseline = 1.0 / n_clusters
        print(f"  随机基线: {random_baseline:.4f}")
        print(f"  超过基线: {test_acc - random_baseline:+.4f}")

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
            confusion_matrix=conf_matrix
        )

    def run_all_classification(self):
        """运行所有分类任务"""
        print("\n" + "=" * 80)
        print("运行分类任务")
        print("=" * 80)

        results = {}

        # 定义任务
        tasks = [
            # 预测投射类别
            ('morph_to_proj', 'Morph', 'Proj', self.morph_vectors),
            ('gene_to_proj', 'Gene', 'Proj', self.gene_vectors),
            ('morph_gene_to_proj', 'Morph+Gene', 'Proj',
             np.hstack([self.morph_vectors, self.gene_vectors])),

            # 预测形态类别
            ('gene_to_morph', 'Gene', 'Morph', self.gene_vectors),
            ('proj_to_morph', 'Proj', 'Morph', self.proj_vectors),
            ('gene_proj_to_morph', 'Gene+Proj', 'Morph',
             np.hstack([self.gene_vectors, self.proj_vectors])),

            # 预测分子类别
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

            # 中间保存
            self.classification_results = results
            self._save_intermediate_results()

        self.classification_results = results
        return results

    def _save_intermediate_results(self):
        """保存中间结果"""
        rows = []
        for name, result in self.classification_results.items():
            row = {
                'task': result.task_name,
                'input': result.input_modality,
                'target': result.target_modality,
                'n_clusters': result.n_clusters,
                'cv_accuracy': result.best_cv_score,
                'train_accuracy': result.train_accuracy,
                'test_accuracy': result.test_accuracy,
                'test_f1_macro': result.test_f1_macro,
                'overfit_gap': result.overfit_gap,
            }
            rows.append(row)

        pd.DataFrame(rows).to_csv('classification_intermediate.csv', index=False)

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("生成可视化")
        print("=" * 80)

        self._plot_clustering_info(output_dir)
        self._plot_accuracy_comparison(output_dir)
        self._plot_multimodal_gain(output_dir)
        self._plot_confusion_matrices(output_dir)

        print(f"✓ 图表已保存到: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_clustering_info(self, output_dir: str):
        """聚类信息图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (name, info) in zip(axes, self.clustering_info.items()):
            sizes = info.cluster_sizes
            ax.bar(range(len(sizes)), sorted(sizes, reverse=True), color='#3498DB', alpha=0.7)
            ax.set_xlabel('Cluster (sorted by size)')
            ax.set_ylabel('Number of neurons')
            ax.set_title(f'{name}: K={info.n_clusters}, Sil={info.silhouette:.3f}',
                        fontweight='bold')
            ax.axhline(y=np.mean(sizes), color='red', linestyle='--',
                      label=f'Mean={np.mean(sizes):.1f}')
            ax.legend()

        plt.suptitle('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_clustering_info.png")

    def _plot_accuracy_comparison(self, output_dir: str):
        """准确率对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        groups = [
            ('预测投射类别', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('预测形态类别', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('预测分子类别', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        colors = ['#3498DB', '#27AE60', '#E74C3C']

        for ax, (title, tasks), color in zip(axes, groups, colors):
            labels = [self.classification_results[t].input_modality for t in tasks]
            train_acc = [self.classification_results[t].train_accuracy for t in tasks]
            test_acc = [self.classification_results[t].test_accuracy for t in tasks]

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width/2, train_acc, width, label='Train', color=color, alpha=0.5)
            ax.bar(x + width/2, test_acc, width, label='Test', color=color, alpha=0.9)

            # 随机基线
            n_clusters = self.classification_results[tasks[0]].n_clusters
            baseline = 1.0 / n_clusters
            ax.axhline(y=baseline, color='gray', linestyle='--',
                      label=f'Random ({baseline:.2f})')

            for i, (tr, te) in enumerate(zip(train_acc, test_acc)):
                ax.annotate(f'{te:.2f}', xy=(i + width/2, te), xytext=(0, 3),
                           textcoords='offset points', ha='center', fontsize=9)

            ax.set_ylabel('Accuracy')
            ax.set_title(f'{title} (K={n_clusters})', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.legend(loc='upper left', fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Classification Accuracy: Train vs Test', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_accuracy_comparison.png")

    def _plot_multimodal_gain(self, output_dir: str):
        """多模态增益图"""
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
            ax.annotate(f'{gain:+.3f}', xy=(bar.get_x() + bar.get_width()/2, gain + offset),
                       ha='center', va=va, fontweight='bold')

        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_ylabel('Accuracy Gain (Multi - Best Single)')
        ax.set_title('Multi-modal Classification Gain (Test Set)', fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_multimodal_gain.png")

    def _plot_confusion_matrices(self, output_dir: str):
        """混淆矩阵图（选择多模态的三个任务）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        multi_tasks = ['morph_gene_to_proj', 'gene_proj_to_morph', 'morph_proj_to_gene']
        titles = ['Morph+Gene → Proj', 'Gene+Proj → Morph', 'Morph+Proj → Gene']

        for ax, task, title in zip(axes, multi_tasks, titles):
            result = self.classification_results[task]
            cm = result.confusion_matrix

            # 归一化
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm_norm, ax=ax, cmap='Blues', annot=False,
                       cbar_kws={'shrink': 0.5})
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{title}\nAcc={result.test_accuracy:.3f}', fontweight='bold')

        plt.suptitle('Confusion Matrices (Normalized, Test Set)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_confusion_matrices.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 聚类信息
        cluster_rows = []
        for name, info in self.clustering_info.items():
            cluster_rows.append({
                'modality': info.modality,
                'n_clusters': info.n_clusters,
                'silhouette': info.silhouette,
                'min_size': info.cluster_sizes.min(),
                'max_size': info.cluster_sizes.max(),
                'mean_size': info.cluster_sizes.mean(),
            })
        pd.DataFrame(cluster_rows).to_csv(f"{output_dir}/clustering_info.csv", index=False)

        # 分类结果
        rows = []
        for name, result in self.classification_results.items():
            row = {
                'task': result.task_name,
                'input': result.input_modality,
                'target': result.target_modality,
                'n_clusters': result.n_clusters,
                'cv_accuracy': result.best_cv_score,
                'train_accuracy': result.train_accuracy,
                'train_f1_macro': result.train_f1_macro,
                'test_accuracy': result.test_accuracy,
                'test_f1_macro': result.test_f1_macro,
                'test_f1_weighted': result.test_f1_weighted,
                'overfit_gap': result.overfit_gap,
                'tuning_time_min': result.tuning_time / 60,
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{output_dir}/classification_results.csv", index=False)

        # 最佳参数
        param_rows = []
        for name, result in self.classification_results.items():
            row = {'task': name}
            row.update(result.best_params)
            param_rows.append(row)
        pd.DataFrame(param_rows).to_csv(f"{output_dir}/best_params.csv", index=False)

        # 保存详细结果
        with open(f"{output_dir}/full_results.pkl", 'wb') as f:
            pickle.dump({
                'clustering_info': self.clustering_info,
                'classification_results': self.classification_results,
            }, f)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./classification_results"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("聚类标签分类实验（两阶段调参）")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  CPU核心: {self.n_jobs}")
        print(f"  候选K值: {self.k_candidates}")
        print(f"  粗搜索: {self.coarse_sample_ratio:.0%}数据, {self.coarse_n_iter}次迭代")
        print(f"  精搜索: 100%数据, {self.fine_n_iter}次迭代")
        print(f"  max_depth范围: [2, 6]")
        print(f"  参数搜索: 连续分布（非离散枚举）")

        n = self.load_all_data()
        if n == 0:
            return

        self.find_optimal_clustering()
        self.run_all_classification()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

    def _print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 80)
        print("实验结果汇总")
        print("=" * 80)

        # 聚类信息
        print(f"\n【聚类信息】")
        for name, info in self.clustering_info.items():
            print(f"  {name}: K={info.n_clusters}, Silhouette={info.silhouette:.4f}")

        # 分类结果
        print(f"\n【分类结果（测试集）】")
        print(f"{'任务':<25} {'输入':<12} {'目标':<8} {'K':>4} {'准确率':>10} {'F1':>10} {'基线':>8}")
        print("-" * 80)

        for name, result in self.classification_results.items():
            baseline = 1.0 / result.n_clusters
            print(f"{name:<25} {result.input_modality:<12} {result.target_modality:<8} "
                  f"{result.n_clusters:>4} {result.test_accuracy:>10.4f} "
                  f"{result.test_f1_macro:>10.4f} {baseline:>8.4f}")

        # 多模态增益
        print(f"\n【多模态增益（测试集准确率）】")
        for target, multi_key, single_keys in [
            ('投射', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('形态', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('分子', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]:
            multi = self.classification_results[multi_key].test_accuracy
            best_single = max(self.classification_results[k].test_accuracy for k in single_keys)
            gain = multi - best_single
            status = "✓" if gain > 0 else "✗"
            print(f"  预测{target}: {multi:.4f} vs {best_single:.4f} (gain: {gain:+.4f} {status})")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./classification_results_v1"

    with ClusterClassificationExperiment(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            test_ratio=0.2,
            k_candidates=[5, 10, 15, 20, 30, 50],  # 候选聚类数
            coarse_sample_ratio=0.1,  # 10%数据做粗搜索
            coarse_n_iter=200,        # 粗搜索500次
            fine_n_iter=15,          # 精搜索100次
            cv_folds=5,
            n_jobs=-1                 # 使用全部CPU
    ) as experiment:
        experiment.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()