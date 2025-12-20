"""
任务1：RandomForest 自动调参实验
===============================================
两阶段调参策略：
1. 粗搜索：10%数据 + 500次RandomizedSearchCV → 找大致范围
2. 精搜索：100%训练数据 + 100次 → 缩小范围精调
3. 最终评估：训练集 vs 测试集，防止过拟合

计算资源：128核CPU全开，GPU暂不用（sklearn RF不支持）

作者: PrometheusTT
日期: 2025-01-xx
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

warnings.filterwarnings('ignore')

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV, KFold, cross_val_predict, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, r2_score
from sklearn.base import clone

# 统计分析
from scipy.stats import pearsonr, uniform, randint

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

# 获取CPU核心数
N_CORES = multiprocessing.cpu_count()
print(f"检测到 {N_CORES} 个CPU核心")


# ==================== 自定义评分函数（向量化版本，更快）====================

def mean_sample_correlation_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算逐样本Pearson相关系数的均值（向量化版本，比循环快10x+）

    对于每个样本i，计算 corr(y_true[i, :], y_pred[i, :])
    """
    # 确保是2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    n_samples = y_true.shape[0]

    # 中心化
    y_true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    y_pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)

    # 计算标准差
    y_true_std = y_true_centered.std(axis=1)
    y_pred_std = y_pred_centered.std(axis=1)

    # 避免除以0
    valid_mask = (y_true_std > 1e-10) & (y_pred_std > 1e-10)

    if not valid_mask.any():
        return 0.0

    # 计算相关系数（向量化）
    # corr = sum(x_centered * y_centered) / (n * std_x * std_y)
    numerator = (y_true_centered * y_pred_centered).sum(axis=1)
    denominator = y_true.shape[1] * y_true_std * y_pred_std

    correlations = np.zeros(n_samples)
    correlations[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # 只对有效样本取均值
    return correlations[valid_mask].mean()


# 创建sklearn scorer
mean_corr_scorer = make_scorer(mean_sample_correlation_vectorized, greater_is_better=True)


@dataclass
class TuningResult:
    """调参结果"""
    task_name: str
    input_modality: str
    target_modality: str
    # 粗搜索结果
    coarse_best_params: Dict
    coarse_best_score: float
    coarse_time: float
    # 精搜索结果
    fine_best_params: Dict
    fine_best_score: float
    fine_time: float
    # 最终评估
    train_corr: float
    train_std: float
    test_corr: float
    test_std: float
    train_r2: float
    test_r2: float
    # 过拟合指标
    overfit_gap: float


@dataclass
class VectorInfo:
    name: str
    original_dims: int
    pca_dims: int
    variance_explained: float


class RFTuningExperiment:
    """RandomForest 调参实验"""

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

    # 粗搜索参数空间（使用连续分布）
    @staticmethod
    def get_coarse_param_distributions():
        """获取粗搜索参数分布（连续范围）"""
        from scipy.stats import randint, uniform

        common_params = {
            'n_estimators': randint(50, 501),        # [50, 500] 整数
            'max_depth': randint(2, 7),             # [5, 30] 整数
            'min_samples_split': randint(2, 51),     # [2, 50] 整数
            'min_samples_leaf': randint(1, 21),      # [1, 20] 整数
            'max_features': uniform(0.1, 0.9),       # [0.1, 1.0] 连续
        }

        return [
            {  # bootstrap=True: 可以用max_samples
                'bootstrap': [True],
                'max_samples': uniform(0.5, 0.5),    # [0.5, 1.0] 连续
                **common_params
            }
        ]

    def __init__(self, uri: str, user: str, password: str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 8.0,
                 pca_variance_threshold: float = 0.95,
                 test_ratio: float = 0.2,
                 coarse_sample_ratio: float = 0.1,
                 coarse_n_iter: int = 200,
                 fine_n_iter: int = 10,
                 cv_folds: int = 5,
                 n_jobs: int = -1):
        """
        参数:
            coarse_sample_ratio: 粗搜索使用的数据比例
            coarse_n_iter: 粗搜索迭代次数
            fine_n_iter: 精搜索迭代次数
            n_jobs: 并行核心数，-1表示全部
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio
        self.coarse_sample_ratio = coarse_sample_ratio
        self.coarse_n_iter = coarse_n_iter
        self.fine_n_iter = fine_n_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs if n_jobs > 0 else N_CORES

        # 并行策略：根据总核心数分配
        self.coarse_cv_jobs = max(4, self.n_jobs // 16)
        self.coarse_rf_jobs = max(8, self.n_jobs // self.coarse_cv_jobs)
        self.fine_cv_jobs = 2
        self.fine_rf_jobs = max(8, self.n_jobs // 2)

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

        self.vector_info: Dict[str, VectorInfo] = {}
        self.tuning_results: Dict[str, TuningResult] = {}

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
        print(f"  Morph: {self.morph_vectors.shape[1]}D, Gene: {self.gene_vectors.shape[1]}D, Proj: {self.proj_vectors.shape[1]}D")

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
        """处理向量（只在训练集上fit）"""
        print("\n处理向量...")
        neurons = self.valid_neuron_ids

        # 形态
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_info = self._process_vector(
            morph_raw, 'Morph', do_log=True)
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
        self.proj_vectors, proj_info = self._process_vector(
            proj_raw, 'Proj', do_log=True)
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

    # ==================== 调参核心 ====================

    def _create_fine_param_dist(self, coarse_best: Dict) -> List[Dict]:
        """
        基于粗搜索结果创建精搜索参数空间（缩小范围的连续分布）
        """
        from scipy.stats import randint, uniform

        # n_estimators: 在最优值±100范围内
        best_n = coarse_best.get('n_estimators', 200)
        n_low = max(50, best_n - 100)
        n_high = min(600, best_n + 100)

        # max_depth: 在最优值±5范围内
        best_depth = coarse_best.get('max_depth', 15)
        if best_depth is None:
            depth_low, depth_high = 20, 40
        else:
            depth_low = max(3, best_depth - 5)
            depth_high = min(40, best_depth + 5)

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
        if isinstance(best_feat, str):
            best_feat = 0.5  # sqrt/log2 近似为0.5
        feat_low = max(0.1, best_feat - 0.2)
        feat_high = min(1.0, best_feat + 0.2)

        common_params = {
            'n_estimators': randint(n_low, n_high + 1),
            'max_depth': randint(depth_low, depth_high + 1),
            'min_samples_split': randint(split_low, split_high),
            'min_samples_leaf': randint(leaf_low, leaf_high),
            'max_features': uniform(feat_low, feat_high - feat_low),
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

    def tune_single_task(self, X: np.ndarray, Y: np.ndarray,
                         task_name: str, input_name: str, target_name: str) -> TuningResult:
        """对单个任务进行两阶段调参"""
        print(f"\n{'='*70}")
        print(f"调参任务: {task_name}")
        print(f"  输入: {input_name} ({X.shape[1]}D) → 目标: {target_name} ({Y.shape[1]}D)")
        print(f"{'='*70}")

        # 获取训练数据
        X_train = X[self.train_idx]
        Y_train = Y[self.train_idx]
        X_test = X[self.test_idx]
        Y_test = Y[self.test_idx]

        print(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        # ========== 阶段1: 粗搜索 ==========
        print(f"\n--- 阶段1: 粗搜索 ({self.coarse_n_iter}次, {self.coarse_sample_ratio:.0%}数据) ---")

        # 采样子集
        n_coarse = int(len(X_train) * self.coarse_sample_ratio)
        coarse_idx = np.random.choice(len(X_train), n_coarse, replace=False)
        X_coarse = X_train[coarse_idx]
        Y_coarse = Y_train[coarse_idx]

        print(f"  粗搜索数据量: {n_coarse}")
        print(f"  并行: CV={self.coarse_cv_jobs}, RF={self.coarse_rf_jobs}")

        # 粗搜索
        start_time = time.time()

        # 显式创建带shuffle的KFold
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        coarse_rf = RandomForestRegressor(random_state=42, n_jobs=self.coarse_rf_jobs)
        coarse_search = RandomizedSearchCV(
            coarse_rf,
            param_distributions=self.get_coarse_param_distributions(),
            n_iter=self.coarse_n_iter,
            cv=cv,
            scoring=mean_corr_scorer,
            n_jobs=self.coarse_cv_jobs,
            random_state=42,
            verbose=1
        )
        coarse_search.fit(X_coarse, Y_coarse)

        coarse_time = time.time() - start_time
        coarse_best_params = coarse_search.best_params_
        coarse_best_score = coarse_search.best_score_

        print(f"\n  粗搜索完成! 用时: {coarse_time/60:.1f}分钟")
        print(f"  最佳得分: {coarse_best_score:.4f}")
        print(f"  最佳参数: {coarse_best_params}")

        # ========== 阶段2: 精搜索 ==========
        print(f"\n--- 阶段2: 精搜索 ({self.fine_n_iter}次, 100%训练数据) ---")

        fine_param_dist = self._create_fine_param_dist(coarse_best_params)
        print(f"  精搜索参数范围: 基于粗搜索结果缩小（连续分布）")
        print(f"  并行: CV={self.fine_cv_jobs}, RF={self.fine_rf_jobs}")

        start_time = time.time()

        # 显式创建带shuffle的KFold
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        fine_rf = RandomForestRegressor(random_state=42, n_jobs=self.fine_rf_jobs)
        fine_search = RandomizedSearchCV(
            fine_rf,
            param_distributions=fine_param_dist,
            n_iter=self.fine_n_iter,
            cv=cv,
            scoring=mean_corr_scorer,
            n_jobs=self.fine_cv_jobs,
            random_state=42,
            verbose=1
        )
        fine_search.fit(X_train, Y_train)

        fine_time = time.time() - start_time
        fine_best_params = fine_search.best_params_
        fine_best_score = fine_search.best_score_

        print(f"\n  精搜索完成! 用时: {fine_time/60:.1f}分钟")
        print(f"  最佳得分: {fine_best_score:.4f}")
        print(f"  最佳参数: {fine_best_params}")

        # ========== 最终评估 ==========
        print(f"\n--- 最终评估 ---")

        final_rf = RandomForestRegressor(**fine_best_params, random_state=42, n_jobs=self.n_jobs)
        final_rf.fit(X_train, Y_train)

        # 训练集评估
        Y_train_pred = final_rf.predict(X_train)
        train_corrs = self._compute_sample_correlations(Y_train, Y_train_pred)
        train_corr = np.mean(train_corrs)
        train_std = np.std(train_corrs)
        # 使用多输出R²（variance_weighted：按各输出方差加权）
        train_r2 = r2_score(Y_train, Y_train_pred, multioutput='variance_weighted')

        # 测试集评估
        Y_test_pred = final_rf.predict(X_test)
        test_corrs = self._compute_sample_correlations(Y_test, Y_test_pred)
        test_corr = np.mean(test_corrs)
        test_std = np.std(test_corrs)
        test_r2 = r2_score(Y_test, Y_test_pred, multioutput='variance_weighted')

        overfit_gap = train_corr - test_corr

        print(f"  训练集: Corr = {train_corr:.4f} ± {train_std:.4f}, R² = {train_r2:.4f}")
        print(f"  测试集: Corr = {test_corr:.4f} ± {test_std:.4f}, R² = {test_r2:.4f}")
        print(f"  过拟合差距: {overfit_gap:.4f}")

        if overfit_gap > 0.1:
            print(f"  ⚠️ 警告: 过拟合较严重!")

        result = TuningResult(
            task_name=task_name,
            input_modality=input_name,
            target_modality=target_name,
            coarse_best_params=coarse_best_params,
            coarse_best_score=coarse_best_score,
            coarse_time=coarse_time,
            fine_best_params=fine_best_params,
            fine_best_score=fine_best_score,
            fine_time=fine_time,
            train_corr=train_corr,
            train_std=train_std,
            test_corr=test_corr,
            test_std=test_std,
            train_r2=train_r2,
            test_r2=test_r2,
            overfit_gap=overfit_gap
        )

        return result

    def _compute_sample_correlations(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """计算逐样本相关系数（向量化版本）"""
        n_samples = Y_true.shape[0]

        # 中心化
        Y_true_centered = Y_true - Y_true.mean(axis=1, keepdims=True)
        Y_pred_centered = Y_pred - Y_pred.mean(axis=1, keepdims=True)

        # 计算标准差
        Y_true_std = Y_true_centered.std(axis=1)
        Y_pred_std = Y_pred_centered.std(axis=1)

        # 避免除以0
        valid_mask = (Y_true_std > 1e-10) & (Y_pred_std > 1e-10)

        # 计算相关系数
        numerator = (Y_true_centered * Y_pred_centered).sum(axis=1)
        denominator = Y_true.shape[1] * Y_true_std * Y_pred_std

        corrs = np.zeros(n_samples)
        corrs[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        return corrs

    # ==================== 运行所有任务 ====================

    def run_all_tuning(self):
        """运行所有9个调参任务"""
        print("\n" + "=" * 80)
        print("开始全部调参任务")
        print(f"CPU核心: {self.n_jobs}, 粗搜索: {self.coarse_n_iter}次, 精搜索: {self.fine_n_iter}次")
        print("=" * 80)

        total_start = time.time()

        # 定义所有任务
        tasks = [
            # 预测投射
            ('morph_to_proj', 'Morph', 'Proj', self.morph_vectors, self.proj_vectors),
            ('gene_to_proj', 'Gene', 'Proj', self.gene_vectors, self.proj_vectors),
            ('morph_gene_to_proj', 'Morph+Gene', 'Proj',
             np.hstack([self.morph_vectors, self.gene_vectors]), self.proj_vectors),

            # 预测形态
            ('gene_to_morph', 'Gene', 'Morph', self.gene_vectors, self.morph_vectors),
            ('proj_to_morph', 'Proj', 'Morph', self.proj_vectors, self.morph_vectors),
            ('gene_proj_to_morph', 'Gene+Proj', 'Morph',
             np.hstack([self.gene_vectors, self.proj_vectors]), self.morph_vectors),

            # 预测分子
            ('morph_to_gene', 'Morph', 'Gene', self.morph_vectors, self.gene_vectors),
            ('proj_to_gene', 'Proj', 'Gene', self.proj_vectors, self.gene_vectors),
            ('morph_proj_to_gene', 'Morph+Proj', 'Gene',
             np.hstack([self.morph_vectors, self.proj_vectors]), self.gene_vectors),
        ]

        results = {}

        for i, (task_name, input_name, target_name, X, Y) in enumerate(tasks):
            print(f"\n\n{'#'*80}")
            print(f"任务 {i+1}/{len(tasks)}: {task_name}")
            print(f"{'#'*80}")

            result = self.tune_single_task(X, Y, task_name, input_name, target_name)
            results[task_name] = result

            # 中间保存
            self.tuning_results = results
            self._save_intermediate_results()

        total_time = time.time() - total_start
        print(f"\n\n{'='*80}")
        print(f"全部调参完成! 总用时: {total_time/3600:.2f}小时")
        print(f"{'='*80}")

        self.tuning_results = results
        return results

    def _save_intermediate_results(self):
        """保存中间结果"""
        rows = []
        for name, result in self.tuning_results.items():
            row = {
                'task': result.task_name,
                'input': result.input_modality,
                'target': result.target_modality,
                'coarse_score': result.coarse_best_score,
                'fine_score': result.fine_best_score,
                'train_corr': result.train_corr,
                'test_corr': result.test_corr,
                'overfit_gap': result.overfit_gap,
            }
            # 添加参数
            for k, v in result.fine_best_params.items():
                row[f'param_{k}'] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv('tuning_intermediate.csv', index=False)

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n生成可视化...")

        self._plot_train_test_comparison(output_dir)
        self._plot_params_heatmap(output_dir)
        self._plot_multimodal_gain(output_dir)

        print(f"✓ 图表已保存到: {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _plot_train_test_comparison(self, output_dir: str):
        """训练集vs测试集对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        groups = [
            ('预测投射', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('预测形态', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('预测分子', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        for ax, (title, tasks) in zip(axes, groups):
            labels = [self.tuning_results[t].input_modality for t in tasks]
            train_corrs = [self.tuning_results[t].train_corr for t in tasks]
            test_corrs = [self.tuning_results[t].test_corr for t in tasks]
            train_stds = [self.tuning_results[t].train_std for t in tasks]
            test_stds = [self.tuning_results[t].test_std for t in tasks]

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width/2, train_corrs, width, yerr=train_stds, capsize=3,
                   label='Train', color='#3498DB', alpha=0.7)
            ax.bar(x + width/2, test_corrs, width, yerr=test_stds, capsize=3,
                   label='Test', color='#E74C3C', alpha=0.7)

            ax.set_ylabel('Correlation')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1)

        plt.suptitle('Train vs Test Performance (Tuned RF)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_train_test_comparison.png")

    def _plot_params_heatmap(self, output_dir: str):
        """参数热力图"""
        params_to_show = ['n_estimators', 'max_depth', 'min_samples_split',
                          'min_samples_leaf', 'max_features']

        fig, ax = plt.subplots(figsize=(12, 8))

        tasks = list(self.tuning_results.keys())
        data = []
        for task in tasks:
            result = self.tuning_results[task]
            row = []
            for param in params_to_show:
                val = result.fine_best_params.get(param, 0)
                if val is None:
                    val = -1  # 用-1表示None
                elif isinstance(val, str):
                    val = {'sqrt': 0.5, 'log2': 0.3}.get(val, 0.5)
                row.append(val)
            data.append(row)

        data = np.array(data)

        # 归一化显示
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

        sns.heatmap(data_norm, annot=data, fmt='.2f', cmap='YlOrRd',
                    xticklabels=params_to_show, yticklabels=tasks, ax=ax)

        ax.set_title('Best Parameters for Each Task', fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_params_heatmap.png")

    def _plot_multimodal_gain(self, output_dir: str):
        """多模态增益图"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        comparisons = [
            ('预测投射', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('预测形态', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('预测分子', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]

        colors = ['#E74C3C', '#3498DB', '#27AE60']

        for ax, (title, multi_key, single_keys), color in zip(axes, comparisons, colors):
            multi_corr = self.tuning_results[multi_key].test_corr
            single_corrs = [self.tuning_results[k].test_corr for k in single_keys]
            best_single = max(single_corrs)
            gain = multi_corr - best_single

            values = single_corrs + [multi_corr]
            labels = [self.tuning_results[k].input_modality for k in single_keys]
            labels.append(self.tuning_results[multi_key].input_modality)

            bars = ax.bar(range(len(values)), values, color=[color]*2 + [color],
                         alpha=[0.4, 0.4, 0.9])

            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                           xytext=(0, 3), textcoords='offset points', ha='center')

            ax.axhline(y=best_single, color='gray', linestyle='--', alpha=0.5)
            ax.text(0.5, 0.95, f'Gain: {gain:+.4f}', transform=ax.transAxes,
                   ha='center', va='top', fontweight='bold',
                   color='green' if gain > 0 else 'red')

            ax.set_title(title, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Multi-modal Gain (Test Set, Tuned RF)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_multimodal_gain.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存完整结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 主结果表
        rows = []
        for name, result in self.tuning_results.items():
            row = {
                'task': result.task_name,
                'input': result.input_modality,
                'target': result.target_modality,
                'coarse_score': result.coarse_best_score,
                'coarse_time_min': result.coarse_time / 60,
                'fine_score': result.fine_best_score,
                'fine_time_min': result.fine_time / 60,
                'train_corr': result.train_corr,
                'train_std': result.train_std,
                'test_corr': result.test_corr,
                'test_std': result.test_std,
                'train_r2': result.train_r2,
                'test_r2': result.test_r2,
                'overfit_gap': result.overfit_gap,
            }
            rows.append(row)

        pd.DataFrame(rows).to_csv(f"{output_dir}/tuning_results.csv", index=False)

        # 参数表
        param_rows = []
        for name, result in self.tuning_results.items():
            row = {'task': name}
            row.update(result.fine_best_params)
            param_rows.append(row)

        pd.DataFrame(param_rows).to_csv(f"{output_dir}/best_params.csv", index=False)

        # 保存详细结果（pickle）
        with open(f"{output_dir}/tuning_results.pkl", 'wb') as f:
            pickle.dump(self.tuning_results, f)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./tuning_results"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("RandomForest 自动调参实验（两阶段 + 连续分布）")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  CPU核心: {self.n_jobs}")
        print(f"  粗搜索: {self.coarse_sample_ratio:.0%}数据, {self.coarse_n_iter}次迭代")
        print(f"  精搜索: 100%数据, {self.fine_n_iter}次迭代")
        print(f"  交叉验证: {self.cv_folds}折")
        print(f"  测试集比例: {self.test_ratio:.0%}")
        print(f"  参数搜索: 连续分布（非离散枚举）")

        n = self.load_all_data()
        if n == 0:
            return

        self.run_all_tuning()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

    def _print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 80)
        print("调参结果汇总")
        print("=" * 80)

        print(f"\n{'任务':<25} {'Train':>10} {'Test':>10} {'Gap':>10} {'最佳参数'}")
        print("-" * 80)

        for name, result in self.tuning_results.items():
            params_str = f"n={result.fine_best_params.get('n_estimators')}, " \
                         f"d={result.fine_best_params.get('max_depth')}"
            print(f"{name:<25} {result.train_corr:>10.4f} {result.test_corr:>10.4f} "
                  f"{result.overfit_gap:>10.4f} {params_str}")

        # 多模态增益
        print(f"\n【多模态增益（测试集）】")
        for target, multi_key, single_keys in [
            ('投射', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('形态', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('分子', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]:
            multi = self.tuning_results[multi_key].test_corr
            best_single = max(self.tuning_results[k].test_corr for k in single_keys)
            gain = multi - best_single
            status = "✓" if gain > 0 else "✗"
            print(f"  预测{target}: {multi:.4f} vs {best_single:.4f} (gain: {gain:+.4f} {status})")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./tuning_results_v1"

    with RFTuningExperiment(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=4.0,
            pca_variance_threshold=0.95,
            test_ratio=0.2,
            coarse_sample_ratio=0.1,  # 10%数据做粗搜索
            coarse_n_iter=200,        # 粗搜索500次
            fine_n_iter=150,          # 精搜索100次
            cv_folds=5,
            n_jobs=-1                 # 全部CPU核心
    ) as experiment:
        experiment.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()