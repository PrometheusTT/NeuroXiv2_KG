"""
任务1：Neuron-level 多模态预测实验 (V12)
===============================================
基于V11修改：
1. 评估指标：逐样本 Pearson Correlation
2. 形态向量：去掉log，直接 Z-score -> PCA
3. 打印训练集和测试集结果，防止数据泄露
4. 三组实验：predict proj、predict morpho、predict mole

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

warnings.filterwarnings('ignore')

# 机器学习
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# 统计分析
from scipy.stats import pearsonr, wilcoxon, gaussian_kde

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# Neo4j
import neo4j

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
class PredictionResult:
    """预测结果数据类"""
    experiment_name: str
    input_modalities: str
    target_modality: str
    model_name: str
    # 训练集指标
    train_corrs: np.ndarray
    train_mean_corr: float
    train_std_corr: float
    train_r2: float
    # 测试集指标
    test_corrs: np.ndarray
    test_mean_corr: float
    test_std_corr: float
    test_r2: float
    # 元信息
    n_train: int
    n_test: int
    n_input_features: int
    n_target_features: int


@dataclass
class VectorInfo:
    """向量处理信息"""
    name: str
    original_dims: int
    pca_dims: int
    variance_explained: float
    pruned_dims: int
    log_transformed: bool


class NeuronMultimodalPredictorV12:
    """神经元多模态投射预测器 V12"""

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
                 search_radius: float = 8.0,
                 pca_variance_threshold: float = 0.95,
                 test_ratio: float = 0.2):
        """
        初始化预测器

        参数:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            data_dir: 数据目录
            database: 数据库名
            search_radius: 搜索半径（体素单位）
            pca_variance_threshold: PCA方差解释阈值
            test_ratio: 测试集比例
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.test_ratio = test_ratio

        # 神经元列表
        self.valid_neuron_ids: List[str] = []

        # 原始特征字典
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}

        # 全局维度
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # PCA处理后的向量
        self.morph_vectors: np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # 训练/测试集索引
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

        # 向量处理信息
        self.vector_info: Dict[str, VectorInfo] = {}

        # 结果
        self.results: Dict[str, PredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def compute_sample_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算单个样本的 Pearson 相关系数"""
        if len(y_true) < 2:
            return 0.0
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        if std_true == 0 or std_pred == 0:
            return 0.0
        corr, _ = pearsonr(y_true, y_pred)
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def compute_all_sample_correlations(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """计算所有样本的逐样本相关系数"""
        n_samples = Y_true.shape[0]
        corrs = np.zeros(n_samples)
        for i in range(n_samples):
            corrs[i] = NeuronMultimodalPredictorV12.compute_sample_correlation(
                Y_true[i], Y_pred[i])
        return corrs

    @staticmethod
    def should_log_transform(data: np.ndarray, ratio_threshold: float = 1000) -> bool:
        """判断是否需要log变换"""
        positive_vals = data[data > 0]
        if len(positive_vals) == 0:
            return False
        max_val = np.max(positive_vals)
        min_val = np.min(positive_vals)
        if min_val > 0:
            return (max_val / min_val) > ratio_threshold
        return False

    # ==================== 数据加载 ====================

    def load_all_data(self) -> int:
        """加载所有数据"""
        print("\n" + "=" * 80)
        print("加载神经元数据 (V12 - 逐样本Pearson Correlation)")
        print("=" * 80)

        # 1. 加载局部分子环境缓存
        self._load_local_gene_features_from_cache()

        # 2. 从Neo4j获取其他数据
        self._get_global_dimensions()
        self._load_all_neuron_features()

        # 3. 过滤有效神经元
        self._filter_valid_neurons()

        # 4. 划分训练/测试集
        self._split_train_test()

        # 5. 处理向量（只在训练集上fit）
        self._process_all_vectors()

        print(f"\n✓ 数据加载完成:")
        print(f"  有效神经元数: {len(self.valid_neuron_ids)}")
        print(f"  训练集: {len(self.train_idx)}, 测试集: {len(self.test_idx)}")
        print(f"  形态向量维度: {self.morph_vectors.shape[1]}")
        print(f"  分子向量维度: {self.gene_vectors.shape[1]}")
        print(f"  投射向量维度: {self.proj_vectors.shape[1]}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        """从缓存加载局部分子环境"""
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"

        if not cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

        print(f"\n加载局部分子环境缓存: {cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.local_gene_features_raw = cache_data['local_environments']
        self.all_subclasses = cache_data['all_subclasses']

        print(f"  加载了 {len(self.local_gene_features_raw)} 个神经元的局部分子环境")
        print(f"  Subclass维度: {len(self.all_subclasses)}")

    def _get_global_dimensions(self):
        """获取全局特征维度"""
        print("\n从Neo4j获取全局维度...")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]

        print(f"  投射目标: {len(self.all_target_regions)} 个脑区")

    def _load_all_neuron_features(self):
        """加载所有神经元特征"""
        print("\n从Neo4j加载神经元特征...")

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

            print(f"  Neo4j中有 {len(records)} 个同时有axon和dendrite的神经元")

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

        print(f"  加载了 {len(self.axon_features_raw)} 个神经元的形态特征")
        print(f"  加载了 {len(self.projection_vectors_raw)} 个神经元的投射向量")

    def _filter_valid_neurons(self):
        """过滤有效神经元"""
        print("\n过滤有效神经元...")

        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())

        self.valid_neuron_ids = sorted(list(candidates))
        print(f"  → 有效神经元: {len(self.valid_neuron_ids)}")

    def _split_train_test(self):
        """划分训练集和测试集"""
        print("\n划分训练/测试集...")

        n = len(self.valid_neuron_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        n_test = int(n * self.test_ratio)
        self.test_idx = indices[:n_test]
        self.train_idx = indices[n_test:]

        print(f"  训练集: {len(self.train_idx)} ({100 * (1 - self.test_ratio):.0f}%)")
        print(f"  测试集: {len(self.test_idx)} ({100 * self.test_ratio:.0f}%)")

    # ==================== 向量处理 ====================

    def _process_all_vectors(self):
        """处理所有向量"""
        print("\n" + "=" * 80)
        print("处理向量（只在训练集上fit，防止数据泄露）")
        print("=" * 80)

        neurons = self.valid_neuron_ids
        n = len(neurons)

        # ===== 1. 形态向量 =====
        print("\n【形态向量】")
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors, morph_info = self._process_morphology_vector(morph_raw)
        self.vector_info['morphology'] = morph_info

        # ===== 2. 分子向量 =====
        print("\n【分子向量】")
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        self.gene_vectors, gene_info = self._process_gene_vector(gene_raw)
        self.vector_info['gene'] = gene_info

        # ===== 3. 投射向量 =====
        print("\n【投射向量】")
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        self.proj_vectors, proj_info = self._process_projection_vector(proj_raw)
        self.vector_info['projection'] = proj_info

        # 打印汇总
        self._print_vector_summary()

    def _process_morphology_vector(self, morph_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """
        处理形态向量（V12: 去掉log）
        流程: Z-score按列 -> PCA 95%
        """
        original_dims = morph_raw.shape[1]
        print(f"  原始维度: {original_dims}")

        # Step 1: Z-score (只在训练集上fit)
        scaler = StandardScaler()
        scaler.fit(morph_raw[self.train_idx])
        morph_zscore = scaler.transform(morph_raw)
        print(f"  Step 1: Z-score标准化（在训练集上fit）")

        # Step 2: PCA (只在训练集上fit)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(morph_zscore[self.train_idx])
        morph_pca = pca.transform(morph_zscore)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"  Step 2: PCA -> {morph_pca.shape[1]} 维 (方差解释: {variance_explained:.2%})")

        info = VectorInfo(
            name='morphology',
            original_dims=original_dims,
            pca_dims=morph_pca.shape[1],
            variance_explained=variance_explained,
            pruned_dims=0,
            log_transformed=False  # V12: 不做log
        )

        return morph_pca, info

    def _process_gene_vector(self, gene_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """
        处理分子向量
        流程: 剪枝(去0 subclass) -> Z-score按列 -> PCA 95%
        """
        original_dims = gene_raw.shape[1]
        print(f"  原始维度: {original_dims}")

        # Step 1: 剪枝 - 基于训练集判断
        col_sums = gene_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        gene_pruned = gene_raw[:, valid_cols]
        n_pruned = original_dims - gene_pruned.shape[1]
        print(f"  Step 1: 剪枝 -> 去掉 {n_pruned} 个零值subclass，剩余 {gene_pruned.shape[1]} 维")

        # Step 2: Z-score (只在训练集上fit)
        scaler = StandardScaler()
        scaler.fit(gene_pruned[self.train_idx])
        gene_zscore = scaler.transform(gene_pruned)
        print(f"  Step 2: Z-score标准化（在训练集上fit）")

        # Step 3: PCA (只在训练集上fit)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(gene_zscore[self.train_idx])
        gene_pca = pca.transform(gene_zscore)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"  Step 3: PCA -> {gene_pca.shape[1]} 维 (方差解释: {variance_explained:.2%})")

        info = VectorInfo(
            name='gene',
            original_dims=original_dims,
            pca_dims=gene_pca.shape[1],
            variance_explained=variance_explained,
            pruned_dims=n_pruned,
            log_transformed=False
        )

        return gene_pca, info

    def _process_projection_vector(self, proj_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """
        处理投射向量
        流程: 剪枝(去0脑区) -> 自动log判断 -> Z-score按列 -> PCA 95%
        """
        original_dims = proj_raw.shape[1]
        print(f"  原始维度: {original_dims}")

        # Step 1: 剪枝 - 基于训练集
        col_sums = proj_raw[self.train_idx].sum(axis=0)
        valid_cols = col_sums > 0
        proj_pruned = proj_raw[:, valid_cols]
        n_pruned = original_dims - proj_pruned.shape[1]
        print(f"  Step 1: 剪枝 -> 去掉 {n_pruned} 个零值脑区，剩余 {proj_pruned.shape[1]} 维")

        # Step 2: 自动判断log (基于训练集)
        do_log = self.should_log_transform(proj_pruned[self.train_idx])
        if do_log:
            proj_transformed = np.log1p(proj_pruned)
            print(f"  Step 2: Log1p变换（值域跨度大）")
        else:
            proj_transformed = proj_pruned
            print(f"  Step 2: 跳过Log变换")

        # Step 3: Z-score (只在训练集上fit)
        scaler = StandardScaler()
        scaler.fit(proj_transformed[self.train_idx])
        proj_zscore = scaler.transform(proj_transformed)
        print(f"  Step 3: Z-score标准化（在训练集上fit）")

        # Step 4: PCA (只在训练集上fit)
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        pca.fit(proj_zscore[self.train_idx])
        proj_pca = pca.transform(proj_zscore)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"  Step 4: PCA -> {proj_pca.shape[1]} 维 (方差解释: {variance_explained:.2%})")

        info = VectorInfo(
            name='projection',
            original_dims=original_dims,
            pca_dims=proj_pca.shape[1],
            variance_explained=variance_explained,
            pruned_dims=n_pruned,
            log_transformed=do_log
        )

        return proj_pca, info

    def _print_vector_summary(self):
        """打印向量处理汇总"""
        print("\n" + "-" * 70)
        print("向量处理汇总:")
        print("-" * 70)
        print(f"{'向量':<12} {'原始维度':>10} {'剪枝后':>10} {'PCA维度':>10} {'方差解释':>12} {'Log变换':>10}")
        print("-" * 70)
        for name, info in self.vector_info.items():
            pruned = info.original_dims - info.pruned_dims
            print(f"{name:<12} {info.original_dims:>10} {pruned:>10} {info.pca_dims:>10} "
                  f"{info.variance_explained:>11.2%} {'是' if info.log_transformed else '否':>10}")

    # ==================== 模型训练 ====================

    def train_and_evaluate(self, X: np.ndarray, Y: np.ndarray,
                           experiment_name: str,
                           input_name: str, target_name: str,
                           model_name: str = "RF") -> PredictionResult:
        """
        训练模型并评估（分别报告训练集和测试集）
        """
        # 划分数据
        X_train, X_test = X[self.train_idx], X[self.test_idx]
        Y_train, Y_test = Y[self.train_idx], Y[self.test_idx]

        # 构建模型
        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 训练
        model.fit(X_train, Y_train)

        # 预测
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)

        # 计算逐样本相关系数
        train_corrs = self.compute_all_sample_correlations(Y_train, Y_train_pred)
        test_corrs = self.compute_all_sample_correlations(Y_test, Y_test_pred)

        # 计算 R²
        train_r2 = r2_score(Y_train.flatten(), Y_train_pred.flatten())
        test_r2 = r2_score(Y_test.flatten(), Y_test_pred.flatten())

        result = PredictionResult(
            experiment_name=experiment_name,
            input_modalities=input_name,
            target_modality=target_name,
            model_name=model_name,
            train_corrs=train_corrs,
            train_mean_corr=np.mean(train_corrs),
            train_std_corr=np.std(train_corrs),
            train_r2=train_r2,
            test_corrs=test_corrs,
            test_mean_corr=np.mean(test_corrs),
            test_std_corr=np.std(test_corrs),
            test_r2=test_r2,
            n_train=len(X_train),
            n_test=len(X_test),
            n_input_features=X.shape[1],
            n_target_features=Y.shape[1]
        )

        return result

    # ==================== 实验运行 ====================

    def run_all_experiments(self, model_name: str = "RF"):
        """运行所有实验"""
        print("\n" + "=" * 80)
        print(f"运行预测实验 (模型: {model_name})")
        print("=" * 80)

        X_morph = self.morph_vectors
        X_gene = self.gene_vectors
        Y_proj = self.proj_vectors

        results = {}

        # ===== 实验1: predict proj =====
        print("\n" + "=" * 60)
        print("【实验1】predict proj (Projection)")
        print("=" * 60)

        results['morph_to_proj'] = self.train_and_evaluate(
            X_morph, Y_proj, 'Exp1', 'Morph', 'Projection', model_name)
        self._print_result(results['morph_to_proj'])

        results['gene_to_proj'] = self.train_and_evaluate(
            X_gene, Y_proj, 'Exp1', 'Gene', 'Projection', model_name)
        self._print_result(results['gene_to_proj'])

        X_morph_gene = np.hstack([X_morph, X_gene])
        results['morph_gene_to_proj'] = self.train_and_evaluate(
            X_morph_gene, Y_proj, 'Exp1', 'Morph+Gene', 'Projection', model_name)
        self._print_result(results['morph_gene_to_proj'])

        # ===== 实验2: predict morpho =====
        print("\n" + "=" * 60)
        print("【实验2】predict morpho (Morphology)")
        print("=" * 60)

        results['gene_to_morph'] = self.train_and_evaluate(
            X_gene, X_morph, 'Exp2', 'Gene', 'Morphology', model_name)
        self._print_result(results['gene_to_morph'])

        results['proj_to_morph'] = self.train_and_evaluate(
            Y_proj, X_morph, 'Exp2', 'Proj', 'Morphology', model_name)
        self._print_result(results['proj_to_morph'])

        X_gene_proj = np.hstack([X_gene, Y_proj])
        results['gene_proj_to_morph'] = self.train_and_evaluate(
            X_gene_proj, X_morph, 'Exp2', 'Gene+Proj', 'Morphology', model_name)
        self._print_result(results['gene_proj_to_morph'])

        # ===== 实验3: predict mole =====
        print("\n" + "=" * 60)
        print("【实验3】predict mole (Gene)")
        print("=" * 60)

        results['morph_to_gene'] = self.train_and_evaluate(
            X_morph, X_gene, 'Exp3', 'Morph', 'Gene', model_name)
        self._print_result(results['morph_to_gene'])

        results['proj_to_gene'] = self.train_and_evaluate(
            Y_proj, X_gene, 'Exp3', 'Proj', 'Gene', model_name)
        self._print_result(results['proj_to_gene'])

        X_morph_proj = np.hstack([X_morph, Y_proj])
        results['morph_proj_to_gene'] = self.train_and_evaluate(
            X_morph_proj, X_gene, 'Exp3', 'Morph+Proj', 'Gene', model_name)
        self._print_result(results['morph_proj_to_gene'])

        self.results = results
        return results

    def _print_result(self, result: PredictionResult):
        """打印单个实验结果"""
        print(f"\n  {result.input_modalities} → {result.target_modality}")
        print(f"    输入维度: {result.n_input_features}, 输出维度: {result.n_target_features}")
        print(f"    训练集 (N={result.n_train}): Corr = {result.train_mean_corr:.4f} ± {result.train_std_corr:.4f}, R² = {result.train_r2:.4f}")
        print(f"    测试集 (N={result.n_test}):  Corr = {result.test_mean_corr:.4f} ± {result.test_std_corr:.4f}, R² = {result.test_r2:.4f}")

        # 过拟合检测
        overfit = result.train_mean_corr - result.test_mean_corr
        if overfit > 0.1:
            print(f"    ⚠️  过拟合警告: 训练-测试差异 = {overfit:.4f}")

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        """统计显著性分析"""
        print("\n" + "=" * 80)
        print("统计显著性分析（测试集）")
        print("=" * 80)

        stats = {}

        # 对每个实验组进行分析
        experiments = [
            ('predict proj', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('predict morpho', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('predict mole', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]

        for exp_name, dual_key, single_keys in experiments:
            print(f"\n【{exp_name}】")

            dual_corrs = self.results[dual_key].test_corrs
            dual_mean = self.results[dual_key].test_mean_corr

            for single_key in single_keys:
                single_corrs = self.results[single_key].test_corrs
                single_mean = self.results[single_key].test_mean_corr

                # Wilcoxon signed-rank test
                stat, p_val = wilcoxon(dual_corrs, single_corrs, alternative='greater')
                diff = dual_corrs - single_corrs
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                improve = dual_mean - single_mean

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

                single_name = self.results[single_key].input_modalities
                dual_name = self.results[dual_key].input_modalities

                print(f"  {dual_name} vs {single_name}: Δ={improve:+.4f}, p={p_val:.2e}{sig}, d={cohens_d:.3f}")

                stats[f'{dual_key}_vs_{single_key}'] = {
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'improvement': improve
                }

        return stats

    # ==================== 可视化 ====================

    def _save_figure(self, fig, output_dir: str, filename: str):
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def visualize_results(self, output_dir: str = "."):
        """生成所有可视化"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_train_test_comparison(output_dir)
        self._plot_experiment_summary(output_dir)
        self._plot_correlation_distribution(output_dir)
        self._plot_vector_info(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_train_test_comparison(self, output_dir: str):
        """训练集vs测试集对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        experiments = [
            ('predict proj', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('predict morpho', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('predict mole', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        for ax, (title, keys) in zip(axes, experiments):
            labels = [self.results[k].input_modalities for k in keys]
            train_means = [self.results[k].train_mean_corr for k in keys]
            test_means = [self.results[k].test_mean_corr for k in keys]
            train_stds = [self.results[k].train_std_corr for k in keys]
            test_stds = [self.results[k].test_std_corr for k in keys]

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds, capsize=3,
                          label='Train', color='#3498DB', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, capsize=3,
                          label='Test', color='#E74C3C', alpha=0.7, edgecolor='black')

            # 标注数值
            for bar, val in zip(bars1, train_means):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
            for bar, val in zip(bars2, test_means):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

            ax.set_ylabel('Pearson Correlation')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.legend(loc='upper left')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Train vs Test Performance (Pearson Correlation)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_train_test_comparison.png")

    def _plot_experiment_summary(self, output_dir: str):
        """实验汇总图（只看测试集）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        experiments = [
            ('predict proj', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj'], '#E74C3C'),
            ('predict morpho', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph'], '#3498DB'),
            ('predict mole', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene'], '#27AE60'),
        ]

        for ax, (title, keys, color) in zip(axes, experiments):
            labels = [self.results[k].input_modalities for k in keys]
            means = [self.results[k].test_mean_corr for k in keys]
            stds = [self.results[k].test_std_corr for k in keys]

            # 最后一个是多模态，颜色深一些
            colors = [color] * len(keys)
            alphas = [0.5, 0.5, 0.9]

            for i, (mean, std, alpha) in enumerate(zip(means, stds, alphas)):
                ax.bar(i, mean, yerr=std, capsize=4, color=colors[i], alpha=alpha,
                      edgecolor='black', linewidth=1.5)
                ax.annotate(f'{mean:.3f}', xy=(i, mean), xytext=(0, 8),
                           textcoords='offset points', ha='center', fontweight='bold')

            # 计算增益
            gain = means[-1] - max(means[:-1])
            ax.text(0.5, 0.95, f'Gain: {gain:+.4f}', transform=ax.transAxes,
                   ha='center', va='top', fontweight='bold',
                   color='#27AE60' if gain > 0 else '#E74C3C',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_ylabel('Pearson Correlation (Test)')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Multi-modal Prediction Performance (Test Set)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_experiment_summary.png")

    def _plot_correlation_distribution(self, output_dir: str):
        """相关系数分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        experiments = [
            ('predict proj', ['morph_to_proj', 'gene_to_proj', 'morph_gene_to_proj']),
            ('predict morpho', ['gene_to_morph', 'proj_to_morph', 'gene_proj_to_morph']),
            ('predict mole', ['morph_to_gene', 'proj_to_gene', 'morph_proj_to_gene']),
        ]

        colors = ['#3498DB', '#27AE60', '#E74C3C']

        for ax, (title, keys) in zip(axes, experiments):
            for key, color in zip(keys, colors):
                corrs = self.results[key].test_corrs
                label = f"{self.results[key].input_modalities} ({np.mean(corrs):.3f})"

                try:
                    kde = gaussian_kde(corrs)
                    x_range = np.linspace(corrs.min() - 0.1, corrs.max() + 0.1, 200)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                    ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
                except:
                    ax.hist(corrs, bins=30, alpha=0.5, color=color, label=label, density=True)

            ax.set_xlabel('Pearson Correlation')
            ax.set_ylabel('Density')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(axis='y', alpha=0.3)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Distribution of Sample-wise Correlations (Test Set)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_correlation_distribution.png")

    def _plot_vector_info(self, output_dir: str):
        """向量维度信息图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：维度变化
        ax1 = axes[0]
        names = ['Morphology', 'Gene', 'Projection']
        original = [self.vector_info['morphology'].original_dims,
                    self.vector_info['gene'].original_dims,
                    self.vector_info['projection'].original_dims]
        pca_dims = [self.vector_info['morphology'].pca_dims,
                    self.vector_info['gene'].pca_dims,
                    self.vector_info['projection'].pca_dims]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, original, width, label='Original', color='#BDC3C7', edgecolor='black')
        bars2 = ax1.bar(x + width/2, pca_dims, width, label='After PCA', color='#3498DB', edgecolor='black')

        ax1.set_ylabel('Dimensions')
        ax1.set_title('Dimensionality Reduction', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars1, original):
            ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar, val in zip(bars2, pca_dims):
            ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

        # 右图：方差解释
        ax2 = axes[1]
        variances = [self.vector_info['morphology'].variance_explained,
                     self.vector_info['gene'].variance_explained,
                     self.vector_info['projection'].variance_explained]

        bars = ax2.bar(names, variances, color=['#3498DB', '#27AE60', '#E74C3C'],
                      edgecolor='black', alpha=0.85)

        ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
        ax2.set_ylabel('Variance Explained')
        ax2.set_title('PCA Variance Explained', fontweight='bold')
        ax2.set_ylim(0.9, 1.0)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, variances):
            ax2.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

        for ax in axes:
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_vector_info.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果到CSV"""
        os.makedirs(output_dir, exist_ok=True)

        # 向量信息
        vector_rows = []
        for name, info in self.vector_info.items():
            vector_rows.append({
                'vector': name,
                'original_dims': info.original_dims,
                'pruned_dims': info.pruned_dims,
                'pca_dims': info.pca_dims,
                'variance_explained': info.variance_explained,
                'log_transformed': info.log_transformed
            })
        pd.DataFrame(vector_rows).to_csv(f"{output_dir}/vector_info.csv", index=False)

        # 预测结果
        if self.results:
            rows = []
            for key, result in self.results.items():
                rows.append({
                    'key': key,
                    'experiment': result.experiment_name,
                    'input': result.input_modalities,
                    'target': result.target_modality,
                    'n_train': result.n_train,
                    'n_test': result.n_test,
                    'n_input_features': result.n_input_features,
                    'n_target_features': result.n_target_features,
                    'train_mean_corr': result.train_mean_corr,
                    'train_std_corr': result.train_std_corr,
                    'train_r2': result.train_r2,
                    'test_mean_corr': result.test_mean_corr,
                    'test_std_corr': result.test_std_corr,
                    'test_r2': result.test_r2,
                })
            pd.DataFrame(rows).to_csv(f"{output_dir}/prediction_results.csv", index=False)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results_v12",
                          model_name: str = "RF"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态预测 (V12)")
        print(f"评估指标: 逐样本 Pearson Correlation")
        print(f"搜索半径: {self.search_radius} 体素 = {self.search_radius * 25}μm")
        print(f"PCA方差阈值: {self.pca_variance_threshold:.0%}")
        print(f"测试集比例: {self.test_ratio:.0%}")
        print("=" * 80)

        n = self.load_all_data()
        if n == 0:
            print("\n✗ 没有有效数据")
            return

        self.run_all_experiments(model_name)
        stats = self.statistical_analysis()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_conclusion()

        print("\n" + "=" * 80)
        print(f"完成! 结果: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self):
        """打印结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        # 向量信息
        print(f"\n【向量处理】")
        for name, info in self.vector_info.items():
            print(f"  {name}: {info.original_dims}D → {info.pca_dims}D ({info.variance_explained:.1%})")

        # 三组实验结果
        print(f"\n【预测结果汇总】(测试集)")
        print("-" * 80)
        print(f"{'Target':<12} {'Input':<12} {'Train Corr':>12} {'Test Corr':>12} {'Δ (overfit)':>12}")
        print("-" * 80)

        for target in ['Projection', 'Morphology', 'Gene']:
            relevant = [(k, v) for k, v in self.results.items()
                       if v.target_modality == target]
            for key, result in sorted(relevant, key=lambda x: -x[1].test_mean_corr):
                delta = result.train_mean_corr - result.test_mean_corr
                print(f"{target:<12} {result.input_modalities:<12} "
                     f"{result.train_mean_corr:>12.4f} {result.test_mean_corr:>12.4f} {delta:>12.4f}")

        # 多模态增益
        print(f"\n【多模态增益分析】(测试集)")
        print("-" * 60)

        experiments = [
            ('predict proj', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
            ('predict morpho', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ('predict mole', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
        ]

        for exp_name, dual_key, single_keys in experiments:
            dual_corr = self.results[dual_key].test_mean_corr
            single_corrs = [self.results[k].test_mean_corr for k in single_keys]
            best_single = max(single_corrs)
            gain = dual_corr - best_single
            status = "✓" if gain > 0 else "✗"

            print(f"  {exp_name}: {dual_corr:.4f} vs {best_single:.4f} (gain: {gain:+.4f} {status})")


def main():
    # ==================== 配置 ====================
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "task1_results_RF_100_99%"
    MODEL_NAME = "RF"
    SEARCH_RADIUS = 4.0
    PCA_VARIANCE = 0.99
    TEST_RATIO = 0.2

    # ==================== 运行 ====================
    with NeuronMultimodalPredictorV12(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=SEARCH_RADIUS,
            pca_variance_threshold=PCA_VARIANCE,
            test_ratio=TEST_RATIO
    ) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()