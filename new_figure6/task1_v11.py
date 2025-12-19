"""
任务1：Neuron-level 多模态预测实验 (V11 - 统一PCA向量处理)
===============================================
基于V10修改：
1. 只使用同时有axon和dendrite的神经元
2. 投射向量：剪枝(去0脑区) -> 自动log判断 -> Z-score按列 -> PCA 95%
3. 分子向量：剪枝(去0 subclass) -> Z-score按列 -> PCA 95%
4. 形态向量：log1p -> Z-score按列 -> PCA 95%
5. 所有预测和评估都使用PCA后的向量

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
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# 统计分析
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, pearsonr, gaussian_kde

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

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
    model_name: str
    condition_name: str
    dataset_name: str
    cosine_similarities: np.ndarray
    per_target_correlations: Dict[str, float]
    mean_cosine_sim: float
    std_cosine_sim: float
    mean_r2: float
    predicted: np.ndarray
    actual: np.ndarray
    n_neurons: int
    n_features: int
    n_target_dims: int


@dataclass
class CrossModalPredictionResult:
    """跨模态预测结果"""
    input_modalities: str
    target_modality: str
    model_name: str
    r2_score: float
    mean_corr: float
    std_corr: float
    per_feature_corrs: np.ndarray
    cosine_similarities: np.ndarray
    mean_cosine_sim: float
    std_cosine_sim: float
    n_neurons: int
    n_input_features: int
    n_target_features: int
    predicted: np.ndarray
    actual: np.ndarray


@dataclass
class VectorInfo:
    """向量处理信息"""
    name: str
    original_dims: int
    pca_dims: int
    variance_explained: float
    pruned_dims: int  # 被剪枝掉的维度数
    log_transformed: bool


class NeuronMultimodalPredictorV11:
    """神经元多模态投射预测器 V11（统一PCA向量处理）"""

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
                 pca_variance_threshold: float = 0.95):
        """
        初始化预测器

        参数:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            data_dir: 数据目录（包含缓存文件）
            database: 数据库名
            search_radius: 搜索半径（体素单位，1体素=25μm）
            pca_variance_threshold: PCA方差解释阈值
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold

        # 神经元列表（只保留同时有axon和dendrite的）
        self.valid_neuron_ids: List[str] = []

        # 原始特征字典
        self.axon_features_raw: Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}

        # 全局维度
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # PCA处理后的向量（核心数据）
        self.morph_vectors: np.ndarray = None  # [N × K_morph]
        self.gene_vectors: np.ndarray = None   # [N × K_gene]
        self.proj_vectors: np.ndarray = None   # [N × K_proj]

        # 向量处理信息
        self.vector_info: Dict[str, VectorInfo] = {}

        # 结果
        self.results: Dict[str, PredictionResult] = {}
        self.cross_modal_results: Dict[str, CrossModalPredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def compute_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_true = np.linalg.norm(y_true)
        norm_pred = np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            return 1 - cosine(y_true, y_pred)
        return 0.0

    @staticmethod
    def should_log_transform(data: np.ndarray, ratio_threshold: float = 1000) -> bool:
        """判断是否需要log变换"""
        # 只考虑正值
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
        print("加载神经元数据 (V11 - 统一PCA向量处理)")
        print("=" * 80)

        # 1. 加载局部分子环境缓存
        self._load_local_gene_features_from_cache()

        # 2. 从Neo4j获取其他数据
        self._get_global_dimensions()
        self._load_all_neuron_features()

        # 3. 过滤：只保留同时有axon和dendrite的神经元
        self._filter_valid_neurons()

        # 4. 处理向量（核心步骤）
        self._process_all_vectors()

        print(f"\n✓ 数据加载完成:")
        print(f"  有效神经元数: {len(self.valid_neuron_ids)}")
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

        # 构建查询
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

                # Axon features
                axon_feats = []
                for f in self.AXONAL_FEATURES:
                    val = record[f]
                    axon_feats.append(float(val) if val is not None else 0.0)
                self.axon_features_raw[neuron_id] = np.array(axon_feats)

                # Dendrite features
                dend_feats = []
                for f in self.DENDRITIC_FEATURES:
                    val = record[f]
                    dend_feats.append(float(val) if val is not None else 0.0)
                self.dendrite_features_raw[neuron_id] = np.array(dend_feats)

                # Projection vectors
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
        """过滤有效神经元：同时有axon、dendrite、local_env、projection"""
        print("\n过滤有效神经元...")

        candidates = set(self.axon_features_raw.keys())
        candidates &= set(self.dendrite_features_raw.keys())
        candidates &= set(self.local_gene_features_raw.keys())
        candidates &= set(self.projection_vectors_raw.keys())

        self.valid_neuron_ids = sorted(list(candidates))

        print(f"  有axon: {len(self.axon_features_raw)}")
        print(f"  有dendrite: {len(self.dendrite_features_raw)}")
        print(f"  有local_env: {len(self.local_gene_features_raw)}")
        print(f"  有projection: {len(self.projection_vectors_raw)}")
        print(f"  → 有效神经元: {len(self.valid_neuron_ids)}")

    # ==================== 向量处理（核心）====================

    def _process_all_vectors(self):
        """处理所有向量：标准化流程"""
        print("\n" + "=" * 80)
        print("处理向量（统一PCA流程）")
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
        print("\n" + "-" * 60)
        print("向量处理汇总:")
        print("-" * 60)
        print(f"{'向量':<12} {'原始维度':>10} {'剪枝后':>10} {'PCA维度':>10} {'方差解释':>12} {'Log变换':>10}")
        print("-" * 60)
        for name, info in self.vector_info.items():
            pruned = info.original_dims - info.pruned_dims
            print(f"{name:<12} {info.original_dims:>10} {pruned:>10} {info.pca_dims:>10} "
                  f"{info.variance_explained:>11.2%} {'是' if info.log_transformed else '否':>10}")

    def _process_morphology_vector(self, morph_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """
        处理形态向量
        流程: log1p -> Z-score按列 -> PCA 95%
        """
        original_dims = morph_raw.shape[1]
        print(f"  原始维度: {original_dims} (axon 32 + dendrite 32)")

        # Step 1: Log1p变换
        morph_log = np.log1p(morph_raw)
        print(f"  Step 1: Log1p变换")

        # Step 2: Z-score按列（每个特征独立标准化）
        scaler = StandardScaler()
        morph_zscore = scaler.fit_transform(morph_log)
        print(f"  Step 2: Z-score标准化（按列）")

        # Step 3: PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        morph_pca = pca.fit_transform(morph_zscore)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"  Step 3: PCA -> {morph_pca.shape[1]} 维 (方差解释: {variance_explained:.2%})")

        info = VectorInfo(
            name='morphology',
            original_dims=original_dims,
            pca_dims=morph_pca.shape[1],
            variance_explained=variance_explained,
            pruned_dims=0,
            log_transformed=True
        )

        return morph_pca, info

    def _process_gene_vector(self, gene_raw: np.ndarray) -> Tuple[np.ndarray, VectorInfo]:
        """
        处理分子向量
        流程: 剪枝(去0 subclass) -> Z-score按列 -> PCA 95%
        """
        original_dims = gene_raw.shape[1]
        print(f"  原始维度: {original_dims} subclasses")

        # Step 1: 剪枝 - 去掉总细胞数=0的subclass
        col_sums = gene_raw.sum(axis=0)
        valid_cols = col_sums > 0
        gene_pruned = gene_raw[:, valid_cols]
        n_pruned = original_dims - gene_pruned.shape[1]
        print(f"  Step 1: 剪枝 -> 去掉 {n_pruned} 个零值subclass，剩余 {gene_pruned.shape[1]} 维")

        # Step 2: Z-score按列（每个subclass独立标准化）
        scaler = StandardScaler()
        gene_zscore = scaler.fit_transform(gene_pruned)
        print(f"  Step 2: Z-score标准化（按列）")

        # Step 3: PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        gene_pca = pca.fit_transform(gene_zscore)
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
        print(f"  原始维度: {original_dims} 脑区")

        # Step 1: 剪枝 - 去掉总投射量=0的脑区
        col_sums = proj_raw.sum(axis=0)
        valid_cols = col_sums > 0
        proj_pruned = proj_raw[:, valid_cols]
        n_pruned = original_dims - proj_pruned.shape[1]
        print(f"  Step 1: 剪枝 -> 去掉 {n_pruned} 个零值脑区，剩余 {proj_pruned.shape[1]} 维")

        # Step 2: 自动判断是否需要log变换
        do_log = self.should_log_transform(proj_pruned)
        if do_log:
            proj_transformed = np.log1p(proj_pruned)
            print(f"  Step 2: Log1p变换（值域跨度大）")
        else:
            proj_transformed = proj_pruned
            print(f"  Step 2: 跳过Log变换（值域跨度小）")

        # Step 3: Z-score按列（每个脑区独立标准化）
        scaler = StandardScaler()
        proj_zscore = scaler.fit_transform(proj_transformed)
        print(f"  Step 3: Z-score标准化（按列）")

        # Step 4: PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        proj_pca = pca.fit_transform(proj_zscore)
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

    # ==================== 模型训练 ====================

    def train_and_predict(self, X: np.ndarray, Y: np.ndarray,
                          condition_name: str, dataset_name: str,
                          model_name: str = "RF",
                          n_folds: int = 5) -> PredictionResult:
        """训练并预测"""
        print(f"\n  训练 {condition_name}...")

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        n_samples = X.shape[0]
        actual_folds = max(3, min(n_folds, n_samples // 10, 10))

        kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # 计算指标
        cosine_sims = np.zeros(n_samples)
        for i in range(n_samples):
            cosine_sims[i] = self.compute_cosine_similarity(Y[i], Y_pred[i])

        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # Per-dimension correlation
        per_target_corrs = {}
        for j in range(Y.shape[1]):
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_target_corrs[f"dim_{j}"] = corr

        result = PredictionResult(
            model_name=model_name,
            condition_name=condition_name,
            dataset_name=dataset_name,
            cosine_similarities=cosine_sims,
            per_target_correlations=per_target_corrs,
            mean_cosine_sim=np.mean(cosine_sims),
            std_cosine_sim=np.std(cosine_sims),
            mean_r2=global_r2,
            predicted=Y_pred,
            actual=Y,
            n_neurons=n_samples,
            n_features=X.shape[1],
            n_target_dims=Y.shape[1]
        )

        print(f"    Cosine: {result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}, R²: {result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_projection_prediction_experiment(self, model_name: str = "RF"):
        """运行投射预测实验"""
        print("\n" + "=" * 80)
        print(f"【实验1】投射预测实验 (模型: {model_name})")
        print("=" * 80)

        X_morph = self.morph_vectors
        X_gene = self.gene_vectors
        Y_proj = self.proj_vectors
        n = len(self.valid_neuron_ids)

        print(f"\n数据维度: N={n}, Morph={X_morph.shape[1]}D, Gene={X_gene.shape[1]}D, Proj={Y_proj.shape[1]}D")

        results = {}

        # 单模态
        results['morph_only'] = self.train_and_predict(
            X_morph, Y_proj, 'Morph-only', 'full', model_name)
        results['gene_only'] = self.train_and_predict(
            X_gene, Y_proj, 'Gene-only', 'full', model_name)

        # 多模态
        X_multi = np.hstack([X_morph, X_gene])
        results['morph_gene'] = self.train_and_predict(
            X_multi, Y_proj, 'Morph+Gene', 'full', model_name)

        # Shuffle对照
        np.random.seed(42)
        X_gene_shuffled = X_gene.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph, X_gene_shuffled])
        results['shuffle'] = self.train_and_predict(
            X_shuffle, Y_proj, 'Shuffle', 'full', model_name)

        self.results = results
        return results

    def run_cross_modal_prediction_experiment(self, model_name: str = "RF"):
        """运行跨模态预测实验"""
        print("\n" + "=" * 80)
        print(f"【实验2】跨模态预测实验 (模型: {model_name})")
        print("=" * 80)

        X_morph = self.morph_vectors
        X_gene = self.gene_vectors
        Y_proj = self.proj_vectors
        n = len(self.valid_neuron_ids)

        print(f"\n数据维度: N={n}, Morph={X_morph.shape[1]}D, Gene={X_gene.shape[1]}D, Proj={Y_proj.shape[1]}D")

        cross_results = {}

        # ===== 预测 Projection =====
        print("\n--- Target: Projection ---")
        X_morph_gene = np.hstack([X_morph, X_gene])
        cross_results['morph_gene_to_proj'] = self._train_cross_modal(
            X_morph_gene, Y_proj, 'Morph+Gene', 'Projection', model_name)
        cross_results['morph_to_proj'] = self._train_cross_modal(
            X_morph, Y_proj, 'Morph', 'Projection', model_name)
        cross_results['gene_to_proj'] = self._train_cross_modal(
            X_gene, Y_proj, 'Gene', 'Projection', model_name)

        # ===== 预测 Gene =====
        print("\n--- Target: Gene ---")
        X_morph_proj = np.hstack([X_morph, Y_proj])
        cross_results['morph_proj_to_gene'] = self._train_cross_modal(
            X_morph_proj, X_gene, 'Morph+Proj', 'Gene', model_name)
        cross_results['morph_to_gene'] = self._train_cross_modal(
            X_morph, X_gene, 'Morph', 'Gene', model_name)
        cross_results['proj_to_gene'] = self._train_cross_modal(
            Y_proj, X_gene, 'Proj', 'Gene', model_name)

        # ===== 预测 Morphology =====
        print("\n--- Target: Morphology ---")
        X_gene_proj = np.hstack([X_gene, Y_proj])
        cross_results['gene_proj_to_morph'] = self._train_cross_modal(
            X_gene_proj, X_morph, 'Gene+Proj', 'Morphology', model_name)
        cross_results['gene_to_morph'] = self._train_cross_modal(
            X_gene, X_morph, 'Gene', 'Morphology', model_name)
        cross_results['proj_to_morph'] = self._train_cross_modal(
            Y_proj, X_morph, 'Proj', 'Morphology', model_name)

        self.cross_modal_results = cross_results
        return cross_results

    def _train_cross_modal(self, X: np.ndarray, Y: np.ndarray,
                           input_name: str, target_name: str,
                           model_name: str = "RF") -> CrossModalPredictionResult:
        """跨模态预测训练"""
        print(f"\n  {input_name} → {target_name}...")

        n_samples = X.shape[0]

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        n_folds = max(3, min(5, n_samples // 10, 10))
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # Cosine Similarity（逐样本）
        cosine_sims = np.zeros(n_samples)
        for i in range(n_samples):
            cosine_sims[i] = self.compute_cosine_similarity(Y[i], Y_pred[i])

        # R²
        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # Per-dimension Correlation
        n_target_features = Y.shape[1]
        per_feature_corrs = np.zeros(n_target_features)
        for j in range(n_target_features):
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_feature_corrs[j] = corr if not np.isnan(corr) else 0

        mean_corr = np.mean(per_feature_corrs)
        std_corr = np.std(per_feature_corrs)

        print(f"    Cosine: {np.mean(cosine_sims):.4f} ± {np.std(cosine_sims):.4f}")
        print(f"    R²: {global_r2:.4f}, Mean Dim Corr: {mean_corr:.4f}")

        return CrossModalPredictionResult(
            input_modalities=input_name,
            target_modality=target_name,
            model_name=model_name,
            r2_score=global_r2,
            mean_corr=mean_corr,
            std_corr=std_corr,
            per_feature_corrs=per_feature_corrs,
            cosine_similarities=cosine_sims,
            mean_cosine_sim=np.mean(cosine_sims),
            std_cosine_sim=np.std(cosine_sims),
            n_neurons=n_samples,
            n_input_features=X.shape[1],
            n_target_features=n_target_features,
            predicted=Y_pred,
            actual=Y
        )

    def run_all_experiments(self, model_name: str = "RF"):
        """运行所有实验"""
        self.run_projection_prediction_experiment(model_name)
        self.run_cross_modal_prediction_experiment(model_name)
        return self.results, self.cross_modal_results

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        """统计显著性分析"""
        print("\n" + "=" * 80)
        print("统计显著性分析")
        print("=" * 80)

        stats = {}

        if 'morph_only' not in self.results:
            return stats

        print(f"\n【投射预测】")

        sim_morph = self.results['morph_only'].cosine_similarities
        sim_gene = self.results['gene_only'].cosine_similarities
        sim_multi = self.results['morph_gene'].cosine_similarities
        sim_shuffle = self.results['shuffle'].cosine_similarities

        comparisons = [
            ('Morph+Gene vs Morph', sim_multi, sim_morph),
            ('Morph+Gene vs Gene', sim_multi, sim_gene),
            ('Morph+Gene vs Shuffle', sim_multi, sim_shuffle),
            ('Shuffle vs Morph', sim_shuffle, sim_morph),
        ]

        for name, sim1, sim2 in comparisons:
            stat, p_val = wilcoxon(sim1, sim2, alternative='greater')
            diff = sim1 - sim2
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            improve = (np.mean(sim1) - np.mean(sim2)) / np.mean(sim2) * 100 if np.mean(sim2) > 0 else 0

            stats[name] = {
                'p_value': p_val, 'cohens_d': cohens_d, 'improvement_pct': improve
            }

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {name}: p={p_val:.2e}{sig}, d={cohens_d:.3f}, Δ={improve:.1f}%")

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

        self._plot_projection_prediction_summary(output_dir)
        self._plot_cross_modal_summary(output_dir)
        self._plot_delta_distribution(output_dir)
        self._plot_vector_info(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_projection_prediction_summary(self, output_dir: str):
        """投射预测汇总图"""
        if not self.results:
            return

        conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        keys = ['morph_only', 'gene_only', 'morph_gene', 'shuffle']
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        fig, ax = plt.subplots(figsize=(10, 6))

        means = [self.results[k].mean_cosine_sim for k in keys]
        stds = [self.results[k].std_cosine_sim for k in keys]

        bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, means):
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontweight='bold', fontsize=11)

        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'Projection Prediction (N={self.results["morph_only"].n_neurons})\n'
                     f'Morph: {self.vector_info["morphology"].pca_dims}D, '
                     f'Gene: {self.vector_info["gene"].pca_dims}D, '
                     f'Proj: {self.vector_info["projection"].pca_dims}D',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_projection_prediction.png")

    def _plot_cross_modal_summary(self, output_dir: str):
        """跨模态预测汇总"""
        if not self.cross_modal_results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        configs = [
            ('Projection', ['morph_gene_to_proj', 'morph_to_proj', 'gene_to_proj'],
             ['Morph+Gene', 'Morph', 'Gene'], '#E74C3C'),
            ('Gene', ['morph_proj_to_gene', 'morph_to_gene', 'proj_to_gene'],
             ['Morph+Proj', 'Morph', 'Proj'], '#27AE60'),
            ('Morphology', ['gene_proj_to_morph', 'gene_to_morph', 'proj_to_morph'],
             ['Gene+Proj', 'Gene', 'Proj'], '#3498DB'),
        ]

        for ax, (target, keys, labels, color) in zip(axes, configs):
            cosines = [self.cross_modal_results[k].mean_cosine_sim for k in keys]
            stds = [self.cross_modal_results[k].std_cosine_sim for k in keys]
            alphas = [0.9, 0.5, 0.5]

            for i, (cos, std, alpha) in enumerate(zip(cosines, stds, alphas)):
                ax.bar(i, cos, yerr=std, capsize=4, color=color, alpha=alpha, edgecolor='black')
                ax.annotate(f'{cos:.3f}', xy=(i, cos), xytext=(0, 8),
                            textcoords='offset points', ha='center', fontweight='bold')

            gain = cosines[0] - max(cosines[1:])
            ax.text(0.5, 0.95, f'Gain: {gain:+.4f}', transform=ax.transAxes,
                    ha='center', va='top', fontweight='bold',
                    color='#27AE60' if gain > 0 else '#E74C3C',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f'Predict: {target}', fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Cross-Modal Prediction (Cosine Similarity)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_cross_modal_cosine.png")

    def _plot_delta_distribution(self, output_dir: str):
        """Delta分布图"""
        if 'morph_only' not in self.results:
            return

        sim_morph = self.results['morph_only'].cosine_similarities
        sim_gene = self.results['gene_only'].cosine_similarities
        sim_multi = self.results['morph_gene'].cosine_similarities
        sim_shuffle = self.results['shuffle'].cosine_similarities

        fig, ax = plt.subplots(figsize=(12, 5))

        for delta, label, color in [
            (sim_multi - sim_morph, f'Multi−Morph (med={np.median(sim_multi - sim_morph):.3f})', '#3498DB'),
            (sim_multi - sim_gene, f'Multi−Gene (med={np.median(sim_multi - sim_gene):.3f})', '#27AE60'),
            (sim_shuffle - sim_morph, f'Shuffle−Morph (med={np.median(sim_shuffle - sim_morph):.3f})', '#95A5A6'),
        ]:
            try:
                kde = gaussian_kde(delta)
                x_range = np.linspace(delta.min() - 0.02, delta.max() + 0.02, 500)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, label=label)
                ax.fill_between(x_range[x_range >= 0], 0, kde(x_range[x_range >= 0]),
                                color=color, alpha=0.15)
            except:
                ax.hist(delta, bins=50, alpha=0.5, color=color, label=label, density=True)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Δ Cosine Similarity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Performance Gain Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_delta_distribution.png")

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

        # 投射预测结果
        if self.results:
            rows = []
            for key, result in self.results.items():
                rows.append({
                    'key': key,
                    'condition': result.condition_name,
                    'n_neurons': result.n_neurons,
                    'n_input_features': result.n_features,
                    'n_target_dims': result.n_target_dims,
                    'mean_cosine_sim': result.mean_cosine_sim,
                    'std_cosine_sim': result.std_cosine_sim,
                    'r2_score': result.mean_r2,
                })
            pd.DataFrame(rows).to_csv(f"{output_dir}/projection_prediction.csv", index=False)

        # 跨模态预测结果
        if self.cross_modal_results:
            cross_rows = []
            for key, result in self.cross_modal_results.items():
                cross_rows.append({
                    'key': key,
                    'input': result.input_modalities,
                    'target': result.target_modality,
                    'r2_score': result.r2_score,
                    'mean_corr': result.mean_corr,
                    'mean_cosine_sim': result.mean_cosine_sim,
                    'std_cosine_sim': result.std_cosine_sim,
                    'n_neurons': result.n_neurons,
                    'n_input_features': result.n_input_features,
                    'n_target_features': result.n_target_features
                })
            pd.DataFrame(cross_rows).to_csv(f"{output_dir}/cross_modal_prediction.csv", index=False)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results_v11",
                          model_name: str = "RF"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态预测 (V11 - 统一PCA向量处理)")
        print(f"搜索半径: {self.search_radius} 体素 = {self.search_radius * 25}μm")
        print(f"PCA方差阈值: {self.pca_variance_threshold:.0%}")
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
        print(f"  半径: {self.search_radius * 25}μm, PCA阈值: {self.pca_variance_threshold:.0%}")
        for name, info in self.vector_info.items():
            print(f"  {name}: {info.original_dims}D → {info.pca_dims}D ({info.variance_explained:.1%})")

        # 投射预测
        if self.results:
            print(f"\n【实验1: 投射预测】(N={self.results['morph_only'].n_neurons})")
            morph = self.results['morph_only'].mean_cosine_sim
            gene = self.results['gene_only'].mean_cosine_sim
            multi = self.results['morph_gene'].mean_cosine_sim
            shuffle = self.results['shuffle'].mean_cosine_sim

            print(f"  Morph: {morph:.4f}, Gene: {gene:.4f}, Multi: {multi:.4f}, Shuffle: {shuffle:.4f}")

            if morph > 0:
                improve = (multi - morph) / morph * 100
                print(f"  Multi vs Morph: {multi - morph:+.4f} ({improve:+.1f}%)")
            if gene > 0:
                improve = (multi - gene) / gene * 100
                print(f"  Multi vs Gene: {multi - gene:+.4f} ({improve:+.1f}%)")

        # 跨模态预测
        if self.cross_modal_results:
            print(f"\n【实验2: 跨模态预测】")
            print("-" * 70)
            print(f"  {'Target':<12} {'Input':<12} {'Cosine':>10} {'R²':>10}")
            print("-" * 70)

            for target in ['Projection', 'Gene', 'Morphology']:
                relevant = [(k, v) for k, v in self.cross_modal_results.items()
                            if v.target_modality == target]
                for key, result in sorted(relevant, key=lambda x: -x[1].mean_cosine_sim):
                    print(f"  {target:<12} {result.input_modalities:<12} "
                          f"{result.mean_cosine_sim:>10.4f} {result.r2_score:>10.4f}")

            # 增益分析
            print(f"\n【双模态增益分析】")
            for target, dual_key, single_keys in [
                ('Projection', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
                ('Gene', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
                ('Morphology', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ]:
                if dual_key not in self.cross_modal_results:
                    continue

                dual_cos = self.cross_modal_results[dual_key].mean_cosine_sim
                single_cos = [self.cross_modal_results[k].mean_cosine_sim for k in single_keys]
                gain = dual_cos - max(single_cos)
                status = "✓" if gain > 0 else "✗"

                print(f"  预测 {target}: {dual_cos:.4f} (gain: {gain:+.4f} {status})")


def main():
    # ==================== 配置 ====================
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./task1_results_RF_cosine_100_pca95"
    MODEL_NAME = "RF"
    SEARCH_RADIUS = 4.0
    PCA_VARIANCE = 0.95

    # ==================== 运行 ====================
    with NeuronMultimodalPredictorV11(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=SEARCH_RADIUS,
            pca_variance_threshold=PCA_VARIANCE
    ) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()