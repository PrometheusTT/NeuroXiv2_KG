"""
任务1：Neuron-level 多模态预测实验 (V10 - 使用精确局部分子环境)
===============================================
基于V9修改：
1.使用soma周围200μm内MERFISH细胞的subclass分布作为分子环境
2.加载预计算的局部分子环境缓存
3.跨模态预测添加Cosine Similarity指标

作者:PrometheusTT
日期:2025-07-30
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score

# 统计分析
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import wilcoxon, pearsonr, gaussian_kde, spearmanr

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
    model_name:str
    condition_name:str
    dataset_name:str
    cosine_similarities:np.ndarray
    jsd_similarities:np.ndarray
    per_target_correlations:Dict[str, float]
    mean_cosine_sim:float
    std_cosine_sim:float
    mean_jsd_sim:float
    std_jsd_sim:float
    mean_r2:float
    predicted:np.ndarray
    actual:np.ndarray
    n_neurons:int
    n_features:int


@dataclass
class CrossModalPredictionResult:
    """跨模态预测结果 - 增强版"""
    input_modalities:str
    target_modality:str
    model_name:str
    # R² 和 相关性
    r2_score:float
    mean_corr:float
    std_corr:float
    per_feature_corrs:np.ndarray
    # Cosine Similarity（逐样本）
    cosine_similarities:np.ndarray
    mean_cosine_sim:float
    std_cosine_sim:float
    # JSD（仅对投射有效）
    jsd_similarities:np.ndarray
    mean_jsd_sim:float
    std_jsd_sim:float
    # 元信息
    n_neurons:int
    n_input_features:int
    n_target_features:int
    predicted:np.ndarray
    actual:np.ndarray


class NeuronMultimodalPredictorV10:
    """神经元多模态投射预测器 V10（使用精确局部分子环境）"""

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

    def __init__(self, uri:str, user:str, password:str,
                 data_dir:str, database:str = "neo4j",
                 search_radius:float = 8.0):
        """
        初始化预测器

        参数:
            uri:Neo4j连接URI
            user:用户名
            password:密码
            data_dir: 数据目录（包含缓存文件）
            database:数据库名
            search_radius: 搜索半径（体素单位，1体素=25μm）
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius

        # 神经元列表
        self.all_neuron_ids:List[str] = []
        self.neurons_with_dendrite:List[str] = []

        # 特征字典
        self.axon_features:Dict[str, np.ndarray] = {}
        self.dendrite_features:Dict[str, np.ndarray] = {}
        self.local_gene_features:Dict[str, np.ndarray] = {}  # 局部分子环境
        self.projection_vectors:Dict[str, np.ndarray] = {}

        # 全局维度
        self.all_subclasses:List[str] = []
        self.all_target_regions:List[str] = []
        self.n_subclasses:int = 0

        # 结果
        self.results:Dict[str, PredictionResult] = {}
        self.cross_modal_results:Dict[str, CrossModalPredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def clr_transform(X:np.ndarray, pseudocount:float = 1e-6) -> np.ndarray:
        """CLR变换"""
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def compute_jsd_similarity(y_true:np.ndarray, y_pred:np.ndarray) -> float:
        """JSD相似度"""
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        sum_true = y_true.sum()
        sum_pred = y_pred.sum()
        if sum_true > 0:
            y_true = y_true / sum_true
        if sum_pred > 0:
            y_pred = y_pred / sum_pred
        if sum_true == 0 or sum_pred == 0:
            return 0.0
        jsd = jensenshannon(y_true, y_pred, base=2)
        if np.isnan(jsd):
            return 0.0
        return 1.0 - jsd

    @staticmethod
    def compute_cosine_similarity(y_true:np.ndarray, y_pred:np.ndarray) -> float:
        """计算余弦相似度"""
        norm_true = np.linalg.norm(y_true)
        norm_pred = np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            return 1 - cosine(y_true, y_pred)
        return 0.0

    # ==================== 数据加载 ====================

    def load_all_data(self) -> Tuple[int, int]:
        """加载所有数据"""
        print("\n" + "=" * 80)
        print("加载神经元数据 (V10 - 使用精确局部分子环境)")
        print("=" * 80)

        # 1.加载局部分子环境缓存
        self._load_local_gene_features_from_cache()

        # 2.从Neo4j获取其他数据
        self._get_global_dimensions()
        self._get_valid_neurons()
        self._load_axon_features()
        self._load_dendrite_features()
        self._load_projection_vectors()

        # 3.过滤有效神经元
        self._filter_valid_neurons()

        print(f"\n✓ 数据加载完成:")
        print(f"  有局部环境的神经元: {len(self.local_gene_features)}")
        print(f"  有效神经元（全部数据）:{len(self.all_neuron_ids)}")
        print(f"  有dendrite的神经元:{len(self.neurons_with_dendrite)}")

        return len(self.all_neuron_ids), len(self.neurons_with_dendrite)

    def _load_local_gene_features_from_cache(self):
        """从缓存加载局部分子环境"""
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"

        if not cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在:{cache_file}")

        print(f"\n加载局部分子环境缓存: {cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # 解析缓存数据
        self.local_gene_features = cache_data['environments']
        self.all_subclasses = cache_data['subclasses']
        self.n_subclasses = len(self.all_subclasses)

        # 统计
        n_neurons = len(self.local_gene_features)
        print(f"  加载了 {n_neurons} 个神经元的局部分子环境")
        print(f"  Subclass维度:{self.n_subclasses}")

        # 诊断
        sample_vectors = list(self.local_gene_features.values())[:1000]
        sample_array = np.array(sample_vectors)
        n_unique = len(np.unique(np.round(sample_array, 6), axis=0))
        zero_ratio = (sample_array == 0).mean()
        nonzero_per_neuron = (sample_array > 0).sum(axis=1).mean()

        print(f"\n局部分子环境质量诊断:")
        print(f"  唯一模式数（前1000）:{n_unique} ({n_unique / len(sample_vectors) * 100:.1f}%)")
        print(f"  零值比例:{zero_ratio:.1%}")
        print(f"  平均非零subclass数:{nonzero_per_neuron:.1f} / {self.n_subclasses}")

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

        print(f"  轴突特征: {len(self.AXONAL_FEATURES)} 维")
        print(f"  树突特征:{len(self.DENDRITIC_FEATURES)} 维")
        print(f"  Subclass: {self.n_subclasses} 种（从缓存）")
        print(f"  投射目标: {len(self.all_target_regions)} 个")

    def _get_valid_neurons(self):
        """获取有效神经元"""
        print("\n从Neo4j获取有效神经元...")

        query = """
        MATCH (n:Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
        WITH n
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        WITH n, COUNT(DISTINCT t) AS n_targets
        WHERE n_targets >= 1
        RETURN n.neuron_id AS neuron_id
        ORDER BY neuron_id
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            self.all_neuron_ids = [r['neuron_id'] for r in result if r['neuron_id']]

        print(f"  Neo4j中有 {len(self.all_neuron_ids)} 个有轴突+有投射的神经元")

    def _load_axon_features(self):
        """加载轴突特征"""
        print("\n加载轴突特征...")

        return_parts = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        query = f"""
        MATCH (n:Neuron) WHERE n.neuron_id = $neuron_id
        RETURN {", ".join(return_parts)}
        """

        loaded = 0
        with self.driver.session(database=self.database) as session:
            for neuron_id in self.all_neuron_ids:
                result = session.run(query, neuron_id=neuron_id)
                record = result.single()
                if record:
                    features = [float(record[f]) if record[f] is not None else 0.0
                    for f in self.AXONAL_FEATURES]
                    self.axon_features[neuron_id] = np.array(features)
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的轴突特征")

    def _load_dendrite_features(self):
        """加载树突特征"""
        print("\n加载树突特征...")

        return_parts = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]
        query = f"""
        MATCH (n:Neuron) WHERE n.neuron_id = $neuron_id
        RETURN {", ".join(return_parts)}
        """

        loaded = 0
        with self.driver.session(database=self.database) as session:
            for neuron_id in self.all_neuron_ids:
                result = session.run(query, neuron_id=neuron_id)
                record = result.single()
                if record:
                    features = []
                    has_data = False
                    for f in self.DENDRITIC_FEATURES:
                        val = record[f]
                        if val is not None and val > 0:
                            has_data = True
                            features.append(float(val))
                        else:
                            features.append(0.0)

                    if has_data:
                        self.dendrite_features[neuron_id] = np.array(features)
                        self.neurons_with_dendrite.append(neuron_id)
                        loaded += 1

        print(f"  加载了 {loaded} 个神经元的树突特征")

    def _load_projection_vectors(self):
        """加载投射向量（与V7一致的处理）"""
        print("\n加载投射向量...")

        query = """
        MATCH (n:Neuron {neuron_id: $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        loaded = 0
        with self.driver.session(database=self.database) as session:
            for neuron_id in self.all_neuron_ids:
                result = session.run(query, neuron_id=neuron_id)
                proj_dict = {r['target']:r['weight'] for r in result
                             if r['target'] and r['weight']}

                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]

                    # 与V7一致：log10变换 + 归一化
                    proj_vector = np.log10(1 + proj_vector)
                    total = proj_vector.sum()
                    if total > 0:
                        proj_vector = proj_vector / total

                    self.projection_vectors[neuron_id] = proj_vector
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _filter_valid_neurons(self):
        """过滤数据完整的神经元"""
        print("\n过滤数据完整的神经元...")

        print(f"  过滤前: {len(self.all_neuron_ids)} 个")
        print(
            f"  有轴突: {len(self.axon_features)}, 有局部环境:{len(self.local_gene_features)}, 有投射:{len(self.projection_vectors)}")

        valid_all = [n for n in self.all_neuron_ids
                     if n in self.axon_features
                     and n in self.local_gene_features
                     and n in self.projection_vectors]

        valid_dendrite = [n for n in valid_all if n in self.dendrite_features]

        self.all_neuron_ids = valid_all
        self.neurons_with_dendrite = valid_dendrite

        print(f"  过滤后 - 全量: {len(self.all_neuron_ids)}, 有dendrite:{len(self.neurons_with_dendrite)}")

    # ==================== 特征准备 ====================

    def prepare_features_full_morph(self) -> Tuple:
        """准备Full-morph特征"""
        print("\n准备Full-morph特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        X_morph_raw = np.array([
            np.concatenate([self.axon_features[nid], self.dendrite_features[nid]])
            for nid in neurons
        ])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (axon+dendrite)")
        print(f"  分子特征:{X_gene_raw.shape[1]} 维 (局部环境)")
        print(f"  投射向量:{Y.shape[1]} 维")

        # Morph: log1p -> RobustScaler
        X_morph_log = np.log1p(X_morph_raw)
        X_morph_scaled = RobustScaler().fit_transform(X_morph_log)

        # Gene:CLR -> StandardScaler
        X_gene_clr = self.clr_transform(X_gene_raw)
        X_gene_scaled = StandardScaler().fit_transform(X_gene_clr)

        # 多模态拼接
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle对照
        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    def prepare_features_axon_only(self) -> Tuple:
        """准备Axon-only特征"""
        print("\n准备Axon-only特征矩阵...")

        neurons = self.all_neuron_ids
        n = len(neurons)

        X_morph_raw = np.array([self.axon_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数:{n}")
        print(f"  形态特征:{X_morph_raw.shape[1]} 维 (axon-only)")

        X_morph_log = np.log1p(X_morph_raw)
        X_morph_scaled = RobustScaler().fit_transform(X_morph_log)

        X_gene_clr = self.clr_transform(X_gene_raw)
        X_gene_scaled = StandardScaler().fit_transform(X_gene_clr)

        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    # ==================== 模型训练 ====================

    def train_and_predict(self, X:np.ndarray, Y:np.ndarray,
                          condition_name:str, dataset_name:str,
                          model_name:str = "RF",
                          n_folds:int = 5) -> PredictionResult:
        """训练并预测"""
        print(f"\n  训练 {condition_name}...")

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model:{model_name}")

        n_samples = X.shape[0]
        actual_folds = max(3, min(n_folds, n_samples // 10, 10))

        kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # 计算指标
        cosine_sims = np.zeros(n_samples)
        jsd_sims = np.zeros(n_samples)

        for i in range(n_samples):
            cosine_sims[i] = self.compute_cosine_similarity(Y[i], Y_pred[i])
            jsd_sims[i] = self.compute_jsd_similarity(Y[i], Y_pred[i])

        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        per_target_corrs = {}
        for j in range(len(self.all_target_regions)):
            target = self.all_target_regions[j]
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_target_corrs[target] = corr

        result = PredictionResult(
            model_name=model_name,
            condition_name=condition_name,
            dataset_name=dataset_name,
            cosine_similarities=cosine_sims,
            jsd_similarities=jsd_sims,
            per_target_correlations=per_target_corrs,
            mean_cosine_sim=np.mean(cosine_sims),
            std_cosine_sim=np.std(cosine_sims),
            mean_jsd_sim=np.mean(jsd_sims),
            std_jsd_sim=np.std(jsd_sims),
            mean_r2=global_r2,
            predicted=Y_pred,
            actual=Y,
            n_neurons=n_samples,
            n_features=X.shape[1]
        )

        print(f"    Cosine: {result.mean_cosine_sim:.4f}, R²:{result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_projection_prediction_experiment(self, model_name:str = "RF"):
        """运行投射预测实验"""
        print("\n" + "=" * 80)
        print(f"【实验1】投射预测实验 (模型:{model_name})")
        print("=" * 80)

        results = {}

        # Full-morph
        print("\n--- Full-morph ---")
        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_full_morph()
        print(f"  N={len(neurons)}, Morph={X_morph.shape[1]}D, Gene={X_gene.shape[1]}D, Proj={Y.shape[1]}D")

        results['full_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'full_morph', model_name)
        results['full_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'full_morph', model_name)
        results['full_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'full_morph', model_name)
        results['full_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'full_morph', model_name)

        # Axon-only
        print("\n--- Axon-only ---")
        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_axon_only()
        print(f"  N={len(neurons)}, Morph={X_morph.shape[1]}D, Gene={X_gene.shape[1]}D, Proj={Y.shape[1]}D")

        results['axon_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'axon_only', model_name)
        results['axon_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'axon_only', model_name)
        results['axon_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'axon_only', model_name)
        results['axon_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'axon_only', model_name)

        self.results = results
        return results

    def run_cross_modal_prediction_experiment(self, model_name:str = "RF"):
        """运行跨模态预测实验（增强版：添加Cosine Similarity）"""
        print("\n" + "=" * 80)
        print(f"【实验2】跨模态预测实验 (模型: {model_name})")
        print("=" * 80)

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        # 准备原始特征
        X_morph_raw = np.array([
            np.concatenate([self.axon_features[nid], self.dendrite_features[nid]])
            for nid in neurons
        ])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        X_proj_raw = np.array([self.projection_vectors[nid] for nid in neurons])

        # 处理特征
        X_morph = RobustScaler().fit_transform(np.log1p(X_morph_raw))
        X_gene = StandardScaler().fit_transform(self.clr_transform(X_gene_raw))
        X_proj = StandardScaler().fit_transform(X_proj_raw)

        print(f"\n  Morph:{X_morph.shape[1]}D, Gene:{X_gene.shape[1]}D, Proj:{X_proj.shape[1]}D, N={n}")

        cross_results = {}

        # ===== 预测 Projection =====
        print("\n--- Target: Projection ---")
        X_morph_gene = np.hstack([X_morph, X_gene])
        cross_results['morph_gene_to_proj'] = self._train_cross_modal(
            X_morph_gene, X_proj, 'Morph+Gene', 'Projection', model_name)
        cross_results['morph_to_proj'] = self._train_cross_modal(
            X_morph, X_proj, 'Morph', 'Projection', model_name)
        cross_results['gene_to_proj'] = self._train_cross_modal(
            X_gene, X_proj, 'Gene', 'Projection', model_name)

        # ===== 预测 Gene =====
        print("\n--- Target: Gene ---")
        X_morph_proj = np.hstack([X_morph, X_proj])
        cross_results['morph_proj_to_gene'] = self._train_cross_modal(
            X_morph_proj, X_gene, 'Morph+Proj', 'Gene', model_name)
        cross_results['morph_to_gene'] = self._train_cross_modal(
            X_morph, X_gene, 'Morph', 'Gene', model_name)
        cross_results['proj_to_gene'] = self._train_cross_modal(
            X_proj, X_gene, 'Proj', 'Gene', model_name)

        # ===== 预测 Morphology =====
        print("\n--- Target: Morphology ---")
        X_gene_proj = np.hstack([X_gene, X_proj])
        cross_results['gene_proj_to_morph'] = self._train_cross_modal(
            X_gene_proj, X_morph, 'Gene+Proj', 'Morphology', model_name)
        cross_results['gene_to_morph'] = self._train_cross_modal(
            X_gene, X_morph, 'Gene', 'Morphology', model_name)
        cross_results['proj_to_morph'] = self._train_cross_modal(
            X_proj, X_morph, 'Proj', 'Morphology', model_name)

        self.cross_modal_results = cross_results
        return cross_results

    def _train_cross_modal(self, X:np.ndarray, Y:np.ndarray,
                           input_name:str, target_name:str,
                           model_name:str = "RF") -> CrossModalPredictionResult:
        """跨模态预测训练（增强版）"""
        print(f"\n  {input_name} → {target_name}...")

        n_samples = X.shape[0]

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model:{model_name}")

        n_folds = max(3, min(5, n_samples // 10, 10))
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # ===== Cosine Similarity（逐样本）=====
        cosine_sims = np.zeros(n_samples)
        jsd_sims = np.zeros(n_samples)

        for i in range(n_samples):
            cosine_sims[i] = self.compute_cosine_similarity(Y[i], Y_pred[i])
            # JSD只对Projection有意义
            if target_name == 'Projection':
                jsd_sims[i] = self.compute_jsd_similarity(Y[i], Y_pred[i])

        # ===== R² =====
        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # ===== Per-dimension Correlation =====
        n_target_features = Y.shape[1]
        per_feature_corrs = np.zeros(n_target_features)
        for j in range(n_target_features):
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_feature_corrs[j] = corr if not np.isnan(corr) else 0

        mean_corr = np.mean(per_feature_corrs)
        std_corr = np.std(per_feature_corrs)

        # 打印结果
        print(f"    Cosine:{np.mean(cosine_sims):.4f} ± {np.std(cosine_sims):.4f}")
        print(f"    R²:{global_r2:.4f}, Mean Dim Corr:{mean_corr:.4f}")

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
            jsd_similarities=jsd_sims,
            mean_jsd_sim=np.mean(jsd_sims) if target_name == 'Projection' else 0.0,
            std_jsd_sim=np.std(jsd_sims) if target_name == 'Projection' else 0.0,
            n_neurons=n_samples,
            n_input_features=X.shape[1],
            n_target_features=n_target_features,
            predicted=Y_pred,
            actual=Y
        )

    def run_all_experiments(self, model_name:str = "RF"):
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

        for dataset in ['full', 'axon']:
            prefix = f'{dataset}_'
            if f'{prefix}morph_only' not in self.results:
                continue

            print(f"\n【{dataset.upper()} - 投射预测】")

            sim_morph = self.results[f'{prefix}morph_only'].cosine_similarities
            sim_gene = self.results[f'{prefix}gene_only'].cosine_similarities
            sim_multi = self.results[f'{prefix}morph_gene'].cosine_similarities
            sim_shuffle = self.results[f'{prefix}shuffle'].cosine_similarities

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

                stats[f'{dataset}_{name}'] = {
                    'p_value':p_val, 'cohens_d':cohens_d, 'improvement_pct':improve
                }

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"  {name}: p={p_val:.2e}{sig}, d={cohens_d:.3f}, Δ={improve:.1f}%")

        return stats

    # ==================== 可视化 ====================

    def _save_figure(self, fig, output_dir:str, filename:str):
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def visualize_results(self, output_dir:str = "."):
        """生成所有可视化"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_projection_prediction_summary(output_dir)
        self._plot_cross_modal_summary(output_dir)
        self._plot_cross_modal_cosine(output_dir)
        self._plot_delta_distribution(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_projection_prediction_summary(self, output_dir:str):
        """投射预测汇总图"""
        datasets = [(p, n) for p, n in [('full', 'Full-morph'), ('axon', 'Axon-only')]
                    if f'{p}_morph_only' in self.results]
        conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6))
        if len(datasets) == 1:
            axes = [axes]

        for ax, (prefix, title) in zip(axes, datasets):
            keys = [f'{prefix}_morph_only', f'{prefix}_gene_only', f'{prefix}_morph_gene', f'{prefix}_shuffle']
            means = [self.results[k].mean_cosine_sim for k in keys]
            stds = [self.results[k].std_cosine_sim for k in keys]

            bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=4,
                          color=colors, alpha=0.85, edgecolor='black')

            for bar, val in zip(bars, means):
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontweight='bold')

            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title(f'{title} (N={self.results[keys[0]].n_neurons})\n'
                         f'Radius: {self.search_radius * 25}μm', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions)
            ax.grid(axis='y', alpha=0.3)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_projection_prediction.png")

    def _plot_cross_modal_summary(self, output_dir:str):
        """跨模态预测汇总（R²版本）"""
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
            r2s = [self.cross_modal_results[k].r2_score for k in keys]
            alphas = [0.9, 0.5, 0.5]

            for i, (r2, alpha) in enumerate(zip(r2s, alphas)):
                ax.bar(i, r2, color=color, alpha=alpha, edgecolor='black')
                ax.annotate(f'{r2:.3f}', xy=(i, r2), xytext=(0, 3),
                            textcoords='offset points', ha='center', fontweight='bold')

            gain = r2s[0] - max(r2s[1:])
            ax.text(0.5, 0.95, f'Gain: {gain:+.4f}', transform=ax.transAxes,
                    ha='center', va='top', fontweight='bold',
                    color='#27AE60' if gain > 0 else '#E74C3C',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_ylabel('R² Score')
            ax.set_title(f'Predict: {target}', fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.grid(axis='y', alpha=0.3)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Cross-Modal Prediction (R² Score)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_cross_modal_r2.png")

    def _plot_cross_modal_cosine(self, output_dir:str):
        """跨模态预测汇总（Cosine Similarity版本）"""
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
            ax.set_title(f'Predict:{target}', fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.grid(axis='y', alpha=0.3)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Cross-Modal Prediction (Cosine Similarity)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_cross_modal_cosine.png")

    def _plot_delta_distribution(self, output_dir:str):
        """Delta分布图"""
        prefix = 'full' if 'full_morph_only' in self.results else 'axon'

        sim_morph = self.results[f'{prefix}_morph_only'].cosine_similarities
        sim_gene = self.results[f'{prefix}_gene_only'].cosine_similarities
        sim_multi = self.results[f'{prefix}_morph_gene'].cosine_similarities
        sim_shuffle = self.results[f'{prefix}_shuffle'].cosine_similarities

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
        self._save_figure(fig, output_dir, "4_delta_distribution.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir:str = "."):
        """保存结果到CSV"""
        os.makedirs(output_dir, exist_ok=True)

        # 投射预测结果
        rows = []
        for key, result in self.results.items():
            rows.append({
                'key':key,
                'dataset':result.dataset_name,
                'condition':result.condition_name,
                'n_neurons':result.n_neurons,
                'n_features':result.n_features,
                'mean_cosine_sim':result.mean_cosine_sim,
                'std_cosine_sim':result.std_cosine_sim,
                'mean_jsd_sim':result.mean_jsd_sim,
                'std_jsd_sim':result.std_jsd_sim,
                'r2_score':result.mean_r2,
                'search_radius_um':self.search_radius * 25
            })
        pd.DataFrame(rows).to_csv(f"{output_dir}/projection_prediction.csv", index=False)

        # 跨模态预测结果
        if self.cross_modal_results:
            cross_rows = []
            for key, result in self.cross_modal_results.items():
                cross_rows.append({
                    'key':key,
                    'input':result.input_modalities,
                    'target':result.target_modality,
                    'r2_score':result.r2_score,
                    'mean_corr':result.mean_corr,
                    'mean_cosine_sim':result.mean_cosine_sim,
                    'std_cosine_sim':result.std_cosine_sim,
                    'n_neurons':result.n_neurons,
                    'n_input_features':result.n_input_features,
                    'n_target_features':result.n_target_features
                })
            pd.DataFrame(cross_rows).to_csv(f"{output_dir}/cross_modal_prediction.csv", index=False)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir:str = "./task1_results_v10", model_name:str = "RF"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态预测 (V10 - 精确局部分子环境)")
        print(f"搜索半径: {self.search_radius} 体素 = {self.search_radius * 25}μm")
        print("=" * 80)

        n_all, n_dendrite = self.load_all_data()
        if n_all == 0:
            print("\n✗ 没有有效数据")
            return

        self.run_all_experiments(model_name)
        stats = self.statistical_analysis()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_conclusion()

        print("\n" + "=" * 80)
        print(f"完成!  结果: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self):
        """打印结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        print(f"\n【设置】半径={self.search_radius * 25}μm")

        # 投射预测
        print(f"\n【实验1:投射预测】")
        for prefix, name in [('full', 'Full-morph'), ('axon', 'Axon-only')]:
            if f'{prefix}_morph_only' not in self.results:
                continue

            morph = self.results[f'{prefix}_morph_only'].mean_cosine_sim
            gene = self.results[f'{prefix}_gene_only'].mean_cosine_sim
            multi = self.results[f'{prefix}_morph_gene'].mean_cosine_sim
            shuffle = self.results[f'{prefix}_shuffle'].mean_cosine_sim

            print(f"\n  {name} (N={self.results[f'{prefix}_morph_only'].n_neurons}):")
            print(f"    Morph: {morph:.4f}, Gene:{gene:.4f}, Multi: {multi:.4f}, Shuffle:{shuffle:.4f}")
            print(f"    Multi vs Morph:{multi - morph:+.4f} ({(multi - morph) / morph * 100:+.1f}%)")
            print(f"    Multi vs Gene:{multi - gene:+.4f} ({(multi - gene) / gene * 100:+.1f}%)")
            print(f"    Shuffle vs Morph:{shuffle - morph:+.4f}")

        # 跨模态预测
        if self.cross_modal_results:
            print(f"\n【实验2:跨模态预测】")
            for target, dual_key, single_keys in [
                ('Projection', 'morph_gene_to_proj', ['morph_to_proj', 'gene_to_proj']),
                ('Gene', 'morph_proj_to_gene', ['morph_to_gene', 'proj_to_gene']),
                ('Morphology', 'gene_proj_to_morph', ['gene_to_morph', 'proj_to_morph']),
            ]:
                dual_cos = self.cross_modal_results[dual_key].mean_cosine_sim
                single_cos = [self.cross_modal_results[k].mean_cosine_sim for k in single_keys]
                gain = dual_cos - max(single_cos)

                dual_r2 = self.cross_modal_results[dual_key].r2_score
                single_r2 = [self.cross_modal_results[k].r2_score for k in single_keys]
                gain_r2 = dual_r2 - max(single_r2)

                status = "✓" if gain > 0 else "✗"
                print(
                    f"  预测{target}: Cos={dual_cos:.4f} (gain={gain:+.4f}{status}), R²={dual_r2:.4f} (gain={gain_r2:+.4f})")


def main():
    # ==================== 配置 ====================
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    DATA_DIR = "/home/wlj/NeuroXiv2/data"  # 包含cache/local_env_r8.0_mirrored.pkl
    OUTPUT_DIR = "./task1_results_v10_local_env"
    MODEL_NAME = "RF"
    SEARCH_RADIUS = 8.0  # 8 × 25μm = 200μm

    # ==================== 运行 ====================
    with NeuronMultimodalPredictorV10(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=SEARCH_RADIUS
    ) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()