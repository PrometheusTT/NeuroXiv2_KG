"""
任务1：Neuron-level 多模态预测实验 (V8 修正版)
===============================================
修正问题：
1.保持与V7完全一致的数据处理和模型训练逻辑
2.Panel 2优化：只显示Delta（提升量）
3.新增三模态交叉预测实验

作者:  PrometheusTT
日期: 2025-07-30
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

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
    model_name: str
    condition_name: str
    dataset_name: str
    cosine_similarities: np.ndarray
    jsd_similarities: np.ndarray
    per_target_correlations: Dict[str, float]
    mean_cosine_sim: float
    std_cosine_sim: float
    mean_jsd_sim: float
    std_jsd_sim: float
    mean_r2: float
    predicted:  np.ndarray
    actual: np.ndarray
    n_neurons: int
    n_features: int


@dataclass
class CrossModalPredictionResult:
    """跨模态预测结果"""
    input_modalities: str
    target_modality: str
    model_name: str
    r2_score: float
    mean_corr: float
    std_corr:  float
    per_feature_corrs: np.ndarray
    n_neurons: int
    n_input_features: int
    n_target_features: int


class NeuronMultimodalPredictorV8:
    """神经元多模态投射预测器 V8（修正版）"""

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

    def __init__(self, uri: str, user: str, password:  str, database: str = "neo4j"):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

        self.all_neuron_ids:  List[str] = []
        self.neurons_with_dendrite: List[str] = []
        self.axon_features: Dict[str, np.ndarray] = {}
        self.dendrite_features: Dict[str, np.ndarray] = {}
        self.local_gene_features: Dict[str, np.ndarray] = {}
        self.projection_vectors: Dict[str, np.ndarray] = {}

        self.all_subclasses:  List[str] = []
        self.all_target_regions: List[str] = []

        self.results: Dict[str, PredictionResult] = {}
        self.cross_modal_results: Dict[str, CrossModalPredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数（与V7完全一致） ====================

    @staticmethod
    def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        """CLR变换 - 与V7完全一致"""
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def compute_jsd_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """JSD相似度 - 与V7完全一致"""
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

    # ==================== 数据加载（与V7完全一致） ====================

    def diagnose_database(self):
        print("\n" + "=" * 80)
        print("数据库诊断")
        print("=" * 80)

        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n: Neuron) RETURN count(n) as count")
            print(f"\n总Neuron数量:  {result.single()['count']}")

            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有axonal数据:  {result.single()['count']}")

            result = session.run("""
                MATCH (n: Neuron) 
                WHERE n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有dendritic数据:  {result.single()['count']}")

            result = session.run("""
                MATCH (n: Neuron)-[p:PROJECT_TO]->(s:Subregion)
                RETURN count(DISTINCT n) as count
            """)
            print(f"有投射关系: {result.single()['count']}")

        print("=" * 80)

    def load_all_data(self) -> Tuple[int, int]:
        print("\n" + "=" * 80)
        print("加载神经元数据")
        print("=" * 80)

        self.diagnose_database()
        self._get_global_dimensions()
        self._get_valid_neurons()
        self._load_axon_features()
        self._load_dendrite_features()
        self._load_projection_vectors()
        self._compute_local_gene_features()
        self._filter_valid_neurons()

        print(f"\n✓ 数据加载完成:")
        print(f"  全量神经元:  {len(self.all_neuron_ids)}")
        print(f"  有dendrite的神经元: {len(self.neurons_with_dendrite)}")

        return len(self.all_neuron_ids), len(self.neurons_with_dendrite)

    def _get_global_dimensions(self):
        print("\n获取全局特征维度...")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Subclass) WHERE s.name IS NOT NULL
                RETURN DISTINCT s.name AS name ORDER BY name
            """)
            self.all_subclasses = [r['name'] for r in result if r['name']]

            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
                WHERE p.weight IS NOT NULL AND p.weight > 0
                RETURN DISTINCT t.acronym AS target ORDER BY target
            """)
            self.all_target_regions = [r['target'] for r in result if r['target']]

        print(f"  轴突特征: {len(self.AXONAL_FEATURES)} 维")
        print(f"  树突特征: {len(self.DENDRITIC_FEATURES)} 维")
        print(f"  Subclass: {len(self.all_subclasses)} 种")
        print(f"  投射目标: {len(self.all_target_regions)} 个")

    def _get_valid_neurons(self):
        print("\n获取有效神经元...")

        query = """
        MATCH (n: Neuron)
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

        print(f"  找到 {len(self.all_neuron_ids)} 个有轴突+有投射的神经元")

    def _load_axon_features(self):
        print("\n加载轴突特征...")

        return_parts = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        query = f"""
        MATCH (n: Neuron) WHERE n.neuron_id = $neuron_id
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
        print("\n加载树突特征...")

        return_parts = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]
        query = f"""
        MATCH (n: Neuron) WHERE n.neuron_id = $neuron_id
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
        """加载投射向量 - 与V7完全一致：log10变换 + 归一化"""
        print("\n加载投射向量...")

        query = """
        MATCH (n: Neuron {neuron_id:  $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        loaded = 0
        with self.driver.session(database=self.database) as session:
            for neuron_id in self.all_neuron_ids:
                result = session.run(query, neuron_id=neuron_id)
                proj_dict = {r['target']:  r['weight'] for r in result
                             if r['target'] and r['weight']}

                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]

                    # ===== 关键：与V7完全一致的处理 =====
                    proj_vector = np.log10(1 + proj_vector)
                    total = proj_vector.sum()
                    if total > 0:
                        proj_vector = proj_vector / total

                    self.projection_vectors[neuron_id] = proj_vector
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _compute_local_gene_features(self):
        """计算局部分子环境 - 与V7完全一致"""
        print("\n计算局部分子环境...")

        query_locate = """
        MATCH (n: Neuron {neuron_id: $neuron_id})-[: LOCATE_AT]->(r:Region)
        OPTIONAL MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN r.acronym AS region, sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        query_base = """
        MATCH (n: Neuron {neuron_id: $neuron_id})
        WITH n.base_region AS base_region WHERE base_region IS NOT NULL
        MATCH (r:Region) WHERE r.acronym = base_region
        OPTIONAL MATCH (r)-[hs:HAS_SUBCLASS]->(sc: Subclass)
        RETURN r.acronym AS region, sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        computed = 0
        with self.driver.session(database=self.database) as session:
            for neuron_id in self.all_neuron_ids:
                result = session.run(query_locate, neuron_id=neuron_id)
                records = list(result)

                if not records or all(r['region'] is None for r in records):
                    result = session.run(query_base, neuron_id=neuron_id)
                    records = list(result)

                subclass_dict = {}
                for r in records:
                    if r['subclass_name'] and r['pct_cells']:
                        subclass_dict[r['subclass_name']] = r['pct_cells']

                gene_vector = np.zeros(len(self.all_subclasses))
                for i, sc in enumerate(self.all_subclasses):
                    if sc in subclass_dict:
                        gene_vector[i] = subclass_dict[sc]

                self.local_gene_features[neuron_id] = gene_vector
                if len(subclass_dict) > 0:
                    computed += 1

        print(f"  计算了 {computed} 个神经元的局部分子环境")

    def _filter_valid_neurons(self):
        print("\n过滤数据完整的神经元...")

        valid_all = [n for n in self.all_neuron_ids
                     if n in self.axon_features
                     and n in self.local_gene_features
                     and n in self.projection_vectors]

        valid_dendrite = [n for n in valid_all if n in self.dendrite_features]

        self.all_neuron_ids = valid_all
        self.neurons_with_dendrite = valid_dendrite

        print(f"  全量数据集:  {len(self.all_neuron_ids)} 个")
        print(f"  有dendrite数据集: {len(self.neurons_with_dendrite)} 个")

    # ==================== 特征准备（与V7完全一致） ====================

    def prepare_features_full_morph(self) -> Tuple:
        """准备Full-morph特征 - 与V7完全一致"""
        print("\n准备Full-morph特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        # 拼接axon + dendrite
        X_morph_raw = np.array([
            np.concatenate([self.axon_features[nid], self.dendrite_features[nid]])
            for nid in neurons
        ])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数:  {n}")
        print(f"  形态特征:  {X_morph_raw.shape[1]} 维 (axon+dendrite)")
        print(f"  分子特征: {X_gene_raw.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # ===== Morph:  log1p -> RobustScaler =====
        X_morph_log = np.log1p(X_morph_raw)
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)
        print("  Morph处理:  log1p -> RobustScaler")

        # ===== Gene: CLR -> StandardScaler =====
        X_gene_clr = self.clr_transform(X_gene_raw)
        scaler_gene = StandardScaler()
        X_gene_scaled = scaler_gene.fit_transform(X_gene_clr)
        print("  Gene处理: CLR -> StandardScaler")

        # 多模态拼接
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle对照
        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    def prepare_features_axon_only(self) -> Tuple:
        """准备Axon-only特征 - 与V7完全一致"""
        print("\n准备Axon-only特征矩阵...")

        neurons = self.all_neuron_ids
        n = len(neurons)

        X_morph_raw = np.array([self.axon_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数:  {n}")
        print(f"  形态特征:  {X_morph_raw.shape[1]} 维 (axon-only)")

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

    def prepare_features_dendrite_only(self) -> Tuple:
        """准备Dendrite-only特征 - 与V7完全一致"""
        print("\n准备Dendrite-only特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        X_morph_raw = np.array([self.dendrite_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数:  {n}")
        print(f"  形态特征:  {X_morph_raw.shape[1]} 维 (dendrite-only)")

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

    # ==================== 模型训练（与V7完全一致） ====================

    def train_and_predict(self, X:  np.ndarray, Y: np.ndarray,
                          condition_name: str, dataset_name: str,
                          model_name: str = "Ridge",
                          n_folds: int = 5) -> PredictionResult:
        """训练并预测 - 与V7完全一致"""
        print(f"\n  训练 {condition_name} ({model_name})...")

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        n_samples = X.shape[0]
        actual_folds = max(3, min(n_folds, n_samples // 10, 10))
        print(f"    使用 {actual_folds} 折交叉验证 (样本数={n_samples})")

        kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # 计算指标 - 与V7完全一致
        cosine_sims = np.zeros(n_samples)
        jsd_sims = np.zeros(n_samples)

        for i in range(n_samples):
            if np.linalg.norm(Y_pred[i]) > 0 and np.linalg.norm(Y[i]) > 0:
                cosine_sims[i] = 1 - cosine(Y_pred[i], Y[i])
            jsd_sims[i] = self.compute_jsd_similarity(Y[i], Y_pred[i])

        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        per_target_corrs = {}
        for j in range(len(self.all_target_regions)):
            target = self.all_target_regions[j]
            if Y[: , j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[: , j], Y_pred[:, j])
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

        print(f"    Cosine Sim:  {result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}")
        print(f"    JSD Sim:     {result.mean_jsd_sim:.4f} ± {result.std_jsd_sim:.4f}")
        print(f"    R² (补充):  {result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_projection_prediction_experiment(self, model_name: str = "Ridge"):
        """运行投射预测实验 - 与V7完全一致"""
        print("\n" + "=" * 80)
        print(f"【实验1】投射预测实验 (模型: {model_name})")
        print("=" * 80)

        results = {}

        # 数据集1:  Full-morph
        print("\n" + "=" * 60)
        print("【数据集1: Full-morph (有dendrite的神经元, axon+dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_full_morph()

        print("\n【条件1: Morph-only】")
        results['full_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'full_morph', model_name)

        print("\n【条件2: Gene-only】")
        results['full_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'full_morph', model_name)

        print("\n【条件3: Morph+Gene】")
        results['full_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'full_morph', model_name)

        print("\n【条件4: Shuffle】")
        results['full_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'full_morph', model_name)

        # 数据集2: Axon-only
        print("\n" + "=" * 60)
        print("【数据集2: Axon-only (全量神经元)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_axon_only()

        print("\n【条件1: Morph-only】")
        results['axon_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'axon_only', model_name)

        print("\n【条件2: Gene-only】")
        results['axon_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'axon_only', model_name)

        print("\n【条件3: Morph+Gene】")
        results['axon_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'axon_only', model_name)

        print("\n【条件4: Shuffle】")
        results['axon_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'axon_only', model_name)

        # 数据集3: Dendrite-only
        print("\n" + "=" * 60)
        print("【数据集3: Dendrite-only (有dendrite的神经元, 仅dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_dendrite_only()

        print("\n【条件1: Morph-only】")
        results['dend_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'dendrite_only', model_name)

        print("\n【条件2: Gene-only】")
        results['dend_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'dendrite_only', model_name)

        print("\n【条件3: Morph+Gene】")
        results['dend_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'dendrite_only', model_name)

        print("\n【条件4: Shuffle】")
        results['dend_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'dendrite_only', model_name)

        self.results = results
        return results

    def run_cross_modal_prediction_experiment(self, model_name: str = "Ridge"):
        """
        运行跨模态预测实验
        测试任意两个模态预测第三个模态的能力
        """
        print("\n" + "=" * 80)
        print(f"【实验2】跨模态预测实验 (模型: {model_name})")
        print("=" * 80)

        # 使用full_morph数据集
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

        print(f"\n特征维度:")
        print(f"  Morph: {X_morph.shape[1]}维")
        print(f"  Gene:  {X_gene.shape[1]}维")
        print(f"  Proj: {X_proj.shape[1]}维")
        print(f"  神经元数: {n}")

        cross_results = {}

        # ===== 1.预测 Proj =====
        print("\n" + "-" * 50)
        print("【目标:  Projection】")
        print("-" * 50)

        X_morph_gene = np.hstack([X_morph, X_gene])
        cross_results['morph_gene_to_proj'] = self._train_cross_modal(
            X_morph_gene, X_proj, 'Morph+Gene', 'Proj', model_name, n)
        cross_results['morph_to_proj'] = self._train_cross_modal(
            X_morph, X_proj, 'Morph', 'Proj', model_name, n)
        cross_results['gene_to_proj'] = self._train_cross_modal(
            X_gene, X_proj, 'Gene', 'Proj', model_name, n)

        # ===== 2.预测 Gene =====
        print("\n" + "-" * 50)
        print("【目标: Gene】")
        print("-" * 50)

        X_morph_proj = np.hstack([X_morph, X_proj])
        cross_results['morph_proj_to_gene'] = self._train_cross_modal(
            X_morph_proj, X_gene, 'Morph+Proj', 'Gene', model_name, n)
        cross_results['morph_to_gene'] = self._train_cross_modal(
            X_morph, X_gene, 'Morph', 'Gene', model_name, n)
        cross_results['proj_to_gene'] = self._train_cross_modal(
            X_proj, X_gene, 'Proj', 'Gene', model_name, n)

        # ===== 3.预测 Morph =====
        print("\n" + "-" * 50)
        print("【目标: Morphology】")
        print("-" * 50)

        X_gene_proj = np.hstack([X_gene, X_proj])
        cross_results['gene_proj_to_morph'] = self._train_cross_modal(
            X_gene_proj, X_morph, 'Gene+Proj', 'Morph', model_name, n)
        cross_results['gene_to_morph'] = self._train_cross_modal(
            X_gene, X_morph, 'Gene', 'Morph', model_name, n)
        cross_results['proj_to_morph'] = self._train_cross_modal(
            X_proj, X_morph, 'Proj', 'Morph', model_name, n)

        self.cross_modal_results = cross_results
        return cross_results

    def _train_cross_modal(self, X:  np.ndarray, Y: np.ndarray,
                           input_mods: str, target_mod: str,
                           model_name: str, n_neurons: int) -> CrossModalPredictionResult:
        """跨模态预测训练"""
        print(f"\n  {input_mods} → {target_mod}...")

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        n_folds = max(3, min(5, n_neurons // 10, 10))
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # R²
        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # 每个特征的相关性
        n_target_features = Y.shape[1]
        per_feature_corrs = np.zeros(n_target_features)
        for j in range(n_target_features):
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_feature_corrs[j] = corr if not np.isnan(corr) else 0

        mean_corr = np.mean(per_feature_corrs)
        std_corr = np.std(per_feature_corrs)

        print(f"    R²: {global_r2:.4f}, Mean Corr: {mean_corr:.4f} ± {std_corr:.4f}")

        return CrossModalPredictionResult(
            input_modalities=input_mods,
            target_modality=target_mod,
            model_name=model_name,
            r2_score=global_r2,
            mean_corr=mean_corr,
            std_corr=std_corr,
            per_feature_corrs=per_feature_corrs,
            n_neurons=n_neurons,
            n_input_features=X.shape[1],
            n_target_features=n_target_features
        )

    def run_all_experiments(self, model_name: str = "Ridge"):
        """运行所有实验"""
        self.run_projection_prediction_experiment(model_name)
        self.run_cross_modal_prediction_experiment(model_name)
        return self.results, self.cross_modal_results

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        print("\n" + "=" * 80)
        print("统计显著性分析")
        print("=" * 80)

        stats = {}

        for dataset in ['full', 'axon', 'dend']:
            print(f"\n【{dataset.upper()} 数据集】")
            print("-" * 70)

            prefix = f'{dataset}_'

            sim_morph = self.results[f'{prefix}morph_only'].cosine_similarities
            sim_gene = self.results[f'{prefix}gene_only'].cosine_similarities
            sim_multi = self.results[f'{prefix}morph_gene'].cosine_similarities
            sim_shuffle = self.results[f'{prefix}shuffle'].cosine_similarities

            comparisons = [
                ('Morph+Gene vs Morph-only', sim_multi, sim_morph),
                ('Morph+Gene vs Gene-only', sim_multi, sim_gene),
                ('Morph+Gene vs Shuffle', sim_multi, sim_shuffle),
            ]

            for name, sim1, sim2 in comparisons:
                stat, p_val = wilcoxon(sim1, sim2, alternative='greater')
                diff = sim1 - sim2
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                improve = (np.mean(sim1) - np.mean(sim2)) / np.mean(sim2) * 100 if np.mean(sim2) > 0 else 0

                stats[f'{dataset}_{name}'] = {
                    'p_value': p_val, 'cohens_d': cohens_d, 'improvement_pct': improve
                }

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"  {name}:  p={p_val:.2e} {sig}, d={cohens_d:.3f}, Δ={improve:.1f}%")

        return stats

    # ==================== 可视化 ====================

    def _save_figure(self, fig, output_dir: str, filename: str):
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def visualize_results(self, output_dir:  str = "."):
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_panel1_delta_distribution(output_dir)
        self._plot_panel2_delta_only(output_dir)
        self._plot_panel3_comparison_summary(output_dir)
        self._plot_panel4_cross_modal_prediction(output_dir)

        print(f"\n✓ 所有图表已保存到:  {output_dir}")

    def _plot_panel1_delta_distribution(self, output_dir: str):
        """Panel 1: Delta分布图"""
        sim_morph = self.results['full_morph_only'].cosine_similarities
        sim_gene = self.results['full_gene_only'].cosine_similarities
        sim_multi = self.results['full_morph_gene'].cosine_similarities

        delta_cos_morph = sim_multi - sim_morph
        delta_cos_gene = sim_multi - sim_gene

        color_morph = '#3498DB'
        color_gene = '#27AE60'

        fig, ax = plt.subplots(figsize=(12, 5))

        x_min = min(delta_cos_morph.min(), delta_cos_gene.min()) - 0.02
        x_max = max(delta_cos_morph.max(), delta_cos_gene.max()) + 0.02
        x_range = np.linspace(x_min, x_max, 500)

        kde_cos_morph = gaussian_kde(delta_cos_morph)
        kde_cos_gene = gaussian_kde(delta_cos_gene)

        y_cos_morph = kde_cos_morph(x_range)
        y_cos_gene = kde_cos_gene(x_range)

        ax.plot(x_range, y_cos_morph, color=color_morph, linewidth=2.5,
                label=f'Multi − Morph (med={np.median(delta_cos_morph):.3f})')
        ax.plot(x_range, y_cos_gene, color=color_gene, linewidth=2.5,
                label=f'Multi − Gene (med={np.median(delta_cos_gene):.3f})')

        x_positive = x_range[x_range >= 0]
        ax.fill_between(x_positive, 0, kde_cos_morph(x_positive), color=color_morph, alpha=0.15)
        ax.fill_between(x_positive, 0, kde_cos_gene(x_positive), color=color_gene, alpha=0.15)

        ax.axvline(x=0, color='#333333', linestyle='-', linewidth=2, alpha=0.8)

        y_max = max(y_cos_morph.max(), y_cos_gene.max())
        ax.annotate('', xy=(0.08, y_max * 0.4), xytext=(0, y_max * 0.4),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
        ax.text(0.04, y_max * 0.47, 'Improvement', fontsize=10, fontweight='bold', color='#E74C3C', ha='center')

        ax.set_xlabel('Δ Cosine Similarity', fontsize=12, fontweight='medium')
        ax.set_ylabel('Density', fontsize=12, fontweight='medium')
        ax.set_title('Performance Gain Distribution', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max * 1.1)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "1_delta_distribution.png")

    def _plot_panel2_delta_only(self, output_dir: str):
        """Panel 2: 只显示Delta的图（去掉基线）"""
        sim_morph = self.results['full_morph_only'].cosine_similarities
        sim_gene = self.results['full_gene_only'].cosine_similarities
        sim_multi = self.results['full_morph_gene'].cosine_similarities

        delta_vs_morph = sim_multi - sim_morph
        delta_vs_gene = sim_multi - sim_gene

        n_neurons = len(delta_vs_morph)

        # ===== 图2a: Delta排序图 =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, (delta, baseline_name, color) in enumerate([
            (delta_vs_morph, 'Morph-only', '#3498DB'),
            (delta_vs_gene, 'Gene-only', '#27AE60')
        ]):
            ax = axes[ax_idx]

            sorted_idx = np.argsort(delta)
            delta_sorted = delta[sorted_idx]
            colors = np.where(delta_sorted > 0, '#27AE60', '#E74C3C')
            x = np.arange(n_neurons)

            # 绘制竖线
            for xi, di, ci in zip(x, delta_sorted, colors):
                ax.plot([xi, xi], [0, di], color=ci, alpha=0.5, linewidth=0.8)

            ax.scatter(x, delta_sorted, c=colors, s=8, alpha=0.7, edgecolors='none')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=5)

            mean_delta = np.mean(delta)
            median_delta = np.median(delta)
            ax.axhline(y=mean_delta, color='#9B59B6', linestyle='--', linewidth=2,
                       label=f'Mean = {mean_delta:.4f}')
            ax.axhline(y=median_delta, color='#E67E22', linestyle=':', linewidth=2,
                       label=f'Median = {median_delta:.4f}')

            n_improved = np.sum(delta > 0)
            pct_improved = n_improved / n_neurons * 100

            ax.set_xlabel('Neurons (sorted by Δ)', fontsize=11)
            ax.set_ylabel('Δ Cosine Similarity', fontsize=11)
            ax.set_title(f'vs {baseline_name}:  {n_improved}/{n_neurons} ({pct_improved:.1f}%) improved',
                         fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Per-Neuron Improvement (Δ = Multimodal − Single)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2a_delta_per_neuron.png")

        # ===== 图2b: 纯净版 =====
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        sorted_idx = np.argsort(delta_vs_morph)
        delta_sorted = delta_vs_morph[sorted_idx]
        n = len(delta_sorted)
        x_norm = np.linspace(0, 1, n)
        colors = np.where(delta_sorted > 0, '#27AE60', '#E74C3C')

        for xi, di, ci in zip(x_norm, delta_sorted, colors):
            ax.plot([xi, xi], [0, di], color=ci, alpha=0.6, linewidth=1)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.axhline(y=np.mean(delta_vs_morph), color='#9B59B6', linestyle='-', linewidth=3)

        ax.set_xlim(-0.02, 1.02)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "2b_delta_pure.png")

    def _plot_panel3_comparison_summary(self, output_dir: str):
        """Panel 3: 数据集汇总"""
        datasets = [('full', 'Full-morph'), ('axon', 'Axon-only'), ('dend', 'Dendrite-only')]
        conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        bar_colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, (prefix, title) in enumerate(datasets):
            ax = axes[ax_idx]

            keys = [f'{prefix}_morph_only', f'{prefix}_gene_only', f'{prefix}_morph_gene', f'{prefix}_shuffle']
            x = np.arange(len(conditions))
            width = 0.6

            cos_means = [self.results[k].mean_cosine_sim for k in keys]
            cos_stds = [self.results[k].std_cosine_sim for k in keys]

            bars = ax.bar(x, cos_means, width, yerr=cos_stds, capsize=4,
                          color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1)

            for bar, val in zip(bars, cos_means):
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')

            ax.set_ylabel('Cosine Similarity', fontsize=11)
            ax.set_title(f'{title}\n(N={self.results[keys[0]].n_neurons})',
                         fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Projection Prediction Performance', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_comparison_summary.png")

    def _plot_panel4_cross_modal_prediction(self, output_dir:  str):
        """Panel 4: 跨模态预测结果"""
        if not self.cross_modal_results:
            print("  ⚠ 没有跨模态预测结果")
            return

        # ===== 图4a: 条形图 =====
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        targets = ['Proj', 'Gene', 'Morph']
        target_colors = ['#E74C3C', '#27AE60', '#3498DB']

        for ax_idx, (target, target_color) in enumerate(zip(targets, target_colors)):
            ax = axes[ax_idx]

            relevant_keys = [k for k in self.cross_modal_results.keys()
                             if k.endswith(f'_to_{target.lower()}')]

            if not relevant_keys:
                continue

            labels = []
            r2_values = []

            for key in relevant_keys:
                result = self.cross_modal_results[key]
                labels.append(result.input_modalities)
                r2_values.append(result.r2_score)

            x = np.arange(len(labels))
            bars = ax.bar(x, r2_values, color=target_color, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, r2_values):
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')

            ax.set_ylabel('R² Score', fontsize=11)
            ax.set_title(f'Predict:  {target}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Cross-Modal Prediction:  Input → Target',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "4a_cross_modal_prediction.png")

        # ===== 图4b: 热力图 =====
        fig, ax = plt.subplots(figsize=(10, 6))

        input_combos = ['Morph', 'Gene', 'Proj', 'Morph+Gene', 'Morph+Proj', 'Gene+Proj']
        targets = ['Proj', 'Gene', 'Morph']

        matrix = np.full((len(input_combos), len(targets)), np.nan)

        key_mapping = {
            ('Morph', 'Proj'): 'morph_to_proj',
            ('Gene', 'Proj'): 'gene_to_proj',
            ('Morph+Gene', 'Proj'): 'morph_gene_to_proj',
            ('Morph', 'Gene'): 'morph_to_gene',
            ('Proj', 'Gene'): 'proj_to_gene',
            ('Morph+Proj', 'Gene'): 'morph_proj_to_gene',
            ('Gene', 'Morph'): 'gene_to_morph',
            ('Proj', 'Morph'): 'proj_to_morph',
            ('Gene+Proj', 'Morph'): 'gene_proj_to_morph',
        }

        for (inp, tgt), key in key_mapping.items():
            if key in self.cross_modal_results:
                i = input_combos.index(inp)
                j = targets.index(tgt)
                matrix[i, j] = self.cross_modal_results[key].r2_score

        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)

        for i in range(len(input_combos)):
            for j in range(len(targets)):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if matrix[i, j] > 0.25 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=color)
                else:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='gray')

        ax.set_xticks(np.arange(len(targets)))
        ax.set_yticks(np.arange(len(input_combos)))
        ax.set_xticklabels(targets, fontsize=11, fontweight='medium')
        ax.set_yticklabels(input_combos, fontsize=10)

        ax.set_xlabel('Target Modality', fontsize=12, fontweight='medium')
        ax.set_ylabel('Input Modality', fontsize=12, fontweight='medium')
        ax.set_title('Cross-Modal Prediction R² Score', fontsize=13, fontweight='bold', pad=15)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('R² Score', fontsize=11)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "4b_cross_modal_heatmap.png")

        # ===== 图4c: 双模态 vs 单模态增益 =====
        fig, ax = plt.subplots(figsize=(10, 6))

        gain_data = []
        for target in ['Proj', 'Gene', 'Morph']:
            # 找双模态结果
            dual_keys = [k for k in self.cross_modal_results.keys()
                         if '_to_' + target.lower() in k and '+' in self.cross_modal_results[k].input_modalities]

            # 找单模态结果
            single_keys = [k for k in self.cross_modal_results.keys()
                           if '_to_' + target.lower() in k and '+' not in self.cross_modal_results[k].input_modalities]

            if dual_keys and single_keys:
                dual_r2 = self.cross_modal_results[dual_keys[0]].r2_score
                best_single_r2 = max(self.cross_modal_results[k].r2_score for k in single_keys)
                gain = dual_r2 - best_single_r2

                gain_data.append({
                    'target': target,
                    'dual_r2': dual_r2,
                    'best_single_r2': best_single_r2,
                    'gain': gain,
                    'dual_name': self.cross_modal_results[dual_keys[0]].input_modalities
                })

        if gain_data:
            targets_plot = [d['target'] for d in gain_data]
            dual_r2s = [d['dual_r2'] for d in gain_data]
            single_r2s = [d['best_single_r2'] for d in gain_data]
            gains = [d['gain'] for d in gain_data]

            x = np.arange(len(targets_plot))
            width = 0.35

            bars1 = ax.bar(x - width / 2, single_r2s, width, label='Best Single Modality',
                           color='#95A5A6', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width / 2, dual_r2s, width, label='Dual Modality',
                           color='#3498DB', alpha=0.8, edgecolor='black')

            # 标注数值和增益
            for i, (xi, gain, dual_r2, single_r2) in enumerate(zip(x, gains, dual_r2s, single_r2s)):
                # 单模态数值
                ax.annotate(f'{single_r2:.3f}',
                            xy=(xi - width / 2, single_r2),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')
                # 双模态数值
                ax.annotate(f'{dual_r2:.3f}',
                            xy=(xi + width / 2, dual_r2),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')
                # 增益
                color = '#27AE60' if gain > 0 else '#E74C3C'
                y_pos = max(dual_r2, single_r2) + 0.03
                ax.annotate(f'{gain: +.3f}',
                            xy=(xi, y_pos),
                            ha='center', fontsize=11, fontweight='bold', color=color)

            ax.set_ylabel('R² Score', fontsize=12, fontweight='medium')
            ax.set_title('Dual Modality Gain over Best Single Modality',
                         fontsize=13, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Predict {t}' for t in targets_plot], fontsize=11)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(axis='y', alpha=0.3)

            # 设置y轴范围
            y_max = max(dual_r2s + single_r2s) + 0.08
            ax.set_ylim(0, y_max)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "4c_dual_vs_single_gain.png")

        # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)

        # 保存投射预测结果
        rows = []
        for key, result in self.results.items():
            rows.append({
                'key': key,
                'dataset': result.dataset_name,
                'condition': result.condition_name,
                'n_neurons': result.n_neurons,
                'n_features': result.n_features,
                'mean_cosine_sim': result.mean_cosine_sim,
                'std_cosine_sim': result.std_cosine_sim,
                'mean_jsd_sim': result.mean_jsd_sim,
                'std_jsd_sim': result.std_jsd_sim,
                'r2_score': result.mean_r2
            })
        pd.DataFrame(rows).to_csv(f"{output_dir}/projection_prediction_results.csv", index=False)

        # 保存跨模态预测结果
        if self.cross_modal_results:
            cross_rows = []
            for key, result in self.cross_modal_results.items():
                cross_rows.append({
                    'key': key,
                    'input_modalities': result.input_modalities,
                    'target_modality': result.target_modality,
                    'n_neurons': result.n_neurons,
                    'n_input_features': result.n_input_features,
                    'n_target_features': result.n_target_features,
                    'r2_score': result.r2_score,
                    'mean_corr': result.mean_corr,
                    'std_corr': result.std_corr
                })
            pd.DataFrame(cross_rows).to_csv(f"{output_dir}/cross_modal_prediction_results.csv", index=False)

        print(f"\n✓ 结果已保存到:  {output_dir}")

        # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results", model_name: str = "Ridge"):
        print("\n" + "=" * 80)
        print("任务1:  Neuron-level 多模态预测 (V8 修正版)")
        print("=" * 80)

        n_all, n_dendrite = self.load_all_data()
        if n_all == 0:
            print("\n✗ 没有有效数据")
            return

        # 运行所有实验
        self.run_all_experiments(model_name)

        # 统计分析
        stats = self.statistical_analysis()

        # 可视化
        self.visualize_results(output_dir)

        # 保存结果
        self.save_results(output_dir)

        # 打印结论
        self._print_conclusion(stats)

        print("\n" + "=" * 80)
        print(f"任务1完成!  结果保存在: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self, stats: Dict):
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        # 投射预测结果
        print("\n【实验1：投射预测】")
        print("-" * 60)
        for prefix, name in [('full', 'Full-morph'), ('axon', 'Axon-only'), ('dend', 'Dendrite-only')]:
            if f'{prefix}_morph_only' in self.results:
                print(f"\n  {name}数据集 (N={self.results[f'{prefix}_morph_only'].n_neurons}):")
                print(
                    f"    Morph-only:   {self.results[f'{prefix}_morph_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_only'].std_cosine_sim:.4f}")
                print(
                    f"    Gene-only:   {self.results[f'{prefix}_gene_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_gene_only'].std_cosine_sim:.4f}")
                print(
                    f"    Morph+Gene:  {self.results[f'{prefix}_morph_gene'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_gene'].std_cosine_sim:.4f}")
                print(
                    f"    Shuffle:     {self.results[f'{prefix}_shuffle'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_shuffle'].std_cosine_sim:.4f}")

        # 跨模态预测结果
        if self.cross_modal_results:
            print("\n【实验2：跨模态预测】")
            print("-" * 60)
            print(f"\n  {'Target':<10} {'Input':<15} {'R²': >10} {'Mean Corr':>12}")
            print("  " + "-" * 50)

            for target in ['Proj', 'Gene', 'Morph']:
                relevant = [(k, v) for k, v in self.cross_modal_results.items()
                            if v.target_modality == target]
                for key, result in sorted(relevant, key=lambda x: -x[1].r2_score):
                    print(
                        f"  {target:<10} {result.input_modalities:<15} {result.r2_score:>10.4f} {result.mean_corr: >12.4f}")

            # 增益分析
            print("\n【双模态增益分析】")
            print("-" * 60)
            for target in ['Proj', 'Gene', 'Morph']:
                dual_keys = [k for k in self.cross_modal_results.keys()
                             if self.cross_modal_results[k].target_modality == target
                             and '+' in self.cross_modal_results[k].input_modalities]
                single_keys = [k for k in self.cross_modal_results.keys()
                               if self.cross_modal_results[k].target_modality == target
                               and '+' not in self.cross_modal_results[k].input_modalities]

                if dual_keys and single_keys:
                    dual_r2 = self.cross_modal_results[dual_keys[0]].r2_score
                    best_single_key = max(single_keys, key=lambda k: self.cross_modal_results[k].r2_score)
                    best_single_r2 = self.cross_modal_results[best_single_key].r2_score
                    gain = dual_r2 - best_single_r2

                    dual_name = self.cross_modal_results[dual_keys[0]].input_modalities
                    single_name = self.cross_modal_results[best_single_key].input_modalities

                    print(f"  预测{target}:")
                    print(f"    双模态 ({dual_name}): R²={dual_r2:.4f}")
                    print(f"    最佳单模态 ({single_name}): R²={best_single_r2:.4f}")
                    print(f"    增益: {gain:+.4f}")

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task1_neuron_multimodal_results_v8_fixed"
    MODEL_NAME = "RF"  # 使用RF模型，与您原来的设置一致

    with NeuronMultimodalPredictorV8(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

if __name__ == "__main__":
    main()