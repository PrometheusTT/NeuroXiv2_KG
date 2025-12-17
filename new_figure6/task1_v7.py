"""
任务1：Neuron-level 多模态预测实验 (V7 可视化优化版)
===============================================
基于V6修复问题：
1. Panel 2缩进bug修复
2. Panel 1调整aspect ratio突出x方向shift
3. Panel 2重新设计：quantify critical improvement

作者: PrometheusTT
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
    jsd_similarities: np.ndarray
    per_target_correlations: Dict[str, float]
    mean_cosine_sim: float
    std_cosine_sim: float
    mean_jsd_sim: float
    std_jsd_sim: float
    mean_r2: float
    predicted: np.ndarray
    actual: np.ndarray
    n_neurons: int
    n_features: int


class NeuronMultimodalPredictorV7:
    """神经元多模态投射预测器 V7（可视化优化版）"""

    # 轴突形态特征（32维）
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

    # 树突形态特征（32维）
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

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

        self.all_neuron_ids: List[str] = []
        self.neurons_with_dendrite: List[str] = []
        self.axon_features: Dict[str, np.ndarray] = {}
        self.dendrite_features: Dict[str, np.ndarray] = {}
        self.local_gene_features: Dict[str, np.ndarray] = {}
        self.projection_vectors: Dict[str, np.ndarray] = {}

        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        self.results: Dict[str, PredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数 ====================

    @staticmethod
    def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def compute_jsd_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    # ==================== 诊断 ====================

    def diagnose_database(self):
        print("\n" + "=" * 80)
        print("数据库诊断")
        print("=" * 80)

        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n:Neuron) RETURN count(n) as count")
            print(f"\n总Neuron数量: {result.single()['count']}")

            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有axonal数据: {result.single()['count']}")

            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有dendritic数据: {result.single()['count']}")

            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(s:Subregion)
                RETURN count(DISTINCT n) as count
            """)
            print(f"有投射关系: {result.single()['count']}")

        print("=" * 80)

    # ==================== 数据加载 ====================

    def load_all_data(self) -> Tuple[int, int]:
        print("\n" + "=" * 80)
        print("加载神经元数据（三数据集方案）")
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
        print(f"  全量神经元（axon-only可用）: {len(self.all_neuron_ids)}")
        print(f"  有dendrite的神经元（full-morph/dendrite-only可用）: {len(self.neurons_with_dendrite)}")

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

        print(f"  找到 {len(self.all_neuron_ids)} 个有轴突+有投射的神经元")

    def _load_axon_features(self):
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

        print(f"  加载了 {loaded} 个神经元的树突特征（有真实数据）")

    def _load_projection_vectors(self):
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
                proj_dict = {r['target']: r['weight'] for r in result
                             if r['target'] and r['weight']}

                if proj_dict:
                    proj_vector = np.zeros(len(self.all_target_regions))
                    for i, target in enumerate(self.all_target_regions):
                        if target in proj_dict:
                            proj_vector[i] = proj_dict[target]

                    # 关键：log变换+归一化
                    proj_vector = np.log10(1 + proj_vector)
                    total = proj_vector.sum()
                    if total > 0:
                        proj_vector = proj_vector / total

                    self.projection_vectors[neuron_id] = proj_vector
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _compute_local_gene_features(self):
        print("\n计算局部分子环境...")

        query_locate = """
        MATCH (n:Neuron {neuron_id: $neuron_id})-[:LOCATE_AT]->(r:Region)
        OPTIONAL MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN r.acronym AS region, sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        query_base = """
        MATCH (n:Neuron {neuron_id: $neuron_id})
        WITH n.base_region AS base_region WHERE base_region IS NOT NULL
        MATCH (r:Region) WHERE r.acronym = base_region
        OPTIONAL MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
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

        print(f"  计算了 {computed} 个神经元的局部分子环境（有数据）")

    def _filter_valid_neurons(self):
        print("\n过滤数据完整的神经元...")

        valid_all = [n for n in self.all_neuron_ids
                     if n in self.axon_features
                     and n in self.local_gene_features
                     and n in self.projection_vectors]

        valid_dendrite = [n for n in valid_all if n in self.dendrite_features]

        self.all_neuron_ids = valid_all
        self.neurons_with_dendrite = valid_dendrite

        print(f"  全量数据集: {len(self.all_neuron_ids)} 个")
        print(f"  有dendrite数据集: {len(self.neurons_with_dendrite)} 个")

    # ==================== 特征准备 ====================

    def prepare_features_axon_only(self) -> Tuple:
        print("\n准备Axon-only特征矩阵...")

        neurons = self.all_neuron_ids
        n = len(neurons)

        X_morph_raw = np.array([self.axon_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (axon-only)")

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

    def prepare_features_full_morph(self) -> Tuple:
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
        print("\n准备Dendrite-only特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        X_morph_raw = np.array([self.dendrite_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (dendrite-only)")

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

    def train_and_predict(self, X: np.ndarray, Y: np.ndarray,
                          condition_name: str, dataset_name: str,
                          model_name: str = "Ridge",
                          n_folds: int = 5) -> PredictionResult:
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

        print(f"    Cosine Sim: {result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}")
        print(f"    JSD Sim:    {result.mean_jsd_sim:.4f} ± {result.std_jsd_sim:.4f}")
        print(f"    R² (补充):  {result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_experiment(self, model_name: str = "Ridge") -> Dict[str, PredictionResult]:
        print("\n" + "=" * 80)
        print(f"运行多模态预测实验 (模型: {model_name})")
        print("=" * 80)

        results = {}

        # 数据集1: Full-morph
        print("\n" + "=" * 60)
        print("【数据集1: Full-morph (有dendrite的神经元, axon+dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_full_morph()

        results['full_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'full_morph', model_name)
        results['full_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'full_morph', model_name)
        results['full_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'full_morph', model_name)
        results['full_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'full_morph', model_name)

        # 数据集2: Axon-only
        print("\n" + "=" * 60)
        print("【数据集2: Axon-only (全量神经元)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_axon_only()

        results['axon_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'axon_only', model_name)
        results['axon_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'axon_only', model_name)
        results['axon_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'axon_only', model_name)
        results['axon_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'axon_only', model_name)

        # 数据集3: Dendrite-only
        print("\n" + "=" * 60)
        print("【数据集3: Dendrite-only (有dendrite的神经元, 仅dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_dendrite_only()

        results['dend_morph_only'] = self.train_and_predict(X_morph, Y, 'Morph-only', 'dendrite_only', model_name)
        results['dend_gene_only'] = self.train_and_predict(X_gene, Y, 'Gene-only', 'dendrite_only', model_name)
        results['dend_morph_gene'] = self.train_and_predict(X_multi, Y, 'Morph+Gene', 'dendrite_only', model_name)
        results['dend_shuffle'] = self.train_and_predict(X_shuffle, Y, 'Shuffle', 'dendrite_only', model_name)

        self.results = results
        return results

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

            print(f"\n--- Cosine Similarity ---")
            print(f"{'Comparison':<28} {'Wilcoxon p':>12} {'Cohen d':>10} {'Δ%':>8}")
            print("-" * 62)

            comparisons_cos = [
                ('Morph+Gene vs Morph-only', sim_multi, sim_morph),
                ('Morph+Gene vs Gene-only', sim_multi, sim_gene),
                ('Morph+Gene vs Shuffle', sim_multi, sim_shuffle),
            ]

            for name, sim1, sim2 in comparisons_cos:
                stat, p_val = wilcoxon(sim1, sim2, alternative='greater')
                diff = sim1 - sim2
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                improve = (np.mean(sim1) - np.mean(sim2)) / np.mean(sim2) * 100 if np.mean(sim2) > 0 else 0

                stats[f'{dataset}_cos_{name}'] = {
                    'wilcoxon_stat': stat, 'p_value': p_val,
                    'cohens_d': cohens_d, 'improvement_pct': improve
                }

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"{name:<28} {p_val:>12.2e} {cohens_d:>10.3f} {improve:>7.2f}% {sig}")

        return stats

    # ==================== 可视化辅助函数 ====================

    def _save_figure(self, fig, output_dir: str, filename: str):
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ {filename}")

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_panel1_delta_distribution(output_dir)
        self._plot_panel2_critical_improvement(output_dir)
        self._plot_panel3_comparison_summary(output_dir)
        self._plot_panel4_target_heatmap(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_panel1_delta_distribution(self, output_dir: str):
        """
        Panel 1: Delta分布图
        优化：使用更扁的aspect ratio突出x方向shift
        """
        # 获取数据
        sim_morph = self.results['full_morph_only'].cosine_similarities
        sim_gene = self.results['full_gene_only'].cosine_similarities
        sim_multi = self.results['full_morph_gene'].cosine_similarities

        jsd_morph = self.results['full_morph_only'].jsd_similarities
        jsd_gene = self.results['full_gene_only'].jsd_similarities
        jsd_multi = self.results['full_morph_gene'].jsd_similarities

        # 计算Delta
        delta_cos_morph = sim_multi - sim_morph
        delta_cos_gene = sim_multi - sim_gene
        delta_jsd_morph = jsd_multi - jsd_morph
        delta_jsd_gene = jsd_multi - jsd_gene

        # 颜色
        color_morph = '#3498DB'
        color_gene = '#27AE60'

        # ===== 合并图：更扁的比例 =====
        fig, ax = plt.subplots(figsize=(12, 5))  # 更扁：12:5

        # 定义x范围
        x_min = min(delta_cos_morph.min(), delta_cos_gene.min(),
                    delta_jsd_morph.min(), delta_jsd_gene.min()) - 0.02
        x_max = max(delta_cos_morph.max(), delta_cos_gene.max(),
                    delta_jsd_morph.max(), delta_jsd_gene.max()) + 0.02
        x_range = np.linspace(x_min, x_max, 500)

        # 计算KDE
        kde_cos_morph = gaussian_kde(delta_cos_morph)
        kde_cos_gene = gaussian_kde(delta_cos_gene)
        kde_jsd_morph = gaussian_kde(delta_jsd_morph)
        kde_jsd_gene = gaussian_kde(delta_jsd_gene)

        y_cos_morph = kde_cos_morph(x_range)
        y_cos_gene = kde_cos_gene(x_range)
        y_jsd_morph = kde_jsd_morph(x_range)
        y_jsd_gene = kde_jsd_gene(x_range)

        # 绘制Cosine组（实线）
        ax.plot(x_range, y_cos_morph, color=color_morph, linewidth=2.5, linestyle='-', alpha=0.9)
        ax.plot(x_range, y_cos_gene, color=color_gene, linewidth=2.5, linestyle='-', alpha=0.9)

        # 绘制JSD组（虚线）
        ax.plot(x_range, y_jsd_morph, color=color_morph, linewidth=2, linestyle='--', alpha=0.7)
        ax.plot(x_range, y_jsd_gene, color=color_gene, linewidth=2, linestyle='--', alpha=0.7)

        # 填充右移区域
        x_positive = x_range[x_range >= 0]
        ax.fill_between(x_positive, 0, kde_cos_morph(x_positive), color=color_morph, alpha=0.12)
        ax.fill_between(x_positive, 0, kde_cos_gene(x_positive), color=color_gene, alpha=0.12)

        # 零线
        ax.axvline(x=0, color='#333333', linestyle='-', linewidth=2, alpha=0.8, zorder=1)

        # 中位数
        med_cos_morph = np.median(delta_cos_morph)
        med_cos_gene = np.median(delta_cos_gene)

        ax.axvline(x=med_cos_morph, color=color_morph, linestyle=':', linewidth=2, alpha=0.8)
        ax.axvline(x=med_cos_gene, color=color_gene, linestyle=':', linewidth=2, alpha=0.8)

        # 中位数标注（放在图的上方，避免挡住曲线）
        y_max = max(y_cos_morph.max(), y_cos_gene.max(), y_jsd_morph.max(), y_jsd_gene.max())
        ax.annotate(f'Δ={med_cos_morph:.3f}', xy=(med_cos_morph, y_max * 0.95),
                    fontsize=10, fontweight='bold', color=color_morph, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color_morph))
        ax.annotate(f'Δ={med_cos_gene:.3f}', xy=(med_cos_gene, y_max * 0.80),
                    fontsize=10, fontweight='bold', color=color_gene, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color_gene))

        # 箭头指示improvement
        arrow_y = y_max * 0.4
        ax.annotate('', xy=(0.06, arrow_y), xytext=(0, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5, mutation_scale=15))
        ax.text(0.03, arrow_y * 1.15, 'Improvement', fontsize=10, fontweight='bold', color='#E74C3C', ha='center')

        # 坐标轴
        ax.set_xlabel('Δ Similarity (Multimodal − Single Modality)', fontsize=12, fontweight='medium')
        ax.set_ylabel('Density', fontsize=12, fontweight='medium')
        ax.set_title('Performance Gain Distribution: Multimodal Integration Effect',
                     fontsize=13, fontweight='bold', pad=15)

        # 图例
        legend_elements = [
            Line2D([0], [0], color=color_morph, linewidth=2.5, linestyle='-', label='Cosine: Multi − Morph'),
            Line2D([0], [0], color=color_gene, linewidth=2.5, linestyle='-', label='Cosine: Multi − Gene'),
            Line2D([0], [0], color=color_morph, linewidth=2, linestyle='--', label='JSD: Multi − Morph'),
            Line2D([0], [0], color=color_gene, linewidth=2, linestyle='--', label='JSD: Multi − Gene'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95)

        # 设置y轴范围，留更少的空白
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max * 1.05)  # 减少顶部空白

        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "1_delta_distribution.png")

    def _plot_panel2_critical_improvement(self, output_dir: str):
        """
        Panel 2: Critical Improvement Analysis
        展示多模态融合带来的"关键性"改进：
        - 左图：Scatter plot (baseline vs multi) with improvement coloring
        - 右图：Improvement分布 + 关键统计量
        """
        sim_morph = self.results['full_morph_only'].cosine_similarities
        sim_gene = self.results['full_gene_only'].cosine_similarities
        sim_multi = self.results['full_morph_gene'].cosine_similarities

        # 计算improvement
        delta_vs_morph = sim_multi - sim_morph
        delta_vs_gene = sim_multi - sim_gene

        # 计算best single modality baseline
        best_single = np.maximum(sim_morph, sim_gene)
        delta_vs_best = sim_multi - best_single

        # ===== 综合图：3 panels =====
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

        # ----- Panel A: Scatter (Best Single vs Multi) -----
        ax1 = fig.add_subplot(gs[0])

        # 按improvement程度着色
        colors = np.where(delta_vs_best > 0.05, '#27AE60',  # 显著改进
                          np.where(delta_vs_best > 0, '#82E0AA',  # 轻微改进
                                   np.where(delta_vs_best > -0.05, '#F5B041', '#E74C3C')))  # 退步

        ax1.scatter(best_single, sim_multi, c=colors, s=15, alpha=0.5, edgecolors='none')

        # y=x线
        lims = [min(best_single.min(), sim_multi.min()) - 0.02,
                max(best_single.max(), sim_multi.max()) + 0.02]
        ax1.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='y = x')

        # 均值点
        ax1.scatter(np.mean(best_single), np.mean(sim_multi), s=200, c='black',
                    marker='*', edgecolors='white', linewidth=1.5, zorder=10,
                    label=f'Mean: ({np.mean(best_single):.3f}, {np.mean(sim_multi):.3f})')

        ax1.set_xlabel('Best Single Modality', fontsize=11, fontweight='medium')
        ax1.set_ylabel('Multimodal (Morph+Gene)', fontsize=11, fontweight='medium')
        ax1.set_title('A. Multimodal vs Best Single', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_aspect('equal')

        for spine in ['top', 'right']:
            ax1.spines[spine].set_visible(False)

        # ----- Panel B: Improvement Histogram -----
        ax2 = fig.add_subplot(gs[1])

        # Histogram of improvement vs best single
        bins = np.linspace(-0.15, 0.15, 40)
        n_improve = np.sum(delta_vs_best > 0)
        n_total = len(delta_vs_best)
        pct_improve = n_improve / n_total * 100

        ax2.hist(delta_vs_best, bins=bins, color='#3498DB', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.axvline(x=0, color='#333333', linestyle='-', linewidth=2, alpha=0.8)
        ax2.axvline(x=np.median(delta_vs_best), color='#E74C3C', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(delta_vs_best):.4f}')

        # 填充正向区域
        ax2.axvspan(0, 0.15, alpha=0.15, color='#27AE60')

        ax2.set_xlabel('Δ Cosine Sim (Multi − Best Single)', fontsize=11, fontweight='medium')
        ax2.set_ylabel('Count', fontsize=11, fontweight='medium')
        ax2.set_title('B. Improvement Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)

        # 统计文本
        ax2.text(0.98, 0.98, f'Improved: {n_improve}/{n_total}\n({pct_improve:.1f}%)',
                 transform=ax2.transAxes, fontsize=10, fontweight='bold',
                 va='top', ha='right', color='#27AE60',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)

        # ----- Panel C: Key Statistics Bar -----
        ax3 = fig.add_subplot(gs[2])

        # 计算关键统计量
        stats_data = {
            'Mean Δ vs Morph': np.mean(delta_vs_morph),
            'Mean Δ vs Gene': np.mean(delta_vs_gene),
            'Mean Δ vs Best': np.mean(delta_vs_best),
            'Median Δ vs Best': np.median(delta_vs_best),
            '% Improved vs Best': pct_improve,
            '% Improved >0.02': np.sum(delta_vs_best > 0.02) / n_total * 100,
        }

        labels = list(stats_data.keys())
        values = list(stats_data.values())

        # 分两组：Delta值 和 百分比
        delta_labels = labels[:4]
        delta_values = values[:4]
        pct_labels = labels[4:]
        pct_values = values[4:]

        # 子图：左侧Delta值，右侧百分比
        x_delta = np.arange(len(delta_labels))
        colors_delta = ['#3498DB', '#27AE60', '#9B59B6', '#E74C3C']
        bars1 = ax3.bar(x_delta, delta_values, color=colors_delta, alpha=0.8, edgecolor='black', linewidth=1)

        # 添加数值标注
        for bar, val in zip(bars1, delta_values):
            height = bar.get_height()
            ax3.annotate(f'{val:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax3.set_xticks(x_delta)
        ax3.set_xticklabels(['vs Morph', 'vs Gene', 'vs Best', 'Median\nvs Best'],
                            fontsize=9, rotation=0)
        ax3.set_ylabel('Mean Δ Cosine Similarity', fontsize=11, fontweight='medium')
        ax3.set_title('C. Key Improvement Metrics', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # 右侧y轴显示百分比
        ax3_right = ax3.twinx()
        x_pct = np.array([len(delta_labels) + 0.5, len(delta_labels) + 1.5])
        bars2 = ax3_right.bar(x_pct, pct_values, color=['#2ECC71', '#1ABC9C'], alpha=0.8,
                              edgecolor='black', linewidth=1, width=0.8)

        for bar, val in zip(bars2, pct_values):
            height = bar.get_height()
            ax3_right.annotate(f'{val:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27AE60')

        ax3_right.set_ylabel('% of Neurons', fontsize=11, fontweight='medium', color='#27AE60')
        ax3_right.tick_params(axis='y', colors='#27AE60')
        ax3_right.set_ylim(0, 100)

        # 设置x轴范围和标签
        ax3.set_xlim(-0.5, len(delta_labels) + 2.5)
        ax3.set_xticks(list(x_delta) + list(x_pct))
        ax3.set_xticklabels(['vs Morph', 'vs Gene', 'vs Best', 'Median\nvs Best',
                             'Improved\n(%)', 'Improved\n>0.02 (%)'],
                            fontsize=9)

        for spine in ['top']:
            ax3.spines[spine].set_visible(False)
            ax3_right.spines[spine].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_critical_improvement.png")

        # ===== 额外：简洁的paired plot（修复版） =====
        self._plot_paired_simple(output_dir, sim_morph, sim_gene, sim_multi)

    def _plot_paired_simple(self, output_dir: str, sim_morph, sim_gene, sim_multi):
        """简洁的paired comparison plot（修复缩进bug）"""
        n_show = min(300, len(sim_morph))
        np.random.seed(42)
        idx = np.random.choice(len(sim_morph), n_show, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for ax_idx, (baseline_sim, baseline_name, baseline_color) in enumerate([
            (sim_morph[idx], 'Morph-only', '#3498DB'),
            (sim_gene[idx], 'Gene-only', '#27AE60')
        ]):
            ax = axes[ax_idx]
            multi_sim = sim_multi[idx]

            improved_mask = multi_sim > baseline_sim

            # 先画退步的线（底层）
            for i in np.where(~improved_mask)[0]:
                ax.plot([0, 1], [baseline_sim[i], multi_sim[i]],
                        color='#E74C3C', alpha=0.25, linewidth=0.6, zorder=1)

            # 再画改进的线（上层）
            for i in np.where(improved_mask)[0]:
                ax.plot([0, 1], [baseline_sim[i], multi_sim[i]],
                        color='#27AE60', alpha=0.35, linewidth=0.8, zorder=2)

            # 散点（在for循环外面！）
            ax.scatter(np.zeros(n_show), baseline_sim, s=25, alpha=0.6, c=baseline_color,
                       edgecolors='white', linewidth=0.5, zorder=3)
            ax.scatter(np.ones(n_show), multi_sim, s=25, alpha=0.6, c='#E74C3C',
                       edgecolors='white', linewidth=0.5, zorder=3)

            # 均值线
            mean_baseline = np.mean(baseline_sim)
            mean_multi = np.mean(multi_sim)
            ax.plot([0, 1], [mean_baseline, mean_multi], 'k-', linewidth=4, zorder=10)
            ax.scatter([0, 1], [mean_baseline, mean_multi], s=150, c='black',
                       edgecolors='white', linewidth=2, zorder=11)

            # 统计
            n_improved = np.sum(improved_mask)
            pct = n_improved / n_show * 100

            ax.set_xticks([0, 1])
            ax.set_xticklabels([baseline_name, 'Morph+Gene'], fontsize=11, fontweight='medium')
            ax.set_ylabel('Cosine Similarity', fontsize=11, fontweight='medium')
            ax.set_title(f'{baseline_name} → Multimodal\n{n_improved}/{n_show} ({pct:.1f}%) improved',
                         fontsize=12, fontweight='bold')

            # 均值标注
            ax.annotate(f'{mean_baseline:.3f}', xy=(0, mean_baseline), xytext=(-0.15, mean_baseline),
                        fontsize=10, fontweight='bold', ha='right')
            ax.annotate(f'{mean_multi:.3f}', xy=(1, mean_multi), xytext=(1.15, mean_multi),
                        fontsize=10, fontweight='bold', ha='left')

            ax.set_xlim(-0.3, 1.3)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "2b_paired_comparison.png")

    def _plot_panel3_comparison_summary(self, output_dir: str):
        """Panel 3: 三个数据集汇总"""
        datasets = [
            ('full', 'Full-morph Dataset'),
            ('axon', 'Axon-only Dataset'),
            ('dend', 'Dendrite-only Dataset')
        ]

        conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        bar_colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, (prefix, title) in enumerate(datasets):
            ax = axes[ax_idx]

            keys = [f'{prefix}_morph_only', f'{prefix}_gene_only', f'{prefix}_morph_gene', f'{prefix}_shuffle']
            x = np.arange(len(conditions))
            width = 0.35

            cos_means = [self.results[k].mean_cosine_sim for k in keys]
            cos_stds = [self.results[k].std_cosine_sim for k in keys]
            bars1 = ax.bar(x - width / 2, cos_means, width, yerr=cos_stds, capsize=3,
                           label='Cosine Sim', color=bar_colors, alpha=0.85,
                           edgecolor='black', linewidth=1)

            jsd_means = [self.results[k].mean_jsd_sim for k in keys]
            jsd_stds = [self.results[k].std_jsd_sim for k in keys]
            bars2 = ax.bar(x + width / 2, jsd_means, width, yerr=jsd_stds, capsize=3,
                           label='JSD Sim', color=bar_colors, alpha=0.4,
                           edgecolor='black', linewidth=1, hatch='//')

            ax.set_ylabel('Similarity', fontsize=11)
            ax.set_title(f'{title}\n(N={self.results[keys[0]].n_neurons})',
                         fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha='right')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(axis='y', alpha=0.3)

            # 标注最高值
            best_idx = np.argmax(cos_means)
            ax.annotate(f'{cos_means[best_idx]:.3f}',
                        xy=(x[best_idx] - width / 2, cos_means[best_idx]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.suptitle('Performance Comparison Across Datasets', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_comparison_summary.png")

    def _plot_panel4_target_heatmap(self, output_dir: str):
        """Panel 4: Per-target heatmap"""
        all_targets = list(self.results['full_morph_only'].per_target_correlations.keys())
        if len(all_targets) == 0:
            print("  ⚠ 没有per-target数据")
            return

        sorted_targets = sorted(all_targets,
                                key=lambda t: self.results['full_morph_gene'].per_target_correlations.get(t, 0),
                                reverse=True)[:30]

        keys = ['full_morph_only', 'full_gene_only', 'full_morph_gene', 'full_shuffle']
        labels = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']

        data = []
        for target in sorted_targets:
            row = [self.results[k].per_target_correlations.get(target, 0) for k in keys]
            data.append(row)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=(8, 10))

        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-0.2, vmax=0.8)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=10, fontweight='medium')
        ax.set_yticks(np.arange(len(sorted_targets)))
        ax.set_yticklabels(sorted_targets, fontsize=8)

        # 标记每行最佳
        for i in range(len(sorted_targets)):
            best_j = np.argmax(data[i])
            if data[i, best_j] > 0:
                ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor='gold', linewidth=2))

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson Correlation', fontsize=10)

        ax.set_title('Per-Target Prediction Correlation\n(Top 30 by Morph+Gene)',
                     fontsize=12, fontweight='bold', pad=10)

        fig.tight_layout()
        self._save_figure(fig, output_dir, "4_target_heatmap.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)

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

        pd.DataFrame(rows).to_csv(f"{output_dir}/experiment_summary.csv", index=False)
        print(f"\n✓ 结果已保存: {output_dir}/experiment_summary.csv")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results", model_name: str = "Ridge"):
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态投射预测 (V7 可视化优化版)")
        print("=" * 80)

        n_all, n_dendrite = self.load_all_data()
        if n_all == 0:
            print("\n✗ 没有有效数据")
            return

        self.run_experiment(model_name)
        stats = self.statistical_analysis()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_conclusion(stats)

        print("\n" + "=" * 80)
        print(f"任务1完成! 结果保存在: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self, stats: Dict):
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        for prefix, name in [('full', 'Full-morph'), ('axon', 'Axon-only'), ('dend', 'Dendrite-only')]:
            print(f"\n【{name}数据集 (N={self.results[f'{prefix}_morph_only'].n_neurons})】")
            print(f"  Morph-only:  {self.results[f'{prefix}_morph_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_only'].std_cosine_sim:.4f}")
            print(f"  Gene-only:   {self.results[f'{prefix}_gene_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_gene_only'].std_cosine_sim:.4f}")
            print(f"  Morph+Gene:  {self.results[f'{prefix}_morph_gene'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_gene'].std_cosine_sim:.4f}")
            print(f"  Shuffle:     {self.results[f'{prefix}_shuffle'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_shuffle'].std_cosine_sim:.4f}")

            if f'{prefix}_cos_Morph+Gene vs Morph-only' in stats:
                p_val = stats[f'{prefix}_cos_Morph+Gene vs Morph-only']['p_value']
                print(f"  → Morph+Gene vs Morph-only: p={p_val:.2e} {'***' if p_val < 0.001 else ''}")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task1_neuron_multimodal_results_v7_RF"
    MODEL_NAME = "RF"

    with NeuronMultimodalPredictorV7(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()