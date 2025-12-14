"""
任务1：Neuron-level 多模态预测实验 (V5 修正版)
===============================================
测试多模态信息（形态+局部分子环境）是否能提高神经元投射预测

V5更新（基于验证过的V4）：
- 新增Dendrite-only数据集（仅使用树突特征）
- 三套数据集完整对比：
  1. Full-morph (axon+dendrite): 有完整重建的神经元
  2. Axon-only: 全量神经元，仅使用轴突特征
  3. Dendrite-only: 有dendrite的神经元，仅使用树突特征

关键：保持V4的所有数据处理逻辑不变，特别是：
- 投射向量：log10变换 + 归一化
- Gene特征：CLR + StandardScaler
- Morph特征：log1p + RobustScaler

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
from scipy.stats import wilcoxon, pearsonr

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# Neo4j
import neo4j

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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


class NeuronMultimodalPredictorV5:
    """神经元多模态投射预测器 V5（基于V4，新增Dendrite-only）"""

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

        # 数据存储
        self.all_neuron_ids: List[str] = []
        self.neurons_with_dendrite: List[str] = []
        self.axon_features: Dict[str, np.ndarray] = {}
        self.dendrite_features: Dict[str, np.ndarray] = {}
        self.local_gene_features: Dict[str, np.ndarray] = {}  # 原始pct_cells
        self.projection_vectors: Dict[str, np.ndarray] = {}

        # 全局维度
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 结果存储
        self.results: Dict[str, PredictionResult] = {}

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 工具函数（与V4完全相同） ====================

    @staticmethod
    def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        """
        Centered Log-Ratio变换（用于compositional数据）
        对于pct_cells这类成分数据，CLR是标准处理方法
        """
        X_pseudo = X + pseudocount
        log_X = np.log(X_pseudo)
        geometric_mean = np.mean(log_X, axis=1, keepdims=True)
        return log_X - geometric_mean

    @staticmethod
    def compute_jsd_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算Jensen-Shannon相似度（1 - JSD）
        JSD适合评估分布预测，语义更符合投射预测任务
        """
        # 确保非负且归一化
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)

        sum_true = y_true.sum()
        sum_pred = y_pred.sum()

        if sum_true > 0:
            y_true = y_true / sum_true
        if sum_pred > 0:
            y_pred = y_pred / sum_pred

        # 处理全零情况
        if sum_true == 0 or sum_pred == 0:
            return 0.0

        # JSD范围是[0, ln(2)]，归一化到[0, 1]
        jsd = jensenshannon(y_true, y_pred, base=2)  # base=2使JSD范围为[0,1]

        if np.isnan(jsd):
            return 0.0

        return 1.0 - jsd  # 转换为相似度

    # ==================== 诊断（与V4相同） ====================

    def diagnose_database(self):
        """诊断数据库"""
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

    # ==================== 数据加载（与V4相同） ====================

    def load_all_data(self) -> Tuple[int, int]:
        """加载数据（双数据集方案）"""
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
        """获取全局维度"""
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
        """获取有轴突数据+有投射的神经元"""
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
        """加载树突特征（只记录有数据的）"""
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
        """
        加载投射向量
        
        关键：Log变换 + 归一化（与V4完全相同）
        """
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

                    # ===== 关键：Log变换+归一化（与V4完全相同） =====
                    proj_vector = np.log10(1 + proj_vector)
                    total = proj_vector.sum()
                    if total > 0:
                        proj_vector = proj_vector / total

                    self.projection_vectors[neuron_id] = proj_vector
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _compute_local_gene_features(self):
        """计算局部分子环境（保存原始pct_cells，后续做CLR）"""
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

                # 保存原始pct_cells向量
                gene_vector = np.zeros(len(self.all_subclasses))
                for i, sc in enumerate(self.all_subclasses):
                    if sc in subclass_dict:
                        gene_vector[i] = subclass_dict[sc]

                self.local_gene_features[neuron_id] = gene_vector
                if len(subclass_dict) > 0:
                    computed += 1

        print(f"  计算了 {computed} 个神经元的局部分子环境（有数据）")

    def _filter_valid_neurons(self):
        """过滤数据完整的神经元"""
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

    def prepare_features_axon_only(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备axon-only数据集的特征（与V4相同）
        """
        print("\n准备Axon-only特征矩阵...")

        neurons = self.all_neuron_ids
        n = len(neurons)

        X_morph_raw = np.array([self.axon_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (axon-only)")
        print(f"  分子特征: {X_gene_raw.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # ===== Morph: log1p + RobustScaler =====
        X_morph_log = np.log1p(X_morph_raw)
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)
        print("  Morph处理: log1p -> RobustScaler")

        # ===== Gene: CLR + StandardScaler =====
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

    def prepare_features_full_morph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备full-morph数据集的特征（与V4相同）
        """
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

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (axon+dendrite)")
        print(f"  分子特征: {X_gene_raw.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # ===== Morph: log1p + RobustScaler =====
        X_morph_log = np.log1p(X_morph_raw)
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)
        print("  Morph处理: log1p -> RobustScaler")

        # ===== Gene: CLR + StandardScaler =====
        X_gene_clr = self.clr_transform(X_gene_raw)
        scaler_gene = StandardScaler()
        X_gene_scaled = scaler_gene.fit_transform(X_gene_clr)
        print("  Gene处理: CLR -> StandardScaler")

        # 多模态拼接
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle
        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    def prepare_features_dendrite_only(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备dendrite-only数据集的特征（V5新增）
        
        使用与full-morph相同的神经元，但只用树突特征
        """
        print("\n准备Dendrite-only特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        # 只用dendrite特征
        X_morph_raw = np.array([self.dendrite_features[nid] for nid in neurons])
        X_gene_raw = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph_raw.shape[1]} 维 (dendrite-only)")
        print(f"  分子特征: {X_gene_raw.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # ===== Morph: log1p + RobustScaler =====
        X_morph_log = np.log1p(X_morph_raw)
        scaler_morph = RobustScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph_log)
        print("  Morph处理: log1p -> RobustScaler")

        # ===== Gene: CLR + StandardScaler =====
        X_gene_clr = self.clr_transform(X_gene_raw)
        scaler_gene = StandardScaler()
        X_gene_scaled = scaler_gene.fit_transform(X_gene_clr)
        print("  Gene处理: CLR -> StandardScaler")

        # 多模态拼接
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle
        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    # ==================== 模型训练（与V4相同） ====================

    def train_and_predict(self, X: np.ndarray, Y: np.ndarray,
                         condition_name: str, dataset_name: str,
                         model_name: str = "Ridge",
                         n_folds: int = 5) -> PredictionResult:
        """训练模型并交叉验证"""
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

        # 计算指标
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

    # ==================== 实验运行（新增Dendrite-only） ====================

    def run_experiment(self, model_name: str = "Ridge") -> Dict[str, PredictionResult]:
        """运行完整实验（三数据集）"""
        print("\n" + "=" * 80)
        print(f"运行多模态预测实验 (模型: {model_name})")
        print("=" * 80)

        results = {}

        # ========== 数据集1: Full-morph (主结果) ==========
        print("\n" + "=" * 60)
        print("【数据集1: Full-morph (有dendrite的神经元, axon+dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_full_morph()

        print("\n【条件1: Morph-only】")
        results['full_morph_only'] = self.train_and_predict(
            X_morph, Y, 'Morph-only', 'full_morph', model_name)

        print("\n【条件2: Gene-only】")
        results['full_gene_only'] = self.train_and_predict(
            X_gene, Y, 'Gene-only', 'full_morph', model_name)

        print("\n【条件3: Morph+Gene】")
        results['full_morph_gene'] = self.train_and_predict(
            X_multi, Y, 'Morph+Gene', 'full_morph', model_name)

        print("\n【条件4: Shuffle】")
        results['full_shuffle'] = self.train_and_predict(
            X_shuffle, Y, 'Shuffle', 'full_morph', model_name)

        # ========== 数据集2: Axon-only (补充结果) ==========
        print("\n" + "=" * 60)
        print("【数据集2: Axon-only (全量神经元)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_axon_only()

        print("\n【条件1: Morph-only】")
        results['axon_morph_only'] = self.train_and_predict(
            X_morph, Y, 'Morph-only', 'axon_only', model_name)

        print("\n【条件2: Gene-only】")
        results['axon_gene_only'] = self.train_and_predict(
            X_gene, Y, 'Gene-only', 'axon_only', model_name)

        print("\n【条件3: Morph+Gene】")
        results['axon_morph_gene'] = self.train_and_predict(
            X_multi, Y, 'Morph+Gene', 'axon_only', model_name)

        print("\n【条件4: Shuffle】")
        results['axon_shuffle'] = self.train_and_predict(
            X_shuffle, Y, 'Shuffle', 'axon_only', model_name)

        # ========== 数据集3: Dendrite-only (V5新增) ==========
        print("\n" + "=" * 60)
        print("【数据集3: Dendrite-only (有dendrite的神经元, 仅dendrite)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_dendrite_only()

        print("\n【条件1: Morph-only】")
        results['dend_morph_only'] = self.train_and_predict(
            X_morph, Y, 'Morph-only', 'dendrite_only', model_name)

        print("\n【条件2: Gene-only】")
        results['dend_gene_only'] = self.train_and_predict(
            X_gene, Y, 'Gene-only', 'dendrite_only', model_name)

        print("\n【条件3: Morph+Gene】")
        results['dend_morph_gene'] = self.train_and_predict(
            X_multi, Y, 'Morph+Gene', 'dendrite_only', model_name)

        print("\n【条件4: Shuffle】")
        results['dend_shuffle'] = self.train_and_predict(
            X_shuffle, Y, 'Shuffle', 'dendrite_only', model_name)

        self.results = results
        return results

    # ==================== 统计分析（与V4相同） ====================

    def statistical_analysis(self) -> Dict:
        """统计显著性分析"""
        print("\n" + "=" * 80)
        print("统计显著性分析")
        print("=" * 80)

        stats = {}

        for dataset in ['full', 'axon', 'dend']:
            print(f"\n【{dataset.upper()} 数据集】")
            print("-" * 70)

            prefix = f'{dataset}_'

            # Cosine similarity
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

    # ==================== 可视化（与V4相同，新增第三个数据集） ====================

    def visualize_results(self, output_dir: str = "."):
        """可视化结果"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_delta_distribution(output_dir)
        self._plot_paired_improvement(output_dir)
        self._plot_comparison_summary(output_dir)
        self._plot_target_heatmap(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_delta_distribution(self, output_dir: str):
        """主图1：Δ Performance分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 使用Full-morph数据集
        sim_morph = self.results['full_morph_only'].cosine_similarities
        sim_gene = self.results['full_gene_only'].cosine_similarities
        sim_multi = self.results['full_morph_gene'].cosine_similarities
        sim_shuffle = self.results['full_shuffle'].cosine_similarities

        # Panel A: Cosine Similarity Δ
        ax = axes[0]

        delta_multi_morph = sim_multi - sim_morph
        delta_multi_gene = sim_multi - sim_gene
        delta_multi_shuffle = sim_multi - sim_shuffle

        sns.kdeplot(delta_multi_morph, ax=ax, color='#3498DB', linewidth=2,
                    label=f'Multi - Morph (med={np.median(delta_multi_morph):.3f})')
        sns.kdeplot(delta_multi_gene, ax=ax, color='#27AE60', linewidth=2,
                    label=f'Multi - Gene (med={np.median(delta_multi_gene):.3f})')
        sns.kdeplot(delta_multi_shuffle, ax=ax, color='#95A5A6', linewidth=2,
                    label=f'Multi - Shuffle (med={np.median(delta_multi_shuffle):.3f})')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=np.median(delta_multi_morph), color='#3498DB', linestyle=':', linewidth=1.5)
        ax.axvline(x=np.median(delta_multi_gene), color='#27AE60', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Δ Cosine Similarity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Performance Gain Distribution (Cosine)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        p_morph = wilcoxon(sim_multi, sim_morph, alternative='greater')[1]
        p_gene = wilcoxon(sim_multi, sim_gene, alternative='greater')[1]
        ax.text(0.02, 0.98, f'vs Morph: p={p_morph:.2e}\nvs Gene: p={p_gene:.2e}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel B: JSD Similarity Δ
        ax = axes[1]

        jsd_morph = self.results['full_morph_only'].jsd_similarities
        jsd_gene = self.results['full_gene_only'].jsd_similarities
        jsd_multi = self.results['full_morph_gene'].jsd_similarities
        jsd_shuffle = self.results['full_shuffle'].jsd_similarities

        delta_multi_morph_jsd = jsd_multi - jsd_morph
        delta_multi_gene_jsd = jsd_multi - jsd_gene
        delta_multi_shuffle_jsd = jsd_multi - jsd_shuffle

        sns.kdeplot(delta_multi_morph_jsd, ax=ax, color='#3498DB', linewidth=2,
                    label=f'Multi - Morph (med={np.median(delta_multi_morph_jsd):.3f})')
        sns.kdeplot(delta_multi_gene_jsd, ax=ax, color='#27AE60', linewidth=2,
                    label=f'Multi - Gene (med={np.median(delta_multi_gene_jsd):.3f})')
        sns.kdeplot(delta_multi_shuffle_jsd, ax=ax, color='#95A5A6', linewidth=2,
                    label=f'Multi - Shuffle (med={np.median(delta_multi_shuffle_jsd):.3f})')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=np.median(delta_multi_morph_jsd), color='#3498DB', linestyle=':', linewidth=1.5)
        ax.axvline(x=np.median(delta_multi_gene_jsd), color='#27AE60', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Δ JSD Similarity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Performance Gain Distribution (JSD)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        p_morph_jsd = wilcoxon(jsd_multi, jsd_morph, alternative='greater')[1]
        p_gene_jsd = wilcoxon(jsd_multi, jsd_gene, alternative='greater')[1]
        ax.text(0.02, 0.98, f'vs Morph: p={p_morph_jsd:.2e}\nvs Gene: p={p_gene_jsd:.2e}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Multimodal Integration Improves Projection Prediction',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_delta_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_delta_distribution.png (主图)")

    def _plot_paired_improvement(self, output_dir: str):
        """主图2：Paired Dot Plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        n_show = min(200, len(self.results['full_morph_only'].cosine_similarities))
        np.random.seed(42)
        idx = np.random.choice(len(self.results['full_morph_only'].cosine_similarities),
                               n_show, replace=False)

        # Panel A: Multi vs Morph-only
        ax = axes[0]
        sim_morph = self.results['full_morph_only'].cosine_similarities[idx]
        sim_multi = self.results['full_morph_gene'].cosine_similarities[idx]

        for i in range(n_show):
            color = '#27AE60' if sim_multi[i] > sim_morph[i] else '#E74C3C'
            ax.plot([0, 1], [sim_morph[i], sim_multi[i]], color=color, alpha=0.3, linewidth=0.5)

        ax.scatter(np.zeros(n_show), sim_morph, s=15, alpha=0.5, c='#3498DB', label='Morph-only')
        ax.scatter(np.ones(n_show), sim_multi, s=15, alpha=0.5, c='#E74C3C', label='Morph+Gene')

        ax.plot([0, 1], [np.mean(sim_morph), np.mean(sim_multi)], 'k-', linewidth=3,
                label=f'Mean: {np.mean(sim_morph):.3f} → {np.mean(sim_multi):.3f}')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Morph-only', 'Morph+Gene'], fontsize=11)
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title('Per-Neuron Improvement\n(Green=↑, Red=↓)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)

        n_improved = np.sum(sim_multi > sim_morph)
        ax.text(0.5, 0.02, f'{n_improved}/{n_show} ({n_improved/n_show:.1%}) improved',
                transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel B: Multi vs Gene-only
        ax = axes[1]
        sim_gene = self.results['full_gene_only'].cosine_similarities[idx]

        for i in range(n_show):
            color = '#27AE60' if sim_multi[i] > sim_gene[i] else '#E74C3C'
            ax.plot([0, 1], [sim_gene[i], sim_multi[i]], color=color, alpha=0.3, linewidth=0.5)

        ax.scatter(np.zeros(n_show), sim_gene, s=15, alpha=0.5, c='#27AE60', label='Gene-only')
        ax.scatter(np.ones(n_show), sim_multi, s=15, alpha=0.5, c='#E74C3C', label='Morph+Gene')

        ax.plot([0, 1], [np.mean(sim_gene), np.mean(sim_multi)], 'k-', linewidth=3,
                label=f'Mean: {np.mean(sim_gene):.3f} → {np.mean(sim_multi):.3f}')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Gene-only', 'Morph+Gene'], fontsize=11)
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title('Per-Neuron Improvement\n(Green=↑, Red=↓)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)

        n_improved = np.sum(sim_multi > sim_gene)
        ax.text(0.5, 0.02, f'{n_improved}/{n_show} ({n_improved/n_show:.1%}) improved',
                transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_paired_improvement.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 2_paired_improvement.png (主图)")

    def _plot_comparison_summary(self, output_dir: str):
        """
        补充图：三个数据集各条件性能汇总
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        datasets = [
            ('full', 'Full-morph Dataset'),
            ('axon', 'Axon-only Dataset'),
            ('dend', 'Dendrite-only Dataset')
        ]

        for ax_idx, (prefix, title) in enumerate(datasets):
            ax = axes[ax_idx]

            conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
            keys = [f'{prefix}_morph_only', f'{prefix}_gene_only', f'{prefix}_morph_gene', f'{prefix}_shuffle']
            colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

            x = np.arange(len(conditions))
            width = 0.35

            # Cosine
            cos_means = [self.results[k].mean_cosine_sim for k in keys]
            cos_stds = [self.results[k].std_cosine_sim for k in keys]
            bars1 = ax.bar(x - width/2, cos_means, width, yerr=cos_stds, capsize=3,
                          label='Cosine Sim', color=colors, alpha=0.8, edgecolor='black')

            # JSD
            jsd_means = [self.results[k].mean_jsd_sim for k in keys]
            jsd_stds = [self.results[k].std_jsd_sim for k in keys]
            bars2 = ax.bar(x + width/2, jsd_means, width, yerr=jsd_stds, capsize=3,
                          label='JSD Sim', color=colors, alpha=0.4, edgecolor='black', hatch='//')

            ax.set_ylabel('Similarity', fontsize=11)
            ax.set_title(f'{title}\n(N={self.results[keys[0]].n_neurons})',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            # 标注Multi最高值
            best_idx = np.argmax(cos_means)
            ax.annotate(f'{cos_means[best_idx]:.3f}',
                       xy=(x[best_idx] - width/2, cos_means[best_idx]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_comparison_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_comparison_summary.png (补充)")

    def _plot_target_heatmap(self, output_dir: str):
        """补充图：Per-target correlation heatmap"""
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

        fig, ax = plt.subplots(figsize=(8, 12))

        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=-0.2, vmax=0.8)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(np.arange(len(sorted_targets)))
        ax.set_yticklabels(sorted_targets, fontsize=8)

        for i in range(len(sorted_targets)):
            if data[i, 2] == data[i].max():
                ax.text(2, i, '★', ha='center', va='center', fontsize=8, color='gold')

        plt.colorbar(im, ax=ax, label='Pearson Correlation')
        ax.set_title('Per-Target Prediction Correlation\n(Top 30 by Morph+Gene)',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_target_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_target_heatmap.png (补充)")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果"""
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
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态投射预测 (V5 三数据集版)")
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
        """打印结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        for prefix, name in [('full', 'Full-morph'), ('axon', 'Axon-only'), ('dend', 'Dendrite-only')]:
            print(f"\n【{name}数据集 (N={self.results[f'{prefix}_morph_only'].n_neurons})】")
            print(f"  Morph-only: {self.results[f'{prefix}_morph_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_only'].std_cosine_sim:.4f}")
            print(f"  Gene-only:  {self.results[f'{prefix}_gene_only'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_gene_only'].std_cosine_sim:.4f}")
            print(f"  Morph+Gene: {self.results[f'{prefix}_morph_gene'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_morph_gene'].std_cosine_sim:.4f}")
            print(f"  Shuffle:    {self.results[f'{prefix}_shuffle'].mean_cosine_sim:.4f} ± {self.results[f'{prefix}_shuffle'].std_cosine_sim:.4f}")

            if f'{prefix}_cos_Morph+Gene vs Morph-only' in stats:
                p_val = stats[f'{prefix}_cos_Morph+Gene vs Morph-only']['p_value']
                print(f"  → Morph+Gene vs Morph-only: p={p_val:.2e} {'***' if p_val < 0.001 else ''}")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task1_neuron_multimodal_results_v5"
    MODEL_NAME = "Ridge"

    with NeuronMultimodalPredictorV5(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()