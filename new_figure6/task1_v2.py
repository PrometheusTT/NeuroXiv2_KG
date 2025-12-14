"""
任务1：Neuron-level 多模态预测实验 (V3 科学严谨版)
===============================================
测试多模态信息（形态+局部分子环境）是否能提高神经元投射预测

修复内容 (V3):
1. 双数据集方案处理缺失dendrite（不引入偏差）
   - 主结果：只用有dendrite的neuron跑完整64维
   - 补充结果：用全量neuron跑axon-only（32维）
2. 移除Ridge的random_state参数（Ridge没有此参数）
3. 交叉验证折数限制在合理范围（3-10）
4. 增加Gene-only条件

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 统计分析
from scipy.spatial.distance import cosine
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
    dataset_name: str  # "full_morph" or "axon_only"
    cosine_similarities: np.ndarray
    per_target_correlations: Dict[str, float]
    mean_cosine_sim: float
    std_cosine_sim: float
    mean_r2: float
    predicted: np.ndarray
    actual: np.ndarray
    n_neurons: int
    n_features: int


class NeuronMultimodalPredictorV3:
    """神经元多模态投射预测器 V3（科学严谨版）"""

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
        self.local_gene_features: Dict[str, np.ndarray] = {}
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

    # ==================== 诊断 ====================

    def diagnose_database(self):
        """诊断数据库"""
        print("\n" + "=" * 80)
        print("数据库诊断")
        print("=" * 80)

        with self.driver.session(database=self.database) as session:
            # Neuron统计
            result = session.run("MATCH (n:Neuron) RETURN count(n) as count")
            print(f"\n总Neuron数量: {result.single()['count']}")

            # 有轴突数据
            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有axonal数据: {result.single()['count']}")

            # 有树突数据
            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
                RETURN count(n) as count
            """)
            print(f"有dendritic数据: {result.single()['count']}")

            # 有投射
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(s:Subregion)
                RETURN count(DISTINCT n) as count
            """)
            print(f"有投射关系: {result.single()['count']}")

        print("=" * 80)

    # ==================== 数据加载 ====================

    def load_all_data(self) -> Tuple[int, int]:
        """
        加载数据（双数据集方案）

        Returns:
            (全量神经元数, 有dendrite神经元数)
        """
        print("\n" + "=" * 80)
        print("加载神经元数据（双数据集方案）")
        print("=" * 80)

        self.diagnose_database()

        # 获取全局维度
        self._get_global_dimensions()

        # 获取有效神经元（有轴突+有投射）
        self._get_valid_neurons()

        # 加载轴突特征
        self._load_axon_features()

        # 加载树突特征（标记哪些有数据）
        self._load_dendrite_features()

        # 加载投射向量
        self._load_projection_vectors()

        # 计算局部分子环境
        self._compute_local_gene_features()

        # 过滤
        self._filter_valid_neurons()

        print(f"\n✓ 数据加载完成:")
        print(f"  全量神经元（axon-only可用）: {len(self.all_neuron_ids)}")
        print(f"  有dendrite的神经元（full-morph可用）: {len(self.neurons_with_dendrite)}")

        return len(self.all_neuron_ids), len(self.neurons_with_dendrite)

    def _get_global_dimensions(self):
        """获取全局维度"""
        print("\n获取全局特征维度...")

        with self.driver.session(database=self.database) as session:
            # Subclass
            result = session.run("""
                MATCH (s:Subclass) WHERE s.name IS NOT NULL
                RETURN DISTINCT s.name AS name ORDER BY name
            """)
            self.all_subclasses = [r['name'] for r in result if r['name']]

            # 投射目标
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
        """加载投射向量"""
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

                    # Log变换+归一化
                    proj_vector = np.log10(1 + proj_vector)
                    total = proj_vector.sum()
                    if total > 0:
                        proj_vector = proj_vector / total

                    self.projection_vectors[neuron_id] = proj_vector
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _compute_local_gene_features(self):
        """计算局部分子环境"""
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
                # 尝试LOCATE_AT
                result = session.run(query_locate, neuron_id=neuron_id)
                records = list(result)

                # 备用base_region
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
        """过滤数据完整的神经元"""
        print("\n过滤数据完整的神经元...")

        # 全量数据集：需要axon + gene + projection
        valid_all = [n for n in self.all_neuron_ids
                    if n in self.axon_features
                    and n in self.local_gene_features
                    and n in self.projection_vectors]

        # 有dendrite数据集：额外需要dendrite
        valid_dendrite = [n for n in valid_all if n in self.dendrite_features]

        self.all_neuron_ids = valid_all
        self.neurons_with_dendrite = valid_dendrite

        print(f"  全量数据集: {len(self.all_neuron_ids)} 个")
        print(f"  有dendrite数据集: {len(self.neurons_with_dendrite)} 个")

    # ==================== 特征准备 ====================

    def prepare_features_axon_only(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备axon-only数据集的特征（使用全量神经元）

        Returns:
            X_morph, X_gene, X_multi, X_shuffle, Y, neuron_ids
        """
        print("\n准备Axon-only特征矩阵...")

        neurons = self.all_neuron_ids
        n = len(neurons)

        X_morph = np.array([self.axon_features[n] for n in neurons])
        X_gene = np.array([self.local_gene_features[n] for n in neurons])
        Y = np.array([self.projection_vectors[n] for n in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph.shape[1]} 维 (axon-only)")
        print(f"  分子特征: {X_gene.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # 标准化
        scaler_morph = StandardScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph)

        scaler_gene = StandardScaler()
        X_gene_scaled = scaler_gene.fit_transform(X_gene)

        # 多模态
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle
        np.random.seed(42)
        X_gene_shuffled = X_gene_scaled.copy()
        np.random.shuffle(X_gene_shuffled)
        X_shuffle = np.hstack([X_morph_scaled, X_gene_shuffled])

        return X_morph_scaled, X_gene_scaled, X_multi, X_shuffle, Y, neurons

    def prepare_features_full_morph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备full-morph数据集的特征（只使用有dendrite的神经元）

        Returns:
            X_morph, X_gene, X_multi, X_shuffle, Y, neuron_ids
        """
        print("\n准备Full-morph特征矩阵...")

        neurons = self.neurons_with_dendrite
        n = len(neurons)

        # 拼接axon + dendrite
        X_morph = np.array([
            np.concatenate([self.axon_features[nid], self.dendrite_features[nid]])
            for nid in neurons
        ])
        X_gene = np.array([self.local_gene_features[nid] for nid in neurons])
        Y = np.array([self.projection_vectors[nid] for nid in neurons])

        print(f"  神经元数: {n}")
        print(f"  形态特征: {X_morph.shape[1]} 维 (axon+dendrite)")
        print(f"  分子特征: {X_gene.shape[1]} 维")
        print(f"  投射向量: {Y.shape[1]} 维")

        # 标准化
        scaler_morph = StandardScaler()
        X_morph_scaled = scaler_morph.fit_transform(X_morph)

        scaler_gene = StandardScaler()
        X_gene_scaled = scaler_gene.fit_transform(X_gene)

        # 多模态
        X_multi = np.hstack([X_morph_scaled, X_gene_scaled])

        # Shuffle
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
        """训练模型并交叉验证"""
        print(f"\n  训练 {condition_name} ({model_name})...")

        # 选择模型（Ridge没有random_state参数！）
        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                         random_state=42, n_jobs=-1)
        elif model_name == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(128, 64),
                                max_iter=500, random_state=42)
        elif model_name == "GBR":
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                             random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 合理的交叉验证折数（避免LOOCV）
        n_samples = X.shape[0]
        actual_folds = max(3, min(n_folds, n_samples // 10, 10))  # 3-10折
        print(f"    使用 {actual_folds} 折交叉验证 (样本数={n_samples})")

        kfold = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # 计算指标
        cosine_sims = np.zeros(n_samples)
        for i in range(n_samples):
            if np.linalg.norm(Y_pred[i]) > 0 and np.linalg.norm(Y[i]) > 0:
                cosine_sims[i] = 1 - cosine(Y_pred[i], Y[i])

        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # Per-target correlations
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
            per_target_correlations=per_target_corrs,
            mean_cosine_sim=np.mean(cosine_sims),
            std_cosine_sim=np.std(cosine_sims),
            mean_r2=global_r2,
            predicted=Y_pred,
            actual=Y,
            n_neurons=n_samples,
            n_features=X.shape[1]
        )

        print(f"    Cosine Sim: {result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}")
        print(f"    R²: {result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_experiment(self, model_name: str = "Ridge") -> Dict[str, PredictionResult]:
        """运行完整实验（双数据集）"""
        print("\n" + "=" * 80)
        print(f"运行多模态预测实验 (模型: {model_name})")
        print("=" * 80)

        results = {}

        # ========== 数据集1: Full-morph (主结果) ==========
        print("\n" + "=" * 60)
        print("【数据集1: Full-morph (有dendrite的神经元)】")
        print("=" * 60)

        X_morph, X_gene, X_multi, X_shuffle, Y, neurons = self.prepare_features_full_morph()

        print("\n【条件1: Morph-only (axon+dendrite)】")
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

        print("\n【条件1: Morph-only (axon)】")
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

        self.results = results
        return results

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        """统计显著性分析"""
        print("\n" + "=" * 80)
        print("统计显著性分析")
        print("=" * 80)

        stats = {}

        for dataset in ['full', 'axon']:
            print(f"\n【{dataset.upper()} 数据集】")
            print("-" * 60)

            prefix = f'{dataset}_'
            sim_morph = self.results[f'{prefix}morph_only'].cosine_similarities
            sim_gene = self.results[f'{prefix}gene_only'].cosine_similarities
            sim_multi = self.results[f'{prefix}morph_gene'].cosine_similarities
            sim_shuffle = self.results[f'{prefix}shuffle'].cosine_similarities

            comparisons = [
                ('Morph+Gene vs Morph-only', sim_multi, sim_morph),
                ('Morph+Gene vs Gene-only', sim_multi, sim_gene),
                ('Morph+Gene vs Shuffle', sim_multi, sim_shuffle),
                ('Morph-only vs Gene-only', sim_morph, sim_gene),
            ]

            print(f"{'Comparison':<28} {'Wilcoxon p':>12} {'Cohen d':>10} {'Δ%':>8}")
            print("-" * 62)

            for name, sim1, sim2 in comparisons:
                stat, p_val = wilcoxon(sim1, sim2, alternative='greater')
                diff = sim1 - sim2
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                improve = (np.mean(sim1) - np.mean(sim2)) / np.mean(sim2) * 100 if np.mean(sim2) > 0 else 0

                stats[f'{dataset}_{name}'] = {
                    'wilcoxon_stat': stat, 'p_value': p_val,
                    'cohens_d': cohens_d, 'improvement_pct': improve
                }

                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"{name:<28} {p_val:>12.2e} {cohens_d:>10.3f} {improve:>7.2f}% {sig}")

        return stats

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """可视化结果"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        self._plot_comparison_both_datasets(output_dir)
        self._plot_similarity_distributions(output_dir)
        self._plot_target_correlations(output_dir)
        self._plot_prediction_scatter(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_comparison_both_datasets(self, output_dir: str):
        """两个数据集的对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, dataset, title in zip(axes, ['full', 'axon'],
                                       ['Full-morph (有dendrite)', 'Axon-only (全量)']):
            prefix = f'{dataset}_'
            conditions = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
            keys = [f'{prefix}morph_only', f'{prefix}gene_only', f'{prefix}morph_gene', f'{prefix}shuffle']
            colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

            means = [self.results[k].mean_cosine_sim for k in keys]
            stds = [self.results[k].std_cosine_sim for k in keys]

            bars = ax.bar(conditions, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax.set_ylabel('Mean Cosine Similarity', fontsize=11)
            ax.set_title(f'{title}\n(N={self.results[keys[0]].n_neurons}, features={self.results[keys[0]].n_features})',
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(means) * 1.3)

            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_comparison_both_datasets.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_comparison_both_datasets.png")

    def _plot_similarity_distributions(self, output_dir: str):
        """相似度分布（full-morph主结果）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        data = {
            'Morph-only': self.results['full_morph_only'].cosine_similarities,
            'Gene-only': self.results['full_gene_only'].cosine_similarities,
            'Morph+Gene': self.results['full_morph_gene'].cosine_similarities,
            'Shuffle': self.results['full_shuffle'].cosine_similarities
        }

        df = pd.DataFrame(data)
        df_melted = df.melt(var_name='Condition', value_name='Cosine Similarity')
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        sns.violinplot(data=df_melted, x='Condition', y='Cosine Similarity',
                      ax=axes[0], palette=colors)
        axes[0].set_title('Full-morph Dataset: Distribution', fontsize=12, fontweight='bold')

        sns.boxplot(data=df_melted, x='Condition', y='Cosine Similarity',
                   ax=axes[1], palette=colors)
        axes[1].set_title('Full-morph Dataset: Box Plot', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_similarity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 2_similarity_distributions.png")

    def _plot_target_correlations(self, output_dir: str):
        """Per-target correlation"""
        # 使用full-morph数据集
        all_targets = list(self.results['full_morph_only'].per_target_correlations.keys())
        if len(all_targets) == 0:
            print("  ⚠ 没有per-target数据")
            return

        # 按Morph+Gene排序取top 50
        sorted_targets = sorted(all_targets,
                               key=lambda t: self.results['full_morph_gene'].per_target_correlations.get(t, 0),
                               reverse=True)[:50]

        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(sorted_targets))
        width = 0.22

        keys = ['full_morph_only', 'full_gene_only', 'full_morph_gene', 'full_shuffle']
        labels = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
            corrs = [self.results[key].per_target_correlations.get(t, 0) for t in sorted_targets]
            ax.bar(x + i * width, corrs, width, label=label, color=color, alpha=0.8)

        ax.set_ylabel('Pearson Correlation', fontsize=11)
        ax.set_title('Per-Target Prediction Correlation (Top 50, Full-morph)', fontsize=12, fontweight='bold')
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(sorted_targets, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_target_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_target_correlations.png")

    def _plot_prediction_scatter(self, output_dir: str):
        """预测散点图"""
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))

        datasets = ['full', 'axon']
        conditions = ['morph_only', 'gene_only', 'morph_gene', 'shuffle']
        titles = ['Morph-only', 'Gene-only', 'Morph+Gene', 'Shuffle']
        colors = ['#3498DB', '#27AE60', '#E74C3C', '#95A5A6']

        for row, dataset in enumerate(datasets):
            for col, (cond, title, color) in enumerate(zip(conditions, titles, colors)):
                ax = axes[row, col]
                key = f'{dataset}_{cond}'
                result = self.results[key]

                y_true = result.actual.flatten()
                y_pred = result.predicted.flatten()

                n_points = min(3000, len(y_true))
                idx = np.random.choice(len(y_true), n_points, replace=False)

                ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=8, c=color)
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

                corr, _ = pearsonr(y_true, y_pred)
                dataset_label = 'Full' if dataset == 'full' else 'Axon'
                ax.set_title(f'{dataset_label}: {title}\nr={corr:.3f}, R²={result.mean_r2:.3f}', fontsize=10)
                ax.set_xlabel('Actual', fontsize=9)
                ax.set_ylabel('Predicted', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_prediction_scatter.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 汇总
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
                'r2_score': result.mean_r2
            })

        pd.DataFrame(rows).to_csv(f"{output_dir}/experiment_summary.csv", index=False)
        print(f"\n✓ 结果已保存: {output_dir}/experiment_summary.csv")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results", model_name: str = "Ridge"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态投射预测 (V3 科学严谨版)")
        print("=" * 80)

        # 加载数据
        n_all, n_dendrite = self.load_all_data()
        if n_all == 0:
            print("\n✗ 没有有效数据")
            return

        # 运行实验
        self.run_experiment(model_name)

        # 统计分析
        stats = self.statistical_analysis()

        # 可视化
        self.visualize_results(output_dir)

        # 保存
        self.save_results(output_dir)

        # 结论
        self._print_conclusion(stats)

        print("\n" + "=" * 80)
        print(f"任务1完成! 结果保存在: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self, stats: Dict):
        """打印结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        print(f"""
【主结果: Full-morph数据集 (N={self.results['full_morph_only'].n_neurons})】
神经元具有完整的轴突+树突形态数据

性能对比:
  - Morph-only: {self.results['full_morph_only'].mean_cosine_sim:.4f}
  - Gene-only:  {self.results['full_gene_only'].mean_cosine_sim:.4f}
  - Morph+Gene: {self.results['full_morph_gene'].mean_cosine_sim:.4f}
  - Shuffle:    {self.results['full_shuffle'].mean_cosine_sim:.4f}

Morph+Gene vs Morph-only:
  - p值: {stats['full_Morph+Gene vs Morph-only']['p_value']:.2e}
  - Cohen's d: {stats['full_Morph+Gene vs Morph-only']['cohens_d']:.3f}
  - 提升: {stats['full_Morph+Gene vs Morph-only']['improvement_pct']:.2f}%

【补充结果: Axon-only数据集 (N={self.results['axon_morph_only'].n_neurons})】
使用全量神经元，仅轴突形态特征

性能对比:
  - Morph-only: {self.results['axon_morph_only'].mean_cosine_sim:.4f}
  - Gene-only:  {self.results['axon_gene_only'].mean_cosine_sim:.4f}
  - Morph+Gene: {self.results['axon_morph_gene'].mean_cosine_sim:.4f}
  - Shuffle:    {self.results['axon_shuffle'].mean_cosine_sim:.4f}
""")


# ==================== 主程序 ====================

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"

    OUTPUT_DIR = "./task1_neuron_multimodal_results_v3"
    MODEL_NAME = "Ridge"

    with NeuronMultimodalPredictorV3(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE) as predictor:
        predictor.run_full_pipeline(output_dir=OUTPUT_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()