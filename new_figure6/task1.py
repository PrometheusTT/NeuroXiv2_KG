"""
任务1：Neuron-level 多模态预测实验
===============================================
测试多模态信息（形态+局部分子环境）是否能提高神经元投射预测

核心科学问题：
1. 单一模态（morphology）能否决定一个神经元的投射模式？
2. 加入局部分子环境（local MERFISH subclass composition）后，是否可以更好地预测投射？

方法：
- Morph-only: 仅使用形态特征预测投射
- Morph+Gene: 使用形态特征+局部分子环境预测投射
- Shuffle Control: 使用形态特征+打乱的分子环境（验证真实效应）

评价指标：
- Cosine similarity
- R² (coefficient of determination)
- Per-target region correlation
- Statistical significance (Wilcoxon signed-rank test, Cohen's d)

依赖：
- neo4j
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn

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
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 统计分析
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, pearsonr, spearmanr
from scipy.spatial.distance import cdist

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# Neo4j
import neo4j

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class PredictionResult:
    """预测结果数据类"""
    model_name: str
    cosine_similarities: np.ndarray
    per_target_correlations: Dict[str, float]
    mean_cosine_sim: float
    std_cosine_sim: float
    mean_r2: float
    predicted: np.ndarray
    actual: np.ndarray


class NeuronMultimodalPredictor:
    """神经元多模态投射预测器"""

    def __init__(self, uri: str, user: str, password: str, radius_um: float = 100.0):
        """
        初始化预测器

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            radius_um: 局部分子环境的搜索半径（微米）
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.radius_um = radius_um

        # 数据存储
        self.neuron_ids: List[str] = []
        self.morph_features: Dict[str, np.ndarray] = {}
        self.local_gene_features: Dict[str, np.ndarray] = {}
        self.projection_vectors: Dict[str, np.ndarray] = {}

        # 全局维度
        self.morph_feature_names: List[str] = []
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # 结果存储
        self.results: Dict[str, PredictionResult] = {}

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 诊断功能 ====================

    def diagnose_database(self):
        """诊断Neo4j数据库结构"""
        print("\n" + "=" * 80)
        print("数据库诊断")
        print("=" * 80)

        with self.driver.session() as session:
            # 1. 检查Neuron节点
            print("\n【Neuron节点诊断】")
            result = session.run("MATCH (n:Neuron) RETURN count(n) as count")
            neuron_count = result.single()['count']
            print(f"  总Neuron数量: {neuron_count}")

            # 检查Neuron属性
            result = session.run("MATCH (n:Neuron) RETURN keys(n) as keys LIMIT 1")
            record = result.single()
            if record:
                print(f"  Neuron属性: {record['keys']}")

            # 检查有形态特征的神经元
            result = session.run("""
                MATCH (n:Neuron) 
                WHERE n.axonal_length IS NOT NULL AND n.axonal_length > 0
                RETURN count(n) as count
            """)
            morph_count = result.single()['count']
            print(f"  有轴突形态数据: {morph_count}")

            # 2. 检查投射关系
            print("\n【投射关系诊断】")
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->()
                RETURN count(p) as count
            """)
            proj_count = result.single()['count']
            print(f"  总PROJECT_TO关系数: {proj_count}")

            # 检查投射到Subregion
            result = session.run("""
                MATCH (n:Neuron)-[p:PROJECT_TO]->(s:Subregion)
                RETURN count(DISTINCT n) as neuron_count, count(p) as proj_count
            """)
            record = result.single()
            print(f"  投射到Subregion的神经元数: {record['neuron_count']}")
            print(f"  Neuron->Subregion关系数: {record['proj_count']}")

            # 3. 检查LOCATE_AT关系
            print("\n【LOCATE_AT关系诊断】")
            result = session.run("""
                MATCH (n:Neuron)-[l:LOCATE_AT]->(r:Region)
                RETURN count(DISTINCT n) as neuron_count
            """)
            locate_count = result.single()['neuron_count']
            print(f"  有LOCATE_AT关系的神经元数: {locate_count}")

            # 4. 检查Region和Subclass
            print("\n【Region-Subclass关系诊断】")
            result = session.run("""
                MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
                RETURN count(DISTINCT r) as region_count, count(DISTINCT s) as subclass_count
            """)
            record = result.single()
            print(f"  有Subclass数据的Region数: {record['region_count']}")
            print(f"  总Subclass数: {record['subclass_count']}")

            # 5. 显示样例数据
            print("\n【样例Neuron数据】")
            result = session.run("""
                MATCH (n:Neuron)
                WHERE n.axonal_length IS NOT NULL
                RETURN n.neuron_id as id, n.base_region as region, 
                       n.axonal_length as axon_len, n.dendritic_length as den_len
                LIMIT 3
            """)
            for record in result:
                print(f"  ID: {record['id']}, Region: {record['region']}, "
                      f"Axon: {record['axon_len']}, Dendrite: {record['den_len']}")

        print("=" * 80)

    # ==================== 数据加载 ====================

    def load_all_data(self) -> int:
        """
        加载所有需要的数据

        Returns:
            有效神经元数量
        """
        print("\n" + "=" * 80)
        print("加载神经元多模态数据")
        print("=" * 80)

        # Step 0: 诊断数据库
        self.diagnose_database()

        # Step 1: 获取全局维度
        self._get_global_dimensions()

        # Step 2: 获取有完整数据的神经元列表
        self._get_valid_neurons()

        if len(self.neuron_ids) == 0:
            print("\n⚠ 没有找到有效神经元，尝试放宽条件...")
            self._get_valid_neurons_relaxed()

        if len(self.neuron_ids) == 0:
            print("\n✗ 仍然没有找到有效神经元，请检查数据库")
            return 0

        # Step 3: 加载形态特征
        self._load_morphology_features()

        # Step 4: 加载投射向量
        self._load_projection_vectors()

        # Step 5: 计算局部分子环境特征
        self._compute_local_gene_features()

        print(f"\n✓ 数据加载完成: {len(self.neuron_ids)} 个有效神经元")
        return len(self.neuron_ids)

    def _get_global_dimensions(self):
        """获取全局特征维度"""
        print("\n获取全局特征维度...")

        # 形态特征名称 - 根据KG构建代码中的实际属性名
        self.morph_feature_names = [
            'axonal_length', 'axonal_branches',
            'axonal_bifurcation_remote_angle', 'axonal_maximum_branch_order',
            'dendritic_length', 'dendritic_branches',
            'dendritic_bifurcation_remote_angle', 'dendritic_maximum_branch_order'
        ]
        print(f"  形态特征: {len(self.morph_feature_names)} 维")

        # 获取所有subclass - 使用正确的节点标签
        query = """
        MATCH (s:Subclass)
        RETURN DISTINCT s.name AS subclass_name
        ORDER BY subclass_name
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_subclasses = [record['subclass_name'] for record in result if record['subclass_name']]
        print(f"  Subclass类型: {len(self.all_subclasses)} 种")

        # 获取所有投射目标区域 - Subregion节点
        query = """
        MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_target_regions = [record['target'] for record in result if record['target']]
        print(f"  投射目标区域: {len(self.all_target_regions)} 个")

    def _get_valid_neurons(self):
        """获取有完整形态和投射数据的神经元"""
        print("\n获取有效神经元列表...")

        # 根据实际KG结构：Neuron有axonal_length属性，且有PROJECT_TO关系到Subregion
        query = """
        MATCH (n:Neuron)
        WHERE n.axonal_length IS NOT NULL 
          AND n.axonal_length > 0
        WITH n
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        WITH n, COUNT(DISTINCT t) AS n_targets
        WHERE n_targets >= 3
        RETURN n.neuron_id AS neuron_id
        ORDER BY neuron_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.neuron_ids = [record['neuron_id'] for record in result if record['neuron_id']]

        print(f"  找到 {len(self.neuron_ids)} 个有效神经元")

    def _get_valid_neurons_relaxed(self):
        """放宽条件获取神经元"""
        print("\n尝试放宽条件获取神经元...")

        # 方案1: 只要有投射关系即可
        query = """
        MATCH (n:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        WITH n, COUNT(DISTINCT t) AS n_targets
        WHERE n_targets >= 1
        RETURN n.neuron_id AS neuron_id
        ORDER BY neuron_id
        LIMIT 5000
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.neuron_ids = [record['neuron_id'] for record in result if record['neuron_id']]

        print(f"  放宽条件后找到 {len(self.neuron_ids)} 个神经元")

        if len(self.neuron_ids) == 0:
            # 方案2: 检查是否有任何神经元
            query = """
            MATCH (n:Neuron)
            WHERE n.axonal_length IS NOT NULL
            RETURN n.neuron_id AS neuron_id
            LIMIT 1000
            """
            with self.driver.session() as session:
                result = session.run(query)
                self.neuron_ids = [record['neuron_id'] for record in result if record['neuron_id']]
            print(f"  仅检查形态数据，找到 {len(self.neuron_ids)} 个神经元")

    def _load_morphology_features(self):
        """加载神经元形态特征"""
        print("\n加载形态特征...")

        # 使用实际的属性名
        query = """
        MATCH (n:Neuron)
        WHERE n.neuron_id = $neuron_id
        RETURN 
            n.axonal_length AS axonal_length,
            n.axonal_branches AS axonal_branches,
            n.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
            n.axonal_maximum_branch_order AS axonal_maximum_branch_order,
            n.dendritic_length AS dendritic_length,
            n.dendritic_branches AS dendritic_branches,
            n.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
            n.dendritic_maximum_branch_order AS dendritic_maximum_branch_order
        """

        loaded = 0
        with self.driver.session() as session:
            for neuron_id in self.neuron_ids:
                result = session.run(query, neuron_id=neuron_id)
                record = result.single()

                if record:
                    features = []
                    for feat_name in self.morph_feature_names:
                        val = record[feat_name]
                        features.append(float(val) if val is not None else 0.0)

                    self.morph_features[neuron_id] = np.array(features)
                    loaded += 1

        print(f"  加载了 {loaded} 个神经元的形态特征")

    def _load_projection_vectors(self):
        """加载神经元投射向量"""
        print("\n加载投射向量...")

        # 根据实际KG结构：Neuron-[:PROJECT_TO]->Subregion，关系有weight属性
        query = """
        MATCH (n:Neuron {neuron_id: $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, p.weight AS weight
        """

        loaded = 0
        with self.driver.session() as session:
            for neuron_id in self.neuron_ids:
                result = session.run(query, neuron_id=neuron_id)

                # 构建投射向量
                proj_dict = {record['target']: record['weight'] for record in result if record['target']}

                if not proj_dict:
                    continue

                # 创建固定维度向量
                proj_vector = np.zeros(len(self.all_target_regions))
                for i, target in enumerate(self.all_target_regions):
                    if target in proj_dict:
                        proj_vector[i] = proj_dict[target]

                # Log变换并归一化
                proj_vector = np.log10(1 + proj_vector)
                total = proj_vector.sum()
                if total > 0:
                    proj_vector = proj_vector / total

                self.projection_vectors[neuron_id] = proj_vector
                loaded += 1

        print(f"  加载了 {loaded} 个神经元的投射向量")

    def _compute_local_gene_features(self):
        """计算每个神经元的局部分子环境特征"""
        print(f"\n计算局部分子环境特征...")

        # 根据实际KG结构：
        # Neuron-[:LOCATE_AT]->Region
        # Region-[:HAS_SUBCLASS]->Subclass (关系有pct_cells属性)

        # 获取神经元所在区域
        query_region = """
        MATCH (n:Neuron {neuron_id: $neuron_id})
        OPTIONAL MATCH (n)-[:LOCATE_AT]->(r:Region)
        RETURN r.acronym AS region_acronym, n.base_region AS base_region
        """

        # 获取区域的subclass组成
        query_subclass = """
        MATCH (r:Region {acronym: $region})-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        computed = 0
        with self.driver.session() as session:
            for neuron_id in self.neuron_ids:
                # 获取神经元所在区域
                result = session.run(query_region, neuron_id=neuron_id)
                record = result.single()

                region = None
                if record:
                    region = record['region_acronym'] or record['base_region']

                if region:
                    # 使用区域的subclass组成作为局部分子环境
                    result_sc = session.run(query_subclass, region=region)

                    subclass_dict = {}
                    for r in result_sc:
                        if r['subclass_name'] and r['pct_cells']:
                            subclass_dict[r['subclass_name']] = r['pct_cells']

                    # 构建特征向量
                    gene_vector = np.zeros(len(self.all_subclasses))
                    for i, sc in enumerate(self.all_subclasses):
                        if sc in subclass_dict:
                            gene_vector[i] = subclass_dict[sc]

                    self.local_gene_features[neuron_id] = gene_vector
                    computed += 1
                else:
                    # 如果没有区域信息，使用零向量
                    self.local_gene_features[neuron_id] = np.zeros(len(self.all_subclasses))

        print(f"  计算了 {computed} 个神经元的局部分子环境")

    # ==================== 特征准备 ====================

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据

        Returns:
            X_morph: 形态特征矩阵 (N x D_morph)
            X_gene: 分子特征矩阵 (N x D_gene)
            X_multi: 多模态特征矩阵 (N x (D_morph + D_gene))
            Y: 投射向量矩阵 (N x K)
            valid_neurons: 有效神经元列表
        """
        print("\n准备特征矩阵...")

        # 过滤有完整数据的神经元
        valid_neurons = [n for n in self.neuron_ids
                        if n in self.morph_features
                        and n in self.local_gene_features
                        and n in self.projection_vectors]

        print(f"  有效神经元数: {len(valid_neurons)}")

        if len(valid_neurons) == 0:
            raise ValueError("没有有效的神经元数据！请检查数据加载步骤。")

        # 构建矩阵
        X_morph = np.array([self.morph_features[n] for n in valid_neurons])
        X_gene = np.array([self.local_gene_features[n] for n in valid_neurons])
        Y = np.array([self.projection_vectors[n] for n in valid_neurons])

        print(f"  原始形态特征维度: {X_morph.shape}")
        print(f"  原始分子特征维度: {X_gene.shape}")
        print(f"  原始投射向量维度: {Y.shape}")

        # 标准化形态特征
        scaler_morph = StandardScaler()
        X_morph = scaler_morph.fit_transform(X_morph)

        # 多模态特征拼接
        X_multi = np.hstack([X_morph, X_gene])

        print(f"  标准化后形态特征维度: {X_morph.shape}")
        print(f"  多模态特征维度: {X_multi.shape}")

        return X_morph, X_gene, X_multi, Y, valid_neurons

    def create_shuffled_features(self, X_morph: np.ndarray,
                                 X_gene: np.ndarray,
                                 random_state: int = 42) -> np.ndarray:
        """
        创建打乱的多模态特征（shuffle control）

        Args:
            X_morph: 形态特征
            X_gene: 分子特征
            random_state: 随机种子

        Returns:
            打乱后的多模态特征
        """
        np.random.seed(random_state)
        X_gene_shuffled = X_gene.copy()
        np.random.shuffle(X_gene_shuffled)

        return np.hstack([X_morph, X_gene_shuffled])

    # ==================== 模型训练与预测 ====================

    def train_and_predict(self, X: np.ndarray, Y: np.ndarray,
                         model_name: str = "Ridge",
                         n_folds: int = 5) -> PredictionResult:
        """
        训练模型并进行交叉验证预测

        Args:
            X: 特征矩阵
            Y: 目标矩阵
            model_name: 模型名称 ("Ridge", "RF", "MLP")
            n_folds: 交叉验证折数

        Returns:
            预测结果
        """
        print(f"\n训练 {model_name} 模型...")

        # 选择模型
        if model_name == "Ridge":
            model = Ridge(alpha=1.0, random_state=42)
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

        # 交叉验证预测
        n_folds = min(n_folds, X.shape[0])  # 确保折数不超过样本数
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(model, X, Y, cv=kfold)

        # 计算评价指标
        n_samples = X.shape[0]
        cosine_sims = np.zeros(n_samples)

        for i in range(n_samples):
            # Cosine similarity
            if np.linalg.norm(Y_pred[i]) > 0 and np.linalg.norm(Y[i]) > 0:
                cosine_sims[i] = 1 - cosine(Y_pred[i], Y[i])
            else:
                cosine_sims[i] = 0

        # 全局R²
        global_r2 = r2_score(Y.flatten(), Y_pred.flatten())

        # Per-target correlations
        per_target_corrs = {}
        n_targets_to_check = min(50, len(self.all_target_regions))
        for j in range(n_targets_to_check):
            target = self.all_target_regions[j]
            if Y[:, j].std() > 0 and Y_pred[:, j].std() > 0:
                corr, _ = pearsonr(Y[:, j], Y_pred[:, j])
                per_target_corrs[target] = corr

        result = PredictionResult(
            model_name=model_name,
            cosine_similarities=cosine_sims,
            per_target_correlations=per_target_corrs,
            mean_cosine_sim=np.mean(cosine_sims),
            std_cosine_sim=np.std(cosine_sims),
            mean_r2=global_r2,
            predicted=Y_pred,
            actual=Y
        )

        print(f"  Mean Cosine Similarity: {result.mean_cosine_sim:.4f} ± {result.std_cosine_sim:.4f}")
        print(f"  Global R²: {result.mean_r2:.4f}")

        return result

    # ==================== 实验运行 ====================

    def run_experiment(self, model_name: str = "Ridge") -> Dict[str, PredictionResult]:
        """
        运行完整实验：比较 Morph-only, Morph+Gene, Shuffle

        Args:
            model_name: 使用的模型

        Returns:
            三种条件的预测结果
        """
        print("\n" + "=" * 80)
        print(f"运行多模态预测实验 (模型: {model_name})")
        print("=" * 80)

        # 准备数据
        X_morph, X_gene, X_multi, Y, valid_neurons = self.prepare_features()
        X_shuffle = self.create_shuffled_features(X_morph, X_gene)

        # 三种条件的预测
        results = {}

        # 1. Morph-only
        print("\n" + "-" * 40)
        print("条件1: Morphology-only")
        print("-" * 40)
        results['morph_only'] = self.train_and_predict(X_morph, Y, model_name)

        # 2. Morph + Gene (多模态)
        print("\n" + "-" * 40)
        print("条件2: Morphology + Local Gene (多模态)")
        print("-" * 40)
        results['morph_gene'] = self.train_and_predict(X_multi, Y, model_name)

        # 3. Shuffle control
        print("\n" + "-" * 40)
        print("条件3: Morphology + Shuffled Gene (控制)")
        print("-" * 40)
        results['shuffle'] = self.train_and_predict(X_shuffle, Y, model_name)

        self.results = results
        return results

    # ==================== 统计分析 ====================

    def statistical_analysis(self) -> Dict:
        """
        进行统计显著性分析

        Returns:
            统计结果字典
        """
        print("\n" + "=" * 80)
        print("统计显著性分析")
        print("=" * 80)

        stats_results = {}

        # 比较 Morph+Gene vs Morph-only
        print("\n比较: Morph+Gene vs Morph-only")
        print("-" * 40)

        sim_multi = self.results['morph_gene'].cosine_similarities
        sim_morph = self.results['morph_only'].cosine_similarities

        # Wilcoxon signed-rank test
        stat, p_value = wilcoxon(sim_multi, sim_morph, alternative='greater')

        # Cohen's d (effect size)
        diff = sim_multi - sim_morph
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        improvement = (np.mean(sim_multi) - np.mean(sim_morph)) / np.mean(sim_morph) * 100 if np.mean(sim_morph) > 0 else 0

        stats_results['multi_vs_morph'] = {
            'wilcoxon_stat': stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'improvement_pct': improvement
        }

        print(f"  Wilcoxon statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Improvement: {improvement:.2f}%")

        # 比较 Morph+Gene vs Shuffle
        print("\n比较: Morph+Gene vs Shuffle Control")
        print("-" * 40)

        sim_shuffle = self.results['shuffle'].cosine_similarities

        stat2, p_value2 = wilcoxon(sim_multi, sim_shuffle, alternative='greater')
        diff2 = sim_multi - sim_shuffle
        cohens_d2 = np.mean(diff2) / np.std(diff2) if np.std(diff2) > 0 else 0

        stats_results['multi_vs_shuffle'] = {
            'wilcoxon_stat': stat2,
            'p_value': p_value2,
            'cohens_d': cohens_d2
        }

        print(f"  Wilcoxon statistic: {stat2:.4f}")
        print(f"  P-value: {p_value2:.2e}")
        print(f"  Cohen's d: {cohens_d2:.4f}")

        return stats_results

    # ==================== 可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """
        可视化实验结果

        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        # 1. Cosine Similarity 分布对比
        self._plot_similarity_distributions(output_dir)

        # 2. Per-target Correlation 热力图
        self._plot_target_correlations(output_dir)

        # 3. 预测 vs 实际的散点图
        self._plot_prediction_scatter(output_dir)

        # 4. 模型性能对比条形图
        self._plot_model_comparison(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_similarity_distributions(self, output_dir: str):
        """绘制相似度分布对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Violin plot
        data = {
            'Morph-only': self.results['morph_only'].cosine_similarities,
            'Morph+Gene': self.results['morph_gene'].cosine_similarities,
            'Shuffle': self.results['shuffle'].cosine_similarities
        }

        df = pd.DataFrame(data)
        df_melted = df.melt(var_name='Condition', value_name='Cosine Similarity')

        sns.violinplot(data=df_melted, x='Condition', y='Cosine Similarity',
                      ax=axes[0], palette=['#3498DB', '#E74C3C', '#95A5A6'])
        axes[0].set_title('Cosine Similarity Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cosine Similarity', fontsize=12)

        # 添加统计标注
        means = [np.mean(v) for v in data.values()]
        for i, (name, mean) in enumerate(zip(data.keys(), means)):
            axes[0].text(i, mean + 0.02, f'{mean:.3f}', ha='center', fontsize=10)

        # Box plot
        sns.boxplot(data=df_melted, x='Condition', y='Cosine Similarity',
                   ax=axes[1], palette=['#3498DB', '#E74C3C', '#95A5A6'])
        axes[1].set_title('Cosine Similarity Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Cosine Similarity', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_similarity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_similarity_distributions.png")

    def _plot_target_correlations(self, output_dir: str):
        """绘制per-target correlation对比"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 获取共同的targets
        targets = list(self.results['morph_only'].per_target_correlations.keys())[:30]

        if len(targets) == 0:
            print("  ⚠ 没有per-target correlation数据")
            plt.close()
            return

        corrs_morph = [self.results['morph_only'].per_target_correlations.get(t, 0) for t in targets]
        corrs_multi = [self.results['morph_gene'].per_target_correlations.get(t, 0) for t in targets]

        x = np.arange(len(targets))
        width = 0.35

        bars1 = ax.bar(x - width/2, corrs_morph, width, label='Morph-only', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, corrs_multi, width, label='Morph+Gene', color='#E74C3C', alpha=0.8)

        ax.set_ylabel('Pearson Correlation', fontsize=12)
        ax.set_xlabel('Target Region', fontsize=12)
        ax.set_title('Per-Target Prediction Correlation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_target_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 2_target_correlations.png")

    def _plot_prediction_scatter(self, output_dir: str):
        """绘制预测vs实际散点图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        conditions = ['morph_only', 'morph_gene', 'shuffle']
        titles = ['Morph-only', 'Morph+Gene', 'Shuffle Control']
        colors = ['#3498DB', '#E74C3C', '#95A5A6']

        for ax, cond, title, color in zip(axes, conditions, titles, colors):
            result = self.results[cond]

            # 展平预测和实际值
            y_true = result.actual.flatten()
            y_pred = result.predicted.flatten()

            # 采样绘制（避免点太多）
            n_points = min(5000, len(y_true))
            idx = np.random.choice(len(y_true), n_points, replace=False)

            ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=10, c=color)

            # 添加对角线
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

            # 计算相关系数
            corr, _ = pearsonr(y_true, y_pred)
            r2 = result.mean_r2

            ax.set_xlabel('Actual', fontsize=11)
            ax.set_ylabel('Predicted', fontsize=11)
            ax.set_title(f'{title}\nr={corr:.3f}, R²={r2:.3f}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_prediction_scatter.png")

    def _plot_model_comparison(self, output_dir: str):
        """绘制模型性能对比条形图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        conditions = ['Morph-only', 'Morph+Gene', 'Shuffle']
        colors = ['#3498DB', '#E74C3C', '#95A5A6']

        # Cosine Similarity
        means = [self.results['morph_only'].mean_cosine_sim,
                self.results['morph_gene'].mean_cosine_sim,
                self.results['shuffle'].mean_cosine_sim]
        stds = [self.results['morph_only'].std_cosine_sim,
               self.results['morph_gene'].std_cosine_sim,
               self.results['shuffle'].std_cosine_sim]

        bars = axes[0].bar(conditions, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel('Mean Cosine Similarity', fontsize=12)
        axes[0].set_title('Prediction Accuracy (Cosine Similarity)', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, max(means) * 1.2 if max(means) > 0 else 0.1)

        for bar, mean in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', fontsize=10)

        # R²
        r2s = [self.results['morph_only'].mean_r2,
              self.results['morph_gene'].mean_r2,
              self.results['shuffle'].mean_r2]

        bars = axes[1].bar(conditions, r2s, color=colors, alpha=0.8)
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('Prediction Accuracy (R²)', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, max(r2s) * 1.2 if max(r2s) > 0 else 0.1)

        for bar, r2 in zip(bars, r2s):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{r2:.3f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_model_comparison.png")

    # ==================== 结果保存 ====================

    def save_results(self, output_dir: str = "."):
        """保存实验结果到CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存汇总结果
        summary = {
            'Condition': ['Morph-only', 'Morph+Gene', 'Shuffle'],
            'Mean_Cosine_Sim': [
                self.results['morph_only'].mean_cosine_sim,
                self.results['morph_gene'].mean_cosine_sim,
                self.results['shuffle'].mean_cosine_sim
            ],
            'Std_Cosine_Sim': [
                self.results['morph_only'].std_cosine_sim,
                self.results['morph_gene'].std_cosine_sim,
                self.results['shuffle'].std_cosine_sim
            ],
            'R2_Score': [
                self.results['morph_only'].mean_r2,
                self.results['morph_gene'].mean_r2,
                self.results['shuffle'].mean_r2
            ]
        }

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/experiment_summary.csv", index=False)
        print(f"\n✓ 结果汇总已保存: {output_dir}/experiment_summary.csv")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task1_results",
                          model_name: str = "Ridge"):
        """
        运行完整的实验流程

        Args:
            output_dir: 输出目录
            model_name: 使用的模型
        """
        print("\n" + "=" * 80)
        print("任务1: Neuron-level 多模态投射预测")
        print("=" * 80)

        # 1. 加载数据
        n_neurons = self.load_all_data()

        if n_neurons == 0:
            print("\n✗ 没有有效数据，无法继续实验")
            return

        # 2. 运行实验
        self.run_experiment(model_name)

        # 3. 统计分析
        stats = self.statistical_analysis()

        # 4. 可视化
        self.visualize_results(output_dir)

        # 5. 保存结果
        self.save_results(output_dir)

        # 6. 打印结论
        self._print_conclusion(stats)

        print("\n" + "=" * 80)
        print("任务1完成!")
        print(f"结果保存在: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self, stats: Dict):
        """打印实验结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        multi_vs_morph = stats['multi_vs_morph']
        multi_vs_shuffle = stats['multi_vs_shuffle']

        print(f"""
【核心发现】

1. 多模态 vs 单模态:
   - Cosine Similarity 提升: {multi_vs_morph['improvement_pct']:.2f}%
   - 统计显著性: p = {multi_vs_morph['p_value']:.2e}
   - 效应量 (Cohen's d): {multi_vs_morph['cohens_d']:.3f}

2. 多模态 vs Shuffle控制:
   - 统计显著性: p = {multi_vs_shuffle['p_value']:.2e}
   - 效应量 (Cohen's d): {multi_vs_shuffle['cohens_d']:.3f}

【生物学意义】
""")

        if multi_vs_morph['p_value'] < 0.05 and multi_vs_morph['cohens_d'] > 0.2:
            print("""
✓ 局部分子环境对神经元输出投射有真实影响
  - 神经元投射模式不是由形态独立决定
  - 分子微环境提供了额外的功能约束

✓ 形态与分子模态信息互补
  - 多模态融合不是简单的特征堆叠
  - 而是恢复了完整的生物学信息

✓ 支持知识图谱设计:
  - "空间 + 分子 + 形态" 是组织原则的三角结构
  - 仅有形态数据无法准确预测投射
""")
        else:
            print("""
⚠ 多模态提升效果有限
  - 可能需要更精细的局部分子特征
  - 或者形态特征本身已包含主要信息
""")


# ==================== 主程序 ====================

def main():
    """主程序入口"""

    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"

    # 输出配置
    OUTPUT_DIR = "./task1_neuron_multimodal_results"
    MODEL_NAME = "Ridge"  # 可选: "Ridge", "RF", "MLP", "GBR"

    print("\n" + "=" * 80)
    print("任务1: Neuron-level 多模态投射预测")
    print("=" * 80)
    print(f"\nNeo4j URI: {NEO4J_URI}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"预测模型: {MODEL_NAME}")

    # 运行分析
    with NeuronMultimodalPredictor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as predictor:
        predictor.run_full_pipeline(
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME
        )


if __name__ == "__main__":
    main()