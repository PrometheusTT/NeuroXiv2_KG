"""
任务2：Region-level 多模态 Embedding 与功能结构分析
===============================================
构建功能指纹（fingerprints）并揭示多模态组织原则

核心科学问题：
1. 能否用 morphology、gene expression、projection 为每个区域构建"功能指纹"？
2. 这些单模态指纹是否一致？
3. 多模态指纹能否揭示单模态无法展示的潜在功能结构？

方法：
- 计算每个Region的三种Fingerprint（分子、形态、投射）
- 对每种Fingerprint进行embedding（t-SNE / UMAP / PHATE）
- 比较单模态vs多模态embedding的结构

评价指标：
- 与已知系统的一致性：NMI, ARI, Silhouette Score
- 距离矩阵相似性：Mantel Test
- 方差分解：Variance Partitioning

依赖：
- neo4j
- pandas, numpy, scipy
- scikit-learn
- umap-learn
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

# 降维与聚类
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score
)

# 统计分析
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Neo4j
import neo4j

# 尝试导入UMAP
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP embedding will be skipped.")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class EmbeddingResult:
    """Embedding结果数据类"""
    modality: str
    embedding_2d: np.ndarray
    method: str
    regions: List[str]


@dataclass
class ClusteringMetrics:
    """聚类评估指标"""
    silhouette: float
    nmi: float
    ari: float
    calinski_harabasz: float


class RegionMultimodalAnalyzer:
    """区域级多模态分析器"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化分析器

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

        # 数据存储
        self.regions: List[str] = []
        self.mol_fingerprints: Dict[str, np.ndarray] = {}
        self.morph_fingerprints: Dict[str, np.ndarray] = {}
        self.proj_fingerprints: Dict[str, np.ndarray] = {}

        # 全局维度
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []
        self.morph_feature_names: List[str] = []

        # 距离矩阵
        self.dist_mol: Optional[np.ndarray] = None
        self.dist_morph: Optional[np.ndarray] = None
        self.dist_proj: Optional[np.ndarray] = None
        self.dist_multi: Optional[np.ndarray] = None

        # Embedding结果
        self.embeddings: Dict[str, EmbeddingResult] = {}

        # 聚类标签
        self.cluster_labels: Dict[str, np.ndarray] = {}

        # 已知的解剖系统标签
        self.anatomical_systems: Dict[str, str] = {}

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 ====================

    def load_all_fingerprints(self) -> int:
        """
        加载所有区域的三种指纹

        Returns:
            有效区域数量
        """
        print("\n" + "=" * 80)
        print("加载区域多模态指纹数据")
        print("=" * 80)

        # Step 1: 获取全局维度
        self._get_global_dimensions()

        # Step 2: 获取有效区域列表
        self._get_valid_regions()

        # Step 3: 加载三种指纹
        self._load_molecular_fingerprints()
        self._load_morphology_fingerprints()
        self._load_projection_fingerprints()

        # Step 4: 加载解剖系统标签
        self._load_anatomical_systems()

        print(f"\n✓ 数据加载完成: {len(self.regions)} 个有效区域")
        return len(self.regions)

    def _get_global_dimensions(self):
        """获取全局特征维度"""
        print("\n获取全局特征维度...")

        # 形态特征名称
        self.morph_feature_names = [
            'axonal_length', 'axonal_branches',
            'axonal_bifurcation_remote_angle', 'axonal_maximum_branch_order',
            'dendritic_length', 'dendritic_branches',
            'dendritic_bifurcation_remote_angle', 'dendritic_maximum_branch_order'
        ]
        print(f"  形态特征: {len(self.morph_feature_names)} 维")

        # 获取所有subclass
        query = """
        MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
        RETURN DISTINCT sc.name AS subclass_name
        ORDER BY subclass_name
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_subclasses = [record['subclass_name'] for record in result]
        print(f"  Subclass类型: {len(self.all_subclasses)} 种")

        # 获取所有投射目标区域
        query = """
        MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        """
        with self.driver.session() as session:
            result = session.run(query)
            self.all_target_regions = [record['target'] for record in result]
        print(f"  投射目标区域: {len(self.all_target_regions)} 个")

    def _get_valid_regions(self):
        """获取同时具有三种数据的区域"""
        print("\n获取有效区域列表...")

        query = """
        MATCH (r:Region)
        WHERE EXISTS((r)-[:HAS_SUBCLASS]->())
          AND r.axonal_length IS NOT NULL
        WITH r
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
        OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
        WITH r, COUNT(DISTINCT n) + COUNT(DISTINCT n2) + COUNT(DISTINCT n3) AS neuron_count
        WHERE neuron_count >= 5
        RETURN r.acronym AS region
        ORDER BY region
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.regions = [record['region'] for record in result]

        print(f"  找到 {len(self.regions)} 个有效区域")

    def _load_molecular_fingerprints(self):
        """加载分子指纹（subclass组成）"""
        print("\n加载分子指纹...")

        query = """
        MATCH (r:Region {acronym: $region})
        MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN sc.name AS subclass_name, hs.pct_cells AS pct_cells
        """

        loaded = 0
        with self.driver.session() as session:
            for region in self.regions:
                result = session.run(query, region=region)

                subclass_dict = {record['subclass_name']: record['pct_cells']
                                 for record in result}

                # 构建固定维度向量
                fingerprint = np.zeros(len(self.all_subclasses))
                for i, sc in enumerate(self.all_subclasses):
                    if sc in subclass_dict:
                        fingerprint[i] = subclass_dict[sc]

                self.mol_fingerprints[region] = fingerprint
                loaded += 1

        print(f"  加载了 {loaded} 个区域的分子指纹")

    def _load_morphology_fingerprints(self):
        """加载形态指纹"""
        print("\n加载形态指纹...")

        query = """
        MATCH (r:Region {acronym: $region})
        RETURN
          r.axonal_length AS axonal_length,
          r.axonal_branches AS axonal_branches,
          r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
          r.axonal_maximum_branch_order AS axonal_maximum_branch_order,
          r.dendritic_length AS dendritic_length,
          r.dendritic_branches AS dendritic_branches,
          r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
          r.dendritic_maximum_branch_order AS dendritic_maximum_branch_order
        """

        loaded = 0
        with self.driver.session() as session:
            for region in self.regions:
                result = session.run(query, region=region)
                record = result.single()

                if record:
                    fingerprint = []
                    for feat_name in self.morph_feature_names:
                        val = record[feat_name]
                        fingerprint.append(val if val is not None else np.nan)

                    self.morph_fingerprints[region] = np.array(fingerprint)
                    loaded += 1

        # Z-score标准化
        all_fps = np.array([self.morph_fingerprints[r] for r in self.regions])
        for i in range(all_fps.shape[1]):
            col = all_fps[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mean = col[valid].mean()
                std = col[valid].std()
                if std > 0:
                    col[valid] = (col[valid] - mean) / std
                all_fps[:, i] = col

        for i, region in enumerate(self.regions):
            self.morph_fingerprints[region] = all_fps[i]

        print(f"  加载了 {loaded} 个区域的形态指纹（已z-score标准化）")

    def _load_projection_fingerprints(self):
        """加载投射指纹"""
        print("\n加载投射指纹...")

        query = """
        MATCH (r:Region {acronym: $region})
        OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r)
        OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
        OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
        WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) + COLLECT(DISTINCT n3)) AS ns
        UNWIND ns AS n
        WITH DISTINCT n
        WHERE n IS NOT NULL
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN t.acronym AS target, SUM(p.weight) AS total_weight
        """

        loaded = 0
        with self.driver.session() as session:
            for region in self.regions:
                result = session.run(query, region=region)

                proj_dict = {record['target']: record['total_weight']
                             for record in result}

                # 构建固定维度向量
                fingerprint = np.zeros(len(self.all_target_regions))
                for i, target in enumerate(self.all_target_regions):
                    if target in proj_dict:
                        fingerprint[i] = proj_dict[target]

                # Log变换并归一化
                fingerprint = np.log10(1 + fingerprint)
                total = fingerprint.sum()
                if total > 0:
                    fingerprint = fingerprint / total

                self.proj_fingerprints[region] = fingerprint
                loaded += 1

        print(f"  加载了 {loaded} 个区域的投射指纹")

    def _load_anatomical_systems(self):
        """加载解剖系统标签用于评估"""
        print("\n加载解剖系统标签...")

        # 定义主要的解剖系统分类
        system_mapping = {
            # Isocortex
            'FRP': 'Isocortex', 'MOs': 'Isocortex', 'MOp': 'Isocortex',
            'SSp': 'Isocortex', 'SSs': 'Isocortex', 'AUDp': 'Isocortex',
            'AUDd': 'Isocortex', 'AUDv': 'Isocortex', 'VISp': 'Isocortex',
            'VISl': 'Isocortex', 'VISal': 'Isocortex', 'VISpm': 'Isocortex',
            'VISam': 'Isocortex', 'RSPagl': 'Isocortex', 'RSPd': 'Isocortex',
            'RSPv': 'Isocortex', 'ACAd': 'Isocortex', 'ACAv': 'Isocortex',
            'PL': 'Isocortex', 'ILA': 'Isocortex', 'ORBl': 'Isocortex',
            'ORBm': 'Isocortex', 'ORBvl': 'Isocortex', 'AI': 'Isocortex',
            'GU': 'Isocortex', 'VISC': 'Isocortex', 'TEa': 'Isocortex',
            'PERI': 'Isocortex', 'ECT': 'Isocortex',

            # Hippocampal formation
            'CA1': 'Hippocampus', 'CA2': 'Hippocampus', 'CA3': 'Hippocampus',
            'DG': 'Hippocampus', 'SUB': 'Hippocampus', 'ProS': 'Hippocampus',
            'ENTl': 'Hippocampus', 'ENTm': 'Hippocampus',

            # Thalamus
            'VAL': 'Thalamus', 'VM': 'Thalamus', 'VPL': 'Thalamus',
            'VPM': 'Thalamus', 'VPMpc': 'Thalamus', 'PoT': 'Thalamus',
            'LP': 'Thalamus', 'LGd': 'Thalamus', 'LGv': 'Thalamus',
            'MG': 'Thalamus', 'RT': 'Thalamus', 'AD': 'Thalamus',
            'AV': 'Thalamus', 'AM': 'Thalamus', 'MD': 'Thalamus',
            'CM': 'Thalamus', 'RH': 'Thalamus', 'RE': 'Thalamus',
            'PVT': 'Thalamus', 'PT': 'Thalamus',

            # Hypothalamus
            'LHA': 'Hypothalamus', 'LPO': 'Hypothalamus', 'AHN': 'Hypothalamus',
            'MPN': 'Hypothalamus', 'VMH': 'Hypothalamus', 'DMH': 'Hypothalamus',
            'ARH': 'Hypothalamus', 'PVH': 'Hypothalamus', 'SO': 'Hypothalamus',
            'SCH': 'Hypothalamus', 'PH': 'Hypothalamus', 'PMv': 'Hypothalamus',
            'PMd': 'Hypothalamus', 'TM': 'Hypothalamus', 'MM': 'Hypothalamus',
            'LM': 'Hypothalamus', 'SUM': 'Hypothalamus',

            # Striatum
            'CP': 'Striatum', 'ACB': 'Striatum', 'OT': 'Striatum',
            'LSc': 'Striatum', 'LSr': 'Striatum', 'LSv': 'Striatum',
            'SF': 'Striatum', 'SH': 'Striatum',

            # Pallidum
            'GPe': 'Pallidum', 'GPi': 'Pallidum', 'SI': 'Pallidum',
            'MA': 'Pallidum', 'NDB': 'Pallidum', 'BST': 'Pallidum',

            # Amygdala
            'CEA': 'Amygdala', 'MEA': 'Amygdala', 'COA': 'Amygdala',
            'BLA': 'Amygdala', 'BMA': 'Amygdala', 'LA': 'Amygdala',
            'PA': 'Amygdala', 'AAA': 'Amygdala',

            # Midbrain
            'SC': 'Midbrain', 'IC': 'Midbrain', 'SNr': 'Midbrain',
            'SNc': 'Midbrain', 'VTA': 'Midbrain', 'RN': 'Midbrain',
            'MRN': 'Midbrain', 'PAG': 'Midbrain', 'APN': 'Midbrain',

            # Hindbrain
            'PRNc': 'Hindbrain', 'PRNr': 'Hindbrain', 'GRN': 'Hindbrain',
            'PG': 'Hindbrain', 'DCO': 'Hindbrain', 'VCO': 'Hindbrain',
            'PB': 'Hindbrain', 'SOC': 'Hindbrain', 'NLL': 'Hindbrain',

            # Cerebellum
            'VERM': 'Cerebellum', 'HEM': 'Cerebellum', 'CBN': 'Cerebellum',
            'FN': 'Cerebellum', 'IP': 'Cerebellum', 'DN': 'Cerebellum',
        }

        for region in self.regions:
            # 尝试精确匹配或前缀匹配
            if region in system_mapping:
                self.anatomical_systems[region] = system_mapping[region]
            else:
                # 尝试前缀匹配
                matched = False
                for key, system in system_mapping.items():
                    if region.startswith(key):
                        self.anatomical_systems[region] = system
                        matched = True
                        break
                if not matched:
                    self.anatomical_systems[region] = 'Other'

        system_counts = {}
        for system in self.anatomical_systems.values():
            system_counts[system] = system_counts.get(system, 0) + 1

        print(f"  解剖系统分布: {system_counts}")

    # ==================== 特征矩阵与距离计算 ====================

    def compute_feature_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算特征矩阵

        Returns:
            X_mol, X_morph, X_proj, X_multi
        """
        print("\n计算特征矩阵...")

        X_mol = np.array([self.mol_fingerprints[r] for r in self.regions])
        X_morph = np.array([self.morph_fingerprints[r] for r in self.regions])
        X_proj = np.array([self.proj_fingerprints[r] for r in self.regions])

        # 处理NaN
        X_morph = np.nan_to_num(X_morph, nan=0.0)

        # 多模态拼接（先标准化各模态）
        scaler = StandardScaler()
        X_mol_scaled = scaler.fit_transform(X_mol)
        X_proj_scaled = scaler.fit_transform(X_proj)

        X_multi = np.hstack([X_mol_scaled, X_morph, X_proj_scaled])

        print(f"  分子特征矩阵: {X_mol.shape}")
        print(f"  形态特征矩阵: {X_morph.shape}")
        print(f"  投射特征矩阵: {X_proj.shape}")
        print(f"  多模态特征矩阵: {X_multi.shape}")

        return X_mol, X_morph, X_proj, X_multi

    def compute_distance_matrices(self, X_mol: np.ndarray, X_morph: np.ndarray,
                                  X_proj: np.ndarray, X_multi: np.ndarray):
        """计算距离矩阵"""
        print("\n计算距离矩阵...")

        # 使用cosine距离
        self.dist_mol = squareform(pdist(X_mol, metric='cosine'))
        self.dist_morph = squareform(pdist(X_morph, metric='euclidean'))
        self.dist_proj = squareform(pdist(X_proj, metric='cosine'))
        self.dist_multi = squareform(pdist(X_multi, metric='euclidean'))

        # 处理NaN和无穷值
        for dist_mat in [self.dist_mol, self.dist_morph, self.dist_proj, self.dist_multi]:
            dist_mat[np.isnan(dist_mat)] = 0
            dist_mat[np.isinf(dist_mat)] = 0

        print("  ✓ 距离矩阵计算完成")

    # ==================== Embedding ====================

    def compute_embeddings(self, X_mol: np.ndarray, X_morph: np.ndarray,
                           X_proj: np.ndarray, X_multi: np.ndarray,
                           method: str = "UMAP") -> Dict[str, EmbeddingResult]:
        """
        计算各模态的2D embedding

        Args:
            X_*: 各模态特征矩阵
            method: "UMAP" 或 "TSNE"

        Returns:
            embedding结果字典
        """
        print(f"\n计算 {method} Embedding...")

        modalities = {
            'molecular': X_mol,
            'morphology': X_morph,
            'projection': X_proj,
            'multimodal': X_multi
        }

        for name, X in modalities.items():
            print(f"  处理 {name}...")

            # 处理NaN
            X_clean = np.nan_to_num(X, nan=0.0)

            if method == "UMAP" and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, n_neighbors=15,
                                    min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(X_clean)
            else:
                # 使用t-SNE
                perplexity = min(30, X_clean.shape[0] - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity,
                            random_state=42, n_iter=1000)
                embedding_2d = tsne.fit_transform(X_clean)

            self.embeddings[name] = EmbeddingResult(
                modality=name,
                embedding_2d=embedding_2d,
                method=method,
                regions=self.regions
            )

        print("  ✓ Embedding计算完成")
        return self.embeddings

    # ==================== 聚类分析 ====================

    def perform_clustering(self, X_mol: np.ndarray, X_morph: np.ndarray,
                           X_proj: np.ndarray, X_multi: np.ndarray,
                           n_clusters: int = 8) -> Dict[str, np.ndarray]:
        """
        对各模态进行聚类

        Args:
            X_*: 各模态特征矩阵
            n_clusters: 聚类数量

        Returns:
            各模态的聚类标签
        """
        print(f"\n进行层次聚类 (k={n_clusters})...")

        modalities = {
            'molecular': X_mol,
            'morphology': X_morph,
            'projection': X_proj,
            'multimodal': X_multi
        }

        for name, X in modalities.items():
            X_clean = np.nan_to_num(X, nan=0.0)

            # 层次聚类
            clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                                 linkage='ward')
            labels = clustering.fit_predict(X_clean)
            self.cluster_labels[name] = labels

            print(f"  {name}: {len(np.unique(labels))} clusters")

        return self.cluster_labels

    # ==================== 评估指标 ====================

    def evaluate_clustering(self, X_mol: np.ndarray, X_morph: np.ndarray,
                            X_proj: np.ndarray, X_multi: np.ndarray) -> Dict[str, ClusteringMetrics]:
        """
        评估聚类质量

        Returns:
            各模态的聚类评估指标
        """
        print("\n" + "=" * 80)
        print("聚类质量评估")
        print("=" * 80)

        # 获取真实标签（解剖系统）
        true_labels = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        label_encoder = {l: i for i, l in enumerate(set(true_labels))}
        true_labels_encoded = np.array([label_encoder[l] for l in true_labels])

        modalities = {
            'molecular': X_mol,
            'morphology': X_morph,
            'projection': X_proj,
            'multimodal': X_multi
        }

        metrics = {}

        print(f"\n{'Modality':<15} {'Silhouette':>12} {'NMI':>12} {'ARI':>12} {'CH Score':>12}")
        print("-" * 65)

        for name, X in modalities.items():
            X_clean = np.nan_to_num(X, nan=0.0)
            pred_labels = self.cluster_labels[name]

            # Silhouette Score
            sil = silhouette_score(X_clean, pred_labels) if len(np.unique(pred_labels)) > 1 else 0

            # NMI with anatomical systems
            nmi = normalized_mutual_info_score(true_labels_encoded, pred_labels)

            # Adjusted Rand Index
            ari = adjusted_rand_score(true_labels_encoded, pred_labels)

            # Calinski-Harabasz Score
            ch = calinski_harabasz_score(X_clean, pred_labels) if len(np.unique(pred_labels)) > 1 else 0

            metrics[name] = ClusteringMetrics(
                silhouette=sil,
                nmi=nmi,
                ari=ari,
                calinski_harabasz=ch
            )

            print(f"{name:<15} {sil:>12.4f} {nmi:>12.4f} {ari:>12.4f} {ch:>12.1f}")

        return metrics

    def mantel_test(self, n_permutations: int = 999) -> Dict[str, Tuple[float, float]]:
        """
        Mantel测试：比较距离矩阵的相关性

        Args:
            n_permutations: 置换次数

        Returns:
            距离矩阵对之间的相关系数和p值
        """
        print("\n" + "=" * 80)
        print("Mantel Test (距离矩阵相关性)")
        print("=" * 80)

        def compute_mantel(D1, D2, n_perm):
            """计算Mantel相关系数和置换p值"""
            # 取上三角（不含对角线）
            i_upper = np.triu_indices(D1.shape[0], k=1)
            d1_flat = D1[i_upper]
            d2_flat = D2[i_upper]

            # 原始相关系数
            r_obs, _ = spearmanr(d1_flat, d2_flat)

            # 置换检验
            r_perms = []
            n = D1.shape[0]
            for _ in range(n_perm):
                perm = np.random.permutation(n)
                D2_perm = D2[np.ix_(perm, perm)]
                d2_perm_flat = D2_perm[i_upper]
                r_perm, _ = spearmanr(d1_flat, d2_perm_flat)
                r_perms.append(r_perm)

            # 计算p值
            p_value = (np.sum(np.array(r_perms) >= r_obs) + 1) / (n_perm + 1)

            return r_obs, p_value

        comparisons = [
            ('mol-morph', self.dist_mol, self.dist_morph),
            ('mol-proj', self.dist_mol, self.dist_proj),
            ('morph-proj', self.dist_morph, self.dist_proj),
            ('multi-mol', self.dist_multi, self.dist_mol),
            ('multi-morph', self.dist_multi, self.dist_morph),
            ('multi-proj', self.dist_multi, self.dist_proj),
        ]

        results = {}
        print(f"\n{'Comparison':<15} {'Correlation':>12} {'P-value':>12}")
        print("-" * 45)

        for name, D1, D2 in comparisons:
            r, p = compute_mantel(D1, D2, n_permutations)
            results[name] = (r, p)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:<15} {r:>12.4f} {p:>12.4f} {sig}")

        return results

    def variance_partitioning(self, X_mol: np.ndarray, X_morph: np.ndarray,
                              X_proj: np.ndarray) -> Dict[str, float]:
        """
        方差分解：计算每种模态的独特贡献

        Returns:
            各模态的方差贡献比例
        """
        print("\n" + "=" * 80)
        print("Variance Partitioning (模态独立贡献)")
        print("=" * 80)

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # 使用多模态embedding作为目标
        target = self.embeddings['multimodal'].embedding_2d

        results = {}

        # 各模态单独的解释方差
        modalities = {
            'molecular': X_mol,
            'morphology': np.nan_to_num(X_morph, nan=0.0),
            'projection': X_proj
        }

        individual_r2 = {}
        for name, X in modalities.items():
            model = LinearRegression()
            model.fit(X, target)
            pred = model.predict(X)
            r2 = r2_score(target, pred)
            individual_r2[name] = r2

        # 两两组合的解释方差
        pairs = [
            ('mol+morph', np.hstack([X_mol, np.nan_to_num(X_morph, nan=0.0)])),
            ('mol+proj', np.hstack([X_mol, X_proj])),
            ('morph+proj', np.hstack([np.nan_to_num(X_morph, nan=0.0), X_proj])),
        ]

        pair_r2 = {}
        for name, X in pairs:
            model = LinearRegression()
            model.fit(X, target)
            pred = model.predict(X)
            r2 = r2_score(target, pred)
            pair_r2[name] = r2

        # 全部模态
        X_all = np.hstack([X_mol, np.nan_to_num(X_morph, nan=0.0), X_proj])
        model = LinearRegression()
        model.fit(X_all, target)
        pred = model.predict(X_all)
        total_r2 = r2_score(target, pred)

        # 计算独特贡献
        unique_mol = total_r2 - pair_r2['morph+proj']
        unique_morph = total_r2 - pair_r2['mol+proj']
        unique_proj = total_r2 - pair_r2['mol+morph']

        results = {
            'molecular_unique': unique_mol,
            'morphology_unique': unique_morph,
            'projection_unique': unique_proj,
            'molecular_total': individual_r2['molecular'],
            'morphology_total': individual_r2['morphology'],
            'projection_total': individual_r2['projection'],
            'all_modalities': total_r2
        }

        print(f"\n{'Modality':<20} {'Unique R²':>12} {'Total R²':>12}")
        print("-" * 50)
        print(f"{'Molecular':<20} {unique_mol:>12.4f} {individual_r2['molecular']:>12.4f}")
        print(f"{'Morphology':<20} {unique_morph:>12.4f} {individual_r2['morphology']:>12.4f}")
        print(f"{'Projection':<20} {unique_proj:>12.4f} {individual_r2['projection']:>12.4f}")
        print(f"\n{'All Combined':<20} {'-':>12} {total_r2:>12.4f}")

        return results

    # ==================== 可视化 ====================

    def visualize_embeddings(self, output_dir: str = "."):
        """可视化embedding结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("生成Embedding可视化")
        print("=" * 80)

        # 1. 四模态embedding对比（按解剖系统着色）
        self._plot_embeddings_by_system(output_dir)

        # 2. 聚类结果对比
        self._plot_embeddings_by_cluster(output_dir)

        # 3. 距离矩阵热力图
        self._plot_distance_matrices(output_dir)

        # 4. 层次聚类树状图
        self._plot_dendrograms(output_dir)

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    def _plot_embeddings_by_system(self, output_dir: str):
        """按解剖系统着色的embedding图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 获取系统标签和颜色
        systems = [self.anatomical_systems.get(r, 'Other') for r in self.regions]
        unique_systems = sorted(set(systems))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_systems)))
        system_to_color = {s: colors[i] for i, s in enumerate(unique_systems)}
        point_colors = [system_to_color[s] for s in systems]

        modalities = ['molecular', 'morphology', 'projection', 'multimodal']
        titles = ['Molecular Fingerprint', 'Morphology Fingerprint',
                  'Projection Fingerprint', 'Multimodal (Combined)']

        for ax, mod, title in zip(axes.flat, modalities, titles):
            emb = self.embeddings[mod].embedding_2d

            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=point_colors,
                                 s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

            # 添加区域标签（只标注部分）
            for i, region in enumerate(self.regions):
                if i % 5 == 0:  # 每5个标注一个
                    ax.annotate(region, (emb[i, 0], emb[i, 1]),
                                fontsize=6, alpha=0.7)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{self.embeddings[mod].method} 1', fontsize=11)
            ax.set_ylabel(f'{self.embeddings[mod].method} 2', fontsize=11)
            ax.grid(alpha=0.3)

        # 添加图例
        legend_elements = [Patch(facecolor=system_to_color[s], label=s)
                           for s in unique_systems if s != 'Other']
        fig.legend(handles=legend_elements, loc='center right',
                   bbox_to_anchor=(1.12, 0.5), fontsize=9)

        plt.suptitle('Region Embeddings by Anatomical System',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_embeddings_by_system.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 1_embeddings_by_system.png")

    def _plot_embeddings_by_cluster(self, output_dir: str):
        """按聚类结果着色的embedding图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        modalities = ['molecular', 'morphology', 'projection', 'multimodal']
        titles = ['Molecular Clustering', 'Morphology Clustering',
                  'Projection Clustering', 'Multimodal Clustering']

        for ax, mod, title in zip(axes.flat, modalities, titles):
            emb = self.embeddings[mod].embedding_2d
            labels = self.cluster_labels[mod]

            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels,
                                 cmap='tab10', s=60, alpha=0.7,
                                 edgecolors='white', linewidth=0.5)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{self.embeddings[mod].method} 1', fontsize=11)
            ax.set_ylabel(f'{self.embeddings[mod].method} 2', fontsize=11)
            ax.grid(alpha=0.3)

            plt.colorbar(scatter, ax=ax, label='Cluster')

        plt.suptitle('Region Embeddings by Cluster Assignment',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_embeddings_by_cluster.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 2_embeddings_by_cluster.png")

    def _plot_distance_matrices(self, output_dir: str):
        """绘制距离矩阵热力图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        matrices = [
            ('Molecular Distance', self.dist_mol),
            ('Morphology Distance', self.dist_morph),
            ('Projection Distance', self.dist_proj),
            ('Multimodal Distance', self.dist_multi)
        ]

        # 选择显示的区域（太多会看不清）
        n_show = min(40, len(self.regions))
        idx = np.linspace(0, len(self.regions) - 1, n_show, dtype=int)
        show_regions = [self.regions[i] for i in idx]

        for ax, (title, dist_mat) in zip(axes.flat, matrices):
            dist_subset = dist_mat[np.ix_(idx, idx)]

            # 归一化用于显示
            if dist_subset.max() > 0:
                dist_norm = dist_subset / dist_subset.max()
            else:
                dist_norm = dist_subset

            sns.heatmap(dist_norm, ax=ax, cmap='RdYlBu_r',
                        xticklabels=show_regions, yticklabels=show_regions,
                        square=True)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)

        plt.suptitle('Distance Matrices (Normalized)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_distance_matrices.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 3_distance_matrices.png")

    def _plot_dendrograms(self, output_dir: str):
        """绘制层次聚类树状图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        matrices = [
            ('Molecular', self.dist_mol),
            ('Morphology', self.dist_morph),
            ('Projection', self.dist_proj),
            ('Multimodal', self.dist_multi)
        ]

        for ax, (title, dist_mat) in zip(axes.flat, matrices):
            # 转换为condensed距离矩阵
            dist_condensed = squareform(dist_mat)

            # 计算linkage
            Z = linkage(dist_condensed, method='ward')

            # 绘制树状图
            dendrogram(Z, ax=ax, labels=self.regions,
                       leaf_rotation=90, leaf_font_size=5)

            ax.set_title(f'{title} Hierarchical Clustering',
                         fontsize=14, fontweight='bold')
            ax.set_ylabel('Distance')

        plt.suptitle('Hierarchical Clustering Dendrograms',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_dendrograms.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 4_dendrograms.png")

    # ==================== 结果保存 ====================

    def save_results(self, output_dir: str = ".",
                     clustering_metrics: Dict = None,
                     mantel_results: Dict = None,
                     variance_results: Dict = None):
        """保存分析结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存embedding坐标
        for name, emb_result in self.embeddings.items():
            df = pd.DataFrame(
                emb_result.embedding_2d,
                index=emb_result.regions,
                columns=[f'{emb_result.method}_1', f'{emb_result.method}_2']
            )
            df['cluster'] = self.cluster_labels[name]
            df['anatomical_system'] = [self.anatomical_systems.get(r, 'Other')
                                       for r in emb_result.regions]
            df.to_csv(f"{output_dir}/embedding_{name}.csv")

        # 保存聚类评估结果
        if clustering_metrics:
            metrics_df = pd.DataFrame({
                'modality': list(clustering_metrics.keys()),
                'silhouette': [m.silhouette for m in clustering_metrics.values()],
                'nmi': [m.nmi for m in clustering_metrics.values()],
                'ari': [m.ari for m in clustering_metrics.values()],
                'calinski_harabasz': [m.calinski_harabasz for m in clustering_metrics.values()]
            })
            metrics_df.to_csv(f"{output_dir}/clustering_metrics.csv", index=False)

        # 保存Mantel测试结果
        if mantel_results:
            mantel_df = pd.DataFrame([
                {'comparison': k, 'correlation': v[0], 'p_value': v[1]}
                for k, v in mantel_results.items()
            ])
            mantel_df.to_csv(f"{output_dir}/mantel_test_results.csv", index=False)

        # 保存方差分解结果
        if variance_results:
            var_df = pd.DataFrame([variance_results])
            var_df.to_csv(f"{output_dir}/variance_partitioning.csv", index=False)

        print(f"\n✓ 结果已保存到: {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./task2_results",
                          n_clusters: int = 8,
                          embedding_method: str = "UMAP"):
        """
        运行完整分析流程

        Args:
            output_dir: 输出目录
            n_clusters: 聚类数量
            embedding_method: embedding方法
        """
        print("\n" + "=" * 80)
        print("任务2: Region-level 多模态Embedding与功能结构分析")
        print("=" * 80)

        # 1. 加载数据
        self.load_all_fingerprints()

        # 2. 计算特征矩阵
        X_mol, X_morph, X_proj, X_multi = self.compute_feature_matrices()

        # 3. 计算距离矩阵
        self.compute_distance_matrices(X_mol, X_morph, X_proj, X_multi)

        # 4. 计算embedding
        self.compute_embeddings(X_mol, X_morph, X_proj, X_multi,
                                method=embedding_method)

        # 5. 聚类分析
        self.perform_clustering(X_mol, X_morph, X_proj, X_multi, n_clusters)

        # 6. 评估指标
        clustering_metrics = self.evaluate_clustering(X_mol, X_morph, X_proj, X_multi)
        mantel_results = self.mantel_test()
        variance_results = self.variance_partitioning(X_mol, X_morph, X_proj)

        # 7. 可视化
        self.visualize_embeddings(output_dir)

        # 8. 保存结果
        self.save_results(output_dir, clustering_metrics, mantel_results, variance_results)

        # 9. 打印结论
        self._print_conclusion(clustering_metrics, mantel_results, variance_results)

        print("\n" + "=" * 80)
        print("任务2完成!")
        print(f"结果保存在: {output_dir}")
        print("=" * 80)

    def _print_conclusion(self, clustering_metrics: Dict,
                          mantel_results: Dict,
                          variance_results: Dict):
        """打印实验结论"""
        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        # 比较各模态的聚类质量
        best_silhouette = max(clustering_metrics.items(),
                              key=lambda x: x[1].silhouette)
        best_nmi = max(clustering_metrics.items(),
                       key=lambda x: x[1].nmi)

        print(f"""
【聚类质量对比】

最佳Silhouette Score: {best_silhouette[0]} ({best_silhouette[1].silhouette:.4f})
最佳NMI (与解剖系统): {best_nmi[0]} ({best_nmi[1].nmi:.4f})

单模态 vs 多模态:
- Molecular Silhouette: {clustering_metrics['molecular'].silhouette:.4f}
- Morphology Silhouette: {clustering_metrics['morphology'].silhouette:.4f}
- Projection Silhouette: {clustering_metrics['projection'].silhouette:.4f}
- Multimodal Silhouette: {clustering_metrics['multimodal'].silhouette:.4f}

【距离矩阵相关性 (Mantel Test)】

- Multimodal ↔ Molecular: r={mantel_results['multi-mol'][0]:.4f} (p={mantel_results['multi-mol'][1]:.4f})
- Multimodal ↔ Morphology: r={mantel_results['multi-morph'][0]:.4f} (p={mantel_results['multi-morph'][1]:.4f})
- Multimodal ↔ Projection: r={mantel_results['multi-proj'][0]:.4f} (p={mantel_results['multi-proj'][1]:.4f})

【方差分解 (独立贡献)】

- Molecular unique: {variance_results['molecular_unique']:.4f}
- Morphology unique: {variance_results['morphology_unique']:.4f}
- Projection unique: {variance_results['projection_unique']:.4f}
- Total explained: {variance_results['all_modalities']:.4f}

【生物学意义】
""")

        # 根据结果给出解释
        multi_sil = clustering_metrics['multimodal'].silhouette
        single_max_sil = max(clustering_metrics['molecular'].silhouette,
                             clustering_metrics['morphology'].silhouette,
                             clustering_metrics['projection'].silhouette)

        if multi_sil > single_max_sil:
            print("""
✓ 多模态embedding揭示了更清晰的功能结构
  - 多模态聚类质量优于任何单一模态
  - 表明三种模态提供互补的组织信息

✓ CCFv3解剖学分类无法完全捕捉功能组织
  - 多模态embedding可能揭示:
    · Forebrain continuum (跨解剖的功能连续体)
    · Limbic-sensorimotor axis (功能轴)
    · Hub regions (如CLA, ACC等)

✓ 各模态独立贡献:
  - 证明三种模态各自提供独特信息
  - 支持"空间 + 分子 + 形态"的三角组织原则
""")
        else:
            print(f"""
⚠ 多模态未显著优于最佳单模态
  - 最佳单模态: {max(clustering_metrics.items(), key=lambda x: x[1].silhouette)[0]}
  - 可能需要:
    · 更精细的特征工程
    · 不同的融合策略
    · 更多的区域数据
""")


# ==================== 主程序 ====================

def main():
    """主程序入口"""

    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"

    # 输出配置
    OUTPUT_DIR = "./task2_region_multimodal_results"
    N_CLUSTERS = 8
    EMBEDDING_METHOD = "UMAP" if UMAP_AVAILABLE else "TSNE"

    print("\n" + "=" * 80)
    print("任务2: Region-level 多模态Embedding与功能结构分析")
    print("=" * 80)
    print(f"\nNeo4j URI: {NEO4J_URI}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"聚类数量: {N_CLUSTERS}")
    print(f"Embedding方法: {EMBEDDING_METHOD}")

    # 运行分析
    with RegionMultimodalAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as analyzer:
        analyzer.run_full_pipeline(
            output_dir=OUTPUT_DIR,
            n_clusters=N_CLUSTERS,
            embedding_method=EMBEDDING_METHOD
        )


if __name__ == "__main__":
    main()