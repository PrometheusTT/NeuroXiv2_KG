"""
聚类数量预实验：搜索最优K值（改进版）
===============================================
添加了：
1.Kneedle算法自动检测拐点
2.二阶导数/曲率可视化
3.Gap Statistic
4.综合评分与投票机制

作者:  PrometheusTT
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# 机器学习
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Neo4j
import neo4j

# 设置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

N_CORES = multiprocessing.cpu_count()
print(f"检测到 {N_CORES} 个CPU核心")


# ==================== Kneedle算法实现 ====================

def find_elbow_point(k_values: np.ndarray, values: np.ndarray,
                     direction: str = 'decreasing',
                     curve:  str = 'convex') -> Tuple[int, np.ndarray]:
    """
    使用Kneedle算法找到拐点

    参数:
        k_values: K值数组
        values: 对应的指标值
        direction: 'increasing' 或 'decreasing'
        curve: 'convex' 或 'concave'

    返回:
        elbow_k: 拐点对应的K值
        normalized_diff: 归一化后的差值曲线
    """
    # 归一化
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    v_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)

    # 根据方向调整
    if direction == 'decreasing':
        v_norm = 1 - v_norm

    # 计算与对角线的差值
    if curve == 'convex':
        diff = v_norm - k_norm
    else:
        diff = k_norm - v_norm

    # 找到最大差值点
    elbow_idx = np.argmax(diff)

    return k_values[elbow_idx], diff


def compute_curvature(k_values: np.ndarray, values: np.ndarray,
                      smooth_sigma: float = 1.5) -> np.ndarray:
    """
    计算曲线的曲率（二阶导数的绝对值）

    参数:
        k_values:  K值
        values: 指标值
        smooth_sigma: 高斯平滑的sigma

    返回:
        curvature: 曲率数组
    """
    # 平滑处理减少噪声
    values_smooth = gaussian_filter1d(values, sigma=smooth_sigma)

    # 一阶导数
    first_deriv = np.gradient(values_smooth, k_values)

    # 二阶导数
    second_deriv = np.gradient(first_deriv, k_values)

    # 曲率公式:  |y''| / (1 + y'^2)^(3/2)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5

    return curvature


def find_elbow_by_curvature(k_values: np.ndarray, values: np.ndarray,
                            min_k: int = 5) -> int:
    """通过曲率最大值找拐点"""
    curvature = compute_curvature(k_values, values)

    # 排除前几个K值（太小没意义）
    valid_mask = k_values >= min_k
    valid_k = k_values[valid_mask]
    valid_curv = curvature[valid_mask]

    # 找曲率最大点
    max_idx = np.argmax(valid_curv)

    return valid_k[max_idx]


# ==================== Gap Statistic ====================

def compute_gap_statistic(X: np.ndarray, k_values:  List[int],
                          n_refs: int = 10, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算Gap Statistic

    Gap(k) = E[log(W_k^*)] - log(W_k)
    其中W_k是类内离差平方和，W_k^*是随机数据的期望值

    参数:
        X: 数据矩阵
        k_values: K值列表
        n_refs: 随机参考数据集数量
        random_state: 随机种子

    返回:
        gaps: Gap值数组
        gap_stds: Gap标准差数组
    """
    np.random.seed(random_state)

    n_samples, n_features = X.shape

    # 数据范围（用于生成均匀随机数据）
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    gaps = []
    gap_stds = []

    for k in k_values:
        # 原始数据的W_k
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(X)
        log_wk = np.log(kmeans.inertia_ + 1e-10)

        # 随机参考数据的W_k^*
        log_wk_refs = []
        for _ in range(n_refs):
            # 生成均匀分布的随机数据
            X_ref = np.random.uniform(mins, maxs, (n_samples, n_features))
            kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            kmeans_ref.fit(X_ref)
            log_wk_refs.append(np.log(kmeans_ref.inertia_ + 1e-10))

        log_wk_refs = np.array(log_wk_refs)
        gap = log_wk_refs.mean() - log_wk
        gap_std = log_wk_refs.std() * np.sqrt(1 + 1/n_refs)

        gaps.append(gap)
        gap_stds.append(gap_std)

    return np.array(gaps), np.array(gap_stds)


def find_optimal_k_gap(k_values: np.ndarray, gaps: np.ndarray,
                       gap_stds: np.ndarray) -> int:
    """
    根据Gap Statistic找最优K
    规则：找最小的k使得 Gap(k) >= Gap(k+1) - s_{k+1}
    """
    for i in range(len(k_values) - 1):
        if gaps[i] >= gaps[i+1] - gap_stds[i+1]:
            return k_values[i]
    return k_values[-1]


@dataclass
class ClusterMetrics:
    """聚类指标"""
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia:  float
    time_seconds: float


def cluster_single_k(args):
    """单个K值的聚类（用于并行）"""
    X, k, n_init, random_state = args

    start = time.time()
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start

    # 计算指标
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    inertia = kmeans.inertia_

    return ClusterMetrics(
        k=k,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        inertia=inertia,
        time_seconds=elapsed
    )


class ClusteringExplorer:
    """聚类数量探索器（改进版）"""

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

    def __init__(self, uri: str, user: str, password:  str,
                 data_dir: str, database: str = "neo4j",
                 search_radius: float = 8.0,
                 pca_variance_threshold:  float = 0.95,
                 k_range:  Tuple[int, int] = (2, 70),
                 n_init:  int = 10,
                 n_jobs: int = -1,
                 compute_gap:  bool = True,
                 gap_n_refs: int = 10):
        """
        参数:
            k_range: K值搜索范围 (min, max)
            n_init: KMeans初始化次数
            n_jobs: 并行数
            compute_gap:  是否计算Gap Statistic（较耗时）
            gap_n_refs: Gap Statistic的参考数据集数量
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.data_dir = Path(data_dir)
        self.search_radius = search_radius
        self.pca_variance_threshold = pca_variance_threshold
        self.k_min, self.k_max = k_range
        self.n_init = n_init
        self.n_jobs = n_jobs if n_jobs > 0 else N_CORES
        self.compute_gap = compute_gap
        self.gap_n_refs = gap_n_refs

        # 数据
        self.valid_neuron_ids:  List[str] = []
        self.axon_features_raw:  Dict[str, np.ndarray] = {}
        self.dendrite_features_raw: Dict[str, np.ndarray] = {}
        self.local_gene_features_raw: Dict[str, np.ndarray] = {}
        self.projection_vectors_raw: Dict[str, np.ndarray] = {}
        self.all_subclasses: List[str] = []
        self.all_target_regions: List[str] = []

        # PCA向量
        self.morph_vectors:  np.ndarray = None
        self.gene_vectors: np.ndarray = None
        self.proj_vectors: np.ndarray = None

        # 结果
        self.results: Dict[str, List[ClusterMetrics]] = {}
        self.gap_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.elbow_points: Dict[str, Dict[str, int]] = {}  # 各方法找到的拐点

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 数据加载 (保持不变) ====================

    def load_all_data(self) -> int:
        """加载数据"""
        print("\n" + "=" * 80)
        print("加载数据")
        print("=" * 80)

        self._load_local_gene_features_from_cache()
        self._get_global_dimensions()
        self._load_all_neuron_features()
        self._filter_valid_neurons()
        self._process_all_vectors()

        print(f"\n✓ 数据加载完成:")
        print(f"  神经元数:  {len(self.valid_neuron_ids)}")
        print(f"  Morph:  {self.morph_vectors.shape}")
        print(f"  Gene:  {self.gene_vectors.shape}")
        print(f"  Proj: {self.proj_vectors.shape}")

        return len(self.valid_neuron_ids)

    def _load_local_gene_features_from_cache(self):
        cache_file = self.data_dir / "cache" / f"local_env_r{self.search_radius}_mirrored.pkl"
        if not cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在:  {cache_file}")

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
        print(f"  投射目标:  {len(self.all_target_regions)} 个脑区")

    def _load_all_neuron_features(self):
        axon_return = [f"n.{feat} AS `{feat}`" for feat in self.AXONAL_FEATURES]
        dend_return = [f"n.{feat} AS `{feat}`" for feat in self.DENDRITIC_FEATURES]

        query = f"""
        MATCH (n: Neuron)
        WHERE n.axonal_total_length IS NOT NULL AND n.axonal_total_length > 0
          AND n.dendritic_total_length IS NOT NULL AND n.dendritic_total_length > 0
        RETURN n.neuron_id AS neuron_id, {", ".join(axon_return)}, {", ".join(dend_return)}
        """

        proj_query = """
        MATCH (n:Neuron {neuron_id:  $neuron_id})-[p:PROJECT_TO]->(t:Subregion)
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
                proj_dict = {r['target']:  r['weight'] for r in proj_result
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
        print(f"  有效神经元:  {len(self.valid_neuron_ids)}")

    def _process_all_vectors(self):
        """处理向量"""
        print("\n处理向量...")
        neurons = self.valid_neuron_ids

        # 形态
        morph_raw = np.array([
            np.concatenate([self.axon_features_raw[nid], self.dendrite_features_raw[nid]])
            for nid in neurons
        ])
        self.morph_vectors = self._process_vector(morph_raw, 'Morph', do_log=True)

        # 分子
        gene_raw = np.array([self.local_gene_features_raw[nid] for nid in neurons])
        col_sums = gene_raw.sum(axis=0)
        gene_raw = gene_raw[: , col_sums > 0]
        self.gene_vectors = self._process_vector(gene_raw, 'Gene')

        # 投射
        proj_raw = np.array([self.projection_vectors_raw[nid] for nid in neurons])
        col_sums = proj_raw.sum(axis=0)
        proj_raw = proj_raw[:, col_sums > 0]
        self.proj_vectors = self._process_vector(proj_raw, 'Proj', do_log=True)

    def _process_vector(self, X_raw: np.ndarray, name: str,
                        do_log: bool = False) -> np.ndarray:
        original_dims = X_raw.shape[1]

        if do_log:
            X = np.log1p(X_raw)
        else:
            X = X_raw

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)

        variance = np.sum(pca.explained_variance_ratio_)
        print(f"  {name}:  {original_dims}D → {X_pca.shape[1]}D ({variance:.1%})")

        return X_pca

    # ==================== 聚类探索 ====================

    def explore_clustering(self, X: np.ndarray, name: str) -> List[ClusterMetrics]:
        """探索不同K值的聚类效果"""
        print(f"\n--- {name} (K={self.k_min} to {self.k_max}) ---")

        k_values = list(range(self.k_min, self.k_max + 1))

        # 准备并行参数
        args_list = [(X, k, self.n_init, 42) for k in k_values]

        results = []

        # 并行执行
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(cluster_single_k, args): args[1]
                       for args in args_list}

            completed = 0
            for future in as_completed(futures):
                k = futures[future]
                try:
                    metric = future.result()
                    results.append(metric)
                    completed += 1
                    if completed % 10 == 0 or completed == len(k_values):
                        print(f"  完成:  {completed}/{len(k_values)}")
                except Exception as e:
                    print(f"  K={k} 失败: {e}")

        # 按K排序
        results.sort(key=lambda x: x.k)

        # 计算Gap Statistic（可选）
        if self.compute_gap:
            print(f"  计算Gap Statistic...")
            # 采样K值以加速（每隔几个取一个）
            gap_k_step = max(1, (self.k_max - self.k_min) // 20)
            gap_k_values = list(range(self.k_min, self.k_max + 1, gap_k_step))
            if self.k_max not in gap_k_values:
                gap_k_values.append(self.k_max)

            gaps, gap_stds = compute_gap_statistic(X, gap_k_values,
                                                    n_refs=self.gap_n_refs)
            self.gap_results[name] = (np.array(gap_k_values), gaps, gap_stds)

        # 找拐点
        self._find_elbow_points(name, results)

        return results

    def _find_elbow_points(self, name:  str, results: List[ClusterMetrics]):
        """使用多种方法找拐点"""
        k_values = np.array([r.k for r in results])
        inertia = np.array([r.inertia for r in results])
        sil = np.array([r.silhouette for r in results])
        ch = np.array([r.calinski_harabasz for r in results])
        db = np.array([r.davies_bouldin for r in results])

        self.elbow_points[name] = {}

        # 1. Kneedle法 - Inertia
        elbow_inertia, _ = find_elbow_point(k_values, inertia,
                                            direction='decreasing', curve='convex')
        self.elbow_points[name]['Kneedle_Inertia'] = elbow_inertia

        # 2.曲率法 - Inertia
        elbow_curv = find_elbow_by_curvature(k_values, inertia, min_k=5)
        self.elbow_points[name]['Curvature_Inertia'] = elbow_curv

        # 3.Silhouette最大值
        self.elbow_points[name]['Max_Silhouette'] = k_values[np.argmax(sil)]

        # 4.CH最大值
        self.elbow_points[name]['Max_CH'] = k_values[np.argmax(ch)]

        # 5.DB最小值
        self.elbow_points[name]['Min_DB'] = k_values[np.argmin(db)]

        # 6.Gap Statistic
        if name in self.gap_results:
            gap_k, gaps, gap_stds = self.gap_results[name]
            optimal_gap_k = find_optimal_k_gap(gap_k, gaps, gap_stds)
            self.elbow_points[name]['Gap_Statistic'] = optimal_gap_k

        print(f"  拐点检测结果:")
        for method, k in self.elbow_points[name].items():
            print(f"    {method}: K={k}")

    def run_exploration(self):
        """运行所有模态的探索"""
        print("\n" + "=" * 80)
        print(f"聚类数量探索 (K={self.k_min} to {self.k_max})")
        print(f"并行核心数:  {self.n_jobs}")
        print("=" * 80)

        modalities = [
            ('Morph', self.morph_vectors),
            ('Gene', self.gene_vectors),
            ('Proj', self.proj_vectors),
        ]

        for name, vectors in modalities:
            self.results[name] = self.explore_clustering(vectors, name)

        return self.results

    # ==================== 改进的可视化 ====================

    def visualize_results(self, output_dir: str = "."):
        """生成可视化"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("生成可视化")
        print("=" * 80)

        self._plot_elbow_with_detection(output_dir)
        self._plot_curvature_analysis(output_dir)
        self._plot_silhouette_with_peaks(output_dir)
        if self.compute_gap:
            self._plot_gap_statistic(output_dir)
        self._plot_comprehensive_view(output_dir)
        self._plot_voting_summary(output_dir)

        print(f"\n✓ 图表已保存到:  {output_dir}")

    def _save_figure(self, fig, output_dir: str, filename: str):
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {filename}")

    def _plot_elbow_with_detection(self, output_dir: str):
        """Elbow曲线 + 自动拐点检测"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = {'Morph': '#3498DB', 'Gene':  '#27AE60', 'Proj': '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            inertia = np.array([r.inertia for r in results])

            # 归一化
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())

            # 主曲线
            ax.plot(k_values, inertia_norm, color=colors[name], linewidth=2,
                    label='Inertia (normalized)')

            # Kneedle拐点
            elbow_k = self.elbow_points[name]['Kneedle_Inertia']
            elbow_idx = np.where(k_values == elbow_k)[0][0]
            ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Kneedle:  K={elbow_k}')
            ax.scatter(elbow_k, inertia_norm[elbow_idx], color='red', s=150,
                       zorder=5, edgecolor='black', linewidth=2)

            # 曲率拐点
            curv_k = self.elbow_points[name]['Curvature_Inertia']
            if curv_k != elbow_k:
                curv_idx = np.where(k_values == curv_k)[0][0]
                ax.axvline(x=curv_k, color='orange', linestyle=':', linewidth=2,
                           alpha=0.8, label=f'Curvature: K={curv_k}')
                ax.scatter(curv_k, inertia_norm[curv_idx], color='orange', s=100,
                           zorder=5, marker='s', edgecolor='black')

            # Kneedle差值曲线
            _, diff = find_elbow_point(k_values, inertia, 'decreasing', 'convex')
            ax2 = ax.twinx()
            ax2.fill_between(k_values, diff, alpha=0.2, color='gray')
            ax2.plot(k_values, diff, color='gray', linestyle='--', alpha=0.5,
                     label='Kneedle diff')
            ax2.set_ylabel('Kneedle Distance', color='gray', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='gray')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Normalized Inertia', fontsize=11)
            ax.set_title(f'{name} - Elbow Detection', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "1_elbow_with_detection.png")

    def _plot_curvature_analysis(self, output_dir:  str):
        """曲率分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            k_values = np.array([r.k for r in results])
            inertia = np.array([r.inertia for r in results])

            # 上排：一阶导数
            ax1 = axes[0, idx]
            inertia_smooth = gaussian_filter1d(inertia, sigma=1.5)
            first_deriv = np.gradient(inertia_smooth, k_values)

            ax1.plot(k_values, first_deriv, color=colors[name], linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.fill_between(k_values, first_deriv, alpha=0.2, color=colors[name])

            ax1.set_xlabel('K')
            ax1.set_ylabel("First Derivative (dInertia/dK)")
            ax1.set_title(f'{name} - First Derivative', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 下排：曲率
            ax2 = axes[1, idx]
            curvature = compute_curvature(k_values, inertia)

            ax2.plot(k_values, curvature, color=colors[name], linewidth=2)
            ax2.fill_between(k_values, curvature, alpha=0.2, color=colors[name])

            # 标记最大曲率点
            max_curv_k = self.elbow_points[name]['Curvature_Inertia']
            max_curv_idx = np.where(k_values == max_curv_k)[0][0]
            ax2.axvline(x=max_curv_k, color='red', linestyle='--', alpha=0.8)
            ax2.scatter(max_curv_k, curvature[max_curv_idx], color='red', s=100,
                        zorder=5, edgecolor='black')
            ax2.annotate(f'K={max_curv_k}', xy=(max_curv_k, curvature[max_curv_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=11, fontweight='bold', color='red')

            ax2.set_xlabel('K')
            ax2.set_ylabel('Curvature')
            ax2.set_title(f'{name} - Curvature (Max at K={max_curv_k})', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Derivative and Curvature Analysis for Elbow Detection',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "2_curvature_analysis.png")

    def _plot_silhouette_with_peaks(self, output_dir: str):
        """Silhouette曲线 + 峰值检测"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = {'Morph': '#3498DB', 'Gene': '#27AE60', 'Proj': '#E74C3C'}

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            k_values = np.array([r.k for r in results])
            sil = np.array([r.silhouette for r in results])

            # 平滑曲线
            sil_smooth = gaussian_filter1d(sil, sigma=1.0)

            ax.plot(k_values, sil, color=colors[name], linewidth=1.5, alpha=0.5,
                    label='Raw')
            ax.plot(k_values, sil_smooth, color=colors[name], linewidth=2.5,
                    label='Smoothed')
            ax.fill_between(k_values, sil, alpha=0.15, color=colors[name])

            # 找峰值
            peaks, properties = find_peaks(sil_smooth, prominence=0.01, distance=3)

            for peak_idx in peaks[: 5]:  # 最多显示5个峰
                ax.scatter(k_values[peak_idx], sil[peak_idx], color='red',
                           s=80, zorder=5, marker='^', edgecolor='black')
                ax.annotate(f'K={k_values[peak_idx]}',
                            xy=(k_values[peak_idx], sil[peak_idx]),
                            xytext=(5, 8), textcoords='offset points',
                            fontsize=9, color='red')

            # 全局最大值
            max_k = self.elbow_points[name]['Max_Silhouette']
            max_idx = np.where(k_values == max_k)[0][0]
            ax.axvline(x=max_k, color='darkred', linestyle='--', linewidth=2,
                       alpha=0.7, label=f'Best: K={max_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Silhouette Score', fontsize=11)
            ax.set_title(f'{name} - Silhouette Analysis (Best K={max_k})',
                         fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.k_min, self.k_max)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "3_silhouette_with_peaks.png")

    def _plot_gap_statistic(self, output_dir: str):
        """Gap Statistic图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]

            if name not in self.gap_results:
                ax.text(0.5, 0.5, 'Gap Statistic not computed',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            gap_k, gaps, gap_stds = self.gap_results[name]

            ax.errorbar(gap_k, gaps, yerr=gap_stds, color=colors[name],
                        linewidth=2, capsize=4, capthick=1.5,
                        label='Gap ± std', marker='o', markersize=6)
            ax.fill_between(gap_k, gaps - gap_stds, gaps + gap_stds,
                            alpha=0.2, color=colors[name])

            # 最优K
            optimal_k = self.elbow_points[name].get('Gap_Statistic', gap_k[np.argmax(gaps)])
            ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Optimal: K={optimal_k}')

            ax.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax.set_ylabel('Gap Statistic', fontsize=11)
            ax.set_title(f'{name} - Gap Statistic (Optimal K={optimal_k})',
                         fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, output_dir, "4_gap_statistic.png")

    def _plot_comprehensive_view(self, output_dir: str):
        """综合视图：所有指标 + 拐点"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        colors = {'Morph': '#3498DB', 'Gene':  '#27AE60', 'Proj': '#E74C3C'}
        modalities = list(self.results.keys())

        for row, name in enumerate(modalities):
            results = self.results[name]
            k_values = np.array([r.k for r in results])
            color = colors[name]

            # 获取拐点
            elbows = self.elbow_points[name]

            # Silhouette
            ax1 = axes[row, 0]
            sil = np.array([r.silhouette for r in results])
            ax1.plot(k_values, sil, color=color, linewidth=2)
            ax1.fill_between(k_values, sil, alpha=0.2, color=color)
            best_k = elbows['Max_Silhouette']
            ax1.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5)
            ax1.set_title(f'{name} - Silhouette (K={best_k})', fontweight='bold', fontsize=10)
            ax1.set_xlabel('K', fontsize=9)
            ax1.grid(True, alpha=0.3)

            # CH
            ax2 = axes[row, 1]
            ch = np.array([r.calinski_harabasz for r in results])
            ax2.plot(k_values, ch, color=color, linewidth=2)
            ax2.fill_between(k_values, ch, alpha=0.2, color=color)
            best_k = elbows['Max_CH']
            ax2.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5)
            ax2.set_title(f'{name} - Calinski-Harabasz (K={best_k})', fontweight='bold', fontsize=10)
            ax2.set_xlabel('K', fontsize=9)
            ax2.grid(True, alpha=0.3)

            # DB
            ax3 = axes[row, 2]
            db = np.array([r.davies_bouldin for r in results])
            ax3.plot(k_values, db, color=color, linewidth=2)
            ax3.fill_between(k_values, db, alpha=0.2, color=color)
            best_k = elbows['Min_DB']
            ax3.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5)
            ax3.set_title(f'{name} - Davies-Bouldin (K={best_k})', fontweight='bold', fontsize=10)
            ax3.set_xlabel('K', fontsize=9)
            ax3.grid(True, alpha=0.3)

            # Inertia with elbow
            ax4 = axes[row, 3]
            inertia = np.array([r.inertia for r in results])
            inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())
            ax4.plot(k_values, inertia_norm, color=color, linewidth=2)
            ax4.fill_between(k_values, inertia_norm, alpha=0.2, color=color)

            # 两种elbow
            k1 = elbows['Kneedle_Inertia']
            k2 = elbows['Curvature_Inertia']
            ax4.axvline(x=k1, color='red', linestyle='--', linewidth=1.5, label=f'Kneedle={k1}')
            ax4.axvline(x=k2, color='orange', linestyle=':', linewidth=1.5, label=f'Curvature={k2}')
            ax4.set_title(f'{name} - Elbow', fontweight='bold', fontsize=10)
            ax4.set_xlabel('K', fontsize=9)
            ax4.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Clustering Metrics Analysis',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        self._save_figure(fig, output_dir, "5_comprehensive_view.png")

    def _plot_voting_summary(self, output_dir: str):
        """投票汇总图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = {'Morph':  '#3498DB', 'Gene': '#27AE60', 'Proj':  '#E74C3C'}

        for idx, name in enumerate(self.results.keys()):
            ax = axes[idx]
            elbows = self.elbow_points[name]

            # 收集所有K值及其投票
            k_votes = {}
            method_names = []
            method_ks = []

            for method, k in elbows.items():
                method_names.append(method)
                method_ks.append(k)
                k_votes[k] = k_votes.get(k, 0) + 1

            # 按K值排序
            sorted_ks = sorted(k_votes.keys())
            votes = [k_votes[k] for k in sorted_ks]

            # 柱状图
            bars = ax.bar(range(len(sorted_ks)), votes, color=colors[name], alpha=0.7)

            ax.set_xticks(range(len(sorted_ks)))
            ax.set_xticklabels([str(k) for k in sorted_ks], fontsize=9)
            ax.set_xlabel('Optimal K (by different methods)', fontsize=11)
            ax.set_ylabel('Number of Votes', fontsize=11)

            # 找最佳K（票数最多）
            best_k = sorted_ks[np.argmax(votes)]
            max_votes = max(votes)

            ax.set_title(f'{name} - K Voting (Recommended: K={best_k}, {max_votes} votes)',
                         fontsize=12, fontweight='bold')

            # 在柱子上标注
            for i, (k, v) in enumerate(zip(sorted_ks, votes)):
                ax.annotate(str(v), xy=(i, v), ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y')

            # 右侧添加方法列表
            method_text = '\n'.join([f'{m}: K={k}' for m, k in zip(method_names, method_ks)])
            ax.text(1.02, 0.5, method_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        self._save_figure(fig, output_dir, "6_voting_summary.png")

    # ==================== 保存结果 ====================

    def save_results(self, output_dir: str = "."):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 每个模态单独保存
        for name, results in self.results.items():
            rows = [{
                'k': r.k,
                'silhouette':  r.silhouette,
                'calinski_harabasz': r.calinski_harabasz,
                'davies_bouldin': r.davies_bouldin,
                'inertia': r.inertia,
                'time_seconds': r.time_seconds,
            } for r in results]

            df = pd.DataFrame(rows)
            df.to_csv(f"{output_dir}/clustering_metrics_{name.lower()}.csv", index=False)

        # 拐点汇总
        elbow_rows = []
        for name, elbows in self.elbow_points.items():
            row = {'modality': name}
            row.update(elbows)
            elbow_rows.append(row)

        pd.DataFrame(elbow_rows).to_csv(f"{output_dir}/elbow_points_summary.csv", index=False)

        # 推荐K值
        recommendations = []
        for name, elbows in self.elbow_points.items():
            k_votes = {}
            for method, k in elbows.items():
                k_votes[k] = k_votes.get(k, 0) + 1

            # 找票数最多的K
            recommended_k = max(k_votes.keys(), key=lambda x: k_votes[x])
            vote_count = k_votes[recommended_k]

            recommendations.append({
                'modality': name,
                'recommended_k':  recommended_k,
                'vote_count': vote_count,
                'total_methods': len(elbows),
                'silhouette_k': elbows.get('Max_Silhouette'),
                'kneedle_k':  elbows.get('Kneedle_Inertia'),
                'curvature_k':  elbows.get('Curvature_Inertia'),
            })

        pd.DataFrame(recommendations).to_csv(f"{output_dir}/recommended_k.csv", index=False)

        print(f"\n✓ 结果已保存到:  {output_dir}")

    # ==================== 主流程 ====================

    def run_full_pipeline(self, output_dir: str = "./clustering_exploration"):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("聚类数量预实验（改进版）")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  K值范围: {self.k_min} - {self.k_max}")
        print(f"  KMeans n_init: {self.n_init}")
        print(f"  并行核心: {self.n_jobs}")
        print(f"  计算Gap Statistic: {self.compute_gap}")

        start_time = time.time()

        n = self.load_all_data()
        if n == 0:
            return

        self.run_exploration()
        self.visualize_results(output_dir)
        self.save_results(output_dir)
        self._print_summary()

        total_time = time.time() - start_time
        print(f"\n总用时: {total_time / 60:.1f} 分钟")

    def _print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 80)
        print("最优K值推荐汇总")
        print("=" * 80)

        for name, elbows in self.elbow_points.items():
            print(f"\n【{name}】")

            # 投票
            k_votes = {}
            for method, k in elbows.items():
                k_votes[k] = k_votes.get(k, 0) + 1
                print(f"  {method: 25s}: K={k}")

            print(f"\n  投票结果:")
            for k in sorted(k_votes.keys()):
                print(f"    K={k: 3d}: {k_votes[k]} 票")

            recommended_k = max(k_votes.keys(), key=lambda x: k_votes[x])
            print(f"\n  ★ 推荐K值: {recommended_k}")

        print("\n" + "=" * 80)
        print("指标解释:")
        print("-" * 80)
        print("  Silhouette Score: 越大越好，衡量样本与自身簇 vs 最近其他簇的相似度")
        print("  Calinski-Harabasz:  越大越好，类间方差/类内方差，对凸形簇效果好")
        print("  Davies-Bouldin:  越小越好，平均每个簇与其最相似簇的相似度")
        print("  Kneedle:  通过归一化曲线与对角线的距离找拐点")
        print("  Curvature: 通过二阶导数（曲率）的最大值找拐点")
        print("  Gap Statistic: 与随机分布对比，Gap值最大的K（或满足Gap规则的最小K）")
        print("=" * 80)


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neuroxiv"
    NEO4J_DATABASE = "neo4j"
    DATA_DIR = "/home/wlj/NeuroXiv2/data"
    OUTPUT_DIR = "./clustering_exploration"

    with ClusteringExplorer(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            data_dir=DATA_DIR,
            database=NEO4J_DATABASE,
            search_radius=8.0,
            pca_variance_threshold=0.95,
            k_range=(2, 70),
            n_init=10,
            n_jobs=-1,
            compute_gap=True,  # 计算Gap Statistic
            gap_n_refs=10      # Gap的参考数据集数量
    ) as explorer:
        explorer.run_full_pipeline(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()