"""
脑区指纹计算与可视化 - 修复版本
根据知识图谱计算分子指纹、形态指纹和投射指纹，并分析区域间的mismatch

依赖：
- neo4j
- pandas
- numpy
- scipy
- matplotlib
- seaborn
"""

import neo4j
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BrainRegionFingerprints:
    """脑区指纹计算类"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j数据库URI (例如: "bolt://localhost:7687")
            user: 用户名
            password: 密码
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

        # 存储计算结果
        self.regions = []
        self.mol_signatures = {}
        self.morph_signatures = {}
        self.proj_signatures = {}

        self.all_subclasses = []
        self.all_target_subregions = []

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 1. 分子指纹 (Molecular Fingerprint) ====================

    def get_all_subclasses(self):
        """获取全局所有subclass的列表"""
        query = """
        MATCH (:Region)-[:HAS_SUBCLASS]->(sc:Subclass)
        RETURN DISTINCT sc.name AS subclass_name
        ORDER BY subclass_name
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.all_subclasses = [record['subclass_name'] for record in result]

        print(f"找到 {len(self.all_subclasses)} 个全局subclass")
        return self.all_subclasses

    def compute_molecular_signature(self, region: str) -> np.ndarray:
        """
        计算单个脑区的分子指纹

        Args:
            region: 脑区acronym

        Returns:
            分子指纹向量
        """
        query = """
        MATCH (r:Region {acronym: $region})
        MATCH (r)-[hs:HAS_SUBCLASS]->(sc:Subclass)
        RETURN
          sc.name AS subclass_name,
          hs.pct_cells AS pct_cells
        ORDER BY subclass_name
        """

        with self.driver.session() as session:
            result = session.run(query, region=region)
            data = {record['subclass_name']: record['pct_cells']
                    for record in result}

        # 构建固定维度的向量
        signature = np.zeros(len(self.all_subclasses))
        for i, subclass in enumerate(self.all_subclasses):
            if subclass in data:
                signature[i] = data[subclass]

        return signature

    def compute_all_molecular_signatures(self):
        """计算所有脑区的分子指纹"""
        print("\n=== 计算分子指纹 ===")

        # 获取所有脑区
        query = """
        MATCH (r:Region)
        WHERE EXISTS((r)-[:HAS_SUBCLASS]->())
        RETURN r.acronym AS acronym
        ORDER BY acronym
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.regions = [record['acronym'] for record in result]

        print(f"找到 {len(self.regions)} 个有分子数据的脑区")

        # 计算每个脑区的指纹
        for region in self.regions:
            sig = self.compute_molecular_signature(region)
            self.mol_signatures[region] = sig

        print(f"完成 {len(self.mol_signatures)} 个分子指纹计算")

    # ==================== 2. 形态指纹 (Morphology Fingerprint) ====================

    def compute_morphology_signature(self, region: str) -> np.ndarray:
        """
        计算单个脑区的形态指纹
        直接从Region节点获取聚合后的形态特征

        Args:
            region: 脑区acronym

        Returns:
            形态指纹向量 [8个特征]
        """
        query = """
        MATCH (r:Region {acronym: $region})
        RETURN
          r.axonal_bifurcation_remote_angle AS axonal_bifurcation_remote_angle,
          r.axonal_length AS axonal_length,
          r.axonal_branches AS axonal_branches,
          r.axonal_maximum_branch_order AS axonal_max_branch_order,
          r.dendritic_bifurcation_remote_angle AS dendritic_bifurcation_remote_angle,
          r.dendritic_length AS dendritic_length,
          r.dendritic_branches AS dendritic_branches,
          r.dendritic_maximum_branch_order AS dendritic_max_branch_order
        """

        with self.driver.session() as session:
            result = session.run(query, region=region)
            record = result.single()

        # 如果没有数据，返回8个NaN
        if not record:
            return np.array([np.nan] * 8)

        # 按照固定顺序提取特征值
        features = [
            'axonal_bifurcation_remote_angle',
            'axonal_length',
            'axonal_branches',
            'axonal_max_branch_order',
            'dendritic_bifurcation_remote_angle',
            'dendritic_length',
            'dendritic_branches',
            'dendritic_max_branch_order'
        ]

        signature = np.array([record[feat] if record[feat] is not None else np.nan
                              for feat in features])
        return signature

    def compute_all_morphology_signatures(self):
        """计算所有脑区的形态指纹"""
        print("\n=== 计算形态指纹 ===")

        for region in self.regions:
            sig = self.compute_morphology_signature(region)
            self.morph_signatures[region] = sig

        # 检查并修复不一致的数组长度
        print("检查形态指纹数组维度...")
        fixed_signatures = {}
        for region in self.regions:
            sig = self.morph_signatures[region]
            # 确保是8维向量
            if len(sig) != 8:
                print(f"警告: {region} 的形态指纹维度不正确 (长度={len(sig)})，填充为8维")
                fixed_sig = np.array([np.nan] * 8)
                fixed_sig[:min(len(sig), 8)] = sig[:min(len(sig), 8)]
                fixed_signatures[region] = fixed_sig
            else:
                fixed_signatures[region] = sig

        self.morph_signatures = fixed_signatures

        # Z-score标准化（跨所有区域）
        all_sigs = np.array([self.morph_signatures[r] for r in self.regions])

        print(f"形态指纹数组形状: {all_sigs.shape}")

        # 对每个特征维度进行z-score
        for i in range(all_sigs.shape[1]):
            col = all_sigs[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                col[valid] = zscore(col[valid])
                all_sigs[:, i] = col

        # 更新为z-score后的值
        for i, region in enumerate(self.regions):
            self.morph_signatures[region] = all_sigs[i]

        print(f"完成 {len(self.morph_signatures)} 个形态指纹计算（已z-score标准化）")

    # ==================== 3. 投射指纹 (Projection Fingerprint) ====================

    def get_all_target_subregions(self):
        """获取全局所有投射目标subregion的列表"""
        query = """
        MATCH (:Neuron)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0
        RETURN DISTINCT t.acronym AS target_subregion
        ORDER BY target_subregion
        """

        with self.driver.session() as session:
            result = session.run(query)
            self.all_target_subregions = [record['target_subregion']
                                          for record in result]

        print(f"找到 {len(self.all_target_subregions)} 个全局投射目标subregion")
        return self.all_target_subregions

    def compute_projection_signature(self, region: str) -> np.ndarray:
        """
        计算单个脑区的投射指纹

        Args:
            region: 脑区acronym

        Returns:
            投射指纹向量 (归一化的概率分布)
        """
        query = """
        MATCH (r:Region {acronym: $region})

        // 找属于这个区域的神经元
        OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT]->(r)
        OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_SUBREGION]->(r)
        OPTIONAL MATCH (n3:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(r)
        WITH r, (COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) + COLLECT(DISTINCT n3)) AS ns
        UNWIND ns AS n
        WITH DISTINCT n
        WHERE n IS NOT NULL

        // 找这些神经元的投射
        MATCH (n)-[p:PROJECT_TO]->(t:Subregion)
        WHERE p.weight IS NOT NULL AND p.weight > 0

        WITH t.acronym AS tgt_subregion,
             SUM(p.weight) AS total_weight_to_tgt
        RETURN
          tgt_subregion,
          total_weight_to_tgt
        ORDER BY total_weight_to_tgt DESC
        """

        with self.driver.session() as session:
            result = session.run(query, region=region)
            data = {record['tgt_subregion']: record['total_weight_to_tgt']
                    for record in result}

        # 构建固定维度的向量
        raw_values = np.zeros(len(self.all_target_subregions))
        for i, tgt in enumerate(self.all_target_subregions):
            if tgt in data:
                raw_values[i] = data[tgt]

        # Log稳定化
        log_values = np.log10(1 + raw_values)

        # 归一化成概率分布
        total = log_values.sum()
        if total > 0:
            signature = log_values / (total + 1e-9)
        else:
            signature = log_values

        return signature

    def compute_all_projection_signatures(self):
        """计算所有脑区的投射指纹"""
        print("\n=== 计算投射指纹 ===")

        for region in self.regions:
            sig = self.compute_projection_signature(region)
            self.proj_signatures[region] = sig

        print(f"完成 {len(self.proj_signatures)} 个投射指纹计算")

    # ==================== 4. 相似度和距离计算 ====================

    def compute_distance_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        计算三种距离矩阵

        Returns:
            (mol_dist, morph_dist, proj_dist)
        """
        n = len(self.regions)
        mol_dist = np.zeros((n, n))
        morph_dist = np.zeros((n, n))
        proj_dist = np.zeros((n, n))

        for i, region_a in enumerate(self.regions):
            for j, region_b in enumerate(self.regions):
                if i == j:
                    continue

                # 分子距离 (1 - cosine similarity)
                try:
                    mol_dist[i, j] = 1 - (1 - cosine(
                        self.mol_signatures[region_a],
                        self.mol_signatures[region_b]
                    ))
                except:
                    mol_dist[i, j] = np.nan

                # 形态距离 (Euclidean on z-scored features)
                sig_a = self.morph_signatures[region_a]
                sig_b = self.morph_signatures[region_b]
                if not np.any(np.isnan(sig_a)) and not np.any(np.isnan(sig_b)):
                    morph_dist[i, j] = euclidean(sig_a, sig_b)
                else:
                    morph_dist[i, j] = np.nan

                # 投射距离 (1 - cosine similarity)
                try:
                    proj_dist[i, j] = 1 - (1 - cosine(
                        self.proj_signatures[region_a],
                        self.proj_signatures[region_b]
                    ))
                except:
                    proj_dist[i, j] = np.nan

        # 转换为DataFrame
        mol_dist_df = pd.DataFrame(mol_dist, index=self.regions, columns=self.regions)
        morph_dist_df = pd.DataFrame(morph_dist, index=self.regions, columns=self.regions)
        proj_dist_df = pd.DataFrame(proj_dist, index=self.regions, columns=self.regions)

        return mol_dist_df, morph_dist_df, proj_dist_df

    def compute_mismatch_matrices(self, mol_dist_df: pd.DataFrame,
                                  morph_dist_df: pd.DataFrame,
                                  proj_dist_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算mismatch矩阵

        Args:
            mol_dist_df: 分子距离矩阵
            morph_dist_df: 形态距离矩阵
            proj_dist_df: 投射距离矩阵

        Returns:
            (mol_morph_mismatch, mol_proj_mismatch)
        """

        # Min-Max归一化到[0,1]
        def minmax_normalize(df):
            values = df.values
            valid = ~np.isnan(values)
            if valid.sum() == 0:
                return df

            vmin = values[valid].min()
            vmax = values[valid].max()

            if vmax - vmin < 1e-9:
                return pd.DataFrame(np.zeros_like(values),
                                    index=df.index, columns=df.columns)

            normalized = (values - vmin) / (vmax - vmin)
            return pd.DataFrame(normalized, index=df.index, columns=df.columns)

        mol_norm = minmax_normalize(mol_dist_df)
        morph_norm = minmax_normalize(morph_dist_df)
        proj_norm = minmax_normalize(proj_dist_df)

        # 计算mismatch
        mol_morph_mismatch = np.abs(mol_norm - morph_norm)
        mol_proj_mismatch = np.abs(mol_norm - proj_norm)

        return mol_morph_mismatch, mol_proj_mismatch

    # ==================== 5. 数据保存 ====================

    def save_fingerprints_to_csv(self, output_dir: str = "."):
        """将三种指纹保存为CSV文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 分子指纹
        mol_df = pd.DataFrame.from_dict(self.mol_signatures, orient='index',
                                        columns=self.all_subclasses)
        mol_df.index.name = 'region'
        mol_df.to_csv(f"{output_dir}/molecular_fingerprints.csv")
        print(f"\n分子指纹已保存: {output_dir}/molecular_fingerprints.csv")

        # 形态指纹
        morph_feature_names = [
            'axonal_bifurcation_remote_angle_mean_zscore',
            'axonal_length_mean_zscore',
            'axonal_branches_mean_zscore',
            'axonal_max_branch_order_mean_zscore',
            'dendritic_bifurcation_remote_angle_mean_zscore',
            'dendritic_length_mean_zscore',
            'dendritic_branches_mean_zscore',
            'dendritic_max_branch_order_mean_zscore'
        ]
        morph_df = pd.DataFrame.from_dict(self.morph_signatures, orient='index',
                                          columns=morph_feature_names)
        morph_df.index.name = 'region'
        morph_df.to_csv(f"{output_dir}/morphology_fingerprints.csv")
        print(f"形态指纹已保存: {output_dir}/morphology_fingerprints.csv")

        # 投射指纹
        proj_df = pd.DataFrame.from_dict(self.proj_signatures, orient='index',
                                         columns=self.all_target_subregions)
        proj_df.index.name = 'region'
        proj_df.to_csv(f"{output_dir}/projection_fingerprints.csv")
        print(f"投射指纹已保存: {output_dir}/projection_fingerprints.csv")

    # ==================== 6. 可视化 ====================

    def select_top_regions_by_neuron_count(self, n: int = 20) -> List[str]:
        """
        根据连接的神经元数量选择top N个脑区

        Args:
            n: 选择的脑区数量

        Returns:
            脑区acronym列表
        """
        query = """
        MATCH (r:Region)
        OPTIONAL MATCH (n:Neuron)-[:LOCATE_AT]->(r)
        WITH r, COUNT(DISTINCT n) AS neuron_count
        WHERE neuron_count > 0
        RETURN r.acronym AS region, neuron_count
        ORDER BY neuron_count DESC
        LIMIT $n
        """

        with self.driver.session() as session:
            result = session.run(query, n=n)
            top_regions = [record['region'] for record in result]

        print(f"\n选择了神经元数量最多的 {len(top_regions)} 个脑区:")
        print(top_regions)

        return top_regions

    def visualize_matrices(self, top_regions: List[str], output_dir: str = "."):
        """
        可视化5个矩阵

        Args:
            top_regions: 要可视化的脑区列表
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 筛选top regions
        valid_regions = [r for r in top_regions if r in self.regions]
        print(f"\n开始可视化 {len(valid_regions)} 个脑区的矩阵...")

        # 重新计算距离矩阵（只针对top regions）
        n = len(valid_regions)
        mol_dist = np.zeros((n, n))
        morph_dist = np.zeros((n, n))
        proj_dist = np.zeros((n, n))

        for i, region_a in enumerate(valid_regions):
            for j, region_b in enumerate(valid_regions):
                if i == j:
                    continue

                # 分子距离
                try:
                    mol_dist[i, j] = 1 - (1 - cosine(
                        self.mol_signatures[region_a],
                        self.mol_signatures[region_b]
                    ))
                except:
                    mol_dist[i, j] = np.nan

                # 形态距离
                sig_a = self.morph_signatures[region_a]
                sig_b = self.morph_signatures[region_b]
                if not np.any(np.isnan(sig_a)) and not np.any(np.isnan(sig_b)):
                    morph_dist[i, j] = euclidean(sig_a, sig_b)
                else:
                    morph_dist[i, j] = np.nan

                # 投射距离
                try:
                    proj_dist[i, j] = 1 - (1 - cosine(
                        self.proj_signatures[region_a],
                        self.proj_signatures[region_b]
                    ))
                except:
                    proj_dist[i, j] = np.nan

        mol_dist_df = pd.DataFrame(mol_dist, index=valid_regions, columns=valid_regions)
        morph_dist_df = pd.DataFrame(morph_dist, index=valid_regions, columns=valid_regions)
        proj_dist_df = pd.DataFrame(proj_dist, index=valid_regions, columns=valid_regions)

        # 计算相似度（1 - distance）
        mol_sim = 1 - mol_dist_df
        morph_sim = 1 - morph_dist_df / morph_dist_df.max().max()  # 标准化
        proj_sim = 1 - proj_dist_df

        # 计算mismatch
        mol_morph_mismatch, mol_proj_mismatch = self.compute_mismatch_matrices(
            mol_dist_df, morph_dist_df, proj_dist_df
        )

        # 创建5个子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Brain Region Similarity and Mismatch Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. 分子相似矩阵
        sns.heatmap(mol_sim, ax=axes[0, 0], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 0].set_title('Molecular Similarity', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Region')

        # 2. 形态相似矩阵
        sns.heatmap(morph_sim, ax=axes[0, 1], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 1].set_title('Morphology Similarity', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Region')
        axes[0, 1].set_ylabel('Region')

        # 3. 投射相似矩阵
        sns.heatmap(proj_sim, ax=axes[0, 2], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 2].set_title('Projection Similarity', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Region')
        axes[0, 2].set_ylabel('Region')

        # 4. 分子-形态 Mismatch
        sns.heatmap(mol_morph_mismatch, ax=axes[1, 0], cmap='YlOrRd',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        axes[1, 0].set_title('Molecular-Morphology Mismatch', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Region')
        axes[1, 0].set_ylabel('Region')

        # 5. 分子-投射 Mismatch
        sns.heatmap(mol_proj_mismatch, ax=axes[1, 1], cmap='YlOrRd',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        axes[1, 1].set_title('Molecular-Projection Mismatch', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Region')

        # 隐藏第6个子图
        axes[1, 2].axis('off')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path = f"{output_dir}/brain_region_similarity_mismatch_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n矩阵可视化已保存: {output_path}")

        plt.close()

        # 找出top mismatch pairs
        self._print_top_mismatch_pairs(mol_morph_mismatch, mol_proj_mismatch,
                                       valid_regions, n=10)

    def _print_top_mismatch_pairs(self, mol_morph_mismatch: pd.DataFrame,
                                  mol_proj_mismatch: pd.DataFrame,
                                  regions: List[str], n: int = 10):
        """打印top N的mismatch脑区对"""
        print("\n" + "=" * 80)
        print("Top Mismatch Region Pairs")
        print("=" * 80)

        # Molecular-Morphology Mismatch
        print(f"\n【分子-形态 Mismatch Top {n}】")
        print("(相同分子背景但形态差异大，或不同分子背景但形态相似)")
        print("-" * 80)

        mm_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mol_morph_mismatch.iloc[i, j]
                if not np.isnan(val):
                    mm_values.append((regions[i], regions[j], val))

        mm_values.sort(key=lambda x: x[2], reverse=True)
        for rank, (r1, r2, val) in enumerate(mm_values[:n], 1):
            print(f"{rank:2d}. {r1:10s} <-> {r2:10s}  |  Mismatch = {val:.4f}")

        # Molecular-Projection Mismatch
        print(f"\n【分子-投射 Mismatch Top {n}】")
        print("(相同分子背景但投射目标不同，或不同分子背景但投射相似)")
        print("-" * 80)

        mp_values = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                val = mol_proj_mismatch.iloc[i, j]
                if not np.isnan(val):
                    mp_values.append((regions[i], regions[j], val))

        mp_values.sort(key=lambda x: x[2], reverse=True)
        for rank, (r1, r2, val) in enumerate(mp_values[:n], 1):
            print(f"{rank:2d}. {r1:10s} <-> {r2:10s}  |  Mismatch = {val:.4f}")

        print("=" * 80 + "\n")

    # ==================== 7. 主流程 ====================

    def run_full_analysis(self, output_dir: str = "./fingerprint_results_v2",
                          top_n_regions: int = 20):
        """
        运行完整分析流程

        Args:
            output_dir: 输出目录
            top_n_regions: 选择多少个神经元数量最多的脑区进行可视化
        """
        print("\n" + "=" * 80)
        print("脑区指纹分析 - 完整流程")
        print("=" * 80)

        # Step 1: 获取全局维度
        self.get_all_subclasses()
        self.get_all_target_subregions()

        # Step 2: 计算三种指纹
        self.compute_all_molecular_signatures()
        self.compute_all_morphology_signatures()
        self.compute_all_projection_signatures()

        # Step 3: 保存指纹到CSV
        self.save_fingerprints_to_csv(output_dir)

        # Step 4: 选择top N脑区
        top_regions = self.select_top_regions_by_neuron_count(top_n_regions)

        # Step 5: 可视化矩阵
        self.visualize_matrices(top_regions, output_dir)

        print("\n" + "=" * 80)
        print("分析完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 80 + "\n")


# ==================== 主程序 ====================

def main():
    """主程序入口"""

    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"  # 修改为你的Neo4j地址
    NEO4J_USER = "neo4j"  # 修改为你的用户名
    NEO4J_PASSWORD = "neuroxiv"  # 修改为你的密码

    # 输出配置
    OUTPUT_DIR = "./fingerprint_results"
    TOP_N_REGIONS = 20

    print("\n" + "=" * 80)
    print("脑区指纹计算与可视化")
    print("=" * 80)
    print(f"\nNeo4j URI: {NEO4J_URI}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"选择前 {TOP_N_REGIONS} 个脑区进行可视化\n")

    # 运行分析
    with BrainRegionFingerprints(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as analyzer:
        analyzer.run_full_analysis(
            output_dir=OUTPUT_DIR,
            top_n_regions=TOP_N_REGIONS
        )


if __name__ == "__main__":
    main()