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

        # ========== 关键修复：处理dendritic特征中的0值 ==========
        # 特征索引：0-3是axonal，4-7是dendritic
        dendritic_indices = [4, 5, 6, 7]  # bifurc_angle, length, branches, max_order

        print("处理dendritic特征的0值（无dendrite的神经元）...")

        # 对dendritic特征，将0值替换为NaN（这样zscore会忽略它们）
        for i in dendritic_indices:
            col = all_sigs[:, i].copy()
            # 将接近0的值（考虑浮点精度）视为无dendrite
            zero_mask = np.abs(col) < 1e-6
            n_zeros = zero_mask.sum()
            if n_zeros > 0:
                feature_names = ['dendritic_bifurcation_angle', 'dendritic_length',
                                 'dendritic_branches', 'dendritic_max_order']
                print(f"  {feature_names[i - 4]:30s}: 排除 {n_zeros}/{len(col)} 个0值")
                col[zero_mask] = np.nan
                all_sigs[:, i] = col

        # 对每个特征维度进行z-score（zscore会自动忽略NaN）
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

    def visualize_mismatch_details(self, top_pairs: Dict, output_dir: str = "."):
        """
        为top mismatch pairs绘制详细对比图

        Args:
            top_pairs: _print_top_mismatch_pairs返回的字典
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("绘制详细对比图...")
        print("=" * 80)

        # 1. 分子-形态 Mismatch 详细图（取top 3）
        print("\n绘制分子-形态 Mismatch 详细对比图...")
        for rank, (r1, r2, mismatch_val) in enumerate(top_pairs['mol_morph'][:3], 1):
            self._plot_mol_morph_comparison(r1, r2, mismatch_val, rank, output_dir)

        # 2. 分子-投射 Mismatch 详细图（取top 3）
        print("\n绘制分子-投射 Mismatch 详细对比图...")
        for rank, (r1, r2, mismatch_val) in enumerate(top_pairs['mol_proj'][:3], 1):
            self._plot_mol_proj_comparison(r1, r2, mismatch_val, rank, output_dir)

        print("\n✓ 所有详细对比图已保存")

    def _plot_mol_morph_comparison(self, region1: str, region2: str,
                                   mismatch: float, rank: int, output_dir: str):
        """
        绘制分子-形态 Mismatch 的详细对比图
        包含：形态雷达图 + 分子组成柱状图
        """
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])

        # === 1. 形态雷达图 ===
        ax_radar = fig.add_subplot(gs[0], projection='polar')

        # 获取形态特征（z-score值）
        morph1_zscore = self.morph_signatures[region1]
        morph2_zscore = self.morph_signatures[region2]

        # 创建雷达图的角度
        feature_names = [
            'Axon\nBifurc\nAngle',
            'Axon\nLength',
            'Axon\nBranches',
            'Axon\nMax\nOrder',
            'Dend\nBifurc\nAngle',
            'Dend\nLength',
            'Dend\nBranches',
            'Dend\nMax\nOrder'
        ]

        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # ========== 改进的归一化方法：使用固定z-score范围 ==========
        # z-score典型范围：[-3, +3] 包含99.7%的数据
        # 我们映射到雷达图的[0.15, 0.85]区间，保持清晰度

        valid_mask = ~(np.isnan(morph1_zscore) | np.isnan(morph2_zscore))

        # 固定的z-score到雷达图的映射
        def zscore_to_radar(zscore_val):
            """将z-score映射到雷达图坐标 [0.15, 0.85]"""
            # z-score在[-3, 3]范围
            # 映射到[0.15, 0.85]，中心0.5对应z-score=0
            if np.isnan(zscore_val):
                return 0.5  # NaN显示在中间

            # 裁剪到[-3, 3]
            z_clipped = np.clip(zscore_val, -3, 3)
            # 线性映射
            radar_val = 0.15 + 0.7 * (z_clipped + 3) / 6
            return radar_val

        # 应用映射
        morph1_plot = np.array([zscore_to_radar(v) for v in morph1_zscore]).tolist()
        morph2_plot = np.array([zscore_to_radar(v) for v in morph2_zscore]).tolist()

        # 闭合
        morph1_plot = morph1_plot + [morph1_plot[0]]
        morph2_plot = morph2_plot + [morph2_plot[0]]

        # 绘制雷达图 - 使用更明显的样式
        ax_radar.plot(angles, morph1_plot, 'o-', linewidth=2.5, markersize=8,
                      label=region1, color='#E74C3C')
        ax_radar.fill(angles, morph1_plot, alpha=0.25, color='#E74C3C')
        ax_radar.plot(angles, morph2_plot, 's-', linewidth=2.5, markersize=8,
                      label=region2, color='#3498DB')
        ax_radar.fill(angles, morph2_plot, alpha=0.25, color='#3498DB')

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(feature_names, size=8)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(['Low', '', 'Mid', '', 'High'], size=10)
        ax_radar.set_title('Morphology Features\n(Normalized Z-scores)',
                           fontsize=11, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax_radar.grid(True, linewidth=0.5, alpha=0.5)

        # 添加参考圆
        for y in [0.25, 0.5, 0.75]:
            ax_radar.plot(angles, [y] * len(angles), '--', linewidth=0.5, color='gray', alpha=0.3)

        # === 2. 分子组成对比 (Top 10 Subclasses) ===
        ax_mol = fig.add_subplot(gs[1])

        mol1 = self.mol_signatures[region1]
        mol2 = self.mol_signatures[region2]

        # 找到两个区域中占比最高的前10个subclass
        top_indices = np.argsort(mol1 + mol2)[-10:][::-1]
        top_subclasses = [self.all_subclasses[i] for i in top_indices]
        top_subclasses_short = [s[:30] + '...' if len(s) > 30 else s for s in top_subclasses]

        mol1_top = mol1[top_indices]
        mol2_top = mol2[top_indices]

        x = np.arange(len(top_subclasses))
        width = 0.35

        ax_mol.barh(x - width / 2, mol1_top, width, label=region1, color='#E74C3C', alpha=0.8)
        ax_mol.barh(x + width / 2, mol2_top, width, label=region2, color='#3498DB', alpha=0.8)

        ax_mol.set_yticks(x)
        ax_mol.set_yticklabels(top_subclasses_short, fontsize=10)
        ax_mol.set_xticklabels(ax_mol.get_xticklabels(),fontsize=10)
        ax_mol.set_xlabel('Cell Type Percentage (%)', fontsize=14)
        ax_mol.set_title('Top 10 Cell Types', fontsize=14, fontweight='bold')
        ax_mol.legend(fontsize=13)
        ax_mol.grid(axis='x', alpha=0.3)

        # === 3. 分子相似度和mismatch说明 ===
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')

        # 计算分子和形态的相似度
        mol_sim = 1 - (1 - cosine(mol1, mol2))
        morph_dist = euclidean(morph1_zscore, morph2_zscore)

        info_text = f"""
Molecular-Morphology Mismatch Analysis
{'=' * 45}

Region Pair: {region1} ↔ {region2}
Rank: #{rank}

Mismatch Score: {mismatch:.4f}
{'=' * 45}

Molecular Similarity: {mol_sim:.4f}
   → Cell type composition similarity

Morphology Distance: {morph_dist:.4f}
   → Euclidean distance of morphology features

{'=' * 45}

Interpretation:
{self._interpret_mol_morph_mismatch(mol_sim, morph_dist, mismatch)}
        """

        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=14, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'Mol-Morph Mismatch #{rank}: {region1} vs {region2}',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        filename = f"{output_dir}/detail_mol_morph_{rank}_{region1}_vs_{region2}.png"
        plt.savefig(filename, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: detail_mol_morph_{rank}_{region1}_vs_{region2}.png")

    def _plot_mol_proj_comparison(self, region1: str, region2: str,
                                  mismatch: float, rank: int, output_dir: str):
        """
        绘制分子-投射 Mismatch 的详细对比图
        包含：投射分布柱状图 + 分子组成柱状图
        """
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

        # === 1. 投射分布对比 (Top 15 Targets) ===
        ax_proj = fig.add_subplot(gs[0])

        proj1 = self.proj_signatures[region1]
        proj2 = self.proj_signatures[region2]

        # 找到两个区域投射最强的前15个目标
        top_indices = np.argsort(proj1 + proj2)[-15:][::-1]
        top_targets = [self.all_target_subregions[i] for i in top_indices]

        proj1_top = proj1[top_indices]
        proj2_top = proj2[top_indices]

        x = np.arange(len(top_targets))
        width = 0.35

        ax_proj.barh(x - width / 2, proj1_top, width, label=region1, color='#E74C3C', alpha=0.8)
        ax_proj.barh(x + width / 2, proj2_top, width, label=region2, color='#3498DB', alpha=0.8)

        ax_proj.set_yticks(x)
        ax_proj.set_yticklabels(top_targets, fontsize=10)
        ax_proj.set_xlabel('Projection Strength (Normalized)', fontsize=14)
        ax_proj.set_title('Top 15 Projection Targets', fontsize=14, fontweight='bold')
        ax_proj.legend(fontsize=9)
        ax_proj.grid(axis='x', alpha=0.3)

        # === 2. 分子组成对比 ===
        ax_mol = fig.add_subplot(gs[1])

        mol1 = self.mol_signatures[region1]
        mol2 = self.mol_signatures[region2]

        top_indices = np.argsort(mol1 + mol2)[-10:][::-1]
        top_subclasses = [self.all_subclasses[i] for i in top_indices]
        top_subclasses_short = [s[:25] + '...' if len(s) > 25 else s for s in top_subclasses]

        mol1_top = mol1[top_indices]
        mol2_top = mol2[top_indices]

        x = np.arange(len(top_subclasses))
        width = 0.35

        ax_mol.barh(x - width / 2, mol1_top, width, label=region1, color='#E74C3C', alpha=0.8)
        ax_mol.barh(x + width / 2, mol2_top, width, label=region2, color='#3498DB', alpha=0.8)

        ax_mol.set_yticks(x)
        ax_mol.set_yticklabels(top_subclasses_short, fontsize=10)
        ax_mol.set_xticklabels(ax_mol.get_xticklabels(), fontsize=10)
        ax_mol.set_xlabel('Cell Type %', fontsize=14)
        ax_mol.set_title('Top 10 Cell Types', fontsize=14, fontweight='bold')
        ax_mol.legend(fontsize=9)
        ax_mol.grid(axis='x', alpha=0.3)

        # === 3. 说明文本 ===
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')

        mol_sim = 1 - (1 - cosine(mol1, mol2))
        proj_sim = 1 - (1 - cosine(proj1, proj2))

        info_text = f"""
Molecular-Projection Mismatch
{'=' * 40}

Region Pair: {region1} ↔ {region2}
Rank: #{rank}

Mismatch Score: {mismatch:.4f}
{'=' * 40}

Molecular Similarity: {mol_sim:.4f}
   → Cell type composition

Projection Similarity: {proj_sim:.4f}
   → Output target pattern

{'=' * 40}

Interpretation:
{self._interpret_mol_proj_mismatch(mol_sim, proj_sim, mismatch)}
        """

        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle(f'Mol-Proj Mismatch #{rank}: {region1} vs {region2}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        filename = f"{output_dir}/detail_mol_proj_{rank}_{region1}_vs_{region2}.png"
        plt.savefig(filename, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: detail_mol_proj_{rank}_{region1}_vs_{region2}.png")

    def _interpret_mol_morph_mismatch(self, mol_sim: float, morph_dist: float,
                                      mismatch: float) -> str:
        """生成分子-形态mismatch的解释文本"""
        if mol_sim > 0.7 and morph_dist > 2.0:
            return """High mismatch: Similar molecular
composition but very different
morphology strategies.
→ Same cell types, different
  wiring patterns."""
        elif mol_sim < 0.3 and morph_dist < 1.0:
            return """High mismatch: Different molecular
composition but similar morphology.
→ Different cell types converge
  to similar morphology."""
        else:
            return """Moderate mismatch: Partial
disagreement between molecular
and morphological organization."""

    def _interpret_mol_proj_mismatch(self, mol_sim: float, proj_sim: float,
                                     mismatch: float) -> str:
        """生成分子-投射mismatch的解释文本"""
        if mol_sim > 0.7 and proj_sim < 0.3:
            return """High mismatch: Similar molecular
composition but very different
projection targets.
→ Same cell types routed to
  different network roles."""
        elif mol_sim < 0.3 and proj_sim > 0.7:
            return """High mismatch: Different molecular
composition but similar projections.
→ Different cell types serve
  similar functional roles."""
        else:
            return """Moderate mismatch: Partial
disagreement between molecular
and projection patterns."""

    # ==================== 7. 主流程 ====================

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
        可视化5个矩阵，并分别保存

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
        # ========== 🔧 修复：使用统一的Min-Max归一化 ==========
        def minmax_normalize_df(df):
            """标准Min-Max归一化"""
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

        # 归一化距离
        mol_dist_norm = minmax_normalize_df(mol_dist_df)
        morph_dist_norm = minmax_normalize_df(morph_dist_df)  # ← 修复
        proj_dist_norm = minmax_normalize_df(proj_dist_df)

        # 计算相似度：1 - normalized_distance
        mol_sim = 1 - mol_dist_norm
        morph_sim = 1 - morph_dist_norm  # ← 修复：统一公式
        proj_sim = 1 - proj_dist_norm

        # 计算mismatch
        mol_morph_mismatch, mol_proj_mismatch = self.compute_mismatch_matrices(
            mol_dist_df, morph_dist_df, proj_dist_df
        )

        # ========== 1. 保存组合图 ==========
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Brain Region Similarity and Mismatch Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        sns.heatmap(mol_sim, ax=axes[0, 0], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 0].set_title('Molecular Similarity', fontsize=16, fontweight='bold')

        sns.heatmap(morph_sim, ax=axes[0, 1], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 1].set_title('Morphology Similarity', fontsize=16, fontweight='bold')

        sns.heatmap(proj_sim, ax=axes[0, 2], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True)
        axes[0, 2].set_title('Projection Similarity', fontsize=16, fontweight='bold')

        sns.heatmap(mol_morph_mismatch, ax=axes[1, 0], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        # axes[1, 0].set_title('Molecular-Morphology Mismatch', fontsize=16, fontweight='bold')

        sns.heatmap(mol_proj_mismatch, ax=axes[1, 1], cmap='RdYlBu_r',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Mismatch'},
                    xticklabels=True, yticklabels=True)
        # axes[1, 1].set_title('Molecular-Projection Mismatch', fontsize=16, fontweight='bold')

        axes[1, 2].axis('off')
        plt.tight_layout()

        combined_path = f"{output_dir}/all_matrices_combined.png"
        plt.savefig(combined_path, dpi=1200, bbox_inches='tight')
        print(f"\n组合矩阵已保存: {combined_path}")
        plt.close()

        # ========== 2. 分别保存每个矩阵 ==========
        print("\n分别保存各个矩阵...")

        # 分子相似性
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mol_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True, annot=False)
        ax.set_title('Molecular fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel('Region', fontsize=20,fontweight='bold')
        ax.set_ylabel('Region', fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_molecular_similarity.png", dpi=1200, bbox_inches='tight')
        plt.close()

        # 形态相似性
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(morph_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True, annot=False)
        ax.set_title('Morphology fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel('Region', fontsize=20,fontweight='bold')
        ax.set_ylabel('Region', fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_morphology_similarity.png", dpi=1200, bbox_inches='tight')
        plt.close()

        # 投射相似性
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(proj_sim, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Similarity'},
                    xticklabels=True, yticklabels=True, annot=False)
        ax.set_title('Projection fingerprint Similarity', fontsize=20, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel('Region', fontsize=20,fontweight='bold')
        ax.set_ylabel('Region', fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_projection_similarity.png", dpi=1200, bbox_inches='tight')
        plt.close()

        # 分子-形态 Mismatch
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mol_morph_mismatch, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True,
                    xticklabels=True, yticklabels=True, annot=False)
        # ax.set_title('Molecular-Morphology Mismatch', fontsize=20, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel('Region', fontsize=20,fontweight='bold')
        ax.set_ylabel('Region', fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_mol_morph_mismatch.png", dpi=1200, bbox_inches='tight')
        plt.close()

        # 分子-投射 Mismatch
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mol_proj_mismatch, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True,xticklabels=True, yticklabels=True, annot=False)
        # ax.set_title('Molecular-Projection Mismatch', fontsize=20, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel('Region', fontsize=20,fontweight='bold')
        ax.set_ylabel('Region', fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_mol_proj_mismatch.png", dpi=1200, bbox_inches='tight')
        plt.close()

        print("✓ 所有矩阵已单独保存")

        # 找出top mismatch pairs并返回（用于后续详细可视化）
        top_pairs = self._print_top_mismatch_pairs(mol_morph_mismatch, mol_proj_mismatch,
                                                   valid_regions, n=10)

        return top_pairs, mol_morph_mismatch, mol_proj_mismatch

    def _print_top_mismatch_pairs(self, mol_morph_mismatch: pd.DataFrame,
                                  mol_proj_mismatch: pd.DataFrame,
                                  regions: List[str], n: int = 10):
        """打印top N的mismatch脑区对，并返回用于详细可视化"""
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
                    # 计算形态差异度（用于筛选有明显差异的对）
                    morph_contrast = self._compute_morphology_contrast(regions[i], regions[j])
                    mm_values.append((regions[i], regions[j], val, morph_contrast))

        # 按mismatch和形态差异度的综合得分排序
        # 综合得分 = mismatch * 0.7 + 形态差异度 * 0.3
        mm_values.sort(key=lambda x: x[2] * 0.7 + x[3] * 0.3, reverse=True)

        for rank, (r1, r2, val, contrast) in enumerate(mm_values[:n], 1):
            print(f"{rank:2d}. {r1:10s} <-> {r2:10s}  |  Mismatch = {val:.4f}  |  形态差异度 = {contrast:.4f}")

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

        # 返回top pairs用于详细可视化（只返回简化的三元组）
        return {
            'mol_morph': [(r1, r2, val) for r1, r2, val, _ in mm_values[:n]],
            'mol_proj': mp_values[:n]
        }

    def _compute_morphology_contrast(self, region1: str, region2: str) -> float:
        """
        计算两个脑区的形态对比度（用于筛选视觉效果好的脑区对）

        对比度高 = 雷达图有明显的差异，适合可视化

        要求：
        1. 8个特征都必须有有效值（非NaN）
        2. dendritic特征必须有实际值（排除无dendrite的脑区）
        3. 两个脑区的差异要明显

        Returns:
            对比度分数 (0-1)，越高越适合可视化
        """
        morph1 = self.morph_signatures[region1]
        morph2 = self.morph_signatures[region2]

        # ========== 关键筛选：必须8个特征都有效 ==========
        valid_mask = ~(np.isnan(morph1) | np.isnan(morph2))
        n_valid = valid_mask.sum()

        # 如果不是全部8个特征都有效，直接返回0
        if n_valid < 8:
            return 0.0

        # ========== 额外检查：dendritic特征必须有实际值 ==========
        # 检查z-score是否过于极端（可能表示原始值是0）
        dendritic_indices = [4, 5, 6, 7]
        for idx in dendritic_indices:
            # 如果dendritic特征的z-score过低（< -2），可能是0值，排除这对
            if morph1[idx] < -2.5 or morph2[idx] < -2.5:
                return 0.0

        morph1_valid = morph1[valid_mask]
        morph2_valid = morph2[valid_mask]

        # 1. 特征标准差（衡量起伏程度）
        std1 = np.std(morph1_valid)
        std2 = np.std(morph2_valid)
        avg_std = (std1 + std2) / 2
        std_score = np.clip(avg_std / 2.0, 0, 1)

        # 2. 两个脑区的差异大小（最重要）
        diff = np.abs(morph1_valid - morph2_valid)
        avg_diff = np.mean(diff)
        diff_score = np.clip(avg_diff / 3.0, 0, 1)

        # 3. 特征范围（max-min）
        range1 = np.max(morph1_valid) - np.min(morph1_valid)
        range2 = np.max(morph2_valid) - np.min(morph2_valid)
        avg_range = (range1 + range2) / 2
        range_score = np.clip(avg_range / 4.0, 0, 1)

        # 4. 完整性（现在总是1.0，因为我们要求8个都有效）
        completeness = 1.0

        # ========== 新增：检查每个特征的差异分布 ==========
        # 如果大部分特征差异都很小，降低分数
        small_diff_count = (diff < 0.5).sum()  # 差异<0.5σ的特征数
        if small_diff_count > 4:  # 超过一半特征差异小
            diff_penalty = 0.5
        else:
            diff_penalty = 1.0

        # 综合得分
        contrast = (
                           std_score * 0.20 +
                           diff_score * 0.40 +  # 提高差异权重
                           range_score * 0.25 +
                           completeness * 0.15
                   ) * diff_penalty

        return contrast

        # python
    def visualize_specific_pairs(
                self,
                mol_morph_pairs=None,
                mol_proj_pairs=None,
                output_dir=".",
                mol_morph_mismatch_df=None,
                mol_proj_mismatch_df=None
        ):
            """
            Manually visualize specified region pairs.

            Args:
                mol_morph_pairs: list of (r1, r2) or (r1, r2, mismatch) for molecular-morphology comparison.
                mol_proj_pairs: list of (r1, r2) or (r1, r2, mismatch) for molecular-projection comparison.
                output_dir: directory to save figures.
                mol_morph_mismatch_df: optional DataFrame produced earlier (mol_morph_mismatch).
                mol_proj_mismatch_df: optional DataFrame produced earlier (mol_proj_mismatch).
            """
            import os
            os.makedirs(output_dir, exist_ok=True)

            if mol_morph_pairs:
                print("\nManual Molecular-Morphology comparisons:")
                for rank, pair in enumerate(mol_morph_pairs, 1):
                    if len(pair) == 3:
                        r1, r2, mismatch = pair
                    else:
                        r1, r2 = pair
                        mismatch = np.nan
                        if mol_morph_mismatch_df is not None:
                            # Try both index orders
                            if r1 in mol_morph_mismatch_df.index and r2 in mol_morph_mismatch_df.columns:
                                mismatch = mol_morph_mismatch_df.loc[r1, r2]
                            elif r2 in mol_morph_mismatch_df.index and r1 in mol_morph_mismatch_df.columns:
                                mismatch = mol_morph_mismatch_df.loc[r2, r1]
                    self._plot_mol_morph_comparison(r1, r2, mismatch, rank, output_dir)

            if mol_proj_pairs:
                print("\nManual Molecular-Projection comparisons:")
                for rank, pair in enumerate(mol_proj_pairs, 1):
                    if len(pair) == 3:
                        r1, r2, mismatch = pair
                    else:
                        r1, r2 = pair
                        mismatch = np.nan
                        if mol_proj_mismatch_df is not None:
                            if r1 in mol_proj_mismatch_df.index and r2 in mol_proj_mismatch_df.columns:
                                mismatch = mol_proj_mismatch_df.loc[r1, r2]
                            elif r2 in mol_proj_mismatch_df.index and r1 in mol_proj_mismatch_df.columns:
                                mismatch = mol_proj_mismatch_df.loc[r2, r1]
                    self._plot_mol_proj_comparison(r1, r2, mismatch, rank, output_dir)

    # ==================== 7. 主流程 ====================

    def run_full_analysis(self, output_dir: str = "./fingerprint_results",
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

        # Step 5: 可视化矩阵（分别保存）
        top_pairs, mol_morph_mismatch, mol_proj_mismatch = self.visualize_matrices(
            top_regions, output_dir
        )

        # Step 6: 绘制详细对比图
        # self.visualize_mismatch_details(top_pairs, output_dir)
        manual_mol_morph = [("CA3", "MOs"),("CA3", "ACAd"), ("CA3", "SUB")]
        manual_mol_proj = [ ("CA3", "MOs"),("CA3", "ACAd"), ("CA3", "SUB")]

        self.visualize_specific_pairs(
            mol_morph_pairs=manual_mol_morph,
            mol_proj_pairs=manual_mol_proj,
            output_dir=output_dir,
            mol_morph_mismatch_df=mol_morph_mismatch,
            mol_proj_mismatch_df=mol_proj_mismatch
        )

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
    OUTPUT_DIR = "./fingerprint_results_v5_RdYlBu_r"
    TOP_N_REGIONS = 30

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