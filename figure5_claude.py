"""
脑区形态特征与分子组成关系的可视化分析
分析region之间的形态特征相似性和分子组成相似性的关系
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BrainRegionAnalyzer:
    """脑区形态与分子组成分析器"""

    def __init__(self, data_dir):
        """
        初始化分析器

        参数:
            data_dir: 包含CSV文件的目录路径
        """
        self.data_dir = Path(data_dir)
        self.regions_df = None
        self.has_subclass_df = None
        self.morph_features = None
        self.molecular_features = None
        self.morph_distances = None
        self.molecular_distances = None
        self.region_pairs = None

        # 形态学属性列表
        self.MORPH_ATTRIBUTES = [
            'axonal_bifurcation_remote_angle',
            'axonal_branches',
            'axonal_length',
            'axonal_maximum_branch_order',
            'dendritic_bifurcation_remote_angle',
            'dendritic_branches',
            'dendritic_length',
            'dendritic_maximum_branch_order'
        ]

    def load_data(self):
        """加载数据文件"""
        # 加载regions.csv
        regions_file = self.data_dir / "nodes" / "regions.csv"
        if not regions_file.exists():
            print(f"错误: {regions_file} 不存在")
            return False

        self.regions_df = pd.read_csv(regions_file)
        print(f"加载了 {len(self.regions_df)} 个region节点")

        # 加载has_subclass.csv
        has_subclass_file = self.data_dir / "relationships" / "has_subclass.csv"
        if not has_subclass_file.exists():
            print(f"错误: {has_subclass_file} 不存在")
            return False

        self.has_subclass_df = pd.read_csv(has_subclass_file)
        print(f"加载了 {len(self.has_subclass_df)} 个has_subclass关系")

        return True

    def prepare_morphological_features(self):
        """准备形态学特征矩阵"""
        # 提取形态学特征列
        morph_cols = []
        for col in self.regions_df.columns:
            for attr in self.MORPH_ATTRIBUTES:
                if attr in col and ':float' in col:
                    morph_cols.append(col)
                    break

        if not morph_cols:
            print("警告: 没有找到形态学特征列")
            return False

        print(f"找到 {len(morph_cols)} 个形态学特征列")

        # 创建形态学特征矩阵
        self.morph_features = self.regions_df[['region_id:ID(Region)'] + morph_cols].copy()
        self.morph_features.columns = ['region_id'] + [col.replace(':float', '') for col in morph_cols]

        # 填充缺失值
        feature_cols = self.morph_features.columns[1:]
        self.morph_features[feature_cols] = self.morph_features[feature_cols].fillna(0)

        # 标准化特征
        scaler = StandardScaler()
        self.morph_features[feature_cols] = scaler.fit_transform(self.morph_features[feature_cols])

        print(f"形态学特征矩阵形状: {self.morph_features.shape}")
        return True

    def prepare_molecular_features(self):
        """准备分子组成特征矩阵（基于subclass）"""
        # 创建region-subclass矩阵
        pivot_df = self.has_subclass_df.pivot_table(
            index=':START_ID(Region)',
            columns=':END_ID(Subclass)',
            values='pct_cells:float',
            fill_value=0
        )

        # 重置索引
        pivot_df.reset_index(inplace=True)
        pivot_df.rename(columns={':START_ID(Region)': 'region_id'}, inplace=True)

        self.molecular_features = pivot_df

        # 标准化特征（每个region的subclass比例和为1，所以可以直接使用）
        # 但为了和形态学特征保持一致，我们也进行标准化
        feature_cols = self.molecular_features.columns[1:]
        scaler = StandardScaler()
        self.molecular_features[feature_cols] = scaler.fit_transform(self.molecular_features[feature_cols])

        print(f"分子组成特征矩阵形状: {self.molecular_features.shape}")
        return True

    def calculate_distances(self):
        """计算形态学和分子组成的距离矩阵"""
        # 确保两个特征矩阵的region顺序一致
        common_regions = set(self.morph_features['region_id']) & set(self.molecular_features['region_id'])
        common_regions = sorted(list(common_regions))

        if len(common_regions) < 2:
            print("错误: 共同的region数量不足")
            return False

        print(f"共同的region数量: {len(common_regions)}")

        # 过滤并排序
        morph_matrix = self.morph_features[self.morph_features['region_id'].isin(common_regions)]
        morph_matrix = morph_matrix.sort_values('region_id')

        molecular_matrix = self.molecular_features[self.molecular_features['region_id'].isin(common_regions)]
        molecular_matrix = molecular_matrix.sort_values('region_id')

        # 计算距离矩阵
        morph_dist = pdist(morph_matrix.iloc[:, 1:].values, metric='euclidean')
        molecular_dist = pdist(molecular_matrix.iloc[:, 1:].values, metric='euclidean')

        # 转换为方阵
        self.morph_distances = squareform(morph_dist)
        self.molecular_distances = squareform(molecular_dist)

        # 保存region列表
        self.region_list = morph_matrix['region_id'].tolist()

        # 创建region对的DataFrame
        pairs = []
        for i in range(len(self.region_list)):
            for j in range(i + 1, len(self.region_list)):
                pairs.append({
                    'region1': self.region_list[i],
                    'region2': self.region_list[j],
                    'morph_distance': self.morph_distances[i, j],
                    'molecular_distance': self.molecular_distances[i, j]
                })

        self.region_pairs = pd.DataFrame(pairs)
        print(f"计算了 {len(self.region_pairs)} 对region之间的距离")

        return True

    def plot_distance_scatter(self, save_path=None):
        """
        绘制图A: 形态特征距离vs分子组成距离的散点图
        """
        if self.region_pairs is None:
            print("错误: 需要先计算距离")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图
        scatter = ax.scatter(
            self.region_pairs['morph_distance'],
            self.region_pairs['molecular_distance'],
            alpha=0.6,
            s=30,
            c=self.region_pairs['morph_distance'] + self.region_pairs['molecular_distance'],
            cmap='viridis'
        )

        # 添加趋势线
        z = np.polyfit(self.region_pairs['morph_distance'],
                       self.region_pairs['molecular_distance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.region_pairs['morph_distance'].min(),
                             self.region_pairs['morph_distance'].max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.6, linewidth=2, label=f'Trend line')

        # 计算相关系数
        correlation = self.region_pairs['morph_distance'].corr(self.region_pairs['molecular_distance'])

        ax.set_xlabel('Morphological Distance', fontsize=12)
        ax.set_ylabel('Molecular Distance', fontsize=12)
        ax.set_title(f'Morphological vs Molecular Distance\n(Correlation: {correlation:.3f})',
                     fontsize=14, pad=20)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Combined Distance', rotation=270, labelpad=20)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图A保存到: {save_path}")

        plt.show()

        return fig

    def find_interesting_pairs(self, n_pairs=3):
        """
        找出有趣的region对
        1. 形态相似但分子组成差异大
        2. 形态差异大但分子组成相似
        """
        if self.region_pairs is None:
            print("错误: 需要先计算距离")
            return None, None

        # 标准化距离用于比较
        morph_norm = (self.region_pairs['morph_distance'] - self.region_pairs['morph_distance'].mean()) / \
                     self.region_pairs['morph_distance'].std()
        molecular_norm = (self.region_pairs['molecular_distance'] - self.region_pairs['molecular_distance'].mean()) / \
                         self.region_pairs['molecular_distance'].std()

        # 计算差异
        self.region_pairs['diff_score'] = molecular_norm - morph_norm  # 正值表示分子差异大于形态差异

        # 找出形态相似但分子差异大的对
        type1_pairs = self.region_pairs.nlargest(n_pairs, 'diff_score')

        # 找出形态差异大但分子相似的对
        type2_pairs = self.region_pairs.nsmallest(n_pairs, 'diff_score')

        print("\n形态相似但分子组成差异大的region对:")
        for _, row in type1_pairs.iterrows():
            print(f"  Region {row['region1']} - Region {row['region2']}: "
                  f"形态距离={row['morph_distance']:.3f}, 分子距离={row['molecular_distance']:.3f}")

        print("\n形态差异大但分子组成相似的region对:")
        for _, row in type2_pairs.iterrows():
            print(f"  Region {row['region1']} - Region {row['region2']}: "
                  f"形态距离={row['morph_distance']:.3f}, 分子距离={row['molecular_distance']:.3f}")

        return type1_pairs, type2_pairs

    def plot_morphology_radar(self, region_pairs, save_path=None):
        """
        绘制图B: 形态差异雷达图
        """
        if region_pairs is None or len(region_pairs) == 0:
            print("错误: 没有region对")
            return

        # 选择第一对region
        pair = region_pairs.iloc[0]
        region1_id = pair['region1']
        region2_id = pair['region2']

        # 获取两个region的形态特征
        region1_features = self.morph_features[self.morph_features['region_id'] == region1_id].iloc[0, 1:].values
        region2_features = self.morph_features[self.morph_features['region_id'] == region2_id].iloc[0, 1:].values

        # 特征名称（简化）
        feature_names = [col.replace('_', '\n') for col in self.morph_features.columns[1:]]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()

        # 闭合图形
        region1_features = np.concatenate((region1_features, [region1_features[0]]))
        region2_features = np.concatenate((region2_features, [region2_features[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 绘制两个region
        ax.plot(angles, region1_features, 'o-', linewidth=2, label=f'Region {region1_id}', color='blue')
        ax.fill(angles, region1_features, alpha=0.25, color='blue')

        ax.plot(angles, region2_features, 's-', linewidth=2, label=f'Region {region2_id}', color='red')
        ax.fill(angles, region2_features, alpha=0.25, color='red')

        # 设置角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, size=8)

        # 设置标题和图例
        ax.set_title(f'Morphological Features Comparison\nRegion {region1_id} vs Region {region2_id}',
                     size=14, pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图B保存到: {save_path}")

        plt.show()

        return fig

    def plot_subclass_comparison(self, region_pairs, top_n=20, save_path=None):
        """
        绘制图C: subclass组成对比条形图
        """
        if region_pairs is None or len(region_pairs) == 0:
            print("错误: 没有region对")
            return

        # 选择第一对region
        pair = region_pairs.iloc[0]
        region1_id = pair['region1']
        region2_id = pair['region2']

        # 获取两个region的subclass组成
        region1_subclass = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region1_id]
        region2_subclass = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region2_id]

        # 合并数据
        region1_dict = dict(zip(region1_subclass[':END_ID(Subclass)'],
                                region1_subclass['pct_cells:float']))
        region2_dict = dict(zip(region2_subclass[':END_ID(Subclass)'],
                                region2_subclass['pct_cells:float']))

        # 获取所有subclass
        all_subclasses = set(region1_dict.keys()) | set(region2_dict.keys())

        # 创建对比数据
        comparison_data = []
        for subclass in all_subclasses:
            comparison_data.append({
                'subclass': f'SC_{subclass}',  # 简化subclass名称
                'region1': region1_dict.get(subclass, 0),
                'region2': region2_dict.get(subclass, 0),
                'diff': abs(region1_dict.get(subclass, 0) - region2_dict.get(subclass, 0))
            })

        # 转换为DataFrame并排序
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.nlargest(top_n, 'diff')

        # 创建条形图
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(comparison_df))
        width = 0.35

        bars1 = ax.bar(x - width / 2, comparison_df['region1'], width,
                       label=f'Region {region1_id}', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, comparison_df['region2'], width,
                       label=f'Region {region2_id}', color='lightcoral', alpha=0.8)

        # 设置标签
        ax.set_xlabel('Subclass', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title(f'Top {top_n} Subclass Composition Comparison\nRegion {region1_id} vs Region {region2_id}',
                     fontsize=14, pad=20)

        # 设置x轴标签（旋转以避免重叠）
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['subclass'], rotation=45, ha='right')

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # 调整布局避免标签被截断
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图C保存到: {save_path}")

        plt.show()

        return fig

    def plot_go_analysis_placeholder(self, region_pairs, save_path=None):
        """
        绘制图D: GO分析占位图
        注意：这是一个占位符，实际的GO分析需要额外的基因本体数据
        """
        if region_pairs is None or len(region_pairs) == 0:
            print("错误: 没有region对")
            return

        # 选择第一对region
        pair = region_pairs.iloc[0]
        region1_id = pair['region1']
        region2_id = pair['region2']

        # 模拟GO terms数据（实际应用中需要真实的GO分析结果）
        go_categories = ['Biological Process', 'Molecular Function', 'Cellular Component']

        # 模拟一些GO terms
        go_terms = {
            'Biological Process': [
                'synaptic transmission', 'neuron development', 'axon guidance',
                'dendrite morphogenesis', 'neurotransmitter secretion'
            ],
            'Molecular Function': [
                'ion channel activity', 'neurotransmitter binding', 'protein kinase activity',
                'transcription factor activity', 'receptor activity'
            ],
            'Cellular Component': [
                'synapse', 'axon', 'dendrite', 'cell body', 'synaptic vesicle'
            ]
        }

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (category, terms) in enumerate(go_terms.items()):
            ax = axes[idx]

            # 模拟富集分数
            np.random.seed(idx + 42)  # 保证可重现
            region1_scores = np.random.uniform(0.2, 1.0, len(terms))
            region2_scores = np.random.uniform(0.2, 1.0, len(terms))

            y_pos = np.arange(len(terms))

            # 创建水平条形图
            ax.barh(y_pos - 0.2, region1_scores, 0.4,
                    label=f'Region {region1_id}', color='steelblue', alpha=0.7)
            ax.barh(y_pos + 0.2, region2_scores, 0.4,
                    label=f'Region {region2_id}', color='darkorange', alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(terms, fontsize=9)
            ax.set_xlabel('Enrichment Score', fontsize=10)
            ax.set_title(category, fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')

            # 设置x轴范围
            ax.set_xlim(0, 1.2)

        fig.suptitle(f'GO Analysis Comparison (Placeholder)\nRegion {region1_id} vs Region {region2_id}',
                     fontsize=14, y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图D保存到: {save_path}")

        plt.show()

        print("注意: 这是GO分析的占位图。实际GO分析需要基因本体数据库和富集分析工具。")

        return fig


def main():
    """主函数：执行完整的分析流程"""

    # 设置数据路径
    data_dir = "./knowledge_graph_v5"  # 请根据实际情况修改路径
    output_dir = Path("./analysis_figures")
    output_dir.mkdir(exist_ok=True)

    # 创建分析器
    analyzer = BrainRegionAnalyzer(data_dir)

    # 1. 加载数据
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    if not analyzer.load_data():
        print("数据加载失败，请检查文件路径")
        return

    # 2. 准备特征矩阵
    print("\n" + "=" * 60)
    print("步骤2: 准备特征矩阵")
    print("=" * 60)

    if not analyzer.prepare_morphological_features():
        print("形态学特征准备失败")
        return

    if not analyzer.prepare_molecular_features():
        print("分子组成特征准备失败")
        return

    # 3. 计算距离
    print("\n" + "=" * 60)
    print("步骤3: 计算距离矩阵")
    print("=" * 60)

    if not analyzer.calculate_distances():
        print("距离计算失败")
        return

    # 4. 绘制图A：距离散点图
    print("\n" + "=" * 60)
    print("步骤4: 绘制图A - 形态vs分子距离散点图")
    print("=" * 60)

    analyzer.plot_distance_scatter(save_path=output_dir / "figure_A_distance_scatter.png")

    # 5. 找出有趣的region对
    print("\n" + "=" * 60)
    print("步骤5: 寻找有趣的region对")
    print("=" * 60)

    type1_pairs, type2_pairs = analyzer.find_interesting_pairs(n_pairs=3)

    # 6. 绘制图B：形态差异雷达图
    print("\n" + "=" * 60)
    print("步骤6: 绘制图B - 形态差异雷达图")
    print("=" * 60)

    if type1_pairs is not None and len(type1_pairs) > 0:
        print("\n绘制形态相似但分子差异大的region对:")
        analyzer.plot_morphology_radar(type1_pairs,
                                       save_path=output_dir / "figure_B_morphology_radar_type1.png")

    if type2_pairs is not None and len(type2_pairs) > 0:
        print("\n绘制形态差异大但分子相似的region对:")
        analyzer.plot_morphology_radar(type2_pairs,
                                       save_path=output_dir / "figure_B_morphology_radar_type2.png")

    # 7. 绘制图C：subclass组成对比
    print("\n" + "=" * 60)
    print("步骤7: 绘制图C - Subclass组成对比条形图")
    print("=" * 60)

    if type1_pairs is not None and len(type1_pairs) > 0:
        print("\n绘制形态相似但分子差异大的region对:")
        analyzer.plot_subclass_comparison(type1_pairs,
                                          save_path=output_dir / "figure_C_subclass_comparison_type1.png")

    if type2_pairs is not None and len(type2_pairs) > 0:
        print("\n绘制形态差异大但分子相似的region对:")
        analyzer.plot_subclass_comparison(type2_pairs,
                                          save_path=output_dir / "figure_C_subclass_comparison_type2.png")

    # 8. 绘制图D：GO分析（占位符）
    print("\n" + "=" * 60)
    print("步骤8: 绘制图D - GO分析")
    print("=" * 60)

    if type1_pairs is not None and len(type1_pairs) > 0:
        analyzer.plot_go_analysis_placeholder(type1_pairs,
                                              save_path=output_dir / "figure_D_go_analysis.png")

    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"所有图表已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()