"""
完整版：MERFISH细胞映射 + 统计分析 + 可视化

功能：
1. 完整的MERFISH数据加载和区域映射
2. 细胞类型组成统计
3. 分子签名可视化（热图）
4. 区域细胞分布分析

作者: wangmajortom & Claude
日期: 2025-10-29
"""

import json
import pandas as pd
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

logger.add("merfish_complete_analysis.log", rotation="10 MB")

# 设置绘图样式
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
sns.set_style("whitegrid")


class CompleteMERFISHAnalyzer:
    """完整的MERFISH分析器：映射 + 统计 + 可视化"""

    def __init__(self, data_path: Path):
        self.data_path = data_path

        # CCF树映射
        self.id_to_acronym = {}
        self.id_to_name = {}
        self.acronym_to_id = {}

        # Annotation
        self.annotation = None

        # MERFISH数据
        self.merfish_cells = None

        # 文件名
        self.coord_files = [
            "ccf_coordinates_1.csv",
            "ccf_coordinates_2.csv",
            "ccf_coordinates_3.csv",
            "ccf_coordinates_4.csv"
        ]

        self.metadata_files = [
            "cell_metadata_with_cluster_annotation_1.csv",
            "cell_metadata_with_cluster_annotation_2.csv",
            "cell_metadata_with_cluster_annotation_3.csv",
            "cell_metadata_with_cluster_annotation_4.csv"
        ]

    def load_ccf_tree(self) -> bool:
        """加载CCF树"""
        logger.info("=" * 80)
        logger.info("Step 1: 加载CCF树")
        logger.info("=" * 80)

        tree_file = self.data_path / "tree_yzx.json"
        logger.info(f"  文件: {tree_file}")

        if not tree_file.exists():
            logger.error(f"  ✗ 文件不存在")
            return False

        with open(tree_file, 'r') as f:
            tree_data = json.load(f)

        logger.info(f"  ✓ 加载了 {len(tree_data)} 个节点")

        for node in tree_data:
            region_id = node.get('id')
            acronym = node.get('acronym', '')
            name = node.get('name', '')

            if region_id and acronym:
                self.id_to_acronym[region_id] = acronym
                self.id_to_name[region_id] = name
                self.acronym_to_id[acronym] = region_id

        logger.info(f"  ✓ 构建了 {len(self.id_to_acronym)} 个区域映射")

        return True

    def load_annotation(self) -> bool:
        """加载Annotation volume"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: 加载Annotation Volume")
        logger.info("=" * 80)

        annotation_file = self.data_path / "annotation_25.nrrd"
        logger.info(f"  文件: {annotation_file}")

        if not annotation_file.exists():
            logger.error(f"  ✗ 文件不存在")
            return False

        self.annotation, header = nrrd.read(str(annotation_file))

        logger.info(f"  ✓ Shape: {self.annotation.shape}")
        logger.info(f"  ✓ 数据类型: {self.annotation.dtype}")

        unique_ids = np.unique(self.annotation)
        logger.info(f"  ✓ 包含 {len(unique_ids)} 个不同的region ID")

        return True

    def load_merfish_data(self) -> bool:
        """加载MERFISH数据（坐标×40转换）"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: 加载MERFISH数据")
        logger.info("=" * 80)

        # 加载坐标
        logger.info("\n  3.1 加载坐标数据...")
        coordinate_dfs = []

        for coord_file in self.coord_files:
            file_path = self.data_path / coord_file
            if file_path.exists():
                logger.info(f"    加载: {coord_file}")
                df = pd.read_csv(file_path)

                if 'x_ccf' not in df.columns and 'x' in df.columns:
                    df = df.rename(columns={'x': 'x_ccf', 'y': 'y_ccf', 'z': 'z_ccf'})

                # 坐标×40转换
                for col in ['x_ccf', 'y_ccf', 'z_ccf']:
                    if col in df.columns:
                        df[col] = df[col] * 40

                df['source_file'] = coord_file
                coordinate_dfs.append(df)
                logger.info(f"      ✓ {len(df):,} 个细胞")

        if not coordinate_dfs:
            logger.error("  ✗ 没有找到坐标文件")
            return False

        coordinates_df = pd.concat(coordinate_dfs, ignore_index=True)
        logger.info(f"  ✓ 总共 {len(coordinates_df):,} 个细胞坐标")

        # 确保cell_label
        if 'cell_label' not in coordinates_df.columns:
            if 'id' in coordinates_df.columns:
                coordinates_df['cell_label'] = coordinates_df['id']
            else:
                coordinates_df['cell_label'] = [f"cell_{i}" for i in range(len(coordinates_df))]

        # 加载元数据
        logger.info("\n  3.2 加载元数据...")
        metadata_dfs = []

        for meta_file in self.metadata_files:
            file_path = self.data_path / meta_file
            if file_path.exists():
                logger.info(f"    加载: {meta_file}")
                df = pd.read_csv(file_path)
                df['source_file'] = meta_file
                metadata_dfs.append(df)
                logger.info(f"      ✓ {len(df):,} 行")

        if metadata_dfs:
            metadata_df = pd.concat(metadata_dfs, ignore_index=True)
            logger.info(f"  ✓ 总共 {len(metadata_df):,} 行元数据")

            if 'cell_label' not in metadata_df.columns:
                if 'id' in metadata_df.columns:
                    metadata_df['cell_label'] = metadata_df['id']
        else:
            metadata_df = pd.DataFrame({'cell_label': coordinates_df['cell_label']})

        # 合并
        logger.info("\n  3.3 合并数据...")
        self.merfish_cells = pd.merge(
            coordinates_df,
            metadata_df,
            on='cell_label',
            how='left',
            suffixes=('', '_meta')
        )

        duplicate_cols = [col for col in self.merfish_cells.columns if col.endswith('_meta')]
        if duplicate_cols:
            self.merfish_cells = self.merfish_cells.drop(columns=duplicate_cols)

        logger.info(f"  ✓ 合并后 {len(self.merfish_cells):,} 个细胞")

        # 质量检查
        valid_coords = (
            self.merfish_cells['x_ccf'].notna() &
            self.merfish_cells['y_ccf'].notna() &
            self.merfish_cells['z_ccf'].notna()
        )
        n_valid = valid_coords.sum()

        if n_valid < len(self.merfish_cells):
            logger.warning(f"  ⚠ 移除 {len(self.merfish_cells) - n_valid:,} 个无效坐标")
            self.merfish_cells = self.merfish_cells[valid_coords].copy()

        logger.info(f"  ✓ 最终 {len(self.merfish_cells):,} 个有效细胞")

        return True

    def map_cells_to_regions(self) -> bool:
        """映射细胞到脑区"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: 映射细胞到脑区")
        logger.info("=" * 80)

        if self.merfish_cells is None or self.annotation is None:
            logger.error("  ✗ 数据未加载")
            return False

        n_cells = len(self.merfish_cells)
        logger.info(f"  处理 {n_cells:,} 个细胞...")

        # 坐标转体素索引（坐标×40后直接就是索引）
        x_idx = np.round(self.merfish_cells['x_ccf'].values).astype(int)
        y_idx = np.round(self.merfish_cells['y_ccf'].values).astype(int)
        z_idx = np.round(self.merfish_cells['z_ccf'].values).astype(int)

        # 限制边界
        shape = self.annotation.shape
        x_idx = np.clip(x_idx, 0, shape[0] - 1)
        y_idx = np.clip(y_idx, 0, shape[1] - 1)
        z_idx = np.clip(z_idx, 0, shape[2] - 1)

        logger.info(f"  ✓ 坐标范围: X[{x_idx.min()}-{x_idx.max()}] Y[{y_idx.min()}-{y_idx.max()}] Z[{z_idx.min()}-{z_idx.max()}]")

        # 查询region ID
        region_ids = self.annotation[x_idx, y_idx, z_idx]

        # 查询名称
        acronyms = np.array([self.id_to_acronym.get(rid, 'Unknown') for rid in region_ids])
        names = np.array([self.id_to_name.get(rid, 'Unknown') for rid in region_ids])

        # 添加到DataFrame
        self.merfish_cells['region_id'] = region_ids
        self.merfish_cells['region_acronym'] = acronyms
        self.merfish_cells['region_name'] = names

        # 统计
        n_mapped = ((region_ids > 0) & (acronyms != 'Unknown')).sum()
        logger.info(f"  ✓ 成功映射: {n_mapped:,} / {n_cells:,} ({n_mapped/n_cells*100:.2f}%)")

        return True

    def compute_region_celltype_composition(
        self,
        target_regions: List[str]
    ) -> pd.DataFrame:
        """计算目标区域的细胞类型组成"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: 计算细胞类型组成")
        logger.info("=" * 80)

        if 'subclass' not in self.merfish_cells.columns:
            logger.error("  ✗ 数据中没有subclass列")
            return pd.DataFrame()

        # 过滤目标区域
        target_cells = self.merfish_cells[
            self.merfish_cells['region_acronym'].isin(target_regions)
        ]

        logger.info(f"  找到 {len(target_cells):,} 个目标区域的细胞")

        if len(target_cells) == 0:
            logger.warning("  ⚠ 目标区域没有细胞")
            return pd.DataFrame()

        # 统计每个区域的细胞类型
        compositions = []

        for region in target_regions:
            region_cells = target_cells[target_cells['region_acronym'] == region]

            if len(region_cells) == 0:
                continue

            # 统计subclass
            subclass_counts = region_cells['subclass'].value_counts()
            total = len(region_cells)

            # Top subclasses
            for subclass, count in subclass_counts.head(10).items():
                pct = count / total * 100

                compositions.append({
                    'region': region,
                    'subclass': subclass,
                    'cell_count': count,
                    'percentage': pct,
                    'total_cells': total
                })

        comp_df = pd.DataFrame(compositions)

        logger.info(f"  ✓ 生成了 {len(comp_df)} 条组成记录")
        logger.info(f"  ✓ 覆盖 {comp_df['region'].nunique()} 个区域")

        return comp_df

    def plot_celltype_composition_heatmap(
        self,
        comp_df: pd.DataFrame,
        output_file: Path,
        top_n: int = 10
    ):
        """绘制细胞类型组成热图"""
        logger.info("\n绘制细胞类型组成热图...")

        # 为每个区域选择top N的subclass
        plot_data = []

        for region in comp_df['region'].unique():
            region_data = comp_df[comp_df['region'] == region]
            region_data = region_data.sort_values('percentage', ascending=False).head(top_n)
            plot_data.append(region_data)

        plot_df = pd.concat(plot_data, ignore_index=True)

        # Pivot为热图格式
        heatmap_data = plot_df.pivot(
            index='region',
            columns='subclass',
            values='percentage'
        ).fillna(0)

        # 创建图形
        fig, ax = plt.subplots(figsize=(16, max(10, len(heatmap_data) * 0.5)))

        # 绘制热图
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': '% of Cells'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        # 标题
        ax.set_title(
            f'Cell Type Composition of Target Brain Regions\n(Top {top_n} Subclasses per Region)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Cell Subclass', fontsize=12, fontweight='bold')
        ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ 保存到: {output_file}")

        plt.close()

    def plot_region_cellcount_barplot(
        self,
        target_regions: List[str],
        output_file: Path
    ):
        """绘制各区域细胞数柱状图"""
        logger.info("\n绘制区域细胞数柱状图...")

        target_cells = self.merfish_cells[
            self.merfish_cells['region_acronym'].isin(target_regions)
        ]

        region_counts = target_cells['region_acronym'].value_counts()

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制柱状图
        regions = region_counts.index
        counts = region_counts.values
        colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))

        bars = ax.bar(range(len(regions)), counts, color=colors)

        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(
                i, count,
                f'{count:,}',
                ha='center', va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # 设置标签
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax.set_title(
            'Cell Distribution Across Target Regions',
            fontsize=14,
            fontweight='bold'
        )

        # 网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ 保存到: {output_file}")

        plt.close()

    def plot_subclass_distribution_stacked(
        self,
        comp_df: pd.DataFrame,
        output_file: Path,
        top_n_subclasses: int = 10
    ):
        """绘制堆叠柱状图显示细胞类型分布"""
        logger.info("\n绘制堆叠柱状图...")

        # 选择top N的subclass
        top_subclasses = comp_df.groupby('subclass')['cell_count'].sum().nlargest(top_n_subclasses).index

        # 过滤数据
        plot_df = comp_df[comp_df['subclass'].isin(top_subclasses)]

        # Pivot
        pivot_df = plot_df.pivot(
            index='region',
            columns='subclass',
            values='percentage'
        ).fillna(0)

        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 8))

        # 堆叠柱状图
        pivot_df.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            colormap='tab20',
            width=0.8
        )

        # 标题和标签
        ax.set_title(
            f'Cell Type Distribution Across Regions\n(Top {top_n_subclasses} Subclasses)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Cells (%)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # 图例
        ax.legend(
            title='Subclass',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True
        )

        # 网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ 保存到: {output_file}")

        plt.close()

    def generate_analysis_report(
        self,
        comp_df: pd.DataFrame,
        target_regions: List[str]
    ) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append("MERFISH细胞区域映射与组成分析报告")
        report.append("=" * 80)
        report.append(f"\n总细胞数: {len(self.merfish_cells):,}")
        report.append(f"目标区域数: {len(target_regions)}")

        # 映射统计
        mapped = (
            (self.merfish_cells['region_id'] > 0) &
            (self.merfish_cells['region_acronym'] != 'Unknown')
        ).sum()
        report.append(f"成功映射细胞数: {mapped:,} ({mapped/len(self.merfish_cells)*100:.2f}%)")

        # 目标区域统计
        target_cells = self.merfish_cells[
            self.merfish_cells['region_acronym'].isin(target_regions)
        ]
        report.append(f"目标区域细胞数: {len(target_cells):,}")

        report.append("\n各区域细胞数:")
        region_counts = target_cells['region_acronym'].value_counts()
        for region, count in region_counts.items():
            report.append(f"  {region}: {count:,} cells")

        # 细胞类型统计
        if not comp_df.empty:
            report.append("\n主要细胞类型 (Top 10):")
            top_subclasses = comp_df.groupby('subclass')['cell_count'].sum().nlargest(10)
            for subclass, count in top_subclasses.items():
                report.append(f"  {subclass}: {count:,} cells")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_results(self, output_path: Path):
        """保存所有结果"""
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: 保存结果")
        logger.info("=" * 80)

        # 完整数据
        output_file = output_path / "merfish_mapped_complete.csv"

        if len(self.merfish_cells) > 1000000:
            self.merfish_cells.head(1000000).to_csv(output_file, index=False)
            logger.info(f"  ✓ 保存前100万行样本到: {output_file}")
        else:
            self.merfish_cells.to_csv(output_file, index=False)
            logger.info(f"  ✓ 保存完整数据到: {output_file}")

        # 区域统计
        stats_file = output_path / "region_statistics.csv"

        valid_cells = self.merfish_cells[
            (self.merfish_cells['region_id'] > 0) &
            (self.merfish_cells['region_acronym'] != 'Unknown')
        ]

        region_stats = valid_cells.groupby(['region_id', 'region_acronym', 'region_name']).size()
        region_stats = region_stats.reset_index(name='cell_count')
        region_stats = region_stats.sort_values('cell_count', ascending=False)

        region_stats.to_csv(stats_file, index=False)
        logger.info(f"  ✓ 区域统计已保存到: {stats_file}")


def main():
    """主函数"""

    # 路径配置
    data_path = Path("/home/wlj/NeuroXiv2/data")
    output_path = Path("./")

    logger.info("=" * 80)
    logger.info("MERFISH完整分析：映射 + 统计 + 可视化")
    logger.info("=" * 80)

    try:
        # 创建分析器
        analyzer = CompleteMERFISHAnalyzer(data_path)

        # Step 1-4: 数据加载和映射
        if not analyzer.load_ccf_tree():
            return

        if not analyzer.load_annotation():
            return

        if not analyzer.load_merfish_data():
            return

        if not analyzer.map_cells_to_regions():
            return

        # 定义目标区域
        target_regions = [
            'ACAd2/3',
            'ACAd5',
            'ACAd6a',
            'ACAv5',
            'AId6a',
            'AIp6a',
            'CP',
            'ENTl2',
            'ENTl5',
            'ENTl6a',
            'EPd',
            'MOp6a',
            'MOs2/3',
            'MOs5',
            'MOs6a',
            'RSPv5',
            'SNr',
        ]

        logger.info(f"\n目标区域: {target_regions}")

        # Step 5: 计算细胞类型组成
        comp_df = analyzer.compute_region_celltype_composition(target_regions)

        if not comp_df.empty:
            # 保存组成数据
            comp_file = output_path / "region_celltype_composition.csv"
            comp_df.to_csv(comp_file, index=False)
            logger.info(f"\n✓ 细胞类型组成已保存到: {comp_file}")

            # 绘图
            logger.info("\n" + "=" * 80)
            logger.info("生成可视化图表")
            logger.info("=" * 80)

            # 热图
            heatmap_file = output_path / "celltype_composition_heatmap.png"
            analyzer.plot_celltype_composition_heatmap(comp_df, heatmap_file, top_n=10)

            # 细胞数柱状图
            barplot_file = output_path / "region_cellcount_barplot.png"
            analyzer.plot_region_cellcount_barplot(target_regions, barplot_file)

            # 堆叠柱状图
            stacked_file = output_path / "subclass_distribution_stacked.png"
            analyzer.plot_subclass_distribution_stacked(comp_df, stacked_file, top_n_subclasses=10)

            # 生成报告
            report = analyzer.generate_analysis_report(comp_df, target_regions)
            print("\n" + report)

            report_file = output_path / "analysis_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"\n✓ 报告已保存到: {report_file}")

        # 保存结果
        analyzer.save_results(output_path)

        logger.info("\n" + "=" * 80)
        logger.info("✓ 全部完成！")
        logger.info("=" * 80)
        logger.info("\n输出文件:")
        logger.info("  1. merfish_mapped_complete.csv - 完整映射数据")
        logger.info("  2. region_statistics.csv - 区域统计")
        logger.info("  3. region_celltype_composition.csv - 细胞类型组成")
        logger.info("  4. celltype_composition_heatmap.png - 组成热图")
        logger.info("  5. region_cellcount_barplot.png - 细胞数柱状图")
        logger.info("  6. subclass_distribution_stacked.png - 堆叠分布图")
        logger.info("  7. analysis_report.txt - 分析报告")
        logger.info("  8. merfish_complete_analysis.log - 详细日志")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()