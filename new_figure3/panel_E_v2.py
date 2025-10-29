"""
完整的MERFISH细胞类型分析流程 - 改进版v2.1
===============================================

改进内容：
1. ✅ 横轴按照生物解剖学远近排序
2. ✅ 确保全部20个目标脑区清晰标记（不遗漏）
3. ✅ 为每个broadcast axis group统计完整的marker基因列表
4. ✅ 更美观的可视化效果

作者: Claude
日期: 2025-10-29
"""

import json
import numpy as np
import pandas as pd
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


class CompleteMERFISHAnalyzerImproved:
    """完整的MERFISH数据分析器 - 改进版"""

    def __init__(self):
        self.merfish_cells = None
        self.annotation = None
        self.ccf_tree = None
        self.hierarchy_data = None

        # 解剖学分组定义（按照距离CLA的远近排序）
        self.anatomical_groups = {
            'Prefrontal/Motor': {
                'regions': [
                    'ACAd5', 'ACAd6a', 'ACAd6b',
                    'MOs5', 'MOs6a', 'MOs6b',
                    'MOp5', 'MOp6a', 'MOp6b'
                ],
                'order': 1,
                'color': '#E74C3C',  # 红色
                'description': '前额叶和运动皮层 - 最近'
            },
            'Orbital': {
                'regions': [
                    'ORBl5', 'ORBl6a', 'ORBl6b',
                    'ORBm5', 'ORBm6a', 'ORBm6b'
                ],
                'order': 2,
                'color': '#F39C12',  # 橙色
                'description': '眶额皮层 - 较近'
            },
            'Entorhinal/Retrosplenial': {
                'regions': [
                    'ENTl6a', 'ENTm6',
                    'RSPv5', 'RSPd5', 'RSPd6a'
                ],
                'order': 3,
                'color': '#27AE60',  # 绿色
                'description': '内嗅和回顾皮层 - 较远'
            },
            'Striatal/Basal Ganglia': {
                'regions': [
                    'CP', 'ACB', 'SNr', 'VTA'
                ],
                'order': 4,
                'color': '#3498DB',  # 蓝色
                'description': '纹状体和基底神经节 - 最远'
            }
        }

    def load_merfish_data(self, csv_path: Path) -> bool:
        """加载MERFISH单细胞数据"""
        logger.info(f"加载MERFISH数据: {csv_path}")

        try:
            self.merfish_cells = pd.read_csv(csv_path)
            logger.info(f"成功加载 {len(self.merfish_cells):,} 个细胞")

            # 检查必需的列
            required_cols = ['x', 'y', 'z', 'subclass']
            missing = [c for c in required_cols if c not in self.merfish_cells.columns]
            if missing:
                logger.error(f"缺少必需的列: {missing}")
                return False

            logger.info(f"数据范围: X[{self.merfish_cells['x'].min():.2f}, {self.merfish_cells['x'].max():.2f}] mm")
            logger.info(f"         Y[{self.merfish_cells['y'].min():.2f}, {self.merfish_cells['y'].max():.2f}] mm")
            logger.info(f"         Z[{self.merfish_cells['z'].min():.2f}, {self.merfish_cells['z'].max():.2f}] mm")

            return True

        except Exception as e:
            logger.error(f"加载MERFISH数据失败: {e}")
            return False

    def load_annotation(self, nrrd_path: Path) -> bool:
        """加载CCF注释volume"""
        logger.info(f"加载注释数据: {nrrd_path}")

        try:
            self.annotation, header = nrrd.read(str(nrrd_path))
            logger.info(f"注释shape: {self.annotation.shape}")
            logger.info(f"注释dtype: {self.annotation.dtype}")
            logger.info(f"唯一区域数: {len(np.unique(self.annotation))}")

            return True

        except Exception as e:
            logger.error(f"加载注释数据失败: {e}")
            return False

    def load_ccf_tree(self, json_path: Path) -> bool:
        """加载CCF树结构"""
        logger.info(f"加载CCF树: {json_path}")

        try:
            with open(json_path, 'r') as f:
                tree_data = json.load(f)

            # 构建ID到信息的映射
            self.ccf_tree = {}

            def parse_node(node):
                node_id = node.get('id')
                if node_id:
                    self.ccf_tree[node_id] = {
                        'acronym': node.get('acronym', ''),
                        'name': node.get('name', ''),
                        'id': node_id
                    }

                for child in node.get('children', []):
                    parse_node(child)

            if isinstance(tree_data, dict):
                parse_node(tree_data)
            elif isinstance(tree_data, list):
                for item in tree_data:
                    parse_node(item)

            logger.info(f"CCF树包含 {len(self.ccf_tree)} 个区域")
            return True

        except Exception as e:
            logger.error(f"加载CCF树失败: {e}")
            return False

    def load_hierarchy_data(self, json_path: Path) -> bool:
        """加载细胞类型层级数据（包含marker基因）"""
        logger.info(f"加载细胞类型层级数据: {json_path}")

        try:
            with open(json_path, 'r') as f:
                self.hierarchy_data = json.load(f)

            logger.info("成功加载细胞类型层级数据")

            # 检查数据结构
            if 'subclass' in self.hierarchy_data:
                logger.info(f"包含 {len(self.hierarchy_data['subclass'])} 个subclass")

            return True

        except Exception as e:
            logger.error(f"加载层级数据失败: {e}")
            return False

    def map_cells_to_regions(self) -> bool:
        """将细胞坐标映射到CCF区域"""
        logger.info("开始映射细胞到CCF区域...")

        if self.merfish_cells is None or self.annotation is None:
            logger.error("MERFISH数据或注释数据未加载")
            return False

        # 坐标转换: mm → voxel indices
        # 关键: mm坐标 * 40 得到25μm分辨率的voxel索引
        voxel_x = (self.merfish_cells['x'] * 40).round().astype(int)
        voxel_y = (self.merfish_cells['y'] * 40).round().astype(int)
        voxel_z = (self.merfish_cells['z'] * 40).round().astype(int)

        # 边界检查
        valid_mask = (
                (voxel_x >= 0) & (voxel_x < self.annotation.shape[2]) &
                (voxel_y >= 0) & (voxel_y < self.annotation.shape[1]) &
                (voxel_z >= 0) & (voxel_z < self.annotation.shape[0])
        )

        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logger.warning(f"{n_invalid:,} 个细胞超出注释边界")

        # 查询注释volume获取region ID
        region_ids = np.full(len(self.merfish_cells), -1, dtype=int)
        region_ids[valid_mask] = self.annotation[
            voxel_z[valid_mask],
            voxel_y[valid_mask],
            voxel_x[valid_mask]
        ]

        # 使用CCF树查找acronym
        region_acronyms = []
        for region_id in region_ids:
            if region_id > 0 and region_id in self.ccf_tree:
                region_acronyms.append(self.ccf_tree[region_id]['acronym'])
            else:
                region_acronyms.append('Unknown')

        self.merfish_cells['region_id'] = region_ids
        self.merfish_cells['region_acronym'] = region_acronyms

        # 统计
        n_mapped = (region_ids > 0).sum()
        logger.info(
            f"成功映射 {n_mapped:,} / {len(self.merfish_cells):,} 个细胞 ({100 * n_mapped / len(self.merfish_cells):.1f}%)")

        unique_regions = self.merfish_cells[self.merfish_cells['region_id'] > 0]['region_acronym'].nunique()
        logger.info(f"细胞分布在 {unique_regions} 个不同区域")

        return True

    def extract_subclass_markers(self, subclass_name: str) -> List[str]:
        """提取subclass的marker基因列表"""
        if not self.hierarchy_data or 'subclass' not in self.hierarchy_data:
            return []

        markers = []
        for subclass in self.hierarchy_data['subclass']:
            if subclass.get('name') == subclass_name or subclass.get('alias') == subclass_name:
                # 提取marker genes
                marker_list = subclass.get('genes', [])
                if isinstance(marker_list, list):
                    markers.extend([m for m in marker_list if m])
                elif isinstance(marker_list, str):
                    markers.append(marker_list)

                # 也尝试从其他字段获取
                for key in ['marker_genes', 'markers', 'gene_symbols']:
                    if key in subclass:
                        additional = subclass[key]
                        if isinstance(additional, list):
                            markers.extend([m for m in additional if m])
                        elif isinstance(additional, str):
                            markers.append(additional)

                break

        # 去重并返回
        return list(set(markers)) if markers else []

    def analyze_region_composition(self, target_regions: List[str]) -> pd.DataFrame:
        """分析目标区域的细胞类型组成"""
        logger.info(f"分析 {len(target_regions)} 个目标区域的细胞类型组成...")

        compositions = []

        for region in target_regions:
            # 筛选该区域的细胞
            region_cells = self.merfish_cells[
                self.merfish_cells['region_acronym'] == region
                ].copy()

            if len(region_cells) == 0:
                logger.warning(f"区域 {region} 没有找到细胞")
                continue

            # 统计每个subclass的细胞数
            subclass_counts = region_cells['subclass'].value_counts()
            total_cells = len(region_cells)

            for subclass, count in subclass_counts.items():
                compositions.append({
                    'region': region,
                    'subclass': subclass,
                    'cell_count': count,
                    'percentage': 100 * count / total_cells,
                    'total_region_cells': total_cells
                })

        comp_df = pd.DataFrame(compositions)

        if len(comp_df) > 0:
            logger.info(f"分析完成: {len(comp_df)} 条记录，覆盖 {comp_df['region'].nunique()} 个区域")
        else:
            logger.warning("未生成任何组成数据")

        return comp_df

    def get_sorted_regions_by_anatomy(self, regions: List[str]) -> List[str]:
        """按照解剖学分组顺序排序区域"""
        # 创建区域到分组的映射
        region_to_group = {}
        region_to_order = {}

        for group_name, group_info in self.anatomical_groups.items():
            order = group_info['order']
            for i, region in enumerate(group_info['regions']):
                region_to_group[region] = group_name
                # order*1000 + i 确保同组内也有顺序
                region_to_order[region] = order * 1000 + i

        # 分类：在定义中的区域和未定义的区域
        defined_regions = [r for r in regions if r in region_to_order]
        undefined_regions = [r for r in regions if r not in region_to_order]

        # 已定义的按照解剖学顺序排序
        defined_sorted = sorted(defined_regions, key=lambda r: region_to_order[r])

        # 未定义的按字母顺序排在最后
        undefined_sorted = sorted(undefined_regions)

        return defined_sorted + undefined_sorted

    def analyze_anatomical_groups(self, comp_df: pd.DataFrame) -> pd.DataFrame:
        """分析每个解剖学分组的细胞类型组成和marker基因"""
        logger.info("分析解剖学分组的特征...")

        group_results = []

        for group_name, group_info in sorted(
                self.anatomical_groups.items(),
                key=lambda x: x[1]['order']
        ):
            regions = group_info['regions']

            # 筛选该分组的所有细胞
            group_data = comp_df[comp_df['region'].isin(regions)]

            if len(group_data) == 0:
                logger.warning(f"分组 {group_name} 没有数据")
                continue

            # 聚合统计：哪个subclass在该分组中最富集
            subclass_totals = group_data.groupby('subclass').agg({
                'cell_count': 'sum',
                'percentage': 'mean'
            }).reset_index()

            subclass_totals = subclass_totals.sort_values('cell_count', ascending=False)

            # 获取top 1 subclass
            if len(subclass_totals) > 0:
                top_subclass = subclass_totals.iloc[0]
                enriched_subclass = top_subclass['subclass']
                total_cells = int(top_subclass['cell_count'])
                avg_percentage = top_subclass['percentage']

                # 提取该subclass的marker基因
                markers = self.extract_subclass_markers(enriched_subclass)

                group_results.append({
                    'group_name': group_name,
                    'order': group_info['order'],
                    'description': group_info['description'],
                    'regions': ', '.join(regions),
                    'n_regions': len(regions),
                    'enriched_subclass': enriched_subclass,
                    'total_cells': total_cells,
                    'avg_percentage': avg_percentage,
                    'markers': ', '.join(markers) if markers else 'N/A',
                    'n_markers': len(markers)
                })

        group_df = pd.DataFrame(group_results)
        group_df = group_df.sort_values('order')

        logger.info(f"生成了 {len(group_df)} 个分组的分析结果")

        return group_df

    def plot_celltype_composition_heatmap(
            self,
            comp_df: pd.DataFrame,
            output_file: Path,
            top_n: int = 10,
            figsize: Tuple[int, int] = (18, 12)
    ):
        """
        绘制细胞类型组成热图（改进版）

        改进：
        1. 横轴按照解剖学远近排序
        2. 确保所有区域都显示完整的Y轴标签
        3. 顶部添加彩色分组条
        """
        logger.info("绘制细胞类型组成热图（改进版）...")

        # 获取所有区域并按解剖学顺序排序
        all_regions = comp_df['region'].unique()
        sorted_regions = self.get_sorted_regions_by_anatomy(list(all_regions))

        logger.info(f"将绘制 {len(sorted_regions)} 个区域（按解剖学分组排序）")

        # 获取top N个subclass
        top_subclasses = (
            comp_df.groupby('subclass')['cell_count']
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )

        # 构建矩阵
        matrix_data = []
        for region in sorted_regions:
            region_data = comp_df[comp_df['region'] == region]

            row = []
            for subclass in top_subclasses:
                pct = region_data[region_data['subclass'] == subclass]['percentage']
                row.append(pct.values[0] if len(pct) > 0 else 0)

            matrix_data.append(row)

        matrix = np.array(matrix_data)

        # 创建图形
        fig = plt.figure(figsize=figsize)

        # 创建子图：顶部分组条 + 主热图
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[0.03, 0.97],
            hspace=0.02
        )

        ax_group = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        # 1. 绘制顶部分组条
        ax_group.set_xlim(0, len(sorted_regions))
        ax_group.set_ylim(0, 1)
        ax_group.axis('off')

        current_x = 0
        for group_name, group_info in sorted(
                self.anatomical_groups.items(),
                key=lambda x: x[1]['order']
        ):
            group_regions = [r for r in sorted_regions if r in group_info['regions']]
            width = len(group_regions)

            if width > 0:
                ax_group.add_patch(Rectangle(
                    (current_x, 0), width, 1,
                    facecolor=group_info['color'],
                    edgecolor='white',
                    linewidth=2
                ))

                # 添加分组标签
                ax_group.text(
                    current_x + width / 2, 0.5,
                    group_name,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white'
                )

                current_x += width

        # 2. 绘制主热图
        im = ax_heat.imshow(
            matrix,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )

        # 设置X轴（细胞类型）
        ax_heat.set_xticks(range(len(top_subclasses)))
        ax_heat.set_xticklabels(
            top_subclasses,
            rotation=45,
            ha='right',
            fontsize=9
        )

        # 设置Y轴（区域）- 确保所有标签都显示
        ax_heat.set_yticks(range(len(sorted_regions)))
        ax_heat.set_yticklabels(
            sorted_regions,
            fontsize=9,
            fontweight='bold'
        )

        # 确保Y轴标签不被截断
        ax_heat.yaxis.set_tick_params(pad=2)

        # 添加网格线
        ax_heat.set_xticks(np.arange(len(top_subclasses)) - 0.5, minor=True)
        ax_heat.set_yticks(np.arange(len(sorted_regions)) - 0.5, minor=True)
        ax_heat.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=11)

        # 标题
        ax_heat.set_title(
            'Cell Type Composition Across Target Regions\n(Sorted by Anatomical Distance from CLA)',
            fontsize=13,
            fontweight='bold',
            pad=20
        )

        ax_heat.set_xlabel('Cell Subclass (Top 10)', fontsize=11)
        ax_heat.set_ylabel('Target Regions', fontsize=11)

        # 保存
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"热图已保存: {output_file}")

    def generate_report(
            self,
            comp_df: pd.DataFrame,
            group_df: pd.DataFrame,
            output_file: Path
    ):
        """生成分析报告"""
        report_lines = [
            "=" * 80,
            "MERFISH细胞类型分析报告 - 改进版v2.1",
            "=" * 80,
            "",
            "## 总体统计",
            f"总细胞数: {len(self.merfish_cells):,}",
            f"已映射细胞数: {(self.merfish_cells['region_id'] > 0).sum():,}",
            f"分析区域数: {comp_df['region'].nunique()}",
            f"细胞类型数: {comp_df['subclass'].nunique()}",
            "",
            "## 解剖学分组分析",
            ""
        ]

        # 添加分组详情
        for _, row in group_df.iterrows():
            report_lines.extend([
                f"### {row['group_name']} ({row['description']})",
                f"  包含区域: {row['regions']}",
                f"  富集细胞类型: {row['enriched_subclass']}",
                f"  总细胞数: {row['total_cells']:,}",
                f"  平均占比: {row['avg_percentage']:.1f}%",
                f"  Marker基因数: {row['n_markers']}",
                f"  Marker基因: {row['markers'][:200]}{'...' if len(row['markers']) > 200 else ''}",
                ""
            ])

        # 每个区域的Top细胞类型
        report_lines.extend([
            "",
            "## 各区域Top 3细胞类型",
            ""
        ])

        for region in comp_df['region'].unique():
            region_data = comp_df[comp_df['region'] == region].nlargest(3, 'percentage')
            report_lines.append(f"### {region}")
            for _, row in region_data.iterrows():
                report_lines.append(
                    f"  {row['subclass']}: {row['percentage']:.1f}% ({row['cell_count']} cells)"
                )
            report_lines.append("")

        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"报告已保存: {output_file}")


def main():
    """主函数"""

    # 配置日志
    logger.add("merfish_analysis_improved.log", rotation="10 MB")

    # 数据路径
    data_dir = Path("/home/wlj/NeuroXiv2/data")
    output_dir = Path("./panel_E")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 初始化分析器
    analyzer = CompleteMERFISHAnalyzerImproved()

    # 1. 加载MERFISH数据
    merfish_file = data_dir / "WMB-10Xv3" / "20230830" / "WMB-10Xv3-TH-Whole_brain_coronal-log2.csv"
    if not analyzer.load_merfish_data(merfish_file):
        logger.error("无法加载MERFISH数据")
        return

    # 2. 加载注释volume
    annotation_file = data_dir / "annotation_25.nrrd"
    if not analyzer.load_annotation(annotation_file):
        logger.error("无法加载注释数据")
        return

    # 3. 加载CCF树
    ccf_tree_file = data_dir / "tree_yzx.json"
    if not analyzer.load_ccf_tree(ccf_tree_file):
        logger.error("无法加载CCF树")
        return

    # 4. 加载细胞类型层级数据（含marker基因）
    hierarchy_file = data_dir / "tran-data-type-tree.json"
    if not analyzer.load_hierarchy_data(hierarchy_file):
        logger.warning("无法加载层级数据，marker基因信息将不可用")

    # 5. 映射细胞到区域
    if not analyzer.map_cells_to_regions():
        logger.error("细胞映射失败")
        return

    # 6. 定义目标区域（全部20个）
    target_regions = [
        # Prefrontal/Motor (9个)
        'ACAd5', 'ACAd6a', 'ACAd6b',
        'MOs5', 'MOs6a', 'MOs6b',
        'MOp5', 'MOp6a', 'MOp6b',

        # Orbital (6个)
        'ORBl5', 'ORBl6a', 'ORBl6b',
        'ORBm5', 'ORBm6a', 'ORBm6b',

        # Entorhinal/Retrosplenial (5个)
        'ENTl6a', 'ENTm6',
        'RSPv5', 'RSPd5', 'RSPd6a',

        # Striatal/Basal Ganglia (4个)
        'CP', 'ACB', 'SNr', 'VTA'
    ]

    logger.info(f"目标区域总数: {len(target_regions)}")

    # 7. 分析区域组成
    comp_df = analyzer.analyze_region_composition(target_regions)

    if len(comp_df) == 0:
        logger.error("未生成组成数据")
        return

    # 保存详细数据
    comp_file = output_dir / "region_celltype_composition_improved.csv"
    comp_df.to_csv(comp_file, index=False)
    logger.info(f"组成数据已保存: {comp_file}")

    # 8. 分析解剖学分组
    group_df = analyzer.analyze_anatomical_groups(comp_df)

    if len(group_df) > 0:
        group_file = output_dir / "anatomical_groups_analysis_improved.csv"
        group_df.to_csv(group_file, index=False)
        logger.info(f"分组分析已保存: {group_file}")

        # 打印分组信息
        logger.info("\n解剖学分组Marker基因统计:")
        for _, row in group_df.iterrows():
            logger.info(f"\n{row['group_name']}:")
            logger.info(f"  区域: {row['regions']}")
            logger.info(f"  富集subclass: {row['enriched_subclass']}")
            logger.info(f"  Marker数量: {row['n_markers']}")
            logger.info(f"  Markers: {row['markers'][:100]}...")

    # 9. 绘制改进版热图
    heatmap_file = output_dir / "celltype_heatmap_improved.png"
    analyzer.plot_celltype_composition_heatmap(
        comp_df,
        heatmap_file,
        top_n=10,
        figsize=(20, 14)
    )

    # 10. 生成报告
    report_file = output_dir / "analysis_report_improved.txt"
    analyzer.generate_report(comp_df, group_df, report_file)

    logger.info("\n" + "=" * 80)
    logger.info("分析完成！")
    logger.info("=" * 80)
    logger.info(f"输出文件:")
    logger.info(f"  - 热图: {heatmap_file}")
    logger.info(f"  - 组成数据: {comp_file}")
    logger.info(f"  - 分组分析: {group_file}")
    logger.info(f"  - 报告: {report_file}")


if __name__ == "__main__":
    main()