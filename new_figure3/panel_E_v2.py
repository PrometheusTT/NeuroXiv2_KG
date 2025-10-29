"""
Panel E: 下游接收端的分子签名分析（改进版）

改进内容：
1. 按照解剖学距离对脑区分组和排序
2. 清晰标记全部20个目标脑区
3. 提取每个broadcast axis group的完整marker基因统计
4. 改进热图可视化

作者: Claude
日期: 2025-10-29
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Set
from loguru import logger
from tqdm import tqdm
import nrrd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['font.weight'] = 'normal'

# 配置日志
logger.add("panel_e_analysis.log", rotation="10 MB")


# 定义解剖学分组（按照离CLA的距离从近到远）
ANATOMICAL_GROUPS = {
    'Prefrontal/Motor': {
        'distance': 1,  # 最近
        'regions': ['ACAd', 'ACAv', 'MOs', 'MOp', 'FRP', 'PL', 'ILA', 'ORBm', 'ORBl'],
        'color': '#E74C3C',  # 红色
        'description': 'Closest to CLA: prefrontal and motor cortices'
    },
    'Orbital': {
        'distance': 2,  # 中等距离
        'regions': ['ORBvl', 'AId', 'AIp', 'AIv', 'GU', 'VISC'],
        'color': '#F39C12',  # 橙色
        'description': 'Intermediate: orbital and insular cortices'
    },
    'Entorhinal/Retrosplenial': {
        'distance': 3,  # 较远
        'regions': ['ENTl', 'ENTm', 'RSPv', 'RSPd', 'RSPagl'],
        'color': '#27AE60',  # 绿色
        'description': 'Far: entorhinal and retrosplenial cortices'
    },
    'Striatal/Basal_Ganglia': {
        'distance': 4,  # 最远
        'regions': ['CP', 'ACB', 'SNr', 'PALd'],
        'color': '#3498DB',  # 蓝色
        'description': 'Farthest: striatal and basal ganglia structures'
    }
}


class DownstreamMolecularProfilerV2:
    """下游微区分子签名分析器（改进版）"""

    def __init__(self, data_path: Path):
        """初始化"""
        self.data_path = data_path

        # MERFISH数据
        self.merfish_cells = None

        # 层级数据
        self.hierarchy_data = None
        self.subclass_to_markers = {}
        self.subclass_to_neurotransmitter = {}

        # CCF树结构
        self.id_to_acronym = {}
        self.acronym_to_id = {}
        self.subregion_acronyms = set()

        # 注释体积
        self.annotation_volume = None

        # 分析结果
        self.subregion_profiles = {}
        self.group_profiles = {}  # 新增：按组的统计

    def load_ccf_tree_structure(self) -> bool:
        """加载CCF树结构"""
        logger.info("加载CCF树结构...")

        tree_file = self.data_path / "tree_yzx.json"
        if not tree_file.exists():
            logger.error(f"CCF树文件不存在: {tree_file}")
            return False

        with open(tree_file, 'r') as f:
            tree_data = json.load(f)

        def traverse_tree(node):
            if not isinstance(node, dict):
                return

            node_id = node.get('id')
            acronym = node.get('acronym')
            name = node.get('name', '')

            if node_id and acronym:
                self.id_to_acronym[node_id] = acronym
                self.acronym_to_id[acronym] = node_id

                if self._is_subregion(acronym, name):
                    self.subregion_acronyms.add(acronym)

            children = node.get('children', [])
            for child in children:
                traverse_tree(child)

        if isinstance(tree_data, list):
            for node in tree_data:
                traverse_tree(node)
        else:
            traverse_tree(tree_data)

        logger.info(f"加载完成: {len(self.id_to_acronym)} 个区域")
        logger.info(f"识别到 {len(self.subregion_acronyms)} 个subregions")

        return True

    def _is_subregion(self, acronym: str, name: str) -> bool:
        """判断是否为subregion"""
        if "-ME" in acronym:
            return False

        if "layer" in name.lower():
            return True

        layer_patterns = ['1', '2/3', '4', '5', '6a', '6b', 'L1', 'L2/3', 'L4', 'L5', 'L6a', 'L6b']
        for pattern in layer_patterns:
            if pattern in acronym:
                return True

        return False

    def load_annotation_volume(self) -> bool:
        """加载CCF注释体积"""
        logger.info("加载CCF注释体积...")

        annotation_file = self.data_path / "annotation_25.nrrd"
        if not annotation_file.exists():
            logger.error(f"注释文件不存在: {annotation_file}")
            return False

        try:
            self.annotation_volume, header = nrrd.read(str(annotation_file))
            logger.info(f"注释体积形状: {self.annotation_volume.shape}")
            return True
        except Exception as e:
            logger.error(f"加载注释体积失败: {e}")
            return False

    def load_merfish_data(self) -> bool:
        """加载MERFISH细胞数据"""
        logger.info("加载MERFISH细胞数据...")

        coord_files = [
            "ccf_coordinates_1.csv",
            "ccf_coordinates_2.csv",
            "ccf_coordinates_3.csv",
            "ccf_coordinates_4.csv"
        ]

        coordinate_dfs = []
        for coord_file in coord_files:
            file_path = self.data_path / coord_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                for col in ['x', 'y', 'z']:
                    if col in df.columns:
                        df[f'{col}_ccf'] = df[col] * 40
                coordinate_dfs.append(df)
                logger.info(f"加载 {coord_file}: {len(df)} 个细胞")

        if not coordinate_dfs:
            logger.error("未找到坐标文件")
            return False

        coordinates_df = pd.concat(coordinate_dfs, ignore_index=True)

        meta_files = [
            "cell_metadata_with_cluster_annotation_1.csv",
            "cell_metadata_with_cluster_annotation_2.csv",
            "cell_metadata_with_cluster_annotation_3.csv",
            "cell_metadata_with_cluster_annotation_4.csv"
        ]

        metadata_dfs = []
        for meta_file in meta_files:
            file_path = self.data_path / meta_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                metadata_dfs.append(df)
                logger.info(f"加载 {meta_file}: {len(df)} 个细胞")

        if not metadata_dfs:
            logger.error("未找到元数据文件")
            return False

        metadata_df = pd.concat(metadata_dfs, ignore_index=True)

        self.merfish_cells = pd.merge(
            coordinates_df,
            metadata_df,
            on='cell_label',
            how='inner'
        )

        logger.info(f"合并后: {len(self.merfish_cells)} 个MERFISH细胞")

        required_cols = ['x_ccf', 'y_ccf', 'z_ccf', 'subclass']
        missing_cols = [col for col in required_cols if col not in self.merfish_cells.columns]
        if missing_cols:
            logger.error(f"缺少必需的列: {missing_cols}")
            return False

        return True

    def map_cells_to_subregions(self) -> bool:
        """将MERFISH细胞映射到subregions"""
        logger.info("将细胞映射到subregions...")

        mapped_count = 0
        subregion_counts = defaultdict(int)

        self.merfish_cells['subregion_acronym'] = None
        self.merfish_cells['parent_region'] = None  # 新增：记录父区域

        for idx, cell in tqdm(self.merfish_cells.iterrows(),
                             total=len(self.merfish_cells),
                             desc="映射细胞"):
            try:
                x_idx = int(cell['x_ccf'])
                y_idx = int(cell['y_ccf'])
                z_idx = int(cell['z_ccf'])

                if (0 <= x_idx < self.annotation_volume.shape[0] and
                    0 <= y_idx < self.annotation_volume.shape[1] and
                    0 <= z_idx < self.annotation_volume.shape[2]):

                    ccf_id = int(self.annotation_volume[x_idx, y_idx, z_idx])

                    if ccf_id in self.id_to_acronym:
                        acronym = self.id_to_acronym[ccf_id]

                        # 提取父区域（去掉layer后缀）
                        parent_region = self._extract_parent_region(acronym)

                        if acronym in self.subregion_acronyms:
                            self.merfish_cells.at[idx, 'subregion_acronym'] = acronym
                            self.merfish_cells.at[idx, 'parent_region'] = parent_region
                            mapped_count += 1
                            subregion_counts[acronym] += 1

            except Exception as e:
                logger.debug(f"映射细胞 {idx} 失败: {e}")
                continue

        logger.info(f"成功映射 {mapped_count} / {len(self.merfish_cells)} 个细胞到subregions")
        logger.info(f"涵盖 {len(subregion_counts)} 个不同的subregions")

        if subregion_counts:
            top_subregions = sorted(subregion_counts.items(),
                                   key=lambda x: x[1],
                                   reverse=True)[:20]
            logger.info("Top 20 subregions (细胞数):")
            for acronym, count in top_subregions:
                logger.info(f"  {acronym}: {count}")

        return mapped_count > 0

    def _extract_parent_region(self, acronym: str) -> str:
        """从subregion acronym提取父区域名称"""
        # 移除常见的layer后缀
        for suffix in ['1', '2/3', '2', '3', '4', '5', '6a', '6b', '6']:
            if acronym.endswith(suffix):
                return acronym[:-len(suffix)]
        return acronym

    def load_hierarchy_data(self) -> bool:
        """加载细胞类型层级数据"""
        logger.info("加载细胞类型层级数据...")

        hierarchy_file = self.data_path / "tran-data-type-tree.json"
        if not hierarchy_file.exists():
            logger.error(f"层级文件不存在: {hierarchy_file}")
            return False

        with open(hierarchy_file, 'r') as f:
            self.hierarchy_data = json.load(f)

        self._extract_subclass_info(self.hierarchy_data)

        logger.info(f"提取到 {len(self.subclass_to_markers)} 个subclass的marker信息")
        logger.info(f"提取到 {len(self.subclass_to_neurotransmitter)} 个subclass的neurotransmitter信息")

        return True

    def _extract_subclass_info(self, node, parent_nt=None):
        """递归提取subclass信息"""
        if not isinstance(node, dict):
            return

        current_nt = node.get('neurotransmitter', parent_nt)

        label = node.get('label', '')
        if 'subclass' in label.lower() or node.get('level') == 'subclass':
            name = node.get('name', '')
            if name:
                markers = node.get('markers', [])
                if markers:
                    self.subclass_to_markers[name] = markers

                if current_nt:
                    self.subclass_to_neurotransmitter[name] = current_nt

        children = node.get('children', [])
        for child in children:
            self._extract_subclass_info(child, current_nt)

    def analyze_target_subregions(self, target_subregions: List[str]) -> bool:
        """分析目标subregions的分子签名"""
        logger.info(f"分析 {len(target_subregions)} 个目标subregions...")

        for subregion in tqdm(target_subregions, desc="分析subregions"):
            subregion_cells = self.merfish_cells[
                self.merfish_cells['subregion_acronym'] == subregion
            ]

            if len(subregion_cells) == 0:
                logger.warning(f"{subregion}: 无细胞数据")
                continue

            subclass_counts = subregion_cells['subclass'].value_counts()
            total_cells = len(subregion_cells)

            top_subclasses = []
            for i, (subclass, count) in enumerate(subclass_counts.head(2).items()):
                percentage = (count / total_cells) * 100
                markers = self.subclass_to_markers.get(subclass, [])
                marker_str = ", ".join(markers[:5]) if markers else "N/A"
                nt = self.subclass_to_neurotransmitter.get(subclass, "Unknown")

                top_subclasses.append({
                    'rank': i + 1,
                    'subclass': subclass,
                    'percentage': percentage,
                    'count': count,
                    'markers': markers,  # 完整的marker列表
                    'marker_str': marker_str,  # 前5个用于显示
                    'neurotransmitter': nt
                })

            if top_subclasses:
                dominant_nt = top_subclasses[0]['neurotransmitter']
            else:
                dominant_nt = "Unknown"

            # 获取父区域
            parent_region = subregion_cells['parent_region'].iloc[0] if len(subregion_cells) > 0 else None

            self.subregion_profiles[subregion] = {
                'total_cells': total_cells,
                'top_subclasses': top_subclasses,
                'dominant_neurotransmitter': dominant_nt,
                'parent_region': parent_region
            }

            logger.info(f"{subregion}: {total_cells} cells, "
                       f"Top: {top_subclasses[0]['subclass'] if top_subclasses else 'N/A'} "
                       f"({top_subclasses[0]['percentage']:.1f}%)")

        logger.info(f"完成分析: {len(self.subregion_profiles)} 个subregions")
        return len(self.subregion_profiles) > 0

    def analyze_anatomical_groups(self):
        """按解剖学分组统计enriched subclass和markers"""
        logger.info("按解剖学分组统计...")

        for group_name, group_info in ANATOMICAL_GROUPS.items():
            group_regions = group_info['regions']

            # 收集该组所有subregions的细胞
            group_cells = self.merfish_cells[
                self.merfish_cells['parent_region'].isin(group_regions)
            ]

            if len(group_cells) == 0:
                logger.warning(f"{group_name}: 无细胞数据")
                continue

            # 统计subclass分布
            subclass_counts = group_cells['subclass'].value_counts()
            total_cells = len(group_cells)

            # 获取top enriched subclass
            enriched_subclasses = []
            for i, (subclass, count) in enumerate(subclass_counts.head(3).items()):
                percentage = (count / total_cells) * 100
                markers = self.subclass_to_markers.get(subclass, [])
                nt = self.subclass_to_neurotransmitter.get(subclass, "Unknown")

                enriched_subclasses.append({
                    'rank': i + 1,
                    'subclass': subclass,
                    'percentage': percentage,
                    'count': count,
                    'markers': markers,
                    'neurotransmitter': nt
                })

            # 找到该组实际存在的subregions
            actual_subregions = []
            for subregion, profile in self.subregion_profiles.items():
                if profile.get('parent_region') in group_regions:
                    actual_subregions.append(subregion)

            self.group_profiles[group_name] = {
                'total_cells': total_cells,
                'enriched_subclasses': enriched_subclasses,
                'actual_subregions': actual_subregions,
                'color': group_info['color'],
                'distance': group_info['distance']
            }

            logger.info(f"{group_name}: {total_cells} cells, "
                       f"{len(actual_subregions)} subregions, "
                       f"Top: {enriched_subclasses[0]['subclass']}")

        logger.info(f"完成分组统计: {len(self.group_profiles)} 个组")

    def generate_improved_heatmap(self, output_path: Path):
        """生成改进版的分子签名热图"""
        logger.info("生成改进版热图...")

        if not self.subregion_profiles:
            logger.error("无profile数据，无法生成热图")
            return

        # 按照解剖学分组排序subregions
        sorted_subregions = self._sort_subregions_by_anatomy()

        if len(sorted_subregions) == 0:
            logger.error("无可排序的subregions")
            return

        n_regions = len(sorted_subregions)

        # 创建图形
        fig = plt.figure(figsize=(20, max(12, n_regions * 0.4)))

        # 创建主热图区域和顶部分组区域
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 20], hspace=0.02)
        ax_group = fig.add_subplot(gs[0])
        ax_main = fig.add_subplot(gs[1])

        # 准备热图数据
        heatmap_data = []
        y_labels = []

        # Neurotransmitter颜色映射
        nt_colors = {
            'Glutamate': '#E74C3C',
            'GABA': '#3498DB',
            'Dopamine': '#9B59B6',
            'Serotonin': '#F39C12',
            'Acetylcholine': '#1ABC9C',
            'Unknown': '#95A5A6'
        }

        # 为每个subregion准备数据
        for subregion in sorted_subregions:
            profile = self.subregion_profiles[subregion]
            top_subclasses = profile['top_subclasses']

            for i in range(2):
                if i < len(top_subclasses):
                    sc = top_subclasses[i]
                    label = f"{sc['subclass'][:40]}\n{sc['marker_str']}"
                    heatmap_data.append([sc['percentage']])
                else:
                    label = "N/A"
                    heatmap_data.append([0])

                y_labels.append(label)

        # 绘制主热图
        heatmap_array = np.array(heatmap_data)
        im = ax_main.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        # 设置Y轴
        ax_main.set_yticks(range(len(y_labels)))
        ax_main.set_yticklabels(y_labels, fontsize=9)

        # 设置X轴 - 显示全部20个脑区（加粗）
        ax_main.set_xticks([0])
        ax_main.set_xticklabels(['Cell Type Composition'], fontsize=11, weight='bold')

        # 添加百分比文本
        for i in range(len(heatmap_data)):
            value = heatmap_data[i][0]
            if value > 0:
                ax_main.text(0, i, f'{value:.1f}%',
                           ha='center', va='center',
                           color='white' if value > 50 else 'black',
                           fontsize=9, weight='bold')

        # 在Y轴左侧添加subregion标签（加粗）
        current_y = 0
        for subregion in sorted_subregions:
            # 在两行的中间位置添加标签
            ax_main.text(-0.15, current_y + 0.5, subregion,
                        ha='right', va='center',
                        fontsize=11, weight='bold',
                        transform=ax_main.transData)
            current_y += 2

        # 绘制顶部分组标签栏
        group_boundaries = self._get_group_boundaries(sorted_subregions)

        for group_name, (start, end) in group_boundaries.items():
            color = ANATOMICAL_GROUPS[group_name]['color']
            # 注意：每个region占2行（top1和top2）
            ax_group.barh(0, end - start, left=start, height=1,
                         color=color, alpha=0.7, edgecolor='white', linewidth=2)
            # 添加组名
            ax_group.text((start + end) / 2, 0, group_name.replace('_', ' '),
                         ha='center', va='center',
                         fontsize=10, weight='bold', color='white')

        ax_group.set_xlim(0, len(sorted_subregions))
        ax_group.set_ylim(-0.5, 0.5)
        ax_group.axis('off')

        # 添加neurotransmitter颜色条（在右侧）
        for i, subregion in enumerate(sorted_subregions):
            profile = self.subregion_profiles[subregion]
            nt = profile['dominant_neurotransmitter']
            color = nt_colors.get(nt, nt_colors['Unknown'])

            # 每个subregion有2行
            y_pos = i * 2
            rect = plt.Rectangle((1.02, y_pos), 0.03, 2,
                               facecolor=color,
                               edgecolor='none',
                               transform=ax_main.transData,
                               clip_on=False)
            ax_main.add_patch(rect)

        # 添加图例
        from matplotlib.patches import Patch

        # Neurotransmitter图例
        nt_legend_elements = [Patch(facecolor=color, label=nt, edgecolor='black')
                             for nt, color in nt_colors.items()
                             if nt != 'Unknown']

        # Anatomical group图例
        group_legend_elements = [Patch(facecolor=ANATOMICAL_GROUPS[name]['color'],
                                       label=name.replace('_', ' '),
                                       alpha=0.7,
                                       edgecolor='white')
                                for name in sorted(ANATOMICAL_GROUPS.keys(),
                                                  key=lambda x: ANATOMICAL_GROUPS[x]['distance'])]

        # 放置图例
        legend1 = ax_main.legend(handles=nt_legend_elements,
                                title='Neurotransmitter',
                                loc='upper left', bbox_to_anchor=(1.08, 1),
                                frameon=True, fontsize=9)

        ax_main.add_artist(legend1)  # 需要add_artist才能显示第二个图例

        legend2 = ax_main.legend(handles=group_legend_elements,
                                title='Anatomical Groups\n(by distance from CLA)',
                                loc='upper left', bbox_to_anchor=(1.08, 0.6),
                                frameon=True, fontsize=9)

        # 标题
        ax_main.set_title('Panel E: Molecular Signatures of CLA Target Subregions\n' +
                         'Organized by Anatomical Distance from CLA',
                         fontsize=16, weight='bold', pad=20)

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage of Cells (%)', rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()

        # 保存
        output_file = output_path / "panel_e_molecular_signature_heatmap_improved.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"热图已保存: {output_file}")

        plt.close()

    def _sort_subregions_by_anatomy(self) -> List[str]:
        """按照解剖学分组排序subregions"""
        sorted_regions = []

        # 按距离排序组
        sorted_groups = sorted(ANATOMICAL_GROUPS.items(),
                              key=lambda x: x[1]['distance'])

        for group_name, group_info in sorted_groups:
            group_regions = group_info['regions']

            # 找到属于该组的subregions
            group_subregions = []
            for subregion, profile in self.subregion_profiles.items():
                parent_region = profile.get('parent_region')
                if parent_region in group_regions:
                    group_subregions.append(subregion)

            # 按名称排序（保持一致性）
            group_subregions.sort()
            sorted_regions.extend(group_subregions)

        return sorted_regions

    def _get_group_boundaries(self, sorted_subregions: List[str]) -> Dict[str, Tuple[int, int]]:
        """获取每个组在排序后的subregions中的边界"""
        boundaries = {}
        current_pos = 0

        sorted_groups = sorted(ANATOMICAL_GROUPS.items(),
                              key=lambda x: x[1]['distance'])

        for group_name, group_info in sorted_groups:
            group_regions = group_info['regions']

            # 统计该组的subregions数量
            count = 0
            for subregion in sorted_subregions[current_pos:]:
                profile = self.subregion_profiles.get(subregion)
                if profile and profile.get('parent_region') in group_regions:
                    count += 1
                else:
                    break

            if count > 0:
                boundaries[group_name] = (current_pos, current_pos + count)
                current_pos += count

        return boundaries

    def save_group_marker_table(self, output_path: Path):
        """保存每个broadcast axis group的marker统计表"""
        logger.info("保存分组marker统计表...")

        # 创建表格数据
        rows = []

        for group_name in sorted(ANATOMICAL_GROUPS.keys(),
                                key=lambda x: ANATOMICAL_GROUPS[x]['distance']):
            if group_name not in self.group_profiles:
                continue

            profile = self.group_profiles[group_name]

            # 获取该组的所有subregions
            subregions_str = ", ".join(sorted(profile['actual_subregions']))

            # 获取top enriched subclasses
            for i, sc_info in enumerate(profile['enriched_subclasses'][:3]):  # 显示top 3
                # 获取完整的marker列表（30-60个）
                all_markers = sc_info['markers']

                # 分为example markers (前5个) 和 additional markers (其余)
                example_markers = ", ".join(all_markers[:5]) if len(all_markers) > 0 else "N/A"
                additional_markers = ", ".join(all_markers[5:30]) if len(all_markers) > 5 else ""

                rows.append({
                    'Broadcast_Axis': group_name.replace('_', '/'),
                    'Rank': i + 1,
                    'Distance_from_CLA': ANATOMICAL_GROUPS[group_name]['distance'],
                    'Recurrent_Target_Subregions': subregions_str if i == 0 else "",  # 只在第一行显示
                    'Enriched_Subclass': sc_info['subclass'],
                    'Percentage': f"{sc_info['percentage']:.2f}%",
                    'Cell_Count': sc_info['count'],
                    'Total_Cells': profile['total_cells'],
                    'Neurotransmitter': sc_info['neurotransmitter'],
                    'Example_Markers': example_markers,
                    'Additional_Markers_30': additional_markers,
                    'Total_Markers': len(all_markers)
                })

        # 保存为CSV
        df = pd.DataFrame(rows)
        csv_file = output_path / "panel_e_group_marker_statistics.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"分组marker统计表已保存: {csv_file}")

        # 同时生成易读的文本报告
        self._generate_group_marker_report(output_path)

    def _generate_group_marker_report(self, output_path: Path):
        """生成易读的分组marker报告"""
        report_lines = [
            "=" * 100,
            "Panel E: Broadcast Axis Group - Enriched Subclass and Marker Statistics",
            "=" * 100,
            "",
            "按照离CLA的解剖学距离排序（近→远）",
            ""
        ]

        for group_name in sorted(ANATOMICAL_GROUPS.keys(),
                                key=lambda x: ANATOMICAL_GROUPS[x]['distance']):
            if group_name not in self.group_profiles:
                continue

            profile = self.group_profiles[group_name]
            group_info = ANATOMICAL_GROUPS[group_name]

            report_lines.append(f"\n{'='*100}")
            report_lines.append(f"Broadcast Axis: {group_name.replace('_', '/')}")
            report_lines.append(f"{'='*100}")
            report_lines.append(f"Distance from CLA: Level {group_info['distance']} - {group_info['description']}")
            report_lines.append(f"Total cells analyzed: {profile['total_cells']:,}")
            report_lines.append(f"\nRecurrent target subregions ({len(profile['actual_subregions'])}):")

            # 按字母排序显示subregions
            for subregion in sorted(profile['actual_subregions']):
                report_lines.append(f"  • {subregion}")

            report_lines.append(f"\nEnriched Subclasses:")

            for i, sc_info in enumerate(profile['enriched_subclasses'][:3], 1):
                report_lines.append(f"\n  Rank {i}: {sc_info['subclass']}")
                report_lines.append(f"    Percentage: {sc_info['percentage']:.2f}%")
                report_lines.append(f"    Cell count: {sc_info['count']:,}")
                report_lines.append(f"    Neurotransmitter: {sc_info['neurotransmitter']}")

                # 显示marker基因
                markers = sc_info['markers']
                if markers:
                    report_lines.append(f"    Total markers: {len(markers)}")
                    report_lines.append(f"    Example markers (top 10):")
                    report_lines.append(f"      {', '.join(markers[:10])}")

                    if len(markers) > 10:
                        report_lines.append(f"    Additional markers (11-30):")
                        report_lines.append(f"      {', '.join(markers[10:30])}")

                    if len(markers) > 30:
                        report_lines.append(f"    ... and {len(markers) - 30} more markers")
                else:
                    report_lines.append(f"    Markers: N/A")

        # 添加总结
        report_lines.append(f"\n\n{'='*100}")
        report_lines.append("Summary Table")
        report_lines.append(f"{'='*100}")
        report_lines.append("")

        # 创建简洁的表格
        table_header = f"{'Broadcast Axis':<30} | {'Target Subregions':<40} | {'Enriched Subclass (top)':<40} | {'Example Markers':<30}"
        report_lines.append(table_header)
        report_lines.append("-" * len(table_header))

        for group_name in sorted(ANATOMICAL_GROUPS.keys(),
                                key=lambda x: ANATOMICAL_GROUPS[x]['distance']):
            if group_name not in self.group_profiles:
                continue

            profile = self.group_profiles[group_name]

            if profile['enriched_subclasses']:
                top_sc = profile['enriched_subclasses'][0]

                # 简化subregions显示
                subregions_short = ", ".join(sorted(profile['actual_subregions'])[:3])
                if len(profile['actual_subregions']) > 3:
                    subregions_short += f"... (+{len(profile['actual_subregions'])-3})"

                # 简化markers显示
                markers_short = ", ".join(top_sc['markers'][:3])
                if len(top_sc['markers']) > 3:
                    markers_short += "..."

                row = f"{group_name.replace('_', '/'):<30} | {subregions_short:<40} | {top_sc['subclass'][:40]:<40} | {markers_short:<30}"
                report_lines.append(row)

        # 保存报告
        report_file = output_path / "panel_e_group_marker_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"分组marker报告已保存: {report_file}")

    def save_profiles_to_csv(self, output_path: Path):
        """保存详细的profile数据到CSV"""
        logger.info("保存profile数据到CSV...")

        rows = []
        for subregion, profile in self.subregion_profiles.items():
            for sc in profile['top_subclasses']:
                rows.append({
                    'Subregion': subregion,
                    'Parent_Region': profile.get('parent_region', 'Unknown'),
                    'Rank': sc['rank'],
                    'Subclass': sc['subclass'],
                    'Percentage': sc['percentage'],
                    'Cell_Count': sc['count'],
                    'Total_Cells': profile['total_cells'],
                    'Marker_Display': sc['marker_str'],
                    'All_Markers': ', '.join(sc['markers']),
                    'Total_Markers': len(sc['markers']),
                    'Neurotransmitter': sc['neurotransmitter']
                })

        df = pd.DataFrame(rows)
        output_file = output_path / "panel_e_subregion_profiles.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Subregion profile CSV已保存: {output_file}")

    def generate_report(self, output_path: Path):
        """生成分析报告"""
        logger.info("生成分析报告...")

        report_lines = [
            "=" * 100,
            "Panel E: 下游接收端分子签名分析报告（改进版）",
            "=" * 100,
            "",
            f"分析的subregions数量: {len(self.subregion_profiles)}",
            f"解剖学分组数量: {len(self.group_profiles)}",
            ""
        ]

        # 按组排序显示
        for group_name in sorted(ANATOMICAL_GROUPS.keys(),
                                key=lambda x: ANATOMICAL_GROUPS[x]['distance']):
            if group_name not in self.group_profiles:
                continue

            profile = self.group_profiles[group_name]

            report_lines.append(f"\n{'='*100}")
            report_lines.append(f"Anatomical Group: {group_name.replace('_', '/')}")
            report_lines.append(f"{'='*100}")

            report_lines.append(f"\nSubregions in this group ({len(profile['actual_subregions'])}):")
            for subregion in sorted(profile['actual_subregions']):
                sr_profile = self.subregion_profiles[subregion]
                report_lines.append(f"\n  {subregion}:")
                report_lines.append(f"    Total cells: {sr_profile['total_cells']}")

                for sc in sr_profile['top_subclasses']:
                    report_lines.append(f"    Rank {sc['rank']}: {sc['subclass']} ({sc['percentage']:.1f}%)")

        # 保存报告
        report_file = output_path / "panel_e_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"报告已保存: {report_file}")


def main():
    """主函数"""
    logger.info("="*100)
    logger.info("Panel E: 下游接收端分子签名分析（改进版）")
    logger.info("="*100)

    # 配置路径
    data_path = Path("/home/wlj/NeuroXiv2/data")
    output_path = Path("./panel_E")
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取目标subregions
    target_subregions_file = output_path / "main_targets.json"

    # if target_subregions_file.exists():
    #     with open(target_subregions_file, 'r') as f:
    #         target_subregions = json.load(f)
    #     logger.info(f"从Panel D加载到 {len(target_subregions)} 个目标subregions")
    # else:
    #     logger.warning("未找到Panel D的输出，使用默认的目标subregions")
    #     # 根据ANATOMICAL_GROUPS生成默认的subregions
    #     target_subregions = []
    #     for group_info in ANATOMICAL_GROUPS.values():
    #         for region in group_info['regions']:
    #             # 添加常见的layer后缀
    #             for layer in ['2/3', '5', '6a']:
    #                 target_subregions.append(f"{region}{layer}")

    target_subregions = [
        "CP",
        "CLA",
        "ENTl5",
        "MOs5",
        "MOs2/3",
        "MOs6a",
        "ACAd5",
        "SNr",
        "ENTl3",
        "AId6a",
        "ACAd2/3",
        "ENTl2",
        "ACAd6a",
        "EPd",
        "ENTl6a",
        "MOp6a",
        "RSPv5",
        "ACAv5",
        "ACAv2/3",
        "AIp6a"
    ]
    try:
        # 创建分析器
        profiler = DownstreamMolecularProfilerV2(data_path)

        # 1. 加载CCF树结构
        if not profiler.load_ccf_tree_structure():
            logger.error("加载CCF树结构失败")
            return 1

        # 2. 加载注释体积
        if not profiler.load_annotation_volume():
            logger.error("加载注释体积失败")
            return 1

        # 3. 加载MERFISH数据
        if not profiler.load_merfish_data():
            logger.error("加载MERFISH数据失败")
            return 1

        # 4. 将细胞映射到subregions
        if not profiler.map_cells_to_subregions():
            logger.error("细胞映射失败")
            return 1

        # 5. 加载层级数据
        if not profiler.load_hierarchy_data():
            logger.error("加载层级数据失败")
            return 1

        # 6. 分析目标subregions
        if not profiler.analyze_target_subregions(target_subregions):
            logger.error("分析subregions失败")
            return 1

        # 7. 按解剖学分组统计
        profiler.analyze_anatomical_groups()

        # 8. 生成改进版可视化
        profiler.generate_improved_heatmap(output_path)

        # 9. 保存分组marker统计表（重点！）
        profiler.save_group_marker_table(output_path)

        # 10. 保存详细数据
        profiler.save_profiles_to_csv(output_path)

        # 11. 生成报告
        profiler.generate_report(output_path)

        logger.info("="*100)
        logger.info("Panel E 分析完成！")
        logger.info("="*100)
        logger.info("\n生成的文件:")
        logger.info("  • panel_e_molecular_signature_heatmap_improved.png - 改进版热图")
        logger.info("  • panel_e_group_marker_statistics.csv - 分组marker统计表 ⭐")
        logger.info("  • panel_e_group_marker_report.txt - 分组marker详细报告 ⭐")
        logger.info("  • panel_e_subregion_profiles.csv - Subregion详细数据")
        logger.info("  • panel_e_analysis_report.txt - 完整分析报告")

        return 0

    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())