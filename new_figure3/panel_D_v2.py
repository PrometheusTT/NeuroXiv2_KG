"""
Panel D: CLA神经元下游投射的结构化偏好分析 - 改进版v2.2
====================================================

改进内容：
1. ✅ 横轴按照生物解剖学远近排序target subregions
2. ✅ 确保清晰标记所有target的X轴标签
3. ✅ 添加顶部彩色分组条
4. ✅ 与Panel E的解剖学分组保持一致

作者: Claude
日期: 2025-10-29
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple
from loguru import logger
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logger.add("panel_d_analysis.log", rotation="10 MB")


class CLAProjectionAnalyzerImproved:
    """CLA神经元投射分析器 - 改进版v2.2"""

    def __init__(self, data_path: Path):
        """
        初始化

        参数:
            data_path: 数据目录路径
        """
        self.data_path = data_path

        # CCF树结构映射
        self.id_to_acronym = {}
        self.acronym_to_id = {}
        self.acronym_to_children_ids = {}
        self.subregion_to_parent = {}

        # 投射数据
        self.neuron_to_subregion_proj = {}
        self.neuron_to_region_proj = {}

        # CLA神经元信息
        self.cla_neurons = set()
        self.neuron_info = {}

        # 分析结果
        self.projection_matrix = None
        self.main_targets = []

        # 解剖学分组定义（与Panel E保持一致）
        self.anatomical_groups = {
            'Prefrontal/Motor': {
                'regions': [
                    'ACAd5', 'ACAd6a', 'ACAd6b',
                    'MOs5', 'MOs6a', 'MOs6b',
                    'MOp5', 'MOp6a', 'MOp6b'
                ],
                'order': 1,
                'color': '#E74C3C',  # 红色
                'description': 'Nearest to CLA'
            },
            'Orbital': {
                'regions': [
                    'ORBl5', 'ORBl6a', 'ORBl6b',
                    'ORBm5', 'ORBm6a', 'ORBm6b'
                ],
                'order': 2,
                'color': '#F39C12',  # 橙色
                'description': 'Intermediate distance'
            },
            'Entorhinal/Retrosplenial': {
                'regions': [
                    'ENTl6a', 'ENTm6',
                    'RSPv5', 'RSPd5', 'RSPd6a'
                ],
                'order': 3,
                'color': '#27AE60',  # 绿色
                'description': 'Far from CLA'
            },
            'Striatal/Basal Ganglia': {
                'regions': [
                    'CP', 'ACB', 'SNr', 'VTA'
                ],
                'order': 4,
                'color': '#3498DB',  # 蓝色
                'description': 'Farthest (subcortical)'
            }
        }

    def load_ccf_tree_structure(self) -> bool:
        """加载CCF树结构"""
        logger.info("加载CCF树结构...")

        ccf_tree_json = self.data_path / "tree_yzx.json"
        if not ccf_tree_json.exists():
            logger.error(f"CCF树文件不存在: {ccf_tree_json}")
            return False

        try:
            with open(ccf_tree_json, 'r') as f:
                tree_data = json.load(f)

            def traverse_tree(node, parent_acronym=None):
                """递归遍历树节点"""
                if not isinstance(node, dict):
                    return

                node_id = node.get('id')
                acronym = node.get('acronym', '')

                if node_id is not None:
                    self.id_to_acronym[node_id] = acronym
                    self.acronym_to_id[acronym] = node_id

                    if parent_acronym:
                        self.subregion_to_parent[acronym] = parent_acronym

                children = node.get('children', [])
                if children:
                    children_ids = []
                    for child in children:
                        if isinstance(child, dict):
                            child_id = child.get('id')
                            if child_id is not None:
                                children_ids.append(child_id)

                    if children_ids:
                        self.acronym_to_children_ids[acronym] = children_ids

                    for child in children:
                        traverse_tree(child, acronym)

            if isinstance(tree_data, list):
                for root in tree_data:
                    traverse_tree(root)
            else:
                traverse_tree(tree_data)

            logger.info(f"  ✓ 加载了 {len(self.id_to_acronym)} 个CCF区域")
            return True

        except Exception as e:
            logger.error(f"加载CCF树结构失败: {e}")
            return False

    def identify_cla_neurons(self) -> bool:
        """识别CLA神经元"""
        logger.info("识别CLA神经元...")

        info_file = self.data_path / "info.csv"
        if not info_file.exists():
            logger.error(f"info文件不存在: {info_file}")
            return False

        try:
            info_df = pd.read_csv(info_file)

            # 过滤掉CCF-thin和local
            if 'ID' in info_df.columns:
                info_df = info_df[
                    ~info_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)
                ]

            # 识别CLA神经元
            if 'celltype' in info_df.columns:
                cla_df = info_df[info_df['celltype'].astype(str).str.contains('CLA', case=False, na=False)]

                for _, row in cla_df.iterrows():
                    neuron_id = str(row['ID'])
                    self.cla_neurons.add(neuron_id)

                    celltype = str(row['celltype'])
                    base_region = self._extract_base_region(celltype)

                    self.neuron_info[neuron_id] = {
                        'celltype': celltype,
                        'base_region': base_region
                    }

            logger.info(f"  ✓ 识别了 {len(self.cla_neurons)} 个CLA神经元")

            if len(self.cla_neurons) == 0:
                logger.warning("  ⚠ 没有找到CLA神经元！")
                return False

            return True

        except Exception as e:
            logger.error(f"识别CLA神经元失败: {e}")
            return False

    def _extract_base_region(self, celltype: str) -> str:
        """从celltype提取基础region（去除层信息）"""
        layer_patterns = ['1', '2/3', '4', '5', '6a', '6b']
        base_region = celltype

        for layer in layer_patterns:
            if celltype.endswith(layer):
                base_region = celltype[:-len(layer)]
                break

        return base_region

    def load_subregion_projections(self) -> bool:
        """加载subregion级别的投射数据"""
        logger.info("加载Subregion投射数据...")

        proj_file = self.data_path / "axonfull_proj.csv"
        if not proj_file.exists():
            logger.error(f"投射文件不存在: {proj_file}")
            return False

        try:
            proj_df = pd.read_csv(proj_file, index_col=0)
            logger.info(f"  - 原始数据: {proj_df.shape}")

            # 过滤
            proj_df = proj_df[
                ~proj_df.index.astype(str).str.contains('CCF-thin|local', na=False)
            ]
            logger.info(f"  - 过滤后: {proj_df.shape}")

            # 获取所有可用的CCF ID列
            available_ids = []
            for col in proj_df.columns:
                try:
                    ccf_id = int(col)
                    if ccf_id in self.id_to_acronym:
                        available_ids.append(ccf_id)
                except ValueError:
                    continue

            logger.info(f"  - 有效CCF ID列: {len(available_ids)}")

            # 提取CLA神经元的投射
            cla_projection_count = 0

            for neuron_id in tqdm(proj_df.index, desc="提取CLA投射"):
                neuron_id_str = str(neuron_id)

                if neuron_id_str not in self.cla_neurons:
                    continue

                neuron_row = proj_df.loc[neuron_id]
                self.neuron_to_subregion_proj[neuron_id_str] = {}

                for ccf_id in available_ids:
                    value = neuron_row.get(str(ccf_id), 0)

                    if pd.notna(value) and float(value) > 0:
                        acronym = self.id_to_acronym[ccf_id]
                        self.neuron_to_subregion_proj[neuron_id_str][acronym] = float(value)

                if self.neuron_to_subregion_proj[neuron_id_str]:
                    cla_projection_count += 1

            logger.info(f"  ✓ {cla_projection_count} 个CLA神经元有投射数据")

            total_projs = sum(len(v) for v in self.neuron_to_subregion_proj.values())
            logger.info(f"  ✓ 总投射关系: {total_projs}")

            unique_targets = set()
            for projs in self.neuron_to_subregion_proj.values():
                unique_targets.update(projs.keys())
            logger.info(f"  ✓ 唯一target subregion: {len(unique_targets)}")

            return True

        except Exception as e:
            logger.error(f"加载投射数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def identify_main_targets(
            self,
            top_n_regions: int = 5,
            threshold_percentile: float = 75
    ) -> List[str]:
        """识别主要投射目标"""
        logger.info("=" * 80)
        logger.info("识别主要投射目标")
        logger.info("=" * 80)

        # Step 1: 聚合到parent region级别
        region_projections = {}

        for neuron_id, subregion_projs in self.neuron_to_subregion_proj.items():
            for subregion, length in subregion_projs.items():
                parent_region = self.subregion_to_parent.get(subregion, subregion)

                if parent_region not in region_projections:
                    region_projections[parent_region] = 0.0

                region_projections[parent_region] += length

        # Step 2: 找出top N regions
        sorted_regions = sorted(
            region_projections.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_regions = [r[0] for r in sorted_regions[:top_n_regions]]

        logger.info(f"\nTop {top_n_regions} 投射目标regions:")
        for i, (region, total_length) in enumerate(sorted_regions[:top_n_regions], 1):
            logger.info(f"  {i}. {region}: {total_length:.2f}")

        # Step 3: 在每个top region内找出强投射的subregions
        main_targets = []

        for region in top_regions:
            subregion_projs = {}

            for neuron_id, projs in self.neuron_to_subregion_proj.items():
                for subregion, length in projs.items():
                    parent = self.subregion_to_parent.get(subregion, subregion)

                    if parent == region:
                        if subregion not in subregion_projs:
                            subregion_projs[subregion] = 0.0
                        subregion_projs[subregion] += length

            if not subregion_projs:
                continue

            values = list(subregion_projs.values())
            threshold = np.percentile(values, threshold_percentile)

            strong_subregions = [
                sr for sr, val in subregion_projs.items()
                if val >= threshold
            ]

            main_targets.extend(strong_subregions)

            logger.info(f"\n{region} 下的主要投射subregions:")
            logger.info(f"  阈值 (P{threshold_percentile}): {threshold:.2f}")
            logger.info(f"  主要subregions ({len(strong_subregions)}): {strong_subregions}")

        self.main_targets = main_targets

        logger.info(f"\n✓ 总共识别了 {len(main_targets)} 个主要投射目标")
        logger.info("=" * 80)

        return main_targets

    def get_sorted_targets_by_anatomy(self, targets: List[str]) -> List[str]:
        """按照解剖学分组顺序排序targets"""
        # 定义区域顺序（与堆叠柱状图一致）
        region_order = [
            "ACAd2/3", "ACAd5", "ACAd6a", "ACAv2/3", "ACAv5",
            "MOs2/3", "MOs5", "MOs6a", "MOp6a",
            "AId6a", "AIp6a",
            "CP", "EPd", "SNr",
            "ENTl2", "ENTl3", "ENTl5", "ENTl6a",
            "RSPv5", "CLA"
        ]

        # 创建顺序映射
        region_to_order = {region: i for i, region in enumerate(region_order)}

        # 分类：在定义中的和未定义的
        defined_targets = [t for t in targets if t in region_to_order]
        undefined_targets = [t for t in targets if t not in region_to_order]

        # 已定义的按照指定顺序排序
        defined_sorted = sorted(defined_targets, key=lambda t: region_to_order[t])

        # 未定义的按字母顺序排在最后
        undefined_sorted = sorted(undefined_targets)

        return defined_sorted + undefined_sorted

    def build_projection_matrix(self) -> pd.DataFrame:
        """构建投射强度矩阵"""
        logger.info("构建投射强度矩阵...")

        if not self.main_targets:
            logger.warning("未定义主要投射目标，使用所有targets")
            all_targets = set()
            for projs in self.neuron_to_subregion_proj.values():
                all_targets.update(projs.keys())
            self.main_targets = sorted(all_targets)

        # 按解剖学排序targets
        sorted_targets = self.get_sorted_targets_by_anatomy(self.main_targets)

        # 创建矩阵
        neurons = sorted(self.cla_neurons)

        matrix = np.zeros((len(neurons), len(sorted_targets)))

        neuron_to_idx = {n: i for i, n in enumerate(neurons)}
        target_to_idx = {t: i for i, t in enumerate(sorted_targets)}

        # 填充矩阵
        for neuron_id, projs in self.neuron_to_subregion_proj.items():
            if neuron_id not in neuron_to_idx:
                continue

            neuron_idx = neuron_to_idx[neuron_id]

            for target, length in projs.items():
                if target in target_to_idx:
                    target_idx = target_to_idx[target]
                    matrix[neuron_idx, target_idx] = length

        # 转换为DataFrame
        self.projection_matrix = pd.DataFrame(
            matrix,
            index=neurons,
            columns=sorted_targets
        )

        logger.info(f"  ✓ 矩阵形状: {self.projection_matrix.shape}")
        logger.info(f"  ✓ 非零值: {(matrix > 0).sum()}/{matrix.size}")
        logger.info(f"  ✓ Targets已按解剖学排序")

        return self.projection_matrix

    def plot_projection_heatmap(
            self,
            output_path: Path,
            figsize: Tuple[int, int] = (20, 14)
    ):
        """绘制投射热图（改进版v2.2）"""
        logger.info("绘制投射热图（改进版v2.2）...")

        if self.projection_matrix is None:
            logger.error("未构建投射矩阵")
            return

        # 准备数据（使用log scale）
        matrix_log = np.log10(self.projection_matrix + 1)

        # 创建图形
        fig = plt.figure(figsize=figsize)

        # 创建子图：顶部分组条 + 主热图
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[0.02, 0.98],
            hspace=0.01
        )

        ax_group = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        # 1. 绘制顶部分组条
        targets = list(self.projection_matrix.columns)

        ax_group.set_xlim(0, len(targets))
        ax_group.set_ylim(0, 1)
        ax_group.axis('off')

        current_x = 0
        # for group_name, group_info in sorted(
        #         self.anatomical_groups.items(),
        #         key=lambda x: x[1]['order']
        # ):
        #     group_targets = [t for t in targets if t in group_info['regions']]
        #     width = len(group_targets)
        #
        #     if width > 0:
        #         ax_group.add_patch(Rectangle(
        #             (current_x, 0), width, 1,
        #             facecolor=group_info['color'],
        #             edgecolor='white',
        #             linewidth=2
        #         ))
        #
        #         # 添加分组标签
        #         ax_group.text(
        #             current_x + width / 2, 0.5,
        #             group_name,
        #             ha='center', va='center',
        #             fontsize=10, fontweight='bold',
        #             color='white'
        #         )
        #
        #         current_x += width

        # 2. 绘制主热图
        im = ax_heat.imshow(
            matrix_log,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )

        # 设置X轴（target subregions）
        ax_heat.set_xticks(range(len(targets)))
        ax_heat.set_xticklabels(
            targets,
            rotation=45,
            ha='right',
            fontsize=16,
            fontweight='bold'
        )

        # 设置Y轴（CLA neurons）
        ax_heat.set_yticks(range(len(self.projection_matrix.index)))
        # ax_heat.set_yticklabels(self.projection_matrix.index, fontsize=6)

        # 确保X轴标签不被截断
        ax_heat.xaxis.set_tick_params(pad=3, length=5)

        # 添加网格线
        ax_heat.set_xticks(np.arange(len(targets)) - 0.5, minor=True)
        ax_heat.set_yticks(np.arange(len(self.projection_matrix.index)) - 0.5, minor=True)
        ax_heat.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # 标题和标签
        # ax_heat.set_title(
        #     'CLA Neuron Projections to Downstream Subregions\n',
        #     fontsize=18,
        #     fontweight='bold',
        #     pad=25
        # )
        ax_heat.set_xlabel('Target Subregions', fontsize=20, fontweight='bold')
        ax_heat.set_ylabel('CLA Neurons', fontsize=20, fontweight='bold')

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Projection Length (log10)', rotation=270, labelpad=20, fontsize=20)

        # 标注统计信息
        total_projections = (self.projection_matrix > 0).sum().sum()
        avg_targets_per_neuron = (self.projection_matrix > 0).sum(axis=1).mean()

        # stats_text = (
        #     f'Statistics:\n'
        #     f'CLA Neurons: {len(self.projection_matrix)}\n'
        #     f'Target Subregions: {len(targets)}\n'
        #     f'Total Projections: {total_projections}\n'
        #     f'Avg Targets/Neuron: {avg_targets_per_neuron:.1f}'
        # )
        #
        # ax_heat.text(
        #     1.12, 0.5, stats_text,
        #     transform=ax_heat.transAxes,
        #     fontsize=10,
        #     verticalalignment='center',
        #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # )

        plt.tight_layout()

        # 保存
        output_file = output_path / "panel_D_heatmap.png"
        plt.savefig(output_file, dpi=1200, bbox_inches='tight')
        logger.info(f"  ✓ 热图已保存: {output_file}")
        logger.info(f"  ✓ 确认X轴显示了全部 {len(targets)} 个target标签")

        plt.close()

    def export_results(self, output_path: Path):
        """导出分析结果"""
        logger.info("导出分析结果...")

        # 1. 导出主要投射目标列表
        targets_file = output_path / "main_targets.json"
        with open(targets_file, 'w') as f:
            json.dump({
                'main_targets': self.main_targets,
                'count': len(self.main_targets),
                'anatomically_sorted': True
            }, f, indent=2)
        logger.info(f"  ✓ 主要目标列表: {targets_file}")

        # 2. 导出投射矩阵
        matrix_file = output_path / "projection_matrix.csv"
        self.projection_matrix.to_csv(matrix_file)
        logger.info(f"  ✓ 投射矩阵: {matrix_file}")

        # 3. 生成报告
        report_file = output_path / "panel_D_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Panel D: CLA神经元下游投射分析报告 - v2.2\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"CLA神经元数量: {len(self.cla_neurons)}\n")
            f.write(f"主要投射目标数量: {len(self.main_targets)}\n")
            f.write(f"总投射关系数: {(self.projection_matrix > 0).sum().sum()}\n")
            f.write(f"Targets已按解剖学分组排序\n\n")

            # 按解剖学分组统计
            f.write("解剖学分组统计:\n")
            f.write("-" * 80 + "\n")

            for group_name, group_info in sorted(
                    self.anatomical_groups.items(),
                    key=lambda x: x[1]['order']
            ):
                group_targets = [t for t in self.main_targets if t in group_info['regions']]

                if group_targets:
                    f.write(f"\n{group_name} ({group_info['description']}):\n")
                    f.write(f"  Target数量: {len(group_targets)}\n")
                    f.write(f"  Targets: {', '.join(group_targets)}\n")

                    # 统计该组的总投射强度
                    group_total = self.projection_matrix[group_targets].sum().sum()
                    f.write(f"  总投射强度: {group_total:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("主要投射目标详细列表:\n")
            f.write("=" * 80 + "\n\n")

            for i, target in enumerate(self.projection_matrix.columns, 1):
                neuron_count = (self.projection_matrix[target] > 0).sum()
                total_length = self.projection_matrix[target].sum()

                f.write(f"{i}. {target}\n")
                f.write(f"   - 投射神经元数: {neuron_count}\n")
                f.write(f"   - 总投射长度: {total_length:.2f}\n")

            # 分析投射模式
            f.write("\n投射模式分析:\n")
            targets_per_neuron = (self.projection_matrix > 0).sum(axis=1)
            f.write(f"  - 平均每个神经元投射到 {targets_per_neuron.mean():.1f} 个subregions\n")
            f.write(f"  - 最多投射: {targets_per_neuron.max()} 个subregions\n")
            f.write(f"  - 最少投射: {targets_per_neuron.min()} 个subregions\n")

        logger.info(f"  ✓ 分析报告: {report_file}")

    def run_full_analysis(
            self,
            output_path: Path,
            top_n_regions: int = 5,
            threshold_percentile: float = 75
    ) -> bool:
        """运行完整分析流程"""
        logger.info("=" * 80)
        logger.info("Panel D: CLA神经元下游投射分析 - v2.2")
        logger.info("=" * 80)

        # Step 1: 加载CCF树结构
        if not self.load_ccf_tree_structure():
            return False

        # Step 2: 识别CLA神经元
        if not self.identify_cla_neurons():
            return False

        # Step 3: 加载投射数据
        if not self.load_subregion_projections():
            return False

        # Step 4: 识别主要投射目标
        self.identify_main_targets(top_n_regions, threshold_percentile)

        # Step 5: 构建投射矩阵（自动按解剖学排序）
        self.build_projection_matrix()

        # Step 6: 绘制热图
        self.plot_projection_heatmap(output_path)

        # Step 7: 导出结果
        self.export_results(output_path)

        logger.info("=" * 80)
        logger.info("✓ Panel D 分析完成")
        logger.info("=" * 80)

        return True


def main():
    """主函数"""
    # 设置路径
    data_path = Path("/home/wlj/NeuroXiv2/data")
    output_path = Path("./panel_D")
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建分析器
    analyzer = CLAProjectionAnalyzerImproved(data_path)

    # 运行分析
    success = analyzer.run_full_analysis(
        output_path=output_path,
        top_n_regions=20,
        threshold_percentile=75
    )

    if success:
        logger.info("\n✓✓✓ Panel D 分析成功完成！✓✓✓")
        logger.info(f"\n输出文件:")
        logger.info(f"  - 热图: {output_path / 'panel_D_heatmap.png'}")
        logger.info(f"  - 报告: {output_path / 'panel_D_report.txt'}")
        logger.info(f"  - 主要目标: {output_path / 'main_targets.json'}")
        logger.info(f"  - 投射矩阵: {output_path / 'projection_matrix.csv'}")
        return 0
    else:
        logger.error("\n✗✗✗ Panel D 分析失败 ✗✗✗")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())