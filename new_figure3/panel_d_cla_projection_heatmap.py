"""
Panel D: CLA神经元下游投射的结构化偏好分析
参考 NeuronProjectionProcessorV5Fixed 的数据处理方式

目的：证明CLA不是随便到处打，而是重复地打到一批特定的下游subregion
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple
from loguru import logger
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logger.add("panel_d_analysis.log", rotation="10 MB")


class CLAProjectionAnalyzer:
    """CLA神经元投射分析器（参考NeuronProjectionProcessorV5Fixed）"""

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

        # 投射数据（参考V5Fixed的结构）
        self.neuron_to_subregion_proj = {}  # {neuron_id: {subregion_acronym: length}}
        self.neuron_to_region_proj = {}  # {neuron_id: {region_acronym: length}}

        # CLA神经元信息
        self.cla_neurons = set()  # CLA神经元ID集合
        self.neuron_info = {}  # {neuron_id: {celltype, region, ...}}

        # 分析结果
        self.projection_matrix = None  # 投射强度矩阵
        self.main_targets = []  # 主要投射目标列表

    def load_ccf_tree_structure(self) -> bool:
        """加载CCF树结构（参考V5Fixed）"""
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

            # 识别CLA神经元（celltype包含"CLA"）
            if 'celltype' in info_df.columns:
                cla_df = info_df[info_df['celltype'].astype(str).str.contains('CLA', case=False, na=False)]

                for _, row in cla_df.iterrows():
                    neuron_id = str(row['ID'])
                    self.cla_neurons.add(neuron_id)

                    # 提取基础region（去除层信息）
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

            # 显示样例
            sample_neurons = list(self.cla_neurons)[:5]
            logger.info(f"  样例神经元ID: {sample_neurons}")

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
        """加载subregion级别的投射数据（参考V5Fixed）"""
        logger.info("加载Subregion投射数据...")

        proj_file = self.data_path / "axonfull_proj.csv"
        if not proj_file.exists():
            logger.error(f"投射文件不存在: {proj_file}")
            return False

        try:
            # 加载投射数据
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

                # 只处理CLA神经元
                if neuron_id_str not in self.cla_neurons:
                    continue

                neuron_row = proj_df.loc[neuron_id]
                self.neuron_to_subregion_proj[neuron_id_str] = {}

                # 提取所有subregion的投射值
                for ccf_id in available_ids:
                    value = neuron_row.get(str(ccf_id), 0)

                    if pd.notna(value) and float(value) > 0:
                        acronym = self.id_to_acronym[ccf_id]
                        self.neuron_to_subregion_proj[neuron_id_str][acronym] = float(value)

                if self.neuron_to_subregion_proj[neuron_id_str]:
                    cla_projection_count += 1

            logger.info(f"  ✓ {cla_projection_count} 个CLA神经元有投射数据")

            # 统计总投射关系数
            total_projs = sum(len(v) for v in self.neuron_to_subregion_proj.values())
            logger.info(f"  ✓ 总投射关系: {total_projs}")

            # 统计唯一target数量
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
        """
        识别主要投射目标

        参数:
            top_n_regions: 考虑前N个投射最强的region
            threshold_percentile: 在每个region内，投射强度超过此百分位数的subregion被视为主要目标

        返回:
            主要投射目标的acronym列表
        """
        logger.info("=" * 80)
        logger.info("识别主要投射目标")
        logger.info("=" * 80)

        # Step 1: 聚合到parent region级别
        region_projections = {}

        for neuron_id, subregion_projs in self.neuron_to_subregion_proj.items():
            for subregion, length in subregion_projs.items():
                # 获取parent region
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
            # 收集该region下所有subregion的投射数据
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

            # 计算阈值
            values = list(subregion_projs.values())
            threshold = np.percentile(values, threshold_percentile)

            # 筛选超过阈值的subregions
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

    def build_projection_matrix(self) -> pd.DataFrame:
        """构建投射强度矩阵"""
        logger.info("构建投射强度矩阵...")

        if not self.main_targets:
            logger.warning("未定义主要投射目标，使用所有targets")
            # 收集所有targets
            all_targets = set()
            for projs in self.neuron_to_subregion_proj.values():
                all_targets.update(projs.keys())
            self.main_targets = sorted(all_targets)

        # 创建矩阵
        neurons = sorted(self.cla_neurons)
        targets = sorted(self.main_targets)

        matrix = np.zeros((len(neurons), len(targets)))

        neuron_to_idx = {n: i for i, n in enumerate(neurons)}
        target_to_idx = {t: i for i, t in enumerate(targets)}

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
            columns=targets
        )

        logger.info(f"  ✓ 矩阵形状: {self.projection_matrix.shape}")
        logger.info(f"  ✓ 非零值: {(matrix > 0).sum()}/{matrix.size}")

        return self.projection_matrix

    def plot_projection_heatmap(
            self,
            output_path: Path,
            figsize: Tuple[int, int] = (16, 12)
    ):
        """
        绘制投射热图

        参数:
            output_path: 输出路径
            figsize: 图形大小
        """
        logger.info("绘制投射热图...")

        if self.projection_matrix is None:
            logger.error("未构建投射矩阵")
            return

        # 准备数据（使用log scale）
        matrix_log = np.log10(self.projection_matrix + 1)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热图
        im = ax.imshow(
            matrix_log,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )

        # 设置刻度
        ax.set_xticks(range(len(self.projection_matrix.columns)))
        ax.set_xticklabels(self.projection_matrix.columns, rotation=45, ha='right', fontsize=8)

        ax.set_yticks(range(len(self.projection_matrix.index)))
        ax.set_yticklabels(self.projection_matrix.index, fontsize=6)

        # 标题和标签
        ax.set_title(
            'CLA Neuron Projections to Downstream Subregions\n'
            'Main Broadcast Set (Structured Preference)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Target Subregions', fontsize=12, fontweight='bold')
        ax.set_ylabel('CLA Neurons', fontsize=12, fontweight='bold')

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Projection Length (log10)', rotation=270, labelpad=20, fontsize=10)

        # 添加网格线
        ax.set_xticks(np.arange(len(self.projection_matrix.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(self.projection_matrix.index)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # 标注统计信息
        total_projections = (self.projection_matrix > 0).sum().sum()
        avg_targets_per_neuron = (self.projection_matrix > 0).sum(axis=1).mean()

        stats_text = (
            f'Statistics:\n'
            f'Neurons: {len(self.projection_matrix)}\n'
            f'Target Subregions: {len(self.projection_matrix.columns)}\n'
            f'Total Projections: {total_projections}\n'
            f'Avg Targets/Neuron: {avg_targets_per_neuron:.1f}'
        )

        ax.text(
            1.15, 0.5, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        # 保存
        output_file = output_path / "panel_d_cla_projection_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ 热图已保存: {output_file}")

        plt.close()

    def export_results(self, output_path: Path):
        """导出分析结果"""
        logger.info("导出分析结果...")

        # 1. 导出主要投射目标列表
        targets_file = output_path / "main_targets.json"
        with open(targets_file, 'w') as f:
            json.dump({
                'main_targets': self.main_targets,
                'count': len(self.main_targets)
            }, f, indent=2)
        logger.info(f"  ✓ 主要目标列表: {targets_file}")

        # 2. 导出投射矩阵
        matrix_file = output_path / "projection_matrix.csv"
        self.projection_matrix.to_csv(matrix_file)
        logger.info(f"  ✓ 投射矩阵: {matrix_file}")

        # 3. 生成报告
        report_file = output_path / "panel_d_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Panel D: CLA神经元下游投射分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"CLA神经元数量: {len(self.cla_neurons)}\n")
            f.write(f"主要投射目标数量: {len(self.main_targets)}\n")
            f.write(f"总投射关系数: {(self.projection_matrix > 0).sum().sum()}\n\n")

            f.write("主要投射目标列表:\n")
            for i, target in enumerate(self.main_targets, 1):
                # 统计投射到该target的神经元数
                neuron_count = (self.projection_matrix[target] > 0).sum()
                total_length = self.projection_matrix[target].sum()

                f.write(f"  {i}. {target}\n")
                f.write(f"     - 投射神经元数: {neuron_count}\n")
                f.write(f"     - 总投射长度: {total_length:.2f}\n")

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
        logger.info("Panel D: CLA神经元下游投射分析")
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

        # Step 5: 构建投射矩阵
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
    output_path = Path("./")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建分析器
    analyzer = CLAProjectionAnalyzer(data_path)

    # 运行分析
    success = analyzer.run_full_analysis(
        output_path=output_path,
        top_n_regions=20,
        threshold_percentile=75
    )

    if success:
        logger.info("\n✓✓✓ Panel D 分析成功完成！✓✓✓")
        logger.info(f"\n输出文件:")
        logger.info(f"  - 热图: {output_path / 'panel_d_cla_projection_heatmap.png'}")
        logger.info(f"  - 报告: {output_path / 'panel_d_report.txt'}")
        logger.info(f"  - 主要目标: {output_path / 'main_targets.json'}")
        logger.info(f"  - 投射矩阵: {output_path / 'projection_matrix.csv'}")
        return 0
    else:
        logger.error("\n✗✗✗ Panel D 分析失败 ✗✗✗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())