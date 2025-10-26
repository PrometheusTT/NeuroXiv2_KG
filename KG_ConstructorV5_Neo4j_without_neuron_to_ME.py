"""
NeuroXiv 2.0 知识图谱构建器 - 完整修复版 V5
整合所有修复：
1. Neuron完整形态学特征
2. Neuron投射关系（到Region和Subregion）
3. MERFISH到Subregion/ME_Subregion的映射和HAS关系
4. 去重neighbouring关系

作者: wangmajortom & Claude
日期: 2025-10-26
版本: V5 Complete Fixed
"""
import json
import sys
import warnings
import ast
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Tuple

import numpy as np
import pandas as pd
import nrrd
from loguru import logger
from neo4j import GraphDatabase
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 导入现有模块
from Subregion_Loader import SubregionLoader
from neuron_subregion_relationship_inserter import (
    NeuronSubregionRelationshipInserter,
    verify_relationships
)

# 导入必要的数据加载函数
from data_loader_enhanced import (
    load_data,
    prepare_analysis_data,
    map_cells_to_regions_fixed
)

# 从原始脚本导入
from KG_ConstructorV4_Neo4j_with_neuron_subregion_subregionrelationship import (
    Neo4jConnector,
    MERFISHHierarchyLoader,
    RegionAnalyzer,
    MorphologyDataLoader,
    NeuronDataLoader,
    KnowledgeGraphBuilderNeo4j,
    CHUNK_SIZE,
    PCT_THRESHOLD,
    BATCH_SIZE,
    MORPH_ATTRIBUTES,
    STAT_ATTRIBUTES,
    NEURON_ATTRIBUTES,
    CONNECTIONS_FILE,
    INFO_FILE
)

# ==================== 新增：Neuron投射关系处理器 ====================

class NeuronProjectionProcessorV5Fixed:
    """修正版：从原始数据提取真实的subregion投射关系"""

    def __init__(self, data_path: Path, ccf_tree_json: Path):
        """
        初始化

        参数:
            data_path: 数据目录
            ccf_tree_json: CCF树结构JSON文件路径（用于构建ID映射）
        """
        self.data_path = data_path
        self.ccf_tree_json = ccf_tree_json

        # 原始投射数据
        self.axon_proj_df = None  # axonfull_proj.csv
        self.dend_proj_df = None  # denfull_proj.csv

        # info数据
        self.info_df = None

        # CCF结构映射
        self.id_to_acronym = {}  # {ccf_id: acronym}
        self.id_to_name = {}  # {ccf_id: name}
        self.acronym_to_children_ids = {}  # {acronym: [child_ids]}
        self.subregion_to_parent = {}  # {subregion_acronym: parent_region_acronym}

        # 投射数据（修正后）
        self.neuron_to_subregion_axon = {}  # {neuron_id: {subregion_acronym: length}}
        self.neuron_to_subregion_dend = {}  # {neuron_id: {subregion_acronym: length}}
        self.neuron_to_region_axon = {}  # {neuron_id: {region_acronym: length}}
        self.neuron_to_region_dend = {}  # {neuron_id: {region_acronym: length}}

    def load_ccf_tree_structure(self) -> bool:
        """加载CCF树结构，构建ID到acronym的映射"""
        logger.info(f"加载CCF树结构: {self.ccf_tree_json}")

        if not self.ccf_tree_json.exists():
            logger.error(f"CCF树文件不存在: {self.ccf_tree_json}")
            return False

        try:
            with open(self.ccf_tree_json, 'r') as f:
                tree_data = json.load(f)

            # 递归遍历树，构建映射
            def traverse_tree(node, parent_acronym=None):
                """递归遍历树节点"""
                if not isinstance(node, dict):
                    return

                # 获取节点信息
                node_id = node.get('id')
                acronym = node.get('acronym', '')
                name = node.get('name', '')

                # 存储映射
                if node_id is not None:
                    self.id_to_acronym[node_id] = acronym
                    self.id_to_name[node_id] = name

                    # 记录父子关系
                    if parent_acronym:
                        self.subregion_to_parent[acronym] = parent_acronym

                # 收集children IDs
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

                    # 递归处理children
                    for child in children:
                        traverse_tree(child, acronym)

            # 开始遍历
            if isinstance(tree_data, list):
                for root in tree_data:
                    traverse_tree(root)
            else:
                traverse_tree(tree_data)

            logger.info(f"  - 加载了 {len(self.id_to_acronym)} 个CCF区域")
            logger.info(f"  - 找到 {len(self.acronym_to_children_ids)} 个具有children的区域")

            return True

        except Exception as e:
            logger.error(f"加载CCF树结构失败: {e}")
            return False

    def load_raw_projection_data(self) -> bool:
        """加载原始投射数据文件"""
        logger.info("加载原始投射数据...")

        # 加载 axonfull_proj.csv
        axon_file = self.data_path / "axonfull_proj.csv"
        if axon_file.exists():
            try:
                self.axon_proj_df = pd.read_csv(axon_file, index_col=0)
                logger.info(f"  - 加载了 axonfull_proj.csv: {self.axon_proj_df.shape}")

                # 过滤CCF-thin和local
                original_len = len(self.axon_proj_df)
                self.axon_proj_df = self.axon_proj_df[
                    ~self.axon_proj_df.index.str.contains('CCF-thin|local', na=False)
                ]
                if original_len > len(self.axon_proj_df):
                    logger.info(f"    过滤掉 {original_len - len(self.axon_proj_df)} 个神经元")

            except Exception as e:
                logger.error(f"加载axonfull_proj.csv失败: {e}")
                return False
        else:
            logger.warning(f"未找到axonfull_proj.csv")

        # # 加载 denfull_proj.csv
        # dend_file = self.data_path / "denfull_proj.csv"
        # if dend_file.exists():
        #     try:
        #         self.dend_proj_df = pd.read_csv(dend_file, index_col=0)
        #         logger.info(f"  - 加载了 denfull_proj.csv: {self.dend_proj_df.shape}")
        #
        #         # 过滤
        #         original_len = len(self.dend_proj_df)
        #         self.dend_proj_df = self.dend_proj_df[
        #             ~self.dend_proj_df.index.str.contains('CCF-thin|local', na=False)
        #         ]
        #         if original_len > len(self.dend_proj_df):
        #             logger.info(f"    过滤掉 {original_len - len(self.dend_proj_df)} 个神经元")
        #
        #     except Exception as e:
        #         logger.error(f"加载denfull_proj.csv失败: {e}")
        #         return False
        # else:
        #     logger.warning(f"未找到denfull_proj.csv")

        if self.axon_proj_df is None:
            logger.error("未能加载任何投射数据文件")
            return False

        return True

    def load_info_data(self) -> bool:
        """加载info.csv"""
        logger.info("加载神经元信息...")

        info_file = self.data_path / "info.csv"
        if not info_file.exists():
            logger.error(f"info文件不存在: {info_file}")
            return False

        try:
            self.info_df = pd.read_csv(info_file)
            logger.info(f"  - 加载了 {len(self.info_df)} 条神经元信息")

            # 过滤
            if 'ID' in self.info_df.columns:
                original_len = len(self.info_df)
                if self.info_df['ID'].dtype == 'object':
                    self.info_df = self.info_df[
                        ~self.info_df['ID'].str.contains('CCF-thin|local', na=False)
                    ]
                    filtered_count = original_len - len(self.info_df)
                    if filtered_count > 0:
                        logger.info(f"    过滤掉 {filtered_count} 个info记录")

            return True

        except Exception as e:
            logger.error(f"加载info文件失败: {e}")
            return False

    def extract_subregion_projections(self,
                                      proj_df: pd.DataFrame,
                                      projection_type: str) -> Tuple[Dict, Dict]:
        """
        从原始投射数据中提取subregion和region级别的投射

        参数:
            proj_df: 原始投射数据（axon或dend）
            projection_type: 'axon' 或 'dend'

        返回:
            (neuron_to_subregion, neuron_to_region)
        """
        logger.info(f"提取{projection_type}投射的subregion数据...")

        neuron_to_subregion = {}
        neuron_to_region = {}

        # 获取所有可用的CCF ID列
        available_ids = []
        for col in proj_df.columns:
            try:
                ccf_id = int(col)
                if ccf_id in self.id_to_acronym:
                    available_ids.append(ccf_id)
            except ValueError:
                continue

        logger.info(f"  - 找到 {len(available_ids)} 个有效的CCF ID列")

        # 处理每个神经元
        for neuron_id in tqdm(proj_df.index, desc=f"处理{projection_type}投射"):
            neuron_row = proj_df.loc[neuron_id]

            neuron_to_subregion[neuron_id] = {}
            neuron_to_region[neuron_id] = {}

            # Step 1: 提取所有subregion的投射值（直接从原始数据）
            for ccf_id in available_ids:
                value = neuron_row.get(str(ccf_id), 0)

                if pd.notna(value) and float(value) > 0:
                    acronym = self.id_to_acronym[ccf_id]
                    neuron_to_subregion[neuron_id][acronym] = float(value)

            # Step 2: 计算region级别的投射（通过children求和）
            for region_acronym, children_ids in self.acronym_to_children_ids.items():
                # 对该region的所有children求和
                region_total = 0
                for child_id in children_ids:
                    if child_id in available_ids:
                        value = neuron_row.get(str(child_id), 0)
                        if pd.notna(value):
                            region_total += float(value)

                if region_total > 0:
                    neuron_to_region[neuron_id][region_acronym] = region_total

        # 统计
        total_subregion_proj = sum(len(v) for v in neuron_to_subregion.values())
        total_region_proj = sum(len(v) for v in neuron_to_region.values())

        logger.info(f"  - 提取了 {total_subregion_proj} 个subregion投射关系")
        logger.info(f"  - 计算了 {total_region_proj} 个region投射关系")

        return neuron_to_subregion, neuron_to_region

    def process_all_projections(self):
        """处理所有投射数据"""
        logger.info("=" * 60)
        logger.info("开始处理投射数据...")
        logger.info("=" * 60)

        # 处理axon投射
        if self.axon_proj_df is not None:
            logger.info("处理Axon投射数据...")
            self.neuron_to_subregion_axon, self.neuron_to_region_axon = \
                self.extract_subregion_projections(self.axon_proj_df, 'axon')

        # # 处理dendrite投射
        # if self.dend_proj_df is not None:
        #     logger.info("处理Dendrite投射数据...")
        #     self.neuron_to_subregion_dend, self.neuron_to_region_dend = \
        #         self.extract_subregion_projections(self.dend_proj_df, 'dend')

        logger.info("=" * 60)
        logger.info("投射数据处理完成！")
        logger.info("=" * 60)

    def verify_data_consistency(self):
        """验证数据一致性：比对Proj_Axon_Final.csv（如果存在）"""
        logger.info("验证数据一致性...")

        final_file = self.data_path / "Proj_Axon_Final.csv"
        if not final_file.exists():
            logger.info("  未找到Proj_Axon_Final.csv，跳过验证")
            return

        try:
            final_df = pd.read_csv(final_file, index_col=0)
            logger.info(f"  加载Proj_Axon_Final.csv用于验证")

            # 获取目标区域列
            target_cols = [col for col in final_df.columns
                           if col.startswith('proj_axon_') and col.endswith('_abs')]

            # 随机选择几个神经元验证
            sample_neurons = list(self.neuron_to_region_axon.keys())[:5]

            for neuron_id in sample_neurons:
                if neuron_id not in final_df.index:
                    continue

                logger.info(f"\n神经元 {neuron_id}:")

                # 提取几个主要区域对比
                for col in target_cols[:10]:  # 只验证前10个区域
                    region_acronym = col.replace('proj_axon_', '').replace('_abs', '')

                    # Final文件中的值
                    final_value = final_df.loc[neuron_id, col]

                    # 我们计算的值
                    calculated_value = self.neuron_to_region_axon.get(neuron_id, {}).get(region_acronym, 0)

                    if pd.notna(final_value) and float(final_value) > 0:
                        diff = abs(float(final_value) - calculated_value)
                        diff_pct = (diff / float(final_value)) * 100 if final_value > 0 else 0

                        status = "✓" if diff_pct < 1 else "✗"
                        logger.info(
                            f"  {status} {region_acronym}: Final={final_value:.2f}, Calculated={calculated_value:.2f} (差异{diff_pct:.2f}%)")

            logger.info("\n验证完成！")

        except Exception as e:
            logger.error(f"验证失败: {e}")

    def export_projection_data(self, output_dir: Optional[Path] = None):
        """导出投射数据"""
        if output_dir is None:
            output_dir = self.data_path / "projection_output"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"导出投射数据到: {output_dir}")

        # 1. 导出Neuron -> Subregion (Axon)
        if self.neuron_to_subregion_axon:
            self._export_projection_dict(
                self.neuron_to_subregion_axon,
                output_dir / "neuron_to_subregion_axon.csv",
                "axon投射"
            )

        # 2. 导出Neuron -> Region (Axon)
        if self.neuron_to_region_axon:
            self._export_projection_dict(
                self.neuron_to_region_axon,
                output_dir / "neuron_to_region_axon.csv",
                "axon投射"
            )

        # # 3. 导出Neuron -> Subregion (Dend)
        # if self.neuron_to_subregion_dend:
        #     self._export_projection_dict(
        #         self.neuron_to_subregion_dend,
        #         output_dir / "neuron_to_subregion_dend.csv",
        #         "dendrite投射"
        #     )
        #
        # # 4. 导出Neuron -> Region (Dend)
        # if self.neuron_to_region_dend:
        #     self._export_projection_dict(
        #         self.neuron_to_region_dend,
        #         output_dir / "neuron_to_region_dend.csv",
        #         "dendrite投射"
        #     )

        logger.info("导出完成！")

    def _export_projection_dict(self, proj_dict: Dict, output_file: Path, proj_type: str):
        """导出投射字典到CSV"""
        records = []

        for neuron_id, projections in proj_dict.items():
            for target, length in projections.items():
                records.append({
                    'neuron_id': neuron_id,
                    'target': target,
                    'projection_length': length,
                    'projection_type': proj_type
                })

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logger.info(f"  - 导出了 {len(df)} 条{proj_type}记录到: {output_file.name}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'axon': {
                'neurons_with_subregion_proj': len(self.neuron_to_subregion_axon),
                'neurons_with_region_proj': len(self.neuron_to_region_axon),
                'total_subregion_proj': sum(len(v) for v in self.neuron_to_subregion_axon.values()),
                'total_region_proj': sum(len(v) for v in self.neuron_to_region_axon.values()),
                'unique_subregions': len(
                    set(s for projs in self.neuron_to_subregion_axon.values() for s in projs.keys())),
                'unique_regions': len(set(r for projs in self.neuron_to_region_axon.values() for r in projs.keys())),
            }
        }

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("投射数据统计")
        logger.info("=" * 60)

        for proj_type in ['axon']:
            if stats[proj_type]['neurons_with_subregion_proj'] > 0:
                logger.info(f"\n{proj_type.upper()}投射:")
                logger.info(f"  有subregion投射的神经元: {stats[proj_type]['neurons_with_subregion_proj']}")
                logger.info(f"  有region投射的神经元: {stats[proj_type]['neurons_with_region_proj']}")
                logger.info(f"  总subregion投射关系: {stats[proj_type]['total_subregion_proj']}")
                logger.info(f"  总region投射关系: {stats[proj_type]['total_region_proj']}")
                logger.info(f"  唯一subregions数: {stats[proj_type]['unique_subregions']}")
                logger.info(f"  唯一regions数: {stats[proj_type]['unique_regions']}")

        logger.info("=" * 60)

    def run_full_pipeline(self) -> bool:
        """运行完整的处理流程"""
        logger.info("=" * 80)
        logger.info("NeuronProjectionProcessor V5 Fixed - 完整流程")
        logger.info("=" * 80)

        # 1. 加载CCF树结构
        if not self.load_ccf_tree_structure():
            return False

        # 2. 加载原始投射数据
        if not self.load_raw_projection_data():
            return False

        # 3. 加载info数据
        if not self.load_info_data():
            return False

        # 4. 处理投射数据
        self.process_all_projections()

        # 5. 验证数据一致性
        self.verify_data_consistency()

        # 6. 打印统计
        self.print_statistics()

        # 7. 导出数据
        self.export_projection_data()

        logger.info("=" * 80)
        logger.info("处理完成！")
        logger.info("=" * 80)

        return True


# ==================== 新增：MERFISH空间映射处理器 ====================

class MERFISHSubregionMapper:
    """MERFISH细胞到Subregion/ME_Subregion的映射器"""

    def __init__(self, data_path: Path, region_analyzer: RegionAnalyzer):
        """
        初始化

        参数:
            data_path: 数据目录
            region_analyzer: 区域分析器
        """
        self.data_path = data_path
        self.region_analyzer = region_analyzer

        # 注释数据
        self.annotation_volume = None
        self.annotation_header = None

        # Subregion映射（从NRRD）
        self.subregion_annotation = None
        self.voxel_to_subregion = {}  # voxel_value -> subregion_acronym

        # ME_Subregion映射（从NRRD）
        self.me_annotation = None
        self.voxel_to_me_subregion = {}  # voxel_value -> me_subregion_acronym

    def load_region_annotation(self) -> bool:
        """加载基础区域注释"""
        logger.info("加载区域注释数据...")

        annotation_file = self.data_path / "annotation_25.nrrd"
        if not annotation_file.exists():
            logger.error(f"注释文件不存在: {annotation_file}")
            return False

        try:
            self.annotation_volume, self.annotation_header = nrrd.read(str(annotation_file))
            logger.info(f"加载了区域注释: {self.annotation_volume.shape}")
            return True
        except Exception as e:
            logger.error(f"加载区域注释失败: {e}")
            return False

    def load_subregion_annotation(self) -> bool:
        """
        加载Subregion注释（CCF层级注释）

        这里假设有一个类似的NRRD文件用于Subregion
        如果没有单独的文件，则从Region+层级信息推断
        """
        logger.info("加载Subregion注释...")

        # 方案1: 如果有专门的subregion nrrd文件
        subregion_file = self.data_path / "subregion_annotation_25.nrrd"
        if subregion_file.exists():
            try:
                self.subregion_annotation, _ = nrrd.read(str(subregion_file))
                logger.info(f"加载了Subregion注释: {self.subregion_annotation.shape}")

                # 需要一个映射文件
                pkl_file = self.data_path / "subregion_mapping.pkl"
                if pkl_file.exists():
                    with open(pkl_file, 'rb') as f:
                        self.voxel_to_subregion = pickle.load(f)
                    logger.info(f"加载了 {len(self.voxel_to_subregion)} 个体素到Subregion的映射")

                return True
            except Exception as e:
                logger.warning(f"加载Subregion注释失败: {e}")

        # 方案2: 从Region注释+层级信息推断
        logger.info("将从Region注释推断Subregion信息")
        # 这需要额外的层级分割算法，暂时返回False
        logger.warning("Subregion注释不可用，将跳过Subregion级别的MERFISH映射")
        return False

    def load_me_subregion_annotation(self) -> bool:
        """加载ME_Subregion注释"""
        logger.info("加载ME_Subregion注释...")

        me_file = self.data_path / "parc_r671_full.nrrd"
        pkl_file = self.data_path / "parc_r671_full.nrrd.pkl"

        if not me_file.exists():
            logger.error(f"ME注释文件不存在: {me_file}")
            return False

        try:
            # 加载NRRD
            self.me_annotation, _ = nrrd.read(str(me_file))
            logger.info(f"加载了ME_Subregion注释: {self.me_annotation.shape}")

            # 加载映射
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    voxel_to_parent = pickle.load(f)
                logger.info(f"加载了 {len(voxel_to_parent)} 个ME体素映射")

                # 加载JSON来获取完整的ME信息
                json_file = self.data_path / "surf_tree_ccf-me.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        tree_data = json.load(f)

                    # 提取ME子区域映射
                    self.voxel_to_me_subregion = self._extract_me_mapping(tree_data)
                    logger.info(f"创建了 {len(self.voxel_to_me_subregion)} 个体素到ME_Subregion的映射")

            return True

        except Exception as e:
            logger.error(f"加载ME_Subregion注释失败: {e}")
            return False

    def _extract_me_mapping(self, tree_data) -> Dict[int, str]:
        """从JSON树中提取ME子区域映射"""
        voxel_to_me = {}

        def traverse(nodes):
            if isinstance(nodes, dict):
                nodes = [nodes]

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                acronym = node.get('acronym', '')
                node_id = node.get('id')

                # 只处理ME子区域
                if '-ME' in acronym and isinstance(node_id, str) and '_' in node_id:
                    parts = node_id.split('_')
                    try:
                        voxel_value = int(parts[1])
                        voxel_to_me[voxel_value] = acronym
                    except (ValueError, IndexError):
                        pass

                # 递归
                children = node.get('children', [])
                if children:
                    traverse(children)

        traverse(tree_data)
        return voxel_to_me

    def map_cells_to_subregions(self, cells_df: pd.DataFrame) -> pd.DataFrame:
        """将MERFISH细胞映射到Subregion"""
        logger.info("将MERFISH细胞映射到Subregion...")

        if self.subregion_annotation is None:
            logger.warning("没有Subregion注释数据")
            cells_df['subregion_acronym'] = None
            return cells_df

        # 转换坐标到体素索引
        resolution = 25.0
        x_indices = (cells_df['x_ccf'] / resolution).astype(int)
        y_indices = (cells_df['y_ccf'] / resolution).astype(int)
        z_indices = (cells_df['z_ccf'] / resolution).astype(int)

        # 查询Subregion
        subregion_acronyms = []
        for x, y, z in zip(x_indices, y_indices, z_indices):
            if (0 <= x < self.subregion_annotation.shape[0] and
                0 <= y < self.subregion_annotation.shape[1] and
                0 <= z < self.subregion_annotation.shape[2]):
                voxel_value = int(self.subregion_annotation[x, y, z])
                acronym = self.voxel_to_subregion.get(voxel_value)
                subregion_acronyms.append(acronym)
            else:
                subregion_acronyms.append(None)

        cells_df['subregion_acronym'] = subregion_acronyms

        mapped_count = cells_df['subregion_acronym'].notna().sum()
        logger.info(f"映射了 {mapped_count}/{len(cells_df)} 个细胞到Subregion")

        return cells_df

    def map_cells_to_me_subregions(self, cells_df: pd.DataFrame) -> pd.DataFrame:
        """将MERFISH细胞映射到ME_Subregion"""
        logger.info("将MERFISH细胞映射到ME_Subregion...")

        if self.me_annotation is None:
            logger.warning("没有ME_Subregion注释数据")
            cells_df['me_subregion_acronym'] = None
            return cells_df

        # 转换坐标到体素索引
        resolution = 25.0
        x_indices = (cells_df['x_ccf'] / resolution).astype(int)
        y_indices = (cells_df['y_ccf'] / resolution).astype(int)
        z_indices = (cells_df['z_ccf'] / resolution).astype(int)

        # 查询ME_Subregion
        me_acronyms = []
        for x, y, z in zip(x_indices, y_indices, z_indices):
            if (0 <= x < self.me_annotation.shape[0] and
                0 <= y < self.me_annotation.shape[1] and
                0 <= z < self.me_annotation.shape[2]):
                voxel_value = int(self.me_annotation[x, y, z])
                acronym = self.voxel_to_me_subregion.get(voxel_value)
                me_acronyms.append(acronym)
            else:
                me_acronyms.append(None)

        cells_df['me_subregion_acronym'] = me_acronyms

        mapped_count = cells_df['me_subregion_acronym'].notna().sum()
        logger.info(f"映射了 {mapped_count}/{len(cells_df)} 个细胞到ME_Subregion")

        return cells_df


# ==================== 扩展KnowledgeGraphBuilderNeo4j类 ====================

class KnowledgeGraphBuilderNeo4jV5(KnowledgeGraphBuilderNeo4j):
    """V5版本的知识图谱构建器 - 包含所有修复"""

    def generate_and_insert_neuron_nodes_with_full_morphology(self, neuron_loader: NeuronDataLoader):
        """
        生成并插入包含完整形态学特征的Neuron节点
        """
        if not neuron_loader or not neuron_loader.neurons_data:
            logger.warning("没有神经元数据可插入")
            return

        logger.info("生成并插入包含完整形态学特征的Neuron节点...")

        # 收集所有唯一神经元
        unique_neurons = {}
        duplicate_count = 0

        for neuron_id, neuron_data in tqdm(neuron_loader.neurons_data.items(),
                                           desc="准备Neuron数据"):
            if neuron_id in unique_neurons:
                duplicate_count += 1
            else:
                unique_neurons[neuron_id] = neuron_data

        if duplicate_count > 0:
            logger.warning(f"发现并移除了 {duplicate_count} 个重复的神经元ID")

        # 批量插入
        batch_nodes = []
        neuron_count = 0

        for neuron_id, neuron_data in tqdm(unique_neurons.items(), desc="插入Neuron节点"):
            # 创建节点字典 - 包含所有形态学特征
            node_dict = {
                'neuron_id': neuron_id,
                'name': neuron_data.get('name', neuron_id),
                'celltype': neuron_data.get('celltype', ''),
                'base_region': neuron_data.get('base_region', '')
            }

            # 添加所有形态学特征
            for feature in neuron_loader.morph_features:
                clean_feature = feature.replace(' ', '_').replace('/', '_').lower()
                value = neuron_data.get(clean_feature, 0.0)
                # 确保是有效的数值
                try:
                    node_dict[clean_feature] = float(value) if value else 0.0
                except:
                    node_dict[clean_feature] = 0.0

            batch_nodes.append(node_dict)

            # 批量插入
            if len(batch_nodes) >= BATCH_SIZE:
                self._insert_neurons_batch_with_merge(batch_nodes)
                neuron_count += len(batch_nodes)
                batch_nodes = []

        # 插入剩余节点
        if batch_nodes:
            self._insert_neurons_batch_with_merge(batch_nodes)
            neuron_count += len(batch_nodes)

        self.stats['neurons_inserted'] = neuron_count
        logger.info(f"成功插入 {neuron_count} 个包含完整形态学特征的Neuron节点")

        # 显示一个样例
        if unique_neurons:
            sample = list(unique_neurons.values())[0]
            logger.info(f"样例神经元特征数: {len(sample)}")

    def generate_and_insert_neuron_projection_relationships(
            self,
            projection_processor: NeuronProjectionProcessorV5Fixed
    ):
        """插入Neuron投射关系（修正版）"""
        logger.info("插入Neuron投射关系...")

        # 1. 插入Axon投射关系
        # 1.1 Region级别
        if projection_processor.neuron_to_region_axon:
            self._insert_projections_batch(
                projection_processor.neuron_to_region_axon,
                target_level='Region',
                projection_type='axon'
            )

        # 1.2 Subregion级别
        if projection_processor.neuron_to_subregion_axon:
            self._insert_projections_batch(
                projection_processor.neuron_to_subregion_axon,
                target_level='Subregion',
                projection_type='axon'
            )

        # # 2. 插入Dendrite投射关系
        # # 2.1 Region级别
        # if projection_processor.neuron_to_region_dend:
        #     self._insert_projections_batch(
        #         projection_processor.neuron_to_region_dend,
        #         target_level='Region',
        #         projection_type='dendrite'
        #     )
        #
        # # 2.2 Subregion级别
        # if projection_processor.neuron_to_subregion_dend:
        #     self._insert_projections_batch(
        #         projection_processor.neuron_to_subregion_dend,
        #         target_level='Subregion',
        #         projection_type='dendrite'
        #     )

    def _insert_projections_batch(
            self,
            projection_dict: Dict[str, Dict[str, float]],
            target_level: str,  # 'Region' or 'Subregion'
            projection_type: str  # 'axon' or 'dendrite'
    ):
        """
        统一的投射关系批量插入方法

        参数:
            projection_dict: {neuron_id: {target_acronym: length}}
            target_level: 目标级别 ('Region' 或 'Subregion')
            projection_type: 投射类型 ('axon' 或 'dendrite')
        """
        logger.info(f"插入Neuron->{target_level} {projection_type}投射关系...")

        batch_relationships = []
        success_count = 0

        for neuron_id, projections in tqdm(
                projection_dict.items(),
                desc=f"处理{projection_type}->{target_level}"
        ):
            for target_acronym, length in projections.items():
                rel = {
                    'neuron_id': str(neuron_id),
                    'target_acronym': target_acronym,
                    'projection_length': float(length),
                    'projection_type': projection_type
                }

                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_projection_insert_batch(
                        batch_relationships,
                        target_level
                    )
                    success_count += count
                    batch_relationships = []

        # 插入剩余
        if batch_relationships:
            count = self._execute_projection_insert_batch(
                batch_relationships,
                target_level
            )
            success_count += count

        logger.info(f"  - 成功插入 {success_count} 个{projection_type}->{target_level}关系")

    def _execute_projection_insert_batch(
            self,
            batch: List[Dict],
            target_level: str
    ):
        """执行投射关系批量插入到Neo4j"""

        if target_level == 'Region':
            # 使用region_id匹配
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (t:Region)
            WHERE t.acronym = rel.target_acronym
            MERGE (n)-[p:PROJECT_TO]->(t)
            SET p.projection_length = rel.projection_length,
                p.projection_type = rel.projection_type
            RETURN count(p) as created_count
            """
        else:  # Subregion
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (s:Subregion {acronym: rel.target_acronym})
            MERGE (n)-[p:PROJECT_TO]->(s)
            SET p.projection_length = rel.projection_length,
                p.projection_type = rel.projection_type
            RETURN count(p) as created_count
            """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入投射关系失败: {e}")
            return 0
    def _insert_neuron_to_region_projections(self, projection_processor):
        """插入Neuron -> Region投射关系"""
        logger.info("插入Neuron -> Region投射关系...")

        batch_relationships = []
        success_count = 0

        for neuron_id, projections in tqdm(
            projection_processor.neuron_to_region_projections.items(),
            desc="处理Neuron->Region投射"
        ):
            for region_id, length in projections.items():
                # 获取区域缩写
                region_acronym = ''
                if self.region_analyzer:
                    region_info = self.region_analyzer.region_info.get(region_id, {})
                    region_acronym = region_info.get('acronym', '')

                rel = {
                    'neuron_id': str(neuron_id),
                    'region_id': int(region_id),
                    'projection_length': float(length),
                    'target_acronym': region_acronym
                }

                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_neuron_projection_batch(
                        batch_relationships,
                        'Region',
                        'region_id'
                    )
                    success_count += count
                    batch_relationships = []

        # 插入剩余
        if batch_relationships:
            count = self._execute_neuron_projection_batch(
                batch_relationships,
                'Region',
                'region_id'
            )
            success_count += count

        logger.info(f"成功插入 {success_count} 个Neuron->Region投射关系")

    def _insert_neuron_to_subregion_projections(self, projection_processor):
        """插入Neuron -> Subregion投射关系"""
        logger.info("插入Neuron -> Subregion投射关系...")

        batch_relationships = []
        success_count = 0

        for neuron_id, projections in tqdm(
            projection_processor.neuron_to_subregion_projections.items(),
            desc="处理Neuron->Subregion投射"
        ):
            for subregion_acronym, length in projections.items():
                rel = {
                    'neuron_id': str(neuron_id),
                    'subregion_acronym': subregion_acronym,
                    'projection_length': float(length)
                }

                batch_relationships.append(rel)

                if len(batch_relationships) >= BATCH_SIZE:
                    count = self._execute_neuron_projection_batch(
                        batch_relationships,
                        'Subregion',
                        'acronym'
                    )
                    success_count += count
                    batch_relationships = []

        # 插入剩余
        if batch_relationships:
            count = self._execute_neuron_projection_batch(
                batch_relationships,
                'Subregion',
                'acronym'
            )
            success_count += count

        logger.info(f"成功插入 {success_count} 个Neuron->Subregion投射关系")

    def _execute_neuron_projection_batch(self, batch, target_label, target_id_field):
        """执行投射关系批量插入"""
        if target_label == 'Region':
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (r:Region {region_id: rel.region_id})
            MERGE (n)-[p:PROJECT_TO]->(r)
            SET p.projection_length = rel.projection_length,
                p.target_acronym = rel.target_acronym
            RETURN count(p) as created_count
            """
        else:  # Subregion
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (sr:Subregion {acronym: rel.subregion_acronym})
            MERGE (n)-[p:PROJECT_TO]->(sr)
            SET p.projection_length = rel.projection_length
            RETURN count(p) as created_count
            """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入投射关系失败: {e}")
            return 0

    def generate_and_insert_merfish_subregion_relationships(
        self,
        merfish_cells: pd.DataFrame,
        level: str  # 'subregion' or 'me_subregion'
    ):
        """
        生成并插入MERFISH细胞类型到Subregion/ME_Subregion的HAS关系

        参数:
            merfish_cells: MERFISH细胞数据（已包含subregion/me_subregion映射）
            level: 'subregion' 或 'me_subregion'
        """
        logger.info(f"生成并插入HAS关系到{level.upper()}...")

        if level == 'subregion':
            region_col = 'subregion_acronym'
            node_label = 'Subregion'
            id_field = 'acronym'
        else:  # me_subregion
            region_col = 'me_subregion_acronym'
            node_label = 'ME_Subregion'
            id_field = 'acronym'

        # 检查必要的列
        if region_col not in merfish_cells.columns:
            logger.warning(f"MERFISH数据中没有 {region_col} 列")
            return

        # 过滤有效细胞
        valid_cells = merfish_cells[merfish_cells[region_col].notna()]

        if len(valid_cells) == 0:
            logger.warning(f"没有细胞映射到 {level}")
            return

        logger.info(f"有 {len(valid_cells)} 个细胞映射到 {level}")

        # 对每个细胞类型级别建立HAS关系
        for cell_type_level in ['class', 'subclass', 'supertype', 'cluster']:
            if cell_type_level not in valid_cells.columns:
                continue

            self._insert_has_relationships_to_subregion(
                valid_cells,
                region_col,
                cell_type_level,
                node_label,
                id_field
            )

    def _insert_has_relationships_to_subregion(
        self,
        cells_df: pd.DataFrame,
        region_col: str,
        cell_type_col: str,
        target_label: str,
        target_id_field: str
    ):
        """插入HAS关系到Subregion/ME_Subregion"""
        logger.info(f"插入HAS_{cell_type_col.upper()}关系到{target_label}...")

        # 获取ID映射
        if cell_type_col == 'class':
            id_map = self.class_id_map
        elif cell_type_col == 'subclass':
            id_map = self.subclass_id_map
        elif cell_type_col == 'supertype':
            id_map = self.supertype_id_map
        else:
            id_map = self.cluster_id_map

        # 筛选有效细胞
        valid_cells = cells_df[
            (cells_df[region_col].notna()) &
            (cells_df[cell_type_col].notna())
        ]

        # 按区域和类型分组计数
        counts_df = valid_cells.groupby([region_col, cell_type_col]).size().reset_index(name='count')

        # 添加比例
        region_totals = valid_cells.groupby(region_col).size().reset_index(name='total')
        counts_df = pd.merge(counts_df, region_totals, on=region_col)
        counts_df['pct'] = counts_df['count'] / counts_df['total']

        # 过滤
        counts_df = counts_df[counts_df['pct'] >= PCT_THRESHOLD]

        # 批量插入
        batch_relationships = []

        for region_acronym, group in tqdm(
            counts_df.groupby(region_col),
            desc=f"处理HAS_{cell_type_col.upper()}->{target_label}"
        ):
            group_sorted = group.sort_values('pct', ascending=False)
            rank = 1

            for _, row in group_sorted.iterrows():
                cell_type = row[cell_type_col]

                if cell_type in id_map:
                    rel = {
                        'region_acronym': str(region_acronym),
                        'cell_type_id': id_map[cell_type],
                        'pct_cells': float(row['pct']),
                        'rank': rank
                    }
                    batch_relationships.append(rel)
                    rank += 1

                    if len(batch_relationships) >= BATCH_SIZE:
                        self._execute_has_subregion_batch(
                            batch_relationships,
                            cell_type_col,
                            target_label,
                            target_id_field
                        )
                        batch_relationships = []

        # 插入剩余
        if batch_relationships:
            self._execute_has_subregion_batch(
                batch_relationships,
                cell_type_col,
                target_label,
                target_id_field
            )

        logger.info(f"插入了HAS_{cell_type_col.upper()}关系到{target_label}")

    def _execute_has_subregion_batch(self, batch, cell_type_col, target_label, target_id_field):
        """执行HAS关系批量插入"""
        cell_type_label = cell_type_col.capitalize()

        query = f"""
        UNWIND $batch AS rel
        MATCH (sr:{target_label} {{{target_id_field}: rel.region_acronym}})
        MATCH (ct:{cell_type_label} {{tran_id: rel.cell_type_id}})
        MERGE (sr)-[r:HAS_{cell_type_col.upper()}]->(ct)
        SET r.pct_cells = rel.pct_cells,
            r.rank = rel.rank
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                session.run(query, batch=batch)
        except Exception as e:
            logger.error(f"批量插入HAS关系失败: {e}")


# ==================== 主函数 ====================

def main(data_dir: str = "../data",
         hierarchy_json: str = None,
         neo4j_uri: str = "bolt://localhost:7687",
         neo4j_user: str = "neo4j",
         neo4j_password: str = "password",
         database_name: str = "neuroxiv",
         clear_database: bool = False):
    """
    主函数 - V5完整版本
    """

    from loguru import logger

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - V5完整修复版")
    logger.info("=" * 60)

    # 初始化Neo4j连接
    logger.info("初始化Neo4j连接...")
    neo4j_conn = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password, database_name)

    if not neo4j_conn.connect():
        logger.error("无法连接到Neo4j数据库")
        return

    try:
        # 清空数据库
        if clear_database:
            neo4j_conn.clear_database_smart()

        # 创建约束和索引
        neo4j_conn.create_constraints()

        # Phase 1: 数据加载
        logger.info("Phase 1: 数据加载")
        data_path = Path(data_dir)
        data = load_data(data_path)
        processed_data = prepare_analysis_data(data)

        region_data = processed_data.get('region_data', pd.DataFrame())
        merfish_cells = processed_data.get('merfish_cells', pd.DataFrame())
        projection_data = processed_data.get('projection_df', pd.DataFrame())

        # 创建builder实例
        builder = KnowledgeGraphBuilderNeo4jV5(neo4j_conn)

        # 加载树结构
        tree_data = processed_data.get('tree', [])
        if tree_data:
            builder.region_analyzer = RegionAnalyzer(tree_data)

        # Phase 2: 加载层级数据
        logger.info("Phase 2: 加载MERFISH层级数据")
        hierarchy_loader = MERFISHHierarchyLoader(
            Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json"
        )

        if not hierarchy_loader.load_hierarchy():
            logger.error("无法加载层级数据")
            return

        # Phase 3: 加载形态数据
        logger.info("Phase 3: 加载形态数据")
        morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
        if morphology_loader.load_morphology_data():
            if projection_data is not None and not projection_data.empty:
                morphology_loader.set_projection_data(projection_data)
            builder.morphology_loader = morphology_loader

        # Phase 3.5: 加载神经元数据
        logger.info("Phase 3.5: 加载神经元数据（完整形态学特征）")
        neuron_loader = NeuronDataLoader(
            data_path,
            builder.region_analyzer,
            builder.morphology_loader
        )

        if neuron_loader.load_neuron_data():
            neuron_loader.process_neuron_data()
            logger.info(f"成功加载 {len(neuron_loader.neurons_data)} 个神经元数据")
            logger.info(f"包含 {len(neuron_loader.morph_features)} 个形态学特征")
        else:
            logger.warning("无法加载神经元数据")
            neuron_loader = None

        # Phase 3.6: 处理投射关系
        logger.info("Phase 3.6: 处理神经元投射数据")
        ccf_tree_json = data_path / "tree_yzx.json"  # 或其他包含CCF结构的JSON文件

        projection_processor = NeuronProjectionProcessorV5Fixed(
            data_path=data_path,
            ccf_tree_json=ccf_tree_json
        )

        if projection_processor.run_full_pipeline():
            logger.info("投射数据处理成功")
        else:
            logger.warning("投射数据处理失败")
            projection_processor = None

        # Phase 3.7: MERFISH空间映射
        logger.info("Phase 3.7: MERFISH细胞空间映射")
        merfish_mapper = MERFISHSubregionMapper(data_path, builder.region_analyzer)

        if merfish_mapper.load_region_annotation():
            # 加载ME_Subregion注释
            if merfish_mapper.load_me_subregion_annotation():
                merfish_cells = merfish_mapper.map_cells_to_me_subregions(merfish_cells)

            # 尝试加载Subregion注释
            if merfish_mapper.load_subregion_annotation():
                merfish_cells = merfish_mapper.map_cells_to_subregions(merfish_cells)

        # Phase 3.8: 加载Subregion数据
        logger.info("Phase 3.8: 加载Subregion和ME_Subregion数据")
        ccf_me_json = data_path / "surf_tree_ccf-me.json"
        subregion_loader = SubregionLoader(ccf_me_json)

        if not subregion_loader.load_subregion_data():
            logger.warning("无法加载Subregion数据")
            subregion_loader = None

        # Phase 4: 知识图谱生成和插入
        logger.info("Phase 4: 知识图谱生成和插入")

        builder.set_hierarchy_loader(hierarchy_loader)

        # 插入Region节点
        builder.generate_and_insert_unified_region_nodes(region_data, merfish_cells)

        # 插入Neuron节点（完整形态学特征）
        if neuron_loader:
            builder.generate_and_insert_neuron_nodes_with_full_morphology(neuron_loader)

        # 插入Subregion和ME_Subregion节点
        if subregion_loader:
            builder.generate_and_insert_subregion_nodes(subregion_loader)
            builder.generate_and_insert_me_subregion_nodes(subregion_loader)

        # 插入MERFISH细胞类型节点
        builder.generate_and_insert_merfish_nodes_from_hierarchy(merfish_cells)

        # Phase 5: 插入关系
        logger.info("Phase 5: 插入关系")

        # HAS关系 - Region级别
        for level in ['class', 'subclass', 'supertype', 'cluster']:
            builder.generate_and_insert_has_relationships_unified(merfish_cells, level)

        # HAS关系 - Subregion级别
        if 'subregion_acronym' in merfish_cells.columns:
            builder.generate_and_insert_merfish_subregion_relationships(
                merfish_cells,
                'subregion'
            )

        # HAS关系 - ME_Subregion级别
        if 'me_subregion_acronym' in merfish_cells.columns:
            builder.generate_and_insert_merfish_subregion_relationships(
                merfish_cells,
                'me_subregion'
            )

        # BELONGS_TO关系
        builder.generate_and_insert_belongs_to_from_hierarchy()

        # PROJECT_TO关系 - Region级别
        builder.generate_and_insert_project_to_relationships(projection_data)

        # 神经元关系
        if neuron_loader:
            builder.generate_and_insert_neuron_relationships(neuron_loader)

        # 神经元投射关系
        if projection_processor:
            builder.generate_and_insert_neuron_projection_relationships(projection_processor)

        # Subregion层级关系
        if subregion_loader:
            builder.generate_and_insert_subregion_relationships(subregion_loader)

        # Neuron-Subregion关系
        logger.info("Phase 6: 插入Neuron-Subregion关系")
        neuron_subregion_inserter = NeuronSubregionRelationshipInserter(
            neo4j_conn, data_path
        )

        if neuron_subregion_inserter.load_neuron_subregion_mapping():
            neuron_subregion_inserter.insert_all_relationships(batch_size=BATCH_SIZE)
            verify_relationships(neo4j_conn, database_name)

        # 打印统计报告
        builder.print_statistics_report_enhanced_with_subregion()

        logger.info("=" * 60)
        logger.info("知识图谱构建完成！")
        logger.info("=" * 60)

    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建 - V5完整版')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data')
    parser.add_argument('--hierarchy_json', type=str,
                       default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, required=True, default='neuroxiv')
    parser.add_argument('--database', type=str, default='neo4j')
    parser.add_argument('--clear_database', action='store_true')

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        hierarchy_json=args.hierarchy_json,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        database_name=args.database,
        clear_database=args.clear_database
    )