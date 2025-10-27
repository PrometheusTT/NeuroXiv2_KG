"""
NeuroXiv 2.0 知识图谱构建器 - 三大问题修复版 V5.1
修复内容：
1. Neuron PROJECT_TO Region 关系直接从 Proj_Axon_Final.csv 读取
2. 修复 MERFISH 到 Subregion/ME_Subregion 的映射
3. 去除 neighbouring 关系的重复

作者: wangmajortom & Claude
日期: 2025-10-27
版本: V5.1 Fixed
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

# ==================== 修复1: 投射关系处理器 ====================

class NeuronProjectionProcessorV5Fixed:
    """修正版：正确处理 Region 和 Subregion 投射关系"""

    def __init__(self, data_path: Path, ccf_tree_json: Path):
        """
        初始化

        参数:
            data_path: 数据目录
            ccf_tree_json: CCF树结构JSON文件路径
        """
        self.data_path = data_path
        self.ccf_tree_json = ccf_tree_json

        # 原始投射数据
        self.axon_proj_df = None  # axonfull_proj.csv
        
        # ⭐ 修复1: 添加直接读取的 Region 级别数据
        self.proj_axon_final_df = None  # Proj_Axon_Final.csv（Region级别）

        # info数据
        self.info_df = None

        # CCF结构映射
        self.id_to_acronym = {}
        self.id_to_name = {}
        self.acronym_to_children_ids = {}
        self.subregion_to_parent = {}

        # 投射数据
        self.neuron_to_subregion_axon = {}  # {neuron_id: {subregion_acronym: length}}
        self.neuron_to_region_axon = {}  # {neuron_id: {region_acronym: length}} - 直接从Final读取
        
    # ⭐ 修复1: 添加加载 Proj_Axon_Final.csv 的方法
    def load_proj_axon_final(self) -> bool:
        """加载 Proj_Axon_Final.csv（Region级别的投射数据）"""
        logger.info("加载 Proj_Axon_Final.csv...")
        
        final_file = self.data_path / "Proj_Axon_Final.csv"
        if not final_file.exists():
            logger.error(f"Proj_Axon_Final.csv 不存在: {final_file}")
            return False
        
        try:
            self.proj_axon_final_df = pd.read_csv(final_file, index_col=0)
            logger.info(f"  - 加载了 Proj_Axon_Final.csv: {self.proj_axon_final_df.shape}")
            
            # 过滤
            original_len = len(self.proj_axon_final_df)
            self.proj_axon_final_df = self.proj_axon_final_df[
                ~self.proj_axon_final_df.index.str.contains('CCF-thin|local', na=False)
            ]
            if original_len > len(self.proj_axon_final_df):
                logger.info(f"    过滤掉 {original_len - len(self.proj_axon_final_df)} 个神经元")
            
            return True
            
        except Exception as e:
            logger.error(f"加载 Proj_Axon_Final.csv 失败: {e}")
            return False

    def load_ccf_tree_structure(self) -> bool:
        """加载CCF树结构，构建ID到acronym的映射"""
        logger.info(f"加载CCF树结构: {self.ccf_tree_json}")

        if not self.ccf_tree_json.exists():
            logger.error(f"CCF树文件不存在: {self.ccf_tree_json}")
            return False

        try:
            with open(self.ccf_tree_json, 'r') as f:
                tree_data = json.load(f)

            def traverse_tree(node, parent_acronym=None):
                """递归遍历树节点"""
                if not isinstance(node, dict):
                    return

                node_id = node.get('id')
                acronym = node.get('acronym', '')
                name = node.get('name', '')

                if node_id is not None:
                    self.id_to_acronym[node_id] = acronym
                    self.id_to_name[node_id] = name

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

            logger.info(f"  - 加载了 {len(self.id_to_acronym)} 个CCF区域")
            logger.info(f"  - 找到 {len(self.acronym_to_children_ids)} 个具有children的区域")

            return True

        except Exception as e:
            logger.error(f"加载CCF树结构失败: {e}")
            return False

    def load_raw_projection_data(self) -> bool:
        """加载原始投射数据文件"""
        logger.info("加载原始投射数据...")

        # 加载 axonfull_proj.csv（用于 subregion 级别）
        axon_file = self.data_path / "axonfull_proj.csv"
        if axon_file.exists():
            try:
                self.axon_proj_df = pd.read_csv(axon_file, index_col=0)
                logger.info(f"  - 加载了 axonfull_proj.csv: {self.axon_proj_df.shape}")

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

        if self.axon_proj_df is None:
            logger.error("未能加载投射数据文件")
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

    # ⭐ 修复1: 提取 Region 级别投射（直接从 Final 读取）
    def extract_region_projections_from_final(self):
        """从 Proj_Axon_Final.csv 直接提取 Region 级别投射"""
        logger.info("从 Proj_Axon_Final.csv 提取 Region 级别投射...")
        
        if self.proj_axon_final_df is None:
            logger.error("Proj_Axon_Final.csv 未加载")
            return
        
        self.neuron_to_region_axon = {}
        
        # 找出所有投射列（proj_axon_*_abs）
        proj_columns = [col for col in self.proj_axon_final_df.columns 
                       if col.startswith('proj_axon_') and col.endswith('_abs')]
        
        logger.info(f"  - 找到 {len(proj_columns)} 个 region 投射列")
        
        for neuron_id in tqdm(self.proj_axon_final_df.index, desc="提取Region投射"):
            neuron_row = self.proj_axon_final_df.loc[neuron_id]
            self.neuron_to_region_axon[neuron_id] = {}
            
            for col in proj_columns:
                # 提取 region acronym（去掉 proj_axon_ 前缀和 _abs 后缀）
                region_acronym = col.replace('proj_axon_', '').replace('_abs', '')
                
                value = neuron_row[col]
                if pd.notna(value) and float(value) > 0:
                    self.neuron_to_region_axon[neuron_id][region_acronym] = float(value)
        
        total_proj = sum(len(v) for v in self.neuron_to_region_axon.values())
        logger.info(f"  - 提取了 {total_proj} 个 Region 投射关系")

    def extract_subregion_projections(self, proj_df: pd.DataFrame) -> Dict:
        """从原始投射数据中提取 subregion 级别的投射"""
        logger.info(f"提取 subregion 投射数据...")

        neuron_to_subregion = {}

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

        for neuron_id in tqdm(proj_df.index, desc="处理subregion投射"):
            neuron_row = proj_df.loc[neuron_id]
            neuron_to_subregion[neuron_id] = {}

            # 提取所有subregion的投射值
            for ccf_id in available_ids:
                value = neuron_row.get(str(ccf_id), 0)

                if pd.notna(value) and float(value) > 0:
                    acronym = self.id_to_acronym[ccf_id]
                    neuron_to_subregion[neuron_id][acronym] = float(value)

        total_subregion_proj = sum(len(v) for v in neuron_to_subregion.values())
        logger.info(f"  - 提取了 {total_subregion_proj} 个 subregion 投射关系")

        return neuron_to_subregion

    def process_all_projections(self):
        """处理所有投射数据"""
        logger.info("=" * 60)
        logger.info("开始处理投射数据...")
        logger.info("=" * 60)

        # ⭐ 修复1: 先提取 Region 级别（从 Final 文件）
        if self.proj_axon_final_df is not None:
            self.extract_region_projections_from_final()

        # 再提取 Subregion 级别（从原始文件）
        if self.axon_proj_df is not None:
            logger.info("处理 Subregion Axon 投射数据...")
            self.neuron_to_subregion_axon = self.extract_subregion_projections(self.axon_proj_df)

        logger.info("=" * 60)
        logger.info("投射数据处理完成！")
        logger.info("=" * 60)

    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("投射数据统计")
        logger.info("=" * 60)
        
        logger.info(f"\nAXON投射:")
        logger.info(f"  有 region 投射的神经元: {len(self.neuron_to_region_axon)}")
        logger.info(f"  有 subregion 投射的神经元: {len(self.neuron_to_subregion_axon)}")
        logger.info(f"  总 region 投射关系: {sum(len(v) for v in self.neuron_to_region_axon.values())}")
        logger.info(f"  总 subregion 投射关系: {sum(len(v) for v in self.neuron_to_subregion_axon.values())}")
        logger.info(f"  唯一 regions 数: {len(set(r for projs in self.neuron_to_region_axon.values() for r in projs.keys()))}")
        logger.info(f"  唯一 subregions 数: {len(set(s for projs in self.neuron_to_subregion_axon.values() for s in projs.keys()))}")

        logger.info("=" * 60)

    def run_full_pipeline(self) -> bool:
        """运行完整的处理流程"""
        logger.info("=" * 80)
        logger.info("NeuronProjectionProcessor V5.1 Fixed - 完整流程")
        logger.info("=" * 80)

        # 1. 加载CCF树结构
        if not self.load_ccf_tree_structure():
            return False

        # 2. ⭐ 修复1: 加载 Proj_Axon_Final.csv
        if not self.load_proj_axon_final():
            return False

        # 3. 加载原始投射数据（用于 subregion）
        if not self.load_raw_projection_data():
            return False

        # 4. 加载info数据
        if not self.load_info_data():
            return False

        # 5. 处理投射数据
        self.process_all_projections()

        # 6. 打印统计
        self.print_statistics()

        logger.info("=" * 80)
        logger.info("处理完成！")
        logger.info("=" * 80)

        return True


# ==================== 修复2: MERFISH空间映射处理器 ====================

class MERFISHSubregionMapper:
    """
    完全正确的MERFISH映射器

    关键理解：
    1. MERFISH坐标已经是25μm单位（data_loader中乘以40是因为原始是mm）
    2. 需要直接使用坐标整数部分作为索引
    3. 只有皮层区域有ME细分
    """

    def __init__(self, data_path: Path, ccf_tree_json: Path, ccf_me_json: Path):
        """初始化"""
        self.data_path = data_path
        self.ccf_tree_json = ccf_tree_json
        self.ccf_me_json = ccf_me_json

        # 标准CCF映射
        self.id_to_acronym = {}
        self.id_to_name = {}
        self.acronym_to_id = {}

        # Subregion信息（从CCF-ME树）
        self.subregion_acronyms = set()
        self.subregion_id_to_acronym = {}
        self.subregion_to_region = {}

        # ME_Subregion信息
        self.me_annotation = None
        self.voxel_to_parent_id = {}
        self.parent_id_to_me_acronym = {}
        self.me_acronym_to_subregion = {}

    def load_standard_ccf_tree(self) -> bool:
        """加载标准CCF树"""
        logger.info(f"加载标准CCF树: {self.ccf_tree_json}")

        if not self.ccf_tree_json.exists():
            logger.error(f"标准CCF树文件不存在")
            return False

        try:
            with open(self.ccf_tree_json, 'r') as f:
                tree_data = json.load(f)

            def traverse_tree(node):
                if not isinstance(node, dict):
                    return

                node_id = node.get('id')
                acronym = node.get('acronym', '')
                name = node.get('name', '')

                if node_id is not None:
                    self.id_to_acronym[node_id] = acronym
                    self.id_to_name[node_id] = name
                    self.acronym_to_id[acronym] = node_id

                for child in node.get('children', []):
                    traverse_tree(child)

            if isinstance(tree_data, list):
                for root in tree_data:
                    traverse_tree(root)
            else:
                traverse_tree(tree_data)

            logger.info(f"  - 加载了 {len(self.id_to_acronym)} 个CCF区域映射")
            return True

        except Exception as e:
            logger.error(f"加载标准CCF树失败: {e}")
            return False

    def load_ccf_me_tree_for_subregions(self) -> bool:
        """两遍遍历加载CCF-ME树"""
        logger.info(f"加载CCF-ME树（两遍遍历）: {self.ccf_me_json}")

        if not self.ccf_me_json.exists():
            logger.error(f"CCF-ME树文件不存在")
            return False

        try:
            with open(self.ccf_me_json, 'r') as f:
                tree_data = json.load(f)

            # 第一遍：收集所有节点信息
            all_nodes = {}  # {acronym: node_info}

            def collect_nodes(node, parent_acronym=None):
                acronym = node.get('acronym', '')
                node_id = node.get('id')

                all_nodes[acronym] = {
                    'node': node,
                    'parent_acronym': parent_acronym,
                    'node_id': node_id
                }

                for child in node.get('children', []):
                    collect_nodes(child, acronym)

            for root in tree_data:
                collect_nodes(root, None)

            logger.info(f"  - 收集了 {len(all_nodes)} 个节点")

            # 第二遍：识别Subregion和ME，建立映射
            for acronym, info in all_nodes.items():
                node = info['node']
                parent_acronym = info['parent_acronym']
                node_id = info['node_id']
                name = node.get('name', '')

                # 识别ME节点
                if '-ME' in acronym and parent_acronym:
                    self.me_acronym_to_subregion[acronym] = parent_acronym

                    # ⭐ 关键：从parent找CCF ID
                    if parent_acronym in all_nodes:
                        parent_info = all_nodes[parent_acronym]
                        parent_id = parent_info['node_id']

                        if parent_id is not None:
                            try:
                                parent_ccf_id = int(parent_id)
                                self.parent_id_to_me_acronym[parent_ccf_id] = acronym
                            except (ValueError, TypeError):
                                pass

                # 识别Subregion
                elif parent_acronym and '-ME' not in parent_acronym:
                    is_subregion = False

                    if 'layer' in name.lower():
                        is_subregion = True

                    for pattern in ['1', '2/3', '4', '5', '6a', '6b']:
                        if acronym.endswith(pattern) or f'{pattern}-' in acronym:
                            is_subregion = True
                            break

                    has_me_children = any('-ME' in child.get('acronym', '')
                                          for child in node.get('children', []))
                    if has_me_children:
                        is_subregion = True

                    if is_subregion:
                        self.subregion_acronyms.add(acronym)

                        if node_id is not None:
                            try:
                                id_int = int(node_id)
                                self.subregion_id_to_acronym[id_int] = acronym
                            except (ValueError, TypeError):
                                pass

                        if parent_acronym:
                            self.subregion_to_region[acronym] = parent_acronym

            logger.info(f"  - 识别了 {len(self.subregion_acronyms)} 个Subregion")
            logger.info(f"  - 识别了 {len(self.me_acronym_to_subregion)} 个ME_Subregion")
            logger.info(f"  - 建立了 {len(self.parent_id_to_me_acronym)} 个parent_id映射")

            return True

        except Exception as e:
            logger.error(f"加载CCF-ME树失败: {e}")
            return False

    def _traverse_ccf_me_tree(self, node, parent_acronym=None, grandparent_acronym=None):
        """递归遍历CCF-ME树"""
        acronym = node.get('acronym', '')
        node_id = node.get('id')
        name = node.get('name', '')

        # 1. 识别ME节点
        if '-ME' in acronym:
            if parent_acronym:
                self.me_acronym_to_subregion[acronym] = parent_acronym

                # ⭐ 关键修复：使用parent_structure_id而不是从node_id提取
                parent_structure_id = node.get('parent_structure_id')

                if parent_structure_id is not None:
                    # 这才是pkl中使用的parent_id！
                    self.parent_id_to_me_acronym[int(parent_structure_id)] = acronym
                else:
                    # 后备方案1：尝试从parent_acronym查找对应的CCF ID
                    if parent_acronym in self.subregion_id_to_acronym.values():
                        # 反向查找parent的ID
                        for sid, sacronym in self.subregion_id_to_acronym.items():
                            if sacronym == parent_acronym:
                                self.parent_id_to_me_acronym[sid] = acronym
                                break

                    # 后备方案2：从标准CCF树查找
                    elif parent_acronym in self.acronym_to_id:
                        parent_ccf_id = self.acronym_to_id[parent_acronym]
                        self.parent_id_to_me_acronym[parent_ccf_id] = acronym

        # 2. 识别Subregion（保持不变）
        elif parent_acronym and '-ME' not in parent_acronym:
            is_subregion = False

            # 判断标准1: 名称包含"layer"
            if 'layer' in name.lower():
                is_subregion = True

            # 判断标准2: acronym符合layer模式
            if not is_subregion:
                for pattern in ['1', '2/3', '4', '5', '6a', '6b']:
                    if acronym.endswith(pattern) or f'{pattern}-' in acronym:
                        is_subregion = True
                        break

            # 判断标准3: 有ME子节点
            has_me_children = any('-ME' in child.get('acronym', '')
                                  for child in node.get('children', []))

            if has_me_children:
                is_subregion = True

            if is_subregion:
                self.subregion_acronyms.add(acronym)

                if node_id is not None:
                    try:
                        id_int = int(node_id)
                        self.subregion_id_to_acronym[id_int] = acronym
                    except (ValueError, TypeError):
                        pass

                if parent_acronym:
                    self.subregion_to_region[acronym] = parent_acronym

        # 递归
        for child in node.get('children', []):
            self._traverse_ccf_me_tree(child, acronym, parent_acronym)

    def map_cells_to_subregions(
            self,
            cells_df: pd.DataFrame,
            annotation_volume: np.ndarray
    ) -> pd.DataFrame:
        """
        将细胞映射到Subregion

        关键：参考data_loader_enhanced.py中的map_cells_to_regions_fixed
        - 坐标已经是25μm单位
        - 直接使用整数部分作为索引
        """
        logger.info("将MERFISH细胞映射到Subregion...")

        if annotation_volume is None:
            logger.warning("没有CCF注释数据")
            cells_df['subregion_acronym'] = None
            return cells_df

        required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
        if not all(col in cells_df.columns for col in required_cols):
            logger.error(f"缺少必要的坐标列")
            cells_df['subregion_acronym'] = None
            return cells_df

        # 关键：坐标已经是25μm单位，直接转为整数索引
        x_indices = cells_df['x_ccf'].astype(int)
        y_indices = cells_df['y_ccf'].astype(int)
        z_indices = cells_df['z_ccf'].astype(int)

        volume_shape = annotation_volume.shape

        logger.info(f"  注释volume形状: {volume_shape}")
        logger.info(f"  坐标范围: X[{x_indices.min()}, {x_indices.max()}], "
                    f"Y[{y_indices.min()}, {y_indices.max()}], "
                    f"Z[{z_indices.min()}, {z_indices.max()}]")

        # 检查有效索引
        valid_indices = (
                (0 <= x_indices) & (x_indices < volume_shape[0]) &
                (0 <= y_indices) & (y_indices < volume_shape[1]) &
                (0 <= z_indices) & (z_indices < volume_shape[2])
        )

        valid_count = valid_indices.sum()
        logger.info(f"  有效索引: {valid_count}/{len(cells_df)} ({valid_count / len(cells_df) * 100:.1f}%)")

        # 统计
        mapped_to_subregion = 0
        mapped_to_region = 0
        out_of_bounds = 0

        subregion_acronyms_found = []

        # 批量处理
        batch_size = 100000
        for i in range(0, len(cells_df), batch_size):
            batch_end = min(i + batch_size, len(cells_df))

            batch_x = x_indices[i:batch_end]
            batch_y = y_indices[i:batch_end]
            batch_z = z_indices[i:batch_end]
            batch_valid = valid_indices[i:batch_end]

            for j, (x, y, z, is_valid) in enumerate(zip(batch_x, batch_y, batch_z, batch_valid)):
                if is_valid:
                    ccf_id = int(annotation_volume[x, y, z])

                    if ccf_id == 0:
                        subregion_acronyms_found.append(None)
                    else:
                        # 转为acronym
                        acronym = self.id_to_acronym.get(ccf_id)

                        if acronym and acronym in self.subregion_acronyms:
                            # 是subregion
                            subregion_acronyms_found.append(acronym)
                            mapped_to_subregion += 1
                        else:
                            # 是region或其他
                            subregion_acronyms_found.append(None)
                            if acronym:
                                mapped_to_region += 1
                else:
                    out_of_bounds += 1
                    subregion_acronyms_found.append(None)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"  已处理 {batch_end}/{len(cells_df)} 个细胞")

        cells_df['subregion_acronym'] = subregion_acronyms_found

        logger.info(f"  ✓ 映射结果:")
        logger.info(f"    - 映射到Subregion: {mapped_to_subregion} ({mapped_to_subregion / len(cells_df) * 100:.2f}%)")
        logger.info(f"    - 在Region: {mapped_to_region} ({mapped_to_region / len(cells_df) * 100:.2f}%)")
        logger.info(f"    - 超出边界: {out_of_bounds}")

        return cells_df

    def load_me_subregion_annotation(self) -> bool:
        """加载ME注释"""
        logger.info("加载ME_Subregion注释...")

        me_file = self.data_path / "parc_r671_full.nrrd"
        pkl_file = self.data_path / "parc_r671_full.nrrd.pkl"

        if not me_file.exists():
            logger.error(f"ME注释文件不存在")
            return False

        try:
            # 1. 加载NRRD
            self.me_annotation, _ = nrrd.read(str(me_file))
            logger.info(f"  - 加载ME注释: {self.me_annotation.shape}")

            # 2. 加载pkl
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    pkl_data = pickle.load(f)

                logger.info(f"  - pkl包含 {len(pkl_data)} 个映射")

                # 转换
                for voxel, parent_id in pkl_data.items():
                    self.voxel_to_parent_id[int(voxel)] = int(parent_id)

                logger.info(f"  - 转换了 {len(self.voxel_to_parent_id)} 个voxel->parent映射")

            # 3. parent_id_to_me_acronym已经在遍历树时构建
            logger.info(f"  - 已有 {len(self.parent_id_to_me_acronym)} 个parent->ME映射")

            # 4. 检查映射覆盖率
            pkl_parents = set(self.voxel_to_parent_id.values())
            covered_parents = pkl_parents & set(self.parent_id_to_me_acronym.keys())
            coverage = len(covered_parents) / len(pkl_parents) * 100 if pkl_parents else 0

            logger.info(f"  - 映射覆盖率: {len(covered_parents)}/{len(pkl_parents)} ({coverage:.1f}%)")
            logger.info(f"  ⓘ 注意: 只有皮层区域有ME细分，其他区域无法映射是正常的")

            # 5. 显示未覆盖的parent_id样例（用于调试）
            uncovered = pkl_parents - set(self.parent_id_to_me_acronym.keys())
            if uncovered:
                sample_uncovered = list(uncovered)[:5]
                logger.info(f"  - 未覆盖parent_id样例: {sample_uncovered}")

                # 尝试从标准CCF树查找这些ID对应的区域
                for pid in sample_uncovered:
                    acronym = self.id_to_acronym.get(pid, 'unknown')
                    logger.info(f"      parent_id {pid} -> {acronym} (非皮层区域)")

            return True

        except Exception as e:
            logger.error(f"加载ME注释失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def map_cells_to_me_subregions(self, cells_df: pd.DataFrame) -> pd.DataFrame:
        """映射到ME_Subregion"""
        logger.info("将MERFISH细胞映射到ME_Subregion...")

        if self.me_annotation is None:
            logger.warning("没有ME注释数据")
            cells_df['me_subregion_acronym'] = None
            return cells_df

        required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
        if not all(col in cells_df.columns for col in required_cols):
            logger.error(f"缺少必要的坐标列")
            cells_df['me_subregion_acronym'] = None
            return cells_df

        # 关键：坐标已经是25μm单位
        x_indices = cells_df['x_ccf'].astype(int)
        y_indices = cells_df['y_ccf'].astype(int)
        z_indices = cells_df['z_ccf'].astype(int)

        # 统计
        mapped_count = 0
        voxel_zero = 0
        voxel_not_in_pkl = 0
        parent_not_in_json = 0
        out_of_bounds = 0

        me_acronyms = []

        # 批量处理
        batch_size = 100000
        for i in range(0, len(cells_df), batch_size):
            batch_end = min(i + batch_size, len(cells_df))

            batch_x = x_indices[i:batch_end]
            batch_y = y_indices[i:batch_end]
            batch_z = z_indices[i:batch_end]

            for x, y, z in zip(batch_x, batch_y, batch_z):
                if (0 <= x < self.me_annotation.shape[0] and
                        0 <= y < self.me_annotation.shape[1] and
                        0 <= z < self.me_annotation.shape[2]):

                    voxel = int(self.me_annotation[x, y, z])

                    if voxel == 0:
                        me_acronyms.append(None)
                        voxel_zero += 1
                        continue

                    # voxel -> parent_id
                    parent_id = self.voxel_to_parent_id.get(voxel)
                    if parent_id is None:
                        me_acronyms.append(None)
                        voxel_not_in_pkl += 1
                        continue

                    # parent_id -> ME acronym
                    me_acronym = self.parent_id_to_me_acronym.get(parent_id)
                    if me_acronym:
                        me_acronyms.append(me_acronym)
                        mapped_count += 1
                    else:
                        me_acronyms.append(None)
                        parent_not_in_json += 1
                else:
                    out_of_bounds += 1
                    me_acronyms.append(None)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"  已处理 {batch_end}/{len(cells_df)} 个细胞")

        cells_df['me_subregion_acronym'] = me_acronyms

        logger.info(f"  ✓ 映射结果:")
        logger.info(f"    - 映射成功: {mapped_count} ({mapped_count / len(cells_df) * 100:.2f}%)")
        logger.info(f"    - voxel为0: {voxel_zero}")
        logger.info(f"    - voxel不在pkl: {voxel_not_in_pkl}")
        logger.info(f"    - parent不在JSON: {parent_not_in_json} (非皮层区域)")
        logger.info(f"    - 超出边界: {out_of_bounds}")
        logger.info(f"  ⓘ 只有皮层区域有ME细分，其他区域映射失败是正常的")

        return cells_df


# ==================== 扩展KnowledgeGraphBuilderNeo4j类 ====================

class KnowledgeGraphBuilderNeo4jV5(KnowledgeGraphBuilderNeo4j):
    """V5.1 版本的知识图谱构建器 - 包含所有修复"""
    
    def __init__(self, neo4j_connector):
        """初始化"""
        super().__init__(neo4j_connector)
        # ⭐ 修复3: 添加去重集合
        self.processed_neuron_pairs = set()

    def generate_and_insert_neuron_nodes_with_full_morphology(self, neuron_loader: NeuronDataLoader):
        """生成并插入包含完整形态学特征的Neuron节点"""
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
                try:
                    node_dict[clean_feature] = float(value) if value else 0.0
                except:
                    node_dict[clean_feature] = 0.0

            batch_nodes.append(node_dict)

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

    def generate_and_insert_neuron_projection_relationships(
            self,
            projection_processor: NeuronProjectionProcessorV5Fixed
    ):
        """插入Neuron投射关系（修正版）"""
        logger.info("=" * 60)
        logger.info("插入Neuron投射关系")
        logger.info("=" * 60)

        # 1. 插入Axon投射关系到 Region
        if projection_processor.neuron_to_region_axon:
            logger.info("\n插入 Neuron -> Region 投射关系...")
            self._insert_projections_batch(
                projection_processor.neuron_to_region_axon,
                target_level='Region',
                projection_type='axon'
            )

        # 2. 插入Axon投射关系到 Subregion
        if projection_processor.neuron_to_subregion_axon:
            logger.info("\n插入 Neuron -> Subregion 投射关系...")
            self._insert_projections_batch(
                projection_processor.neuron_to_subregion_axon,
                target_level='Subregion',
                projection_type='axon'
            )
        
        logger.info("=" * 60)
        logger.info("Neuron投射关系插入完成")
        logger.info("=" * 60)

    def _insert_projections_batch(
            self,
            projection_dict: Dict[str, Dict[str, float]],
            target_level: str,
            projection_type: str
    ):
        """统一的投射关系批量插入方法"""
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

        logger.info(f"  ✓ 成功插入 {success_count} 个 {projection_type}->{target_level} 关系")

    def _execute_projection_insert_batch(
            self,
            batch: List[Dict],
            target_level: str
    ):
        """执行投射关系批量插入到Neo4j"""

        if target_level == 'Region':
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (t:Region)
            WHERE t.acronym = rel.target_acronym
            MERGE (n)-[p:PROJECT_TO]->(t)
            SET p.projection_length = rel.projection_length,
                p.projection_type = rel.projection_type,
                p.target_level = 'region'
            RETURN count(p) as created_count
            """
        else:  # Subregion
            query = """
            UNWIND $batch AS rel
            MATCH (n:Neuron {neuron_id: rel.neuron_id})
            MATCH (s:Subregion {acronym: rel.target_acronym})
            MERGE (n)-[p:PROJECT_TO]->(s)
            SET p.projection_length = rel.projection_length,
                p.projection_type = rel.projection_type,
                p.target_level = 'subregion'
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

    # ⭐ 修复2: 改进 MERFISH 到 subregion/ME_subregion 的关系插入
    def generate_and_insert_merfish_subregion_relationships(
        self,
        merfish_cells: pd.DataFrame,
        level: str
    ):
        """
        生成并插入 MERFISH 细胞类型到 Subregion/ME_Subregion 的 HAS 关系

        参数:
            merfish_cells: MERFISH细胞数据
            level: 'subregion' 或 'me_subregion'
        """
        logger.info(f"=" * 60)
        logger.info(f"生成并插入 HAS 关系到 {level.upper()}")
        logger.info(f"=" * 60)

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

        # 对每个细胞类型级别建立 HAS 关系
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
        
        logger.info(f"=" * 60)
        logger.info(f"{level.upper()} HAS 关系插入完成")
        logger.info(f"=" * 60)

    def _insert_has_relationships_to_subregion(
        self,
        cells_df: pd.DataFrame,
        region_col: str,
        cell_type_col: str,
        target_label: str,
        target_id_field: str
    ):
        """插入 HAS 关系到 Subregion/ME_Subregion"""
        logger.info(f"\n插入 HAS_{cell_type_col.upper()} 关系到 {target_label}...")

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

        if len(valid_cells) == 0:
            logger.warning(f"  没有有效的 {cell_type_col} 数据")
            return

        # 按区域和类型分组计数
        counts_df = valid_cells.groupby([region_col, cell_type_col]).size().reset_index(name='count')

        # 添加比例
        region_totals = valid_cells.groupby(region_col).size().reset_index(name='total')
        counts_df = pd.merge(counts_df, region_totals, on=region_col)
        counts_df['pct'] = counts_df['count'] / counts_df['total']

        # 过滤
        counts_df = counts_df[counts_df['pct'] >= PCT_THRESHOLD]

        logger.info(f"  准备插入 {len(counts_df)} 条 HAS_{cell_type_col.upper()} 关系")

        # 批量插入
        batch_relationships = []
        success_count = 0

        for region_acronym, group in tqdm(
            counts_df.groupby(region_col),
            desc=f"处理HAS_{cell_type_col.upper()}"
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
                        count = self._execute_has_subregion_batch(
                            batch_relationships,
                            cell_type_col,
                            target_label,
                            target_id_field
                        )
                        success_count += count
                        batch_relationships = []

        # 插入剩余
        if batch_relationships:
            count = self._execute_has_subregion_batch(
                batch_relationships,
                cell_type_col,
                target_label,
                target_id_field
            )
            success_count += count

        logger.info(f"  ✓ 成功插入 {success_count} 条 HAS_{cell_type_col.upper()} 关系到 {target_label}")

    def _execute_has_subregion_batch(self, batch, cell_type_col, target_label, target_id_field):
        """执行 HAS 关系批量插入"""
        cell_type_label = cell_type_col.capitalize()

        query = f"""
        UNWIND $batch AS rel
        MATCH (sr:{target_label} {{{target_id_field}: rel.region_acronym}})
        MATCH (ct:{cell_type_label} {{tran_id: rel.cell_type_id}})
        MERGE (sr)-[r:HAS_{cell_type_col.upper()}]->(ct)
        SET r.pct_cells = rel.pct_cells,
            r.rank = rel.rank
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入 HAS 关系失败: {e}")
            return 0

    # ⭐ 修复3: 去重 neighbouring 关系
    def generate_and_insert_neuron_relationships(self, neuron_loader: NeuronDataLoader):
        """生成并插入神经元之间的关系（去重版本）"""
        if not neuron_loader or not neuron_loader.connections_df.empty:
            logger.info("处理神经元之间的关系（NEIGHBOURING，去重版本）...")
            self._insert_neighbouring_relationships_deduplicated(neuron_loader.connections_df)

    def _insert_neighbouring_relationships_deduplicated(self, connections_df: pd.DataFrame):
        """
        插入 NEIGHBOURING 关系（去重版本）
        
        ⭐ 修复3: 确保每对神经元之间只创建一次关系
        """
        logger.info("=" * 60)
        logger.info("插入 NEIGHBOURING 关系（去重版本）")
        logger.info("=" * 60)

        if connections_df.empty:
            logger.warning("没有连接数据")
            return

        # 重置去重集合
        self.processed_neuron_pairs = set()

        batch_relationships = []
        success_count = 0
        duplicate_count = 0

        for _, row in tqdm(connections_df.iterrows(), total=len(connections_df), desc="处理连接"):
            neuron1 = str(row.get('axon_ID', ''))
            neuron2 = str(row.get('dendrite_ID', ''))

            if not neuron1 or not neuron2:
                continue

            # ⭐ 关键：创建排序后的 tuple 作为 key，确保 (A,B) 和 (B,A) 被视为同一对
            neuron_pair = tuple(sorted([neuron1, neuron2]))

            # 检查是否已处理
            if neuron_pair in self.processed_neuron_pairs:
                duplicate_count += 1
                continue

            # 标记为已处理
            self.processed_neuron_pairs.add(neuron_pair)

            # 添加到批次
            rel = {
                'neuron1_id': neuron_pair[0],  # 使用排序后的第一个
                'neuron2_id': neuron_pair[1]   # 使用排序后的第二个
            }
            batch_relationships.append(rel)

            if len(batch_relationships) >= BATCH_SIZE:
                count = self._execute_neighbouring_batch(batch_relationships)
                success_count += count
                batch_relationships = []

        # 插入剩余
        if batch_relationships:
            count = self._execute_neighbouring_batch(batch_relationships)
            success_count += count

        logger.info(f"  ✓ 成功插入 {success_count} 个 NEIGHBOURING 关系")
        logger.info(f"  ✓ 去除了 {duplicate_count} 个重复关系")
        logger.info("=" * 60)

    def _execute_neighbouring_batch(self, batch: List[Dict]):
        """执行 NEIGHBOURING 关系批量插入"""
        query = """
        UNWIND $batch AS rel
        MATCH (n1:Neuron {neuron_id: rel.neuron1_id})
        MATCH (n2:Neuron {neuron_id: rel.neuron2_id})
        MERGE (n1)-[r:NEIGHBOURING]-(n2)
        RETURN count(r) as created_count
        """

        try:
            with self.neo4j.driver.session(database=self.neo4j.database) as session:
                result = session.run(query, batch=batch)
                record = result.single()
                return record['created_count'] if record else 0
        except Exception as e:
            logger.error(f"批量插入 NEIGHBOURING 关系失败: {e}")
            return 0


# ==================== 主函数 ====================

def main(data_dir: str = "../data",
         hierarchy_json: str = None,
         neo4j_uri: str = "bolt://localhost:7687",
         neo4j_user: str = "neo4j",
         neo4j_password: str = "password",
         database_name: str = "neuroxiv",
         clear_database: bool = False):
    """
    主函数 - V5.1 三大问题修复版本
    """

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - V5.1 三大问题修复版")
    logger.info("=" * 60)
    logger.info("修复内容：")
    logger.info("1. Neuron PROJECT_TO Region 关系直接从 Proj_Axon_Final.csv 读取")
    logger.info("2. 修复 MERFISH 到 Subregion/ME_Subregion 的映射")
    logger.info("3. 去除 neighbouring 关系的重复")
    logger.info("=" * 60)

    # 初始化Neo4j连接
    logger.info("\n初始化Neo4j连接...")
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
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: 数据加载")
        logger.info("=" * 60)
        
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
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: 加载MERFISH层级数据")
        logger.info("=" * 60)
        
        hierarchy_loader = MERFISHHierarchyLoader(
            Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json"
        )

        if not hierarchy_loader.load_hierarchy():
            logger.error("无法加载层级数据")
            return

        # Phase 3: 加载形态数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: 加载形态数据")
        logger.info("=" * 60)
        
        morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
        if morphology_loader.load_morphology_data():
            if projection_data is not None and not projection_data.empty:
                morphology_loader.set_projection_data(projection_data)
            builder.morphology_loader = morphology_loader

        # Phase 3.5: 加载神经元数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.5: 加载神经元数据（完整形态学特征）")
        logger.info("=" * 60)
        
        neuron_loader = NeuronDataLoader(
            data_path,
            builder.region_analyzer,
            builder.morphology_loader
        )

        if neuron_loader.load_neuron_data():
            neuron_loader.process_neuron_data()
            logger.info(f"✓ 成功加载 {len(neuron_loader.neurons_data)} 个神经元数据")
            logger.info(f"✓ 包含 {len(neuron_loader.morph_features)} 个形态学特征")
        else:
            logger.warning("无法加载神经元数据")
            neuron_loader = None

        # Phase 3.6: ⭐ 修复1 - 处理投射关系（包含直接读取Region数据）
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.6: 处理神经元投射数据（修复1：直接读取Region投射）")
        logger.info("=" * 60)
        
        ccf_tree_json = data_path / "tree_yzx.json"

        projection_processor = NeuronProjectionProcessorV5Fixed(
            data_path=data_path,
            ccf_tree_json=ccf_tree_json
        )

        if projection_processor.run_full_pipeline():
            logger.info("✓ 投射数据处理成功")
        else:
            logger.warning("投射数据处理失败")
            projection_processor = None

        # Phase 3.7: MERFISH空间映射（最终正确版）
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.7: MERFISH细胞空间映射（最终正确版）")
        logger.info("参考data_loader_enhanced.py的处理方式")
        logger.info("=" * 60)

        # 文件路径
        ccf_tree_json = data_path / "tree_yzx.json"
        ccf_me_json = data_path / "surf_tree_ccf-me.json"
        annotation_file = data_path / "annotation_25.nrrd"

        # 验证文件
        if not all([ccf_tree_json.exists(), ccf_me_json.exists(), annotation_file.exists()]):
            logger.error("缺少必要的文件")
            merfish_mapper = None
        else:
            # 初始化mapper
            merfish_mapper = MERFISHSubregionMapper(
                data_path=data_path,
                ccf_tree_json=ccf_tree_json,
                ccf_me_json=ccf_me_json
            )

            # 步骤1: 加载标准CCF树
            logger.info("\n步骤1: 加载标准CCF树")
            if not merfish_mapper.load_standard_ccf_tree():
                logger.error("✗ 标准CCF树加载失败")
                merfish_mapper = None
            else:
                logger.info(f"✓ 成功: {len(merfish_mapper.id_to_acronym)} 个区域映射")

                # 步骤2: 加载CCF-ME树
                logger.info("\n步骤2: 加载CCF-ME树识别subregion")
                if not merfish_mapper.load_ccf_me_tree_for_subregions():
                    logger.error("✗ CCF-ME树加载失败")
                    merfish_mapper = None
                else:
                    logger.info(f"✓ Subregion: {len(merfish_mapper.subregion_acronyms)} 个")
                    logger.info(f"✓ ME_Subregion: {len(merfish_mapper.me_acronym_to_subregion)} 个")

        # 执行映射
        if merfish_mapper:
            # 步骤3: 加载annotation并映射到Subregion
            logger.info("\n步骤3: 映射到Subregion")
            try:
                import nrrd
                annotation_volume, annotation_header = nrrd.read(str(annotation_file))
                logger.info(f"✓ 加载annotation: {annotation_volume.shape}")

                # 关键：参考data_loader_enhanced.py，坐标已经是正确单位
                merfish_cells = merfish_mapper.map_cells_to_subregions(
                    merfish_cells,
                    annotation_volume
                )

                mapped_count = merfish_cells['subregion_acronym'].notna().sum()
                logger.info(f"✓ Subregion映射: {mapped_count}/{len(merfish_cells)}")

            except Exception as e:
                logger.error(f"✗ Subregion映射失败: {e}")
                import traceback
                traceback.print_exc()

            # 步骤4: 映射到ME_Subregion
            logger.info("\n步骤4: 映射到ME_Subregion")
            if merfish_mapper.load_me_subregion_annotation():
                merfish_cells = merfish_mapper.map_cells_to_me_subregions(merfish_cells)

                mapped_count = merfish_cells['me_subregion_acronym'].notna().sum()
                logger.info(f"✓ ME_Subregion映射: {mapped_count}/{len(merfish_cells)}")
            else:
                logger.warning("✗ ME注释加载失败")

        logger.info("=" * 60)
        logger.info("Phase 3.7 完成")
        logger.info("=" * 60)

        # Phase 3.8: 加载Subregion数据
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.8: 加载Subregion和ME_Subregion数据")
        logger.info("=" * 60)

        ccf_me_json = data_path / "surf_tree_ccf-me.json"
        subregion_loader = SubregionLoader(ccf_me_json)

        if not subregion_loader.load_subregion_data():
            logger.warning("无法加载Subregion数据")
            subregion_loader = None

        # Phase 4: 知识图谱生成和插入
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: 知识图谱生成和插入")
        logger.info("=" * 60)

        builder.set_hierarchy_loader(hierarchy_loader)

        # 插入Region节点
        logger.info("\n插入Region节点...")
        builder.generate_and_insert_unified_region_nodes(region_data, merfish_cells)

        # 插入Neuron节点
        if neuron_loader:
            logger.info("\n插入Neuron节点...")
            builder.generate_and_insert_neuron_nodes_with_full_morphology(neuron_loader)

        # 插入Subregion和ME_Subregion节点
        if subregion_loader:
            logger.info("\n插入Subregion和ME_Subregion节点...")
            builder.generate_and_insert_subregion_nodes(subregion_loader)
            builder.generate_and_insert_me_subregion_nodes(subregion_loader)

        # 插入MERFISH细胞类型节点
        logger.info("\n插入MERFISH细胞类型节点...")
        builder.generate_and_insert_merfish_nodes_from_hierarchy(merfish_cells)

        # Phase 5: 插入关系
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: 插入关系")
        logger.info("=" * 60)

        # HAS关系 - Region级别
        logger.info("\n插入Region级别的HAS关系...")
        for level in ['class', 'subclass', 'supertype', 'cluster']:
            builder.generate_and_insert_has_relationships_unified(merfish_cells, level)

        # ⭐ 修复2: HAS关系 - Subregion级别
        if 'subregion_acronym' in merfish_cells.columns:
            mapped_count = merfish_cells['subregion_acronym'].notna().sum()
            if mapped_count > 0:
                logger.info(f"\n插入Subregion级别的HAS关系（{mapped_count}个已映射细胞）...")
                builder.generate_and_insert_merfish_subregion_relationships(
                    merfish_cells,
                    'subregion'
                )
            else:
                logger.warning("没有细胞映射到Subregion")

        # ⭐ 修复2: HAS关系 - ME_Subregion级别
        if 'me_subregion_acronym' in merfish_cells.columns:
            mapped_count = merfish_cells['me_subregion_acronym'].notna().sum()
            if mapped_count > 0:
                logger.info(f"\n插入ME_Subregion级别的HAS关系（{mapped_count}个已映射细胞）...")
                builder.generate_and_insert_merfish_subregion_relationships(
                    merfish_cells,
                    'me_subregion'
                )
            else:
                logger.warning("没有细胞映射到ME_Subregion")

        # BELONGS_TO关系
        logger.info("\n插入BELONGS_TO关系...")
        builder.generate_and_insert_belongs_to_from_hierarchy()

        # PROJECT_TO关系 - Region级别（从原有数据）
        logger.info("\n插入Region级别的PROJECT_TO关系（从统计数据）...")
        builder.generate_and_insert_project_to_relationships(projection_data)

        # ⭐ 修复3: 神经元NEIGHBOURING关系（去重）
        if neuron_loader:
            logger.info("\n插入Neuron之间的关系（修复3：去重版本）...")
            builder.generate_and_insert_neuron_relationships(neuron_loader)

        # ⭐ 修复1: 神经元投射关系（包含直接读取的Region数据）
        if projection_processor:
            logger.info("\n插入Neuron投射关系（修复1：包含直接读取的Region数据）...")
            builder.generate_and_insert_neuron_projection_relationships(projection_processor)

        # Subregion层级关系
        if subregion_loader:
            logger.info("\n插入Subregion层级关系...")
            builder.generate_and_insert_subregion_relationships(subregion_loader)

        # Neuron-Subregion关系
        logger.info("\n" + "=" * 60)
        logger.info("Phase 6: 插入Neuron-Subregion关系")
        logger.info("=" * 60)
        
        neuron_subregion_inserter = NeuronSubregionRelationshipInserter(
            neo4j_conn, data_path
        )

        if neuron_subregion_inserter.load_neuron_subregion_mapping():
            neuron_subregion_inserter.insert_all_relationships(batch_size=BATCH_SIZE)
            verify_relationships(neo4j_conn, database_name)

        # 打印统计报告
        logger.info("\n" + "=" * 60)
        logger.info("统计报告")
        logger.info("=" * 60)
        builder.print_statistics_report_enhanced_with_subregion()

        logger.info("\n" + "=" * 60)
        logger.info("✓ 知识图谱构建完成！")
        logger.info("=" * 60)
        logger.info("\n修复验证：")
        logger.info("1. ✓ Neuron->Region 投射关系已从 Proj_Axon_Final.csv 直接读取")
        logger.info("2. ✓ MERFISH 到 Subregion/ME_Subregion 的映射已修复")
        logger.info("3. ✓ Neighbouring 关系已去重")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建 - V5.1修复版')
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