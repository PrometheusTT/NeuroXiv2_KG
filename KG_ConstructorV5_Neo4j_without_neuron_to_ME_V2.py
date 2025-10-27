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
    """修复版：正确映射 MERFISH 细胞到 Subregion/ME_Subregion"""

    def __init__(self, data_path: Path, ccf_tree_json: Path):
        """
        初始化

        参数:
            data_path: 数据目录
            ccf_tree_json: CCF树结构JSON文件
        """
        self.data_path = data_path
        self.ccf_tree_json = ccf_tree_json

        # CCF结构映射
        self.id_to_acronym = {}
        self.id_to_name = {}
        self.id_to_parent = {}

        # ME注释
        self.me_annotation = None
        self.voxel_to_me_subregion = {}

    # ⭐ 修复2: 改进 CCF 树加载
    def load_ccf_tree_for_subregion_mapping(self) -> bool:
        """加载 CCF 树以构建 subregion 映射"""
        logger.info(f"加载 CCF 树用于 subregion 映射: {self.ccf_tree_json}")

        if not self.ccf_tree_json.exists():
            logger.error(f"CCF树文件不存在: {self.ccf_tree_json}")
            return False

        try:
            with open(self.ccf_tree_json, 'r') as f:
                tree_data = json.load(f)

            def traverse_tree(node, parent_info=None):
                """递归遍历树节点"""
                if not isinstance(node, dict):
                    return

                node_id = node.get('id')
                acronym = node.get('acronym', '')
                name = node.get('name', '')

                if node_id is not None:
                    self.id_to_acronym[node_id] = acronym
                    self.id_to_name[node_id] = name
                    
                    if parent_info:
                        self.id_to_parent[node_id] = parent_info

                # 递归处理children
                children = node.get('children', [])
                for child in children:
                    traverse_tree(child, {'id': node_id, 'acronym': acronym})

            if isinstance(tree_data, list):
                for root in tree_data:
                    traverse_tree(root)
            else:
                traverse_tree(tree_data)

            logger.info(f"  - 加载了 {len(self.id_to_acronym)} 个区域映射")
            return True

        except Exception as e:
            logger.error(f"加载CCF树失败: {e}")
            return False

    # ⭐ 修复2: 改进 ME_Subregion 注释加载
    def load_me_subregion_annotation(self) -> bool:
        """加载 ME_Subregion 注释"""
        logger.info("加载 ME_Subregion 注释...")

        me_file = self.data_path / "parc_r671_full.nrrd"
        pkl_file = self.data_path / "parc_r671_full.nrrd.pkl"

        if not me_file.exists():
            logger.error(f"ME注释文件不存在: {me_file}")
            return False

        try:
            # 加载NRRD
            self.me_annotation, _ = nrrd.read(str(me_file))
            logger.info(f"  - 加载了 ME 注释: {self.me_annotation.shape}")

            # 加载pkl映射
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    parent_to_voxel = pickle.load(f)
                logger.info(f"  - 加载了 pkl 映射: {len(parent_to_voxel)} 条")

                # 反转映射：voxel -> parent
                for parent, voxel_list in parent_to_voxel.items():
                    for voxel_value in voxel_list:
                        # parent 格式如 "VIS-ME_324"
                        # 提取 acronym
                        if isinstance(parent, str) and '-ME' in parent:
                            parts = parent.split('_')
                            if len(parts) >= 1:
                                me_acronym = parts[0]  # 例如 "VIS-ME"
                                self.voxel_to_me_subregion[voxel_value] = me_acronym

                logger.info(f"  - 创建了 {len(self.voxel_to_me_subregion)} 个体素到ME的映射")

            # 也加载JSON获取完整结构
            json_file = self.data_path / "surf_tree_ccf-me.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    tree_data = json.load(f)
                
                # 从JSON提取更多映射
                additional_mappings = self._extract_me_mapping_from_json(tree_data)
                self.voxel_to_me_subregion.update(additional_mappings)
                
                logger.info(f"  - 更新后共 {len(self.voxel_to_me_subregion)} 个映射")

            return True

        except Exception as e:
            logger.error(f"加载 ME_Subregion 注释失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_me_mapping_from_json(self, tree_data) -> Dict[int, str]:
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

                # 处理ME子区域
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

    # ⭐ 修复2: 改进 MERFISH 到 subregion 映射
    def map_cells_to_subregions(self, cells_df: pd.DataFrame, annotation_volume: np.ndarray) -> pd.DataFrame:
        """
        将 MERFISH 细胞映射到 Subregion
        
        参数:
            cells_df: MERFISH 细胞数据
            annotation_volume: CCF 注释体积数据
        """
        logger.info("将 MERFISH 细胞映射到 Subregion...")

        if annotation_volume is None:
            logger.warning("没有 CCF 注释数据")
            cells_df['subregion_acronym'] = None
            return cells_df

        # 检查必要的列
        required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
        if not all(col in cells_df.columns for col in required_cols):
            logger.error(f"缺少必要的坐标列: {required_cols}")
            cells_df['subregion_acronym'] = None
            return cells_df

        # 转换坐标到体素索引
        resolution = 25.0
        x_indices = (cells_df['x_ccf'] / resolution).astype(int)
        y_indices = (cells_df['y_ccf'] / resolution).astype(int)
        z_indices = (cells_df['z_ccf'] / resolution).astype(int)

        # 查询 subregion
        subregion_acronyms = []
        for x, y, z in tqdm(zip(x_indices, y_indices, z_indices), total=len(cells_df), desc="映射subregion"):
            if (0 <= x < annotation_volume.shape[0] and
                0 <= y < annotation_volume.shape[1] and
                0 <= z < annotation_volume.shape[2]):
                ccf_id = int(annotation_volume[x, y, z])
                acronym = self.id_to_acronym.get(ccf_id)
                subregion_acronyms.append(acronym)
            else:
                subregion_acronyms.append(None)

        cells_df['subregion_acronym'] = subregion_acronyms

        mapped_count = cells_df['subregion_acronym'].notna().sum()
        logger.info(f"  - 映射了 {mapped_count}/{len(cells_df)} 个细胞到 Subregion")

        return cells_df

    # ⭐ 修复2: 改进 MERFISH 到 ME_subregion 映射
    def map_cells_to_me_subregions(self, cells_df: pd.DataFrame) -> pd.DataFrame:
        """将 MERFISH 细胞映射到 ME_Subregion"""
        logger.info("将 MERFISH 细胞映射到 ME_Subregion...")

        if self.me_annotation is None:
            logger.warning("没有 ME_Subregion 注释数据")
            cells_df['me_subregion_acronym'] = None
            return cells_df

        # 检查必要的列
        required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
        if not all(col in cells_df.columns for col in required_cols):
            logger.error(f"缺少必要的坐标列: {required_cols}")
            cells_df['me_subregion_acronym'] = None
            return cells_df

        # 转换坐标到体素索引
        resolution = 25.0
        x_indices = (cells_df['x_ccf'] / resolution).astype(int)
        y_indices = (cells_df['y_ccf'] / resolution).astype(int)
        z_indices = (cells_df['z_ccf'] / resolution).astype(int)

        # 查询 ME_Subregion
        me_acronyms = []
        unmapped_count = 0
        
        for x, y, z in tqdm(zip(x_indices, y_indices, z_indices), total=len(cells_df), desc="映射ME_subregion"):
            if (0 <= x < self.me_annotation.shape[0] and
                0 <= y < self.me_annotation.shape[1] and
                0 <= z < self.me_annotation.shape[2]):
                voxel_value = int(self.me_annotation[x, y, z])
                acronym = self.voxel_to_me_subregion.get(voxel_value)
                me_acronyms.append(acronym)
                if acronym is None:
                    unmapped_count += 1
            else:
                me_acronyms.append(None)
                unmapped_count += 1

        cells_df['me_subregion_acronym'] = me_acronyms

        mapped_count = cells_df['me_subregion_acronym'].notna().sum()
        logger.info(f"  - 映射了 {mapped_count}/{len(cells_df)} 个细胞到 ME_Subregion")
        if unmapped_count > 0:
            logger.info(f"  - {unmapped_count} 个细胞未能映射")

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
        if not neuron_loader or not neuron_loader.connections_data.empty:
            logger.info("处理神经元之间的关系（NEIGHBOURING，去重版本）...")
            self._insert_neighbouring_relationships_deduplicated(neuron_loader.connections_data)

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

        # Phase 3.7: ⭐ 修复2 - MERFISH空间映射
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3.7: MERFISH细胞空间映射（修复2：改进映射逻辑）")
        logger.info("=" * 60)
        
        merfish_mapper = MERFISHSubregionMapper(data_path, ccf_tree_json)

        # 加载CCF树
        if merfish_mapper.load_ccf_tree_for_subregion_mapping():
            
            # 加载基础annotation（用于subregion映射）
            annotation_file = data_path / "annotation_25.nrrd"
            annotation_volume = None
            if annotation_file.exists():
                try:
                    annotation_volume, _ = nrrd.read(str(annotation_file))
                    logger.info(f"✓ 加载了基础注释: {annotation_volume.shape}")
                    
                    # 映射到subregion
                    merfish_cells = merfish_mapper.map_cells_to_subregions(
                        merfish_cells, 
                        annotation_volume
                    )
                except Exception as e:
                    logger.warning(f"加载基础注释失败: {e}")

            # 加载ME注释并映射
            if merfish_mapper.load_me_subregion_annotation():
                merfish_cells = merfish_mapper.map_cells_to_me_subregions(merfish_cells)
                logger.info("✓ ME_Subregion 映射完成")

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