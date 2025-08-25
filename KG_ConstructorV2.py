"""
NeuroXiv 2.0 知识图谱构建器 - 修复版
整合MERFISH层级JSON和层特异性形态计算

作者: PrometheusTT
日期: 2025-08-25
"""
import csv
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings('ignore')

# 导入必要的数据加载函数
from data_loader_enhanced import (
    load_data,
    prepare_analysis_data,
    map_cells_to_regions_fixed
)

# ==================== 配置常量 ====================

# 性能参数
CHUNK_SIZE = 100000
N_WORKERS = 100
BATCH_SIZE = 10000
PCT_THRESHOLD = 0.01

# 层定义
LAYERS = ['L1', 'L2/3', 'L4', 'L5', 'L6a', 'L6b']

# 形态学属性列表
MORPH_ATTRIBUTES = [
    'axonal_bifurcation_remote_angle', 'axonal_branches', 'axonal_length',
    'axonal_maximum_branch_order', 'dendritic_bifurcation_remote_angle',
    'dendritic_branches', 'dendritic_length', 'dendritic_maximum_branch_order'
]

# 统计属性列表
STAT_ATTRIBUTES = [
    'number_of_apical_dendritic_morphologies', 'number_of_axonal_morphologies',
    'number_of_dendritic_morphologies', 'number_of_neuron_morphologies',
    'number_of_transcriptomic_neurons'
]

# 合并到完整形态特征列表
FULL_MORPH_FEATURES = MORPH_ATTRIBUTES + [
    'Area', 'Volume', 'Width', 'Height', 'Depth', 'Slimness', 'Flatness',
    'Average Contraction', 'Max Euclidean Distance', 'Max Path Distance',
    'Average Euclidean Distance', 'Average Path Distance', '2D Density', '3D Density'
]


# ==================== 工具函数 ====================

def setup_logger(log_file: str = "kg_builder.log"):
    """设置日志"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="100 MB", level="DEBUG")


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def extract_layer_from_celltype(celltype: str) -> str:
    """从celltype字符串提取层信息"""
    if pd.isna(celltype):
        return 'Unknown'

    celltype = str(celltype)

    # 直接匹配层模式
    layer_patterns = {
        'L1': ['L1', '1'],
        'L2/3': ['L2/3', 'L2', 'L3', '2/3', '2', '3'],
        'L4': ['L4', '4'],
        'L5': ['L5', 'L5a', 'L5b', '5'],
        'L6a': ['L6a', '6a'],
        'L6b': ['L6b', '6b']
    }

    for layer, patterns in layer_patterns.items():
        for pattern in patterns:
            # 检查是否以层模式结尾
            if celltype.endswith(pattern):
                return layer
            # 或者包含层模式（用于处理如"SSp-m4"这样的格式）
            if pattern in celltype and pattern != '1':  # 避免'1'匹配到其他数字
                return layer

    return 'Unknown'

def extract_base_region_from_celltype(celltype: str) -> str:
    """从celltype提取基础区域名（去除层信息）"""
    if pd.isna(celltype):
        return celltype

    celltype = str(celltype)

    # 移除层后缀
    layer_suffixes = ['1', '2/3', '2', '3', '4', '5', '5a', '5b', '6', '6a', '6b']
    for suffix in sorted(layer_suffixes, key=len, reverse=True):  # 从长到短匹配
        if celltype.endswith(suffix):
            base = celltype[:-len(suffix)]
            # 移除可能的连接符
            if base.endswith('-') or base.endswith('_'):
                base = base[:-1]
            return base

    return celltype

# ==================== MERFISH层级数据加载器 ====================

class MERFISHHierarchyLoader:
    """MERFISH层级JSON数据加载器"""

    def __init__(self, json_file: Path):
        self.json_file = Path(json_file)
        self.hierarchy_tree = None
        self.class_data = {}
        self.subclass_data = {}
        self.supertype_data = {}
        self.cluster_data = {}
        self.hierarchy_relations = {
            'subclass_to_class': {},
            'supertype_to_subclass': {},
            'cluster_to_supertype': {}
        }

    def load_hierarchy(self) -> bool:
        """加载层级JSON文件"""
        if not self.json_file.exists():
            logger.error(f"层级JSON文件不存在: {self.json_file}")
            return False

        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.hierarchy_tree = json.load(f)
            logger.info(f"成功加载层级JSON，包含 {len(self.hierarchy_tree)} 个顶级类")
            self._parse_hierarchy()
            return True
        except Exception as e:
            logger.error(f"加载层级JSON失败: {e}")
            return False

    def _parse_hierarchy(self):
        """解析层级结构"""
        class_id = 0
        subclass_id = 0
        supertype_id = 0
        cluster_id = 0

        for class_node in self.hierarchy_tree:
            class_id += 1

            # 解析Class节点
            self.class_data[class_node['label']] = {
                'tran_id': class_id,
                'original_id': int(class_node['id']),
                'name': class_node['label'],
                'neighborhood': class_node.get('neighborhood', ''),
                'color': class_node.get('color', ''),
                'number_of_child_types': len(class_node.get('child', [])),
                'number_of_neurons': 0  # 后续计算
            }

            # 解析Subclass
            for subclass_node in class_node.get('child', []):
                subclass_id += 1

                self.subclass_data[subclass_node['label']] = {
                    'tran_id': subclass_id,
                    'original_id': int(subclass_node['id']),
                    'name': subclass_node['label'],
                    'markers': subclass_node.get('markers', ''),
                    'transcription_factor_markers': subclass_node.get('transcription_factor_markers', ''),
                    'dominant_neurotransmitter_type': subclass_node.get('dominent_neurotransmitter_type', ''),
                    'neighborhood': subclass_node.get('neighborhood', ''),
                    'color': subclass_node.get('color', ''),
                    'number_of_child_types': len(subclass_node.get('child', [])),
                    'number_of_neurons': 0
                }

                # 记录层级关系
                self.hierarchy_relations['subclass_to_class'][subclass_node['label']] = class_node['label']

                # 解析Supertype
                for supertype_node in subclass_node.get('child', []):
                    supertype_id += 1

                    self.supertype_data[supertype_node['label']] = {
                        'tran_id': supertype_id,
                        'original_id': int(supertype_node['id']),
                        'name': supertype_node['label'],
                        'markers': supertype_node.get('markers', ''),
                        'within_subclass_markers': supertype_node.get('within_subclass_markers', ''),
                        'neighborhood': supertype_node.get('neighborhood', ''),
                        'color': supertype_node.get('color', ''),
                        'number_of_child_types': len(supertype_node.get('child', [])),
                        'number_of_neurons': 0
                    }

                    # 记录层级关系
                    self.hierarchy_relations['supertype_to_subclass'][supertype_node['label']] = subclass_node['label']

                    # 解析Cluster
                    for cluster_node in supertype_node.get('child', []):
                        cluster_id += 1

                        self.cluster_data[cluster_node['label']] = {
                            'tran_id': cluster_id,
                            'original_id': int(cluster_node['id']),
                            'name': cluster_node['label'],
                            'neighborhood': cluster_node.get('neighborhood', ''),
                            'anatomical_annotation': cluster_node.get('anatomical_annotation', ''),
                            'broad_region_distribution': cluster_node.get('broad_region_distribution', ''),
                            'acronym_region_distribution': cluster_node.get('acronym_region_distribution', ''),
                            'dominant_neurotransmitter_type': cluster_node.get('dominent_neurotransmitter_type', ''),
                            'neurotransmitter_mark_genes': cluster_node.get('neurotransmitter_mark_genes', ''),
                            'neuropeptide_mark_genes': cluster_node.get('neuropeptide_mark_genes', ''),
                            'markers': cluster_node.get('markers', ''),
                            'merfish_markers': cluster_node.get('merfish_markers', ''),
                            'transcription_factor_markers': cluster_node.get('transcription_factor_markers', ''),
                            'within_subclass_markers': cluster_node.get('within_subclass_markers', ''),
                            'color': cluster_node.get('color', ''),
                            'number_of_neurons': 0
                        }

                        # 记录层级关系
                        self.hierarchy_relations['cluster_to_supertype'][cluster_node['label']] = supertype_node['label']

        logger.info(f"解析完成: {len(self.class_data)} Class, {len(self.subclass_data)} Subclass, "
                    f"{len(self.supertype_data)} Supertype, {len(self.cluster_data)} Cluster")


class RegionAnalyzer:
    """区域分析器，用于识别皮层区域和标准CCF区域"""

    def __init__(self, tree_data: List[Dict]):
        """
        初始化区域分析器

        参数:
            tree_data: CCF树结构数据
        """
        self.tree_data = tree_data
        self.region_info = {}
        self.cortical_regions = set()
        self.standard_regions = set()  # CCF标准区域（不含层信息）
        self._build_region_hierarchy()

    def _build_region_hierarchy(self):
        """构建区域层级信息"""
        # 构建区域信息字典
        for node in self.tree_data:
            region_id = node.get('id')
            if region_id:
                self.region_info[region_id] = {
                    'acronym': node.get('acronym', ''),
                    'name': node.get('name', ''),
                    'parent_id': node.get('parent_id'),
                    'depth': node.get('depth', 0)
                }

                # 标准CCF区域（所有在树中定义的区域）
                self.standard_regions.add(region_id)

        # 识别皮层区域
        self._identify_cortical_regions()

    def _identify_cortical_regions(self):
        """识别皮层区域"""
        # 皮层相关的关键词和缩写
        cortical_keywords = [
            'cortex', 'Cortex', 'CORTEX',
            'cortical', 'Cortical',
            'isocortex', 'Isocortex',
            'neocortex', 'Neocortex'
        ]

        cortical_acronyms = [
            'SSp', 'SSs', 'MOp', 'MOs',  # 体感和运动皮层
            'VIS', 'AUD', 'TEa', 'ECT',  # 视觉、听觉、颞叶皮层
            'RSP', 'ACA', 'PL', 'ILA',  # 后部、前扣带、边缘皮层
            'ORB', 'AI', 'GU', 'VISC',  # 眶额、岛叶、味觉、内脏皮层
            'FRP', 'PRL',  # 额极皮层
            'ENT', 'PERI', 'POST',  # 内嗅、周围、后皮层
            'PTLp', 'VISpor'  # 顶叶后部皮层
        ]

        # 查找所有皮层区域
        for region_id, info in self.region_info.items():
            acronym = info['acronym']
            name = info['name']

            # 检查是否是皮层区域
            is_cortical = False

            # 检查名称中是否包含皮层关键词
            for keyword in cortical_keywords:
                if keyword in name:
                    is_cortical = True
                    break

            # 检查缩写是否匹配皮层区域
            if not is_cortical:
                for cortical_prefix in cortical_acronyms:
                    if acronym.startswith(cortical_prefix):
                        is_cortical = True
                        break

            # 检查父区域是否是皮层（递归检查）
            if not is_cortical and info['parent_id']:
                is_cortical = self._has_cortical_ancestor(info['parent_id'])

            if is_cortical:
                self.cortical_regions.add(region_id)

    def _has_cortical_ancestor(self, region_id: int, visited: Set[int] = None) -> bool:
        """检查是否有皮层祖先"""
        if visited is None:
            visited = set()

        if region_id in visited:
            return False

        visited.add(region_id)

        if region_id in self.cortical_regions:
            return True

        info = self.region_info.get(region_id)
        if info and info['parent_id']:
            return self._has_cortical_ancestor(info['parent_id'], visited)

        return False

    def is_cortical_region(self, region_id: int) -> bool:
        """判断是否是皮层区域"""
        return region_id in self.cortical_regions

    def is_standard_region(self, region_id: int) -> bool:
        """判断是否是标准CCF区域"""
        return region_id in self.standard_regions

    def get_region_info(self, region_id: int) -> Dict:
        """获取区域信息"""
        return self.region_info.get(region_id, {})
# ==================== 层特异性形态计算 ====================

class LayerSpecificMorphologyCalculator:
    """层特异性形态学计算器"""

    def __init__(self, data_dir: Path, region_analyzer: RegionAnalyzer = None):
        self.data_dir = Path(data_dir)
        self.info_df = None
        self.axon_df = None
        self.dendrite_df = None
        self.region_name_id_map = {}
        self.region_analyzer = region_analyzer  # 添加区域分析器

    def load_morphology_with_layers(self) -> bool:
        """加载形态数据和层信息，处理celltype映射"""
        # 加载info.csv
        info_file = self.data_dir / "info.csv"
        if not info_file.exists():
            logger.error(f"info.csv不存在: {info_file}")
            return False

        self.info_df = pd.read_csv(info_file)
        logger.info(f"加载了 {len(self.info_df)} 条神经元信息")

        # 提取基础区域和层信息
        if 'celltype' in self.info_df.columns:
            # 提取基础区域（不含层信息）
            self.info_df['base_region'] = self.info_df['celltype'].apply(extract_base_region_from_celltype)
            # 提取层信息
            self.info_df['layer'] = self.info_df['celltype'].apply(extract_layer_from_celltype)

            layer_counts = self.info_df['layer'].value_counts()
            logger.info(f"层分布: {layer_counts.to_dict()}")

            base_region_counts = self.info_df['base_region'].value_counts()
            logger.info(f"找到 {len(base_region_counts)} 个基础区域")

            # 加载tree_yzx.json获取acronym到region_id的映射
            tree_file = self.data_dir / "tree_yzx.json"
            acronym_to_id = {}

            if tree_file.exists():
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)

                for node in tree_data:
                    if 'id' in node and 'acronym' in node:
                        acronym_to_id[node['acronym']] = node['id']

                # 将base_region映射到region_id
                self.info_df['region_id'] = self.info_df['base_region'].map(acronym_to_id)
                valid_region_count = self.info_df['region_id'].notna().sum()
                logger.info(f"从base_region映射到region_id: {valid_region_count}/{len(self.info_df)}个有效映射")
        else:
            logger.warning("info.csv缺少celltype列")
            self.info_df['layer'] = 'Unknown'
            self.info_df['base_region'] = 'Unknown'

        # 加载轴突形态数据
        axon_file = self.data_dir / "axonfull_morpho.csv"
        if axon_file.exists():
            self.axon_df = pd.read_csv(axon_file)
            if 'name' in self.axon_df.columns:
                self.axon_df = self.axon_df[~self.axon_df['name'].str.contains('CCF-thin|local', na=False)]
            logger.info(f"加载了 {len(self.axon_df)} 条轴突形态数据")

        # 加载树突形态数据
        dendrite_file = self.data_dir / "denfull_morpho.csv"
        if dendrite_file.exists():
            self.dendrite_df = pd.read_csv(dendrite_file)
            if 'name' in self.dendrite_df.columns:
                self.dendrite_df = self.dendrite_df[~self.dendrite_df['name'].str.contains('CCF-thin|local', na=False)]
            logger.info(f"加载了 {len(self.dendrite_df)} 条树突形态数据")

        return True

    def calculate_regionlayer_morphology(self, region_data: pd.DataFrame,
                                         merfish_cells: pd.DataFrame = None) -> pd.DataFrame:
        """计算层特异性形态特征 - 只为皮层区域创建层"""
        logger.info("计算层特异性形态特征（仅皮层区域）...")

        if region_data.empty:
            logger.warning("输入的region_data为空")
            return pd.DataFrame()

        if self.axon_df is None or self.dendrite_df is None or self.info_df is None:
            logger.warning("缺少必要的形态数据")
            return pd.DataFrame()

        # 构建区域名称到ID的映射
        for _, region in region_data.iterrows():
            region_id = region.get('region_id')
            region_name = region.get('name', '')
            region_acronym = region.get('acronym', '')

            if region_id and not pd.isna(region_id):
                # 使用acronym或name作为映射键
                if region_acronym:
                    self.region_name_id_map[region_acronym] = region_id
                if region_name:
                    self.region_name_id_map[region_name] = region_id

        # 添加region_id到info_df（基于base_region）
        if 'region_id' not in self.info_df.columns:
            self.info_df['region_id'] = self.info_df['base_region'].map(self.region_name_id_map).fillna(-1).astype(int)

        # 创建结果数据框
        regionlayer_data = []

        # 形态特征映射
        morph_mapping = {
            'Average Bifurcation Angle Remote': 'bifurcation_remote_angle',
            'Number of Bifurcations': 'branches',
            'Total Length': 'length',
            'Max Branch Order': 'maximum_branch_order'
        }

        # 统计皮层和非皮层区域
        cortical_count = 0
        non_cortical_count = 0

        # 为每个区域处理
        for _, region in region_data.iterrows():
            region_id = region.get('region_id')
            if not region_id or pd.isna(region_id):
                continue

            region_name = region.get('name', f'Region {region_id}')

            # 判断是否是皮层区域
            is_cortical = False
            if self.region_analyzer:
                is_cortical = self.region_analyzer.is_cortical_region(region_id)
            else:
                # 如果没有区域分析器，使用简单规则判断
                region_acronym = region.get('acronym', '')
                cortical_prefixes = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'RSP', 'ACA', 'ENT', 'AI']
                is_cortical = any(region_acronym.startswith(prefix) for prefix in cortical_prefixes)

            if is_cortical:
                cortical_count += 1
                # 皮层区域：为每个层创建节点
                layers_to_create = LAYERS
            else:
                non_cortical_count += 1
                # 非皮层区域：创建单个"全区域"节点
                layers_to_create = ['ALL']  # 使用ALL表示整个区域

            # 获取该区域的所有神经元
            region_neurons = self.info_df[self.info_df['region_id'] == region_id]

            for layer in layers_to_create:
                if layer == 'ALL':
                    # 非皮层区域，使用所有神经元
                    layer_neurons = region_neurons
                    layer_neuron_ids = set(layer_neurons['ID'].values) if 'ID' in layer_neurons.columns else set()
                    rl_id = f"{region_id}_ALL"
                else:
                    # 皮层区域，筛选特定层的神经元
                    layer_neurons = region_neurons[region_neurons['layer'] == layer]
                    layer_neuron_ids = set(layer_neurons['ID'].values) if 'ID' in layer_neurons.columns else set()
                    rl_id = f"{region_id}_{layer}"

                # 计算统计属性
                axon_count = 0
                if self.axon_df is not None and 'ID' in self.axon_df.columns and len(layer_neuron_ids) > 0:
                    layer_axons = self.axon_df[self.axon_df['ID'].isin(layer_neuron_ids)]
                    axon_count = len(layer_axons['ID'].unique())

                dendrite_count = 0
                apical_dendrite_count = 0
                if self.dendrite_df is not None and 'ID' in self.dendrite_df.columns and len(layer_neuron_ids) > 0:
                    layer_dendrites = self.dendrite_df[self.dendrite_df['ID'].isin(layer_neuron_ids)]
                    dendrite_count = len(layer_dendrites['ID'].unique())

                    if 'type' in layer_dendrites.columns:
                        apical_dendrites = layer_dendrites[
                            layer_dendrites['type'].str.contains('apical', case=False, na=False)]
                        apical_dendrite_count = len(apical_dendrites['ID'].unique())

                neuron_count = len(layer_neuron_ids)

                # 转录组神经元数量
                transcriptomic_count = 0
                if merfish_cells is not None and not merfish_cells.empty:
                    if 'region_id' in merfish_cells.columns:
                        if layer == 'ALL':
                            # 非皮层区域，计算所有细胞
                            region_cells = merfish_cells[merfish_cells['region_id'] == region_id]
                            transcriptomic_count = len(region_cells)
                        else:
                            # 皮层区域，计算特定层的细胞
                            if 'celltype' in merfish_cells.columns:
                                merfish_cells_copy = merfish_cells.copy()
                                merfish_cells_copy['layer'] = merfish_cells_copy['celltype'].apply(
                                    extract_layer_from_celltype)
                                layer_cells = merfish_cells_copy[
                                    (merfish_cells_copy['region_id'] == region_id) &
                                    (merfish_cells_copy['layer'] == layer)
                                    ]
                                transcriptomic_count = len(layer_cells)

                # 创建RegionLayer记录
                rl_dict = {
                    'rl_id': rl_id,
                    'region_id': region_id,
                    'layer': layer,
                    'region_name': region_name,
                    'is_cortical': is_cortical,
                    'number_of_neuron_morphologies': neuron_count,
                    'number_of_axonal_morphologies': axon_count,
                    'number_of_dendritic_morphologies': dendrite_count,
                    'number_of_apical_dendritic_morphologies': apical_dendrite_count,
                    'number_of_transcriptomic_neurons': transcriptomic_count
                }

                # 计算形态特征
                if len(layer_neuron_ids) >= 5:
                    # 计算轴突特征
                    if self.axon_df is not None and 'ID' in self.axon_df.columns:
                        layer_axons = self.axon_df[self.axon_df['ID'].isin(layer_neuron_ids)]
                        if len(layer_axons) > 0:
                            for source_feat, target_feat in morph_mapping.items():
                                if source_feat in layer_axons.columns:
                                    values = layer_axons[source_feat].dropna()
                                    if len(values) > 0:
                                        rl_dict[f'axonal_{target_feat}'] = values.mean()
                                    else:
                                        rl_dict[f'axonal_{target_feat}'] = 0.0

                    # 计算树突特征
                    if self.dendrite_df is not None and 'ID' in self.dendrite_df.columns:
                        layer_dendrites = self.dendrite_df[self.dendrite_df['ID'].isin(layer_neuron_ids)]
                        if len(layer_dendrites) > 0:
                            for source_feat, target_feat in morph_mapping.items():
                                if source_feat in layer_dendrites.columns:
                                    values = layer_dendrites[source_feat].dropna()
                                    if len(values) > 0:
                                        rl_dict[f'dendritic_{target_feat}'] = values.mean()
                                    else:
                                        rl_dict[f'dendritic_{target_feat}'] = 0.0
                else:
                    # 神经元太少，使用默认值
                    for attr in MORPH_ATTRIBUTES:
                        rl_dict[attr] = 0.0

                regionlayer_data.append(rl_dict)

        # 转换为DataFrame
        result_df = pd.DataFrame(regionlayer_data)

        logger.info(f"生成了 {len(result_df)} 个RegionLayer节点")
        logger.info(f"  - 皮层区域: {cortical_count} 个 (生成 {cortical_count * len(LAYERS)} 个层节点)")
        logger.info(f"  - 非皮层区域: {non_cortical_count} 个 (生成 {non_cortical_count} 个全区域节点)")

        return result_df


# ==================== 增强的知识图谱构建器 ====================

class EnhancedKnowledgeGraphBuilder:
    """增强的知识图谱构建器"""

    def __init__(self, output_dir: Path):
        self.region_name_id_map = None
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        self.nodes_dir = self.output_dir / "nodes"
        self.relationships_dir = self.output_dir / "relationships"
        ensure_dir(self.nodes_dir)
        ensure_dir(self.relationships_dir)

        # 存储ID映射
        self.class_id_map = {}
        self.subclass_id_map = {}
        self.supertype_id_map = {}
        self.cluster_id_map = {}

        # 存储层级数据
        self.hierarchy_loader = None

        # 初始化区域分析器
        self.region_analyzer = None

        # 有效的脑区acronym列表
        self.valid_acronyms = set([
            'AAA', 'ACAd', 'ACAv', 'ACB', 'ACVII', 'AD', 'ADP', 'AHN', 'AId', 'AIp', 'AIv',
            'AM', 'AMB', 'AN', 'AOB', 'AON', 'AP', 'APN', 'APr', 'ARH', 'ASO', 'AT', 'AUDd',
            'AUDp', 'AUDpo', 'AUDv', 'AV', 'AVP', 'AVPV', 'Acs5', 'B', 'BA', 'BAC', 'BLA',
            'BMA', 'BST', 'CA1', 'CA2', 'CA3', 'CEA', 'CENT', 'CL', 'CLA', 'CLI', 'CM',
            'COAa', 'COAp', 'COPY', 'CP', 'CS', 'CU', 'CUL', 'CUN', 'DCO', 'DEC', 'DG',
            'DMH', 'DMX', 'DN', 'DP', 'DR', 'DT', 'DTN', 'ECT', 'ECU', 'ENTl', 'ENTm',
            'EPd', 'EPv', 'EW', 'FC', 'FL', 'FN', 'FOTU', 'FRP', 'FS', 'GPe', 'GPi', 'GR',
            'GRN', 'GU', 'HATA', 'I5', 'IA', 'IAD', 'IAM', 'IC', 'ICB', 'IF', 'IG', 'IGL',
            'III', 'ILA', 'IMD', 'IO', 'IP', 'IPN', 'IRN', 'ISN', 'IV', 'IntG', 'L1', 'L2/3',
            'L4', 'L5', 'L6a', 'L6b', 'LA', 'LAV', 'LC', 'LD', 'LDT', 'LGd', 'LGv', 'LH',
            'LHA', 'LIN', 'LING', 'LM', 'LP', 'LPO', 'LRN', 'LSc', 'LSr', 'LSv', 'LT', 'MA',
            'MA3', 'MARN', 'MD', 'MDRN', 'MDRNd', 'MDRNv', 'ME', 'MEA', 'MEPO', 'MEV', 'MG',
            'MH', 'MM', 'MOB', 'MOp', 'MOs', 'MPN', 'MPO', 'MPT', 'MRN', 'MS', 'MT', 'MV',
            'NB', 'NDB', 'NI', 'NLL', 'NLOT', 'NOD', 'NOT', 'NPC', 'NR', 'NTB', 'NTS', 'OP',
            'ORBl', 'ORBm', 'ORBvl', 'OT', 'OV', 'P5', 'PA', 'PAA', 'PAG', 'PAR', 'PARN',
            'PAS', 'PB', 'PBG', 'PC5', 'PCG', 'PCN', 'PD', 'PDTg', 'PERI', 'PF', 'PFL', 'PG',
            'PGRNd', 'PGRNl', 'PH', 'PIL', 'PIR', 'PL', 'PMd', 'PMv', 'PN', 'PO', 'POL',
            'POST', 'PP', 'PPN', 'PPT', 'PPY', 'PR', 'PRE', 'PRM', 'PRNc', 'PRNr', 'PRP',
            'PS', 'PST', 'PSTN', 'PSV', 'PT', 'PVH', 'PVHd', 'PVT', 'PVa', 'PVi', 'PVp',
            'PVpo', 'PYR', 'Pa4', 'Pa5', 'PeF', 'PoT', 'ProS', 'RCH', 'RE', 'RH', 'RL', 'RM',
            'RN', 'RO', 'RPA', 'RPO', 'RR', 'RSPagl', 'RSPd', 'RSPv', 'RT', 'SAG', 'SBPV',
            'SCH', 'SCO', 'SCm', 'SCs', 'SF', 'SFO', 'SG', 'SGN', 'SH', 'SI', 'SIM', 'SLC',
            'SLD', 'SMT', 'SNc', 'SNr', 'SO', 'SOC', 'SPA', 'SPFm', 'SPFp', 'SPIV', 'SPVC',
            'SPVI', 'SPVO', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul',
            'SSp-un', 'SSs', 'STN', 'SUB', 'SUM', 'SUT', 'SUV', 'SubG', 'TEa', 'TMd', 'TMv',
            'TR', 'TRN', 'TRS', 'TT', 'TU', 'UVU', 'V', 'VAL', 'VCO', 'VI', 'VII', 'VISC',
            'VISa', 'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl', 'VISpm', 'VISpor',
            'VISrl', 'VLPO', 'VM', 'VMH', 'VMPO', 'VPL', 'VPLpc', 'VPM', 'VPMpc', 'VTA',
            'VTN', 'VeCB', 'XII', 'Xi', 'ZI', 'fiber tracts'
        ])

    def set_hierarchy_loader(self, hierarchy_loader: MERFISHHierarchyLoader):
        """设置层级加载器"""
        self.hierarchy_loader = hierarchy_loader

        # 更新ID映射
        for name, data in hierarchy_loader.class_data.items():
            self.class_id_map[name] = data['tran_id']

        for name, data in hierarchy_loader.subclass_data.items():
            self.subclass_id_map[name] = data['tran_id']

        for name, data in hierarchy_loader.supertype_data.items():
            self.supertype_id_map[name] = data['tran_id']

        for name, data in hierarchy_loader.cluster_data.items():
            self.cluster_id_map[name] = data['tran_id']

    def generate_region_nodes(self, region_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """生成Region节点，使用tree_yzx.json中的acronym作为名称"""
        logger.info("生成Region节点...")

        # 保存region_data供其他方法使用
        self.region_data = region_data

        # 确保有region_analyzer
        if not hasattr(self, 'region_analyzer') or not self.region_analyzer:
            logger.warning("没有初始化region_analyzer，尝试加载tree_yzx.json")
            try:
                tree_file = Path("./data/tree_yzx.json")
                possible_locations = [
                    self.output_dir.parent / "data" / "tree_yzx.json",
                    Path("./data/tree_yzx.json"),
                    Path("/home/wlj/NeuroXiv2/data/tree_yzx.json")
                ]

                for loc in possible_locations:
                    if loc.exists():
                        tree_file = loc
                        break

                if tree_file.exists():
                    with open(tree_file, 'r') as f:
                        tree_data = json.load(f)

                    self.region_analyzer = RegionAnalyzer(tree_data)
                    logger.info(f"成功加载tree_yzx.json并初始化region_analyzer")
                else:
                    logger.warning("找不到tree_yzx.json文件")
            except Exception as e:
                logger.error(f"加载tree_yzx.json失败: {e}")

        # 诊断信息
        logger.info(f"区域数据形状: {region_data.shape}")
        logger.info(
            f"树信息条目数: {len(self.region_analyzer.region_info) if hasattr(self, 'region_analyzer') and self.region_analyzer else 0}")

        # 跟踪已处理的区域ID
        processed_ids = set()
        regions = []

        # 1. 首先处理region_data中的区域
        for _, region in region_data.iterrows():
            region_id = region.get('region_id')
            if pd.isna(region_id):
                continue

            try:
                region_id_int = int(region_id)
                processed_ids.add(region_id_int)
            except (ValueError, TypeError):
                logger.warning(f"跳过非数字region_id: {region_id}")
                continue

            # 获取区域信息
            region_info = {}
            if hasattr(self, 'region_analyzer') and self.region_analyzer:
                region_info = self.region_analyzer.get_region_info(region_id_int)

            # 获取acronym
            acronym = region_info.get('acronym', '') or region.get('acronym', '')

            # 如果没有acronym，继续下一个区域
            if not acronym:
                logger.warning(f"区域 {region_id_int} 没有acronym，使用默认名称")
                acronym = f"Region_{region_id_int}"

            # 创建区域字典
            region_dict = {
                'region_id:ID(Region)': region_id_int,
                'name': str(acronym),
                'full_name': str(region_info.get('name', '') or region.get('name', acronym)),
                'acronym': str(acronym)
            }

            # 添加parent_id
            if 'parent_id' in region_info:
                parent_id = region_info['parent_id']
                if parent_id is not None:
                    region_dict['parent_id:int'] = int(parent_id)
            elif 'parent_id' in region and not pd.isna(region['parent_id']):
                region_dict['parent_id:int'] = int(region['parent_id'])
            else:
                region_dict['parent_id:int'] = 0

            # 添加颜色
            if 'color' in region_info:
                region_dict['color:int[]'] = region_info['color']

            # 添加形态学属性
            for attr in MORPH_ATTRIBUTES:
                if attr in region:
                    region_dict[f'{attr}:float'] = float(region[attr])
                else:
                    region_dict[f'{attr}:float'] = 0.0

            # 计算统计属性
            stat_values = self._calculate_region_statistics(region_id_int, region_data, merfish_cells)

            # 添加统计属性
            for attr in STAT_ATTRIBUTES:
                region_dict[f'{attr}:int'] = stat_values.get(attr, 0)

            regions.append(region_dict)

        # 2. 处理有效acronym列表中的区域，但不在region_data中的
        if hasattr(self, 'region_analyzer') and self.region_analyzer:
            for acronym in self.valid_acronyms:
                region_id = self.region_analyzer.get_region_by_acronym(acronym)
                if region_id and region_id not in processed_ids:
                    region_info = self.region_analyzer.get_region_info(region_id)

                    # 创建区域字典
                    region_dict = {
                        'region_id:ID(Region)': region_id,
                        'name': acronym,
                        'full_name': region_info.get('name', acronym),
                        'acronym': acronym,
                        'parent_id:int': region_info.get('parent_id', 0) or 0
                    }

                    # 添加颜色
                    if 'color' in region_info:
                        region_dict['color:int[]'] = region_info['color']

                    # 添加默认的形态学属性
                    for attr in MORPH_ATTRIBUTES:
                        region_dict[f'{attr}:float'] = 0.0

                    # 添加默认的统计属性
                    for attr in STAT_ATTRIBUTES:
                        region_dict[f'{attr}:int'] = 0

                    regions.append(region_dict)
                    processed_ids.add(region_id)

        # 保存到CSV
        self._save_nodes(regions, "regions")
        logger.info(f"保存了 {len(regions)} 个Region节点")

    def _calculate_morphology_statistics(self, region_id, layer=None):
        """
        计算区域或区域层的形态统计数据

        参数:
            region_id: 区域ID
            layer: 层名称(如'L1'，'L2/3'等)，为None表示计算整个区域

        返回:
            包含统计数据的字典
        """
        stats = {}

        # 确保layer_calculator存在且已加载数据
        if not hasattr(self, 'layer_calculator') or not self.layer_calculator or self.layer_calculator.info_df is None:
            logger.warning(f"没有可用的神经元形态数据，无法计算统计信息")
            return {attr: 0 for attr in STAT_ATTRIBUTES}

        info_df = self.layer_calculator.info_df

        # 步骤1: 找到属于该区域的神经元
        if 'region_id' in info_df.columns:
            # 直接使用region_id过滤
            region_neurons = info_df[info_df['region_id'] == region_id]
        else:
            # 通过celltype前缀匹配
            region_acronym = self._get_region_acronym(region_id)
            if not region_acronym:
                logger.warning(f"无法获取区域{region_id}的acronym，无法匹配神经元")
                return {attr: 0 for attr in STAT_ATTRIBUTES}

            # 使用前缀匹配
            region_neurons = info_df[info_df['celltype'].str.startswith(region_acronym, na=False)]

        # 步骤2: 如果指定了层，进一步过滤
        if layer and layer != 'ALL':
            if 'layer' in region_neurons.columns:
                layer_neurons = region_neurons[region_neurons['layer'] == layer]
            else:
                # 尝试从celltype中提取层信息
                layer_neurons = region_neurons[region_neurons['celltype'].apply(
                    lambda x: extract_layer_from_celltype(x) == layer if isinstance(x, str) else False
                )]
        else:
            layer_neurons = region_neurons

        # 记录匹配到的神经元数量
        neuron_count = len(layer_neurons)
        layer_str = f"层{layer}" if layer else "所有层"
        logger.info(f"区域{region_id}{layer_str}匹配到 {neuron_count} 个神经元")

        # 步骤3: 计算统计数据
        # 1. 总神经元数
        stats['number_of_neuron_morphologies'] = neuron_count

        # 2. 使用has_apical, has_recon_axon, has_recon_den列计算
        if 'has_apical' in layer_neurons.columns:
            stats['number_of_apical_dendritic_morphologies'] = int(layer_neurons['has_apical'].sum())

        if 'has_recon_axon' in layer_neurons.columns:
            stats['number_of_axonal_morphologies'] = int(layer_neurons['has_recon_axon'].sum())

        if 'has_recon_den' in layer_neurons.columns:
            stats['number_of_dendritic_morphologies'] = int(layer_neurons['has_recon_den'].sum())

        # 3. 如果没有这些列，尝试从axon_df和dendrite_df中统计
        neuron_ids = set(layer_neurons['ID'].tolist()) if 'ID' in layer_neurons.columns else set()

        if neuron_ids and 'number_of_axonal_morphologies' not in stats and self.layer_calculator.axon_df is not None:
            axon_df = self.layer_calculator.axon_df
            if 'ID' in axon_df.columns:
                axon_neurons = axon_df[axon_df['ID'].isin(neuron_ids)]
                stats['number_of_axonal_morphologies'] = len(axon_neurons['ID'].unique())

        if neuron_ids and 'number_of_dendritic_morphologies' not in stats and self.layer_calculator.dendrite_df is not None:
            dendrite_df = self.layer_calculator.dendrite_df
            if 'ID' in dendrite_df.columns:
                dendrite_neurons = dendrite_df[dendrite_df['ID'].isin(neuron_ids)]
                stats['number_of_dendritic_morphologies'] = len(dendrite_neurons['ID'].unique())

                # 计算顶树突数量
                if 'number_of_apical_dendritic_morphologies' not in stats and 'type' in dendrite_df.columns:
                    apical_neurons = dendrite_neurons[
                        dendrite_neurons['type'].str.contains('apical', case=False, na=False)]
                    stats['number_of_apical_dendritic_morphologies'] = len(apical_neurons['ID'].unique())

        # 确保所有统计属性都有值
        for attr in STAT_ATTRIBUTES:
            if attr not in stats:
                stats[attr] = 0

        return stats

    def _get_region_acronym(self, region_id):
        """获取区域的acronym"""
        # 从region_analyzer获取
        if hasattr(self, 'region_analyzer') and self.region_analyzer:
            region_info = self.region_analyzer.get_region_info(region_id)
            if region_info and 'acronym' in region_info:
                return region_info['acronym']

        # 从region_data获取
        if hasattr(self, 'region_data') and self.region_data is not None:
            region_row = self.region_data[self.region_data['region_id'] == region_id]
            if not region_row.empty and 'acronym' in region_row.columns:
                return region_row.iloc[0]['acronym']

        return None

    def _calculate_region_statistics(self, region_id, region_data, merfish_cells):
        """为Region节点计算统计属性"""
        # 使用共享方法计算形态统计数据
        stats = self._calculate_morphology_statistics(region_id)

        # 计算转录组神经元数量
        if merfish_cells is not None and not merfish_cells.empty and 'region_id' in merfish_cells.columns:
            region_cells = merfish_cells[merfish_cells['region_id'] == region_id]
            stats['number_of_transcriptomic_neurons'] = len(region_cells)

        return stats

    def generate_regionlayer_nodes(self, regionlayer_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """生成包含完整形态学和统计属性的RegionLayer节点"""
        logger.info("生成RegionLayer节点...")

        # 存储region_data供_get_region_acronym使用
        if 'region_data' not in vars(self):
            self.region_data = regionlayer_data

        regionlayers = []
        for _, rl in regionlayer_data.iterrows():
            # 获取RegionLayer ID
            rl_id = rl.get('rl_id', '')

            # 解析region_id和layer
            parts = rl_id.split('_')
            if len(parts) < 2:
                logger.warning(f"跳过无效的rl_id: {rl_id}")
                continue

            region_id_str = parts[0]
            layer = parts[1]

            try:
                region_id = int(region_id_str)
            except (ValueError, TypeError):
                logger.warning(f"跳过非数字region_id: {region_id_str}")
                continue

            # 创建RegionLayer字典
            rl_dict = {
                'rl_id:ID(RegionLayer)': rl_id,
                'region_id:int': region_id,
                'layer': layer,
                'name': f"{rl.get('region_name', f'Region {region_id}')} {layer}",
            }

            # 添加形态学属性
            for attr in MORPH_ATTRIBUTES:
                if attr in rl:
                    rl_dict[f'{attr}:float'] = float(rl[attr])
                else:
                    rl_dict[f'{attr}:float'] = 0.0

            # 计算统计属性 - 使用共享的统计计算方法
            stats = self._calculate_morphology_statistics(region_id, layer)

            # 添加统计属性
            for attr in STAT_ATTRIBUTES:
                if attr in stats:
                    rl_dict[f'{attr}:int'] = stats[attr]
                else:
                    rl_dict[f'{attr}:int'] = 0

            # 添加MERFISH特有的属性
            if merfish_cells is not None and not merfish_cells.empty:
                # 计算该层区域中的转录组神经元数量
                if 'region_id' in merfish_cells.columns:
                    layer_cells = merfish_cells[(merfish_cells['region_id'] == region_id)]

                    # 如果有层信息，进一步筛选
                    if 'layer' in merfish_cells.columns:
                        layer_cells = layer_cells[layer_cells['layer'] == layer]

                    # 更新转录组神经元数量
                    rl_dict['number_of_transcriptomic_neurons:int'] = len(layer_cells)

            regionlayers.append(rl_dict)

        # 保存到CSV
        self._save_nodes(regionlayers, "regionlayers")
        logger.info(f"保存了 {len(regionlayers)} 个RegionLayer节点")

    def generate_merfish_nodes_from_hierarchy(self, merfish_cells: pd.DataFrame):
        """从层级数据生成MERFISH节点，包含所有必要的元数据"""
        if not self.hierarchy_loader:
            logger.error("未设置层级加载器")
            return

        # 计算每个类型的细胞数量
        if not merfish_cells.empty:
            class_counts = merfish_cells['class'].value_counts().to_dict() if 'class' in merfish_cells.columns else {}
            subclass_counts = merfish_cells[
                'subclass'].value_counts().to_dict() if 'subclass' in merfish_cells.columns else {}
            supertype_counts = merfish_cells[
                'supertype'].value_counts().to_dict() if 'supertype' in merfish_cells.columns else {}
            cluster_counts = merfish_cells[
                'cluster'].value_counts().to_dict() if 'cluster' in merfish_cells.columns else {}
        else:
            class_counts = subclass_counts = supertype_counts = cluster_counts = {}

        # 生成Class节点
        logger.info("生成Class节点（从层级数据）...")
        class_nodes = []
        for name, data in self.hierarchy_loader.class_data.items():
            node = {
                'tran_id:ID(Class)': data['tran_id'],
                'id': data['original_id'] + 300,
                'name': name,
                'neighborhood': data['neighborhood'],
                'number_of_child_types': data['number_of_child_types'],
                'number_of_neurons': class_counts.get(name, 0),
                # 添加缺少的元数据
                'dominant_neurotransmitter_type': data.get('dominant_neurotransmitter_type', ''),
                'markers': data.get('markers', '')
            }
            class_nodes.append(node)

        if class_nodes:
            df = pd.DataFrame(class_nodes)
            output_file = self.nodes_dir / "class.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个Class节点")

        # 生成Subclass节点
        logger.info("生成Subclass节点（从层级数据）...")
        subclass_nodes = []
        for name, data in self.hierarchy_loader.subclass_data.items():
            node = {
                'tran_id:ID(Subclass)': data['tran_id'],
                'id': data['original_id'] + 350,
                'name': name,
                'neighborhood': data['neighborhood'],
                'dominant_neurotransmitter_type': data['dominant_neurotransmitter_type'],
                'number_of_child_types': data['number_of_child_types'],
                'number_of_neurons': subclass_counts.get(name, 0),
                'markers': data['markers'],
                'transcription_factor_markers': data['transcription_factor_markers']
            }
            subclass_nodes.append(node)

        if subclass_nodes:
            df = pd.DataFrame(subclass_nodes)
            output_file = self.nodes_dir / "subclass.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个Subclass节点")

        # 生成Supertype节点
        logger.info("生成Supertype节点（从层级数据）...")
        supertype_nodes = []
        for name, data in self.hierarchy_loader.supertype_data.items():
            node = {
                'tran_id:ID(Supertype)': data['tran_id'],
                'id': data['original_id'] + 600,
                'name': name,
                'number_of_child_types': data['number_of_child_types'],
                'number_of_neurons': supertype_counts.get(name, 0),
                'markers': data['markers'],
                'within_subclass_markers': data['within_subclass_markers']
            }
            supertype_nodes.append(node)

        if supertype_nodes:
            df = pd.DataFrame(supertype_nodes)
            output_file = self.nodes_dir / "supertype.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个Supertype节点")

        # 生成Cluster节点
        logger.info("生成Cluster节点（从层级数据）...")
        cluster_nodes = []
        for name, data in self.hierarchy_loader.cluster_data.items():
            node = {
                'tran_id:ID(Cluster)': data['tran_id'],
                'id': data['original_id'] + 1800,
                'name': name,
                'anatomical_annotation': data['anatomical_annotation'],
                'broad_region_distribution': data['broad_region_distribution'],
                'dominant_neurotransmitter_type': data['dominant_neurotransmitter_type'],
                'number_of_neurons': cluster_counts.get(name, 0),
                'markers': data['markers'],
                'neuropeptide_mark_genes': data['neuropeptide_mark_genes'],
                'neurotransmitter_mark_genes': data['neurotransmitter_mark_genes'],
                'transcription_factor_markers': data['transcription_factor_markers'],
                'within_subclass_markers': data['within_subclass_markers']
            }
            cluster_nodes.append(node)

        if cluster_nodes:
            df = pd.DataFrame(cluster_nodes)
            output_file = self.nodes_dir / "cluster.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个Cluster节点")

    def generate_has_layer_relationships(self, region_data: pd.DataFrame, regionlayer_data: pd.DataFrame):
        """生成HAS_LAYER关系 - 处理非数字region_id"""
        logger.info("生成HAS_LAYER关系...")

        # 从regionlayer_data获取实际存在的region-layer组合
        existing_combinations = set()
        regionlayer_id_map = {}  # 存储rl_id到(region_id, layer)的映射

        for _, rl in regionlayer_data.iterrows():
            rl_id = rl.get('rl_id')
            region_id_raw = rl.get('region_id')
            layer = rl.get('layer')

            if rl_id and region_id_raw is not None and layer:
                # 处理非数字region_id
                if isinstance(region_id_raw, str) and not region_id_raw.isdigit():
                    # 使用映射的数字ID
                    if hasattr(self, 'region_id_mapping') and region_id_raw in self.region_id_mapping:
                        region_id_int = self.region_id_mapping[region_id_raw]
                    else:
                        # 如果没有映射，创建一个并记录
                        if not hasattr(self, 'region_id_mapping'):
                            self.region_id_mapping = {}
                            self.next_region_id = -1000

                        self.region_id_mapping[region_id_raw] = self.next_region_id
                        region_id_int = self.next_region_id
                        self.next_region_id -= 1
                        logger.info(f"在关系中为非数字region_id '{region_id_raw}' 创建映射ID: {region_id_int}")
                else:
                    # 对于数字ID，直接转换
                    try:
                        region_id_int = int(region_id_raw)
                    except (ValueError, TypeError):
                        # 如果转换失败，跳过
                        logger.warning(f"跳过无效的region_id: {region_id_raw}")
                        continue

                # 重建rl_id以确保一致性
                consistent_rl_id = f"{region_id_int}_{layer}"
                existing_combinations.add((region_id_int, layer))
                regionlayer_id_map[consistent_rl_id] = (region_id_int, layer)

        relationships = []
        for region_id, layer in existing_combinations:
            rl_id = f"{region_id}_{layer}"

            rel = {
                ':START_ID(Region)': region_id,
                ':END_ID(RegionLayer)': rl_id,
                ':TYPE': 'HAS_LAYER'
            }
            relationships.append(rel)

        self._save_relationships_batch(relationships, "has_layer")
        logger.info(f"保存了 {len(relationships)} 个HAS_LAYER关系")

    def generate_has_relationships_optimized(self, merfish_cells: pd.DataFrame, level: str):
        """优化的HAS_*关系生成，支持非皮层区域"""
        logger.info(f"生成HAS_{level.upper()}关系...")

        if level not in merfish_cells.columns:
            logger.warning(f"没有{level}数据")
            return

        if 'region_id' not in merfish_cells.columns:
            logger.warning("细胞缺少region_id")
            return

        # 获取ID映射
        if level == 'class':
            id_map = self.class_id_map
        elif level == 'subclass':
            id_map = self.subclass_id_map
        elif level == 'supertype':
            id_map = self.supertype_id_map
        else:
            id_map = self.cluster_id_map

        if not id_map:
            logger.warning(f"{level} ID映射为空")
            return

        # 筛选有效细胞
        valid_cells = merfish_cells[(merfish_cells[level].notna()) & (merfish_cells['region_id'] > 0)]

        # 按区域和类型分组计数
        grouped = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')
        region_totals = valid_cells.groupby('region_id').size().to_dict()

        # 获取已存在的RegionLayer节点
        regionlayer_file = self.nodes_dir / "regionlayers.csv"
        existing_regionlayers = set()
        if regionlayer_file.exists():
            try:
                rl_df = pd.read_csv(regionlayer_file)
                # 提取所有rl_id，可能是'rl_id:ID(RegionLayer)'列
                id_col = [col for col in rl_df.columns if 'rl_id' in col][0] if any(
                    'rl_id' in col for col in rl_df.columns) else None
                if id_col:
                    existing_regionlayers = set(rl_df[id_col].values)
            except Exception as e:
                logger.error(f"读取RegionLayer节点失败: {e}")

        relationships = []
        for region_id in grouped['region_id'].unique():
            region_group = grouped[grouped['region_id'] == region_id]
            total = region_totals.get(region_id, 1)

            region_group = region_group.copy()
            region_group['pct'] = region_group['count'] / total
            region_group = region_group[region_group['pct'] >= PCT_THRESHOLD]
            region_group['rank'] = region_group['pct'].rank(ascending=False, method='dense').astype(int)

            # 检查是否为皮层区域
            is_cortical = False
            if hasattr(self, 'region_analyzer') and self.region_analyzer:
                is_cortical = self.region_analyzer.is_cortical_region(region_id)

            # 确定要处理的层
            if is_cortical:
                # 皮层区域处理所有层
                layers_to_process = LAYERS
            else:
                # 非皮层区域只处理ALL层
                layers_to_process = ['ALL']

            # 为每个层创建关系
            for layer in layers_to_process:
                rl_id = f"{int(region_id)}_{layer}"

                # 检查RegionLayer节点是否存在
                if rl_id not in existing_regionlayers:
                    logger.warning(f"RegionLayer节点 {rl_id} 不存在，跳过")
                    continue

                for _, row in region_group.iterrows():
                    cell_type = row[level]
                    if cell_type in id_map:
                        rel = {
                            ':START_ID(RegionLayer)': rl_id,
                            f':END_ID({level.capitalize()})': id_map[cell_type],
                            'pct_cells:float': float(row['pct']),
                            'rank:int': int(row['rank']),
                            ':TYPE': f'HAS_{level.upper()}'
                        }
                        relationships.append(rel)

        self._save_relationships_batch(relationships, f"has_{level}")
        logger.info(f"保存了 {len(relationships)} 个HAS_{level.upper()}关系")

    def generate_dominant_transcriptomic_relationships(self, merfish_cells: pd.DataFrame):
        """为所有层级(Class, Subclass, Supertype, Cluster)生成Dominant_transcriptomic关系"""
        logger.info("生成Dominant_transcriptomic关系...")

        if merfish_cells.empty:
            logger.warning("没有细胞数据，无法生成转录组关系")
            return

        # 验证必要的列
        hierarchy_levels = ['class', 'subclass', 'supertype', 'cluster']
        missing_cols = [col for col in hierarchy_levels if col not in merfish_cells.columns]
        if missing_cols:
            logger.warning(f"细胞数据缺少必要的列: {missing_cols}")
            hierarchy_levels = [col for col in hierarchy_levels if col not in missing_cols]

        if not hierarchy_levels:
            logger.error("没有可用的层级列，无法继续")
            return

        if 'region_id' not in merfish_cells.columns:
            logger.warning("细胞数据缺少region_id列")
            return

        # 为每个层级生成关系
        for level in hierarchy_levels:
            self._generate_dominant_transcriptomic_for_level(merfish_cells, level)

    def _generate_dominant_transcriptomic_for_level(self, merfish_cells: pd.DataFrame, level: str):
        """为特定层级生成Dominant_transcriptomic关系，支持非皮层区域"""
        logger.info(f"生成{level.capitalize()}级别的Dominant_transcriptomic关系...")

        # 获取ID映射
        if level == 'class':
            id_map = self.class_id_map
        elif level == 'subclass':
            id_map = self.subclass_id_map
        elif level == 'supertype':
            id_map = self.supertype_id_map
        else:  # cluster
            id_map = self.cluster_id_map

        if not id_map:
            logger.warning(f"{level} ID映射为空")
            return

        # 按区域和类型分组计数
        valid_cells = merfish_cells[(merfish_cells[level].notna()) & (merfish_cells['region_id'] > 0)]
        if len(valid_cells) == 0:
            logger.warning(f"没有同时有{level}和区域ID的细胞")
            return

        # 计算每个区域中每种类型的细胞数量
        grouped = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')

        # 计算区域总数
        region_totals = valid_cells.groupby('region_id').size().to_dict()

        # 计算比例并找出每个区域的主导类型
        relationships = []
        processed_regions = set()

        for region_id in grouped['region_id'].unique():
            region_group = grouped[grouped['region_id'] == region_id]
            total = region_totals.get(region_id, 1)

            # 计算比例
            region_group = region_group.copy()
            region_group['pct'] = region_group['count'] / total

            # 按比例排序并取前3个
            top_types = region_group.sort_values('pct', ascending=False).head(3)

            for i, (_, row) in enumerate(top_types.iterrows()):
                cell_type = row[level]
                if cell_type in id_map:
                    rel = {
                        ':START_ID(Region)': int(region_id),
                        f':END_ID({level.capitalize()})': id_map[cell_type],
                        'pct:float': float(row['pct']),
                        'rank:int': i + 1,
                        ':TYPE': f'DOMINANT_TRANSCRIPTOMIC_{level.upper()}'
                    }
                    relationships.append(rel)
                    processed_regions.add(region_id)
                else:
                    logger.warning(f"{level}类型 '{cell_type}' 不在ID映射中")

        # 保存关系
        if relationships:
            # 确保输出目录存在
            rel_dir = self.relationships_dir
            rel_dir.mkdir(parents=True, exist_ok=True)

            # 确定输出文件
            output_file = rel_dir / f"dominant_transcriptomic_{level}.csv"

            # 创建DataFrame并保存
            df = pd.DataFrame(relationships)
            df.to_csv(output_file, index=False)

            logger.info(f"保存了 {len(relationships)} 个{level}级别的Dominant_transcriptomic关系到 {output_file}")
        else:
            logger.warning(f"没有生成{level}级别的Dominant_transcriptomic关系")
    def generate_belongs_to_from_hierarchy(self):
        """从层级数据生成BELONGS_TO关系"""
        if not self.hierarchy_loader:
            logger.error("未设置层级加载器")
            return

        logger.info("生成BELONGS_TO关系...")
        relationships = []

        # Subclass -> Class
        for subclass, class_name in self.hierarchy_loader.hierarchy_relations['subclass_to_class'].items():
            if subclass in self.subclass_id_map and class_name in self.class_id_map:
                rel = {
                    ':START_ID(Subclass)': self.subclass_id_map[subclass],
                    ':END_ID(Class)': self.class_id_map[class_name],
                    ':TYPE': 'BELONGS_TO'
                }
                relationships.append(rel)

        # Supertype -> Subclass
        for supertype, subclass in self.hierarchy_loader.hierarchy_relations['supertype_to_subclass'].items():
            if supertype in self.supertype_id_map and subclass in self.subclass_id_map:
                rel = {
                    ':START_ID(Supertype)': self.supertype_id_map[supertype],
                    ':END_ID(Subclass)': self.subclass_id_map[subclass],
                    ':TYPE': 'BELONGS_TO'
                }
                relationships.append(rel)

        # Cluster -> Supertype
        for cluster, supertype in self.hierarchy_loader.hierarchy_relations['cluster_to_supertype'].items():
            if cluster in self.cluster_id_map and supertype in self.supertype_id_map:
                rel = {
                    ':START_ID(Cluster)': self.cluster_id_map[cluster],
                    ':END_ID(Supertype)': self.supertype_id_map[supertype],
                    ':TYPE': 'BELONGS_TO'
                }
                relationships.append(rel)

        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "belongs_to.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个BELONGS_TO关系")

    def generate_project_to_relationships(self, projection_data: pd.DataFrame):
        """
        从神经元级别的投射数据生成区域间PROJECT_TO关系

        数据格式：
        - 行：神经元
        - 列：proj_axon_[区域]_rela 和 proj_axon_[区域]_abs
        """
        if projection_data is None or projection_data.empty:
            logger.warning("没有投影数据")
            return

        logger.info("生成PROJECT_TO关系...")
        logger.info(f"投影数据形状: {projection_data.shape}")

        # 1. 提取所有目标区域名称
        target_regions = set()
        region_to_id = {}  # 区域名称到ID的映射

        # 加载区域ID映射
        tree_data = None
        if hasattr(self, 'region_analyzer') and self.region_analyzer:
            # 从region_analyzer获取映射
            for region_id, info in self.region_analyzer.region_info.items():
                acronym = info.get('acronym', '')
                if acronym:
                    region_to_id[acronym] = region_id
        else:
            # 尝试加载tree_yzx.json
            tree_file = Path(f"{self.output_dir.parent}/tree_yzx.json")
            if tree_file.exists():
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)

                for node in tree_data:
                    if 'id' in node and 'acronym' in node:
                        region_to_id[node['acronym']] = node['id']

                logger.info(f"从tree_yzx.json加载了 {len(region_to_id)} 个区域ID映射")

        # 2. 解析列名中的区域名称
        for col in projection_data.columns:
            if col.startswith('proj_axon_') and (col.endswith('_rela') or col.endswith('_abs')):
                parts = col.split('_')
                if len(parts) >= 3:
                    # 提取中间的区域名称部分
                    region = '_'.join(parts[2:-1])  # 处理可能包含下划线的区域名
                    target_regions.add(region)

        logger.info(f"从列名中提取到 {len(target_regions)} 个目标区域: {sorted(list(target_regions))}")

        # 3. 确定每个神经元的源区域
        # 先检查是否有info.csv以获取神经元的源区域
        neuron_to_source = {}
        source_regions = set()

        # 3.1 尝试从神经元ID中提取源区域信息
        for neuron_id in projection_data.index:
            if isinstance(neuron_id, str):
                # 从ID中提取区域部分(如果格式允许)
                source_region = None

                # 检查是否为SEU-ALLEN_full_15257格式
                parts = neuron_id.split('_')
                if len(parts) >= 3 and parts[0] == 'SEU-ALLEN' and parts[1] == 'full':
                    try:
                        # 假设第3部分是区域ID
                        region_id = int(parts[2])
                        if region_id in self.region_analyzer.region_info:
                            acronym = self.region_analyzer.region_info[region_id].get('acronym', '')
                            if acronym:
                                source_region = acronym
                                source_regions.add(acronym)
                                neuron_to_source[neuron_id] = acronym
                    except (ValueError, IndexError):
                        pass

        # 3.2 如果有layer_calculator，尝试从其info_df中获取源区域
        if not neuron_to_source and hasattr(self, 'layer_calculator') and self.layer_calculator:
            if self.layer_calculator.info_df is not None:
                info_df = self.layer_calculator.info_df

                # 将ID转换为字符串以匹配
                if 'ID' in info_df.columns:
                    info_df['ID_str'] = info_df['ID'].astype(str)

                    for neuron_id in projection_data.index:
                        matches = info_df[info_df['ID_str'] == str(neuron_id)]
                        if not matches.empty:
                            if 'base_region' in matches.columns:
                                source_region = matches.iloc[0]['base_region']
                                neuron_to_source[neuron_id] = source_region
                                source_regions.add(source_region)
                            elif 'celltype' in matches.columns:
                                # 从celltype提取区域部分
                                celltype = matches.iloc[0]['celltype']
                                base_region = extract_base_region_from_celltype(celltype)
                                neuron_to_source[neuron_id] = base_region
                                source_regions.add(base_region)

        # 3.3 如果还是找不到源区域，使用一个默认值
        if not neuron_to_source:
            logger.warning("无法确定神经元的源区域，使用默认值'Unknown'")
            for neuron_id in projection_data.index:
                neuron_to_source[neuron_id] = 'Unknown'
            source_regions.add('Unknown')

        logger.info(f"确定了 {len(neuron_to_source)}/{len(projection_data)} 个神经元的源区域")
        logger.info(f"源区域: {sorted(list(source_regions))}")

        # 4. 计算区域间的投射总和 (使用相对值rela列)
        region_connections = {}  # (源区域, 目标区域) -> 投射强度总和

        for source in source_regions:
            # 获取该源区域的所有神经元
            source_neurons = [nid for nid, src in neuron_to_source.items() if src == source]

            if not source_neurons:
                continue

            # 计算到每个目标区域的投射总和
            for target in target_regions:
                rela_col = f"proj_axon_{target}_rela"

                if rela_col not in projection_data.columns:
                    continue

                # 计算所有该源区域神经元到此目标区域的投射总和
                total_projection = 0
                count = 0

                for neuron_id in source_neurons:
                    if neuron_id in projection_data.index:
                        val = projection_data.loc[neuron_id, rela_col]
                        if pd.notna(val) and val > 0:
                            total_projection += val
                            count += 1

                # 只保存有投射的连接
                if total_projection > 0:
                    region_connections[(source, target)] = {
                        'total': total_projection,
                        'count': count,
                        'average': total_projection / count if count > 0 else 0
                    }

        logger.info(f"计算了 {len(region_connections)} 个区域间连接")

        # 5. 生成PROJECT_TO关系
        relationships = []
        missing_source_ids = set()
        missing_target_ids = set()

        for (source, target), stats in region_connections.items():
            # 获取源区域和目标区域的ID
            source_id = region_to_id.get(source)
            target_id = region_to_id.get(target)

            if not source_id:
                missing_source_ids.add(source)
                continue

            if not target_id:
                missing_target_ids.add(target)
                continue

            # 创建关系
            rel = {
                ':START_ID(Region)': source_id,
                ':END_ID(Region)': target_id,
                'weight:float': float(stats['average']),  # 使用平均投射强度
                'total:float': float(stats['total']),  # 总投射强度
                'neuron_count:int': int(stats['count']),  # 参与投射的神经元数量
                ':TYPE': 'PROJECT_TO'
            }

            relationships.append(rel)

        if missing_source_ids:
            logger.warning(f"以下源区域没有找到ID映射: {sorted(list(missing_source_ids))}")

        if missing_target_ids:
            logger.warning(f"以下目标区域没有找到ID映射: {sorted(list(missing_target_ids))}")

        # 6. 保存关系
        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "project_to.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个PROJECT_TO关系到 {output_file}")
        else:
            logger.warning("没有生成任何PROJECT_TO关系")

    def generate_import_script(self):
        """生成Neo4j导入脚本"""
        script = """#!/bin/bash
# Neo4j批量导入脚本
NEO4J_HOME=/path/to/neo4j
DATABASE_NAME=neuroxiv

$NEO4J_HOME/bin/neo4j stop
rm -rf $NEO4J_HOME/data/databases/$DATABASE_NAME

$NEO4J_HOME/bin/neo4j-admin import \\
   --database=$DATABASE_NAME \\
   --nodes=Region=nodes/regions.csv \\
   --nodes=RegionLayer=nodes/regionlayers.csv \\
   --nodes=Class=nodes/class.csv \\
   --nodes=Subclass=nodes/subclass.csv \\
   --nodes=Supertype=nodes/supertype.csv \\
   --nodes=Cluster=nodes/cluster.csv \\
   --relationships=relationships/*.csv \\
   --skip-bad-relationships=true \\
   --skip-duplicate-nodes=true

$NEO4J_HOME/bin/neo4j start
echo "导入完成！"
"""
        script_file = self.output_dir / "import_to_neo4j.sh"
        with open(script_file, 'w') as f:
            f.write(script)

        import os
        os.chmod(script_file, 0o755)
        logger.info(f"生成了Neo4j导入脚本: {script_file}")

    def generate_statistics_report(self, region_data, regionlayer_data, merfish_cells):
        """生成统计报告"""
        report = []
        report.append("=" * 60)
        report.append("NeuroXiv 2.0 知识图谱统计报告")
        report.append("=" * 60)
        report.append(f"节点统计:")
        report.append(f"  - Region节点: {len(region_data)}")
        report.append(f"  - RegionLayer节点: {len(regionlayer_data)}")

        if self.hierarchy_loader:
            report.append(f"  - Class节点: {len(self.hierarchy_loader.class_data)}")
            report.append(f"  - Subclass节点: {len(self.hierarchy_loader.subclass_data)}")
            report.append(f"  - Supertype节点: {len(self.hierarchy_loader.supertype_data)}")
            report.append(f"  - Cluster节点: {len(self.hierarchy_loader.cluster_data)}")

        report.append(f"\n数据统计:")
        report.append(f"  - 总细胞数: {len(merfish_cells)}")
        report.append(f"生成时间: {pd.Timestamp.now()}")
        report.append("=" * 60)

        report_text = "\n".join(report)
        report_file = self.output_dir / "statistics_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        logger.info(f"生成了统计报告: {report_file}")
        print(report_text)

    def _save_nodes(self, nodes: List[Dict], node_type: str):
        """保存节点数据到CSV文件"""
        if not nodes:
            logger.warning(f"没有{node_type}节点需要保存")
            return

        output_file = self.nodes_dir / f"{node_type}.csv"
        df = pd.DataFrame(nodes)
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
        logger.info(f"保存了{len(nodes)}个{node_type}节点到 {output_file}")

    def _save_relationships_batch(self, relationships: List[Dict], prefix: str):
        """批量保存关系"""
        if not relationships:
            return

        df = pd.DataFrame(relationships)
        output_file = self.relationships_dir / f"{prefix}.csv"
        df.to_csv(output_file, index=False)
        logger.debug(f"保存了 {len(df)} 个关系到 {output_file}")


# ==================== 主函数 ====================

def main(data_dir: str = "../data",
         output_dir: str = "./knowledge_graph",
         hierarchy_json: str = None):
    """主函数 - 修改版"""

    setup_logger()

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - 修复版")
    logger.info("=" * 60)

    # 初始化
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Phase 1: 数据加载
    logger.info("Phase 1: 数据加载")

    # 加载数据
    data = load_data(data_path)
    processed_data = prepare_analysis_data(data)
    region_data = processed_data.get('region_data', pd.DataFrame())
    merfish_cells = processed_data.get('merfish_cells', pd.DataFrame())
    projection_data = processed_data.get('projection_df', pd.DataFrame())

    # 加载树结构用于区域分析
    tree_data = processed_data.get('tree', [])
    builder = EnhancedKnowledgeGraphBuilder(output_path)

    if tree_data:
        logger.info("初始化区域分析器...")
        builder.region_analyzer = RegionAnalyzer(tree_data)
        logger.info(f"识别出 {len(builder.region_analyzer.cortical_regions)} 个皮层区域")
        logger.info(f"识别出 {len(builder.region_analyzer.standard_regions)} 个标准CCF区域")

    # Phase 2: 加载层级数据
    logger.info("Phase 2: 加载MERFISH层级数据")

    hierarchy_loader = MERFISHHierarchyLoader(Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json")
    if not hierarchy_loader.load_hierarchy():
        logger.error("无法加载层级数据")
        return

    # Phase 3: 计算层特异性形态数据
    logger.info("Phase 3: 计算层特异性形态数据")
    builder = EnhancedKnowledgeGraphBuilder(output_path)

    # 修改这一行，传入region_analyzer
    layer_calculator = LayerSpecificMorphologyCalculator(data_path, builder.region_analyzer)

    if layer_calculator.load_morphology_with_layers():
        # 设置layer_calculator
        builder.layer_calculator = layer_calculator

        # 计算层特异性形态数据
        regionlayer_data = layer_calculator.calculate_regionlayer_morphology(region_data, merfish_cells)
    else:
        logger.error("无法加载形态学数据和层信息")
        regionlayer_data = pd.DataFrame()

    # Phase 4: 知识图谱生成
    logger.info("Phase 4: 知识图谱生成")


    builder.set_hierarchy_loader(hierarchy_loader)
    builder.layer_calculator = layer_calculator

    # 生成节点
    logger.info("生成节点...")
    builder.generate_region_nodes(region_data, merfish_cells)  # 使用修改后的方法
    builder.generate_regionlayer_nodes(regionlayer_data, merfish_cells)
    builder.generate_merfish_nodes_from_hierarchy(merfish_cells)

    # 生成关系
    logger.info("生成关系...")
    builder.generate_has_layer_relationships(region_data, regionlayer_data)  # 使用修改后的方法

    for level in ['class', 'subclass', 'supertype', 'cluster']:
        builder.generate_has_relationships_optimized(merfish_cells, level)

    # 添加下面这行来生成主导转录组关系
    builder.generate_dominant_transcriptomic_relationships(merfish_cells)

    builder.generate_belongs_to_from_hierarchy()
    builder.generate_project_to_relationships(projection_data)

    # 后处理
    builder.generate_import_script()
    builder.generate_statistics_report(region_data, regionlayer_data, merfish_cells)

    logger.info("=" * 60)
    logger.info("知识图谱构建完成！")
    logger.info(f"输出目录: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./knowledge_graph',
                        help='输出目录路径')
    parser.add_argument('--hierarchy_json', type=str, default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json',
                        help='MERFISH层级JSON文件路径')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.hierarchy_json)