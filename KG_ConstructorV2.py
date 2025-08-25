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
        """加载形态数据和层信息"""
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
        self.region_name_id_map = {}
        self.region_id_map = {}

        # 存储层级数据
        self.hierarchy_loader = None

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
        """生成Region节点 - 只使用标准CCF区域"""
        logger.info("生成Region节点（标准CCF区域）...")

        # 如果有区域分析器，过滤出标准区域
        if hasattr(self, 'region_analyzer') and self.region_analyzer:
            # 只保留标准CCF区域
            standard_regions = []
            for _, region in region_data.iterrows():
                region_id = region.get('region_id')
                if region_id and self.region_analyzer.is_standard_region(region_id):
                    standard_regions.append(region)

            if standard_regions:
                logger.info(f"从 {len(region_data)} 个区域中筛选出 {len(standard_regions)} 个标准CCF区域")
                region_data_filtered = pd.DataFrame(standard_regions)
            else:
                region_data_filtered = region_data
        else:
            # 如果没有区域分析器，使用基于acronym的过滤
            # 移除包含层信息的区域（如ENTm2）
            filtered_regions = []
            for _, region in region_data.iterrows():
                region_acronym = region.get('acronym', '')
                region_name = region.get('name', '')

                # 检查是否包含层后缀
                has_layer_suffix = False
                layer_suffixes = ['1', '2', '3', '4', '5', '6', '2/3', '5a', '5b', '6a', '6b']
                for suffix in layer_suffixes:
                    if region_acronym.endswith(suffix) or region_name.endswith(suffix):
                        has_layer_suffix = True
                        break

                if not has_layer_suffix:
                    filtered_regions.append(region)

            logger.info(f"从 {len(region_data)} 个区域中筛选出 {len(filtered_regions)} 个标准区域（移除带层信息的）")
            region_data_filtered = pd.DataFrame(filtered_regions)

        regions = []
        for _, region in region_data_filtered.iterrows():
            region_id = region.get('region_id')
            if pd.isna(region_id):
                continue

            region_name = region.get('name', f'Region_{region_id}')
            region_acronym = region.get('acronym', '')

            # 计算统计属性
            stat_values = self._calculate_region_statistics(region_id, region_data_filtered, merfish_cells)

            # 创建区域字典
            region_dict = {
                'region_id:ID(Region)': int(region_id),
                'name': str(region_name),
                'acronym': str(region_acronym) if region_acronym else '',
            }

            # 添加形态学属性
            for attr in MORPH_ATTRIBUTES:
                if attr in region:
                    region_dict[f'{attr}:float'] = float(region[attr])
                else:
                    region_dict[f'{attr}:float'] = 0.0

            # 添加统计属性
            for attr in STAT_ATTRIBUTES:
                region_dict[f'{attr}:int'] = stat_values.get(attr, 0)

            regions.append(region_dict)

        # 保存到CSV
        self._save_nodes(regions, "regions")
        logger.info(f"保存了 {len(regions)} 个标准CCF Region节点")

    def _calculate_region_statistics(self, region_id, region_data, merfish_cells):
        """计算区域的统计属性"""
        stats = {}

        # 从形态数据计算
        if hasattr(self, 'layer_calculator') and self.layer_calculator:
            if self.layer_calculator.info_df is not None:
                region_neurons = self.layer_calculator.info_df[
                    self.layer_calculator.info_df['region_id'] == region_id
                ]
                neuron_ids = set(region_neurons['ID'].values) if 'ID' in region_neurons.columns else set()

                # 轴突形态数量
                if self.layer_calculator.axon_df is not None and 'ID' in self.layer_calculator.axon_df.columns:
                    axon_ids = self.layer_calculator.axon_df[
                        self.layer_calculator.axon_df['ID'].isin(neuron_ids)
                    ]['ID'].unique()
                    stats['number_of_axonal_morphologies'] = len(axon_ids)

                # 树突形态数量
                if self.layer_calculator.dendrite_df is not None and 'ID' in self.layer_calculator.dendrite_df.columns:
                    dendrite_ids = self.layer_calculator.dendrite_df[
                        self.layer_calculator.dendrite_df['ID'].isin(neuron_ids)
                    ]['ID'].unique()
                    stats['number_of_dendritic_morphologies'] = len(dendrite_ids)

                    # 顶树突数量
                    if 'type' in self.layer_calculator.dendrite_df.columns:
                        apical_dendrites = self.layer_calculator.dendrite_df[
                            (self.layer_calculator.dendrite_df['ID'].isin(neuron_ids)) &
                            (self.layer_calculator.dendrite_df['type'].str.contains('apical', case=False, na=False))
                        ]
                        stats['number_of_apical_dendritic_morphologies'] = len(apical_dendrites['ID'].unique())

                stats['number_of_neuron_morphologies'] = len(neuron_ids)

        # 从MERFISH数据计算转录组神经元数量
        if merfish_cells is not None and not merfish_cells.empty:
            if 'region_id' in merfish_cells.columns:
                region_cells = merfish_cells[merfish_cells['region_id'] == region_id]
                stats['number_of_transcriptomic_neurons'] = len(region_cells)

        # 填充缺失值
        for attr in STAT_ATTRIBUTES:
            if attr not in stats:
                stats[attr] = 0

        return stats

    def generate_regionlayer_nodes(self, regionlayer_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """生成RegionLayer节点"""
        logger.info("生成RegionLayer节点...")

        regionlayers = []
        for _, rl in regionlayer_data.iterrows():
            rl_id = rl.get('rl_id', '')
            region_id = rl.get('region_id', -1)
            layer = rl.get('layer', 'Unknown')

            # 创建RegionLayer字典
            rl_dict = {
                'rl_id:ID(RegionLayer)': rl_id,
                'region_id:int': int(region_id),
                'layer': layer,
                'name': f"{rl.get('region_name', f'Region {region_id}')} {layer}",
            }

            # 添加形态学属性
            for attr in MORPH_ATTRIBUTES:
                if attr in rl:
                    rl_dict[f'{attr}:float'] = float(rl[attr])
                else:
                    rl_dict[f'{attr}:float'] = 0.0

            # 添加统计属性
            for attr in STAT_ATTRIBUTES:
                if attr in rl:
                    rl_dict[f'{attr}:int'] = int(rl[attr])
                else:
                    rl_dict[f'{attr}:int'] = 0

            regionlayers.append(rl_dict)

        # 保存到CSV
        self._save_nodes(regionlayers, "regionlayers")
        logger.info(f"保存了 {len(regionlayers)} 个RegionLayer节点")

    def generate_merfish_nodes_from_hierarchy(self, merfish_cells: pd.DataFrame):
        """从层级数据生成MERFISH节点"""
        if not self.hierarchy_loader:
            logger.error("未设置层级加载器")
            return

        # 计算每个类型的细胞数量
        if not merfish_cells.empty:
            class_counts = merfish_cells['class'].value_counts().to_dict() if 'class' in merfish_cells.columns else {}
            subclass_counts = merfish_cells['subclass'].value_counts().to_dict() if 'subclass' in merfish_cells.columns else {}
            supertype_counts = merfish_cells['supertype'].value_counts().to_dict() if 'supertype' in merfish_cells.columns else {}
            cluster_counts = merfish_cells['cluster'].value_counts().to_dict() if 'cluster' in merfish_cells.columns else {}
        else:
            class_counts = subclass_counts = supertype_counts = cluster_counts = {}

        # 生成Class节点
        logger.info("生成Class节点...")
        class_nodes = []
        for name, data in self.hierarchy_loader.class_data.items():
            node = {
                'tran_id:ID(Class)': data['tran_id'],
                'name': name,
                'number_of_neurons': class_counts.get(name, 0)
            }
            class_nodes.append(node)

        if class_nodes:
            self._save_nodes(class_nodes, "class")

        # 生成Subclass节点
        logger.info("生成Subclass节点...")
        subclass_nodes = []
        for name, data in self.hierarchy_loader.subclass_data.items():
            node = {
                'tran_id:ID(Subclass)': data['tran_id'],
                'name': name,
                'number_of_neurons': subclass_counts.get(name, 0)
            }
            subclass_nodes.append(node)

        if subclass_nodes:
            self._save_nodes(subclass_nodes, "subclass")

        # 生成Supertype节点
        logger.info("生成Supertype节点...")
        supertype_nodes = []
        for name, data in self.hierarchy_loader.supertype_data.items():
            node = {
                'tran_id:ID(Supertype)': data['tran_id'],
                'name': name,
                'number_of_neurons': supertype_counts.get(name, 0)
            }
            supertype_nodes.append(node)

        if supertype_nodes:
            self._save_nodes(supertype_nodes, "supertype")

        # 生成Cluster节点
        logger.info("生成Cluster节点...")
        cluster_nodes = []
        for name, data in self.hierarchy_loader.cluster_data.items():
            node = {
                'tran_id:ID(Cluster)': data['tran_id'],
                'name': name,
                'number_of_neurons': cluster_counts.get(name, 0)
            }
            cluster_nodes.append(node)

        if cluster_nodes:
            self._save_nodes(cluster_nodes, "cluster")

    def generate_has_layer_relationships(self, region_data: pd.DataFrame, regionlayer_data: pd.DataFrame):
        """生成HAS_LAYER关系 - 基于实际存在的RegionLayer节点"""
        logger.info("生成HAS_LAYER关系...")

        # 从regionlayer_data获取实际存在的region-layer组合
        existing_combinations = set()
        for _, rl in regionlayer_data.iterrows():
            region_id = rl.get('region_id')
            layer = rl.get('layer')
            if region_id and layer:
                existing_combinations.add((int(region_id), layer))

        relationships = []
        for region_id, layer in existing_combinations:
            if layer == 'ALL':
                rl_id = f"{region_id}_ALL"
            else:
                rl_id = f"{region_id}_{layer}"

            rel = {
                ':START_ID(Region)': int(region_id),
                ':END_ID(RegionLayer)': rl_id,
                ':TYPE': 'HAS_LAYER'
            }
            relationships.append(rel)

        self._save_relationships_batch(relationships, "has_layer")
        logger.info(f"保存了 {len(relationships)} 个HAS_LAYER关系")

    def generate_has_relationships_optimized(self, merfish_cells: pd.DataFrame, level: str):
        """优化的HAS_*关系生成"""
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

        relationships = []
        for region_id in grouped['region_id'].unique():
            region_group = grouped[grouped['region_id'] == region_id]
            total = region_totals.get(region_id, 1)

            region_group = region_group.copy()
            region_group['pct'] = region_group['count'] / total
            region_group = region_group[region_group['pct'] >= PCT_THRESHOLD]
            region_group['rank'] = region_group['pct'].rank(ascending=False, method='dense').astype(int)

            for layer in LAYERS:
                rl_id = f"{int(region_id)}_{layer}"

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
        """生成PROJECT_TO关系"""
        if projection_data is None or projection_data.empty:
            logger.warning("没有投影数据")
            return

        logger.info("生成PROJECT_TO关系...")
        relationships = []

        # 检查必要的列
        source_col = None
        target_col = None

        for col in ['source_region', 'source_region_id', 'source', 'from_region']:
            if col in projection_data.columns:
                source_col = col
                break

        for col in ['target_region', 'target_region_id', 'target', 'to_region']:
            if col in projection_data.columns:
                target_col = col
                break

        if not source_col or not target_col:
            logger.error(f"投影数据缺少必要的列")
            return

        for _, proj in projection_data.iterrows():
            try:
                source = int(proj[source_col]) if pd.notna(proj[source_col]) else 0
                target = int(proj[target_col]) if pd.notna(proj[target_col]) else 0

                if source > 0 and target > 0:
                    rel = {
                        ':START_ID(Region)': source,
                        ':END_ID(Region)': target,
                        ':TYPE': 'PROJECT_TO'
                    }
                    relationships.append(rel)
            except (ValueError, TypeError):
                continue

        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "project_to.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个PROJECT_TO关系")

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
    region_analyzer = None
    if tree_data:
        logger.info("初始化区域分析器...")
        region_analyzer = RegionAnalyzer(tree_data)
        logger.info(f"识别出 {len(region_analyzer.cortical_regions)} 个皮层区域")
        logger.info(f"识别出 {len(region_analyzer.standard_regions)} 个标准CCF区域")

    # Phase 2: 加载层级数据
    logger.info("Phase 2: 加载MERFISH层级数据")

    hierarchy_loader = MERFISHHierarchyLoader(Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json")
    if not hierarchy_loader.load_hierarchy():
        logger.error("无法加载层级数据")
        return

    # Phase 3: 计算层特异性形态数据
    logger.info("Phase 3: 计算层特异性形态数据")

    # 传入区域分析器
    layer_calculator = LayerSpecificMorphologyCalculator(data_path, region_analyzer)
    if layer_calculator.load_morphology_with_layers():
        regionlayer_data = layer_calculator.calculate_regionlayer_morphology(region_data, merfish_cells)
    else:
        logger.error("无法加载形态学数据")
        regionlayer_data = pd.DataFrame()

    # Phase 4: 知识图谱生成
    logger.info("Phase 4: 知识图谱生成")

    builder = EnhancedKnowledgeGraphBuilder(output_path)
    builder.set_hierarchy_loader(hierarchy_loader)
    builder.layer_calculator = layer_calculator
    builder.region_analyzer = region_analyzer  # 设置区域分析器

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