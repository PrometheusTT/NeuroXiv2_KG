"""
NeuroXiv 2.0 知识图谱构建器 - 统一区域节点版本
整合MERFISH层级JSON和形态计算

作者: wangmajortom
日期: 2025-08-26
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

    # 特殊区域列表 - 这些区域名称中的数字不应被移除
    special_regions = ['CA1', 'CA2', 'CA3', 'LGd1', 'LGd2', 'LGd3', 'LGd4', 'LGd5', 'LGd6']

    # 检查是否为特殊区域
    for special in special_regions:
        if celltype.startswith(special):
            return special

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

    def get_region_by_acronym(self, acronym: str) -> int:
        """通过acronym获取区域ID"""
        for region_id, info in self.region_info.items():
            if info.get('acronym') == acronym:
                return region_id
        return None

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


# ==================== 形态数据加载器 ====================

class MorphologyDataLoader:
    """形态学数据加载器 - 按脑区计算"""

    def __init__(self, data_dir: Path, region_analyzer: RegionAnalyzer = None):
        self.data_dir = Path(data_dir)
        self.info_df = None
        self.axon_df = None
        self.dendrite_df = None
        self.region_analyzer = region_analyzer

    def load_morphology_data(self) -> bool:
        """加载形态数据"""
        # 加载info.csv
        info_file = self.data_dir / "info.csv"
        if not info_file.exists():
            logger.error(f"info.csv不存在: {info_file}")
            return False

        self.info_df = pd.read_csv(info_file)
        logger.info(f"加载了 {len(self.info_df)} 条神经元信息")

        # 过滤掉包含CCF-thin和local的行
        if 'ID' in self.info_df.columns:
            orig_len = len(self.info_df)
            self.info_df = self.info_df[~self.info_df['ID'].astype(str).str.contains('CCF-thin|local', na=False)]
            filtered_count = orig_len - len(self.info_df)
            logger.info(f"从info表中过滤掉了 {filtered_count} 条包含CCF-thin或local的行")

        # 提取基础区域和层信息
        if 'celltype' in self.info_df.columns:
            # 提取基础区域（不含层信息）
            self.info_df['base_region'] = self.info_df['celltype']
            # 提取层信息作为元数据
            self.info_df['layer'] = self.info_df['celltype'].apply(extract_layer_from_celltype)

            # 将base_region映射到region_id
            if self.region_analyzer:
                self.info_df['region_id'] = self.info_df['base_region'].apply(
                    lambda x: self.region_analyzer.get_region_by_acronym(x) if pd.notna(x) else None
                )
                valid_region_count = self.info_df['region_id'].notna().sum()
                logger.info(f"从base_region映射到region_id: {valid_region_count}/{len(self.info_df)}个有效映射")
        else:
            logger.warning("info.csv缺少celltype列")
            self.info_df['layer'] = 'Unknown'
            self.info_df['base_region'] = 'Unknown'

        # 加载轴突形态数据 - 过滤掉CCF-thin和local数据
        axon_file = self.data_dir / "axonfull_morpho.csv"
        if axon_file.exists():
            self.axon_df = pd.read_csv(axon_file)
            if 'name' in self.axon_df.columns:
                orig_len = len(self.axon_df)
                self.axon_df = self.axon_df[~self.axon_df['name'].str.contains('CCF-thin|local', na=False)]
                filtered_count = orig_len - len(self.axon_df)
                logger.info(f"加载了 {len(self.axon_df)} 条轴突形态数据 (过滤了 {filtered_count} 条CCF-thin或local数据)")
            else:
                logger.info(f"加载了 {len(self.axon_df)} 条轴突形态数据")

        # 加载树突形态数据 - 过滤掉CCF-thin和local数据
        dendrite_file = self.data_dir / "denfull_morpho.csv"
        if dendrite_file.exists():
            self.dendrite_df = pd.read_csv(dendrite_file)
            if 'name' in self.dendrite_df.columns:
                orig_len = len(self.dendrite_df)
                self.dendrite_df = self.dendrite_df[~self.dendrite_df['name'].str.contains('CCF-thin|local', na=False)]
                filtered_count = orig_len - len(self.dendrite_df)
                logger.info(f"加载了 {len(self.dendrite_df)} 条树突形态数据 (过滤了 {filtered_count} 条CCF-thin或local数据)")
            else:
                logger.info(f"加载了 {len(self.dendrite_df)} 条树突形态数据")

        return True

    def calculate_region_morphology(self, region_id: int) -> Dict:
        """计算特定区域的形态学特征"""
        stats = {}

        # 确保数据已加载
        if self.info_df is None or self.axon_df is None or self.dendrite_df is None:
            logger.warning(f"缺少必要的形态数据，无法计算区域{region_id}的统计信息")
            return {attr: 0 for attr in MORPH_ATTRIBUTES + STAT_ATTRIBUTES}

        # 找到属于该区域的神经元
        region_neurons = self.info_df[self.info_df['region_id'] == region_id]
        neuron_count = len(region_neurons)

        if neuron_count == 0:
            logger.debug(f"区域{region_id}没有匹配到神经元")
            return {attr: 0 for attr in MORPH_ATTRIBUTES + STAT_ATTRIBUTES}

        logger.info(f"区域{region_id}匹配到 {neuron_count} 个神经元")

        # 计算基本统计数据
        stats['number_of_neuron_morphologies'] = neuron_count
        stats['number_of_axonal_morphologies'] = 0
        stats['number_of_dendritic_morphologies'] = 0
        stats['number_of_apical_dendritic_morphologies'] = 0

        # 形态特征初始值
        for attr in MORPH_ATTRIBUTES:
            stats[attr] = 0.0

        # 获取神经元ID集合
        if 'ID' not in region_neurons.columns:
            return stats

        neuron_ids = set(region_neurons['ID'].dropna().astype(str).tolist())
        if not neuron_ids:
            return stats

        # 定义列名映射 - 添加键值对确保所有MORPH_ATTRIBUTES都有映射
        column_mapping = {
            "Average Bifurcation Angle Remote": ["axonal_bifurcation_remote_angle", "dendritic_bifurcation_remote_angle"],
            "Number of Bifurcations": ["axonal_branches", "dendritic_branches"],
            "Total Length": ["axonal_length", "dendritic_length"],
            "Max Branch Order": ["axonal_maximum_branch_order", "dendritic_maximum_branch_order"]
        }

        # 计算轴突特征
        if self.axon_df is not None:
            # 确保ID为字符串类型以便匹配
            self.axon_df['ID_str'] = self.axon_df['ID'].astype(str)
            axon_neurons = self.axon_df[self.axon_df['ID_str'].isin(neuron_ids)]

            if not axon_neurons.empty:
                axon_count = len(axon_neurons['ID_str'].unique())
                stats['number_of_axonal_morphologies'] = axon_count

                # 计算平均形态特征
                for csv_col, feature_cols in column_mapping.items():
                    if csv_col in axon_neurons.columns:
                        values = axon_neurons[csv_col].dropna()
                        if len(values) > 0:
                            # 只使用轴突相关的特征列
                            for feature_col in feature_cols:
                                if feature_col.startswith('axonal_'):
                                    stats[feature_col] = values.mean()

        # 计算树突特征
        if self.dendrite_df is not None:
            # 确保ID为字符串类型以便匹配
            self.dendrite_df['ID_str'] = self.dendrite_df['ID'].astype(str)
            dendrite_neurons = self.dendrite_df[self.dendrite_df['ID_str'].isin(neuron_ids)]

            if not dendrite_neurons.empty:
                dendrite_count = len(dendrite_neurons['ID_str'].unique())
                stats['number_of_dendritic_morphologies'] = dendrite_count

                # 计算顶树突数量
                if 'type' in dendrite_neurons.columns:
                    apical_neurons = dendrite_neurons[dendrite_neurons['type'].str.contains('apical', case=False, na=False)]
                    stats['number_of_apical_dendritic_morphologies'] = len(apical_neurons['ID_str'].unique())

                # 计算平均形态特征
                for csv_col, feature_cols in column_mapping.items():
                    if csv_col in dendrite_neurons.columns:
                        values = dendrite_neurons[csv_col].dropna()
                        if len(values) > 0:
                            # 只使用树突相关的特征列
                            for feature_col in feature_cols:
                                if feature_col.startswith('dendritic_'):
                                    stats[feature_col] = values.mean()

        # 添加调试信息
        logger.info(f"区域{region_id}形态统计: 神经元={stats['number_of_neuron_morphologies']}, "
                    f"轴突={stats['number_of_axonal_morphologies']}, "
                    f"树突={stats['number_of_dendritic_morphologies']}")

        # 记录非零形态特征
        non_zero_features = {k: v for k, v in stats.items() if k in MORPH_ATTRIBUTES and v > 0}
        if non_zero_features:
            logger.info(f"区域{region_id}非零形态特征: {non_zero_features}")

        return stats


# ==================== 知识图谱构建器 ====================

class KnowledgeGraphBuilder:
    """知识图谱构建器 - 统一区域节点版本"""

    def __init__(self, output_dir: Path):
        self.morphology_loader = None
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

    def generate_unified_region_nodes(self, region_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """
        生成统一的Region节点，使用最细粒度的脑区

        从MERFISH数据和info表中获取最细粒度的脑区，取并集作为Region节点
        不再区分Region和RegionLayer
        """
        logger.info("生成统一的Region节点（最细粒度脑区）...")

        # 1. 获取MERFISH数据中的区域（如果有）
        merfish_regions = set()
        if merfish_cells is not None and not merfish_cells.empty and 'region_id' in merfish_cells.columns:
            merfish_regions = set(merfish_cells['region_id'].dropna().unique())
            logger.info(f"从MERFISH数据中提取了 {len(merfish_regions)} 个区域ID")

        # 2. 从info表中提取区域
        info_regions = set()
        layer_mapping = {}  # 存储区域ID到layer的映射

        if hasattr(self, 'morphology_loader') and self.morphology_loader and self.morphology_loader.info_df is not None:
            info_df = self.morphology_loader.info_df

            # 2.1 如果info表有region_id列，直接使用
            if 'region_id' in info_df.columns:
                info_regions = set(info_df['region_id'].dropna().unique())

                # 提取layer信息
                if 'layer' in info_df.columns:
                    for region_id in info_regions:
                        # 获取该区域的所有细胞
                        cells = info_df[info_df['region_id'] == region_id]
                        # 找出最常见的layer
                        if not cells.empty:
                            layer_counts = cells['layer'].value_counts()
                            if not layer_counts.empty:
                                dominant_layer = layer_counts.idxmax()
                                layer_mapping[region_id] = dominant_layer

            # 2.2 如果info表只有celltype列，需要转换为region_id
            elif 'celltype' in info_df.columns and hasattr(self, 'region_analyzer') and self.region_analyzer:
                # 从celltype提取基础区域
                info_df['base_region'] = info_df['celltype']

                # 将base_region转换为region_id
                for region_name in info_df['base_region'].dropna().unique():
                    region_id = self.region_analyzer.get_region_by_acronym(region_name)
                    if region_id:
                        info_regions.add(region_id)

                        # 提取layer信息
                        if 'layer' in info_df.columns:
                            cells = info_df[info_df['base_region'] == region_name]
                            if not cells.empty:
                                layer_counts = cells['layer'].value_counts()
                                if not layer_counts.empty:
                                    dominant_layer = layer_counts.idxmax()
                                    layer_mapping[region_id] = dominant_layer

            logger.info(f"从info表中提取了 {len(info_regions)} 个区域ID")

        # 3. 合并区域列表
        all_region_ids = merfish_regions.union(info_regions)
        logger.info(f"合并后共有 {len(all_region_ids)} 个唯一区域ID")

        # 4. 生成Region节点
        regions = []

        for region_id in all_region_ids:
            if pd.isna(region_id):
                continue

            try:
                region_id_int = int(region_id)
            except (ValueError, TypeError):
                logger.warning(f"跳过非数字region_id: {region_id}")
                continue

            # 获取区域信息
            region_info = {}
            if hasattr(self, 'region_analyzer') and self.region_analyzer:
                region_info = self.region_analyzer.get_region_info(region_id_int)

            # 获取acronym
            acronym = region_info.get('acronym', '') or f"Region_{region_id_int}"

            # 创建区域字典
            region_dict = {
                'region_id:ID(Region)': region_id_int,
                'name': str(acronym),
                'full_name': str(region_info.get('name', acronym)),
                'acronym': str(acronym)
            }

            # 添加layer信息（如果有）
            if region_id_int in layer_mapping:
                region_dict['layer'] = layer_mapping[region_id_int]
            else:
                region_dict['layer'] = 'Unknown'

            # 添加parent_id
            if 'parent_id' in region_info:
                parent_id = region_info['parent_id']
                if parent_id is not None:
                    region_dict['parent_id:int'] = int(parent_id)
            else:
                region_dict['parent_id:int'] = 0

            # 添加颜色信息
            if 'color' in region_info:
                region_dict['color:int[]'] = region_info['color']

            # 添加形态学属性
            for attr in MORPH_ATTRIBUTES:
                region_dict[f'{attr}:float'] = 0.0  # 默认值

            # 统计属性稍后会更新
            for attr in STAT_ATTRIBUTES:
                region_dict[f'{attr}:int'] = 0

            regions.append(region_dict)

        # 5. 计算形态学特征和统计数据
        regions_df = pd.DataFrame(regions)
        logger.info(f"生成了 {len(regions_df)} 个Region节点基本信息")

        # 更新形态学特征和统计数据
        updated_regions = self._update_region_features(regions_df, merfish_cells)

        # 保存到CSV
        self._save_nodes(updated_regions, "regions")
        logger.info(f"保存了 {len(updated_regions)} 个统一的Region节点")

        # 保存region_data供其他方法使用
        self.region_data = pd.DataFrame(updated_regions)

        return updated_regions

    def _update_region_features(self, regions_df: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """更新区域的形态学特征和统计数据"""
        logger.info("更新区域的形态学特征和统计数据...")

        # 转换为列表字典以便更新
        regions = regions_df.to_dict('records')

        # 1. 计算形态学特征
        if hasattr(self, 'morphology_loader') and self.morphology_loader:
            for i, region in enumerate(regions):
                region_id = region['region_id:ID(Region)']

                # 使用新的计算方法
                stats = self.morphology_loader.calculate_region_morphology(region_id)

                # 更新形态学属性
                for attr in MORPH_ATTRIBUTES:
                    if attr in stats:
                        regions[i][f'{attr}:float'] = float(stats.get(attr, 0.0))

                # 更新统计属性
                for attr in STAT_ATTRIBUTES:
                    if attr in stats:
                        regions[i][f'{attr}:int'] = int(stats.get(attr, 0))

        # 2. 计算MERFISH相关统计（如果有）
        if merfish_cells is not None and not merfish_cells.empty and 'region_id' in merfish_cells.columns:
            for i, region in enumerate(regions):
                region_id = region['region_id:ID(Region)']

                # 计算该区域的MERFISH细胞数量
                region_cells = merfish_cells[merfish_cells['region_id'] == region_id]
                cell_count = len(region_cells)

                # 更新转录组神经元数量
                regions[i]['number_of_transcriptomic_neurons:int'] = cell_count

        return regions

    def generate_project_to_relationships(self, projection_data: pd.DataFrame):
        """
        从神经元级别的投射数据生成区域间PROJECT_TO关系
        采用新的计算逻辑：A脑区到B脑区的投射强度 = A脑区中到B脑区有投射的神经元的投射abs值之和/这批神经元数量
        """
        if projection_data is None or projection_data.empty:
            logger.warning("没有投影数据")
            return

        logger.info("生成PROJECT_TO关系...")
        logger.info(f"投影数据形状: {projection_data.shape}")

        # 确保我们有info表数据
        if not hasattr(self, 'morphology_loader') or self.morphology_loader is None or self.morphology_loader.info_df is None:
            logger.error("缺少info表数据，无法匹配源区域")
            return

        info_df = self.morphology_loader.info_df
        logger.info(f"info表形状: {info_df.shape}")

        # 确保ID列在info表中
        if 'ID' not in info_df.columns:
            logger.error("info表中没有ID列，无法匹配神经元")
            return

        # 创建标准化ID函数 - 处理ID格式差异
        def normalize_id(id_str):
            """标准化ID格式，移除CCF-thin或CCFv3后缀"""
            id_str = str(id_str)
            id_str = id_str.replace('CCF-thin', '').replace('CCFv3', '').strip('_')
            return id_str

        # 标准化info表中的ID
        info_df['normalized_id'] = info_df['ID'].apply(normalize_id)

        # 获取区域ID映射
        region_to_id = {}  # 区域名称到ID的映射
        id_to_acronym = {}  # ID到区域名称的映射

        if hasattr(self, 'region_analyzer') and self.region_analyzer:
            for region_id, info in self.region_analyzer.region_info.items():
                acronym = info.get('acronym', '')
                if acronym:
                    region_to_id[acronym] = region_id
                    id_to_acronym[region_id] = acronym

        # 解析列名中的目标区域名称
        target_regions = []
        for col in projection_data.columns:
            if col.startswith('proj_axon_') and col.endswith('_abs'):
                # 提取区域名称
                region = col.replace('proj_axon_', '').replace('_abs', '')
                target_regions.append(region)

        logger.info(f"找到 {len(target_regions)} 个目标区域")

        # 将投射数据索引转换为规范化ID进行匹配
        proj_normalized_ids = {}
        for idx in projection_data.index:
            proj_normalized_ids[normalize_id(idx)] = idx

        # 构建神经元ID到源区域的映射
        neuron_source_map = {}  # neuron_id -> (source_id, source_acronym)

        # 使用规范化ID进行匹配
        for normalized_id, info_rows in info_df.groupby('normalized_id'):
            if normalized_id in proj_normalized_ids:
                orig_idx = proj_normalized_ids[normalized_id]

                # 获取源区域信息
                if 'region_id' in info_rows.columns:
                    source_id = info_rows.iloc[0]['region_id']
                    if pd.notna(source_id):
                        source_acronym = id_to_acronym.get(source_id, '')
                        if source_acronym:
                            neuron_source_map[orig_idx] = (source_id, source_acronym)
                elif 'base_region' in info_rows.columns:
                    source_acronym = info_rows.iloc[0]['base_region']
                    if pd.notna(source_acronym):
                        source_id = region_to_id.get(source_acronym)
                        if source_id:
                            neuron_source_map[orig_idx] = (source_id, source_acronym)

        logger.info(f"规范化ID匹配: {len(neuron_source_map)}/{len(projection_data)} 个神经元匹配")

        # 计算区域间投射
        region_connections = {}  # (source_id, target_id) -> {'total': 值, 'count': 数量}

        # 遍历每个有源区域的神经元
        for neuron_id, (source_id, source_acronym) in neuron_source_map.items():
            # 获取该神经元的投射数据
            neuron_data = projection_data.loc[neuron_id]

            # 处理该神经元到各目标区域的投射
            for target_acronym in target_regions:
                # 跳过同源投射
                if source_acronym == target_acronym:
                    continue

                # 获取目标区域ID
                target_id = region_to_id.get(target_acronym)
                if not target_id:
                    continue

                # 获取投射强度列名
                proj_col = f"proj_axon_{target_acronym}_abs"  # 使用绝对值列
                if proj_col not in projection_data.columns:
                    continue

                # 获取投射强度
                try:
                    proj_value = neuron_data[proj_col]

                    # 只处理有效的投射值（大于0）
                    if pd.notna(proj_value) and proj_value > 0:
                        conn_key = (source_id, target_id)
                        if conn_key not in region_connections:
                            region_connections[conn_key] = {
                                'total': 0,
                                'count': 0,
                                'source_acronym': source_acronym,
                                'target_acronym': target_acronym
                            }

                        region_connections[conn_key]['total'] += float(proj_value)
                        region_connections[conn_key]['count'] += 1
                except Exception as e:
                    logger.debug(f"处理投射数据时出错: {e}")
                    continue

        # 生成PROJECT_TO关系
        relationships = []

        for (source_id, target_id), stats in region_connections.items():
            # 计算平均投射强度
            avg_strength = stats['total'] / stats['count'] if stats['count'] > 0 else 0

            # 创建关系
            rel = {
                ':START_ID(Region)': source_id,
                ':END_ID(Region)': target_id,
                'weight:float': float(avg_strength),
                'total:float': float(stats['total']),
                'neuron_count:int': int(stats['count']),
                'source_acronym': stats['source_acronym'],
                'target_acronym': stats['target_acronym'],
                ':TYPE': 'PROJECT_TO'
            }

            relationships.append(rel)

        # 保存关系
        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "project_to.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个PROJECT_TO关系到 {output_file}")

            # 打印前几条记录
            logger.info("投射关系示例:")
            for i, row in df.head(5).iterrows():
                logger.info(
                    f"  {row['source_acronym']} -> {row['target_acronym']}: {row['weight:float']:.4f} (神经元数: {row['neuron_count:int']})")
        else:
            logger.warning("没有生成任何PROJECT_TO关系")

    def generate_has_relationships_unified(self, merfish_cells: pd.DataFrame, level: str):
        """
        为统一的Region节点生成HAS关系
        """
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
        counts_df = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')

        # 添加比例列
        region_totals = valid_cells.groupby('region_id')['region_id'].count().reset_index(name='total')
        counts_df = pd.merge(counts_df, region_totals, on='region_id')
        counts_df['pct'] = counts_df['count'] / counts_df['total']

        # 过滤低于阈值的行
        counts_df = counts_df[counts_df['pct'] >= PCT_THRESHOLD]

        # 为每个区域计算rank
        relationships = []

        for region_id in counts_df['region_id'].unique():
            # 获取该区域的所有细胞类型
            region_df = counts_df[counts_df['region_id'] == region_id].copy()

            # 按比例降序排序
            region_df = region_df.sort_values('pct', ascending=False)

            # 明确分配rank (1-based)
            region_df['rank'] = range(1, len(region_df) + 1)

            # 创建关系
            for _, row in region_df.iterrows():
                cell_type = row[level]

                if cell_type in id_map:
                    rel = {
                        ':START_ID(Region)': int(region_id),
                        f':END_ID({level.capitalize()})': id_map[cell_type],
                        'pct_cells:float': float(row['pct']),
                        'rank:int': int(row['rank']),
                        ':TYPE': f'HAS_{level.upper()}'
                    }
                    relationships.append(rel)

        # 保存关系
        self._save_relationships_batch(relationships, f"has_{level}")
        logger.info(f"保存了 {len(relationships)} 个HAS_{level.upper()}关系")

        # 打印前几条关系示例
        if relationships:
            logger.info(f"HAS_{level.upper()} 关系示例:")
            for i, rel in enumerate(relationships[:5]):
                logger.info(f"  区域 {rel[':START_ID(Region)']} -> {level} {rel[f':END_ID({level.capitalize()})']}:"
                           f" 比例={rel['pct_cells:float']:.4f}, 排名={rel['rank:int']}")

    def generate_merfish_nodes_from_hierarchy(self, merfish_cells: pd.DataFrame):
        """从层级数据生成MERFISH节点，包含所有必要的元数据"""
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
        """为特定层级生成Dominant_transcriptomic关系，每个区域只保留一个主导类型"""
        logger.info(f"生成{level.capitalize()}级别的Dominant_transcriptomic关系...")

        # 获取ID映射
        id_map = getattr(self, f"{level}_id_map", {})
        if not id_map:
            logger.warning(f"{level} ID映射为空")
            return

        # 筛选有效细胞
        valid_cells = merfish_cells[(merfish_cells[level].notna()) & (merfish_cells['region_id'] > 0)]
        if len(valid_cells) == 0:
            logger.warning(f"没有同时有{level}和区域ID的细胞")
            return

        # 计算每个区域中各细胞类型的数量
        relationships = []

        # 为每个区域找出主导类型
        for region_id in valid_cells['region_id'].unique():
            # 该区域的所有细胞
            region_cells = valid_cells[valid_cells['region_id'] == region_id]
            total = len(region_cells)

            if total == 0:
                continue

            # 计算各类型的比例
            type_counts = region_cells[level].value_counts()
            type_data = []

            for cell_type, count in type_counts.items():
                if cell_type in id_map:
                    pct = count / total
                    type_data.append({
                        'type': cell_type,
                        'count': count,
                        'pct': pct,
                        'type_id': id_map[cell_type]
                    })

            # 按比例排序
            type_data.sort(key=lambda x: x['pct'], reverse=True)

            # 只取第一个（最主要的）类型
            if type_data:
                dominant_type = type_data[0]
                rel = {
                    ':START_ID(Region)': int(region_id),
                    f':END_ID({level.capitalize()})': dominant_type['type_id'],
                    'pct:float': float(dominant_type['pct']),
                    'rank:int': 1,  # 固定为1，因为每个区域只有一个主导类型
                                        ':TYPE': f'DOMINANT_TRANSCRIPTOMIC_{level.upper()}'
                }
                relationships.append(rel)

        # 保存关系
        if relationships:
            output_file = self.relationships_dir / f"dominant_transcriptomic_{level}.csv"
            df = pd.DataFrame(relationships)
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(relationships)} 个{level}级别的Dominant_transcriptomic关系")
        else:
            logger.warning(f"没有生成{level}级别的Dominant_transcriptomic关系")

    def generate_import_script(self):
        """生成Neo4j导入脚本 - 统一区域节点版本"""
        script = """#!/bin/bash
# Neo4j批量导入脚本
NEO4J_HOME=/path/to/neo4j
DATABASE_NAME=neuroxiv

$NEO4J_HOME/bin/neo4j stop
rm -rf $NEO4J_HOME/data/databases/$DATABASE_NAME

$NEO4J_HOME/bin/neo4j-admin import \\
   --database=$DATABASE_NAME \\
   --nodes=Region=nodes/regions.csv \\
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

    def generate_statistics_report(self, region_data, merfish_cells):
        """生成统计报告 - 统一区域节点版本"""
        report = []
        report.append("=" * 60)
        report.append("NeuroXiv 2.0 知识图谱统计报告 (统一区域节点版本)")
        report.append("=" * 60)
        report.append(f"节点统计:")
        report.append(f"  - Region节点: {len(region_data)}")

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
    """主函数 - 使用统一区域节点的新版本"""

    setup_logger()

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - 统一区域节点版本")
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

    # 创建builder实例
    builder = KnowledgeGraphBuilder(output_path)

    # 加载树结构用于区域分析
    tree_data = processed_data.get('tree', [])
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

    # Phase 3: 加载形态数据
    logger.info("Phase 3: 加载形态数据")

    morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
    if morphology_loader.load_morphology_data():
        builder.morphology_loader = morphology_loader
    else:
        logger.warning("无法加载形态学数据")

    # Phase 4: 知识图谱生成
    logger.info("Phase 4: 知识图谱生成")

    builder.set_hierarchy_loader(hierarchy_loader)

    # 生成节点
    logger.info("生成节点...")

    # 使用新方法生成统一的Region节点
    builder.generate_unified_region_nodes(region_data, merfish_cells)

    # 生成MERFISH细胞类型节点
    builder.generate_merfish_nodes_from_hierarchy(merfish_cells)

    # 生成关系
    logger.info("生成关系...")

    # 使用新方法生成HAS关系
    for level in ['class', 'subclass', 'supertype', 'cluster']:
        builder.generate_has_relationships_unified(merfish_cells, level)

    # 生成层级关系
    builder.generate_belongs_to_from_hierarchy()

    # 生成投射关系
    builder.generate_project_to_relationships(projection_data)

    # 生成主导转录组关系
    builder.generate_dominant_transcriptomic_relationships(merfish_cells)

    # 后处理
    builder.generate_import_script()
    builder.generate_statistics_report(builder.region_data, merfish_cells)

    logger.info("=" * 60)
    logger.info("知识图谱构建完成！")
    logger.info(f"输出目录: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./knowledge_graph_v4',
                        help='输出目录路径')
    parser.add_argument('--hierarchy_json', type=str, default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json',
                        help='MERFISH层级JSON文件路径')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.hierarchy_json)