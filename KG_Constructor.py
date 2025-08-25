"""
NeuroXiv 2.0 知识图谱构建器 - 完整增强版
整合MERFISH层级JSON和层特异性形态计算

作者: PrometheusTT
日期: 2025-08-25
"""

import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

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
N_WORKERS = 32
BATCH_SIZE = 10000
PCT_THRESHOLD = 0.01

# 层定义
LAYERS = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']

# 完整的形态特征列表
FULL_MORPH_FEATURES = [
    'axonal_bifurcation_remote_angle',
    'axonal_branches',
    'axonal_length',
    'axonal_maximum_branch_order',
    'dendritic_bifurcation_remote_angle',
    'dendritic_branches',
    'dendritic_length',
    'dendritic_maximum_branch_order',
    'local_traced_dendritic_bifurcation_remote_angle',
    'local_traced_dendritic_branches',
    'local_traced_dendritic_length',
    'local_traced_dendritic_maximum_branch_order'
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
        'L6': ['L6', 'L6a', 'L6b', '6'],
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
                        self.hierarchy_relations['cluster_to_supertype'][cluster_node['label']] = supertype_node[
                            'label']

        logger.info(f"解析完成: {len(self.class_data)} Class, {len(self.subclass_data)} Subclass, "
                    f"{len(self.supertype_data)} Supertype, {len(self.cluster_data)} Cluster")


# ==================== 层特异性形态计算 ====================

class LayerSpecificMorphologyCalculator:
    """层特异性形态学计算器"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.info_df = None
        self.axon_df = None
        self.dendrite_df = None

    def load_morphology_with_layers(self) -> bool:
        """加载形态数据和层信息"""
        # 加载info.csv
        info_file = self.data_dir / "info.csv"
        if not info_file.exists():
            logger.error(f"info.csv不存在: {info_file}")
            return False

        self.info_df = pd.read_csv(info_file)
        logger.info(f"加载了 {len(self.info_df)} 条神经元信息")

        # 提取层信息
        if 'celltype' in self.info_df.columns:
            self.info_df['layer'] = self.info_df['celltype'].apply(extract_layer_from_celltype)
            layer_counts = self.info_df['layer'].value_counts()
            logger.info(f"层分布: {layer_counts.to_dict()}")
        else:
            logger.warning("info.csv缺少celltype列")
            self.info_df['layer'] = 'Unknown'

        # 加载轴突形态数据
        axon_file = self.data_dir / "axonfull_morpho.csv"
        if axon_file.exists():
            self.axon_df = pd.read_csv(axon_file)
            # 过滤掉CCF-thin和local
            self.axon_df = self.axon_df[~self.axon_df['name'].str.contains('CCF-thin|local', na=False)]
            logger.info(f"加载了 {len(self.axon_df)} 条轴突形态数据")

        # 加载树突形态数据
        dendrite_file = self.data_dir / "denfull_morpho.csv"
        if dendrite_file.exists():
            self.dendrite_df = pd.read_csv(dendrite_file)
            # 过滤掉CCF-thin和local
            self.dendrite_df = self.dendrite_df[~self.dendrite_df['name'].str.contains('CCF-thin|local', na=False)]
            logger.info(f"加载了 {len(self.dendrite_df)} 条树突形态数据")

        return True

    def calculate_regionlayer_morphology(self, region_data: pd.DataFrame) -> pd.DataFrame:
        """计算RegionLayer的层特异性形态特征"""
        logger.info("计算层特异性形态特征...")

        regionlayer_data = []

        # 从info.csv提取区域信息
        if 'celltype' in self.info_df.columns:
            # 提取基础区域名（去除层信息）
            self.info_df['base_region'] = self.info_df['celltype'].apply(self._extract_base_region)

        for _, region in region_data.iterrows():
            region_id = region.get('region_id', region.name)
            region_name = region.get('region_name', region.get('name', f'Region_{region_id}'))

            # 获取该区域的神经元
            if 'base_region' in self.info_df.columns:
                region_neurons = self.info_df[self.info_df['base_region'] == region_name]
            else:
                region_neurons = pd.DataFrame()

            for layer in LAYERS:
                rl_id = f"{region_id}_{layer}"

                # 获取该层的神经元
                if not region_neurons.empty:
                    layer_neurons = region_neurons[region_neurons['layer'] == layer]
                    neuron_ids = set(layer_neurons['ID'].values) if 'ID' in layer_neurons.columns else set()
                else:
                    layer_neurons = pd.DataFrame()
                    neuron_ids = set()

                # 初始化记录
                rl_dict = {
                    'rl_id': rl_id,
                    'region_name': region_name,
                    'layer': layer,
                    'morph_neuron_count': len(neuron_ids)
                }

                # 计算该层的形态特征
                if len(neuron_ids) > 0:
                    # 计算轴突特征
                    if self.axon_df is not None and 'ID' in self.axon_df.columns:
                        layer_axon = self.axon_df[self.axon_df['ID'].isin(neuron_ids)]
                        if len(layer_axon) > 0:
                            rl_dict['axonal_length'] = layer_axon[
                                'Total Length'].mean() if 'Total Length' in layer_axon.columns else 0
                            rl_dict['axonal_branches'] = layer_axon[
                                'Number of Bifurcations'].mean() if 'Number of Bifurcations' in layer_axon.columns else 0
                            rl_dict['axonal_bifurcation_remote_angle'] = layer_axon[
                                'Average Bifurcation Angle Remote'].mean() if 'Average Bifurcation Angle Remote' in layer_axon.columns else 0
                            rl_dict['axonal_maximum_branch_order'] = layer_axon[
                                'Max Branch Order'].mean() if 'Max Branch Order' in layer_axon.columns else 0

                    # 计算树突特征
                    if self.dendrite_df is not None and 'ID' in self.dendrite_df.columns:
                        layer_dendrite = self.dendrite_df[self.dendrite_df['ID'].isin(neuron_ids)]
                        if len(layer_dendrite) > 0:
                            rl_dict['dendritic_length'] = layer_dendrite[
                                'Total Length'].mean() if 'Total Length' in layer_dendrite.columns else 0
                            rl_dict['dendritic_branches'] = layer_dendrite[
                                'Number of Bifurcations'].mean() if 'Number of Bifurcations' in layer_dendrite.columns else 0
                            rl_dict['dendritic_bifurcation_remote_angle'] = layer_dendrite[
                                'Average Bifurcation Angle Remote'].mean() if 'Average Bifurcation Angle Remote' in layer_dendrite.columns else 0
                            rl_dict['dendritic_maximum_branch_order'] = layer_dendrite[
                                'Max Branch Order'].mean() if 'Max Branch Order' in layer_dendrite.columns else 0
                else:
                    # 如果该层没有神经元，使用区域平均值
                    for feat in FULL_MORPH_FEATURES:
                        rl_dict[feat] = float(region.get(feat, 0.0))

                # 确保所有特征都存在
                for feat in FULL_MORPH_FEATURES:
                    if feat not in rl_dict:
                        rl_dict[feat] = 0.0

                regionlayer_data.append(rl_dict)

        df = pd.DataFrame(regionlayer_data)
        logger.info(f"生成了 {len(df)} 个RegionLayer节点的层特异性形态数据")
        return df

    def _extract_base_region(self, celltype: str) -> str:
        """提取基础区域名（去除层信息）"""
        if pd.isna(celltype):
            return 'Unknown'

        celltype = str(celltype)

        # 移除层后缀
        layer_suffixes = ['1', '2/3', '2', '3', '4', '5', '5a', '5b', '6', '6a', '6b']
        for suffix in layer_suffixes:
            if celltype.endswith(suffix):
                return celltype[:-len(suffix)]

        return celltype


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

    def generate_merfish_nodes_from_hierarchy(self, merfish_cells: pd.DataFrame):
        """从层级数据生成MERFISH节点"""
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
                'number_of_neurons': class_counts.get(name, 0)
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

        logger.info("生成BELONGS_TO关系（从层级数据）...")
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

    # 保留原有的其他方法...
    def generate_region_nodes(self, region_data: pd.DataFrame):
        """生成Region节点CSV"""
        logger.info("生成Region节点...")

        nodes = []
        for idx, region in region_data.iterrows():
            region_id = region.get('region_id', idx)

            node = {
                'region_id:ID(Region)': int(region_id),
                'name': str(region.get('region_name', region.get('name', f'Region_{region_id}'))),
                'atlas_versions': json.dumps({'me_subregions': []}),
                'number_of_apical_dendritic_morphologies': float(
                    region.get('number_of_apical_dendritic_morphologies', 0)),
                'number_of_axonal_morphologies': float(region.get('number_of_axonal_morphologies', 0)),
                'number_of_dendritic_morphologies': float(region.get('number_of_dendritic_morphologies', 0)),
                'number_of_local_traced_dendritic_morphologies': float(
                    region.get('number_of_local_traced_dendritic_morphologies', 0)),
                'number_of_neuron_morphologies': float(region.get('number_of_neuron_morphologies', 0)),
                'number_of_transcriptomic_neurons': float(region.get('number_of_transcriptomic_neurons', 0))
            }

            # 添加所有形态特征
            for feat in FULL_MORPH_FEATURES:
                node[feat] = float(region.get(feat, 0.0))

            nodes.append(node)

        df = pd.DataFrame(nodes)
        output_file = self.nodes_dir / "regions.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"保存了 {len(df)} 个Region节点到 {output_file}")

    def generate_regionlayer_nodes(self, regionlayer_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """生成RegionLayer节点CSV（包含层特异性数据）"""
        logger.info("生成RegionLayer节点...")

        # 如果有MERFISH数据，计算细胞数量
        if merfish_cells is not None and not merfish_cells.empty and 'region_id' in merfish_cells.columns:
            region_cell_counts = merfish_cells.groupby('region_id').size().to_dict()
        else:
            region_cell_counts = {}

        # 更新merfish_cell_count
        for idx, row in regionlayer_data.iterrows():
            region_id = row['rl_id'].split('_')[0]
            try:
                region_id_int = int(region_id)
                regionlayer_data.at[idx, 'merfish_cell_count'] = region_cell_counts.get(region_id_int, 0)
            except:
                regionlayer_data.at[idx, 'merfish_cell_count'] = 0

        # 重命名列以匹配Neo4j格式
            # 重命名列以匹配Neo4j格式
            regionlayer_data = regionlayer_data.rename(columns={'rl_id': 'rl_id:ID(RegionLayer)'})

            # 确保所有必要的列都存在
            for feat in FULL_MORPH_FEATURES:
                if feat not in regionlayer_data.columns:
                    regionlayer_data[feat] = 0.0

            # 添加额外字段（如果缺失）
            if 'subclass_pct' not in regionlayer_data.columns:
                regionlayer_data['subclass_pct'] = '{}'
            if 'module_score' not in regionlayer_data.columns:
                regionlayer_data['module_score'] = '{}'

            output_file = self.nodes_dir / "regionlayers.csv"
            regionlayer_data.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(regionlayer_data)} 个RegionLayer节点到 {output_file}")

    def generate_has_layer_relationships(self, region_data: pd.DataFrame):
        """生成HAS_LAYER关系"""
        logger.info("生成HAS_LAYER关系...")

        relationships = []
        for _, region in region_data.iterrows():
            region_id = region.get('region_id', region.name)

            for layer in LAYERS:
                rl_id = f"{region_id}_{layer}"

                rel = {
                    ':START_ID(Region)': int(region_id),
                    ':END_ID(RegionLayer)': rl_id,
                    ':TYPE': 'HAS_LAYER'
                }
                relationships.append(rel)

        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "has_layer.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个HAS_LAYER关系到 {output_file}")

    def generate_has_relationships_optimized(self,
                                             merfish_cells: pd.DataFrame,
                                             level: str):
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
        else:  # cluster
            id_map = self.cluster_id_map

        if not id_map:
            logger.warning(f"{level} ID映射为空")
            return

        # 按区域和类型分组计数
        valid_cells = merfish_cells[merfish_cells[level].notna()]
        grouped = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')

        # 计算每个区域的总数
        region_totals = valid_cells.groupby('region_id').size().to_dict()

        # 生成关系
        relationships = []

        for region_id in grouped['region_id'].unique():
            region_group = grouped[grouped['region_id'] == region_id]
            total = region_totals.get(region_id, 1)

            # 计算比例并排序
            region_group = region_group.copy()
            region_group['pct'] = region_group['count'] / total
            region_group = region_group[region_group['pct'] >= PCT_THRESHOLD]
            region_group['rank'] = region_group['pct'].rank(ascending=False, method='dense').astype(int)

            # 为每个层生成关系
            for layer in LAYERS:
                rl_id = f"{region_id}_{layer}"

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

                # 批量保存
                if len(relationships) >= BATCH_SIZE:
                    self._save_relationships_batch(relationships, f"has_{level}")
                    relationships = []

        # 保存剩余的关系
        if relationships:
            self._save_relationships_batch(relationships, f"has_{level}")

        logger.info(f"完成HAS_{level.upper()}关系生成")

    def generate_dominant_transcriptomic_relationships(self,
                                                       region_data: pd.DataFrame,
                                                       merfish_cells: pd.DataFrame):
        """生成Dominant_transcriptomic关系"""
        logger.info("生成Dominant_transcriptomic关系...")

        if 'region_id' not in merfish_cells.columns:
            logger.warning("细胞缺少region_id")
            return

        relationships = []

        for level in ['class', 'subclass', 'supertype', 'cluster']:
            if level not in merfish_cells.columns:
                continue

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
                continue

            # 找出每个区域的主导类型
            valid_cells = merfish_cells[merfish_cells[level].notna()]

            for region_id in region_data['region_id'].unique():
                region_cells = valid_cells[valid_cells['region_id'] == region_id]

                if len(region_cells) > 0:
                    # 找出最常见的类型
                    dominant_type = region_cells[level].mode()
                    if len(dominant_type) > 0:
                        dominant_type = dominant_type[0]

                        if dominant_type in id_map:
                            rel = {
                                ':START_ID(Region)': int(region_id),
                                f':END_ID({level.capitalize()})': id_map[dominant_type],
                                ':TYPE': f'DOMINANT_TRANSCRIPTOMIC_{level.upper()}'
                            }
                            relationships.append(rel)

        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "dominant_transcriptomic.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个Dominant_transcriptomic关系到 {output_file}")

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
        length_col = None

        # 尝试不同的列名
        for col in ['source_region', 'source_region_id', 'source', 'from_region']:
            if col in projection_data.columns:
                source_col = col
                break

        for col in ['target_region', 'target_region_id', 'target', 'to_region']:
            if col in projection_data.columns:
                target_col = col
                break

        for col in ['length', 'projection_length', 'axon_length', 'distance']:
            if col in projection_data.columns:
                length_col = col
                break

        if not source_col or not target_col:
            logger.error(f"投影数据缺少必要的列。找到的列: {projection_data.columns.tolist()}")
            return

        for _, proj in projection_data.iterrows():
            try:
                source = int(proj[source_col]) if pd.notna(proj[source_col]) else 0
                target = int(proj[target_col]) if pd.notna(proj[target_col]) else 0

                if source > 0 and target > 0:
                    rel = {
                        ':START_ID(Region)': source,
                        ':END_ID(Region)': target,
                        'length:float': float(proj[length_col]) if length_col and pd.notna(
                            proj.get(length_col)) else 0.0,
                        ':TYPE': 'PROJECT_TO'
                    }
                    relationships.append(rel)
            except (ValueError, TypeError) as e:
                continue

        if relationships:
            df = pd.DataFrame(relationships)
            output_file = self.relationships_dir / "project_to.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存了 {len(df)} 个PROJECT_TO关系到 {output_file}")

    def _save_relationships_batch(self, relationships: List[Dict], prefix: str):
        """批量保存关系"""
        if not relationships:
            return

        df = pd.DataFrame(relationships)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.relationships_dir / f"{prefix}_{timestamp}_{len(df)}.csv"
        df.to_csv(output_file, index=False)
        logger.debug(f"保存了 {len(df)} 个关系到 {output_file}")

    # ==================== 数据加载器 ====================

    class DataLoader:
        """完整数据加载器"""

        def __init__(self, data_dir: Path):
            self.data_dir = Path(data_dir)
            self.region_data = None
            self.merfish_cells = None
            self.projection_data = None
            self.tree_data = None

        def load_region_morphology(self) -> pd.DataFrame:
            """加载区域形态数据"""
            logger.info("加载区域形态数据...")

            # 使用已有的加载函数
            data = load_data(self.data_dir)
            processed_data = prepare_analysis_data(data)

            if 'region_data' not in processed_data:
                raise ValueError("无法加载区域形态数据")

            self.region_data = processed_data['region_data']

            # 确保包含所有形态特征
            for feat in FULL_MORPH_FEATURES:
                if feat not in self.region_data.columns:
                    self.region_data[feat] = 0.0

            logger.info(f"加载了 {len(self.region_data)} 个区域的形态数据")
            return self.region_data

        def load_merfish_cells_parallel(self) -> pd.DataFrame:
            """并行加载MERFISH细胞数据"""
            logger.info("并行加载MERFISH细胞数据...")

            # 获取所有MERFISH文件
            coord_files = sorted(self.data_dir.glob("ccf_coordinates_*.csv"))
            meta_files = sorted(self.data_dir.glob("cell_metadata_with_cluster_annotation_*.csv"))

            if not coord_files:
                logger.warning("未找到坐标文件")
                return pd.DataFrame()

            # 并行读取
            all_cells = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                coord_futures = [executor.submit(self._load_merfish_file, f, i) for i, f in enumerate(coord_files)]

                for future in coord_futures:
                    df = future.result()
                    if df is not None:
                        all_cells.append(df)

            if all_cells:
                self.merfish_cells = pd.concat(all_cells, ignore_index=True)
                logger.info(f"加载了 {len(self.merfish_cells)} 个细胞")

                # 验证并修复坐标列
                required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
                missing_cols = [col for col in required_cols if col not in self.merfish_cells.columns]

                if missing_cols:
                    logger.warning(f"合并后的细胞数据缺少必要的坐标列: {missing_cols}")
                    logger.warning(f"可用列: {self.merfish_cells.columns.tolist()}")

                    # 尝试从其他列推断坐标列
                    coord_candidates = {
                        'x_ccf': ['x', 'X', 'x_coord', 'x_25um', 'x_position', 'ccf_x'],
                        'y_ccf': ['y', 'Y', 'y_coord', 'y_25um', 'y_position', 'ccf_y'],
                        'z_ccf': ['z', 'Z', 'z_coord', 'z_25um', 'z_position', 'ccf_z']
                    }

                    for missing_col in missing_cols:
                        for candidate in coord_candidates.get(missing_col, []):
                            if candidate in self.merfish_cells.columns:
                                self.merfish_cells[missing_col] = self.merfish_cells[candidate]
                                logger.info(f"使用列 '{candidate}' 作为 '{missing_col}'")
                                break

                    # 再次检查是否仍有缺失列
                    still_missing = [col for col in required_cols if col not in self.merfish_cells.columns]
                    if still_missing:
                        logger.error(f"无法修复所有缺失的坐标列: {still_missing}")
            else:
                self.merfish_cells = pd.DataFrame()

            return self.merfish_cells

        def _load_merfish_file(self, coord_file: Path, index: int) -> pd.DataFrame:
            """加载单个MERFISH文件"""
            try:
                # 加载坐标
                coords = pd.read_csv(coord_file)

                # 确保坐标列名正确 - 处理不同的列名情况
                coord_mapping = {
                    'x': 'x_ccf',
                    'y': 'y_ccf',
                    'z': 'z_ccf',
                    'X': 'x_ccf',
                    'Y': 'y_ccf',
                    'Z': 'z_ccf',
                    'x_coord': 'x_ccf',
                    'y_coord': 'y_ccf',
                    'z_coord': 'z_ccf',
                    'ccf_x': 'x_ccf',
                    'ccf_y': 'y_ccf',
                    'ccf_z': 'z_ccf'
                }

                # 重命名坐标列
                for old_col, new_col in coord_mapping.items():
                    if old_col in coords.columns and new_col not in coords.columns:
                        coords = coords.rename(columns={old_col: new_col})
                        logger.debug(f"将列 '{old_col}' 重命名为 '{new_col}'")

                # 尝试加载对应的元数据
                meta_file = coord_file.parent / f"cell_metadata_with_cluster_annotation_{index + 1}.csv"
                if meta_file.exists():
                    meta = pd.read_csv(meta_file)

                    # 合并坐标和元数据
                    if 'cell_label' in coords.columns and 'cell_label' in meta.columns:
                        coords = pd.merge(coords, meta, on='cell_label', how='left')
                    elif len(coords) == len(meta):
                        # 如果长度相同，假设顺序对应
                        for col in meta.columns:
                            if col not in coords.columns:
                                coords[col] = meta[col].values

                # 最后再次检查必要的坐标列是否存在
                missing_cols = [col for col in ['x_ccf', 'y_ccf', 'z_ccf'] if col not in coords.columns]
                if missing_cols:
                    logger.warning(f"文件 {coord_file.name} 处理后仍缺少坐标列: {missing_cols}")
                    logger.warning(f"可用列: {coords.columns.tolist()}")

                return coords

            except Exception as e:
                logger.error(f"加载文件 {coord_file} 失败: {e}")
                return None

        def load_projection_data(self) -> pd.DataFrame:
            """加载投影数据"""
            proj_file = self.data_dir / "Proj_Axon_Final.csv"
            if proj_file.exists():
                self.projection_data = pd.read_csv(proj_file)
                logger.info(f"加载了 {len(self.projection_data)} 条投影数据")
            else:
                logger.warning("未找到投影数据文件")
                self.projection_data = pd.DataFrame()
            return self.projection_data

        def load_tree_structure(self) -> Dict:
            """加载CCF树结构"""
            tree_file = self.data_dir / "tree_yzx.json"
            if tree_file.exists():
                with open(tree_file, 'r') as f:
                    self.tree_data = json.load(f)
                logger.info("加载了CCF树结构")
            else:
                self.tree_data = {}
            return self.tree_data

    # ==================== 主函数 ====================

def main(data_dir: str = "../data",
         output_dir: str = "./knowledge_graph",
         hierarchy_json: str = None):
    """主函数"""

    setup_logger()

    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - 完整增强版")
    logger.info("=" * 60)

    # 初始化
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Phase 1: 数据加载
    logger.info("Phase 1: 数据加载")
    loader = EnhancedKnowledgeGraphBuilder.DataLoader(data_path)

    # 加载区域形态数据
    region_data = loader.load_region_morphology()

    # 并行加载MERFISH细胞数据
    merfish_cells = loader.load_merfish_cells_parallel()

    # 映射细胞到区域
    if not merfish_cells.empty and 'region_id' not in merfish_cells.columns:
        logger.info("映射细胞到区域...")
        data = load_data(data_path)
        if 'annotation' in data:
            # 确保所有必要的坐标列都存在
            required_cols = ['x_ccf', 'y_ccf', 'z_ccf']
            missing_cols = [col for col in required_cols if col not in merfish_cells.columns]

            if missing_cols:
                logger.warning(f"映射前检测到缺少坐标列: {missing_cols}")

                # 尝试找到并复制坐标列
                coord_mapping = {
                    'x_ccf': ['x', 'X', 'x_coord', 'ccf_x', 'x_position'],
                    'y_ccf': ['y', 'Y', 'y_coord', 'ccf_y', 'y_position'],
                    'z_ccf': ['z', 'Z', 'z_coord', 'ccf_z', 'z_position']
                }

                for target_col in missing_cols:
                    for source_col in coord_mapping.get(target_col, []):
                        if source_col in merfish_cells.columns:
                            merfish_cells[target_col] = merfish_cells[source_col]
                            logger.info(f"使用 '{source_col}' 作为 '{target_col}'")
                            break

            # 最终检查
            missing_cols = [col for col in required_cols if col not in merfish_cells.columns]
            if not missing_cols:
                try:
                    merfish_cells = map_cells_to_regions_fixed(
                        merfish_cells,
                        data['annotation']['volume'],
                        data['annotation']['header']
                    )
                except KeyError as e:
                    logger.error(f"映射过程中出现键错误: {e}")
                    logger.error(f"可用列: {merfish_cells.columns.tolist()}")
                    # 添加空的region_id列
                    merfish_cells['region_id'] = 0
            else:
                logger.error(f"无法修复所有坐标列，跳过映射: {missing_cols}")
                merfish_cells['region_id'] = 0

    # 加载其他数据
    projection_data = loader.load_projection_data()
    tree_data = loader.load_tree_structure()

    # Phase 2: 加载MERFISH层级数据
    logger.info("Phase 2: 加载MERFISH层级数据")

    # 如果没有指定JSON文件，尝试默认路径
    if not hierarchy_json:
        possible_files = [
            data_path / "merfish_hierarchy.json",
            data_path / "allen_merfish_hierarchy.json",
            data_path / "cell_type_hierarchy.json"
        ]
        for f in possible_files:
            if f.exists():
                hierarchy_json = str(f)
                break

    hierarchy_loader = None
    if hierarchy_json:
        hierarchy_loader = MERFISHHierarchyLoader(hierarchy_json)
        if hierarchy_loader.load_hierarchy():
            logger.info("成功加载MERFISH层级数据")
        else:
            logger.warning("无法加载MERFISH层级数据，将使用细胞数据推断")
            hierarchy_loader = None
    else:
        logger.warning("未找到MERFISH层级JSON文件")

    # Phase 3: 计算层特异性形态数据
    logger.info("Phase 3: 计算层特异性形态数据")

    layer_calculator = LayerSpecificMorphologyCalculator(data_path)
    if layer_calculator.load_morphology_with_layers():
        regionlayer_data = layer_calculator.calculate_regionlayer_morphology(region_data)
    else:
        logger.warning("无法计算层特异性形态数据，使用简化版本")
        # 使用简化的RegionLayer数据
        regionlayer_data = []
        for _, region in region_data.iterrows():
            region_id = region.get('region_id', region.name)
            region_name = region.get('region_name', region.get('name', f'Region_{region_id}'))

            for layer in LAYERS:
                rl_id = f"{region_id}_{layer}"
                rl_dict = {
                    'rl_id': rl_id,
                    'region_name': region_name,
                    'layer': layer,
                    'merfish_cell_count': 0,
                    'morph_neuron_count': 0
                }

                # 使用区域平均值
                for feat in FULL_MORPH_FEATURES:
                    rl_dict[feat] = float(region.get(feat, 0.0))

                regionlayer_data.append(rl_dict)

        regionlayer_data = pd.DataFrame(regionlayer_data)

    # Phase 4: 图谱生成
    logger.info("Phase 4: 知识图谱生成")
    builder = EnhancedKnowledgeGraphBuilder(output_path)

    # 如果有层级数据，设置层级加载器
    if hierarchy_loader:
        builder.set_hierarchy_loader(hierarchy_loader)

    # 生成所有节点
    logger.info("生成节点...")
    builder.generate_region_nodes(region_data)
    builder.generate_regionlayer_nodes(regionlayer_data, merfish_cells)

    if hierarchy_loader:
        # 使用层级数据生成MERFISH节点
        builder.generate_merfish_nodes_from_hierarchy(merfish_cells)
    else:
        # 使用原始方法（从细胞数据推断）
        logger.warning("使用细胞数据生成MERFISH节点（可能缺少详细属性）")
        # 这里可以调用原始的生成方法

    # 生成所有关系
    logger.info("生成关系...")

    # HAS_LAYER关系
    builder.generate_has_layer_relationships(region_data)

    # HAS_* 关系
    for level in ['class', 'subclass', 'supertype', 'cluster']:
        builder.generate_has_relationships_optimized(merfish_cells, level)

    # BELONGS_TO层级关系
    if hierarchy_loader:
        builder.generate_belongs_to_from_hierarchy()

    # Dominant_transcriptomic关系
    builder.generate_dominant_transcriptomic_relationships(region_data, merfish_cells)

    # PROJECT_TO关系
    builder.generate_project_to_relationships(projection_data)

    # 生成Neo4j导入脚本
    generate_import_script(output_path)

    # 生成统计报告
    generate_statistics_report(output_path, region_data, merfish_cells, regionlayer_data, hierarchy_loader)

    logger.info("=" * 60)
    logger.info("知识图谱构建完成！")
    logger.info(f"输出目录: {output_path}")
    logger.info("=" * 60)

def generate_import_script(output_dir: Path):
    """生成Neo4j导入脚本"""
    script = """#!/bin/bash
    # Neo4j批量导入脚本
    # 使用方法: ./import_to_neo4j.sh

    # 设置Neo4j路径
    NEO4J_HOME=/path/to/neo4j
    DATABASE_NAME=neuroxiv

    # 停止Neo4j（如果正在运行）
    $NEO4J_HOME/bin/neo4j stop

    # 删除旧数据库（如果存在）
    rm -rf $NEO4J_HOME/data/databases/$DATABASE_NAME

    # 执行导入
    $NEO4J_HOME/bin/neo4j-admin import \\
       --database=$DATABASE_NAME \\
       --nodes=Region=nodes/regions.csv \\
       --nodes=RegionLayer=nodes/regionlayers.csv \\
       --nodes=Class=nodes/class.csv \\
       --nodes=Subclass=nodes/subclass.csv \\
       --nodes=Supertype=nodes/supertype.csv \\
       --nodes=Cluster=nodes/cluster.csv \\
       --relationships=HAS_LAYER=relationships/has_layer.csv \\
       --relationships=HAS_CLASS=relationships/has_class_*.csv \\
       --relationships=HAS_SUBCLASS=relationships/has_subclass_*.csv \\
       --relationships=HAS_SUPERTYPE=relationships/has_supertype_*.csv \\
       --relationships=HAS_CLUSTER=relationships/has_cluster_*.csv \\
       --relationships=BELONGS_TO=relationships/belongs_to.csv \\
       --relationships=DOMINANT_TRANSCRIPTOMIC=relationships/dominant_transcriptomic.csv \\
       --relationships=PROJECT_TO=relationships/project_to.csv \\
       --skip-bad-relationships=true \\
       --skip-duplicate-nodes=true \\
       --high-io=true

    # 启动Neo4j
    $NEO4J_HOME/bin/neo4j start

    echo "导入完成！"
    """

    script_file = output_dir / "import_to_neo4j.sh"
    with open(script_file, 'w') as f:
        f.write(script)

    # 使脚本可执行
    import os
    os.chmod(script_file, 0o755)

    logger.info(f"生成了Neo4j导入脚本: {script_file}")

def generate_statistics_report(output_dir: Path,
                               region_data: pd.DataFrame,
                               merfish_cells: pd.DataFrame,
                               regionlayer_data: pd.DataFrame,
                               hierarchy_loader: MERFISHHierarchyLoader = None):
    """生成统计报告"""
    report = []
    report.append("=" * 60)
    report.append("NeuroXiv 2.0 知识图谱统计报告")
    report.append("=" * 60)
    report.append("")

    # 节点统计
    report.append("节点统计:")
    report.append(f"  - Region节点: {len(region_data)}")
    report.append(f"  - RegionLayer节点: {len(regionlayer_data)}")

    if hierarchy_loader:
        report.append(f"  - Class节点: {len(hierarchy_loader.class_data)}")
        report.append(f"  - Subclass节点: {len(hierarchy_loader.subclass_data)}")
        report.append(f"  - Supertype节点: {len(hierarchy_loader.supertype_data)}")
        report.append(f"  - Cluster节点: {len(hierarchy_loader.cluster_data)}")
    elif not merfish_cells.empty:
        if 'class' in merfish_cells.columns:
            report.append(f"  - Class节点: {merfish_cells['class'].nunique()}")
        if 'subclass' in merfish_cells.columns:
            report.append(f"  - Subclass节点: {merfish_cells['subclass'].nunique()}")
        if 'supertype' in merfish_cells.columns:
            report.append(f"  - Supertype节点: {merfish_cells['supertype'].nunique()}")
        if 'cluster' in merfish_cells.columns:
            report.append(f"  - Cluster节点: {merfish_cells['cluster'].nunique()}")

    report.append("")
    report.append("数据统计:")
    report.append(f"  - 总细胞数: {len(merfish_cells)}")
    report.append(
        f"  - 有region_id的细胞数: {merfish_cells['region_id'].notna().sum() if 'region_id' in merfish_cells.columns else 0}")
    report.append(
        f"  - 平均每个区域的细胞数: {len(merfish_cells) / len(region_data) if len(region_data) > 0 else 0:.0f}")

    # 层特异性统计
    if 'morph_neuron_count' in regionlayer_data.columns:
        layer_stats = regionlayer_data.groupby('layer')['morph_neuron_count'].sum()
        report.append("")
        report.append("层特异性神经元分布:")
        for layer, count in layer_stats.items():
            report.append(f"  - {layer}: {count} 个神经元")

    report.append("")
    report.append(f"生成时间: {pd.Timestamp.now()}")
    report.append("=" * 60)

    report_text = "\n".join(report)

    report_file = output_dir / "statistics_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    logger.info(f"生成了统计报告: {report_file}")
    print(report_text)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建 - 完整增强版')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./knowledge_graph',
                        help='输出目录路径')
    parser.add_argument('--hierarchy_json', type=str, default=None,
                        help='MERFISH层级JSON文件路径')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.hierarchy_json)