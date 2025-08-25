"""
NeuroXiv 2.0 知识图谱构建器 - 完整增强版
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
from typing import Dict, List

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
        # 区域名称到ID的映射
        self.region_name_id_map = {}

        # 已使用的区域ID集合，用于确保唯一性
        self.used_region_ids = set()

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

            # 检查并记录axon_df的列
            logger.info(f"轴突数据列: {self.axon_df.columns.tolist()}")

            # 检查是否存在'name'列，如果不存在则跳过基于name的过滤
            if 'name' in self.axon_df.columns:
                # 过滤掉CCF-thin和local
                self.axon_df = self.axon_df[~self.axon_df['name'].str.contains('CCF-thin|local', na=False)]
                logger.info(f"基于'name'列过滤后的轴突数据行数: {len(self.axon_df)}")
            else:
                # 尝试找到可能的替代列
                name_alternatives = ['neuron_name', 'cell_name', 'morphology_name', 'reconstruction_name', 'label',
                                     'id', 'ID']
                filtered = False

                for alt_col in name_alternatives:
                    if alt_col in self.axon_df.columns:
                        logger.info(f"使用 '{alt_col}' 列替代 'name' 列进行过滤")
                        before_count = len(self.axon_df)
                        self.axon_df = self.axon_df[
                            ~self.axon_df[alt_col].astype(str).str.contains('CCF-thin|local', na=False)]
                        after_count = len(self.axon_df)
                        logger.info(
                            f"过滤前: {before_count}行, 过滤后: {after_count}行, 移除了: {before_count - after_count}行")
                        filtered = True
                        break

                if not filtered:
                    logger.warning("轴突数据中没有找到可用于过滤的'name'列或替代列，跳过过滤步骤")

                logger.info(f"轴突形态数据行数: {len(self.axon_df)}")

        # 加载树突形态数据
        dendrite_file = self.data_dir / "denfull_morpho.csv"
        if dendrite_file.exists():
            self.dendrite_df = pd.read_csv(dendrite_file)

            # 检查并记录dendrite_df的列
            logger.info(f"树突数据列: {self.dendrite_df.columns.tolist()}")

            # 检查是否存在'name'列，如果不存在则跳过基于name的过滤
            if 'name' in self.dendrite_df.columns:
                # 过滤掉CCF-thin和local
                self.dendrite_df = self.dendrite_df[~self.dendrite_df['name'].str.contains('CCF-thin|local', na=False)]
                logger.info(f"基于'name'列过滤后的树突数据行数: {len(self.dendrite_df)}")
            else:
                # 尝试找到可能的替代列
                name_alternatives = ['neuron_name', 'cell_name', 'morphology_name', 'reconstruction_name', 'label',
                                     'id', 'ID']
                filtered = False

                for alt_col in name_alternatives:
                    if alt_col in self.dendrite_df.columns:
                        logger.info(f"使用 '{alt_col}' 列替代 'name' 列进行过滤")
                        before_count = len(self.dendrite_df)
                        self.dendrite_df = self.dendrite_df[
                            ~self.dendrite_df[alt_col].astype(str).str.contains('CCF-thin|local', na=False)]
                        after_count = len(self.dendrite_df)
                        logger.info(
                            f"过滤前: {before_count}行, 过滤后: {after_count}行, 移除了: {before_count - after_count}行")
                        filtered = True
                        break

                if not filtered:
                    logger.warning("树突数据中没有找到可用于过滤的'name'列或替代列，跳过过滤步骤")

                logger.info(f"树突形态数据行数: {len(self.dendrite_df)}")

        return True

    def calculate_regionlayer_morphology(self, region_data: pd.DataFrame) -> pd.DataFrame:
        """计算层特异性形态特征，处理缺少region_id的情况"""
        logger.info("计算层特异性形态特征...")

        # 验证输入数据
        if region_data.empty:
            logger.warning("输入的region_data为空")
            return pd.DataFrame()

        # 检查必要的信息
        if self.axon_df is None or self.dendrite_df is None or self.info_df is None:
            logger.warning("缺少必要的形态数据")
            return pd.DataFrame()

        # 记录统计信息
        logger.info(f"轴突数据: {len(self.axon_df) if self.axon_df is not None else 0} 行")
        logger.info(f"树突数据: {len(self.dendrite_df) if self.dendrite_df is not None else 0} 行")
        logger.info(f"神经元信息: {len(self.info_df) if self.info_df is not None else 0} 行")

        # 检查并处理info_df中缺少的列
        required_columns = ['ID', 'layer', 'region_id']
        missing_columns = [col for col in required_columns if col not in self.info_df.columns]

        if missing_columns:
            logger.warning(f"神经元信息中缺少列: {missing_columns}")

            # 处理缺少的layer列
            if 'layer' in missing_columns:
                logger.warning("缺少layer列，尝试从celltype提取")
                if 'celltype' in self.info_df.columns:
                    self.info_df['layer'] = self.info_df['celltype'].apply(self._extract_layer_from_celltype)
                else:
                    logger.warning("无法提取layer信息，使用默认值'Unknown'")
                    self.info_df['layer'] = 'Unknown'

            # 处理缺少的region_id列
            if 'region_id' in missing_columns:
                logger.warning("缺少region_id列，尝试从其他列提取或使用区域名称匹配")

                # 方法1：如果有region或brain_region列
                if 'region' in self.info_df.columns:
                    logger.info("从'region'列提取region_id")
                    self.info_df['region_id'] = self.info_df['region'].apply(self._extract_region_id)
                elif 'brain_region' in self.info_df.columns:
                    logger.info("从'brain_region'列提取region_id")
                    self.info_df['region_id'] = self.info_df['brain_region'].apply(self._extract_region_id)
                # 方法2：如果有celltype列，可能包含区域信息
                elif 'celltype' in self.info_df.columns:
                    logger.info("从'celltype'列提取region_id")
                    self.info_df['region_id'] = self.info_df['celltype'].apply(self._extract_region_id_from_celltype)
                else:
                    # 如果无法提取，使用默认值并记录警告
                    logger.warning("无法提取region_id，将使用-1作为默认值")
                    self.info_df['region_id'] = -1

        # 记录处理后的神经元信息统计
        valid_region_count = (self.info_df['region_id'] > 0).sum() if 'region_id' in self.info_df.columns else 0
        valid_layer_count = (self.info_df['layer'] != 'Unknown').sum() if 'layer' in self.info_df.columns else 0
        logger.info(f"有效region_id数量: {valid_region_count}")
        logger.info(f"有效layer数量: {valid_layer_count}")

        # 准备合并所需的列
        info_cols = ['ID']
        if 'layer' in self.info_df.columns:
            info_cols.append('layer')
        if 'region_id' in self.info_df.columns:
            info_cols.append('region_id')

        # 合并ID和层信息
        axon_with_info = pd.merge(self.axon_df, self.info_df[info_cols], on='ID', how='inner')
        dendrite_with_info = pd.merge(self.dendrite_df, self.info_df[info_cols], on='ID', how='inner')

        logger.info(f"合并后的轴突数据: {len(axon_with_info)} 行")
        logger.info(f"合并后的树突数据: {len(dendrite_with_info)} 行")

        # 创建结果数据框
        regionlayer_data = []

        # 定义形态学和统计属性
        morph_attributes = [
            'axonal_bifurcation_remote_angle', 'axonal_branches', 'axonal_length',
            'axonal_maximum_branch_order', 'dendritic_bifurcation_remote_angle',
            'dendritic_branches', 'dendritic_length', 'dendritic_maximum_branch_order'
        ]

        stat_attributes = [
            'number_of_apical_dendritic_morphologies', 'number_of_axonal_morphologies',
            'number_of_dendritic_morphologies', 'number_of_neuron_morphologies',
            'number_of_transcriptomic_neurons'
        ]

        # 为每个区域的每个层计算特征
        for _, region in region_data.iterrows():
            region_id = region.get('region_id')

            # 跳过无效区域ID
            if not region_id or pd.isna(region_id):
                continue

            region_name = region.get('name', f'Region {region_id}')

            # 如果info_df中有region_id列，过滤特定区域的神经元
            if 'region_id' in self.info_df.columns:
                region_axons = axon_with_info[axon_with_info['region_id'] == region_id]
                region_dendrites = dendrite_with_info[dendrite_with_info['region_id'] == region_id]
            else:
                # 如果没有region_id，尝试通过区域名称匹配
                if 'region' in self.info_df.columns:
                    region_axons = axon_with_info[axon_with_info['region'] == region_name]
                    region_dendrites = dendrite_with_info[dendrite_with_info['region'] == region_name]
                else:
                    # 如果无法匹配区域，使用所有数据（不理想，但至少提供一些默认值）
                    logger.warning(f"无法为区域 {region_name} 过滤神经元，使用全局数据")
                    region_axons = axon_with_info
                    region_dendrites = dendrite_with_info

            # 为每个层计算特征
            for layer in LAYERS:
                # 如果有layer列，过滤该层的神经元
                if 'layer' in axon_with_info.columns:
                    layer_axons = region_axons[region_axons['layer'] == layer]
                    layer_dendrites = region_dendrites[region_dendrites['layer'] == layer]
                else:
                    # 如果没有layer信息，使用所有神经元（不理想）
                    logger.warning(f"无法过滤层 {layer} 的神经元，使用所有区域神经元")
                    layer_axons = region_axons
                    layer_dendrites = region_dendrites

                # 记录神经元数量
                axon_count = len(layer_axons)
                dendrite_count = len(layer_dendrites)
                unique_neuron_ids = set(layer_axons['ID'].tolist() + layer_dendrites['ID'].tolist())
                neuron_count = len(unique_neuron_ids)

                # 创建基础区域层字典
                rl_dict = {
                    'rl_id': f"{region_id}_{layer}",
                    'region_id': region_id,
                    'layer': layer,
                    'region_name': region_name,
                    'number_of_neuron_morphologies': neuron_count,
                    'number_of_axonal_morphologies': axon_count,
                    'number_of_dendritic_morphologies': dendrite_count,
                    'number_of_apical_dendritic_morphologies': 0,  # 默认值
                    'number_of_transcriptomic_neurons': 0  # 默认值，将在后续通过MERFISH数据更新
                }

                # 计算特定形态特征映射
                morph_mapping = {
                    'Average Bifurcation Angle Remote': 'bifurcation_remote_angle',
                    'Number of Bifurcations': 'branches',
                    'Total Length': 'length',
                    'Max Branch Order': 'maximum_branch_order'
                }

                # 如果有足够的神经元，计算层特异性特征
                if neuron_count >= 5:  # 最低阈值，确保统计意义
                    # 计算轴突特征
                    if axon_count > 0:
                        for source_feat, target_feat in morph_mapping.items():
                            if source_feat in layer_axons.columns:
                                rl_dict[f'axonal_{target_feat}'] = layer_axons[source_feat].mean(skipna=True)
                            else:
                                rl_dict[f'axonal_{target_feat}'] = 0.0
                    else:
                        # 设置默认值
                        for target_feat in ['bifurcation_remote_angle', 'branches', 'length', 'maximum_branch_order']:
                            rl_dict[f'axonal_{target_feat}'] = 0.0

                    # 计算树突特征
                    if dendrite_count > 0:
                        for source_feat, target_feat in morph_mapping.items():
                            if source_feat in layer_dendrites.columns:
                                rl_dict[f'dendritic_{target_feat}'] = layer_dendrites[source_feat].mean(skipna=True)
                            else:
                                rl_dict[f'dendritic_{target_feat}'] = 0.0
                    else:
                        # 设置默认值
                        for target_feat in ['bifurcation_remote_angle', 'branches', 'length', 'maximum_branch_order']:
                            rl_dict[f'dendritic_{target_feat}'] = 0.0
                else:
                    # 神经元太少，使用区域平均值或默认值
                    for attr in morph_attributes:
                        if attr in region:
                            rl_dict[attr] = region[attr]
                        else:
                            rl_dict[attr] = 0.0

                # 添加其他形态特征
                for feat in FULL_MORPH_FEATURES:
                    if feat not in rl_dict and feat in region:
                        rl_dict[feat] = region[feat]

                regionlayer_data.append(rl_dict)

        # 转换为DataFrame
        result_df = pd.DataFrame(regionlayer_data)

        logger.info(f"生成了 {len(result_df)} 个RegionLayer节点的层特异性形态数据")
        return result_df

    def _get_numeric_region_id(self, region_id):
        """
        将区域ID转换为数字ID

        参数:
            region_id: 区域ID，可以是数字、字符串或其他类型

        返回:
            数字区域ID
        """
        # 处理NaN值
        if pd.isna(region_id):
            return -1

        # 处理数字值
        if isinstance(region_id, (int, float)):
            return int(region_id)

        # 处理字符串值
        if isinstance(region_id, str):
            # 如果是纯数字字符串
            if region_id.isdigit():
                return int(region_id)

            # 尝试从字符串中提取数字
            import re
            match = re.search(r'(\d+)', region_id)
            if match:
                return int(match.group(1))

            # 如果是区域名称，尝试从区域名称映射中查找
            if hasattr(self, 'region_name_id_map') and region_id in self.region_name_id_map:
                return self.region_name_id_map[region_id]

        # 如果是其他类型或无法转换，使用哈希值作为ID
        try:
            return abs(hash(str(region_id))) % (10 ** 9)
        except:
            # 如果所有方法都失败，返回默认值
            return -1
    def _extract_region_id(self, region_name):
        """从区域名称提取区域ID"""
        if pd.isna(region_name):
            return -1

        # 如果region_name已经是ID
        if isinstance(region_name, (int, float)):
            return int(region_name)

        # 如果是字符串，检查是否为数字
        if isinstance(region_name, str):
            # 如果是纯数字
            if region_name.isdigit():
                return int(region_name)

            # 尝试从字符串中提取数字ID
            import re
            match = re.search(r'(\d+)', region_name)
            if match:
                return int(match.group(1))

        # 如果无法提取，返回默认值
        return -1

    def _extract_region_id_from_celltype(self, celltype):
        """从celltype字符串提取区域ID"""
        if pd.isna(celltype):
            return -1

        if not isinstance(celltype, str):
            return -1

        # 尝试提取区域部分（通常在celltype的开头部分）
        parts = celltype.split(' ', 1)
        if len(parts) > 0:
            # 检查第一部分是否是区域ID
            if parts[0].isdigit():
                return int(parts[0])

            # 否则尝试在整个字符串中查找区域名称
            import re

            # 查找常见区域缩写模式
            region_patterns = [
                r'\b(SSp|SSs|MOp|MOs|ACA|AI|RSP|PTLp|VIS|AUD)\b',  # 常见皮层区域缩写
                r'\b(CLA|STR|PAL|TH|HY|MB|P|MY|CB)\b',  # 常见皮层下区域缩写
            ]

            for pattern in region_patterns:
                match = re.search(pattern, celltype)
                if match:
                    # 这里需要将区域名称映射到区域ID
                    # 简化处理，返回一个占位值
                    return 1

        # 如果无法提取，返回默认值
        return -1

    def _extract_layer_from_celltype(self, celltype):
        """从celltype字符串提取层信息"""
        if pd.isna(celltype):
            return "Unknown"

        if not isinstance(celltype, str):
            return "Unknown"

        # 转换为小写以便匹配
        celltype_lower = celltype.lower()

        # 匹配常见层模式
        layer_patterns = {
            'l1': 'L1',
            'l2/3': 'L2/3',
            'l2': 'L2/3',
            'l3': 'L2/3',
            'l4': 'L4',
            'l5': 'L5',
            'l6a': 'L6a',
            'l6b': 'L6b',
            'l6': 'L6a'  # 默认归为L6a
        }

        for pattern, layer in layer_patterns.items():
            if pattern in celltype_lower:
                return layer

        return "Unknown"
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
        self.region_name_id_map = {}
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

    def _get_numeric_region_id(self, region_id):
        """
        将区域ID转换为数字ID

        参数:
            region_id: 区域ID，可以是数字、字符串或其他类型

        返回:
            数字区域ID
        """
        # 处理NaN值
        if pd.isna(region_id):
            return -1

        # 处理数字值
        if isinstance(region_id, (int, float)):
            return int(region_id)

        # 处理字符串值
        if isinstance(region_id, str):
            # 如果是纯数字字符串
            if region_id.isdigit():
                return int(region_id)

            # 尝试从字符串中提取数字
            import re
            match = re.search(r'(\d+)', region_id)
            if match:
                return int(match.group(1))

            # 如果是区域名称，尝试从区域名称映射中查找
            if hasattr(self, 'region_name_id_map') and region_id in self.region_name_id_map:
                return self.region_name_id_map[region_id]

        # 如果是其他类型或无法转换，使用哈希值作为ID
        try:
            return abs(hash(str(region_id))) % (10 ** 9)
        except:
            # 如果所有方法都失败，返回默认值
            return -1
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

    def load_hierarchy(self, hierarchy_json_path: str) -> None:
        """
        从JSON文件加载细胞类型层级数据

        参数:
            hierarchy_json_path: 层级JSON文件的路径
        """
        import json

        logger.info(f"从文件加载层级数据: {hierarchy_json_path}")

        try:
            # 读取JSON文件
            with open(hierarchy_json_path, 'r', encoding='utf-8') as f:
                hierarchy_data = json.load(f)

            # 解析层级数据
            self._parse_hierarchy(hierarchy_data)

            # 日志记录映射大小
            logger.info(f"加载完成，共解析到 {len(self.class_id_map)} 个Class, "
                        f"{len(self.subclass_id_map)} 个Subclass, "
                        f"{len(self.supertype_id_map)} 个Supertype, "
                        f"{len(self.cluster_id_map)} 个Cluster")

        except FileNotFoundError:
            logger.error(f"层级文件不存在: {hierarchy_json_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"层级文件格式不正确: {hierarchy_json_path}")
            raise
        except Exception as e:
            logger.error(f"加载层级数据时发生错误: {str(e)}")
            raise

    def _parse_hierarchy(self, hierarchy_data) -> None:
        """
        解析特定格式的层级数据，提取类别映射

        参数:
            hierarchy_data: 层级JSON数据（列表格式）
        """
        logger.info("解析细胞类型层级数据...")

        # 计数器
        class_count = 0
        subclass_count = 0
        supertype_count = 0
        cluster_count = 0

        # 存储映射关系
        class_map = {}
        subclass_map = {}
        supertype_map = {}
        cluster_map = {}

        # 验证数据类型
        if not isinstance(hierarchy_data, list):
            logger.error(f"层级数据必须是列表，实际类型: {type(hierarchy_data)}")
            raise ValueError(f"层级数据必须是列表，实际类型: {type(hierarchy_data)}")

        # 遍历顶级类别（Class层级）
        for class_item in hierarchy_data:
            if not isinstance(class_item, dict):
                logger.warning(f"跳过非字典类型的Class项: {class_item}")
                continue

            # 获取类别名称和标签
            class_name = class_item.get('name')
            class_label = class_item.get('label')

            # 验证这确实是Class层级
            if class_name != "class":
                logger.warning(f"跳过非Class项: {class_item}")
                continue

            # 使用label作为实际类别名称
            if not class_label:
                logger.warning(f"Class项缺少label: {class_item}")
                continue

            # 生成Class ID
            class_id = f"Class_{class_count}"
            class_map[class_label] = class_id
            class_count += 1

            logger.info(f"处理Class: {class_label}")

            # 获取子类列表（Subclass层级）
            subclass_items = class_item.get('child', [])

            # 遍历Subclass
            for subclass_item in subclass_items:
                if not isinstance(subclass_item, dict):
                    continue

                # 获取子类名称和标签
                subclass_name = subclass_item.get('name')
                subclass_label = subclass_item.get('label')

                # 验证这是Subclass层级
                if subclass_name != "subclass":
                    continue

                # 使用label作为实际子类名称
                if not subclass_label:
                    continue

                # 生成Subclass ID
                subclass_id = f"Subclass_{subclass_count}"
                subclass_map[subclass_label] = subclass_id
                subclass_count += 1

                # 获取超类列表（Supertype层级）
                supertype_items = subclass_item.get('child', [])

                # 遍历Supertype
                for supertype_item in supertype_items:
                    if not isinstance(supertype_item, dict):
                        continue

                    # 获取超类名称和标签
                    supertype_name = supertype_item.get('name')
                    supertype_label = supertype_item.get('label')

                    # 验证这是Supertype层级
                    if supertype_name != "supertype":
                        continue

                    # 使用label作为实际超类名称
                    if not supertype_label:
                        continue

                    # 生成Supertype ID
                    supertype_id = f"Supertype_{supertype_count}"
                    supertype_map[supertype_label] = supertype_id
                    supertype_count += 1

                    # 获取集群列表（Cluster层级）
                    cluster_items = supertype_item.get('child', [])

                    # 遍历Cluster
                    for cluster_item in cluster_items:
                        if not isinstance(cluster_item, dict):
                            continue

                        # 获取集群名称和标签
                        cluster_name = cluster_item.get('name')
                        cluster_label = cluster_item.get('label')

                        # 验证这是Cluster层级
                        if cluster_name != "cluster":
                            continue

                        # 使用label作为实际集群名称
                        if not cluster_label:
                            continue

                        # 生成Cluster ID
                        cluster_id = f"Cluster_{cluster_count}"
                        cluster_map[cluster_label] = cluster_id

                        # 处理MERFISH数据中的格式: '5312 Microglia NN_1'
                        if ' ' in cluster_label:
                            parts = cluster_label.split(' ', 1)
                            if parts[0].isdigit():
                                id_prefix = parts[0]
                                # 存储带ID前缀的映射
                                for suffix in [parts[1], f"{parts[1]}_1", f"{parts[1]}_2"]:
                                    full_name = f"{id_prefix} {suffix}"
                                    cluster_map[full_name] = cluster_id

                        # 存储不带空格的版本
                        no_space = cluster_label.replace(' ', '')
                        cluster_map[no_space] = cluster_id

                        # 尝试其他常见格式
                        cluster_map[cluster_label.lower()] = cluster_id

                        cluster_count += 1

        # 保存ID映射
        self.class_id_map = class_map
        self.subclass_id_map = subclass_map
        self.supertype_id_map = supertype_map
        self.cluster_id_map = cluster_map

        logger.info(
            f"解析完成: {class_count} Class, {subclass_count} Subclass, {supertype_count} Supertype, {cluster_count} Cluster")

    def extract_hierarchy_from_cells(self, merfish_cells: pd.DataFrame) -> None:
        """
        从细胞数据中提取层级信息

        参数:
            merfish_cells: 包含细胞类型信息的DataFrame
        """
        logger.info("从细胞数据中提取层级信息...")

        # 检查必要的列
        hierarchy_levels = ['class', 'subclass', 'supertype', 'cluster']
        missing_cols = [col for col in hierarchy_levels if col not in merfish_cells.columns]

        if missing_cols:
            logger.warning(f"细胞数据缺少必要的列: {missing_cols}")
            return

        # 初始化映射
        class_map = {}
        subclass_map = {}
        supertype_map = {}
        cluster_map = {}

        # 提取唯一值并创建映射
        if 'class' in merfish_cells.columns:
            unique_classes = merfish_cells['class'].dropna().unique()
            for i, class_name in enumerate(unique_classes):
                class_map[class_name] = f"Class_{i}"

        if 'subclass' in merfish_cells.columns:
            unique_subclasses = merfish_cells['subclass'].dropna().unique()
            for i, subclass_name in enumerate(unique_subclasses):
                subclass_map[subclass_name] = f"Subclass_{i}"

        if 'supertype' in merfish_cells.columns:
            unique_supertypes = merfish_cells['supertype'].dropna().unique()
            for i, supertype_name in enumerate(unique_supertypes):
                supertype_map[supertype_name] = f"Supertype_{i}"

        if 'cluster' in merfish_cells.columns:
            unique_clusters = merfish_cells['cluster'].dropna().unique()
            for i, cluster_name in enumerate(unique_clusters):
                cluster_id = f"Cluster_{i}"
                cluster_map[cluster_name] = cluster_id

                # 处理MERFISH数据中的格式: '5312 Microglia NN_1'
                if ' ' in cluster_name:
                    parts = cluster_name.split(' ', 1)
                    if parts[0].isdigit():
                        id_prefix = parts[0]
                        # 存储带ID前缀的映射
                        for suffix in [parts[1], f"{parts[1]}_1", f"{parts[1]}_2"]:
                            full_name = f"{id_prefix} {suffix}"
                            cluster_map[full_name] = cluster_id

                # 存储不带空格的版本
                no_space = cluster_name.replace(' ', '')
                cluster_map[no_space] = cluster_id

        # 保存ID映射
        self.class_id_map = class_map
        self.subclass_id_map = subclass_map
        self.supertype_id_map = supertype_map
        self.cluster_id_map = cluster_map

        logger.info(f"从细胞数据提取完成: {len(class_map)} Class, {len(subclass_map)} Subclass, "
                    f"{len(supertype_map)} Supertype, {len(cluster_map)} Cluster")
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
        """生成Region节点，使用acronym作为名称"""
        logger.info("生成Region节点...")

        # 诊断信息
        logger.info(f"区域数据形状: {region_data.shape}")

        # 验证是否有acronym列
        has_acronym = 'acronym' in region_data.columns
        if not has_acronym:
            logger.warning("区域数据缺少acronym列，将使用区域ID作为名称")

        # 形态学属性列表
        morph_attributes = [
            'axonal_bifurcation_remote_angle', 'axonal_branches', 'axonal_length',
            'axonal_maximum_branch_order', 'dendritic_bifurcation_remote_angle',
            'dendritic_branches', 'dendritic_length', 'dendritic_maximum_branch_order'
        ]

        # 统计属性列表
        stat_attributes = [
            'number_of_apical_dendritic_morphologies', 'number_of_axonal_morphologies',
            'number_of_dendritic_morphologies', 'number_of_neuron_morphologies',
            'number_of_transcriptomic_neurons'
        ]

        # 首先构建区域名称到ID的映射
        for _, region in region_data.iterrows():
            region_id = region.get('region_id')
            if pd.isna(region_id):
                continue

            region_name = region.get('name', f'Region_{region_id}')
            if region_name and not pd.isna(region_name):
                self.region_name_id_map[region_name] = region_id

        regions = []
        for _, region in region_data.iterrows():
            # 获取区域ID
            region_id = region.get('region_id')

            if pd.isna(region_id):
                logger.warning(f"跳过缺少region_id的区域: {region.get('name', 'Unknown')}")
                continue

            # 使用中央映射获取数字ID
            region_id_int = self._get_numeric_region_id(region_id)

            # 使用acronym作为名称，如果有的话
            if has_acronym and not pd.isna(region.get('acronym')):
                name = region.get('acronym')
            else:
                name = region.get('name', f'Region_{region_id}')

            # 创建区域字典
            region_dict = {
                'region_id:ID(Region)': region_id_int,
                'original_id': str(region_id),
                'name': str(name),  # 使用acronym作为名称
                'full_name': str(region.get('name', '')),  # 保留完整名称
            }

            # 添加acronym如果存在
            if has_acronym:
                region_dict['acronym'] = str(region.get('acronym', ''))

            # 添加其他基本属性
            if 'color' in region:
                region_dict['color:int[]'] = region.get('color', [200, 200, 200])
            if 'parent_id' in region:
                region_dict['parent_id:int'] = int(region.get('parent_id', 0))
            if 'depth' in region:
                region_dict['depth:int'] = int(region.get('depth', 0))
            if 'graph_order' in region:
                region_dict['graph_order:int'] = int(region.get('graph_order', 0))

            # 添加神经元计数
            if 'complete_neuron_count' in region:
                region_dict['complete_neuron_count:int'] = int(region.get('complete_neuron_count', 0))
            elif 'morpho_count' in region:
                region_dict['complete_neuron_count:int'] = int(region.get('morpho_count', 0))

            # 添加形态学属性
            for attr in morph_attributes:
                if attr in region:
                    region_dict[f'{attr}:float'] = float(region[attr])
                else:
                    region_dict[f'{attr}:float'] = 0.0

            # 添加统计属性
            for attr in stat_attributes:
                if attr in region:
                    region_dict[f'{attr}:int'] = int(region[attr])
                else:
                    # 尝试从其他列推断
                    if attr == 'number_of_neuron_morphologies' and 'neuron_count' in region:
                        region_dict[f'{attr}:int'] = int(region['neuron_count'])
                    elif attr == 'number_of_transcriptomic_neurons' and 'complete_neuron_count' in region:
                        region_dict[f'{attr}:int'] = int(region['complete_neuron_count'])
                    else:
                        region_dict[f'{attr}:int'] = 0

            # 添加其他形态特征 (如果有的话)
            for col in region.index:
                if col not in region_dict and col not in ['region_id', 'name', 'acronym'] and not pd.isna(region[col]):
                    try:
                        # 尝试转换为适当的类型
                        if isinstance(region[col], (int, np.integer)):
                            region_dict[f'{col}:int'] = int(region[col])
                        elif isinstance(region[col], (float, np.floating)):
                            region_dict[f'{col}:float'] = float(region[col])
                        else:
                            region_dict[f'{col}'] = str(region[col])
                    except (ValueError, TypeError):
                        # 如果转换失败，保留为字符串
                        region_dict[f'{col}'] = str(region[col])

            regions.append(region_dict)

        # 保存到CSV
        self._save_nodes(regions, "regions")
        logger.info(f"保存了 {len(regions)} 个Region节点")

    def generate_regionlayer_nodes(self, regionlayer_data: pd.DataFrame, merfish_cells: pd.DataFrame = None):
        """生成包含完整形态学和统计属性的RegionLayer节点"""
        logger.info("生成RegionLayer节点...")

        # 形态学属性列表
        morph_attributes = [
            'axonal_bifurcation_remote_angle', 'axonal_branches', 'axonal_length',
            'axonal_maximum_branch_order', 'dendritic_bifurcation_remote_angle',
            'dendritic_branches', 'dendritic_length', 'dendritic_maximum_branch_order'
        ]

        # 统计属性列表
        stat_attributes = [
            'number_of_apical_dendritic_morphologies', 'number_of_axonal_morphologies',
            'number_of_dendritic_morphologies', 'number_of_neuron_morphologies',
            'number_of_transcriptomic_neurons'
        ]

        regionlayers = []
        for _, rl in regionlayer_data.iterrows():
            # 获取RegionLayer ID
            rl_id = rl.get('rl_id', '')

            # 如果rl_id格式是"region_id_layer"，提取region_id部分
            region_id_str = rl_id.split('_')[0] if '_' in rl_id else rl_id

            # 使用中央映射获取数字ID
            region_id = self._get_numeric_region_id(region_id_str)

            # 获取层信息
            layer = rl.get('layer', 'Unknown')

            # 生成合成ID：region_id_layer
            synth_id = f"{region_id}_{layer}"

            # 创建RegionLayer字典
            rl_dict = {
                'rl_id:ID(RegionLayer)': synth_id,
                'original_id': rl_id,  # 保留原始ID
                'region_id:int': region_id,
                'layer': layer,
                'name': f"{rl.get('region_name', f'Region {region_id}')} {layer}",
                'morph_neuron_count:int': int(rl.get('morph_neuron_count', 0))
            }

            # 添加形态学属性
            for attr in morph_attributes:
                if attr in rl:
                    rl_dict[f'{attr}:float'] = float(rl[attr])
                else:
                    rl_dict[f'{attr}:float'] = 0.0

            # 添加统计属性
            for attr in stat_attributes:
                if attr in rl:
                    rl_dict[f'{attr}:int'] = int(rl[attr])
                else:
                    # 尝试从其他列推断
                    if attr == 'number_of_neuron_morphologies' and 'morph_neuron_count' in rl:
                        rl_dict[f'{attr}:int'] = int(rl['morph_neuron_count'])
                    else:
                        rl_dict[f'{attr}:int'] = 0

            # 添加其他形态特征
            for feat in FULL_MORPH_FEATURES:
                if feat in rl and feat not in morph_attributes:
                    try:
                        rl_dict[f'{feat}:float'] = float(rl[feat])
                    except (ValueError, TypeError):
                        rl_dict[f'{feat}:float'] = 0.0

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

                    # 如果有class信息，计算类型统计
                    if 'class' in layer_cells.columns:
                        top_classes = layer_cells['class'].value_counts().head(3)
                        rl_dict['top_classes'] = '; '.join([f"{c}:{n}" for c, n in top_classes.items()])

            regionlayers.append(rl_dict)

        # 保存到CSV
        self._save_nodes(regionlayers, "regionlayers")
        logger.info(f"保存了 {len(regionlayers)} 个RegionLayer节点")

    def generate_has_layer_relationships(self, region_data: pd.DataFrame):
        """生成HAS_LAYER关系，处理非数字区域ID"""
        logger.info("生成HAS_LAYER关系...")

        # 创建区域ID映射（如果还没有）
        if not hasattr(self, 'region_id_map'):
            self.region_id_map = {}

        relationships = []
        for _, region in region_data.iterrows():
            # 获取区域ID
            region_id = region.get('region_id', region.name)

            # 处理非数字区域ID
            if isinstance(region_id, (int, float)):
                numeric_id = int(region_id)
            elif isinstance(region_id, str) and region_id.isdigit():
                numeric_id = int(region_id)
            else:
                # 检查是否已有映射
                if region_id in self.region_id_map:
                    numeric_id = self.region_id_map[region_id]
                else:
                    # 创建新映射 - 使用哈希生成负整数ID
                    numeric_id = -abs(hash(str(region_id)) % 1000000)
                    self.region_id_map[region_id] = numeric_id
                    logger.info(f"为非数字区域ID '{region_id}' 创建映射: {numeric_id}")

            # 为每一层创建关系
            for layer in LAYERS:
                rl_id = f"{numeric_id}_{layer}"

                # 创建关系字典
                rel = {
                    ':START_ID(Region)': numeric_id,
                    ':END_ID(RegionLayer)': rl_id,
                    ':TYPE': 'HAS_LAYER'
                }
                relationships.append(rel)

        # 保存关系
        self._save_relationships_batch(relationships, "has_layer")
        logger.info(f"保存了 {len(relationships)} 个HAS_LAYER关系")

    def generate_has_relationships_optimized(self,
                                             merfish_cells: pd.DataFrame,
                                             level: str):
        """优化的HAS_*关系生成，处理缺失列的情况"""
        logger.info(f"生成HAS_{level.upper()}关系...")

        # 验证必要的列是否存在
        if level not in merfish_cells.columns:
            logger.warning(f"没有{level}数据，列不存在")
            # 尝试创建替代列
            if level == 'class' and 'cell_class' in merfish_cells.columns:
                merfish_cells = merfish_cells.copy()
                merfish_cells[level] = merfish_cells['cell_class']
                logger.info(f"使用 'cell_class' 列替代 '{level}' 列")
            elif level == 'subclass' and 'cell_type' in merfish_cells.columns:
                merfish_cells = merfish_cells.copy()
                merfish_cells[level] = merfish_cells['cell_type']
                logger.info(f"使用 'cell_type' 列替代 '{level}' 列")
            elif level == 'cluster' and 'cluster_label' in merfish_cells.columns:
                merfish_cells = merfish_cells.copy()
                merfish_cells[level] = merfish_cells['cluster_label']
                logger.info(f"使用 'cluster_label' 列替代 '{level}' 列")
            else:
                logger.error(f"无法找到 '{level}' 列的替代，无法继续")
                return

        if 'region_id' not in merfish_cells.columns:
            logger.warning("细胞缺少region_id")
            # 检查是否有替代列
            if 'brain_region_id' in merfish_cells.columns:
                merfish_cells = merfish_cells.copy()
                merfish_cells['region_id'] = merfish_cells['brain_region_id']
                logger.info("使用 'brain_region_id' 替代 'region_id'")
            else:
                logger.error("无法找到 'region_id' 的替代，无法继续")
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

        # 记录有效数据的统计信息
        valid_cells = merfish_cells[merfish_cells[level].notna()]
        logger.info(f"总细胞数: {len(merfish_cells)}, 有{level}值的细胞: {len(valid_cells)}")

        valid_region_cells = valid_cells[valid_cells['region_id'] > 0]
        logger.info(f"有region_id的{level}细胞: {len(valid_region_cells)}")

        if len(valid_region_cells) == 0:
            logger.error(f"没有同时有{level}和region_id的细胞，无法生成关系")
            return

        # 按区域和类型分组计数
        grouped = valid_region_cells.groupby(['region_id', level]).size().reset_index(name='count')

        # 计算每个区域的总数
        region_totals = valid_region_cells.groupby('region_id').size().to_dict()

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
                    else:
                        logger.warning(f"类型 '{cell_type}' 不在 {level} ID映射中")

                # 批量保存
                if len(relationships) >= BATCH_SIZE:
                    self._save_relationships_batch(relationships, f"has_{level}")
                    relationships = []

        # 保存剩余的关系
        if relationships:
            self._save_relationships_batch(relationships, f"has_{level}")

        logger.info(f"完成HAS_{level.upper()}关系生成，共生成 {len(relationships)} 个关系")

    def generate_relationships_for_layerless_regions(self, region_data: pd.DataFrame, merfish_cells: pd.DataFrame):
        """为没有层信息的脑区创建直接关系"""
        logger.info("为没有层信息的脑区生成直接关系...")

        # 获取所有区域ID
        all_region_ids = set(region_data['region_id'].tolist())

        # 验证区域层节点中的区域ID
        regionlayer_file = self.output_dir / "regionlayers.csv"
        if not regionlayer_file.exists():
            logger.warning("找不到RegionLayer节点文件，无法确定哪些区域没有层信息")
            return

        # 读取RegionLayer数据
        regionlayer_df = pd.read_csv(regionlayer_file)

        # 提取具有层信息的区域ID
        if 'region_id:int' in regionlayer_df.columns:
            regions_with_layers = set(regionlayer_df['region_id:int'].unique())
        else:
            # 尝试从rl_id中提取
            regions_with_layers = set()
            if 'rl_id:ID(RegionLayer)' in regionlayer_df.columns:
                for rl_id in regionlayer_df['rl_id:ID(RegionLayer)']:
                    parts = str(rl_id).split('_')
                    if len(parts) > 1 and parts[0].isdigit():
                        regions_with_layers.add(int(parts[0]))

        # 找出没有层信息的区域
        layerless_regions = all_region_ids - regions_with_layers

        if not layerless_regions:
            logger.info("所有区域都有层信息，无需创建直接关系")
            return

        logger.info(f"发现 {len(layerless_regions)} 个没有层信息的区域")

        # 为这些区域创建直接关系
        # 1. 直接连接到细胞类型
        self._create_direct_has_relationships(layerless_regions, merfish_cells)

    def _create_direct_has_relationships(self, layerless_regions: set, merfish_cells: pd.DataFrame):
        """为没有层信息的区域创建直接HAS关系到细胞类型"""
        if merfish_cells.empty:
            logger.warning("没有细胞数据，无法创建直接关系")
            return

        # 验证必要的列
        hierarchy_levels = ['class', 'subclass', 'supertype', 'cluster']
        missing_cols = [col for col in hierarchy_levels if col not in merfish_cells.columns]
        if missing_cols:
            logger.warning(f"细胞数据缺少必要的列: {missing_cols}")
            hierarchy_levels = [col for col in hierarchy_levels if col not in missing_cols]

        if 'region_id' not in merfish_cells.columns:
            logger.warning("细胞数据缺少region_id列")
            return

        # 筛选出这些区域的细胞
        valid_cells = merfish_cells[merfish_cells['region_id'].isin(layerless_regions)]

        if valid_cells.empty:
            logger.warning("没有找到位于无层区域的细胞")
            return

        # 为每个层级创建关系
        for level in hierarchy_levels:
            self._create_direct_has_for_level(layerless_regions, valid_cells, level)

    def _create_direct_has_for_level(self, layerless_regions: set, cells: pd.DataFrame, level: str):
        """为特定层级创建直接HAS关系"""
        logger.info(f"为无层区域创建直接HAS_{level.upper()}关系...")

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

        # 筛选有效细胞
        valid_cells = cells[cells[level].notna()]

        if valid_cells.empty:
            logger.warning(f"无层区域没有{level}数据")
            return

        # 按区域和类型分组计数
        grouped = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')

        # 计算区域总数
        region_totals = valid_cells.groupby('region_id').size().to_dict()

        # 生成关系
        relationships = []
        PCT_THRESHOLD = 0.01  # 1%的阈值

        for region_id in grouped['region_id'].unique():
            # 跳过非无层区域
            if region_id not in layerless_regions:
                continue

            region_group = grouped[grouped['region_id'] == region_id]
            total = region_totals.get(region_id, 1)

            # 计算比例并排序
            region_group = region_group.copy()
            region_group['pct'] = region_group['count'] / total
            region_group = region_group[region_group['pct'] >= PCT_THRESHOLD]
            region_group['rank'] = region_group['pct'].rank(ascending=False, method='dense').astype(int)

            for _, row in region_group.iterrows():
                cell_type = row[level]
                if cell_type in id_map:
                    rel = {
                        ':START_ID(Region)': int(region_id),
                        f':END_ID({level.capitalize()})': id_map[cell_type],
                        'pct_cells:float': float(row['pct']),
                        'rank:int': int(row['rank']),
                        'direct:boolean': True,  # 标记为直接关系
                        ':TYPE': f'HAS_{level.upper()}_DIRECT'
                    }
                    relationships.append(rel)
                else:
                    logger.warning(f"类型 '{cell_type}' 不在 {level} ID映射中")

        # 保存关系
        if relationships:
            # 确保输出目录存在
            rel_dir = self.output_dir / "relationships"
            rel_dir.mkdir(parents=True, exist_ok=True)

            # 确定输出文件
            output_file = rel_dir / f"has_{level}_direct.csv"

            # 创建DataFrame并保存
            df = pd.DataFrame(relationships)
            df.to_csv(output_file, index=False)

            logger.info(f"保存了 {len(relationships)} 个直接HAS_{level.upper()}关系")
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
        """为特定层级生成Dominant_transcriptomic关系"""
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
            rel_dir = self.output_dir / "relationships"
            rel_dir.mkdir(parents=True, exist_ok=True)

            # 确定输出文件
            output_file = rel_dir / f"dominant_transcriptomic_{level}.csv"

            # 创建DataFrame并保存
            df = pd.DataFrame(relationships)
            df.to_csv(output_file, index=False)

            logger.info(f"保存了 {len(relationships)} 个{level}级别的Dominant_transcriptomic关系到 {output_file}")
        else:
            logger.warning(f"没有生成{level}级别的Dominant_transcriptomic关系")

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

    def _save_nodes(self, nodes: List[Dict], node_type: str):
        """
        保存节点数据到CSV文件

        参数:
            nodes: 节点数据列表
            node_type: 节点类型名称
        """
        if not nodes:
            logger.warning(f"没有{node_type}节点需要保存")
            return

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确定输出文件路径
        output_file = self.output_dir / f"{node_type}.csv"

        # 获取所有可能的列名
        all_columns = set()
        for node in nodes:
            all_columns.update(node.keys())

        # 将列名排序，使ID列在前
        columns = sorted(all_columns, key=lambda x: (0 if ':ID' in x else 1, x))

        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(nodes)
        df = df.reindex(columns=columns)  # 重新排列列顺序

        # 保存为CSV，注意设置编码和引号选项
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')

        logger.info(f"已将{len(nodes)}个{node_type}节点保存到 {output_file}")


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
            """并行加载MERFISH细胞数据并映射到区域"""
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

                # 1. 验证并修复坐标列
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

                # 2. 调用 data_loader_enhanced 中的函数映射细胞到区域
                from data_loader_enhanced import map_cells_to_regions_fixed
                try:
                    # 加载注释体积
                    logger.info("加载注释体积以映射细胞...")
                    data = load_data(self.data_dir)

                    if 'annotation' in data and 'volume' in data['annotation'] and 'header' in data['annotation']:
                        logger.info("开始映射细胞到区域...")
                        self.merfish_cells = map_cells_to_regions_fixed(
                            self.merfish_cells,
                            data['annotation']['volume'],
                            data['annotation']['header']
                        )
                        logger.info(f"映射完成，有区域ID的细胞数: {(self.merfish_cells['region_id'] > 0).sum()}")
                    else:
                        logger.error("无法加载注释体积数据，无法映射细胞到区域")
                except Exception as e:
                    logger.error(f"映射细胞到区域时出错: {e}")

                # 3. 添加区域名称和层级信息
                try:
                    # 加载树结构
                    tree_file = self.data_dir / "tree_yzx.json"
                    if tree_file.exists():
                        logger.info("加载树结构以获取区域信息...")
                        with open(tree_file, 'r') as f:
                            tree_data = json.load(f)

                        # 创建区域ID到信息的映射
                        region_info = {}
                        for node in tree_data:
                            if 'id' in node and 'acronym' in node:
                                region_id = node['id']
                                region_info[region_id] = {
                                    'acronym': node.get('acronym', ''),
                                    'name': node.get('name', ''),
                                    'parent_id': node.get('parent_id', 0)
                                }

                        # 添加区域名称列
                        self.merfish_cells['region_acronym'] = self.merfish_cells['region_id'].map(
                            lambda x: region_info.get(x, {}).get('acronym', '') if x > 0 else ''
                        )

                        self.merfish_cells['region_name'] = self.merfish_cells['region_id'].map(
                            lambda x: region_info.get(x, {}).get('name', '') if x > 0 else ''
                        )

                        logger.info("已添加区域名称信息")

                        # 4. 尝试识别缺失的level列
                        # 如果merfish_cells中没有level列(class, subclass等)，尝试从元数据提取
                        level_columns = ['class', 'subclass', 'supertype', 'cluster']
                        missing_levels = [col for col in level_columns if col not in self.merfish_cells.columns]

                        if missing_levels:
                            logger.warning(f"缺少细胞层级列: {missing_levels}")

                            # 尝试从其他列推断
                            possible_sources = ['cell_class', 'cell_type', 'cell_cluster', 'type', 'cluster_label']

                            for level in missing_levels:
                                # 尝试从可能的源列中查找
                                for source in possible_sources:
                                    if source in self.merfish_cells.columns:
                                        self.merfish_cells[level] = self.merfish_cells[source]
                                        logger.info(f"使用列 '{source}' 作为 '{level}'")
                                        break

                        # 记录层级列的可用性
                        for level in level_columns:
                            if level in self.merfish_cells.columns:
                                unique_values = self.merfish_cells[level].nunique()
                                logger.info(f"列 '{level}' 有 {unique_values} 个唯一值")
                            else:
                                logger.warning(f"列 '{level}' 不存在")

                    else:
                        logger.warning(f"树结构文件不存在: {tree_file}")
                except Exception as e:
                    logger.error(f"添加区域名称和层级信息时出错: {e}")
            else:
                self.merfish_cells = pd.DataFrame()

            return self.merfish_cells

        def _load_merfish_file(self, coord_file: Path, index: int) -> pd.DataFrame:
            """
            加载单个MERFISH文件，并进行坐标标准化和缩放处理

            参数:
                coord_file: 坐标文件路径
                index: 文件索引

            返回:
                处理后的细胞数据DataFrame
            """
            try:
                # 记录开始加载
                logger.info(f"加载MERFISH坐标文件: {coord_file.name}")

                # 加载坐标文件
                coords = pd.read_csv(coord_file)
                logger.info(f"成功加载 {len(coords)} 行坐标数据")

                # 记录原始列
                logger.debug(f"原始坐标文件列: {coords.columns.tolist()}")

                # 1. 标准化坐标列名 - 确保坐标列使用标准名称
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

                # 应用列名映射
                for old_col, new_col in coord_mapping.items():
                    if old_col in coords.columns and new_col not in coords.columns:
                        coords = coords.rename(columns={old_col: new_col})
                        logger.debug(f"将列 '{old_col}' 重命名为 '{new_col}'")

                # 2. 检查并缩放坐标 - 确保坐标使用25μm分辨率
                for col in ['x_ccf', 'y_ccf', 'z_ccf']:
                    if col in coords.columns:
                        # 检查是否需要缩放 (如果最大值小于20，假设是毫米单位)
                        max_val = coords[col].max()
                        if max_val is not None and max_val < 20:
                            # 记录原始范围
                            orig_range = f"{coords[col].min():.2f} - {max_val:.2f}"

                            # 应用缩放因子40 (将毫米转换为25μm单位)
                            coords[col] = coords[col] * 40

                            # 记录缩放后范围
                            new_max = coords[col].max()
                            new_range = f"{coords[col].min():.2f} - {new_max:.2f}"

                            logger.info(f"将{col}从mm转换为25μm分辨率 (×40): {orig_range} -> {new_range}")

                # 3. 验证所需的坐标列是否存在
                missing_cols = [col for col in ['x_ccf', 'y_ccf', 'z_ccf'] if col not in coords.columns]
                if missing_cols:
                    logger.warning(f"坐标文件 {coord_file.name} 缺少必要的列: {missing_cols}")
                    logger.warning(f"可用列: {coords.columns.tolist()}")

                # 4. 加载元数据文件并合并
                meta_file = coord_file.parent / f"cell_metadata_with_cluster_annotation_{index + 1}.csv"
                if meta_file.exists():
                    logger.info(f"加载对应的元数据文件: {meta_file.name}")
                    try:
                        meta = pd.read_csv(meta_file)
                        logger.info(f"成功加载 {len(meta)} 行元数据")

                        # 记录元数据列
                        logger.debug(f"元数据文件列: {meta.columns.tolist()}")

                        # 尝试基于cell_label合并
                        if 'cell_label' in coords.columns and 'cell_label' in meta.columns:
                            logger.info("使用'cell_label'列合并坐标和元数据")
                            # 记录合并前行数
                            pre_merge_rows = len(coords)

                            # 执行合并
                            coords = pd.merge(coords, meta, on='cell_label', how='left')

                            # 验证合并结果
                            logger.info(f"合并后行数: {len(coords)} (原始坐标行数: {pre_merge_rows})")

                            # 检查缺失值
                            null_pct = (coords.isnull().sum() / len(coords)).max() * 100
                            if null_pct > 10:
                                logger.warning(f"合并后存在较多缺失值 (最大缺失率: {null_pct:.1f}%)")

                        # 如果没有公共的cell_label列但行数相同，假设顺序对应
                        elif len(coords) == len(meta):
                            logger.info("坐标和元数据行数相同，假设顺序对应")

                            # 跳过已存在的列(避免重复)
                            for col in meta.columns:
                                if col not in coords.columns:
                                    coords[col] = meta[col].values

                            logger.info(f"已添加 {len(meta.columns)} 个元数据列")
                        else:
                            logger.warning("无法合并元数据：没有共同的cell_label列且行数不同")
                    except Exception as e:
                        logger.error(f"加载或合并元数据时出错: {e}")
                else:
                    logger.warning(f"元数据文件不存在: {meta_file}")

                # 5. 添加文件来源信息
                coords['source_file'] = coord_file.name

                # 6. 数据清理和标准化

                # 确保有cell_label列
                if 'cell_label' not in coords.columns:
                    # 尝试从其他列创建
                    for id_col in ['id', 'cell_id', 'ID', 'Cell_ID']:
                        if id_col in coords.columns:
                            coords['cell_label'] = coords[id_col]
                            logger.info(f"使用 '{id_col}' 创建 'cell_label' 列")
                            break
                    else:
                        # 如果没有合适的列，创建序列号作为cell_label
                        coords['cell_label'] = [f"cell_{coord_file.stem}_{i}" for i in range(len(coords))]
                        logger.info("没有找到ID列，创建序列号作为cell_label")

                # 记录处理后的列
                logger.debug(f"处理后的列: {coords.columns.tolist()}")
                logger.info(f"完成文件 {coord_file.name} 的处理，返回 {len(coords)} 行数据")

                return coords

            except Exception as e:
                logger.error(f"加载文件 {coord_file} 失败: {e}")
                # 返回一个空的DataFrame而不是None，这样合并时不会出错
                return pd.DataFrame()

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
    """更新的主函数，集成所有改进"""

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

    # 加载其他数据
    projection_data = loader.load_projection_data()
    tree_data = loader.load_tree_structure()

    # 并行加载MERFISH细胞数据
    merfish_cells = loader.load_merfish_cells_parallel()

    # Phase 2: 加载MERFISH层级数据
    logger.info("Phase 2: 加载MERFISH层级数据")
    builder = EnhancedKnowledgeGraphBuilder(output_path)

    if hierarchy_json:
        logger.info(f"从指定文件加载层级数据: {hierarchy_json}")
        builder.load_hierarchy(hierarchy_json)
    else:
        hierarchy_file = data_path / "hierarchy.json"
        if hierarchy_file.exists():
            logger.info(f"从默认位置加载层级数据: {hierarchy_file}")
            builder.load_hierarchy(hierarchy_file)
        else:
            logger.warning("找不到层级JSON文件，将尝试从细胞数据中提取")
            builder.extract_hierarchy_from_cells(merfish_cells)

    logger.info("成功加载MERFISH层级数据")

    # Phase 3: 计算层特异性形态数据
    logger.info("Phase 3: 计算层特异性形态数据")
    layer_calculator = LayerSpecificMorphologyCalculator(data_path)

    if layer_calculator.load_morphology_with_layers():
        regionlayer_data = layer_calculator.calculate_regionlayer_morphology(region_data)
    else:
        logger.error("无法加载形态学数据和层信息")
        regionlayer_data = pd.DataFrame()

    # Phase 4: 知识图谱生成
    logger.info("Phase 4: 知识图谱生成")

    # 生成节点
    logger.info("生成节点...")
    builder.generate_region_nodes(region_data)
    builder.generate_regionlayer_nodes(regionlayer_data, merfish_cells)
    builder.generate_merfish_nodes_from_hierarchy()

    # 生成关系
    logger.info("生成关系...")
    builder.generate_has_layer_relationships(region_data)

    # 处理没有层信息的区域
    builder.generate_relationships_for_layerless_regions(region_data, merfish_cells)

    # 生成HAS关系 (细胞类型分布)
    for level in ['class', 'subclass', 'supertype', 'cluster']:
        builder.generate_has_relationships_optimized(merfish_cells, level)

    # 生成其他关系
    builder.generate_belongs_to_from_hierarchy()

    # 生成所有层级的Dominant_transcriptomic关系
    builder.generate_dominant_transcriptomic_relationships(merfish_cells)

    # 投影关系
    builder.generate_project_to_relationships(projection_data)

    # 后处理
    logger.info("生成Neo4j导入脚本...")
    builder.generate_import_script()

    # 生成统计报告
    builder.generate_statistics_report(region_data, regionlayer_data, merfish_cells)

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
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./knowledge_graph',
                        help='输出目录路径')
    parser.add_argument('--hierarchy_json', type=str, default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json',
                        help='MERFISH层级JSON文件路径')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.hierarchy_json)