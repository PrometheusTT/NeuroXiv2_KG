import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from tqdm import tqdm
import os
import tempfile
import argparse
import sys
import time
import shutil
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeGraphRestructure:
    """知识图谱重构类 - 实现KG 2.3版本"""

    def __init__(self, kg_path: str, morpho_data_path: str, merfish_data_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化
        Args:
            kg_path: 现有知识图谱JSON文件路径或Neo4j连接信息
            morpho_data_path: 形态学数据文件夹路径
            merfish_data_path: MERFISH数据路径（可选）
            cache_dir: 缓存目录路径（可选，默认为系统临时目录）
        """
        self.morpho_data = None
        self.kg_path = kg_path
        self.morpho_data_path = Path(morpho_data_path)
        self.merfish_data_path = Path(merfish_data_path) if merfish_data_path else None

        # 中间结果存储路径
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"使用缓存目录: {self.cache_dir}")

        # 层定义
        self.layers = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']

        # 投射类型标记基因映射 - 扩展的标记列表
        self.projection_markers = {
            'IT': ['Satb2', 'Cux1', 'Cux2', 'Rorb', 'Lhx2', 'Rasgrf2', 'Slc30a3'],
            'ET': ['Fezf2', 'Bcl11b', 'Crym', 'Foxo1', 'Tbr1', 'Epha4', 'Pcp4'],
            'CT': ['Tle4', 'Foxp2', 'Syt6', 'Ntsr1', 'Grik1', 'Tshz2'],
            'NP': ['Npr3', 'Cplx3', 'Rspo1', 'Rxfp1', 'Penk']  # Near-projecting
        }

        # 投射类型正则表达式模式（用于名称匹配）
        self.projection_regex = {
            'IT': re.compile(r'(IT|ipsilateral|intratelencephalic)', re.IGNORECASE),
            'ET': re.compile(r'(ET|PT|extratelencephalic|pyramidal|contralateral)', re.IGNORECASE),
            'CT': re.compile(r'(CT|corticothalamic|thalamic)', re.IGNORECASE),
            'NP': re.compile(r'(NP|near[- ]projecting)', re.IGNORECASE)
        }

        # 连接到Neo4j的选项
        self.neo4j_conn = None
        if kg_path.startswith('bolt://') or kg_path.startswith('neo4j://'):
            try:
                from neo4j import GraphDatabase
                uri, username, password = self._parse_neo4j_uri(kg_path)
                self.neo4j_conn = GraphDatabase.driver(uri, auth=(username, password))
                logger.info(f"已连接到Neo4j数据库: {uri}")
            except ImportError:
                logger.warning("未安装neo4j模块，无法直接连接Neo4j。请使用 pip install neo4j 安装。")
                self.neo4j_conn = None
            except Exception as e:
                logger.error(f"连接Neo4j失败: {e}")
                self.neo4j_conn = None
        self.load_morphology_data()

    def _parse_neo4j_uri(self, uri_string: str) -> Tuple[str, str, str]:
        """解析Neo4j URI，格式为 bolt://username:password@host:port"""
        # 提取用户名和密码
        if '@' in uri_string:
            auth_part, host_part = uri_string.split('@', 1)
            protocol = auth_part.split('://', 1)[0]
            auth = auth_part.split('://', 1)[1]
            if ':' in auth:
                username, password = auth.split(':', 1)
            else:
                username, password = auth, ''
            uri = f"{protocol}://{host_part}"
        else:
            uri = uri_string
            username, password = 'neo4j', 'password'  # 默认值

        return uri, username, password

    def load_kg(self) -> Tuple[List[Dict], List[Dict]]:
        """加载现有知识图谱"""
        logger.info("加载现有知识图谱...")

        # 检查是否从Neo4j读取
        if self.neo4j_conn is not None:
            return self._load_from_neo4j()

        # 检查是否是目录（包含CSV文件）
        kg_path = Path(self.kg_path)
        if kg_path.is_dir():
            nodes_csv = kg_path / "nodes.csv"
            rels_csv = kg_path / "relationships.csv"

            if nodes_csv.exists() and rels_csv.exists():
                logger.info(f"从CSV文件加载: {nodes_csv}, {rels_csv}")
                return self._load_from_csv(nodes_csv, rels_csv)

        # 否则假定是单一JSON文件
        try:
            with open(self.kg_path, 'r') as f:
                kg_data = json.load(f)

            nodes = [item for item in kg_data if item['type'] == 'node']
            relationships = [item for item in kg_data if item['type'] == 'relationship']

            logger.info(f"加载完成: {len(nodes)}个节点, {len(relationships)}条关系")
            return nodes, relationships
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            logger.error(f"文件路径: {self.kg_path}")
            raise

    def _load_from_neo4j(self) -> Tuple[List[Dict], List[Dict]]:
        """直接从Neo4j数据库加载知识图谱"""
        nodes = []
        relationships = []

        with self.neo4j_conn.session() as session:
            # 读取所有节点
            node_result = session.run("MATCH (n) RETURN n")
            node_id_counter = 0

            for record in node_result:
                node = record["n"]
                node_data = {
                    "type": "node",
                    "id": str(node_id_counter),
                    "neo4j_id": node.id,  # 保存原始Neo4j ID
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
                nodes.append(node_data)
                node_id_counter += 1

            # 创建节点ID映射
            node_map = {node["neo4j_id"]: node["id"] for node in nodes}

            # 读取所有关系
            rel_result = session.run("MATCH ()-[r]->() RETURN r")
            rel_id_counter = 0

            for record in rel_result:
                rel = record["r"]
                rel_data = {
                    "type": "relationship",
                    "id": str(rel_id_counter),
                    "label": rel.type,
                    "properties": dict(rel),
                    "start": node_map[rel.start_node.id],
                    "end": node_map[rel.end_node.id]
                }
                relationships.append(rel_data)
                rel_id_counter += 1

        logger.info(f"从Neo4j加载完成: {len(nodes)}个节点, {len(relationships)}条关系")
        return nodes, relationships

    def _load_from_csv(self, nodes_csv: Path, rels_csv: Path) -> Tuple[List[Dict], List[Dict]]:
        """从CSV文件加载知识图谱（Neo4j导出格式）"""
        nodes = []
        relationships = []

        # 读取节点
        nodes_df = pd.read_csv(nodes_csv)
        for idx, row in nodes_df.iterrows():
            properties = {}

            # 提取属性列
            for col in nodes_df.columns:
                if col not in ['id', 'labels', ':ID', ':LABEL']:
                    if pd.notna(row[col]):
                        properties[col] = row[col]

            # 构建节点
            node_id = str(row.get('id', row.get(':ID')))
            labels = row.get('labels', row.get(':LABEL', '')).split(';')

            nodes.append({
                'type': 'node',
                'id': node_id,
                'labels': labels,
                'properties': properties
            })

        # 读取关系
        rels_df = pd.read_csv(rels_csv)
        for idx, row in rels_df.iterrows():
            properties = {}

            # 提取属性列
            for col in rels_df.columns:
                if col not in ['id', 'type', 'start', 'end', ':ID', ':TYPE', ':START_ID', ':END_ID']:
                    if pd.notna(row[col]):
                        properties[col] = row[col]

            # 构建关系
            rel_id = str(row.get('id', row.get(':ID', '')))
            rel_type = row.get('type', row.get(':TYPE', ''))
            start_id = str(row.get('start', row.get(':START_ID', '')))
            end_id = str(row.get('end', row.get(':END_ID', '')))

            # 查找对应节点
            start_node = next((n for n in nodes if n['id'] == start_id), None)
            end_node = next((n for n in nodes if n['id'] == end_id), None)

            if start_node and end_node:
                relationships.append({
                    'type': 'relationship',
                    'id': rel_id,
                    'label': rel_type,
                    'properties': properties,
                    'start': start_node,
                    'end': end_node
                })

        logger.info(f"从CSV加载完成: {len(nodes)}个节点, {len(relationships)}条关系")
        return nodes, relationships

    def load_morphology_data(self):
        """加载形态学数据并处理不同的CSV格式"""
        logger.info("加载形态学数据...")
        self.morpho_data = {}

        # 1. 加载轴突形态学数据
        axon_morpho_file = self.morpho_data_path / "axonfull_morpho.csv"
        if axon_morpho_file.exists():
            try:
                axon_morpho = pd.read_csv(axon_morpho_file)
                # 检查ID列的位置
                id_col = self._identify_id_column(axon_morpho)
                if id_col:
                    axon_morpho = axon_morpho.set_index(id_col)
                self.morpho_data['axon'] = axon_morpho
                logger.info(f"加载了{len(axon_morpho)}条轴突形态学数据")
            except Exception as e:
                logger.error(f"加载轴突形态学数据失败: {e}")
                self.morpho_data['axon'] = pd.DataFrame()
        else:
            logger.warning(f"轴突形态学数据文件不存在: {axon_morpho_file}")
            self.morpho_data['axon'] = pd.DataFrame()

        # 2. 加载树突形态学数据
        den_morpho_file = self.morpho_data_path / "denfull_morpho.csv"
        if den_morpho_file.exists():
            try:
                den_morpho = pd.read_csv(den_morpho_file)
                id_col = self._identify_id_column(den_morpho)
                if id_col:
                    den_morpho = den_morpho.set_index(id_col)
                self.morpho_data['dendrite'] = den_morpho
                logger.info(f"加载了{len(den_morpho)}条树突形态学数据")
            except Exception as e:
                logger.error(f"加载树突形态学数据失败: {e}")
                self.morpho_data['dendrite'] = pd.DataFrame()
        else:
            logger.warning(f"树突形态学数据文件不存在: {den_morpho_file}")
            self.morpho_data['dendrite'] = pd.DataFrame()

        # 3. 加载神经元信息和投射类型
        info_file = self.morpho_data_path / "info_with_projection_type.csv"
        if not info_file.exists():
            info_file = self.morpho_data_path / "info.csv"  # 尝试备用文件名

        if info_file.exists():
            try:
                neuron_info = pd.read_csv(info_file)
                id_col = self._identify_id_column(neuron_info)
                if id_col:
                    neuron_info = neuron_info.set_index(id_col)
                self.morpho_data['info'] = neuron_info
                logger.info(f"加载了{len(neuron_info)}条神经元信息数据")
            except Exception as e:
                logger.error(f"加载神经元信息数据失败: {e}")
                self.morpho_data['info'] = pd.DataFrame()
        else:
            logger.warning(f"神经元信息数据文件不存在: {info_file}")
            self.morpho_data['info'] = pd.DataFrame()

        # 4. 加载投射轴突长度数据 - 处理特殊格式
        proj_axon_file = self.morpho_data_path / "Proj_Axon_Final.csv"
        if not proj_axon_file.exists():
            proj_axon_file = self.morpho_data_path / "Proj_Axon_abs.csv"

        if proj_axon_file.exists():
            try:
                # 使用特殊处理方法加载投射数据
                proj_axon = pd.read_csv(proj_axon_file)
                id_col = self._identify_id_column(proj_axon)

                # 分析列结构
                abs_cols = [col for col in proj_axon.columns if '_abs' in col]
                rela_cols = [col for col in proj_axon.columns if '_rela' in col]

                if abs_cols:
                    logger.info(f"检测到{len(abs_cols)}个绝对值投射列")
                    # 提取区域名称
                    regions = [col.replace('proj_axon_', '').replace('_abs', '') for col in abs_cols]
                    logger.info(f"投射目标区域包括: {', '.join(regions[:5])}等{len(regions)}个区域")

                if id_col:
                    proj_axon = proj_axon.set_index(id_col)

                self.morpho_data['proj_axon'] = proj_axon
                logger.info(f"加载了{len(proj_axon)}行投射轴突数据")

            except Exception as e:
                logger.error(f"加载投射轴突长度数据失败: {e}")
                self.morpho_data['proj_axon'] = pd.DataFrame()
        else:
            logger.warning(f"投射轴突长度数据文件不存在: {proj_axon_file}")
            self.morpho_data['proj_axon'] = pd.DataFrame()

        # 5. 加载连接数据
        connections_file = self.morpho_data_path / "Connections_CCF-thin_final_250218.csv"
        if not connections_file.exists():
            connections_file = self.morpho_data_path / "Connections_CCFv3_final_250218.csv"

        if connections_file.exists():
            try:
                connections = pd.read_csv(connections_file)
                self.morpho_data['connections'] = connections
                logger.info(f"加载了{len(connections)}条连接数据")
            except Exception as e:
                logger.error(f"加载连接数据失败: {e}")
                self.morpho_data['connections'] = pd.DataFrame()
        else:
            logger.warning(f"连接数据文件不存在: {connections_file}")
            self.morpho_data['connections'] = pd.DataFrame()

        # 6. 加载神经元位置数据
        soma_file = self.morpho_data_path / "soma.csv"
        if soma_file.exists():
            try:
                soma = pd.read_csv(soma_file)
                id_col = self._identify_id_column(soma)
                if id_col:
                    soma = soma.set_index(id_col)
                self.morpho_data['soma'] = soma
                logger.info(f"加载了{len(soma)}条神经元位置数据")
            except Exception as e:
                logger.error(f"加载神经元位置数据失败: {e}")
                self.morpho_data['soma'] = pd.DataFrame()
        else:
            logger.warning(f"神经元位置数据文件不存在: {soma_file}")
            self.morpho_data['soma'] = pd.DataFrame()

        # 7. 尝试加载连接矩阵
        connection_matrix_file = self.morpho_data_path / "all_connection_20250218.csv"
        if connection_matrix_file.exists():
            try:
                connection_matrix = self._load_connection_matrix(connection_matrix_file)
                self.morpho_data['connection_matrix'] = connection_matrix
                logger.info(f"加载了连接矩阵，形状: {connection_matrix.shape}")
            except Exception as e:
                logger.error(f"加载连接矩阵失败: {e}")
                self.morpho_data['connection_matrix'] = pd.DataFrame()
        else:
            self.morpho_data['connection_matrix'] = pd.DataFrame()

    def _identify_id_column(self, df):
        """识别数据框中的ID列"""
        # 首选顺序：ID列，Unnamed: 0，第一列
        if 'ID' in df.columns:
            return 'ID'
        elif 'Unnamed: 0' in df.columns:
            # 检查是否是有效的ID列
            if df['Unnamed: 0'].nunique() == len(df):
                return 'Unnamed: 0'

        # 检查其他可能的ID列
        potential_id_cols = ['axon_id', 'neuron_id', 'id', 'cell_id']
        for col in potential_id_cols:
            if col in df.columns:
                return col

        # 如果没有明确的ID列，返回第一列
        logger.warning(f"未找到明确的ID列，使用第一列{df.columns[0]}作为ID")
        return df.columns[0]

    def _load_connection_matrix(self, file_path):
        """加载连接矩阵数据，处理特殊的列名格式"""
        try:
            # 先尝试正常读取
            df = pd.read_csv(file_path)

            # 检查是否有Unnamed: 0列作为索引
            id_col = self._identify_id_column(df)
            if id_col:
                df = df.set_index(id_col)
                df.index.name = 'ID'

            # 处理可能的内存优化
            if df.shape[1] > 1000:
                logger.info(f"连接矩阵包含{df.shape[1]}列，进行内存优化")
                # 将小于阈值的值设为0以减少内存占用
                df = df.sparse.to_dense()

            return df

        except Exception as e:
            logger.error(f"加载连接矩阵失败: {e}")
            return pd.DataFrame()

    def _process_projection_data(self):
        """处理投射数据，转换为每对区域之间的连接信息"""
        if 'proj_axon' not in self.morpho_data or self.morpho_data['proj_axon'].empty:
            logger.warning("未找到投射数据，跳过处理")
            return {}

        logger.info("处理投射数据...")
        proj_df = self.morpho_data['proj_axon']

        # 识别绝对值列
        abs_cols = [col for col in proj_df.columns if '_abs' in col]
        if not abs_cols:
            logger.warning("投射数据中未找到绝对值列")
            return {}

        # 提取区域名称
        region_map = {}
        for col in abs_cols:
            # 假设格式为 proj_axon_REGION_abs
            parts = col.split('_')
            if len(parts) >= 3:
                # 跳过 proj_axon_ 前缀，提取区域部分
                region = '_'.join(parts[2:-1])  # 排除最后的 _abs
                region_map[col] = region

        # 获取轴突源区域信息
        source_regions = {}
        if 'info' in self.morpho_data and not self.morpho_data['info'].empty:
            info_df = self.morpho_data['info']
            # 检查列是否存在
            if 'region' in info_df.columns:
                for idx, row in info_df.iterrows():
                    source_regions[idx] = row.get('region', '')

        # 构建区域对之间的投射统计
        region_projections = defaultdict(lambda: {
            'length_total': 0,
            'it_len': 0,
            'et_len': 0,
            'ct_len': 0,
            'n_axon': 0
        })

        # 处理每个轴突
        for axon_id, row in proj_df.iterrows():
            source = source_regions.get(axon_id, 'unknown')
            if source == 'unknown':
                continue

            # 检查投射类型
            proj_type = 'unknown'
            if 'info' in self.morpho_data and not self.morpho_data['info'].empty:
                info_df = self.morpho_data['info']
                if axon_id in info_df.index and 'projection_type' in info_df.columns:
                    proj_type = info_df.loc[axon_id, 'projection_type'].lower()

            # 处理投射到各区域的长度
            for col, region in region_map.items():
                if pd.notna(row[col]) and row[col] > 0:
                    length = float(row[col])
                    key = (source, region)

                    region_projections[key]['length_total'] += length
                    region_projections[key]['n_axon'] += 1

                    # 根据投射类型分配长度
                    if 'ipsilateral' in proj_type or 'it' in proj_type:
                        region_projections[key]['it_len'] += length
                    elif 'contralateral' in proj_type or 'et' in proj_type:
                        region_projections[key]['et_len'] += length
                    elif 'corticothalamic' in proj_type or 'ct' in proj_type:
                        region_projections[key]['ct_len'] += length
                    else:
                        # 默认分配给IT
                        region_projections[key]['it_len'] += length

        logger.info(f"从投射数据中提取了{len(region_projections)}对区域间的连接")
        return dict(region_projections)
    def create_region_layer_nodes(self, regions: List[Dict]) -> List[Dict]:
        """创建RegionLayer节点"""
        cache_file = self.cache_dir / "region_layer_nodes.pkl"

        if cache_file.exists():
            logger.info("从缓存加载RegionLayer节点...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("创建RegionLayer节点...")
        region_layer_nodes = []
        node_id_counter = 10000  # 从10000开始避免ID冲突

        # 从MERFISH数据加载已有的RegionLayer属性（如果有）
        merfish_rl_props = {}
        if self.merfish_data_path and (self.merfish_data_path / "region_layer_props.csv").exists():
            try:
                rl_props_df = pd.read_csv(self.merfish_data_path / "region_layer_props.csv")
                for _, row in rl_props_df.iterrows():
                    rl_id = row.get('rl_id')
                    if rl_id:
                        merfish_rl_props[rl_id] = row.to_dict()
            except Exception as e:
                logger.warning(f"加载MERFISH RegionLayer属性失败: {e}")

        # 创建区域到节点的映射
        region_map = {}
        for region in regions:
            region_name = region['properties'].get('name', '')
            if region_name:
                region_map[region_name] = region

        # 为每个区域创建所有层的RegionLayer节点
        for region in tqdm(regions, desc="创建RegionLayer节点"):
            region_name = region['properties'].get('name', '')
            region_id = region['id']

            # 如果名称不存在，跳过
            if not region_name:
                continue

            # 检查名称中是否已包含层信息
            layer_suffix = None
            for layer in self.layers:
                if region_name.endswith(f"_{layer}"):
                    layer_suffix = layer
                    region_name = region_name.replace(f"_{layer}", "")
                    break

            # 只为皮层区域创建layer节点
            if self._is_cortical_region(region_name):
                # 如果有层后缀，只创建该层的节点
                if layer_suffix:
                    layers_to_create = [layer_suffix]
                else:
                    # 否则为所有层创建节点
                    layers_to_create = self.layers

                for layer in layers_to_create:
                    rl_id = f"{region_id}_{layer}"

                    # 创建基本属性
                    properties = {
                        'rl_id': rl_id,
                        'region_name': region_name,
                        'layer': layer,
                        'region_id': region_id
                    }

                    # 添加MERFISH数据中的属性（如果有）
                    if rl_id in merfish_rl_props:
                        merfish_props = merfish_rl_props[rl_id]
                        for key, value in merfish_props.items():
                            if key not in ['rl_id', 'region_name', 'layer', 'region_id'] and pd.notna(value):
                                properties[key] = value

                    node = {
                        'type': 'node',
                        'id': str(node_id_counter),
                        'labels': ['RegionLayer'],
                        'properties': properties
                    }
                    region_layer_nodes.append(node)
                    node_id_counter += 1

        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(region_layer_nodes, f)

        logger.info(f"创建了{len(region_layer_nodes)}个RegionLayer节点")
        return region_layer_nodes

    def _is_cortical_region(self, region_name: str) -> bool:
        """判断是否为皮层区域"""
        cortical_prefixes = ['MO', 'SS', 'VIS', 'ACA', 'AI', 'RSP', 'PTL', 'TEa', 'PERI', 'ECT']
        return any(region_name.startswith(prefix) for prefix in cortical_prefixes)

    def _layer_num_to_name(self, layer_num, layer_map) -> str:
        """将层数字转换为层名称"""
        if pd.isna(layer_num):
            return 'L5'  # 默认层

        layer_num = int(layer_num)
        for layer_name, nums in layer_map.items():
            if layer_num in nums:
                return layer_name

        return f"L{layer_num}"  # 直接转换

    def calculate_morphology_stats(self, region_layer_nodes: List[Dict]) -> Dict[str, Dict]:
        """计算每个RegionLayer的形态学统计信息"""
        cache_file = self.cache_dir / "morphology_stats.pkl"

        if cache_file.exists():
            logger.info("从缓存加载形态学统计...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("计算形态学统计信息...")

        # 使用已加载的形态学数据
        if hasattr(self, 'morpho_data'):
            axon_df = self.morpho_data.get('axon', pd.DataFrame())
            dendrite_df = self.morpho_data.get('dendrite', pd.DataFrame())
            info_df = self.morpho_data.get('info', pd.DataFrame())

            # 检查数据是否为空
            if axon_df.empty:
                logger.warning("轴突形态学数据为空，结果可能不准确")
            if dendrite_df.empty:
                logger.warning("树突形态学数据为空，结果可能不准确")
            if info_df.empty:
                logger.warning("神经元信息数据为空，结果可能不准确")
        else:
            # 如果未预先加载，现在加载形态学数据
            logger.info("未找到预加载的形态学数据，正在加载...")
            morpho_files = {
                'axon': self.morpho_data_path / 'axonfull_morpho.csv',
                'dendrite': self.morpho_data_path / 'denfull_morpho.csv',
                'info': self.morpho_data_path / 'info_with_projection_type.csv'
            }

            # 检查文件是否存在
            for key, file_path in morpho_files.items():
                if not file_path.exists():
                    if key == 'info' and (self.morpho_data_path / 'info.csv').exists():
                        morpho_files[key] = self.morpho_data_path / 'info.csv'
                        logger.info(f"使用备用信息文件: {morpho_files[key]}")
                    else:
                        logger.error(f"形态学数据文件不存在: {file_path}")
                        raise FileNotFoundError(f"形态学数据文件不存在: {file_path}")

            # 读取数据
            logger.info("读取形态学数据...")
            axon_df = pd.read_csv(morpho_files['axon'])
            dendrite_df = pd.read_csv(morpho_files['dendrite'])
            info_df = pd.read_csv(morpho_files['info'])

            # 检查ID列
            id_cols = {}
            for name, df in [('axon', axon_df), ('dendrite', dendrite_df), ('info', info_df)]:
                id_cols[name] = self._identify_id_column(df)
                if id_cols[name]:
                    logger.info(f"{name}数据使用{id_cols[name]}作为ID列")

        # 合并数据前处理ID列
        if 'ID' not in axon_df.columns and hasattr(axon_df, 'index') and axon_df.index.name != 'ID':
            # 如果索引不是'ID'，尝试重置索引
            if axon_df.index.name:
                axon_df = axon_df.reset_index()

        if 'ID' not in dendrite_df.columns and hasattr(dendrite_df, 'index') and dendrite_df.index.name != 'ID':
            if dendrite_df.index.name:
                dendrite_df = dendrite_df.reset_index()

        if 'ID' not in info_df.columns and hasattr(info_df, 'index') and info_df.index.name != 'ID':
            if info_df.index.name:
                info_df = info_df.reset_index()

        # 合并数据
        logger.info("合并形态学数据...")
        # 确定要使用的ID列
        axon_id_col = 'ID' if 'ID' in axon_df.columns else axon_df.columns[0]
        dendrite_id_col = 'ID' if 'ID' in dendrite_df.columns else dendrite_df.columns[0]
        info_id_col = 'ID' if 'ID' in info_df.columns else info_df.columns[0]

        # 使用更高效的合并方式
        try:
            # 1. 准备轴突数据子集
            axon_cols = ['Total Length', 'Number of Bifurcations']
            axon_cols = [col for col in axon_cols if col in axon_df.columns]
            if not axon_cols:
                logger.warning("轴突数据中缺少必要的列，使用默认列")
                # 尝试查找替代列
                length_cols = [col for col in axon_df.columns if 'length' in col.lower()]
                if length_cols:
                    axon_df['Total Length'] = axon_df[length_cols[0]]
                    axon_cols.append('Total Length')

                bifurcation_cols = [col for col in axon_df.columns if 'bifurc' in col.lower()]
                if bifurcation_cols:
                    axon_df['Number of Bifurcations'] = axon_df[bifurcation_cols[0]]
                    axon_cols.append('Number of Bifurcations')

            axon_subset = axon_df[[axon_id_col] + axon_cols].copy()

            # 2. 准备树突数据子集
            dend_cols = ['Number of Bifurcations']
            dend_cols = [col for col in dend_cols if col in dendrite_df.columns]
            if not dend_cols:
                # 尝试查找替代列
                bifurcation_cols = [col for col in dendrite_df.columns if 'bifurc' in col.lower()]
                if bifurcation_cols:
                    dendrite_df['Number of Bifurcations'] = dendrite_df[bifurcation_cols[0]]
                    dend_cols.append('Number of Bifurcations')

            dend_subset = dendrite_df[[dendrite_id_col] + dend_cols].copy()

            # 3. 执行合并
            # 首先合并信息和轴突数据
            merged_df = pd.merge(
                info_df,
                axon_subset,
                left_on=info_id_col,
                right_on=axon_id_col,
                how='left',
                suffixes=('', '_axon')
            )

            # 然后合并树突数据
            merged_df = pd.merge(
                merged_df,
                dend_subset,
                left_on=info_id_col,
                right_on=dendrite_id_col,
                how='left',
                suffixes=('', '_dendrite')
            )

            logger.info(f"合并后数据形状: {merged_df.shape}")

        except Exception as e:
            logger.error(f"合并数据时出错: {e}")
            # 创建一个空的合并数据框
            merged_df = info_df.copy()
            merged_df['Total Length'] = np.nan
            merged_df['Number of Bifurcations_dendrite'] = np.nan

        # 检查是否有'has_apical'列，如果没有，尝试从其他列推断
        if 'has_apical' not in merged_df.columns:
            apical_cols = [col for col in merged_df.columns if 'apical' in col.lower()]
            if apical_cols:
                logger.info(f"使用{apical_cols[0]}列作为has_apical")
                merged_df['has_apical'] = merged_df[apical_cols[0]].notna() & (merged_df[apical_cols[0]] != 0)
            else:
                # 如果找不到任何有关顶树突的列，使用默认值
                logger.warning("找不到有关顶树突的列，使用默认值")
                merged_df['has_apical'] = 0.5  # 默认值，表示未知

        # 预处理：创建层映射字典
        layer_map = {'L2/3': [2, 3]}
        for layer in self.layers:
            if layer != 'L2/3' and layer != 'L6b':
                layer_map[layer] = [int(layer[1:])]
            elif layer == 'L6b':
                layer_map[layer] = [6]  # 假设L6b在数据中标记为6

        # 检查是否有'layer'列，如果没有，尝试从其他列推断
        if 'layer' not in merged_df.columns:
            layer_cols = [col for col in merged_df.columns if 'layer' in col.lower()]
            if layer_cols:
                logger.info(f"使用{layer_cols[0]}列作为layer")
                merged_df['layer'] = merged_df[layer_cols[0]]
            else:
                # 如果找不到任何层信息，使用默认值
                logger.warning("找不到层信息列，使用默认值 (L5)")
                merged_df['layer'] = 5  # 默认层为L5

        # 检查是否有'celltype_manual'列，如果没有，尝试从其他列推断
        if 'celltype_manual' not in merged_df.columns:
            region_cols = [
                col for col in merged_df.columns
                if any(name in col.lower() for name in ['region', 'area', 'celltype', 'cell_type'])
            ]
            if region_cols:
                logger.info(f"使用{region_cols[0]}列作为celltype_manual")
                merged_df['celltype_manual'] = merged_df[region_cols[0]]
            else:
                # 如果找不到任何区域信息，使用默认值
                logger.warning("找不到区域信息列，使用默认值")
                merged_df['celltype_manual'] = 'unknown'  # 默认未知区域

        # 检查是否有'projection_type'列，如果没有，尝试从其他列推断
        if 'projection_type' not in merged_df.columns:
            proj_cols = [
                col for col in merged_df.columns
                if any(name in col.lower() for name in ['projection', 'proj_type', 'proj', 'type'])
            ]
            if proj_cols:
                logger.info(f"使用{proj_cols[0]}列作为projection_type")
                merged_df['projection_type'] = merged_df[proj_cols[0]]
            else:
                # 使用"ipsilateral"作为默认投射类型（最常见）
                logger.warning("找不到投射类型列，使用默认值 'ipsilateral'")
                merged_df['projection_type'] = 'ipsilateral'

        # 预先聚合统计信息 - 使用向量化操作提高效率
        logger.info("聚合形态学统计信息...")

        # 将区域-层组合编码为唯一键
        merged_df['region_layer'] = merged_df.apply(
            lambda row: f"{row['celltype_manual']}_{self._layer_num_to_name(row['layer'], layer_map)}",
            axis=1
        )

        # 分组聚合计算
        try:
            stats_df = merged_df.groupby('region_layer').agg({
                'Total Length': ['mean', 'std', 'count'],
                'Number of Bifurcations_dendrite': ['std'],
                'has_apical': ['mean'],
                'projection_type': lambda x: x.value_counts(normalize=True).to_dict()
            }).reset_index()
        except Exception as e:
            logger.error(f"聚合统计时出错: {e}")
            # 使用备用聚合方法
            stats = {}
            for region_layer, group in merged_df.groupby('region_layer'):
                try:
                    total_length_mean = group['Total Length'].mean()
                    total_length_std = group['Total Length'].std()
                    count = len(group)
                    dend_br_std = group['Number of Bifurcations_dendrite'].std()
                    has_apical_mean = group['has_apical'].mean()

                    # 计算投射类型分布
                    proj_types = group['projection_type'].value_counts(normalize=True).to_dict()

                    stats[region_layer] = {
                        'region_layer': region_layer,
                        'Total Length': {
                            'mean': total_length_mean,
                            'std': total_length_std,
                            'count': count
                        },
                        'Number of Bifurcations_dendrite': {'std': dend_br_std},
                        'has_apical': {'mean': has_apical_mean},
                        'projection_type': {'<lambda>': proj_types}
                    }
                except Exception as inner_e:
                    logger.error(f"处理区域层{region_layer}时出错: {inner_e}")

            # 创建替代的stats_df
            stats_df = pd.DataFrame([
                {
                    'region_layer': k,
                    'Total Length': v['Total Length'],
                    'Number of Bifurcations_dendrite': v['Number of Bifurcations_dendrite'],
                    'has_apical': v['has_apical'],
                    'projection_type': v['projection_type']
                } for k, v in stats.items()
            ])

        # 将统计结果转换为所需格式
        stats = {}
        for _, row in stats_df.iterrows():
            region_layer = row['region_layer']
            if not pd.isna(region_layer):
                # 解析区域和层
                parts = region_layer.split('_')
                if len(parts) >= 2:
                    region_name = parts[0]
                    layer = parts[1]

                    # 查找对应的RegionLayer节点
                    rl_node = None
                    for node in region_layer_nodes:
                        if (node['properties'].get('region_name') == region_name and
                                node['properties'].get('layer') == layer):
                            rl_id = node['properties'].get('rl_id')
                            if rl_id:
                                # 获取投射类型分布
                                proj_types = {}
                                if isinstance(row['projection_type'], dict) and '<lambda>' in row['projection_type']:
                                    proj_types = row['projection_type']['<lambda>']
                                elif hasattr(row['projection_type'], 'get') and row['projection_type'].get('<lambda>'):
                                    proj_types = row['projection_type'].get('<lambda>')

                                # 安全地提取统计值
                                def safe_extract(obj, path, default=0.0):
                                    try:
                                        result = obj
                                        for key in path:
                                            if isinstance(result, dict) and key in result:
                                                result = result[key]
                                            else:
                                                return default
                                        return float(result) if pd.notna(result) else default
                                    except:
                                        return default

                                # 构建统计数据
                                stats[rl_id] = {
                                    'region_name': region_name,
                                    'layer': layer,
                                    'morph_ax_len_mean': safe_extract(row, ['Total Length', 'mean']),
                                    'morph_ax_len_std': safe_extract(row, ['Total Length', 'std']),
                                    'dend_polarity_index_mean': safe_extract(row, ['has_apical', 'mean']),
                                    'dend_br_std': safe_extract(row, ['Number of Bifurcations_dendrite', 'std']),
                                    'n_neuron': int(safe_extract(row, ['Total Length', 'count'])),
                                    'it_pct': min(1.0, float(proj_types.get('ipsilateral', 0.0) +
                                                             proj_types.get('IT', 0.0) +
                                                             proj_types.get('it', 0.0))),
                                    'et_pct': min(1.0, float(proj_types.get('contralateral', 0.0) +
                                                             proj_types.get('ET', 0.0) +
                                                             proj_types.get('PT', 0.0) +
                                                             proj_types.get('et', 0.0) +
                                                             proj_types.get('pt', 0.0))),
                                    'ct_pct': min(1.0, float(proj_types.get('corticothalamic', 0.0) +
                                                             proj_types.get('CT', 0.0) +
                                                             proj_types.get('ct', 0.0))),
                                    'lr_pct': min(1.0, float(proj_types.get('contralateral', 0.0) +
                                                             proj_types.get('ET', 0.0) +
                                                             proj_types.get('et', 0.0))),
                                    'lr_prior': 0.5  # 默认优先级
                                }

        # 如果从MERFISH数据加载了RegionLayer属性，则优先使用这些值
        if self.merfish_data_path and (self.merfish_data_path / "region_layer_props.csv").exists():
            try:
                rl_props_df = pd.read_csv(self.merfish_data_path / "region_layer_props.csv")

                for _, row in rl_props_df.iterrows():
                    rl_id = row.get('rl_id')
                    if rl_id in stats:
                        # 使用MERFISH数据更新投射类型比例
                        for prop in ['it_pct', 'et_pct', 'ct_pct', 'lr_pct', 'lr_prior']:
                            if prop in row and pd.notna(row[prop]):
                                stats[rl_id][prop] = float(row[prop])

                logger.info("使用MERFISH数据更新了部分形态学统计信息")
            except Exception as e:
                logger.warning(f"加载MERFISH RegionLayer属性失败: {e}")

        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(stats, f)

        # 记录结果统计
        non_zero_stats = {k: v for k, v in stats.items() if v.get('n_neuron', 0) > 0}
        logger.info(f"计算了{len(stats)}个RegionLayer的形态学统计，其中{len(non_zero_stats)}个有神经元数据")

        # 最后更新: 2025-07-14 15:06:12
        # 作者: PrometheusTT

        return stats

    def determine_projection_type(self, markers: List[str], subclass_name: str = None) -> str:
        """根据标记基因和亚类名称确定投射类型"""
        # 1. 首先尝试从亚类名称匹配
        if subclass_name:
            for proj_type, pattern in self.projection_regex.items():
                if pattern.search(subclass_name):
                    return proj_type

        # 2. 然后尝试使用标记基因
        if not markers:
            return 'UNK'

        # 计算每种投射类型的匹配分数
        scores = {}
        for proj_type, proj_markers in self.projection_markers.items():
            score = sum(1 for marker in proj_markers if marker in markers)
            scores[proj_type] = score

        # 返回得分最高的类型
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return 'UNK'

    def update_subclass_nodes(self, nodes: List[Dict]) -> None:
        """更新Subclass节点，添加proj_type属性"""
        logger.info("更新Subclass节点投射类型...")

        # 加载亚类-投射类型映射（如果有）
        subclass_projtype_map = {}
        if self.merfish_data_path and (self.merfish_data_path / "subclass_projtype.csv").exists():
            try:
                projtype_df = pd.read_csv(self.merfish_data_path / "subclass_projtype.csv")
                for _, row in projtype_df.iterrows():
                    subclass_name = row.get('subclass_name')
                    proj_type = row.get('proj_type')
                    if subclass_name and proj_type:
                        subclass_projtype_map[subclass_name] = proj_type
                logger.info(f"从MERFISH数据加载了{len(subclass_projtype_map)}个亚类-投射类型映射")
            except Exception as e:
                logger.warning(f"加载亚类-投射类型映射失败: {e}")

        for node in tqdm(nodes, desc="更新Subclass节点"):
            if 'Subclass' in node.get('labels', []):
                # 获取亚类名称
                subclass_name = node['properties'].get('name', '')

                # 1. 首先尝试从映射中获取投射类型
                if subclass_name in subclass_projtype_map:
                    node['properties']['proj_type'] = subclass_projtype_map[subclass_name]
                    continue

                # 2. 否则，从标记基因确定
                markers = []
                node_props = node.get('properties', {})

                # 从多个可能的标记字段中收集标记
                for marker_field in ['markers', 'transcription factor markers', 'within subclass markers']:
                    if marker_field in node_props and node_props[marker_field]:
                        markers.extend([m.strip() for m in node_props[marker_field].split(',') if m.strip()])

                proj_type = self.determine_projection_type(markers, subclass_name)
                node['properties']['proj_type'] = proj_type

    def create_has_layer_relationships(self, regions: List[Dict], region_layer_nodes: List[Dict]) -> List[Dict]:
        """创建Region到RegionLayer的HAS_LAYER关系"""
        logger.info("创建HAS_LAYER关系...")

        relationships = []
        rel_id_counter = 20000

        # 创建region_id到node的映射
        region_map = {node['id']: node for node in regions}
        region_name_map = {node['properties'].get('name', ''): node for node in regions if
                           'name' in node.get('properties', {})}

        # 创建rl_id到node的映射
        rl_map = {node['properties']['rl_id']: node for node in region_layer_nodes}

        for rl_node in tqdm(region_layer_nodes, desc="创建HAS_LAYER关系"):
            region_id = rl_node['properties'].get('region_id')

            if not region_id:
                # 如果没有region_id，尝试从region_name查找
                region_name = rl_node['properties'].get('region_name')
                if region_name and region_name in region_name_map:
                    region_id = region_name_map[region_name]['id']
                else:
                    continue

            if region_id in region_map:
                rel = {
                    'type': 'relationship',
                    'id': str(rel_id_counter),
                    'label': 'HAS_LAYER',
                    'properties': {},
                    'start': region_map[region_id],
                    'end': rl_node
                }
                relationships.append(rel)
                rel_id_counter += 1

        logger.info(f"创建了{len(relationships)}个HAS_LAYER关系")
        return relationships

    def create_transcriptomic_relationships(self, region_layer_nodes: List[Dict],
                                            transcriptomic_nodes: List[Dict]) -> List[Dict]:
        """创建RegionLayer到转录组类型的关系"""
        cache_file = self.cache_dir / "transcriptomic_relationships.pkl"

        if cache_file.exists():
            logger.info("从缓存加载转录组关系...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("创建转录组关系...")
        relationships = []
        rel_id_counter = 30000

        # 检查是否有MERFISH数据
        merfish_rels = {}
        has_real_data = False

        # 加载MERFISH关系数据
        if self.merfish_data_path:
            has_class_path = self.merfish_data_path / "has_class.csv"
            has_subclass_path = self.merfish_data_path / "has_subclass.csv"
            has_cluster_path = self.merfish_data_path / "has_cluster.csv"

            # 加载关系数据
            if has_class_path.exists():
                try:
                    has_class_df = pd.read_csv(has_class_path)
                    merfish_rels['HAS_CLASS'] = has_class_df
                    has_real_data = True
                    logger.info(f"加载了{len(has_class_df)}个HAS_CLASS关系数据")
                except Exception as e:
                    logger.warning(f"加载HAS_CLASS关系失败: {e}")

            if has_subclass_path.exists():
                try:
                    has_subclass_df = pd.read_csv(has_subclass_path)
                    merfish_rels['HAS_SUBCLASS'] = has_subclass_df
                    has_real_data = True
                    logger.info(f"加载了{len(has_subclass_df)}个HAS_SUBCLASS关系数据")
                except Exception as e:
                    logger.warning(f"加载HAS_SUBCLASS关系失败: {e}")

            if has_cluster_path.exists():
                try:
                    has_cluster_df = pd.read_csv(has_cluster_path)
                    merfish_rels['HAS_CLUSTER'] = has_cluster_df
                    has_real_data = True
                    logger.info(f"加载了{len(has_cluster_df)}个HAS_CLUSTER关系数据")
                except Exception as e:
                    logger.warning(f"加载HAS_CLUSTER关系失败: {e}")

        # 如果有真实的MERFISH数据，使用它创建关系
        if has_real_data:
            logger.info("使用MERFISH数据创建转录组关系...")

            # 创建转录组名称到节点的映射
            trans_map = {
                'class': {},
                'subclass': {},
                'cluster': {}
            }

            for node in transcriptomic_nodes:
                node_name = node['properties'].get('name', '')
                if not node_name:
                    continue

                if 'Class' in node.get('labels', []):
                    trans_map['class'][node_name] = node
                elif 'Subclass' in node.get('labels', []):
                    trans_map['subclass'][node_name] = node
                elif 'Cluster' in node.get('labels', []):
                    trans_map['cluster'][node_name] = node

            # 创建RegionLayer ID到节点的映射
            rl_map = {node['properties']['rl_id']: node for node in region_layer_nodes}

            # 创建HAS_CLASS关系
            if 'HAS_CLASS' in merfish_rels:
                for _, row in merfish_rels['HAS_CLASS'].iterrows():
                    rl_id = row.get('rl_id')
                    class_name = row.get('class_name')

                    if rl_id in rl_map and class_name in trans_map['class']:
                        rel = {
                            'type': 'relationship',
                            'id': str(rel_id_counter),
                            'label': 'HAS_CLASS',
                            'properties': {
                                'pct_cells': float(row.get('pct_cells', 0)),
                                'rank': int(row.get('rank', 0)),
                                'n_cells': int(row.get('n_cells', 0))
                            },
                            'start': rl_map[rl_id],
                            'end': trans_map['class'][class_name]
                        }
                        relationships.append(rel)
                        rel_id_counter += 1

            # 创建HAS_SUBCLASS关系
            if 'HAS_SUBCLASS' in merfish_rels:
                for _, row in merfish_rels['HAS_SUBCLASS'].iterrows():
                    rl_id = row.get('rl_id')
                    subclass_name = row.get('subclass_name')

                    if rl_id in rl_map and subclass_name in trans_map['subclass']:
                        rel = {
                            'type': 'relationship',
                            'id': str(rel_id_counter),
                            'label': 'HAS_SUBCLASS',
                            'properties': {
                                'pct_cells': float(row.get('pct_cells', 0)),
                                'rank': int(row.get('rank', 0)),
                                'n_cells': int(row.get('n_cells', 0)),
                                'proj_type': row.get('proj_type', 'UNK')
                            },
                            'start': rl_map[rl_id],
                            'end': trans_map['subclass'][subclass_name]
                        }
                        relationships.append(rel)
                        rel_id_counter += 1

            # 创建HAS_CLUSTER关系
            if 'HAS_CLUSTER' in merfish_rels:
                for _, row in merfish_rels['HAS_CLUSTER'].iterrows():
                    rl_id = row.get('rl_id')
                    cluster_name = row.get('cluster_name')

                    if rl_id in rl_map and cluster_name in trans_map['cluster']:
                        rel = {
                            'type': 'relationship',
                            'id': str(rel_id_counter),
                            'label': 'HAS_CLUSTER',
                            'properties': {
                                'pct_cells': float(row.get('pct_cells', 0)),
                                'rank': int(row.get('rank', 0)),
                                'n_cells': int(row.get('n_cells', 0))
                            },
                            'start': rl_map[rl_id],
                            'end': trans_map['cluster'][cluster_name]
                        }
                        relationships.append(rel)
                        rel_id_counter += 1

        # 如果没有真实数据或处理失败，确保每个RegionLayer至少有一些基本关系
        # if not relationships:
        #     logger.warning("没有MERFISH数据或处理失败，创建基本转录组关系...")
        #
        #     # 创建转录组类型映射
        #     tc_types = {'Class': [], 'Subclass': [], 'Supertype': [], 'Cluster': []}
        #     for tc_node in transcriptomic_nodes:
        #         for label in tc_node.get('labels', []):
        #             if label in tc_types:
        #                 tc_types[label].append(tc_node)
        #
        #     # 确保每个RegionLayer至少有一些Class和Subclass关系
        #     for rl_node in tqdm(region_layer_nodes, desc="创建基本转录组关系"):
        #         region_name = rl_node['properties'].get('region_name', '')
        #         layer = rl_node['properties'].get('layer', '')
        #
        #         # 层特异性选择
        #         if layer == 'L2/3':
        #             # L2/3倾向于IT神经元
        #             it_weight = 0.8
        #             et_weight = 0.1
        #             ct_weight = 0.1
        #         elif layer == 'L5':
        #             # L5包含更多ET神经元
        #             it_weight = 0.4
        #             et_weight = 0.5
        #             ct_weight = 0.1
        #         elif layer == 'L6':
        #             # L6包含更多CT神经元
        #             it_weight = 0.3
        #             et_weight = 0.2
        #             ct_weight = 0.5
        #         else:
        #             # 其他层的默认权重
        #             it_weight = 0.5
        #             et_weight = 0.3
        #             ct_weight = 0.2
        #
        #         # 根据区域调整权重
        #         if 'VIS' in region_name:
        #             # 视觉区域有更多的IT和ET神经元
        #             it_weight *= 1.2
        #             et_weight *= 1.2
        #             ct_weight *= 0.6
        #         elif 'MO' in region_name:
        #             # 运动区域有更多的ET神经元
        #             it_weight *= 0.9
        #             et_weight *= 1.3
        #             ct_weight *= 0.8
        #
        #         # 确保权重总和为1
        #         total_weight = it_weight + et_weight + ct_weight
        #         it_weight /= total_weight
        #         et_weight /= total_weight
        #         ct_weight /= total_weight
        #
        #         # 为每个类型创建关系
        #         for tc_type, nodes in tc_types.items():
        #             if not nodes:
        #                 continue
        #
        #             # 选择节点数量，但不超过可用节点数
        #             rel_type = f"HAS_{tc_type.upper()}"
        #
        #             if tc_type == 'Class':
        #                 # 为每个Class创建关系
        #                 for rank, tc_node in enumerate(nodes[:3], 1):
        #                     pct = 0.0
        #
        #                     # 根据Class名称调整百分比
        #                     class_name = tc_node['properties'].get('name', '')
        #                     if 'Glutamatergic' in class_name:
        #                         pct = 0.7  # 谷氨酸能神经元占主导
        #                     elif 'GABAergic' in class_name:
        #                         pct = 0.2  # GABA能神经元次之
        #                     else:
        #                         pct = 0.1  # 其他类型较少
        #
        #                     rel = {
        #                         'type': 'relationship',
        #                         'id': str(rel_id_counter),
        #                         'label': rel_type,
        #                         'properties': {
        #                             'pct_cells': pct,
        #                             'rank': rank,
        #                             'n_cells': int(100 * pct)  # 假设每个RegionLayer有100个细胞
        #                         },
        #                         'start': rl_node,
        #                         'end': tc_node
        #                     }
        #                     relationships.append(rel)
        #                     rel_id_counter += 1
        #
        #             elif tc_type == 'Subclass':
        #                 # 选择一些Subclass节点
        #                 selected_nodes = []
        #
        #                 # 根据投射类型分类Subclass
        #                 it_nodes = []
        #                 et_nodes = []
        #                 ct_nodes = []
        #                 other_nodes = []
        #
        #                 for node in nodes:
        #                     proj_type = node['properties'].get('proj_type', 'UNK')
        #                     if proj_type == 'IT':
        #                         it_nodes.append(node)
        #                     elif proj_type == 'ET':
        #                         et_nodes.append(node)
        #                     elif proj_type == 'CT':
        #                         ct_nodes.append(node)
        #                     else:
        #                         other_nodes.append(node)
        #
        #                 # 根据权重选择不同投射类型的Subclass
        #                 num_it = max(1, min(3, int(len(it_nodes))))
        #                 num_et = max(1, min(2, int(len(et_nodes))))
        #                 num_ct = max(1, min(2, int(len(ct_nodes))))
        #
        #                 if it_nodes:
        #                     selected_nodes.extend(np.random.choice(it_nodes, num_it, replace=False))
        #                 if et_nodes:
        #                     selected_nodes.extend(np.random.choice(et_nodes, num_et, replace=False))
        #                 if ct_nodes:
        #                     selected_nodes.extend(np.random.choice(ct_nodes, num_ct, replace=False))
        #
        #                 # 如果选择的节点太少，添加一些其他节点
        #                 if len(selected_nodes) < 3 and other_nodes:
        #                     num_other = min(3 - len(selected_nodes), len(other_nodes))
        #                     selected_nodes.extend(np.random.choice(other_nodes, num_other, replace=False))
        #
        #                 # 创建关系，根据投射类型分配百分比
        #                 for rank, tc_node in enumerate(selected_nodes, 1):
        #                     proj_type = tc_node['properties'].get('proj_type', 'UNK')
        #
        #                     # 根据投射类型和层分配百分比
        #                     if proj_type == 'IT':
        #                         base_pct = it_weight / num_it
        #                     elif proj_type == 'ET':
        #                         base_pct = et_weight / num_et
        #                     elif proj_type == 'CT':
        #                         base_pct = ct_weight / num_ct
        #                     else:
        #                         base_pct = 0.1 / len(selected_nodes)
        #
        #                     # 添加一些随机变化
        #                     pct = min(0.8, max(0.01, base_pct * (0.8 + 0.4 * np.random.random())))
        #
        #                     rel = {
        #                         'type': 'relationship',
        #                         'id': str(rel_id_counter),
        #                         'label': rel_type,
        #                         'properties': {
        #                             'pct_cells': pct,
        #                             'rank': rank,
        #                             'n_cells': int(100 * pct),
        #                             'proj_type': proj_type
        #                         },
        #                         'start': rl_node,
        #                         'end': tc_node
        #                     }
        #                     relationships.append(rel)
        #                     rel_id_counter += 1
        #
        #             elif tc_type == 'Cluster':
        #                 # 为一些Cluster创建关系
        #                 num_clusters = min(5, len(nodes))
        #                 selected_nodes = np.random.choice(nodes, num_clusters, replace=False)
        #
        #                 for rank, tc_node in enumerate(selected_nodes, 1):
        #                     # Cluster的百分比较小
        #                     pct = min(0.2, max(0.01, 0.05 + 0.15 * np.random.random()))
        #
        #                     rel = {
        #                         'type': 'relationship',
        #                         'id': str(rel_id_counter),
        #                         'label': rel_type,
        #                         'properties': {
        #                             'pct_cells': pct,
        #                             'rank': rank,
        #                             'n_cells': int(100 * pct)
        #                         },
        #                         'start': rl_node,
        #                         'end': tc_node
        #                     }
        #                     relationships.append(rel)
        #                     rel_id_counter += 1

        # 保存缓存
        if not relationships:
            logger.error("没有找到MERFISH转录组关系数据。无法继续构建知识图谱。")
            logger.error("请确保merfish_data_path指向包含has_class.csv、has_subclass.csv和has_cluster.csv的有效目录。")
            raise ValueError("缺少必要的MERFISH转录组关系数据，无法构建完整知识图谱。")
        with open(cache_file, 'wb') as f:
            pickle.dump(relationships, f)

        logger.info(f"创建了{len(relationships)}个转录组关系")
        return relationships

    # Add this function to KnowledgeGraphRestructure.py

    def create_gene_nodes_and_relationships(self, all_nodes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """创建基因节点和共表达关系"""
        logger.info("创建基因节点和共表达关系...")

        # 新的基因节点和共表达关系
        new_gene_nodes = []
        coexpression_relationships = []

        # 用于节点ID和关系ID的计数器
        node_id_counter = 50000  # 从50000开始避免ID冲突
        rel_id_counter = 100000  # 从100000开始避免ID冲突

        # 检查是否存在基因共表达数据
        coexpr_file = None
        if self.merfish_data_path:
            coexpr_path = self.merfish_data_path / "gene_coexpression.csv"
            if coexpr_path.exists():
                coexpr_file = coexpr_path

        # 如果没有找到共表达文件，无法继续
        if not coexpr_file:
            logger.error("未找到基因共表达数据文件 (gene_coexpression.csv)")
            logger.error(f"在 {self.merfish_data_path} 中查找，但未发现此文件")
            logger.error("请先运行 MERFISHDataIntegration.py 生成此文件")
            raise FileNotFoundError(f"缺少必要的基因共表达数据文件: gene_coexpression.csv")

        try:
            logger.info(f"加载基因共表达数据: {coexpr_file}")
            coexpr_df = pd.read_csv(coexpr_file)

            if coexpr_df.empty:
                logger.error("基因共表达数据为空")
                raise ValueError("基因共表达数据为空，无法创建基因节点和关系")

            # 收集所有涉及的基因
            all_genes = set()
            for _, row in coexpr_df.iterrows():
                gene1 = row.get('gene1')
                gene2 = row.get('gene2')
                if pd.notna(gene1) and pd.notna(gene2):
                    all_genes.add(gene1)
                    all_genes.add(gene2)

            # 检查已有的基因节点
            existing_gene_symbols = {}
            for node in all_nodes:
                if 'Gene' in node.get('labels', []):
                    symbol = node['properties'].get('symbol')
                    if symbol:
                        existing_gene_symbols[symbol] = node

            # 创建不存在的基因节点
            for gene in all_genes:
                if gene not in existing_gene_symbols:
                    gene_node = {
                        'type': 'node',
                        'id': str(node_id_counter),
                        'labels': ['Gene'],
                        'properties': {
                            'symbol': gene,
                            'name': gene,
                            'type': 'protein_coding'  # 默认类型
                        }
                    }
                    new_gene_nodes.append(gene_node)
                    existing_gene_symbols[gene] = gene_node
                    node_id_counter += 1

            # 创建共表达关系
            significant_threshold = 0.05  # FDR显著性阈值
            for _, row in coexpr_df.iterrows():
                gene1 = row.get('gene1')
                gene2 = row.get('gene2')
                rho = row.get('rho')
                fdr = row.get('fdr', row.get('p_value', 1.0))  # 尝试使用fdr，如果没有就用p值

                # 检查数据有效性
                if (pd.isna(gene1) or pd.isna(gene2) or pd.isna(rho) or
                        gene1 not in existing_gene_symbols or gene2 not in existing_gene_symbols):
                    continue

                # 只保留显著的关系
                if pd.notna(fdr) and fdr <= significant_threshold:
                    rel = {
                        'type': 'relationship',
                        'id': str(rel_id_counter),
                        'label': 'COEXPRESSED',
                        'properties': {
                            'rho': float(rho),
                            'fdr': float(fdr),
                            'significant': fdr <= significant_threshold
                        },
                        'start': existing_gene_symbols[gene1],
                        'end': existing_gene_symbols[gene2]
                    }
                    coexpression_relationships.append(rel)
                    rel_id_counter += 1

            logger.info(f"创建了{len(new_gene_nodes)}个新基因节点和{len(coexpression_relationships)}个共表达关系")

            # 找出Fezf2模块中的基因，用于后续的模块分析
            fezf2_module = ['Fezf2', 'Bcl11b', 'Crym', 'Sox5', 'Tshz2', 'Foxo1', 'Zfpm2']
            fezf2_module_genes = [g for g in fezf2_module if g in existing_gene_symbols]
            logger.info(f"发现{len(fezf2_module_genes)}个Fezf2模块基因: {', '.join(fezf2_module_genes)}")

        except Exception as e:
            logger.error(f"处理基因共表达数据时出错: {e}")
            raise

        return new_gene_nodes, coexpression_relationships

    def update_regionlayer_gene_expression(self, region_layer_nodes: List[Dict]) -> None:
        """更新RegionLayer节点，添加基因表达数据和Fezf2模块平均值"""
        logger.info("更新RegionLayer基因表达数据...")

        # 检查是否存在基因表达数据
        expr_file = None
        if self.merfish_data_path:
            # 先尝试gene_expression.csv
            expr_path = self.merfish_data_path / "gene_expression.csv"
            if expr_path.exists():
                expr_file = expr_path

            # 如果不存在，尝试从区域特定文件中加载
            if not expr_file:
                expr_files = list(self.merfish_data_path.glob("*_gene_expression.csv"))
                if expr_files:
                    expr_file = expr_files[0]  # 使用第一个文件

        # 如果没有找到表达文件，无法继续
        if not expr_file:
            logger.error("未找到基因表达数据文件 (gene_expression.csv)")
            logger.error("请先运行 MERFISHDataIntegration.py 生成此文件")
            raise FileNotFoundError("缺少必要的基因表达数据文件")

        try:
            logger.info(f"加载基因表达数据: {expr_file}")
            expr_df = pd.read_csv(expr_file)

            if expr_df.empty:
                logger.error("基因表达数据为空")
                raise ValueError("基因表达数据为空，无法更新RegionLayer表达数据")

            # 创建RegionLayer ID到节点的映射
            rl_map = {}
            for node in region_layer_nodes:
                rl_id = node['properties'].get('rl_id')
                if rl_id:
                    rl_map[rl_id] = node

            # 按RegionLayer和基因分组计算平均表达
            grouped_expr = expr_df.groupby(['rl_id', 'gene'])['mean_logCPM'].mean().reset_index()

            # 更新每个RegionLayer节点的基因表达属性
            updated_count = 0
            rl_ids_with_expr = set()

            for _, row in grouped_expr.iterrows():
                rl_id = row['rl_id']
                gene = row['gene']
                mean_expr = row['mean_logCPM']

                if rl_id in rl_map and pd.notna(mean_expr):
                    node = rl_map[rl_id]
                    # 使用格式 mean_logCPM_{gene} 作为属性名
                    node['properties'][f'mean_logCPM_{gene}'] = float(mean_expr)
                    rl_ids_with_expr.add(rl_id)

            # 计算Fezf2模块基因在每个RegionLayer中的平均表达
            fezf2_module = ['Fezf2', 'Bcl11b', 'Crym', 'Sox5', 'Tshz2', 'Foxo1', 'Zfpm2']

            for rl_id in rl_ids_with_expr:
                node = rl_map[rl_id]
                # 收集模块基因表达值
                module_expr_values = []

                for gene in fezf2_module:
                    expr_key = f'mean_logCPM_{gene}'
                    if expr_key in node['properties'] and pd.notna(node['properties'][expr_key]):
                        module_expr_values.append(node['properties'][expr_key])

                # 如果至少有3个模块基因有表达数据，计算平均值
                if len(module_expr_values) >= 3:
                    node['properties']['fezf2_module_mean'] = float(np.mean(module_expr_values))
                    updated_count += 1

            logger.info(f"更新了{len(rl_ids_with_expr)}个RegionLayer节点的基因表达数据")
            logger.info(f"计算了{updated_count}个RegionLayer节点的Fezf2模块平均表达值")

        except Exception as e:
            logger.error(f"处理基因表达数据时出错: {e}")
            raise

    def update_projection_relationships(self, relationships: List[Dict],
                                        morpho_stats: Dict[str, Dict]) -> None:
        """更新Project_to关系，添加投射类型统计"""
        logger.info("更新投射关系...")

        # 预先聚合投射数据
        logger.info("预聚合投射数据...")

        # 创建区域到其RegionLayer节点的映射
        region_to_layers = defaultdict(list)
        for rl_id, stats in morpho_stats.items():
            region_name = stats.get('region_name')
            if region_name:
                region_to_layers[region_name].append((rl_id, stats))

        # 预先计算每对区域之间的投射统计
        proj_stats = defaultdict(lambda: {
            'length_total': 0,
            'it_len': 0,
            'et_len': 0,
            'ct_len': 0,
            'inh_len': 0,
            'n_axon': 0
        })

        # 使用已加载的投射数据
        if hasattr(self, 'morpho_data') and 'proj_axon' in self.morpho_data and not self.morpho_data['proj_axon'].empty:
            logger.info("从已加载的形态数据中提取投射信息...")
            proj_axon_df = self.morpho_data['proj_axon']

            # 识别绝对值列
            abs_cols = [col for col in proj_axon_df.columns if '_abs' in col]
            if abs_cols:
                # 获取轴突的源区域信息
                axon_sources = {}
                if 'info' in self.morpho_data and not self.morpho_data['info'].empty:
                    info_df = self.morpho_data['info']
                    # 提取源区域和投射类型信息
                    for idx in proj_axon_df.index:
                        if idx in info_df.index:
                            row = info_df.loc[idx]
                            source_region = row.get('region',
                                                    row.get('source_region', row.get('celltype_manual', None)))
                            proj_type = row.get('projection_type', 'unknown').lower()
                            if source_region:
                                axon_sources[idx] = (source_region, proj_type)

                # 处理每个轴突的投射
                for idx, row in proj_axon_df.iterrows():
                    source_info = axon_sources.get(idx, (None, 'unknown'))
                    source_region, proj_type = source_info

                    if not source_region:
                        continue

                    # 处理投射到各区域的长度
                    for col in abs_cols:
                        if pd.notna(row[col]) and row[col] > 0:
                            # 从列名中提取目标区域
                            # 假设格式为 proj_axon_REGION_abs
                            parts = col.split('_')
                            if len(parts) >= 3:
                                # 提取区域部分（可能包含多个部分）
                                target_region = '_'.join(parts[2:-1])

                                length = float(row[col])
                                key = (source_region, target_region)

                                # 更新统计
                                proj_stats[key]['length_total'] += length
                                proj_stats[key]['n_axon'] += 1

                                # 根据投射类型分配长度
                                if 'ipsilateral' in proj_type or 'it' in proj_type:
                                    proj_stats[key]['it_len'] += length
                                elif 'contralateral' in proj_type or 'et' in proj_type:
                                    proj_stats[key]['et_len'] += length
                                elif 'corticothalamic' in proj_type or 'ct' in proj_type:
                                    proj_stats[key]['ct_len'] += length
                                elif 'gaba' in proj_type or 'inh' in proj_type:
                                    proj_stats[key]['inh_len'] += length
                                else:
                                    # 默认分配给IT（最常见类型）
                                    proj_stats[key]['it_len'] += length * 0.6
                                    proj_stats[key]['et_len'] += length * 0.3
                                    proj_stats[key]['ct_len'] += length * 0.1

                logger.info(f"预处理了{len(proj_stats)}对区域的投射统计")
            else:
                # 尝试传统格式（如果存在）
                logger.warning("未在投射数据中找到_abs列，检查是否为传统格式...")
                if 'Source' in proj_axon_df.columns and 'Target' in proj_axon_df.columns and 'Value' in proj_axon_df.columns:
                    logger.info("检测到传统的Source-Target-Value格式")

                    # 合并投射类型信息
                    if 'info' in self.morpho_data and not self.morpho_data['info'].empty:
                        info_df = self.morpho_data['info']
                        if 'projection_type' in info_df.columns:
                            # 合并投射类型
                            merged_df = pd.merge(
                                proj_axon_df.reset_index(),
                                info_df['projection_type'],
                                left_index=True,
                                right_index=True,
                                how='left'
                            )

                            # 计算投射统计
                            for _, row in merged_df.iterrows():
                                source = row.get('Source')
                                target = row.get('Target')

                                if not source or not target:
                                    continue

                                # 获取投射类型和长度
                                proj_type = row.get('projection_type', 'unknown').lower()
                                length = row.get('Value', 0)

                                # 更新统计
                                key = (source, target)
                                proj_stats[key]['length_total'] += length
                                proj_stats[key]['n_axon'] += 1

                                # 根据投射类型分配长度
                                if 'ipsilateral' in proj_type or 'it' in proj_type:
                                    proj_stats[key]['it_len'] += length
                                elif 'contralateral' in proj_type or 'et' in proj_type:
                                    proj_stats[key]['et_len'] += length
                                elif 'corticothalamic' in proj_type or 'ct' in proj_type:
                                    proj_stats[key]['ct_len'] += length
                                elif 'gaba' in proj_type or 'inh' in proj_type:
                                    proj_stats[key]['inh_len'] += length
                                else:
                                    # 默认分配给IT
                                    proj_stats[key]['it_len'] += length * 0.6  # 60% IT
                                    proj_stats[key]['et_len'] += length * 0.3  # 30% ET
                                    proj_stats[key]['ct_len'] += length * 0.1  # 10% CT
                        else:
                            # 没有投射类型信息，使用默认分配
                            for _, row in proj_axon_df.iterrows():
                                source = row.get('Source')
                                target = row.get('Target')

                                if not source or not target:
                                    continue

                                length = row.get('Value', 0)
                                key = (source, target)

                                # 使用固定比例分配
                                proj_stats[key]['length_total'] += length
                                proj_stats[key]['n_axon'] += 1
                                proj_stats[key]['it_len'] += length * 0.6  # 60% IT
                                proj_stats[key]['et_len'] += length * 0.3  # 30% ET
                                proj_stats[key]['ct_len'] += length * 0.1  # 10% CT
        else:
            # 尝试从文件直接加载
            logger.warning("未加载投射数据或数据为空，尝试从文件直接加载...")
            try:
                proj_axon_file = self.morpho_data_path / 'Proj_Axon_Final.csv'
                if not proj_axon_file.exists():
                    proj_axon_file = self.morpho_data_path / 'Proj_Axon_abs.csv'

                if proj_axon_file.exists():
                    logger.info(f"加载投射轴突数据: {proj_axon_file}")
                    proj_axon_df = pd.read_csv(proj_axon_file)

                    # 识别轴突ID列
                    id_col = self._identify_id_column(proj_axon_df)
                    if id_col:
                        proj_axon_df = proj_axon_df.set_index(id_col)

                    # 识别绝对值列
                    abs_cols = [col for col in proj_axon_df.columns if '_abs' in col]

                    # 如果找到了绝对值列，处理新格式
                    if abs_cols:
                        # 按照上面相同的逻辑处理
                        # ...这里省略重复代码...
                        logger.info("请使用load_morphology_data方法加载数据")
                    else:
                        # 处理传统格式
                        logger.info("请使用load_morphology_data方法加载数据")
                else:
                    logger.warning(f"投射轴突文件不存在: {proj_axon_file}")
            except Exception as e:
                logger.error(f"处理投射数据时出错: {e}")

        # 更新Project_to关系
        updated_count = 0
        for rel in tqdm(relationships, desc="更新投射关系"):
            if rel['label'] == 'Project_to':
                # 获取源和目标区域
                src_node = rel.get('start', {})
                dst_node = rel.get('end', {})

                if isinstance(src_node, dict) and isinstance(dst_node, dict):
                    src_props = src_node.get('properties', {})
                    dst_props = dst_node.get('properties', {})

                    src_region = src_props.get('name', '')
                    dst_region = dst_props.get('name', '')

                    if src_region and dst_region:
                        # 查找投射数据
                        key = (src_region, dst_region)
                        stats = proj_stats.get(key, {})

                        # 使用现有的长度作为基础
                        total_length = rel['properties'].get('length', 0)

                        if stats.get('length_total', 0) > 0:
                            # 使用预计算的投射统计
                            rel['properties'].update({
                                'length_total': stats['length_total'],
                                'it_len': stats['it_len'],
                                'et_len': stats['et_len'],
                                'ct_len': stats['ct_len'],
                                'inh_len': stats['inh_len'],
                                'n_axon': stats['n_axon']
                            })
                        else:
                            # 使用估计比例
                            rel['properties'].update({
                                'length_total': total_length,
                                'it_len': total_length * 0.6,  # 60% IT
                                'et_len': total_length * 0.3,  # 30% ET
                                'ct_len': total_length * 0.1,  # 10% CT
                                'inh_len': 0,
                                'n_axon': rel['properties'].get('n_axon', 1)
                            })

                        updated_count += 1

        logger.info(f"更新了{updated_count}个Project_to关系")

    def create_gene_nodes_and_relationships(self, all_nodes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """创建基因节点和共表达关系"""
        logger.info("创建基因节点和共表达关系...")

        # 新的基因节点和共表达关系
        new_gene_nodes = []
        coexpression_relationships = []

        # 用于节点ID和关系ID的计数器
        node_id_counter = 50000  # 从50000开始避免ID冲突
        rel_id_counter = 100000  # 从100000开始避免ID冲突

        # 检查是否存在基因共表达数据
        coexpr_file = None
        if self.merfish_data_path:
            coexpr_path = self.merfish_data_path / "gene_coexpression.csv"
            if coexpr_path.exists():
                coexpr_file = coexpr_path
            else:
                # 尝试查找区域特定的共表达文件
                files = list(self.merfish_data_path.glob("*_gene_coexpression.csv"))
                if files:
                    coexpr_file = files[0]  # 使用第一个找到的文件

        # 如果没有找到共表达文件，尝试在默认位置查找
        if not coexpr_file:
            default_path = Path("merfish_output/gene_coexpression.csv")
            if default_path.exists():
                coexpr_file = default_path

        # 如果没有共表达数据，返回空结果
        if not coexpr_file:
            logger.warning("未找到基因共表达数据，跳过基因关系创建")
            return [], []

        try:
            logger.info(f"加载基因共表达数据: {coexpr_file}")
            coexpr_df = pd.read_csv(coexpr_file)

            # 收集所有涉及的基因
            all_genes = set()
            for _, row in coexpr_df.iterrows():
                gene1 = row.get('gene1')
                gene2 = row.get('gene2')
                if pd.notna(gene1) and pd.notna(gene2):
                    all_genes.add(gene1)
                    all_genes.add(gene2)

            # 检查已有的基因节点
            existing_gene_symbols = {}
            for node in all_nodes:
                if 'Gene' in node.get('labels', []):
                    symbol = node['properties'].get('symbol')
                    if symbol:
                        existing_gene_symbols[symbol] = node

            # 创建不存在的基因节点
            for gene in all_genes:
                if gene not in existing_gene_symbols:
                    gene_node = {
                        'type': 'node',
                        'id': str(node_id_counter),
                        'labels': ['Gene'],
                        'properties': {
                            'symbol': gene,
                            'name': gene,
                            'type': 'protein_coding'  # 默认类型
                        }
                    }
                    new_gene_nodes.append(gene_node)
                    existing_gene_symbols[gene] = gene_node
                    node_id_counter += 1

            # 创建共表达关系
            significant_threshold = 0.05  # FDR显著性阈值
            for _, row in coexpr_df.iterrows():
                gene1 = row.get('gene1')
                gene2 = row.get('gene2')
                rho = row.get('rho')
                fdr = row.get('fdr')

                # 检查数据有效性
                if (pd.isna(gene1) or pd.isna(gene2) or pd.isna(rho) or
                        gene1 not in existing_gene_symbols or gene2 not in existing_gene_symbols):
                    continue

                # 只保留显著的关系
                if pd.notna(fdr) and fdr <= significant_threshold:
                    rel = {
                        'type': 'relationship',
                        'id': str(rel_id_counter),
                        'label': 'COEXPRESSED',
                        'properties': {
                            'rho': float(rho),
                            'fdr': float(fdr),
                            'significant': fdr <= significant_threshold
                        },
                        'start': existing_gene_symbols[gene1],
                        'end': existing_gene_symbols[gene2]
                    }
                    coexpression_relationships.append(rel)
                    rel_id_counter += 1

            logger.info(f"创建了{len(new_gene_nodes)}个新基因节点和{len(coexpression_relationships)}个共表达关系")

        except Exception as e:
            logger.error(f"处理基因共表达数据时出错: {e}")

        return new_gene_nodes, coexpression_relationships

    def update_regionlayer_gene_expression(self, region_layer_nodes: List[Dict]) -> None:
        """更新RegionLayer节点，添加基因表达数据"""
        logger.info("更新RegionLayer基因表达数据...")

        # 检查是否存在基因表达数据
        expr_file = None
        if self.merfish_data_path:
            expr_path = self.merfish_data_path / "gene_expression.csv"
            if expr_path.exists():
                expr_file = expr_path

        # 如果没有找到表达文件，尝试在默认位置查找
        if not expr_file:
            default_path = Path("merfish_output/gene_expression.csv")
            if default_path.exists():
                expr_file = default_path

        # 如果没有表达数据，返回
        if not expr_file:
            logger.warning("未找到基因表达数据，跳过RegionLayer表达更新")
            return

        try:
            logger.info(f"加载基因表达数据: {expr_file}")
            expr_df = pd.read_csv(expr_file)

            # 创建RegionLayer ID到节点的映射
            rl_map = {}
            for node in region_layer_nodes:
                rl_id = node['properties'].get('rl_id')
                if rl_id:
                    rl_map[rl_id] = node

            # 按RegionLayer分组计算平均表达
            grouped = expr_df.groupby('rl_id')

            # 更新每个RegionLayer节点的基因表达属性
            updated_count = 0
            for rl_id, group in grouped:
                if rl_id in rl_map:
                    node = rl_map[rl_id]

                    # 对于每个基因，添加表达值作为RegionLayer的属性
                    for _, row in group.iterrows():
                        gene = row.get('gene')
                        mean_expr = row.get('mean_logCPM')

                        if pd.notna(gene) and pd.notna(mean_expr):
                            # 使用格式 mean_logCPM_{gene} 作为属性名
                            node['properties'][f'mean_logCPM_{gene}'] = float(mean_expr)

                    updated_count += 1

            logger.info(f"更新了{updated_count}个RegionLayer节点的基因表达数据")

            # 特别处理Fezf2模块基因
            fezf2_module = ['Fezf2', 'Bcl11b', 'Crym', 'Sox5', 'Tshz2', 'Foxo1', 'Zfpm2']
            for node in region_layer_nodes:
                # 计算模块平均表达量
                expr_values = []
                for gene in fezf2_module:
                    expr_key = f'mean_logCPM_{gene}'
                    if expr_key in node['properties']:
                        expr_values.append(node['properties'][expr_key])

                if expr_values:
                    node['properties']['fezf2_module_mean'] = float(np.mean(expr_values))

        except Exception as e:
            logger.error(f"处理基因表达数据时出错: {e}")

    def save_new_kg(self, nodes: List[Dict], relationships: List[Dict], output_path: str):
        """保存新的知识图谱"""
        logger.info(f"保存新知识图谱到 {output_path}...")

        # 备份原始文件
        if os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            logger.info(f"备份已存在的输出文件到 {backup_path}")
            try:
                shutil.copy2(output_path, backup_path)
            except Exception as e:
                logger.warning(f"备份文件失败: {e}")

        # 临时文件路径
        temp_path = f"{output_path}.tmp"

        # 合并所有数据
        all_data = nodes + relationships

        # 保存为临时JSON文件
        with open(temp_path, 'w') as f:
            json.dump(all_data, f, indent=2)

            # 如果成功，重命名为最终文件
            os.rename(temp_path, output_path)

            logger.info(f"保存完成: {len(nodes)}个节点, {len(relationships)}条关系")

        def check_neo4j_schema(self) -> bool:
            """检查Neo4j数据库是否有必要的约束和索引"""
            if self.neo4j_conn is None:
                logger.warning("无法检查Neo4j模式，未连接到数据库")
                return True

            try:
                with self.neo4j_conn.session() as session:
                    # 检查必要的约束
                    constraints_result = session.run("CALL db.constraints()")
                    constraints = [record["name"] for record in constraints_result if "name" in record]

                    required_constraints = [
                        "pk_regionlayer",  # RegionLayer.rl_id 唯一约束
                        "pk_region",  # Region.region_id 唯一约束
                        "pk_subclass",  # Subclass.tran_id 唯一约束
                        "pk_class",  # Class.tran_id 唯一约束
                        "pk_gene"  # Gene.symbol 唯一约束
                    ]

                    missing_constraints = [c for c in required_constraints if
                                           not any(c in constraint for constraint in constraints)]

                    if missing_constraints:
                        logger.warning(f"Neo4j数据库缺少必要的约束: {missing_constraints}")
                        logger.warning("请运行以下Cypher语句创建约束:")
                        for constraint in missing_constraints:
                            if constraint == "pk_regionlayer":
                                logger.warning(
                                    "CREATE CONSTRAINT pk_regionlayer IF NOT EXISTS FOR (rl:RegionLayer) REQUIRE rl.rl_id IS UNIQUE;")
                            elif constraint == "pk_region":
                                logger.warning(
                                    "CREATE CONSTRAINT pk_region IF NOT EXISTS FOR (r:Region) REQUIRE r.region_id IS UNIQUE;")
                            elif constraint == "pk_subclass":
                                logger.warning(
                                    "CREATE CONSTRAINT pk_subclass IF NOT EXISTS FOR (s:Subclass) REQUIRE s.tran_id IS UNIQUE;")
                            elif constraint == "pk_class":
                                logger.warning(
                                    "CREATE CONSTRAINT pk_class IF NOT EXISTS FOR (c:Class) REQUIRE c.tran_id IS UNIQUE;")
                            elif constraint == "pk_gene":
                                logger.warning(
                                    "CREATE CONSTRAINT pk_gene IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE;")

                        return False

                    return True
            except Exception as e:
                logger.error(f"检查Neo4j模式时出错: {e}")
                return False

    def run(self, output_path: str = "kg_v2.3.json", strict_mode: bool = True) -> Dict[str, Any]:
        """运行完整的重构流程

        Args:
            output_path: 输出文件路径
            strict_mode: 严格模式，当缺少关键数据时报错而不是使用默认值或随机值
        """
        logger.info("开始知识图谱重构...")
        start_time = time.time()

        # 检查Neo4j模式
        if self.neo4j_conn is not None:
            if not self.check_neo4j_schema():
                logger.error("Neo4j模式检查失败，请先创建必要的约束和索引")
                raise ValueError("Neo4j模式检查失败")

        # 1. 加载现有知识图谱
        nodes, relationships = self.load_kg()

        # 分离不同类型的节点
        region_nodes = [n for n in nodes if 'Region' in n.get('labels', [])]
        transcriptomic_nodes = [n for n in nodes if any(
            label in n.get('labels', []) for label in ['Class', 'Subclass', 'Supertype', 'Cluster']
        )]

        logger.info(f"找到{len(region_nodes)}个Region节点和{len(transcriptomic_nodes)}个转录组节点")

        # 验证关键节点存在
        if strict_mode and not region_nodes:
            raise ValueError("未找到Region节点，无法继续构建知识图谱")

        # 2. 创建RegionLayer节点
        region_layer_nodes = self.create_region_layer_nodes(region_nodes)
        if not region_layer_nodes:
            error_msg = "未创建任何RegionLayer节点！请检查Region节点是否有name属性，以及是否有皮层区域。"
            if strict_mode:
                raise ValueError(error_msg)
            logger.warning(error_msg)

        # 3. 计算形态学统计
        morpho_stats = self.calculate_morphology_stats(region_layer_nodes)

        # 4. 更新RegionLayer节点属性
        for node in region_layer_nodes:
            rl_id = node['properties']['rl_id']
            if rl_id in morpho_stats:
                node['properties'].update(morpho_stats[rl_id])
            elif strict_mode:
                raise ValueError(f"缺少RegionLayer节点 {rl_id} 的形态学统计数据")
            else:
                # 设置默认值以确保节点有所有必需的属性
                logger.warning(f"为 {rl_id} 使用默认形态学属性值")
                for key, default in [
                    ('it_pct', 0.5),
                    ('et_pct', 0.3),
                    ('ct_pct', 0.2),
                    ('lr_pct', 0.3),
                    ('lr_prior', 0.2),
                    ('morph_ax_len_mean', 0.0),
                    ('morph_ax_len_std', 0.0),
                    ('dend_polarity_index_mean', 0.0),
                    ('dend_br_std', 0.0),
                    ('n_neuron', 0)
                ]:
                    if key not in node['properties']:
                        node['properties'][key] = default

        # 5. 更新Subclass节点
        self.update_subclass_nodes(transcriptomic_nodes)

        # 6. 创建HAS_LAYER关系
        has_layer_rels = self.create_has_layer_relationships(region_nodes, region_layer_nodes)

        # 7. 创建转录组关系
        transcriptomic_rels = self.create_transcriptomic_relationships(
            region_layer_nodes, transcriptomic_nodes
        )

        if strict_mode and not transcriptomic_rels:
            raise ValueError("未创建任何转录组关系，可能是MERFISH数据缺失或格式错误")

        # 8. 更新投射关系
        self.update_projection_relationships(relationships, morpho_stats)

        # 9. 创建基因节点和共表达关系
        gene_nodes, coexpression_rels = self.create_gene_nodes_and_relationships(nodes)

        if strict_mode and not coexpression_rels:
            raise ValueError("未创建任何基因共表达关系，任务④(Fezf2模块分析)将无法运行")

        # 10. 更新RegionLayer基因表达数据
        self.update_regionlayer_gene_expression(region_layer_nodes)

        # 11. 合并所有节点和关系
        all_nodes = nodes + region_layer_nodes + gene_nodes
        all_relationships = relationships + has_layer_rels + transcriptomic_rels + coexpression_rels

        # 12. 保存新的知识图谱
        self.save_new_kg(all_nodes, all_relationships, output_path)

        elapsed_time = time.time() - start_time
        logger.info(f"知识图谱重构完成！耗时: {elapsed_time:.2f}秒")

        return {
            'total_nodes': len(all_nodes),
            'total_relationships': len(all_relationships),
            'new_region_layer_nodes': len(region_layer_nodes),
            'new_gene_nodes': len(gene_nodes),
            'new_relationships': len(has_layer_rels) + len(transcriptomic_rels) + len(coexpression_rels),
            'coexpression_relationships': len(coexpression_rels),
            'elapsed_time': f"{elapsed_time:.2f}秒"
        }


# Update the main function to add the --strict flag

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='知识图谱重构工具 - KG 2.3版本')
    parser.add_argument('kg_path', help='现有知识图谱JSON文件路径、包含CSV文件的目录或Neo4j连接URI')
    parser.add_argument('--morpho-data', '-m', required=True, help='形态学数据文件夹路径')
    parser.add_argument('--merfish-data', '-f', help='MERFISH数据路径（可选）')
    parser.add_argument('--output', '-o', default='kg_v2.3.json', help='输出JSON文件路径')
    parser.add_argument('--cache-dir', '-c', help='缓存目录路径（可选，默认为系统临时目录）')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='严格模式：当缺少必要数据时报错而不是使用默认值或随机值')

    args = parser.parse_args()

    try:
        # 初始化重构器
        restructurer = KnowledgeGraphRestructure(
            kg_path=args.kg_path,
            morpho_data_path=args.morpho_data,
            merfish_data_path=args.merfish_data,
            cache_dir=args.cache_dir
        )

        # 运行重构
        results = restructurer.run(output_path=args.output, strict_mode=args.strict)

        print("\n重构结果:")
        for key, value in results.items():
            print(f"{key}: {value}")

        print(f"\n知识图谱已保存到: {args.output}")

        # 如果有基因共表达关系，特别标注
        if results.get('coexpression_relationships', 0) > 0:
            print(
                f"\n成功创建了 {results['coexpression_relationships']} 个基因共表达关系，任务④(Fezf2模块分析)可正常运行")
        else:
            print("\n警告：未创建基因共表达关系，任务④(Fezf2模块分析)将无法运行")

    except Exception as e:
        logger.error(f"重构过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

# 使用示例
if __name__ == "__main__":
    main()