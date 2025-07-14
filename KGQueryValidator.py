import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import argparse
import sys
import os
import tempfile
from io import TextIOWrapper
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KGQueryValidator:
    """知识图谱查询和验证工具"""

    def __init__(self, kg_path: str, output_dir: Optional[str] = None, allow_missing: bool = False):
        """
        初始化查询验证器

        Args:
            kg_path: 知识图谱文件路径（JSON、CSV目录或Neo4j URI）
            output_dir: 输出文件夹路径（可选，默认为当前目录）
            allow_missing: 是否允许缺失属性（使用默认值代替）
        """
        self.kg_path = kg_path
        self.output_dir = Path(output_dir) if output_dir else Path("./kg_validator_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.allow_missing = allow_missing

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

        # 加载知识图谱
        self.graph = None
        self.load_kg()

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

    def load_kg(self):
        """加载知识图谱到NetworkX图形"""
        logger.info("加载知识图谱数据...")

        # 创建有向图
        self.graph = nx.DiGraph()

        # 根据不同的数据源加载图形
        if self.neo4j_conn is not None:
            self._load_from_neo4j()
        elif os.path.isdir(self.kg_path):
            self._load_from_csv_dir()
        else:
            # 使用流式处理加载JSON以节省内存
            self._load_from_json_streaming()

        logger.info(f"知识图谱加载完成: {self.graph.number_of_nodes()}个节点, {self.graph.number_of_edges()}条边")

    def _load_from_neo4j(self):
        """从Neo4j数据库加载知识图谱"""
        try:
            with self.neo4j_conn.session() as session:
                # 查询过滤，只获取我们需要的子图节点
                node_query = """
                MATCH (n) 
                WHERE n:Region OR n:RegionLayer OR n:Class OR n:Subclass OR n:Cluster OR n:Gene
                RETURN n
                """
                node_result = session.run(node_query)

                # 添加节点
                for record in node_result:
                    node = record["n"]
                    node_id = str(node.id)
                    labels = list(node.labels)
                    properties = dict(node)

                    # 添加到图中
                    self.graph.add_node(node_id, labels=labels, properties=properties)

                # 查询过滤，只获取RegionLayer相关的边
                edge_query = """
                MATCH (r)-[rel]->(s)
                WHERE r:Region OR r:RegionLayer OR s:RegionLayer OR s:Class OR s:Subclass OR s:Cluster
                OR type(rel) = 'Project_to' OR type(rel) = 'HAS_LAYER' OR type(rel) = 'HAS_CLASS' OR type(rel) = 'HAS_SUBCLASS'
                RETURN r, rel, s
                """
                edge_result = session.run(edge_query)

                # 添加边
                for record in edge_result:
                    source_id = str(record["r"].id)
                    target_id = str(record["s"].id)
                    rel = record["rel"]
                    rel_type = rel.type
                    properties = dict(rel)

                    # 添加到图中
                    self.graph.add_edge(source_id, target_id, type=rel_type, properties=properties)

        except Exception as e:
            logger.error(f"从Neo4j加载知识图谱失败: {e}")
            raise

    def _load_from_csv_dir(self):
        """从CSV文件目录加载知识图谱"""
        nodes_csv = os.path.join(self.kg_path, "nodes.csv")
        edges_csv = os.path.join(self.kg_path, "relationships.csv")

        if not os.path.exists(nodes_csv) or not os.path.exists(edges_csv):
            raise FileNotFoundError(f"在{self.kg_path}中未找到nodes.csv或relationships.csv")

        # 加载节点
        try:
            nodes_df = pd.read_csv(nodes_csv)
            for _, row in nodes_df.iterrows():
                node_id = str(row.get(':ID', row.get('id')))
                labels = row.get(':LABEL', row.get('labels', '')).split(';')

                # 只保留我们需要的节点类型
                keep_node = any(
                    label in ['Region', 'RegionLayer', 'Class', 'Subclass', 'Cluster', 'Gene'] for label in labels)
                if not keep_node:
                    continue

                # 构建属性字典
                properties = {}
                for col in nodes_df.columns:
                    if col not in [':ID', 'id', ':LABEL', 'labels'] and pd.notna(row[col]):
                        properties[col] = row[col]

                # 添加到图中
                self.graph.add_node(node_id, labels=labels, properties=properties)

            # 加载边
            edges_df = pd.read_csv(edges_csv)
            for _, row in edges_df.iterrows():
                source_id = str(row.get(':START_ID', row.get('start')))
                target_id = str(row.get(':END_ID', row.get('end')))
                rel_type = row.get(':TYPE', row.get('type'))

                # 只保留我们需要的边类型
                if rel_type not in ['Project_to', 'HAS_LAYER', 'HAS_CLASS', 'HAS_SUBCLASS', 'HAS_CLUSTER']:
                    continue

                # 检查源节点和目标节点是否存在
                if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
                    continue

                # 构建属性字典
                properties = {}
                for col in edges_df.columns:
                    if col not in [':START_ID', 'start', ':END_ID', 'end', ':TYPE', 'type'] and pd.notna(row[col]):
                        properties[col] = row[col]

                # 添加到图中
                self.graph.add_edge(source_id, target_id, type=rel_type, properties=properties)

        except Exception as e:
            logger.error(f"从CSV加载知识图谱失败: {e}")
            raise

    def _load_from_json_streaming(self):
        """流式处理加载JSON知识图谱文件，减少内存使用"""
        try:
            # 创建ID到节点标签的映射（用于过滤要保留的节点）
            id_to_labels = {}

            # 首先扫描所有节点，只保留特定类型的节点
            with open(self.kg_path, 'r') as f:
                # 读取文件开头的"["
                f.read(1)

                # 使用迭代器逐个处理JSON对象
                for obj in self._json_objects(f):
                    if obj.get('type') == 'node':
                        node_id = obj.get('id')
                        labels = obj.get('labels', [])

                        # 只保留我们需要的节点类型
                        keep_node = any(
                            label in ['Region', 'RegionLayer', 'Class', 'Subclass', 'Cluster', 'Gene'] for label in
                            labels)
                        if keep_node:
                            id_to_labels[node_id] = labels
                            properties = obj.get('properties', {})

                            # 添加到图中
                            self.graph.add_node(node_id, labels=labels, properties=properties)

            # 再次扫描所有关系，只保留与保留节点相关的关系
            with open(self.kg_path, 'r') as f:
                # 读取文件开头的"["
                f.read(1)

                # 使用迭代器逐个处理JSON对象
                for obj in self._json_objects(f):
                    if obj.get('type') == 'relationship':
                        rel_type = obj.get('label')
                        start_id = obj.get('start', {}).get('id')
                        end_id = obj.get('end', {}).get('id')

                        # 只保留特定类型的关系且两端节点都存在
                        if rel_type in ['Project_to', 'HAS_LAYER', 'HAS_CLASS', 'HAS_SUBCLASS', 'HAS_CLUSTER'] and \
                                start_id in id_to_labels and end_id in id_to_labels:
                            properties = obj.get('properties', {})

                            # 添加到图中
                            self.graph.add_edge(start_id, end_id, type=rel_type, properties=properties)

        except Exception as e:
            logger.error(f"从JSON加载知识图谱失败: {e}")
            raise

    def _json_objects(self, file_obj: TextIOWrapper):
        """迭代读取JSON文件中的对象"""
        buffer = ""
        depth = 0
        in_string = False
        escape_next = False

        for chunk in iter(lambda: file_obj.read(8192), ''):
            buffer += chunk

            i = 0
            start = 0

            while i < len(buffer):
                char = buffer[i]

                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj_str = buffer[start:i + 1]
                                obj = json.loads(obj_str)
                                yield obj
                            except json.JSONDecodeError:
                                pass

                i += 1

            # 保留未处理完的部分
            if depth > 0:
                buffer = buffer[start:]
            else:
                buffer = ""

    def query_region_layer_properties(self, region_name: Optional[str] = None) -> pd.DataFrame:
        """查询RegionLayer节点的属性"""
        # 找到所有RegionLayer节点
        region_layers = []

        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'RegionLayer' in labels:
                props = node_attrs.get('properties', {})

                # 如果指定了区域，过滤非匹配区域
                if region_name and props.get('region_name') != region_name:
                    continue

                # 检查必要属性
                rl_id = props.get('rl_id')
                region = props.get('region_name')
                layer = props.get('layer')

                # 收集投射类型比例
                it_pct = props.get('it_pct', 0.0 if self.allow_missing else None)
                et_pct = props.get('et_pct', 0.0 if self.allow_missing else None)
                ct_pct = props.get('ct_pct', 0.0 if self.allow_missing else None)
                lr_pct = props.get('lr_pct', 0.0 if self.allow_missing else None)
                lr_prior = props.get('lr_prior', 0.0 if self.allow_missing else None)

                # 收集形态学属性
                ax_len_mean = props.get('morph_ax_len_mean', 0.0 if self.allow_missing else None)
                ax_len_std = props.get('morph_ax_len_std', 0.0 if self.allow_missing else None)
                dend_polarity = props.get('dend_polarity_index_mean', 0.0 if self.allow_missing else None)

                # 添加到结果列表
                region_layers.append({
                    'rl_id': rl_id,
                    'region_name': region,
                    'layer': layer,
                    'it_pct': it_pct,
                    'et_pct': et_pct,
                    'ct_pct': ct_pct,
                    'lr_pct': lr_pct,
                    'lr_prior': lr_prior,
                    'morph_ax_len_mean': ax_len_mean,
                    'morph_ax_len_std': ax_len_std,
                    'dend_polarity_index_mean': dend_polarity,
                    'n_neuron': props.get('n_neuron', 0)
                })

        # 转换为DataFrame并排序
        df = pd.DataFrame(region_layers)
        if len(df) > 0:
            df = df.sort_values(['region_name', 'layer'])

            # 保存结果
            output_file = self.output_dir / f"region_layer_properties{'_' + region_name if region_name else ''}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存RegionLayer属性到: {output_file}")

            # 生成可视化
            self._plot_projection_types(df, region_name)

        return df

    def _plot_projection_types(self, df: pd.DataFrame, region_name: Optional[str] = None):
        """为RegionLayer节点的投射类型生成可视化"""
        if len(df) == 0:
            return

        # 准备投射类型数据
        plot_data = df[['region_name', 'layer', 'it_pct', 'et_pct', 'ct_pct']].copy()

        # 区域分组
        regions = plot_data['region_name'].unique()

        # 为每个区域创建一个图表
        for region in regions:
            if region_name and region != region_name:
                continue

            region_data = plot_data[plot_data['region_name'] == region]

            # 确保数据按层顺序排列
            layer_order = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']
            region_data['layer_order'] = region_data['layer'].apply(
                lambda x: layer_order.index(x) if x in layer_order else 999)
            region_data = region_data.sort_values('layer_order')

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 堆叠条形图
            bottom = np.zeros(len(region_data))

            for col, color, label in [
                ('it_pct', 'skyblue', 'IT'),
                ('et_pct', 'coral', 'ET'),
                ('ct_pct', 'lightgreen', 'CT')
            ]:
                values = region_data[col].values
                ax.bar(region_data['layer'], values, bottom=bottom, label=label, color=color)
                bottom += values

            ax.set_title(f'{region} 投射类型分布')
            ax.set_xlabel('层')
            ax.set_ylabel('比例')
            ax.legend()

            # 保存图表
            output_file = self.output_dir / f"{region}_projection_types.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"保存投射类型可视化到: {output_file}")

    def query_transcriptomic_distribution(self, region_name: str, layer: str) -> pd.DataFrame:
        """查询特定RegionLayer的细胞类型分布"""
        # 找到指定的RegionLayer节点
        rl_node_id = None
        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'RegionLayer' in labels:
                props = node_attrs.get('properties', {})
                if props.get('region_name') == region_name and props.get('layer') == layer:
                    rl_node_id = node_id
                    break

        if rl_node_id is None:
            logger.warning(f"未找到RegionLayer节点: {region_name}_{layer}")
            return pd.DataFrame()

        # 查找从RegionLayer出发的HAS_SUBCLASS关系
        subclass_distributions = []

        for _, target_id, edge_attrs in self.graph.out_edges(rl_node_id, data=True):
            edge_type = edge_attrs.get('type')
            if edge_type == 'HAS_SUBCLASS':
                edge_props = edge_attrs.get('properties', {})

                # 获取目标亚类节点的属性
                target_attrs = self.graph.nodes[target_id]
                target_props = target_attrs.get('properties', {})

                # 检查必要属性
                subclass_name = target_props.get('name')
                pct_cells = edge_props.get('pct_cells')
                rank = edge_props.get('rank')
                n_cells = edge_props.get('n_cells')
                proj_type = edge_props.get('proj_type', target_props.get('proj_type', 'UNK'))

                # 添加到结果列表
                subclass_distributions.append({
                    'subclass': subclass_name,
                    'pct_cells': pct_cells,
                    'rank': rank,
                    'n_cells': n_cells,
                    'proj_type': proj_type
                })

        # 转换为DataFrame并排序
        df = pd.DataFrame(subclass_distributions)
        if len(df) > 0:
            df = df.sort_values('rank')

            # 保存结果
            output_file = self.output_dir / f"{region_name}_{layer}_subclass_distribution.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存亚类分布到: {output_file}")

            # 生成可视化
            self._plot_subclass_distribution(df, region_name, layer)

        return df

    def _plot_subclass_distribution(self, df: pd.DataFrame, region_name: str, layer: str):
        """为亚类分布生成可视化"""
        if len(df) == 0:
            return

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))

        # 按投射类型为条形图上色
        colors = {
            'IT': 'skyblue',
            'ET': 'coral',
            'CT': 'lightgreen',
            'NP': 'plum',
            'UNK': 'lightgray'
        }

        # 获取前10个亚类
        plot_df = df.head(10).copy()

        # 创建条形图
        bars = ax.barh(plot_df['subclass'], plot_df['pct_cells'],
                       color=[colors.get(p, 'lightgray') for p in plot_df['proj_type']])

        # 添加标签
        ax.set_title(f'{region_name} {layer} 亚类分布')
        ax.set_xlabel('细胞比例')
        ax.set_ylabel('亚类')

        # 添加投射类型图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=pt) for pt, color in colors.items() if
                           pt in plot_df['proj_type'].values]
        ax.legend(handles=legend_elements, loc='upper right')

        # 保存图表
        output_file = self.output_dir / f"{region_name}_{layer}_subclass_distribution.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存亚类分布可视化到: {output_file}")

    def query_projections_between_regions(self, source_region: str, target_region: str) -> pd.DataFrame:
        """查询两个区域之间的投射关系"""
        # 找到所有Region节点
        region_nodes = {}
        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'Region' in labels:
                props = node_attrs.get('properties', {})
                region_name = props.get('name')
                if region_name:
                    region_nodes[region_name] = node_id

        # 检查源区域和目标区域是否存在
        if source_region not in region_nodes:
            logger.warning(f"未找到源区域: {source_region}")
            return pd.DataFrame()

        if target_region not in region_nodes:
            logger.warning(f"未找到目标区域: {target_region}")
            return pd.DataFrame()

        # 查找从源区域到目标区域的Project_to关系
        projections = []

        for _, target_id, edge_attrs in self.graph.out_edges(region_nodes[source_region], data=True):
            edge_type = edge_attrs.get('type')
            target_attrs = self.graph.nodes[target_id]
            target_props = target_attrs.get('properties', {})
            target_name = target_props.get('name')

            if edge_type == 'Project_to' and target_name == target_region:
                edge_props = edge_attrs.get('properties', {})

                # 检查必要属性
                length_total = edge_props.get('length_total', edge_props.get('length', 0))
                it_len = edge_props.get('it_len', 0.0 if self.allow_missing else None)
                et_len = edge_props.get('et_len', 0.0 if self.allow_missing else None)
                ct_len = edge_props.get('ct_len', 0.0 if self.allow_missing else None)
                inh_len = edge_props.get('inh_len', 0.0 if self.allow_missing else None)
                n_axon = edge_props.get('n_axon', 0)

                # 添加到结果列表
                projections.append({
                    'source': source_region,
                    'target': target_region,
                    'length_total': length_total,
                    'it_len': it_len,
                    'et_len': et_len,
                    'ct_len': ct_len,
                    'inh_len': inh_len,
                    'n_axon': n_axon
                })

        # 转换为DataFrame
        df = pd.DataFrame(projections)
        if len(df) > 0:
            # 保存结果
            output_file = self.output_dir / f"{source_region}_to_{target_region}_projection.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存投射关系到: {output_file}")

            # 生成可视化
            self._plot_projection_composition(df, source_region, target_region)

        return df

    def _plot_projection_composition(self, df: pd.DataFrame, source_region: str, target_region: str):
        """为投射组成生成可视化"""
        if len(df) == 0:
            return

        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 8))

        # 提取投射类型长度
        row = df.iloc[0]

        # 检查是否有所有投射类型长度
        if any(pd.isna(row[col]) for col in ['it_len', 'et_len', 'ct_len', 'inh_len']):
            logger.warning(f"投射数据缺少必要的长度属性")
            return

        # 计算各类型占比
        lengths = [row['it_len'], row['et_len'], row['ct_len'], row['inh_len']]
        labels = ['IT', 'ET', 'CT', 'Inh']
        colors = ['skyblue', 'coral', 'lightgreen', 'plum']

        # 移除零值
        non_zero = [(label, length, color) for label, length, color in zip(labels, lengths, colors) if length > 0]
        if non_zero:
            labels, lengths, colors = zip(*non_zero)
        else:
            logger.warning(f"投射数据中所有长度都为0")
            return

        # 创建饼图
        wedges, texts, autotexts = ax.pie(
            lengths,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )

        # 调整文本
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        # 添加标题
        ax.set_title(f'{source_region} 到 {target_region} 投射组成\n总长度: {row["length_total"]:.1f} um')

        # 保存图表
        output_file = self.output_dir / f"{source_region}_to_{target_region}_projection_composition.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存投射组成可视化到: {output_file}")

    def query_gene_coexpression_morphology(self, gene_symbol: str) -> pd.DataFrame:
        """查询基因的共表达和形态学关联"""
        # 找到Gene节点
        gene_node_id = None
        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'Gene' in labels:
                props = node_attrs.get('properties', {})
                if props.get('symbol') == gene_symbol:
                    gene_node_id = node_id
                    break

        if gene_node_id is None:
            logger.warning(f"未找到Gene节点: {gene_symbol}")
            return pd.DataFrame()

        # 查找与该基因有COEXPRESSED关系的基因
        coexpressed_genes = []

        # 检查入边和出边
        for source_id, _, edge_attrs in list(self.graph.in_edges(gene_node_id, data=True)) + list(
                self.graph.out_edges(gene_node_id, data=True)):
            edge_type = edge_attrs.get('type')
            if edge_type == 'COEXPRESSED':
                edge_props = edge_attrs.get('properties', {})

                # 获取关联基因节点
                other_gene_attrs = self.graph.nodes[source_id]
                other_gene_props = other_gene_attrs.get('properties', {})
                other_gene = other_gene_props.get('symbol')

                # 添加到结果列表
                coexpressed_genes.append({
                    'gene': other_gene,
                    'rho': edge_props.get('rho', 0.0),
                    'fdr': edge_props.get('fdr', 1.0)
                })

        # 转换为DataFrame并排序
        df = pd.DataFrame(coexpressed_genes)
        if len(df) > 0:
            df = df.sort_values('rho', ascending=False)

            # 保存结果
            output_file = self.output_dir / f"{gene_symbol}_coexpression.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"保存共表达关系到: {output_file}")

            # 生成可视化
            self._plot_gene_coexpression(df, gene_symbol)

        return df

    def _plot_gene_coexpression(self, df: pd.DataFrame, gene_symbol: str):
        """为基因共表达关系生成可视化"""
        if len(df) == 0:
            return

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 获取前10个共表达基因
        plot_df = df.head(10).copy()

        # 创建条形图
        bars = ax.barh(plot_df['gene'], plot_df['rho'], color='skyblue')

        # 根据相关系数方向上色
        for i, v in enumerate(plot_df['rho']):
            if v < 0:
                bars[i].set_color('coral')

        # 添加标签
        ax.set_title(f'{gene_symbol} 共表达关系')
        ax.set_xlabel('Spearman相关系数 (rho)')
        ax.set_ylabel('基因')

        # 添加垂直零线
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

        # 保存图表
        output_file = self.output_dir / f"{gene_symbol}_coexpression.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存共表达可视化到: {output_file}")

    def validate_kg(self) -> Dict[str, Any]:
        """验证知识图谱数据"""
        validation_results = {
            'region_layer_nodes': 0,
            'region_layer_with_complete_props': 0,
            'has_subclass_rels': 0,
            'has_class_rels': 0,
            'projection_rels': 0,
            'projection_rels_with_types': 0,
            'missing_properties': [],
            'passed': True
        }

        # 检查RegionLayer节点
        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'RegionLayer' in labels:
                validation_results['region_layer_nodes'] += 1

                # 检查必要属性
                props = node_attrs.get('properties', {})
                missing_props = []

                # 必要属性列表
                required_props = [
                    'rl_id', 'region_name', 'layer', 'it_pct', 'et_pct', 'ct_pct',
                    'lr_pct', 'lr_prior', 'morph_ax_len_mean'
                ]

                for prop in required_props:
                    if prop not in props:
                        missing_props.append(prop)

                if not missing_props:
                    validation_results['region_layer_with_complete_props'] += 1
                else:
                    validation_results['missing_properties'].append({
                        'node_type': 'RegionLayer',
                        'node_id': node_id,
                        'rl_id': props.get('rl_id', 'unknown'),
                        'missing': missing_props
                    })

        # 检查HAS_SUBCLASS关系
        for source, target, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('type')
            if edge_type == 'HAS_SUBCLASS':
                validation_results['has_subclass_rels'] += 1

                # 检查必要属性
                props = attrs.get('properties', {})
                missing_props = []

                # 必要属性列表
                required_props = ['pct_cells', 'rank', 'n_cells', 'proj_type']

                for prop in required_props:
                    if prop not in props:
                        missing_props.append(prop)

                if missing_props:
                    validation_results['missing_properties'].append({
                        'edge_type': 'HAS_SUBCLASS',
                        'source': source,
                        'target': target,
                        'missing': missing_props
                    })

            elif edge_type == 'HAS_CLASS':
                validation_results['has_class_rels'] += 1

                # 检查必要属性
                props = attrs.get('properties', {})
                missing_props = []

                # 必要属性列表
                required_props = ['pct_cells', 'rank', 'n_cells']

                for prop in required_props:
                    if prop not in props:
                        missing_props.append(prop)

                if missing_props:
                    validation_results['missing_properties'].append({
                        'edge_type': 'HAS_CLASS',
                        'source': source,
                        'target': target,
                        'missing': missing_props
                    })

            elif edge_type == 'Project_to':
                validation_results['projection_rels'] += 1

                # 检查必要属性
                props = attrs.get('properties', {})
                missing_props = []

                # 必要属性列表
                required_props = ['length_total', 'it_len', 'et_len', 'ct_len', 'n_axon']

                for prop in required_props:
                    if prop not in props:
                        missing_props.append(prop)

                if not missing_props:
                    validation_results['projection_rels_with_types'] += 1
                else:
                    validation_results['missing_properties'].append({
                        'edge_type': 'Project_to',
                        'source': source,
                        'target': target,
                        'missing': missing_props
                    })

        # 检查验证是否通过
        if validation_results['region_layer_nodes'] == 0:
            validation_results['passed'] = False
            logger.error("验证失败：无RegionLayer节点")
        elif validation_results['region_layer_with_complete_props'] / validation_results['region_layer_nodes'] < 0.95:
            validation_results['passed'] = False
            logger.error(
                f"验证失败：只有{validation_results['region_layer_with_complete_props']}/{validation_results['region_layer_nodes']}个RegionLayer节点具有完整属性")

        if validation_results['has_subclass_rels'] == 0:
            validation_results['passed'] = False
            logger.error("验证失败：无HAS_SUBCLASS关系")

        if validation_results['has_class_rels'] == 0:
            validation_results['passed'] = False
            logger.error("验证失败：无HAS_CLASS关系")

        if validation_results['projection_rels'] > 0 and validation_results['projection_rels_with_types'] / \
                validation_results['projection_rels'] < 0.9:
            validation_results['passed'] = False
            logger.error(
                f"验证失败：只有{validation_results['projection_rels_with_types']}/{validation_results['projection_rels']}个Project_to关系具有完整属性")

        # 保存验证结果
        output_file = self.output_dir / "validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"保存验证结果到: {output_file}")

        return validation_results


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='知识图谱查询和验证工具')
    parser.add_argument('kg_path', help='知识图谱文件路径（JSON、CSV目录或Neo4j URI）')
    parser.add_argument('--output-dir', '-o', help='输出文件夹路径')
    parser.add_argument('--allow-missing', '-a', action='store_true', help='允许缺失属性（使用默认值代替）')
    parser.add_argument('--validate', '-v', action='store_true', help='验证知识图谱结构')
    parser.add_argument('--query-region', '-r', help='查询特定区域的RegionLayer属性')
    parser.add_argument('--query-layer', '-l', nargs=2, metavar=('REGION', 'LAYER'),
                        help='查询特定RegionLayer的细胞类型分布')
    parser.add_argument('--query-projection', '-p', nargs=2, metavar=('SOURCE', 'TARGET'),
                        help='查询两个区域之间的投射关系')
    parser.add_argument('--query-gene', '-g', help='查询基因的共表达和形态学关联')

    args = parser.parse_args()

    try:
        # 初始化查询验证器
        validator = KGQueryValidator(args.kg_path, args.output_dir, args.allow_missing)

        # 验证知识图谱
        if args.validate:
            results = validator.validate_kg()
            print("\n验证结果:")
            print(f"RegionLayer节点: {results['region_layer_nodes']}")
            print(f"属性完整的RegionLayer节点: {results['region_layer_with_complete_props']}")
            print(f"HAS_SUBCLASS关系: {results['has_subclass_rels']}")
            print(f"HAS_CLASS关系: {results['has_class_rels']}")
            print(f"Project_to关系: {results['projection_rels']}")
            print(f"具有投射类型的Project_to关系: {results['projection_rels_with_types']}")
            print(f"验证通过: {'是' if results['passed'] else '否'}")

            if not results['passed']:
                sys.exit(1)

        # 执行查询
        if args.query_region:
            df = validator.query_region_layer_properties(args.query_region)
            print(f"\n区域{args.query_region}的RegionLayer属性:")
            print(df.to_string(index=False))

        if args.query_layer:
            region, layer = args.query_layer
            df = validator.query_transcriptomic_distribution(region, layer)
            print(f"\n{region} {layer}的细胞类型分布:")
            print(df.to_string(index=False))

        if args.query_projection:
            source, target = args.query_projection
            df = validator.query_projections_between_regions(source, target)
            print(f"\n{source}到{target}的投射关系:")
            print(df.to_string(index=False))

        if args.query_gene:
            df = validator.query_gene_coexpression_morphology(args.query_gene)
            print(f"\n基因{args.query_gene}的共表达关系:")
            print(df.to_string(index=False))

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()