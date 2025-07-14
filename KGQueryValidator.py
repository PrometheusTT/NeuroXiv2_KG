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
from scipy import stats

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
                WHERE r:Region OR r:RegionLayer OR s:RegionLayer OR s:Class OR s:Subclass OR s:Cluster OR r:Gene OR s:Gene
                OR type(rel) = 'PROJECT_TO' OR type(rel) = 'HAS_LAYER' OR type(rel) = 'HAS_CLASS' OR type(rel) = 'HAS_SUBCLASS'
                OR type(rel) = 'COEXPRESSED'
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
                if rel_type not in ['Project_to', 'HAS_LAYER', 'HAS_CLASS', 'HAS_SUBCLASS', 'HAS_CLUSTER',
                                    'COEXPRESSED']:
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
            # 直接加载整个JSON文件 - 对单一文件更简单和高效
            with open(self.kg_path, 'r') as f:
                kg_data = json.load(f)

            # 先处理所有节点
            for obj in kg_data:
                if obj.get('type') == 'node':
                    node_id = obj.get('id')
                    labels = obj.get('labels', [])

                    # 只保留我们需要的节点类型
                    keep_node = any(
                        label in ['Region', 'RegionLayer', 'Class', 'Subclass', 'Cluster', 'Gene'] for label in labels)
                    if keep_node:
                        properties = obj.get('properties', {})
                        self.graph.add_node(node_id, labels=labels, properties=properties)

            # 然后处理所有关系
            for obj in kg_data:
                if obj.get('type') == 'relationship':
                    rel_type = obj.get('label')

                    # 处理start和end属性
                    start_info = obj.get('start')
                    end_info = obj.get('end')

                    # 获取节点ID
                    if isinstance(start_info, dict):
                        start_id = start_info.get('id')
                    else:
                        start_id = start_info

                    if isinstance(end_info, dict):
                        end_id = end_info.get('id')
                    else:
                        end_id = end_info

                    # 只保留我们需要的关系类型且两端节点存在
                    if rel_type in ['Project_to', 'HAS_LAYER', 'HAS_CLASS', 'HAS_SUBCLASS', 'HAS_CLUSTER',
                                    'COEXPRESSED'] and \
                            start_id in self.graph.nodes and end_id in self.graph.nodes:
                        properties = obj.get('properties', {})
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

                # 收集基因表达属性 (任务④)
                fezf2_expr = props.get('mean_logCPM_Fezf2',
                                       props.get('fezf2_module_mean', 0.0 if self.allow_missing else None))

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
                    'fezf2_expression': fezf2_expr,
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

    def kruskal_wallis_test(self, df: pd.DataFrame, groupby_col: str, value_col: str) -> Dict[str, Any]:
        """执行Kruskal-Wallis H检验"""
        # 检查是否有足够的分组
        groups = df[groupby_col].unique()
        if len(groups) < 2:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'groups': groups,
                'valid': False,
                'message': f"需要至少2个分组，但只找到{len(groups)}个"
            }

        # 收集各组数据
        samples = []
        for group in groups:
            group_data = df[df[groupby_col] == group][value_col].dropna()
            if len(group_data) > 0:
                samples.append(group_data)

        # 执行检验
        if len(samples) >= 2:
            statistic, pvalue = stats.kruskal(*samples)
            return {
                'statistic': statistic,
                'pvalue': pvalue,
                'groups': groups,
                'valid': True,
                'significant': pvalue < 0.05,
                'message': f"Kruskal-Wallis H = {statistic:.4f}, p = {pvalue:.4f}"
            }
        else:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'groups': groups,
                'valid': False,
                'message': "有效分组不足，无法执行检验"
            }

    def mann_whitney_u_test(self, data1: pd.Series, data2: pd.Series, label1: str = "Group 1",
                            label2: str = "Group 2") -> Dict[str, Any]:
        """执行Mann-Whitney U检验"""
        # 清除空值
        data1 = data1.dropna()
        data2 = data2.dropna()

        # 检查数据量是否足够
        if len(data1) < 3 or len(data2) < 3:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'valid': False,
                'message': f"数据量不足 ({len(data1)} vs {len(data2)})，需要至少3个样本"
            }

        # 执行检验
        statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'group1': label1,
            'group2': label2,
            'n1': len(data1),
            'n2': len(data2),
            'mean1': data1.mean(),
            'mean2': data2.mean(),
            'valid': True,
            'significant': pvalue < 0.05,
            'message': f"Mann-Whitney U = {statistic:.4f}, p = {pvalue:.4f}"
        }

    def spearman_correlation_test(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """执行Spearman相关性检验"""
        # 清除空值
        clean_data = pd.DataFrame({'x': x, 'y': y}).dropna()

        # 检查数据量是否足够
        if len(clean_data) < 5:
            return {
                'rho': np.nan,
                'pvalue': np.nan,
                'valid': False,
                'message': f"数据量不足，只有{len(clean_data)}个有效样本，需要至少5个"
            }

        # 执行检验
        rho, pvalue = stats.spearmanr(clean_data['x'], clean_data['y'])
        return {
            'rho': rho,
            'pvalue': pvalue,
            'n': len(clean_data),
            'valid': True,
            'significant': pvalue < 0.05,
            'message': f"Spearman rho = {rho:.4f}, p = {pvalue:.4f}"
        }

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

    def it_et_ct_axon_length_analysis(self) -> pd.DataFrame:
        """
        任务①：分析不同投射类型的轴突长度（IT、ET、CT）

        Returns:
            轴突长度分析结果DataFrame
        """
        logger.info("执行投射类型轴突长度分析...")

        # 找到所有L5 RegionLayer节点
        l5_layers = []
        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            props = node_attrs.get('properties', {})

            if 'RegionLayer' in labels and props.get('layer') == 'L5':
                # 提取关键属性
                l5_layers.append({
                    'rl_id': props.get('rl_id'),
                    'region_name': props.get('region_name'),
                    'layer': 'L5',
                    'it_pct': props.get('it_pct', 0.0),
                    'et_pct': props.get('et_pct', 0.0),
                    'ct_pct': props.get('ct_pct', 0.0),
                    'morph_ax_len_mean': props.get('morph_ax_len_mean', 0.0)
                })

        # 转换为DataFrame
        df = pd.DataFrame(l5_layers)

        if len(df) == 0:
            logger.warning("未找到L5 RegionLayer数据，无法执行分析")
            return pd.DataFrame()

        # 创建投射类型列，基于最高百分比
        df['proj_type'] = df.apply(
            lambda row: 'IT' if row['it_pct'] >= max(row['et_pct'], row['ct_pct']) else
            'ET' if row['et_pct'] >= row['ct_pct'] else 'CT',
            axis=1
        )

        # 执行Kruskal-Wallis检验
        test_result = self.kruskal_wallis_test(df, 'proj_type', 'morph_ax_len_mean')

        # 创建结果可视化
        self._plot_axon_length_by_projtype(df)

        # 输出检验结果
        output_file = self.output_dir / "it_et_ct_axon_length.csv"
        df.to_csv(output_file, index=False)

        # 保存统计结果
        stats_file = self.output_dir / "it_et_ct_axon_length_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(test_result, f, indent=2)

        logger.info(f"投射类型轴突长度分析完成，结果保存到: {output_file}")
        logger.info(f"统计检验结果: {test_result['message']}")

        return df

    def _plot_axon_length_by_projtype(self, df: pd.DataFrame):
        """为不同投射类型的轴突长度创建箱线图"""
        if len(df) == 0:
            return

        # 创建箱线图
        plt.figure(figsize=(10, 6))

        # 使用seaborn风格
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
            ax = sns.boxplot(x='proj_type', y='morph_ax_len_mean', data=df)
            sns.stripplot(x='proj_type', y='morph_ax_len_mean', data=df,
                          color='black', alpha=0.5, jitter=True)
        except ImportError:
            # 如果没有seaborn，使用matplotlib
            boxplot = plt.boxplot([df[df['proj_type'] == pt]['morph_ax_len_mean']
                                   for pt in ['IT', 'ET', 'CT'] if len(df[df['proj_type'] == pt]) > 0],
                                  labels=[pt for pt in ['IT', 'ET', 'CT'] if len(df[df['proj_type'] == pt]) > 0])

        plt.title('不同投射类型的轴突长度比较')
        plt.xlabel('投射类型')
        plt.ylabel('平均轴突长度 (μm)')

        # 添加样本数标注
        for i, pt in enumerate(['IT', 'ET', 'CT']):
            n = len(df[df['proj_type'] == pt])
            if n > 0:
                plt.annotate(f'n = {n}', xy=(i, df[df['proj_type'] == pt]['morph_ax_len_mean'].min()),
                             ha='center', va='top', xytext=(0, -20), textcoords='offset points')

        # 保存图表
        output_file = self.output_dir / "it_et_ct_axon_length.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"保存投射类型轴突长度可视化到: {output_file}")

    def vip_dendrite_polarity_analysis(self) -> pd.DataFrame:
        """
        任务②：分析VIP细胞在上层vs下层的树突极性差异

        Returns:
            VIP细胞树突极性分析结果DataFrame
        """
        logger.info("执行VIP细胞树突极性分析...")

        # 找到所有包含VIP亚类的RegionLayer节点
        vip_data = []

        # 查找所有HAS_SUBCLASS关系，寻找包含VIP的亚类
        for source_id, target_id, edge_attrs in self.graph.edges(data=True):
            if edge_attrs.get('type') == 'HAS_SUBCLASS':
                # 获取目标亚类节点
                target_attrs = self.graph.nodes[target_id]
                target_props = target_attrs.get('properties', {})

                # 检查是否是VIP亚类
                subclass_name = target_props.get('name', '')
                if 'Vip' in subclass_name:
                    # 获取源RegionLayer节点
                    source_attrs = self.graph.nodes[source_id]
                    source_props = source_attrs.get('properties', {})

                    # 提取关键属性
                    region_name = source_props.get('region_name')
                    layer = source_props.get('layer')
                    dend_polarity = source_props.get('dend_polarity_index_mean', 0.0)

                    # 计算层类别（上层/下层）
                    if layer in ['L1', 'L2/3', 'L4']:
                        layer_group = 'upper'
                    elif layer in ['L5', 'L6', 'L6b']:
                        layer_group = 'deep'
                    else:
                        layer_group = 'other'

                    # 收集亚类信息
                    edge_props = edge_attrs.get('properties', {})
                    pct_cells = edge_props.get('pct_cells', 0.0)

                    # 添加数据点
                    vip_data.append({
                        'region_name': region_name,
                        'layer': layer,
                        'layer_group': layer_group,
                        'subclass': subclass_name,
                        'dend_polarity_index': dend_polarity,
                        'pct_cells': pct_cells
                    })

        # 转换为DataFrame
        df = pd.DataFrame(vip_data)

        if len(df) == 0:
            logger.warning("未找到VIP亚类数据，无法执行分析")
            return pd.DataFrame()

        # 只保留upper和deep层组
        df = df[df['layer_group'].isin(['upper', 'deep'])]

        # 执行Mann-Whitney U检验
        upper_polarity = df[df['layer_group'] == 'upper']['dend_polarity_index']
        deep_polarity = df[df['layer_group'] == 'deep']['dend_polarity_index']

        test_result = self.mann_whitney_u_test(
            upper_polarity, deep_polarity,
            label1="Upper layers (L1-L4)", label2="Deep layers (L5-L6b)"
        )

        # 创建结果可视化
        self._plot_vip_dendrite_polarity(df)

        # 输出检验结果
        output_file = self.output_dir / "vip_dendrite_polarity.csv"
        df.to_csv(output_file, index=False)

        # 保存统计结果
        stats_file = self.output_dir / "vip_dendrite_polarity_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(test_result, f, indent=2)

        logger.info(f"VIP细胞树突极性分析完成，结果保存到: {output_file}")
        logger.info(f"统计检验结果: {test_result['message']}")

        return df

    def _plot_vip_dendrite_polarity(self, df: pd.DataFrame):
        """为VIP细胞树突极性创建箱线图"""
        if len(df) == 0:
            return

        # 创建箱线图
        plt.figure(figsize=(10, 6))

        # 使用seaborn风格
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
            ax = sns.boxplot(x='layer_group', y='dend_polarity_index', data=df,
                             order=['upper', 'deep'])
            sns.stripplot(x='layer_group', y='dend_polarity_index', data=df,
                          color='black', alpha=0.5, jitter=True, order=['upper', 'deep'])
        except ImportError:
            # 如果没有seaborn，使用matplotlib
            boxplot = plt.boxplot([df[df['layer_group'] == g]['dend_polarity_index']
                                   for g in ['upper', 'deep'] if len(df[df['layer_group'] == g]) > 0],
                                  labels=['Upper layers (L1-L4)', 'Deep layers (L5-L6b)'])

        plt.title('VIP细胞在上层vs下层的树突极性比较')
        plt.xlabel('皮层层位置')
        plt.ylabel('树突极性指数')

        # 将x轴标签改为更可读的形式
        plt.xticks([0, 1], ['Upper layers (L1-L4)', 'Deep layers (L5-L6b)'])

        # 添加样本数标注
        for i, g in enumerate(['upper', 'deep']):
            n = len(df[df['layer_group'] == g])
            if n > 0:
                plt.annotate(f'n = {n}', xy=(i, df[df['layer_group'] == g]['dend_polarity_index'].min()),
                             ha='center', va='top', xytext=(0, -20), textcoords='offset points')

        # 保存图表
        output_file = self.output_dir / "vip_dendrite_polarity.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"保存VIP细胞树突极性可视化到: {output_file}")

    def query_rare_subtype_locator(self, subtype_name: str = "Sst-Chodl") -> pd.DataFrame:
        """
        任务③：定位稀有亚型 (例如Sst-Chodl) 在不同皮层层的分布

        Args:
            subtype_name: 要查找的稀有亚型名称，默认为"Sst-Chodl"

        Returns:
            稀有亚型分布结果DataFrame
        """
        logger.info(f"查找稀有亚型 {subtype_name} 的分布...")

        # 收集亚型在各层的分布数据
        subtype_distribution = []

        # 查找所有HAS_SUBCLASS或HAS_CLUSTER关系，寻找指定亚型
        for source_id, target_id, edge_attrs in self.graph.edges(data=True):
            if edge_attrs.get('type') in ['HAS_SUBCLASS', 'HAS_CLUSTER']:
                # 获取目标亚类/簇节点
                target_attrs = self.graph.nodes[target_id]
                target_props = target_attrs.get('properties', {})

                # 检查是否是指定亚型
                node_name = target_props.get('name', '')
                if subtype_name.lower() in node_name.lower():
                    # 获取源RegionLayer节点
                    source_attrs = self.graph.nodes[source_id]
                    source_props = source_attrs.get('properties', {})

                    # 提取关键属性
                    region_name = source_props.get('region_name')
                    layer = source_props.get('layer')
                    morph_ax_len = source_props.get('morph_ax_len_mean', 0.0)
                    lr_pct = source_props.get('lr_pct', 0.0)

                    # 收集亚型分布信息
                    edge_props = edge_attrs.get('properties', {})
                    pct_cells = edge_props.get('pct_cells', 0.0)
                    n_cells = edge_props.get('n_cells', 0)

                    # 添加数据点
                    subtype_distribution.append({
                        'region_name': region_name,
                        'layer': layer,
                        'subtype': node_name,
                        'pct_cells': pct_cells,
                        'n_cells': n_cells,
                        'morph_ax_len_mean': morph_ax_len,
                        'lr_pct': lr_pct
                    })

        # 转换为DataFrame
        df = pd.DataFrame(subtype_distribution)

        if len(df) == 0:
            logger.warning(f"未找到亚型 {subtype_name} 的分布数据")
            return pd.DataFrame()

        # 排序
        df = df.sort_values(['region_name', 'layer', 'pct_cells'], ascending=[True, True, False])

        # 输出结果
        output_file = self.output_dir / f"{subtype_name.replace('-', '_').lower()}_distribution.csv"
        df.to_csv(output_file, index=False)

        # 创建结果可视化
        self._plot_subtype_distribution(df, subtype_name)

        logger.info(f"稀有亚型 {subtype_name} 分布分析完成，结果保存到: {output_file}")

        return df

    def _plot_subtype_distribution(self, df: pd.DataFrame, subtype_name: str):
        """为稀有亚型分布创建可视化"""
        if len(df) == 0:
            return

        # 1. 按区域分布热图
        plt.figure(figsize=(12, 8))

        # 创建区域-层级透视表
        try:
            pivot_df = df.pivot_table(
                values='pct_cells',
                index='region_name',
                columns='layer',
                aggfunc='mean'
            )

            # 调整层顺序
            layer_order = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b']
            available_layers = [l for l in layer_order if l in pivot_df.columns]
            pivot_df = pivot_df[available_layers]

            # 绘制热图
            try:
                import seaborn as sns
                ax = sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".3f",
                                 cbar_kws={'label': '细胞占比'})
            except ImportError:
                # 使用matplotlib
                im = plt.imshow(pivot_df.values, cmap="YlGnBu")
                plt.colorbar(im, label='细胞占比')

                # 添加标注
                for i in range(len(pivot_df.index)):
                    for j in range(len(pivot_df.columns)):
                        value = pivot_df.iloc[i, j]
                        plt.text(j, i, f"{value:.3f}", ha="center", va="center", color="black")

                plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
                plt.yticks(range(len(pivot_df.index)), pivot_df.index)

            plt.title(f'{subtype_name} 亚型在各区域各层的分布')
            plt.tight_layout()

            # 保存热图
            output_file = self.output_dir / f"{subtype_name.replace('-', '_').lower()}_distribution_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"创建热图失败: {e}")

        # 2. 按层分布条形图
        try:
            plt.figure(figsize=(10, 6))

            # 按层分组计算平均值
            layer_avg = df.groupby('layer')['pct_cells'].mean().reindex(layer_order)
            layer_std = df.groupby('layer')['pct_cells'].std().reindex(layer_order)

            # 过滤有数据的层
            valid_layers = layer_avg.dropna().index

            # 创建条形图
            plt.bar(valid_layers, layer_avg[valid_layers], yerr=layer_std[valid_layers],
                    color='skyblue', capsize=5)

            plt.title(f'{subtype_name} 亚型在各层的平均分布')
            plt.xlabel('皮层层')
            plt.ylabel('平均细胞比例')

            # 添加样本数标注
            for i, layer in enumerate(valid_layers):
                n = len(df[df['layer'] == layer])
                plt.annotate(f'n = {n}', xy=(i, layer_avg[layer] + layer_std[layer]),
                             ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

            plt.tight_layout()

            # 保存条形图
            output_file = self.output_dir / f"{subtype_name.replace('-', '_').lower()}_distribution_by_layer.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"创建条形图失败: {e}")

        logger.info(f"保存{subtype_name}分布可视化到输出目录")

    def fezf2_module_axon_length_analysis(self) -> pd.DataFrame:
        """
        任务④：分析Fezf2模块表达与轴突长度的关系

        Returns:
            Fezf2模块表达与轴突长度关系的DataFrame
        """
        logger.info("执行Fezf2模块表达与轴突长度关系分析...")

        # 找到所有有Fezf2表达数据的RegionLayer节点
        fezf2_data = []

        for node_id, node_attrs in self.graph.nodes(data=True):
            labels = node_attrs.get('labels', [])
            if 'RegionLayer' in labels:
                props = node_attrs.get('properties', {})

                # 检查是否有Fezf2表达数据
                fezf2_expr_key = None
                for key in props:
                    if key == 'mean_logCPM_Fezf2' or key == 'fezf2_module_mean':
                        fezf2_expr_key = key
                        break

                if fezf2_expr_key:
                    # 提取关键属性
                    region_name = props.get('region_name')
                    layer = props.get('layer')
                    fezf2_expr = props.get(fezf2_expr_key, 0.0)
                    morph_ax_len = props.get('morph_ax_len_mean', 0.0)

                    # 只保留有效数据点
                    if pd.notna(fezf2_expr) and pd.notna(morph_ax_len) and morph_ax_len > 0:
                        fezf2_data.append({
                            'rl_id': props.get('rl_id'),
                            'region_name': region_name,
                            'layer': layer,
                            'fezf2_expression': fezf2_expr,
                            'morph_ax_len_mean': morph_ax_len
                        })

        # 转换为DataFrame
        df = pd.DataFrame(fezf2_data)

        if len(df) < 5:
            logger.warning("Fezf2表达数据点不足，无法进行相关性分析")
            return pd.DataFrame()

        # 执行Spearman相关性检验
        test_result = self.spearman_correlation_test(
            df['fezf2_expression'], df['morph_ax_len_mean']
        )

        # 创建结果可视化
        self._plot_fezf2_axon_correlation(df)

        # 输出检验结果
        output_file = self.output_dir / "fezf2_axon_length.csv"
        df.to_csv(output_file, index=False)

        # 保存统计结果
        stats_file = self.output_dir / "fezf2_axon_length_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(test_result, f, indent=2)

        logger.info(f"Fezf2模块表达与轴突长度关系分析完成，结果保存到: {output_file}")
        logger.info(f"相关性检验结果: {test_result['message']}")

        return df

    def _plot_fezf2_axon_correlation(self, df: pd.DataFrame):
        """绘制Fezf2表达与轴突长度的散点图和相关性"""
        if len(df) < 5:
            return

        plt.figure(figsize=(10, 8))

        # 绘制散点图
        try:
            import seaborn as sns
            sns.set_style("whitegrid")

            # 按层着色的散点图
            ax = sns.scatterplot(
                data=df,
                x='fezf2_expression',
                y='morph_ax_len_mean',
                hue='layer',
                palette='viridis',
                s=80,
                alpha=0.7
            )

            # 添加回归线
            sns.regplot(
                data=df,
                x='fezf2_expression',
                y='morph_ax_len_mean',
                scatter=False,
                color='red'
            )

        except ImportError:
            # 使用matplotlib
            plt.scatter(df['fezf2_expression'], df['morph_ax_len_mean'], alpha=0.7)

            # 添加回归线
            z = np.polyfit(df['fezf2_expression'], df['morph_ax_len_mean'], 1)
            p = np.poly1d(z)
            plt.plot(np.sort(df['fezf2_expression']),
                     p(np.sort(df['fezf2_expression'])),
                     "r--", alpha=0.7)

        # 计算相关系数和p值
        rho, p = stats.spearmanr(df['fezf2_expression'], df['morph_ax_len_mean'])

        plt.title(f'Fezf2表达与轴突长度的相关性\nSpearman ρ = {rho:.3f}, p = {p:.3e}')
        plt.xlabel('Fezf2模块表达 (log CPM)')
        plt.ylabel('平均轴突长度 (μm)')

        # 保存图表
        output_file = self.output_dir / "fezf2_axon_correlation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"保存Fezf2-轴突长度相关性可视化到: {output_file}")

        # 创建按层分组的箱线图
        try:
            plt.figure(figsize=(12, 6))

            # 创建分组
            median_expr = df['fezf2_expression'].median()
            df['fezf2_group'] = df['fezf2_expression'].apply(
                lambda x: 'High Fezf2' if x > median_expr else 'Low Fezf2'
            )

            # 使用seaborn绘制
            try:
                sns.set_style("whitegrid")
                ax = sns.boxplot(x='layer', y='morph_ax_len_mean', hue='fezf2_group', data=df)

                # 添加数据点
                sns.stripplot(x='layer', y='morph_ax_len_mean', hue='fezf2_group',
                              data=df, dodge=True, size=4, alpha=0.6, jitter=True)

                # 修复图例
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles[:2], labels[:2], title="Fezf2表达组")

            except (ImportError, NameError):
                # 使用matplotlib
                layers = df['layer'].unique()
                x_pos = np.arange(len(layers))
                width = 0.35

                # 分组
                high_df = df[df['fezf2_group'] == 'High Fezf2']
                low_df = df[df['fezf2_group'] == 'Low Fezf2']

                high_means = [high_df[high_df['layer'] == l]['morph_ax_len_mean'].mean() for l in layers]
                low_means = [low_df[low_df['layer'] == l]['morph_ax_len_mean'].mean() for l in layers]

                plt.bar(x_pos - width / 2, high_means, width, label='High Fezf2', color='coral')
                plt.bar(x_pos + width / 2, low_means, width, label='Low Fezf2', color='skyblue')

                plt.xticks(x_pos, layers)
                plt.legend()

            plt.title('按皮层层和Fezf2表达水平分组的轴突长度')
            plt.xlabel('皮层层')
            plt.ylabel('平均轴突长度 (μm)')
            plt.tight_layout()

            # 保存图表
            output_file = self.output_dir / "fezf2_axon_by_layer.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"创建按层分组的箱线图失败: {e}")

        def query_it_projection_flow(self) -> pd.DataFrame:
            """
            任务⑤：分析IT型L2/3细胞的主要投射靶点

            Returns:
                IT投射流DataFrame
            """
            logger.info("分析IT型L2/3细胞的投射流...")

            # 收集所有L2/3层的RegionLayer节点ID和它们的IT百分比
            l23_it_layers = {}

            for node_id, node_attrs in self.graph.nodes(data=True):
                labels = node_attrs.get('labels', [])
                props = node_attrs.get('properties', {})

                if 'RegionLayer' in labels and props.get('layer') == 'L2/3':
                    region_name = props.get('region_name')
                    it_pct = props.get('it_pct', 0.0)

                    # 记录L2/3层的IT百分比
                    if region_name and it_pct > 0:
                        l23_it_layers[node_id] = {
                            'region_name': region_name,
                            'it_pct': it_pct
                        }

            # 寻找这些区域的投射目标
            it_projections = []

            # 找到所有源区域对应的Region节点
            region_nodes = {}
            for node_id, node_attrs in self.graph.nodes(data=True):
                labels = node_attrs.get('labels', [])
                props = node_attrs.get('properties', {})

                if 'Region' in labels:
                    region_name = props.get('name')
                    if region_name:
                        region_nodes[region_name] = node_id

            # 查找所有投射关系
            for source_id, target_id, edge_attrs in self.graph.edges(data=True):
                if edge_attrs.get('type') == 'Project_to':
                    edge_props = edge_attrs.get('properties', {})

                    # 获取源节点和目标节点
                    source_attrs = self.graph.nodes[source_id]
                    target_attrs = self.graph.nodes[target_id]

                    source_props = source_attrs.get('properties', {})
                    target_props = target_attrs.get('properties', {})

                    source_region = source_props.get('name')
                    target_region = target_props.get('name')

                    # 检查源区域是否有L2/3层的IT细胞
                    if source_region in [info['region_name'] for info in l23_it_layers.values()]:
                        # 提取IT轴突长度
                        it_len = edge_props.get('it_len', 0.0)
                        total_len = edge_props.get('length_total', edge_props.get('length', 0.0))

                        # 如果有IT投射，记录数据
                        if it_len > 0:
                            # 找到对应的L2/3层IT百分比
                            it_pct = 0
                            for info in l23_it_layers.values():
                                if info['region_name'] == source_region:
                                    it_pct = info['it_pct']
                                    break

                            it_projections.append({
                                'source_region': source_region,
                                'target_region': target_region,
                                'it_len': it_len,
                                'total_len': total_len,
                                'it_pct': it_pct,
                                'it_weight': it_len * it_pct  # 加权投射强度
                            })

            # 转换为DataFrame
            df = pd.DataFrame(it_projections)

            if len(df) == 0:
                logger.warning("未找到IT型L2/3细胞的投射数据")
                return pd.DataFrame()

            # 按源区域分组，计算每个目标区域的投射比例
            result_rows = []
            for source, group in df.groupby('source_region'):
                # 按目标区域汇总IT投射
                target_sums = group.groupby('target_region')['it_weight'].sum().reset_index()

                # 计算总投射
                total_weight = target_sums['it_weight'].sum()

                # 计算每个目标的比例
                if total_weight > 0:
                    for _, row in target_sums.iterrows():
                        target = row['target_region']
                        weight = row['it_weight']

                        result_rows.append({
                            'source_region': source,
                            'target_region': target,
                            'it_projection_weight': weight,
                            'it_projection_proportion': weight / total_weight
                        })

            # 创建结果DataFrame
            result_df = pd.DataFrame(result_rows)

            if len(result_df) > 0:
                # 排序
                result_df = result_df.sort_values(['source_region', 'it_projection_proportion'],
                                                  ascending=[True, False])

                # 输出结果
                output_file = self.output_dir / "it_l23_projection_flow.csv"
                result_df.to_csv(output_file, index=False)

                # 创建可视化
                self._plot_it_projection_flow(result_df)

                logger.info(f"IT型L2/3细胞投射流分析完成，结果保存到: {output_file}")

            return result_df

        def _plot_it_projection_flow(self, df: pd.DataFrame):
            """为IT型L2/3投射流创建Sankey图和条形图"""
            if len(df) == 0:
                return

            # 1. 创建条形图 - 为每个源区域显示主要投射靶点
            for source, group in df.groupby('source_region'):
                try:
                    plt.figure(figsize=(12, 6))

                    # 排序并获取前10个目标区域
                    top_targets = group.sort_values('it_projection_proportion', ascending=False).head(10)

                    # 创建条形图
                    plt.bar(top_targets['target_region'], top_targets['it_projection_proportion'], color='skyblue')

                    plt.title(f'{source} L2/3 IT细胞的主要投射靶点')
                    plt.xlabel('目标区域')
                    plt.ylabel('投射比例')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # 保存图表
                    output_file = self.output_dir / f"{source}_it_l23_projection_targets.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()

                except Exception as e:
                    logger.warning(f"创建{source}的投射目标条形图失败: {e}")

            # 2. 尝试创建Sankey图
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # 准备Sankey数据
                source_labels = df['source_region'].unique().tolist()
                target_labels = df['target_region'].unique().tolist()

                # 创建标签映射
                all_labels = source_labels + target_labels
                label_dict = {label: i for i, label in enumerate(all_labels)}

                # 准备Sankey数据
                sources = [label_dict[row['source_region']] for _, row in df.iterrows()]
                targets = [label_dict[row['target_region']] for _, row in df.iterrows()]
                values = df['it_projection_weight'].tolist()

                # 创建Sankey图
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_labels,
                        color="blue"
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color="rgba(100, 100, 200, 0.3)"
                    )
                )])

                fig.update_layout(
                    title_text="L2/3 IT细胞投射网络",
                    font_size=12
                )

                # 保存为HTML和图片
                html_output = self.output_dir / "it_l23_projection_network.html"
                fig.write_html(str(html_output))

                png_output = self.output_dir / "it_l23_projection_network.png"
                fig.write_image(str(png_output), width=1200, height=800)

                logger.info(f"保存Sankey图到: {html_output} 和 {png_output}")

            except ImportError:
                logger.warning("未安装plotly，无法创建Sankey图")
            except Exception as e:
                logger.warning(f"创建Sankey图失败: {e}")

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
                'gene_nodes': 0,
                'coexpressed_rels': 0,
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

                elif 'Gene' in labels:
                    validation_results['gene_nodes'] += 1

            # 检查关系
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

                elif edge_type == 'COEXPRESSED':
                    validation_results['coexpressed_rels'] += 1

            # 检查验证是否通过
            if validation_results['region_layer_nodes'] == 0:
                validation_results['passed'] = False
                logger.error("验证失败：无RegionLayer节点")
            elif validation_results['region_layer_with_complete_props'] / validation_results[
                'region_layer_nodes'] < 0.95:
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

            # 检查任务④需要的基因共表达关系
            if validation_results['coexpressed_rels'] == 0:
                logger.warning("警告：没有找到基因共表达关系，任务④可能无法运行")

            # 保存验证结果
            output_file = self.output_dir / "validation_results.json"
            with open(output_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"保存验证结果到: {output_file}")

            return validation_results

        def run_all_tasks(self):
            """运行所有五个演练任务，生成完整结果"""
            logger.info("开始运行全部演练任务...")

            # 任务①：分析不同投射类型的轴突长度（IT、ET、CT）
            logger.info("执行任务①: IT-ET-CT长程通路轴突长度分析")
            task1_result = self.it_et_ct_axon_length_analysis()

            # 任务②：分析VIP细胞在上层vs下层的树突极性差异
            logger.info("执行任务②: VIP细胞上层vs下层树突极性分析")
            task2_result = self.vip_dendrite_polarity_analysis()

            # 任务③：定位稀有亚型Sst-Chodl
            logger.info("执行任务③: Sst-Chodl稀有亚型定位")
            task3_result = self.query_rare_subtype_locator("Sst-Chodl")

            # 任务④：分析Fezf2模块表达与轴突长度的关系
            logger.info("执行任务④: Fezf2模块与轴突长度相关性分析")
            task4_result = self.fezf2_module_axon_length_analysis()

            # 任务⑤：分析IT型L2/3细胞的主要投射靶点
            logger.info("执行任务⑤: IT型L2/3细胞投射靶点分析")
            task5_result = self.query_it_projection_flow()

            # 生成汇总报告
            summary = {
                "task1_completed": not task1_result.empty,
                "task2_completed": not task2_result.empty,
                "task3_completed": not task3_result.empty,
                "task4_completed": not task4_result.empty,
                "task5_completed": not task5_result.empty,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            summary_file = self.output_dir / "tasks_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"所有任务执行完成，结果保存在: {self.output_dir}")
            logger.info(f"任务完成情况: 任务① {'完成' if not task1_result.empty else '失败'}, " +
                        f"任务② {'完成' if not task2_result.empty else '失败'}, " +
                        f"任务③ {'完成' if not task3_result.empty else '失败'}, " +
                        f"任务④ {'完成' if not task4_result.empty else '失败'}, " +
                        f"任务⑤ {'完成' if not task5_result.empty else '失败'}")

            return summary

    # Add these functions to KGQueryValidator.py - they will properly implement
    # the statistical tests needed for all 5 tasks

    def kruskal_wallis_test(self, df: pd.DataFrame, groupby_col: str, value_col: str) -> Dict[str, Any]:
        """执行Kruskal-Wallis H检验"""
        # 检查是否有足够的分组
        groups = df[groupby_col].unique()
        if len(groups) < 2:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'groups': groups.tolist(),
                'valid': False,
                'message': f"需要至少2个分组，但只找到{len(groups)}个"
            }

        # 收集各组数据
        samples = []
        group_data = {}
        for group in groups:
            group_values = df[df[groupby_col] == group][value_col].dropna()
            if len(group_values) > 0:
                samples.append(group_values)
                group_data[group] = {
                    'n': len(group_values),
                    'mean': group_values.mean(),
                    'median': group_values.median(),
                    'std': group_values.std()
                }

        # 执行检验
        if len(samples) >= 2:
            statistic, pvalue = stats.kruskal(*samples)
            result = {
                'statistic': float(statistic),
                'pvalue': float(pvalue),
                'groups': [str(g) for g in groups],
                'group_data': group_data,
                'valid': True,
                'significant': pvalue < 0.05,
                'message': f"Kruskal-Wallis H = {statistic:.4f}, p = {pvalue:.4f}" +
                           (", 差异显著" if pvalue < 0.05 else ", 差异不显著")
            }

            # 保存结果
            result_file = self.output_dir / f"kruskal_wallis_{groupby_col}_{value_col}.json"
            with open(result_file, 'w') as f:
                # 将numpy类型转换为Python原生类型以便JSON序列化
                json_result = {k: (v if not isinstance(v, dict) else
                                   {gk: ({sk: float(sv) for sk, sv in gv.items()} if isinstance(gv, dict) else gv)
                                    for gk, gv in v.items()})
                               for k, v in result.items()}
                json.dump(json_result, f, ensure_ascii=False, indent=2)

            return result
        else:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'groups': [str(g) for g in groups],
                'valid': False,
                'message': "有效分组不足，无法执行检验"
            }

    def mann_whitney_u_test(self, data1: pd.Series, data2: pd.Series,
                            label1: str = "Group 1", label2: str = "Group 2") -> Dict[str, Any]:
        """执行Mann-Whitney U检验"""
        # 清除空值
        data1 = data1.dropna()
        data2 = data2.dropna()

        # 检查数据量是否足够
        if len(data1) < 3 or len(data2) < 3:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'valid': False,
                'message': f"数据量不足 ({len(data1)} vs {len(data2)})，需要至少3个样本"
            }

        # 执行检验
        statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        result = {
            'statistic': float(statistic),
            'pvalue': float(pvalue),
            'group1': label1,
            'group2': label2,
            'n1': int(len(data1)),
            'n2': int(len(data2)),
            'mean1': float(data1.mean()),
            'mean2': float(data2.mean()),
            'median1': float(data1.median()),
            'median2': float(data2.median()),
            'std1': float(data1.std()),
            'std2': float(data2.std()),
            'valid': True,
            'significant': pvalue < 0.05,
            'message': f"Mann-Whitney U = {statistic:.4f}, p = {pvalue:.4f}" +
                       (", 差异显著" if pvalue < 0.05 else ", 差异不显著")
        }

        # 保存结果
        result_file = self.output_dir / f"mann_whitney_{label1}_vs_{label2}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def spearman_correlation_test(self, x: pd.Series, y: pd.Series,
                                  x_label: str = "X", y_label: str = "Y") -> Dict[str, Any]:
        """执行Spearman相关性检验"""
        # 创建数据框并清除空值
        clean_data = pd.DataFrame({x_label: x, y_label: y}).dropna()

        # 检查数据量是否足够
        if len(clean_data) < 5:
            return {
                'rho': np.nan,
                'pvalue': np.nan,
                'valid': False,
                'message': f"数据量不足，只有{len(clean_data)}个有效样本，需要至少5个"
            }

        # 执行检验
        rho, pvalue = stats.spearmanr(clean_data[x_label], clean_data[y_label])
        result = {
            'rho': float(rho),
            'pvalue': float(pvalue),
            'n': int(len(clean_data)),
            'x_label': x_label,
            'y_label': y_label,
            'x_mean': float(clean_data[x_label].mean()),
            'y_mean': float(clean_data[y_label].mean()),
            'x_std': float(clean_data[x_label].std()),
            'y_std': float(clean_data[y_label].std()),
            'valid': True,
            'significant': pvalue < 0.05,
            'direction': 'positive' if rho > 0 else 'negative',
            'strength': 'strong' if abs(rho) > 0.7 else 'moderate' if abs(rho) > 0.3 else 'weak',
            'message': f"Spearman rho = {rho:.4f}, p = {pvalue:.4f}" +
                       (", 相关性显著" if pvalue < 0.05 else ", 相关性不显著")
        }

        # 保存结果
        result_file = self.output_dir / f"spearman_{x_label}_vs_{y_label}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result


# Update the main function to add --strict flag

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
    parser.add_argument('--run-tasks', '-t', action='store_true', help='运行所有五个演练任务')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='严格模式：当缺少必要数据时报错')

    args = parser.parse_args()

    try:
        # 初始化查询验证器
        validator = KGQueryValidator(args.kg_path, args.output_dir,
                                     allow_missing=not args.strict and args.allow_missing)

        # 验证知识图谱
        if args.validate or args.strict or args.run_tasks:
            results = validator.validate_kg()
            print("\n验证结果:")
            print(f"RegionLayer节点: {results['region_layer_nodes']}")
            print(f"属性完整的RegionLayer节点: {results['region_layer_with_complete_props']}")
            print(f"HAS_SUBCLASS关系: {results['has_subclass_rels']}")
            print(f"HAS_CLASS关系: {results['has_class_rels']}")
            print(f"Project_to关系: {results['projection_rels']}")
            print(f"具有投射类型的Project_to关系: {results['projection_rels_with_types']}")
            print(f"Gene节点: {results['gene_nodes']}")
            print(f"COEXPRESSED关系: {results['coexpressed_rels']}")
            print(f"验证通过: {'是' if results['passed'] else '否'}")

            if args.strict and not results['passed']:
                print("\n严格模式：因验证失败而终止执行")
                sys.exit(1)

            # 检查特定任务所需数据
            if args.run_tasks:
                # 任务④需要Gene节点和COEXPRESSED关系
                if results['gene_nodes'] == 0 or results['coexpressed_rels'] == 0:
                    print("\n警告：缺少Gene节点或COEXPRESSED关系，任务④(Fezf2模块分析)将无法运行")
                    if args.strict:
                        print("严格模式：因缺少必要数据而终止执行")
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

        # 运行所有演练任务
        if args.run_tasks:
            summary = validator.run_all_tasks()
            print("\n演练任务执行结果:")
            for task, status in summary.items():
                if task.startswith('task'):
                    task_num = task.replace('task', '').replace('_completed', '')
                    task_names = {
                        '1': 'IT-ET-CT长程通路轴突长度分析',
                        '2': 'VIP细胞上层vs下层树突极性分析',
                        '3': 'Sst-Chodl稀有亚型定位',
                        '4': 'Fezf2模块与轴突长度相关性分析',
                        '5': 'IT型L2/3细胞投射靶点分析'
                    }
                    task_name = task_names.get(task_num, f'任务{task_num}')
                    print(f"任务{task_num} ({task_name}): {'✅ 完成' if status else '❌ 失败'}")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()