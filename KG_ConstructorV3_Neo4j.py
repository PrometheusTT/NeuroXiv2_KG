"""
NeuroXiv 2.0 知识图谱构建器 - Neo4j实时插入版本
整合MERFISH层级JSON和形态计算，实时插入Neo4j数据库

作者: wangmajortom (修改版)
日期: 2025-08-27
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set, Optional

import numpy as np
import pandas as pd
from loguru import logger
from neo4j import GraphDatabase
from tqdm import tqdm

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
PCT_THRESHOLD = 0.01
BATCH_SIZE = 1000  # Neo4j批量插入大小

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

# ==================== Neo4j连接管理器 ====================

class Neo4jConnector:
    """Neo4j数据库连接管理器"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password",
                 database: str = "neuroxiv"):
        """
        初始化Neo4j连接
        
        参数:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        
    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 测试连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"成功连接到Neo4j数据库: {self.uri}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("关闭Neo4j连接")
    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        logger.warning("清空数据库中...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("数据库已清空")
    
    def create_constraints(self):
        """创建约束和索引"""
        logger.info("创建数据库约束和索引...")
        
        constraints = [
            # 唯一性约束
            "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (r:Region) REQUIRE r.region_id IS UNIQUE",
            "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.tran_id IS UNIQUE",
            "CREATE CONSTRAINT subclass_id IF NOT EXISTS FOR (s:Subclass) REQUIRE s.tran_id IS UNIQUE",
            "CREATE CONSTRAINT supertype_id IF NOT EXISTS FOR (s:Supertype) REQUIRE s.tran_id IS UNIQUE",
            "CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:Cluster) REQUIRE c.tran_id IS UNIQUE",
            
            # 索引
            "CREATE INDEX region_name IF NOT EXISTS FOR (r:Region) ON (r.name)",
            "CREATE INDEX region_acronym IF NOT EXISTS FOR (r:Region) ON (r.acronym)",
            "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX subclass_name IF NOT EXISTS FOR (s:Subclass) ON (s.name)",
            "CREATE INDEX supertype_name IF NOT EXISTS FOR (s:Supertype) ON (s.name)",
            "CREATE INDEX cluster_name IF NOT EXISTS FOR (c:Cluster) ON (c.name)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"创建: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"约束/索引可能已存在: {e}")
    
    def insert_node(self, label: str, properties: Dict):
        """插入单个节点"""
        with self.driver.session(database=self.database) as session:
            # 构建属性字符串
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            query = f"CREATE (n:{label} {{{props_str}}})"
            session.run(query, **properties)
    
    def insert_nodes_batch(self, label: str, nodes: List[Dict]):
        """批量插入节点"""
        if not nodes:
            return
        
        with self.driver.session(database=self.database) as session:
            # 使用UNWIND批量创建
            query = f"""
            UNWIND $batch AS props
            CREATE (n:{label})
            SET n = props
            """
            session.run(query, batch=nodes)
    
    def upsert_node(self, label: str, id_field: str, properties: Dict):
        """更新或插入节点（MERGE操作）"""
        with self.driver.session(database=self.database) as session:
            query = f"""
            MERGE (n:{label} {{{id_field}: ${id_field}}})
            SET n += $props
            """
            session.run(query, **{id_field: properties[id_field], 'props': properties})
    
    def insert_relationship(self, start_label: str, start_id_field: str, start_id: any,
                           end_label: str, end_id_field: str, end_id: any,
                           rel_type: str, rel_properties: Dict = None):
        """插入关系"""
        with self.driver.session(database=self.database) as session:
            if rel_properties:
                props_str = ", ".join([f"{k}: ${k}" for k in rel_properties.keys()])
                query = f"""
                MATCH (a:{start_label} {{{start_id_field}: $start_id}})
                MATCH (b:{end_label} {{{end_id_field}: $end_id}})
                CREATE (a)-[r:{rel_type} {{{props_str}}}]->(b)
                """
                params = {'start_id': start_id, 'end_id': end_id}
                params.update(rel_properties)
            else:
                query = f"""
                MATCH (a:{start_label} {{{start_id_field}: $start_id}})
                MATCH (b:{end_label} {{{end_id_field}: $end_id}})
                CREATE (a)-[r:{rel_type}]->(b)
                """
                params = {'start_id': start_id, 'end_id': end_id}
            
            session.run(query, **params)
    
    def insert_relationships_batch(self, relationships: List[Dict]):
        """批量插入关系"""
        if not relationships:
            return
        
        with self.driver.session(database=self.database) as session:
            # 按关系类型分组处理
            rel_types = {}
            for rel in relationships:
                rel_type = rel['rel_type']
                if rel_type not in rel_types:
                    rel_types[rel_type] = []
                rel_types[rel_type].append(rel)
            
            for rel_type, rels in rel_types.items():
                # 根据关系类型构建查询
                if rel_type == 'PROJECT_TO':
                    query = """
                    UNWIND $batch AS rel
                    MATCH (a:Region {region_id: rel.start_id})
                    MATCH (b:Region {region_id: rel.end_id})
                    CREATE (a)-[r:PROJECT_TO {
                        weight: rel.weight,
                        total: rel.total,
                        neuron_count: rel.neuron_count,
                        source_acronym: rel.source_acronym,
                        target_acronym: rel.target_acronym
                    }]->(b)
                    """
                elif rel_type.startswith('HAS_'):
                    # HAS_CLASS, HAS_SUBCLASS等
                    entity_type = rel_type.replace('HAS_', '').capitalize()
                    query = f"""
                    UNWIND $batch AS rel
                    MATCH (a:Region {{region_id: rel.start_id}})
                    MATCH (b:{entity_type} {{tran_id: rel.end_id}})
                    CREATE (a)-[r:{rel_type} {{
                        pct_cells: rel.pct_cells,
                        rank: rel.rank
                    }}]->(b)
                    """
                elif rel_type == 'BELONGS_TO':
                    query = """
                    UNWIND $batch AS rel
                    MATCH (a) WHERE id(a) = rel.start_node_id
                    MATCH (b) WHERE id(b) = rel.end_node_id
                    CREATE (a)-[r:BELONGS_TO]->(b)
                    """
                else:
                    continue
                
                # 批量执行
                for i in range(0, len(rels), BATCH_SIZE):
                    batch = rels[i:i+BATCH_SIZE]
                    try:
                        session.run(query, batch=batch)
                    except Exception as e:
                        logger.error(f"批量插入关系失败: {e}")


# ==================== 继承原有类并添加Neo4j支持 ====================

# 继续使用原有的MERFISHHierarchyLoader、RegionAnalyzer、MorphologyDataLoader类
# 这里省略，因为它们不需要修改

from KG_ConstructorV3 import (
    MERFISHHierarchyLoader,
    RegionAnalyzer, 
    MorphologyDataLoader
)


# ==================== 增强的知识图谱构建器 ====================

class KnowledgeGraphBuilderNeo4j:
    """知识图谱构建器 - Neo4j实时插入版本"""
    
    def __init__(self, neo4j_connector: Neo4jConnector):
        """
        初始化
        
        参数:
            neo4j_connector: Neo4j连接器实例
        """
        self.neo4j = neo4j_connector
        self.morphology_loader = None
        
        # 存储ID映射
        self.class_id_map = {}
        self.subclass_id_map = {}
        self.supertype_id_map = {}
        self.cluster_id_map = {}
        
        # 存储层级数据
        self.hierarchy_loader = None
        
        # 初始化区域分析器
        self.region_analyzer = None
        
        # 统计信息
        self.stats = {
            'regions_inserted': 0,
            'classes_inserted': 0,
            'subclasses_inserted': 0,
            'supertypes_inserted': 0,
            'clusters_inserted': 0,
            'relationships_inserted': 0
        }
    
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
    
    def generate_and_insert_unified_region_nodes(self, region_data: pd.DataFrame, 
                                                 merfish_cells: pd.DataFrame = None):
        """生成并实时插入统一的Region节点"""
        logger.info("生成并插入统一的Region节点（聚合脑区）...")
        
        # 1. 获取已聚合的MERFISH数据中的区域
        merfish_regions = set()
        if merfish_cells is not None and not merfish_cells.empty and 'region_id' in merfish_cells.columns:
            # 对MERFISH数据进行聚合
            merfish_cells_copy = merfish_cells.copy()
            merfish_cells_copy['target_region_id'] = merfish_cells_copy['region_id'].apply(
                lambda x: self.region_analyzer.region_to_target.get(x, x) if pd.notna(x) else None
            )
            merfish_cells_copy = merfish_cells_copy[merfish_cells_copy['target_region_id'].notna()]
            merfish_cells_copy['region_id'] = merfish_cells_copy['target_region_id']
            
            merfish_regions = set(merfish_cells_copy['region_id'].dropna().unique())
            logger.info(f"从聚合后的MERFISH数据中提取了 {len(merfish_regions)} 个区域ID")
            
            # 保存修改后的MERFISH数据供后续使用
            self.aggregated_merfish_cells = merfish_cells_copy
        
        # 2. 从info表中提取区域
        info_regions = set()
        if hasattr(self, 'morphology_loader') and self.morphology_loader and self.morphology_loader.info_df is not None:
            info_df = self.morphology_loader.info_df
            if 'region_id' in info_df.columns:
                info_regions = set(info_df['region_id'].dropna().unique())
                logger.info(f"从形态数据中提取了 {len(info_regions)} 个区域ID")
        
        # 3. 合并区域列表
        all_region_ids = merfish_regions.union(info_regions)
        logger.info(f"合并后共有 {len(all_region_ids)} 个唯一区域ID")
        
        # 4. 逐个处理并插入Region节点
        batch_nodes = []
        
        for region_id in tqdm(all_region_ids, desc="处理Region节点"):
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
            acronym = region_info.get('acronym', '') or self.region_analyzer.get_region_acronym(region_id_int)
            
            # 创建区域字典
            region_dict = {
                'region_id': region_id_int,
                'name': str(acronym),
                'full_name': str(region_info.get('name', acronym)),
                'acronym': str(acronym),
                'parent_id': int(region_info.get('parent_id', 0))
            }
            
            # 计算形态学特征
            if hasattr(self, 'morphology_loader') and self.morphology_loader:
                stats = self.morphology_loader.calculate_region_morphology(region_id_int)
                
                # 更新形态学属性
                for attr in MORPH_ATTRIBUTES:
                    region_dict[attr] = float(stats.get(attr, 0.0))
                
                # 更新统计属性
                for attr in STAT_ATTRIBUTES:
                    if attr != 'number_of_transcriptomic_neurons':
                        region_dict[attr] = int(stats.get(attr, 0))
            
            # 计算MERFISH相关统计
            if hasattr(self, 'aggregated_merfish_cells'):
                region_cells = self.aggregated_merfish_cells[
                    self.aggregated_merfish_cells['region_id'] == region_id_int
                ]
                region_dict['number_of_transcriptomic_neurons'] = len(region_cells)
            else:
                region_dict['number_of_transcriptomic_neurons'] = 0
            
            # 检查是否有数据
            has_morphology = any(region_dict.get(attr, 0) > 0 for attr in MORPH_ATTRIBUTES)
            has_morphology = has_morphology or any(
                region_dict.get(attr, 0) > 0 for attr in STAT_ATTRIBUTES 
                if attr != 'number_of_transcriptomic_neurons'
            )
            has_merfish = region_dict.get('number_of_transcriptomic_neurons', 0) > 0
            
            # 仅插入有数据的节点
            if has_morphology or has_merfish:
                batch_nodes.append(region_dict)
                
                # 批量插入
                if len(batch_nodes) >= BATCH_SIZE:
                    self.neo4j.insert_nodes_batch('Region', batch_nodes)
                    self.stats['regions_inserted'] += len(batch_nodes)
                    batch_nodes = []
        
        # 插入剩余的节点
        if batch_nodes:
            self.neo4j.insert_nodes_batch('Region', batch_nodes)
            self.stats['regions_inserted'] += len(batch_nodes)
        
        logger.info(f"成功插入 {self.stats['regions_inserted']} 个Region节点")
        
        # 保存region_data供其他方法使用
        self.region_data = pd.DataFrame([
            {'region_id': region_id} for region_id in all_region_ids
        ])
    
    def generate_and_insert_merfish_nodes_from_hierarchy(self, merfish_cells: pd.DataFrame):
        """从层级数据生成并插入MERFISH节点"""
        if not self.hierarchy_loader:
            logger.error("未设置层级加载器")
            return
        
        # 使用聚合后的MERFISH数据
        if hasattr(self, 'aggregated_merfish_cells'):
            merfish_cells = self.aggregated_merfish_cells
        
        # 计算每个类型的细胞数量
        if not merfish_cells.empty:
            class_counts = merfish_cells['class'].value_counts().to_dict() if 'class' in merfish_cells.columns else {}
            subclass_counts = merfish_cells['subclass'].value_counts().to_dict() if 'subclass' in merfish_cells.columns else {}
            supertype_counts = merfish_cells['supertype'].value_counts().to_dict() if 'supertype' in merfish_cells.columns else {}
            cluster_counts = merfish_cells['cluster'].value_counts().to_dict() if 'cluster' in merfish_cells.columns else {}
        else:
            class_counts = subclass_counts = supertype_counts = cluster_counts = {}
        
        # 生成并插入Class节点
        logger.info("生成并插入Class节点...")
        class_nodes = []
        for name, data in tqdm(self.hierarchy_loader.class_data.items(), desc="处理Class节点"):
            node = {
                'tran_id': data['tran_id'],
                'id': data['original_id'] + 300,
                'name': name,
                'neighborhood': data['neighborhood'],
                'number_of_child_types': data['number_of_child_types'],
                'number_of_neurons': class_counts.get(name, 0),
                'dominant_neurotransmitter_type': data.get('dominant_neurotransmitter_type', ''),
                'markers': data.get('markers', '')
            }
            class_nodes.append(node)
            
            if len(class_nodes) >= BATCH_SIZE:
                self.neo4j.insert_nodes_batch('Class', class_nodes)
                self.stats['classes_inserted'] += len(class_nodes)
                class_nodes = []
        
        if class_nodes:
            self.neo4j.insert_nodes_batch('Class', class_nodes)
            self.stats['classes_inserted'] += len(class_nodes)
        
        logger.info(f"插入了 {self.stats['classes_inserted']} 个Class节点")
        
        # 生成并插入Subclass节点
        logger.info("生成并插入Subclass节点...")
        subclass_nodes = []
        for name, data in tqdm(self.hierarchy_loader.subclass_data.items(), desc="处理Subclass节点"):
            node = {
                'tran_id': data['tran_id'],
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
            
            if len(subclass_nodes) >= BATCH_SIZE:
                self.neo4j.insert_nodes_batch('Subclass', subclass_nodes)
                self.stats['subclasses_inserted'] += len(subclass_nodes)
                subclass_nodes = []
        
        if subclass_nodes:
            self.neo4j.insert_nodes_batch('Subclass', subclass_nodes)
            self.stats['subclasses_inserted'] += len(subclass_nodes)
        
        logger.info(f"插入了 {self.stats['subclasses_inserted']} 个Subclass节点")
        
        # 生成并插入Supertype节点
        logger.info("生成并插入Supertype节点...")
        supertype_nodes = []
        for name, data in tqdm(self.hierarchy_loader.supertype_data.items(), desc="处理Supertype节点"):
            node = {
                'tran_id': data['tran_id'],
                'id': data['original_id'] + 600,
                'name': name,
                'number_of_child_types': data['number_of_child_types'],
                'number_of_neurons': supertype_counts.get(name, 0),
                'markers': data['markers'],
                'within_subclass_markers': data['within_subclass_markers']
            }
            supertype_nodes.append(node)
            
            if len(supertype_nodes) >= BATCH_SIZE:
                self.neo4j.insert_nodes_batch('Supertype', supertype_nodes)
                self.stats['supertypes_inserted'] += len(supertype_nodes)
                supertype_nodes = []
        
        if supertype_nodes:
            self.neo4j.insert_nodes_batch('Supertype', supertype_nodes)
            self.stats['supertypes_inserted'] += len(supertype_nodes)
        
        logger.info(f"插入了 {self.stats['supertypes_inserted']} 个Supertype节点")
        
        # 生成并插入Cluster节点
        logger.info("生成并插入Cluster节点...")
        cluster_nodes = []
        for name, data in tqdm(self.hierarchy_loader.cluster_data.items(), desc="处理Cluster节点"):
            node = {
                'tran_id': data['tran_id'],
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
            
            if len(cluster_nodes) >= BATCH_SIZE:
                self.neo4j.insert_nodes_batch('Cluster', cluster_nodes)
                self.stats['clusters_inserted'] += len(cluster_nodes)
                cluster_nodes = []
        
        if cluster_nodes:
            self.neo4j.insert_nodes_batch('Cluster', cluster_nodes)
            self.stats['clusters_inserted'] += len(cluster_nodes)
        
        logger.info(f"插入了 {self.stats['clusters_inserted']} 个Cluster节点")
    
    def generate_and_insert_project_to_relationships(self, projection_data: pd.DataFrame):
        """生成并插入PROJECT_TO关系"""
        if projection_data is None or projection_data.empty:
            logger.warning("没有投影数据")
            return
        
        logger.info("生成并插入PROJECT_TO关系...")
        
        # 确保我们有info表数据
        if not hasattr(self, 'morphology_loader') or self.morphology_loader is None or self.morphology_loader.info_df is None:
            logger.error("缺少info表数据，无法匹配源区域")
            return
        
        info_df = self.morphology_loader.info_df
        
        # 创建标准化ID函数
        def normalize_id(id_str):
            id_str = str(id_str)
            id_str = id_str.replace('CCF-thin', '').replace('CCFv3', '').strip('_')
            return id_str
        
        # 标准化info表中的ID
        info_df['normalized_id'] = info_df['ID'].apply(normalize_id)
        
        # 获取区域ID映射
        region_to_id = {}
        id_to_acronym = {}
        
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
                region = col.replace('proj_axon_', '').replace('_abs', '')
                target_regions.append(region)
        
        logger.info(f"找到 {len(target_regions)} 个目标区域")
        
        # 将投射数据索引转换为规范化ID进行匹配
        proj_normalized_ids = {}
        for idx in projection_data.index:
            norm_id = normalize_id(idx)
            proj_normalized_ids[norm_id] = idx
        
        # 构建神经元ID到源区域的映射
        neuron_source_map = {}
        
        for normalized_id, info_rows in info_df.groupby('normalized_id'):
            if normalized_id in proj_normalized_ids:
                orig_idx = proj_normalized_ids[normalized_id]
                
                if 'region_id' in info_rows.columns:
                    source_id = info_rows.iloc[0]['region_id']
                    if pd.notna(source_id):
                        source_acronym = id_to_acronym.get(source_id, '')
                        if source_acronym:
                            neuron_source_map[orig_idx] = (source_id, source_acronym)
        
        logger.info(f"规范化ID匹配: {len(neuron_source_map)}/{len(projection_data)} 个神经元匹配")
        
        # 计算区域间投射并实时插入
        region_connections = {}
        
        for neuron_id, (source_id, source_acronym) in tqdm(neuron_source_map.items(), desc="处理投射数据"):
            neuron_data = projection_data.loc[neuron_id]
            
            for target_acronym in target_regions:
                target_id = region_to_id.get(target_acronym)
                if not target_id:
                    continue
                
                proj_col = f"proj_axon_{target_acronym}_abs"
                if proj_col not in projection_data.columns:
                    continue
                
                try:
                    proj_value = neuron_data[proj_col]
                    
                    if pd.notna(proj_value) and proj_value > 0:
                        # 使用聚合后的区域ID
                        if hasattr(self.region_analyzer, 'region_to_target'):
                            source_target_id = self.region_analyzer.region_to_target.get(source_id, source_id)
                            target_target_id = self.region_analyzer.region_to_target.get(target_id, target_id)
                            
                            conn_key = (source_target_id, target_target_id)
                            
                            source_target_acronym = id_to_acronym.get(source_target_id, source_acronym)
                            target_target_acronym = id_to_acronym.get(target_target_id, target_acronym)
                        else:
                            conn_key = (source_id, target_id)
                            source_target_acronym = source_acronym
                            target_target_acronym = target_acronym
                        
                        if conn_key not in region_connections:
                            region_connections[conn_key] = {
                                'total': 0,
                                'count': 0,
                                'source_acronym': source_target_acronym,
                                'target_acronym': target_target_acronym
                            }
                        
                        region_connections[conn_key]['total'] += float(proj_value)
                        region_connections[conn_key]['count'] += 1
                except Exception as e:
                    logger.debug(f"处理投射数据时出错: {e}")
                    continue
        
        # 批量插入PROJECT_TO关系
        batch_relationships = []
        
        for (source_id, target_id), stats in tqdm(region_connections.items(), desc="插入PROJECT_TO关系"):
            avg_strength = stats['total'] / stats['count'] if stats['count'] > 0 else 0
            
            rel = {
                'start_id': int(source_id),
                'end_id': int(target_id),
                'weight': float(avg_strength),
                'total': float(stats['total']),
                'neuron_count': int(stats['count']),
                'source_acronym': stats['source_acronym'],
                'target_acronym': stats['target_acronym'],
                'rel_type': 'PROJECT_TO'
            }
            
            batch_relationships.append(rel)
            
            # 批量插入
            if len(batch_relationships) >= BATCH_SIZE:
                self.neo4j.insert_relationships_batch(batch_relationships)
                self.stats['relationships_inserted'] += len(batch_relationships)
                batch_relationships = []
        
        # 插入剩余的关系
        if batch_relationships:
            self.neo4j.insert_relationships_batch(batch_relationships)
            self.stats['relationships_inserted'] += len(batch_relationships)
        
        logger.info(f"插入了 {len(region_connections)} 个PROJECT_TO关系")
    
    def generate_and_insert_has_relationships_unified(self, merfish_cells: pd.DataFrame, level: str):
        """生成并插入HAS关系"""
        logger.info(f"生成并插入HAS_{level.upper()}关系...")
        
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
        
        # 使用已聚合的MERFISH数据
        if hasattr(self, 'aggregated_merfish_cells'):
            merfish_cells_copy = self.aggregated_merfish_cells
        else:
            merfish_cells_copy = merfish_cells.copy()
            if hasattr(self.region_analyzer, 'region_to_target'):
                merfish_cells_copy['target_region_id'] = merfish_cells_copy['region_id'].apply(
                    lambda x: self.region_analyzer.region_to_target.get(x, x) if pd.notna(x) else None
                )
                merfish_cells_copy = merfish_cells_copy[merfish_cells_copy['target_region_id'].notna()]
                merfish_cells_copy['region_id'] = merfish_cells_copy['target_region_id']
        
        # 筛选有效细胞
        valid_cells = merfish_cells_copy[(merfish_cells_copy[level].notna()) & (merfish_cells_copy['region_id'] > 0)]
        
        # 按区域和类型分组计数
        counts_df = valid_cells.groupby(['region_id', level]).size().reset_index(name='count')
        
        # 添加比例列
        region_totals = valid_cells.groupby('region_id').size().reset_index(name='total')
        counts_df = pd.merge(counts_df, region_totals, on='region_id')
        counts_df['pct'] = counts_df['count'] / counts_df['total']
        
        # 过滤低于阈值的行
        counts_df = counts_df[counts_df['pct'] >= PCT_THRESHOLD]
        
        # 批量插入关系
        batch_relationships = []
        
        for region_id, group in tqdm(counts_df.groupby('region_id'), desc=f"处理HAS_{level.upper()}关系"):
            # 按比例降序排序
            group_sorted = group.sort_values('pct', ascending=False)
            
            # 创建连续的rank值
            rank = 1
            
            for _, row in group_sorted.iterrows():
                cell_type = row[level]
                
                if cell_type in id_map:
                    rel = {
                        'start_id': int(region_id),
                        'end_id': id_map[cell_type],
                        'pct_cells': float(row['pct']),
                        'rank': rank,
                        'rel_type': f'HAS_{level.upper()}'
                    }
                    batch_relationships.append(rel)
                    rank += 1
                    
                    # 批量插入
                    if len(batch_relationships) >= BATCH_SIZE:
                        self.neo4j.insert_relationships_batch(batch_relationships)
                        self.stats['relationships_inserted'] += len(batch_relationships)
                        batch_relationships = []
        
        # 插入剩余的关系
        if batch_relationships:
            self.neo4j.insert_relationships_batch(batch_relationships)
            self.stats['relationships_inserted'] += len(batch_relationships)
        
        logger.info(f"插入了 {counts_df.shape[0]} 个HAS_{level.upper()}关系")
    
    def generate_and_insert_belongs_to_from_hierarchy(self):
        """生成并插入BELONGS_TO关系"""
        if not self.hierarchy_loader:
            logger.error("未设置层级加载器")
            return
        
        logger.info("生成并插入BELONGS_TO关系...")
        
        # 使用简化的BELONGS_TO关系插入
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            # Subclass -> Class
            for subclass, class_name in tqdm(
                self.hierarchy_loader.hierarchy_relations['subclass_to_class'].items(), 
                desc="插入Subclass->Class关系"
            ):
                if subclass in self.subclass_id_map and class_name in self.class_id_map:
                    query = """
                    MATCH (s:Subclass {tran_id: $subclass_id})
                    MATCH (c:Class {tran_id: $class_id})
                    CREATE (s)-[r:BELONGS_TO]->(c)
                    """
                    try:
                        session.run(query, 
                                  subclass_id=self.subclass_id_map[subclass],
                                  class_id=self.class_id_map[class_name])
                        self.stats['relationships_inserted'] += 1
                    except Exception as e:
                        logger.error(f"插入关系失败: {e}")
            
            # Supertype -> Subclass
            for supertype, subclass in tqdm(
                self.hierarchy_loader.hierarchy_relations['supertype_to_subclass'].items(),
                desc="插入Supertype->Subclass关系"
            ):
                if supertype in self.supertype_id_map and subclass in self.subclass_id_map:
                    query = """
                    MATCH (st:Supertype {tran_id: $supertype_id})
                    MATCH (sc:Subclass {tran_id: $subclass_id})
                    CREATE (st)-[r:BELONGS_TO]->(sc)
                    """
                    try:
                        session.run(query,
                                  supertype_id=self.supertype_id_map[supertype],
                                  subclass_id=self.subclass_id_map[subclass])
                        self.stats['relationships_inserted'] += 1
                    except Exception as e:
                        logger.error(f"插入关系失败: {e}")
            
            # Cluster -> Supertype
            for cluster, supertype in tqdm(
                self.hierarchy_loader.hierarchy_relations['cluster_to_supertype'].items(),
                desc="插入Cluster->Supertype关系"
            ):
                if cluster in self.cluster_id_map and supertype in self.supertype_id_map:
                    query = """
                    MATCH (cl:Cluster {tran_id: $cluster_id})
                    MATCH (st:Supertype {tran_id: $supertype_id})
                    CREATE (cl)-[r:BELONGS_TO]->(st)
                    """
                    try:
                        session.run(query,
                                  cluster_id=self.cluster_id_map[cluster],
                                  supertype_id=self.supertype_id_map[supertype])
                        self.stats['relationships_inserted'] += 1
                    except Exception as e:
                        logger.error(f"插入关系失败: {e}")
        
        logger.info(f"插入了BELONGS_TO关系")
    
    def print_statistics_report(self):
        """打印统计报告"""
        report = []
        report.append("=" * 60)
        report.append("NeuroXiv 2.0 知识图谱Neo4j导入统计报告")
        report.append("=" * 60)
        report.append("节点统计:")
        report.append(f"  - Region节点: {self.stats['regions_inserted']}")
        report.append(f"  - Class节点: {self.stats['classes_inserted']}")
        report.append(f"  - Subclass节点: {self.stats['subclasses_inserted']}")
        report.append(f"  - Supertype节点: {self.stats['supertypes_inserted']}")
        report.append(f"  - Cluster节点: {self.stats['clusters_inserted']}")
        report.append(f"  - 总节点数: {sum([self.stats['regions_inserted'], self.stats['classes_inserted'], self.stats['subclasses_inserted'], self.stats['supertypes_inserted'], self.stats['clusters_inserted']])}")
        report.append(f"\n关系统计:")
        report.append(f"  - 总关系数: {self.stats['relationships_inserted']}")
        report.append(f"\n生成时间: {pd.Timestamp.now()}")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 也保存到日志
        for line in report:
            logger.info(line)


# ==================== 工具函数 ====================

def setup_logger(log_file: str = "kg_builder_neo4j.log"):
    """设置日志"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="100 MB", level="DEBUG")


# ==================== 主函数 ====================

def main(data_dir: str = "../data",
         hierarchy_json: str = None,
         neo4j_uri: str = "bolt://localhost:7687",
         neo4j_user: str = "neo4j",
         neo4j_password: str = "password",
         database_name: str = "neuroxiv",
         clear_database: bool = False):
    """
    主函数 - Neo4j实时插入版本
    
    参数:
        data_dir: 数据目录路径
        hierarchy_json: MERFISH层级JSON文件路径
        neo4j_uri: Neo4j数据库URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        database_name: 数据库名称
        clear_database: 是否清空数据库
    """
    
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("NeuroXiv 2.0 知识图谱构建 - Neo4j实时插入版本")
    logger.info("=" * 60)
    
    # 初始化Neo4j连接
    logger.info("初始化Neo4j连接...")
    neo4j_conn = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password, database_name)
    
    if not neo4j_conn.connect():
        logger.error("无法连接到Neo4j数据库")
        return
    
    try:
        # 清空数据库（如果需要）
        if clear_database:
            neo4j_conn.clear_database()
        
        # 创建约束和索引
        neo4j_conn.create_constraints()
        
        # Phase 1: 数据加载
        logger.info("Phase 1: 数据加载")
        
        data_path = Path(data_dir)
        data = load_data(data_path)
        processed_data = prepare_analysis_data(data)
        region_data = processed_data.get('region_data', pd.DataFrame())
        merfish_cells = processed_data.get('merfish_cells', pd.DataFrame())
        projection_data = processed_data.get('projection_df', pd.DataFrame())
        
        # 创建builder实例
        builder = KnowledgeGraphBuilderNeo4j(neo4j_conn)
        
        # 加载树结构用于区域分析
        tree_data = processed_data.get('tree', [])
        if tree_data:
            logger.info("初始化区域分析器...")
            builder.region_analyzer = RegionAnalyzer(tree_data)
            logger.info(f"从树数据中加载了 {len(builder.region_analyzer.region_info)} 个区域信息")
        
        # Phase 2: 加载层级数据
        logger.info("Phase 2: 加载MERFISH层级数据")
        
        hierarchy_loader = MERFISHHierarchyLoader(
            Path(hierarchy_json) if hierarchy_json else data_path / "hierarchy.json")
        
        if not hierarchy_loader.load_hierarchy():
            logger.error("无法加载层级数据")
            return
        
        # Phase 3: 加载形态数据
        logger.info("Phase 3: 加载形态数据和投射数据")
        
        morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
        if morphology_loader.load_morphology_data():
            if projection_data is not None and not projection_data.empty:
                morphology_loader.set_projection_data(projection_data)
            builder.morphology_loader = morphology_loader
        else:
            logger.warning("无法加载形态学数据")
        
        # Phase 4: 知识图谱生成和插入
        logger.info("Phase 4: 知识图谱生成和实时插入")
        
        builder.set_hierarchy_loader(hierarchy_loader)
        
        # 生成并插入节点
        logger.info("生成并插入节点...")
        
        # 生成并插入统一的Region节点
        builder.generate_and_insert_unified_region_nodes(region_data, merfish_cells)
        
        # 生成并插入MERFISH细胞类型节点
        builder.generate_and_insert_merfish_nodes_from_hierarchy(merfish_cells)
        
        # 生成并插入关系
        logger.info("生成并插入关系...")
        
        # 生成并插入HAS关系
        for level in ['class', 'subclass', 'supertype', 'cluster']:
            builder.generate_and_insert_has_relationships_unified(merfish_cells, level)
        
        # 生成并插入层级关系
        builder.generate_and_insert_belongs_to_from_hierarchy()
        
        # 生成并插入投射关系
        builder.generate_and_insert_project_to_relationships(projection_data)
        
        # 打印统计报告
        builder.print_statistics_report()
        
        logger.info("=" * 60)
        logger.info("知识图谱构建和导入完成！")
        logger.info("=" * 60)
        
    finally:
        # 关闭Neo4j连接
        neo4j_conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroXiv 2.0 知识图谱构建 - Neo4j实时插入版本')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data',
                        help='数据目录路径')
    parser.add_argument('--hierarchy_json', type=str, default='/home/wlj/NeuroXiv2/data/tran-data-type-tree.json',
                        help='MERFISH层级JSON文件路径')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j数据库URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j',
                        help='Neo4j用户名')
    parser.add_argument('--neo4j_password', type=str, required=True,
                        help='Neo4j密码')
    parser.add_argument('--database', type=str, default='neuroxiv',
                        help='数据库名称')
    parser.add_argument('--clear_database', action='store_true',
                        help='清空数据库后重新导入')
    
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
