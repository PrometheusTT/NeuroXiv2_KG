"""
Schema-Aware Path Planning
==========================
对应设计图B: Schema views exposed to planner + Dynamic Schema Path Planner

包含:
- Schema View: 暴露给planner的schema视图
- Dynamic Schema Path Planner: 动态路径规划
- Path Optimization: 路径优化

Author: Lijun
Date: 2025-01
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core_structures import (
    Entity, SchemaPath, Modality, PlannerType
)

logger = logging.getLogger(__name__)


# ==================== Schema Definition ====================

@dataclass
class SchemaNode:
    """Schema节点"""
    name: str
    properties: List[str] = field(default_factory=list)
    modality: Optional[Modality] = None


@dataclass
class SchemaRelationship:
    """Schema关系"""
    name: str
    from_node: str
    to_node: str
    properties: List[str] = field(default_factory=list)


class KnowledgeGraphSchema:
    """
    知识图谱Schema - 对应设计图B的Schema views exposed to planner

    定义了NeuroXiv知识图谱的完整schema
    """

    # 节点类型定义
    NODES = {
        # 分子层级
        'Class': SchemaNode('Class', ['class_name', 'neurotransmitter'], Modality.MOLECULAR),
        'Subclass': SchemaNode('Subclass', ['subclass_name', 'markers'], Modality.MOLECULAR),
        'Supertype': SchemaNode('Supertype', ['supertype_name'], Modality.MOLECULAR),
        'Cluster': SchemaNode('Cluster', ['cluster_id', 'cell_count', 'markers', 'enrichment'], Modality.MOLECULAR),

        # 空间层级
        'BrainRegion': SchemaNode('BrainRegion', ['acronym', 'full_name', 'structure_id'], None),
        'TargetRegion': SchemaNode('TargetRegion', ['acronym', 'full_name'], Modality.PROJECTION),

        # 形态层级
        'Morphology': SchemaNode('Morphology', ['reconstruction_id', 'axon_length', 'dendrite_length',
                                                'branch_count', 'soma_depth'], Modality.MORPHOLOGICAL),
    }

    # 关系类型定义
    RELATIONSHIPS = {
        # 分类层级关系
        'HAVE_CLASS': SchemaRelationship('HAVE_CLASS', 'Cluster', 'Class'),
        'HAVE_SUBCLASS': SchemaRelationship('HAVE_SUBCLASS', 'Cluster', 'Subclass'),
        'HAVE_SUPERTYPE': SchemaRelationship('HAVE_SUPERTYPE', 'Cluster', 'Supertype'),

        # 空间关系
        'ENRICHED_IN': SchemaRelationship('ENRICHED_IN', 'Cluster', 'BrainRegion', ['percentage']),
        'LOCATED_IN': SchemaRelationship('LOCATED_IN', 'Morphology', 'BrainRegion'),

        # 投射关系
        'PROJECT_TO': SchemaRelationship('PROJECT_TO', 'BrainRegion', 'TargetRegion',
                                         ['projection_weight', 'normalized_weight']),

        # 形态关联
        'HAS_MORPHOLOGY': SchemaRelationship('HAS_MORPHOLOGY', 'Cluster', 'Morphology'),
    }

    @classmethod
    def get_node(cls, name: str) -> Optional[SchemaNode]:
        """获取节点定义"""
        return cls.NODES.get(name)

    @classmethod
    def get_relationship(cls, name: str) -> Optional[SchemaRelationship]:
        """获取关系定义"""
        return cls.RELATIONSHIPS.get(name)

    @classmethod
    def get_adjacent_nodes(cls, node_name: str) -> List[Tuple[str, str, str]]:
        """获取邻接节点 - (关系名, 方向, 目标节点)"""
        adjacent = []
        for rel_name, rel in cls.RELATIONSHIPS.items():
            if rel.from_node == node_name:
                adjacent.append((rel_name, 'outgoing', rel.to_node))
            elif rel.to_node == node_name:
                adjacent.append((rel_name, 'incoming', rel.from_node))
        return adjacent

    @classmethod
    def get_modality_nodes(cls, modality: Modality) -> List[str]:
        """获取特定模态的节点"""
        return [name for name, node in cls.NODES.items() if node.modality == modality]


# ==================== Path Planning ====================

class DynamicSchemaPathPlanner:
    """
    动态Schema路径规划器 - 对应设计图B的Dynamic Schema Path Planner

    根据输入实体和约束，通过图搜索生成最优查询路径
    """

    def __init__(self, schema: type = KnowledgeGraphSchema):
        self.schema = schema
        self._build_adjacency_graph()

    def _build_adjacency_graph(self):
        """构建邻接图用于路径搜索"""
        self.adjacency = defaultdict(list)

        for rel_name, rel in self.schema.RELATIONSHIPS.items():
            self.adjacency[rel.from_node].append((rel.to_node, rel_name, 'forward'))
            self.adjacency[rel.to_node].append((rel.from_node, rel_name, 'backward'))

    def plan_paths(self,
                   entities: List[Entity],
                   focus_modalities: List[Modality],
                   planner_type: PlannerType,
                   max_paths: int = 5) -> List[SchemaPath]:
        """
        规划查询路径

        Args:
            entities: 输入实体
            focus_modalities: 焦点模态
            planner_type: 规划器类型
            max_paths: 最大路径数

        Returns:
            规划的路径列表
        """
        paths = []
        path_id = 0

        # 1. 确定起始节点
        start_nodes = self._determine_start_nodes(entities)

        # 2. 确定目标节点（基于模态）
        target_nodes = set()
        for modality in focus_modalities:
            target_nodes.update(self.schema.get_modality_nodes(modality))

        # 如果没有明确目标，添加所有模态节点
        if not target_nodes:
            for modality in Modality:
                target_nodes.update(self.schema.get_modality_nodes(modality))

        # 3. 根据planner类型生成路径
        if planner_type == PlannerType.ADAPTIVE:
            # 简单直接路径
            paths = self._generate_direct_paths(start_nodes, target_nodes, max_paths=3)

        elif planner_type == PlannerType.FOCUS_DRIVEN:
            # 深度优先路径 - 覆盖所有模态
            paths = self._generate_comprehensive_paths(start_nodes, focus_modalities, max_paths)

        elif planner_type == PlannerType.COMPARATIVE:
            # 并行比较路径
            paths = self._generate_comparative_paths(entities, focus_modalities, max_paths)

        # 4. 路径优化和排序
        paths = self._optimize_paths(paths)

        logger.debug(f"Planned {len(paths)} paths for {planner_type.value}")
        return paths[:max_paths]

    def _determine_start_nodes(self, entities: List[Entity]) -> Set[str]:
        """根据实体类型确定起始节点"""
        start_nodes = set()

        for entity in entities:
            if entity.type == 'Marker':
                start_nodes.add('Cluster')  # marker通常关联cluster
            elif entity.type == 'Region':
                start_nodes.add('BrainRegion')
            elif entity.type == 'Class':
                start_nodes.add('Class')
            elif entity.type == 'Subclass':
                start_nodes.add('Subclass')

        # 默认从Cluster开始
        if not start_nodes:
            start_nodes.add('Cluster')

        return start_nodes

    def _generate_direct_paths(self, start_nodes: Set[str],
                               target_nodes: Set[str],
                               max_paths: int) -> List[SchemaPath]:
        """生成直接路径（用于简单查询）"""
        paths = []
        path_id = 0

        for start in start_nodes:
            # BFS寻找到目标的最短路径
            visited = {start}
            queue = [(start, [start], [])]

            while queue and len(paths) < max_paths:
                current, node_path, rel_path = queue.pop(0)

                if current in target_nodes and len(node_path) > 1:
                    paths.append(SchemaPath(
                        path_id=path_id,
                        nodes=node_path,
                        relationships=rel_path,
                        description=f"Direct path: {' -> '.join(node_path)}",
                        estimated_cost=len(node_path),
                    ))
                    path_id += 1

                # 扩展邻居
                for neighbor, rel_name, direction in self.adjacency[current]:
                    if neighbor not in visited and len(node_path) < 4:
                        visited.add(neighbor)
                        queue.append((
                            neighbor,
                            node_path + [neighbor],
                            rel_path + [rel_name]
                        ))

        return paths

    def _generate_comprehensive_paths(self, start_nodes: Set[str],
                                      focus_modalities: List[Modality],
                                      max_paths: int) -> List[SchemaPath]:
        """生成综合分析路径（用于Focus Driven）"""
        paths = []
        path_id = 0

        # 预定义的综合分析路径模板
        COMPREHENSIVE_TEMPLATES = [
            # 分子特征路径
            {
                'nodes': ['Cluster', 'Class'],
                'relationships': ['HAVE_CLASS'],
                'description': 'Molecular: Get cell class information',
                'modality': Modality.MOLECULAR,
            },
            {
                'nodes': ['Cluster', 'Subclass'],
                'relationships': ['HAVE_SUBCLASS'],
                'description': 'Molecular: Get subclass markers',
                'modality': Modality.MOLECULAR,
            },
            {
                'nodes': ['Cluster', 'BrainRegion'],
                'relationships': ['ENRICHED_IN'],
                'description': 'Molecular: Spatial distribution',
                'modality': Modality.MOLECULAR,
            },
            # 形态特征路径
            {
                'nodes': ['Cluster', 'Morphology'],
                'relationships': ['HAS_MORPHOLOGY'],
                'description': 'Morphological: Neuron structure',
                'modality': Modality.MORPHOLOGICAL,
            },
            {
                'nodes': ['Morphology', 'BrainRegion'],
                'relationships': ['LOCATED_IN'],
                'description': 'Morphological: Reconstruction location',
                'modality': Modality.MORPHOLOGICAL,
            },
            # 投射特征路径
            {
                'nodes': ['BrainRegion', 'TargetRegion'],
                'relationships': ['PROJECT_TO'],
                'description': 'Projection: Output targets',
                'modality': Modality.PROJECTION,
            },
            # 闭环路径 - 对应设计图中的CLOSED LOOP
            {
                'nodes': ['Cluster', 'BrainRegion', 'TargetRegion'],
                'relationships': ['ENRICHED_IN', 'PROJECT_TO'],
                'description': 'Closed Loop: Source region to projection targets',
                'modality': Modality.PROJECTION,
            },
        ]

        # 根据焦点模态筛选路径
        for template in COMPREHENSIVE_TEMPLATES:
            if not focus_modalities or template['modality'] in focus_modalities:
                paths.append(SchemaPath(
                    path_id=path_id,
                    nodes=template['nodes'],
                    relationships=template['relationships'],
                    description=template['description'],
                    estimated_cost=len(template['nodes']),
                ))
                path_id += 1

        return paths

    def _generate_comparative_paths(self, entities: List[Entity],
                                    focus_modalities: List[Modality],
                                    max_paths: int) -> List[SchemaPath]:
        """生成比较分析路径"""
        paths = []
        path_id = 0

        # 为每个实体生成相似的查询路径
        base_paths = [
            {
                'nodes': ['Cluster'],
                'relationships': [],
                'description': 'Compare: Basic cluster info for {entity}',
            },
            {
                'nodes': ['Cluster', 'BrainRegion'],
                'relationships': ['ENRICHED_IN'],
                'description': 'Compare: Spatial distribution of {entity}',
            },
            {
                'nodes': ['Cluster', 'Morphology'],
                'relationships': ['HAS_MORPHOLOGY'],
                'description': 'Compare: Morphology of {entity}',
            },
        ]

        for entity in entities:
            for base in base_paths:
                paths.append(SchemaPath(
                    path_id=path_id,
                    nodes=base['nodes'],
                    relationships=base['relationships'],
                    description=base['description'].format(entity=entity.name),
                    estimated_cost=len(base['nodes']),
                ))
                path_id += 1

        return paths

    def _optimize_paths(self, paths: List[SchemaPath]) -> List[SchemaPath]:
        """优化路径顺序"""

        # 按成本和覆盖度排序
        # 优先执行：
        # 1. 短路径（快速获取基础信息）
        # 2. 高覆盖路径（闭环路径）

        def path_priority(path: SchemaPath) -> Tuple[int, int]:
            is_closed_loop = 'Closed Loop' in path.description
            return (0 if is_closed_loop else 1, path.estimated_cost)

        return sorted(paths, key=path_priority)


# ==================== Query Generator ====================

class CypherQueryGenerator:
    """
    Cypher查询生成器

    根据SchemaPath生成Neo4j Cypher查询
    """

    def __init__(self, schema: type = KnowledgeGraphSchema):
        self.schema = schema

    def generate_query(self, path: SchemaPath,
                       constraints: Dict[str, Any]) -> str:
        """
        根据路径和约束生成Cypher查询

        Args:
            path: Schema路径
            constraints: 查询约束（如marker名、region等）

        Returns:
            Cypher查询字符串
        """
        # 构建MATCH子句
        match_parts = []
        var_names = []

        for i, node in enumerate(path.nodes):
            var_name = f"n{i}"
            var_names.append(var_name)

            if i == 0:
                # 第一个节点可能有约束
                match_parts.append(f"({var_name}:{node})")
            else:
                rel_name = path.relationships[i - 1]
                rel = self.schema.get_relationship(rel_name)

                if rel and rel.from_node == path.nodes[i - 1]:
                    match_parts.append(f"-[:{rel_name}]->({var_name}:{node})")
                else:
                    match_parts.append(f"<-[:{rel_name}]-({var_name}:{node})")

        match_clause = "MATCH " + "".join(match_parts)

        # 构建WHERE子句
        where_conditions = []

        if 'marker' in constraints:
            marker = constraints['marker']
            where_conditions.append(
                f"(n0.markers CONTAINS '{marker}' OR n0.cluster_id CONTAINS '{marker}')"
            )

        if 'region' in constraints:
            region = constraints['region']
            for i, node in enumerate(path.nodes):
                if node == 'BrainRegion':
                    where_conditions.append(f"n{i}.acronym = '{region}'")
                    break

        where_clause = ""
        if where_conditions:
            where_clause = " WHERE " + " AND ".join(where_conditions)

        # 构建RETURN子句
        return_items = []
        for i, (var, node) in enumerate(zip(var_names, path.nodes)):
            node_def = self.schema.get_node(node)
            if node_def:
                props = ", ".join([f"{var}.{p} AS {node}_{p}" for p in node_def.properties[:3]])
                return_items.append(props)

        return_clause = " RETURN " + ", ".join(return_items) if return_items else f" RETURN {var_names[0]}"

        # 添加LIMIT
        limit_clause = " LIMIT 100"

        return match_clause + where_clause + return_clause + limit_clause

    def generate_aggregation_query(self, base_query: str,
                                   aggregation_type: str) -> str:
        """生成聚合查询"""
        if aggregation_type == 'count':
            return base_query.replace('RETURN', 'RETURN COUNT(DISTINCT n0) AS count,')
        elif aggregation_type == 'sum':
            return base_query.replace('RETURN', 'RETURN SUM(n0.cell_count) AS total,')
        return base_query


# ==================== Export ====================

__all__ = [
    'SchemaNode',
    'SchemaRelationship',
    'KnowledgeGraphSchema',
    'DynamicSchemaPathPlanner',
    'CypherQueryGenerator',
]