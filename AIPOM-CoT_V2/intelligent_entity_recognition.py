"""
Intelligent Entity Recognition System
=====================================
KG-powered entity discovery with NLP and fuzzy matching

优化点:
1. 一次性构建完整KG索引 (初始化时完成)
2. 使用rapidfuzz加速模糊匹配
3. 支持中英文、缩写、全名多种形式
4. 上下文感知的实体聚类

Author: Claude & PrometheusTT
Date: 2025-01-12
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging

import spacy
from rapidfuzz import fuzz, process

from neo4j_exec import Neo4jExec
from aipom_cot_true_agent_v2 import RealSchemaCache

logger = logging.getLogger(__name__)


# ==================== Entity Data Structures ====================

@dataclass
class EntityMatch:
    """单个实体匹配"""
    text: str  # 原始文本
    entity_id: str  # KG中的ID或值
    entity_type: str  # Node label或类别
    match_type: str  # 'node_name' | 'property_value' | 'acronym'
    confidence: float  # 0-1
    context: Dict  # 额外信息 {name, props, ...}


@dataclass
class EntityCluster:
    """相关实体的聚合"""
    primary_entity: EntityMatch
    related_entities: List[EntityMatch]
    cluster_type: str  # 'gene_marker' | 'region' | 'cell_type'
    relevance_score: float  # 与问题的相关性


# ==================== Entity Index Builder ====================

class KGEntityIndexer:
    """
    构建完整KG实体索引 (初始化时一次性完成)
    优化: 使用内存索引 + rapidfuzz加速查询
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema

        # 索引结构
        self.indices = {
            'nodes': {},  # {label: [(id, name, acronym, props), ...]}
            'gene_markers': set(),  # Set of gene names
            'regions': {},  # {acronym: (id, full_name, props)}
            'clusters': {},  # {cluster_name: (id, markers, props)}
        }

        logger.info("🔍 Building comprehensive KG entity index...")
        self._build_all_indices()
        logger.info("✅ Entity index ready!")

    def _build_all_indices(self):
        """构建所有索引"""
        self._index_regions()
        self._index_clusters_and_genes()
        self._index_cell_types()
        self._index_neurons()

    def _index_regions(self):
        """索引所有脑区"""
        query = """
        MATCH (r:Region)
        RETURN elementId(r) AS id,
               r.acronym AS acronym,
               r.name AS name,
               r.full_name AS full_name,
               properties(r) AS props
        LIMIT 5000
        """

        result = self.db.run(query)
        if not result['success']:
            logger.warning("Failed to index regions")
            return

        for row in result['data']:
            acronym = row['acronym']
            if acronym:
                self.indices['regions'][acronym.upper()] = {
                    'id': row['id'],
                    'name': row['name'],
                    'full_name': row['full_name'],
                    'props': row['props']
                }

        logger.info(f"  ✓ Indexed {len(self.indices['regions'])} regions")

    def _index_clusters_and_genes(self):
        """索引Cluster和提取gene markers"""
        query = """
        MATCH (c:Cluster)
        RETURN elementId(c) AS id,
               c.name AS name,
               c.markers AS markers,
               c.number_of_neurons AS neuron_count,
               properties(c) AS props
        LIMIT 3000
        """

        result = self.db.run(query)
        if not result['success']:
            logger.warning("Failed to index clusters")
            return

        for row in result['data']:
            cluster_name = row['name']
            if cluster_name:
                self.indices['clusters'][cluster_name] = {
                    'id': row['id'],
                    'markers': row['markers'],
                    'neuron_count': row['neuron_count'],
                    'props': row['props']
                }

            # 提取gene markers
            markers = row['markers']
            if markers:
                genes = [g.strip() for g in markers.split(',')]
                self.indices['gene_markers'].update(genes)

        logger.info(f"  ✓ Indexed {len(self.indices['clusters'])} clusters")
        logger.info(f"  ✓ Extracted {len(self.indices['gene_markers'])} unique gene markers")

    def _index_cell_types(self):
        """索引细胞类型 (Class, Subclass, Supertype)"""
        for label in ['Class', 'Subclass', 'Supertype']:
            query = f"""
            MATCH (n:{label})
            RETURN elementId(n) AS id,
                   n.name AS name,
                   properties(n) AS props
            LIMIT 1000
            """

            result = self.db.run(query)
            if result['success']:
                self.indices['nodes'][label] = [
                    {
                        'id': row['id'],
                        'name': row['name'],
                        'props': row['props']
                    }
                    for row in result['data']
                ]
                logger.info(f"  ✓ Indexed {len(self.indices['nodes'][label])} {label} nodes")

    def _index_neurons(self):
        """索引神经元样本 (用于形态学查询)"""
        query = """
        MATCH (n:Neuron)
        RETURN elementId(n) AS id,
               properties(n) AS props
        LIMIT 500
        """

        result = self.db.run(query)
        if result['success']:
            self.indices['nodes']['Neuron'] = [
                {'id': row['id'], 'props': row['props']}
                for row in result['data']
            ]
            logger.info(f"  ✓ Indexed {len(self.indices['nodes']['Neuron'])} neurons")


# ==================== Intelligent Entity Recognizer ====================

class IntelligentEntityRecognizer:
    """
    主实体识别器

    Pipeline:
    1. NLP tokenization (spaCy)
    2. Multi-level matching (exact, fuzzy, pattern)
    3. Context-aware filtering
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema

        # 加载spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # 构建索引
        self.indexer = KGEntityIndexer(db, schema)

    def recognize_entities(self, question: str) -> List[EntityMatch]:
        """
        识别实体 (修复版 - 不依赖缺失的方法)

        🔧 修复:
        1. 移除对 _match_neurons 的调用
        2. 添加调试日志
        3. 简化流程
        """
        logger.info(f"🔍 Recognizing entities in: {question}")

        all_matches = []

        # Extract tokens
        tokens = self._extract_tokens(question)
        logger.debug(f"   Tokens extracted: {tokens[:10]}")

        # 1. Gene markers
        try:
            gene_matches = self._match_gene_markers(tokens, question)
            all_matches.extend(gene_matches)
            if gene_matches:
                logger.debug(f"   Found {len(gene_matches)} gene markers")
        except Exception as e:
            logger.error(f"   Error matching gene markers: {e}")

        # 2. Regions (最重要!)
        try:
            region_matches = self._match_regions(tokens, question)
            all_matches.extend(region_matches)
            if region_matches:
                logger.debug(f"   Found {len(region_matches)} regions")
        except Exception as e:
            logger.error(f"   Error matching regions: {e}")
            import traceback
            traceback.print_exc()

        # 3. Cell types
        try:
            cell_type_matches = self._match_cell_types(tokens, question)
            all_matches.extend(cell_type_matches)
            if cell_type_matches:
                logger.debug(f"   Found {len(cell_type_matches)} cell types")
        except Exception as e:
            logger.error(f"   Error matching cell types: {e}")

        # 🔧 移除对 _match_neurons 的调用 (因为该方法不存在)
        # neuron_matches = self._match_neurons(tokens, question)
        # all_matches.extend(neuron_matches)

        # Report results
        if all_matches:
            logger.info(f"   ✅ Found {len(all_matches)} entities")
            for m in all_matches[:5]:
                logger.debug(f"      • {m.text} ({m.entity_type}) via {m.context.get('source', 'unknown')}")
        else:
            logger.warning(f"   ⚠️ No entities found in: {question}")

        # Deduplicate
        seen = set()
        unique_matches = []
        for match in all_matches:
            key = (match.entity_id, match.entity_type)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        # Sort by confidence
        unique_matches.sort(key=lambda x: x.confidence, reverse=True)

        return unique_matches

    def _extract_tokens(self, text: str) -> List[str]:
        """
        提取有意义的tokens

        使用:
        - NLP词性标注
        - 领域特定模式 (如Car3+, L5, IT-type)
        - 大写缩写识别
        """
        doc = self.nlp(text)
        tokens = []

        # 1. NLP entities
        for ent in doc.ents:
            tokens.append(ent.text)

        # 2. Nouns and proper nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2:
                tokens.append(token.text)

        # 3. Gene-like patterns (e.g., Car3, Pvalb, Sst)
        gene_pattern = r'\b[A-Z][a-z]{2,}[0-9]?\b'
        genes = re.findall(gene_pattern, text)
        tokens.extend(genes)

        # 4. Gene+ patterns (e.g., Car3+)
        plus_pattern = r'\b([A-Z][a-z]+[0-9]*)\+\b'
        plus_genes = re.findall(plus_pattern, text)
        tokens.extend(plus_genes)

        # 5. Uppercase acronyms (2-6 letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        tokens.extend(acronyms)

        # 6. Special patterns
        # "layer 5" -> "L5"
        layer_match = re.findall(r'layer\s+(\d+)', text, re.IGNORECASE)
        tokens.extend([f"L{l}" for l in layer_match])

        # Deduplicate
        return list(set(tokens))

    def _match_gene_markers(self, tokens: List[str], full_text: str) -> List[EntityMatch]:
        """匹配基因marker"""
        matches = []

        gene_markers = self.indexer.indices['gene_markers']

        for token in tokens:
            # Exact match
            if token in gene_markers:
                # Check if "+" nearby (higher confidence)
                confidence = 0.95 if (token + '+') in full_text else 0.85

                matches.append(EntityMatch(
                    text=token,
                    entity_id=token,
                    entity_type='GeneMarker',
                    match_type='exact',
                    confidence=confidence,
                    context={'source': 'Cluster.markers'}
                ))
            else:
                # Fuzzy match
                best_matches = process.extract(
                    token,
                    gene_markers,
                    scorer=fuzz.ratio,
                    limit=3,
                    score_cutoff=80
                )

                for gene, score, _ in best_matches:
                    matches.append(EntityMatch(
                        text=token,
                        entity_id=gene,
                        entity_type='GeneMarker',
                        match_type='fuzzy',
                        confidence=score / 100.0 * 0.9,  # Slight penalty for fuzzy
                        context={'matched_gene': gene, 'source': 'Cluster.markers'}
                    ))

        return matches

    def _match_regions(self, tokens: List[str], full_text: str) -> List[EntityMatch]:
        """匹配regions (带停用词过滤)"""
        matches = []

        # 🔧 停用词列表 (避免误匹配常见词)
        STOPWORDS = {
            'ME', 'US', 'IT', 'IS', 'IN', 'ON', 'AT', 'TO', 'OF', 'AND', 'OR',
            'THE', 'A', 'AN', 'FOR', 'WITH', 'AS', 'BY', 'FROM', 'UP', 'OUT'
        }

        # 获取或重建region index
        region_acronyms = self.indexer.indices.get('region_acronyms', {})

        if not region_acronyms:
            logger.warning("   Region index empty, rebuilding...")
            try:
                query = "MATCH (r:Region) RETURN r.acronym AS acronym LIMIT 500"
                result = self.indexer.db.run(query)
                if result['success'] and result['data']:
                    region_acronyms = {row['acronym']: row['acronym'] for row in result['data'] if row.get('acronym')}
                    self.indexer.indices['region_acronyms'] = region_acronyms
                    logger.info(f"   Rebuilt region index: {len(region_acronyms)} regions")
            except Exception as e:
                logger.error(f"   Failed to rebuild region index: {e}")
                return matches

        import re

        # Strategy 1: Direct token matching (with stopword filter)
        for token in tokens:
            token_upper = token.strip('.,!?;: ').upper()

            # 🔧 跳过停用词
            if token_upper in STOPWORDS:
                continue

            if token_upper in region_acronyms:
                if not any(m.entity_id == token_upper for m in matches):
                    matches.append(EntityMatch(
                        text=token_upper,
                        entity_id=token_upper,
                        entity_type='Region',
                        match_type='exact',
                        confidence=0.95,
                        context={'source': 'token'}
                    ))

        # Strategy 2: "Compare A and B" pattern
        pattern = r'compare\s+(\w+)\s+and\s+(\w+)'
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            for idx in [1, 2]:
                entity = m.group(idx).upper()

                # 🔧 跳过停用词
                if entity in STOPWORDS:
                    continue

                if entity in region_acronyms:
                    if not any(match.entity_id == entity for match in matches):
                        matches.append(EntityMatch(
                            text=entity,
                            entity_id=entity,
                            entity_type='Region',
                            match_type='pattern',
                            confidence=0.95,
                            context={'source': 'compare'}
                        ))

        # Strategy 3: "A vs B" pattern
        pattern = r'(\w+)\s+vs\.?\s+(\w+)'
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            for idx in [1, 2]:
                entity = m.group(idx).upper()

                # 🔧 跳过停用词
                if entity in STOPWORDS:
                    continue

                if entity in region_acronyms:
                    if not any(match.entity_id == entity for match in matches):
                        matches.append(EntityMatch(
                            text=entity,
                            entity_id=entity,
                            entity_type='Region',
                            match_type='pattern',
                            confidence=0.95,
                            context={'source': 'vs'}
                        ))

        # Strategy 4: Word-by-word fallback (with stopword filter)
        if not matches:
            for word in re.findall(r'\b\w+\b', full_text):
                word_upper = word.upper()

                # 🔧 跳过停用词
                if word_upper in STOPWORDS:
                    continue

                if word_upper in region_acronyms:
                    if not any(m.entity_id == word_upper for m in matches):
                        matches.append(EntityMatch(
                            text=word_upper,
                            entity_id=word_upper,
                            entity_type='Region',
                            match_type='exact',
                            confidence=0.90,
                            context={'source': 'fallback'}
                        ))

        if matches:
            logger.info(f"   Matched regions: {[m.entity_id for m in matches]}")

        return matches

    def _match_cell_types(self, tokens: List[str], full_text: str) -> List[EntityMatch]:
        """匹配细胞类型"""
        matches = []

        # Predefined cell type keywords
        known_types = {
            'IT': 'Intratelencephalic',
            'ET': 'Extratelencephalic',
            'CT': 'Corticothalamic',
            'PT': 'Pyramidal tract',
            'NP': 'Near-projecting',
            'interneuron': 'Interneuron',
            'pyramidal': 'Pyramidal',
            'excitatory': 'Excitatory',
            'inhibitory': 'Inhibitory'
        }

        for token in tokens:
            token_lower = token.lower()

            if token in known_types or token_lower in known_types:
                cell_type = known_types.get(token) or known_types.get(token_lower)
                matches.append(EntityMatch(
                    text=token,
                    entity_id=cell_type,
                    entity_type='CellType',
                    match_type='keyword',
                    confidence=0.85,
                    context={'full_name': cell_type}
                ))

        # Match against Subclass nodes
        for label in ['Subclass', 'Class', 'Supertype']:
            if label in self.indexer.indices['nodes']:
                nodes = self.indexer.indices['nodes'][label]
                node_names = [n['name'] for n in nodes if n['name']]

                for token in tokens:
                    best_matches = process.extract(
                        token.lower(),
                        [n.lower() for n in node_names],
                        scorer=fuzz.ratio,
                        limit=2,
                        score_cutoff=80
                    )

                    for matched_name, score, _ in best_matches:
                        # Find original name
                        original = next(n for n in node_names if n.lower() == matched_name)
                        node_info = next(n for n in nodes if n['name'] == original)

                        matches.append(EntityMatch(
                            text=token,
                            entity_id=node_info['id'],
                            entity_type=label,
                            match_type='node_name',
                            confidence=score / 100.0 * 0.8,
                            context={'name': original}
                        ))

        return matches

    def _deduplicate(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """去重,保留最高confidence的匹配"""
        seen = {}

        for match in matches:
            key = (match.entity_type, match.entity_id)

            if key not in seen or match.confidence > seen[key].confidence:
                seen[key] = match

        return list(seen.values())


# ==================== Entity Clustering ====================

class EntityClusteringEngine:
    """
    将识别的实体聚类成有意义的组

    例如:
    - Car3 (gene) + MOs (region) -> 'gene_in_region' cluster
    - Pvalb (gene) + Sst (gene) -> 'gene_comparison' cluster
    """

    def __init__(self, db: Neo4jExec, schema: RealSchemaCache):
        self.db = db
        self.schema = schema

    def cluster_entities(self,
                         matches: List[EntityMatch],
                         question: str) -> List[EntityCluster]:
        """
        聚类实体

        策略:
        1. 按类型分组
        2. 查询KG找相关性
        3. 计算relevance score
        """
        clusters = []

        # Group by type
        genes = [m for m in matches if m.entity_type == 'GeneMarker']
        regions = [m for m in matches if m.entity_type == 'Region']
        cell_types = [m for m in matches if m.entity_type in ['CellType', 'Subclass', 'Class']]

        # Create clusters
        if genes:
            cluster = self._create_gene_cluster(genes, regions, question)
            if cluster:
                clusters.append(cluster)

        if regions and not genes:
            cluster = self._create_region_cluster(regions, question)
            if cluster:
                clusters.append(cluster)

        if cell_types:
            cluster = self._create_celltype_cluster(cell_types, question)
            if cluster:
                clusters.append(cluster)

        # Rank by relevance
        clusters.sort(key=lambda c: c.relevance_score, reverse=True)

        return clusters

    def _create_gene_cluster(self,
                             genes: List[EntityMatch],
                             regions: List[EntityMatch],
                             question: str) -> Optional[EntityCluster]:
        """创建基因为中心的cluster"""
        if not genes:
            return None

        primary_gene = genes[0]
        gene_name = primary_gene.entity_id

        # Query KG for related clusters
        query = """
        MATCH (c:Cluster)
        WHERE c.markers CONTAINS $gene
        RETURN c.name AS cluster,
               c.markers AS markers,
               elementId(c) AS id
        LIMIT 10
        """

        result = self.db.run(query, {'gene': gene_name})

        related = []
        if result['success']:
            for row in result['data']:
                related.append(EntityMatch(
                    text=row['cluster'],
                    entity_id=row['id'],
                    entity_type='Cluster',
                    match_type='related_to_gene',
                    confidence=0.8,
                    context={'markers': row['markers']}
                ))

        # Add regions if any
        related.extend(regions)

        # Calculate relevance
        relevance = 0.9
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['gene', 'marker', 'express']):
            relevance *= 1.2

        return EntityCluster(
            primary_entity=primary_gene,
            related_entities=related,
            cluster_type='gene_marker',
            relevance_score=min(1.0, relevance)
        )

    def _create_region_cluster(self,
                               regions: List[EntityMatch],
                               question: str) -> Optional[EntityCluster]:
        """创建区域为中心的cluster"""
        if not regions:
            return None

        primary_region = regions[0]

        relevance = 0.85
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['region', 'area', 'brain']):
            relevance *= 1.2

        return EntityCluster(
            primary_entity=primary_region,
            related_entities=regions[1:],
            cluster_type='region',
            relevance_score=min(1.0, relevance)
        )

    def _create_celltype_cluster(self,
                                 cell_types: List[EntityMatch],
                                 question: str) -> Optional[EntityCluster]:
        """创建细胞类型cluster"""
        if not cell_types:
            return None

        primary = cell_types[0]

        relevance = 0.8
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['cell', 'neuron', 'type']):
            relevance *= 1.2

        return EntityCluster(
            primary_entity=primary,
            related_entities=cell_types[1:],
            cluster_type='cell_type',
            relevance_score=min(1.0, relevance)
        )


# ==================== Test ====================

if __name__ == "__main__":
    import os
    from neo4j_exec import Neo4jExec
    from aipom_cot_true_agent_v2 import RealSchemaCache

    # Initialize
    db = Neo4jExec(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )

    schema = RealSchemaCache("./schema_output/schema.json")

    # Test entity recognition
    recognizer = IntelligentEntityRecognizer(db, schema)

    test_questions = [
        "Tell me about Car3+ neurons",
        "Compare Pvalb and Sst interneurons in MOs",
        "What are the projection targets of claustrum?",
        "Analyze layer 5 pyramidal neurons morphology"
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print('=' * 60)

        matches = recognizer.recognize_entities(q)

        for m in matches[:5]:
            print(f"  • {m.text} ({m.entity_type}) [{m.confidence:.2f}]")
            print(f"    Match: {m.match_type}, ID: {m.entity_id}")