"""
Baseline Methods for AIPOM-CoT Benchmark (v2.1 - GPT-5)
========================================================
包含4个baseline方法：
1. Direct GPT-5 - 最强LLM baseline
2. Template-KG - 模板化KG查询
3. RAG - 检索增强生成 (with GPT-5)
4. ReAct - 推理+行动 (with GPT-5)

Changes in v2.1:
- 使用GPT-5替代所有LLM调用
- 移除o1-preview（使用GPT-5作为SOTA baseline）

Author: Claude & PrometheusTT
Date: 2025-01-15
Version: 2.1
"""

import time
import json
import logging
import re
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== Abstract Base Class ====================

class BaselineAgent(ABC):
    """Baseline抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        """回答问题"""
        pass


# ==================== Baseline 1: Direct GPT-5 ====================

class DirectGPT5Baseline(BaselineAgent):
    """
    Direct GPT-5 Baseline (SOTA LLM)

    特点：
    - 使用最新的GPT-5模型
    - 无KG访问
    - 纯粹依赖预训练知识
    - 单次推理（fast）

    优势：
    - SOTA语言理解和推理能力
    - 速度快
    - 对常识性问题表现好

    劣势：
    - 无法访问最新/专有数据
    - 可能产生幻觉
    - 无系统分析能力
    """

    def __init__(self, openai_client):
        super().__init__("Direct GPT-5")
        self.client = openai_client
        self.model = "gpt-5"

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        """使用GPT-5直接回答"""
        start_time = time.time()

        # 🔧 高质量的system prompt
        system_prompt = """You are an expert neuroscientist with deep knowledge of:
- Brain anatomy and neuroanatomy (Allen Mouse Brain Atlas)
- Cell types and molecular markers (Pvalb, Sst, VIP, Car3, etc.)
- Neuronal morphology and electrophysiology
- Brain connectivity and neural circuits
- Mouse brain regions and their functions

Provide scientifically accurate, detailed answers based on your knowledge.
Include specific quantitative data when possible (neuron counts, connectivity strengths, etc.).
If you're uncertain about specific details, acknowledge it rather than speculate."""

        user_prompt = f"""Question about neuroscience:

{question}

Please provide a comprehensive, scientifically rigorous answer that includes:
1. Direct answer to the question
2. Relevant molecular markers or cell types (if applicable)
3. Brain regions involved (if applicable)
4. Quantitative data when available
5. Key scientific context

Answer:"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1500,
                timeout=timeout
            )

            answer = completion.choices[0].message.content
            execution_time = time.time() - start_time

            # 提取实体
            entities_recognized = self._extract_entities_heuristic(answer)

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': [{
                    'purpose': 'Direct GPT-5 inference',
                    'modality': None,
                }],
                'schema_paths_used': [],
                'execution_time': execution_time,
                'total_steps': 1,
                'confidence_score': 0.75,  # 高置信（SOTA LLM）
                'success': True,
                'method': 'Direct GPT-5',
            }

        except Exception as e:
            logger.error(f"Direct GPT-5 failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_entities_heuristic(self, answer: str) -> List[Dict]:
        """启发式提取实体"""
        entities = []

        # 提取脑区缩写 (2-5个大写字母)
        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)
        for r in set(regions):
            # 排除常见非脑区词
            if r not in ['DNA', 'RNA', 'ATP', 'GABA', 'LLM', 'GPT', 'USA', 'PHD']:
                entities.append({
                    'text': r,
                    'type': 'Region',
                    'confidence': 0.7,
                })

        # 提取基因名 (首字母大写 + 小写字母)
        genes = re.findall(r'\b[A-Z][a-z]{2,8}\+?\b', answer)
        for g in set(genes):
            # 排除常见非基因词
            if g not in ['The', 'This', 'That', 'There', 'These', 'Their', 'When', 'Where', 'Which']:
                entities.append({
                    'text': g.rstrip('+'),
                    'type': 'Gene',
                    'confidence': 0.6,
                })

        return entities[:15]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Direct GPT-5',
            'error': error,
        }


# ==================== Baseline 2: Template-KG ====================

class TemplateKGBaseline(BaselineAgent):
    """
    Template-based Knowledge Graph Query Baseline

    特点：
    - 使用预定义查询模板
    - 有KG访问（公平对比）
    - 无自适应能力
    - 使用GPT-5合成答案
    """

    def __init__(self, neo4j_exec, openai_client):
        super().__init__("Template-KG")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-5"  # 🔧 使用GPT-5
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict:
        """构建查询模板库"""
        return {
            # 模板1：基因 → 细胞簇
            'gene_to_clusters': """
                MATCH (c:Cluster)
                WHERE c.markers CONTAINS $gene
                RETURN c.name AS cluster, 
                       c.number_of_neurons AS neurons,
                       c.broad_region_distribution AS regions,
                       c.markers AS markers
                ORDER BY c.number_of_neurons DESC
                LIMIT 20
            """,

            # 模板2：脑区 → 细胞簇
            'region_to_clusters': """
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE r.acronym = $region
                RETURN r.name AS region_name, 
                       c.name AS cluster,
                       c.markers AS markers, 
                       c.number_of_neurons AS neurons
                ORDER BY c.number_of_neurons DESC
                LIMIT 30
            """,

            # 模板3：脑区 → 投射
            'region_projections': """
                MATCH (r:Region)-[p:PROJECT_TO]->(t:Region)
                WHERE r.acronym = $region
                RETURN r.name AS source, 
                       t.acronym AS target, 
                       t.name AS target_name,
                       p.weight AS weight,
                       p.neuron_count AS neuron_count
                ORDER BY p.weight DESC
                LIMIT 20
            """,

            # 模板4：脑区 → 形态
            'region_morphology': """
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region)
                WHERE r.acronym = $region
                RETURN r.name AS region,
                       count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon_length,
                       avg(n.dendritic_length) AS avg_dendrite_length,
                       avg(n.axonal_branches) AS avg_axon_branches,
                       avg(n.dendritic_branches) AS avg_dendrite_branches
            """,

            # 模板5：基因 → 脑区（enrichment）
            'gene_to_regions': """
                MATCH (r:Region)-[:HAS_CLUSTER]->(c:Cluster)
                WHERE c.markers CONTAINS $gene
                WITH r, count(c) AS cluster_count, sum(c.number_of_neurons) AS total_neurons
                RETURN r.acronym AS region,
                       r.name AS region_name,
                       cluster_count,
                       total_neurons
                ORDER BY total_neurons DESC
                LIMIT 15
            """,
        }

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        """使用模板回答问题"""
        start_time = time.time()

        try:
            # Step 1: 分类问题
            question_type = self._classify_question(question)
            logger.info(f"  Template-KG: Classified as '{question_type}'")

            # Step 2: 提取参数
            params = self._extract_parameters(question)
            logger.info(f"  Template-KG: Extracted params: {params}")

            if not params:
                return self._fallback_answer(question, time.time() - start_time)

            # Step 3: 执行模板
            results = []
            executed_steps = []

            if question_type == 'gene_profiling':
                results, executed_steps = self._execute_gene_profiling(params)

            elif question_type == 'region_analysis':
                results, executed_steps = self._execute_region_analysis(params)

            elif question_type == 'projection_query':
                results, executed_steps = self._execute_projection_query(params)

            else:
                results, executed_steps = self._execute_simple_lookup(params)

            # Step 4: 合成答案（使用GPT-5）
            if not results or not any(r.get('success') for r in results):
                return self._fallback_answer(question, time.time() - start_time)

            answer = self._synthesize_answer(question, results)

            execution_time = time.time() - start_time

            # 提取实体
            entities_recognized = []
            for key, value in params.items():
                if value:
                    entities_recognized.append({
                        'text': value,
                        'type': 'Gene' if key == 'gene' else 'Region',
                        'confidence': 1.0,
                    })

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': executed_steps,
                'schema_paths_used': [s['template'] for s in executed_steps],
                'execution_time': execution_time,
                'total_steps': len(executed_steps),
                'confidence_score': 0.7,
                'success': True,
                'method': 'Template-KG',
            }

        except Exception as e:
            logger.error(f"Template-KG failed: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(question, str(e), time.time() - start_time)

    def _classify_question(self, question: str) -> str:
        """分类问题类型"""
        q_lower = question.lower()

        if any(kw in q_lower for kw in ['tell me about', 'about', 'profile', 'characterize']):
            if any(kw in q_lower for kw in ['+', 'neuron', 'cell', 'interneuron']):
                return 'gene_profiling'
            else:
                return 'region_analysis'

        if any(kw in q_lower for kw in ['projection', 'project', 'target', 'connectivity']):
            return 'projection_query'

        return 'simple_lookup'

    def _extract_parameters(self, question: str) -> Dict:
        """提取查询参数"""
        params = {}

        # 提取基因名
        genes = re.findall(r'\b([A-Z][a-z]{2,8})\+?', question)
        if genes:
            stopwords = {'What', 'Which', 'Where', 'Tell', 'Give', 'Show', 'Find', 'The', 'This', 'That'}
            for g in genes:
                if g not in stopwords:
                    params['gene'] = g
                    break

        # 提取脑区缩写
        regions = re.findall(r'\b([A-Z]{2,5})\b', question)
        known_regions = {
            'MOp', 'MOs', 'SSp', 'SSs', 'VISp', 'VISal', 'VISam', 'VISl', 'VISpm',
            'AUDp', 'AUDpo', 'AUDv', 'ACA', 'PL', 'ILA', 'ORB',
            'RSP', 'CLA', 'HPF', 'HIP', 'TH', 'HY'
        }
        for r in regions:
            if r in known_regions:
                params['region'] = r
                break

        return params

    def _execute_gene_profiling(self, params: Dict) -> tuple:
        """执行基因profiling模板序列"""
        gene = params.get('gene')
        if not gene:
            return [], []

        results = []
        steps = []

        # Step 1: Gene -> Clusters
        result1 = self.db.run(self.templates['gene_to_clusters'], {'gene': gene})
        results.append(result1)
        steps.append({
            'purpose': f'Find clusters expressing {gene}',
            'template': 'gene_to_clusters',
            'modality': 'molecular',
            'success': result1.get('success', False),
        })

        # Step 2: Gene -> Regions
        result2 = self.db.run(self.templates['gene_to_regions'], {'gene': gene})
        results.append(result2)
        steps.append({
            'purpose': f'Find regions enriched for {gene}',
            'template': 'gene_to_regions',
            'modality': 'molecular',
            'success': result2.get('success', False),
        })

        # Step 3: 如果找到了primary region，查询morphology
        if result2.get('success') and result2.get('data'):
            top_region = result2['data'][0].get('region')
            if top_region:
                result3 = self.db.run(self.templates['region_morphology'], {'region': top_region})
                results.append(result3)
                steps.append({
                    'purpose': f'Morphology of {top_region}',
                    'template': 'region_morphology',
                    'modality': 'morphological',
                    'success': result3.get('success', False),
                })

        return results, steps

    def _execute_region_analysis(self, params: Dict) -> tuple:
        """执行脑区分析模板序列"""
        region = params.get('region')
        if not region:
            return [], []

        results = []
        steps = []

        # Step 1: Region -> Clusters
        result1 = self.db.run(self.templates['region_to_clusters'], {'region': region})
        results.append(result1)
        steps.append({
            'purpose': f'Cell types in {region}',
            'template': 'region_to_clusters',
            'modality': 'molecular',
            'success': result1.get('success', False),
        })

        # Step 2: Region -> Morphology
        result2 = self.db.run(self.templates['region_morphology'], {'region': region})
        results.append(result2)
        steps.append({
            'purpose': f'Morphology of {region}',
            'template': 'region_morphology',
            'modality': 'morphological',
            'success': result2.get('success', False),
        })

        # Step 3: Region -> Projections
        result3 = self.db.run(self.templates['region_projections'], {'region': region})
        results.append(result3)
        steps.append({
            'purpose': f'Projections from {region}',
            'template': 'region_projections',
            'modality': 'projection',
            'success': result3.get('success', False),
        })

        return results, steps

    def _execute_projection_query(self, params: Dict) -> tuple:
        """执行投射查询"""
        region = params.get('region')
        if not region:
            return [], []

        results = []
        steps = []

        result = self.db.run(self.templates['region_projections'], {'region': region})
        results.append(result)
        steps.append({
            'purpose': f'Projections from {region}',
            'template': 'region_projections',
            'modality': 'projection',
            'success': result.get('success', False),
        })

        return results, steps

    def _execute_simple_lookup(self, params: Dict) -> tuple:
        """执行简单查询"""
        results = []
        steps = []

        if 'gene' in params:
            result = self.db.run(self.templates['gene_to_clusters'], params)
            results.append(result)
            steps.append({
                'purpose': f'Lookup {params["gene"]}',
                'template': 'gene_to_clusters',
                'modality': 'molecular',
                'success': result.get('success', False),
            })

        elif 'region' in params:
            result = self.db.run(self.templates['region_to_clusters'], params)
            results.append(result)
            steps.append({
                'purpose': f'Lookup {params["region"]}',
                'template': 'region_to_clusters',
                'modality': 'molecular',
                'success': result.get('success', False),
            })

        return results, steps

    def _synthesize_answer(self, question: str, results: List[Dict]) -> str:
        """合成答案（使用GPT-5）"""
        # 收集所有成功的数据
        all_data = []
        for result in results:
            if result.get('success') and result.get('data'):
                all_data.extend(result['data'][:10])

        if not all_data:
            return "No data found in knowledge graph."

        # 格式化为context
        context = "Data from Knowledge Graph:\n"
        for i, row in enumerate(all_data[:20], 1):
            context += f"\n{i}. "
            context += ", ".join(f"{k}: {v}" for k, v in list(row.items())[:5])

        # 🔧 使用GPT-5合成
        prompt = f"""Based on the following data from a neuroscience knowledge graph, provide a comprehensive answer.

Question: {question}

{context}

Provide a detailed, scientific answer using ONLY the data above. Include quantitative details and be precise."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a neuroscience expert analyzing knowledge graph data."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000,
                timeout=30
            )

            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"GPT-5 synthesis failed: {e}")
            return f"Based on knowledge graph data: Found {len(all_data)} relevant entries. " + context[:500]

    def _fallback_answer(self, question: str, elapsed: float) -> Dict:
        """Fallback answer"""
        return {
            'question': question,
            'answer': "Unable to extract parameters or execute templates for this question.",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Template-KG',
        }

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'Template-KG',
            'error': error,
        }


# ==================== Baseline 3: RAG (with GPT-5) ====================

class RAGBaseline(BaselineAgent):
    """RAG baseline (使用GPT-5)"""

    def __init__(self, neo4j_exec, openai_client):
        super().__init__("RAG")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-5"  # 🔧 使用GPT-5

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        start_time = time.time()

        # 提取关键词
        keywords = self._extract_keywords(question)
        logger.info(f"  RAG keywords: {keywords}")

        # 检索文档
        docs = self._retrieve_documents(keywords, top_k=10)
        logger.info(f"  RAG retrieved {len(docs)} documents")

        # 构建prompt
        if docs:
            context = self._format_documents(docs)
        else:
            context = "No relevant documents found in the knowledge graph."

        # 生成答案（使用GPT-5）
        try:
            answer = self._generate_answer(question, context, timeout)
            execution_time = time.time() - start_time

            entities_recognized = self._extract_entities_from_docs(docs)

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': [{
                    'purpose': f'Retrieved {len(docs)} documents from KG',
                    'modality': 'retrieval',
                }],
                'schema_paths_used': [],
                'execution_time': execution_time,
                'total_steps': 1,
                'confidence_score': 0.6,
                'success': True,
                'method': 'RAG',
            }

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词"""
        keywords = re.findall(r'\b[A-Z]{2,5}\b', question)
        keywords.extend(re.findall(r'\b[A-Z][a-z]{2,8}\+?\b', question))

        neuro_terms = [
            'neuron', 'neurons', 'cell', 'cells', 'cortex', 'region',
            'brain', 'axon', 'dendrite', 'projection', 'marker', 'cluster'
        ]
        q_lower = question.lower()
        keywords.extend([term for term in neuro_terms if term in q_lower])

        return list(set(keywords))[:5]

    def _retrieve_documents(self, keywords: List[str], top_k: int = 10) -> List[Dict]:
        """检索文档"""
        docs = []

        for keyword in keywords:
            # Region
            query_region = """
            MATCH (r:Region)
            WHERE r.acronym CONTAINS $keyword OR r.name CONTAINS $keyword
            RETURN 'Region' AS type, r.acronym AS acronym, r.name AS name, 
                   r.number_of_transcriptomic_neurons AS neuron_count
            LIMIT 3
            """
            result = self.db.run(query_region, {'keyword': keyword})
            if result.get('success') and result.get('data'):
                docs.extend(result['data'])

            # Cluster
            query_cluster = """
            MATCH (c:Cluster)
            WHERE c.markers CONTAINS $keyword
            RETURN 'Cluster' AS type, c.name AS cluster_name, 
                   c.markers AS markers, c.number_of_neurons AS neurons
            ORDER BY c.number_of_neurons DESC
            LIMIT 3
            """
            result = self.db.run(query_cluster, {'keyword': keyword})
            if result.get('success') and result.get('data'):
                docs.extend(result['data'])

        # 去重
        seen = set()
        unique_docs = []
        for doc in docs:
            key = json.dumps(doc, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs[:top_k]

    def _format_documents(self, docs: List[Dict]) -> str:
        """格式化文档"""
        if not docs:
            return "No documents found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            doc_type = doc.get('type', 'Unknown')

            if doc_type == 'Region':
                text = f"Region: {doc.get('name', 'N/A')} ({doc.get('acronym', 'N/A')})"
                if doc.get('neuron_count'):
                    text += f"\n  Neurons: {doc['neuron_count']:,}"

            elif doc_type == 'Cluster':
                text = f"Cluster: {doc.get('cluster_name', 'N/A')}"
                if doc.get('markers'):
                    text += f"\n  Markers: {doc['markers']}"
                if doc.get('neurons'):
                    text += f"\n  Neurons: {doc['neurons']:,}"

            else:
                text = json.dumps(doc, indent=2)

            formatted.append(f"Document {i}:\n{text}")

        return "\n\n".join(formatted)

    def _generate_answer(self, question: str, context: str, timeout: int) -> str:
        """生成答案（使用GPT-5）"""

        system_prompt = """You are a neuroscience expert analyzing data from a knowledge graph.
Use ONLY the provided documents to answer the question.
Be precise and cite specific data from the documents.
If the documents don't contain sufficient information, acknowledge it."""

        user_prompt = f"""Based on the following documents from a neuroscience knowledge graph, answer the question.

Documents:
{context}

Question: {question}

Provide a detailed, scientific answer using ONLY information from the documents above.

Answer:"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            
            max_completion_tokens=1200,
            timeout=timeout
        )

        return completion.choices[0].message.content

    def _extract_entities_from_docs(self, docs: List[Dict]) -> List[Dict]:
        """从文档提取实体"""
        entities = []

        for doc in docs:
            doc_type = doc.get('type')

            if doc_type == 'Region':
                entities.append({
                    'text': doc.get('acronym', ''),
                    'type': 'Region',
                    'confidence': 1.0,
                })

            elif doc_type == 'Cluster':
                markers = doc.get('markers', '')
                if markers:
                    for marker in markers.split(',')[:3]:
                        entities.append({
                            'text': marker.strip(),
                            'type': 'Gene',
                            'confidence': 0.9,
                        })

        return entities[:10]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'RAG',
            'error': error,
        }


# ==================== Baseline 4: ReAct (with GPT-5) ====================

class ReActBaseline(BaselineAgent):
    """ReAct baseline (使用GPT-5，增加max_iterations)"""

    def __init__(self, neo4j_exec, openai_client, max_iterations=5):
        super().__init__("ReAct")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = "gpt-5"  # 🔧 使用GPT-5
        self.max_iterations = max_iterations

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        start_time = time.time()

        history = []
        executed_steps = []
        entities_recognized = []

        system_prompt = """You are a neuroscience expert with access to a knowledge graph database.

You can execute Cypher queries to retrieve information.

Use the ReAct framework:
1. Thought: Reason about what information you need
2. Action: Either "query" to execute a Cypher query, or "answer" to provide final answer
3. Query: If action is "query", provide a Cypher query
4. Observation: System will provide query results
5. Repeat until you can answer

Respond in JSON format:
{
  "thought": "your reasoning about what to do next",
  "action": "query" or "answer",
  "query": "MATCH ... RETURN ..." (if action is "query", null otherwise),
  "final_answer": "your answer" (if action is "answer", null otherwise)
}

Keep queries simple and focused. Limit results to 20 rows."""

        try:
            for iteration in range(self.max_iterations):
                logger.info(f"  ReAct iteration {iteration + 1}/{self.max_iterations}")

                if history:
                    context = "\n\n".join(history)
                else:
                    context = "Start your reasoning."

                prompt = f"""Question: {question}

Previous steps:
{context}

What's your next step? Respond in JSON format."""

                # LLM推理（使用GPT-5）
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=800,
                    timeout=timeout // self.max_iterations
                )

                result = json.loads(completion.choices[0].message.content)

                thought = result.get('thought', '')
                action = result.get('action', '')

                history.append(f"Thought: {thought}")
                logger.info(f"    Thought: {thought[:80]}...")

                # 回答
                if action == 'answer':
                    final_answer = result.get('final_answer', '')

                    execution_time = time.time() - start_time

                    return {
                        'question': question,
                        'answer': final_answer,
                        'entities_recognized': entities_recognized,
                        'executed_steps': executed_steps,
                        'schema_paths_used': [],
                        'execution_time': execution_time,
                        'total_steps': len(executed_steps),
                        'confidence_score': 0.7,
                        'success': True,
                        'method': 'ReAct',
                    }

                # 查询
                elif action == 'query':
                    query = result.get('query', '')

                    if not query:
                        logger.warning(f"    Empty query, skipping")
                        continue

                    history.append(f"Action: Execute query")
                    logger.info(f"    Executing query: {query[:80]}...")

                    db_result = self.db.run(query)

                    if db_result.get('success'):
                        data = db_result.get('data', [])[:20]
                        observation = f"Query returned {len(data)} results"

                        entities_recognized.extend(self._extract_entities_from_data(data))

                    else:
                        error = db_result.get('error', 'Unknown error')
                        observation = f"Query failed: {error}"
                        data = []

                    history.append(f"Observation: {observation}")
                    logger.info(f"    {observation}")

                    executed_steps.append({
                        'purpose': thought,
                        'query': query,
                        'result_count': len(data),
                        'success': db_result.get('success', False),
                        'modality': self._infer_modality(query),
                    })

            # 达到最大迭代
            logger.warning(f"  ReAct reached max iterations")

            final_answer = "Unable to complete analysis within iteration limit. "
            if executed_steps:
                final_answer += f"Executed {len(executed_steps)} queries but need more steps."
            else:
                final_answer += "Could not generate valid queries."

            return {
                'question': question,
                'answer': final_answer,
                'entities_recognized': entities_recognized,
                'executed_steps': executed_steps,
                'schema_paths_used': [],
                'execution_time': time.time() - start_time,
                'total_steps': len(executed_steps),
                'confidence_score': 0.4,
                'success': False,
                'method': 'ReAct',
            }

        except Exception as e:
            logger.error(f"ReAct failed: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(question, str(e), time.time() - start_time)

    def _infer_modality(self, query: str) -> str:
        """推断查询的modality"""
        query_lower = query.lower()

        if 'project' in query_lower or 'target' in query_lower:
            return 'projection'
        elif 'morpholog' in query_lower or 'axon' in query_lower or 'dendrit' in query_lower:
            return 'morphological'
        elif 'cluster' in query_lower or 'marker' in query_lower:
            return 'molecular'
        else:
            return None

    def _extract_entities_from_data(self, data: List[Dict]) -> List[Dict]:
        """从数据提取实体"""
        entities = []

        for row in data[:5]:
            for key, value in row.items():
                if isinstance(value, str):
                    if len(value) >= 2 and len(value) <= 5 and value.isupper():
                        entities.append({
                            'text': value,
                            'type': 'Region',
                            'confidence': 0.8,
                        })
                    elif len(value) >= 3 and value[0].isupper():
                        entities.append({
                            'text': value,
                            'type': 'Gene',
                            'confidence': 0.6,
                        })

        seen = set()
        unique = []
        for e in entities:
            key = (e['text'], e['type'])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:10]

    def _error_response(self, question: str, error: str, elapsed: float) -> Dict:
        return {
            'question': question,
            'answer': f"Error: {error}",
            'entities_recognized': [],
            'executed_steps': [],
            'schema_paths_used': [],
            'execution_time': elapsed,
            'total_steps': 0,
            'confidence_score': 0.0,
            'success': False,
            'method': 'ReAct',
            'error': error,
        }


# ==================== Factory Function ====================

def create_baseline(baseline_type: str, **kwargs) -> BaselineAgent:
    """工厂函数创建baseline"""

    if baseline_type == 'direct-gpt5':
        return DirectGPT5Baseline(
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'template-kg':
        return TemplateKGBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'rag':
        return RAGBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client']
        )

    elif baseline_type == 'react':
        return ReActBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client'],
            max_iterations=kwargs.get('max_iterations', 5)
        )

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# ==================== Test ====================

if __name__ == "__main__":
    print("✅ Updated baselines.py v2.1 (GPT-5) loaded successfully!")
    print("\nAvailable baselines:")
    print("1. Direct GPT-5 - SOTA LLM (no KG)")
    print("2. Template-KG - Template-based KG query (with GPT-5)")
    print("3. RAG - Retrieval + Generation (with GPT-5)")
    print("4. ReAct - Reasoning + Acting (with GPT-5, max_iter=5)")