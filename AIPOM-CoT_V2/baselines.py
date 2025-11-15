"""
Baseline Methods for AIPOM-CoT Benchmark
=========================================
实现3个高质量的baseline方法：
1. Direct LLM (GPT-4)
2. RAG (Retrieval-Augmented Generation)
3. ReAct (Reasoning + Acting)

设计原则：
- 不故意削弱baseline
- 给予baseline充分的提示工程
- 但展示AIPOM-CoT的独特优势

Author: Claude & PrometheusTT
Date: 2025-01-15
"""

import time
import json
import logging
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
        """
        回答问题

        Returns:
            {
                'question': str,
                'answer': str,
                'entities_recognized': List[Dict],
                'executed_steps': List[Dict],
                'schema_paths_used': List[Dict],
                'execution_time': float,
                'total_steps': int,
                'confidence_score': float,
                'success': bool,
                'method': str,
            }
        """
        pass


# ==================== Baseline 1: Direct LLM ====================

class DirectLLMBaseline(BaselineAgent):
    """
    Baseline 1: 直接使用LLM回答

    特点：
    - 无KG访问
    - 纯粹依赖预训练知识
    - 快速但可能幻觉

    优势：
    - 速度快
    - 对简单问题可能正确

    劣势：
    - 无法访问最新/详细数据
    - 容易幻觉
    - 无法进行多跳推理
    """

    def __init__(self, openai_client, model="gpt-4o"):
        super().__init__("Direct LLM")
        self.client = openai_client
        self.model = model

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        start_time = time.time()

        # 🔧 高质量的system prompt
        system_prompt = """You are an expert neuroscientist with deep knowledge of:
- Brain anatomy and neuroanatomy
- Cell types and molecular markers
- Neuronal morphology
- Brain connectivity and circuits
- Mouse brain atlas (Allen Brain Atlas)

Provide scientifically accurate, detailed answers based on your knowledge.
Include specific data when possible (numbers, region names, gene names).
If you're uncertain, acknowledge it rather than speculate."""

        # 🔧 高质量的user prompt
        user_prompt = f"""Question: {question}

Please provide a comprehensive, scientifically rigorous answer that includes:
1. Direct answer to the question
2. Relevant molecular markers or cell types (if applicable)
3. Brain regions involved (if applicable)
4. Quantitative data when available
5. Citations to established knowledge

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 低温度减少幻觉
                max_tokens=1500,
                timeout=timeout
            )

            answer = response.choices[0].message.content
            execution_time = time.time() - start_time

            # 尝试从答案中提取实体（简单启发式）
            entities_recognized = self._extract_entities_heuristic(answer)

            return {
                'question': question,
                'answer': answer,
                'entities_recognized': entities_recognized,
                'executed_steps': [{
                    'purpose': 'Direct LLM inference',
                    'modality': None,
                }],
                'schema_paths_used': [],
                'execution_time': execution_time,
                'total_steps': 1,
                'confidence_score': 0.5,  # 中等置信度（无验证）
                'success': True,
                'method': 'Direct LLM',
            }

        except Exception as e:
            logger.error(f"Direct LLM failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_entities_heuristic(self, answer: str) -> List[Dict]:
        """启发式提取实体（简单方法）"""
        import re

        entities = []

        # 提取大写缩写（可能是脑区）
        regions = re.findall(r'\b[A-Z]{2,5}\b', answer)
        for r in set(regions):
            if r not in ['DNA', 'RNA', 'ATP', 'GABA']:  # 排除常见非region缩写
                entities.append({
                    'text': r,
                    'type': 'Region',
                    'confidence': 0.6,
                })

        # 提取基因名模式
        genes = re.findall(r'\b[A-Z][a-z]{2,8}\+?\b', answer)
        for g in set(genes):
            if g not in ['The', 'This', 'These', 'That']:
                entities.append({
                    'text': g.rstrip('+'),
                    'type': 'Gene',
                    'confidence': 0.5,
                })

        return entities[:10]  # 最多返回10个

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
            'method': 'Direct LLM',
            'error': error,
        }


# ==================== Baseline 2: RAG ====================

class RAGBaseline(BaselineAgent):
    """
    Baseline 2: Retrieval-Augmented Generation

    特点：
    - 检索相关KG节点
    - 基于检索内容生成答案

    优势：
    - 有KG数据支持
    - 比Direct LLM更准确

    劣势：
    - 检索可能不精准
    - 无多跳推理
    - 无闭环能力
    """

    def __init__(self, neo4j_exec, openai_client, model="gpt-4o"):
        super().__init__("RAG")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = model

    def answer(self, question: str, timeout: int = 120) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: 提取关键词
        keywords = self._extract_keywords(question)
        logger.info(f"  RAG keywords: {keywords}")

        # Step 2: 检索相关文档
        docs = self._retrieve_documents(keywords, top_k=10)
        logger.info(f"  RAG retrieved {len(docs)} documents")

        # Step 3: 构建prompt
        if docs:
            context = self._format_documents(docs)
        else:
            context = "No relevant documents found in the knowledge graph."

        # Step 4: 生成答案
        try:
            answer = self._generate_answer(question, context, timeout)
            execution_time = time.time() - start_time

            # 从docs提取实体
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
                'confidence_score': 0.6,  # 有KG支持，置信度稍高
                'success': True,
                'method': 'RAG',
            }

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词（简单方法）"""
        import re

        # 提取大写缩写
        keywords = re.findall(r'\b[A-Z]{2,5}\b', question)

        # 提取基因名模式
        keywords.extend(re.findall(r'\b[A-Z][a-z]{2,8}\+?\b', question))

        # 提取常见神经科学术语
        neuro_terms = [
            'neuron', 'neurons', 'cell', 'cells', 'cortex', 'region',
            'brain', 'axon', 'dendrite', 'projection', 'marker', 'cluster'
        ]
        q_lower = question.lower()
        keywords.extend([term for term in neuro_terms if term in q_lower])

        return list(set(keywords))[:5]  # 最多5个关键词

    def _retrieve_documents(self, keywords: List[str], top_k: int = 10) -> List[Dict]:
        """检索相关文档"""
        docs = []

        for keyword in keywords:
            # 🔧 多种检索策略

            # 策略1: 匹配Region
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

            # 策略2: 匹配Cluster (基因marker)
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

            # 策略3: 匹配Subclass
            query_subclass = """
            MATCH (s:Subclass)
            WHERE s.name CONTAINS $keyword OR s.markers CONTAINS $keyword
            RETURN 'Subclass' AS type, s.name AS subclass_name, s.markers AS markers
            LIMIT 2
            """
            result = self.db.run(query_subclass, {'keyword': keyword})
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
        """格式化文档为prompt"""
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

            elif doc_type == 'Subclass':
                text = f"Subclass: {doc.get('subclass_name', 'N/A')}"
                if doc.get('markers'):
                    text += f"\n  Markers: {doc['markers']}"

            else:
                text = json.dumps(doc, indent=2)

            formatted.append(f"Document {i}:\n{text}")

        return "\n\n".join(formatted)

    def _generate_answer(self, question: str, context: str, timeout: int) -> str:
        """基于检索内容生成答案"""

        system_prompt = """You are a neuroscience expert analyzing data from a knowledge graph.
Use ONLY the provided documents to answer the question.
Be precise and cite specific data from the documents.
If the documents don't contain sufficient information, acknowledge it."""

        user_prompt = f"""Based on the following documents from a neuroscience knowledge graph, answer the question.

Documents:
{context}

Question: {question}

Provide a detailed, scientific answer using ONLY information from the documents above.
Include specific numbers, region names, and markers when available.

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1200,
            timeout=timeout
        )

        return response.choices[0].message.content

    def _extract_entities_from_docs(self, docs: List[Dict]) -> List[Dict]:
        """从检索文档中提取实体"""
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
                # 提取markers
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


# ==================== Baseline 3: ReAct ====================

class ReActBaseline(BaselineAgent):
    """
    Baseline 3: ReAct (Reasoning + Acting)

    特点：
    - 交替进行推理和行动
    - 可以执行多步查询
    - 有一定推理能力

    优势：
    - 可以进行简单的多跳推理
    - 有KG访问能力

    劣势：
    - 无schema引导
    - 无自适应深度
    - 无闭环机制
    - 推理步骤有限
    """

    def __init__(self, neo4j_exec, openai_client, model="gpt-4o", max_iterations=3):
        super().__init__("ReAct")
        self.db = neo4j_exec
        self.client = openai_client
        self.model = model
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

                # 构建context
                if history:
                    context = "\n\n".join(history)
                else:
                    context = "Start your reasoning."

                prompt = f"""Question: {question}

Previous steps:
{context}

What's your next step? Respond in JSON format."""

                # LLM推理
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=800,
                    timeout=timeout // self.max_iterations
                )

                result = json.loads(response.choices[0].message.content)

                thought = result.get('thought', '')
                action = result.get('action', '')

                history.append(f"Thought: {thought}")
                logger.info(f"    Thought: {thought[:80]}...")

                # 如果决定回答
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
                        'confidence_score': 0.7,  # ReAct有推理，置信度较高
                        'success': True,
                        'method': 'ReAct',
                    }

                # 如果决定查询
                elif action == 'query':
                    query = result.get('query', '')

                    if not query:
                        logger.warning(f"    Empty query, skipping")
                        continue

                    history.append(f"Action: Execute query")
                    logger.info(f"    Executing query: {query[:80]}...")

                    # 执行查询
                    db_result = self.db.run(query)

                    if db_result.get('success'):
                        data = db_result.get('data', [])[:20]  # 限制20行
                        observation = f"Query returned {len(data)} results"

                        # 提取实体
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
                    })

            # 达到最大迭代次数
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
                'confidence_score': 0.4,  # 未完成，低置信度
                'success': False,
                'method': 'ReAct',
            }

        except Exception as e:
            logger.error(f"ReAct failed: {e}")
            return self._error_response(question, str(e), time.time() - start_time)

    def _extract_entities_from_data(self, data: List[Dict]) -> List[Dict]:
        """从查询结果提取实体"""
        entities = []

        for row in data[:5]:  # 只处理前5行
            for key, value in row.items():
                if isinstance(value, str):
                    # Region acronym pattern
                    if len(value) >= 2 and len(value) <= 5 and value.isupper():
                        entities.append({
                            'text': value,
                            'type': 'Region',
                            'confidence': 0.8,
                        })
                    # Gene pattern
                    elif len(value) >= 3 and value[0].isupper():
                        entities.append({
                            'text': value,
                            'type': 'Gene',
                            'confidence': 0.6,
                        })

        # 去重
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

    if baseline_type == 'direct_llm':
        return DirectLLMBaseline(
            openai_client=kwargs['openai_client'],
            model=kwargs.get('model', 'gpt-4o')
        )

    elif baseline_type == 'rag':
        return RAGBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client'],
            model=kwargs.get('model', 'gpt-4o')
        )

    elif baseline_type == 'react':
        return ReActBaseline(
            neo4j_exec=kwargs['neo4j_exec'],
            openai_client=kwargs['openai_client'],
            model=kwargs.get('model', 'gpt-4o'),
            max_iterations=kwargs.get('max_iterations', 3)
        )

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# ==================== Test ====================

if __name__ == "__main__":
    # 简单测试
    print("Baseline implementations loaded successfully!")
    print("\nAvailable baselines:")
    print("1. Direct LLM - No KG access, pure LLM knowledge")
    print("2. RAG - Retrieval + Generation")
    print("3. ReAct - Reasoning + Acting with KG queries")