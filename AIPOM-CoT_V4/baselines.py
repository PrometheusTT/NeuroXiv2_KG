"""
Baseline Agents for Comparison
===============================
实现用于对比的Baseline方法

Baselines:
1. Direct LLM - 直接问LLM，无规划无反思
2. ReAct - 有观察反馈，但无结构化反思
3. Simple RAG - 简单检索增强

这些baseline有能力天花板限制，确保公平对比

Author: Lijun
Date: 2025-01
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from core_structures import (
    Modality, AnalysisDepth, AnalysisState,
    EvidenceBuffer, EvidenceRecord
)
from llm_intelligence import LLMClient

logger = logging.getLogger(__name__)


# ==================== Base Agent ====================

class BaselineAgent(ABC):
    """Baseline Agent基类"""

    def __init__(self,
                 db_executor,
                 llm_client: LLMClient,
                 name: str = "BaselineAgent"):
        self.db = db_executor
        self.llm = llm_client
        self.name = name

        # 能力天花板限制
        self.capability_limits = {
            'think': 1.0,
            'plan': 1.0,
            'reflect': 1.0,
            'act': 1.0,
        }

    @abstractmethod
    def answer(self, question: str, max_iterations: int = 5) -> Dict[str, Any]:
        """回答问题"""
        pass

    def _build_result(self,
                      question: str,
                      answer: str,
                      steps: List[Dict],
                      start_time: float,
                      metadata: Dict = None) -> Dict:
        """构建标准结果格式"""
        return {
            'question': question,
            'answer': answer,
            'method': self.name,
            'executed_steps': steps,
            'total_steps': len(steps),
            'execution_time': time.time() - start_time,
            'modalities_covered': list(set(s.get('modality', '') for s in steps if s.get('modality'))),
            'metadata': metadata or {},
            'capability_limits': self.capability_limits,
            'success': True
        }


# ==================== Direct LLM Baseline ====================

class DirectLLMAgent(BaselineAgent):
    """
    Direct LLM Baseline

    特点：
    - 直接问LLM，不查询知识图谱
    - 无规划、无反思
    - 完全依赖LLM预训练知识

    能力天花板：
    - Think: 0.30 (只有基础理解)
    - Plan: 0.10 (无规划能力)
    - Reflect: 0.05 (无反思能力)
    - Act: 0.20 (无执行能力)
    """

    def __init__(self, db_executor, llm_client: LLMClient):
        super().__init__(db_executor, llm_client, "Direct LLM")

        self.capability_limits = {
            'think': 0.30,
            'plan': 0.10,
            'reflect': 0.05,
            'act': 0.20,
        }

    def answer(self, question: str, max_iterations: int = 5) -> Dict[str, Any]:
        """直接问LLM"""
        logger.info(f"[Direct LLM] Answering: {question[:50]}...")
        start_time = time.time()

        system_prompt = """You are a neuroscience expert. Answer the question based on your knowledge.
Be specific and cite relevant facts when possible."""

        user_prompt = f"""Question: {question}

Please provide a comprehensive answer about this neuroscience topic."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            answer = self.llm.chat(messages, temperature=0.3, max_tokens=1500)

            return self._build_result(
                question=question,
                answer=answer,
                steps=[{'step': 1, 'action': 'direct_llm_query', 'modality': None}],
                start_time=start_time,
                metadata={'approach': 'direct_llm_only'}
            )

        except Exception as e:
            return {
                'question': question,
                'answer': f"Error: {e}",
                'method': self.name,
                'success': False,
                'error': str(e)
            }


# ==================== ReAct Baseline ====================

class ReActAgent(BaselineAgent):
    """
    ReAct Baseline (Reasoning + Acting)

    特点：
    - 有思考和行动交替
    - 有观察反馈
    - 但无结构化反思和重规划

    能力天花板：
    - Think: 0.60 (有基础推理)
    - Plan: 0.50 (有简单规划)
    - Reflect: 0.40 (有观察但无深度反思)
    - Act: 0.70 (能执行查询)
    """

    def __init__(self, db_executor, llm_client: LLMClient):
        super().__init__(db_executor, llm_client, "ReAct")

        self.capability_limits = {
            'think': 0.60,
            'plan': 0.50,
            'reflect': 0.40,
            'act': 0.70,
        }

    def answer(self, question: str, max_iterations: int = 5) -> Dict[str, Any]:
        """ReAct循环"""
        logger.info(f"[ReAct] Answering: {question[:50]}...")
        start_time = time.time()

        steps = []
        observations = []

        for i in range(max_iterations):
            # Think: 决定下一步行动
            thought, action, action_input = self._think(question, observations)

            if action == "finish":
                break

            # Act: 执行行动
            observation = self._act(action, action_input)
            observations.append({
                'thought': thought,
                'action': action,
                'observation': observation
            })

            steps.append({
                'step': i + 1,
                'thought': thought,
                'action': action,
                'observation_rows': len(observation) if isinstance(observation, list) else 0,
                'modality': self._infer_modality(action)
            })

        # 生成最终答案
        answer = self._synthesize(question, observations)

        return self._build_result(
            question=question,
            answer=answer,
            steps=steps,
            start_time=start_time,
            metadata={'observations': len(observations)}
        )

    def _think(self, question: str, observations: List[Dict]) -> tuple:
        """思考下一步"""
        obs_str = ""
        for i, obs in enumerate(observations[-3:], 1):
            obs_str += f"\nStep {i}: {obs['action']} -> {len(obs['observation']) if isinstance(obs['observation'], list) else 'N/A'} results"

        system_prompt = """You are a ReAct agent for neuroscience knowledge graph queries.

Available actions:
- search_gene(gene_name): Find clusters expressing a gene
- search_region(region_acronym): Get region information
- search_projection(region): Find projection targets
- search_morphology(region): Get morphological features
- finish: Complete the task

Think step by step, then choose ONE action."""

        user_prompt = f"""Question: {question}

Previous observations: {obs_str if obs_str else "None"}

What should I do next? Return JSON:
{{
    "thought": "Your reasoning",
    "action": "action_name",
    "action_input": "parameter"
}}"""

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
            return (
                result.get('thought', ''),
                result.get('action', 'finish'),
                result.get('action_input', '')
            )
        except:
            return ('Unable to decide', 'finish', '')

    def _act(self, action: str, action_input: str) -> Any:
        """执行行动"""
        queries = {
            'search_gene': """
                MATCH (c:Cluster)
                WHERE c.markers CONTAINS $input
                RETURN c.name AS cluster, c.markers AS markers, 
                       c.number_of_neurons AS neurons
                ORDER BY c.number_of_neurons DESC
                LIMIT 10
            """,
            'search_region': """
                MATCH (r:Region {acronym: $input})
                OPTIONAL MATCH (r)-[:HAS_CLUSTER]->(c:Cluster)
                RETURN r.name AS region_name, r.acronym AS acronym,
                       count(c) AS cluster_count
            """,
            'search_projection': """
                MATCH (r:Region {acronym: $input})-[p:PROJECT_TO]->(t:Region)
                RETURN t.acronym AS target, t.name AS target_name,
                       p.weight AS weight
                ORDER BY p.weight DESC
                LIMIT 10
            """,
            'search_morphology': """
                MATCH (n:Neuron)-[:LOCATE_AT]->(r:Region {acronym: $input})
                RETURN count(n) AS neuron_count,
                       avg(n.axonal_length) AS avg_axon,
                       avg(n.dendritic_length) AS avg_dendrite
            """
        }

        query = queries.get(action)
        if not query:
            return []

        result = self.db.run(query, {'input': action_input})
        return result.get('data', [])

    def _infer_modality(self, action: str) -> Optional[str]:
        """推断模态"""
        modality_map = {
            'search_gene': 'molecular',
            'search_region': 'spatial',
            'search_projection': 'projection',
            'search_morphology': 'morphological'
        }
        return modality_map.get(action)

    def _synthesize(self, question: str, observations: List[Dict]) -> str:
        """综合答案"""
        obs_summary = []
        for obs in observations:
            data = obs['observation']
            if isinstance(data, list) and data:
                obs_summary.append(f"- {obs['action']}: Found {len(data)} results")
                if data:
                    obs_summary.append(f"  Sample: {json.dumps(data[0], default=str)[:200]}")

        system_prompt = "You are synthesizing findings from a knowledge graph search."

        user_prompt = f"""Question: {question}

Findings:
{chr(10).join(obs_summary)}

Write a concise answer based on these findings."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self.llm.chat(messages, temperature=0.3, max_tokens=1000)
        except:
            return "Unable to synthesize answer."


# ==================== Simple RAG Baseline ====================

class SimpleRAGAgent(BaselineAgent):
    """
    Simple RAG Baseline

    特点：
    - 先检索相关数据
    - 然后一次性回答
    - 无迭代、无反思

    能力天花板：
    - Think: 0.40 (有检索意图)
    - Plan: 0.30 (简单检索规划)
    - Reflect: 0.20 (无反思)
    - Act: 0.60 (能检索)
    """

    def __init__(self, db_executor, llm_client: LLMClient):
        super().__init__(db_executor, llm_client, "Simple RAG")

        self.capability_limits = {
            'think': 0.40,
            'plan': 0.30,
            'reflect': 0.20,
            'act': 0.60,
        }

    def answer(self, question: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Simple RAG: 检索然后回答"""
        logger.info(f"[Simple RAG] Answering: {question[:50]}...")
        start_time = time.time()

        steps = []
        retrieved_data = {}

        # Step 1: 提取关键词
        keywords = self._extract_keywords(question)
        steps.append({'step': 1, 'action': 'extract_keywords', 'keywords': keywords})

        # Step 2: 检索数据
        for keyword in keywords[:3]:
            data = self._retrieve(keyword)
            if data:
                retrieved_data[keyword] = data
                steps.append({
                    'step': len(steps) + 1,
                    'action': f'retrieve_{keyword}',
                    'rows': len(data),
                    'modality': 'molecular'
                })

        # Step 3: 生成答案
        answer = self._generate_answer(question, retrieved_data)

        return self._build_result(
            question=question,
            answer=answer,
            steps=steps,
            start_time=start_time,
            metadata={'keywords': keywords, 'retrieved_count': len(retrieved_data)}
        )

    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词"""
        import re

        keywords = []

        # 基因名模式
        gene_pattern = r'\b([A-Z][a-z]{2,8}\d*)\+?\b'
        for match in re.finditer(gene_pattern, question):
            keywords.append(match.group(1))

        # 脑区模式
        region_pattern = r'\b([A-Z]{2,5})\b'
        for match in re.finditer(region_pattern, question):
            text = match.group(1)
            if text.lower() not in {'what', 'which', 'where', 'about'}:
                keywords.append(text)

        return list(set(keywords))[:5]

    def _retrieve(self, keyword: str) -> List[Dict]:
        """检索数据"""
        # 尝试作为基因检索
        query = """
        MATCH (c:Cluster)
        WHERE c.markers CONTAINS $keyword
        RETURN c.name AS cluster, c.markers AS markers,
               c.number_of_neurons AS neurons
        ORDER BY c.number_of_neurons DESC
        LIMIT 10
        """
        result = self.db.run(query, {'keyword': keyword})

        if result.get('data'):
            return result['data']

        # 尝试作为区域检索
        query = """
        MATCH (r:Region)
        WHERE r.acronym = $keyword OR r.name CONTAINS $keyword
        RETURN r.acronym AS region, r.name AS name
        LIMIT 5
        """
        result = self.db.run(query, {'keyword': keyword})
        return result.get('data', [])

    def _generate_answer(self, question: str, retrieved_data: Dict) -> str:
        """生成答案"""
        context = []
        for keyword, data in retrieved_data.items():
            context.append(f"Data for '{keyword}':")
            for item in data[:5]:
                context.append(f"  - {json.dumps(item, default=str)[:200]}")

        system_prompt = "Answer the question using the retrieved data."

        user_prompt = f"""Question: {question}

Retrieved Data:
{chr(10).join(context) if context else "No relevant data found."}

Provide a clear answer."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self.llm.chat(messages, temperature=0.3, max_tokens=1000)
        except:
            return "Unable to generate answer."


# ==================== Factory ====================

def create_baseline_agents(db_executor, llm_client: LLMClient) -> Dict[str, BaselineAgent]:
    """创建所有baseline agents"""
    return {
        'Direct LLM': DirectLLMAgent(db_executor, llm_client),
        'ReAct': ReActAgent(db_executor, llm_client),
        'Simple RAG': SimpleRAGAgent(db_executor, llm_client),
    }


# ==================== Export ====================

__all__ = [
    'BaselineAgent',
    'DirectLLMAgent',
    'ReActAgent',
    'SimpleRAGAgent',
    'create_baseline_agents',
]