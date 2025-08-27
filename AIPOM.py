"""
CoT-KG Agent: 修正版 - 基于实际知识图谱结构
正确生成Cypher查询

Author: NeuroXiv Team
Date: 2025-12-19
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 核心数据结构 ====================

@dataclass
class Thought:
    """单个思考步骤"""
    step: int
    question: str
    reasoning: str
    kg_query: str
    kg_result: Any
    insight: str
    next_question: Optional[str] = None
    confidence: float = 0.0

@dataclass
class ReasoningChain:
    """完整的推理链"""
    initial_question: str
    thoughts: List[Thought]
    final_answer: str
    discoveries: List[str]
    confidence: float

# ==================== 原子知识图谱接口 ====================

class KGInterface:
    """
    最小化但完备的知识图谱接口
    只提供原子操作，让LLM决定如何组合使用
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_cypher(self, query: str, **params) -> List[Dict]:
        """
        执行任意Cypher查询 - 这是唯一需要的接口
        LLM会生成适当的查询
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Cypher查询执行失败: {e}")
                logger.error(f"查询: {query}")
                return []

    def get_schema(self) -> Dict:
        """获取图谱schema供LLM参考 - 基于实际的数据结构"""
        schema = {
            'nodes': {
                'Region': {
                    'properties': [
                        'region_id (INT - unique identifier)',
                        'name (STRING)',
                        'acronym (STRING)',
                        'axonal_length (FLOAT - μm)',
                        'dendritic_length (FLOAT - μm)',
                        'axonal_branches (FLOAT)',
                        'dendritic_branches (FLOAT)',
                        'axonal_bifurcation_remote_angle (FLOAT)',
                        'dendritic_bifurcation_remote_angle (FLOAT)',
                        'axonal_maximum_branch_order (FLOAT)',
                        'dendritic_maximum_branch_order (FLOAT)',
                        'number_of_neuron_morphologies (INT)',
                        'number_of_transcriptomic_neurons (INT)'
                    ],
                    'description': '337 aggregated brain regions with morphological averages'
                },
                'Cluster': {
                    'properties': [
                        'tran_id (INT - unique identifier)',
                        'name (STRING)',
                        'markers (STRING - comma separated gene list)',
                        'dominant_neurotransmitter_type (STRING - e.g., GABA, Glutamate)'
                    ],
                    'description': 'Cell type clusters'
                },
                'Subclass': {
                    'properties': ['tran_id', 'name', 'markers', 'dominant_neurotransmitter_type'],
                    'description': 'Cell subclasses'
                },
                'Supertype': {
                    'properties': ['tran_id', 'name', 'markers'],
                    'description': 'Cell supertypes'
                },
                'Class': {
                    'properties': ['tran_id', 'name'],
                    'description': 'Cell classes'
                }
            },
            'relationships': {
                'PROJECT_TO': {
                    'pattern': '(Region)-[PROJECT_TO]->(Region)',
                    'properties': ['weight (FLOAT)', 'neuron_count (INT)', 'source_acronym', 'target_acronym'],
                    'description': 'Projection between regions'
                },
                'HAS_CLUSTER': {
                    'pattern': '(Region)-[HAS_CLUSTER]->(Cluster)',
                    'properties': ['pct_cells (FLOAT - percentage)', 'rank (INT - 1 is highest)'],
                    'description': 'Cell type composition of region'
                },
                'HAS_SUBCLASS': {
                    'pattern': '(Region)-[HAS_SUBCLASS]->(Subclass)',
                    'properties': ['pct_cells', 'rank']
                },
                'HAS_SUPERTYPE': {
                    'pattern': '(Region)-[HAS_SUPERTYPE]->(Supertype)',
                    'properties': ['pct_cells', 'rank']
                },
                'HAS_CLASS': {
                    'pattern': '(Region)-[HAS_CLASS]->(Class)',
                    'properties': ['pct_cells', 'rank']
                },
                'BELONGS_TO': {
                    'pattern': 'Cluster->Supertype->Subclass->Class hierarchy',
                    'description': 'Cell type hierarchy'
                }
            },
            'important_notes': [
                'Region nodes contain AGGREGATED morphological data (means), not individual neurons',
                'No single neuron data - only statistical aggregates at region level',
                'Cell type percentages are at region level via HAS_* relationships',
                'Use region_id for matching Region nodes, tran_id for cell type nodes'
            ]
        }
        return schema

    def close(self):
        self.driver.close()

# ==================== Chain-of-Thought 推理引擎 ====================

class ChainOfThoughtEngine:
    """
    CoT推理引擎 - LLM主导的推理过程
    """

    def __init__(self, kg: KGInterface, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.kg = kg
        self.kg_schema = kg.get_schema()
        self.max_thinking_steps = 10

        # Cypher查询示例，帮助LLM生成正确的查询
        self.example_queries = """
Example Cypher queries for this knowledge graph:

1. Find regions with longest axons:
MATCH (r:Region)
WHERE r.axonal_length IS NOT NULL
RETURN r.acronym, r.axonal_length
ORDER BY r.axonal_length DESC
LIMIT 10

2. Get cell type composition of a region:
MATCH (r:Region {acronym: 'MOp'})-[h:HAS_CLUSTER]->(c:Cluster)
RETURN c.name, h.pct_cells, h.rank
ORDER BY h.rank

3. Find projection targets of a region:
MATCH (r:Region {acronym: 'MOp'})-[p:PROJECT_TO]->(target:Region)
RETURN target.acronym, p.weight, p.neuron_count
ORDER BY p.weight DESC

4. Compare morphology between regions:
MATCH (r1:Region {acronym: 'MOp'})
MATCH (r2:Region {acronym: 'SSp'})
RETURN r1.acronym, r1.axonal_length, r2.acronym, r2.axonal_length

5. Find regions dominated by inhibitory neurons:
MATCH (r:Region)-[h:HAS_CLUSTER]->(c:Cluster)
WHERE c.dominant_neurotransmitter_type = 'GABA' AND h.rank = 1
RETURN r.acronym, c.name, h.pct_cells

IMPORTANT: 
- Never use parameters like :param in queries
- Use WHERE clauses for filtering
- Region matching uses region_id or acronym
- Cell type matching uses tran_id or name
"""

    def think(self, question: str) -> ReasoningChain:
        """
        执行链式思考推理
        LLM逐步分解问题，查询知识图谱，综合得出答案
        """
        thoughts = []
        current_question = question
        context = {
            'initial_question': question,
            'kg_schema': self.kg_schema,
            'example_queries': self.example_queries,
            'previous_thoughts': []
        }

        logger.info(f"开始推理: {question}")

        for step in range(self.max_thinking_steps):
            # Step 1: LLM思考下一步
            thought = self._generate_thought(current_question, context)

            # Step 2: 如果需要查询KG，生成并执行查询
            if thought.kg_query:
                thought.kg_result = self._execute_kg_query(thought.kg_query)

            # Step 3: LLM解释查询结果，提取洞察
            thought.insight = self._extract_insight(thought, context)

            # Step 4: 决定是否继续或结束
            thought.next_question = self._generate_next_question(thought, context)

            thoughts.append(thought)
            context['previous_thoughts'].append(thought)

            # 如果没有新问题，结束推理
            if not thought.next_question:
                break

            current_question = thought.next_question

        # Step 5: 综合所有思考，生成最终答案
        final_answer = self._synthesize_answer(thoughts, question)

        # Step 6: 识别新发现
        discoveries = self._identify_discoveries(thoughts)

        return ReasoningChain(
            initial_question=question,
            thoughts=thoughts,
            final_answer=final_answer,
            discoveries=discoveries,
            confidence=self._calculate_confidence(thoughts)
        )

    def _generate_thought(self, question: str, context: Dict) -> Thought:
        """
        LLM生成下一个思考步骤
        """
        prompt = f"""
You are a neuroscience research assistant analyzing a brain region knowledge graph.

Current question: {question}
Initial question: {context['initial_question']}

The knowledge graph contains:
- 337 aggregated brain regions with morphological data (axon/dendrite lengths and branches)
- Cell type composition data (via HAS_SUBCLASS relationships)  
- Projection data between regions (via PROJECT_TO relationships)

IMPORTANT: All morphological data are AVERAGES at the region level, not individual neurons.

Schema:
{json.dumps(context['kg_schema'], indent=2)}

Example queries:
{context['example_queries']}

Previous thoughts:
{self._format_previous_thoughts(context.get('previous_thoughts', []))}

Please generate the next reasoning step:
1. What information is needed to answer the current question?
2. Your reasoning process
3. If you need to query the knowledge graph, generate a Cypher query
4. Explain your reasoning logic

CRITICAL: 
- Do NOT use :param syntax in queries
- Use WHERE clauses for filtering
- Match regions by region_id or acronym
- All queries should be complete and executable

Return JSON format:
{{
    "reasoning": "Your reasoning process",
    "kg_query": "Complete executable Cypher query (if needed), otherwise null",
    "query_purpose": "What this query will tell us"
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a neuroscience expert skilled in logical reasoning and Cypher query generation."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        # Clean up the query if present
        if result.get('kg_query'):
            # Remove any parameter declarations
            query = result['kg_query']
            # Remove :param lines
            lines = query.split('\n')
            cleaned_lines = [line for line in lines if not line.strip().startswith(':param')]
            result['kg_query'] = '\n'.join(cleaned_lines).strip()

        return Thought(
            step=len(context.get('previous_thoughts', [])) + 1,
            question=question,
            reasoning=result['reasoning'],
            kg_query=result.get('kg_query'),
            kg_result=None,
            insight=""
        )

    def _execute_kg_query(self, query: str) -> List[Dict]:
        """执行知识图谱查询"""
        try:
            logger.info(f"执行查询: {query[:100]}...")
            result = self.kg.execute_cypher(query)
            logger.info(f"返回 {len(result)} 条结果")
            return result
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return []

    def _extract_insight(self, thought: Thought, context: Dict) -> str:
        """
        从查询结果中提取洞察
        """
        if not thought.kg_result:
            return thought.reasoning

        prompt = f"""
Based on the following reasoning and data, extract key insights:

Reasoning: {thought.reasoning}
Query results: {json.dumps(thought.kg_result[:20], indent=2, default=str)}  # Limit data

Please extract:
1. Key findings from the data
2. Whether it supports or refutes the hypothesis
3. Unexpected discoveries
4. Statistical significance (if applicable)

Return a concise insight description.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analysis expert in neuroscience."},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content

    def _generate_next_question(self, thought: Thought, context: Dict) -> Optional[str]:
        """
        基于当前思考生成下一个问题
        """
        prompt = f"""
Based on the current reasoning chain, decide the next step:

Initial question: {context['initial_question']}
Current insight: {thought.insight}
Steps completed: {thought.step}

Previous findings:
{self._format_insights(context.get('previous_thoughts', []))}

Please decide:
1. Have we sufficiently answered the initial question?
2. Are there new leads to explore?
3. Do we need to verify a hypothesis?

If exploration should continue, return the next specific question.
If complete, return null.

Return JSON:
{{
    "continue": true/false,
    "next_question": "next question" or null,
    "reason": "reasoning for decision"
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a logical reasoning expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        if result.get('continue') and result.get('next_question'):
            return result['next_question']
        return None

    def _synthesize_answer(self, thoughts: List[Thought], original_question: str) -> str:
        """综合所有思考步骤，生成最终答案"""
        prompt = f"""
Please synthesize the following reasoning chain to answer the original question:

Original question: {original_question}

Reasoning process:
{self._format_thought_chain(thoughts)}

Please generate:
1. Direct answer to the original question
2. Key supporting evidence
3. Confidence assessment
4. Limitations or areas needing further verification

Provide an accurate, complete, and logical answer.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in neuroscience, skilled in comprehensive analysis."},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content

    def _identify_discoveries(self, thoughts: List[Thought]) -> List[str]:
        """识别推理过程中的新发现"""
        discoveries = []

        for thought in thoughts:
            if thought.insight:
                # 简单规则：包含特定关键词的洞察可能是发现
                keywords = ['unexpected', 'discovered', 'unusual', 'significant',
                           'correlation', 'different from', 'special', 'surprisingly']
                if any(keyword in thought.insight.lower() for keyword in keywords):
                    discoveries.append(thought.insight)

        return discoveries[:5]  # Return top 5 discoveries

    def _calculate_confidence(self, thoughts: List[Thought]) -> float:
        """计算推理链的置信度"""
        if not thoughts:
            return 0.0

        base_confidence = 0.5
        data_supported = sum(1 for t in thoughts if t.kg_result)
        confidence = base_confidence + (data_supported / len(thoughts)) * 0.3

        if len(thoughts) > 7:
            confidence *= 0.9

        return min(confidence, 0.95)

    def _format_previous_thoughts(self, thoughts: List[Thought]) -> str:
        """格式化之前的思考"""
        if not thoughts:
            return "None"

        formatted = []
        for t in thoughts[-3:]:
            formatted.append(f"Step {t.step}: {t.insight[:100] if t.insight else 'Processing...'}")

        return "\n".join(formatted)

    def _format_insights(self, thoughts: List[Thought]) -> str:
        """格式化洞察"""
        insights = [t.insight for t in thoughts if t.insight]
        return "\n".join(insights[-5:])

    def _format_thought_chain(self, thoughts: List[Thought]) -> str:
        """格式化完整思考链"""
        formatted = []
        for t in thoughts:
            formatted.append(f"""
Step {t.step}:
  Question: {t.question}
  Reasoning: {t.reasoning[:200]}
  Query: {t.kg_query[:100] if t.kg_query else 'None'}
  Results: {len(t.kg_result) if t.kg_result else 0} records
  Insight: {t.insight[:200]}
""")
        return "\n".join(formatted)

# ==================== 主Agent ====================

class CoTKGAgent:
    """
    主Agent - 整合CoT推理和知识图谱
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str):
        self.kg = KGInterface(neo4j_uri, neo4j_user, neo4j_password)
        self.cot_engine = ChainOfThoughtEngine(self.kg, openai_api_key)

    def answer(self, question: str) -> Dict:
        """
        回答问题 - 主接口
        """
        logger.info(f"收到问题: {question}")

        # 执行CoT推理
        reasoning_chain = self.cot_engine.think(question)

        # 构造返回结果
        result = {
            'question': question,
            'answer': reasoning_chain.final_answer,
            'reasoning_steps': [
                {
                    'step': t.step,
                    'question': t.question,
                    'reasoning': t.reasoning,
                    'query': t.kg_query,
                    'num_results': len(t.kg_result) if t.kg_result else 0,
                    'insight': t.insight
                }
                for t in reasoning_chain.thoughts
            ],
            'discoveries': reasoning_chain.discoveries,
            'confidence': reasoning_chain.confidence,
            'num_steps': len(reasoning_chain.thoughts)
        }

        return result

    def explore(self, topic: str = None) -> Dict:
        """自主探索模式 - 发现新知识"""
        if topic:
            question = f"Explore interesting patterns and associations related to {topic}"
        else:
            question = "Find unexpected patterns or associations in the brain region data"

        return self.answer(question)

    def validate_hypothesis(self, hypothesis: str) -> Dict:
        """验证假说"""
        question = f"Validate the following hypothesis: {hypothesis}"
        result = self.answer(question)

        # 添加假说验证的特殊处理
        result['hypothesis'] = hypothesis
        result['validation_result'] = self._assess_validation(result)

        return result

    def _assess_validation(self, result: Dict) -> str:
        """评估假说验证结果"""
        answer = result.get('answer', '').lower()

        if 'support' in answer or 'confirm' in answer or 'correct' in answer or 'true' in answer:
            return "Hypothesis Supported"
        elif 'refute' in answer or 'not support' in answer or 'incorrect' in answer or 'false' in answer:
            return "Hypothesis Refuted"
        elif 'partial' in answer or 'some' in answer or 'mixed' in answer:
            return "Partially Supported"
        else:
            return "Insufficient Evidence"

    def close(self):
        """关闭连接"""
        self.kg.close()

# ==================== 使用示例 ====================

def main():
    """使用示例"""

    # 初始化Agent
    agent = CoTKGAgent(
        neo4j_uri="bolt://10.133.56.119:7687",  # 根据实际情况修改
        neo4j_user="neo4j",
        neo4j_password="neuroxiv",  # 使用实际密码
        openai_api_key="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"
    )

    try:
        print("=" * 60)
        print("Example 1: molecule Analysis")
        # result = agent.answer("Which areas violate the typical excitation-inhibition balance?")
        result = agent.answer("Which regions have a mismatch between morphological complexity and functional importance?")

        print(f"\nQuestion: {result['question']}")
        print(f"\nReasoning steps ({result['num_steps']} steps):")
        for step in result['reasoning_steps']:
            print(f"  Step {step['step']}: Generated {step['num_results']} results")
            if step['insight']:
                print(f"    Insight: {step['insight'][:100]}...")

        print(f"\nFinal Answer:")
        print(result['answer'][:500])
        # # 示例1: 形态学问题
        # print("=" * 60)
        # print("Example 1: Morphology Analysis")
        # result = agent.answer("Which brain regions have the longest axons and what might this indicate about their function?")
        #
        # print(f"\nQuestion: {result['question']}")
        # print(f"\nReasoning steps ({result['num_steps']} steps):")
        # for step in result['reasoning_steps']:
        #     print(f"  Step {step['step']}: Generated {step['num_results']} results")
        #     if step['insight']:
        #         print(f"    Insight: {step['insight'][:100]}...")
        #
        # print(f"\nFinal Answer:")
        # print(result['answer'][:500])
        #
        # # 示例2: 细胞类型分析
        # print("\n" + "=" * 60)
        # print("Example 2: Cell Type Composition")
        # result2 = agent.answer("Which regions are dominated by inhibitory (GABAergic) neurons?")
        #
        # print(f"\nQuestion: {result2['question']}")
        # print(f"Number of reasoning steps: {result2['num_steps']}")
        # print(f"Confidence: {result2['confidence']:.2f}")
        #
        # # 示例3: 连接性分析
        # print("\n" + "=" * 60)
        # print("Example 3: Connectivity Analysis")
        # result3 = agent.answer("What are the main projection targets of the motor cortex (MOp)?")
        #
        # print(f"\nQuestion: {result3['question']}")
        # if result3['discoveries']:
        #     print(f"\nDiscoveries:")
        #     for discovery in result3['discoveries']:
        #         print(f"  - {discovery}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    finally:
        agent.close()
        print("\n分析完成")

if __name__ == "__main__":
    main()