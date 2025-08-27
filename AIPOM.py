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
        # Cypher查询示例，帮助LLM生成正确的查询
        self.example_queries = """
        Example Cypher queries for this knowledge graph:

        1. Find regions with longest axons (USING CONTAINS AND NULL CHECKING):
        MATCH (r:Region)
        WHERE r.axonal_length IS NOT NULL
        RETURN r.acronym, r.axonal_length
        ORDER BY r.axonal_length DESC
        LIMIT 10

        2. Get cell type composition of a region (USING OPTIONAL MATCH):
        MATCH (r:Region)
        WHERE r.acronym CONTAINS 'MOp' 
        OPTIONAL MATCH (r)-[h:HAS_SUBCLASS]->(c:Subclass)
        RETURN r.acronym, c.name, h.pct_cells, h.rank
        ORDER BY h.rank

        3. Find projection targets of a region (PROPER UNION EXAMPLE):
        MATCH (r:Region)-[p:PROJECT_TO]->(target:Region)
        WHERE r.acronym = 'MOp'
        RETURN target.acronym AS region, p.weight AS value, 'projection weight' AS metric
        ORDER BY p.weight DESC
        LIMIT 5
        UNION
        MATCH (r:Region)-[p:PROJECT_TO]->(target:Region)
        WHERE r.acronym = 'MOp'
        RETURN target.acronym AS region, p.neuron_count AS value, 'neuron count' AS metric
        ORDER BY p.neuron_count DESC
        LIMIT 5

        4. Compare morphology between regions (STARTING BROAD THEN FILTERING):
        MATCH (r:Region)
        WHERE r.acronym IN ['MOp', 'SSp']
        RETURN r.acronym, r.axonal_length, r.dendritic_length

        5. Find regions dominated by inhibitory neurons (CONTAINS FOR FUZZY MATCHING):
        MATCH (r:Region)-[h:HAS_SUBCLASS]->(c:Subclass)
        WHERE toLower(c.dominant_neurotransmitter_type) CONTAINS 'gaba' AND h.rank = 1
        RETURN r.acronym, c.name, h.pct_cells

        IMPORTANT TIPS: 
        - Use CONTAINS for string matching instead of exact equals
        - Always check if properties exist before filtering on them
        - Start with broader queries and then narrow down results
        """
        # 定义一致的指标体系
        self.metric_definitions = {
            "morphological_complexity": {
                "primary": "axonal_length + dendritic_length",
                "secondary": "axonal_branches + dendritic_branches",
                "tertiary": "max(axonal_maximum_branch_order, dendritic_maximum_branch_order)"
            },
            "functional_importance": {
                "primary": "sum(PROJECT_TO.weight)",
                "secondary": "count(distinct target:Region)",
                "tertiary": "percentage of excitatory neurons"
            }
        }

        # 添加Cypher最佳实践
        self.cypher_best_practices = """
        CYPHER BEST PRACTICES:
        1. String matching: Use CONTAINS instead of = (WHERE r.name CONTAINS 'CA3')
        2. Missing relationships: Use OPTIONAL MATCH for relations that might not exist
        3. UNION queries: Ensure both parts have EXACTLY the same column names
        4. Property existence: Check if properties exist (WHERE prop IS NOT NULL) before filtering
        5. Always use params in brackets: ['val1', 'val2'] not {param_name}
        6. Handle aggregation properly: Use WITH clauses for aggregated values
        7. Start broad then narrow: Begin with simple queries before adding complex filters
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
        metric_definitions = """
        CONSISTENT METRIC DEFINITIONS:
        1. Morphological Complexity:
           - Primary: Combined axonal_length + dendritic_length
           - Secondary: Combined axonal_branches + dendritic_branches
           - Tertiary: Maximum branch order (axonal/dendritic)

        2. Functional Importance:
           - Primary: Sum of outgoing projection weights (sum of PROJECT_TO weights)
           - Secondary: Number of target regions receiving projections
           - Tertiary: Dominance of excitatory vs inhibitory neuron types

        3. Mismatch Definition:
           - Regions with high morphological complexity but low functional importance
           - OR regions with low morphological complexity but high functional importance
        """

        prompt = f"""
        You are a neuroscience research assistant analyzing a brain region knowledge graph.

        Current question: {question}
        Initial question: {context['initial_question']}

        {metric_definitions}

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

        CYPHER QUERY GUIDELINES:
        1. Use CONTAINS for flexible string matching: WHERE r.name CONTAINS 'CA3' 
        2. Use OPTIONAL MATCH to handle missing relationships
        3. For UNION queries, ensure both sides have EXACTLY the same column names
        4. NEVER include multiple queries separated by semicolons
        5. Use CONSISTENT METRICS as defined above

        Please generate the next reasoning step:
        1. What information is needed to answer the current question?
        2. Your reasoning process
        3. If you need to query the knowledge graph, generate a Cypher query
        4. Explain your reasoning logic

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
                {"role": "system",
                 "content": "You are a neuroscience expert skilled in logical reasoning and Cypher query generation."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        # Clean up the query if present
        if result.get('kg_query'):
            query = result['kg_query']
            # Remove any parameter declarations
            lines = query.split('\n')
            cleaned_lines = [line for line in lines if not line.strip().startswith(':param')]
            # Remove any trailing semicolons
            query = '\n'.join(cleaned_lines).strip()
            if query.endswith(';'):
                query = query[:-1]
            result['kg_query'] = query

        return Thought(
            step=len(context.get('previous_thoughts', [])) + 1,
            question=question,
            reasoning=result['reasoning'],
            kg_query=result.get('kg_query'),
            kg_result=None,
            insight=""
        )

    def _clean_query(self, query: str) -> str:
        """清理并验证Cypher查询"""
        if not query:
            return None

        # 移除参数声明和尾部分号
        lines = query.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith(':param')]
        query = '\n'.join(cleaned_lines).strip()
        if query.endswith(';'):
            query = query[:-1]

        # 检查UNION语法，确保列名一致
        if ' UNION ' in query.upper():
            parts = query.upper().split(' UNION ')
            if len(parts) >= 2:
                # 提取第一部分的RETURN列
                import re
                return_pattern = r'RETURN\s+(.*?)(?:ORDER BY|LIMIT|$)'
                first_return = re.search(return_pattern, parts[0], re.IGNORECASE)

                if first_return:
                    columns = [c.strip().split(' AS ')[-1] for c in first_return.group(1).split(',')]

                    # 将修正建议添加到查询注释中
                    query = "// 注意: UNION查询需要相同的列名: " + ", ".join(columns) + "\n" + query

        return query

    def _clean_parameters(self, query: str) -> str:
        """处理查询中的参数问题"""
        if not query:
            return query

        # 查找 {parameter_name} 形式的参数并替换为适当的语法
        import re
        param_pattern = r'IN\s+{([^}]+)}'

        def replace_param(match):
            param_name = match.group(1)
            # 替换为固定值或列表语法
            return f"IN ['placeholder']  /* 替换 {{{param_name}}} 为实际值列表 */"

        fixed_query = re.sub(param_pattern, replace_param, query)
        return fixed_query
    def _execute_kg_query(self, query: str) -> List[Dict]:
        """执行知识图谱查询"""
        try:
            # 清理并验证查询
            query = self._clean_query(query)
            query = self._clean_parameters(query)
            if not query:
                return []

            logger.info(f"执行查询: {query[:100]}...")
            result = self.kg.execute_cypher(query)
            logger.info(f"返回 {len(result)} 条结果")

            # 如果查询失败并包含UNION，尝试分开执行
            if len(result) == 0 and ' UNION ' in query.upper():
                logger.info("UNION查询失败，尝试分开执行部分...")
                parts = query.split(' UNION ', 1)
                first_part = parts[0].strip()

                result = self.kg.execute_cypher(first_part)
                logger.info(f"第一部分查询返回 {len(result)} 条结果")

            return result
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return []

    def _extract_insight(self, thought: Thought, context: Dict) -> str:
        """从查询结果中提取洞察 - 增加空结果处理"""
        if not thought.kg_result:
            # 分析为什么结果为空，提供诊断信息
            empty_result_analysis = self._analyze_empty_result(thought.kg_query)

            return f"""
    No results returned from query. Possible reasons:
    {empty_result_analysis}

    Original reasoning: {thought.reasoning}

    Suggested next steps:
    1. Try broader matching with CONTAINS instead of exact equality
    2. Check if properties exist with IS NOT NULL clauses before filtering
    3. Use OPTIONAL MATCH for relationships that might not exist
    4. Start with simpler queries to explore data availability
    """

        # 原有的正常结果处理...
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

    def _analyze_empty_result(self, query: str) -> str:
        """分析空结果的可能原因"""
        reasons = []

        if not query:
            return "No query was provided."

        # 检查可能的原因
        if '=' in query and 'CONTAINS' not in query:
            reasons.append("- Using exact equality (=) instead of CONTAINS for string matching")

        if 'MATCH' in query and 'OPTIONAL MATCH' not in query:
            reasons.append("- Using MATCH instead of OPTIONAL MATCH for potentially missing relationships")

        if 'UNION' in query:
            reasons.append("- UNION query may have syntax issues with column names not matching")

        if 'WHERE' in query and ('IS NOT NULL' not in query):
            reasons.append("- Not checking if properties exist with IS NOT NULL")

        if len(reasons) == 0:
            reasons.append("- Data may simply not exist that matches these specific criteria")
            reasons.append("- Try exploring with broader, simpler queries first")

        return "\n".join(reasons)

    def _generate_next_question(self, thought: Thought, context: Dict) -> Optional[str]:
        """
        基于当前思考生成下一个问题
        """
        if thought.kg_query and not thought.kg_result:
            # 尝试生成更宽松的查询作为下一步
            return f"How can I modify my approach to find data related to {thought.question}? Let me try a broader query strategy."

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

    # Modify these methods in the ChainOfThoughtEngine class to prevent truncation

    def _format_previous_thoughts(self, thoughts: List[Thought]) -> str:
        """格式化之前的思考 - 不截断"""
        if not thoughts:
            return "None"

        formatted = []
        for t in thoughts[-3:]:  # 仍然保留最近3个，但不截断内容
            formatted.append(f"Step {t.step}: {t.insight if t.insight else 'Processing...'}")

        return "\n".join(formatted)

    def _format_insights(self, thoughts: List[Thought]) -> str:
        """格式化洞察 - 不截断"""
        insights = [t.insight for t in thoughts if t.insight]
        return "\n".join(insights[-5:])

    def _format_thought_chain(self, thoughts: List[Thought]) -> str:
        """格式化完整思考链 - 不截断"""
        formatted = []
        for t in thoughts:
            formatted.append(f"""
    Step {t.step}:
      Question: {t.question}
      Reasoning: {t.reasoning}
      Query: {t.kg_query if t.kg_query else 'None'}
      Results: {len(t.kg_result) if t.kg_result else 0} records
      Insight: {t.insight}
    """)
        return "\n".join(formatted)

    def _synthesize_answer(self, thoughts: List[Thought], original_question: str) -> str:
        """综合所有思考步骤，生成最终答案 - 确保完整性"""
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

    Provide a complete answer - do not truncate or cut off mid-sentence.
    If the answer is long, structure it with clear headings and bullet points.
    """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in neuroscience, skilled in comprehensive analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16000,  # 增加token上限确保完整回答
        )

        return response.choices[0].message.content
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
        result = agent.answer("Analyze the projection pattern of the brain region with the highest proportion of Car3 transcriptome subclass neurons")
        # result = agent.answer("Which regions have a mismatch between morphological complexity and functional importance?")

        print(f"\nQuestion: {result['question']}")
        print(f"\nReasoning steps ({result['num_steps']} steps):")
        for step in result['reasoning_steps']:
            print(f"  Step {step['step']}: Generated {step['num_results']} results")
            if step['insight']:
                print(f"    Insight: {step['insight']}...")

        print(f"\nFinal Answer:")
        print(result['answer'])


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