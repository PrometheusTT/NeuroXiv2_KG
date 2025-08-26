"""
CoT-KG Agent: Chain-of-Thought驱动的知识图谱推理系统
核心理念：LLM负责推理，KG提供事实，两者协同发现新知识

Author: NeuroXiv Team
Date: 2025-08-26
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
import openai
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
            result = session.run(query, **params)
            return [record.data() for record in result]

    def get_schema(self) -> Dict:
        """获取图谱schema供LLM参考"""
        schema = {
            'nodes': {
                'Region': {
                    'properties': ['region_id', 'name', 'acronym', 'axonal_length',
                                   'dendritic_length', 'axonal_branches', 'dendritic_branches',
                                   'number_of_neuron_morphologies', 'number_of_transcriptomic_neurons'],
                    'description': '337个聚合的脑区'
                },
                'Cluster': {
                    'properties': ['tran_id', 'name', 'markers', 'dominant_neurotransmitter_type'],
                    'description': '细胞类型簇'
                },
                'Subclass': {
                    'properties': ['tran_id', 'name', 'markers', 'dominant_neurotransmitter_type'],
                    'description': '细胞亚类'
                },
                'Supertype': {
                    'properties': ['tran_id', 'name', 'markers'],
                    'description': '细胞超类型'
                },
                'Class': {
                    'properties': ['tran_id', 'name'],
                    'description': '细胞大类'
                }
            },
            'relationships': {
                'PROJECT_TO': '(Region)-[PROJECT_TO {weight, neuron_count}]->(Region)',
                'HAS_CLUSTER': '(Region)-[HAS_CLUSTER {pct_cells, rank}]->(Cluster)',
                'HAS_SUBCLASS': '(Region)-[HAS_SUBCLASS {pct_cells, rank}]->(Subclass)',
                'HAS_SUPERTYPE': '(Region)-[HAS_SUPERTYPE {pct_cells, rank}]->(Supertype)',
                'HAS_CLASS': '(Region)-[HAS_CLASS {pct_cells, rank}]->(Class)',
                'BELONGS_TO': 'Cluster->Supertype->Subclass->Class层级关系'
            }
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
        self.kg = kg
        openai.api_key = openai_api_key
        self.kg_schema = kg.get_schema()
        self.max_thinking_steps = 10

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
你是一个神经科学研究助手，正在分析脑区知识图谱。

当前问题: {question}
初始问题: {context['initial_question']}

知识图谱包含337个聚合脑区，具有形态学数据(轴突/树突长度和分支)、转录组数据(细胞类型组成)和连接数据(投射关系)。

之前的思考:
{self._format_previous_thoughts(context.get('previous_thoughts', []))}

请进行下一步推理:
1. 分析当前问题需要什么信息
2. 生成推理过程
3. 如果需要查询知识图谱，生成Cypher查询
4. 解释你的推理逻辑

返回JSON格式:
{{
    "reasoning": "你的推理过程",
    "kg_query": "Cypher查询(如果需要)，否则为null",
    "query_purpose": "查询目的"
}}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是神经科学专家，擅长逻辑推理和数据分析"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)

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
基于以下推理和数据，提取关键洞察：

推理: {thought.reasoning}
查询结果: {json.dumps(thought.kg_result[:10], indent=2, default=str)}  # 限制数据量

请提取：
1. 数据显示的关键发现
2. 是否支持或反驳了假设
3. 意外的发现
4. 统计上的显著性（如果适用）

返回简洁的洞察描述。
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是数据分析专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content

    def _generate_next_question(self, thought: Thought, context: Dict) -> Optional[str]:
        """
        基于当前思考生成下一个问题
        """
        prompt = f"""
基于当前的推理链，决定下一步：

初始问题: {context['initial_question']}
当前洞察: {thought.insight}
已完成步骤数: {thought.step}

之前的发现:
{self._format_insights(context.get('previous_thoughts', []))}

请决定:
1. 是否已经充分回答了初始问题？
2. 是否发现了需要深入探索的新线索？
3. 是否需要验证某个假设？

如果需要继续探索，返回下一个具体问题。
如果已经完成，返回null。

返回JSON:
{{
    "continue": true/false,
    "next_question": "下一个问题" 或 null,
    "reason": "决定的理由"
}}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是逻辑推理专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)

        if result['continue'] and result.get('next_question'):
            return result['next_question']
        return None

    def _synthesize_answer(self, thoughts: List[Thought], original_question: str) -> str:
        """
        综合所有思考步骤，生成最终答案
        """
        prompt = f"""
请综合以下推理链，回答原始问题：

原始问题: {original_question}

推理过程:
{self._format_thought_chain(thoughts)}

请生成：
1. 直接回答原始问题
2. 支持答案的关键证据
3. 推理的可信度评估
4. 潜在的局限性或需要进一步验证的方面

要求答案准确、完整、有逻辑性。
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是神经科学专家，擅长综合分析"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content

    def _identify_discoveries(self, thoughts: List[Thought]) -> List[str]:
        """
        识别推理过程中的新发现
        """
        discoveries = []

        for thought in thoughts:
            if thought.insight:
                # 简单规则：包含特定关键词的洞察可能是发现
                keywords = ['意外', '发现', '异常', '显著', '相关', '不同于', '特殊']
                if any(keyword in thought.insight for keyword in keywords):
                    discoveries.append(thought.insight)

        # 使用LLM进一步提炼
        if discoveries:
            prompt = f"""
从以下洞察中，识别真正有价值的科学发现：

{json.dumps(discoveries, ensure_ascii=False, indent=2)}

请返回最重要的3-5个发现，每个用一句话概括。
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是科学发现评估专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            refined = response.choices[0].message.content.split('\n')
            discoveries = [d.strip() for d in refined if d.strip()]

        return discoveries

    def _calculate_confidence(self, thoughts: List[Thought]) -> float:
        """计算推理链的置信度"""
        if not thoughts:
            return 0.0

        # 简单策略：基于步骤数和数据支持
        base_confidence = 0.5

        # 有数据支持的步骤增加置信度
        data_supported = sum(1 for t in thoughts if t.kg_result)
        confidence = base_confidence + (data_supported / len(thoughts)) * 0.3

        # 步骤过多降低置信度
        if len(thoughts) > 7:
            confidence *= 0.9

        return min(confidence, 0.95)

    def _format_previous_thoughts(self, thoughts: List[Thought]) -> str:
        """格式化之前的思考"""
        if not thoughts:
            return "无"

        formatted = []
        for t in thoughts[-3:]:  # 只显示最近3个
            formatted.append(f"Step {t.step}: {t.insight[:100]}")

        return "\n".join(formatted)

    def _format_insights(self, thoughts: List[Thought]) -> str:
        """格式化洞察"""
        insights = [t.insight for t in thoughts if t.insight]
        return "\n".join(insights[-5:])  # 最近5个洞察

    def _format_thought_chain(self, thoughts: List[Thought]) -> str:
        """格式化完整思考链"""
        formatted = []
        for t in thoughts:
            formatted.append(f"""
Step {t.step}:
  问题: {t.question}
  推理: {t.reasoning[:200]}
  洞察: {t.insight[:200]}
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
        """
        自主探索模式 - 发现新知识
        """
        if topic:
            question = f"探索{topic}相关的有趣模式和关联"
        else:
            question = "在脑区数据中寻找意外的模式或关联"

        return self.answer(question)

    def validate_hypothesis(self, hypothesis: str) -> Dict:
        """
        验证假说
        """
        question = f"验证以下假说：{hypothesis}"
        result = self.answer(question)

        # 添加假说验证的特殊处理
        result['hypothesis'] = hypothesis
        result['validation_result'] = self._assess_validation(result)

        return result

    def _assess_validation(self, result: Dict) -> str:
        """评估假说验证结果"""
        # 简单规则评估
        answer = result.get('answer', '').lower()

        if '支持' in answer or '确认' in answer or '正确' in answer:
            return "支持假说"
        elif '反驳' in answer or '不支持' in answer or '错误' in answer:
            return "反驳假说"
        elif '部分' in answer or '一定程度' in answer:
            return "部分支持"
        else:
            return "证据不足"

    def close(self):
        """关闭连接"""
        self.kg.close()


# ==================== 使用示例 ====================

def main():
    """使用示例"""

    # 初始化Agent
    agent = CoTKGAgent(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="your-api-key"
    )

    try:
        # 示例1: 开放性问题
        print("=" * 60)
        print("示例1: 开放性问题")
        result = agent.answer("为什么某些脑区的轴突特别长？这与它们的功能有什么关系？")

        print(f"\n问题: {result['question']}")
        print(f"\n推理步骤 ({result['num_steps']}步):")
        for step in result['reasoning_steps']:
            print(f"  Step {step['step']}: {step['insight'][:100]}...")

        print(f"\n最终答案:")
        print(result['answer'][:500])

        if result['discoveries']:
            print(f"\n新发现:")
            for discovery in result['discoveries']:
                print(f"  - {discovery}")

        print(f"\n置信度: {result['confidence']:.2f}")

        # 示例2: 探索模式
        print("\n" + "=" * 60)
        print("示例2: 自主探索")
        explore_result = agent.explore("形态学和细胞类型的关系")

        print(f"探索主题: 形态学和细胞类型的关系")
        print(f"发现了 {len(explore_result['discoveries'])} 个有趣的模式")

        # 示例3: 验证假说
        print("\n" + "=" * 60)
        print("示例3: 假说验证")
        hypothesis = "长轴突的脑区倾向于有更多的谷氨酸能神经元"
        validation = agent.validate_hypothesis(hypothesis)

        print(f"假说: {hypothesis}")
        print(f"验证结果: {validation['validation_result']}")
        print(f"关键证据: {validation['answer'][:300]}...")

    finally:
        agent.close()
        print("\n分析完成")


if __name__ == "__main__":
    main()