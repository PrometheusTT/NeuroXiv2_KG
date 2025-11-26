"""
LLM-Driven Intent Classifier
=============================
使用LLM进行真正的意图理解，而非关键词匹配

对齐Figure 2A: IntentClassifier

核心功能：
1. 问题意图分类
2. 分析深度确定
3. 规划器推荐
4. 实体类型预测

Author: Claude & Lijun
Date: 2025-01-15
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from core_structures import (
    QuestionIntent,
    AnalysisDepth,
    PlannerType,
    Modality
)

logger = logging.getLogger(__name__)


@dataclass
class IntentClassification:
    """意图分类结果"""
    intent: QuestionIntent
    confidence: float
    reasoning: str

    # 推荐
    recommended_depth: AnalysisDepth
    recommended_planner: PlannerType
    expected_modalities: List[Modality]

    # 预测的实体类型
    expected_entity_types: List[str]

    # 问题分解
    sub_questions: List[str]


class LLMIntentClassifier:
    """
    LLM驱动的意图分类器

    核心改进：
    1. 使用LLM推理而非关键词匹配
    2. 输出结构化的分类结果
    3. 提供决策reasoning
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

    def classify(self, question: str, context: str = "") -> IntentClassification:
        """
        分类问题意图

        Args:
            question: 用户问题
            context: 可选的会话上下文

        Returns:
            IntentClassification对象
        """
        logger.info(f"🧠 LLM Intent Classification: {question[:50]}...")

        prompt = self._build_classification_prompt(question, context)

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            classification = self._parse_result(result)

            logger.info(f"   Intent: {classification.intent.value}")
            logger.info(f"   Depth: {classification.recommended_depth.value}")
            logger.info(f"   Planner: {classification.recommended_planner.value}")
            logger.info(f"   Confidence: {classification.confidence:.2f}")

            return classification

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._fallback_classification(question)

    def _get_system_prompt(self) -> str:
        return """You are an expert neuroscience query analyzer for a brain knowledge graph system.

Your task is to deeply understand user questions and classify them to determine:
1. The underlying INTENT (what the user really wants to know)
2. The appropriate DEPTH of analysis needed
3. Which PLANNER should handle this query
4. What MODALITIES of data will be needed

You must reason carefully about the question's semantics, not just match keywords.

INTENT TYPES:
- simple_query: Direct factual questions ("What is the full name of CLA?")
- deep_profiling: Comprehensive analysis of an entity ("Tell me about Car3+ neurons")
- comparison: Comparing two or more entities ("Compare Pvalb and Sst interneurons")
- screening: Finding entities matching criteria ("Which regions show highest mismatch?")
- explanation: Understanding mechanisms ("Why do these regions connect?")

DEPTH LEVELS:
- shallow: 1-2 steps, quick factual answer
- medium: 3-4 steps, multi-modal but focused
- deep: 5-8 steps, comprehensive with closed-loop analysis

PLANNERS:
- focus_driven: For deep analysis of a primary entity (Figure 3 style)
- comparative: For systematic screening or pairwise comparison (Figure 4 style)
- adaptive: For simple queries or exploratory questions

MODALITIES:
- molecular: Gene expression, cell types, clusters
- morphological: Neuron structure, axon/dendrite features
- projection: Connectivity, projection targets
- spatial: Brain regions, coordinates"""

    def _build_classification_prompt(self, question: str, context: str) -> str:
        return f"""Analyze this neuroscience question and classify it.

**Question:** {question}

{f"**Previous Context:** {context}" if context else ""}

Think step by step:
1. What is the user really asking for?
2. What entities are mentioned or implied?
3. How deep should the analysis go?
4. What data modalities will be needed?
5. Can this question be broken into sub-questions?

Return JSON:
{{
    "intent": "simple_query|deep_profiling|comparison|screening|explanation",
    "intent_confidence": 0.0-1.0,
    "intent_reasoning": "Why you classified it this way",

    "recommended_depth": "shallow|medium|deep",
    "depth_reasoning": "Why this depth is appropriate",

    "recommended_planner": "focus_driven|comparative|adaptive",
    "planner_reasoning": "Why this planner is best suited",

    "expected_modalities": ["molecular", "morphological", "projection"],
    "modality_reasoning": "Why these modalities are needed",

    "expected_entity_types": ["GeneMarker", "Region", "Cluster", ...],

    "sub_questions": [
        "Sub-question 1 that needs to be answered",
        "Sub-question 2...",
        ...
    ]
}}

Be precise and provide clear reasoning for each decision."""

    def _parse_result(self, result: Dict) -> IntentClassification:
        """解析LLM返回的JSON"""

        # Intent
        intent_str = result.get('intent', 'unknown')
        intent_map = {
            'simple_query': QuestionIntent.SIMPLE_QUERY,
            'deep_profiling': QuestionIntent.DEEP_PROFILING,
            'comparison': QuestionIntent.COMPARISON,
            'screening': QuestionIntent.SCREENING,
            'explanation': QuestionIntent.EXPLANATION
        }
        intent = intent_map.get(intent_str, QuestionIntent.UNKNOWN)

        # Depth
        depth_str = result.get('recommended_depth', 'medium')
        depth_map = {
            'shallow': AnalysisDepth.SHALLOW,
            'medium': AnalysisDepth.MEDIUM,
            'deep': AnalysisDepth.DEEP
        }
        depth = depth_map.get(depth_str, AnalysisDepth.MEDIUM)

        # Planner
        planner_str = result.get('recommended_planner', 'adaptive')
        planner_map = {
            'focus_driven': PlannerType.FOCUS_DRIVEN,
            'comparative': PlannerType.COMPARATIVE,
            'adaptive': PlannerType.ADAPTIVE
        }
        planner = planner_map.get(planner_str, PlannerType.ADAPTIVE)

        # Modalities
        modality_strs = result.get('expected_modalities', [])
        modality_map = {
            'molecular': Modality.MOLECULAR,
            'morphological': Modality.MORPHOLOGICAL,
            'projection': Modality.PROJECTION,
            'spatial': Modality.SPATIAL
        }
        modalities = [modality_map[m] for m in modality_strs if m in modality_map]

        # 构建分类reasoning
        reasoning_parts = [
            f"Intent: {result.get('intent_reasoning', 'N/A')}",
            f"Depth: {result.get('depth_reasoning', 'N/A')}",
            f"Planner: {result.get('planner_reasoning', 'N/A')}"
        ]

        return IntentClassification(
            intent=intent,
            confidence=result.get('intent_confidence', 0.5),
            reasoning="\n".join(reasoning_parts),
            recommended_depth=depth,
            recommended_planner=planner,
            expected_modalities=modalities,
            expected_entity_types=result.get('expected_entity_types', []),
            sub_questions=result.get('sub_questions', [])
        )

    def _fallback_classification(self, question: str) -> IntentClassification:
        """Fallback：当LLM调用失败时使用规则"""
        logger.warning("Using fallback rule-based classification")

        q_lower = question.lower()

        # 简单规则fallback
        if any(w in q_lower for w in ['compare', 'versus', 'vs', 'difference']):
            intent = QuestionIntent.COMPARISON
            planner = PlannerType.COMPARATIVE
            depth = AnalysisDepth.MEDIUM
        elif any(w in q_lower for w in ['which', 'find', 'screen', 'highest', 'top']):
            intent = QuestionIntent.SCREENING
            planner = PlannerType.COMPARATIVE
            depth = AnalysisDepth.MEDIUM
        elif any(w in q_lower for w in ['tell me about', 'comprehensive', 'analyze']):
            intent = QuestionIntent.DEEP_PROFILING
            planner = PlannerType.FOCUS_DRIVEN
            depth = AnalysisDepth.DEEP
        elif any(w in q_lower for w in ['why', 'how', 'explain']):
            intent = QuestionIntent.EXPLANATION
            planner = PlannerType.ADAPTIVE
            depth = AnalysisDepth.MEDIUM
        else:
            intent = QuestionIntent.SIMPLE_QUERY
            planner = PlannerType.ADAPTIVE
            depth = AnalysisDepth.SHALLOW

        return IntentClassification(
            intent=intent,
            confidence=0.6,  # Lower confidence for fallback
            reasoning="Fallback rule-based classification (LLM unavailable)",
            recommended_depth=depth,
            recommended_planner=planner,
            expected_modalities=[Modality.MOLECULAR],
            expected_entity_types=['GeneMarker', 'Region'],
            sub_questions=[]
        )


class PlannerRouter:
    """
    规划器路由器 - 对齐Figure 2A Planner Router

    基于IntentClassification选择并配置规划器
    """

    def __init__(self, focus_planner, comparative_planner, adaptive_planner):
        self.planners = {
            PlannerType.FOCUS_DRIVEN: focus_planner,
            PlannerType.COMPARATIVE: comparative_planner,
            PlannerType.ADAPTIVE: adaptive_planner
        }

    def route(self,
              classification: IntentClassification,
              state: 'AnalysisState') -> Tuple[any, Dict]:
        """
        路由到合适的规划器

        Returns:
            (planner_instance, planner_config)
        """
        planner_type = classification.recommended_planner
        planner = self.planners.get(planner_type)

        if planner is None:
            logger.warning(f"Planner {planner_type} not found, using adaptive")
            planner = self.planners[PlannerType.ADAPTIVE]
            planner_type = PlannerType.ADAPTIVE

        # 配置规划器
        config = self._build_planner_config(classification, state)

        logger.info(f"🎯 Routed to {planner_type.value} planner")

        return planner, config

    def _build_planner_config(self,
                              classification: IntentClassification,
                              state: 'AnalysisState') -> Dict:
        """构建规划器配置"""

        # 基于分类结果配置规划器行为
        config = {
            'target_depth': classification.recommended_depth,
            'expected_modalities': classification.expected_modalities,
            'sub_questions': classification.sub_questions,
            'intent': classification.intent
        }

        # 根据意图调整配置
        if classification.intent == QuestionIntent.SCREENING:
            config['max_regions'] = 30
            config['require_statistics'] = True
            config['require_fdr'] = True

        elif classification.intent == QuestionIntent.DEEP_PROFILING:
            config['require_closed_loop'] = True
            config['deep_characterization'] = True

        elif classification.intent == QuestionIntent.COMPARISON:
            config['pairwise_analysis'] = True
            config['require_effect_size'] = True

        return config