"""
LLM-Driven Intelligence Module
==============================
真正利用LLM能力的智能理解模块

核心设计：
1. 意图理解 - LLM深度分析问题语义
2. 实体识别 - LLM+KG验证的混合识别
3. 策略推荐 - LLM选择最优分析路径
4. 自由推理 - 不受硬编码模式限制

Author: Lijun
Date: 2025-01
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core_structures import (
    Entity, EntityCluster, QuestionIntent, AnalysisDepth,
    PlannerType, Modality
)

logger = logging.getLogger(__name__)


# ==================== LLM Client Interface ====================

class LLMClient(ABC):
    """LLM客户端抽象接口"""

    @abstractmethod
    def chat(self,
             messages: List[Dict],
             temperature: float = 0.2,
             max_tokens: int = 2000,
             json_mode: bool = False) -> str:
        """发送聊天请求"""
        pass

    def generate_json(self,
                      system_prompt: str,
                      user_prompt: str,
                      temperature: float = 0.1) -> Dict:
        """生成JSON响应"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.chat(messages, temperature=temperature, json_mode=True)

        # 清理可能的markdown
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


class OpenAIClient(LLMClient):
    """OpenAI API客户端"""

    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def chat(self,
             messages: List[Dict],
             temperature: float = 0.2,
             max_tokens: int = 2000,
             json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# ==================== Intent Classification ====================

@dataclass
class IntentClassification:
    """意图分类结果"""
    intent: QuestionIntent
    confidence: float
    reasoning: str

    recommended_depth: AnalysisDepth
    recommended_planner: PlannerType
    expected_modalities: List[Modality]

    expected_entity_types: List[str]
    sub_questions: List[str]

    key_concepts: List[str]
    analysis_goals: List[str]


class LLMIntentClassifier:
    """
    LLM驱动的意图分类器

    核心能力：
    1. 深度语义理解
    2. 问题分解
    3. 策略推荐
    4. 多模态需求识别
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(self, question: str, context: str = "") -> IntentClassification:
        """分类问题意图"""
        logger.info(f"🧠 Classifying intent: {question[:60]}...")

        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(question, context)

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
            classification = self._parse_result(result)

            logger.info(f"   Intent: {classification.intent.value}")
            logger.info(f"   Depth: {classification.recommended_depth.value}")
            logger.info(f"   Planner: {classification.recommended_planner.value}")

            return classification

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._fallback_classification(question)

    def _get_system_prompt(self) -> str:
        return """You are an expert neuroscience query analyzer for a brain knowledge graph (NeuroXiv-KG).

The knowledge graph contains:
- **Regions**: Brain regions with acronyms (MOp, VISp, SSp, etc.) and full names
- **Clusters**: Cell type clusters with markers and neuron counts
- **Subclass**: Higher-level cell type taxonomy (Pvalb, Sst, Vip, Lamp5, etc.)
- **Neurons**: Individual neuron morphologies with axon/dendrite measurements
- **Projections**: Region-to-region connectivity with weights

Your task is to deeply understand user questions and determine:
1. The semantic INTENT (what they really want to know)
2. The appropriate analysis DEPTH
3. Which PLANNER strategy is optimal
4. What data MODALITIES are needed
5. How to decompose into sub-questions

INTENT TYPES:
- definition: "What is X?" / "What does X stand for?" / "Define X"
- profiling: "Tell me about X" / "Comprehensive analysis of X"
- comparison: "Compare A and B" / "Difference between A and B"
- screening: "Which regions..." / "Find all..." / "Top N..."
- connectivity: "Where does X project?" / "Inputs to X"
- composition: "What cell types in X?" / "Clusters expressing X"
- quantification: "How many..." / "Count..."
- mechanism: "Why..." / "How does..."

DEPTH:
- shallow: Quick factual answer (1-2 steps)
- medium: Multi-modal analysis (3-4 steps)
- deep: Comprehensive closed-loop analysis (5-8 steps)

PLANNERS:
- focus_driven: Deep analysis of single entity with closed-loop
- comparative: Systematic comparison or screening
- adaptive: Exploratory or simple queries

Think carefully about the user's real information need, not just keywords."""

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Analyze this neuroscience question:

**Question:** {question}

{f"**Context:** {context}" if context else ""}

Return JSON with your analysis:
{{
    "intent": "definition|profiling|comparison|screening|connectivity|composition|quantification|mechanism",
    "intent_confidence": 0.0-1.0,
    "intent_reasoning": "Why you classified it this way",

    "recommended_depth": "shallow|medium|deep",
    "depth_reasoning": "Why this depth",

    "recommended_planner": "focus_driven|comparative|adaptive",
    "planner_reasoning": "Why this planner",

    "expected_modalities": ["molecular", "morphological", "projection"],
    "modality_reasoning": "Why these modalities",

    "expected_entity_types": ["GeneMarker", "Region", "CellType"],

    "key_concepts": ["concept1", "concept2"],

    "sub_questions": [
        "Sub-question 1",
        "Sub-question 2"
    ],

    "analysis_goals": [
        "Goal 1: What we need to find",
        "Goal 2: What analysis to perform"
    ]
}}"""

    def _parse_result(self, result: Dict) -> IntentClassification:
        # Intent mapping
        intent_map = {
            'definition': QuestionIntent.DEFINITION,
            'profiling': QuestionIntent.PROFILING,
            'comparison': QuestionIntent.COMPARISON,
            'screening': QuestionIntent.SCREENING,
            'connectivity': QuestionIntent.CONNECTIVITY,
            'composition': QuestionIntent.COMPOSITION,
            'quantification': QuestionIntent.QUANTIFICATION,
            'mechanism': QuestionIntent.MECHANISM,
        }
        intent = intent_map.get(result.get('intent', ''), QuestionIntent.UNKNOWN)

        # Depth mapping
        depth_map = {
            'shallow': AnalysisDepth.SHALLOW,
            'medium': AnalysisDepth.MEDIUM,
            'deep': AnalysisDepth.DEEP,
        }
        depth = depth_map.get(result.get('recommended_depth', ''), AnalysisDepth.MEDIUM)

        # Planner mapping
        planner_map = {
            'focus_driven': PlannerType.FOCUS_DRIVEN,
            'comparative': PlannerType.COMPARATIVE,
            'adaptive': PlannerType.ADAPTIVE,
        }
        planner = planner_map.get(result.get('recommended_planner', ''), PlannerType.ADAPTIVE)

        # Modalities
        modality_map = {
            'molecular': Modality.MOLECULAR,
            'morphological': Modality.MORPHOLOGICAL,
            'projection': Modality.PROJECTION,
            'spatial': Modality.SPATIAL,
        }
        modalities = [modality_map[m] for m in result.get('expected_modalities', [])
                      if m in modality_map]

        return IntentClassification(
            intent=intent,
            confidence=result.get('intent_confidence', 0.5),
            reasoning=result.get('intent_reasoning', ''),
            recommended_depth=depth,
            recommended_planner=planner,
            expected_modalities=modalities or [Modality.MOLECULAR],
            expected_entity_types=result.get('expected_entity_types', []),
            sub_questions=result.get('sub_questions', []),
            key_concepts=result.get('key_concepts', []),
            analysis_goals=result.get('analysis_goals', []),
        )

    def _fallback_classification(self, question: str) -> IntentClassification:
        """Fallback规则分类"""
        q_lower = question.lower()

        # 简单规则
        if any(w in q_lower for w in ['stand for', 'full name', 'abbreviation', 'define', 'what is']):
            intent = QuestionIntent.DEFINITION
            depth = AnalysisDepth.SHALLOW
            planner = PlannerType.ADAPTIVE
        elif any(w in q_lower for w in ['compare', 'versus', 'vs', 'difference']):
            intent = QuestionIntent.COMPARISON
            depth = AnalysisDepth.MEDIUM
            planner = PlannerType.COMPARATIVE
        elif any(w in q_lower for w in ['which', 'find', 'top', 'highest', 'screen']):
            intent = QuestionIntent.SCREENING
            depth = AnalysisDepth.MEDIUM
            planner = PlannerType.COMPARATIVE
        elif any(w in q_lower for w in ['tell me about', 'comprehensive', 'analyze', 'profile']):
            intent = QuestionIntent.PROFILING
            depth = AnalysisDepth.DEEP
            planner = PlannerType.FOCUS_DRIVEN
        elif any(w in q_lower for w in ['project', 'connect', 'target', 'output']):
            intent = QuestionIntent.CONNECTIVITY
            depth = AnalysisDepth.MEDIUM
            planner = PlannerType.FOCUS_DRIVEN
        else:
            intent = QuestionIntent.UNKNOWN
            depth = AnalysisDepth.MEDIUM
            planner = PlannerType.ADAPTIVE

        return IntentClassification(
            intent=intent,
            confidence=0.5,
            reasoning="Fallback rule-based classification",
            recommended_depth=depth,
            recommended_planner=planner,
            expected_modalities=[Modality.MOLECULAR],
            expected_entity_types=['GeneMarker', 'Region'],
            sub_questions=[],
            key_concepts=[],
            analysis_goals=[],
        )


# ==================== Entity Recognition ====================

class LLMEntityRecognizer:
    """
    LLM+KG混合实体识别器

    策略：
    1. LLM理解语义，提取候选实体
    2. KG验证实体存在性
    3. 返回验证通过的实体
    """

    def __init__(self, llm: LLMClient, db_executor):
        self.llm = llm
        self.db = db_executor

        # 实体缓存
        self._entity_cache: Dict[str, List[Dict]] = {}

    def recognize(self, question: str) -> List[Entity]:
        """识别问题中的实体"""
        logger.info(f"🔍 Recognizing entities...")

        # Step 1: LLM提取候选实体
        candidates = self._llm_extract_candidates(question)
        logger.info(f"   LLM candidates: {len(candidates)}")

        # Step 2: KG验证
        validated = []
        for candidate in candidates:
            entity = self._validate_in_kg(candidate)
            if entity:
                validated.append(entity)
                logger.info(f"   ✓ Validated: {entity.name} ({entity.entity_type})")

        # Step 3: Fallback - 正则提取 + KG验证
        if not validated:
            logger.info("   Using regex fallback...")
            regex_candidates = self._regex_extract(question)
            for candidate in regex_candidates:
                entity = self._validate_in_kg(candidate)
                if entity:
                    validated.append(entity)

        logger.info(f"   Final: {len(validated)} entities")
        return validated

    def _llm_extract_candidates(self, question: str) -> List[Dict]:
        """使用LLM提取候选实体"""
        system_prompt = """You are a neuroscience entity extractor.

Extract entities from the question that might be in a brain knowledge graph:
- Gene markers (e.g., Car3, Pvalb, Sst, Vip, Lamp5, Gad1)
- Brain regions (e.g., MOp, VISp, SSp, MOs, HIP, TH)
- Cell types (e.g., interneuron, pyramidal, GABAergic)

Be precise. Only extract actual scientific terms, not common words."""

        user_prompt = f"""Extract entities from this question:

"{question}"

Return JSON:
{{
    "entities": [
        {{"text": "Car3", "type": "GeneMarker", "confidence": 0.95}},
        {{"text": "MOp", "type": "Region", "confidence": 0.9}}
    ]
}}

Only include entities you're confident are scientific terms."""

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
            return result.get('entities', [])
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []

    def _regex_extract(self, question: str) -> List[Dict]:
        """正则提取候选实体"""
        candidates = []

        # 脑区缩写 (2-5个大写字母)
        for match in re.finditer(r'\b([A-Z]{2,5})\b', question):
            text = match.group(1)
            # 过滤常见非实体词
            if text.lower() not in {'what', 'which', 'where', 'when', 'how', 'why'}:
                candidates.append({'text': text, 'type': 'Region', 'confidence': 0.5})

        # 基因名 (首字母大写 + 小写 + 可选数字)
        for match in re.finditer(r'\b([A-Z][a-z]{2,8}\d*)\+?\b', question):
            text = match.group(1)
            if text.lower() not in {'tell', 'about', 'what', 'which', 'where', 'cells', 'neurons'}:
                candidates.append({'text': text, 'type': 'GeneMarker', 'confidence': 0.5})

        return candidates

    def _validate_in_kg(self, candidate: Dict) -> Optional[Entity]:
        """在KG中验证实体"""
        text = candidate.get('text', '')
        entity_type = candidate.get('type', '')

        if not text or len(text) < 2:
            return None

        if entity_type == 'Region':
            result = self._check_region(text)
        elif entity_type == 'GeneMarker':
            result = self._check_gene(text)
        elif entity_type == 'CellType':
            result = self._check_celltype(text)
        else:
            # 尝试所有类型
            result = self._check_region(text) or self._check_gene(text)

        return result

    def _check_region(self, text: str) -> Optional[Entity]:
        """检查Region"""
        query = """
        MATCH (r:Region)
        WHERE r.acronym = $name OR toLower(r.name) CONTAINS toLower($name)
        RETURN r.acronym AS acronym, r.name AS full_name
        LIMIT 1
        """
        result = self.db.run(query, {'name': text})

        if result.get('success') and result.get('data'):
            row = result['data'][0]
            return Entity(
                name=row.get('acronym', text),
                entity_type='Region',
                canonical_name=row.get('full_name', text),
                confidence=1.0,
                metadata={'full_name': row.get('full_name')}
            )
        return None

    def _check_gene(self, text: str) -> Optional[Entity]:
        """检查GeneMarker"""
        query = """
        MATCH (c:Cluster)
        WHERE c.markers CONTAINS $name
        RETURN $name AS gene, count(c) AS cluster_count
        LIMIT 1
        """
        result = self.db.run(query, {'name': text})

        if result.get('success') and result.get('data'):
            return Entity(
                name=text,
                entity_type='GeneMarker',
                canonical_name=text,
                confidence=1.0,
                metadata={'cluster_count': result['data'][0].get('cluster_count', 0)}
            )
        return None

    def _check_celltype(self, text: str) -> Optional[Entity]:
        """检查CellType/Subclass"""
        query = """
        MATCH (s:Subclass)
        WHERE toLower(s.name) CONTAINS toLower($name)
        RETURN s.name AS name
        LIMIT 1
        """
        result = self.db.run(query, {'name': text})

        if result.get('success') and result.get('data'):
            return Entity(
                name=result['data'][0]['name'],
                entity_type='CellType',
                canonical_name=result['data'][0]['name'],
                confidence=1.0,
            )
        return None

    def cluster_entities(self, entities: List[Entity]) -> List[EntityCluster]:
        """聚类相关实体"""
        if not entities:
            return []

        clusters = []

        # 按类型分组
        genes = [e for e in entities if e.entity_type == 'GeneMarker']
        regions = [e for e in entities if e.entity_type == 'Region']
        celltypes = [e for e in entities if e.entity_type == 'CellType']

        # 基因为主的cluster
        if genes:
            clusters.append(EntityCluster(
                primary=genes[0],
                related=regions + celltypes + genes[1:],
                cluster_type='gene_marker',
                relevance_score=0.9
            ))

        # 区域为主的cluster
        elif regions:
            clusters.append(EntityCluster(
                primary=regions[0],
                related=regions[1:] + celltypes,
                cluster_type='region',
                relevance_score=0.85
            ))

        # 细胞类型为主
        elif celltypes:
            clusters.append(EntityCluster(
                primary=celltypes[0],
                related=celltypes[1:],
                cluster_type='cell_type',
                relevance_score=0.8
            ))

        return clusters


# ==================== Strategy Recommender ====================

class LLMStrategyRecommender:
    """
    LLM驱动的策略推荐器

    根据问题和当前状态，推荐最优分析策略
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def recommend_strategy(self,
                           question: str,
                           entities: List[Entity],
                           classification: IntentClassification,
                           current_state: Dict = None) -> Dict:
        """推荐分析策略"""

        system_prompt = """You are a neuroscience analysis strategist.

Given a question, entities, and intent classification, recommend the optimal analysis strategy.

Consider:
1. What queries should be executed first?
2. What modalities are essential vs optional?
3. Is closed-loop analysis needed (source → projection → target)?
4. What statistical validation is required?

Be specific and actionable."""

        entity_str = ", ".join([f"{e.name}({e.entity_type})" for e in entities])

        user_prompt = f"""Recommend analysis strategy:

**Question:** {question}

**Entities Found:** {entity_str}

**Intent:** {classification.intent.value}
**Recommended Depth:** {classification.recommended_depth.value}
**Sub-questions:** {classification.sub_questions}

**Current State:** {json.dumps(current_state) if current_state else 'Initial'}

Return JSON:
{{
    "strategy_name": "focus_driven|comparative|adaptive",
    "strategy_reasoning": "Why this strategy",

    "phase_1": {{
        "goal": "What to achieve",
        "modality": "molecular|morphological|projection",
        "priority_queries": ["Query purpose 1", "Query purpose 2"]
    }},

    "phase_2": {{
        "goal": "Next phase goal",
        "modality": "...",
        "depends_on": "phase_1 results"
    }},

    "closed_loop_needed": true/false,
    "closed_loop_plan": "How to complete the loop if needed",

    "statistical_validation": ["FDR correction", "Permutation test"],

    "expected_insights": ["Insight 1", "Insight 2"]
}}"""

        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
            return result
        except Exception as e:
            logger.warning(f"Strategy recommendation failed: {e}")
            return {
                'strategy_name': classification.recommended_planner.value,
                'strategy_reasoning': 'Fallback to classification recommendation',
                'phase_1': {'goal': 'Initial exploration', 'modality': 'molecular'},
            }


# ==================== Export ====================

__all__ = [
    'LLMClient',
    'OpenAIClient',
    'IntentClassification',
    'LLMIntentClassifier',
    'LLMEntityRecognizer',
    'LLMStrategyRecommender',
]