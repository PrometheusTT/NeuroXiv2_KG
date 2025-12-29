"""
Intelligent Intent Routing
==========================
对应设计图A: User Question → Entity & Intent Layer → Planner Router

包含:
- Intelligent Entity Recognizer: 智能实体识别
- Intent Classifier: 意图分类
- Planner Router: 规划器路由

Author: Lijun
Date: 2025-01
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from core_structures import (
    Entity, Intent, PlannerType, Modality, ThinkResult
)

logger = logging.getLogger(__name__)


# ==================== Entity Recognition ====================

class IntelligentEntityRecognizer:
    """
    智能实体识别器 - 对应设计图A的Entity & Intent Layer

    功能:
    1. 识别神经科学实体（Marker, Region, Class等）
    2. 解析Focus F = (M, T, C) - morphology, transcriptomics, connectivity
    3. 识别别名和缩写
    """

    # 标记基因/蛋白质
    KNOWN_MARKERS = {
        'VIP': {'full': 'vasoactive intestinal peptide', 'type': 'Marker'},
        'SST': {'full': 'somatostatin', 'type': 'Marker'},
        'PVALB': {'full': 'parvalbumin', 'aliases': ['PV', 'Pvalb'], 'type': 'Marker'},
        'GAD1': {'full': 'glutamic acid decarboxylase 1', 'type': 'Marker'},
        'GAD2': {'full': 'glutamic acid decarboxylase 2', 'type': 'Marker'},
        'SLC17A7': {'full': 'vesicular glutamate transporter 1', 'aliases': ['VGLUT1'], 'type': 'Marker'},
        'CAR3': {'full': 'carbonic anhydrase 3', 'aliases': ['Car3'], 'type': 'Marker'},
        'LAMP5': {'full': 'lysosomal associated membrane protein 5', 'type': 'Marker'},
        'NOS1': {'full': 'nitric oxide synthase 1', 'type': 'Marker'},
        'CHAT': {'full': 'choline acetyltransferase', 'type': 'Marker'},
    }

    # 脑区缩写
    KNOWN_REGIONS = {
        'MOP': {'full': 'Primary motor area', 'aliases': ['MOp', 'M1'], 'type': 'Region'},
        'MOS': {'full': 'Secondary motor area', 'aliases': ['MOs', 'M2'], 'type': 'Region'},
        'SSP': {'full': 'Primary somatosensory area', 'aliases': ['SSp', 'S1'], 'type': 'Region'},
        'SSS': {'full': 'Secondary somatosensory area', 'aliases': ['SSs', 'S2'], 'type': 'Region'},
        'VISP': {'full': 'Primary visual area', 'aliases': ['VISp', 'V1'], 'type': 'Region'},
        'ACA': {'full': 'Anterior cingulate area', 'type': 'Region'},
        'RSP': {'full': 'Retrosplenial area', 'type': 'Region'},
        'PL': {'full': 'Prelimbic area', 'type': 'Region'},
        'ILA': {'full': 'Infralimbic area', 'type': 'Region'},
        'TH': {'full': 'Thalamus', 'type': 'Region'},
        'HY': {'full': 'Hypothalamus', 'type': 'Region'},
        'CP': {'full': 'Caudoputamen', 'aliases': ['Striatum'], 'type': 'Region'},
        'HIP': {'full': 'Hippocampus', 'aliases': ['CA1', 'CA3', 'DG'], 'type': 'Region'},
    }

    # 细胞类型
    KNOWN_CELL_TYPES = {
        'GLUTAMATERGIC': {'aliases': ['Glut', 'excitatory'], 'type': 'Class'},
        'GABAERGIC': {'aliases': ['GABA', 'inhibitory', 'interneuron'], 'type': 'Class'},
        'ASTROCYTE': {'type': 'Class'},
        'MICROGLIA': {'type': 'Class'},
        'OLIGODENDROCYTE': {'type': 'Class'},
    }

    def __init__(self):
        # 构建别名映射
        self._build_alias_map()

    def _build_alias_map(self):
        """构建别名到标准名的映射"""
        self.alias_map = {}

        for name, info in self.KNOWN_MARKERS.items():
            self.alias_map[name.lower()] = (name, info)
            if 'aliases' in info:
                for alias in info['aliases']:
                    self.alias_map[alias.lower()] = (name, info)

        for name, info in self.KNOWN_REGIONS.items():
            self.alias_map[name.lower()] = (name, info)
            if 'aliases' in info:
                for alias in info['aliases']:
                    self.alias_map[alias.lower()] = (name, info)

        for name, info in self.KNOWN_CELL_TYPES.items():
            self.alias_map[name.lower()] = (name, info)
            if 'aliases' in info:
                for alias in info['aliases']:
                    self.alias_map[alias.lower()] = (name, info)

    def recognize(self, text: str) -> List[Entity]:
        """
        识别文本中的实体

        Args:
            text: 输入文本（通常是问题）

        Returns:
            识别到的实体列表
        """
        entities = []
        text_lower = text.lower()

        # 1. 精确匹配已知实体
        for alias, (canonical, info) in self.alias_map.items():
            # 单词边界匹配
            pattern = rf'\b{re.escape(alias)}\b'
            if re.search(pattern, text_lower):
                entity = Entity(
                    name=canonical,
                    type=info.get('type', 'Unknown'),
                    confidence=0.95,
                    aliases=info.get('aliases', [])
                )
                # 避免重复
                if not any(e.name == entity.name for e in entities):
                    entities.append(entity)

        # 2. 模式匹配识别潜在实体
        # 识别基因名模式（大写字母+数字）
        gene_pattern = r'\b([A-Z][a-z]*[0-9]*[a-z]*)\b'
        for match in re.finditer(gene_pattern, text):
            name = match.group(1)
            if len(name) >= 2 and name.upper() not in [e.name for e in entities]:
                # 检查是否可能是基因/marker名
                if self._is_likely_gene(name):
                    entities.append(Entity(
                        name=name,
                        type='Marker',
                        confidence=0.7,
                    ))

        # 3. 识别神经元类型描述
        neuron_patterns = [
            (r'(\w+)\+?\s*neurons?', 'Marker'),
            (r'(\w+)-expressing', 'Marker'),
            (r'(\w+)-positive', 'Marker'),
        ]

        for pattern, entity_type in neuron_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                if name.lower() not in ['the', 'a', 'an', 'these', 'those']:
                    if not any(e.name.lower() == name.lower() for e in entities):
                        entities.append(Entity(
                            name=name,
                            type=entity_type,
                            confidence=0.8,
                        ))

        logger.debug(f"Recognized {len(entities)} entities: {[e.name for e in entities]}")
        return entities

    def _is_likely_gene(self, name: str) -> bool:
        """判断是否可能是基因名"""
        # 基因名通常: 3-10字符，包含大写，可能有数字
        if len(name) < 3 or len(name) > 10:
            return False
        if name.lower() in ['the', 'and', 'for', 'with', 'from', 'that', 'this']:
            return False
        # 至少有一个大写字母
        if not any(c.isupper() for c in name):
            return False
        return True

    def get_entity_info(self, name: str) -> Optional[Dict]:
        """获取实体详细信息"""
        name_lower = name.lower()
        if name_lower in self.alias_map:
            canonical, info = self.alias_map[name_lower]
            return {'canonical': canonical, **info}
        return None


# ==================== Intent Classification ====================

class IntentClassifier:
    """
    意图分类器 - 对应设计图A的Intent Classifier

    将用户问题分类为:
    - SIMPLE_QA: 简单问答（定义、缩写展开等）
    - FOCUS_DRIVEN: 深度分析（综合分析某类神经元）
    - COMPARATIVE: 比较分析（A vs B）
    - SCREENING: 筛选排序（找最高/最多的）
    """

    # 意图模式
    INTENT_PATTERNS = {
        Intent.SIMPLE_QA: [
            r'\bwhat does \w+ stand for\b',
            r'\bwhat is the (full name|abbreviation)\b',
            r'\bdefine\s+\w+',
            r'\bwhat is\s+\w+\s*\?',
            r'\bhow many\s+\w+',
            r'\blist\s+(all\s+)?\w+',
        ],
        Intent.COMPARATIVE: [
            r'\bcompare\b',
            r'\bversus\b|\bvs\.?\b',
            r'\bdifference(s)?\s+between\b',
            r'\bsimilarit(y|ies)\s+between\b',
            r'\bhow\s+(do|does)\s+\w+\s+differ\b',
            r'\bcontrast\b',
        ],
        Intent.SCREENING: [
            r'\bwhich\s+\w+\s+(has|have|show)\s+(the\s+)?(highest|lowest|most|least)\b',
            r'\btop\s+\d+\b',
            r'\brank(ing)?\b',
            r'\bfind\s+\w+\s+with\s+(highest|lowest|most|least)\b',
            r'\bidentify\s+\w+\s+that\s+(have|has|show)\b',
        ],
        Intent.FOCUS_DRIVEN: [
            r'\bcomprehensive\b',
            r'\btell\s+me\s+(all\s+)?about\b',
            r'\banalyze\b|\banalysis\b',
            r'\bcharacterize\b',
            r'\bprofile\b|\bprofiling\b',
            r'\bdescribe\s+(in\s+detail|comprehensively)\b',
            r'\bwhat\s+(are|is)\s+the\s+.*(features|properties|characteristics)\b',
        ],
    }

    # 焦点模态关键词
    MODALITY_KEYWORDS = {
        Modality.MOLECULAR: [
            'marker', 'gene', 'expression', 'molecular', 'transcriptom',
            'rna', 'protein', 'cluster', 'subclass', 'type',
        ],
        Modality.MORPHOLOGICAL: [
            'morpholog', 'axon', 'dendrite', 'soma', 'shape', 'structure',
            'branch', 'arbor', 'length', 'volume',
        ],
        Modality.PROJECTION: [
            'project', 'connect', 'target', 'pathway', 'circuit',
            'input', 'output', 'afferent', 'efferent',
        ],
    }

    def classify(self, question: str, entities: List[Entity]) -> Tuple[Intent, List[Modality]]:
        """
        分类问题意图和焦点模态

        Args:
            question: 用户问题
            entities: 识别到的实体

        Returns:
            (intent, focus_modalities)
        """
        question_lower = question.lower()

        # 1. 模式匹配确定意图
        intent_scores = {intent: 0 for intent in Intent}

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    intent_scores[intent] += 1

        # 2. 根据实体数量调整
        if len(entities) >= 2:
            # 多个实体可能是比较
            intent_scores[Intent.COMPARATIVE] += 0.5

        # 3. 检查问题长度和复杂度
        word_count = len(question.split())
        if word_count > 15:
            intent_scores[Intent.FOCUS_DRIVEN] += 0.5

        # 4. 确定最终意图
        best_intent = max(intent_scores, key=intent_scores.get)

        # 如果没有明确模式，根据问题特征判断
        if intent_scores[best_intent] == 0:
            if word_count <= 8:
                best_intent = Intent.SIMPLE_QA
            else:
                best_intent = Intent.FOCUS_DRIVEN

        # 5. 确定焦点模态
        focus_modalities = self._detect_focus_modalities(question_lower)

        # 默认至少包含molecular
        if not focus_modalities:
            focus_modalities = [Modality.MOLECULAR]

        logger.debug(f"Classified intent: {best_intent.value}, modalities: {[m.value for m in focus_modalities]}")
        return best_intent, focus_modalities

    def _detect_focus_modalities(self, question_lower: str) -> List[Modality]:
        """检测问题关注的模态"""
        modalities = []

        for modality, keywords in self.MODALITY_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                modalities.append(modality)

        return modalities


# ==================== Planner Router ====================

class PlannerRouter:
    """
    规划器路由 - 对应设计图A的Planner Router

    根据意图将任务路由到合适的Planner:
    - SIMPLE_QA → Adaptive Planner
    - FOCUS_DRIVEN → Focus Driven Planner
    - COMPARATIVE/SCREENING → Comparative Analysis Planner
    """

    INTENT_TO_PLANNER = {
        Intent.SIMPLE_QA: PlannerType.ADAPTIVE,
        Intent.FOCUS_DRIVEN: PlannerType.FOCUS_DRIVEN,
        Intent.COMPARATIVE: PlannerType.COMPARATIVE,
        Intent.SCREENING: PlannerType.COMPARATIVE,
    }

    def route(self, intent: Intent, entities: List[Entity],
              modalities: List[Modality]) -> Tuple[PlannerType, Dict[str, Any]]:
        """
        路由到合适的规划器

        Args:
            intent: 分类的意图
            entities: 识别的实体
            modalities: 焦点模态

        Returns:
            (planner_type, planner_config)
        """
        planner_type = self.INTENT_TO_PLANNER.get(intent, PlannerType.ADAPTIVE)

        # 根据planner类型构建配置
        config = {
            'entities': entities,
            'focus_modalities': modalities,
        }

        if planner_type == PlannerType.ADAPTIVE:
            config['max_steps'] = 3
            config['early_exit'] = True

        elif planner_type == PlannerType.FOCUS_DRIVEN:
            config['max_steps'] = 10
            config['require_all_modalities'] = len(modalities) >= 2
            config['depth_first'] = True

        elif planner_type == PlannerType.COMPARATIVE:
            config['max_steps'] = 8
            config['parallel_analysis'] = len(entities) >= 2
            config['require_statistics'] = True

        logger.debug(f"Routed to {planner_type.value} planner")
        return planner_type, config


# ==================== Integrated Intent Router ====================

class IntentRouter:
    """
    集成的意图路由器 - 对应设计图A的完整流程

    User Question → Entity Recognition → Intent Classification → Planner Routing
    """

    def __init__(self):
        self.entity_recognizer = IntelligentEntityRecognizer()
        self.intent_classifier = IntentClassifier()
        self.planner_router = PlannerRouter()

    def process(self, question: str) -> ThinkResult:
        """
        处理用户问题，执行Think阶段

        Args:
            question: 用户问题

        Returns:
            ThinkResult包含实体、意图、模态等
        """
        # 1. 实体识别
        entities = self.entity_recognizer.recognize(question)

        # 2. 意图分类
        intent, focus_modalities = self.intent_classifier.classify(question, entities)

        # 3. 规划器路由
        planner_type, planner_config = self.planner_router.route(
            intent, entities, focus_modalities
        )

        # 4. 构建约束
        constraints = {
            'planner_type': planner_type,
            'planner_config': planner_config,
        }

        # 5. 生成推理说明
        reasoning = self._generate_think_reasoning(
            question, entities, intent, focus_modalities, planner_type
        )

        return ThinkResult(
            entities=entities,
            intent=intent,
            focus_modalities=focus_modalities,
            key_constraints=constraints,
            reasoning=reasoning,
        )

    def _generate_think_reasoning(self, question: str, entities: List[Entity],
                                  intent: Intent, modalities: List[Modality],
                                  planner_type: PlannerType) -> str:
        """生成Think阶段的推理说明"""
        entity_names = [e.name for e in entities]
        modality_names = [m.value for m in modalities]

        reasoning = f"""[THINK] Question Analysis:
- Identified entities: {entity_names}
- Classified intent: {intent.value}
- Focus modalities: {modality_names}
- Selected planner: {planner_type.value}

Reasoning: The question requires {'deep analysis' if intent == Intent.FOCUS_DRIVEN else
        'comparison' if intent == Intent.COMPARATIVE else
        'screening' if intent == Intent.SCREENING else
        'direct answer'}. 
{'Multiple entities detected, enabling parallel analysis.' if len(entities) > 1 else ''}
{'Will prioritize ' + modality_names[0] + ' data.' if modalities else ''}"""

        return reasoning


# ==================== Export ====================

__all__ = [
    'IntelligentEntityRecognizer',
    'IntentClassifier',
    'PlannerRouter',
    'IntentRouter',
]