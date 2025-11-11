"""
AIPOM-CoT V8: Advanced Iterative Planning with Orchestrated Multi-modal CoT
专为神经科学知识图谱推理设计，Nature子刊级别实现

核心创新:
1. Neuroscience-aware query decomposition
2. Multi-modal fingerprint analysis
3. Statistical validation integration
4. Enhanced reflection with data quality checks
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from neo4j_exec import Neo4jExec
from schema_cache import SchemaCache
from operators import validate_and_fix_cypher, generate_safe_cypher
from llm import LLMClient, ToolSpec

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """查询复杂度级别"""
    SIMPLE = 1  # 单跳检索
    PATTERN = 2  # 模式识别
    MULTIHOP = 3  # 多跳推理
    HYPOTHESIS = 4  # 假设生成


class ModalityType(Enum):
    """数据模态类型"""
    MOLECULAR = "molecular"
    MORPHOLOGICAL = "morphological"
    PROJECTION = "projection"
    SPATIAL = "spatial"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    purpose: str
    query: str
    modality: ModalityType
    success: bool
    data: List[Dict]
    statistics: Optional[Dict] = None
    quality_score: float = 0.0


@dataclass
class Fingerprint:
    """区域指纹"""
    region_id: str
    molecular: np.ndarray
    morphological: np.ndarray
    projection: np.ndarray

    def similarity(self, other: 'Fingerprint', metric: str = 'cosine') -> Dict[str, float]:
        """计算与另一个指纹的相似度"""
        return {
            'molecular': self._compute_similarity(self.molecular, other.molecular, metric),
            'morphological': self._compute_similarity(self.morphological, other.morphological, metric),
            'projection': self._compute_similarity(self.projection, other.projection, metric)
        }

    @staticmethod
    def _compute_similarity(v1: np.ndarray, v2: np.ndarray, metric: str) -> float:
        """计算向量相似度"""
        if metric == 'cosine':
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        elif metric == 'euclidean':
            return float(1.0 / (1.0 + np.linalg.norm(v1 - v2)))
        else:  # correlation
            if len(v1) < 2:
                return 0.0
            corr, _ = stats.pearsonr(v1, v2)
            return float(corr) if not np.isnan(corr) else 0.0


class StatisticalTools:
    """统计分析工具集"""

    @staticmethod
    def correlation_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Pearson相关性检验"""
        if len(x) < 3 or len(y) < 3:
            return {'r': 0.0, 'p_value': 1.0}

        r, p = stats.pearsonr(x, y)
        return {
            'r': float(r) if not np.isnan(r) else 0.0,
            'p_value': float(p) if not np.isnan(p) else 1.0
        }

    @staticmethod
    def compare_distributions(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, Any]:
        """比较两个分布"""
        # Mann-Whitney U test (非参数检验)
        statistic, p_value = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')

        # Effect size (Cohen's d)
        mean_diff = np.mean(dist1) - np.mean(dist2)
        pooled_std = np.sqrt((np.var(dist1) + np.var(dist2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'mean_diff': float(mean_diff)
        }

    @staticmethod
    def compute_enrichment(observed: np.ndarray,
                           expected: np.ndarray) -> Dict[str, float]:
        """计算富集程度"""
        if np.sum(expected) == 0:
            return {'fold_change': 0.0, 'p_value': 1.0}

        fold_change = np.mean(observed) / np.mean(expected)

        # 使用t检验
        t_stat, p_value = stats.ttest_ind(observed, expected)

        return {
            'fold_change': float(fold_change),
            'p_value': float(p_value) if not np.isnan(p_value) else 1.0
        }


class FingerprintAnalyzer:
    """多模态指纹分析器"""

    def __init__(self, neo4j: Neo4jExec):
        self.neo4j = neo4j

    def compute_region_fingerprint(self, region_acronym: str) -> Optional[Fingerprint]:
        """计算区域的三维指纹"""
        try:
            # 1. Molecular fingerprint (subclass distribution)
            molecular = self._get_molecular_fingerprint(region_acronym)

            # 2. Morphological fingerprint
            morphological = self._get_morphological_fingerprint(region_acronym)

            # 3. Projection fingerprint
            projection = self._get_projection_fingerprint(region_acronym)

            if molecular is None or morphological is None or projection is None:
                return None

            return Fingerprint(
                region_id=region_acronym,
                molecular=molecular,
                morphological=morphological,
                projection=projection
            )

        except Exception as e:
            logger.error(f"计算指纹失败: {e}")
            return None

    def _get_molecular_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """获取分子指纹（细胞类型分布）"""
        query = """
        MATCH (r:Region {acronym: $acronym})-[h:HAS_SUBCLASS]->(s:Subclass)
        RETURN s.name AS subclass, h.pct_cells AS pct
        ORDER BY s.name
        """
        result = self.neo4j.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # 构建向量（按subclass排序）
        subclass_dict = {row['subclass']: row['pct'] for row in result['data']}

        # 使用标准subclass顺序
        standard_subclasses = self._get_standard_subclasses()
        vector = np.array([subclass_dict.get(sc, 0.0) for sc in standard_subclasses])

        return vector

    def _get_morphological_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """获取形态学指纹"""
        query = """
        MATCH (r:Region {acronym: $acronym})
        RETURN r.axonal_length AS axon_len,
               r.dendritic_length AS dend_len,
               r.axonal_branches AS axon_br,
               r.dendritic_branches AS dend_br,
               r.axonal_maximum_branch_order AS axon_order,
               r.dendritic_maximum_branch_order AS dend_order
        """
        result = self.neo4j.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        data = result['data'][0]
        vector = np.array([
            data.get('axon_len', 0.0) or 0.0,
            data.get('dend_len', 0.0) or 0.0,
            data.get('axon_br', 0.0) or 0.0,
            data.get('dend_br', 0.0) or 0.0,
            data.get('axon_order', 0.0) or 0.0,
            data.get('dend_order', 0.0) or 0.0
        ], dtype=float)

        # 标准化
        if np.sum(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def _get_projection_fingerprint(self, region: str) -> Optional[np.ndarray]:
        """获取投射指纹"""
        query = """
        MATCH (r:Region {acronym: $acronym})-[p:PROJECT_TO]->(t:Region)
        RETURN t.acronym AS target, p.weight AS weight
        ORDER BY t.acronym
        """
        result = self.neo4j.run(query, {'acronym': region})

        if not result['success'] or not result['data']:
            return None

        # 构建投射向量
        target_dict = {row['target']: row['weight'] for row in result['data']}

        # 使用标准目标区域顺序
        standard_targets = self._get_standard_targets()
        vector = np.array([target_dict.get(t, 0.0) for t in standard_targets])

        # 标准化
        if np.sum(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def _get_standard_subclasses(self) -> List[str]:
        """获取标准subclass列表"""
        query = "MATCH (s:Subclass) RETURN s.name AS name ORDER BY s.name LIMIT 100"
        result = self.neo4j.run(query)

        if result['success'] and result['data']:
            return [row['name'] for row in result['data']]

        # 默认列表
        return ['IT', 'ET', 'CT', 'PT', 'NP', 'Pvalb', 'Sst', 'Vip']

    def _get_standard_targets(self) -> List[str]:
        """获取标准投射目标列表"""
        query = """
        MATCH (r:Region)-[:PROJECT_TO]->(t:Region)
        RETURN DISTINCT t.acronym AS target
        ORDER BY target
        LIMIT 100
        """
        result = self.neo4j.run(query)

        if result['success'] and result['data']:
            return [row['target'] for row in result['data']]

        # 默认主要靶区
        return ['MOs', 'ACAd', 'ENTl', 'CP', 'TH', 'HY']


class NeuroscienceQueryPlanner:
    """神经科学特定的查询规划器 - Schema-Guided版本"""

    def __init__(self, schema: SchemaCache, llm: LLMClient):
        self.schema = schema
        self.llm = llm

        # ⭐ 核心改动：集成Schema-Guided CoT Generator
        from .schema_guided_cot import SchemaGuidedCoTGenerator
        self.cot_generator = SchemaGuidedCoTGenerator(schema)

    def classify_query(self, question: str) -> QueryComplexity:
        """分类查询复杂度"""
        # 使用LLM进行分类
        prompt = f"""Classify this neuroscience query by complexity:

Query: {question}

Complexity levels:
1. SIMPLE: Single fact retrieval (e.g., "Find Car3+ regions")
2. PATTERN: Pattern recognition (e.g., "Regions with similar projections")
3. MULTIHOP: Multi-hop reasoning (e.g., "Molecular features of projection targets")
4. HYPOTHESIS: Hypothesis generation (e.g., "Why do regions differ?")

Return only the level number (1-4)."""

        response = self.llm.run_planner_json(
            "You classify neuroscience queries.",
            prompt
        )

        try:
            level = int(response.strip())
            return QueryComplexity(level)
        except:
            # 默认为PATTERN
            return QueryComplexity.PATTERN

    def decompose_query(self, question: str,
                        complexity: QueryComplexity) -> Dict[str, Any]:
        """
        ⭐ 核心改动：首先尝试Schema-Guided CoT生成
        如果成功，直接使用；否则回退到LLM分解
        """

        # 1. 尝试Schema-Guided CoT生成
        try:
            cot_result = self.cot_generator.generate_cot(question)

            # 检查是否成功生成了有意义的推理链
            if cot_result.get('reasoning_chain') and len(cot_result['reasoning_chain']) > 1:
                logger.info(f"✓ Schema-Guided CoT generated {len(cot_result['reasoning_chain'])} steps")

                # 转换为标准格式
                return self._convert_cot_to_plan(cot_result)

        except Exception as e:
            logger.warning(f"Schema-Guided CoT failed: {e}, falling back to LLM")

        # 2. 回退到LLM分解
        logger.info("Using LLM-based decomposition")

        # 根据复杂度使用不同的分解策略
        if complexity == QueryComplexity.SIMPLE:
            return self._decompose_simple(question)
        elif complexity == QueryComplexity.PATTERN:
            return self._decompose_pattern(question)
        elif complexity == QueryComplexity.MULTIHOP:
            return self._decompose_multihop(question)
        else:  # HYPOTHESIS
            return self._decompose_hypothesis(question)

    def _convert_cot_to_plan(self, cot_result: Dict) -> Dict[str, Any]:
        """
        将Schema-Guided CoT结果转换为执行计划
        """
        reasoning_chain = cot_result['reasoning_chain']

        steps = []
        for step_info in reasoning_chain:
            step = {
                'purpose': step_info.get('purpose', 'Unknown'),
                'modality': step_info.get('modality', 'molecular'),
                'query': step_info.get('query', ''),
                'depends_on': step_info.get('depends_on', []),
                'schema_path': step_info.get('schema_path', []),
                'action': step_info.get('action', 'query')
            }

            # 如果没有查询，尝试生成
            if not step['query'] and step.get('query_template'):
                step['query'] = self._generate_from_template(
                    step['query_template'],
                    cot_result.get('primary_entity', {})
                )

            steps.append(step)

        return {
            'steps': steps,
            'analysis_plan': f"Schema-guided analysis of {cot_result.get('primary_entity', {}).get('text', 'entities')}",
            'expected_modalities': cot_result.get('expected_modalities', []),
            'complexity': cot_result.get('complexity', 'pattern'),
            'source': 'schema_guided'
        }

    def _generate_from_template(self, template_name: str, entity: Dict) -> str:
        """从模板生成查询"""
        # 这里可以扩展更多模板
        templates = {
            'region_by_gene': f"""
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.name CONTAINS '{entity.get('text', '')}'
RETURN r.acronym AS region, 
       r.region_id AS region_id,
       avg(h.pct_cells) AS enrichment
ORDER BY enrichment DESC
LIMIT 10
            """.strip()
        }

        return templates.get(template_name, '')

    # 保持原有的LLM分解方法作为后备

    def _decompose_simple(self, question: str) -> Dict[str, Any]:
        """分解简单查询"""
        system = "You are a neuroscience data analyst. Return JSON only."

        user = f"""Break down this retrieval query into Cypher:

Question: {question}

Schema:
{self.schema.summary_text()}

Return JSON:
{{
  "steps": [
    {{
      "purpose": "retrieval goal",
      "modality": "molecular|morphological|projection",
      "query": "MATCH ... RETURN ... LIMIT 20"
    }}
  ],
  "expected_output": "description"
}}"""

        response = self.llm.run_planner_json(system, user)
        return json.loads(response)

    def _decompose_pattern(self, question: str) -> Dict[str, Any]:
        """分解模式识别查询"""
        system = "You are a neuroscience pattern analyst. Return JSON only."

        user = f"""Break down this pattern recognition query:

Question: {question}

This requires:
1. Retrieve relevant entities
2. Compute features/fingerprints
3. Compare/cluster

Schema:
{self.schema.summary_text()}

Return JSON:
{{
  "steps": [
    {{
      "purpose": "step description",
      "modality": "molecular|morphological|projection",
      "query": "MATCH ... RETURN ... LIMIT 50",
      "analysis": "correlation|clustering|enrichment"
    }}
  ],
  "analysis_plan": "how to find patterns"
}}"""

        response = self.llm.run_planner_json(system, user)
        return json.loads(response)

    def _decompose_multihop(self, question: str) -> Dict[str, Any]:
        """分解多跳推理查询"""
        system = "You are a neuroscience reasoning analyst. Return JSON only."

        user = f"""Break down this multi-hop query into sequential steps:

Question: {question}

Strategy:
1. Identify start entities
2. Follow relationships
3. Retrieve end properties
4. Integrate across modalities

Schema:
{self.schema.summary_text()}

Return JSON:
{{
  "steps": [
    {{
      "hop": 1,
      "purpose": "find Car3+ regions",
      "modality": "molecular",
      "query": "MATCH (r:Region)-[:HAS_SUBCLASS]->... LIMIT 20",
      "output_var": "regions"
    }},
    {{
      "hop": 2,
      "purpose": "find projection targets",
      "modality": "projection",
      "query": "MATCH (r:Region)-[:PROJECT_TO]->(t) WHERE r.acronym IN $regions ...",
      "output_var": "targets",
      "depends_on": ["regions"]
    }}
  ],
  "synthesis": "how to combine results"
}}"""

        response = self.llm.run_planner_json(system, user)
        return json.loads(response)

    def _decompose_hypothesis(self, question: str) -> Dict[str, Any]:
        """分解假设生成查询"""
        system = "You are a neuroscience hypothesis generator. Return JSON only."

        user = f"""Plan hypothesis generation for:

Question: {question}

Strategy:
1. Collect evidence across modalities
2. Identify patterns/anomalies
3. Generate explanations
4. Support with statistics

Schema:
{self.schema.summary_text()}

Return JSON:
{{
  "evidence_steps": [
    {{
      "purpose": "gather molecular evidence",
      "modality": "molecular",
      "query": "MATCH ... RETURN ...",
      "expected": "subclass distributions"
    }}
  ],
  "analysis_steps": [
    {{
      "type": "correlation|comparison|enrichment",
      "variables": ["var1", "var2"]
    }}
  ],
  "reasoning_framework": "computational|developmental|functional"
}}"""

        response = self.llm.run_planner_json(system, user)
        return json.loads(response)


class EnhancedReflection:
    """增强的反思机制"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reflect_on_results(self,
                           question: str,
                           steps: List[ReasoningStep],
                           complexity: QueryComplexity) -> Dict[str, Any]:
        """对结果进行反思"""

        # 数据质量检查
        quality_issues = self._check_data_quality(steps)

        # 完整性检查
        completeness = self._check_completeness(steps, complexity)

        # 一致性检查
        consistency = self._check_consistency(steps)

        # 决定是否需要更多步骤
        if quality_issues or not completeness['is_complete']:
            return self._generate_followup_plan(
                question, steps, quality_issues, completeness
            )

        # 生成最终总结
        return {
            'continue': False,
            'quality_score': self._compute_quality_score(steps),
            'summary': self._generate_summary(question, steps)
        }

    def _check_data_quality(self, steps: List[ReasoningStep]) -> List[str]:
        """检查数据质量"""
        issues = []

        for step in steps:
            # 检查空结果
            if not step.data:
                issues.append(f"Step {step.step_id}: No data returned")

            # 检查数据量
            if len(step.data) < 3 and step.success:
                issues.append(f"Step {step.step_id}: Insufficient data ({len(step.data)} rows)")

            # 检查NULL值
            if step.data:
                null_fields = [k for k, v in step.data[0].items() if v is None]
                if len(null_fields) > len(step.data[0]) / 2:
                    issues.append(f"Step {step.step_id}: Many NULL values in {null_fields}")

        return issues

    def _check_completeness(self,
                            steps: List[ReasoningStep],
                            complexity: QueryComplexity) -> Dict[str, Any]:
        """检查完整性"""

        # 按复杂度要求不同的模态覆盖
        required_modalities = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.PATTERN: 2,
            QueryComplexity.MULTIHOP: 2,
            QueryComplexity.HYPOTHESIS: 3
        }

        covered_modalities = set(step.modality for step in steps)
        required = required_modalities.get(complexity, 1)

        is_complete = len(covered_modalities) >= required

        return {
            'is_complete': is_complete,
            'covered_modalities': list(covered_modalities),
            'required': required,
            'missing': required - len(covered_modalities) if not is_complete else 0
        }

    def _check_consistency(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        """检查一致性"""
        # 检查不同步骤的结果是否一致
        # 例如：region在不同查询中的数据是否匹配

        inconsistencies = []

        # 提取所有提到的region
        all_regions = set()
        for step in steps:
            for row in step.data:
                if 'region' in row:
                    all_regions.add(row['region'])
                if 'acronym' in row:
                    all_regions.add(row['acronym'])

        # 简单检查：相同region在不同步骤中的数据是否合理
        # 这里可以扩展更复杂的一致性检查

        return {
            'is_consistent': len(inconsistencies) == 0,
            'issues': inconsistencies
        }

    def _generate_followup_plan(self,
                                question: str,
                                steps: List[ReasoningStep],
                                quality_issues: List[str],
                                completeness: Dict) -> Dict[str, Any]:
        """生成后续步骤计划"""

        system = "You plan next steps for neuroscience analysis."

        # 总结当前状态
        current_state = {
            'completed_steps': len(steps),
            'quality_issues': quality_issues,
            'missing_modalities': completeness.get('missing', 0)
        }

        user = f"""Current analysis state:

Question: {question}

Completed steps: {current_state['completed_steps']}
Quality issues: {quality_issues}
Completeness: {completeness}

Recent data samples:
{self._format_recent_data(steps)}

Generate 1-2 follow-up queries to address gaps.

Return JSON:
{{
  "continue": true,
  "reason": "what's missing",
  "next_steps": [
    {{
      "purpose": "fill gap",
      "modality": "molecular|morphological|projection",
      "query": "MATCH ... RETURN ... LIMIT 30"
    }}
  ]
}}"""

        response = self.llm.run_planner_json(system, user)
        return json.loads(response)

    def _format_recent_data(self, steps: List[ReasoningStep]) -> str:
        """格式化最近的数据样本"""
        recent = steps[-2:] if len(steps) >= 2 else steps

        lines = []
        for step in recent:
            sample = step.data[:3] if step.data else []
            lines.append(f"Step {step.step_id}: {step.purpose}")
            lines.append(f"  Rows: {len(step.data)}, Sample: {sample}")

        return "\n".join(lines)

    def _compute_quality_score(self, steps: List[ReasoningStep]) -> float:
        """计算整体质量分数"""
        if not steps:
            return 0.0

        scores = []
        for step in steps:
            score = 0.0

            # 成功执行 +0.3
            if step.success:
                score += 0.3

            # 有数据 +0.3
            if step.data:
                score += 0.3

            # 数据量充足 +0.2
            if len(step.data) >= 5:
                score += 0.2

            # 有统计信息 +0.2
            if step.statistics:
                score += 0.2

            scores.append(score)

        return float(np.mean(scores))

    def _generate_summary(self,
                          question: str,
                          steps: List[ReasoningStep]) -> str:
        """生成最终总结"""

        system = "You write concise neuroscience analysis summaries."

        # 整理所有数据
        all_data = []
        for step in steps:
            all_data.append({
                'purpose': step.purpose,
                'modality': step.modality.value,
                'rows': len(step.data),
                'sample': step.data[:5] if step.data else [],
                'statistics': step.statistics
            })

        user = f"""Summarize this neuroscience analysis:

Question: {question}

Analysis steps and data:
{json.dumps(all_data, indent=2, ensure_ascii=False)}

Write a 2-3 paragraph summary:
1. What was found
2. Key patterns/statistics
3. Answer to the original question

Be specific and cite data."""

        return self.llm.summarize(json.dumps(all_data, ensure_ascii=False))


class AIPOMCoTV8:
    """
    AIPOM-CoT V8: 完整实现

    核心特性:
    1. 神经科学感知的查询规划
    2. 多模态指纹分析
    3. 统计验证集成
    4. 增强反思机制
    """

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_pwd: str,
                 database: str,
                 openai_api_key: Optional[str] = None,
                 planner_model: str = "gpt-5",
                 summarizer_model: str = "gpt-4o"):

        # 初始化组件
        self.db = Neo4jExec(neo4j_uri, neo4j_user, neo4j_pwd, database=database)
        self.schema = SchemaCache()

        with self.db.driver.session(database=database) as s:
            self.schema.load_from_db(s)

        self.llm = LLMClient(
            api_key=openai_api_key,
            planner_model=planner_model,
            summarizer_model=summarizer_model
        )

        # 专用模块
        self.planner = NeuroscienceQueryPlanner(self.schema, self.llm)
        self.fingerprint = FingerprintAnalyzer(self.db)
        self.reflection = EnhancedReflection(self.llm)
        self.stats = StatisticalTools()

        # 注册工具
        self.tools = self._register_tools()

        logger.info("AIPOM-CoT V8 initialized")

    def _register_tools(self) -> List[ToolSpec]:
        """注册所有工具"""
        return [
            ToolSpec(
                name="neo4j_query",
                description="Execute read-only Cypher query. Must include LIMIT.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["query"]
                }
            ),
            ToolSpec(
                name="compute_fingerprint",
                description="Compute multi-modal fingerprint for a brain region.",
                parameters={
                    "type": "object",
                    "properties": {
                        "region_acronym": {"type": "string"}
                    },
                    "required": ["region_acronym"]
                }
            ),
            ToolSpec(
                name="compare_fingerprints",
                description="Compare fingerprints of two regions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "region1": {"type": "string"},
                        "region2": {"type": "string"},
                        "metric": {"type": "string", "enum": ["cosine", "euclidean", "correlation"]}
                    },
                    "required": ["region1", "region2"]
                }
            ),
            ToolSpec(
                name="correlation_test",
                description="Test correlation between two variables.",
                parameters={
                    "type": "object",
                    "properties": {
                        "variable1": {"type": "array", "items": {"type": "number"}},
                        "variable2": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["variable1", "variable2"]
                }
            ),
            ToolSpec(
                name="enrichment_analysis",
                description="Test if observed values are enriched vs expected.",
                parameters={
                    "type": "object",
                    "properties": {
                        "observed": {"type": "array", "items": {"type": "number"}},
                        "expected": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["observed", "expected"]
                }
            )
        ]

    def _tool_router(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """路由工具调用"""

        if name == "neo4j_query":
            query = validate_and_fix_cypher(self.schema, args["query"])
            return self.db.run(query, args.get("params"))

        elif name == "compute_fingerprint":
            fp = self.fingerprint.compute_region_fingerprint(args["region_acronym"])
            if fp is None:
                return {"error": "Failed to compute fingerprint"}
            return {
                "region": fp.region_id,
                "molecular": fp.molecular.tolist(),
                "morphological": fp.morphological.tolist(),
                "projection": fp.projection.tolist()
            }

        elif name == "compare_fingerprints":
            fp1 = self.fingerprint.compute_region_fingerprint(args["region1"])
            fp2 = self.fingerprint.compute_region_fingerprint(args["region2"])

            if fp1 is None or fp2 is None:
                return {"error": "Failed to compute fingerprints"}

            metric = args.get("metric", "cosine")
            similarities = fp1.similarity(fp2, metric)

            # 计算mismatch
            mismatch = abs(similarities['morphological'] - similarities['molecular'])

            return {
                "region1": args["region1"],
                "region2": args["region2"],
                "similarities": similarities,
                "mismatch_index": float(mismatch)
            }

        elif name == "correlation_test":
            x = np.array(args["variable1"])
            y = np.array(args["variable2"])
            return self.stats.correlation_test(x, y)

        elif name == "enrichment_analysis":
            obs = np.array(args["observed"])
            exp = np.array(args["expected"])
            return self.stats.compute_enrichment(obs, exp)

        else:
            raise ValueError(f"Unknown tool: {name}")

    def answer(self,
               question: str,
               max_rounds: int = 3) -> Dict[str, Any]:
        """
        主入口：回答问题

        Args:
            question: 用户问题
            max_rounds: 最大推理轮数

        Returns:
            完整的推理结果
        """

        logger.info(f"Processing question: {question}")
        start_time = time.time()

        # 1. 分类查询复杂度
        complexity = self.planner.classify_query(question)
        logger.info(f"Query complexity: {complexity.name}")

        # 2. 分解查询
        plan = self.planner.decompose_query(question, complexity)
        logger.info(f"Generated plan with {len(plan.get('steps', []))} steps")

        # 3. 执行推理循环
        all_steps: List[ReasoningStep] = []
        current_round = 0

        while current_round < max_rounds:
            current_round += 1
            logger.info(f"Round {current_round}/{max_rounds}")

            # 执行当前计划的步骤
            round_steps = self._execute_plan(plan, all_steps)
            all_steps.extend(round_steps)

            # 反思
            reflection = self.reflection.reflect_on_results(
                question, all_steps, complexity
            )

            # 如果完成，退出
            if not reflection.get('continue', False):
                summary = reflection.get('summary', '')
                quality = reflection.get('quality_score', 0.0)

                elapsed = time.time() - start_time

                return {
                    'question': question,
                    'complexity': complexity.name,
                    'rounds': current_round,
                    'total_steps': len(all_steps),
                    'steps': [self._serialize_step(s) for s in all_steps],
                    'quality_score': quality,
                    'answer': summary,
                    'elapsed_time': elapsed
                }

            # 否则，更新计划继续
            plan = {
                'steps': reflection.get('next_steps', [])
            }

        # 达到最大轮数，强制总结
        logger.warning(f"Reached max rounds ({max_rounds})")

        summary = self.reflection._generate_summary(question, all_steps)
        quality = self.reflection._compute_quality_score(all_steps)
        elapsed = time.time() - start_time

        return {
            'question': question,
            'complexity': complexity.name,
            'rounds': max_rounds,
            'total_steps': len(all_steps),
            'steps': [self._serialize_step(s) for s in all_steps],
            'quality_score': quality,
            'answer': summary,
            'warning': 'Incomplete - reached max rounds',
            'elapsed_time': elapsed
        }

    def _execute_plan(self,
                      plan: Dict[str, Any],
                      previous_steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """
        ⭐ 核心改动：支持步骤依赖和变量传递
        """

        steps = plan.get('steps', [])
        executed_steps = []

        # 存储中间结果供后续步骤使用
        step_outputs = {}  # step_id -> data

        for i, step_plan in enumerate(steps):
            step_id = len(previous_steps) + i + 1

            # 解析步骤
            purpose = step_plan.get('purpose', 'Unknown')
            modality_str = step_plan.get('modality', 'molecular')
            query = step_plan.get('query', '')

            # 转换modality
            try:
                modality = ModalityType(modality_str)
            except ValueError:
                modality = ModalityType.MOLECULAR

            # ⭐ 处理依赖：替换查询中的变量
            if step_plan.get('depends_on'):
                query = self._resolve_dependencies(
                    query,
                    step_plan['depends_on'],
                    step_outputs
                )

            # 执行查询
            result = self._execute_step(query, step_plan.get('params', {}))

            # 计算统计（如果需要）
            statistics = None
            if step_plan.get('analysis'):
                statistics = self._compute_statistics(
                    result['data'],
                    step_plan['analysis']
                )

            # 创建步骤记录
            step = ReasoningStep(
                step_id=step_id,
                purpose=purpose,
                query=query,
                modality=modality,
                success=result['success'],
                data=result['data'][:50],  # 限制数据量
                statistics=statistics,
                quality_score=self._evaluate_step_quality(result)
            )

            executed_steps.append(step)

            # ⭐ 保存输出供后续步骤使用
            step_outputs[step_id] = self._extract_key_values(result['data'], purpose)

            logger.info(f"Step {step_id}: {purpose} - {len(result['data'])} rows")

        return executed_steps

    def _resolve_dependencies(self,
                              query: str,
                              depends_on: List[int],
                              step_outputs: Dict[int, Dict]) -> str:
        """
        ⭐ 新增：解析查询依赖，替换变量

        Example:
            query: "WHERE r.acronym IN $enriched_regions"
            depends_on: [2]
            step_outputs: {2: {'regions': ['CLA', 'ACAd']}}

            → "WHERE r.acronym IN ['CLA', 'ACAd']"
        """

        # 收集所有依赖步骤的输出
        available_vars = {}
        for dep_step_id in depends_on:
            if dep_step_id in step_outputs:
                available_vars.update(step_outputs[dep_step_id])

        # 替换查询中的变量
        import re

        # 查找所有 $variable 模式
        variables = re.findall(r'\$(\w+)', query)

        for var in variables:
            if var in available_vars:
                value = available_vars[var]

                # 根据类型格式化
                if isinstance(value, list):
                    # 列表 → ['item1', 'item2']
                    formatted = str(value)
                elif isinstance(value, str):
                    # 字符串 → 'value'
                    formatted = f"'{value}'"
                else:
                    formatted = str(value)

                query = query.replace(f'${var}', formatted)

        return query

    def _extract_key_values(self, data: List[Dict], purpose: str) -> Dict[str, any]:
        """
        ⭐ 新增：从步骤输出提取关键值

        根据步骤目的智能提取需要传递给下游的值
        """
        if not data:
            return {}

        extracted = {}

        # 根据purpose推断需要提取什么
        if 'enriched' in purpose.lower() or 'find region' in purpose.lower():
            # 提取区域列表
            if 'region' in data[0]:
                extracted['enriched_regions'] = [row['region'] for row in data if 'region' in row]
            elif 'acronym' in data[0]:
                extracted['enriched_regions'] = [row['acronym'] for row in data if 'acronym' in row]

        if 'target' in purpose.lower() or 'projection' in purpose.lower():
            # 提取投射目标
            if 'target' in data[0]:
                extracted['target_regions'] = list(set(row['target'] for row in data if 'target' in row))

        # 通用：提取所有列表型值
        for key in data[0].keys():
            if key in ['region', 'acronym', 'source', 'target']:
                extracted[key + 's'] = list(set(row[key] for row in data if key in row and row[key]))

        return extracted

    def _execute_step(self, query: str, params: Dict) -> Dict[str, Any]:
        """执行单个步骤"""

        # 验证和修复查询
        validated_query = validate_and_fix_cypher(self.schema, query)

        # 执行
        result = self.db.run(validated_query, params)

        return result

    def _compute_statistics(self,
                            data: List[Dict],
                            analysis_type: str) -> Dict[str, Any]:
        """计算统计信息"""

        if not data:
            return {}

        if analysis_type == 'correlation':
            # 提取数值列
            numeric_cols = self._get_numeric_columns(data)
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[:2]
                x = np.array([row[col1] for row in data if row[col1] is not None])
                y = np.array([row[col2] for row in data if row[col2] is not None])

                return self.stats.correlation_test(x, y)

        elif analysis_type == 'enrichment':
            # 简单的富集计算
            numeric_cols = self._get_numeric_columns(data)
            if numeric_cols:
                col = numeric_cols[0]
                values = np.array([row[col] for row in data if row[col] is not None])

                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n': len(values)
                }

        return {}

    def _get_numeric_columns(self, data: List[Dict]) -> List[str]:
        """获取数值型列"""
        if not data:
            return []

        sample = data[0]
        numeric_cols = []

        for key, value in sample.items():
            if isinstance(value, (int, float)):
                numeric_cols.append(key)

        return numeric_cols

    def _evaluate_step_quality(self, result: Dict) -> float:
        """评估步骤质量"""
        score = 0.0

        if result['success']:
            score += 0.5

        if result['data']:
            score += 0.3

        if len(result['data']) >= 5:
            score += 0.2

        return score

    def _serialize_step(self, step: ReasoningStep) -> Dict[str, Any]:
        """序列化步骤"""
        return {
            'step_id': step.step_id,
            'purpose': step.purpose,
            'modality': step.modality.value,
            'success': step.success,
            'data_rows': len(step.data),
            'sample_data': step.data[:3],
            'statistics': step.statistics,
            'quality_score': step.quality_score
        }