"""
LLM-Driven Reflection Module
=============================
真正的自我反思 - LLM评估结果质量并决定下一步

核心能力：
1. 结果验证 - 评估执行结果是否符合预期
2. 发现洞察 - 识别关键发现和意外结果
3. 决策推理 - 决定继续/深化/转向/终止
4. 替代生成 - 当失败时生成替代方案

Author: Lijun
Date: 2025-01
"""

import json
import logging
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core_structures import (
    AnalysisState, StructuredReflection,
    ReflectionDecision, ValidationStatus,
    Modality
)
from llm_intelligence import LLMClient

logger = logging.getLogger(__name__)


class LLMReflector:
    """
    LLM驱动的反思引擎

    不再是简单的规则检查，而是真正的深度反思
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.rule_reflector = RuleBasedReflector()

    def reflect(self,
                step_number: int,
                purpose: str,
                expected_result: str,
                actual_result: Dict[str, Any],
                state: AnalysisState,
                use_llm: bool = True) -> StructuredReflection:
        """
        对执行结果进行结构化反思
        """
        # 快速路径：空结果或失败
        if not actual_result.get('success') or not actual_result.get('data'):
            return self.rule_reflector.quick_reflect(
                step_number, purpose, expected_result, actual_result
            )

        # LLM深度反思
        if use_llm:
            try:
                return self._llm_reflect(
                    step_number, purpose, expected_result, actual_result, state
                )
            except Exception as e:
                logger.warning(f"LLM reflection failed: {e}")

        # Fallback
        return self.rule_reflector.reflect(
            step_number, purpose, expected_result, actual_result
        )

    def _llm_reflect(self,
                     step_number: int,
                     purpose: str,
                     expected_result: str,
                     actual_result: Dict,
                     state: AnalysisState) -> StructuredReflection:
        """使用LLM进行深度反思"""

        logger.info(f"🤔 LLM Reflection on step {step_number}...")

        data = actual_result.get('data', [])
        data_summary = self._summarize_data(data)

        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(
            step_number, purpose, expected_result,
            data_summary, actual_result, state
        )

        result = self.llm.generate_json(system_prompt, user_prompt)
        reflection = self._parse_reflection(result, step_number)

        logger.info(f"   Decision: {reflection.decision.value}")
        logger.info(f"   Confidence: {reflection.confidence_score:.2f}")

        return reflection

    def _get_system_prompt(self) -> str:
        return """You are a scientific reasoning engine performing metacognitive reflection.

Your task:
1. EVALUATE: Did the step achieve its purpose? Does data match expectations?
2. DIAGNOSE: If issues exist, what are likely causes?
3. DISCOVER: What new insights or unexpected findings emerged?
4. DECIDE: What should happen next?

DECISION OPTIONS:
- continue: Step succeeded, proceed with plan
- deepen: Results are interesting, worth exploring further
- pivot: Results suggest different direction would be more valuable
- replan: Step failed or unexpected results, need new approach
- terminate: Analysis complete, no more steps needed

Be scientifically rigorous but also creative in identifying opportunities."""

    def _build_prompt(self,
                      step_number: int,
                      purpose: str,
                      expected_result: str,
                      data_summary: Dict,
                      actual_result: Dict,
                      state: AnalysisState) -> str:

        progress = state.get_progress_summary()

        return f"""Reflect on this analysis step:

**Step {step_number}: {purpose}**

**Expected Result:** {expected_result}

**Actual Result Summary:**
- Success: {actual_result.get('success')}
- Row count: {data_summary['row_count']}
- Columns: {', '.join(data_summary['columns'][:10])}
- Sample: {json.dumps(data_summary['sample'][:3], indent=2, default=str)}

**Numeric Stats:** {json.dumps(data_summary.get('numeric_stats', {}), indent=2)}

**Progress:**
- Steps completed: {progress['steps_executed']}
- Modalities: {progress['modalities_covered']}
- Entities: {progress['entities_found']}
- Confidence: {progress['evidence_confidence']:.2f}
- Depth: {progress['target_depth']}

**Question:** {progress['question']}

Return JSON:
{{
    "validation_status": "passed|partial|failed|empty|unexpected",
    "validation_reasoning": "Why",

    "key_findings": ["Finding 1", "Finding 2"],
    "surprising_results": ["Surprise 1"] or [],

    "uncertainty_level": 0.0-1.0,
    "uncertainty_sources": ["Source 1"],

    "decision": "continue|deepen|pivot|replan|terminate",
    "decision_reasoning": "Why this decision",

    "next_step_suggestions": ["Suggestion 1"],
    "alternative_approaches": ["Alternative 1"] or [],

    "confidence_score": 0.0-1.0,
    "confidence_factors": {{"data_quality": 0.8, "expectation_match": 0.9}},

    "summary": "One paragraph summary"
}}"""

    def _summarize_data(self, data: List[Dict]) -> Dict:
        """生成数据摘要"""
        if not data:
            return {'row_count': 0, 'columns': [], 'sample': [], 'numeric_stats': {}}

        columns = list(data[0].keys()) if data else []
        sample = data[:5]

        # 数值统计
        numeric_stats = {}
        for col in columns:
            values = [row.get(col) for row in data
                      if isinstance(row.get(col), (int, float))]
            if values and len(values) >= 2:
                numeric_stats[col] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'count': len(values)
                }

        return {
            'row_count': len(data),
            'columns': columns,
            'sample': sample,
            'numeric_stats': numeric_stats
        }

    def _parse_reflection(self, result: Dict, step_number: int) -> StructuredReflection:
        """解析LLM反思结果"""

        # Validation status
        status_map = {
            'passed': ValidationStatus.PASSED,
            'partial': ValidationStatus.PARTIAL,
            'failed': ValidationStatus.FAILED,
            'empty': ValidationStatus.EMPTY,
            'unexpected': ValidationStatus.UNEXPECTED
        }
        validation_status = status_map.get(
            result.get('validation_status', 'passed'),
            ValidationStatus.PASSED
        )

        # Decision
        decision_map = {
            'continue': ReflectionDecision.CONTINUE,
            'deepen': ReflectionDecision.DEEPEN,
            'pivot': ReflectionDecision.PIVOT,
            'replan': ReflectionDecision.REPLAN,
            'terminate': ReflectionDecision.TERMINATE
        }
        decision = decision_map.get(
            result.get('decision', 'continue'),
            ReflectionDecision.CONTINUE
        )

        return StructuredReflection(
            step_number=step_number,
            validation_status=validation_status,
            validation_reasoning=result.get('validation_reasoning', ''),
            uncertainty_level=result.get('uncertainty_level', 0.5),
            uncertainty_sources=result.get('uncertainty_sources', []),
            key_findings=result.get('key_findings', []),
            surprising_results=result.get('surprising_results', []),
            decision=decision,
            decision_reasoning=result.get('decision_reasoning', ''),
            next_step_suggestions=result.get('next_step_suggestions', []),
            alternative_approaches=result.get('alternative_approaches', []),
            confidence_score=result.get('confidence_score', 0.5),
            confidence_factors=result.get('confidence_factors', {}),
            summary=result.get('summary', 'Reflection completed.')
        )


class RuleBasedReflector:
    """规则反思器 - 快速路径"""

    def quick_reflect(self,
                      step_number: int,
                      purpose: str,
                      expected_result: str,
                      actual_result: Dict) -> StructuredReflection:
        """快速反思（失败/空结果）"""

        if not actual_result.get('success'):
            return StructuredReflection(
                step_number=step_number,
                validation_status=ValidationStatus.FAILED,
                validation_reasoning=f"Query failed: {actual_result.get('error', 'Unknown')}",
                uncertainty_level=1.0,
                uncertainty_sources=["Execution failure"],
                key_findings=[],
                surprising_results=[],
                decision=ReflectionDecision.REPLAN,
                decision_reasoning="Query failed, need to adjust",
                next_step_suggestions=["Check query syntax", "Verify entity names"],
                alternative_approaches=["Try alternative path"],
                confidence_score=0.0,
                confidence_factors={'execution': 0.0},
                summary=f"Step {step_number} failed. Replanning required."
            )

        data = actual_result.get('data', [])
        if not data:
            return StructuredReflection(
                step_number=step_number,
                validation_status=ValidationStatus.EMPTY,
                validation_reasoning="Query returned no results",
                uncertainty_level=0.8,
                uncertainty_sources=["No data"],
                key_findings=[],
                surprising_results=["Expected data not found"],
                decision=ReflectionDecision.REPLAN,
                decision_reasoning="Empty result, entity may not exist",
                next_step_suggestions=["Verify entity", "Relax constraints"],
                alternative_approaches=["Broader search"],
                confidence_score=0.1,
                confidence_factors={'data': 0.0},
                summary=f"Step {step_number} returned no data."
            )

        return self.reflect(step_number, purpose, expected_result, actual_result)

    def reflect(self,
                step_number: int,
                purpose: str,
                expected_result: str,
                actual_result: Dict) -> StructuredReflection:
        """标准规则反思"""

        data = actual_result.get('data', [])
        row_count = len(data)

        if row_count >= 10:
            validation_status = ValidationStatus.PASSED
            decision = ReflectionDecision.CONTINUE
            confidence = 0.8
        elif row_count >= 3:
            validation_status = ValidationStatus.PARTIAL
            decision = ReflectionDecision.CONTINUE
            confidence = 0.6
        else:
            validation_status = ValidationStatus.PARTIAL
            decision = ReflectionDecision.CONTINUE
            confidence = 0.4

        return StructuredReflection(
            step_number=step_number,
            validation_status=validation_status,
            validation_reasoning=f"Retrieved {row_count} rows",
            uncertainty_level=1.0 - confidence,
            uncertainty_sources=["Rule-based"],
            key_findings=[f"Found {row_count} results"],
            surprising_results=[],
            decision=decision,
            decision_reasoning="Rule-based continuation",
            next_step_suggestions=["Proceed with plan"],
            alternative_approaches=[],
            confidence_score=confidence,
            confidence_factors={'row_count': min(1.0, row_count / 20)},
            summary=f"Step {step_number}: {row_count} results (rule-based)"
        )


class ReflectionAggregator:
    """反思聚合器 - 综合多个步骤的反思"""

    def aggregate(self, reflections: List[StructuredReflection]) -> Dict:
        """聚合多个反思结果"""
        if not reflections:
            return {
                'overall_confidence': 0.0,
                'overall_decision': ReflectionDecision.TERMINATE,
                'key_issues': [],
                'major_findings': []
            }

        # 平均置信度
        avg_confidence = sum(r.confidence_score for r in reflections) / len(reflections)

        # 收集发现
        all_findings = []
        for r in reflections:
            all_findings.extend(r.key_findings)

        # 收集问题
        issues = []
        for r in reflections:
            if r.validation_status in [ValidationStatus.FAILED, ValidationStatus.EMPTY]:
                issues.append(f"Step {r.step_number}: {r.validation_reasoning}")

        # 决定方向
        decisions = [r.decision for r in reflections]
        if ReflectionDecision.TERMINATE in decisions:
            overall_decision = ReflectionDecision.TERMINATE
        elif decisions.count(ReflectionDecision.REPLAN) > len(decisions) / 2:
            overall_decision = ReflectionDecision.REPLAN
        else:
            overall_decision = ReflectionDecision.CONTINUE

        return {
            'overall_confidence': avg_confidence,
            'overall_decision': overall_decision,
            'key_issues': issues,
            'major_findings': all_findings[:10],
            'reflections_count': len(reflections),
            'passed_count': sum(1 for r in reflections
                                if r.validation_status == ValidationStatus.PASSED)
        }


# ==================== Export ====================

__all__ = [
    'LLMReflector',
    'RuleBasedReflector',
    'ReflectionAggregator',
]