"""
LLM-Driven Structured Reflection
=================================
使用LLM进行真正的自我反思，而非纯规则计算

对齐Figure 2C: Reflect phase - "evaluate evidence & decide next step"

核心改进：
1. LLM评估结果质量
2. LLM生成替代假设
3. LLM决定下一步行动
4. 保留规则作为快速路径

Author: Claude & Lijun
Date: 2025-01-15
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core_structures import (
    ValidationStatus,
    ReflectionDecision,
    StructuredReflection,
    EvidenceRecord,
    AnalysisState,
    Modality
)

logger = logging.getLogger(__name__)


class LLMReflector:
    """
    LLM驱动的反思引擎

    关键改进：
    1. 使用LLM进行深度反思
    2. 生成有意义的替代假设
    3. 做出有根据的决策
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model
        self.rule_reflector = RuleBasedReflector()  # 快速路径

    def reflect(self,
                step_number: int,
                purpose: str,
                expected_result: str,
                actual_result: Dict[str, Any],
                state: AnalysisState,
                use_llm: bool = True) -> StructuredReflection:
        """
        对执行结果进行结构化反思

        Args:
            step_number: 步骤编号
            purpose: 步骤目的
            expected_result: 预期结果描述
            actual_result: 实际执行结果
            state: 当前分析状态
            use_llm: 是否使用LLM (False时用规则快速路径)
        """

        # 快速路径：空结果或失败时用规则
        if not actual_result.get('success') or not actual_result.get('data'):
            return self.rule_reflector.quick_reflect(
                step_number, purpose, expected_result, actual_result
            )

        # 正常路径：使用LLM深度反思
        if use_llm:
            return self._llm_reflect(
                step_number, purpose, expected_result, actual_result, state
            )
        else:
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

        # 准备数据摘要
        data = actual_result.get('data', [])
        data_summary = self._summarize_data(data)

        prompt = self._build_reflection_prompt(
            step_number, purpose, expected_result,
            data_summary, actual_result, state
        )

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1200
            )

            result = json.loads(response.choices[0].message.content)
            reflection = self._parse_reflection(result, step_number)

            logger.info(f"   Decision: {reflection.decision.value}")
            logger.info(f"   Confidence: {reflection.confidence_score:.2f}")
            logger.info(f"   Summary: {reflection.summary[:80]}...")

            # 记录LLM调用
            state.increment_budget('llm')

            return reflection

        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            return self.rule_reflector.reflect(
                step_number, purpose, expected_result, actual_result
            )

    def _get_system_prompt(self) -> str:
        return """You are a scientific reasoning engine performing metacognitive reflection on neuroscience analysis steps.

Your task is to:
1. EVALUATE: Did the step achieve its purpose? Does the data match expectations?
2. DIAGNOSE: If there are issues, what are the likely causes?
3. DISCOVER: What new insights or unexpected findings emerged?
4. DECIDE: What should happen next?

DECISION OPTIONS:
- continue: The step succeeded, proceed with the plan
- replan: The step failed or produced unexpected results, need new approach
- deepen: Results are interesting, worth exploring further
- pivot: Results suggest a different direction would be more valuable
- terminate: Analysis is complete, no more steps needed

Be scientifically rigorous but also creative in identifying opportunities."""

    def _build_reflection_prompt(self,
                                 step_number: int,
                                 purpose: str,
                                 expected_result: str,
                                 data_summary: Dict,
                                 actual_result: Dict,
                                 state: AnalysisState) -> str:

        progress = state.get_progress_summary()

        return f"""Reflect on this analysis step:

**Step {step_number}: {purpose}**

**Expected Result:** 
{expected_result}

**Actual Result Summary:**
- Success: {actual_result.get('success')}
- Row count: {data_summary['row_count']}
- Columns: {', '.join(data_summary['columns'][:10])}
- Sample data: {json.dumps(data_summary['sample'][:3], indent=2, default=str)}

**Numeric Statistics (if applicable):**
{json.dumps(data_summary.get('numeric_stats', {}), indent=2)}

**Current Analysis Progress:**
- Steps completed: {progress['steps_executed']}
- Modalities covered: {progress['modalities_covered']}
- Entities found: {progress['entities_found']}
- Overall confidence: {progress['evidence_confidence']:.2f}
- Target depth: {progress['target_depth']}

**Original Question:** {progress['question']}

---

Now reflect deeply:

1. **Validation**: Does the data match what was expected? Are there anomalies?

2. **Key Findings**: What are the 2-3 most important discoveries from this step?

3. **Surprises**: Anything unexpected that warrants attention?

4. **Uncertainty**: What are the sources of uncertainty in these results?

5. **Decision**: What should happen next?
   - If data matches expectations and analysis is progressing → continue
   - If data is empty or wrong → replan
   - If interesting patterns emerged → deepen
   - If results suggest different direction → pivot
   - If question is fully answered → terminate

6. **Next Steps**: If continuing, what specific steps would be most valuable?

7. **Alternatives**: If this approach isn't working, what alternatives exist?

Return JSON:
{{
    "validation_status": "passed|partial|failed|empty|unexpected",
    "validation_reasoning": "Why you judged it this way",

    "key_findings": ["Finding 1", "Finding 2", ...],
    "surprising_results": ["Surprise 1", ...] or [],

    "uncertainty_level": 0.0-1.0,
    "uncertainty_sources": ["Source 1", "Source 2", ...],

    "decision": "continue|replan|deepen|pivot|terminate",
    "decision_reasoning": "Detailed explanation of why this decision",

    "next_step_suggestions": ["Suggestion 1", "Suggestion 2", ...],
    "alternative_approaches": ["Alternative 1", ...] or [],

    "confidence_score": 0.0-1.0,
    "confidence_factors": {{
        "data_quality": 0.0-1.0,
        "expectation_match": 0.0-1.0,
        "scientific_validity": 0.0-1.0
    }},

    "summary": "One paragraph summary of this reflection"
}}"""

    def _summarize_data(self, data: List[Dict]) -> Dict:
        """生成数据摘要"""
        if not data:
            return {
                'row_count': 0,
                'columns': [],
                'sample': [],
                'numeric_stats': {}
            }

        columns = list(data[0].keys()) if data else []
        sample = data[:5]

        # 数值统计
        numeric_stats = {}
        for col in columns:
            values = [row.get(col) for row in data if isinstance(row.get(col), (int, float))]
            if values and len(values) >= 2:
                import statistics
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
        """解析LLM返回的反思结果"""

        # Validation status
        status_str = result.get('validation_status', 'passed')
        status_map = {
            'passed': ValidationStatus.PASSED,
            'partial': ValidationStatus.PARTIAL,
            'failed': ValidationStatus.FAILED,
            'empty': ValidationStatus.EMPTY,
            'unexpected': ValidationStatus.UNEXPECTED
        }
        validation_status = status_map.get(status_str, ValidationStatus.PASSED)

        # Decision
        decision_str = result.get('decision', 'continue')
        decision_map = {
            'continue': ReflectionDecision.CONTINUE,
            'replan': ReflectionDecision.REPLAN,
            'deepen': ReflectionDecision.DEEPEN,
            'pivot': ReflectionDecision.PIVOT,
            'terminate': ReflectionDecision.TERMINATE
        }
        decision = decision_map.get(decision_str, ReflectionDecision.CONTINUE)

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
    """
    规则反思器 - 快速路径

    用于：
    - 空结果
    - 执行失败
    - 简单验证
    """

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
                validation_reasoning=f"Query execution failed: {actual_result.get('error', 'Unknown')}",
                uncertainty_level=1.0,
                uncertainty_sources=["Execution failure"],
                key_findings=[],
                surprising_results=[],
                decision=ReflectionDecision.REPLAN,
                decision_reasoning="Query failed, need to adjust approach",
                next_step_suggestions=[
                    "Check query syntax",
                    "Verify entity names exist",
                    "Try alternative schema path"
                ],
                alternative_approaches=["Use different relationship", "Relax constraints"],
                confidence_score=0.0,
                confidence_factors={'execution': 0.0},
                summary=f"Step {step_number} failed to execute. Replanning required."
            )

        data = actual_result.get('data', [])
        if not data:
            return StructuredReflection(
                step_number=step_number,
                validation_status=ValidationStatus.EMPTY,
                validation_reasoning="Query returned no results",
                uncertainty_level=0.8,
                uncertainty_sources=["No data available"],
                key_findings=[],
                surprising_results=["Expected data not found"],
                decision=ReflectionDecision.REPLAN,
                decision_reasoning="Empty result suggests entity may not exist or query is too restrictive",
                next_step_suggestions=[
                    "Verify entity exists in database",
                    "Relax query constraints",
                    "Try alternative entity names"
                ],
                alternative_approaches=["Broader search", "Check spelling variations"],
                confidence_score=0.1,
                confidence_factors={'data_availability': 0.0},
                summary=f"Step {step_number} returned no data. Consider relaxing constraints."
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

        # 简单启发式
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
            validation_reasoning=f"Retrieved {row_count} rows (rule-based evaluation)",
            uncertainty_level=1.0 - confidence,
            uncertainty_sources=["Rule-based assessment"],
            key_findings=[f"Found {row_count} results"],
            surprising_results=[],
            decision=decision,
            decision_reasoning="Rule-based continuation",
            next_step_suggestions=["Proceed with plan"],
            alternative_approaches=[],
            confidence_score=confidence,
            confidence_factors={'row_count': min(1.0, row_count / 20)},
            summary=f"Step {step_number} completed with {row_count} results (rule-based reflection)"
        )


class ReflectionAggregator:
    """
    反思聚合器

    综合多个步骤的反思，生成整体分析评估
    """

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

        # 收集所有发现
        all_findings = []
        for r in reflections:
            all_findings.extend(r.key_findings)

        # 收集所有问题
        issues = []
        for r in reflections:
            if r.validation_status in [ValidationStatus.FAILED, ValidationStatus.EMPTY]:
                issues.append(f"Step {r.step_number}: {r.validation_reasoning}")

        # 决定整体方向
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
            'major_findings': all_findings[:10],  # Top 10
            'reflections_count': len(reflections),
            'passed_count': sum(1 for r in reflections if r.validation_status == ValidationStatus.PASSED)
        }