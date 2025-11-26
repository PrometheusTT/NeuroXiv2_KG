"""
Intelligent Termination Decider
================================
使用LLM推理来决定何时终止分析

对齐Figure 2C的"schema-adaptive, evidence-seeking agent"核心特性

关键功能：
1. 基于证据的终止决策
2. 目标完成度评估
3. 多模态覆盖检查
4. 闭环验证

Author: Claude & Lijun
Date: 2025-01-15
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from core_structures import (
    AnalysisState,
    AnalysisDepth,
    Modality,
    QuestionIntent,
    EvidenceBuffer
)

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """终止原因"""
    QUESTION_ANSWERED = "question_answered"
    MODALITIES_COMPLETE = "modalities_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_MORE_STEPS = "no_more_steps"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    CLOSED_LOOP_ACHIEVED = "closed_loop_achieved"
    LLM_DECISION = "llm_decision"


@dataclass
class TerminationDecision:
    """终止决策结果"""
    should_terminate: bool
    reason: TerminationReason
    confidence: float
    reasoning: str
    missing_elements: List[str]  # 如果不终止，还缺什么


class IntelligentTerminator:
    """
    智能终止决策器

    核心改进：
    1. 不只是数步数
    2. 评估问题是否真正被回答
    3. 检查多模态覆盖
    4. 验证闭环是否完成
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model

        # 各深度的期望
        self.depth_expectations = {
            AnalysisDepth.SHALLOW: {
                'min_steps': 1,
                'max_steps': 2,
                'required_modalities': 1,
                'require_closed_loop': False
            },
            AnalysisDepth.MEDIUM: {
                'min_steps': 2,
                'max_steps': 4,
                'required_modalities': 2,
                'require_closed_loop': False
            },
            AnalysisDepth.DEEP: {
                'min_steps': 4,
                'max_steps': 8,
                'required_modalities': 3,
                'require_closed_loop': True
            }
        }

    def should_terminate(self,
                         state: AnalysisState,
                         use_llm: bool = True) -> TerminationDecision:
        """
        决定是否应该终止分析

        Args:
            state: 当前分析状态
            use_llm: 是否使用LLM进行决策

        Returns:
            TerminationDecision
        """
        logger.info("🎯 Evaluating termination...")

        # 1. 硬性检查
        hard_check = self._hard_termination_check(state)
        if hard_check.should_terminate:
            return hard_check

        # 2. 软性检查（规则）
        soft_check = self._soft_termination_check(state)

        # 3. LLM决策（如果启用且规则不确定）
        if use_llm and not soft_check.should_terminate:
            llm_decision = self._llm_termination_check(state)

            # 综合规则和LLM
            if llm_decision.should_terminate and llm_decision.confidence > 0.7:
                return llm_decision
            elif soft_check.confidence > llm_decision.confidence:
                return soft_check
            else:
                return llm_decision

        return soft_check

    def _hard_termination_check(self, state: AnalysisState) -> TerminationDecision:
        """硬性终止检查（必须终止的情况）"""

        # 预算耗尽
        budget = state.check_budget()
        if not budget['can_continue']:
            reasons = []
            if not budget['steps_ok']:
                reasons.append("max steps reached")
            if not budget['cypher_ok']:
                reasons.append("max cypher calls reached")
            if not budget['time_ok']:
                reasons.append("time limit exceeded")

            return TerminationDecision(
                should_terminate=True,
                reason=TerminationReason.BUDGET_EXHAUSTED,
                confidence=1.0,
                reasoning=f"Budget exhausted: {', '.join(reasons)}",
                missing_elements=[]
            )

        return TerminationDecision(
            should_terminate=False,
            reason=TerminationReason.QUESTION_ANSWERED,
            confidence=0.0,
            reasoning="Hard checks passed",
            missing_elements=[]
        )

    def _soft_termination_check(self, state: AnalysisState) -> TerminationDecision:
        """软性终止检查（基于规则）"""

        expectations = self.depth_expectations[state.target_depth]
        current_steps = len(state.executed_steps)

        # 检查项
        checks = {
            'min_steps_reached': current_steps >= expectations['min_steps'],
            'modalities_covered': len(state.modalities_covered) >= expectations['required_modalities'],
            'confidence_ok': state.evidence_buffer.get_overall_confidence() > 0.6
        }

        # 闭环检查（仅DEEP模式）
        if expectations['require_closed_loop']:
            checks['closed_loop'] = self._check_closed_loop(state)

        # 计算完成度
        completion_rate = sum(checks.values()) / len(checks)

        # 检查缺失元素
        missing = []
        if not checks.get('min_steps_reached'):
            missing.append(f"Need at least {expectations['min_steps']} steps")
        if not checks.get('modalities_covered'):
            covered = [m.value for m in state.modalities_covered]
            missing.append(f"Need more modalities (current: {covered})")
        if not checks.get('confidence_ok'):
            missing.append("Confidence too low")
        if not checks.get('closed_loop', True):
            missing.append("Closed loop not achieved")

        # 决策
        if completion_rate >= 0.8 and current_steps >= expectations['min_steps']:
            return TerminationDecision(
                should_terminate=True,
                reason=TerminationReason.MODALITIES_COMPLETE,
                confidence=completion_rate,
                reasoning=f"Completion rate: {completion_rate:.0%}",
                missing_elements=[]
            )

        return TerminationDecision(
            should_terminate=False,
            reason=TerminationReason.QUESTION_ANSWERED,
            confidence=completion_rate,
            reasoning=f"Completion rate: {completion_rate:.0%}, continuing",
            missing_elements=missing
        )

    def _check_closed_loop(self, state: AnalysisState) -> bool:
        """检查是否完成了闭环分析"""

        # 闭环 = 有projection步骤 + 有target分子分析步骤
        has_projection = False
        has_target_molecular = False

        for step in state.executed_steps:
            purpose = step.get('purpose', '').lower()
            modality = step.get('modality', '')

            if 'projection' in purpose or 'target' in purpose:
                has_projection = True

            if 'target' in purpose and ('composition' in purpose or 'molecular' in purpose):
                has_target_molecular = True

            if modality == 'projection':
                has_projection = True

        return has_projection and has_target_molecular

    def _llm_termination_check(self, state: AnalysisState) -> TerminationDecision:
        """使用LLM进行终止决策"""

        prompt = self._build_termination_prompt(state)

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)

            return TerminationDecision(
                should_terminate=result.get('should_terminate', False),
                reason=TerminationReason.LLM_DECISION,
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', 'LLM decision'),
                missing_elements=result.get('missing_elements', [])
            )

        except Exception as e:
            logger.error(f"LLM termination check failed: {e}")
            return TerminationDecision(
                should_terminate=False,
                reason=TerminationReason.LLM_DECISION,
                confidence=0.3,
                reasoning=f"LLM check failed: {e}",
                missing_elements=[]
            )

    def _get_system_prompt(self) -> str:
        return """You are evaluating whether a neuroscience knowledge graph analysis has collected enough evidence to answer the original question.

Consider:
1. Has the question been directly answered?
2. Are there enough modalities covered (molecular, morphological, projection)?
3. Is the confidence in the findings sufficient?
4. For comprehensive questions, is there a closed-loop analysis (source → targets → target composition)?
5. Would additional steps significantly improve the answer quality?

Be efficient - don't recommend more steps if the question can already be answered well."""

    def _build_termination_prompt(self, state: AnalysisState) -> str:
        progress = state.get_progress_summary()

        steps_summary = "\n".join([
            f"  {i + 1}. {s.get('purpose', 'Unknown')} ({s.get('row_count', 0)} results)"
            for i, s in enumerate(state.executed_steps)
        ])

        return f"""Should we terminate or continue analysis?

**Original Question:** {state.question}

**Question Intent:** {progress['intent']}
**Target Depth:** {progress['target_depth']}

**Executed Steps:**
{steps_summary}

**Current Status:**
- Modalities covered: {progress['modalities_covered']}
- Entities found: {progress['entities_found']}
- Evidence confidence: {progress['evidence_confidence']:.2f}
- Has primary focus: {progress['has_primary_focus']}

**Budget:**
- Steps used: {progress['steps_executed']} / {state.budget['max_steps']}
- Replanning count: {progress['replanning_count']}

**Decision Question:**
Given the question and what we've gathered, should we:
1. TERMINATE - We have enough to answer well
2. CONTINUE - More analysis would significantly improve the answer

Return JSON:
{{
    "should_terminate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation",
    "missing_elements": ["Element 1 if continuing", ...] or []
}}"""


class GoalTracker:
    """
    目标追踪器

    追踪问题的子目标完成情况
    """

    def __init__(self):
        self.goals: List[Dict] = []
        self.completed_goals: List[str] = []

    def set_goals_from_classification(self, classification: 'IntentClassification'):
        """从意图分类设置目标"""
        self.goals = []

        # 基于sub_questions设置目标
        for i, sub_q in enumerate(classification.sub_questions):
            self.goals.append({
                'id': f"subq_{i}",
                'description': sub_q,
                'type': 'sub_question',
                'completed': False
            })

        # 基于expected_modalities设置目标
        for modality in classification.expected_modalities:
            self.goals.append({
                'id': f"mod_{modality.value}",
                'description': f"Cover {modality.value} modality",
                'type': 'modality',
                'completed': False
            })

    def mark_completed(self, goal_id: str):
        """标记目标完成"""
        for goal in self.goals:
            if goal['id'] == goal_id:
                goal['completed'] = True
                self.completed_goals.append(goal_id)

    def check_modality_goals(self, covered_modalities: List[Modality]):
        """检查模态目标"""
        for goal in self.goals:
            if goal['type'] == 'modality':
                modality_name = goal['id'].replace('mod_', '')
                if modality_name in [m.value for m in covered_modalities]:
                    goal['completed'] = True

    def get_completion_rate(self) -> float:
        """获取完成率"""
        if not self.goals:
            return 1.0

        completed = sum(1 for g in self.goals if g['completed'])
        return completed / len(self.goals)

    def get_incomplete_goals(self) -> List[Dict]:
        """获取未完成的目标"""
        return [g for g in self.goals if not g['completed']]