"""
Self-evaluation and correction mechanisms for enhanced reasoning.
Provides error detection, confidence calibration, and adaptive correction capabilities.
"""

import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .reasoning_engine import ReasoningState, Thought, Action, Observation, Reflection

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of reasoning errors that can be detected."""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OVERCONFIDENCE = "overconfidence"
    CIRCULAR_REASONING = "circular_reasoning"
    PREMATURE_CONCLUSION = "premature_conclusion"
    TOOL_MISUSE = "tool_misuse"
    INFORMATION_OVERLOAD = "information_overload"
    SCOPE_DRIFT = "scope_drift"


class CorrectionStrategy(Enum):
    """Strategies for correcting reasoning errors."""
    BACKTRACK = "backtrack"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    REDUCE_CONFIDENCE = "reduce_confidence"
    REFOCUS_SCOPE = "refocus_scope"
    CHANGE_APPROACH = "change_approach"
    SEEK_ALTERNATIVE = "seek_alternative"
    DECOMPOSE_PROBLEM = "decompose_problem"


@dataclass
class ReasoningError:
    """Represents a detected reasoning error."""
    error_type: ErrorType
    description: str
    severity: float  # 0.0 to 1.0
    step_id: str
    evidence: List[str] = field(default_factory=list)
    suggested_correction: Optional[CorrectionStrategy] = None
    confidence: float = 0.0


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for reasoning assessment."""
    logical_consistency: float = 0.0
    evidence_strength: float = 0.0
    reasoning_depth: float = 0.0
    coherence_score: float = 0.0
    confidence_calibration: float = 0.0
    goal_alignment: float = 0.0
    efficiency_score: float = 0.0
    overall_quality: float = 0.0


@dataclass
class CorrectionAction:
    """Represents a correction action to be taken."""
    strategy: CorrectionStrategy
    description: str
    target_step: Optional[int] = None
    new_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


class SelfEvaluationEngine:
    """
    Self-evaluation engine that monitors reasoning quality and suggests corrections.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.error_patterns = self._initialize_error_patterns()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.correction_history = []

    def evaluate_reasoning_state(self, state: ReasoningState, question: str) -> Tuple[QualityMetrics, List[ReasoningError]]:
        """
        Comprehensive evaluation of current reasoning state.
        """
        # Compute quality metrics
        metrics = self._compute_quality_metrics(state, question)

        # Detect reasoning errors
        errors = self._detect_reasoning_errors(state, question, metrics)

        # Log evaluation results
        logger.info(f"Reasoning evaluation - Quality: {metrics.overall_quality:.3f}, Errors: {len(errors)}")

        return metrics, errors

    def suggest_corrections(self, errors: List[ReasoningError], state: ReasoningState) -> List[CorrectionAction]:
        """
        Suggest correction actions based on detected errors.
        """
        corrections = []

        for error in errors:
            correction = self._generate_correction_for_error(error, state)
            if correction:
                corrections.append(correction)

        # Sort corrections by priority and expected improvement
        corrections.sort(key=lambda c: (c.priority, -c.expected_improvement))

        return corrections

    def calibrate_confidence(self, state: ReasoningState, metrics: QualityMetrics) -> float:
        """
        Calibrate confidence based on reasoning quality and consistency.
        """
        base_confidence = state.confidence_score

        # Adjust based on quality metrics
        quality_adjustment = (metrics.overall_quality - 0.5) * 0.3
        consistency_adjustment = (metrics.logical_consistency - 0.5) * 0.2
        evidence_adjustment = (metrics.evidence_strength - 0.5) * 0.2

        calibrated_confidence = base_confidence + quality_adjustment + consistency_adjustment + evidence_adjustment

        # Apply bounds
        calibrated_confidence = max(0.1, min(0.95, calibrated_confidence))

        return calibrated_confidence

    def monitor_reasoning_progress(self, state: ReasoningState, question: str) -> Dict[str, Any]:
        """
        Monitor reasoning progress and provide ongoing guidance.
        """
        metrics, errors = self.evaluate_reasoning_state(state, question)
        corrections = self.suggest_corrections(errors, state)

        progress_assessment = {
            "current_quality": metrics.overall_quality,
            "trajectory": self._assess_quality_trajectory(state),
            "bottlenecks": self._identify_bottlenecks(state, metrics),
            "efficiency": metrics.efficiency_score,
            "next_actions": corrections[:3],  # Top 3 corrections
            "should_continue": self._should_continue_reasoning(metrics, state),
            "estimated_completion": self._estimate_completion_progress(state, question)
        }

        return progress_assessment

    def _compute_quality_metrics(self, state: ReasoningState, question: str) -> QualityMetrics:
        """Compute comprehensive quality metrics."""

        # Logical consistency
        logical_consistency = self._assess_logical_consistency(state)

        # Evidence strength
        evidence_strength = self._assess_evidence_strength(state)

        # Reasoning depth
        reasoning_depth = self._assess_reasoning_depth(state)

        # Coherence score
        coherence_score = self._assess_coherence(state)

        # Confidence calibration
        confidence_calibration = self._assess_confidence_calibration(state)

        # Goal alignment
        goal_alignment = self._assess_goal_alignment(state, question)

        # Efficiency score
        efficiency_score = self._assess_efficiency(state)

        # Overall quality (weighted combination)
        overall_quality = (
            logical_consistency * 0.25 +
            evidence_strength * 0.20 +
            reasoning_depth * 0.15 +
            coherence_score * 0.15 +
            confidence_calibration * 0.10 +
            goal_alignment * 0.10 +
            efficiency_score * 0.05
        )

        return QualityMetrics(
            logical_consistency=logical_consistency,
            evidence_strength=evidence_strength,
            reasoning_depth=reasoning_depth,
            coherence_score=coherence_score,
            confidence_calibration=confidence_calibration,
            goal_alignment=goal_alignment,
            efficiency_score=efficiency_score,
            overall_quality=overall_quality
        )

    def _detect_reasoning_errors(self, state: ReasoningState, question: str, metrics: QualityMetrics) -> List[ReasoningError]:
        """Detect various types of reasoning errors."""
        errors = []

        # Detect logical inconsistencies
        if metrics.logical_consistency < 0.3:
            errors.append(ReasoningError(
                error_type=ErrorType.LOGICAL_INCONSISTENCY,
                description="Logical inconsistencies detected in reasoning chain",
                severity=1.0 - metrics.logical_consistency,
                step_id="multiple",
                suggested_correction=CorrectionStrategy.BACKTRACK
            ))

        # Detect insufficient evidence
        if metrics.evidence_strength < 0.4 and len(state.observations) < 2:
            errors.append(ReasoningError(
                error_type=ErrorType.INSUFFICIENT_EVIDENCE,
                description="Insufficient evidence to support conclusions",
                severity=0.6,
                step_id="evidence_gathering",
                suggested_correction=CorrectionStrategy.GATHER_MORE_EVIDENCE
            ))

        # Detect overconfidence
        confidence_vs_quality_gap = state.confidence_score - metrics.overall_quality
        if confidence_vs_quality_gap > 0.3:
            errors.append(ReasoningError(
                error_type=ErrorType.OVERCONFIDENCE,
                description=f"Confidence ({state.confidence_score:.2f}) significantly exceeds reasoning quality ({metrics.overall_quality:.2f})",
                severity=confidence_vs_quality_gap,
                step_id="confidence_calibration",
                suggested_correction=CorrectionStrategy.REDUCE_CONFIDENCE
            ))

        # Detect circular reasoning
        if self._detect_circular_reasoning(state):
            errors.append(ReasoningError(
                error_type=ErrorType.CIRCULAR_REASONING,
                description="Circular reasoning detected in thought progression",
                severity=0.7,
                step_id="reasoning_loop",
                suggested_correction=CorrectionStrategy.CHANGE_APPROACH
            ))

        # Detect scope drift
        if self._detect_scope_drift(state, question):
            errors.append(ReasoningError(
                error_type=ErrorType.SCOPE_DRIFT,
                description="Reasoning has drifted from original question scope",
                severity=0.5,
                step_id="scope_management",
                suggested_correction=CorrectionStrategy.REFOCUS_SCOPE
            ))

        return errors

    def _assess_logical_consistency(self, state: ReasoningState) -> float:
        """Assess logical consistency of reasoning chain."""
        if len(state.thoughts) < 2:
            return 0.5

        consistency_scores = []

        # Check for contradictions between thoughts
        for i in range(len(state.thoughts) - 1):
            current_thought = state.thoughts[i].content.lower()
            next_thought = state.thoughts[i + 1].content.lower()

            # Simple contradiction detection
            contradiction_indicators = [
                ("is" in current_thought and "is not" in next_thought),
                ("should" in current_thought and "should not" in next_thought),
                ("will" in current_thought and "will not" in next_thought)
            ]

            if any(contradiction_indicators):
                consistency_scores.append(0.2)
            else:
                consistency_scores.append(0.8)

        return statistics.mean(consistency_scores) if consistency_scores else 0.5

    def _assess_evidence_strength(self, state: ReasoningState) -> float:
        """Assess strength of evidence supporting reasoning."""
        if not state.observations:
            return 0.1

        evidence_quality = []
        for obs in state.observations:
            if obs.success:
                # Quality based on insights generated
                insight_score = min(len(obs.insights) / 3.0, 1.0)
                surprise_penalty = min(len(obs.surprises) / 5.0, 0.3)
                quality = max(0.1, insight_score - surprise_penalty)
                evidence_quality.append(quality)
            else:
                evidence_quality.append(0.1)

        return statistics.mean(evidence_quality)

    def _assess_reasoning_depth(self, state: ReasoningState) -> float:
        """Assess depth of reasoning process."""
        depth_indicators = {
            "num_steps": min(state.current_step / 5.0, 1.0),
            "thought_complexity": self._assess_thought_complexity(state.thoughts),
            "reflection_depth": min(len(state.reflections) / 3.0, 1.0),
            "hypothesis_refinement": len(state.working_hypotheses) > 0
        }

        return sum(depth_indicators.values()) / len(depth_indicators)

    def _assess_coherence(self, state: ReasoningState) -> float:
        """Assess coherence of reasoning flow."""
        if len(state.thoughts) < 2:
            return 0.5

        coherence_indicators = []

        for i in range(len(state.thoughts) - 1):
            current = state.thoughts[i].content.lower()
            next_thought = state.thoughts[i + 1].content.lower()

            # Check for logical connectors
            connectors = ["therefore", "thus", "however", "furthermore", "because", "since", "consequently"]
            has_connector = any(connector in next_thought for connector in connectors)

            # Check for topic continuity
            current_words = set(current.split())
            next_words = set(next_thought.split())
            word_overlap = len(current_words & next_words) / len(current_words | next_words)

            coherence_score = (0.6 if has_connector else 0.3) + (word_overlap * 0.4)
            coherence_indicators.append(min(coherence_score, 1.0))

        return statistics.mean(coherence_indicators)

    def _assess_confidence_calibration(self, state: ReasoningState) -> float:
        """Assess how well calibrated the confidence scores are."""
        if not state.thoughts:
            return 0.5

        # Check consistency of confidence across thoughts
        confidences = [t.confidence for t in state.thoughts if hasattr(t, 'confidence')]
        if not confidences:
            return 0.5

        # Well-calibrated confidence should not vary too wildly without good reason
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0
        calibration_score = max(0.1, 1.0 - (confidence_variance * 2))

        return calibration_score

    def _assess_goal_alignment(self, state: ReasoningState, question: str) -> float:
        """Assess how well reasoning aligns with original question."""
        question_keywords = set(question.lower().split())

        alignment_scores = []
        for thought in state.thoughts:
            thought_keywords = set(thought.content.lower().split())
            overlap = len(question_keywords & thought_keywords) / len(question_keywords)
            alignment_scores.append(overlap)

        return statistics.mean(alignment_scores) if alignment_scores else 0.3

    def _assess_efficiency(self, state: ReasoningState) -> float:
        """Assess efficiency of reasoning process."""
        if state.current_step == 0:
            return 1.0

        # Efficiency based on insight generation per step
        total_insights = sum(len(obs.insights) for obs in state.observations)
        insights_per_step = total_insights / state.current_step

        # Normalize to 0-1 scale
        efficiency = min(insights_per_step / 2.0, 1.0)

        # Penalize excessive steps with low progress
        if state.current_step > 8 and total_insights < 3:
            efficiency *= 0.5

        return efficiency

    def _assess_thought_complexity(self, thoughts: List[Thought]) -> float:
        """Assess complexity and sophistication of thoughts."""
        if not thoughts:
            return 0.0

        complexity_indicators = []
        for thought in thoughts:
            content = thought.content

            # Length-based complexity
            length_score = min(len(content) / 200.0, 1.0)

            # Vocabulary sophistication
            sophisticated_words = [
                "hypothesis", "evidence", "conclusion", "analysis", "correlation",
                "implication", "methodology", "systematic", "comprehensive"
            ]
            vocab_score = sum(1 for word in sophisticated_words if word in content.lower()) / len(sophisticated_words)

            complexity = (length_score + vocab_score) / 2
            complexity_indicators.append(complexity)

        return statistics.mean(complexity_indicators)

    def _detect_circular_reasoning(self, state: ReasoningState) -> bool:
        """Detect if reasoning has become circular."""
        if len(state.thoughts) < 3:
            return False

        # Check for repeated concepts in recent thoughts
        recent_thoughts = state.thoughts[-4:]
        thought_contents = [t.content.lower() for t in recent_thoughts]

        # Simple circular detection: same key concepts repeated
        key_concepts = []
        for content in thought_contents:
            words = content.split()
            key_words = [w for w in words if len(w) > 5 and w.isalpha()]
            key_concepts.extend(key_words[:3])  # Top 3 key words per thought

        # If more than 50% of concepts are repeated, it might be circular
        unique_concepts = len(set(key_concepts))
        total_concepts = len(key_concepts)

        return (total_concepts - unique_concepts) / total_concepts > 0.5 if total_concepts > 0 else False

    def _detect_scope_drift(self, state: ReasoningState, question: str) -> bool:
        """Detect if reasoning has drifted from original question scope."""
        if not state.thoughts:
            return False

        question_keywords = set(question.lower().split())

        # Check alignment of recent thoughts with original question
        recent_thoughts = state.thoughts[-3:]
        alignment_scores = []

        for thought in recent_thoughts:
            thought_keywords = set(thought.content.lower().split())
            alignment = len(question_keywords & thought_keywords) / len(question_keywords)
            alignment_scores.append(alignment)

        avg_alignment = statistics.mean(alignment_scores)
        return avg_alignment < 0.2  # Significant drift threshold

    def _generate_correction_for_error(self, error: ReasoningError, state: ReasoningState) -> Optional[CorrectionAction]:
        """Generate specific correction action for a detected error."""

        strategy_map = {
            ErrorType.LOGICAL_INCONSISTENCY: self._create_backtrack_correction,
            ErrorType.INSUFFICIENT_EVIDENCE: self._create_evidence_gathering_correction,
            ErrorType.OVERCONFIDENCE: self._create_confidence_reduction_correction,
            ErrorType.CIRCULAR_REASONING: self._create_approach_change_correction,
            ErrorType.SCOPE_DRIFT: self._create_refocus_correction
        }

        correction_creator = strategy_map.get(error.error_type)
        if correction_creator:
            return correction_creator(error, state)

        return None

    def _create_backtrack_correction(self, error: ReasoningError, state: ReasoningState) -> CorrectionAction:
        """Create correction action for logical inconsistency."""
        return CorrectionAction(
            strategy=CorrectionStrategy.BACKTRACK,
            description="Backtrack to previous consistent state and reconsider reasoning",
            target_step=max(1, state.current_step - 2),
            expected_improvement=0.6,
            priority=1
        )

    def _create_evidence_gathering_correction(self, error: ReasoningError, state: ReasoningState) -> CorrectionAction:
        """Create correction action for insufficient evidence."""
        return CorrectionAction(
            strategy=CorrectionStrategy.GATHER_MORE_EVIDENCE,
            description="Gather additional evidence before drawing conclusions",
            new_parameters={"min_evidence_threshold": 3},
            expected_improvement=0.7,
            priority=1
        )

    def _create_confidence_reduction_correction(self, error: ReasoningError, state: ReasoningState) -> CorrectionAction:
        """Create correction action for overconfidence."""
        return CorrectionAction(
            strategy=CorrectionStrategy.REDUCE_CONFIDENCE,
            description="Reduce confidence to better match reasoning quality",
            new_parameters={"confidence_adjustment": -0.2},
            expected_improvement=0.4,
            priority=2
        )

    def _create_approach_change_correction(self, error: ReasoningError, state: ReasoningState) -> CorrectionAction:
        """Create correction action for circular reasoning."""
        return CorrectionAction(
            strategy=CorrectionStrategy.CHANGE_APPROACH,
            description="Change reasoning approach to break circular pattern",
            expected_improvement=0.8,
            priority=1
        )

    def _create_refocus_correction(self, error: ReasoningError, state: ReasoningState) -> CorrectionAction:
        """Create correction action for scope drift."""
        return CorrectionAction(
            strategy=CorrectionStrategy.REFOCUS_SCOPE,
            description="Refocus reasoning on original question scope",
            expected_improvement=0.5,
            priority=2
        )

    def _assess_quality_trajectory(self, state: ReasoningState) -> str:
        """Assess the trajectory of reasoning quality over time."""
        if len(state.thoughts) < 3:
            return "insufficient_data"

        # Simple trajectory assessment based on thought progression
        recent_confidences = [t.confidence for t in state.thoughts[-3:] if hasattr(t, 'confidence')]

        if len(recent_confidences) < 2:
            return "stable"

        if recent_confidences[-1] > recent_confidences[0]:
            return "improving"
        elif recent_confidences[-1] < recent_confidences[0]:
            return "declining"
        else:
            return "stable"

    def _identify_bottlenecks(self, state: ReasoningState, metrics: QualityMetrics) -> List[str]:
        """Identify bottlenecks in reasoning process."""
        bottlenecks = []

        if metrics.evidence_strength < 0.4:
            bottlenecks.append("insufficient_evidence_collection")

        if metrics.logical_consistency < 0.5:
            bottlenecks.append("logical_reasoning_quality")

        if metrics.coherence_score < 0.4:
            bottlenecks.append("reasoning_flow_coherence")

        if metrics.efficiency_score < 0.3:
            bottlenecks.append("reasoning_efficiency")

        return bottlenecks

    def _should_continue_reasoning(self, metrics: QualityMetrics, state: ReasoningState) -> bool:
        """Determine if reasoning should continue based on quality metrics."""

        # Continue if quality is improving and not yet sufficient
        if metrics.overall_quality < 0.7 and state.current_step < 8:
            return True

        # Stop if quality is high enough
        if metrics.overall_quality >= 0.8:
            return False

        # Stop if efficiency is very low (spinning wheels)
        if metrics.efficiency_score < 0.2 and state.current_step > 4:
            return False

        return True

    def _estimate_completion_progress(self, state: ReasoningState, question: str) -> float:
        """Estimate how close reasoning is to completion."""

        progress_indicators = {
            "evidence_gathered": min(len(state.observations) / 3.0, 1.0),
            "hypotheses_formed": len(state.working_hypotheses) > 0,
            "reflections_made": len(state.reflections) > 0,
            "conclusions_drawn": any("conclusion" in t.content.lower() for t in state.thoughts[-2:])
        }

        return sum(progress_indicators.values()) / len(progress_indicators)

    def _initialize_error_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for error detection."""
        return {
            "contradiction_words": ["but", "however", "contradicts", "opposite"],
            "uncertainty_words": ["maybe", "perhaps", "possibly", "might"],
            "overconfidence_words": ["definitely", "certainly", "absolutely", "undoubtedly"],
            "circular_indicators": ["as mentioned before", "as we established", "returning to"]
        }

    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds for various metrics."""
        return {
            "logical_consistency_min": 0.6,
            "evidence_strength_min": 0.5,
            "coherence_min": 0.4,
            "overall_quality_target": 0.75,
            "confidence_calibration_min": 0.5
        }