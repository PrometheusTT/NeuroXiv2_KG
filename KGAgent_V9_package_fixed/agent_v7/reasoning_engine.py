"""
Enhanced Chain-of-Thought and Knowledge Graph guided reasoning engine.
Implements think-act-observe-reflect pattern for powerful reasoning capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Thought:
    """Represents a single thought in the reasoning chain."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    type: str = "thought"  # thought, hypothesis, question, insight
    content: str = ""
    confidence: float = 0.5
    reasoning_step: int = 0
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Action:
    """Represents an action taken during reasoning."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_outcome: str = ""
    reasoning_step: int = 0


@dataclass
class Observation:
    """Represents observations from actions or analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action_id: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    insights: List[str] = field(default_factory=list)
    surprises: List[str] = field(default_factory=list)
    reasoning_step: int = 0


@dataclass
class Reflection:
    """Represents reflections on thoughts, actions, and observations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: str = ""
    type: str = "general"  # general, error_analysis, pattern_recognition, meta_cognitive
    confidence_change: float = 0.0
    new_hypotheses: List[str] = field(default_factory=list)
    reasoning_adjustments: List[str] = field(default_factory=list)
    reasoning_step: int = 0


@dataclass
class ReasoningState:
    """Current state of the reasoning process."""
    question: str = ""
    current_step: int = 0
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    reflections: List[Reflection] = field(default_factory=list)
    working_hypotheses: List[str] = field(default_factory=list)
    confirmed_facts: List[str] = field(default_factory=list)
    rejected_hypotheses: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    reasoning_path: List[str] = field(default_factory=list)


class EnhancedCoTReasoningEngine:
    """
    Enhanced Chain-of-Thought reasoning engine that integrates with Knowledge Graph exploration.
    Implements think-act-observe-reflect pattern for sophisticated reasoning.
    """

    def __init__(self, llm_client, db_executor, schema_cache, tool_registry):
        self.llm = llm_client
        self.db = db_executor
        self.schema = schema_cache
        self.tools = tool_registry
        self.reasoning_state = ReasoningState()
        self.memory = ReasoningMemory()

    def reset_state(self, question: str):
        """Reset reasoning state for a new question."""
        self.reasoning_state = ReasoningState(question=question)

    def think(self, context: str = "") -> Thought:
        """
        Generate thoughtful analysis of the current situation.
        This is the 'Think' step in think-act-observe-reflect.
        """
        self.reasoning_state.current_step += 1

        # Construct thinking prompt with full context
        thinking_prompt = self._construct_thinking_prompt(context)

        # Generate thought using LLM
        thought_response = self.llm.run_with_tools(
            system="You are a sophisticated reasoning engine. Think step by step, consider multiple perspectives, and generate insights.",
            user=thinking_prompt,
            tools=[],
            tool_router=lambda n, a: {}
        )

        # Parse the thought response
        thought = self._parse_thought_response(thought_response)
        thought.reasoning_step = self.reasoning_state.current_step

        self.reasoning_state.thoughts.append(thought)
        self.reasoning_state.reasoning_path.append(f"THINK: {thought.content}")

        logger.info(f"Step {self.reasoning_state.current_step} - THINK: {thought.content}")
        return thought

    def act(self, thought: Thought) -> Action:
        """
        Choose and execute an action based on current thought.
        This is the 'Act' step in think-act-observe-reflect.
        """
        # Generate action plan
        action_prompt = self._construct_action_prompt(thought)

        action_response = self.llm.run_with_tools(
            system="You are an action planner. Given the current thought, choose the best action to take next. Return JSON with tool_name, parameters, and rationale.",
            user=action_prompt,
            tools=self.tools,
            tool_router=self._tool_router_wrapper
        )

        # Parse and create action
        action = self._parse_action_response(action_response, thought)
        action.reasoning_step = self.reasoning_state.current_step

        self.reasoning_state.actions.append(action)
        self.reasoning_state.reasoning_path.append(f"ACT: {action.tool_name} - {action.rationale}")

        logger.info(f"Step {self.reasoning_state.current_step} - ACT: {action.tool_name}")
        return action

    def observe(self, action: Action) -> Observation:
        """
        Execute action and observe results.
        This is the 'Observe' step in think-act-observe-reflect.
        """
        # Execute the action using appropriate tool
        try:
            result = self._execute_action(action)
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False

        # Analyze the results for insights and surprises
        observation = Observation(
            action_id=action.id,
            result=result,
            success=success,
            reasoning_step=self.reasoning_state.current_step
        )

        # Generate insights from the observation
        insights, surprises = self._analyze_observation_results(result, action.expected_outcome)
        observation.insights = insights
        observation.surprises = surprises

        self.reasoning_state.observations.append(observation)
        self.reasoning_state.reasoning_path.append(f"OBSERVE: {len(insights)} insights, {len(surprises)} surprises")

        logger.info(f"Step {self.reasoning_state.current_step} - OBSERVE: {success}, {len(insights)} insights")
        return observation

    def reflect(self, observation: Observation) -> Reflection:
        """
        Reflect on observations to generate insights and adjust reasoning.
        This is the 'Reflect' step in think-act-observe-reflect.
        """
        reflection_prompt = self._construct_reflection_prompt(observation)

        reflection_response = self.llm.run_with_tools(
            system="You are a reflective analyst. Analyze the observation deeply, identify patterns, generate new hypotheses, and suggest reasoning adjustments.",
            user=reflection_prompt,
            tools=[],
            tool_router=lambda n, a: {}
        )

        reflection = self._parse_reflection_response(reflection_response)
        reflection.reasoning_step = self.reasoning_state.current_step

        # Update reasoning state based on reflection
        self._update_reasoning_state_from_reflection(reflection)

        self.reasoning_state.reflections.append(reflection)
        self.reasoning_state.reasoning_path.append(f"REFLECT: {reflection.type} - {len(reflection.new_hypotheses)} new hypotheses")

        logger.info(f"Step {self.reasoning_state.current_step} - REFLECT: {reflection.type}")
        return reflection

    def should_continue_reasoning(self) -> Tuple[bool, str]:
        """
        Determine if reasoning should continue or if we have sufficient answer.
        """
        # Check various stopping conditions
        max_steps = 10
        confidence_threshold = 0.8

        if self.reasoning_state.current_step >= max_steps:
            return False, "Maximum reasoning steps reached"

        if self.reasoning_state.confidence_score >= confidence_threshold:
            return False, "High confidence achieved"

        if len(self.reasoning_state.knowledge_gaps) == 0 and len(self.reasoning_state.working_hypotheses) > 0:
            return False, "No significant knowledge gaps remaining"

        # Check for reasoning loops or stagnation
        recent_paths = self.reasoning_state.reasoning_path[-6:]
        if len(set(recent_paths)) < 3 and len(recent_paths) >= 6:
            return False, "Reasoning appears to be in a loop"

        return True, "Continue reasoning"

    def synthesize_answer(self) -> Dict[str, Any]:
        """
        Synthesize final answer from all reasoning steps.
        """
        synthesis_prompt = self._construct_synthesis_prompt()

        final_answer = self.llm.run_with_tools(
            system="You are a synthesis expert. Integrate all reasoning steps, thoughts, observations, and reflections into a comprehensive, well-reasoned answer.",
            user=synthesis_prompt,
            tools=[],
            tool_router=lambda n, a: {}
        )

        return {
            "answer": final_answer,
            "reasoning_trace": self.reasoning_state,
            "confidence": self.reasoning_state.confidence_score,
            "evidence": self._extract_evidence(),
            "methodology": self._extract_methodology(),
            "limitations": self._extract_limitations()
        }

    def reason_through_question(self, question: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Main reasoning loop: think-act-observe-reflect until answer is found.
        """
        self.reset_state(question)

        logger.info(f"Starting reasoning for: {question}")

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # THINK: Generate thoughtful analysis
            context = self._build_context_for_thinking()
            thought = self.think(context)

            # ACT: Choose and plan action
            action = self.act(thought)

            # OBSERVE: Execute and collect results
            observation = self.observe(action)

            # REFLECT: Analyze and adjust
            reflection = self.reflect(observation)

            # Check if we should continue
            should_continue, reason = self.should_continue_reasoning()
            if not should_continue:
                logger.info(f"Stopping reasoning: {reason}")
                break

        # Synthesize final answer
        return self.synthesize_answer()

    def _construct_thinking_prompt(self, context: str) -> str:
        """Construct prompt for thinking step."""
        return f"""
Current Question: {self.reasoning_state.question}

Current Context: {context}

Previous Reasoning Steps: {len(self.reasoning_state.thoughts)}

Working Hypotheses: {self.reasoning_state.working_hypotheses}

Knowledge Gaps: {self.reasoning_state.knowledge_gaps}

Confirmed Facts: {self.reasoning_state.confirmed_facts}

Think deeply about this situation. Consider:
1. What do we know for certain?
2. What are our current hypotheses?
3. What don't we know yet?
4. What patterns or connections might we be missing?
5. What would be the most valuable information to gather next?

Generate thoughtful analysis and insights. Be specific and actionable.
"""

    def _construct_action_prompt(self, thought: Thought) -> str:
        """Construct prompt for action planning."""
        available_tools = [tool.name for tool in self.tools]

        return f"""
Based on this thought: {thought.content}

Available tools: {available_tools}

Current knowledge state:
- Confirmed facts: {self.reasoning_state.confirmed_facts}
- Working hypotheses: {self.reasoning_state.working_hypotheses}
- Knowledge gaps: {self.reasoning_state.knowledge_gaps}

Choose the best action to take next. Consider:
1. Which tool would provide the most valuable information?
2. What specific parameters should be used?
3. What outcome do you expect?

Return JSON:
{{
    "tool_name": "selected_tool",
    "parameters": {{"param1": "value1"}},
    "rationale": "why this action",
    "expected_outcome": "what we expect to learn"
}}
"""

    def _construct_reflection_prompt(self, observation: Observation) -> str:
        """Construct prompt for reflection step."""
        return f"""
Action taken: {observation.action_id}
Result: {json.dumps(observation.result, indent=2)}
Success: {observation.success}

Reflect deeply on this observation:

1. What did we learn? (insights)
2. What was unexpected? (surprises)
3. How does this change our understanding?
4. What new hypotheses does this generate?
5. What should we investigate next?
6. How confident are we in our current direction?

Generate thoughtful reflection focusing on:
- Pattern recognition
- Hypothesis refinement
- Error analysis (if applicable)
- Meta-cognitive assessment

Return insights about the reasoning process itself.
"""

    def _construct_synthesis_prompt(self) -> str:
        """Construct prompt for final answer synthesis."""
        return f"""
Original Question: {self.reasoning_state.question}

Complete Reasoning Trace:
{json.dumps([
    {"step": i+1, "thought": t.content, "confidence": t.confidence}
    for i, t in enumerate(self.reasoning_state.thoughts)
], indent=2)}

Key Observations:
{json.dumps([
    {"insights": obs.insights, "surprises": obs.surprises}
    for obs in self.reasoning_state.observations
], indent=2)}

Reflections:
{json.dumps([
    {"type": r.type, "content": r.content, "new_hypotheses": r.new_hypotheses}
    for r in self.reasoning_state.reflections
], indent=2)}

Synthesize a comprehensive answer that:
1. Directly addresses the original question
2. Integrates all reasoning steps and evidence
3. Acknowledges limitations and uncertainties
4. Provides clear methodology explanation
5. Suggests follow-up investigations if appropriate

Be thorough but clear and well-structured.
"""

    def _parse_thought_response(self, response: str) -> Thought:
        """Parse LLM response into Thought object."""
        # Simple parsing - could be enhanced with structured output
        return Thought(
            content=response.strip(),
            type="analytical",
            confidence=0.7  # Could be extracted from response
        )

    def _parse_action_response(self, response: str, thought: Thought) -> Action:
        """Parse LLM response into Action object."""
        try:
            action_data = json.loads(response)
            return Action(
                tool_name=action_data.get("tool_name", ""),
                parameters=action_data.get("parameters", {}),
                rationale=action_data.get("rationale", ""),
                expected_outcome=action_data.get("expected_outcome", "")
            )
        except:
            # Fallback for malformed JSON
            return Action(
                tool_name="enhanced_neo4j_query",
                parameters={"query": "MATCH (n) RETURN count(n) as total_nodes LIMIT 1"},
                rationale="Fallback action due to parsing error",
                expected_outcome="Basic graph statistics"
            )

    def _parse_reflection_response(self, response: str) -> Reflection:
        """Parse LLM response into Reflection object."""
        return Reflection(
            content=response.strip(),
            type="general",
            confidence_change=0.0  # Could be extracted
        )

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute an action using the appropriate tool."""
        # This would integrate with the tool router
        tool_router = getattr(self, '_tool_router', lambda n, a: {"error": "Tool router not available"})
        return tool_router(action.tool_name, action.parameters)

    def _tool_router_wrapper(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for tool router to integrate with reasoning."""
        # This should be connected to the main agent's tool router
        return {"placeholder": "Tool execution result"}

    def _analyze_observation_results(self, result: Dict[str, Any], expected: str) -> Tuple[List[str], List[str]]:
        """Analyze observation results to extract insights and surprises."""
        insights = []
        surprises = []

        if "error" in result:
            surprises.append(f"Unexpected error: {result['error']}")
        elif "data" in result:
            data = result["data"]
            if isinstance(data, list):
                insights.append(f"Retrieved {len(data)} data points")
                if len(data) == 0:
                    surprises.append("No data found - may indicate incorrect assumptions")

        return insights, surprises

    def _update_reasoning_state_from_reflection(self, reflection: Reflection):
        """Update reasoning state based on reflection."""
        if reflection.new_hypotheses:
            self.reasoning_state.working_hypotheses.extend(reflection.new_hypotheses)

        # Update confidence based on reflection
        self.reasoning_state.confidence_score += reflection.confidence_change
        self.reasoning_state.confidence_score = max(0.0, min(1.0, self.reasoning_state.confidence_score))

    def _build_context_for_thinking(self) -> str:
        """Build context string for thinking step."""
        context_parts = []

        if self.reasoning_state.observations:
            latest_obs = self.reasoning_state.observations[-1]
            context_parts.append(f"Latest observation: {latest_obs.insights}")

        if self.reasoning_state.reflections:
            latest_ref = self.reasoning_state.reflections[-1]
            context_parts.append(f"Latest reflection: {latest_ref.content[:200]}...")

        return " | ".join(context_parts)

    def _extract_evidence(self) -> List[Dict[str, Any]]:
        """Extract evidence from reasoning trace."""
        evidence = []
        for obs in self.reasoning_state.observations:
            if obs.success and obs.insights:
                evidence.append({
                    "type": "empirical",
                    "source": obs.action_id,
                    "content": obs.insights
                })
        return evidence

    def _extract_methodology(self) -> Dict[str, Any]:
        """Extract methodology used in reasoning."""
        return {
            "reasoning_steps": self.reasoning_state.current_step,
            "tools_used": list(set(action.tool_name for action in self.reasoning_state.actions)),
            "reasoning_pattern": "think-act-observe-reflect",
            "confidence_progression": [t.confidence for t in self.reasoning_state.thoughts]
        }

    def _extract_limitations(self) -> List[str]:
        """Extract limitations identified during reasoning."""
        limitations = []

        # Extract from knowledge gaps
        limitations.extend(self.reasoning_state.knowledge_gaps)

        # Extract from failed actions
        for obs in self.reasoning_state.observations:
            if not obs.success:
                limitations.append(f"Unable to execute: {obs.action_id}")

        return limitations


class ReasoningMemory:
    """Memory system for storing and retrieving reasoning patterns and insights."""

    def __init__(self):
        self.episodic_memory = []  # Specific reasoning episodes
        self.semantic_memory = {}  # General patterns and knowledge
        self.procedural_memory = {}  # Reasoning strategies that work

    def store_reasoning_episode(self, question: str, reasoning_state: ReasoningState, outcome: Dict[str, Any]):
        """Store a complete reasoning episode."""
        episode = {
            "question": question,
            "reasoning_state": reasoning_state,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "success": outcome.get("confidence", 0) > 0.7
        }
        self.episodic_memory.append(episode)

        # Extract patterns for semantic memory
        self._extract_patterns_from_episode(episode)

    def retrieve_similar_episodes(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar reasoning episodes."""
        # Simple similarity based on keyword matching
        # Could be enhanced with semantic similarity
        question_words = set(question.lower().split())

        similarities = []
        for episode in self.episodic_memory:
            episode_words = set(episode["question"].lower().split())
            similarity = len(question_words & episode_words) / len(question_words | episode_words)
            similarities.append((similarity, episode))

        similarities.sort(reverse=True)
        return [episode for _, episode in similarities[:top_k]]

    def _extract_patterns_from_episode(self, episode: Dict[str, Any]):
        """Extract patterns from reasoning episode for future use."""
        # Extract successful reasoning patterns
        if episode["success"]:
            reasoning_state = episode["reasoning_state"]
            pattern_key = f"successful_reasoning_{len(reasoning_state.thoughts)}_steps"

            if pattern_key not in self.semantic_memory:
                self.semantic_memory[pattern_key] = []

            self.semantic_memory[pattern_key].append({
                "tools_used": [action.tool_name for action in reasoning_state.actions],
                "reasoning_path": reasoning_state.reasoning_path,
                "final_confidence": reasoning_state.confidence_score
            })