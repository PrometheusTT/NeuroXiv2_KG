
import json
import os
import logging
from typing import Any, Dict, List, Optional

from dataclasses import dataclass

logger = logging.getLogger(__name__)

# We support Responses API (preferred) with a fallback to Chat Completions.
# Toggle via env: USE_RESPONSES_API=1/0
USE_RESPONSES = os.getenv("USE_RESPONSES_API", "1") not in ("0", "false", "False")

try:
    from openai import OpenAI  # Official SDK v1+
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]


def _tools_to_openai_schema(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
        } for t in tools
    ]


class LLMClient:
    def __init__(self, api_key: Optional[str] = None,
                 planner_model: str = "gpt-5",
                 summarizer_model: str = "gpt-4o"):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. `pip install openai`")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.planner_model = planner_model
        self.summarizer_model = summarizer_model

    # ---------- Responses API path ----------
    def _responses_loop(self, system_prompt: str, user_prompt: str, tools: List[ToolSpec],
                        tool_router) -> str:
        response = self.client.responses.create(
            model=self.planner_model,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=_tools_to_openai_schema(tools)
        )
        convo_id = response.id
        # Iterate until no more tool calls
        while True:
            made_call = False
            for item in response.output:
                if getattr(item, "type", "") == "tool_call":
                    made_call = True
                    tool_name = item.tool.name
                    args = json.loads(item.tool.arguments or "{}")
                    tool_result = tool_router(tool_name, args)
                    response = self.client.responses.create(
                        model=self.planner_model,
                        reasoning={"effort": "medium"},
                        input=[{"role": "tool", "tool_call_id": item.id,
                                "content": json.dumps(tool_result)}],
                        conversation=convo_id
                    )
                    break
            if not made_call:
                break
        # Concatenate text outputs
        chunks = []
        for o in response.output:
            if getattr(o, "content", None):
                try:
                    chunks.append(o.content[0].text.value)
                except Exception:
                    pass
        return "\n".join(chunks).strip()

    # ---------- Chat Completions fallback ----------
    def _chat_loop(self, system_prompt: str, user_prompt: str, tools: List[ToolSpec],
                   tool_router) -> str:
        # Convert tool schema for chat.completions
        tool_schema = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
        } for t in tools]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        while True:
            resp = self.client.chat.completions.create(
                model=self.planner_model,
                messages=messages,
                tools=tool_schema,
                tool_choice="auto",
                # temperature=0.2
            )
            choice = resp.choices[0]
            fn_call = getattr(choice.message, "tool_calls", None) or getattr(choice.message, "function_call", None)
            if fn_call:
                # Tool calls may be parallel; handle sequentially for simplicity
                if isinstance(fn_call, list):
                    for call in fn_call:
                        name = call.function.name
                        args = json.loads(call.function.arguments or "{}")
                        result = tool_router(name, args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": name,
                            "content": json.dumps(result)
                        })
                    continue
                else:
                    name = fn_call.name
                    args = json.loads(fn_call.arguments or "{}")
                    result = tool_router(name, args)
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result)
                    })
                    continue
            # No tool call -> final text
            return (choice.message.content or "").strip()

    def run_with_tools(self, system_prompt: str, user_prompt: str, tools: List[ToolSpec], tool_router) -> str:
        if USE_RESPONSES:
            try:
                return self._responses_loop(system_prompt, user_prompt, tools, tool_router)
            except Exception as e:
                logger.warning(f"Responses API failed, falling back to chat.completions: {e}")
        return self._chat_loop(system_prompt, user_prompt, tools, tool_router)

    def summarize(self, text: str, title: str = "Analysis Summary") -> str:
        prompt = f"Title: {title}\n\nSummarize the following analysis for a neuroscience KG audience:\n\n{text}"
        try:
            resp = self.client.chat.completions.create(
                model=self.summarizer_model,
                messages=[
                    {"role": "system", "content": "You write concise, technically accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Summarizer failed: {e}")
            return text[:4000]

    def run_planner_json(self, system_prompt: str, user_prompt: str) -> str:
        # 强制 JSON schema
        schema = {
            "name": "PlannerSchema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cypher_attempts", "analysis_plan"],
                "properties": {
                    "cypher_attempts": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["purpose", "query"],
                            "additionalProperties": False,
                            "properties": {
                                "purpose": {"type": "string"},
                                "query": {"type": "string"}
                            }
                        }
                    },
                    "analysis_plan": {"type": "string"}
                }
            },
            "strict": True
        }

        if USE_RESPONSES:
            try:
                resp = self.client.responses.create(
                    model=self.planner_model,
                    reasoning={"effort": "medium"},
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_schema", "json_schema": schema},
                )
                # 拼接文本
                out = []
                for o in resp.output:
                    if getattr(o, "content", None):
                        try:
                            out.append(o.content[0].text.value)
                        except Exception:
                            pass
                return "\n".join(out).strip()
            except Exception as e:
                # 回退到 chat
                pass

        # Chat Completions 回退，强制 JSON 对象
        resp = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn ONLY a JSON object."}
            ],
            response_format={"type": "json_object"}
        )
        return (resp.choices[0].message.content or "").strip()

    # 在LLMClient类中添加新方法

    def run_advanced_reasoning(self, system_prompt: str, user_prompt: str) -> str:
        """
        Run advanced reasoning with chain-of-thought
        """
        enhanced_system = f"""{system_prompt}

        You should think step-by-step, showing your reasoning process.
        Be specific, quantitative, and cite evidence from the data.
        Focus on biological significance and mechanistic interpretation."""

        if USE_RESPONSES:
            try:
                resp = self.client.responses.create(
                    model=self.planner_model,
                    reasoning={"effort": "high"},  # Use high effort for insights
                    input=[
                        {"role": "system", "content": enhanced_system},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                # Extract response
                chunks = []
                for o in resp.output:
                    if getattr(o, "content", None):
                        try:
                            chunks.append(o.content[0].text.value)
                        except Exception:
                            pass
                return "\n".join(chunks).strip()
            except Exception as e:
                logger.warning(f"Advanced reasoning failed, falling back: {e}")

        # Fallback to standard completion with CoT prompt
        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": user_prompt + "\n\nThink step by step and show your reasoning."}
        ]

        resp = self.client.chat.completions.create(
            model=self.planner_model,
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused reasoning
            max_tokens=2000
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_neuroscience_summary(self, analysis_results: Dict) -> str:
        """
        Generate publication-quality neuroscience summary
        """
        system = """You are a computational neuroscientist writing for a high-impact journal.
        Your summary should be:
        1. Quantitative and specific
        2. Biologically interpretable
        3. Connected to broader neuroscience knowledge
        4. Proposing testable hypotheses

        Use standard neuroscience terminology and cite specific data points."""

        user = f"""Generate a comprehensive summary of this analysis:

    {json.dumps(analysis_results, indent=2)}

    Structure your response as:

    ### Key Findings
    - Primary discovery with specific numbers
    - Supporting evidence from multiple analyses
    - Statistical significance where applicable

    ### Biological Interpretation
    - Mechanistic implications
    - Relationship to known brain function
    - Clinical or evolutionary relevance

    ### Network Organization
    - Connectivity patterns and their significance
    - Hub regions and their roles
    - Hierarchical organization if detected

    ### Morphology-Molecular Relationships
    - Correlation or mismatch findings
    - Implications for cell type classification
    - Functional interpretations

    ### Future Directions
    - Specific testable hypotheses
    - Proposed experiments
    - Potential applications

    Be specific, use actual region names and numbers from the data."""

        return self.run_advanced_reasoning(system, user)
