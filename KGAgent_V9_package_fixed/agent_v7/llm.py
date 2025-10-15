
import json
import os
import logging
from typing import Any, Dict, List, Optional

from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Choose LLM provider: "openai" or "qwen"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen").lower()

# We support Responses API (preferred) with a fallback to Chat Completions.
# Toggle via env: USE_RESPONSES_API=1/0
USE_RESPONSES = os.getenv("USE_RESPONSES_API", "1") not in ("0", "false", "False")

try:
    from openai import OpenAI  # Official SDK v1+ (used for both OpenAI and Qwen compatibility)
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
                 planner_model: str = None,
                 summarizer_model: str = None):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. `pip install openai`")

        self.provider = LLM_PROVIDER

        if self.provider == "qwen":
            # Qwen configuration - OpenAI-compatible API
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            # Default Qwen models (latest 2025 models)
            self.planner_model = planner_model or os.getenv("PLANNER_MODEL", "qwen-max-2025-01-25")
            self.summarizer_model = summarizer_model or os.getenv("SUMMARIZER_MODEL", "qwen2.5-72b-instruct")
            logger.info(f"Initialized Qwen client with models: {self.planner_model}, {self.summarizer_model}")
        else:
            # OpenAI configuration (commented for now, but kept functional)
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.planner_model = planner_model or os.getenv("PLANNER_MODEL", "gpt-5")
            self.summarizer_model = summarizer_model or os.getenv("SUMMARIZER_MODEL", "gpt-4o")
            logger.info(f"Initialized OpenAI client with models: {self.planner_model}, {self.summarizer_model}")

        # Model capabilities detection
        self.is_qwen = "qwen" in self.planner_model.lower()
        self.is_gpt5 = "gpt-5" in self.planner_model
        self.is_o1 = "o1" in self.planner_model

    # ---------- Responses API path (disabled for Qwen) ----------
    def _responses_loop(self, system_prompt: str, user_prompt: str, tools: List[ToolSpec],
                        tool_router) -> str:
        """
        OpenAI Responses API implementation - disabled for Qwen as it uses Chat Completions API
        """
        # Qwen doesn't support Responses API, fall back to chat completions
        if self.provider == "qwen":
            logger.debug("Qwen doesn't support Responses API, using Chat Completions")
            return self._chat_loop(system_prompt, user_prompt, tools, tool_router)

        # Original OpenAI implementation (commented for reference)
        create_params = {
            "model": self.planner_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "tools": _tools_to_openai_schema(tools)
        }

        # GPT-5 and o1 models don't support reasoning parameter
        if not self.is_gpt5 and not self.is_o1:
            create_params["reasoning"] = {"effort": "medium"}

        response = self.client.responses.create(**create_params)
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

                    follow_params = {
                        "model": self.planner_model,
                        "input": [{"role": "tool", "tool_call_id": item.id,
                                  "content": json.dumps(tool_result)}],
                        "conversation": convo_id
                    }

                    # GPT-5 and o1 models don't support reasoning parameter
                    if not self.is_gpt5 and not self.is_o1:
                        follow_params["reasoning"] = {"effort": "medium"}

                    response = self.client.responses.create(**follow_params)
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

    # ---------- Chat Completions (primary for Qwen) ----------
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
            create_params = {
                "model": self.planner_model,
                "messages": messages,
                "tools": tool_schema,
                "tool_choice": "auto"
            }

            # Add temperature for models that support it (Qwen supports it, GPT-5 and o1 don't)
            if not self.is_gpt5 and not self.is_o1:
                create_params["temperature"] = 0.2

            logger.debug(f"Calling {self.provider} chat completions with model: {self.planner_model}")
            resp = self.client.chat.completions.create(**create_params)
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
        # Qwen uses Chat Completions API exclusively
        if self.provider == "qwen":
            logger.debug("Using Qwen Chat Completions API")
            return self._chat_loop(system_prompt, user_prompt, tools, tool_router)

        # OpenAI: GPT-5 has issues with Responses API, use Chat Completions instead
        if USE_RESPONSES and not self.is_gpt5:
            try:
                return self._responses_loop(system_prompt, user_prompt, tools, tool_router)
            except Exception as e:
                logger.warning(f"Responses API failed, falling back to chat.completions: {e}")
        return self._chat_loop(system_prompt, user_prompt, tools, tool_router)

    def summarize(self, text: str, title: str = "Analysis Summary") -> str:
        prompt = f"Title: {title}\n\nSummarize the following analysis for a neuroscience KG audience:\n\n{text}"
        try:
            create_params = {
                "model": self.summarizer_model,
                "messages": [
                    {"role": "system", "content": "You write concise, technically accurate summaries."},
                    {"role": "user", "content": prompt}
                ]
            }

            # Add temperature for models that support it
            if not self.is_gpt5 and not self.is_o1:
                create_params["temperature"] = 0.2

            logger.debug(f"Summarizing with {self.provider} model: {self.summarizer_model}")
            resp = self.client.chat.completions.create(**create_params)
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

        # Qwen uses Chat Completions exclusively, OpenAI GPT-5 has compatibility issues with Responses API
        if USE_RESPONSES and self.provider == "openai" and not self.is_gpt5:
            try:
                create_params = {
                    "model": self.planner_model,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_schema", "json_schema": schema},
                }

                # GPT-5 and o1 models don't support reasoning parameter
                if not self.is_gpt5 and not self.is_o1:
                    create_params["reasoning"] = {"effort": "medium"}

                resp = self.client.responses.create(**create_params)
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
                logger.warning(f"Responses API failed: {e}, falling back to Chat Completions")

        # Chat Completions fallback
        try:
            # GPT-5 doesn't support json_object response format, Qwen supports it
            if self.is_gpt5:
                # GPT-5 text mode
                create_params = {
                    "model": self.planner_model,
                    "messages": [
                        {"role": "system", "content": system_prompt + " Return valid JSON only."},
                        {"role": "user", "content": user_prompt + "\n\nIMPORTANT: Return ONLY a valid JSON object with no other text."}
                    ]
                }
            else:
                # Qwen and other models can use json_object format
                create_params = {
                    "model": self.planner_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nReturn ONLY a JSON object."}
                    ],
                    "response_format": {"type": "json_object"}
                }

            # Add temperature for models that support it
            if not self.is_gpt5 and not self.is_o1:
                create_params["temperature"] = 0.1

            logger.debug(f"JSON planning with {self.provider} model: {self.planner_model}")
            resp = self.client.chat.completions.create(**create_params)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"JSON response format failed: {e}, using text mode")
            # Final fallback - just ask for JSON in text
            create_params = {
                "model": self.planner_model,
                "messages": [
                    {"role": "system", "content": system_prompt + " Return valid JSON only."},
                    {"role": "user", "content": user_prompt + "\n\nIMPORTANT: Return ONLY a valid JSON object with no other text."}
                ]
            }
            if not self.is_gpt5 and not self.is_o1:
                create_params["temperature"] = 0.1
            resp = self.client.chat.completions.create(**create_params)
            return (resp.choices[0].message.content or "").strip()
