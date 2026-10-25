"""The agent loop: the model chooses tools, our code runs them, until the model answers.

Usage: from src.agent.agent import ask; answer = ask("How much will FOODS_3_090 sell next week?")
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.agent.config import MAX_TOKENS, MAX_TOOL_CALLS, MODEL
from src.agent.grounding import RETRY, ungrounded_numbers
from src.agent.prompt import SYSTEM_PROMPT
from src.agent.schemas import TOOL_SCHEMAS
from src.agent.tools import run_tool

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ToolCall:
    name: str
    inputs: dict
    result: dict
    seconds: float


@dataclass
class Answer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list = field(default_factory=list)  # the whole conversation, for follow-ups
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: bool = False  # True if the loop hit its limit or the reply was cut off
    ungrounded: list[str] = field(default_factory=list)  # numbers no tool supports, after retry
    retried: bool = False  # True if the first answer had ungrounded numbers


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


def _text(response) -> str:
    """The reply's text blocks; thinking and tool_use blocks are not shown to the user."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def ask(
    question: str,
    history: list | None = None,
    client=None,
    on_tool_call: Callable[[ToolCall], None] | None = None,
    model: str = MODEL,
) -> Answer:
    """Answer one question, calling tools as the model asks, up to MAX_TOOL_CALLS of them.

    history: Answer.messages from the previous question, so follow-ups keep their context.
    on_tool_call: called after each tool runs (the chat prints it, the app shows it).
    model: the Claude model; the evaluation passes another one to compare.
    """
    client = client or _client()
    messages = list(history or []) + [{"role": "user", "content": question}]
    answer = Answer(text="")
    for _ in range(MAX_TOOL_CALLS + 3):  # +3: the first request, a retry and the final answer
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        answer.input_tokens += response.usage.input_tokens
        answer.output_tokens += response.usage.output_tokens
        # The whole reply goes back next time, thinking blocks included: the API needs them
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = _text(response)
            bad = ungrounded_numbers(text, messages)
            if bad and not answer.retried:  # one second chance, then answer with a warning
                answer.retried = True
                messages.append({"role": "user", "content": RETRY.format(", ".join(bad))})
                continue
            answer.text = text
            answer.ungrounded = bad
            answer.stopped_early = response.stop_reason == "max_tokens"
            answer.messages = messages
            return answer

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if len(answer.tool_calls) >= MAX_TOOL_CALLS:
                result = {"error": "Tool-call limit reached. Answer with the results you have."}
                seconds = 0.0
            else:
                start = time.perf_counter()
                result = run_tool(block.name, dict(block.input))
                seconds = time.perf_counter() - start
            call = ToolCall(block.name, dict(block.input), result, seconds)
            answer.tool_calls.append(call)
            if on_tool_call:
                on_tool_call(call)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": "error" in result,
                }
            )
        messages.append({"role": "user", "content": results})

    answer.text = "I could not finish within the tool-call limit."
    answer.stopped_early = True
    answer.messages = messages
    return answer
