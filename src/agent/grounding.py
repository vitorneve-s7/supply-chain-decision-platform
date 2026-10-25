"""Check that every number in an answer comes from the conversation, mostly from tool results.

A number is grounded if it matches, after rounding to the precision written, a number in a tool
result, the user's questions, the system prompt or the tool descriptions (all documented facts).
Shares may be written as percentages (0.9773 -> 97.7%). Item ids, scenario names and small counts
(12 or less) are ignored.
"""

import json
import re

from src.agent.prompt import SYSTEM_PROMPT
from src.agent.schemas import TOOL_SCHEMAS

RETRY = (
    "Check: your answer contains numbers that no tool result supports: {}. Rewrite the answer "
    "using only numbers from the tool results (call a tool if you need another figure), or "
    "drop those numbers. Reply with the corrected answer only."
)
SMALL = 12  # counts like "4 policies" or "2 weeks" are not checked
NAMES = re.compile(r"FOODS_\d+_\d+|\bS[0-4]\b|\bq(?:50|90|95)\b|\bCA_\d|\bTX_\d|\bWI_\d", re.I)
# A number, with thousands separated by "," "." or a space ("1 134,8" in Portuguese)
NUMBER = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d[\d.,]*")
SPACES = re.compile(r"[ \u00a0\u202f]")
THOUSANDS = re.compile(r"(\d[\d.,]*)\s*(k|mil|thousand)\b", re.I)  # "23.9k", "24 mil"


def _values(token: str) -> set[float]:
    """The values a written number can mean: 1,258 / 1.258,5 / 0,9773 / 97.7 ..."""
    token = token.strip(".,")
    candidates = {
        token.replace(",", ""),  # 1,258.5 (English)
        token.replace(".", "").replace(",", "."),  # 1.258,5 (Portuguese)
        token.replace(",", "."),  # 0,9773
    }
    values = set()
    for c in candidates:
        try:
            values.add(float(c))
        except ValueError:
            pass
    return values


def _decimals(token: str) -> int:
    """Digits after the decimal separator as written (the last . or , if 1-2 or 4+ digits)."""
    token = token.strip(".,")
    match = re.search(r"[.,](\d+)$", token)
    if not match or len(match.group(1)) == 3:  # 1,258 or 1.585: a thousands separator
        return 0
    return len(match.group(1))


def numbers_in(text: str) -> list[str]:
    """Numbers written in a text, without those inside item ids and scenario names."""
    return [SPACES.sub("", n) for n in NUMBER.findall(NAMES.sub(" ", text))]


def _collect(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for token in numbers_in(obj):
            out.update(_values(token))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, out)


def known_numbers(messages: list) -> set[float]:
    """Every number in the tool results and user questions so far, plus the documented context."""
    known: set[float] = set()
    _collect(SYSTEM_PROMPT, known)
    _collect([s["description"] for s in TOOL_SCHEMAS], known)
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            if not content.startswith(RETRY[:20]):  # the retry note's numbers are not facts
                _collect(content, known)
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                _collect(json.loads(block["content"]), known)
    return known | {k * 100 for k in known if k <= 1}  # shares written as percentages


def _matches(value: float, decimals: int, known: set[float]) -> bool:
    tolerance = 0.5 * 10**-decimals + 1e-9
    return any(abs(value - k) <= tolerance for k in known)


def ungrounded_numbers(answer: str, messages: list) -> list[str]:
    """Numbers in the answer that no tool result, question or documented fact supports."""
    known = known_numbers(messages)
    in_thousands = {m.group(1).strip(".,") for m in THOUSANDS.finditer(answer)}
    bad = []
    for token in numbers_in(answer):
        token = token.strip(".,")
        values = _values(token)
        if not values or any(v <= SMALL and v == int(v) for v in values):
            continue
        scale = 1000 if token in in_thousands else 1
        known_here = {k / scale for k in known}
        if not any(_matches(v, _decimals(token), known_here) for v in values):
            bad.append(token)
    return bad
