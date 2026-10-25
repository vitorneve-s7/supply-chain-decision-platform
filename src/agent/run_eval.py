"""Ask the agent the 30 evaluation questions and score each answer.

Usage: uv run python -m src.agent.run_eval                     (all 30, the default model)
       uv run python -m src.agent.run_eval --only F1 C2        (some questions)
       uv run python -m src.agent.run_eval --model claude-sonnet-5-5 --out agent_eval_sonnet.csv
Writes results/agent_eval.csv: one row per question with its scores, tools, tokens and answer.

A question is correct when all four checks pass: the expected tools were called, every expected
number appears in the answer (computed by the tools themselves, so the key never goes stale),
every expected name or keyword appears, and the answer has no ungrounded numbers.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from src.agent.agent import ask
from src.agent.config import MODEL
from src.agent.grounding import THOUSANDS, _decimals, _values, numbers_in
from src.agent.tools import run_tool

ROOT = Path(__file__).resolve().parents[2]
EVAL_SET = Path(__file__).with_name("eval_set.json")
RESULTS = ROOT / "results"
REL_TOL = 0.01  # an expected number matches if the answer is within 1% (or its rounding)


def load_questions() -> list[dict]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def resolve(result, field: str):
    """Read a field like 'q50_total_units', 'daily.5.q50' or 'rows[policy=(s, S)].fill_rate'."""
    value = result
    for part in field.split("."):
        match = re.fullmatch(r"(\w+)\[(\w+)=(.+)\]", part)
        if match:
            key, attr, wanted = match.groups()
            value = next(row for row in value[key] if str(row[attr]) == wanted)
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def expected_values(question: dict) -> list[float]:
    """The numbers the answer must contain, computed now by the same tools the agent uses."""
    values = []
    for check in question.get("checks", []):
        if "value" in check:
            values.append(float(check["value"]))
        else:
            values.append(float(resolve(run_tool(check["tool"], check["inputs"]), check["field"])))
    return values


def mentions(answer: str, expected: float) -> bool:
    """True if the answer writes the expected number, allowing rounding, % for shares and 'k'."""
    in_thousands = {m.group(1).strip(".,") for m in THOUSANDS.finditer(answer)}
    targets = {abs(expected)} | ({abs(expected) * 100} if abs(expected) <= 1 else set())
    for token in numbers_in(answer):
        token = token.strip(".,")
        scale = 1000 if token in in_thousands else 1
        for value in _values(token):
            for target in targets:
                tolerance = max(REL_TOL * target, 0.5 * 10 ** -_decimals(token) * scale)
                if abs(value * scale - target) <= tolerance + 1e-9:
                    return True
    return False


def score(question: dict, answer, expected: list[float]) -> dict:
    called = [c.name for c in answer.tool_calls]
    wanted = question.get("tools")
    if wanted is None:
        tools_ok = True
    elif wanted == []:
        tools_ok = not called
    else:
        tools_ok = all(t in called for t in wanted)
    text = answer.text.lower()
    names_ok = all(any(n.lower() in text for n in group) for group in question.get("names", []))
    result = {
        "tools_ok": tools_ok,
        "numbers_ok": all(mentions(answer.text, v) for v in expected),
        "names_ok": names_ok,
        "grounded_ok": not answer.ungrounded and not answer.stopped_early,
    }
    result["correct"] = all(result.values())
    return result


def run_one(question: dict, model: str) -> dict:
    expected = expected_values(question)
    start = time.perf_counter()
    history, tokens_in, tokens_out = None, 0, 0
    if question.get("setup"):
        first = ask(question["setup"], model=model)
        history, tokens_in, tokens_out = first.messages, first.input_tokens, first.output_tokens
    answer = ask(question["question"], history, model=model)
    return {
        "id": question["id"],
        "type": question["type"],
        **score(question, answer, expected),
        "retried": answer.retried,
        "ungrounded": " ".join(answer.ungrounded),
        "tools_called": " ".join(c.name for c in answer.tool_calls),
        "expected": " ".join(f"{v:g}" for v in expected),
        "input_tokens": tokens_in + answer.input_tokens,
        "output_tokens": tokens_out + answer.output_tokens,
        "seconds": round(time.perf_counter() - start, 1),
        "question": question["question"],
        "answer": answer.text,
    }


def main(only: list[str] | None, model: str, out: str) -> None:
    questions = [q for q in load_questions() if not only or q["id"] in only]
    rows = []
    for q in questions:
        try:
            row = run_one(q, model)
        except Exception as e:  # an API error should not lose the other answers
            row = {"id": q["id"], "type": q["type"], "correct": False, "answer": f"ERROR: {e}"}
            row |= {k: False for k in ("tools_ok", "numbers_ok", "names_ok", "grounded_ok")}
        mark = "ok  " if row["correct"] else "FAIL"
        failed = [
            k[:-3] for k in ("tools_ok", "numbers_ok", "names_ok", "grounded_ok") if not row[k]
        ]
        print(f"{mark} {q['id']:3} {q['type']:14} {', '.join(failed)}")
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / out, index=False)
    by_type = runs.groupby("type")["correct"].agg(["sum", "count"])
    print(f"\n{by_type.to_string()}")
    correct = runs["correct"]
    print(f"\nScore: {correct.sum()} / {len(runs)} ({correct.mean():.0%}) with {model}")
    print(f"Tokens: {runs['input_tokens'].sum():,} in / {runs['output_tokens'].sum():,} out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="question ids, e.g. F1 C2")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="agent_eval.csv")
    args = parser.parse_args()
    main(args.only, args.model, args.out)
