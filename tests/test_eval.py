from types import SimpleNamespace

import pytest

from src.agent.run_eval import expected_values, load_questions, mentions, resolve, score
from src.agent.tools import TOOLS


def test_the_set_has_30_questions_of_7_types():
    questions = load_questions()
    assert len(questions) == 30
    assert len({q["id"] for q in questions}) == 30
    assert len({q["type"] for q in questions}) == 7
    for q in questions:
        assert all(t in TOOLS for t in q.get("tools") or [])


def test_every_expected_value_resolves_to_a_number():
    for q in load_questions():
        assert all(isinstance(v, float) for v in expected_values(q)), q["id"]


def test_resolve_reads_nested_fields():
    result = {"rows": [{"policy": "(s, S)", "x": 1.5}], "daily": [{"q50": 3.0}, {"q50": 4.0}]}
    assert resolve(result, "rows[policy=(s, S)].x") == 1.5
    assert resolve(result, "daily.1.q50") == 4.0


@pytest.mark.parametrize(
    "answer, expected, ok",
    [
        ("About 369 units", 368.6, True),
        ("cerca de 1.580 unidades", 1580.5, True),
        ("fill rate of 97.7%", 0.9773, True),
        ("about $23.9k", 23889.0, True),
        ("about 400 units", 368.6, False),
    ],
)
def test_mentions_allows_rounding_and_formats(answer, expected, ok):
    assert mentions(answer, expected) is ok


def test_score_needs_all_four_checks():
    answer = SimpleNamespace(
        text="(s, S) loses 989 USD.",
        tool_calls=[SimpleNamespace(name="compare_policies")],
        ungrounded=[],
        stopped_early=False,
    )
    question = {"tools": ["compare_policies"], "names": [["(s, S)"]]}
    assert score(question, answer, [989.0])["correct"]
    assert not score(question, answer, [1258.0])["correct"]
    assert not score({**question, "tools": ["run_scenario"]}, answer, [989.0])["correct"]
