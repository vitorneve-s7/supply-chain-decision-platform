import json

from src.agent.agent import ask
from src.agent.grounding import RETRY, numbers_in, ungrounded_numbers
from tests.test_agent import FakeClient, text, tool_use


def conversation(result: dict, question: str = "Q") -> list:
    tool_result = {"type": "tool_result", "tool_use_id": "x", "content": json.dumps(result)}
    return [{"role": "user", "content": question}, {"role": "user", "content": [tool_result]}]


FORECAST = {"from": "2016-05-23", "to": "2016-05-29", "q50_total_units": 368.6, "q95": 579.8}


def test_an_invented_number_is_flagged():
    assert ungrounded_numbers("About 400 units next week.", conversation(FORECAST)) == ["400"]


def test_rounded_and_reformatted_numbers_pass():
    messages = conversation({"s": 1263.3, "fill_rate": 0.9773, "cost": 23905.0, "change": -10.8})
    answer = "s = 1.263,3 (or 1,263); fill rate 97,7% (0.977); 23,905 USD, about 23.9k; −10,8 USD"
    assert ungrounded_numbers(answer, messages) == []


def test_space_as_thousands_separator():
    messages = conversation({"q95_total_units": 1134.8, "cost": 52214.0})
    assert ungrounded_numbers("até 1 134,8 unidades; custo de 52 214 USD", messages) == []
    assert numbers_in("de 23 a 29 de maio") == ["23", "29"]


def test_ids_dates_and_small_counts_are_not_checked():
    answer = "FOODS_3_090 in S1: about 369 units from 23 to 29 May 2016, across 4 policies."
    assert ungrounded_numbers(answer, conversation(FORECAST)) == []


def test_numbers_from_the_question_are_allowed():
    messages = conversation(FORECAST, question="Is 500 units enough?")
    assert ungrounded_numbers("No: 500 is below the 95% total of 580.", messages) == []


def test_the_retry_note_does_not_ground_its_own_numbers():
    messages = conversation(FORECAST) + [{"role": "user", "content": RETRY.format("400")}]
    assert ungrounded_numbers("About 400 units.", messages) == ["400"]


def test_agent_retries_once_with_the_unsupported_numbers():
    first = tool_use("get_forecast", {"item": "FOODS_3_090", "days": 7})
    client = FakeClient([first, text("About 400 units."), text("About 369 units.")])
    answer = ask("How much next week?", client=client)
    assert answer.retried and answer.ungrounded == []
    assert answer.text == "About 369 units."
    assert "400" in client.requests[2]["messages"][-1]["content"]  # the retry note


def test_agent_reports_numbers_still_unsupported_after_the_retry():
    first = tool_use("get_forecast", {"item": "FOODS_3_090", "days": 7})
    client = FakeClient([first, text("About 400 units.")])  # repeats the mistake
    answer = ask("How much next week?", client=client)
    assert answer.ungrounded == ["400"]
