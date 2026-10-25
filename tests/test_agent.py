"""The agent loop, tested with a fake client: no API key, no cost."""

from types import SimpleNamespace

from src.agent.agent import ask
from src.agent.config import MAX_TOOL_CALLS


def tool_use(name, inputs, block_id="call_1"):
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=inputs)
    return _reply([SimpleNamespace(type="thinking"), block], "tool_use")


def text(words):
    return _reply([SimpleNamespace(type="text", text=words)], "end_turn")


def _reply(content, stop_reason):
    usage = SimpleNamespace(input_tokens=100, output_tokens=10)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


class FakeClient:
    """Plays back scripted replies and records what each request sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_runs_the_tool_and_sends_the_result_back():
    client = FakeClient(
        [tool_use("get_forecast", {"item": "FOODS_3_090", "days": 7}), text("About 369 units.")]
    )
    answer = ask("How much will FOODS_3_090 sell next week?", client=client)
    assert answer.text == "About 369 units."
    assert [c.name for c in answer.tool_calls] == ["get_forecast"]
    assert answer.tool_calls[0].result["q50_total_units"] == 368.6
    sent = client.requests[1]["messages"][-1]["content"][0]  # what went back to the model
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "call_1"
    assert answer.input_tokens == 200 and not answer.stopped_early


def test_errors_are_flagged_to_the_model():
    client = FakeClient([tool_use("get_forecast", {"item": "FOODS_9_999"}), text("Unknown item.")])
    ask("Forecast FOODS_9_999", client=client)
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_stops_at_the_tool_call_limit():
    client = FakeClient([tool_use("compare_policies", {"scenario": "all"})])  # never answers
    answer = ask("Loop forever", client=client)
    real = [c for c in answer.tool_calls if "error" not in c.result]
    assert len(real) == MAX_TOOL_CALLS
    assert answer.stopped_early


def test_follow_up_keeps_the_history():
    first = ask("Hi", client=FakeClient([text("Hello.")]))
    client = FakeClient([text("Still here.")])
    ask("And now?", history=first.messages, client=client)
    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "Hi"
