"""Chat with the agent in the terminal.

Usage: uv run python -m src.agent.chat   (an empty line quits)
"""

import json

from src.agent.agent import ToolCall, ask


def show(call: ToolCall) -> None:
    flag = "  ERROR" if "error" in call.result else ""
    print(f"  [{call.name}] {json.dumps(call.inputs)} ({call.seconds:.1f} s){flag}")


def main() -> None:
    print("Ask about CA_1 / FOODS_3 forecasts, stock levels or disruptions. Empty line quits.")
    history = None
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        answer = ask(question, history, on_tool_call=show)
        print(f"\n{answer.text}")
        if answer.ungrounded:
            print(f"\n  WARNING: numbers no tool supports: {', '.join(answer.ungrounded)}")
        elif answer.retried:
            print("\n  (the first answer had an unsupported number and was corrected)")
        tokens = f"{answer.input_tokens} in / {answer.output_tokens} out tokens"
        print(f"\n  ({len(answer.tool_calls)} tool calls, {tokens})")
        history = answer.messages


if __name__ == "__main__":
    main()
