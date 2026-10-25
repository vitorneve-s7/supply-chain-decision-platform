"""Agent settings in one place."""

MODEL = "claude-haiku-5-5"
MAX_TOOL_CALLS = 6  # per question; then the agent must answer with what it has
MAX_TOKENS = 2048  # per model reply, thinking included
