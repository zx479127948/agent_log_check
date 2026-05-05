"""Tool usage metrics: call count, errors, rate, subagent calls."""

from collections import Counter
from parser import LogSession
from .base import BaseMetric


class ToolUsageMetric(BaseMetric):
    name = "工具调用"
    category = "tool"

    def compute(self, session: LogSession) -> dict:
        tool_call_count = 0
        tool_error_count = 0
        subagent_count = 0
        tool_names = Counter()

        for msg in session.messages:
            for block in msg.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    tool_names[block.tool_name or "unknown"] += 1
                    if block.tool_name == "Agent":
                        subagent_count += 1
                elif block.type == "tool_result" and block.is_error:
                    tool_error_count += 1

        error_rate = tool_error_count / tool_call_count if tool_call_count > 0 else 0

        return {
            "tool_call_count": tool_call_count,
            "tool_error_count": tool_error_count,
            "tool_error_rate": round(error_rate, 4),
            "subagent_count": subagent_count,
            "tool_name_distribution": dict(tool_names),
        }
