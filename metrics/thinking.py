"""Thinking metrics: count, total chars, average chars."""

from parser import LogSession
from .base import BaseMetric


class ThinkingMetric(BaseMetric):
    name = "深度思考"
    category = "thinking"

    def compute(self, session: LogSession) -> dict:
        thinking_count = 0
        thinking_total_chars = 0

        for msg in session.messages:
            for block in msg.content:
                if block.type == "thinking" and block.text:
                    thinking_count += 1
                    thinking_total_chars += len(block.text)

        avg_chars = thinking_total_chars / thinking_count if thinking_count > 0 else 0

        return {
            "thinking_count": thinking_count,
            "thinking_total_chars": thinking_total_chars,
            "thinking_avg_chars": round(avg_chars, 1),
        }
