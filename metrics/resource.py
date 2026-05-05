"""Resource consumption metrics: tokens, duration, speed, steps."""

from datetime import datetime
from parser import LogSession
from .base import BaseMetric


class ResourceMetric(BaseMetric):
    name = "资源消耗"
    category = "resource"

    def compute(self, session: LogSession) -> dict:
        total_input = 0
        total_output = 0
        assistant_count = 0

        for msg in session.messages:
            if msg.type == "assistant" and msg.usage:
                total_input += msg.usage.input_tokens + msg.usage.cache_read_input_tokens
                total_output += msg.usage.output_tokens
                assistant_count += 1

        # Duration
        duration_sec = 0.0
        if session.first_timestamp and session.last_timestamp:
            try:
                t1 = datetime.fromisoformat(session.first_timestamp.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(session.last_timestamp.replace("Z", "+00:00"))
                duration_sec = (t2 - t1).total_seconds()
            except (ValueError, TypeError):
                duration_sec = 0.0

        tokens_per_sec = total_output / duration_sec if duration_sec > 0 else 0

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "session_duration_sec": round(duration_sec, 1),
            "tokens_per_second": round(tokens_per_sec, 2),
            "total_steps": assistant_count,
        }
