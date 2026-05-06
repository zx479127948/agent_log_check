"""Metric computation pipeline."""

from parser import LogSession
from metrics.base import BaseMetric


class MetricPipeline:
    def __init__(self):
        self.metrics: list[BaseMetric] = []

    def register(self, metric: BaseMetric) -> None:
        self.metrics.append(metric)

    def run(self, session: LogSession) -> dict:
        """Run all metrics. Returns {category: {metric_key: value}}."""
        results = {}
        for m in self.metrics:
            result = m.compute(session)
            results[m.category] = results.get(m.category, {})
            results[m.category].update(result)
        return results

    def run_all(self, sessions: list[LogSession]) -> list[dict]:
        """Run all metrics on each session. Returns list of result dicts."""
        return [self.run(s) for s in sessions]
