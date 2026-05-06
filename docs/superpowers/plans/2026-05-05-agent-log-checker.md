# Agent Log Quality Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that parses Claude Code JSONL logs, computes quality metrics across 4 categories, compares baseline vs upgraded logs, calls claude -p for risk assessment, and generates a detailed HTML report.

**Architecture:** Strategy pattern for metric detectors (BaseMetric → ResourceMetric, ToolUsageMetric, ThinkingMetric, ContentQualityMetric), pipeline orchestration, JSONL parser to unified data model, claude -p for LLM judgment, Jinja2-style string templates for HTML generation.

**Tech Stack:** Python 3.11, stdlib only (no external dependencies), subprocess for claude -p, dataclasses for models

---

## File Structure

| File | Responsibility |
|------|---------------|
| `main.py` | CLI entry point, argparse, orchestrates parse→compute→judge→report |
| `parser.py` | JSONL parsing to LogSession/LogGroup dataclasses |
| `metrics/base.py` | BaseMetric ABC with name, category, compute() |
| `metrics/resource.py` | ResourceMetric - tokens, duration, speed, steps |
| `metrics/tool_usage.py` | ToolUsageMetric - calls, errors, rate, subagent count |
| `metrics/thinking.py` | ThinkingMetric - count, chars, avg |
| `metrics/content_quality.py` | ContentQualityMetric - vague words, repeated sentences |
| `metrics/__init__.py` | Re-exports all metric classes |
| `pipeline.py` | MetricPipeline - register + run all metrics on a LogSession |
| `llm_judge.py` | Build prompt, call `claude -p`, parse structured JSON result |
| `reporter.py` | Generate self-contained HTML report from comparison data |

---

### Task 1: Data Models and Parser

**Files:**
- Create: `parser.py`

- [ ] **Step 1: Write the data model classes and parser**

Create `parser.py` with all dataclasses and the `parse_log_dir` function:

```python
"""JSONL log parser for Claude Code agent sessions."""

import json
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ContentBlock:
    type: str  # text / thinking / tool_use / tool_result
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    error_content: str | None = None


@dataclass
class Message:
    type: str  # user / assistant
    timestamp: str | None = None
    role: str | None = None
    content: list[ContentBlock] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass
class LogSession:
    name: str
    log_path: str
    messages: list[Message] = field(default_factory=list)
    first_timestamp: str | None = None
    last_timestamp: str | None = None


@dataclass
class LogGroup:
    main: LogSession
    subagents: list[dict]  # list of {"meta": dict, "session": LogSession}


def _parse_content_block(block: dict) -> ContentBlock:
    """Parse a single content block from JSONL."""
    btype = block.get("type", "")
    if btype == "text":
        return ContentBlock(type="text", text=block.get("text", ""))
    elif btype == "thinking":
        return ContentBlock(type="thinking", text=block.get("thinking", ""))
    elif btype == "tool_use":
        return ContentBlock(
            type="tool_use",
            tool_name=block.get("name"),
            tool_input=block.get("input"),
            tool_use_id=block.get("id"),
        )
    elif btype == "tool_result":
        is_err = block.get("is_error", False)
        err_content = None
        if is_err:
            rc = block.get("content", "")
            err_content = rc if isinstance(rc, str) else str(rc)
        return ContentBlock(
            type="tool_result",
            tool_use_id=block.get("tool_use_id"),
            is_error=bool(is_err),
            error_content=err_content,
        )
    return ContentBlock(type=btype)


def _parse_message(line: dict) -> Message | None:
    """Parse a single JSONL line into a Message."""
    msg_type = line.get("type", "")
    if msg_type not in ("user", "assistant"):
        return None

    timestamp = line.get("timestamp")
    msg = line.get("message", {})
    role = msg.get("role") if isinstance(msg, dict) else None

    # Parse content blocks
    content_raw = msg.get("content", []) if isinstance(msg, dict) else []
    blocks = []
    if isinstance(content_raw, str):
        if content_raw:
            blocks.append(ContentBlock(type="text", text=content_raw))
    elif isinstance(content_raw, list):
        for c in content_raw:
            if isinstance(c, dict):
                blocks.append(_parse_content_block(c))

    # Parse usage
    usage = None
    if isinstance(msg, dict) and "usage" in msg:
        u = msg["usage"]
        usage = TokenUsage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
        )

    return Message(
        type=msg_type,
        timestamp=timestamp,
        role=role,
        content=blocks,
        usage=usage,
    )


def parse_jsonl(filepath: str) -> LogSession:
    """Parse a single JSONL file into a LogSession."""
    messages = []
    timestamps = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = _parse_message(obj)
            if msg is not None:
                messages.append(msg)
                if msg.timestamp:
                    timestamps.append(msg.timestamp)

    name = os.path.splitext(os.path.basename(filepath))[0]
    return LogSession(
        name=name,
        log_path=filepath,
        messages=messages,
        first_timestamp=timestamps[0] if timestamps else None,
        last_timestamp=timestamps[-1] if timestamps else None,
    )


def parse_log_dir(log_dir: str) -> LogGroup:
    """Parse a log directory into a LogGroup (main + subagents)."""
    # Find main agent JSONL (root level .jsonl file)
    main_jsonl = None
    for f in os.listdir(log_dir):
        if f.endswith(".jsonl"):
            main_jsonl = os.path.join(log_dir, f)
            break

    if main_jsonl is None:
        raise FileNotFoundError(f"No .jsonl file found in {log_dir}")

    main_session = parse_jsonl(main_jsonl)

    # Find subagents
    subagents = []
    sub_dir = os.path.join(log_dir, main_session.name, "subagents")
    if os.path.isdir(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            if f.endswith(".jsonl"):
                sub_path = os.path.join(sub_dir, f)
                meta_path = sub_path.replace(".jsonl", ".meta.json")
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                sub_session = parse_jsonl(sub_path)
                subagents.append({"meta": meta, "session": sub_session})

    return LogGroup(main=main_session, subagents=subagents)
```

- [ ] **Step 2: Verify parser works on sample data**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
from parser import parse_log_dir
g = parse_log_dir('sample/log/log_glm5.1_20260505')
print('Main:', g.main.name, 'msgs:', len(g.main.messages))
print('First ts:', g.main.first_timestamp)
print('Last ts:', g.main.last_timestamp)
print('Subagents:', len(g.subagents))
for s in g.subagents:
    print(f'  {s[\"session\"].name}: meta={s[\"meta\"]}, msgs={len(s[\"session\"].messages)}')
"
```

Expected: Main session with ~131 messages, 5 subagents with meta info.

- [ ] **Step 3: Commit**

```bash
git add parser.py
git commit -m "feat: add JSONL log parser with data models"
```

---

### Task 2: BaseMetric and Metric Pipeline

**Files:**
- Create: `metrics/__init__.py`
- Create: `metrics/base.py`
- Create: `pipeline.py`

- [ ] **Step 1: Write BaseMetric ABC and MetricPipeline**

Create `metrics/__init__.py`:
```python
from .base import BaseMetric
from .resource import ResourceMetric
from .tool_usage import ToolUsageMetric
from .thinking import ThinkingMetric
from .content_quality import ContentQualityMetric

__all__ = [
    "BaseMetric",
    "ResourceMetric",
    "ToolUsageMetric",
    "ThinkingMetric",
    "ContentQualityMetric",
]
```

Create `metrics/base.py`:
```python
"""Base class for metric detectors."""

from abc import ABC, abstractmethod
from parser import LogSession


class BaseMetric(ABC):
    name: str = ""
    category: str = ""  # resource / tool / thinking / content

    @abstractmethod
    def compute(self, session: LogSession) -> dict:
        """Compute metrics for a session. Returns {metric_key: value}."""
```

Create `pipeline.py`:
```python
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
```

- [ ] **Step 2: Verify pipeline can be instantiated and registered**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from pipeline import MetricPipeline
from metrics.base import BaseMetric
p = MetricPipeline()
print('Pipeline created, metrics:', len(p.metrics))
"
```

Expected: `Pipeline created, metrics: 0`

- [ ] **Step 3: Commit**

```bash
git add metrics/__init__.py metrics/base.py pipeline.py
git commit -m "feat: add BaseMetric ABC and MetricPipeline"
```

---

### Task 3: ResourceMetric

**Files:**
- Create: `metrics/resource.py`

- [ ] **Step 1: Write ResourceMetric**

Create `metrics/resource.py`:
```python
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
```

- [ ] **Step 2: Verify on sample data**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from parser import parse_log_dir
from metrics.resource import ResourceMetric
g = parse_log_dir('sample/log/log_glm5.1_20260505')
m = ResourceMetric()
print('Main:', m.compute(g.main))
print('Sub:', m.compute(g.subagents[0]['session']))
"
```

Expected: Main shows ~3080863 input tokens, ~28593 output tokens, ~15 min duration, ~5 subagents.

- [ ] **Step 3: Commit**

```bash
git add metrics/resource.py
git commit -m "feat: add ResourceMetric - tokens, duration, speed, steps"
```

---

### Task 4: ToolUsageMetric

**Files:**
- Create: `metrics/tool_usage.py`

- [ ] **Step 1: Write ToolUsageMetric**

Create `metrics/tool_usage.py`:
```python
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
```

- [ ] **Step 2: Verify on sample data**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from parser import parse_log_dir
from metrics.tool_usage import ToolUsageMetric
g = parse_log_dir('sample/log/log_glm5.1_20260505')
m = ToolUsageMetric()
print('Main:', m.compute(g.main))
"
```

Expected: Main shows ~17 tool calls, ~1 error, ~5 Agent calls.

- [ ] **Step 3: Commit**

```bash
git add metrics/tool_usage.py
git commit -m "feat: add ToolUsageMetric - calls, errors, rate, subagents"
```

---

### Task 5: ThinkingMetric

**Files:**
- Create: `metrics/thinking.py`

- [ ] **Step 1: Write ThinkingMetric**

Create `metrics/thinking.py`:
```python
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
```

- [ ] **Step 2: Verify on sample data**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from parser import parse_log_dir
from metrics.thinking import ThinkingMetric
g = parse_log_dir('sample/log/log_glm5.1_20260505')
m = ThinkingMetric()
print('Main:', m.compute(g.main))
"
```

Expected: Main shows ~10 thinking blocks, ~6221 total chars.

- [ ] **Step 3: Commit**

```bash
git add metrics/thinking.py
git commit -m "feat: add ThinkingMetric - count, chars, average"
```

---

### Task 6: ContentQualityMetric

**Files:**
- Create: `metrics/content_quality.py`

- [ ] **Step 1: Write ContentQualityMetric**

Create `metrics/content_quality.py`:
```python
"""Content quality metrics: vague words, repeated sentences."""

import re
from collections import Counter
from parser import LogSession
from .base import BaseMetric

VAGUE_WORDS = [
    "可能", "大概", "或许", "建议", "疑似", "也许", "似乎", "好像",
    "应该", "不确定", "不确定是否", "无法确定", "未必", "不一定",
]

# Split text into sentences by Chinese/English sentence-end punctuation
_SENTENCE_SPLIT_RE = re.compile(r"[。！？.!?\n]+")


class ContentQualityMetric(BaseMetric):
    name = "内容质量"
    category = "content"

    def __init__(self, vague_words: list[str] | None = None):
        self.vague_words = vague_words or VAGUE_WORDS
        # Build regex that matches any of the vague words
        self._vague_re = re.compile(
            "|".join(re.escape(w) for w in sorted(self.vague_words, key=len, reverse=True))
        )

    def compute(self, session: LogSession) -> dict:
        vague_hits = 0
        all_sentences: list[str] = []

        for msg in session.messages:
            for block in msg.content:
                if block.type == "text" and block.text:
                    vague_hits += len(self._vague_re.findall(block.text))
                    parts = _SENTENCE_SPLIT_RE.split(block.text)
                    for p in parts:
                        stripped = p.strip()
                        if len(stripped) >= 6:
                            all_sentences.append(stripped)

        # Count sentences appearing >= 3 times
        sentence_counts = Counter(all_sentences)
        repeat_3plus = sum(1 for c in sentence_counts.values() if c >= 3)

        return {
            "vague_word_hits": vague_hits,
            "sentence_repeat_3plus": repeat_3plus,
        }
```

- [ ] **Step 2: Verify on sample data**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from parser import parse_log_dir
from metrics.content_quality import ContentQualityMetric
g = parse_log_dir('sample/log/log_glm5.1_20260505')
m = ContentQualityMetric()
print('Main:', m.compute(g.main))
"
```

Expected: Returns vague_word_hits and sentence_repeat_3plus counts (integers >= 0).

- [ ] **Step 3: Commit**

```bash
git add metrics/content_quality.py
git commit -m "feat: add ContentQualityMetric - vague words, repeated sentences"
```

---

### Task 7: LLM Judge

**Files:**
- Create: `llm_judge.py`

- [ ] **Step 1: Write llm_judge.py**

Create `llm_judge.py`:
```python
"""LLM-based risk judgment via claude -p."""

import json
import subprocess
import sys


def _build_prompt(
    baseline_name: str,
    baseline_metrics: dict,
    upgraded_name: str,
    upgraded_metrics: dict,
    baseline_log_path: str,
    upgraded_log_path: str,
) -> str:
    return f"""你是一个 Agent 日志质量检测专家。请对比以下两组日志的指标数据，判断模型升级后是否存在退化风险。

## Baseline 日志: {baseline_name}
路径: {baseline_log_path}
指标数据:
{json.dumps(baseline_metrics, indent=2, ensure_ascii=False)}

## Upgraded 日志: {upgraded_name}
路径: {upgraded_log_path}
指标数据:
{json.dumps(upgraded_metrics, indent=2, ensure_ascii=False)}

## 请分析以下方面并返回 JSON 格式的结果:

1. **overall_risk**: 整体退化风险等级，取值 "高"/"中"/"低"
2. **indicator_risks**: 各指标的退化风险判断，格式为 {{"指标key": {{"risk": "高/中/低", "reason": "原因"}}}}
3. **suggestions**: 处理建议列表，格式为 ["建议1", "建议2", ...]

关注以下退化信号:
- token使用量大幅增加（资源浪费）
- 工具调用失败率上升
- 深度思考减少（可能意味着推理质量下降）
- 模糊不确定性词汇增多（输出信心下降）
- 句子重复出现（陷入循环）

请只返回 JSON，不要返回其他内容。"""


def judge(
    baseline_name: str,
    baseline_metrics: dict,
    upgraded_name: str,
    upgraded_metrics: dict,
    baseline_log_path: str,
    upgraded_log_path: str,
) -> dict | None:
    """Call claude -p to get risk judgment. Returns parsed JSON or None on failure."""
    prompt = _build_prompt(
        baseline_name, baseline_metrics,
        upgraded_name, upgraded_metrics,
        baseline_log_path, upgraded_log_path,
    )

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json",
             "--dangerously-skip-permissions"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"claude -p failed: {result.stderr}", file=sys.stderr)
            return None

        # The --output-format json wraps the result in a JSON envelope
        raw = result.stdout.strip()
        try:
            envelope = json.loads(raw)
            # Extract the text content from the envelope
            if isinstance(envelope, dict):
                text = envelope.get("result", raw)
            else:
                text = raw
        except json.JSONDecodeError:
            text = raw

        # Parse the LLM's JSON output
        # Try to find JSON in the text (might have markdown fences)
        json_match = text
        if "```json" in text:
            json_match = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_match = text.split("```")[1].split("```")[0]

        return json.loads(json_match.strip())

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"LLM judge error: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 2: Verify llm_judge can be imported without error**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from llm_judge import _build_prompt
p = _build_prompt('glm5.1', {'resource': {'total_steps': 40}}, 'minimax2.7', {'resource': {'total_steps': 34}}, '/path/a', '/path/b')
print('Prompt length:', len(p))
print('Has JSON instruction:', 'JSON' in p)
"
```

Expected: Prompt length > 200, "Has JSON instruction: True"

- [ ] **Step 3: Commit**

```bash
git add llm_judge.py
git commit -m "feat: add LLM judge module with claude -p integration"
```

---

### Task 8: HTML Reporter

**Files:**
- Create: `reporter.py`

- [ ] **Step 1: Write reporter.py**

Create `reporter.py`:
```python
"""HTML report generator for agent log quality comparison."""

import html
import json
from datetime import datetime


def _delta_str(baseline_val, upgraded_val, higher_is_worse=True) -> str:
    """Compute change percentage and CSS class for a metric."""
    if baseline_val is None or upgraded_val is None:
        return "<td>-</td><td>-</td>"
    if isinstance(baseline_val, dict) or isinstance(upgraded_val, dict):
        return "<td>-</td><td>-</td>"

    try:
        b = float(baseline_val)
        u = float(upgraded_val)
    except (TypeError, ValueError):
        return f"<td>{html.escape(str(upgraded_val))}</td><td>-</td>"

    if b == 0:
        if u == 0:
            css = "neutral"
            pct = "0%"
        else:
            css = "worse" if higher_is_worse else "better"
            pct = "+∞"
    else:
        change = (u - b) / abs(b) * 100
        if abs(change) < 5:
            css = "neutral"
        elif (change > 0) == higher_is_worse:
            css = "worse"
        else:
            css = "better"
        pct = f"{change:+.1f}%"

    return f'<td class="{css}">{html.escape(str(u))}</td><td class="{css}">{pct}</td>'


def _metric_table(
    title: str,
    metrics_baseline: dict,
    metrics_upgraded: dict,
    metric_labels: dict,
    higher_is_worse_map: dict,
) -> str:
    """Generate an HTML table for a category of metrics."""
    rows = []
    for key, label in metric_labels.items():
        if key in metrics_baseline or key in metrics_upgraded:
            b_val = metrics_baseline.get(key, "-")
            u_val = metrics_upgraded.get(key, "-")
            hiw = higher_is_worse_map.get(key, True)
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(b_val))}</td>"
                f"{_delta_str(b_val, u_val, hiw)}</tr>"
            )

    return f"""<h3>{html.escape(title)}</h3>
<table>
<tr><th>指标</th><th>Baseline</th><th>Upgraded</th><th>变化幅度</th></tr>
{"".join(rows)}
</table>"""


def _risk_badge(risk: str) -> str:
    colors = {"高": "#e74c3c", "中": "#f39c12", "低": "#27ae60"}
    color = colors.get(risk, "#95a5a6")
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:4px;font-weight:bold;">{html.escape(risk)}</span>'


_METRIC_CONFIG = {
    "resource": {
        "title": "资源消耗",
        "labels": {
            "total_input_tokens": "输入Token总量",
            "total_output_tokens": "输出Token总量",
            "session_duration_sec": "会话时长(秒)",
            "tokens_per_second": "每秒输出Token数",
            "total_steps": "执行总步骤数",
        },
        "higher_is_worse": {
            "total_input_tokens": True,
            "total_output_tokens": True,
            "session_duration_sec": True,
            "tokens_per_second": False,
            "total_steps": True,
        },
    },
    "tool": {
        "title": "工具调用",
        "labels": {
            "tool_call_count": "工具调用次数",
            "tool_error_count": "调用失败次数",
            "tool_error_rate": "失败占比",
            "subagent_count": "子Agent调用数",
        },
        "higher_is_worse": {
            "tool_call_count": True,
            "tool_error_count": True,
            "tool_error_rate": True,
            "subagent_count": True,
        },
    },
    "thinking": {
        "title": "深度思考",
        "labels": {
            "thinking_count": "Think次数",
            "thinking_total_chars": "Think总字符数",
            "thinking_avg_chars": "Think平均字符数",
        },
        "higher_is_worse": {
            "thinking_count": False,
            "thinking_total_chars": False,
            "thinking_avg_chars": False,
        },
    },
    "content": {
        "title": "内容质量",
        "labels": {
            "vague_word_hits": "模糊词汇命中次数",
            "sentence_repeat_3plus": "重复3次以上句子数",
        },
        "higher_is_worse": {
            "vague_word_hits": True,
            "sentence_repeat_3plus": True,
        },
    },
}


def generate_report(
    baseline_name: str,
    baseline_main_metrics: dict,
    baseline_sub_metrics: list[dict],
    upgraded_name: str,
    upgraded_main_metrics: dict,
    upgraded_sub_metrics: list[dict],
    baseline_log_path: str,
    upgraded_log_path: str,
    llm_result: dict | None,
    sub_metas_baseline: list[dict],
    sub_metas_upgraded: list[dict],
) -> str:
    """Generate a self-contained HTML report."""

    # Main agent comparison tables
    main_tables = []
    for cat, cfg in _METRIC_CONFIG.items():
        b_cat = baseline_main_metrics.get(cat, {})
        u_cat = upgraded_main_metrics.get(cat, {})
        main_tables.append(
            _metric_table(
                f"主Agent - {cfg['title']}", b_cat, u_cat,
                cfg["labels"], cfg["higher_is_worse"],
            )
        )

    # Sub-agent sections
    sub_sections = []
    max_subs = max(len(baseline_sub_metrics), len(upgraded_sub_metrics))
    for i in range(max_subs):
        b_sub = baseline_sub_metrics[i] if i < len(baseline_sub_metrics) else {}
        u_sub = upgraded_sub_metrics[i] if i < len(upgraded_sub_metrics) else {}
        b_meta = sub_metas_baseline[i] if i < len(sub_metas_baseline) else {}
        u_meta = sub_metas_upgraded[i] if i < len(sub_metas_upgraded) else {}

        b_desc = b_meta.get("description", f"子Agent #{i+1}")
        u_desc = u_meta.get("description", f"子Agent #{i+1}")
        meta_html = f"<p>Baseline: {html.escape(b_desc)} | Upgraded: {html.escape(u_desc)}</p>"

        tables = []
        for cat, cfg in _METRIC_CONFIG.items():
            b_cat = b_sub.get(cat, {})
            u_cat = u_sub.get(cat, {})
            tables.append(
                _metric_table(
                    f"子Agent #{i+1} - {cfg['title']}", b_cat, u_cat,
                    cfg["labels"], cfg["higher_is_worse"],
                )
            )
        sub_sections.append(f'<div class="subagent-section"><h2>子Agent #{i+1}</h2>{meta_html}{"".join(tables)}</div>')

    # LLM analysis section
    llm_html = ""
    if llm_result:
        risk = llm_result.get("overall_risk", "未知")
        indicator_risks = llm_result.get("indicator_risks", {})
        suggestions = llm_result.get("suggestions", [])

        risk_rows = []
        for k, v in indicator_risks.items():
            risk_rows.append(
                f"<tr><td>{html.escape(k)}</td>"
                f"<td>{_risk_badge(v.get('risk', '未知'))}</td>"
                f"<td>{html.escape(v.get('reason', ''))}</td></tr>"
            )

        suggestion_items = "".join(f"<li>{html.escape(s)}</li>" for s in suggestions)

        llm_html = f"""<div class="llm-section">
<h2>退化风险分析</h2>
<p>整体风险等级: {_risk_badge(risk)}</p>
<h3>各指标风险判断</h3>
<table>
<tr><th>指标</th><th>风险等级</th><th>原因</th></tr>
{"".join(risk_rows)}
</table>
<h3>处理建议</h3>
<ol>{suggestion_items}</ol>
</div>"""
    else:
        llm_html = '<div class="llm-section"><h2>退化风险分析</h2><p class="error">LLM判断调用失败，请检查claude命令是否可用。</p></div>'

    # Assemble full HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent日志质量对比报告</title>
<style>
body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
h3 {{ color: #7f8c8d; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #2c3e50; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.better {{ color: #27ae60; font-weight: bold; }}
.worse {{ color: #e74c3c; font-weight: bold; }}
.neutral {{ color: #7f8c8d; }}
.subagent-section {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.llm-section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid #3498db; }}
.error {{ color: #e74c3c; font-style: italic; }}
.overview {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
.overview p {{ margin: 5px 0; }}
.main-section {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>Agent日志质量对比报告</h1>
<p style="color:#95a5a6;">生成时间: {html.escape(now)}</p>

<div class="overview">
<h2>概览</h2>
<p><strong>Baseline:</strong> {html.escape(baseline_name)} ({html.escape(baseline_log_path)})</p>
<p><strong>Upgraded:</strong> {html.escape(upgraded_name)} ({html.escape(upgraded_log_path)})</p>
{f'<p>整体风险等级: {_risk_badge(llm_result.get("overall_risk", "未知"))}</p>' if llm_result else ''}
</div>

<div class="main-section">
<h2>主Agent指标对比</h2>
{"".join(main_tables)}
</div>

<h2>子Agent指标对比</h2>
{"".join(sub_sections)}

{llm_html}

</body>
</html>"""
```

- [ ] **Step 2: Verify reporter generates valid HTML**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
import sys; sys.path.insert(0, '.')
from reporter import generate_report
html = generate_report(
    'glm5.1', {'resource': {'total_steps': 40}}, [],
    'minimax2.7', {'resource': {'total_steps': 34}}, [],
    '/path/a', '/path/b', None, [], []
)
print('HTML length:', len(html))
print('Has DOCTYPE:', '<!DOCTYPE html>' in html)
print('Has table:', '<table>' in html)
"
```

Expected: HTML length > 1000, has DOCTYPE and table tags.

- [ ] **Step 3: Commit**

```bash
git add reporter.py
git commit -m "feat: add HTML report generator with comparison tables"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

Create `main.py`:
```python
"""CLI entry point for Agent Log Quality Checker."""

import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_log_dir
from pipeline import MetricPipeline
from metrics import ResourceMetric, ToolUsageMetric, ThinkingMetric, ContentQualityMetric
from llm_judge import judge
from reporter import generate_report


def main():
    parser = argparse.ArgumentParser(description="Agent运行日志质量检测器")
    parser.add_argument("--baseline", required=True, help="Baseline日志目录路径")
    parser.add_argument("--upgraded", required=True, help="Upgraded日志目录路径")
    parser.add_argument("--output", default="report.html", help="输出HTML报告路径")
    parser.add_argument("--skip-llm", action="store_true", help="跳过LLM判断步骤")
    args = parser.parse_args()

    # Parse logs
    print(f"[1/4] 解析Baseline日志: {args.baseline}")
    baseline_group = parse_log_dir(args.baseline)
    print(f"  主Agent: {len(baseline_group.main.messages)} 条消息, {len(baseline_group.subagents)} 个子Agent")

    print(f"[2/4] 解析Upgraded日志: {args.upgraded}")
    upgraded_group = parse_log_dir(args.upgraded)
    print(f"  主Agent: {len(upgraded_group.main.messages)} 条消息, {len(upgraded_group.subagents)} 个子Agent")

    # Compute metrics
    pipeline = MetricPipeline()
    pipeline.register(ResourceMetric())
    pipeline.register(ToolUsageMetric())
    pipeline.register(ThinkingMetric())
    pipeline.register(ContentQualityMetric())

    baseline_main_metrics = pipeline.run(baseline_group.main)
    upgraded_main_metrics = pipeline.run(upgraded_group.main)

    baseline_sub_metrics = [pipeline.run(s["session"]) for s in baseline_group.subagents]
    upgraded_sub_metrics = [pipeline.run(s["session"]) for s in upgraded_group.subagents]

    baseline_sub_metas = [s["meta"] for s in baseline_group.subagents]
    upgraded_sub_metas = [s["meta"] for s in upgraded_group.subagents]

    # LLM judgment
    llm_result = None
    if not args.skip_llm:
        print("[3/4] 调用LLM进行退化风险判断...")
        llm_result = judge(
            baseline_name=args.baseline,
            baseline_metrics={
                "main": baseline_main_metrics,
                "subagents": baseline_sub_metrics,
            },
            upgraded_name=args.upgraded,
            upgraded_metrics={
                "main": upgraded_main_metrics,
                "subagents": upgraded_sub_metrics,
            },
            baseline_log_path=os.path.abspath(args.baseline),
            upgraded_log_path=os.path.abspath(args.upgraded),
        )
        if llm_result:
            print(f"  整体风险等级: {llm_result.get('overall_risk', '未知')}")
        else:
            print("  LLM判断失败")
    else:
        print("[3/4] 跳过LLM判断 (--skip-llm)")

    # Generate report
    print(f"[4/4] 生成HTML报告: {args.output}")
    report_html = generate_report(
        baseline_name=args.baseline,
        baseline_main_metrics=baseline_main_metrics,
        baseline_sub_metrics=baseline_sub_metrics,
        upgraded_name=args.upgraded,
        upgraded_main_metrics=upgraded_main_metrics,
        upgraded_sub_metrics=upgraded_sub_metrics,
        baseline_log_path=os.path.abspath(args.baseline),
        upgraded_log_path=os.path.abspath(args.upgraded),
        llm_result=llm_result,
        sub_metas_baseline=baseline_sub_metas,
        sub_metas_upgraded=upgraded_sub_metas,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end with --skip-llm**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python main.py --baseline sample/log/log_glm5.1_20260505 --upgraded sample/log/log_minimax2.7_20260505 --output report.html --skip-llm
```

Expected: Report generated at `report.html` with metrics data in tables.

- [ ] **Step 3: Open the HTML report and verify it renders correctly**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
with open('report.html', 'r', encoding='utf-8') as f:
    content = f.read()
print('File size:', len(content))
print('Has baseline data:', 'glm5.1' in content or 'log_glm5.1' in content)
print('Has metric tables:', 'total_input_tokens' in content)
print('Has color coding:', 'worse' in content or 'better' in content)
"
```

Expected: File size > 2000, all checks True.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with full pipeline orchestration"
```

---

### Task 10: End-to-End Test with LLM Judge

**Files:**
- No new files (integration test of existing code)

- [ ] **Step 1: Run full pipeline including LLM judge**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python main.py --baseline sample/log/log_glm5.1_20260505 --upgraded sample/log/log_minimax2.7_20260505 --output report.html
```

Expected: All 4 steps complete, LLM returns risk assessment, HTML has structured analysis section.

- [ ] **Step 2: Verify the HTML report has LLM analysis content**

Run:
```bash
cd C:/Users/47912/Desktop/agent_log_check && python -c "
with open('report.html', 'r', encoding='utf-8') as f:
    content = f.read()
print('Has LLM section:', '退化风险分析' in content)
print('Has risk badge:', 'risk' in content.lower() or '风险' in content)
print('Has suggestions:', '建议' in content or 'suggestion' in content.lower())
"
```

Expected: All True.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Agent Log Quality Checker with LLM judgment"
```

---

## Self-Review

**1. Spec coverage:**
- Data models (LogSession, Message, ContentBlock, TokenUsage, LogGroup) → Task 1
- Parser (parse_log_dir, JSONL parsing, subagent handling) → Task 1
- BaseMetric ABC → Task 2
- MetricPipeline → Task 2
- ResourceMetric (5 indicators) → Task 3
- ToolUsageMetric (5 indicators) → Task 4
- ThinkingMetric (3 indicators) → Task 5
- ContentQualityMetric (2 indicators) → Task 6
- LLM judge (claude -p, structured JSON) → Task 7
- HTML reporter (overview, main comparison, sub-agent, LLM analysis, no raw output) → Task 8
- CLI entry point → Task 9
- End-to-end test → Task 10
- All edge cases from spec → handled in each metric's zero-division guards

**2. Placeholder scan:** No TBD, TODO, "implement later", or vague steps. All code blocks are complete.

**3. Type consistency:** All metric classes use `compute(self, session: LogSession) -> dict` matching BaseMetric. Pipeline.run() takes LogSession matching what parser returns. Reporter function signatures match main.py's call sites.
