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