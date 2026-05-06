"""LLM-based risk judgment via claude -p.

Calls claude -p, saves the markdown output to a file, and parses key fields.
"""

import json
import os
import re
import shutil
import string
import subprocess
import sys

_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts", "judge_prompt.md"
)

ANALYSIS_MD_FILENAME = "analysis.md"

_CN_TO_EN_KEY = {
    "输入Token总量": "total_input_tokens",
    "输出Token总量": "total_output_tokens",
    "会话时长(秒)": "session_duration_sec",
    "每秒输出Token数": "tokens_per_second",
    "执行总步骤数": "total_steps",
    "工具调用次数": "tool_call_count",
    "调用失败次数": "tool_error_count",
    "失败占比": "tool_error_rate",
    "子Agent调用数": "subagent_count",
    "Think次数": "thinking_count",
    "Think总字符数": "thinking_total_chars",
    "Think平均字符数": "thinking_avg_chars",
    "模糊词汇命中次数": "vague_word_hits",
    "重复3次以上句子数": "sentence_repeat_3plus",
}


def _build_subagent_sections(baseline_metrics: dict, upgraded_metrics: dict) -> str:
    """Build the per-sub-agent analysis template sections."""
    b_subs = baseline_metrics.get("subagents", [])
    u_subs = upgraded_metrics.get("subagents", [])
    max_subs = max(len(b_subs), len(u_subs))

    sections = []
    for i in range(max_subs):
        b_data = b_subs[i] if i < len(b_subs) else {}
        u_data = u_subs[i] if i < len(u_subs) else {}
        sections.append(
            f"### 子Agent #{i+1}\n\n"
            f"[1-2句总结该子Agent的关键变化]\n\n"
            f"| 指标 | Baseline | Upgraded | 变化幅度 | 风险等级 | 分析说明 |\n"
            f"|------|----------|----------|----------|----------|----------|\n"
            f"| 输入Token总量 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 输出Token总量 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 会话时长(秒) | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 每秒输出Token数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 执行总步骤数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 工具调用次数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 调用失败次数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 失败占比 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| Think次数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| Think总字符数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| Think平均字符数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 模糊词汇命中次数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n"
            f"| 重复3次以上句子数 | 值 | 值 | +X% | 高/中/低 | 详细分析 |\n\n"
            f"Baseline子Agent指标数据:\n```json\n"
            f"{json.dumps(b_data, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Upgraded子Agent指标数据:\n```json\n"
            f"{json.dumps(u_data, indent=2, ensure_ascii=False)}\n```"
        )
    return "\n\n".join(sections)


def _build_prompt(
    baseline_name: str,
    baseline_metrics: dict,
    upgraded_name: str,
    upgraded_metrics: dict,
    baseline_log_path: str,
    upgraded_log_path: str,
    template_path: str | None = None,
) -> str:
    """Build the LLM prompt from a markdown template file."""
    tpl_path = template_path or _PROMPT_TEMPLATE_PATH
    with open(tpl_path, "r", encoding="utf-8") as f:
        template = f.read()

    subagent_sections = _build_subagent_sections(baseline_metrics, upgraded_metrics)

    tpl = string.Template(template)
    return tpl.safe_substitute(
        baseline_name=baseline_name,
        baseline_log_path=baseline_log_path,
        baseline_metrics=json.dumps(baseline_metrics.get("main", baseline_metrics), indent=2, ensure_ascii=False),
        upgraded_name=upgraded_name,
        upgraded_log_path=upgraded_log_path,
        upgraded_metrics=json.dumps(upgraded_metrics.get("main", upgraded_metrics), indent=2, ensure_ascii=False),
        subagent_sections=subagent_sections,
    )


def _parse_risk_level(md: str) -> str:
    """Extract overall risk level from markdown."""
    m = re.search(r"## 整体风险等级\s*\n\s*(高|中|低)", md)
    return m.group(1) if m else "未知"


def _parse_action(md: str) -> str:
    """Extract action recommendation from markdown."""
    m = re.search(r"## 行动建议\s*\n\s*(建议上线|建议回滚|需要更多测试)", md)
    return m.group(1) if m else ""


def _parse_table_rows(md: str) -> list[dict]:
    """Parse markdown table rows by splitting on |, returning list of {indicator, risk, analysis}."""
    results = []
    for line in md.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        # Skip separator rows and header rows
        if len(cells) < 6:
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        # Skip header rows (first cell is "指标")
        if cells[0] == "指标":
            continue
        risk_level = cells[4].strip()
        if risk_level not in ("高", "中", "低"):
            continue
        results.append({
            "indicator": cells[0].strip(),
            "risk": risk_level,
            "analysis": cells[5].strip(),
        })
    return results


def _parse_indicator_risks(md: str) -> dict:
    """Parse per-indicator risk levels from the main agent markdown tables only."""
    risks = {}
    # Only parse tables before the sub-agent section
    main_md = md.split("## 子Agent分析")[0] if "## 子Agent分析" in md else md
    for row in _parse_table_rows(main_md):
        en_key = _CN_TO_EN_KEY.get(row["indicator"], row["indicator"])
        risks[en_key] = {"risk": row["risk"], "analysis": row["analysis"]}
    return risks


def _parse_subagent_risks(md: str) -> list[dict]:
    """Parse per-sub-agent indicator risks from the markdown."""
    sub_risks = []
    # Split by sub-agent section headers like "### 子Agent #1" or "## 子Agent #1"
    parts = re.split(r"(?:#{2,3})\s*子Agent\s*#(\d+)", md)
    # parts: [before, "1", content1, "2", content2, ...]
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1  # 0-based
        content = parts[i + 1] if i + 1 < len(parts) else ""

        sub_indicator_risks = {}
        for row in _parse_table_rows(content):
            en_key = _CN_TO_EN_KEY.get(row["indicator"], row["indicator"])
            sub_indicator_risks[en_key] = {"risk": row["risk"], "analysis": row["analysis"]}

        # Extract sub-agent summary (text before the first table)
        summary_match = re.match(r"\s*\n+(.*?)(?=\n\s*\|)", content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Ensure list is long enough
        while len(sub_risks) <= idx:
            sub_risks.append({"indicator_risks": {}, "summary": ""})
        sub_risks[idx] = {"indicator_risks": sub_indicator_risks, "summary": summary}
    return sub_risks


def _find_claude_cmd() -> str | None:
    """Find the claude command, trying multiple approaches."""
    # 1. Direct which/where lookup
    found = shutil.which("claude")
    if found:
        return found
    # 2. Common Windows install paths
    if sys.platform == "win32":
        home = os.environ.get("USERPROFILE", "")
        appdata = os.environ.get("APPDATA", "")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            os.path.join(appdata, "npm", "claude.cmd"),
            os.path.join(local_appdata, "npm", "claude.cmd"),
            os.path.join(home, "AppData", "Roaming", "npm", "claude.cmd"),
            os.path.join(home, "AppData", "Local", "npm", "claude.cmd"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    # 3. Try npx
    if shutil.which("npx"):
        return "npx"
    return None


def judge(
    baseline_name: str,
    baseline_metrics: dict,
    upgraded_name: str,
    upgraded_metrics: dict,
    baseline_log_path: str,
    upgraded_log_path: str,
    output_dir: str,
    template_path: str | None = None,
    max_retries: int = 2,
    timeout: int = 600,
) -> dict | None:
    """Call claude -p, save markdown to output_dir/analysis.md, return parsed fields."""
    prompt = _build_prompt(
        baseline_name, baseline_metrics,
        upgraded_name, upgraded_metrics,
        baseline_log_path, upgraded_log_path,
        template_path=template_path,
    )

    md_path = os.path.join(output_dir, ANALYSIS_MD_FILENAME)

    # Find claude command
    claude_cmd = _find_claude_cmd()
    if not claude_cmd:
        print("  未找到claude命令，尝试使用缓存的分析结果...", file=sys.stderr)
        return load_analysis(output_dir)

    # Build command arguments
    if claude_cmd == "npx":
        cmd = ["npx", "@anthropic-ai/claude-code", "-p", "--dangerously-skip-permissions"]
    else:
        cmd = [claude_cmd, "-p", "--dangerously-skip-permissions"]

    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                import time
                wait = 10 * attempt
                print(f"  第{attempt}次重试 (等待{wait}秒)...", file=sys.stderr)
                time.sleep(wait)

            print(f"  调用claude -p (尝试 {attempt}/{max_retries})...")
            result = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8", errors="replace")
                print(f"  claude -p 返回非零 ({result.returncode}): {stderr_text[:500]}", file=sys.stderr)
                if attempt < max_retries:
                    continue
                # Last attempt failed — try loading cached
                print("  所有重试失败，尝试使用缓存的分析结果...", file=sys.stderr)
                return load_analysis(output_dir)

            raw = result.stdout.decode("utf-8", errors="replace").strip()

            # Validate output is not empty
            if not raw or len(raw) < 50:
                print(f"  claude -p 输出为空或过短 ({len(raw)}字符)", file=sys.stderr)
                if attempt < max_retries:
                    continue
                return load_analysis(output_dir)

            # Strip markdown code fences if present
            if raw.startswith("```markdown"):
                raw = raw[len("```markdown"):]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            # Save to file
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(raw)

            # Parse key fields
            overall_risk = _parse_risk_level(raw)
            action = _parse_action(raw)
            indicator_risks = _parse_indicator_risks(raw)
            subagent_risks = _parse_subagent_risks(raw)

            return {
                "overall_risk": overall_risk,
                "action": action,
                "indicator_risks": indicator_risks,
                "subagent_risks": subagent_risks,
                "analysis_md_path": md_path,
            }

        except subprocess.TimeoutExpired:
            print(f"  claude -p 超时 ({timeout}秒)，尝试 {attempt}/{max_retries}", file=sys.stderr)
            if attempt < max_retries:
                continue
            print("  所有重试超时，尝试使用缓存的分析结果...", file=sys.stderr)
            return load_analysis(output_dir)

        except FileNotFoundError:
            print(f"  命令不存在: {cmd[0]}", file=sys.stderr)
            return load_analysis(output_dir)

        except Exception as e:
            print(f"  claude -p 异常: {e}", file=sys.stderr)
            if attempt < max_retries:
                continue
            return load_analysis(output_dir)

    return load_analysis(output_dir)


def load_analysis(output_dir: str) -> dict | None:
    """Load a previously saved analysis.md and parse it. Returns dict or None."""
    md_path = os.path.join(output_dir, ANALYSIS_MD_FILENAME)
    if not os.path.exists(md_path):
        return None
    with open(md_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return {
        "overall_risk": _parse_risk_level(raw),
        "action": _parse_action(raw),
        "indicator_risks": _parse_indicator_risks(raw),
        "subagent_risks": _parse_subagent_risks(raw),
        "analysis_md_path": md_path,
    }
