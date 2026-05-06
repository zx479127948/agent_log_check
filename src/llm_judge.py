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

    tpl = string.Template(template)
    return tpl.safe_substitute(
        baseline_name=baseline_name,
        baseline_log_path=baseline_log_path,
        baseline_metrics=json.dumps(baseline_metrics, indent=2, ensure_ascii=False),
        upgraded_name=upgraded_name,
        upgraded_log_path=upgraded_log_path,
        upgraded_metrics=json.dumps(upgraded_metrics, indent=2, ensure_ascii=False),
    )


def _parse_risk_level(md: str) -> str:
    """Extract overall risk level from markdown."""
    m = re.search(r"## 整体风险等级\s*\n\s*(高|中|低)", md)
    return m.group(1) if m else "未知"


def _parse_action(md: str) -> str:
    """Extract action recommendation from markdown."""
    m = re.search(r"## 行动建议\s*\n\s*(建议上线|建议回滚|需要更多测试)", md)
    return m.group(1) if m else ""


def _parse_indicator_risks(md: str) -> dict:
    """Parse per-indicator risk levels from the markdown tables."""
    risks = {}
    rows = re.findall(
        r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(高|中|低)\s*\|\s*(.+?)\s*\|",
        md,
    )
    for row in rows:
        indicator = row[0].strip()
        risk_level = row[4].strip()
        analysis = row[5].strip()
        if indicator and risk_level:
            risks[indicator] = {"risk": risk_level, "analysis": analysis}
    return risks


def judge(
    baseline_name: str,
    baseline_metrics: dict,
    upgraded_name: str,
    upgraded_metrics: dict,
    baseline_log_path: str,
    upgraded_log_path: str,
    output_dir: str,
    template_path: str | None = None,
) -> dict | None:
    """Call claude -p, save markdown to output_dir/analysis.md, return parsed fields."""
    prompt = _build_prompt(
        baseline_name, baseline_metrics,
        upgraded_name, upgraded_metrics,
        baseline_log_path, upgraded_log_path,
        template_path=template_path,
    )

    claude_cmd = "claude"
    if sys.platform == "win32":
        claude_cmd = shutil.which("claude") or "claude"

    md_path = os.path.join(output_dir, ANALYSIS_MD_FILENAME)

    try:
        result = subprocess.run(
            [claude_cmd, "-p", "--dangerously-skip-permissions"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")
            print(f"claude -p failed: {stderr_text}", file=sys.stderr)
            return None

        raw = result.stdout.decode("utf-8", errors="replace").strip()

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

        return {
            "overall_risk": overall_risk,
            "action": action,
            "indicator_risks": indicator_risks,
            "analysis_md_path": md_path,
        }

    except subprocess.TimeoutExpired:
        print("LLM judge error: timed out after 300s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"LLM judge error: {e}", file=sys.stderr)
        return None


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
        "analysis_md_path": md_path,
    }
