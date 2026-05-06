"""HTML report generator for agent log quality comparison."""

import html
import json
import os
import re
from datetime import datetime


def _markdown_to_html(md: str) -> str:
    """Minimal markdown-to-HTML converter (tables, headers, paragraphs, bold, lists)."""
    lines = md.split("\n")
    out = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            tag = "th" if not in_table else "td"
            row = "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
            if not in_table:
                out.append("<table>")
                in_table = True
                row = row.replace("<td>", "<th>").replace("</td>", "</th>")
            out.append(row)
            continue
        else:
            if in_table:
                out.append("</table>")
                in_table = False

        if stripped.startswith("### "):
            out.append(f"<h4>{html.escape(stripped[4:])}</h4>")
        elif stripped.startswith("## "):
            out.append(f"<h3>{html.escape(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            out.append(f"<h2>{html.escape(stripped[2:])}</h2>")
        elif stripped.startswith("- "):
            out.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped:
            escaped = html.escape(stripped)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            out.append(f"<p>{escaped}</p>")

    if in_table:
        out.append("</table>")

    return "\n".join(out)


def _delta_cell(baseline_val, upgraded_val, higher_is_worse=True) -> str:
    """Compute change percentage and return a single <td> with CSS class."""
    if baseline_val is None or upgraded_val is None:
        return '<td class="neutral">-</td>'
    if isinstance(baseline_val, dict) or isinstance(upgraded_val, dict):
        return '<td class="neutral">-</td>'

    try:
        b = float(baseline_val)
        u = float(upgraded_val)
    except (TypeError, ValueError):
        return f'<td class="neutral">{html.escape(str(upgraded_val))}</td>'

    if b == 0:
        if u == 0:
            return '<td class="neutral">0%</td>'
        css = "worse" if higher_is_worse else "better"
        return f'<td class="{css}">+∞</td>'

    change = (u - b) / abs(b) * 100
    if abs(change) < 5:
        css = "neutral"
    elif (change > 0) == higher_is_worse:
        css = "worse"
    else:
        css = "better"
    return f'<td class="{css}">{change:+.1f}%</td>'


def _risk_badge(risk: str) -> str:
    colors = {"高": "#e74c3c", "中": "#f39c12", "低": "#27ae60"}
    color = colors.get(risk, "#95a5a6")
    return f'<span class="risk-badge" style="background:{color}">{html.escape(risk)}</span>'


_METRIC_CONFIG = {
    "resource": {
        "title": "资源消耗",
        "icon": "&#9889;",
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
        "icon": "&#128295;",
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
        "icon": "&#129504;",
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
        "icon": "&#128221;",
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


def _metric_table(
    title: str,
    icon: str,
    metrics_baseline: dict,
    metrics_upgraded: dict,
    metric_labels: dict,
    higher_is_worse_map: dict,
    indicator_risks: dict | None = None,
) -> str:
    """Generate an HTML table for a category of metrics, with optional AI analysis column."""
    has_ai = bool(indicator_risks)
    rows = []
    for key, label in metric_labels.items():
        if key in metrics_baseline or key in metrics_upgraded:
            b_val = metrics_baseline.get(key, "-")
            u_val = metrics_upgraded.get(key, "-")
            hiw = higher_is_worse_map.get(key, True)

            ai_cell = ""
            if has_ai:
                risk_info = indicator_risks.get(key, {})
                risk_level = risk_info.get("risk", "")
                risk_analysis = risk_info.get("analysis", risk_info.get("reason", ""))
                badge = _risk_badge(risk_level) if risk_level else ""
                ai_cell = f"<td>{badge} {html.escape(risk_analysis)}</td>"

            rows.append(
                f"<tr>"
                f"<td class='metric-name'>{html.escape(label)}</td>"
                f"<td class='val'>{html.escape(str(b_val))}</td>"
                f"<td class='val'>{html.escape(str(u_val))}</td>"
                f"{_delta_cell(b_val, u_val, hiw)}"
                f"{ai_cell}"
                f"</tr>"
            )

    ai_header = "<th>AI分析</th>" if has_ai else ""
    return f"""<div class="card">
<h3>{icon} {html.escape(title)}</h3>
<table>
<tr><th>指标</th><th>Baseline</th><th>Upgraded</th><th>变化幅度</th>{ai_header}</tr>
{"".join(rows)}
</table>
</div>"""


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

    # Extract indicator_risks from LLM result for per-cell analysis
    indicator_risks = llm_result.get("indicator_risks", {}) if llm_result else {}

    # Main agent comparison tables (with AI analysis column)
    main_tables = []
    for cat, cfg in _METRIC_CONFIG.items():
        b_cat = baseline_main_metrics.get(cat, {})
        u_cat = upgraded_main_metrics.get(cat, {})
        main_tables.append(
            _metric_table(
                cfg["title"], cfg["icon"], b_cat, u_cat,
                cfg["labels"], cfg["higher_is_worse"],
                indicator_risks=indicator_risks,
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

        tables = []
        for cat, cfg in _METRIC_CONFIG.items():
            b_cat = b_sub.get(cat, {})
            u_cat = u_sub.get(cat, {})
            tables.append(
                _metric_table(
                    cfg["title"], cfg["icon"], b_cat, u_cat,
                    cfg["labels"], cfg["higher_is_worse"],
                )
            )
        sub_sections.append(
            f'<div class="card"><h2>子Agent #{i+1}</h2>'
            f'<p class="meta">Baseline: {html.escape(b_desc)} | Upgraded: {html.escape(u_desc)}</p>'
            f'{"".join(tables)}</div>'
        )

    # LLM analysis section
    llm_html = ""
    summary_html = ""
    if llm_result:
        risk = llm_result.get("overall_risk", "未知")
        action = llm_result.get("action", "")
        md_path = llm_result.get("analysis_md_path", "")

        # Read the analysis markdown file
        analysis_md = ""
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                analysis_md = f.read()

        # Extract overall summary from the markdown
        overall_summary = ""
        if analysis_md and "整体总结" in analysis_md:
            parts = analysis_md.split("整体总结")
            if len(parts) > 1:
                summary_part = parts[-1]
                for prefix in ["\n# ", "\n## ", "\n### ", "\n"]:
                    if summary_part.startswith(prefix):
                        summary_part = summary_part[len(prefix):]
                        break
                overall_summary = summary_part.strip()

        # Render full markdown analysis
        analysis_html = _markdown_to_html(analysis_md) if analysis_md else ""

        # Extract suggestions from the markdown
        suggestion_items = ""
        if analysis_md:
            suggestion_lines = re.findall(r"^- (.+)$", analysis_md, re.MULTILINE)
            suggestion_items = "".join(f"<li>{html.escape(s)}</li>" for s in suggestion_lines)

        # Action badge
        action_html = ""
        if action:
            action_colors = {"建议上线": "#27ae60", "建议回滚": "#e74c3c", "需要更多测试": "#f39c12"}
            action_color = action_colors.get(action, "#3498db")
            action_html = f"""<div class="action-badge">
<span style="background:{action_color};color:white;padding:12px 32px;border-radius:8px;font-size:20px;font-weight:bold;display:inline-block;">{html.escape(action)}</span>
</div>"""

        # Summary section (placed at top of report)
        summary_html = f"""<div class="summary-card">
<div class="summary-header">
<h2>分析总结</h2>
<div class="risk-overview">
<span class="risk-label">整体风险等级</span> {_risk_badge(risk)}
</div>
</div>
{f'<div class="summary-text">{_markdown_to_html(overall_summary)}</div>' if overall_summary else ''}
{action_html}
</div>"""

        # Detailed analysis section
        llm_html = f"""<div class="card analysis-card">
<h2>详细分析报告</h2>
{analysis_html}
</div>"""
    else:
        llm_html = '<div class="card"><h2>退化风险分析</h2><p class="error">LLM判断调用失败，请检查claude命令是否可用。</p></div>'

    # Assemble full HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent日志质量对比报告</title>
<style>
:root {{ --primary: #1a73e8; --bg: #f8f9fa; --card-bg: #ffffff; --text: #202124; --text-secondary: #5f6368; --border: #dadce0; --success: #1e8e3e; --warning: #f9ab00; --danger: #d93025; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Microsoft YaHei", "Segoe UI", -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px; background: var(--bg); color: var(--text); line-height: 1.6; }}
h1 {{ color: var(--primary); font-size: 28px; margin-bottom: 4px; }}
h2 {{ color: #202124; font-size: 20px; margin-top: 0; }}
h3 {{ color: var(--text-secondary); font-size: 16px; margin: 16px 0 8px; }}
h4 {{ color: var(--text-secondary); font-size: 14px; margin: 12px 0 6px; }}
.card {{ background: var(--card-bg); border-radius: 12px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06); }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 14px; }}
th, td {{ border: 1px solid var(--border); padding: 10px 14px; text-align: left; }}
th {{ background: #f1f3f4; color: var(--text); font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.02em; }}
tr:hover {{ background: #f8f9fa; }}
.metric-name {{ font-weight: 500; white-space: nowrap; }}
.val {{ font-variant-numeric: tabular-nums; text-align: right; }}
.better {{ color: var(--success); font-weight: 600; }}
.worse {{ color: var(--danger); font-weight: 600; }}
.neutral {{ color: var(--text-secondary); }}
.risk-badge {{ display: inline-block; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 6px; }}
.overview {{ background: var(--card-bg); border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid var(--primary); }}
.overview p {{ margin: 4px 0; color: var(--text-secondary); }}
.overview strong {{ color: var(--text); }}
.summary-card {{ background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%); border-radius: 16px; padding: 24px 28px; margin: 16px 0; box-shadow: 0 2px 8px rgba(26,115,232,0.1); }}
.summary-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.summary-header h2 {{ margin: 0; }}
.risk-overview {{ display: flex; align-items: center; gap: 8px; }}
.risk-label {{ color: var(--text-secondary); font-size: 14px; }}
.summary-text {{ background: white; border-radius: 8px; padding: 16px; margin: 12px 0; line-height: 1.8; }}
.summary-text p {{ margin: 6px 0; }}
.summary-text h2, .summary-text h3, .summary-text h4 {{ margin-top: 8px; }}
.action-badge {{ text-align: center; margin: 16px 0; }}
.analysis-card {{ border-left: 4px solid var(--primary); }}
.analysis-card table {{ margin: 12px 0; }}
.meta {{ color: var(--text-secondary); font-size: 13px; margin-bottom: 12px; }}
.suggestions {{ padding-left: 20px; }}
.suggestions li {{ margin: 6px 0; line-height: 1.6; }}
.error {{ color: var(--danger); font-style: italic; }}
.timestamp {{ color: var(--text-secondary); font-size: 13px; }}
</style>
</head>
<body>
<h1>Agent日志质量对比报告</h1>
<p class="timestamp">生成时间: {html.escape(now)}</p>

<div class="overview">
<h2>概览</h2>
<p><strong>Baseline:</strong> {html.escape(baseline_name)}</p>
<p><strong>Upgraded:</strong> {html.escape(upgraded_name)}</p>
</div>

{summary_html}

<div class="card">
<h2>主Agent指标对比</h2>
{"".join(main_tables)}
</div>

<h2 style="margin-top:24px;padding-left:8px;">子Agent指标对比</h2>
{"".join(sub_sections)}

{llm_html}

</body>
</html>"""
