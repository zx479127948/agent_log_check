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