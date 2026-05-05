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