"""
Benchmark 命令行入口

使用 submitit 调度多个 benchmark 任务到 SLURM。
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import submitit

from .benchmark_config import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelConfig,
    MODEL_PRESETS,
    MODEL_DEVICE_MAP,
    CONTEXT_LENGTHS,
)
from .benchmark_runner import run_benchmark

warnings.filterwarnings("ignore", category=UserWarning, module="submitit")


# ============================================================
# 命令行参数解析
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Benchmark Transformer models with various configurations")

    # 模型和配置选择
    parser.add_argument(
        "--model_sizes",
        type=str,
        default="all",
        help="Model sizes to benchmark: Small,Medium,Large,xl,2.7B (comma-separated or 'all')",
    )
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="all",
        help="Context lengths: 128,256,512,1024 (comma-separated or 'all')",
    )

    # Benchmark 参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--measure_steps",
        type=int,
        default=10,
        help="Number of measurement steps (default: 10)",
    )

    # 模式选择
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["bf16", "fp16"],
        default=None,
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Only run forward pass (inference mode)",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Enable memory profiling (saves .pickle snapshots)",
    )

    # 输出
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark.md",
        help="Output markdown file path",
    )

    return parser.parse_args()


# ============================================================
# 配置构建
# ============================================================


def _parse_model_sizes(sizes_str: str) -> list[str]:
    """解析模型大小参数"""
    if sizes_str.lower() == "all":
        return list(MODEL_PRESETS.keys())

    sizes = [s.strip() for s in sizes_str.split(",")]
    invalid = set(sizes) - set(MODEL_PRESETS.keys())
    if invalid:
        raise ValueError(f"Invalid model sizes: {invalid}. Valid options: {list(MODEL_PRESETS.keys())}")
    return sizes


def _parse_context_lengths(lengths_str: str) -> list[int]:
    """解析 context length 参数"""
    if lengths_str.lower() == "all":
        return CONTEXT_LENGTHS.copy()

    try:
        lengths = [int(s.strip()) for s in lengths_str.split(",")]
    except ValueError:
        raise ValueError(f"Invalid context length format. Must be integers: {CONTEXT_LENGTHS}")

    invalid = set(lengths) - set(CONTEXT_LENGTHS)
    if invalid:
        raise ValueError(f"Invalid context lengths: {invalid}. Valid options: {CONTEXT_LENGTHS}")
    return lengths


def build_configs(args: argparse.Namespace) -> list[BenchmarkConfig]:
    """根据命令行参数构建配置列表"""
    model_sizes = _parse_model_sizes(args.model_sizes)
    context_lengths = _parse_context_lengths(args.context_lengths)

    configs: list[BenchmarkConfig] = []

    # 生成所有配置组合
    for size_name in model_sizes:
        model = MODEL_PRESETS[size_name]
        device_id = MODEL_DEVICE_MAP[size_name]

        for ctx_len in context_lengths:
            config = BenchmarkConfig(
                model=model,
                context_length=ctx_len,
                device_id=device_id,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                mixed_precision=args.mixed_precision,
                forward_only=args.forward_only,
                profile_memory=args.profile_memory,
            )
            configs.append(config)

    return configs


# ============================================================
# Submitit 调度
# ============================================================


def submit_and_collect(
    configs: list[BenchmarkConfig],
) -> list[BenchmarkResult]:
    """
    使用 submitit 提交所有任务并收集结果。

    按 context_length 分批提交，同一批次内的任务并行执行。
    """
    executor = submitit.AutoExecutor(folder="logs/slurm_logs")
    executor.update_parameters(timeout_min=60)

    # 按 context_length 分组
    from collections import defaultdict

    by_context: dict[int, list[BenchmarkConfig]] = defaultdict(list)
    for cfg in configs:
        by_context[cfg.context_length].append(cfg)

    all_results: list[BenchmarkResult] = []

    # 逐批提交
    for ctx_len in sorted(by_context.keys()):
        batch_configs = by_context[ctx_len]
        print(f"\n>>> Submitting batch: context_length={ctx_len}, n_jobs={len(batch_configs)}")

        jobs: list[submitit.Job] = []
        with executor.batch():
            for cfg in batch_configs:
                job = executor.submit(run_benchmark, cfg)
                jobs.append(job)

        # 收集结果
        for cfg, job in zip(batch_configs, jobs):
            try:
                result = job.result()
                all_results.append(result)
            except Exception as e:
                # Job 执行失败
                all_results.append(
                    BenchmarkResult(
                        config=cfg,
                        status="error",
                        error_msg=str(e)[:50],
                    )
                )

    return all_results


# ============================================================
# 结果输出
# ============================================================


def _build_report_title(args: argparse.Namespace) -> str:
    """构建报告标题"""
    parts = [f"warmup={args.warmup_steps}"]

    if args.mixed_precision:
        parts.append(f"dtype={args.mixed_precision}")
    else:
        parts.append("dtype=fp32")

    if args.forward_only:
        parts.append("mode=forward_only")
    else:
        parts.append("mode=full_step")

    if args.profile_memory:
        parts.append("memory_profiling=ON")

    return f"### Benchmark Results ({', '.join(parts)})"


def write_markdown_report(
    results: list[BenchmarkResult],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """将结果写入 Markdown 文件"""

    # 构建表格数据
    rows = []
    for r in results:
        if r.status == "success":
            row = {
                "Model": r.config.model.name,
                "Context": r.config.context_length,
                "Status": "✓",
                "Fwd Mean (ms)": f"{r.fwd_mean:.2f}",
                "Fwd Std (ms)": f"{r.fwd_std:.2f}",
                "Bwd Mean (ms)": f"{r.bwd_mean:.2f}" if r.bwd_times else "-",
                "Bwd Std (ms)": f"{r.bwd_std:.2f}" if r.bwd_times else "-",
                "Total (ms)": f"{r.total_mean:.2f}",
            }
        else:
            row = {
                "Model": r.config.model.name,
                "Context": r.config.context_length,
                "Status": f"{r.status}: {r.error_msg}",
                "Fwd Mean (ms)": "-",
                "Fwd Std (ms)": "-",
                "Bwd Mean (ms)": "-",
                "Bwd Std (ms)": "-",
                "Total (ms)": "-",
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    table_md = df.to_markdown(index=False) or ""

    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    title = _build_report_title(args)

    with open(output, "a", encoding="utf-8") as f:
        f.write(f"\n{title}\n\n")
        f.write(table_md)
        f.write("\n\n")

    # 同时打印到控制台
    print(f"\n{title}")
    print(table_md)
    print(f"\n>>> Results appended to: {output}")


# ============================================================
# 主函数
# ============================================================


def main():
    args = parse_args()

    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print(f"Model sizes: {args.model_sizes}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision or 'disabled'}")
    print(f"Forward only: {args.forward_only}")
    print(f"Memory profiling: {args.profile_memory}")
    print("=" * 60)

    # 构建配置
    configs = build_configs(args)
    print(f"\nTotal configurations: {len(configs)}")

    # 提交并收集结果
    results = submit_and_collect(configs)

    # 输出报告
    write_markdown_report(results, args, args.output)


if __name__ == "__main__":
    main()
