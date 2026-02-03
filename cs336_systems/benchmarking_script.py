import timeit
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import pandas as pd
import submitit
import warnings
import os
import numpy as np
import pathlib
from contextlib import nullcontext

warnings.filterwarnings("ignore", category=UserWarning, module="submitit")


def generate_data(vocab_size, batch_size, context_length, device: str = "cpu"):
    mock_input = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    mock_target = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return mock_input, mock_target


def classify_error(e: Exception) -> tuple[str, str]:
    """
    分类错误类型并返回简洁的错误信息
    返回 (error_type, short_message)
    """
    error_str = str(e)

    # 检测 CUDA OOM
    if (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in error_str.lower()
        or "CUDA out of memory" in error_str
    ):
        return "OOM", "CUDA Out of Memory"

    # 检测设备错误
    if "No CUDA GPUs" in error_str or "invalid device" in error_str.lower():
        return "DeviceError", "GPU not available"

    # 其他错误：截取第一行，限制长度
    first_line = error_str.split("\n")[0][:50]
    # 移除可能破坏 markdown 的字符
    clean_msg = first_line.replace("|", "/").replace("\n", " ")
    return "Error", clean_msg


def benchmark_one_step(
    d_model,
    d_ff,
    num_layers,
    num_heads,
    vocab_size,
    context_length,
    batch_size,
    warmup_step,
    device_id,
    enable_mixed_precision=False,
    mix_dtype="bf16",
    pf_mem = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'=' * 20} Benchmark Configuration {'=' * 20}")
    print(f"Device: {device}")
    print(f"d_model: {d_model}")
    print(f"d_ff: {d_ff}")
    print(f"num_layers: {num_layers}")
    print(f"num_heads: {num_heads}")
    print(f"vocab_size: {vocab_size}")
    print(f"context_length: {context_length}")
    print(f"batch_size: {batch_size}")
    print(f"warmup_step: {warmup_step}")
    print(f"device_id: {device_id}")
    print(f"{'=' * 53}\n")
    if enable_mixed_precision:
        use_dtype = torch.bfloat16 if mix_dtype == "bf16" else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=use_dtype)
    else:
        autocast_ctx = nullcontext()

    print("Init model")
    with torch.cuda.nvtx.range("Init model"):
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=10000,
        )
        if device == "cuda":
            model.to(device)

    optimizer = AdamW(params=model.parameters())

    for _ in range(warmup_step):
        with autocast_ctx:
            optimizer.zero_grad()
            inputs, targets = generate_data(vocab_size, batch_size, context_length, device=device)
            y_hat = model.forward(inputs)
            loss = cross_entropy(y_hat, targets)
            loss.backward()
            optimizer.step()

    

    inputs, targets = generate_data(vocab_size, batch_size, context_length, device=device)

    fwd_times = []
    bwd_times = []

    if device == "cuda":
        torch.cuda.synchronize()
    for step in range(10):
        enable_pf_mem = pf_mem and step == 0
        optimizer.zero_grad()
        torch.cuda.nvtx.range_push(f"step_{step}")
        if enable_pf_mem:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        
        try:
            with autocast_ctx, torch.cuda.nvtx.range("Forward"):
                start_forward_time = timeit.default_timer()
                
                y_hat = model.forward(inputs)
                loss = cross_entropy(y_hat, targets)
                if enable_pf_mem:
                    forward_pickle_path = f"results/memory_snapshot_forward_only_{d_model}_{context_length}.pickle"
                    if enable_mixed_precision:
                        forward_pickle_path = f"results/memory_snapshot_forward_only_{d_model}_{context_length}_mixed_{mix_dtype}.pickle"
                    torch.cuda.memory._dump_snapshot(forward_pickle_path)

            if device == "cuda":
                torch.cuda.synchronize()

            forward_duration = (timeit.default_timer() - start_forward_time) * 1000  # 转换为 ms
            fwd_times.append(forward_duration)

            with autocast_ctx, torch.cuda.nvtx.range("Backward"):
                start_backward_time = timeit.default_timer()
                loss.backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                backward_duration = (timeit.default_timer() - start_backward_time) * 1000  # 转换为 ms
                bwd_times.append(backward_duration)

            with torch.cuda.nvtx.range("Optimize"):
                optimizer.step()

            torch.cuda.nvtx.range_pop()
            if enable_pf_mem:
                fullstep_pickle_path = f"results/memory_snapshot_{d_model}_{context_length}.pickle"
                if enable_mixed_precision:
                    fullstep_pickle_path = f"results/memory_snapshot_{d_model}_{context_length}_mixed_{mix_dtype}.pickle"
                torch.cuda.memory._dump_snapshot(fullstep_pickle_path)
        finally:
            if enable_pf_mem:
                torch.cuda.memory._record_memory_history(enabled=None)

    return {
        "fwd_mean": np.mean(fwd_times),
        "fwd_std": np.std(fwd_times),
        "bwd_mean": np.mean(bwd_times),
        "bwd_std": np.std(bwd_times),
        "total_mean": np.mean(fwd_times) + np.mean(bwd_times),
    }


def _get_valid_model_sizes(selected_sizes: str) -> set:
    """
    Parse comma-separated model sizes and return valid ones.
    Returns set of valid model sizes or all if 'all' is specified.
    """
    all_sizes = {"Small", "Medium", "Large", "xl", "2.7B"}
    if selected_sizes.lower() == "all":
        return all_sizes
    selected = {s.strip() for s in selected_sizes.split(",")}
    invalid = selected - all_sizes
    if invalid:
        raise ValueError(f"Invalid model sizes: {invalid}. Valid options: {sorted(all_sizes)}")
    return selected


def _get_valid_context_lengths(selected_lengths: str) -> set:
    """
    Parse comma-separated context lengths and return valid ones.
    Returns set of valid context lengths or all if 'all' is specified.
    """
    all_lengths = {128, 256, 512, 1024}
    if selected_lengths.lower() == "all":
        return all_lengths
    try:
        selected = {int(s.strip()) for s in selected_lengths.split(",")}
    except ValueError:
        raise ValueError(f"Invalid context length format. Must be integers: 128, 256, 512, 1024")
    invalid = selected - all_lengths
    if invalid:
        raise ValueError(f"Invalid context lengths: {invalid}. Valid options: {sorted(all_lengths)}")
    return selected


def run_benchmark(
    enable_mix_precision: bool, mix_type: str, pf_mem:bool, model_size_filter: str = "all", context_len_filter: str = "all"
):
    model_sizes = ["Small", "Medium", "Large", "xl", "2.7B"]
    d_model = [768, 1024, 1280, 1600, 2560]
    d_ff = [3072, 4096, 5120, 6400, 10240]
    num_layers = [12, 24, 36, 48, 32]
    num_heads = [12, 16, 20, 25, 32]
    device_ids = [0, 4, 5, 6, 7]
    vocab_size = 10000
    context_lengths = [128, 256, 512, 1024]
    batch_size = 4
    warmup_step = 5

    executor = submitit.AutoExecutor(folder="logs/slurm_logs")
    executor.update_parameters(timeout_min=60)

    valid_sizes = _get_valid_model_sizes(model_size_filter)
    valid_context_lengths = _get_valid_context_lengths(context_len_filter)

    filtered_configs = [
        (size, d_m, d_f, n_l, n_h, d_id)
        for size, d_m, d_f, n_l, n_h, d_id in zip(model_sizes, d_model, d_ff, num_layers, num_heads, device_ids)
        if size in valid_sizes
    ]

    if not filtered_configs:
        print(f"No valid model sizes found matching: {model_size_filter}")
        return

    filtered_context_lengths = [cl for cl in context_lengths if cl in valid_context_lengths]

    if not filtered_context_lengths:
        print(f"No valid context lengths found matching: {context_len_filter}")
        return

    results = []

    for context_length in filtered_context_lengths:
        jobs = []
        job_configs = []
        with executor.batch():
            for size, d_m, d_f, n_l, n_h, d_id in filtered_configs:
                job = executor.submit(
                    benchmark_one_step,
                    d_m,
                    d_f,
                    n_l,
                    n_h,
                    vocab_size,
                    context_length,
                    batch_size,
                    warmup_step,
                    d_id,
                    enable_mix_precision,
                    mix_type,
                    pf_mem
                )
                jobs.append(job)
                job_configs.append(
                    {
                        "model_size": size,
                        "d_model": d_m,
                        "d_ff": d_f,
                        "num_layers": n_l,
                        "num_heads": n_h,
                        "context_length": context_length,
                    }
                )

        for config, job in zip(job_configs, jobs):
            try:
                r = job.result()
                results.append(
                    {
                        **config,
                        "status": "success",
                        "Fwd Mean (ms)": f"{r['fwd_mean']:.2f}",
                        "Fwd Std (ms)": f"{r['fwd_std']:.2f}",
                        "Bwd Mean (ms)": f"{r['bwd_mean']:.2f}",
                        "Bwd Std (ms)": f"{r['bwd_std']:.2f}",
                        "Total (ms)": f"{r['total_mean']:.2f}",
                    }
                )
            except Exception as e:
                error_type, error_msg = classify_error(e)
                results.append(
                    {
                        **config,
                        "status": f"{error_type}: {error_msg}",
                        "Fwd Mean (ms)": "-",
                        "Fwd Std (ms)": "-",
                        "Bwd Mean (ms)": "-",
                        "Bwd Std (ms)": "-",
                        "Total (ms)": "-",
                    }
                )

    df = pd.DataFrame(results)
    file_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark.md"
    with open(file_path, "a", encoding="utf-8") as f:
        if enable_mix_precision:
            f.write(
                f"\n### Model Benchmarking Results with warm-up step:{warmup_step}, with mixed precision:{mix_type}\n"
            )
        else:
            f.write(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}\n")
        f.write(df.to_markdown(index=False))

    print(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}")
    print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark your training infra")
    parser.add_argument("--mix", action="store_true", help="turn on mixed precision")
    parser.add_argument("--mix_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="mixed precision dtype")
    parser.add_argument("--memory", action="store_true", help="profile memory")
    parser.add_argument(
        "--model_size",
        type=str,
        default="all",
        help="model sizes to benchmark: Small, Medium, Large, xl, 2.7B (comma-separated or 'all')",
    )
    parser.add_argument(
        "--context_len",
        type=str,
        default="all",
        help="context lengths to benchmark: 128, 256, 512, 1024 (comma-separated or 'all')",
    )
    parser.add_argument("--d_model", type=int, default=768, help="d_model")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads")

    args = parser.parse_args()

    run_benchmark(args.mix, args.mix_dtype,args.memory, args.model_size, args.context_len)
    return


if __name__ == "__main__":
    main()
