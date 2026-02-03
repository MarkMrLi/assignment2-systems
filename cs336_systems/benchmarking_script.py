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
    d_model, d_ff, num_layers, num_heads, vocab_size, context_length, batch_size, warmup_step, device_id,
    enable_mixed_precision = False, mix_dtype = "bf16"
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

    if device == "cuda":
        torch.cuda.synchronize()

    inputs, targets = generate_data(vocab_size, batch_size, context_length, device=device)

    fwd_times = []
    bwd_times = []

    for step in range(10):
        optimizer.zero_grad()
        torch.cuda.nvtx.range_push(f"step_{step}")
        
        with autocast_ctx, torch.cuda.nvtx.range("Forward"):
            start_forward_time = timeit.default_timer()
            y_hat = model.forward(inputs)
            loss = cross_entropy(y_hat, targets)

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

    return {
        "fwd_mean": np.mean(fwd_times),
        "fwd_std": np.std(fwd_times),
        "bwd_mean": np.mean(bwd_times),
        "bwd_std": np.std(bwd_times),
        "total_mean": np.mean(fwd_times) + np.mean(bwd_times),
    }


def run_benchmark(enable_mix_precision:bool, mix_type: str):
    model_size = ["Small", "Medium", "Large", "xl", "2.7B"]
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

    
    results = []
    
    for context_length in context_lengths:
        jobs = []
        job_configs = []
        with executor.batch():
            for idx, (d_m, d_f, n_l, n_h, d_id) in enumerate(zip(d_model, d_ff, num_layers, num_heads, device_ids)):
                job = executor.submit(
                    benchmark_one_step, d_m, d_f, n_l, n_h, vocab_size, context_length, batch_size, warmup_step, d_id,
                    enable_mix_precision, mix_type
                )
                jobs.append(job)
                job_configs.append(
                    {
                        "model_size": model_size[idx],
                        "d_model": d_model[idx],
                        "d_ff": d_ff[idx],
                        "num_layers": num_layers[idx],
                        "num_heads": num_heads[idx],
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
        if enable_mix_precision :
            f.write(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}, with mixed precision:{mix_type}\n")
        else:
            f.write(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}\n")
        f.write(df.to_markdown(index=False))

    print(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}")
    print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark your training infra")
    parser.add_argument("--eval", action="store_true", help="benchmark with default configs")
    parser.add_argument("--mix", action="store_true", help="turn on mixed precision")
    parser.add_argument("--mix_dtype", type=str, default="bf16", choices = ["bf16","fp16"], help="mixed precision dtype")
    parser.add_argument("--d_model", type=int, default=768, help="d_model")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads")
    parser.add_argument("--context_length", type=int, default=256, help="context length")
    args = parser.parse_args()

    if args.eval:
        run_benchmark(args.mix,args.mix_dtype)
        return
    else:
        benchmark_one_step(
            args.d_model, args.d_ff, args.num_layers, args.num_heads, 10000, args.context_length, 4, 5, device_id=0,
            enable_mixed_precision=args.mix,mix_dtype=args.mix_dtype
        )


if __name__ == "__main__":
    main()
