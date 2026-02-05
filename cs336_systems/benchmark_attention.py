from dataclasses import dataclass, field, asdict
import argparse
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from cs336_basics.nn_utils import softmax
import timeit
from cs336_systems.benchmark_runner import _classify_error
import pandas as pd
from pathlib import Path
import math


@dataclass
class BenchmarkConfig:
    d_model: int
    seq_len: int
    batch_size: int = 8
    warmup_steps: int = 5
    steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    fwd_times: list[float] = field(default_factory=list)
    bwd_times: list[float] = field(default_factory=list)
    memory_after_forward: list[float] = field(default_factory=list)
    add_memory: list[float] = field(default_factory=list)
    status: str = "success"

    @property
    def fwd_time_mean(self):
        return np.mean(self.fwd_times) if self.fwd_times else 0.0

    @property
    def bwd_time_mean(self):
        return np.mean(self.bwd_times) if self.bwd_times else 0.0

    @property
    def peak_memory_mean(self):
        return np.mean(self.memory_after_forward) if self.memory_after_forward else 0.0

    @property
    def add_memory_mean(self):
        return np.mean(self.add_memory) if self.add_memory else 0.0


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size seq_len d_model"],
    K: Float[torch.Tensor, "batch_size seq_len d_model"],
    V: Float[torch.Tensor, "batch_size seq_len d_model"]
) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
    d_model = Q.shape[-1]
    attn_score = einsum(Q, K, "... seq_q d_m, ... seq_k d_m -> ... seq_q seq_k") / math.sqrt(d_model)
    attn_weight = softmax(attn_score)
    return einsum(attn_weight, V, "... seq_q seq_k, ... seq_k d_m -> ... seq_q d_m")


def get_q_k_v(config: BenchmarkConfig, requires_grad: bool = False):
    """模拟生成 QKV"""
    batch_size = config.batch_size
    d_model = config.d_model
    seq_len = config.seq_len
    device = config.device

    Q = torch.randn([batch_size, seq_len, d_model], requires_grad=requires_grad, device=device)
    K = torch.randn([batch_size, seq_len, d_model], requires_grad=requires_grad, device=device)
    V = torch.randn([batch_size, seq_len, d_model], requires_grad=requires_grad, device=device)

    return Q, K, V


def fwd_only(config: BenchmarkConfig):
    """Inference forward"""
    Q, K, V = get_q_k_v(config, requires_grad=False)

    for _ in range(config.warmup_steps):
        with torch.no_grad():
            scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    fwd_times = []
    for _ in range(config.steps):
        with torch.no_grad():
            start = timeit.default_timer()
            scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            duration = timeit.default_timer() - start
            fwd_times.append(duration * 1000)

    return fwd_times


def fwd_bwd(config: BenchmarkConfig):
    """foward and backward"""
    Q, K, V = get_q_k_v(config, requires_grad=True)
    
    for _ in range(config.warmup_steps):
        out = scaled_dot_product_attention(Q, K, V)
        out = out.sum()
        out.backward()
    torch.cuda.synchronize()

    bwd_times = []
    peak_memory = []
    add_memory = []
    
    for _ in range(config.steps):
        if Q.grad is not None: 
            Q.grad.zero_()
        if K.grad is not None: 
            K.grad.zero_()
        if V.grad is not None: 
            V.grad.zero_()
        
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        out = scaled_dot_product_attention(Q, K, V)
        out = out.sum()
        torch.cuda.synchronize()
        
        peak_memory.append(torch.cuda.max_memory_allocated() / (1024**2))
        add_memory.append((torch.cuda.memory_allocated() - base_mem) / (1024**2))
        
        start = timeit.default_timer()
        out.backward()
        torch.cuda.synchronize()
        duration = timeit.default_timer() - start
        bwd_times.append(duration * 1000)

    return bwd_times, peak_memory, add_memory


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark Attention"""
    result = BenchmarkResult(config=config)
    
    try:
        result.fwd_times = fwd_only(config)
        result.bwd_times, result.memory_after_forward, result.add_memory = fwd_bwd(config)
        result.status = "success"
    except Exception as e:
        error_type, short_meg = _classify_error(e)
        result.status = f"{error_type}:{short_meg}"
    finally:
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d_model",
        type=str,
        default="all",
        help="d_model: 16,32,64,128 or all"
    )
    parser.add_argument(
        "--seq_len",
        type=str,
        default="all",
        help="seq_len:256, 1024, 4096, 8192, 16384 or all"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/profile_attention/pytorch.md",
        help="output_file_path"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> list[BenchmarkConfig]:
    default_d_model = [16, 32, 64, 128]
    default_seq_len = [256, 1024, 4096, 8192, 16384]
    
    if args.d_model == "all":
        d_models = default_d_model
    else:
        d_models = [int(d.strip()) for d in args.d_model.split(',')]

    if args.seq_len == "all":
        seq_lens = default_seq_len
    else:
        seq_lens = [int(s.strip()) for s in args.seq_len.split(',')]

    configs = []
    for d in d_models:
        for s in seq_lens:
            config = BenchmarkConfig(d_model=d, seq_len=s)
            configs.append(config)

    return configs


def process_result(results: list[BenchmarkResult], output_path: str):
    format_results = []
    for r in results:
        res_dict = asdict(r.config)
        res_dict.update({
            "fwd_ms": r.fwd_time_mean,
            "bwd_ms": r.bwd_time_mean,
            "mem_max_MB": r.peak_memory_mean,
            "add_mem_MB": r.add_memory_mean,
            "status": r.status
        })
        format_results.append(res_dict)
    
    df = pd.DataFrame(format_results)
    result_md = df.to_markdown(index=False) or ""

    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        f.write(result_md)


def main():
    """
    1. 准备 config
    2. 运行benchmark
    3. 整理结果
    """
    args = parse_args()
    configs = build_config(args)

    results: list[BenchmarkResult] = []
    for config in configs:
        torch.cuda.empty_cache()
        results.append(run_benchmark(config))

    process_result(results, args.output_path)


if __name__ == "__main__":
    main()