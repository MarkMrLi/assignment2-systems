"""
Benchmark 执行器

这个模块包含单次 benchmark 的执行逻辑。
submitit 会调用 run_benchmark() 函数。
"""

import os
import timeit
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from .benchmark_config import BenchmarkConfig, BenchmarkResult


# ============================================================
# 主入口函数 (submitit 调用此函数)
# ============================================================


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    执行单次 benchmark 的主入口。

    Args:
        config: Benchmark 配置

    Returns:
        BenchmarkResult: 包含计时结果或错误信息
    """
    # 设置 CUDA 设备 (必须在 import torch 前或重新 import)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)

    _print_config(config)

    try:
        device = _setup_device()
        model = _create_model(config, device)
        optimizer = _create_optimizer(model) if not config.forward_only else None

        # 预热
        _warmup(model, optimizer, config, device)
        if optimizer is not None:
            optimizer.zero_grad()

        # 根据是否需要内存 profiling 选择执行路径
        if config.profile_memory:
            return _run_with_memory_profile(model, optimizer, config, device)
        else:
            return _run_timing_benchmark(model, optimizer, config, device)

    except Exception as e:
        return _handle_error(config, e)


# ============================================================
# 设备和模型初始化
# ============================================================


def _print_config(config: BenchmarkConfig) -> None:
    """打印配置信息"""
    print(f"{'=' * 20} Benchmark Configuration {'=' * 20}")
    print(f"Model: {config.model.name}")
    print(f"  d_model: {config.model.d_model}")
    print(f"  d_ff: {config.model.d_ff}")
    print(f"  num_layers: {config.model.num_layers}")
    print(f"  num_heads: {config.model.num_heads}")
    print(f"Context Length: {config.context_length}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Mixed Precision: {config.mixed_precision or 'fp32'}")
    print(f"Forward Only: {config.forward_only}")
    print(f"Profile Memory: {config.profile_memory}")
    print(f"{'=' * 53}\n")


def _setup_device() -> torch.device:
    """设置并返回计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _create_model(config: BenchmarkConfig, device: torch.device) -> BasicsTransformerLM:
    """创建并初始化模型"""
    print("Initializing model...")
    with torch.cuda.nvtx.range("Init model"):
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            rope_theta=10000,
        )
        model.to(device)
    return model


def _create_optimizer(model: BasicsTransformerLM) -> AdamW:
    """创建优化器"""
    return AdamW(params=model.parameters())


# ============================================================
# 数据生成
# ============================================================


def _generate_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """生成随机输入数据"""
    inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
    targets = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
    return inputs, targets


# ============================================================
# 上下文管理器辅助
# ============================================================


def _get_autocast_context(config: BenchmarkConfig):
    """获取混合精度上下文管理器"""
    if config.mixed_precision:
        dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _get_inference_context(config: BenchmarkConfig):
    """获取推理模式上下文管理器"""
    return torch.inference_mode() if config.forward_only else nullcontext()


# ============================================================
# 预热
# ============================================================


def _warmup(
    model: BasicsTransformerLM,
    optimizer: Optional[AdamW],
    config: BenchmarkConfig,
    device: torch.device,
) -> None:
    """执行预热步骤"""
    print(f"Warming up ({config.warmup_steps} steps)...")

    autocast_ctx = _get_autocast_context(config)
    inference_ctx = _get_inference_context(config)

    for _ in range(config.warmup_steps):
        inputs, targets = _generate_batch(config, device)

        if config.forward_only:
            with inference_ctx, autocast_ctx:
                model(inputs)
        else:
            assert optimizer is not None
            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()


# ============================================================
# 计时测量
# ============================================================


def _measure_forward_step(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    config: BenchmarkConfig,
) -> float:
    """
    测量单次前向传播耗时。

    Returns:
        forward_time_ms: 前向耗时 (毫秒)
    """
    autocast_ctx = _get_autocast_context(config)
    inference_ctx = _get_inference_context(config)

    with inference_ctx:
        with torch.cuda.nvtx.range("Forward"):
            torch.cuda.synchronize()
            start = timeit.default_timer()

            with autocast_ctx:
                model(inputs)

            torch.cuda.synchronize()
            elapsed_ms = (timeit.default_timer() - start) * 1000

    return elapsed_ms


def _measure_full_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
) -> tuple[float, float]:
    """
    测量完整训练步耗时 (前向 + 后向)。

    Returns:
        (forward_time_ms, backward_time_ms): 前向和后向耗时 (毫秒)
    """
    autocast_ctx = _get_autocast_context(config)

    optimizer.zero_grad()

    # Forward
    with torch.cuda.nvtx.range("Forward"):
        torch.cuda.synchronize()
        start_fwd = timeit.default_timer()

        with autocast_ctx:
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

        torch.cuda.synchronize()
        fwd_ms = (timeit.default_timer() - start_fwd) * 1000

    # Backward
    with torch.cuda.nvtx.range("Backward"):
        torch.cuda.synchronize()
        start_bwd = timeit.default_timer()

        loss.backward()

        torch.cuda.synchronize()
        bwd_ms = (timeit.default_timer() - start_bwd) * 1000

    # Optimizer step
    with torch.cuda.nvtx.range("Optimizer"):
        optimizer.step()

    return fwd_ms, bwd_ms


# ============================================================
# 执行路径：计时 Benchmark
# ============================================================


def _run_timing_benchmark(
    model: BasicsTransformerLM,
    optimizer: Optional[AdamW],
    config: BenchmarkConfig,
    device: torch.device,
) -> BenchmarkResult:
    """执行计时 benchmark (不进行内存 profiling)"""

    fwd_times: list[float] = []
    bwd_times: list[float] = []

    inputs, targets = _generate_batch(config, device)

    for step in range(config.measure_steps):
        torch.cuda.nvtx.range_push(f"step_{step}")

        try:
            if config.forward_only:
                fwd_ms = _measure_forward_step(model, inputs, config)
                fwd_times.append(fwd_ms)
            else:
                assert optimizer is not None
                fwd_ms, bwd_ms = _measure_full_step(model, optimizer, inputs, targets, config)
                fwd_times.append(fwd_ms)
                bwd_times.append(bwd_ms)
        finally:
            torch.cuda.nvtx.range_pop()

    return BenchmarkResult(
        config=config,
        fwd_times=fwd_times,
        bwd_times=bwd_times,
        status="success",
    )


# ============================================================
# 执行路径：内存 Profiling
# ============================================================


def _get_memory_snapshot_path(config: BenchmarkConfig) -> Path:
    """生成内存快照文件路径"""
    results_dir = Path("results/memory")
    results_dir.mkdir(parents=True, exist_ok=True)

    mode = "fwd_only" if config.forward_only else "full_step"
    dtype = config.mixed_precision or "fp32"

    filename = f"memory_{config.model.name}_ctx{config.context_length}_bs{config.batch_size}_{mode}_{dtype}.pickle"
    return results_dir / filename


def _run_with_memory_profile(
    model: BasicsTransformerLM,
    optimizer: Optional[AdamW],
    config: BenchmarkConfig,
    device: torch.device,
) -> BenchmarkResult:
    """执行带内存 profiling 的 benchmark (只跑 1 步)"""

    inputs, targets = _generate_batch(config, device)
    fwd_times: list[float] = []
    bwd_times: list[float] = []

    # 开始记录内存
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    try:
        torch.cuda.nvtx.range_push("profiled_step")

        if config.forward_only:
            fwd_ms = _measure_forward_step(model, inputs, config)
            fwd_times.append(fwd_ms)
        else:
            assert optimizer is not None
            fwd_ms, bwd_ms = _measure_full_step(model, optimizer, inputs, targets, config)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)

        torch.cuda.nvtx.range_pop()

        # 保存内存快照
        snapshot_path = _get_memory_snapshot_path(config)
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        print(f"Memory snapshot saved to: {snapshot_path}")

    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    return BenchmarkResult(
        config=config,
        fwd_times=fwd_times,
        bwd_times=bwd_times,
        status="success",
    )


# ============================================================
# 错误处理
# ============================================================


def _classify_error(e: Exception) -> tuple[str, str]:
    """
    分类错误类型。

    Returns:
        (error_type, short_message)
    """
    error_str = str(e)

    # CUDA OOM
    if (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in error_str.lower()
        or "CUDA out of memory" in error_str
    ):
        return "oom", "CUDA Out of Memory"

    # Device error
    if "No CUDA GPUs" in error_str or "invalid device" in error_str.lower():
        return "error", "GPU not available"

    # Other errors
    first_line = error_str.split("\n")[0][:50]
    clean_msg = first_line.replace("|", "/").replace("\n", " ")
    return "error", clean_msg


def _handle_error(config: BenchmarkConfig, e: Exception) -> BenchmarkResult:
    """处理异常并返回错误结果"""
    status, msg = _classify_error(e)
    print(f"Benchmark failed: {msg}")

    return BenchmarkResult(
        config=config,
        fwd_times=[],
        bwd_times=[],
        status=status,  # type: ignore
        error_msg=msg,
    )
