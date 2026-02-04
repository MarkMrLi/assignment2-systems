"""
Benchmark 配置定义

包含:
- ModelConfig: 模型架构参数
- BenchmarkConfig: 单次 benchmark 的完整配置
- BenchmarkResult: benchmark 结果
- MODEL_PRESETS: 预设模型配置
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    """模型架构配置 (不可变)"""

    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass
class BenchmarkConfig:
    """单次 Benchmark 的完整配置"""

    model: ModelConfig
    context_length: int
    device_id: int
    batch_size: int = 4
    warmup_steps: int = 5
    measure_steps: int = 10
    vocab_size: int = 10000
    mixed_precision: Optional[Literal["bf16", "fp16"]] = None
    forward_only: bool = False
    profile_memory: bool = False

    def __repr__(self) -> str:
        """简洁的字符串表示，用于日志"""
        mode = "fwd" if self.forward_only else "full"
        dtype = self.mixed_precision or "fp32"
        return (
            f"BenchmarkConfig({self.model.name}, "
            f"ctx={self.context_length}, bs={self.batch_size}, "
            f"mode={mode}, dtype={dtype})"
        )


@dataclass
class BenchmarkResult:
    """Benchmark 结果"""

    config: BenchmarkConfig
    fwd_times: list[float] = field(default_factory=list)
    bwd_times: list[float] = field(default_factory=list)
    status: Literal["success", "oom", "error"] = "success"
    error_msg: Optional[str] = None

    # 统计属性

    @property
    def fwd_mean(self) -> float:
        """前向传播平均耗时 (ms)"""
        return float(np.mean(self.fwd_times)) if self.fwd_times else 0.0

    @property
    def fwd_std(self) -> float:
        """前向传播耗时标准差 (ms)"""
        return float(np.std(self.fwd_times)) if self.fwd_times else 0.0

    @property
    def bwd_mean(self) -> float:
        """后向传播平均耗时 (ms)，如果没有后向则返回 0"""
        return float(np.mean(self.bwd_times)) if self.bwd_times else 0.0

    @property
    def bwd_std(self) -> float:
        """后向传播耗时标准差 (ms)，如果没有后向则返回 0"""
        return float(np.std(self.bwd_times)) if self.bwd_times else 0.0

    @property
    def total_mean(self) -> float:
        """总耗时 = fwd_mean + bwd_mean"""
        return self.fwd_mean + self.bwd_mean


# 预设模型配置
MODEL_PRESETS: dict[str, ModelConfig] = {
    "Small": ModelConfig("Small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "Medium": ModelConfig("Medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "Large": ModelConfig("Large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig("xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig("2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


# 模型名称到 device_id 的映射 (保持原脚本的固定分配)
MODEL_DEVICE_MAP: dict[str, int] = {
    "Small": 0,
    "Medium": 4,
    "Large": 5,
    "xl": 6,
    "2.7B": 7,
}


# 支持的 context lengths
CONTEXT_LENGTHS = [128, 256, 512, 1024]
