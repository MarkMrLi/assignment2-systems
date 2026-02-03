# CS336 Learning Ledger
**Last Updated:** 2026-02-03

---

## 🧠 Knowledge Graph (Mastery Levels)

### 🟢 Mastered
- PyTorch `zero_grad()` 机制与梯度累积
- Forward/Backward Pass 边界定义（计算图构建 vs 消费）
- `torch.cuda.synchronize()` 用于准确 GPU 时间测量
- 单机多卡环境下 `torch.cuda.set_device()` 使用
- PyTorch CUDA memory caching allocator behavior
- Submitit LocalExecutor execution model and batch submission semantics
- GPU resource contention diagnosis methodology
- Concurrency control via loop restructuring
- **Benchmarking fundamentals**: Warmup importance for stable measurements
- **NVTX annotation**: Using `torch.cuda.nvtx.range()` for profiling phase marking
- **Memory-bound vs Compute-bound**: Small models memory-bound (Bwd/Fwd ~1.1x), large models compute-bound (~2x)
- **GEMM kernel naming**: cuBLAS convention (`tn_n`=left transpose, `nn_n`=no transpose, tile sizes)
- **Dynamic Range vs Precision distinction**: Exponent bits → range, Mantissa bits → precision
- **FP16/BF16/FP32 trade-offs**: BF16 stable (same range as FP32), FP16 precise but unstable
- **Loss Scaling mechanism**: Shifts gradients to avoid FP16's representational dead zone
- **FP16 Spacing/ULP**: Gap between representable FP16 numbers grows exponentially with magnitude
- **Round-to-Nearest Behavior**: IEEE 754 rounding and its impact on accumulation accuracy
- **Mixed Precision Accumulation Pattern**: Accumulator must use FP32 even when operands are FP16 - this is Master Weights core principle
- **Autocast Mechanism**: 参数存储类型 vs 运算精度的区别，黑白名单概念
- **Device-Specific Autocast Behavior**: CPU vs GPU 黑白名单差异（LayerNorm 在 GPU 黑名单，CPU 可能不是）
- **BF16 Trade-offs**: 动态范围足够但累加精度较低（7-bit 尾数 vs FP16 的 10-bit）
- **LayerNorm Precision Sensitivity**: 方差计算容易下溢，除法容易损失精度
- **Mixed Precision Training Flow**: Forward/Backward in autocast, Optimizer outside (Master Weights)
- **Conditional Autocast**: Using `nullcontext` for no-op precision switching
- **Warmup Strategy**: Must match target precision (FP32 warmup → BF16 measure = JIT overhead contamination)

### 🟡 Developing
- Slurm + CUDA_VISIBLE_DEVICES 交互机制
- submitit 多进程任务调度
- Distributed training setup (SLURM vs non-SLURM environments)
- Multi-GPU benchmarking best practices
- CUDA context management across subprocess boundaries
- Memory profiling and OOM debugging techniques
- **Nsys profiling workflow**: Can use GUI effectively, need more practice with CLI stats extraction
- **Kernel-level analysis**: Can identify top kernels, building intuition for optimization
- **Tile size trade-offs**: 128×128 for high arithmetic intensity, 64×64 when register pressure high
- **Arithmetic Intensity**: Can explain concept, need practice calculating for specific ops
- **Mixed Precision Training**: Hands-on implementation with BF16 benchmark completed
- **Speedup Analysis**: Understanding why BF16 speedup plateaus and doesn't scale with model size
- **Accumulation Error Visualization**: Can predict error patterns, need to implement visualization tools

### 🔴 Blind Spots
- Slurm 集群环境下的 GPU 分配与隔离策略
- **FlashAttention implementation**: Understand motivation (Softmax memory overhead), haven't implemented
- Quantization (FP8/FP4) memory savings calculation
- Gradient checkpointing trade-offs
- Tensor Parallelism vs Pipeline Parallelism decision criteria
- AdamW optimizer state memory overhead calculation
- **Tensor Core utilization measurement**: Know they accelerate GEMM, don't know how to measure
- **Safe Softmax implementation**: Know the principle (subtract max), need to verify in code
- **TPU/Mixed Precision**: Google TPU's BF16 strategy vs NVIDIA GPU differences
- **Stochastic Rounding**: Alternative rounding strategies for training stability
- **Error Propagation in Deep Networks**: How numerical errors compound across transformer layers
- **GradScaler Mechanics**: FP16 needs loss scaling, BF16 doesn't - understand implementation details
- **Long-term Training Stability**: BF16 accumulation error impact over 100k+ steps

---

## 📉 Action Items & Review Queue

### Immediate (Next Session)
- [ ] Implement FlashAttention and compare profiling results
- [ ] Investigate OOM causes for XL/2.7B models at context_length=1024
- [ ] Implement safe softmax and test numerical stability
- [ ] Compare BF16 vs FP16 accumulation experiment (0.01 × 1000 test)
- [ ] Read PyTorch Autocast 源码，理解黑白名单的注册机制 (`torch/_autocast_utils.py`)
- [ ] Add Nsight Compute analysis to benchmark，采集算术强度和带宽利用率指标

### Short-term (This Week)
- [ ] 了解 Slurm `--gres=gpu:N` 与 CUDA_VISIBLE_DEVICES 的映射关系
- [ ] Review FlashAttention paper - understand the SRAM vs HBM movement
- [ ] Implement memory profiling using `torch.cuda.memory_summary()`
- [ ] Learn `nsys stats` CLI commands for automated data extraction
- [ ] Practice calculating arithmetic intensity for Attention operations
- [ ] Implement mixed precision training with dynamic loss scaling
- [ ] Compare FP32 vs BF16 vs FP16 training convergence on small model
- [ ] Cross-Platform Test: 在 CPU 上复现 BF16 实验，验证 LayerNorm 输出 dtype 差异

### Long-term (Before Assignment Due)
- [ ] Compare benchmark results with theoretical FLOPs calculations
- [ ] Explore mixed precision training impact on memory and speed
- [ ] Study Tensor Core architecture and how to maximize utilization
- [ ] Implement custom BF16 LayerNorm and test numerical stability
- [ ] Compare with JAX/Flax BF16 strategy

---

## 📚 Session History

| Date | Topic | Key Outcome |
|------|-------|-------------|
| 2026-01-30 | Benchmarking Code Review | 掌握 Forward/Backward 边界、GPU 设备选择机制 |
| 2026-01-30 | OOM Diagnosis & Concurrency Fix | Diagnosed submitit OOM by restructuring batch submission pattern |
| 2026-01-31 | Nsys Profiling & Kernel Analysis | Mastered GEMM kernel naming, understood memory-bound vs compute-bound, analyzed Softmax overhead motivating FlashAttention |
| 2026-01-31 | Mixed Precision Training | Clarified Dynamic Range vs Precision, understood FP16/BF16/FP32 trade-offs, Loss Scaling mechanism |
| 2026-02-02 | Mixed Precision Accumulation | Discovered FP16 spacing impact on accumulation, understood Master Weights design rationale, verified Round-to-Nearest behavior |
| **2026-02-03** | **BF16 Autocast & Benchmarking** | **Discovered CPU vs GPU autocast behavior differences, implemented BF16 benchmarking, analyzed speedup trends (small models 3x, large models 2.2x), understood why optimizer must be outside autocast** |

---

## 🎯 Course Goals (Assignment 2 - Systems)
- [x] Build benchmarking infrastructure with multiple model sizes
- [x] Profile Forward/Backward with Nsys, analyze kernel distribution
- [x] Implement mixed precision training with BF16 autocast
- [ ] Implement memory-efficient attention (FlashAttention)
- [ ] Implement distributed training strategies
- [ ] Optimize training throughput

---

## 💡 Key Insights Archive

### 2026-02-03: BF16 Speedup Plateau
> "BF16 achieves ~3x speedup for Small models (memory-bound) but only ~2.2x for 2.7B model (compute-bound).
> Speedup doesn't scale with parameter count because 2.7B has fewer layers (32 vs 48), meaning higher GEMM percentage.
> This confirms: arithmetic intensity, not raw parameters, determines mixed precision gains."

### 2026-02-03: Memory Savings > Speed Gains
> "BF16's primary benefit is preventing OOM (XL model runs at context_length=1024 in BF16 but OOM in FP32).
> The ~2-3x speedup is secondary to the ability to train larger models/batches."

### 2026-02-03: Autocast is Device-Specific
> "LayerNorm outputs float32 on GPU (blacklisted) but bfloat16 on CPU (not blacklisted).
> PyTorch maintains separate allow/deny lists per device backend. This can cause cross-platform inconsistencies."

### 2026-02-03: Warmup Must Match Target Precision
> "If you FP32-warmup then BF16-measure, the first BF16 iteration includes JIT kernel compilation overhead.
> Fair comparison requires separate warmup for each precision mode."

### 2026-02-03: Optimizer Outside Autocast
> "Master Weights require optimizer.step() to remain in FP32. Placing it inside autocast doesn't break anything
> (autocast doesn't affect optimizer), but semantically it belongs outside - autocast is for compute, not parameter update."

### 2026-02-03: Small Models More Memory-Bound
> "Contrary to intuition, Small models show higher BF16 speedup (3x) than Large (2.9x) or 2.7B (2.2x).
> Reason: Small models spend more time on memory-bound ops (LayerNorm, residuals, data movement).
> BF16 halves HBM traffic, benefiting memory-bound workloads most."

### 2026-02-02: FP16 Accumulation Error is Non-uniform
> "FP16 spacing increases with magnitude. Adding 0.01 to 16.0 in FP16 yields 16.015625 
> (spacing=0.0156), while adding to 1.0 yields 1.009765625. The error direction 
> varies by range - unpredictable and unidirectional. This is why pure FP16 
> gradient accumulation fails."

### 2026-02-02: Master Weights Principle
> "Even when gradients are computed and stored as FP16, the accumulation buffer 
> must be FP32. This eliminates the spacing problem entirely - FP32's spacing 
> at typical gradient magnitudes is negligible. This is the numerical foundation 
> for Master Weights in mixed precision training."

### 2026-02-02: FP16 Representation of 0.01
> "torch.tensor(0.01, dtype=torch.float16) stores 0.01000213623046875, not 0.01. 
> The representation error starts immediately, not just at large values. This 
> explains why even early-stage FP16 accumulation shows systematic bias."

### 2026-01-31: Dynamic Range vs Precision
> "Dynamic range (exponent bits) determines if a value can be represented at all.
> Precision (mantissa bits) determines if two close values can be distinguished.
> FP16 has narrow range but decent precision; BF16 has full range but low precision.
> This is why BF16 is 'stable but affects performance' - no overflow, but small updates get swallowed."

### 2026-01-31: Loss Scaling Essence
> "Loss Scaling doesn't change FP16's dynamic range - it shifts gradient values 
> away from the 'dead zone' (values too small to represent). 
> Like moving your data to a different floor in a building, not making the building taller."

### 2026-01-31: Attention Overflow Point
> "QK^T accumulates d_k products. With exp() in softmax, overflow occurs when QK^T > ln(65504) ≈ 11.
> This is why we need: (1) / sqrt(d_k) scaling, (2) safe softmax (subtract max before exp)."

### 2026-01-31: FLOPs ≠ Runtime
> "Softmax has ~100x fewer FLOPs than Attention GEMM, but only runs ~8.5x faster. 
> End-to-end latency is nearly identical (~400μs vs ~300μs) due to memory overhead. 
> This is why FlashAttention fuses Softmax with Attention computation in SRAM."

### 2026-01-31: Backward uses smaller tiles
> "Backward pass uses 64×64 tiles instead of Forward's 128×128 because it needs 
> to store both activations and gradients, creating higher register pressure."

### 2026-01-31: Memory-bound to Compute-bound transition
> "Small models show Bwd/Fwd ratio ~1.1x (both memory-bound), while large models 
> approach theoretical 2x ratio (compute-bound). This explains why optimization 
> strategies differ by model scale."

---

*This ledger is updated at the end of each study session. Say "End Session" to generate updates.*
