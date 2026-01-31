# CS336 Learning Ledger
**Last Updated:** 2026-01-31

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
- **Benchmarking fundamentals**: Warmup importance for stable measurements *(新增)*
- **NVTX annotation**: Using `torch.cuda.nvtx.range()` for profiling phase marking *(新增)*
- **Memory-bound vs Compute-bound**: Small models memory-bound (Bwd/Fwd ~1.1x), large models compute-bound (~2x) *(新增)*
- **GEMM kernel naming**: cuBLAS convention (`tn_n`=left transpose, `nn_n`=no transpose, tile sizes) *(新增)*

### 🟡 Developing
- Slurm + CUDA_VISIBLE_DEVICES 交互机制
- submitit 多进程任务调度
- Distributed training setup (SLURM vs non-SLURM environments)
- Multi-GPU benchmarking best practices
- CUDA context management across subprocess boundaries
- Memory profiling and OOM debugging techniques
- **Nsys profiling workflow**: Can use GUI effectively, need more practice with CLI stats extraction *(更新)*
- **Kernel-level analysis**: Can identify top kernels, building intuition for optimization *(新增)*
- **Tile size trade-offs**: 128×128 for high arithmetic intensity, 64×64 when register pressure high *(新增)*
- **Arithmetic Intensity**: Can explain concept, need practice calculating for specific ops *(新增)*

### 🔴 Blind Spots
- Slurm 集群环境下的 GPU 分配与隔离策略
- **FlashAttention implementation**: Understand motivation (Softmax memory overhead), haven't implemented *(更新)*
- Quantization (FP8/FP4) memory savings calculation
- Gradient checkpointing trade-offs
- Tensor Parallelism vs Pipeline Parallelism decision criteria
- AdamW optimizer state memory overhead calculation
- **Tensor Core utilization measurement**: Know they accelerate GEMM, don't know how to measure *(新增)*

---

## 📉 Action Items & Review Queue

### Immediate (Next Session)
- [x] 验证 benchmarking 脚本在 8 卡环境下并行运行结果
- [x] Verify the concurrency fix resolves all OOM cases in benchmark
- [x] 学习如何用 Nsight Systems 分析 NVTX 标记的时间线 *(完成)*
- [ ] Implement FlashAttention and compare profiling results *(新增)*
- [ ] Investigate OOM causes for XL/2.7B models at context_length=1024 *(新增)*

### Short-term (This Week)
- [ ] 了解 Slurm `--gres=gpu:N` 与 CUDA_VISIBLE_DEVICES 的映射关系
- [ ] Review FlashAttention paper - understand the SRAM vs HBM movement
- [ ] Implement memory profiling using `torch.cuda.memory_summary()`
- [ ] Learn `nsys stats` CLI commands for automated data extraction *(新增)*
- [ ] Practice calculating arithmetic intensity for Attention operations *(新增)*

### Long-term (Before Assignment Due)
- [ ] Compare benchmark results with theoretical FLOPs calculations
- [ ] Explore mixed precision training impact on memory and speed
- [ ] Study Tensor Core architecture and how to maximize utilization *(新增)*

---

## 📚 Session History

| Date | Topic | Key Outcome |
|------|-------|-------------|
| 2026-01-30 | Benchmarking Code Review | 掌握 Forward/Backward 边界、GPU 设备选择机制 |
| 2026-01-30 | OOM Diagnosis & Concurrency Fix | Diagnosed submitit OOM by restructuring batch submission pattern |
| **2026-01-31** | **Nsys Profiling & Kernel Analysis** | **Mastered GEMM kernel naming, understood memory-bound vs compute-bound, analyzed Softmax overhead motivating FlashAttention** |

---

## 🎯 Course Goals (Assignment 2 - Systems)
- [x] Build benchmarking infrastructure with multiple model sizes
- [x] Profile Forward/Backward with Nsys, analyze kernel distribution *(新完成)*
- [ ] Implement memory-efficient attention (FlashAttention)
- [ ] Implement distributed training strategies
- [ ] Optimize training throughput

---

## 💡 Key Insights Archive

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
