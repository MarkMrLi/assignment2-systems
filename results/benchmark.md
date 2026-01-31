# Model Benchmarking Analysis Report

## set up
h20 × 8

## 1. Benchmarking Results (warm-up step: 5) - Test A

| Model size | d_model | d_ff | num_layers | num_heads | Fwd Mean (s) | Fwd Std | Bwd Mean (s) | Bwd Std | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 768 | 3072 | 12 | 12 | 0.02 | 0 | 0.03 | 0 | 0.05 |
| Medium | 1024 | 4096 | 24 | 16 | 0.05 | 0 | 0.09 | 0 | 0.14 |
| Large | 1280 | 5120 | 36 | 20 | 0.11 | 0 | 0.2 | 0 | 0.32 |
| xl | 1600 | 6400 | 48 | 25 | 0.19 | 0 | 0.38 | 0 | 0.57 |
| 2.7B | 2560 | 10240 | 32 | 32 | 0.33 | 0.01 | 0.58 | 0 | 0.91 |

## 2. Benchmarking Results (warm-up step: 0)

| Model size | d_model | d_ff | num_layers | num_heads | Fwd Mean (s) | Fwd Std | Bwd Mean (s) | Bwd Std | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 768 | 3072 | 12 | 12 | 0.08 | 0.14 | 0.04 | 0.03 | 0.12 |
| Medium | 1024 | 4096 | 24 | 16 | 0.1 | 0.15 | 0.1 | 0.03 | 0.2 |
| Large | 1280 | 5120 | 36 | 20 | 0.17 | 0.16 | 0.21 | 0.02 | 0.38 |
| xl | 1600 | 6400 | 48 | 25 | 0.23 | 0.12 | 0.39 | 0.02 | 0.61 |
| 2.7B | 2560 | 10240 | 32 | 32 | 0.37 | 0.12 | 0.59 | 0.02 | 0.96 |

## 3. Benchmarking Results (warm-up step: 2)

| Model size | d_model | d_ff | num_layers | num_heads | Fwd Mean (s) | Fwd Std | Bwd Mean (s) | Bwd Std | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 768 | 3072 | 12 | 12 | 0.03 | 0.01 | 0.03 | 0 | 0.06 |
| Medium | 1024 | 4096 | 24 | 16 | 0.05 | 0 | 0.09 | 0 | 0.14 |
| Large | 1280 | 5120 | 36 | 20 | 0.12 | 0.01 | 0.2 | 0 | 0.32 |
| xl | 1600 | 6400 | 48 | 25 | 0.19 | 0 | 0.38 | 0 | 0.57 |
| 2.7B | 2560 | 10240 | 32 | 32 | 0.33 | 0.01 | 0.58 | 0 | 0.91 |

## 4. Benchmarking Results (warm-up step: 5) - Test B

| Model size | d_model | d_ff | num_layers | num_heads | Fwd Mean (s) | Fwd Std | Bwd Mean (s) | Bwd Std | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 768 | 3072 | 12 | 12 | 0.03 | 0 | 0.04 | 0 | 0.07 |
| Medium | 1024 | 4096 | 24 | 16 | 0.06 | 0.01 | 0.09 | 0 | 0.15 |
| Large | 1280 | 5120 | 36 | 20 | 0.17 | 0.02 | 0.32 | 0.03 | 0.49 |
| xl | 1600 | 6400 | 48 | 25 | 0.18 | 0 | 0.38 | 0 | 0.56 |
| 2.7B | 2560 | 10240 | 32 | 32 | 0.3 | 0 | 0.58 | 0 | 0.88 |

## 5. Benchmarking Results (warm-up step: 5) - Detailed (ms)

| Model size | d_model | d_ff | num_layers | num_heads | Fwd Mean (ms) | Fwd Std | Bwd Mean (ms) | Bwd Std | Total (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 768 | 3072 | 12 | 12 | 29.482 | 1.299 | 35.492 | 4.126 | 64.975 |
| Medium | 1024 | 4096 | 24 | 16 | 57.514 | 11.972 | 93.866 | 0.265 | 151.38 |
| Large | 1280 | 5120 | 36 | 20 | 172.18 | 23.171 | 322.045 | 36.435 | 494.224 |
| xl | 1600 | 6400 | 48 | 25 | 177.814 | 0.114 | 385.153 | 0.736 | 562.967 |
| 2.7B | 2560 | 10240 | 32 | 32 | 300.147 | 0.139 | 581.466 | 1.201 | 881.612 |

## Analyse
- (b) 向前传播时间从 29ms 增加到 300 ms，反向传播时间从 35ms 增加到 581 ms，反向传播时间逐渐接近向前传播两倍的理论值，可能是由于小模型的 forward 和 backward 过程主要是 memory bound，同时可以看到标准差很小(小于均值 1%)，说明训练进入稳定阶段
- (c) 不进行预热 forward 时间会显著提升且标准差极大，主要是因为首轮迭代包括一些 CUDA 内核加载、显存分配、算子优化等一次性开销
---

### Model Benchmarking Results with warm-up step:5

| **model_size** | **d_model** | **d_ff** | **num_layers** | **num_heads** | **context_length** | **status** | **Fwd Mean (ms)** | **Fwd Std (ms)** | **Bwd Mean (ms)** | **Bwd Std (ms)** | **Total (ms)** | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| Small | 768 | 3072 | 12 | 12 | 128 | success | 76.80 | 8.16 | 85.83 | 4.41 | 162.63 | 
| Medium | 1024 | 4096 | 24 | 16 | 128 | success | 143.23 | 4.05 | 159.63 | 3.29 | 302.86 | 
| Large | 1280 | 5120 | 36 | 20 | 128 | success | 202.31 | 3.79 | 220.45 | 4.02 | 422.76 | 
| xl | 1600 | 6400 | 48 | 25 | 128 | success | 266.70 | 2.61 | 290.30 | 3.39 | 557.00 | 
| 2.7B | 2560 | 10240 | 32 | 32 | 128 | success | 186.83 | 4.52 | 318.73 | 0.27 | 505.56 | 
| Small | 768 | 3072 | 12 | 12 | 256 | success | 75.67 | 4.76 | 86.34 | 9.13 | 162.01 | 
| Medium | 1024 | 4096 | 24 | 16 | 256 | success | 131.90 | 3.41 | 142.61 | 2.14 | 274.51 | 
| Large | 1280 | 5120 | 36 | 20 | 256 | success | 199.17 | 4.02 | 222.90 | 3.93 | 422.07 | 
| xl | 1600 | 6400 | 48 | 25 | 256 | success | 259.30 | 2.71 | 384.28 | 0.62 | 643.58 | 
| 2.7B | 2560 | 10240 | 32 | 32 | 256 | success | 328.13 | 8.34 | 581.11 | 1.05 | 909.24 | 
| Small | 768 | 3072 | 12 | 12 | 512 | success | 72.14 | 2.61 | 87.16 | 6.22 | 159.30 | 
| Medium | 1024 | 4096 | 24 | 16 | 512 | success | 142.20 | 9.61 | 184.14 | 0.31 | 326.34 | 
| Large | 1280 | 5120 | 36 | 20 | 512 | success | 206.08 | 0.82 | 391.87 | 0.58 | 597.96 | 
| xl | 1600 | 6400 | 48 | 25 | 512 | success | 386.92 | 0.22 | 768.69 | 1.29 | 1155.61 | 
| 2.7B | 2560 | 10240 | 32 | 32 | 512 | success | 670.15 | 8.85 | 1167.95 | 1.64 | 1838.10 | 
| Small | 768 | 3072 | 12 | 12 | 1024 | success | 70.98 | 1.31 | 132.42 | 0.17 | 203.40 | 
| Medium | 1024 | 4096 | 24 | 16 | 1024 | success | 204.93 | 0.53 | 398.73 | 0.26 | 603.65 | 
| Large | 1280 | 5120 | 36 | 20 | 1024 | success | 422.26 | 0.14 | 825.80 | 0.67 | 1248.06 | 
| xl | 1600 | 6400 | 48 | 25 | 1024 | OOM: CUDA Out of Memory | - | - | - | - | - | 
| 2.7B | 2560 | 10240 | 32 | 32 | 1024 | OOM: CUDA Out of Memory | - | - | - | - | - |

(a) 与使用 python 标准库测试的结果相同。

(b)

---

## 模型性能 Profiling 分析(一次)

### 1. Forward 阶段 Kernel 耗时统计

| 排名 | 占比 | 总耗时 | 实例数 | 平均耗时 | 中位数 | 最小值 | 最大值 | 标准差 | Kernel 名称 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Top 1** | 64.8% | 66.765 ms | 217 | 307.671 μs | 141.825 μs | 140.674 μs | 1.045 ms | 235.537 μs | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| **Top 2** | 21.7% | 22.349 ms | 72 | 310.405 μs | 311.315 μs | 40.928 μs | 580.006 μs | 270.468 μs | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |

---

### 2. Backward 阶段 Kernel 耗时统计

| 排名 | 占比 | 总耗时 | 实例数 | 平均耗时 | 中位数 | 最小值 | 最大值 | 标准差 | Kernel 名称 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Top 1** | 30.8% | 59.884 ms | 216 | 277.241 μs | 146.113 μs | 145.377 μs | 540.997 μs | 186.093 μs | `sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| **Top 2** | 18.5% | 35.878 ms | 72 | 498.309 μs | 498.149 μs | 495.877 μs | 500.741 μs | 1.045 μs | `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)` |

---

(c) 除了之前分析的矩阵乘法（GEMM）外，Forward 阶段排在第三、四位的 Kernel 均属于 **Element-wise（逐元素操作）** 类型。这类 Kernel 通常是 **Memory Bound（受显存带宽限制）** 的。


---

### 1. Forward 阶段辅助 Kernel 耗时统计 (Top 3 & Top 4)

| 排名 | 占比 | 总耗时 | 实例数 | 平均耗时 | 中位数 | 最小值 | 最大值 | 标准差 | Kernel 名称 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Top 3** | 2.8% | 2.930 ms | 434 | 6.750 μs | 6.752 μs | 5.984 μs | 8.352 μs | 589 ns | `void at::native::elementwise_kernel<MulFunctor<float>>` (逐元素乘法) |
| **Top 4** | 1.4% | 1.493 ms | 216 | 6.911 μs | 6.432 μs | 6.240 μs | 9.344 μs | 933 ns | `void at::native::elementwise_kernel<direct_copy_kernel_cuda>` (数据拷贝/类型转换) |

(d) 训练全流程相对于向前传播（forward only）矩阵乘法占比从 86.5% 下降到 69.4%，增加占比的是逐元素的一些算子从 5% ~ 8% 提升到 25% ~ 30%。
分析原因是整个训练过程中增加了激活函数的导数运算、梯度累加、特别是 AdamW 的逐元素更新参数操作

(e) 矩阵乘法和 softmax 对比

| 维度 | 矩阵乘法 (MatMul) | Softmax 操作 | 差异倍数 |
| --- | --- | --- | --- |
| **理论计算量 (FLOPs)** | 巨大 (TFLOPS 级别) | 极小 (GFLOPS 级别) | **100x ~ 1000x** |
| **算子实际执行 (GPU Time)** | ~300 μs | ~35 μs | **~8.5x** |
| **端到端感知 (Total Latency)** | ~300 μs | **~400 μs** | **< 1x (Softmax 反而更慢)** |

**结论：** 在现代 GPU 架构中，**FLOPs 已经完全无法代表运行时间**。Softmax 虽然 FLOPs 几乎为零，但由于其**破碎的 Kernel 结构**和**频繁的显存交换**，导致其在端到端耗时上反而成为了比 MatMul 更严重的瓶颈。

---

## Methodology Notes

- **Python timeit**: Measured wall-clock time with `torch.cuda.synchronize()` before timing to ensure GPU completion
- **Nsys NVTX**: Used `torch.cuda.nvtx.range()` to mark Forward/Backward phases, then filtered kernels in Nsight Systems GUI
- **Softmax end-to-end**: Added explicit NVTX annotation around the full `F.softmax()` call to capture all associated kernels

## Kernel Instance Analysis

For the Large model (36 layers), the top GEMM kernel has 217 instances:
- Per-layer GEMMs: 4 (Attention Q/K/V/O) + 2 (FFN up/down) = 6
- Total: 36 layers × 6 GEMMs = 216 ≈ 217 (+ embedding projection)

## GEMM Tile Size Observation

- **Forward** uses `tilesize128x128x8` (larger tiles, higher arithmetic intensity)
- **Backward** uses `tilesize64x64x8` (smaller tiles, due to higher register pressure from storing both activations and gradients simultaneously)

## Key Insights

1. **Memory-bound vs Compute-bound**: Small models are memory-bound (Fwd/Bwd ratio ~1.1x), while larger models become compute-bound (ratio approaches theoretical 2x)
2. **Warmup importance**: First iteration includes CUDA kernel compilation, memory allocation, and operator optimization overhead
3. **FlashAttention motivation**: The observed Softmax memory overhead (~400μs end-to-end vs ~35μs kernel time) demonstrates why fusing Softmax with Attention in SRAM is beneficial
