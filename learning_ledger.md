# CS336 Learning Ledger
**Last Updated:** 2026-02-05

## 🧠 Knowledge Graph (Mastery Levels)

### 🟢 Mastered
- **GPU Benchmarking Fundamentals**: Warmup discipline; `torch.cuda.synchronize()` for accurate timing; NVTX ranges for phase labeling; separate warmup vs measurement phases
- **Memory Profiling**: `torch.cuda.memory_allocated()` vs `max_memory_allocated()`; memory measurement timing; decomposing peak memory into components (parameters, activations, CUDA overhead)
- **Naïve Attention Mechanics**: Full matrix materialization O(seq_len²); attention score computation (QK^T/√d_k); softmax operation; output computation (PV)
- **PyTorch Autograd Internals**: Leaf nodes vs intermediate tensors; lazy gradient computation; why intermediate tensors (S, P, O) don't persist `.grad`; memory reuse during backward

### 🟡 Developing
- **Memory Optimization Strategies**: Gradient checkpointing trade-offs (compute vs memory); understanding when to apply checkpointing
- **Scaling Analysis**: O(N²) vs O(N) memory components; estimating OOM boundaries; GPU memory budget planning

### 🔴 Blind Spots
- **FlashAttention Implementation**: Tiled attention algorithm; block-wise softmax with online normalization; SRAM vs HBM data movement optimization
- **Numerical Stability**: Online softmax algorithm for LSE (log-sum-exp); FP16/FP8 precision considerations
- **Quantization**: FP8/FP4 memory accounting; mixed-gate architectures
- **Advanced Parallelism**: Tensor Parallel (TP) vs Pipeline Parallel (PP) decision criteria
- **Tensor Core Utilization**: Measuring actual utilization; kernel-level profiling

## 📉 Action Items & Review Queue

### Immediate (Next Session)
1. **Implement FlashAttention-2 PyTorch version**:
   - Follow FlashAttention-2 paper algorithm with tiling
   - Implement online softmax with log-sum-exp (LSE) saving
   - Complete `get_flashattention_autograd_function_pytorch()` in `tests/adapters.py`

2. **Verify FlashAttention correctness**:
   - Compare outputs with naïve attention (forward pass)
   - Compare gradients with naïve attention (backward pass)
   - Test on attention test cases (including causal masking)

### Short-term
3. **Memory comparison**:
   - Benchmark FlashAttention memory vs naïve attention
   - Verify O(N × block_size) vs O(N²) scaling claim
   - Measure actual memory savings at seq_len=4096, 8192, 16384

4. **Performance comparison**:
   - Time FlashAttention forward and backward
   - Compare with naïve attention timings
   - Identify compute-vs-memory trade-off sweet spot

## 📊 Session Summary (2026-02-05)

### Deliverables Completed
- ✅ Created `cs336_systems/benchmark_attention.py` - dedicated attention benchmarking script
- ✅ Generated benchmark results across 20 configurations (d_model × seq_len)
- ✅ Produced comprehensive analysis document at `results/profile_attention/pytorch.md`

### Key Findings
| Metric | Observation |
|--------|-------------|
| Memory scaling | `add_mem_MB ≈ 2 × seq_len²` (FP32, batch=8) |
| seq_len=16384 memory | ~16 GB (activations) + 64 MB (cuBLAS) = ~33 GB peak |
| Forward time scaling | ~O(seq_len²) - dominated by matmul operations |
| OOM boundary | Between seq_len=16384 (~33 GB) and seq_len=32768 (~130 GB) |

### Memory Decomposition (seq_len=256)
```
Peak Memory = cuBLAS workspace (64 MB) 
            + QKV + gradients (~4.5 MB) 
            + Attention scores S, P (~4 MB)
            ≈ 72.77 MB (measured) vs 69 MB (theoretical)
```

### Core Insight
The naïve attention implementation's O(N²) activation memory is the fundamental bottleneck. For seq_len=16384, just the attention scores require ~16 GB (FP32), exceeding many GPUs. FlashAttention-2 addresses this by:
1. **Tiling**: Process attention in blocks (e.g., 64×64)
2. **Recomputation**: Don't store full S, P; recompute during backward
3. **Fusion**: Single kernel reduces memory traffic
