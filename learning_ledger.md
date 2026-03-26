# CS336 Learning Ledger
**Last Updated:** 2026-02-05

## Knowledge Graph (Mastery Levels)

- Mastered:
  - GPU benchmarking basics: warmup discipline; `torch.cuda.synchronize()` for accurate timings; separating warmup vs measurement.
  - CUDA memory profiling: `torch.cuda.memory_allocated()` vs `torch.cuda.max_memory_allocated()`; using `torch.cuda.reset_peak_memory_stats()` to measure phase-local peaks.
  - Naive Attention mechanics and scaling: standard attention materializes score/probability matrices (`S = QK^T/sqrt(d)`, `P = softmax(S)`) with O(seq_len^2) activation memory.
  - Metric interpretation: `add_mem_MB` as a proxy for activations that remain live after forward for backward; `mem_max_MB` as transient peak driven by intermediates/workspaces.

- Developing:
  - `torch.compile` performance model: kernel fusion and layout/contiguity effects; when compile overhead dominates; why speedups improve at long sequence lengths.
  - Distinguishing compile-time allocations/caches from steady-state runtime memory and time.
  - Attribution in end-to-end models: separating Attention vs MLP vs LayerNorm/residual contributions to speed and memory.

- Blind Spots:
  - FlashAttention-2 algorithm: tiling strategy; online softmax (LSE) bookkeeping; SRAM vs HBM data movement; backward recomputation vs saved statistics.
  - Numerics: stability of online softmax; FP16/FP8 interactions.
  - Parallelism and kernel-level analysis: using profiler traces to reason about bandwidth vs compute limits.

## Action Items and Review Queue

1. Part (b) benchmark: compile the full Transformer with `torch.compile(model)` and report (a) forward-only timing and (b) full step timing (forward+backward+optimizer).
2. Isolate compile overhead: record first-iteration latency separately from steady-state; increase warmup; document whether numbers include compilation.
3. Use a profiler (e.g., `torch.profiler` or `nsys`) to compare kernel counts and identify whether speedups come from fewer launches vs faster kernels.
4. FlashAttention prep: restate the key goal as eliminating full `[B, N, N]` materialization in HBM; review online softmax derivation and what must be saved for backward.
