# CS336 Learning Ledger
**Last Updated:** 2026-02-04

## 🧠 Knowledge Graph (Mastery Levels)
- 🟢 Mastered: Benchmarking warmup discipline; NVTX ranges for phase labeling; `torch.cuda.synchronize()` for accurate GPU timing; mixed precision fundamentals (FP16/BF16 vs FP32, autocast allow/deny lists); AdamW state concept (per-parameter m/v); PyTorch CUDA memory snapshot workflow (`_record_memory_history` + `_dump_snapshot` + memory_viz); inference (`inference_mode`) vs training forward memory semantics; autocast weight-cast cache behavior and `cache_enabled` effect on memory timelines.
- 🟡 Developing: Designing clean profiling runs (separating warmup vs record window) while keeping stack traces attributable; interpreting “Active” vs allocator artifacts; organizing peak memory extraction across many configurations.
- 🔴 Blind Spots: FlashAttention implementation; quantization memory accounting; gradient checkpointing trade-offs; TP/PP decision criteria; measuring Tensor Core utilization; long-run numerical stability effects.

## 📉 Action Items & Review Queue
- Extract exact peak “Active memory” for ctx {128, 256, 512} for (forward-only, full-step) × (FP32, BF16) and compile the required table.
- Capture and save the two required memory_viz screenshots for the 2.7B model (forward-only and full-step).
- For (e), ensure stack traces are captured reliably by adjusting recording window and using `stacks='all'`/appropriate context if allocations show “no frames”.
