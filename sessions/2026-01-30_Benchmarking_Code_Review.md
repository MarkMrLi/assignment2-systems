# Session Log: Benchmarking Script Code Review
**Date:** 2026-01-30
**Focus:** 审查 PyTorch 训练性能测量脚本的正确性

---

## Key Concepts Discussed

### 1. 梯度清零 (`zero_grad()`) 的必要性
- **问题**: 测量循环中缺少 `optimizer.zero_grad()`，导致梯度累积
- **影响**: 主要影响测量一致性和数值稳定性，而非计算时间本身
- **结论**: 每次迭代应独立，确保 std 统计有意义

### 2. CUDA 设备选择机制
- `CUDA_VISIBLE_DEVICES`: 环境变量，必须在 `import torch` 之前设置
- `torch.cuda.set_device()`: 可以在运行时调用，适合单机多卡
- **Slurm 场景**: Slurm 会重映射 GPU 编号，需要特殊处理
- **单机场景**: 直接使用 `set_device(physical_id)` 即可 ✅

### 3. Forward Pass 的边界定义
- **核心理解**: Forward Pass = 构建计算图的全过程
- `model.forward(x)` → 构建计算图 → Forward
- `cross_entropy(y_hat, targets)` → 构建计算图 → Forward
- `loss.backward()` → 消费计算图 → Backward
- **结论**: Loss 计算属于 Forward，因为它是 backward 的起点

---

## Technical Breakthroughs
- 理解了 PyTorch 中 Forward/Backward 的本质区分：是否在构建 autograd 计算图
- 掌握了 `torch.cuda.synchronize()` 在 GPU 时间测量中的必要性
- 理解了单机多卡 vs 集群环境下 GPU 选择的不同策略

---

## Code Improvements Made
- [x] 添加 `zero_grad()` 到测量循环
- [x] 将 `cross_entropy` 移入 Forward 测量范围
- [x] 使用 `pathlib` 替代硬编码路径
- [x] `eval()` 重命名为 `run_benchmark()`
- [x] 使用 `torch.cuda.set_device()` 替代环境变量方式
- [x] 修复 `args.context_length` 未使用的问题

---

## Unresolved Issues
- `device_ids = [0, 4, 5, 6, 7]` 配置是否符合预期？（跳过了 1, 2, 3）
