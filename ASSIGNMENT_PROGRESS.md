# CS336 Assignment 2 进度追踪看板

**最后更新**: 2026-02-05  
**当前会话**: FlashAttention-2 实现阶段

---

## 📊 总体进度

| 模块 | 总分 | 已完成 | 剩余 | 状态 | 完成度 |
|------|-----|-------|-----|------|-------|
| 1.1 Profiling & Benchmarking | 16 pts | 16 pts | 0 pts | ✅ 完成 | 100% |
| 1.2-1.3 FlashAttention-2 | 29 pts | 2 pts | 27 pts | 🔴 当前重点 | 7% |
| 2. Distributed Data Parallel | 32 pts | 0 pts | 32 pts | ⚪ 未开始 | 0% |
| 3. Optimizer State Sharding | 20 pts | 0 pts | 20 pts | ⚪ 未开始 | 0% |
| **总计** | **97 pts** | **18 pts** | **79 pts** | - | **18.6%** |

---

## ✅ 已完成详细清单

### 1.1 Profiling & Benchmarking (16/16 pts)

| 问题 | 分数 | 状态 | 关键产出 | 相关文件 |
|------|-----|------|---------|---------|
| `benchmarking_script` | 4 pts | ✅ | 完整的 benchmark pipeline | `benchmark_runner.py`, `benchmark_cli.py` |
| `nsys_profile` | 5 pts | ✅ | NVTX 标注、kernel 分析 | `sessions/2026-01-31.md` |
| `mixed_precision_accumulation` | 1 pt | ✅ | FP16 spacing 理解 | `mixed_precision_accumulation.py` |
| `benchmarking_mixed_precision` | 2 pts | ✅ | BF16 性能对比表 | `sessions/2026-02-03.md` |
| `memory_profiling` | 4 pts | ✅ | `memory_viz` 分析 | `sessions/2026-02-04.md` |

### 1.2 PyTorch Attention Benchmarking (2 pts)

| 问题 | 分数 | 状态 | 关键发现 |
|------|-----|------|---------|
| `pytorch_attention` | 2 pts | ✅ | O(N²) activation memory 是瓶颈，seq_len=16384 时约 33 GB |

**已完成的核心成果**:
- ✅ 20 种配置 (d_model × seq_len) 的完整 benchmark
- ✅ 内存分解：cuBLAS workspace (64 MB) + QKV + Attention scores
- ✅ 理论 OOM 边界估算
- ✅ PyTorch Autograd 内部机制完全理解

---

## 🔴 当前任务 (阻塞中)

### FlashAttention-2 实现 (27 pts 剩余)

```
当前 Action Item: get_flashattention_autograd_function_pytorch()
文件位置: tests/adapters.py (Line 9-19)
状态: 🔴 raise NotImplementedError
Blocker: 需要从理解 -> 实现 的转化
```

| 子任务 | 分数 | 状态 | 预估时间 | 依赖 |
|-------|-----|------|---------|-----|
| PyTorch Forward | 7 pts | 🔴 未开始 | 1-2 天 | 无 |
| Triton Forward | 8 pts | ⚪ 未开始 | 2-3 天 | PyTorch 版本正确 |
| Causal Masking | - | ⚪ 未开始 | 0.5 天 | Triton Forward |
| Backward | 5 pts | ⚪ 未开始 | 1 天 | Forward 正确 |
| Benchmarking | 5 pts | ⚪ 未开始 | 0.5 天 | 全部正确 |
| Leaderboard (opt) | - | ⚪ 未开始 | 2-3 天 | 基础实现 |

**Algorithm 1 FlashAttention-2 Forward Pass - 实现检查清单**:

- [ ] 实现 query tiling 外循环 (split Q into Tq tiles)
- [ ] 实现 key/value tiling 内循环
- [ ] 实现 S_ij = Q_i @ K_j^T / sqrt(d)
- [ ] 实现 running maximum m_ij = max(m_ij-1, rowmax(S_ij))
- [ ] 实现 running softmax denominator l_ij
- [ ] 实现输出累加 O_ij
- [ ] 实现最终归一化 O_i = O_i / l_i
- [ ] 计算 logsumexp L_i = m_i + log(l_i)
- [ ] 保存 L, Q, K, V, O 用于 backward
- [ ] 通过 test_flash_forward_pass_pytorch

**关键代码结构**:
```python
class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # TODO: 实现 Algorithm 1
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("留给下一部分")
```

---

## ⚪ 待办模块 (未开始)

### 2. Distributed Data Parallel (32 pts)

| 子模块 | 分数 | 难度 | 预估时间 |
|-------|-----|------|---------|
| 2.1 Single-Node Communication | 5 pts | ⭐⭐ | 1 天 |
| 2.2 Naïve DDP | 8 pts | ⭐⭐⭐ | 2 天 |
| 2.3 Improved DDP | 19 pts | ⭐⭐⭐⭐ | 3-4 天 |
| 2.4 4D Parallelism | 10 pts | ⭐⭐⭐ | 1-2 天 |

**需实现的 Adapters**:
- `get_ddp_individual_parameters()` - Line 38-56
- `ddp_individual_parameters_on_after_backward()` - Line 59-71
- `get_ddp_bucketed()` - Line 74-92
- `ddp_bucketed_on_after_backward()` - Line 95-107
- `ddp_bucketed_on_train_batch_start()` - Line 110-120

### 3. Optimizer State Sharding (20 pts)

| 子模块 | 分数 | 难度 | 预估时间 |
|-------|-----|------|---------|
| `optimizer_state_sharding` | 15 pts | ⭐⭐⭐⭐ | 2-3 天 |
| `optimizer_state_sharding_accounting` | 5 pts | ⭐⭐⭐ | 1 天 |

**需实现的 Adapter**:
- `get_sharded_optimizer()` - Line 123-139

---

## 📅 时间规划

### 本周计划 (02-05 至 02-12) - FlashAttention 冲刺周

```
Day 1 (Today - 02-05): FlashAttention PyTorch Forward
├── 目标: 完成 tests/adapters.py::get_flashattention_autograd_function_pytorch()
├── 任务:
│   ├── [ ] 实现基本的 tiling 循环
│   ├── [ ] 实现 Online Softmax (m, l 的更新)
│   └── [ ] 通过 test_flash_forward_pass_pytorch
└── 产出: 可工作的 PyTorch FlashAttention forward

Day 2 (02-06): FlashAttention PyTorch Forward 调试
├── 目标: 确保数值正确性
├── 任务:
│   ├── [ ] 与 naïve attention 对比输出
│   ├── [ ] 修复 numerical precision 问题
│   └── [ ] 通过所有 edge cases
└── 产出: 正确的 reference implementation

Day 3-4 (02-07 to 02-08): FlashAttention Triton Forward
├── 目标: 完成 Triton kernel
├── 任务:
│   ├── [ ] 将 PyTorch 逻辑移植到 Triton
│   ├── [ ] 实现 causal masking
│   └── [ ] 通过 test_flash_forward_pass_triton
└── 产出: Triton FlashAttention

Day 5 (02-09): FlashAttention Backward
├── 目标: 完成 backward
├── 任务:
│   ├── [ ] 使用 torch.compile 实现 backward
│   ├── [ ] 实现 D vector 计算
│   └── [ ] 通过 test_flash_backward
└── 产出: 完整的 FlashAttention-2

Day 6 (02-10): Benchmarking
├── 目标: 性能对比
├── 任务:
│   ├── [ ] 编写 benchmark 脚本
│   ├── [ ] 对比 PyTorch vs FlashAttention
│   └── [ ] 生成对比表格
└── 产出: 性能报告

Day 7 (02-11): Buffer / Leaderboard (可选)
└── 产出: Leaderboard 提交 (如有时间)
```

### 长期规划 (至作业截止)

```
Week 1 (现在): FlashAttention (29 pts)
├── 目标: 全部完成
└── 截止: 2026-02-12

Week 2: DDP 基础 (18 pts)
├── 目标: 完成 2.1, 2.2, 2.3.1
└── 截止: 2026-02-19

Week 3: DDP 优化 + 4D Parallelism (24 pts)
├── 目标: 完成 2.3.2, 2.3.3, 2.4
└── 截止: 2026-02-26

Week 4: Optimizer Sharding (20 pts)
├── 目标: 全部完成
└── 截止: 2026-03-05

Week 5: Buffer / 润色 / 最终提交
└── 截止: 作业 DDL
```

---

## 🎯 每日执行检查清单

### 今日 (2026-02-05) 任务

- [ ] 30 分钟时间盒：写出 FlashAttention2 类框架
- [ ] 实现 Algorithm 1 的伪代码转 Python
- [ ] 通过至少 1 个简单的 forward pass test
- [ ] 更新本文件中的进度状态

### 通用每日模板

```markdown
## YYYY-MM-DD

### 今日目标
[1-2 个具体、可衡量的目标]

### 完成任务
- [ ] 任务 1
- [ ] 任务 2
- [ ] 任务 3

### 遇到的问题
- 问题 1: [描述]
- 问题 2: [描述]

### 明日计划
- 计划 1
- 计划 2

### 代码产出
[链接或文件路径]
```

---

## 🔍 关键障碍与解决策略

### 当前主要障碍

| 障碍 | 影响 | 解决策略 |
|-----|-----|---------|
| 理论 -> 实现转化慢 | FlashAttention 停滞 | 30 分钟时间盒，允许不完美实现 |
| 完美主义倾向 | 迟迟不敢开始写代码 | 明确区分 reference impl vs optimized impl |
| 实现信心不足 | 担心写错 | 用 PyTorch 版本作为调试基准 |

### 突破策略

1. **接受"不完美实现"的心态转变**
   - PyTorch FlashAttention 版本不需要优化，只需要正确
   - 允许自己写出丑陋的第一版

2. **30 分钟时间盒法**
   - Set a timer for 30 minutes
   - Goal: 完成 forward() 函数的框架，哪怕里面全是 pass

3. **从伪代码到实现**
   ```
   Step 1: 抄写 Algorithm 1 为 Python 伪代码 (15 min)
   Step 2: 为每个变量添加 shape 注释 (10 min)
   Step 3: 实现外循环 (query tiles) (20 min)
   Step 4: 实现内循环 (key tiles) (30 min)
   Step 5: 实现 online softmax 更新 (30 min)
   Step 6: 调试至通过测试 (剩余时间)
   ```

---

## 📈 学习速度追踪

### 速度指标

| 指标 | 当前 | 目标 | 状态 |
|-----|-----|-----|-----|
| 理论理解速度 | 快 | 正常 | ✅ 优秀 |
| 实现转化速度 | 慢 | 正常 | 🔴 需提升 |
| 调试迭代速度 | 未评估 | - | ⚪ 待观察 |
| 端到端交付速度 | 慢 | 正常 | 🔴 需提升 |

### 模式切换提醒

```
当前模式: 理解模式 (Student Mode)
需要切换: 建造模式 (Engineer Mode)

切换信号:
- 当你发现自己读论文超过 30 分钟而没有写代码时
- 当你在想"我是否完全理解了"而不是"我先试试看"时
- 当你在优化伪代码而不是写实际代码时
```

---

## 📚 相关资源链接

### 代码文件
- `tests/adapters.py` - 需要实现的接口
- `tests/test_attention.py` - FlashAttention 测试
- `cs336_systems/benchmark_attention.py` - 已完成 benchmark

### 学习记录
- `sessions/2026-02-05.md` - 当前会话
- `learning_ledger.md` - 知识图谱

### 参考资料
- FlashAttention-2 Paper (Algorithm 1)
- Triton Documentation
- PyTorch Autograd Function Tutorial

---

## 📝 更新日志

| 日期 | 更新内容 | 进度变化 |
|-----|---------|---------|
| 2026-02-05 | 创建进度追踪看板 | 初始版本 18.6% |

---

**下一步行动**: 打开 `tests/adapters.py`，开始实现 `get_flashattention_autograd_function_pytorch()`
