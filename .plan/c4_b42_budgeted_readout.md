# Phase C4-B4.2: Module Budgeted Readout (预算型读出)

## 1. 动机与数学必要性

### 1.1 B4.0/B4.1 的数学局限

B4.0 和 B4.1 使用的 concave 读出（power-law x^α）本质上是 **齐次函数的份额归一化**：

```
R_f[m] = (raw_m)^α / Σ_k (raw_k)^α
```

这是 **share-preserving** 变换——对于 K 个模块，`R_f` 仍然满足 `Σ R_f = 1`，且保序（单调性不变）。即使 α→0+，`R_f[m]` 的大小关系仍然由 `raw_m` 的大小关系完全决定。

**因此**：只要 module 0 的 effective_size 仍是最大（即使 alive_frac_gap=0.589，effective 192×0.41=79 vs 16×1.0=16，ratio 4.9:1），任何 share 型读出都不会翻转 winner。

### 1.2 B4.2 的突破点

引入 **外部非比例约束** (cap)：

```
score_m = min(raw_m, cap)                    # Hard cap
score_m = cap · (1 - exp(-raw_m / cap))      # Soft cap (saturating)
score_m = cap · log(1 + raw_m / cap)         # Soft cap (log)
```

**关键区别**：cap 不依赖于 raw_m 的相对比例，而是一个 **绝对上界**。

当 module 0 的 raw_0 >> cap 时，score_0 ≈ cap，而小模块 raw_7 < cap 时 score_7 ≈ raw_7。

此时 score ratio = cap / raw_7 << raw_0 / raw_7，**打破了线性累加不变量**。

### 1.3 路线1终局目标

> B4.2 不是为了让系统"变聪明"，而是验证：
> **当读出层不再等价于线性计数时，P2 是否仍然必然 module 0 赢？**
>
> - 若仍然：动力学层完全 IID-like，要转路线2（供给侧 Poiseuille）或路线3（任务驱动输入）
> - 若翻转：读出层的非线性约束可以传导到竞争结构 → 闭环 B4.3

---

## 2. 实验设计

### 2.1 底盘配置（继承 B4.1 Q_EX champion）

| 参数 | 值 | 来源 |
|------|-----|------|
| N | 512 (GRID_SIZE=8) | C4 标准 |
| K | 8 | C4 标准 |
| P2 partition | [192, 96, 64, 48, 48, 32, 16, 16] | C4-A 标准 |
| max_frac | 0.375 (192/512) | — |
| μ | 20.0 | 窗口点 |
| λ | 0.01 | 窗口点 |
| β (extra-kill) | 0.050 per-step | B3.4 sweet spot |
| τ (queue) | 15 | B3.4 sweet spot |
| warmup | 500 steps | B3.4 标准 |
| direction | **invert** | B4.1 Q_EX 配置 |
| kill_map | **bottom2** | B4.1 Q_EX 配置 |
| queue_mode | **extra_only** | B4.1 Q_EX champion |
| κ (kill gain) | 2.0 | B4.1 Q_EX 配置 |
| steps | 6000 | C4 标准 |
| RANK_INTERVAL | 100 | B4.1 标准 |
| EMA_EPS | 0.2 | B4.1 标准 |
| TRACK_INTERVAL | 500 | C4 标准 |

### 2.2 Cap 设计

**自适应 cap** = `total_raw / K`（每 interval 的总 spikes 除以模块数）

- 物理含义：如果每个模块"理应"贡献相等份额，cap 就是该平均值
- 自适应性：随系统状态变化（不需要调参猜 fr）
- 对称性：cap 对所有模块相同 → 只惩罚"超额"模块

同时测试 **median-based cap** = `median(raw_m)` 作为对照：
- 更保守（中位数通常小于均值，因为大模块拉高均值）
- 当 P2 高度不等分时，median 比 mean 更能保护中间模块

### 2.3 L0 Cell 矩阵（8 cells × 1 seed）

| # | Label | Readout | Cap Source | Formula | 目的 |
|---|-------|---------|-----------|---------|------|
| 1 | RAW | raw sum | — | score_m = raw_m | 基线（必须 = B4.1 CTRL） |
| 2 | CONC | concave α=0.25 | — | score_m = raw_m^0.25 / Σ | 历史对照（B4.0 最强） |
| 3 | HC_EQ | hard cap | total/K | score_m = min(raw_m, total/K) | 均等预算硬上限 |
| 4 | HC_MED | hard cap | median | score_m = min(raw_m, median(raw)) | 中位数硬上限 |
| 5 | SC_EQ | soft cap (1-exp) | total/K | score_m = cap·(1−exp(−raw/cap)) | 均等预算软上限 |
| 6 | SC_MED | soft cap (1-exp) | median | score_m = cap·(1−exp(−raw/cap)) | 中位数软上限 |
| 7 | LOG_EQ | soft cap (log) | total/K | score_m = cap·log(1+raw/cap) | 对数软上限 |
| 8 | TOPQ | top-q pooling | — | score_m = mean(top_q(spikes_i in m)), q=5 | 非加法聚合读出 |

> TOPQ 作为第二类机制的代表：不截断总量，而是换聚合方式（per-unit 密度的 top-q 均值）。
> 如果 TOPQ 翻转而 HC/SC 不翻转，说明问题在聚合方式而非上限约束。

### 2.4 用法：两层读出（Measurement Layer + Ranking Layer）

**B4.2 测试两种部署模式**：

**Mode A: 纯测量层**（像 B4.0）
- 引擎 dynamics 不变，只在 TRACK_INTERVAL 用 budgeted readout 重新计算 winner
- 目的：验证 "如果我们用预算读出衡量，谁赢？"

**Mode B: 闭环排名层**（像 B4.1）
- 把 budgeted readout 替换 B4.1 的 signal_raw 作为 P3 ranking signal
- 目的：验证 "如果我们用预算读出驱动 kill ranking，能否翻转？"

**L0 先跑 Mode A**（8 cells），看哪些 readout 在测量层翻转 winner。
**若有翻转 → L1 跑 Mode B**（闭环），看翻转是否能传导到 raw layer。

### 2.5 L1 / EQ 设计

**L1（3 seeds × top 2 cells）**：
- 从 L0 选出 winner_budget ≠ 0 的 cells（如果存在），或 dom95_budget 最小的 cells
- P2 partition，seeds = {42, 0, 1}

**EQ Control（3 seeds × 1 cell）**：
- 选 L1 最佳 cell，换 EQ partition (64×8)
- 验证预算读出在等分条件下的行为

---

## 3. 关键诊断列（CSV Schema）

### 3.1 新增列（B4.2 特有）

| Column | Type | Description |
|--------|------|-------------|
| `readout_type` | str | 'raw', 'concave', 'hard_cap_eq', 'hard_cap_med', 'soft_cap_eq', 'soft_cap_med', 'log_eq', 'topq' |
| `cap_source` | str | 'none', 'total_div_K', 'median' |
| `mode` | str | 'measurement' (A) or 'closed_loop' (B) |
| **Budget Layer Metrics** | | |
| `winner_budget` | int | Module with highest budgeted score |
| `dom95s_budget` | float | 95th %ile budgeted dominance (late half) |
| `margin_budget_mean` | float | Mean(score_sorted[0] - score_sorted[1]) |
| `winner_budget_stability` | float | Fraction of intervals budget winner stays same |
| `cap_hit_ratio_m` | str | Per-module cap hit ratio (JSON array, K floats) — 模块有多少比例 interval 达到 cap |
| `cap_hit_ratio_0` | float | Module 0 的 cap hit ratio（核心：应接近 1.0） |
| `cap_mean` | float | Mean cap value across intervals |
| `overflow_0_mean` | float | Module 0 mean(raw_0 - cap) / cap — 超额比例 |
| **Cross-Layer Agreement** | | |
| `agree_rate_budget_raw` | float | Fraction of intervals where winner_budget == winner_raw |
| `flip_rate_budget` | float | winner_budget ≠ winner_raw 的比例 |
| `rank_budget_0_mode` | int | Module 0 在 budget readout 中的众数排名 |
| **Raw Layer (继承 B4.1)** | | |
| `winner_raw` | int | 原始 spike-sum winner |
| `dom95s_raw` | float | 原始 dominance |
| `margin_raw_mean` | float | 原始 margin |
| **Workforce (继承 B3.4/B4.1)** | | |
| `alive_frac_gap_p95` | float | 活力不对称 |
| `alive_frac_min_p05` | float | 最小活力 |
| `corr_rank_af` | float | 排名→活力因果链 |
| **System Health** | | |
| `dR` | float | 窗口指标 |
| `fr` | float | 发火率 |
| `dr` | float | 死亡率 |
| `health` | str | ALIVE/MARGINAL/DEAD |

### 3.2 cap_hit_ratio 的计算

```python
# 每个 interval，对每个模块 m:
cap_hit[m] += 1 if raw_m >= cap * 0.95  # 95% threshold (not exact, avoids edge noise)

# 最终:
cap_hit_ratio_m[m] = cap_hit[m] / n_intervals
```

---

## 4. Verdict 判据

### 4.1 Tier 0: 实现安全检查

```
T0-1: RAW cell 的 winner_budget == winner_raw AND dom95s_budget ≈ dom95s_raw (±0.001)
      (RAW readout 必须和原始完全一致)
T0-2: ALL cells health ∈ {ALIVE, MARGINAL}
      (预算读出不应该影响 dynamics)
T0-3: CONC cell 的 winner_budget 与 B4.0 P025 结果一致
      (历史对照)
```

### 4.2 Tier 1: Budget Winner Flip 分类（Mode A）

对每个 budget cell (HC_EQ, HC_MED, SC_EQ, SC_MED, LOG_EQ, TOPQ):

```
IF winner_budget ≠ 0 (module 0 不是 budget winner):
  → 标记 BUDGET_FLIP

  IF winner_budget_stability > 0.6:
    → BUDGET_FLIP_STABLE ★★★
  ELIF winner_budget_stability > 0.3:
    → BUDGET_FLIP_MODERATE ★★
  ELSE:
    → BUDGET_FLIP_NOISY ★

ELIF dom95s_budget < 1/K + 0.02 (≈ 0.145):
  → BUDGET_EQUALIZED (读出层已去规模化，但无稳定新赢家)

ELIF cap_hit_ratio_0 > 0.8 AND margin_budget_mean < margin_raw_mean * 0.3:
  → BUDGET_CAPPED (大模块确实被限制，但 margin 仍正)

ELSE:
  → BUDGET_INEFFECTIVE
```

### 4.3 Tier 2: 闭环传导检查（Mode B，仅当 Tier 1 有 BUDGET_FLIP）

```
IF winner_raw ≠ 0 (闭环传导到 raw layer):
  → RAW_FLIP ★★★★★ (路线1突破！)

ELIF dom95s_raw < dom95s_raw(RAW baseline) - 0.03:
  → RAW_SHIFT (raw layer 有响应但未翻转)

ELSE:
  → FEEDBACK_ABSORBED (闭环效果被吸收)
```

### 4.4 Global Verdict

```
IF any RAW_FLIP:
  → VERDICT: BUDGET_BREAKS_INVARIANT
  → Next: B4.3 稳定化 + 参数扫描

ELIF any BUDGET_FLIP_STABLE and FEEDBACK_ABSORBED:
  → VERDICT: BUDGET_FLIP_NO_PROPAGATION
  → 读出层已去规模化，但动力学层 IID → 需路线2/3

ELIF any BUDGET_EQUALIZED:
  → VERDICT: BUDGET_EQUALIZED_NO_WINNER
  → 预算去规模化成功但无新赢家 → 干净的负结论

ELIF all BUDGET_INEFFECTIVE:
  → VERDICT: BUDGET_INEFFECTIVE
  → cap 约束不够强 or 实现问题

ELSE:
  → VERDICT: PARTIAL (部分有效)
```

---

## 5. 实现计划

### 5.1 文件结构

```
experiments/phase_c4/run_c4b42_budgeted.py    # 新脚本（~1000 行）
```

### 5.2 代码结构

```python
# ── Imports ──
from experiments.phase_c4.run_c4a_bottleneck import (
    make_labels_scatter, P2_SIZES, K_FIXED, MAX_FRAC, MAX_MODULE_ID,
    TRACK_INTERVAL, STEADY_N, write_csv,
)
from experiments.phase_c4.run_c4b4_0_budget import concave_transform  # 复用

# ── Constants (B3.4 + B4.1 Q_EX sweet spot) ──
# 继承 B4.1 闭环底盘 + 新增 cap 参数

# ── budgeted_score() ──
def budgeted_score(raw_m, readout_type, **kwargs):
    """核心：计算预算得分。
    返回 (K,) array 的 budgeted scores。
    """

# ── _run_c4b42_worker() ──
# 继承 B4.1 的 PRE/CYCLE/POST 管道
# Mode A: 只在 TRACK_INTERVAL 计算 budgeted readout（测量层）
# Mode B: 在 RANK_INTERVAL 用 budgeted readout 替换 signal_raw 驱动 P3 kill ranking

# ── build_l0_cells() ──
# 8 cells (上面的矩阵)

# ── L0 → L1 gate → L1 → EQ ──
# 三层实验

# ── print_verdict() ──
# 上面的 4 层判据

# ── main() ──
# --smoke, --l0-only, --steps, --seeds, --workers
```

### 5.3 继承与复用

- `concave_transform()`：从 B4.0 import（历史对照 cell）
- `make_labels_scatter()`：从 C4-A import（散列分区）
- `write_csv()`：从 C4-A import（CSV 输出）
- PRE/CYCLE/POST 管道：从 B4.1 复制并修改 P3 ranking（Mode B）

### 5.4 新增函数

```python
def budgeted_score(raw_m, readout_type, cap_source='total_div_K'):
    """
    raw_m: (K,) per-module raw spike counts for interval
    readout_type: str
    cap_source: 'total_div_K' or 'median'
    Returns: (K,) budgeted scores
    """
    if readout_type == 'raw':
        return raw_m.copy()
    elif readout_type == 'concave':
        return concave_transform(raw_m, 'power', alpha=0.25)

    # Compute cap
    if cap_source == 'total_div_K':
        cap = raw_m.sum() / len(raw_m)
    elif cap_source == 'median':
        cap = np.median(raw_m)
    else:
        cap = raw_m.sum() / len(raw_m)  # fallback

    cap = max(cap, 1e-12)  # safety floor

    if readout_type == 'hard_cap':
        return np.minimum(raw_m, cap)
    elif readout_type == 'soft_cap':
        return cap * (1.0 - np.exp(-raw_m / cap))
    elif readout_type == 'log_cap':
        return cap * np.log1p(raw_m / cap)
    elif readout_type == 'topq':
        # 需要 per-unit spikes，此处 raw_m 不够 → 见 5.5 特殊处理
        raise NotImplementedError("TOPQ requires per-unit data")

    return raw_m.copy()
```

### 5.5 TOPQ 特殊处理

TOPQ 需要 per-unit spikes（不是 per-module 聚合），因此在 worker 内需要额外记录：

```python
# 在 cycle 后记录 per-unit spikes
unit_spikes = grid.spike.copy()  # (N,) 0/1 binary

# 在 TRACK_INTERVAL 计算 TOPQ：
for m in range(K):
    mask_m = labels == m
    spk_m = interval_unit_spikes[mask_m]  # 该 interval 内的累积 spikes
    n_alive = (grid.alive_mask[mask_m] == 1).sum()
    if n_alive >= q:
        # top_q mean of per-unit firing in this interval
        fr_m = spk_m / n_interval_steps
        topq_vals = np.partition(fr_m, -q)[-q:]
        score_topq[m] = topq_vals.mean()
    else:
        score_topq[m] = spk_m.sum() / (n_alive + 1e-12) / n_interval_steps
```

这需要在 worker 中维护 `interval_unit_spikes[N]` 累加器（每 TRACK_INTERVAL 重置）。

---

## 6. 预期运行规模

| Layer | Cells | Seeds | Runs | 预计时间 |
|-------|-------|-------|------|---------|
| L0 (Mode A) | 8 | 1 | 8 | ~2 min |
| L1 (Mode A) | 2 best | 3 | 6 | ~2 min |
| L1 (Mode B, if flip) | 2 best | 3 | 6 | ~2 min |
| EQ | 1 best | 3 | 3 | ~1 min |
| **Total** | | | **17-23** | **~5-7 min** |

轻量实验，信息密度极高。

---

## 7. 输出路径

```
output/phase_c4a_bottleneck/c4b42_l0_results.csv
output/phase_c4a_bottleneck/c4b42_results.csv     # L0 + L1 + EQ combined
```

---

## 8. 判断路线1是否到终点

B4.2 结束后的决策树：

```
IF BUDGET_BREAKS_INVARIANT:
  → 路线1 继续：B4.3 参数扫描 + 稳定化

IF BUDGET_FLIP_NO_PROPAGATION:
  → 路线1 终结：读出层可去规模化但动力学层 IID
  → 转路线2（V4-Flow Poiseuille 供给侧竞争）或路线3（任务驱动输入）

IF BUDGET_EQUALIZED_NO_WINNER:
  → 路线1 终结（干净负结论）
  → 去规模化成功但竞争力均等 → 没有新赢家 → dynamics 是根本原因

IF BUDGET_INEFFECTIVE:
  → 可尝试 TOPQ + 极端 cap（cap = min(raw_m)）
  → 若仍无效 → 路线1 彻底终结
```
