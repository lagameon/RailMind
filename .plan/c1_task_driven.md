# Phase C.1 — Minimal Task-Driven Demand Asymmetry Experiment

## 目标

验证：在非对称输入统计（prototype routing + frequency bias）下，
高频 prototype 对应的**小模块**是否能突破 Σspikes ∝ |m| 的 counting ceiling。

这是对 IID Regime Theorem 的最后一次反例尝试——只改需求层，不动供给/读出/死亡。

---

## 设计决策

### 注入点：activation.py 的 content match (A-1)

引擎的 A-1~A-4 流程：
```
A-1: m_i = z_i · x_t          (cosine match)
A-2: s_i = Σ_j w_ij · spike_j (synaptic input)
A-3: a_i = σ(g·(m_i + s_i) - θ_i) · (1-f_i)
A-4: spike_i = 1[a_i > A_FIRE]
```

在 A-1 之后，乘以 per-unit `input_boost`：`m_i *= boost_i`。
这只放大 input-driven 项，不影响 recurrent (s_i)、BCPNN、fatigue、metabolism。

### Prototype-Dependent Boost 机制

**关键挑战：** boost 依赖当前 step 的 proto_idx，但 proto_idx 在 cycle() 内部由 world.step() 产生。

**方案：在 engine.py 的 cycle() 中添加 hook**

Worker 在 cycle() 前设置一个 per-prototype boost mapping（静态，只设一次）：
```python
grid._input_boost_map = {
    0: boost_proto_A,  # (N,) array - module 2 units boosted
    1: boost_proto_B,  # (N,) array - module 1 units boosted
    2: boost_proto_C,  # (N,) array - module 0 units boosted
}
```

Engine 在 world.step() 返回后、compute_activation() 调用前，自动选择：
```python
if self._input_boost_map is not None:
    self._input_boost = self._input_boost_map.get(proto_idx)
```

然后 activation.py 在 A-1 后应用：
```python
m = content @ x_t
if input_boost is not None:
    m *= input_boost
```

### Prototype 频率不对称

不修改 world.py。在 worker 内 monkey-patch `grid.world.step()`：
```python
original_prototypes = grid.world.prototypes  # (3, D)
custom_rng = np.random.RandomState(seed + 99999)
def custom_step():
    proto_idx = custom_rng.choice(3, p=[0.6, 0.3, 0.1])
    x_t = original_prototypes[proto_idx] + custom_rng.randn(D) * INPUT_NOISE_STD
    x_t /= (np.linalg.norm(x_t) + 1e-12)
    return x_t, proto_idx
grid.world.step = custom_step
```

这样 world 产生自定义频率的 proto，engine 根据 proto_idx 选择 boost——完全对齐。

### 总引擎改动量
- `activation.py`: +3 行 (input_boost param + multiply after A-1)
- `engine.py`: +6 行 (init attributes + cycle() hook + pass to compute_activation)
- 完全向后兼容 (`_input_boost_map=None` → 无 boost → 行为不变)

---

## 实验矩阵

### 模块-Prototype 偏好分配

| Module | Size | Preferred Proto | Proto Freq |
|--------|------|:---:|:---:|
| 0 (largest) | 192 | C | 10% (lowest) |
| 1 | 96 | B | 30% |
| 2 (target) | 64 | A | 60% (highest) |
| 3-7 | 48-16 | none | — |

**设计意图：** 最大模块 0 对应最低频 prototype，最小目标模块 2 对应最高频 prototype。
当 proto=A (60% 时间) 出现时，module 2 的 content match 乘以 (1+delta)。

### Cell 矩阵 (6 cells × 3 seeds = 18 runs)

| # | Label | delta | θ adapt | Purpose |
|---|-------|-------|---------|---------|
| 1 | CTRL | 0.0 | ON | 纯 baseline，有频率差但无 boost |
| 2 | C1_D05 | 0.5 | ON | 温和 boost |
| 3 | C1_D10 | 1.0 | ON | 中等 boost |
| 4 | C1_D20 | 2.0 | ON | 强 boost |
| 5 | C1b_D10 | 1.0 | OFF | θ 关闭对照 |
| 6 | C1b_D20 | 2.0 | OFF | θ 关闭对照 |

**θ 关闭实现：** `THETA_ADAPT_RATE = 0.0` (θ 固定在初始值，不自适应)

### 底盘参数

```python
BASE_PARAMS = {
    'MU_EXCITOTOX': 20.0, 'A_FIRE': 0.3, 'LAMBDA_BCPNN': 0.0,
    'GRID_SIZE': 8,  # N=512
    'B1_ENABLED': False, 'B12_ENABLED': False,
    'B3_ENABLED': True, 'B3_FREEZE_LABELS': True,
    'B3_ETA_ENERGY': 0.0, 'B3_ETA_CAP': 0.0,
    'B4_ENABLED': False,
    'B5_ENABLED': False, 'B5_ETA': 0,
    'PHI_COMPUTE_INTERVAL': 999999,
}
```

**无 extra-kill, 无 concave readout, 无 Poiseuille, 无 DUAL_MU。纯 IID baseline + input boost。**

---

## 诊断指标

### Per-module (每 TRACK_INTERVAL=100 记录)

| Metric | Formula | Purpose |
|--------|---------|---------|
| `fr_m[m]` | spikes_m / (size_m × n_steps) | 每单元平均 firing rate |
| `total_spk_m[m]` | Σspikes_m | 总 spike 数 |
| `density_m[m]` | act_m / (alive_m × n_steps) | 激活密度 |
| `death_rate_m[m]` | deaths_m / (size_m × n_steps) | 死亡率 |
| `alive_frac_m[m]` | alive_m / size_m | 存活比例 |
| `R_spk[m]` | total_spk_m / Σtotal_spk | Raw spike 份额 |

### 聚合指标

| Metric | Formula | Purpose |
|--------|---------|---------|
| `fr_std` | std(fr_m) across 3 target modules | Firing rate 方差 — Tier 1 |
| `fr_ratio_2_vs_0` | fr_m[2] / fr_m[0] | per-unit fr 比 |
| `spk_ratio_2_vs_0` | total_spk[2] / total_spk[0] | 总 spike 比 — Tier 2/3 |
| `dom_raw` | argmax(R_spk) | Raw winner |
| `dom95s_raw` | p95(dom_frac_raw) | Raw dominance |
| `theta_std` | std(θ across all units) | θ 分布扩散 |
| `theta_mean_m[m]` | mean(θ | labels==m) | per-module θ 均值 |

### Per-prototype 诊断

| Metric | Purpose |
|--------|---------|
| `fr_when_proto_A[m]` | proto=A 时 module m 的 fr |
| `fr_lift_m2` | module 2 在 proto=A 时的 fr lift |
| `proto_freq_actual` | 实际 proto 出现频率验证 |

---

## Verdict 判据

### Tier 0 — 安全检查
- CTRL: dom95s_raw ∈ [0.34, 0.42]
- CTRL: winner_raw == 0
- ALL: health ∈ {ALIVE, MARGINAL}

### Tier 1 — Firing Equalization 是否被打破？
```
fr_std = std(fr_m[0:3]):  # modules 0, 1, 2
  < 0.005  → THETA_ABSORBS_ALL
  ≥ 0.005  → FIRING_DIVERGED → Tier 2
```

### Tier 2 — fr 差异方向 + 规模效应
```
if fr_m[2] > fr_m[0]:  # 小模块 per-unit fr 更高
  but R_spk[2] < R_spk[0]:  # 大模块仍赢
  → FR_DIFF_SIZE_WINS
```

### Tier 3 — 真翻转
```
if R_spk[2] > R_spk[0]:
  → DEMAND_BREAKS_CEILING ***
```

### C1b 对照 (θ OFF)
```
if C1b flips but C1 doesn't → THETA_IS_ROOT_INVARIANT
if C1b also doesn't flip → INVARIANT_DEEPER_THAN_THETA
```

---

## 输出路径

```
output/phase_c1_task/c1_l0_results.csv
output/phase_c1_task/c1_results.csv
```

## 运行命令

```bash
python3 experiments/phase_c1_task/run_c1_task_driven.py --smoke
python3 experiments/phase_c1_task/run_c1_task_driven.py --l0-only
python3 experiments/phase_c1_task/run_c1_task_driven.py --seeds 3 --steps 6000
```

预计 ~18 runs, ~5-7 min

---

## 实施步骤

### Step 1: 修改 `railmind_2a/activation.py` (+3 行)
添加 `input_boost` 可选参数，A-1 之后 `m *= input_boost`。

### Step 2: 修改 `railmind_2a/engine.py` (+6 行)
- `__init__`: `self._input_boost_map = None`, `self._input_boost = None`
- `cycle()`: world.step() 之后插入 boost selection hook
- `compute_activation()` 调用传入 `input_boost=self._input_boost`

### Step 3: 验证 backward compatibility
```bash
python3 -c "from railmind_2a import ContentGrid; g = ContentGrid(); g.cycle(); print('OK')"
```

### Step 4: 创建 `experiments/phase_c1_task/run_c1_task_driven.py` (~600 行)

### Step 5: Smoke test → L0 → Full pipeline

---

## ⚠️ 不改的东西

- ✅ 不改 μ (metabolic rate 固定 20)
- ✅ 不改死亡机制
- ✅ 不改 readout (raw spike sum)
- ✅ 不加 extra-kill
- ✅ 不加 Poiseuille
- ✅ 不改 BCPNN (λ=0)
- ✅ 只改 input → content match 项
