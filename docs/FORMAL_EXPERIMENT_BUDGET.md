# EGMAP Formal 实验协议与算力预算

> 口径：同协议 **EGMAP formal** vs **MASPO formal baseline**（冻结 prompt、disjoint eval、residual selector；MASPO 无 experience）。  
> 算力估算基于 2026-06-09 实测日志（单卡 H100 + vLLM，Qwen3.5 系列）。

---

## 1. 当前已完成的测试量

### 1.1 配置矩阵（已完成部分）

| 维度 | 已完成 | 全量目标 |
|------|--------|----------|
| 数据集 | math500、gpqa（2/8） | 8（见 §2） |
| Seed | 123（1/3） | 123, 42, 456 |
| 拓扑 / 模式 | Parallel `llm_agg` na=3（1/2） | Parallel + Sequential `reflect` nr=2 |
| 模型 | Dual：Work 4B + Strong 9B（1/2） | Dual + Single 9B |
| 方法 | EGMAP + MASPO 均已跑 | 每格两方法 |

**完成度**：2 / (8×3×2×2) = **2/96 ≈ 2.1%**（按「数据集×seed×拓扑×模型配置×方法」计）；  
若只数 **EGMAP+MASPO 成对单元格**：2 / 48 = **4.2%**（Dual 模型下）。

### 1.2 单格协议内的题量（Formal）

固定超参（`run_egmap_formal_one_seed.py` / `run_maspo_formal_baseline.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `opt_size` | 100 | 优化池 / Stage1 建 bank |
| `sample_size` | 200 | Stage2 / MASPO 评测上限 |
| `depth` | 3 | Prompt 优化深度 |
| `na` / `nr` | 3 / 2 | Parallel na=3；Sequential nr=2 |
| `bank_size` / `top_k` | 100 / 3 | Experience 库上限与检索数 |
| `disjoint` | 是 | eval 与 opt 无 `unique_id` 重叠（`split_opt_eval_items` 启动时断言） |
| `split manifest` | 是 | `splits/{tag}_split.json` 锁定 EGMAP/MASPO 同一批 eval id |
| Stage2 `write_experience` | **False** | 评测阶段不向 bank 写入 |

**每个 (dataset, seed) 的实际评测题数**：

```
eval_pool = |D| - 100
eval_run  = min(200, eval_pool)
```

| 数据集 | \|D\| | eval_pool | **实际评测** | Stage1 bank |
|--------|------|-----------|--------------|-------------|
| math500 | 500 | 400 | **200** | 100 |
| agieval | 1000 | 900 | **200** | 100 |
| aqua | 254 | 154 | **154** | 100 |
| gpqa | 198 | 98 | **98** | 100 |
| humaneval | 164 | 64 | **64** | 100 |
| vqarad | 451 | 351 | **200** | 100 |
| slake | 1061 | 961 | **200** | 100 |
| chartqa | 2500 | 2400 | **200** | 100 |

### 1.3 当前已跑的有效 LLM 调用规模（粗算）

以 **Parallel + Dual 4B/9B** 为例，单题约 **8–15 次** work 解码 + **1–3 次** strong 调用（优化器 / residual selector / CODE judge 另计）。

**已完成（math500 + gpqa，seed=123）**：

| 阶段 | math500 | gpqa | 合计 |
|------|---------|------|------|
| EGMAP optimize（prompt+handoff） | 0（skip） | 1× | — |
| EGMAP Stage1（100 题建 bank） | 100 | 100 | 200 题 |
| EGMAP Stage2 eval | 200 | 98 | **298 题** |
| MASPO eval（同 prompt，无 experience） | 200 | 98 | **298 题** |
| **评测总题次** | | | **596 题次** |

结果文件：

- `result/egmap_formal_*_seed123_b100k3.json`
- `result/maspo_formal_*_seed123.json`
- 对照表：`result/comparison_table.md`

### 1.4 当前实测墙钟时间（seed=123，Parallel，Dual 4B+9B）

| 任务 | 墙钟 | 日志 |
|------|------|------|
| EGMAP math500（skip optimize） | Stage1 23m + Stage2 46m ≈ **69 min** | `logs/egmap_formal_math500_gpqa_seed123.log` |
| EGMAP gpqa（含 optimize） | **150 min**（03:01→05:32 UTC） | 同上 |
| MASPO math500 eval 200 | **39 min** | `logs/maspo_formal_math500_gpqa_seed123.log` |
| MASPO gpqa eval 98 | **51 min** | 同上 |
| **本批合计墙钟** | ≈ **5.8 h**（单拓扑、单 seed、2 数据集） | |

---

## 1.5 防数据泄露（已实现）

| 阶段 | 数据 | 是否见 gold / 是否进 bank |
|------|------|---------------------------|
| Prompt optimize | 仅 `opt` 池 100 题（问题文本） | 无标签；强模型 pairwise，不用标答 |
| Stage1 建 bank | 仅 `opt` 100 题 | `write_experience=True`；bank 清空后重建 |
| Stage2 EGMAP eval | `eval_run` held-out | `write_experience=False`；只读冻结 bank |
| MASPO eval | 与 Stage2 **同一** `eval_unique_ids_run` | 无 experience；读 manifest |

**代码保障**：

- 单一划分：`data_loaders.split_opt_eval_items()`（`load_opt_and_eval` / formal 脚本共用）
- 运行前断言：`verify_disjoint(opt, eval)`
- Stage1 后：`verify_bank_from_opt_only(bank, opt_ids)`
- 清单：`splits/egmap_formal_*_split.json`；MASPO 优先按 manifest 取题，避免二次 `random.sample` 漂移
- 检索注入：默认只注入 `error_type` + `advice`（无 gold）；`EGMAP_SHOW_MEMORY_EXAMPLES=1` 才会带 `prior_problem`

**审计命令**：

```bash
python scripts/verify_formal_no_leakage.py --dataset math500 --seed 123 --opt-size 100 --sample-size 200
python scripts/verify_formal_no_leakage.py \
  --manifest splits/egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3_split.json \
  --egmap-result result/egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3.json \
  --maspo-result result/maspo_formal_math500_llm_agg_na3_d3s200o100seed123.json
```

---

## 2. 全量泛化范围

与 `scripts/run_exhandoff_text_na3.sh` / `run_egmap_qwen9b_full_benchmark.sh` 对齐：

**文本（5）**：math500, agieval, aqua, gpqa, humaneval  
**VQA（3）**：vqarad, slake, chartqa  

（`textvqa` / `pmcvqa` 在 `config.py` 中但不在当前 8 数据集主矩阵内。）

### 2.1 实验因子

| 因子 | 水平 | 代码 |
|------|------|------|
| **Seed** | 3 | 123, 42, 456 |
| **模式 / 拓扑** | 2 | Parallel `llm_agg --na 3`；Sequential `reflect --nr 2` |
| **模型** | 2 | 见下表 |

**两种模型配置**：

| 配置 | Work（Agent） | Strong（优化/选择/Judge） | vLLM 部署 | 参考脚本 |
|------|---------------|---------------------------|-----------|----------|
| **Dual（默认 formal）** | Qwen3.5-4B @8005 | Qwen3.5-9B @8004 | **2×H100** 常开 | `scripts/env_unified.sh` |
| **Single 9B** | Qwen3.5-9B @8001 | 同左 | **1×H100** | `scripts/run_egmap_qwen9b_full_benchmark.sh` |

### 2.2 每个「单元格」要跑什么

对每个 **(dataset, seed, topology, model_config)**：

1. **EGMAP**：optimize → Stage1（100 题 `write_experience=True`）→ Stage2（`eval_run` 题，`write_experience=False`）
2. **MASPO**：仅 Stage2 同题量（加载 EGMAP 的 prompt/handoff，`residual_selector=True`，无 experience）

单元格数：**8 数据集 × 3 seed × 2 拓扑 = 48**（每种模型配置各 48 格；每格含 EGMAP+MASPO 两条流水线）。

---

## 3. H100 卡时估算

### 3.1 估算假设

1. **卡时定义**：1 张 H100 满载运行 1 小时 = **1 GPU·h**。
2. **Dual 配置**：4B 与 9B 各占 1 卡，实验期间双卡同开 → 墙钟 1h ≈ **2 GPU·h**。
3. **Single 9B**：单卡 → 墙钟 1h = **1 GPU·h**；同题量墙钟约为 Dual 的 **1.25×**（math500 全流水线：88 min vs ~70 min 量级）。
4. **Sequential** 较 Parallel 墙钟 ×**1.15**（reflect 更深，实测 nr2 与 na3 同集接近或略慢）。
5. 每格耗时 = **EGMAP（含 optimize）+ MASPO eval**；不同 seed 的 opt 划分不同，**不可共用 optimize**。
6. VQA 集 `max_concurrent=2`（文本为 4），耗时按实测上浮。

### 3.2 单格墙钟先验（Parallel，Dual 4B+9B）

| 数据集 | EGMAP (h) | MASPO (h) | 合计 (h) | 依据 |
|--------|-----------|-----------|----------|------|
| math500 | 1.45 | 0.65 | **2.10** | optimize~45m + eval~69m；MASPO 39m |
| agieval | 1.45 | 0.65 | **2.10** | 同 math500 题量结构 |
| aqua | 1.10 | 0.50 | **1.60** | eval 154 题 |
| gpqa | 2.50 | 0.85 | **3.35** | 实测 150m + 51m |
| humaneval | 0.70 | 0.30 | **1.00** | eval 64 题 + CODE judge |
| vqarad / slake / chartqa | 2.00 | 0.75 | **2.75** | VQA 并发 2，约为文本 1.3× |

> optimize 耗时与数据集强相关；上表为 **含 optimize** 的 EGMAP 总时长。math500 若已有 prompt 可 `--skip-optimize`，省 ~45 min，但 **换 seed 必须重跑 optimize**。

### 3.3 全矩阵汇总

**48 单元格 / 每种模型配置**（8 ds × 3 seed × 2 topo；每格 EGMAP+MASPO）

| 模型配置 | 墙钟（约） | **GPU·h（约）** | 说明 |
|----------|------------|-----------------|------|
| **Dual 4B + 9B**（2 卡） | **~166 h**（≈6.9 天） | **~330** | 48 格 × 平均 ~2.1h ×2 卡；含 Sequential ×1.15 |
| **Single 9B**（1 卡） | **~207 h**（≈8.6 天） | **~207** | 墙钟 ×1.25，单卡计费 |
| **两种模型都跑** | — | **~540** | 330 + 207 |

**仅 Dual、仅 Parallel、仅 3 seed 的文本 5 集**（不做 VQA、不做 Sequential、不做 Single 9B）：

- 单元格：5 × 3 = **15**
- GPU·h ≈ **15 × 2.1h × 2 ≈ 63 GPU·h**（墙钟 ~32 h）

### 3.4 与当前进度的差距

| 项目 | 当前 | 全量 Dual | 剩余 |
|------|------|-----------|------|
| 单元格（EGMAP+MASPO 对） | 2 | 48 | **46** |
| GPU·h（Dual 估算） | ~12 | ~330 | **~318** |
| 墙钟（2×H100 满负载） | ~5.8 h | ~166 h | **~160 h** |

---

## 4. 推荐执行顺序与省钱策略

1. **先文本 5 集 × Parallel × 3 seed × Dual**（主表）→ ~63 GPU·h。  
2. **补 Sequential nr2**（同数据同 seed）→ 再 ×1.15。  
3. **VQA 3 集**放后（单格 ~2.75h，最耗时）。  
4. **Single 9B** 若只为消融：可只对 math500+gpqa 抽样 1 seed → ~15 GPU·h，再决定是否全跑。  
5. **任务级续跑**：结果 JSON 存在则跳过（见 `run_egmap_qwen9b_full_benchmark.sh`）；无单任务内 checkpoint，中途 kill 需整格重跑。  
6. **MASPO 不重复 optimize**：在 EGMAP 同格 prompt 产出后立刻跑 MASPO，避免空等。

### 4.1 启动命令模板

```bash
# Dual formal（单数据集单 seed）
cd /mnt/afs/L202500372/Experience-Guided-Multi-Agent-Prompting
source scripts/env_unified.sh

python run_egmap_formal_one_seed.py \
  --dataset math500 --graph llm_agg --na 3 --seed 123 \
  --opt-size 100 --sample-size 200 --depth 3

python run_maspo_formal_baseline.py \
  --dataset math500 --graph llm_agg --na 3 --seed 123 \
  --opt-size 100 --sample-size 200 --depth 3

python scripts/export_egmap_maspo_table.py --auto --output result/comparison_table.md
```

```bash
# 全矩阵 Dual（需先起 4B@8005 + 9B@8004）
bash /mnt/afs/L202500372/scripts/run_egmap_qwen9b_full_benchmark.sh  # 仅 Single 9B 版；Dual 需改端口/模型 env
```

---

## 5. 产出物清单

| 产物 | 路径模式 |
|------|----------|
| EGMAP 结果 | `result/egmap_formal_{ds}_{graph}_na3_d3s200o100seed{S}_b100k3.json` |
| MASPO 结果 | `result/maspo_formal_{ds}_{graph}_na3_d3s200o100seed{S}.json` |
| Prompt / Handoff | `prompt/egmap_formal_*_prompts.json`, `*_handoffs.json` |
| Experience bank | `memory/egmap_formal_*_bank.jsonl` |
| 论文对照表 | `result/comparison_table.md`（`export_egmap_maspo_table.py`） |

---

## 6. 不确定性说明

| 因素 | 影响 |
|------|------|
| gpqa / humaneval 题少 | eval 题数 <200，总卡时低于 math500 |
| 长输出题（CoT、化学） | 单题 30–90s，方差大（见 MASPO gpqa 日志） |
| vLLM 批大小 / `max_concurrent` | 文本 4、VQA 2；调高可能省墙钟但 OOM 风险 |
| CODE `CodeJudgeAgent` | humaneval 额外 CPU/子进程，GPU 利用偏低 |
| 双卡 vs 单卡串行加载 | 若只有 1×H100 跑 Dual，墙钟↑但 GPU·h↓ |

**结论（主答案）**：把当前 formal 操作泛化到 **8 数据集 × 3 seed × 2 拓扑 ×（EGMAP+MASPO）**：

- **仅 Dual 4B+9B（2×H100）**：约 **330 GPU·h**（墙钟 ~7 天）。  
- **仅 Single 9B（1×H100）**：约 **207 GPU·h**（墙钟 ~8.5 天）。  
- **两种模型配置全跑**：约 **540 GPU·h**。  

当前仅完成 **~12 GPU·h（~2%）**，有效评测 **596 题次**（298×2 方法）。

---

*文档生成：2026-06-09；实测日志 `logs/egmap_formal_math500_gpqa_seed123.log`、`logs/maspo_formal_math500_gpqa_seed123.log`。*
