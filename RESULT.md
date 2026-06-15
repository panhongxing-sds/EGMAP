# 实验结果与审计状态

> 最后更新：2026-06-11  
> 协议定义见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)、[METHOD.md](METHOD.md)

---

## 1. 关键结论：旧 MASPO 列全部无效

`result/maspo_formal_*.json`（2026-06-11 之前）在 `split_info` 中记录为：

- `handoff_source`: `prompt/egmap_formal_*_handoffs.json`
- `residual_selector`: `true`
- `disagreement_handoff`: `true`

这是 **EGMAP 推理栈去掉 experience bank**，**不是** 官方 MASPO。  
因此：

| 产物 | 状态 |
|------|------|
| `result/comparison_table.md`（当前版） | **作废** — 基于错误 MASPO |
| `result/maspo_formal_*.json`（旧） | **作废** — 需按 `run_maspo_formal_one_seed.py` 重跑 |
| `prompt/maspo_formal_*_prompts.json` | **尚未生成** — MASPO 需独立优化 |
| `result/egmap_formal_*.json` | **部分有效** — 见下表，仍有截断/bank 问题 |

**在官方 MASPO 重跑完成前，不得报告 EGMAP vs MASPO 的 Δ。**

---

## 2. EGMAP 结果状态（Parallel `llm_agg`, `na=3`, seed 123 为主）

> 数字来自 prune 后 `graph_types.llm_agg.accuracy`；**MASPO 列留空待重跑**。

### 2.1 文本 / Math（tok8192 前后混用 — 以重跑后为准）

| Dataset | EGMAP s123 | EGMAP s42 | EGMAP s456 | 官方 MASPO | 备注 |
|---------|:----------:|:---------:|:----------:|:----------:|------|
| math500 | ~~89.7%~~ **待重跑** | ~~93.9%~~ **待重跑** | ~~93.8%~~ **待重跑** | **全格待重跑** | s123 bank=0；旧 MASPO 无效；tok8192 |
| aqua | 90.8% | — | — | **待重跑** | 与旧伪-MASPO 差距 <1% |
| gpqa | 86.5% | — | — | **待重跑** | 截断严重，EGMAP 建议 tok8192 重跑 |
| agieval | 89.3% | — | — | **待重跑** | 截断中等 |
| humaneval | 91.8% | 缺 | 缺 | **待重跑** | 仅 s123；评分逻辑已修 |

旧 **无效** 伪-MASPO（seed123，勿引用）：

| Dataset | 旧 maspo_formal（无效） |
|---------|:----------------------:|
| math500 | 88.7% |
| aqua | 91.4% |
| gpqa | 82.4% |
| agieval | 89.3% |
| humaneval | 86.9% |

### 2.2 VQA

| Dataset | EGMAP eval | MASPO | 备注 |
|---------|:----------:|:-----:|------|
| vqarad | 无 | 无 | 仅有 stage1 memory build |
| slake | 无 | 无 | 同上 |
| chartqa | 无 | 无 | 同上 |

### 2.3 Reflect 变体（`nr=2`）

未纳入主表；无完整 formal 三联 seed 结果。

---

## 3. 已知问题（影响 EGMAP 数字可信度）

### 3.1 Prompt / 输出截断

| 数据集 | 严重程度 | 建议 |
|--------|:--------:|------|
| gpqa | 高 | tok8192 下重跑 EGMAP + MASPO |
| agieval | 中 | 建议重跑 |
| aqua | 低 | 可暂保留，重跑更稳 |
| math500 | 低–中 | **建议三 seed 全重跑**（s123 bank 空；统一 tok8192） |

环境：`scripts/formal_apply_tok8192_env.sh` → `MASPO_WORK_MAX_TOKENS=8192`, `MASPO_WORK_MAX_PROMPT_CHARS=0`。

### 3.2 Experience bank（math500 seed123）

| 指标 | 值 |
|------|-----|
| 旧 stage1 timeout 比例 | 67/100 |
| prune 后 bank 条数 | **0** |
| smoke（tok8192, 2026-06-15） | 3 题 PASS；旧 timeout 题可解 |

结论：math500 seed123 正式 EGMAP 需 **独占 GPU 重建 stage1 bank**，再 frozen eval。

### 3.3 不可评分题

已提供 `scripts/prune_unscoreable_formal.py`：从 formal json 剔除无法自动评分的题目并重算 accuracy。  
prune 后 seed123 文本宏平均（仅 EGMAP，旧伪-MASPO 勿用）：

- EGMAP：**89.6%**
- 旧伪-MASPO：87.7%（**无效 baseline**）

### 3.4 EGMAP vs 旧伪-MASPO 差距成因（math500 / aqua，仅供参考）

在错误 baseline 下差距很小（每 seed 1–2 题），主因包括：

1. compress / 截断导致双路径同错  
2. residual selector 过保守（`KEEP_BASE` 在 challenger 已对时仍保留 base）  
3. seed123 experience 几乎为空，略损双路径收益  
4. aqua 无 head-to-head 独赢题  

**官方 MASPO 重跑后需重新做 head-to-head 分析。**

---

## 4. 产物清单

### 4.1 有效 / 待刷新

```
result/egmap_formal_{math500,aqua,gpqa,agieval,humaneval}_llm_agg_na3_d3s200o100seed{123,42,456}_b100k3.json
memory/egmap_formal_*_bank.jsonl
splits/egmap_formal_*_split.json
prompt/egmap_formal_*_{prompts,handoffs}.json
```

### 4.2 作废（勿删，作审计对照）

```
result/maspo_formal_*   # 含 handoff/residual 字段
result/comparison_table.md
```

### 4.3 待生成（官方 MASPO）

```
prompt/maspo_formal_{ds}_llm_agg_na3_d3s200o100seed{S}_prompts.json
result/maspo_formal_{ds}_llm_agg_na3_d3s200o100seed{S}.json   # split_info.handoff=false
```

---

## 5. 论文主表模板（重跑后填写）

```
| Benchmark | MASPO (official) | EGMAP | Δ |
|-----------|------------------|-------|---|
| math500   | __._ ± _._       | __._  |    |
| aqua      |                  |       |    |
| gpqa      |                  |       |    |
| agieval   |                  |       |    |
| humaneval |                  |       |    |
| Macro avg |                  |       |    |
```

填写规则：

- 每格 3-seed mean ± std（或 median [IQR]）
- 仅使用 prune 后、tok8192 正式协议下的 json
- 导出：`python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md`（脚本需识别新 MASPO 协议字段）

---

<!-- RESULT_LEDGER_START -->
### 运行台账（graph=`llm_agg`，seed=123）

> 每完成一格 formal run 自动更新。MASPO 需 `protocol_ok=yes` 才可锁定 baseline。

| Dataset | Method | Model | Acc | Bank | Protocol | Updated | File |
|---------|--------|-------|----:|-----:|:--------:|---------|------|
| math500 | MASPO | m4b | 85.6% | — | yes | 2026-06-15 06:29 UTC | `maspo_formal_math500_llm_agg_na3_d3s200o100seed123_m4b.json` |
| math500 | EGMAP | legacy | 89.7% | 0 | yes | 2026-06-15 03:06 UTC | `egmap_formal_math500_llm_agg_na3_d3s200o100seed123_b100k3.json` |
| aqua | MASPO | legacy | 91.4% | — | **no** invalid pseudo-MASPO | 2026-06-15 03:06 UTC | `maspo_formal_aqua_llm_agg_na3_d3s200o100seed123.json` |
| aqua | MASPO | m4b | 90.1% | — | yes | 2026-06-15 07:30 UTC | `maspo_formal_aqua_llm_agg_na3_d3s200o100seed123_m4b.json` |
| aqua | EGMAP | legacy | 90.8% | 8 | yes | 2026-06-15 03:06 UTC | `egmap_formal_aqua_llm_agg_na3_d3s200o100seed123_b100k3.json` |
| gpqa | MASPO | legacy | 82.4% | — | **no** invalid pseudo-MASPO | 2026-06-15 03:06 UTC | `maspo_formal_gpqa_llm_agg_na3_d3s200o100seed123.json` |
| gpqa | EGMAP | legacy | 86.5% | 7 | yes | 2026-06-15 03:06 UTC | `egmap_formal_gpqa_llm_agg_na3_d3s200o100seed123_b100k3.json` |
| agieval | MASPO | legacy | 89.3% | — | **no** invalid pseudo-MASPO | 2026-06-15 03:06 UTC | `maspo_formal_agieval_llm_agg_na3_d3s200o100seed123.json` |
| agieval | EGMAP | legacy | 89.3% | 24 | yes | 2026-06-15 03:06 UTC | `egmap_formal_agieval_llm_agg_na3_d3s200o100seed123_b100k3.json` |
| humaneval | MASPO | legacy | 86.9% | — | **no** invalid pseudo-MASPO | 2026-06-15 03:06 UTC | `maspo_formal_humaneval_llm_agg_na3_d3s200o100seed123.json` |
| humaneval | EGMAP | legacy | 91.8% | — | yes | 2026-06-15 03:06 UTC | `egmap_formal_humaneval_llm_agg_na3_d3s200o100seed123_b100k3.json` |

<!-- RESULT_LEDGER_END -->

<!-- MASPO_BASELINE_LEDGER_START -->
### Phase 1 锁定 baseline（已合并至运行台账，保留占位）

<!-- MASPO_BASELINE_LEDGER_END -->

## 6. 下一步（执行顺序）

1. **Phase 1（当前）** — `bash scripts/run_maspo_official_phase1.sh`：8 数据集 × seed123 官方 MASPO，逐格锁定 baseline  
2. **Phase 2** — 同 8 数据集 × seed123 EGMAP 全重跑（tok8192 + bank）  
3. **Phase 3** — seeds 42/456 扩满 MASPO → EGMAP  
4. prune + 更新 `comparison_table.md`

---

## 7. Smoke / 工具验证记录

| 日期 | 项 | 结果 |
|------|-----|------|
| 2026-06-15 | `scripts/smoke_bank_gpu.sh` fast 3 题 | PASS (~3 min) |
| 2026-06-15 | 旧 timeout 题 525/297 @ tok8192 | 正确 |
| 2026-06-15 | vLLM 僵尸请求清理 + 重启 | 已恢复 |

Smoke 输出示例：`memory/egmap_formal_math500_*_smoke_bank.jsonl`（非正式表数字来源）。
