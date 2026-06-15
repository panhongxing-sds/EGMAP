# EGMAP 正式实验计划

> 最后更新：2026-06-11  
> 仓库基线：[OFFICIAL_BASE.md](OFFICIAL_BASE.md)（官方 MASPO @ `e79aa8e`）

---

## 1. 核心原则：两条独立协议

| 维度 | **官方 MASPO（Baseline）** | **EGMAP / ExHandoff（Ours）** |
|------|---------------------------|------------------------------|
| 代码入口 | `run_maspo_formal_one_seed.py` 或 `run_maspo.py`（**无** `--experience-guided`） | `run_egmap_formal_one_seed.py` 或 `run_maspo.py --experience-guided` |
| Prompt 优化 | 仅 **节点 prompt**；官方 fixed-rounds + beam-refresh + misleading-sampling + lookahead-score | 节点 prompt + **handoff 优化** + structured meta-prompt |
| 推理执行 | **单路** `MAS.arun()`，默认拓扑 | handoff + disagreement verification + **residual selector** + experience retrieval |
| Handoff map | **不使用** | `prompt/egmap_formal_*_handoffs.json` |
| Experience bank | **不使用** | `memory/egmap_formal_*_bank.jsonl`（仅 opt 错题构建） |
| 结果 tag | `maspo_formal_{ds}_...` | `egmap_formal_{ds}_..._b100k3` |

### ⚠️ 已废弃的错误 baseline

此前 `run_maspo_formal_baseline.py` 错误地开启了：

- `use_handoff=True`
- `use_disagreement_handoff=True`
- `use_residual_selector=True`
- 读取 `egmap_formal_*_handoffs.json`

这 **不是** 官方 MASPO，而是「EGMAP 去掉 experience」的变体。  
`result/maspo_formal_*.json`（旧版）**全部作废**，不得写入论文或 `comparison_table.md`。

公平对照的唯一合法定义：

- **同数据集、同 seed、同 opt/eval 划分**（共用 `splits/egmap_formal_*_split.json`）
- **MASPO**：官方优化出的 `maspo_formal_*_prompts.json` + 单路推理
- **EGMAP**：EGMAP 优化出的 prompts/handoffs + bank + 全套 ExHandoff 推理栈

---

## 2. 环境与数据

### 2.1 两套模型配置（分开跑，不混用 tag）

| Profile | 名称 | Work | Strong | 端口 | 后缀 | 何时跑 |
|---------|------|------|--------|------|------|--------|
| **`m4b`** | Single 4B（**先跑，1×GPU**） | 4B | 4B | :8005 | `_m4b` | **Campaign 1** |
| **`m4b9b`** | Dual 4B+9B（论文主设置，需 2×GPU 或足够显存） | 4B | 9B | :8005/:8004 | `_m4b9b` | 有双卡时 |
| **`m9b`** | Single 9B | 9B | 9B | :8001 | `_m9b` | **Campaign 2（后跑）** |

```bash
# Campaign 1 — 小模型 4B（单卡默认）
bash /mnt/afs/L202500372/bootstrap/serve-qwen35.sh 4b --port 8005
MODEL_PROFILE=single_4b bash scripts/run_maspo_official_phase1.sh

# Campaign 2 — 大模型 9B（单独一轮，不覆盖 Campaign 1 产物）
bash /mnt/afs/L202500372/bootstrap/serve-qwen35.sh 9b --port 8001
MODEL_PROFILE=single_9b bash scripts/run_maspo_official_phase1.sh

# 可选：双卡论文配置 Dual
bash scripts/start_vllm_dual_4b9b.sh
MODEL_PROFILE=dual_4b_9b bash scripts/run_maspo_official_phase1.sh
```

- 产物示例：`result/maspo_formal_math500_..._seed123_m4b9b.json` vs `..._m9b.json`
- **prompt / bank / eval 按 profile 独立**；`splits/egmap_formal_*_split.json`（无后缀）各数据集共享
- 每格跑完 → `scripts/update_result_ledger.py`（台账含 **Model** 列）

```bash
export HANDOFF_DATASET_ROOT=/mnt/afs/L202500372/data/egmap_handoff
```
| 图拓扑 | `llm_agg`（并行聚合，主表） |
| Agent 数 | `na=3` |
| Reflect 变体 | `nr=2`（单独一行，未跑完前不进主表） |
| 随机种子 | `123`, `42`, `456` |
| Opt 池 | `opt_size=100`（与 eval 不交） |
| Eval 子集 | `sample_size=200`（从 eval pool 按 seed 抽样） |
| 优化深度 | `depth=3`, `rounds_per_turn=3` |
| Bank | `bank_size=100`, `top_k=3`（仅 EGMAP） |

### 防截断（math / 文本正式格必开）

```bash
source scripts/formal_apply_tok8192_env.sh   # MASPO_WORK_MAX_TOKENS=8192, MASPO_WORK_MAX_PROMPT_CHARS=0
```

---

## 3. 数据集矩阵

### 3.1 主表（Parallel `llm_agg`, `na=3`, `nr=1`）

| 域 | 数据集 | EGMAP×3 seeds | 官方 MASPO×3 seeds | 备注 |
|----|--------|:-------------:|:------------------:|------|
| Math | math500 | ❌ **需重跑**（3 seeds） | ❌ **需重跑**（3 seeds） | s123 bank=0；MASPO 为伪 baseline；建议 tok8192 全重跑 |
| Math | aqua | ✅ | ❌ **需重跑** | |
| Reasoning | gpqa | ✅（截断严重，建议重跑 EGMAP） | ❌ **需重跑** | tok8192 优先 |
| Reasoning | agieval | ✅（建议重跑） | ❌ **需重跑** | |
| Code | humaneval | seed123 ✅ | ❌ **需重跑** | seed42/456 缺 |
| VQA | vqarad / slake / chartqa | stage1 only | ❌ 全缺 | 无正式 eval json |

### 3.2 变体（不进主表直至 parallel 收敛）

- `graph=reflect`, `nr=2`
- 其他拓扑（`chain`, `debate` 等）

---

## 4. 运行命令

### 4.1 官方 MASPO（单 seed，优化 + eval）

```bash
cd /mnt/afs/L202500372/Experience-Guided-Multi-Agent-Prompting
source scripts/formal_apply_tok8192_env.sh

python run_maspo_formal_one_seed.py \
  --dataset math500 --graph llm_agg --na 3 --nr 1 \
  --seed 123 --opt-size 100 --sample-size 200 --depth 3
```

- 产出：`prompt/maspo_formal_*_prompts.json`、`result/maspo_formal_*.json`
- **不会**写 handoffs、bank、residual 字段

仅重跑 eval（已有 MASPO prompts）：

```bash
python run_maspo_formal_one_seed.py --dataset math500 --seed 123 --skip-optimize
# 或
python run_maspo_formal_baseline.py --dataset math500 --seed 123
```

### 4.2 EGMAP（单 seed，三阶段）

```bash
python run_egmap_formal_one_seed.py \
  --dataset math500 --graph llm_agg --na 3 --nr 1 \
  --seed 123 --opt-size 100 --sample-size 200 --depth 3 \
  --bank-size 100 --top-k 3
```

阶段：

1. **Optimize**：structured meta + handoff optimize → `prompt/egmap_formal_*_{prompts,handoffs}.json`
2. **Stage1 bank build**：仅在 opt 100 题上跑，写错题 → `memory/egmap_formal_*_bank.jsonl`
3. **Frozen eval**：固定 bank，在 eval 200 题上推理 → `result/egmap_formal_*.json`

跳过优化（仅 stage1+2）：

```bash
python run_egmap_formal_one_seed.py --dataset math500 --seed 123 --skip-optimize
```

### 4.3 等价 CLI（`run_maspo.py`）

**官方 MASPO**（无 ExHandoff）：

```bash
python run_maspo.py --dataset math500 --graph llm_agg --na 3 \
  --disjoint-eval --opt-size 100 --sample-size 200 --seed 123 \
  --optimize --fixed-rounds --beam-refresh --misleading-sampling --lookahead-score \
  --depth 3
# 不要加：--experience-guided --handoff --disagreement-handoff --residual-selector
```

**EGMAP 全套**：

```bash
python run_maspo.py --dataset math500 --graph llm_agg --na 3 \
  --disjoint-eval --opt-size 100 --sample-size 200 --seed 123 \
  --optimize --fixed-rounds --beam-refresh --misleading-sampling --lookahead-score \
  --experience-guided --handoff --handoff-optimize --structured-meta-prompt \
  --disagreement-handoff --residual-selector \
  --experience-bank memory/... --experience-top-k 3 --write-experience
```

### 4.4 批量重跑脚本

| 脚本 | 用途 |
|------|------|
| `scripts/rerun_textmath_tok8192.sh` | math500 / aqua / gpqa / agieval EGMAP+MASPO |
| `scripts/rerun_humaneval_tok8192.sh` | humaneval |
| `scripts/run_smoke_bank_gpu.sh` | bank 构建 smoke（独占 GPU） |
| `scripts/prune_unscoreable_formal.py` | 剔除不可评分题，重算 accuracy |

重跑官方 MASPO 全矩阵（示例）：

```bash
for ds in math500 aqua gpqa agieval humaneval; do
  for seed in 123 42 456; do
    python run_maspo_formal_one_seed.py --dataset "$ds" --seed "$seed" \
      2>&1 | tee "logs/maspo_official_${ds}_seed${seed}.log"
  done
done
```

---

## 5. 产物与命名

| 类型 | 路径模式 |
|------|----------|
| EGMAP split | `splits/egmap_formal_{ds}_llm_agg_na3_d3s200o100seed{S}_b100k3_split.json` |
| MASPO prompts | `prompt/maspo_formal_{ds}_llm_agg_na3_d3s200o100seed{S}_prompts.json` |
| EGMAP prompts / handoffs | `prompt/egmap_formal_*_{prompts,handoffs}.json` |
| EGMAP bank | `memory/egmap_formal_*_bank.jsonl` |
| MASPO eval（有效） | `result/maspo_formal_{ds}_...seed{S}.json`，`split_info.handoff=false` |
| EGMAP eval | `result/egmap_formal_{ds}_...seed{S}_b100k3.json` |

导出对照表（**仅在新 MASPO 跑完后**）：

```bash
python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
```

---

## 6. 质量门禁（跑前 / 跑后）

### 跑前

- [ ] vLLM 就绪：`curl -s http://127.0.0.1:8001/v1/models`
- [ ] 无其他任务抢占 `:8001`（humaneval 与 bank smoke 互斥）
- [ ] 文本/math 格已 `formal_apply_tok8192_env`
- [ ] MASPO 脚本 **未** 读取 `egmap_formal_*_handoffs.json`

### 跑后

- [ ] MASPO json 中 `handoff=false`, `residual_selector=false`, `disagreement_handoff=false`
- [ ] EGMAP bank 仅含 opt 错题；`verify_bank_from_opt_only` 通过
- [ ] `prune_unscoreable_formal.py` 已执行（formal 结果）
- [ ] 三 seed 齐全再报均值 ± std

---

## 7. 分阶段执行计划（当前策略）

**原则：先单 seed 打通全数据集，MASPO baseline 跑完即锁定数值；再跑 EGMAP 对照；最后扩 3 seeds。**

### Phase 1 — 官方 MASPO × seed=123 × 全主表（**Campaign A: m4b9b 先跑**）

| 顺序 | 数据集 | Profile | 动作 |
|:----:|--------|---------|------|
| 1–8 | 全主表 | **m4b9b** | `MODEL_PROFILE=dual_4b_9b bash scripts/run_maspo_official_phase1.sh` |
| 1–8 | 全主表 | m9b（后） | `MODEL_PROFILE=single_9b bash scripts/run_maspo_official_phase1.sh` |

> 当前后台若在用 single 9B @:8001 跑 math500，应停掉后改用 **dual m4b9b** 重跑（产物 tag 不同，不冲突）。

```bash
bash scripts/run_maspo_official_phase1.sh          # 8 数据集串行
DATASETS=math500 bash scripts/run_maspo_official_phase1.sh   # 单格调试
FORCE=1 bash scripts/run_maspo_official_phase1.sh  # 覆盖旧伪-MASPO json
```

跑完每格 → `scripts/update_result_ledger.py` 写入 [RESULT.md](RESULT.md) **运行台账**。  
**MASPO 锁定条件**：`protocol_ok=yes`（无 handoff/residual）；**锁定后不改 protocol**。

旧无效 json 自动归档到 `result/_invalid_pseudo_maspo/`。

### Phase 1.5 — EGMAP 预检门禁（Phase 2 前必过）

每数据集在正式 EGMAP 跑之前执行：

```bash
# 静态：handoff 边覆盖、split/bank 隔离、bank schema
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123

# GPU：bank 构建 smoke（tok8192，fast 约 3min；full 含 residual 更慢）
source scripts/formal_common.sh && formal_apply_env ... && formal_apply_tok8192_env
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123 --smoke --fast
```

| 检查项 | 说明 |
|--------|------|
| handoff 边覆盖 | 每条 MAS 边有 sender/receiver 契约 |
| bank 隔离 | `unique_id` 仅来自 opt 100 |
| bank schema | 仅错题；`correct=True` 行视为 FAIL |
| smoke | opt 子集可写 bank、无大面积 timeout |
| post-eval | result 含 `residual` + `experience` + `raw_trace` |

当前静态预检（seed123）：math500/aqua/gpqa/humaneval **PASS**；**agieval bank 第 15 行含 correct=True，Phase 2 需 FORCE 重建 bank**。

### Phase 2 — EGMAP × seed=123 × 全主表（MASPO 锁定 + 预检通过后）

```bash
bash scripts/run_egmap_official_phase2a.sh
DATASETS=math500 bash scripts/run_egmap_official_phase2a.sh
```

每格完成后：`preflight --check-eval` + `update_result_ledger.py` 更新台账。

### Phase 3 — 扩 seeds 42 / 456

先 MASPO 全矩阵（24 格），再 EGMAP 全矩阵；三 seed mean±std 进论文主表。

### Phase 4 — 扩展

- reflect `nr=2`
- ablation（去 bank / residual / disagreement）

### 旧优先级备忘

| 项 | 说明 |
|----|------|
| 全部伪 `maspo_formal_*` | Phase 1 覆盖重跑 |
| math500 EGMAP s123 bank | Phase 2 重建 |
| gpqa/agieval EGMAP tok8192 | Phase 2 一并重跑 |

---

## 8. 论文表结构（目标）

主表列：**Dataset | MASPO (official) | EGMAP (Ours) | Δ**  
行：上表 5+3 个数据集 × 3-seed mean±std（或 median）。

副表：

- Ablation：去 bank / 去 residual / 去 disagreement
- Bank 规模与 top-k 敏感性
- Opt 池大小（100 vs 50）

详细数字与审计见 [RESULT.md](RESULT.md)；方法定义见 [METHOD.md](METHOD.md)。
