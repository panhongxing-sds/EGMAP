# EGMAP 正式实验计划与运行手册

> 最后更新：2026-06-17  
> 面向：**新服务器从零复现实验** · 协议见 [METHOD.md](METHOD.md) · 数字台账 [RESULT.md](RESULT.md) · 操作细节 [RUN.md](RUN.md)

---

## 0. 读本文能做什么

| 你想… | 跳转到 |
|--------|--------|
| 在新机器上从零跑通 8 数据集 | **§5 标准流水线** |
| 理解 MASPO / EGMAP 区别与顺序 | **§1、§6** |
| 保证论文数字公平可比 | **§7 公平协议** |
| 单格重跑 / 补 MASPO 缺题 | **§8** |
| 查当前 seed123 进度与数字 | **§11** |

**铁律**：论文主表只用带 `fair_eval.policy` 的 `result/*.json`；MASPO 必须先于 EGMAP 锁定；**禁止**使用含 `handoff`/`residual_selector` 的旧 `maspo_formal_*.json`。

---

## 1. 两条协议（Baseline vs Ours）

| 维度 | **官方 MASPO** | **EGMAP** |
|------|----------------|-----------|
| 入口 | `run_maspo_formal_one_seed.py` | `run_egmap_formal_one_seed.py` |
| Prompt | 仅 **节点 prompt** optimize | 节点 prompt + **handoff** optimize |
| 推理 | 单路 `MAS.arun()` | handoff + disagreement + residual + **experience 检索** |
| Handoff | ❌ | `prompt/egmap_formal_*_handoffs.json` |
| Bank | ❌ | `memory/egmap_formal_*_bank.jsonl`（仅 opt 错题写入） |
| 结果文件 | `maspo_formal_{ds}_..._{profile}.json` | `egmap_formal_{ds}_..._b100k3_{profile}.json` |

固定超参（主表）：`graph=llm_agg`, `na=3`, `nr=1`, `opt_size=100`, `sample_size=200`, `depth=3`, `bank_size=100`, `top_k=3`, seeds `123/42/456`。

---

## 2. 新服务器部署清单

按顺序打勾，**全部通过后再跑正式实验**。

### 2.1 代码与 Python

```bash
git clone <your-EGMAP-repo-url>
cd Experience-Guided-Multi-Agent-Prompting   # 或你的克隆目录
export EGMAP_ROOT="$(pwd)"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 存储路径（按机器改）

```bash
export AFS_HOME=/path/to/storage          # 例：/mnt/afs/L202500372
export HANDOFF_DATASET_ROOT="${AFS_HOME}/data/egmap_handoff"
```

需存在：各 benchmark 的 handoff 数据（与现网 `egmap_handoff` 布局一致）。

### 2.3 模型权重

| Profile | 变量名 | 路径示例 |
|---------|--------|----------|
| m4b | `Qwen3.5-4B` | `${AFS_HOME}/models/Qwen3.5-4B` |
| m9b | `Qwen3.5-9B` | `${AFS_HOME}/models/Qwen3.5-9B` |
| m4b9b | 4B + 9B 各一份 | work :8005，strong :8004 |

### 2.4 vLLM 环境

需已安装 vLLM（本仓库常用 `/tmp/vllm-cu124` 或 `bootstrap` 脚本配套环境）。  
**文本阶段**可用 `--language-model-only`；**VQA 必须多模态**（见 §4）。

### 2.5 就绪探测

```bash
cd "${EGMAP_ROOT}"
source scripts/formal_model_profiles.sh
formal_apply_model_profile single_4b    # 或 single_9b / dual_4b_9b
formal_check_vllm_profile             # 应通过
curl -sf http://127.0.0.1:8005/v1/models && echo OK
```

### 2.6 可选：EGMAP 预检（建议每数据集正式跑前）

```bash
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123 --smoke --fast
```

---

## 3. 模型 Profile 与产物后缀

| Profile | 命令 | Work | Strong | 端口 | 后缀 |
|---------|------|------|--------|------|------|
| **m4b** | `MODEL_PROFILE=single_4b` | 4B | 4B | :8005 | `_m4b` |
| **m9b** | `MODEL_PROFILE=single_9b` | 9B | 9B | :8001 | `_m9b` |
| **m4b9b** | `MODEL_PROFILE=dual_4b_9b` | 4B | 9B | :8005/:8004 | `_m4b9b` |

**不同 profile 的 prompt / bank / result 互不覆盖**；`splits/egmap_formal_*_split.json` 按数据集+seed 共享（无 profile 后缀）。

防截断（文本/math **必开**，脚本内已 `source`）：

```bash
# 等价于 MASPO_WORK_MAX_TOKENS=8192, MASPO_WORK_MAX_PROMPT_CHARS=0
source scripts/formal_model_profiles.sh && formal_apply_tok8192_env
```

---

## 4. 启动 vLLM（文本 vs VQA）

### 4.1 文本 5 集（math500 … humaneval）

```bash
# 标准 4B 文本服务（language-model-only 可接受）
bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 4b --port 8005
```

### 4.2 VQA 3 集（vqarad / slake / chartqa）

**必须多模态**，否则报 `At most 0 image(s) may be provided` 且结果无效：

```bash
cd "${EGMAP_ROOT}"
bash scripts/restart_vllm_4b_multimodal.sh
# 脚本含：--enforce-eager --gdn-prefill-backend triton --gpu-memory-utilization 0.68
```

跑 VQA 实验时：

```bash
export RUN_VQA=1
```

### 4.3 切换文本 ↔ VQA

同一端口 `:8005` 上 **文本与多模态互斥**，换阶段需重启 vLLM：

1. 跑完文本 → `restart_vllm_4b_multimodal.sh`
2. 跑完 VQA → 若还要跑文本，再 `serve-qwen35.sh 4b --port 8005`

### 4.4 勿用错误 baseline 的环境

- ❌ 用 **text-only** vLLM 跑 VQA  
- ❌ 不设 `RUN_VQA=1` 跑 VQA（脚本会 skip）  
- ❌ 混用无 `_m4b` 后缀的旧 json 与新版 official MASPO

---

## 5. 标准流水线（新服务器推荐）

### 5.1 总览

```
┌─────────────────────────────────────────────────────────────┐
│  Phase A：文本 5 集  (RUN_VQA=0, 文本 vLLM @ :8005)          │
│    run_m4b_text.sh                                          │
│    → MASPO ×5 → EGMAP ×5 → fair_all_pairs → ledger          │
├─────────────────────────────────────────────────────────────┤
│  切换 vLLM → 多模态 4B                                       │
├─────────────────────────────────────────────────────────────┤
│  Phase B：VQA 3 集  (RUN_VQA=1)                              │
│    run_m4b_vqa.sh                                           │
│    → MASPO ×3 → EGMAP ×3 → fair_all_pairs → ledger          │
└─────────────────────────────────────────────────────────────┘
```

**顺序原因**：必须先 **MASPO** 再 **EGMAP**（baseline 锁定）；每对完成后做 **fair**（§7）。EGMAP 每格内部顺序见 §6。

### 5.2 Phase A — 文本（一键）

```bash
cd "${EGMAP_ROOT}"
export AFS_HOME=/path/to/storage
export MODEL_PROFILE=single_4b
export SEED=123

# 确认文本 vLLM
curl -sf http://127.0.0.1:8005/v1/models || bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 4b --port 8005

nohup bash scripts/run_m4b_text.sh >> logs/campaign_m4b_text.nohup 2>&1 &
tail -f logs/campaign_m4b_text.log
```

`run_m4b_text.sh` 内部：

| 步骤 | 脚本 | 数据集 |
|------|------|--------|
| 1 | `run_maspo_official_phase1.sh` | math500 aqua gpqa agieval humaneval |
| 2 | `run_egmap_official_phase2a.sh` | 同上 |
| 3 | `RERUN_MASPO=1 fair_all_pairs.sh` | 补 MASPO 缺题 + 公平后处理 |
| 4 | `update_result_ledger.py` | 写 RESULT.md |

已有合法 `_m4b` json 的格会 **skip**；强制重跑：`FORCE=1`。

### 5.3 Phase B — VQA（一键）

```bash
cd "${EGMAP_ROOT}"
bash scripts/restart_vllm_4b_multimodal.sh    # 等 API ready

export RUN_VQA=1
export MODEL_PROFILE=single_4b
export SEED=123
# 可选加速：export MAX_CONCURRENT=12 EGMAP_MAX_CONCURRENT=8

nohup bash scripts/run_m4b_vqa.sh >> logs/campaign_m4b_vqa.nohup 2>&1 &
tail -f logs/campaign_m4b_vqa.log
```

监控（可选）：`bash scripts/monitor_vqa_campaign.sh` 或 `tail -f logs/vqa_monitor.log`。

### 5.4 扩 seed / 换 profile

```bash
# seed 42 文本
export SEED=42 MODEL_PROFILE=single_4b
bash scripts/run_m4b_text.sh

# seed 42 VQA（记得 RUN_VQA=1 + 多模态 vLLM）
export SEED=42 RUN_VQA=1
bash scripts/run_m4b_vqa.sh

# 9B 单机（端口 :8001）
export MODEL_PROFILE=single_9b
bash scripts/run_campaign_m9b.sh   # 见 RUN.md
```

### 5.5 脚本索引（按用途）

| 脚本 | 何时用 |
|------|--------|
| **`scripts/run_m4b_text.sh`** | **新服：文本 5 集全流程（推荐）** |
| **`scripts/run_m4b_vqa.sh`** | **新服：VQA 3 集全流程（推荐）** |
| `scripts/run_maspo_official_phase1.sh` | 仅 MASPO；可 `DATASETS=gpqa` |
| `scripts/run_egmap_official_phase2a.sh` | 仅 EGMAP；MASPO 完成后 |
| `scripts/fair_all_pairs.sh` | 批量公平后处理；`RERUN_MASPO=1` 补题 |
| `scripts/rerun_maspo_fair_eval.sh` | 单/多集 MASPO `--skip-optimize` 补 manifest |
| `scripts/fair_pair_postprocess.py` | 单对 fair `--write` |
| `scripts/update_result_ledger.py` | 更新 RESULT.md |
| `scripts/preflight_egmap.py` | 跑前/跑后 EGMAP 门禁 |
| `scripts/export_egmap_maspo_table.py` | 导出 comparison 表（fair 后） |
| `scripts/restart_vllm_4b_multimodal.sh` | VQA 前启多模态 vLLM |

**不要**用仅 `prune_unscoreable_formal.py` 代替 fair；prune 是 fair 流程的子步骤之一。

---

## 6. EGMAP 单格内部三阶段

每格 `run_egmap_formal_one_seed.py`（`skip_optimize=False` 时）：

```
① Optimize（opt 100 题）
   ├─ 节点 prompt（depth=3, beam-refresh…）
   └─ handoff 契约优化 → prompts.json + handoffs.json
        ↓
② Stage1 建 bank（仍 opt 100 题，write_experience=True）
   └─ memory/*_bank.jsonl
        ↓
③ Stage2 frozen eval（manifest eval 题，write_experience=False）
   └─ result/egmap_formal_*_{profile}.json
```

MASPO 只有「节点 prompt optimize + eval」，**无 handoff、无 bank**。

`--skip-optimize`：跳过 ①，加载已有 prompts/handoffs，直接从 ②③ 开始（重跑 eval / 重建 bank 时用）。

---

## 7. 公平对比协议（论文主表必做）

### 7.1 原则

对每一对 `(dataset, seed, profile)`：

1. **同题集**：共用 `splits/egmap_formal_*_split.json` 的 `eval_unique_ids_run`
2. **同评分**：`rescore_formal_clean` 双边重算
3. **同分母**：**union** 剔除两边均不可自动评分的题（截断等），两边都删

### 7.2 命令

```bash
cd "${EGMAP_ROOT}"

# 单对
.venv/bin/python scripts/fair_pair_postprocess.py \
  --dataset math500 --seed 123 --model-suffix m4b --write

# 批量（MASPO 若缺 manifest 题，先补跑）
export SEED=123 MODEL_SUFFIX=m4b
export DATASETS="math500 aqua gpqa agieval humaneval"   # 或含 VQA
RERUN_MASPO=1 bash scripts/fair_all_pairs.sh

.venv/bin/python scripts/update_result_ledger.py --seed 123 --graph llm_agg
```

### 7.3 如何确认已 fair

```bash
.venv/bin/python -c "
import json; d=json.load(open('result/maspo_formal_math500_llm_agg_na3_d3s200o100seed123_m4b.json'))
print(d.get('fair_eval',{}).get('policy'))
"
# 应输出: manifest_sync + rescore + union_unscoreable_prune
```

### 7.4 常见不公平来源（务必避免）

| 问题 | 处理 |
|------|------|
| MASPO eval 题数 < manifest | `bash scripts/rerun_maspo_fair_eval.sh <ds>` 或 `RERUN_MASPO=1 fair_all_pairs.sh` |
| 用旧伪 MASPO（含 handoff） | 脚本自动归档到 `result/_invalid_pseudo_maspo/`；`FORCE=1` 重跑 |
| 只 prune 不做 pair sync | 必须 `fair_pair_postprocess.py --write` |
| VQA 用 text-only vLLM | 重跑；旧 json 作废 |
| 论文引 raw 分母 | 只引 `fair_eval` 后 `graph_types.llm_agg` |

### 7.5 fair n 说明

`fair n` ≤ manifest n 因数据集规模与 union 剔除（见 §11 脚注）。**公平性靠同题同分母，不靠把 n 撑大**；gpqa/humaneval 的 n 偏小是协议 `eval_run=min(200,|D|-100)` 所致，正文脚注即可。

---

## 8. 单格调试与补跑

```bash
cd "${EGMAP_ROOT}"
source scripts/formal_model_profiles.sh
formal_apply_model_profile single_4b
formal_apply_tok8192_env
export HANDOFF_DATASET_ROOT=...

# 官方 MASPO 整格（optimize + eval）
.venv/bin/python run_maspo_formal_one_seed.py \
  --dataset gpqa --graph llm_agg --na 3 --seed 123 \
  --opt-size 100 --sample-size 200 --depth 3 --max-concurrent 12

# MASPO 仅 eval（已有 prompts）
.venv/bin/python run_maspo_formal_one_seed.py --dataset gpqa --seed 123 --skip-optimize

# EGMAP 整格
.venv/bin/python run_egmap_formal_one_seed.py \
  --dataset gpqa --graph llm_agg --na 3 --seed 123 \
  --opt-size 100 --sample-size 200 --depth 3 \
  --bank-size 100 --top-k 3 --max-concurrent 8

# EGMAP 仅 bank + eval（已有 prompts/handoffs）
.venv/bin/python run_egmap_formal_one_seed.py --dataset gpqa --seed 123 --skip-optimize

# VQA 单格：先 export RUN_VQA=1 + 多模态 vLLM
```

强制覆盖已有结果：`FORCE=1 bash scripts/run_maspo_official_phase1.sh`（或 phase2a）。

---

## 9. 跑前 / 跑后门禁

### 跑前

- [ ] `curl -s http://127.0.0.1:<port>/v1/models` 成功
- [ ] VQA：`RUN_VQA=1` 且 vLLM **无** `--language-model-only`
- [ ] `HANDOFF_DATASET_ROOT` 指向正确数据
- [ ] 文本/math 已 tok8192（正式脚本默认已设）
- [ ] GPU 独占（避免多实验抢同一 vLLM）

### 跑后（每格或每对）

- [ ] MASPO json：`split_info` 无 handoff / residual（或 `protocol_ok`）
- [ ] EGMAP：`verify_bank_from_opt_only` 通过（preflight 可验）
- [ ] 双边 json 齐全 → `fair_pair_postprocess.py --write`
- [ ] `fair_eval.policy` 存在 → `update_result_ledger.py`
- [ ] 三 seed 齐再报 mean±std

---

## 10. 产物命名

| 类型 | 路径 |
|------|------|
| Split manifest | `splits/egmap_formal_{ds}_llm_agg_na3_d3s200o100seed{S}_b100k3_split.json` |
| MASPO prompts | `prompt/maspo_formal_{ds}_..._seed{S}_m4b_prompts.json` |
| EGMAP prompts / handoffs | `prompt/egmap_formal_*_{prompts,handoffs}.json` |
| EGMAP bank | `memory/egmap_formal_*_b100k3_m4b_bank.jsonl` |
| MASPO eval | `result/maspo_formal_{ds}_..._seed{S}_m4b.json` |
| EGMAP eval | `result/egmap_formal_{ds}_..._b100k3_m4b.json` |
| 日志 | `logs/{maspo,egmap}_formal_{ds}_..._m4b_official.log` |
| Fair 日志 | `logs/fair_{ds}_seed{S}_m4b.log` |

---

## 11. 当前进度与结果（m4b · seed123）

> 更新于 2026-06-17；详情见 [RESULT.md](RESULT.md)

### 11.1 流水线状态

| 阶段 | 状态 |
|------|------|
| Phase A 文本：MASPO + EGMAP + fair | ✅ 完成 |
| Phase B VQA：MASPO | ✅ 3/3（raw，待 fair） |
| Phase B VQA：EGMAP | 🔄 vqarad optimize 中 |
| Phase B VQA：fair | ❌ 待 EGMAP 完成 |

### 11.2 文本 5 集 — 公平主表（可写论文）

| 数据集 | manifest | fair n | MASPO | EGMAP | Δ |
|--------|:--------:|:------:|:-----:|:-----:|--:|
| math500 | 200 | 196 | 85.7% | 85.2% | −0.5 |
| aqua | 154 | 154 | 89.6% | 92.2% | +2.6 |
| gpqa | 98 | 94 | 71.3% | 75.5% | +4.3 |
| agieval | 156 | 151 | 84.1% | 88.7% | +4.6 |
| humaneval | 64 | 63 | 84.1% | 93.7% | +9.5 |
| **Macro（5 集等权）** | | | **83.0%** | **87.1%** | **+4.1** |

### 11.3 VQA 3 集 — 进行中（raw，未 fair）

| 数据集 | MASPO (raw) | EGMAP | Fair |
|--------|:-----------:|:-----:|:----:|
| vqarad | 123/200=61.5% | 进行中 | ❌ |
| slake | 126/200=63.0% | 排队 | ❌ |
| chartqa | 160/200=80.0% | 排队 | ❌ |

### 11.4 每集 eval 上限（协议）

| 数据集 | \|D\| | eval_run 上限 |
|--------|------:|:-------------:|
| math500 | 500 | 200 |
| aqua | 254 | 154 |
| gpqa | 198 | 98 |
| agieval | 256* | 156 |
| humaneval | 164 | 64 |
| vqarad / slake / chartqa | 451 / 1061 / 2500 | 200 |

\*agieval 以 `load_test_data` 为准；公式 `eval_run = min(200, |D|-100)`。

---

## 12. 后续待跑（优先级）

| 优先级 | 任务 | 命令要点 |
|:------:|------|----------|
| P0 | VQA EGMAP + fair | 等当前 `run_m4b_vqa.sh` 跑完；或手动 phase2a + `fair_all_pairs.sh` |
| P1 | m4b seed 42/456 | `SEED=42 bash scripts/run_m4b_text.sh` + `RUN_VQA=1 bash scripts/run_m4b_vqa.sh` |
| P2 | m4b9b / m9b 全矩阵 | `dual_4b_9b` / `single_9b` + 对应 vLLM |
| P3 | reflect `nr=2`、ablation | 副表；不进主表 |

---

## 13. 常见故障

| 现象 | 原因 | 处理 |
|------|------|------|
| VQA skip / 0 image | text-only vLLM 或 `RUN_VQA=0` | `restart_vllm_4b_multimodal.sh` + `RUN_VQA=1` |
| vLLM OOM | `gpu-memory-utilization` 过高 | multimodal 脚本已设 0.68；降 `MAX_CONCURRENT` |
| vLLM 启动极慢 | GDN JIT 编译 | 用 `restart_vllm_4b_multimodal.sh`（enforce-eager + triton） |
| MASPO fair 报缺题 | manifest 未满 | `rerun_maspo_fair_eval.sh` |
| EGMAP bank 含 eval 题 | 泄露 | `preflight_egmap.py`；重建 bank |
| 数字与 MASPO 不可比 | 未 fair / 伪 baseline | §7 |

---

## 14. 论文表目标

主表列：**Dataset | MASPO (official) | EGMAP | Δ**  
行：8 benchmark × 3-seed mean±std（当前仅 seed123 fair 可填文本 5 行）。

导出（fair 完成后）：

```bash
.venv/bin/python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
```

---

*维护：实验协议变更时同步更新 §5–§7；结果数字以 `fair_eval.policy` 存在的 json 为准。*
