# EGMAP 正式实验运行手册（双机分工）

> 仓库：[panhongxing-sds/EGMAP](https://github.com/panhongxing-sds/EGMAP)  
> 协议细节：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) · 方法定义：[METHOD.md](METHOD.md) · 结果台账：[RESULT.md](RESULT.md)

---

## 1. 分工总览

两台机器**并行、互不覆盖**，靠结果文件名后缀区分模型：

| 机器 | Profile | 模型 | vLLM 端口 | 结果后缀 | 负责 Campaign |
|------|---------|------|-----------|----------|---------------|
| **本机（小模型）** | `single_4b` / **m4b** | Qwen3.5-4B work+strong+judge | `:8005` | `_m4b` | **Campaign 1** |
| **另一台（大模型）** | `single_9b` / **m9b** | Qwen3.5-9B work+strong+judge | `:8001` | `_m9b` | **Campaign 2** |

**不要混跑**：同一数据集、同一 seed 上，m4b 与 m9b 各产一份 json，论文主表可分行或分表汇报。

### 实验矩阵（每台机器都要跑完整套）

- **8 数据集**：`math500 aqua gpqa agieval humaneval vqarad slake chartqa`
- **2 方法**：官方 **MASPO**（baseline）→ **EGMAP**（ours）
- **阶段顺序**：先 MASPO 全矩阵锁定 baseline → 再 EGMAP 全矩阵
- **当前 seed**：`123`（打通后再扩 `42`、`456`）
- **图拓扑**：`llm_agg`，`na=3`，`nr=1`（reflect `nr=2` 另做，不进主表）

---

## 2. 环境准备（两台机器相同步骤）

### 2.1 克隆与 Python

```bash
git clone https://github.com/panhongxing-sds/EGMAP.git
cd EGMAP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 路径变量（必设）

```bash
# 项目根目录（按实际路径改）
export EGMAP_ROOT="$(pwd)"

# 数据与模型根目录（按实际路径改；可与 EGMAP_ROOT 不同）
export AFS_HOME=/path/to/your/storage   # 例：/mnt/afs/L202500372

export HANDOFF_DATASET_ROOT="${AFS_HOME}/data/egmap_handoff"
```

数据目录需包含各 benchmark 的 handoff 格式数据（与当前 `egmap_handoff` 布局一致）。

### 2.3 模型权重

| Profile | 需要下载 |
|---------|----------|
| m4b | `Qwen3.5-4B` → `${AFS_HOME}/models/Qwen3.5-4B` |
| m9b | `Qwen3.5-9B` → `${AFS_HOME}/models/Qwen3.5-9B` |

### 2.4 vLLM 服务

**必须用标准 serve，不要用 turbo 脚本**（turbo 会触发 flashinfer GDN JIT，首启极慢且易占满显存）。

```bash
# m4b 机器
bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 4b --port 8005

# m9b 机器
bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 9b --port 8001
```

就绪检查：

```bash
curl -s http://127.0.0.1:8005/v1/models   # m4b
curl -s http://127.0.0.1:8001/v1/models   # m9b
```

VQA 数据集（`vqarad slake chartqa`）需要 **4B/9B 多模态** vLLM（去掉 `--language-model-only`）。可用：

```bash
bash scripts/restart_vllm_4b_multimodal.sh   # m4b 端
# m9b 端同理改端口与模型路径
```

### 2.5 防截断（文本/math 必开）

```bash
source scripts/formal_apply_tok8192_env.sh
# MASPO_WORK_MAX_TOKENS=8192, MASPO_WORK_MAX_PROMPT_CHARS=0
```

---

## 3. Campaign 1 — 本机跑 4B（m4b）

### 3.1 一键全流程（推荐）

```bash
cd "${EGMAP_ROOT}"
export AFS_HOME=/mnt/afs/L202500372    # 按本机实际路径

# 后台跑：等 vLLM → MASPO 剩余集 → EGMAP 8 集 → prune + 台账
nohup bash scripts/run_campaign_m4b_now.sh > logs/campaign_m4b.nohup 2>&1 &
tail -f logs/campaign_m4b_now.log
```

`run_campaign_m4b_now.sh` 逻辑：

1. 等待 `:8005` vLLM ready（未就绪则自动 `serve-qwen35.sh 4b`）
2. **MASPO**：`gpqa agieval humaneval vqarad slake chartqa`（`math500`/`aqua` 若已有有效 `_m4b` json 会 skip）
3. **EGMAP**：8 数据集全跑，`SKIP_PREFLIGHT=1`（正式跑前建议先单格 preflight）
4. `prune_unscoreable_formal.py` + `update_result_ledger.py`

### 3.2 分步跑（调试 / 单格重跑）

```bash
export MODEL_PROFILE=single_4b
export SEED=123
export MAX_CONCURRENT=16          # MASPO 并发
export EGMAP_MAX_CONCURRENT=8     # EGMAP 并发（传给 phase2a 前 export MAX_CONCURRENT=8）

# Phase 1 — 官方 MASPO
bash scripts/run_maspo_official_phase1.sh
DATASETS=gpqa FORCE=1 bash scripts/run_maspo_official_phase1.sh   # 单格强制重跑

# Phase 2 — EGMAP（MASPO 锁定后）
bash scripts/run_egmap_official_phase2a.sh
DATASETS=math500 FORCE=1 bash scripts/run_egmap_official_phase2a.sh
```

### 3.3 EGMAP 预检（建议每数据集跑正式格之前）

```bash
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123
.venv/bin/python scripts/preflight_egmap.py --dataset math500 --seed 123 --smoke --fast
```

### 3.4 本机当前进度（seed123）

| 数据集 | MASPO m4b | 说明 |
|--------|-----------|------|
| math500 | ✅ 85.6% | 已锁定 |
| aqua | ✅ 90.1% | 已锁定 |
| gpqa / agieval / humaneval / VQA×3 | ❌ 待跑 | campaign 进行中 |

EGMAP `_m4b` 后缀结果：**8 集待重跑**（旧无后缀 json 勿用于论文）。

---

## 4. Campaign 2 — 另一台机器跑 9B（m9b）

### 4.1 一键全流程

```bash
cd "${EGMAP_ROOT}"
export AFS_HOME=/path/to/your/storage   # 9B 机器上的根路径

nohup bash scripts/run_campaign_m9b.sh > logs/campaign_m9b.nohup 2>&1 &
tail -f logs/campaign_m9b_now.log
```

### 4.2 分步跑

```bash
# 先起 9B vLLM
bash "${AFS_HOME}/bootstrap/serve-qwen35.sh" 9b --port 8001

export MODEL_PROFILE=single_9b
export SEED=123
export MAX_CONCURRENT=12
export AFS_HOME=/path/to/your/storage

# 1) 官方 MASPO 全 8 集
bash scripts/run_maspo_official_phase1.sh

# 2) EGMAP 全 8 集
export MAX_CONCURRENT=6
bash scripts/run_egmap_official_phase2a.sh
```

### 4.3 产物命名

```
result/maspo_formal_{ds}_llm_agg_na3_d3s200o100seed123_m9b.json
result/egmap_formal_{ds}_llm_agg_na3_d3s200o100seed123_b100k3_m9b.json
prompt/maspo_formal_*_m9b_prompts.json
prompt/egmap_formal_*_m9b_{prompts,handoffs}.json
memory/egmap_formal_*_m9b_bank.jsonl
```

`splits/egmap_formal_*_split.json` **无后缀**，两台机器共用（按 seed 一致即可）。

---

## 5. 协议核对（跑后必查）

### 官方 MASPO（有效 baseline）

```bash
python -c "
import json, sys
d=json.load(open(sys.argv[1]))
si=d.get('split_info') or {}
bad = d.get('residual_selector') or si.get('residual_selector') or si.get('handoff')
print('INVALID' if bad else 'OK official MASPO')
" result/maspo_formal_math500_*_m4b.json
```

必须：`handoff=false`, `residual_selector=false`, `disagreement_handoff=false`。

### EGMAP

- bank 仅含 opt 100 错题，无 `correct=True` 行
- result 含 `residual` / `experience` 字段
- `preflight_egmap.py --check-eval` 通过

### prune

```bash
.venv/bin/python scripts/prune_unscoreable_formal.py --write result/maspo_formal_*_m4b.json
.venv/bin/python scripts/prune_unscoreable_formal.py --write result/egmap_formal_*_m4b.json
```

---

## 6. 双机结果合并

1. 各机跑完执行：`python scripts/update_result_ledger.py --seed 123 --graph llm_agg`
2. 将 `result/*_m9b.json`（及可选 `prompt/`、`memory/`）拷回本机或 git 同步
3. 导出对照表（**仅当官方 MASPO 有效 json 齐全后**）：

```bash
python scripts/export_egmap_maspo_table.py --auto -o result/comparison_table.md
```

4. 更新 [RESULT.md](RESULT.md) 台账；**在 m4b 与 m9b 均完成前不要报跨模型 Δ**

---

## 7. Phase 3 — 扩 seed 42 / 456

两台机器同步扩：

```bash
export SEED=42   # 或 456
export MODEL_PROFILE=single_4b   # 或 single_9b
bash scripts/run_maspo_official_phase1.sh
bash scripts/run_egmap_official_phase2a.sh
```

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| `APIConnectionError` / vLLM 未就绪 | 先 `curl :8005/:8001/v1/models`；未 ready 不要开跑 |
| turbo vLLM + flashinfer JIT 卡死 | `pkill -f vllm.*8005; pkill -f nvcc.*flashinfer` 后改用标准 `serve-qwen35.sh` |
| 显存不足 `Free memory ... less than desired` | 杀掉残留 `VLLM::EngineCore` / nvcc 后再启 vLLM |
| 旧 `maspo_formal_*` 无 `_m4b` 后缀 | **全部作废**（含 residual/handoff）；`FORCE=1` 重跑 |
| math500 EGMAP bank=0 | 独占 GPU 重建 stage1 bank 后再 eval |
| VQA 无分数 | 确认多模态 vLLM 已启，非 `language-model-only` |

---

## 9. 日志位置

| 日志 | 路径 |
|------|------|
| m4b campaign | `logs/campaign_m4b_now.log` |
| m9b campaign | `logs/campaign_m9b_now.log` |
| 单格 MASPO | `logs/maspo_formal_{ds}_*_m4b_official.log` |
| 单格 EGMAP | `logs/egmap_formal_{ds}_*_m4b_official.log` |
| vLLM | `${AFS_HOME}/logs/vllm-Qwen3.5-4B-8005.log` 或 `...-9B-8001.log` |

---

## 10. 最小命令速查

```bash
# === 本机 m4b ===
export AFS_HOME=/mnt/afs/L202500372 EGMAP_ROOT=$PWD
bash scripts/run_campaign_m4b_now.sh

# === 远端 m9b ===
export AFS_HOME=/your/path EGMAP_ROOT=$PWD
bash scripts/run_campaign_m9b.sh

# === 单 seed 单数据集 MASPO ===
MODEL_PROFILE=single_4b python run_maspo_formal_one_seed.py --dataset math500 --seed 123

# === 单 seed 单数据集 EGMAP ===
MODEL_PROFILE=single_4b python run_egmap_formal_one_seed.py --dataset math500 --seed 123
```
