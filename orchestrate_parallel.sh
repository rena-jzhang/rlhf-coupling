#!/usr/bin/env bash
# Parallel orchestrator for 4-GPU pods. Runs variants A,B,C,D concurrently,
# one per GPU, then evals each. Total wall time ≈ time of slowest variant.
#
# Use this on a 4×A100 (or 4×H100) pod. For 1-GPU pods, use orchestrate.sh.
#
# Same env contract as orchestrate.sh: GH_PAT, AUTO_SHUTDOWN, RUNPOD_POD_ID.

set -uo pipefail

VARIANTS=${VARIANTS:-"A B C D"}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
mkdir -p results checkpoints logs

# Container disk is 20 GB; workspace volume is 80 GB. Force all HF caches to /workspace.
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Pull RUNPOD_* env from PID 1 (sshd login shells drop them)
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ]; then
    while IFS= read -r -d '' kv; do
        case "$kv" in RUNPOD_*) export "$kv" ;; esac
    done < /proc/1/environ
fi

# Git auth
if [ -n "${GH_PAT:-}" ]; then
    REMOTE_URL=$(git remote get-url origin)
    if [[ "$REMOTE_URL" != https://x-access-token:* ]]; then
        REPO_PATH=${REMOTE_URL#https://github.com/}
        REPO_PATH=${REPO_PATH#https://*@github.com/}
        git remote set-url origin "https://x-access-token:${GH_PAT}@github.com/${REPO_PATH}"
    fi
    git config user.email "pod@rlhf-coupling.local"
    git config user.name "pod-bot"
fi

push_results() {
    git pull --rebase --autostash 2>&1 | tail -3 || true
    git add results/ logs/ 2>/dev/null
    git commit -m "$1" 2>&1 | tail -2 || echo "(nothing to commit)"
    git push 2>&1 | tail -3 || echo "(push failed; continuing)"
}

heartbeat() {
    echo "$(date -u +%FT%TZ) | $1 | $2" >> logs/STATUS.md
}

# Verify GPU count
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NGPU GPUs"
if [ "$NGPU" -lt 4 ]; then
    echo "⚠️  This script expects ≥4 GPUs. Found $NGPU. Falling back to serial orchestrate.sh recommended."
    exit 1
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 1 — train all variants in parallel, one per GPU
# ────────────────────────────────────────────────────────────────────
echo "=== parallel train: $VARIANTS ==="
heartbeat "ORCH" "parallel-train start"
push_results "[ORCH] parallel-train start ($NGPU GPUs)"

declare -a PIDS
GPU_IDX=0
for v in $VARIANTS; do
    if [ -f "results/eval_${v}.json" ]; then
        echo "[$v] already done — skipping"
        continue
    fi
    echo "[$v] launching on GPU $GPU_IDX"
    heartbeat "$v" "train start (gpu=$GPU_IDX)"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU_IDX python train_dpo.py --variant '$v' --out 'checkpoints/$v' > 'logs/train_${v}.log' 2>&1" < /dev/null &
    PIDS+=("$!:$v:$GPU_IDX")
    GPU_IDX=$((GPU_IDX + 1))
done

# Wait for all + track failures
echo "=== waiting for $(echo $VARIANTS | wc -w) parallel trains ==="
for entry in "${PIDS[@]}"; do
    pid=${entry%%:*}
    rest=${entry#*:}
    v=${rest%%:*}
    wait "$pid"
    rc=$?
    has_ckpt="no"
    [ -d "checkpoints/$v" ] && [ -n "$(ls -A checkpoints/$v 2>/dev/null)" ] && has_ckpt="yes"
    echo "[$v] wait rc=$rc ckpt=$has_ckpt" | tee -a logs/STATUS.md
    if [ "$rc" = "0" ] && [ "$has_ckpt" = "yes" ]; then
        echo "$(date -u +%FT%TZ) | $v | train done" >> logs/STATUS.md
    else
        echo "$(date -u +%FT%TZ) | $v | train FAILED rc=$rc ckpt=$has_ckpt" >> logs/STATUS.md
    fi
done
push_results "[ORCH] parallel-train all done"

# ────────────────────────────────────────────────────────────────────
# PHASE 2 — eval all in parallel
# ────────────────────────────────────────────────────────────────────
echo "=== parallel eval ==="
heartbeat "ORCH" "parallel-eval start"

declare -a EVAL_PIDS
GPU_IDX=0
for v in $VARIANTS; do
    if [ -f "results/eval_${v}.json" ]; then continue; fi
    if [ ! -d "checkpoints/$v" ]; then echo "[$v] no checkpoint, skip eval"; continue; fi
    echo "[$v] eval on GPU $GPU_IDX"
    CUDA_VISIBLE_DEVICES=$GPU_IDX python eval.py --adapter "checkpoints/$v" --tag "$v" \
        > "logs/eval_${v}.log" 2>&1 &
    EVAL_PIDS+=("$!:$v")
    GPU_IDX=$((GPU_IDX + 1))
done

# Base policy eval on whichever GPU is free (use GPU 0)
if [ ! -f "results/eval_base.json" ]; then
    echo "[base] eval on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python eval.py --tag base \
        > "logs/eval_base.log" 2>&1 &
    EVAL_PIDS+=("$!:base")
fi

for entry in "${EVAL_PIDS[@]}"; do
    pid=${entry%%:*}
    v=${entry##*:}
    if wait "$pid"; then
        heartbeat "$v" "eval done"
    else
        heartbeat "$v" "eval FAILED rc=$?"
    fi
done
push_results "[ORCH] all evals done"

# ────────────────────────────────────────────────────────────────────
# PHASE 3 — analysis + shutdown
# ────────────────────────────────────────────────────────────────────
echo "=== analyze ==="
pip install -q altair pandas vl-convert-python 2>&1 | tail -1
python analyze.py 2>&1 | tee logs/analyze.log

heartbeat "ORCH" "all done"
push_results "[ORCH] complete; SUMMARY.json + REPORT.md written"

if [ "$AUTO_SHUTDOWN" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== shutting down pod $RUNPOD_POD_ID in 60s ==="
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
echo "=== orchestrator finished ==="
