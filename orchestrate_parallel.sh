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
if [ "$NGPU" -lt 1 ]; then
    echo "⚠️  No GPUs found."
    exit 1
fi

# ────────────────────────────────────────────────────────────────────
# PHASE 1 — queue variants across NGPU GPUs (more variants than GPUs is fine)
# ────────────────────────────────────────────────────────────────────
echo "=== queued train: $VARIANTS on $NGPU GPUs ==="
heartbeat "ORCH" "queued-train start"
push_results "[ORCH] queued-train start ($NGPU GPUs)"

# Filter pending variants
PENDING=()
for v in $VARIANTS; do
    if [ -f "results/eval_${v}.json" ]; then
        echo "[$v] already done — skipping"
    else
        PENDING+=("$v")
    fi
done

# GPU pool (free list)
FREE_GPUS=()
for ((i=0; i<NGPU; i++)); do FREE_GPUS+=("$i"); done

# pid -> variant, pid -> gpu
declare -A PID2V
declare -A PID2G

launch_next() {
    local v="$1"
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    echo "[$v] launching on GPU $gpu"
    heartbeat "$v" "train start (gpu=$gpu)"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu python train_dpo.py --variant '$v' --out 'checkpoints/$v' > 'logs/train_${v}.log' 2>&1" < /dev/null &
    local pid=$!
    PID2V[$pid]="$v"
    PID2G[$pid]="$gpu"
}

# Initial fill
while [ ${#PENDING[@]} -gt 0 ] && [ ${#FREE_GPUS[@]} -gt 0 ]; do
    v="${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
    launch_next "$v"
done

# Drain queue: wait for any to finish, free its GPU, launch next
while [ ${#PID2V[@]} -gt 0 ]; do
    wait -n
    rc=$?
    # Find which pid finished (kill -0 on each known pid)
    for pid in "${!PID2V[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            v="${PID2V[$pid]}"
            gpu="${PID2G[$pid]}"
            has_ckpt="no"
            [ -d "checkpoints/$v" ] && [ -n "$(ls -A checkpoints/$v 2>/dev/null)" ] && has_ckpt="yes"
            echo "[$v] finished rc=$rc ckpt=$has_ckpt (gpu=$gpu freed)"
            if [ "$has_ckpt" = "yes" ]; then
                echo "$(date -u +%FT%TZ) | $v | train done" >> logs/STATUS.md
            else
                echo "$(date -u +%FT%TZ) | $v | train FAILED rc=$rc ckpt=$has_ckpt" >> logs/STATUS.md
            fi
            unset PID2V[$pid] PID2G[$pid]
            FREE_GPUS+=("$gpu")
            # Launch next pending on freed GPU
            if [ ${#PENDING[@]} -gt 0 ]; then
                next="${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
                launch_next "$next"
            fi
            break
        fi
    done
done
push_results "[ORCH] queued-train all done"

# ────────────────────────────────────────────────────────────────────
# PHASE 2 — eval all in parallel
# ────────────────────────────────────────────────────────────────────
echo "=== parallel eval ==="
heartbeat "ORCH" "parallel-eval start"

EVAL_QUEUE=()
for v in $VARIANTS; do
    if [ -f "results/eval_${v}.json" ]; then continue; fi
    if [ ! -d "checkpoints/$v" ]; then echo "[$v] no checkpoint, skip eval"; continue; fi
    EVAL_QUEUE+=("$v:checkpoints/$v")
done
[ ! -f "results/eval_base.json" ] && EVAL_QUEUE+=("base:")

declare -a EVAL_PIDS
EVAL_FREE=()
for ((i=0; i<NGPU; i++)); do EVAL_FREE+=("$i"); done

launch_eval() {
    local entry="$1"
    local v="${entry%%:*}"
    local adapter="${entry#*:}"
    local gpu="${EVAL_FREE[0]}"
    EVAL_FREE=("${EVAL_FREE[@]:1}")
    echo "[$v] eval on GPU $gpu"
    if [ -n "$adapter" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python eval.py --adapter "$adapter" --tag "$v" > "logs/eval_${v}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$gpu python eval.py --tag "$v" > "logs/eval_${v}.log" 2>&1 &
    fi
    EVAL_PIDS+=("$!:$v:$gpu")
}

while [ ${#EVAL_QUEUE[@]} -gt 0 ] && [ ${#EVAL_FREE[@]} -gt 0 ]; do
    e="${EVAL_QUEUE[0]}"; EVAL_QUEUE=("${EVAL_QUEUE[@]:1}")
    launch_eval "$e"
done

while [ ${#EVAL_PIDS[@]} -gt 0 ]; do
    wait -n 2>/dev/null
    NEW_PIDS=()
    for entry in "${EVAL_PIDS[@]}"; do
        pid=${entry%%:*}; rest=${entry#*:}; v=${rest%%:*}; gpu=${rest##*:}
        if kill -0 "$pid" 2>/dev/null; then
            NEW_PIDS+=("$entry")
        else
            echo "[$v] eval done (gpu=$gpu freed)"
            heartbeat "$v" "eval done"
            EVAL_FREE+=("$gpu")
            if [ ${#EVAL_QUEUE[@]} -gt 0 ]; then
                e="${EVAL_QUEUE[0]}"; EVAL_QUEUE=("${EVAL_QUEUE[@]:1}")
                launch_eval "$e"
                NEW_PIDS+=("${EVAL_PIDS[-1]}")
            fi
        fi
    done
    EVAL_PIDS=("${NEW_PIDS[@]}")
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
