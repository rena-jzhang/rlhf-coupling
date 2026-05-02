#!/usr/bin/env bash
# C-prime two-seed experiment: ELEPHANT-filtered variant C with two seeds in parallel.
# Reuses A baseline from the prior run (already in results/eval_A.json on the repo).
#
# Required:
#   GH_PAT          — GitHub fine-grained PAT
#   OPENAI_API_KEY  — for filter_elephant.py (only if not already run)
#
# Optional:
#   AUTO_SHUTDOWN=1 — runpodctl stop pod after run completes
#   SEEDS="1 2"     — space-separated seeds to run (default "1 2")
#   ELEPHANT_N=-1   — passes to filter_elephant.py --n (default all 61135 pairs)

set -uo pipefail

SEEDS=${SEEDS:-"1 2"}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
ELEPHANT_N=${ELEPHANT_N:--1}

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
mkdir -p results checkpoints logs

# HF caches → /workspace volume
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# RUNPOD env from PID 1 (sshd login shells drop them)
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

# ────────────────────────────────────────────────────────────────────
# PHASE 0 — ELEPHANT filter (CPU/API only, runs once)
# ────────────────────────────────────────────────────────────────────
if [ ! -f "results/elephant_scores.json" ]; then
    echo "=== ELEPHANT filter pass ==="
    heartbeat "ELEPHANT" "filter start"
    push_results "[ELEPHANT] filter pass starting"
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: OPENAI_API_KEY not set; cannot run filter pass."
        exit 1
    fi
    python filter_elephant.py --model gpt-4o-mini --n "$ELEPHANT_N" --workers 30 --resume 2>&1 | tee logs/elephant.log
    heartbeat "ELEPHANT" "filter done"
    push_results "[ELEPHANT] filter scores written to results/elephant_scores.json"
else
    echo "elephant_scores.json exists; skipping filter pass"
fi

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NGPU GPUs"
if [ "$NGPU" -lt 1 ]; then exit 1; fi

# ────────────────────────────────────────────────────────────────────
# PHASE 1 — train Cp on each requested seed in parallel, queue across N GPUs
# ────────────────────────────────────────────────────────────────────
PENDING=()
for s in $SEEDS; do
    if [ -f "results/eval_Cp_seed${s}.json" ]; then
        echo "[Cp seed=$s] already done — skipping"
    else
        PENDING+=("$s")
    fi
done

FREE_GPUS=()
for ((i=0; i<NGPU; i++)); do FREE_GPUS+=("$i"); done
declare -A PID2S
declare -A PID2G

launch_cp() {
    local s="$1"
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    echo "[Cp seed=$s] launching on GPU $gpu"
    heartbeat "Cp_seed${s}" "train start (gpu=$gpu)"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu python train_dpo.py --variant Cp --seed $s --out 'checkpoints/Cp_seed${s}' > 'logs/train_Cp_seed${s}.log' 2>&1" < /dev/null &
    local pid=$!
    PID2S[$pid]="$s"
    PID2G[$pid]="$gpu"
}

while [ ${#PENDING[@]} -gt 0 ] && [ ${#FREE_GPUS[@]} -gt 0 ]; do
    s="${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
    launch_cp "$s"
done

while [ ${#PID2S[@]} -gt 0 ]; do
    wait -n
    rc=$?
    for pid in "${!PID2S[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            s="${PID2S[$pid]}"; gpu="${PID2G[$pid]}"
            ckpt_dir="checkpoints/Cp_seed${s}"
            has_ckpt="no"
            [ -d "$ckpt_dir" ] && [ -n "$(ls -A $ckpt_dir 2>/dev/null)" ] && has_ckpt="yes"
            echo "[Cp seed=$s] train finished rc=$rc ckpt=$has_ckpt (gpu=$gpu freed)"
            if [ "$has_ckpt" = "yes" ]; then
                heartbeat "Cp_seed${s}" "train done"
            else
                heartbeat "Cp_seed${s}" "train FAILED rc=$rc"
            fi
            unset PID2S[$pid] PID2G[$pid]
            FREE_GPUS+=("$gpu")
            if [ ${#PENDING[@]} -gt 0 ]; then
                next="${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
                launch_cp "$next"
            fi
            break
        fi
    done
done
push_results "[Cp] all seeds trained"

# ────────────────────────────────────────────────────────────────────
# PHASE 2 — eval each seed in parallel across GPUs
# ────────────────────────────────────────────────────────────────────
EVAL_FREE=()
for ((i=0; i<NGPU; i++)); do EVAL_FREE+=("$i"); done
declare -a EVAL_PIDS
declare -A EPID2S
declare -A EPID2G

for s in $SEEDS; do
    if [ -f "results/eval_Cp_seed${s}.json" ]; then continue; fi
    if [ ! -d "checkpoints/Cp_seed${s}" ]; then echo "[Cp seed=$s] no ckpt; skip eval"; continue; fi
    while [ ${#EVAL_FREE[@]} -eq 0 ]; do
        wait -n
        for pid in "${!EPID2S[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                ds="${EPID2S[$pid]}"; dg="${EPID2G[$pid]}"
                heartbeat "Cp_seed${ds}" "eval done"
                EVAL_FREE+=("$dg"); unset EPID2S[$pid] EPID2G[$pid]; break
            fi
        done
    done
    gpu="${EVAL_FREE[0]}"; EVAL_FREE=("${EVAL_FREE[@]:1}")
    echo "[Cp seed=$s] eval on GPU $gpu"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu python eval.py --adapter 'checkpoints/Cp_seed${s}' --tag 'Cp_seed${s}' --n-verb 300 --n-syco 500 > 'logs/eval_Cp_seed${s}.log' 2>&1" < /dev/null &
    pid=$!
    EPID2S[$pid]="$s"; EPID2G[$pid]="$gpu"
done

while [ ${#EPID2S[@]} -gt 0 ]; do
    wait -n
    for pid in "${!EPID2S[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            ds="${EPID2S[$pid]}"; dg="${EPID2G[$pid]}"
            heartbeat "Cp_seed${ds}" "eval done"
            EVAL_FREE+=("$dg"); unset EPID2S[$pid] EPID2G[$pid]; break
        fi
    done
done
push_results "[Cp] all evals done"

# ────────────────────────────────────────────────────────────────────
# PHASE 3 — quick summary (no analyze.py rewrite needed)
# ────────────────────────────────────────────────────────────────────
python - << 'PY'
import json, os, glob
print("\n=== C-prime two-seed summary ===")
print(f"{'tag':16s} {'verbosity':>10s} {'sycophancy':>12s}")
# Load A baseline + each Cp seed
files = ["results/eval_A.json"] + sorted(glob.glob("results/eval_Cp_seed*.json"))
rows = []
for fp in files:
    if not os.path.exists(fp): continue
    d = json.load(open(fp))
    tag = d.get("tag", os.path.basename(fp))
    verb = d.get("verbosity", {}).get("mean", float("nan"))
    syco = d.get("sycophancy", {}).get("flip_rate", float("nan"))
    rows.append((tag, verb, syco))
    print(f"{tag:16s} {verb:10.2f} {syco:12.3f}")
# Compute Cp average + delta vs A
cp_rows = [r for r in rows if r[0].startswith("Cp_seed")]
if cp_rows and any(r[0] == "A" for r in rows):
    a_syco = next(r[2] for r in rows if r[0] == "A")
    cp_syco = sum(r[2] for r in cp_rows) / len(cp_rows)
    print(f"\nCp mean syco across seeds: {cp_syco:.3f}")
    print(f"Δ syco vs A baseline:      {cp_syco - a_syco:+.3f} ({100*(cp_syco - a_syco):+.1f}pp)")
PY

heartbeat "ORCH" "Cp run complete"
push_results "[Cp] complete; per-seed eval JSONs + summary in logs"

if [ "$AUTO_SHUTDOWN" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== shutting down pod $RUNPOD_POD_ID in 60s ==="
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
echo "=== Cp orchestrator finished ==="
