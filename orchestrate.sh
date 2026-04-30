#!/usr/bin/env bash
# Idempotent end-to-end orchestrator. Re-run after a crash; finished work skips.
#
# What it does:
#   - For each variant in A B C D:
#       train (resumable from checkpoint) → eval → git push results
#   - Eval base policy
#   - git push final summary
#   - shutdown pod (if RUNPOD_POD_ID set and AUTO_SHUTDOWN=1)
#
# Required env vars (set on the pod):
#   GH_PAT           — GitHub fine-grained PAT with write access to this repo
#   RUNPOD_POD_ID    — auto-injected by RunPod, no action needed
# Optional:
#   AUTO_SHUTDOWN    — "1" to runpodctl stop the pod when done (default: 0)
#   VARIANTS         — space-separated subset, e.g. "A B" (default: "A B C D")

set -uo pipefail

VARIANTS=${VARIANTS:-"A B C D"}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
MODE=${1:-run}   # "check" = print env diagnostics and exit, "run" = full pipeline

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

mkdir -p results checkpoints logs

# Force HF caches to /workspace volume (80GB) instead of container disk (20GB).
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# RunPod injects pod env on PID 1 but ssh login shells drop them. Pull them in.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ]; then
    while IFS= read -r -d '' kv; do
        case "$kv" in RUNPOD_*) export "$kv" ;; esac
    done < /proc/1/environ
fi

if [ "$MODE" = "check" ]; then
    echo "=== orchestrate.sh self-check ==="
    echo "ROOT: $ROOT"
    echo "VARIANTS: $VARIANTS"
    echo "AUTO_SHUTDOWN: $AUTO_SHUTDOWN"
    echo "GH_PAT set: $([ -n "${GH_PAT:-}" ] && echo yes || echo no)"
    echo "RUNPOD_POD_ID: ${RUNPOD_POD_ID:-(unset)}"
    echo "runpodctl: $(command -v runpodctl 2>/dev/null || echo MISSING)"
    echo "python: $(python --version 2>&1)"
    echo "torch + cuda:"
    python -c "import torch; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" 2>&1
    echo "git remote:"
    git remote -v 2>&1 | head -2
    echo "pre-existing results:"
    ls results/ 2>/dev/null || echo "  (none)"
    echo "pre-existing checkpoints:"
    ls checkpoints/ 2>/dev/null || echo "  (none)"
    if [ "$AUTO_SHUTDOWN" = "1" ] && ! command -v runpodctl >/dev/null; then
        echo ""
        echo "⚠️  AUTO_SHUTDOWN=1 but runpodctl not found. Pod won't auto-stop."
        echo "    Install with: pip install runpod && which runpodctl"
    fi
    echo ""
    echo "=== end self-check ==="
    exit 0
fi

# Git auth via PAT
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
    local msg="$1"
    git pull --rebase --autostash 2>&1 | tail -3 || true
    git add results/ logs/ 2>/dev/null
    git commit -m "$msg" 2>&1 | tail -2 || echo "(nothing to commit)"
    git push 2>&1 | tail -3 || echo "(push failed; continuing)"
}

heartbeat() {
    # one-line timestamped status to logs/STATUS.md (committed periodically)
    local phase="$1"; local detail="$2"
    echo "$(date -u +%FT%TZ) | $phase | $detail" >> logs/STATUS.md
}

run_variant() {
    local v="$1"
    local done_flag="results/eval_${v}.json"
    if [ -f "$done_flag" ]; then
        echo "[$v] already done — skipping (delete $done_flag to re-run)"
        return 0
    fi

    heartbeat "$v" "train start"
    push_results "[$v] starting train"

    # Training. --max-steps -1 = full epoch. resume_from_checkpoint handled inside.
    python train_dpo.py --variant "$v" --out "checkpoints/$v" 2>&1 | tee "logs/train_${v}.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        heartbeat "$v" "train FAILED rc=$rc"
        push_results "[$v] train FAILED rc=$rc"
        return 1
    fi

    heartbeat "$v" "eval start"
    python eval.py --adapter "checkpoints/$v" --tag "$v" 2>&1 | tee "logs/eval_${v}.log"
    local rc2=${PIPESTATUS[0]}
    if [ "$rc2" -ne 0 ]; then
        heartbeat "$v" "eval FAILED rc=$rc2"
        push_results "[$v] eval FAILED rc=$rc2"
        return 1
    fi

    heartbeat "$v" "done"
    push_results "[$v] complete: train + eval"
}

echo "=== orchestrator start: variants=[$VARIANTS] ==="
heartbeat "ORCH" "start"

for v in $VARIANTS; do
    run_variant "$v" || echo "[$v] failed; continuing"
done

# Base eval (no adapter)
if [ ! -f "results/eval_base.json" ]; then
    echo "=== eval base ==="
    heartbeat "base" "eval start"
    python eval.py --tag base 2>&1 | tee "logs/eval_base.log"
    heartbeat "base" "done"
    push_results "[base] eval complete"
fi

# Final summary
python - << 'PY'
import json, os
results = {}
for v in ["A", "B", "C", "D", "base"]:
    p = f"results/eval_{v}.json"
    if os.path.exists(p):
        results[v] = json.load(open(p))
with open("results/SUMMARY.json", "w") as f:
    json.dump(results, f, indent=2)
print("=== SUMMARY ===")
for v, r in results.items():
    if not r:
        continue
    verb = r.get("verbosity", {}).get("mean", float("nan"))
    syco = r.get("sycophancy", {}).get("flip_rate", float("nan"))
    print(f"  {v:5s}  verbosity={verb:7.1f}  sycophancy={syco:.3f}")
PY

heartbeat "ORCH" "all done"

# Auto-analysis
echo "=== running analyze.py ==="
pip install -q altair pandas vl-convert-python 2>&1 | tail -1
python analyze.py 2>&1 | tee logs/analyze.log

push_results "[ORCH] all variants complete; SUMMARY.json + REPORT.md written"

if [ "$AUTO_SHUTDOWN" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== shutting down pod $RUNPOD_POD_ID in 60s ==="
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
echo "=== orchestrator finished ==="
