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

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

mkdir -p results checkpoints logs

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
push_results "[ORCH] all variants complete; SUMMARY.json written"

if [ "$AUTO_SHUTDOWN" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== shutting down pod $RUNPOD_POD_ID in 60s ==="
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
echo "=== orchestrator finished ==="
