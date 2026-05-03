#!/usr/bin/env bash
# Pod-side orchestrator for the same-data different-annotator experiment.
# Drops in next to score_chosen_vs_rejected.py / score_skywork_v2.py / score_armorm.py.
#
# Required:
#   GH_PAT (optional, only if you want results auto-pushed)
#
# Optional env:
#   N=500        # pairs to score (must match seed=42 sample of UltraFeedback)
#   AUTO_SHUTDOWN=1  # runpodctl stop pod after completion
#
# Expected pod: 1× A100 PCIe 40GB or H100 (8B reward models in bf16 ≈ 16GB).
# Expected wall: ~1 hr Skywork + 30 min ArmoRM + summary.

set -uo pipefail
cd "$(dirname "$0")"

N=${N:-500}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}

mkdir -p results logs

# HF caches → /workspace volume (persistent across pod restarts)
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Git auth (only if GH_PAT set)
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
    [ -z "${GH_PAT:-}" ] && return 0
    git pull --rebase --autostash 2>&1 | tail -3 || true
    git add -f results/ logs/ 2>/dev/null
    git commit -m "$1" 2>&1 | tail -2 || echo "(nothing to commit)"
    git push 2>&1 | tail -3 || echo "(push failed; continuing)"
}

echo "=== installing deps ==="
pip install -q transformers accelerate datasets torch tqdm 2>&1 | tail -3

# ──────────────────────────────────────────────────────────
# PHASE 1: Skywork-Reward-V2-Llama-3.1-8B
# ──────────────────────────────────────────────────────────
if [ ! -f "results/skywork_v2_scores.json" ]; then
    echo "=== Skywork-V2 scoring ==="
    python score_skywork_v2.py --n "$N" 2>&1 | tee logs/skywork_v2.log
    push_results "[annotator-swap] Skywork-V2 scores (n=$N)"
else
    echo "skywork_v2_scores.json exists; skipping"
fi

# ──────────────────────────────────────────────────────────
# PHASE 2: ArmoRM-Llama3-8B
# ──────────────────────────────────────────────────────────
if [ ! -f "results/armorm_scores.json" ]; then
    echo "=== ArmoRM scoring ==="
    python score_armorm.py --n "$N" 2>&1 | tee logs/armorm.log
    push_results "[annotator-swap] ArmoRM scores (n=$N)"
else
    echo "armorm_scores.json exists; skipping"
fi

# ──────────────────────────────────────────────────────────
# PHASE 3: cross-annotator summary
# ──────────────────────────────────────────────────────────
echo "=== summary ==="
python summarize_annotator_swap.py 2>&1 | tee logs/annotator_swap_summary.log
push_results "[annotator-swap] summary done"

if [ "$AUTO_SHUTDOWN" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== shutting down pod $RUNPOD_POD_ID in 60s ==="
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
echo "=== annotator-swap orchestrator finished ==="
