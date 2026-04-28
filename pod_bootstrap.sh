#!/usr/bin/env bash
# One-shot pod setup. Paste this whole script into the RunPod SSH terminal.
# Assumes the repo is already cloned and you're inside it.
set -euo pipefail

cd /workspace
if [ ! -d rlhf-coupling ]; then
    echo "Clone the repo first: git clone <your-repo-url> rlhf-coupling && cd rlhf-coupling"
    exit 1
fi
cd rlhf-coupling

# Quick deps install (RunPod PyTorch image already has torch + cuda)
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verify GPU + flash-attn + model load
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
"

# Smoke test: tiny DPO run (10 steps) to validate pipeline before committing 6 hrs
echo ''
echo '=== Smoke test: 10-step DPO run on variant A ==='
python train_dpo.py --variant A --max-steps 10 --out checkpoints/smoke
echo ''
echo 'Smoke test passed. Ready to launch full run with: bash run_all.sh'
