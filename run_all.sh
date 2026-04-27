#!/usr/bin/env bash
# Train all 4 DPO variants and eval each. Run on a single A100/H100.
# Total: ~25-35 GPU-hrs, ~$50-75 on RunPod spot.
set -euo pipefail

for v in A B C D; do
    echo "=== train $v ==="
    python train_dpo.py --variant $v
    echo "=== eval $v ==="
    python eval.py --adapter checkpoints/$v --tag $v
done

echo "=== eval base (no adapter) ==="
python eval.py --tag base

echo "Done. Results in ./results/"
