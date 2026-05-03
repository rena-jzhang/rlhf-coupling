"""
Score UltraFeedback chosen + rejected with ArmoRM-Llama3-8B-v0.1.

ArmoRM is a multi-objective reward model with 19 attribute heads. Most relevant
to the structural-sycophancy story:
  - ultrafeedback-helpfulness  (Likert 1-10 → normalized)
  - ultrafeedback-truthfulness
  - ultrafeedback-honesty
  - ultrafeedback-instruction_following
plus the gating/aggregated MoE score.

Same indices (seed=42, n=500) as score_chosen_vs_rejected.py.

Output: results/armorm_scores.json — list of
  {idx, chosen_score, rejected_score, chosen_attrs[19], rejected_attrs[19],
   agrees_with_uf_label}

Usage (pod, ~16GB VRAM, ~30 min for 500 pairs):
    pip install transformers accelerate datasets torch
    python score_armorm.py [--n 500] [--seed 42]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

# Per ArmoRM v0.1 model card; index → attribute name
ARMORM_ATTRS = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity",
    "ultrafeedback-overall_score", "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness",
    "beavertails-is_safe", "prometheus-score",
    "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation", "code-instruction-following",
    "code-readability",
]


def score_pair(rm, tok, prompt: str, response: str, device):
    """Return (gated_scalar, per_attr_19) under ArmoRM."""
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    ids = tok.apply_chat_template(convo, return_tensors="pt").to(device)
    with torch.no_grad():
        out = rm(ids)
    # ArmoRM output: out.score (scalar via gating), out.rewards (19 attribute heads)
    return float(out.score.item()), [float(x) for x in out.rewards[0].tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/armorm_scores.json")
    ap.add_argument("--max-chars", type=int, default=4000)
    args = ap.parse_args()

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    rm = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    rm.eval()
    device = next(rm.parameters()).device
    print(f"loaded on {device}")

    print("Loading UltraFeedback...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    pbar = tqdm(indices, desc="armorm")
    for i in pbar:
        ex = ds[i]
        user_msg = next((m["content"] for m in ex["chosen"] if m["role"] == "user"), "")
        chosen_resp = ex["chosen"][-1]["content"]
        rejected_resp = ex["rejected"][-1]["content"]
        try:
            cs, ca = score_pair(rm, tok, user_msg[: args.max_chars], chosen_resp[: args.max_chars], device)
            rs, ra = score_pair(rm, tok, user_msg[: args.max_chars], rejected_resp[: args.max_chars], device)
            agree = cs > rs
        except Exception as e:
            print(f"  err idx={i}: {e}", file=sys.stderr)
            cs = rs = float("nan"); ca = ra = []; agree = None
        results.append({
            "idx": i,
            "chosen_score": cs,
            "rejected_score": rs,
            "chosen_attrs": ca,
            "rejected_attrs": ra,
            "agrees_with_uf_label": agree,
        })
        if len(results) % 50 == 0:
            json.dump(results, open(out_path, "w"))
    json.dump(results, open(out_path, "w"))

    # Summary
    valid = [r for r in results if r["agrees_with_uf_label"] is not None]
    n_valid = len(valid)
    n_agree = sum(1 for r in valid if r["agrees_with_uf_label"])
    print("\n=== summary ===")
    print(f"valid pairs:           {n_valid}")
    print(f"gated chosen>rejected: {n_agree} ({100*n_agree/n_valid:.1f}%)")
    # Per-attribute agreement on the four UF dimensions
    uf_idx = {n: ARMORM_ATTRS.index(n) for n in ARMORM_ATTRS if n.startswith("ultrafeedback-")}
    print("\nper-attribute agreement (chosen-attr > rejected-attr):")
    for name, j in uf_idx.items():
        n_a = sum(1 for r in valid if r["chosen_attrs"] and r["chosen_attrs"][j] > r["rejected_attrs"][j])
        print(f"  {name:42s}: {n_a:>3d}/{n_valid} ({100*n_a/n_valid:.1f}%)")


if __name__ == "__main__":
    main()
