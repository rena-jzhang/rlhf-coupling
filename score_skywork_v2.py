"""
Score UltraFeedback chosen + rejected with Skywork-Reward-V2-Llama-3.1-8B.

Same-data, different-annotator companion to score_chosen_vs_rejected.py:
  - score_chosen_vs_rejected.py: scores pairs with gpt-4o-mini ELEPHANT framing
  - this script:                 scores pairs with Skywork-Reward-V2 (scalar)

By default uses the SAME indices (seed=42, n=500) as score_chosen_vs_rejected.py
so the two outputs are directly comparable per-index.

Output: results/skywork_v2_scores.json — list of
  {idx, chosen_score, rejected_score, agrees_with_uf_label (chosen > rejected)}

Usage (pod, ~16GB VRAM, ~1 hr for 500 pairs):
    pip install transformers accelerate datasets torch
    python score_skywork_v2.py [--n 500] [--seed 42]
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

MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


def score_pair(rm, tok, prompt: str, response: str, device) -> float:
    """Return the scalar reward for (prompt, response) under Skywork-V2."""
    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tok.apply_chat_template(convo, tokenize=False)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = rm(**inputs)
    return float(out.logits[0][0].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/skywork_v2_scores.json")
    ap.add_argument("--max-chars", type=int, default=4000, help="truncate long responses")
    args = ap.parse_args()

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rm = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
        attn_implementation="sdpa",  # avoids needing flash-attn install on pod
    )
    rm.eval()
    device = next(rm.parameters()).device
    print(f"loaded on {device}")

    print("Loading UltraFeedback...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n)
    print(f"Scoring {args.n} pairs (seed={args.seed}) on Skywork-V2.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    pbar = tqdm(indices, desc="skywork-v2")
    for i in pbar:
        ex = ds[i]
        user_msg = next((m["content"] for m in ex["chosen"] if m["role"] == "user"), "")
        chosen_resp = ex["chosen"][-1]["content"]
        rejected_resp = ex["rejected"][-1]["content"]
        try:
            cs = score_pair(rm, tok, user_msg[: args.max_chars], chosen_resp[: args.max_chars], device)
            rs = score_pair(rm, tok, user_msg[: args.max_chars], rejected_resp[: args.max_chars], device)
        except Exception as e:
            print(f"  err idx={i}: {e}", file=sys.stderr)
            cs = rs = float("nan")
        results.append({
            "idx": i,
            "chosen_score": cs,
            "rejected_score": rs,
            "agrees_with_uf_label": (cs > rs) if not (cs != cs or rs != rs) else None,
        })
        if len(results) % 50 == 0:
            json.dump(results, open(out_path, "w"))
    json.dump(results, open(out_path, "w"))

    # Summary
    valid = [r for r in results if r["agrees_with_uf_label"] is not None]
    n_valid = len(valid)
    n_agree = sum(1 for r in valid if r["agrees_with_uf_label"])
    chosen_higher = sum(1 for r in valid if r["chosen_score"] > r["rejected_score"])
    mean_gap = sum(r["chosen_score"] - r["rejected_score"] for r in valid) / max(1, n_valid)
    print("\n=== summary ===")
    print(f"valid pairs:        {n_valid}")
    print(f"chosen > rejected:  {chosen_higher} ({100*chosen_higher/n_valid:.1f}%)  [Skywork-V2 agreement with UF labels]")
    print(f"mean (chosen-rej):  {mean_gap:+.3f}")
    print(f"interpretation: if Skywork-V2 agrees with UltraFeedback labels at >>50%, the bias propagated; if ~50%, bias is GPT-4-mini-specific (or random).")


if __name__ == "__main__":
    main()
