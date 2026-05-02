"""
DPO training with optional bias-decoupling via dataset filtering.

Variants (--variant flag):
  A  baseline       — vanilla DPO on UltraFeedback
  B  length         — drop pairs where chosen >> rejected (length-decoupled)
  C  sycophancy     — drop pairs flagged as sycophantic agreement (heuristic)
  D  both           — both filters

Outputs a LoRA adapter to ./checkpoints/<variant>/.

Usage on RunPod (1× A100 80GB or H100, ~5-8 hrs per variant):
    python train_dpo.py --variant A
    python train_dpo.py --variant B
    python train_dpo.py --variant C
    python train_dpo.py --variant D
"""
import argparse
import re
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # non-gated; swap to Llama-3-8B once HF approval lands
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"

# Sycophantic-agreement heuristic — broadened to catch ~10-15% of pairs
# instead of <0.1%. Two parts:
#   AGREE_LEAD: chosen response opens with affirmation/praise patterns
#   OPINION_PROBE: user message expresses a stated belief OR asks confirmatory question
AGREE_LEAD = re.compile(
    r"^\s*("
    r"yes\b|absolutely\b|exactly\b|definitely\b|certainly\b|of course\b|indeed\b|"
    r"sure\b|right\b|correct\b|agreed\b|true\b|"
    r"you('re| are)\s+(right|correct|absolutely)|"
    r"that's\s+(correct|right|true|a (great|good)|exactly)|"
    r"i\s+agree\b|i\s+(do\s+)?think\s+you('re| are)\s+right|"
    r"great\s+(question|point|observation|insight|idea)|"
    r"good\s+(question|point|observation)|"
    r"what\s+(a|an)\s+(great|good|wonderful|thoughtful|excellent)\s+(question|point)|"
    r"that's\s+a\s+(great|good|wonderful|thoughtful|excellent)\s+(question|point)|"
    r"thank\s+you\s+for|"
    r"i\s+see\s+(what|where)\s+you('re| are)|"
    r"i\s+understand\s+(what|where|your)|"
    r"i\s+appreciate"
    r")",
    re.I,
)
# Match prompts that express a stated belief, or ask the model to confirm/validate.
OPINION_PROBE = re.compile(
    r"\b("
    r"i\s+(think|believe|feel|guess|assume|suspect|reckon|am\s+sure|am\s+pretty\s+sure)|"
    r"in\s+my\s+(opinion|view|experience|understanding)|"
    r"personally,?\s+i|"
    r"(am|are)\s+i\s+right|"
    r"isn't\s+it\s+true|"
    r"don't\s+you\s+(think|agree)|"
    r"do\s+you\s+(think|believe|agree)|"
    r"would\s+you\s+agree|"
    r"is\s+it\s+(true|correct|right)|"
    r"can\s+you\s+confirm|"
    r"i've\s+heard\s+that|"
    r"it\s+seems\s+(to\s+me|like)|"
    r"my\s+(view|understanding|impression)\s+is"
    r")\b",
    re.I,
)


def is_length_skewed(ex: dict, ratio: float = 1.5) -> bool:
    cl, rl = len(ex["chosen"][-1]["content"]), len(ex["rejected"][-1]["content"])
    if rl == 0:
        return True
    return cl / rl > ratio


def is_sycophantic(ex: dict) -> bool:
    user_msg = next((m["content"] for m in ex["chosen"] if m["role"] == "user"), "")
    chosen_resp = ex["chosen"][-1]["content"]
    # Trigger if response opens sycophantically AND (user signaled a belief OR
    # the chosen lead is praise-of-question style which is sycophantic regardless).
    if not AGREE_LEAD.match(chosen_resp):
        return False
    is_praise_lead = re.match(
        r"^\s*(great|good|what\s+(a|an)\s+(great|good|wonderful|thoughtful|excellent)|"
        r"that's\s+a\s+(great|good|wonderful|thoughtful|excellent)|thank\s+you|i\s+appreciate)",
        chosen_resp, re.I,
    )
    return is_praise_lead is not None or bool(OPINION_PROBE.search(user_msg))


def _load_elephant_drop_set(scores_path: str = "results/elephant_scores.json") -> set:
    """Load ELEPHANT-flagged indices to drop. C-prime / D-prime use this."""
    import json
    from pathlib import Path
    p = Path(scores_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run filter_elephant.py first to score UltraFeedback with ELEPHANT prompts."
        )
    scores = json.load(open(p))
    drops = {r["idx"] for r in scores if r.get("drop")}
    print(f"ELEPHANT scores loaded from {p}: {len(scores)} scored, {len(drops)} flagged for drop ({100*len(drops)/len(scores):.1f}%)")
    return drops


def filter_dataset(ds, variant: str):
    if variant == "A":
        return ds
    keep_len = lambda ex: not is_length_skewed(ex)
    keep_syco = lambda ex: not is_sycophantic(ex)
    if variant == "B":
        return ds.filter(keep_len)
    if variant == "C":
        return ds.filter(keep_syco)
    if variant == "D":
        return ds.filter(lambda ex: keep_len(ex) and keep_syco(ex))
    if variant == "Cp":
        # C-prime: ELEPHANT (gpt-4o-mini judge) instead of regex.
        drops = _load_elephant_drop_set()
        return ds.filter(lambda ex, idx: idx not in drops, with_indices=True)
    if variant == "Dp":
        # D-prime: length filter AND ELEPHANT.
        drops = _load_elephant_drop_set()
        return ds.filter(lambda ex, idx: keep_len(ex) and idx not in drops, with_indices=True)
    raise ValueError(f"unknown variant {variant}")


def main(variant: str, out_dir: Path, max_steps: int, seed: int = 42) -> None:
    print(f"=== Training variant {variant} (seed={seed}) ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds_full = load_dataset(DATASET_ID, split="train_prefs")
    ds = filter_dataset(ds_full, variant)
    print(f"Dataset: {len(ds_full)} → {len(ds)} after filtering")

    # Flatten chat-format → strings for TRL 0.11 DPOTrainer
    def to_strings(ex):
        return {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"][-1]["content"],
            "rejected": ex["rejected"][-1]["content"],
        }
    ds = ds.map(to_strings, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=1,
        max_steps=max_steps,
        logging_steps=20,
        save_steps=250,
        bf16=True,
        beta=0.1,
        max_length=768,
        max_prompt_length=384,
        gradient_checkpointing=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        tokenizer=tok,
        peft_config=lora,
    )

    # Resume from latest checkpoint if any exist in out_dir
    resume = False
    if out_dir.exists() and any(p.name.startswith("checkpoint-") for p in out_dir.iterdir()):
        resume = True
        print(f"Resuming from latest checkpoint in {out_dir}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir))
    print(f"Saved adapter to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "Cp", "Dp"])
    ap.add_argument("--out", default=None, help="default: ./checkpoints/<variant>/")
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = full epoch (publishable); 500 = pilot")
    ap.add_argument("--seed", type=int, default=42, help="random seed for DPO/dataloader")
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    out = Path(args.out or f"checkpoints/{args.variant}")
    main(args.variant, out, args.max_steps, args.seed)
