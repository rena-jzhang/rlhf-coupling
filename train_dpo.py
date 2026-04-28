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

# Sycophantic-agreement heuristic: chosen response leads with affirmation phrasing
# right after a user opinion claim. Cheap proxy; not perfect.
AGREE_LEAD = re.compile(
    r"^\s*(yes|absolutely|exactly|you('re| are) right|that's correct|i agree|great point|i think you're right)",
    re.I,
)
OPINION_PROBE = re.compile(r"\bI (think|believe|feel|guess|assume)\b", re.I)


def is_length_skewed(ex: dict, ratio: float = 1.5) -> bool:
    cl, rl = len(ex["chosen"][-1]["content"]), len(ex["rejected"][-1]["content"])
    if rl == 0:
        return True
    return cl / rl > ratio


def is_sycophantic(ex: dict) -> bool:
    user_msg = next((m["content"] for m in ex["chosen"] if m["role"] == "user"), "")
    if not OPINION_PROBE.search(user_msg):
        return False
    chosen_resp = ex["chosen"][-1]["content"]
    return bool(AGREE_LEAD.match(chosen_resp))


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
    raise ValueError(f"unknown variant {variant}")


def main(variant: str, out_dir: Path, max_steps: int) -> None:
    print(f"=== Training variant {variant} ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds_full = load_dataset(DATASET_ID, split="train_prefs")
    ds = filter_dataset(ds_full, variant)
    print(f"Dataset: {len(ds_full)} → {len(ds)} after filtering")

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
        save_steps=500,
        bf16=True,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        tokenizer=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved adapter to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--out", default=None, help="default: ./checkpoints/<variant>/")
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = full epoch")
    args = ap.parse_args()
    out = Path(args.out or f"checkpoints/{args.variant}")
    main(args.variant, out, args.max_steps)
