# rlhf-coupling

**Question:** When you decouple one bias in an RLHF reward model (length), does another bias (sycophancy) get worse in the trained policy?

**Plan:** Qwen2.5-7B-Instruct (non-gated) + LoRA + DPO on UltraFeedback. 4 variants × 3 metrics. ~$75 on RunPod spot, ~25-35 GPU-hrs.

## Deploy checklist (do these in order)

**Once, locally:** Create a fine-grained PAT at https://github.com/settings/personal-access-tokens/new — repo: `rlhf-coupling`, contents: read+write. Save the value.

**On the pod (web terminal, 4 pastes):**

1. **Clone + install:**
   ```bash
   cd /workspace && git clone https://github.com/rena-jzhang/rlhf-coupling.git && \
     cd rlhf-coupling && pip install -r requirements.txt && pip install runpod
   ```
2. **Self-check** (verifies torch/CUDA/git/runpodctl):
   ```bash
   GH_PAT=<your-pat> AUTO_SHUTDOWN=1 bash orchestrate.sh check
   ```
   Expected: `cuda=True`, GPU shown, GH_PAT set, runpodctl found, RUNPOD_POD_ID present.
3. **Smoke** (5-step DPO, ~30s on A100):
   ```bash
   bash pod_bootstrap.sh
   ```
4. **Launch** (~30 hrs end-to-end):
   ```bash
   GH_PAT=<your-pat> AUTO_SHUTDOWN=1 \
     nohup bash orchestrate.sh > logs/orchestrate.log 2>&1 &
   ```
   You can close the browser. Pod auto-commits results + STATUS.md after each variant.

**On your laptop, while it runs:**
```bash
cd ~/projects/rlhf-coupling
python monitor.py --watch
```
Every 60s pulls latest, prints which variants are done, last 5 events.

**Crash recovery:** if the pod dies or gets killed, redeploy and paste step 4 again. It skips finished variants (eval_*.json present) and resumes partial training from the latest checkpoint inside `checkpoints/<V>/`.

**Auto-shutdown:** when all 4 variants + base eval finish, pod runs `runpodctl stop pod $RUNPOD_POD_ID` after 60s. You won't burn idle $/hr. The volume is preserved; you can restart later if needed.

Full proposal: [`re-exploration/research/evals/b-llm-judge-bias/proposal-v3-cross-bias.md`](../re-exploration/research/evals/b-llm-judge-bias/proposal-v3-cross-bias.md)

## Variants

| Tag | Filter | Hypothesis if winning |
|---|---|---|
| **A** | none (vanilla) | baseline |
| **B** | drop pairs where chosen >> rejected length | sycophancy goes UP (H2 — coupling) |
| **C** | drop pairs where chosen leads with sycophantic agreement | verbosity may go UP |
| **D** | both filters | both biases lower; quality may drop |

Sycophancy filter is heuristic (regex on chosen response after detecting opinion-claim in prompt). Cheap proxy; documented in `train_dpo.py`.

## Metrics

- **Verbosity** = mean response length (tokens) on AlpacaEval prompts
- **Sycophancy** = TriviaQA "are you sure?" flip-rate (Sharma 2023 protocol)

## RunPod setup (one-time)

```bash
# pod: 1× A100 80GB or H100, PyTorch 2.4 image, 100GB volume
git clone <this-repo>
cd rlhf-coupling
pip install -r requirements.txt
huggingface-cli login   # only needed for private datasets; Qwen is open
bash run_all.sh
```

`run_all.sh` trains A→D (~6 hrs each) and evals each policy. Results in `results/eval_{A,B,C,D,base}.json`.

## Cost

| Item | Hours | $/hr (spot) | Total |
|---|---|---|---|
| Train A | 6 | $1.89 | $11 |
| Train B | 5 | $1.89 | $9 |
| Train C | 6 | $1.89 | $11 |
| Train D | 5 | $1.89 | $9 |
| Eval ×5 | 3 | $1.89 | $6 |
| Buffer (debug, restarts) | 8 | $1.89 | $15 |
| **Total** | ~33 | | **~$60-75** |

## Layout

```
rlhf-coupling/
├── train_dpo.py        # one variant per --variant flag
├── eval.py             # verbosity + sycophancy on a policy
├── run_all.sh          # orchestrate all 4 variants + base
├── requirements.txt
└── results/            # eval JSONs land here (gitignored)
```

## What's NOT in the repo (yet)

- Result analysis / plotting (will add after first eval JSON lands)
- Multi-seed runs (only after seed-1 results justify it)
- Real ODIN / PWP-style RM training (we use cheap dataset filtering as a Phase-0 stand-in)
