# rlhf-coupling

**Question:** When you decouple one bias in an RLHF reward model (length), does another bias (sycophancy) get worse in the trained policy?

**Plan:** Llama-3-8B + LoRA + DPO on UltraFeedback. 4 variants × 3 metrics. ~$75 on RunPod spot, ~25-35 GPU-hrs.

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
huggingface-cli login   # need Llama-3-8B-Instruct access on HF
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
