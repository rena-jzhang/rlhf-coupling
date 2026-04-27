# rlhf-coupling

**Question:** When you decouple one bias in an RLHF reward model (length, à la ODIN), does another bias (sycophancy) get worse in the trained policy?

**Plan:** Llama-3-8B + LoRA + DPO on UltraFeedback. 4 RM variants × 3 metrics. ~$100, 2 weeks. See [proposal](../re-exploration/research/evals/b-llm-judge-bias/proposal-v3-cross-bias.md).

## Status

- **Day 0:** Preflight skipped — local M1 Pro / 16GB cannot run Llama-3-8B; preflight only on smaller models is not informative for the real experiment. Going straight to rented GPU for Day 1.
- **Day 1:** TBD — DPO pipeline setup on rented GPU.
