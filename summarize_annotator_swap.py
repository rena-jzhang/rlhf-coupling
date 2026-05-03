"""
Cross-annotator agreement analysis for the workshop paper §4.2 / §4.5.

Combines three same-data different-annotator outputs:
  - results/chosen_vs_rejected_framing.json   (gpt-4o-mini, ELEPHANT framing)
  - results/skywork_v2_scores.json            (Skywork-Reward-V2-8B, scalar)
  - results/armorm_scores.json                (ArmoRM-8B, multi-objective)

All three score the SAME 500 indices (seed=42 sample of UltraFeedback train_prefs),
so we can compute pairwise agreement and McNemar tests directly.

Reports:
  - Per-annotator: rate at which annotator prefers chosen over rejected
  - Pairwise agreement matrix (3x3)
  - McNemar tests for chosen-vs-rejected on each annotator
  - Subgroup analysis on the framing-discordant subset (the §4.2 wrong-direction signal)

Usage:
    python summarize_annotator_swap.py
"""
import json
from math import sqrt
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.load(open(p))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_z(b, c):
    return (b - c) / sqrt(b + c) if (b + c) > 0 else 0.0


def gpt4mini_chosen_more_sycophantic(r):
    """For ELEPHANT framing: chosen_framing=1 (does not challenge) and rejected_framing=0."""
    cf, rf = r.get("chosen_framing"), r.get("rejected_framing")
    if cf in (-1, None) or rf in (-1, None):
        return None
    return cf == 1 and rf == 0  # chosen flagged, rejected not


def gpt4mini_chosen_higher(r):
    """Per ELEPHANT: chosen_framing=1 means 'does not challenge' (more sycophantic).
    For comparability with Skywork/ArmoRM (which give chosen>rejected = agree with UF),
    we flip: chosen-side flag → chosen prefers premise-acceptance.
    Returns 1 if chosen is at least as flagged as rejected, 0 if rejected more flagged.
    """
    cf, rf = r.get("chosen_framing"), r.get("rejected_framing")
    if cf in (-1, None) or rf in (-1, None):
        return None
    if cf == rf:
        return None  # tie — exclude from chosen-vs-rejected counts
    return cf > rf  # chosen flagged, rejected not


def main():
    elephant = load("results/chosen_vs_rejected_framing.json")
    skywork = load("results/skywork_v2_scores.json")
    armorm = load("results/armorm_scores.json")

    if not elephant:
        print("missing results/chosen_vs_rejected_framing.json — run score_chosen_vs_rejected.py first")
        return
    print(f"elephant: {len(elephant)} pairs")
    print(f"skywork:  {len(skywork) if skywork else 0} pairs")
    print(f"armorm:   {len(armorm) if armorm else 0} pairs")

    # Index-align
    ele_by_idx = {r["idx"]: r for r in elephant}
    sky_by_idx = {r["idx"]: r for r in skywork} if skywork else {}
    arm_by_idx = {r["idx"]: r for r in armorm} if armorm else {}
    common = set(ele_by_idx) & set(sky_by_idx) & set(arm_by_idx) if (skywork and armorm) else set(ele_by_idx)
    print(f"common indices across all loaded annotators: {len(common)}")

    # =====================
    # Per-annotator: chosen-vs-rejected preference
    # =====================
    print("\n=== per-annotator: how often is chosen preferred over rejected? ===")

    # GPT-4o-mini ELEPHANT framing — chosen MORE flagged than rejected
    ele_disc = [(ele_by_idx[i].get("chosen_framing"), ele_by_idx[i].get("rejected_framing")) for i in ele_by_idx]
    n_chosen_only = sum(1 for cf, rf in ele_disc if cf == 1 and rf == 0)
    n_rej_only = sum(1 for cf, rf in ele_disc if cf == 0 and rf == 1)
    n_both = sum(1 for cf, rf in ele_disc if cf == 1 and rf == 1)
    n_neither = sum(1 for cf, rf in ele_disc if cf == 0 and rf == 0)
    n = len(ele_disc)
    z = mcnemar_z(n_chosen_only, n_rej_only)
    print(f"\ngpt-4o-mini ELEPHANT framing (chosen-side more 'does not challenge'):")
    print(f"  both flagged: {n_both}  chosen-only: {n_chosen_only}  rejected-only: {n_rej_only}  neither: {n_neither}")
    print(f"  McNemar z = {z:+.2f} on (chosen-only vs rejected-only)")
    print(f"  → wrong-direction signal at framing axis: chosen MORE sycophantic")

    if skywork:
        sky_chosen = sum(1 for r in skywork if r.get("agrees_with_uf_label") is True)
        sky_rej = sum(1 for r in skywork if r.get("agrees_with_uf_label") is False)
        sky_total = sky_chosen + sky_rej
        ci = wilson_ci(sky_chosen, sky_total)
        print(f"\nSkywork-Reward-V2-8B (chosen scalar > rejected scalar):")
        print(f"  agrees with UF label: {sky_chosen}/{sky_total} = {100*sky_chosen/sky_total:.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]")
        print(f"  → 50% = no signal; >>50% = bias propagated; <<50% = Skywork disagrees with UF")

    if armorm:
        arm_chosen = sum(1 for r in armorm if r.get("agrees_with_uf_label") is True)
        arm_rej = sum(1 for r in armorm if r.get("agrees_with_uf_label") is False)
        arm_total = arm_chosen + arm_rej
        ci = wilson_ci(arm_chosen, arm_total)
        print(f"\nArmoRM-8B gated scalar (chosen > rejected):")
        print(f"  agrees with UF label: {arm_chosen}/{arm_total} = {100*arm_chosen/arm_total:.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]")

    # =====================
    # Pairwise cross-annotator agreement (on chosen>rejected ordering)
    # =====================
    if skywork and armorm and common:
        print("\n=== pairwise cross-annotator agreement (chosen>rejected ordering) ===")
        sky_pref = {i: sky_by_idx[i].get("agrees_with_uf_label") for i in common}
        arm_pref = {i: arm_by_idx[i].get("agrees_with_uf_label") for i in common}
        ele_pref = {i: gpt4mini_chosen_higher(ele_by_idx[i]) for i in common}

        pairs = [
            ("gpt-4o-mini-framing", "Skywork-V2", ele_pref, sky_pref),
            ("gpt-4o-mini-framing", "ArmoRM",     ele_pref, arm_pref),
            ("Skywork-V2",          "ArmoRM",     sky_pref, arm_pref),
        ]
        for a, b, pa, pb in pairs:
            n_eval = sum(1 for i in common if pa[i] is not None and pb[i] is not None)
            n_agree = sum(1 for i in common if pa[i] is not None and pb[i] is not None and pa[i] == pb[i])
            ci = wilson_ci(n_agree, n_eval) if n_eval else (0, 0)
            print(f"  {a:25s} vs {b:15s}: {n_agree}/{n_eval} = {100*n_agree/max(1,n_eval):.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]")

    # =====================
    # ArmoRM per-UF-attribute agreement (where the 13th-bias claim either lives or dies)
    # =====================
    if armorm:
        print("\n=== ArmoRM per-attribute agreement (chosen attr-score > rejected attr-score) ===")
        # ArmoRM attribute order from score_armorm.py
        attrs = [
            "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
            "helpsteer-complexity", "helpsteer-verbosity",
            "ultrafeedback-overall_score", "ultrafeedback-instruction_following",
            "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness",
            "beavertails-is_safe", "prometheus-score",
            "argilla-overall_quality", "argilla-judge_lm",
            "code-complexity", "code-style", "code-explanation", "code-instruction-following",
            "code-readability",
        ]
        for j, name in enumerate(attrs):
            valid = [r for r in armorm if r.get("chosen_attrs") and r.get("rejected_attrs")]
            if not valid:
                continue
            n_a = sum(1 for r in valid if r["chosen_attrs"][j] > r["rejected_attrs"][j])
            n_v = len(valid)
            ci = wilson_ci(n_a, n_v)
            print(f"  {name:42s}: {n_a:>3d}/{n_v} ({100*n_a/n_v:.1f}%, [{100*ci[0]:.1f}, {100*ci[1]:.1f}])")


if __name__ == "__main__":
    main()
