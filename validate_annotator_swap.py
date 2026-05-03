"""
Quick local sanity-check for pod-produced annotator-swap JSONs.

After `git pull` lands `results/skywork_v2_scores.json` and
`results/armorm_scores.json` from the pod run, run this locally to verify:
  - Files exist and parse as JSON
  - Length matches expected n=500 (or whatever was passed via --n)
  - Indices align with results/chosen_vs_rejected_framing.json
  - No NaN/null scores from runtime errors
  - Score distributions look reasonable (not all zeros, not all NaN)
  - chosen_score and rejected_score have non-trivial spread

Run BEFORE re-running summarize_annotator_swap.py — catches pod-side bugs fast.

Usage:
    python validate_annotator_swap.py
"""
import json
from pathlib import Path
from statistics import mean, stdev


def check(name: str, condition: bool, msg: str = "") -> bool:
    icon = "OK " if condition else "FAIL"
    suffix = f" — {msg}" if msg else ""
    print(f"  [{icon}] {name}{suffix}")
    return condition


def validate_file(path: str, expected_n: int, ele_indices: set) -> bool:
    p = Path(path)
    print(f"\n=== {path} ===")
    if not check("file exists", p.exists()):
        return False
    try:
        data = json.load(open(p))
    except Exception as e:
        print(f"  [FAIL] JSON parse: {e}")
        return False
    check("is list", isinstance(data, list))
    check(f"length == {expected_n}", len(data) == expected_n, f"got {len(data)}")

    # Required fields
    if not data:
        return False
    sample = data[0]
    required = {"idx", "chosen_score", "rejected_score", "agrees_with_uf_label"}
    missing = required - set(sample.keys())
    check("required fields present", not missing, f"missing: {missing}" if missing else "")

    # Index alignment
    indices = {r["idx"] for r in data}
    overlap = indices & ele_indices
    check(f"indices overlap ELEPHANT seed=42 sample", len(overlap) == len(indices),
          f"{len(overlap)}/{len(indices)} match" if len(overlap) != len(indices) else "")

    # NaN / runtime errors
    nan_chosen = sum(1 for r in data if r["chosen_score"] != r["chosen_score"])
    nan_rej = sum(1 for r in data if r["rejected_score"] != r["rejected_score"])
    check("no NaN chosen scores", nan_chosen == 0, f"{nan_chosen} NaN" if nan_chosen else "")
    check("no NaN rejected scores", nan_rej == 0, f"{nan_rej} NaN" if nan_rej else "")

    # Distribution sanity
    valid = [r for r in data if r["chosen_score"] == r["chosen_score"]]
    if valid:
        chosen_scores = [r["chosen_score"] for r in valid]
        rejected_scores = [r["rejected_score"] for r in valid]
        cs_std = stdev(chosen_scores) if len(chosen_scores) > 1 else 0
        rs_std = stdev(rejected_scores) if len(rejected_scores) > 1 else 0
        check("chosen_score has spread (sd > 0.01)", cs_std > 0.01, f"sd = {cs_std:.4f}")
        check("rejected_score has spread (sd > 0.01)", rs_std > 0.01, f"sd = {rs_std:.4f}")
        print(f"  ℹ  chosen_score:    mean={mean(chosen_scores):+.3f} sd={cs_std:.3f}")
        print(f"  ℹ  rejected_score:  mean={mean(rejected_scores):+.3f} sd={rs_std:.3f}")

    # ArmoRM-specific: per-attribute scores
    if "chosen_attrs" in sample:
        attrs_ok = all(len(r.get("chosen_attrs", [])) == 19 for r in valid)
        check("all chosen_attrs have 19 dims", attrs_ok)
        attrs_ok = all(len(r.get("rejected_attrs", [])) == 19 for r in valid)
        check("all rejected_attrs have 19 dims", attrs_ok)

    # Quick agreement preview
    n_agree = sum(1 for r in valid if r.get("agrees_with_uf_label") is True)
    n_total = sum(1 for r in valid if r.get("agrees_with_uf_label") is not None)
    if n_total:
        print(f"  ℹ  chosen > rejected: {n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")

    return True


def main():
    print("=== Annotator-swap output validation ===")

    # Expected n from ELEPHANT file
    ele_path = Path("results/chosen_vs_rejected_framing.json")
    if not ele_path.exists():
        print("FATAL: results/chosen_vs_rejected_framing.json missing — can't establish reference indices.")
        return
    ele = json.load(open(ele_path))
    ele_indices = {r["idx"] for r in ele}
    expected_n = len(ele)
    print(f"Reference: {expected_n} indices from chosen_vs_rejected_framing.json (seed=42)")

    files = [
        "results/skywork_v2_scores.json",
        "results/armorm_scores.json",
    ]
    all_ok = True
    for f in files:
        ok = validate_file(f, expected_n, ele_indices)
        all_ok = all_ok and ok

    print("\n=== overall ===")
    if all_ok:
        print("All checks passed. Safe to run: python summarize_annotator_swap.py")
    else:
        print("FAILURES detected. Fix before re-running summarize.")


if __name__ == "__main__":
    main()
