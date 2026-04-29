"""
Local monitor — pull latest from GitHub and print pod status.

Usage:
    python monitor.py            # one-shot
    python monitor.py --watch    # poll every 60s

Run from your laptop. The pod auto-pushes results + logs/STATUS.md after
each phase, so this gives you progress visibility without SSH.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def git_pull() -> None:
    subprocess.run(["git", "pull", "--quiet"], cwd=ROOT, check=False)


def status() -> str:
    lines = ["=" * 60, f"rlhf-coupling status @ {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 60]

    status_file = ROOT / "logs" / "STATUS.md"
    if status_file.exists():
        last = status_file.read_text().strip().splitlines()[-5:]
        lines.append("Last 5 status events:")
        for ln in last:
            lines.append(f"  {ln}")
        lines.append("")

    lines.append("Phase results:")
    for v in ["A", "B", "C", "D", "base"]:
        p = ROOT / "results" / f"eval_{v}.json"
        if p.exists():
            try:
                r = json.loads(p.read_text())
                verb = r.get("verbosity", {}).get("mean", "?")
                syco = r.get("sycophancy", {}).get("flip_rate", "?")
                verb_s = f"{verb:.1f}" if isinstance(verb, (int, float)) else str(verb)
                syco_s = f"{syco:.3f}" if isinstance(syco, (int, float)) else str(syco)
                lines.append(f"  {v:5s} ✓  verbosity={verb_s:>8s}  sycophancy={syco_s}")
            except Exception:
                lines.append(f"  {v:5s} ?  (json parse failed)")
        else:
            lines.append(f"  {v:5s} —  pending")

    summary = ROOT / "results" / "SUMMARY.json"
    report = ROOT / "results" / "REPORT.md"
    if summary.exists():
        lines.append("")
        lines.append(f"✅ SUMMARY.json present — orchestrator finished")
    if report.exists():
        lines.append(f"✅ REPORT.md present — view with: cat results/REPORT.md")
    # Surface any sanity warnings from the eval JSONs
    warns = []
    for v in ["A", "B", "C", "D", "base"]:
        p = ROOT / "results" / f"eval_{v}.json"
        if p.exists():
            try:
                r = json.loads(p.read_text())
                for w in r.get("warnings", []):
                    warns.append(f"  {v}: {w}")
            except Exception:
                pass
    if warns:
        lines.append("")
        lines.append("⚠️  Sanity warnings:")
        lines.extend(warns)

    return "\n".join(lines)


def main(watch: bool) -> None:
    while True:
        git_pull()
        print(status())
        if not watch:
            break
        print("\n(refreshing in 60s — Ctrl-C to stop)\n")
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="poll every 60s")
    args = ap.parse_args()
    main(args.watch)
