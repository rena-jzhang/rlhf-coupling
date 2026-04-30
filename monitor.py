"""
Local monitor — pull latest from GitHub and print pod status.

Usage:
    python monitor.py            # one-shot
    python monitor.py --watch    # poll every 60s, fire macOS notifications
                                 # on state changes (variant done, warning, failure)

Run from your laptop. The pod auto-pushes results + logs/STATUS.md after
each phase, so this gives you progress visibility without SSH.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".monitor_state.json"  # local cache, gitignored


def notify(title: str, body: str, sound: str = "Glass") -> None:
    """macOS native notification + optional ntfy.sh push to phone."""
    # macOS native
    if platform.system() == "Darwin":
        try:
            msg = f'display notification "{body}" with title "{title}" sound name "{sound}"'
            subprocess.run(["osascript", "-e", msg], check=False, timeout=5)
        except Exception:
            pass
    # Phone via ntfy.sh — set NTFY_TOPIC env var to enable
    import os, urllib.request
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=body.encode("utf-8"),
                headers={"Title": title, "Priority": "default", "Tags": "robot"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2))


def diff_and_alert(prev: dict, curr: dict) -> dict:
    """Compare prev vs curr state; fire notifications on transitions."""
    new_state = dict(curr)

    # Variant completion: was pending → now has result
    for v in ["A", "B", "C", "D", "base"]:
        was = prev.get("variants", {}).get(v, "pending")
        now = curr["variants"].get(v, "pending")
        if was == "pending" and now == "done":
            r = curr["results"].get(v, {})
            verb = r.get("verbosity_mean", "?")
            syco = r.get("sycophancy_flip", "?")
            verb_s = f"{verb:.1f}" if isinstance(verb, (int, float)) else str(verb)
            syco_s = f"{syco:.3f}" if isinstance(syco, (int, float)) else str(syco)
            notify(f"✅ Variant {v} done", f"verbosity={verb_s}  sycophancy={syco_s}")

    # New sanity warnings
    prev_warns = set(tuple(w) for w in prev.get("warnings", []))
    curr_warns = set(tuple(w) for w in curr["warnings"])
    new_warns = curr_warns - prev_warns
    for v, w in new_warns:
        notify(f"⚠️ Sanity warning: {v}", w, sound="Sosumi")

    # Failure detection from STATUS.md
    prev_fails = set(prev.get("failures", []))
    curr_fails = set(curr["failures"])
    for f in (curr_fails - prev_fails):
        notify("❌ Run failure", f, sound="Basso")

    # Final completion
    if not prev.get("done") and curr["done"]:
        notify("🎉 Run complete", "All variants finished. REPORT.md is ready.")

    return new_state


def git_pull() -> None:
    subprocess.run(["git", "pull", "--quiet"], cwd=ROOT, check=False)


def collect_state() -> dict:
    """Snapshot of current pulled state — used both for display and diff."""
    state = {"variants": {}, "results": {}, "warnings": [], "failures": [], "done": False}
    for v in ["A", "B", "C", "D", "base"]:
        p = ROOT / "results" / f"eval_{v}.json"
        if p.exists():
            state["variants"][v] = "done"
            try:
                r = json.loads(p.read_text())
                state["results"][v] = {
                    "verbosity_mean": r.get("verbosity", {}).get("mean"),
                    "sycophancy_flip": r.get("sycophancy", {}).get("flip_rate"),
                }
                for w in r.get("warnings", []):
                    state["warnings"].append([v, w])
            except Exception:
                state["variants"][v] = "broken"
        else:
            state["variants"][v] = "pending"

    state["done"] = (ROOT / "results" / "SUMMARY.json").exists()

    # Pull failures from STATUS.md
    sf = ROOT / "logs" / "STATUS.md"
    if sf.exists():
        for line in sf.read_text().splitlines():
            if "FAILED" in line:
                state["failures"].append(line.strip())
    return state


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
    prev = load_state()
    if watch:
        notify("Monitor started", "Watching rlhf-coupling. Will alert on completion / failure.")
    while True:
        git_pull()
        curr = collect_state()
        if watch:
            diff_and_alert(prev, curr)
            save_state(curr)
            prev = curr
        print(status())
        if not watch:
            break
        if curr["done"]:
            print("\n✅ Run complete. Stopping watch.\n")
            break
        print("\n(refreshing in 60s — Ctrl-C to stop)\n")
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="poll every 60s")
    args = ap.parse_args()
    main(args.watch)
