from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance


RUNLOG = Path(
    "/mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/"
    "qframe_5seed_20260701"
)

LABEL_RE = re.compile(
    r"^(?P<method>qframe_v1|longclip_image|longclip_fused)_"
    r"(?P<task>rc3|rc5|rc9|intercept|shell)_s(?P<seed>\d+)_5seed_20260701$"
)


def launcher_summary_path(label: str) -> Path | None:
    log_path = RUNLOG / f"{label}.launcher.log"
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        if "[eval]" in line and " summary=" in line:
            return Path(line.split(" summary=", 1)[1].split(" log=", 1)[0])
        if line.startswith("SUMMARY="):
            return Path(line.split("=", 1)[1])
    return None


def iter_started_labels() -> list[str]:
    master = RUNLOG / "master.log"
    labels: list[str] = []
    if not master.exists():
        return labels
    for line in master.read_text(errors="replace").splitlines():
        if line.startswith("[start]") and " label=" in line:
            labels.append(line.split(" label=", 1)[1].split(" ", 1)[0])
    return labels


def summarize(values: list[float]) -> tuple[float, float]:
    return mean(values), pvariance(values)


def fmt_mean_var(values: list[float]) -> str:
    if not values:
        return ""
    avg, var = summarize(values)
    return f"{avg:.3f} / {var:.5f}"


def main() -> None:
    rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    missing: list[str] = []
    incomplete: list[str] = []

    for label in iter_started_labels():
        match = LABEL_RE.match(label)
        if not match:
            continue
        summary_path = launcher_summary_path(label)
        if summary_path is None:
            incomplete.append(label)
            continue
        if not summary_path.exists():
            missing.append(f"{label}: {summary_path}")
            continue
        data = json.loads(summary_path.read_text())
        rows[(match["method"], match["task"])].append(
            {
                "seed": int(match["seed"]),
                "success_once": float(data["success_once"]),
                "success_at_end": float(data["success_at_end"]),
                "num_episodes": int(data["num_episodes"]),
                "summary": str(summary_path),
            }
        )

    methods = ["qframe_v1", "longclip_image", "longclip_fused"]
    tasks = ["rc3", "rc5", "rc9", "intercept", "shell"]
    print("SUCCESS_ONCE mean / variance")
    print("| method | RC3 | RC5 | RC9 | Intercept | Shell |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        cells = []
        for task in tasks:
            vals = [float(row["success_once"]) for row in rows[(method, task)]]
            cells.append(fmt_mean_var(vals))
        print(f"| {method} | " + " | ".join(cells) + " |")

    print()
    print("SUCCESS_AT_END mean / variance")
    print("| method | RC3 | RC5 | RC9 | Intercept | Shell |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        cells = []
        for task in tasks:
            vals = [float(row["success_at_end"]) for row in rows[(method, task)]]
            cells.append(fmt_mean_var(vals))
        print(f"| {method} | " + " | ".join(cells) + " |")

    print()
    print("Per-seed values")
    for method in methods:
        for task in tasks:
            vals = sorted(rows[(method, task)], key=lambda row: int(row["seed"]))
            if not vals:
                continue
            once = ", ".join(
                f"{row['seed']}:{float(row['success_once']):.3f}" for row in vals
            )
            end = ", ".join(
                f"{row['seed']}:{float(row['success_at_end']):.3f}" for row in vals
            )
            episodes = ", ".join(
                f"{row['seed']}:{int(row['num_episodes'])}" for row in vals
            )
            print(f"{method}/{task} once [{once}] end [{end}] episodes [{episodes}]")

    if incomplete:
        print()
        print("Incomplete labels")
        for label in incomplete:
            print(label)
    if missing:
        print()
        print("Missing summaries")
        for item in missing:
            print(item)


if __name__ == "__main__":
    main()
