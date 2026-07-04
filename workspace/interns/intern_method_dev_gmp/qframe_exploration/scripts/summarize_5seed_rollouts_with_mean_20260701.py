from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance


BASE_RUNLOG = Path(
    "/mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/"
    "qframe_5seed_20260701"
)
SHELL_RUNLOG = Path(
    "/mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/"
    "longclip_shell_5seed_20260701"
)
OUT_PATH = BASE_RUNLOG / "summary_table_with_mean.md"

LABEL_RE = re.compile(
    r"^(?P<method>qframe_v1|longclip_image|longclip_fused)_"
    r"(?P<task>rc3|rc5|rc9|intercept|shell)_s(?P<seed>\d+)_5seed_20260701$"
)


def iter_started_labels(runlog: Path) -> list[str]:
    master = runlog / "master.log"
    labels: list[str] = []
    if not master.exists():
        return labels
    for line in master.read_text(errors="replace").splitlines():
        if line.startswith("[start]") and " label=" in line:
            labels.append(line.split(" label=", 1)[1].split(" ", 1)[0])
    return labels


def launcher_summary_path(runlog: Path, label: str) -> Path | None:
    log_path = runlog / f"{label}.launcher.log"
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        if "[eval]" in line and " summary=" in line:
            return Path(line.split(" summary=", 1)[1].split(" log=", 1)[0])
        if line.startswith("SUMMARY="):
            return Path(line.split("=", 1)[1])
    return None


def collect_rows() -> tuple[dict[tuple[str, str], list[dict[str, object]]], list[str]]:
    rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    incomplete: list[str] = []
    seen: set[str] = set()

    for runlog in (BASE_RUNLOG, SHELL_RUNLOG):
        for label in iter_started_labels(runlog):
            if label in seen:
                continue
            seen.add(label)
            match = LABEL_RE.match(label)
            if not match:
                continue
            summary_path = launcher_summary_path(runlog, label)
            if summary_path is None or not summary_path.exists():
                incomplete.append(label)
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
    return rows, incomplete


def fmt_mean_var(values: list[float]) -> str:
    if not values:
        return ""
    return f"{mean(values):.3f} / {pvariance(values):.5f}"


def task_values(
    rows: dict[tuple[str, str], list[dict[str, object]]],
    method: str,
    task: str,
    metric: str,
) -> list[float]:
    return [float(row[metric]) for row in rows[(method, task)]]


def method_mean_cell(
    rows: dict[tuple[str, str], list[dict[str, object]]],
    method: str,
    tasks: list[str],
    metric: str,
) -> str:
    task_means: list[float] = []
    missing = False
    for task in tasks:
        values = task_values(rows, method, task, metric)
        if not values:
            missing = True
            continue
        task_means.append(mean(values))
        if len(values) < 5:
            missing = True
    if not task_means:
        return ""
    suffix = "*" if missing else ""
    return f"{mean(task_means):.3f}{suffix}"


def render_metric_table(
    rows: dict[tuple[str, str], list[dict[str, object]]],
    metric: str,
    title: str,
) -> list[str]:
    methods = ["qframe_v1", "longclip_image", "longclip_fused"]
    tasks = ["rc3", "rc5", "rc9", "intercept", "shell"]
    labels = ["RC3", "RC5", "RC9", "Intercept", "Shell"]
    lines = [title]
    lines.append("| method | " + " | ".join(labels) + " | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        cells = [
            fmt_mean_var(task_values(rows, method, task, metric)) for task in tasks
        ]
        cells.append(method_mean_cell(rows, method, tasks, metric))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def render_per_seed(
    rows: dict[tuple[str, str], list[dict[str, object]]]
) -> list[str]:
    methods = ["qframe_v1", "longclip_image", "longclip_fused"]
    tasks = ["rc3", "rc5", "rc9", "intercept", "shell"]
    lines = ["Per-seed values"]
    for method in methods:
        for task in tasks:
            values = sorted(rows[(method, task)], key=lambda row: int(row["seed"]))
            if not values:
                continue
            once = ", ".join(
                f"{row['seed']}:{float(row['success_once']):.3f}" for row in values
            )
            end = ", ".join(
                f"{row['seed']}:{float(row['success_at_end']):.3f}"
                for row in values
            )
            episodes = ", ".join(
                f"{row['seed']}:{int(row['num_episodes'])}" for row in values
            )
            lines.append(
                f"{method}/{task} once [{once}] end [{end}] episodes [{episodes}]"
            )
    return lines


def main() -> None:
    rows, incomplete = collect_rows()
    sections: list[str] = []
    sections.extend(
        render_metric_table(rows, "success_once", "SUCCESS_ONCE mean / variance")
    )
    sections.append("")
    sections.extend(
        render_metric_table(rows, "success_at_end", "SUCCESS_AT_END mean / variance")
    )
    sections.append("")
    sections.extend(render_per_seed(rows))
    if incomplete:
        sections.append("")
        sections.append("Incomplete labels")
        sections.extend(incomplete)

    text = "\n".join(sections) + "\n"
    OUT_PATH.write_text(text)
    print(text, end="")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
