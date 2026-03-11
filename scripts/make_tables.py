#!/usr/bin/env python3
"""
scripts/make_tables.py

Generate paper-ready tables from project outputs.

Inputs (default):
  - results/metrics/metrics.csv
  - results/bench/latency.txt

Outputs:
  - results/tables/main_results.csv
  - results/tables/main_results.tex
  - results/tables/ablation_template.tex

Usage:
  python scripts/make_tables.py
  python scripts/make_tables.py --metrics results/metrics/metrics.csv --latency results/bench/latency.txt --out_dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


def read_latency_txt(path: Path) -> Dict[str, float | str]:
    """
    Parse latency.txt written by scripts/benchmark_latency.py.
    Expected lines like:
      device=cuda
      steps=8
      ms_per_iter=42.123
      fps=23.80
    """
    out: Dict[str, float | str] = {}
    if not path.exists():
        return out

    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k = k.strip()
        v = v.strip()
        # float if possible
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", v):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def load_metrics_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found: {path}")

    if pd is not None:
        df = pd.read_csv(path)
        return df

    # fallback: manual CSV
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def summarize_metrics(metrics) -> Dict[str, float]:
    """
    Returns mean values for columns present.
    Expected columns (from scripts/evaluate.py):
      teacher_flicker, student_flicker,
      teacher_sync, student_sync,
      teacher_idcos, student_idcos
    """
    if pd is not None and hasattr(metrics, "columns"):
        means = metrics.mean(numeric_only=True).to_dict()
        # ensure floats
        return {k: float(v) for k, v in means.items()}

    # fallback list[dict]
    cols = [k for k in metrics[0].keys() if k != "sample"]
    sums = {c: 0.0 for c in cols}
    n = 0
    for r in metrics:
        n += 1
        for c in cols:
            try:
                sums[c] += float(r[c])
            except Exception:
                pass
    return {c: (sums[c] / max(n, 1)) for c in cols}


def write_csv(path: Path, header: List[str], rows: List[List[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def latex_table_main(
    method_rows: List[Dict[str, str]],
    caption: str,
    label: str
) -> str:
    """
    Build a LaTeX table block for IEEE-style papers.
    """
    # Columns must match your paper format; adjust if needed.
    # Here: Method, Steps, ms/frame, FPS, Flicker, ID, Sync
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Steps & ms/frame$\downarrow$ & FPS$\uparrow$ & Flicker$\downarrow$ & ID$\uparrow$ & Sync$\uparrow$\\")
    lines.append(r"\midrule")
    for row in method_rows:
        lines.append(
            f"{row['method']} & {row['steps']} & {row['ms']} & {row['fps']} & {row['flicker']} & {row['id']} & {row['sync']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def latex_table_ablation_template(
    base_vals: Dict[str, str],
    caption: str,
    label: str
) -> str:
    """
    Create an ablation template.
    If you later run extra evals for each ablation variant, replace the placeholders.
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Flicker$\downarrow$ & ID$\uparrow$ & Sync$\uparrow$\\")
    lines.append(r"\midrule")
    lines.append(rf"Full TempoSyncDiff & {base_vals['flicker']} & {base_vals['id']} & {base_vals['sync']} \\")
    lines.append(r"w/o identity loss ($\mathcal{L}_{id}$) & \textit{run+fill} & \textit{run+fill} & \textit{run+fill} \\")
    lines.append(r"w/o temporal loss ($\mathcal{L}_{temp}$) & \textit{run+fill} & \textit{run+fill} & \textit{run+fill} \\")
    lines.append(r"w/o sync loss ($\mathcal{L}_{sync}$) & \textit{run+fill} & \textit{run+fill} & \textit{run+fill} \\")
    lines.append(r"w/o distillation (teacher/25 steps) & \textit{run+fill} & \textit{run+fill} & \textit{run+fill} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics/metrics.csv")
    ap.add_argument("--latency", default="results/bench/latency.txt")
    ap.add_argument("--out_dir", default="results/tables")
    ap.add_argument("--student_steps", type=int, default=None, help="Override steps shown for student row")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    latency_path = Path(args.latency)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics_csv(metrics_path)
    means = summarize_metrics(metrics)

    lat = read_latency_txt(latency_path)
    student_steps = args.student_steps
    if student_steps is None:
        # try parse from latency.txt
        if "steps" in lat:
            student_steps = int(lat["steps"])  # type: ignore
        else:
            student_steps = 8  # fallback

    # Extract summary values (these exist in your eval CSV)
    # teacher_* are from a 1-step refine in this scaffold
    t_flick = means.get("teacher_flicker", float("nan"))
    s_flick = means.get("student_flicker", float("nan"))
    t_sync = means.get("teacher_sync", float("nan"))
    s_sync = means.get("student_sync", float("nan"))
    t_id = means.get("teacher_idcos", float("nan"))
    s_id = means.get("student_idcos", float("nan"))

    # Latency fields (if present)
    ms = lat.get("ms_per_iter", float("nan"))
    fps = lat.get("fps", float("nan"))

    # Build table rows
    rows = [
        {
            "method": "Teacher (toy 1-step)",
            "steps": "50 (train) / 1 (eval)",
            "ms": "—",
            "fps": "—",
            "flicker": fmt(t_flick, 4) if t_flick == t_flick else "—",
            "id": fmt(t_id, 4) if t_id == t_id else "—",
            "sync": fmt(t_sync, 4) if t_sync == t_sync else "—",
        },
        {
            "method": "TempoSyncDiff (Student)",
            "steps": str(student_steps),
            "ms": fmt(float(ms), 3) if isinstance(ms, float) and ms == ms else "—",
            "fps": fmt(float(fps), 2) if isinstance(fps, float) and fps == fps else "—",
            "flicker": fmt(s_flick, 4) if s_flick == s_flick else "—",
            "id": fmt(s_id, 4) if s_id == s_id else "—",
            "sync": fmt(s_sync, 4) if s_sync == s_sync else "—",
        },
    ]

    # Write CSV
    csv_out = out_dir / "main_results.csv"
    header = ["Method", "Steps", "ms_per_frame", "FPS", "Flicker", "ID", "Sync"]
    csv_rows = [
        [r["method"], r["steps"], r["ms"], r["fps"], r["flicker"], r["id"], r["sync"]]
        for r in rows
    ]
    write_csv(csv_out, header, csv_rows)

    # Write LaTeX main table
    tex_out = out_dir / "main_results.tex"
    tex_out.write_text(
        latex_table_main(
            rows,
            caption="Main results produced from results/metrics/metrics.csv and results/bench/latency.txt (synthetic demo). Replace with real-dataset metrics for the paper.",
            label="tab:main_results_auto",
        ),
        encoding="utf-8",
    )

    # Ablation template (auto-fills the full-model row with student means)
    abla_out = out_dir / "ablation_template.tex"
    abla_out.write_text(
        latex_table_ablation_template(
            base_vals={
                "flicker": fmt(s_flick, 4) if s_flick == s_flick else "—",
                "id": fmt(s_id, 4) if s_id == s_id else "—",
                "sync": fmt(s_sync, 4) if s_sync == s_sync else "—",
            },
            caption="Ablation template. Full row is auto-filled from the current run; fill remaining rows by running ablated configs.",
            label="tab:ablation_auto",
        ),
        encoding="utf-8",
    )

    print("Wrote:")
    print(" -", csv_out)
    print(" -", tex_out)
    print(" -", abla_out)


if __name__ == "__main__":
    main()
