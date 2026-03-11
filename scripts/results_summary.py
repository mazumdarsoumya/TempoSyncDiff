#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NUM_TYPES = (int, float)

def is_number(x: Any) -> bool:
    return isinstance(x, NUM_TYPES) and not isinstance(x, bool) and math.isfinite(float(x))

def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, key))
    elif isinstance(d, list):
        # Skip lists unless they are small numeric lists (rare for eval reports)
        # To avoid polluting tables with per-frame arrays.
        if len(d) <= 8 and all(is_number(x) for x in d):
            for i, v in enumerate(d):
                out[f"{prefix}[{i}]"] = float(v)
    else:
        if prefix and is_number(d):
            out[prefix] = float(d)
    return out

def try_load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def choose_best_json_in_run(run_dir: Path) -> Tuple[Optional[Path], Dict[str, float]]:
    """
    Prefer report-like JSON files with many numeric metrics.
    """
    candidates = []
    for p in run_dir.rglob("*.json"):
        if p.name.endswith("_torchload_verify.json"):
            continue
        obj = try_load_json(p)
        if obj is None:
            continue
        metrics = flatten_numeric(obj)
        if metrics:
            score = len(metrics)
            # Prefer names like report/eval/metrics
            name_bonus = 0
            lname = p.name.lower()
            if "report" in lname:
                name_bonus += 5
            if "eval" in lname:
                name_bonus += 3
            if "metric" in lname:
                name_bonus += 2
            candidates.append((score + name_bonus, p, metrics))
    if not candidates:
        return None, {}
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    _, p, m = candidates[0]
    return p, m

def extract_last_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of the last valid JSON object from a log.
    """
    # Find all positions of '{' and try from the end
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):
        snippet = text[s:].strip()
        # progressively trim lines from the end until parseable
        lines = snippet.splitlines()
        for end in range(len(lines), 0, -1):
            block = "\n".join(lines[:end]).strip()
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None

def parse_metrics_from_log(log_path: Path) -> Dict[str, float]:
    """
    Fallback parser if no JSON report exists.
    Parses:
      - JSON object printed in log (preferred)
      - key=value numeric patterns
      - key: value numeric patterns
    """
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return {}

    # 1) Try JSON block
    obj = extract_last_json_block(text)
    if isinstance(obj, dict):
        m = flatten_numeric(obj)
        if m:
            return m

    # 2) Regex key=value / key: value
    metrics: Dict[str, float] = {}
    # key=value
    for k, v in re.findall(r'([A-Za-z][A-Za-z0-9_.\-\/]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text):
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    # key: value
    for k, v in re.findall(r'([A-Za-z][A-Za-z0-9_.\-\/ ]{1,60})\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text):
        k2 = k.strip().replace(" ", "_")
        if k2 not in metrics:
            try:
                metrics[k2] = float(v)
            except Exception:
                pass
    return metrics

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Return {metric_key: {n, mean, std, min, max}}
    """
    all_keys = sorted({k for r in rows for k in r.keys() if k not in {"run_id", "run_dir", "source_file"}})
    out: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        vals = []
        for r in rows:
            v = r.get(key)
            if is_number(v):
                vals.append(float(v))
        if not vals:
            continue
        mean_v = statistics.mean(vals)
        std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = {
            "n": float(len(vals)),
            "mean": mean_v,
            "std": std_v,
            "min": min(vals),
            "max": max(vals),
        }
    return out

def write_per_run_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_summary_csv(summary: Dict[str, Dict[str, float]], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "std", "min", "max"])
        for k in sorted(summary.keys()):
            s = summary[k]
            w.writerow([k, int(s["n"]), s["mean"], s["std"], s["min"], s["max"]])

def latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("_", "\\_")
             .replace("%", "\\%")
             .replace("&", "\\&")
             .replace("#", "\\#")
             .replace("{", "\\{")
             .replace("}", "\\}"))

def write_summary_md(summary: Dict[str, Dict[str, float]], out_md: Path) -> None:
    lines = []
    lines.append("# Repeated Evaluation Summary (mean ± std)\n")
    lines.append("| Metric | n | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in sorted(summary.keys()):
        s = summary[k]
        lines.append(f"| `{k}` | {int(s['n'])} | {s['mean']:.6f} | {s['std']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    out_md.write_text("\n".join(lines) + "\n")

def write_summary_latex(summary: Dict[str, Dict[str, float]], out_tex: Path) -> None:
    lines = []
    lines.append("% Auto-generated by scripts/results_summary.py")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append("Metric & n & Mean & Std & Min & Max \\\\")
    lines.append("\\hline")
    for k in sorted(summary.keys()):
        s = summary[k]
        lines.append(f"{latex_escape(k)} & {int(s['n'])} & {s['mean']:.6f} & {s['std']:.6f} & {s['min']:.6f} & {s['max']:.6f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_tex.write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Aggregate repeated eval runs into mean/std summary.")
    ap.add_argument("--session_dir", required=True, help="e.g., results/terminal2_eval/live_eval_YYYY-MM-DD_HHMMSS")
    ap.add_argument("--out_dir", default=None, help="Output directory for summary files (default: <session_dir>/summary)")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (session_dir / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_runs_root = session_dir / "eval_runs"
    if not eval_runs_root.exists():
        print(f"[WARN] eval_runs directory not found: {eval_runs_root}")
        rows = []
    else:
        rows = []
        run_dirs = sorted([p for p in eval_runs_root.glob("run_*") if p.is_dir()],
                          key=lambda p: int(re.search(r'run_(\d+)$', p.name).group(1)) if re.search(r'run_(\d+)$', p.name) else p.name)
        for run_dir in run_dirs:
            row: Dict[str, Any] = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "source_file": "",
            }

            # 1) Prefer JSON report
            chosen_json, metrics = choose_best_json_in_run(run_dir)
            if chosen_json is not None and metrics:
                row["source_file"] = str(chosen_json)
                row.update(metrics)
                rows.append(row)
                continue

            # 2) Fallback to evaluate.log parse
            log_path = run_dir / "evaluate.log"
            if log_path.exists():
                metrics = parse_metrics_from_log(log_path)
                if metrics:
                    row["source_file"] = str(log_path)
                    row.update(metrics)
                    rows.append(row)
                    continue

            # 3) No metrics extracted
            row["source_file"] = "NO_METRICS_FOUND"
            rows.append(row)

    # Write per-run CSV
    per_run_csv = out_dir / "per_run_metrics.csv"
    write_per_run_csv(rows, per_run_csv)

    # Summarize numeric metrics
    summary = summarize(rows)
    summary_csv = out_dir / "aggregate_mean_std.csv"
    summary_json = out_dir / "aggregate_mean_std.json"
    summary_md = out_dir / "aggregate_mean_std.md"
    summary_tex = out_dir / "aggregate_mean_std.tex"

    write_summary_csv(summary, summary_csv)
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)
    write_summary_md(summary, summary_md)
    write_summary_latex(summary, summary_tex)

    # Console summary (prioritize common metrics if present)
    preferred = [
        "psnr_db", "ssim", "lpips", "mse",
        "vs_vae_recon.teacher.psnr_db", "vs_vae_recon.student.psnr_db",
        "vs_vae_recon.delta_psnr_student_minus_teacher_db",
        "temporal.temporal_l1.teacher", "temporal.temporal_l1.student",
        "temporal.flicker_std.teacher", "temporal.flicker_std.student",
        "val_loss", "val_acc"
    ]
    print(f"[OK] Session: {session_dir}")
    print(f"[OK] Parsed runs: {len(rows)}")
    print(f"[OK] Per-run CSV: {per_run_csv}")
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Summary MD : {summary_md}")
    print(f"[OK] Summary TeX: {summary_tex}")
    print("\nTop available metrics (mean ± std):")
    shown = set()
    for k in preferred:
        if k in summary:
            s = summary[k]
            print(f"  - {k}: {s['mean']:.6f} ± {s['std']:.6f} (n={int(s['n'])})")
            shown.add(k)
    # Show a few more if preferred list misses everything
    if not shown:
        for k in list(sorted(summary.keys()))[:20]:
            s = summary[k]
            print(f"  - {k}: {s['mean']:.6f} ± {s['std']:.6f} (n={int(s['n'])})")

if __name__ == "__main__":
    main()
