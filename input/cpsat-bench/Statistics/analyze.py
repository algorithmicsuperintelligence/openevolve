#!/usr/bin/env python3
"""CP-SAT raw-data statistical analysis + visualization.

Scans ../raw-data/*.meta.jsonl and produces:
  - report.txt        : text report with distributions + correlations
  - per_instance.csv  : flat table for spreadsheet exploration
  - plots/*.png       : histograms, scatters, boxplots, heatmap

Metrics analyzed:
  - cpsat_status.elapsed_ms  (runtime)
  - features.num_variables / num_bool / num_int
  - features.num_constraints

Usage:
  .venv/bin/python Statistics/analyze.py
  .venv/bin/python Statistics/analyze.py --raw-dir <path> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE.parent / "raw-data"
DEFAULT_OUT = HERE


# ---------- loading ----------

def load_meta(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.meta.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARN bad json {path.name}: {e}", file=sys.stderr)
    return rows


def extract(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        feats = r.get("features") or {}
        status = r.get("cpsat_status") or {}
        applied = r.get("cpsat_applied_params") or {}
        stats = r.get("cpsat_response_stats") or {}
        out.append(
            {
                "problem_sha256": r.get("problem_sha256"),
                "applied_params_hash": r.get("applied_params_hash"),
                "status": status.get("result"),
                "elapsed_ms": status.get("elapsed_ms"),
                "num_variables": feats.get("num_variables"),
                "num_bool": feats.get("num_bool"),
                "num_int": feats.get("num_int"),
                "num_constraints": feats.get("num_constraints"),
                "num_workers": applied.get("num_search_workers"),
                "num_conflicts": stats.get("num_conflicts"),
                "num_branches": stats.get("num_branches"),
                "deterministic_time": stats.get("deterministic_time"),
            }
        )
    return out


# ---------- stats ----------

def summarize(values: Iterable, label: str) -> dict:
    arr = np.array(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=float,
    )
    if arr.size == 0:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.3f}"
    return str(v)


def table(summaries: list[dict]) -> str:
    cols = ["label", "count", "mean", "std", "min", "p25", "median", "p75", "p95", "max"]
    widths = {c: max(len(c), 10) for c in cols}
    for s in summaries:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(s.get(c, ""))))
    lines = []
    lines.append(" | ".join(c.rjust(widths[c]) for c in cols))
    lines.append("-+-".join("-" * widths[c] for c in cols))
    for s in summaries:
        lines.append(" | ".join(_fmt(s.get(c, "")).rjust(widths[c]) for c in cols))
    return "\n".join(lines)


def correlations(records: list[dict]) -> dict:
    def pairs(key: str):
        xs, ys = [], []
        for r in records:
            x, y = r.get(key), r.get("elapsed_ms")
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    out = {}
    for key in (
        "num_variables",
        "num_bool",
        "num_int",
        "num_constraints",
        "num_conflicts",
        "num_branches",
    ):
        x, y = pairs(key)
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            out[key] = {"n": int(x.size), "pearson": None, "spearman": None}
            continue
        pearson = float(np.corrcoef(x, y)[0, 1])
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(rx, ry)[0, 1])
        out[key] = {"n": int(x.size), "pearson": pearson, "spearman": spearman}
    return out


def status_breakdown(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.get("status") or "UNKNOWN"] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def per_hash_groups(records: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("applied_params_hash") or "UNKNOWN"].append(r)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def write_report(records: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# CP-SAT raw-data statistical report")
    lines.append(f"instances: {len(records)}")
    lines.append(
        f"unique problems: {len({r['problem_sha256'] for r in records if r.get('problem_sha256')})}"
    )
    lines.append(
        f"unique applied_params_hash: {len({r['applied_params_hash'] for r in records if r.get('applied_params_hash')})}"
    )

    lines.append("\n## Status breakdown")
    for k, v in status_breakdown(records).items():
        lines.append(f"  {k:12s} {v}")

    lines.append("\n## Overall distributions")
    metrics = [
        ("elapsed_ms", [r["elapsed_ms"] for r in records]),
        ("num_variables", [r["num_variables"] for r in records]),
        ("num_bool", [r["num_bool"] for r in records]),
        ("num_int", [r["num_int"] for r in records]),
        ("num_constraints", [r["num_constraints"] for r in records]),
    ]
    lines.append(table([summarize(vs, name) for name, vs in metrics]))

    lines.append("\n## Runtime correlations (Pearson / Spearman vs elapsed_ms)")
    corr = correlations(records)
    lines.append(f"{'feature':22s} {'n':>6s} {'pearson':>10s} {'spearman':>10s}")
    for k, v in corr.items():
        p = f"{v['pearson']:.4f}" if v["pearson"] is not None else "n/a"
        s = f"{v['spearman']:.4f}" if v["spearman"] is not None else "n/a"
        lines.append(f"{k:22s} {v['n']:>6d} {p:>10s} {s:>10s}")

    lines.append("\n## Per applied_params_hash")
    for h, recs in per_hash_groups(records):
        lines.append(f"\n### hash={h}  n={len(recs)}")
        sub = [
            ("elapsed_ms", [r["elapsed_ms"] for r in recs]),
            ("num_variables", [r["num_variables"] for r in recs]),
            ("num_constraints", [r["num_constraints"] for r in recs]),
        ]
        lines.append(table([summarize(vs, name) for name, vs in sub]))
        sb = status_breakdown(recs)
        lines.append("  status: " + ", ".join(f"{k}={v}" for k, v in sb.items()))

    # size-bucketed runtime
    arr = np.array([r["num_variables"] for r in records if r.get("num_variables") is not None], dtype=float)
    if arr.size:
        edges = np.percentile(arr, [0, 25, 50, 75, 100])
        lines.append("\n## Runtime by num_variables quartile")
        lines.append(f"edges: {[int(e) for e in edges]}")
        sums = []
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            bucket = []
            for r in records:
                v, ms = r.get("num_variables"), r.get("elapsed_ms")
                if v is None or ms is None:
                    continue
                in_range = lo <= v <= hi if i == 3 else lo <= v < hi
                if in_range:
                    bucket.append(ms)
            sums.append(summarize(bucket, f"Q{i + 1}[{int(lo)}..{int(hi)}]"))
        lines.append(table(sums))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"wrote {path}")


# ---------- viz ----------

STATUS_COLORS = {
    "OPTIMAL": "#1f77b4",
    "FEASIBLE": "#2ca02c",
    "INFEASIBLE": "#d62728",
    "UNKNOWN": "#7f7f7f",
    "MODEL_INVALID": "#ff7f0e",
}


def _color_for(status: str | None) -> str:
    return STATUS_COLORS.get(status or "UNKNOWN", "#9467bd")


def _safe_log_array(values, floor=1e-6):
    arr = np.array([v for v in values if v is not None and v > 0], dtype=float)
    arr[arr <= 0] = floor
    return arr


def plot_hist_runtime(records, out: Path) -> None:
    vals = _safe_log_array([r["elapsed_ms"] for r in records])
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(max(vals.min(), 1.0)), np.log10(vals.max()), 40)
    ax.hist(vals, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("elapsed_ms (log)")
    ax.set_ylabel("count")
    ax.set_title(f"Runtime distribution (n={vals.size})")
    ax.axvline(np.median(vals), color="red", linestyle="--", label=f"median={np.median(vals):.0f} ms")
    ax.axvline(np.mean(vals), color="orange", linestyle="--", label=f"mean={np.mean(vals):.0f} ms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_hist_features(records, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    spec = [
        ("num_variables", "#1f77b4"),
        ("num_bool", "#2ca02c"),
        ("num_constraints", "#d62728"),
    ]
    for ax, (key, color) in zip(axes, spec):
        vals = np.array(
            [r[key] for r in records if r.get(key) is not None and r[key] > 0],
            dtype=float,
        )
        if vals.size == 0:
            ax.set_title(f"{key} (empty)")
            continue
        bins = np.logspace(np.log10(max(vals.min(), 1.0)), np.log10(vals.max()), 30)
        ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(f"{key} (log)")
        ax.set_ylabel("count")
        ax.set_title(f"{key} (median={int(np.median(vals)):,})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_scatter_runtime(records, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key in zip(axes, ("num_variables", "num_constraints")):
        xs, ys, cs = [], [], []
        for r in records:
            x, y = r.get(key), r.get("elapsed_ms")
            if x is None or y is None or x <= 0 or y <= 0:
                continue
            xs.append(x)
            ys.append(y)
            cs.append(_color_for(r.get("status")))
        ax.scatter(xs, ys, c=cs, alpha=0.7, s=22, edgecolors="black", linewidths=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{key} (log)")
        ax.set_ylabel("elapsed_ms (log)")
        if len(xs) > 2:
            lx, ly = np.log(xs), np.log(ys)
            slope, intercept = np.polyfit(lx, ly, 1)
            xx = np.linspace(min(lx), max(lx), 50)
            ax.plot(np.exp(xx), np.exp(slope * xx + intercept), color="black", linestyle="--", linewidth=1,
                    label=f"log-fit slope={slope:.2f}")
            pearson = np.corrcoef(lx, ly)[0, 1]
            ax.set_title(f"elapsed_ms vs {key} (log-log r={pearson:.3f})")
            ax.legend(loc="upper left", fontsize=9)
        else:
            ax.set_title(f"elapsed_ms vs {key}")

    # status legend
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=k)
               for k, c in STATUS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(STATUS_COLORS), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_box_by_group(records, key: str, title: str, out: Path) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("elapsed_ms") and r["elapsed_ms"] > 0:
            groups[str(r.get(key) or "UNKNOWN")].append(r["elapsed_ms"])
    if not groups:
        return
    items = sorted(groups.items(), key=lambda kv: -np.median(kv[1]))
    fig, ax = plt.subplots(figsize=(max(7, 1.0 + 0.9 * len(items)), 5))
    data = [v for _, v in items]
    labels = [k for k, _ in items]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    ax.set_yscale("log")
    ax.set_ylabel("elapsed_ms (log)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for i, (label, vals) in enumerate(items, start=1):
        ax.text(i, np.median(vals), f"n={len(vals)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_heatmap_size(records, out: Path) -> None:
    xs, ys, ms = [], [], []
    for r in records:
        v, c, t = r.get("num_variables"), r.get("num_constraints"), r.get("elapsed_ms")
        if not v or not c or not t or v <= 0 or c <= 0 or t <= 0:
            continue
        xs.append(v)
        ys.append(c)
        ms.append(t)
    if len(xs) < 5:
        return
    xs, ys, ms = np.array(xs), np.array(ys), np.array(ms)
    x_edges = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 16)
    y_edges = np.logspace(np.log10(ys.min()), np.log10(ys.max()), 16)
    # mean elapsed per cell
    H_sum, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=ms)
    H_cnt, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        H_mean = np.where(H_cnt > 0, H_sum / H_cnt, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(x_edges, y_edges, H_mean.T, norm=LogNorm(vmin=max(np.nanmin(H_mean[H_mean > 0]), 1.0),
                                                                  vmax=np.nanmax(H_mean)),
                          cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("num_variables (log)")
    ax.set_ylabel("num_constraints (log)")
    ax.set_title("Mean elapsed_ms across size grid")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("mean elapsed_ms (log)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_plots(records: list[dict], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_hist_runtime(records, plot_dir / "01_runtime_hist.png")
    plot_hist_features(records, plot_dir / "02_features_hist.png")
    plot_scatter_runtime(records, plot_dir / "03_runtime_vs_size_scatter.png")
    plot_box_by_group(records, "status", "Runtime by cpsat_status.result", plot_dir / "04_box_by_status.png")
    plot_box_by_group(records, "applied_params_hash", "Runtime by applied_params_hash", plot_dir / "05_box_by_hash.png")
    plot_heatmap_size(records, plot_dir / "06_heatmap_vars_constraints.png")
    print(f"wrote {len(list(plot_dir.glob('*.png')))} plots to {plot_dir}")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        print(f"ERROR not a directory: {args.raw_dir}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_meta(args.raw_dir)
    records = extract(rows)
    print(f"loaded {len(records)} instances from {args.raw_dir}")

    write_report(records, args.out / "report.txt")
    write_csv(records, args.out / "per_instance.csv")

    if not args.no_plots:
        make_plots(records, args.out / "plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
