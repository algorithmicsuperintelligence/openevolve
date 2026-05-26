#!/usr/bin/env python3
"""Find CP-SAT runtime outliers and diagnose root cause.

Pipeline:
  1. Load all meta + features.
  2. Fit log-log baseline: log10(elapsed_ms) ~ a*log10(num_variables) + b*log10(num_constraints) + c
  3. Residual r = log10(actual) - log10(predicted). Large positive r = exponential blow-up.
  4. Top-K outliers: decode .cpsat.pb to extract constraint kinds, domain sizes,
     coefficient magnitudes, objective presence.
  5. Compare outliers vs rest (mean diff, log-fold).
  6. Write outliers_report.txt + outliers_top.csv + plots/07_residual_*.png.

Run:
  .venv/bin/python Statistics/analyze_outliers.py --top-k 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortools.sat import cp_model_pb2

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE.parent / "raw-data"
DEFAULT_OUT = HERE


# ---------- meta load ----------

def load_meta(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.meta.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def flatten(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        f = r.get("features") or {}
        st = r.get("cpsat_status") or {}
        sd = r.get("cpsat_response_stats") or {}
        ap = r.get("cpsat_applied_params") or {}
        out.append({
            "problem_sha256": r.get("problem_sha256"),
            "applied_params_hash": r.get("applied_params_hash"),
            "status": st.get("result"),
            "elapsed_ms": st.get("elapsed_ms"),
            "num_variables": f.get("num_variables"),
            "num_bool": f.get("num_bool"),
            "num_int": f.get("num_int"),
            "num_constraints": f.get("num_constraints"),
            "num_conflicts": sd.get("num_conflicts"),
            "num_branches": sd.get("num_branches"),
            "num_binary_propagations": sd.get("num_binary_propagations"),
            "num_integer_propagations": sd.get("num_integer_propagations"),
            "num_restarts": sd.get("num_restarts"),
            "deterministic_time": sd.get("deterministic_time"),
            "num_workers": ap.get("num_search_workers"),
            "pb_path": str(Path(r.get("problem_filename") or "")),
        })
    return out


# ---------- regression / residuals ----------

def fit_residuals(records: list[dict]) -> tuple[np.ndarray, dict]:
    """log10(elapsed_ms) ~ a*log10(vars) + b*log10(constraints) + c."""
    keep = []
    for r in records:
        v, c, t = r.get("num_variables"), r.get("num_constraints"), r.get("elapsed_ms")
        if not v or not c or not t or t <= 0:
            continue
        keep.append(r)
    X = np.array([[math.log10(r["num_variables"]), math.log10(r["num_constraints"]), 1.0] for r in keep])
    y = np.array([math.log10(r["elapsed_ms"]) for r in keep])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    pred = X @ coef
    resid = y - pred
    for r, p, rs in zip(keep, pred, resid):
        r["log10_elapsed"] = float(math.log10(r["elapsed_ms"]))
        r["log10_pred"] = float(p)
        r["residual"] = float(rs)  # log10 units; +1.0 = 10x slower than expected
    info = {
        "a_vars": float(a),
        "b_constraints": float(b),
        "intercept": float(c),
        "n": int(len(keep)),
        "r2": float(1.0 - np.var(resid) / np.var(y)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }
    return np.array(resid), info, keep


# ---------- pb features ----------

KIND_KEYS = (
    "linear",
    "bool_or",
    "bool_and",
    "at_most_one",
    "exactly_one",
    "bool_xor",
    "all_diff",
    "element",
    "circuit",
    "routes",
    "table",
    "automaton",
    "inverse",
    "reservoir",
    "interval",
    "no_overlap",
    "no_overlap_2d",
    "cumulative",
    "lin_max",
    "int_div",
    "int_mod",
    "int_prod",
)


def pb_features(pb_path: Path) -> dict:
    m = cp_model_pb2.CpModelProto()
    m.ParseFromString(pb_path.read_bytes())
    kinds = Counter()
    enforce_count = 0
    linear_coef_max = 0.0
    linear_coef_abs_sum = 0.0
    linear_terms_max = 0
    linear_terms_total = 0
    for c in m.constraints:
        kind = c.WhichOneof("constraint")
        kinds[kind] += 1
        if c.enforcement_literal:
            enforce_count += 1
        if kind == "linear":
            coefs = c.linear.coeffs
            if coefs:
                amax = max(abs(v) for v in coefs)
                if amax > linear_coef_max:
                    linear_coef_max = float(amax)
                linear_coef_abs_sum += float(sum(abs(v) for v in coefs))
                tlen = len(coefs)
                linear_terms_total += tlen
                if tlen > linear_terms_max:
                    linear_terms_max = tlen
    # variable domain stats
    dsizes = []
    wide_int_count = 0
    max_domain_max = 0
    min_domain_min = 0
    for v in m.variables:
        dom = v.domain
        sz = 0
        for i in range(0, len(dom), 2):
            sz += dom[i + 1] - dom[i] + 1
            if dom[i + 1] > max_domain_max:
                max_domain_max = int(dom[i + 1])
            if dom[i] < min_domain_min:
                min_domain_min = int(dom[i])
        dsizes.append(sz)
        if sz > 1000:
            wide_int_count += 1
    dsizes_arr = np.array(dsizes, dtype=float) if dsizes else np.array([0.0])
    has_obj = m.HasField("objective") or m.HasField("floating_point_objective")
    obj_terms = len(m.objective.vars) if m.HasField("objective") else 0
    num_assumptions = len(m.assumptions)
    num_search_hint = len(m.search_strategy)
    return {
        "n_vars": int(len(m.variables)),
        "n_cons": int(len(m.constraints)),
        "kinds": dict(kinds),
        "n_enforce": int(enforce_count),
        "linear_coef_max": linear_coef_max,
        "linear_coef_abs_sum": linear_coef_abs_sum,
        "linear_terms_max": int(linear_terms_max),
        "linear_terms_total": int(linear_terms_total),
        "domain_size_max": float(dsizes_arr.max()),
        "domain_size_median": float(np.median(dsizes_arr)),
        "wide_int_count": int(wide_int_count),
        "domain_min": int(min_domain_min),
        "domain_max": int(max_domain_max),
        "has_objective": bool(has_obj),
        "objective_terms": int(obj_terms),
        "n_assumptions": int(num_assumptions),
        "n_search_strategy": int(num_search_hint),
    }


# ---------- aggregation ----------

def summarize_kinds(kinds_list: list[dict]) -> dict[str, dict]:
    all_keys = set()
    for d in kinds_list:
        all_keys.update(d.keys())
    out = {}
    for k in sorted(all_keys):
        vals = np.array([d.get(k, 0) for d in kinds_list], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
            "share_nonzero": float((vals > 0).mean()),
        }
    return out


def fmt_block(title: str, lines: list[str]) -> str:
    return f"\n## {title}\n" + "\n".join(lines)


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=30, help="Number of outliers to deep-dive")
    ap.add_argument("--baseline-k", type=int, default=60, help="Inlier sample size for comparison")
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        print(f"ERROR not a directory: {args.raw_dir}", file=sys.stderr)
        return 1

    rows = load_meta(args.raw_dir)
    records = flatten(rows)
    print(f"loaded {len(records)} instances")

    _resid, info, kept = fit_residuals(records)
    print(f"baseline log10 fit: a_vars={info['a_vars']:.3f} b_cons={info['b_constraints']:.3f} "
          f"intercept={info['intercept']:.3f} R^2={info['r2']:.3f} rmse={info['rmse']:.3f}")

    # rank by residual (positive = slower than predicted)
    kept_sorted = sorted(kept, key=lambda r: -r["residual"])
    outliers = kept_sorted[: args.top_k]
    inliers = [r for r in kept_sorted if abs(r["residual"]) <= 0.3][-args.baseline_k:]
    if not inliers:
        inliers = kept_sorted[-args.baseline_k:]

    print(f"top-{args.top_k} outliers: residual range "
          f"{outliers[-1]['residual']:.3f} .. {outliers[0]['residual']:.3f} "
          f"(=> {10 ** outliers[-1]['residual']:.1f}x .. {10 ** outliers[0]['residual']:.1f}x slowdown)")

    # decode pb for outliers + baseline
    def decode_set(group, label):
        out = []
        for i, r in enumerate(group):
            pb = args.raw_dir / r["pb_path"]
            if not pb.exists():
                continue
            try:
                feats = pb_features(pb)
            except Exception as e:
                print(f"  WARN failed to decode {pb.name}: {e}", file=sys.stderr)
                continue
            feats["_residual"] = r["residual"]
            feats["_elapsed_ms"] = r["elapsed_ms"]
            feats["_num_variables"] = r["num_variables"]
            feats["_num_constraints"] = r["num_constraints"]
            feats["_status"] = r["status"]
            feats["_num_conflicts"] = r["num_conflicts"]
            feats["_num_branches"] = r["num_branches"]
            feats["_num_binary_propagations"] = r["num_binary_propagations"]
            feats["_num_integer_propagations"] = r["num_integer_propagations"]
            feats["_num_restarts"] = r["num_restarts"]
            feats["_applied_params_hash"] = r["applied_params_hash"]
            feats["_sha"] = r["problem_sha256"]
            out.append(feats)
            if (i + 1) % 10 == 0:
                print(f"  decoded {label} {i + 1}/{len(group)}")
        return out

    print("decoding outlier pb files...")
    out_feats = decode_set(outliers, "outliers")
    print("decoding baseline pb files...")
    base_feats = decode_set(inliers, "baseline")

    # ---- aggregate comparison ----
    def agg_scalar(group, key):
        arr = np.array([g[key] for g in group if g.get(key) is not None], dtype=float)
        return arr

    scalar_keys = [
        "n_vars", "n_cons", "n_enforce",
        "linear_coef_max", "linear_coef_abs_sum",
        "linear_terms_max", "linear_terms_total",
        "domain_size_max", "domain_size_median",
        "wide_int_count", "domain_max",
        "objective_terms", "n_assumptions", "n_search_strategy",
        "_num_conflicts", "_num_branches",
        "_num_binary_propagations", "_num_integer_propagations",
        "_num_restarts",
    ]
    print("\n## Outlier vs baseline scalar comparison (geomean, log-fold)")
    table_lines = [f"{'feature':28s} {'outlier_med':>14s} {'base_med':>14s} {'ratio':>8s}"]
    for k in scalar_keys:
        o = agg_scalar(out_feats, k)
        b = agg_scalar(base_feats, k)
        if o.size == 0 or b.size == 0:
            continue
        om = float(np.median(o))
        bm = float(np.median(b))
        ratio = om / bm if bm > 0 else float("inf") if om > 0 else 0.0
        table_lines.append(f"{k:28s} {om:>14,.2f} {bm:>14,.2f} {ratio:>8.2f}")
    table = "\n".join(table_lines)
    print(table)

    # ---- constraint kinds histograms ----
    out_kinds = summarize_kinds([g["kinds"] for g in out_feats])
    base_kinds = summarize_kinds([g["kinds"] for g in base_feats])
    all_keys = sorted(set(out_kinds) | set(base_kinds))
    kind_lines = [f"{'kind':18s} {'out_mean':>10s} {'base_mean':>10s} {'ratio':>8s} {'out_share':>10s}"]
    for k in all_keys:
        om = out_kinds.get(k, {}).get("mean", 0.0)
        bm = base_kinds.get(k, {}).get("mean", 0.0)
        ratio = om / bm if bm > 0 else float("inf") if om > 0 else 0.0
        sh = out_kinds.get(k, {}).get("share_nonzero", 0.0)
        kind_lines.append(f"{k:18s} {om:>10,.1f} {bm:>10,.1f} {ratio:>8.2f} {sh:>10.2f}")
    kind_table = "\n".join(kind_lines)
    print("\n## Constraint-kind histogram (outlier vs baseline)")
    print(kind_table)

    # ---- per-outlier detail csv ----
    csv_path = args.out / "outliers_top.csv"
    fields = [
        "_sha", "_status", "_applied_params_hash", "_residual",
        "_elapsed_ms", "_num_variables", "_num_constraints",
        "_num_conflicts", "_num_branches",
        "_num_binary_propagations", "_num_integer_propagations", "_num_restarts",
        "n_vars", "n_cons", "n_enforce",
        "linear_coef_max", "linear_terms_max", "linear_terms_total",
        "domain_size_max", "domain_size_median", "wide_int_count", "domain_max",
        "has_objective", "objective_terms", "n_assumptions", "n_search_strategy",
    ] + [f"kind_{k}" for k in KIND_KEYS]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for g in out_feats:
            row = []
            for fld in fields:
                if fld.startswith("kind_"):
                    row.append(g["kinds"].get(fld[5:], 0))
                else:
                    row.append(g.get(fld, ""))
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    # ---- residual scatter ----
    plot_dir = args.out / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    rec_arr = np.array([[r["num_variables"], r["num_constraints"], r["elapsed_ms"], r["residual"]] for r in kept])
    sc = ax.scatter(rec_arr[:, 0] * rec_arr[:, 1], rec_arr[:, 2], c=rec_arr[:, 3], cmap="coolwarm",
                    s=22, edgecolors="black", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("num_variables * num_constraints (log)")
    ax.set_ylabel("elapsed_ms (log)")
    ax.set_title(f"Runtime residual (log10) on size product\n"
                 f"a_vars={info['a_vars']:.2f} b_cons={info['b_constraints']:.2f} R²={info['r2']:.2f}")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("residual log10(actual/predicted)")
    for r in outliers[:8]:
        ax.annotate(r["problem_sha256"][:6], (r["num_variables"] * r["num_constraints"], r["elapsed_ms"]),
                    fontsize=7, color="darkred", xytext=(4, 2), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_residual_scatter.png", dpi=120)
    plt.close(fig)

    # constraint kind comparison bar chart
    if all_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        idx = np.arange(len(all_keys))
        w = 0.42
        out_vals = [out_kinds.get(k, {}).get("mean", 0.0) for k in all_keys]
        base_vals = [base_kinds.get(k, {}).get("mean", 0.0) for k in all_keys]
        ax.bar(idx - w / 2, out_vals, w, label=f"outliers (n={len(out_feats)})", color="#d62728")
        ax.bar(idx + w / 2, base_vals, w, label=f"baseline (n={len(base_feats)})", color="#1f77b4")
        ax.set_xticks(idx)
        ax.set_xticklabels(all_keys, rotation=25, ha="right")
        ax.set_ylabel("mean count per instance")
        ax.set_yscale("symlog")
        ax.set_title("Constraint-kind composition: outlier vs baseline")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "08_outlier_constraint_kinds.png", dpi=120)
        plt.close(fig)

    # ---- write text report ----
    report = [
        "# CP-SAT runtime outlier diagnosis",
        f"records: {len(records)}, regression-kept: {info['n']}",
        f"log10 baseline: log10(elapsed_ms) = {info['a_vars']:.3f}*log10(vars) "
        f"+ {info['b_constraints']:.3f}*log10(cons) + {info['intercept']:.3f}",
        f"R^2 = {info['r2']:.3f}, RMSE(log10) = {info['rmse']:.3f}",
        f"top-{args.top_k} outliers slowdown range: "
        f"{10 ** outliers[-1]['residual']:.1f}x .. {10 ** outliers[0]['residual']:.1f}x",
        "",
        "## Top outliers (residual = log10(actual/predicted))",
        f"{'sha':10s} {'hash':10s} {'status':10s} {'elapsed_ms':>12s} {'vars':>8s} {'cons':>8s} "
        f"{'pred_ms':>12s} {'resid':>7s} {'conflicts':>12s} {'branches':>14s}",
    ]
    for r in outliers:
        pred_ms = 10 ** r["log10_pred"]
        report.append(
            f"{r['problem_sha256'][:8]} {(r['applied_params_hash'] or '')[:8]} "
            f"{(r['status'] or '')[:10]:10s} {r['elapsed_ms']:>12,.1f} {r['num_variables']:>8d} "
            f"{r['num_constraints']:>8d} {pred_ms:>12,.1f} {r['residual']:>7.2f} "
            f"{(r['num_conflicts'] or 0):>12,d} {(r['num_branches'] or 0):>14,d}"
        )

    report += ["", "## Scalar feature comparison: outlier median vs baseline median", table]
    report += ["", "## Constraint kinds: outlier vs baseline mean per instance", kind_table]

    # interpretation hints
    report.append("\n## Interpretation hints")
    hints = []
    for k in scalar_keys:
        o = agg_scalar(out_feats, k)
        b = agg_scalar(base_feats, k)
        if o.size == 0 or b.size == 0:
            continue
        om, bm = float(np.median(o)), float(np.median(b))
        if bm <= 0 and om <= 0:
            continue
        ratio = om / bm if bm > 0 else float("inf")
        if ratio >= 2.0 or (ratio > 0 and ratio <= 0.5):
            hints.append(f"  - {k}: outlier median {om:,.1f} vs baseline {bm:,.1f}  ({ratio:.2f}x)")
    if hints:
        report.append("Features where outliers differ substantially (>=2x or <=0.5x):")
        report.extend(hints)
    else:
        report.append("No scalar feature showed a >=2x median shift; the blow-up is concentrated in solver-state stats (conflicts/branches), suggesting structural/search-space hardness rather than raw size.")

    (args.out / "outliers_report.txt").write_text("\n".join(report) + "\n")
    print(f"wrote {args.out / 'outliers_report.txt'}")
    print(f"wrote {plot_dir / '07_residual_scatter.png'}")
    print(f"wrote {plot_dir / '08_outlier_constraint_kinds.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
