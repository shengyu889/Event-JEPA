#!/usr/bin/env python3
"""Evaluate and summarize Event-JEPA SEA-RAFT v8/v8.2 experiments.

Run from the Event-JEPA project root:
  python scripts/report_searaft_v8.py                  # report already-evaluated runs
  CUDA_VISIBLE_DEVICES=6 python scripts/report_searaft_v8.py --eval-missing

Only reads existing training checkpoints; never trains or rewrites them.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

TEST_SEQS = ("indoor_flying1", "indoor_flying2", "indoor_flying3")
EXPECTED_SAMPLES = (2190, 2649, 2936)

# (label, training dir, evaluation dir, code, fusion, reference, protocol)
RUNS = [
    ("Raw SEA-RAFT (pretrained)", None,
     "searaft_v8_original_pretrained_eval_all_seed0", "v8", "none", None, "raw"),
    ("Raw + GEP (fusion frozen)", "searaft_v8_2_rawbase_gep_full_seed0",
     "searaft_v8_2_rawbase_gep_full_seed0_eval", "v82", "gep", "raw", "raw"),
    ("Raw + JEPA (fusion frozen)", "searaft_v8_2_rawbase_jepa_full_seed0",
     "searaft_v8_2_rawbase_jepa_full_seed0_eval", "v82", "jepa", "raw", "raw"),
    ("Full FT LR=2e-6 BS32", "searaft_v8_fullbase_lr2e6_bs32_seed0",
     "searaft_v8_fullbase_lr2e6_bs32_eval", "v8", "none", "raw", "raw"),
    ("Full FT LR=2e-5 BS8", "searaft_v8_fullbase_seed0",
     "searaft_v8_fullbase_eval_seed0", "v8", "none", None, "old"),
    ("Continued (old base)", "searaft_v8_fullbase_continued_seed0",
     "searaft_v8_fullbase_continued_seed0_eval", "v8", "none", "oldbase", "old"),
    ("GEP (old base)", "searaft_v8_gep_fusion_seed0",
     "searaft_v8_gep_fusion_seed0_eval", "v8", "gep", "continued", "old"),
    ("JEPA (old base)", "searaft_v8_jepa_fusion_seed0",
     "searaft_v8_jepa_fusion_seed0_eval", "v8", "jepa", "continued", "old"),
]

def valid_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        obj = json.loads(path.read_text())
        r = obj["results"]
        return all(
            isinstance(r[s]["EPE"], (float, int))
            and math.isfinite(r[s]["EPE"])
            and int(r[s].get("eval_samples", -1)) == expected
            for s, expected in zip(TEST_SEQS, EXPECTED_SAMPLES)
        )
    except (ValueError, KeyError, TypeError, OverflowError):
        return False

def evaluate_missing(root: Path, gpu: str | None, workers: int) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for label, train_name, eval_name, code, fusion, _, _ in RUNS:
        eval_path = root / "runs" / eval_name
        metrics = eval_path / "metrics_final.json"
        if valid_metrics(metrics):
            print(f"[SKIP] {label}: full evaluation already exists", flush=True)
            continue
        checkpoint = root / "runs" / train_name / "last.pt" if train_name else None
        if checkpoint and not checkpoint.is_file():
            print(f"[MISSING CHECKPOINT] {label}: {checkpoint}", flush=True)
            continue
        script = root / "scripts" / (
            "sea_raft_fullbackbone_v8_2.py" if code == "v82" else "sea_raft_fullbackbone_v8.py"
        )
        if not script.is_file():
            print(f"[MISSING SCRIPT] {script}", flush=True)
            continue
        eval_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(script),
            "--mode", "eval-trained" if checkpoint else "eval-pretrained",
            "--fusion", fusion,
            "--root", "datasets/MVSEC",
            "--output-dir", str(eval_path.relative_to(root)),
            "--workers", str(workers),
        ]
        if checkpoint:
            cmd += ["--resume", str(checkpoint.relative_to(root))]
        env["PYTHONPATH"] = os.pathsep.join([
            str(root / "dinov2"), str(root / "src"), str(root), env.get("PYTHONPATH", "")
        ])
        log_file = log_dir / f"{eval_name}.log"
        print(f"\n[EVALUATING] {label}\n$ {' '.join(cmd)}", flush=True)
        print(f"[LOG] {log_file}", flush=True)
        with log_file.open("w") as log:
            proc = subprocess.Popen(
                cmd, cwd=root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if any(tag in line for tag in ("[EVAL]", "[RESULTS]", "Traceback", "Error", "RuntimeError")):
                    print(line, end="", flush=True)
            status = proc.wait()
        if status:
            print(f"[ERROR] {label}: exit={status}, see {log_file}", flush=True)
        elif not valid_metrics(metrics):
            print(f"[WARNING] {label}: metrics missing/incomplete after eval", flush=True)

def read_results(root: Path):
    rows = []
    for label, train_name, eval_name, _, fusion, ref, group in RUNS:
        metrics = root / "runs" / eval_name / "metrics_final.json"
        ckpt = root / "runs" / train_name / "last.pt" if train_name else None
        training = root / "runs" / train_name / "train_history.json" if train_name else None
        entry = {"group": group, "method": label, "fusion": fusion, "reference": ref,
                 "checkpoint": str(ckpt) if ckpt else "original pretrained weights",
                 "metrics": str(metrics), "status": "NOT_EVALUATED"}
        if valid_metrics(metrics):
            obj = json.loads(metrics.read_text())
            epe = [float(obj["results"][s]["EPE"]) for s in TEST_SEQS]
            out = [float(obj["results"][s]["Outlier%"]) for s in TEST_SEQS]
            cos = [float(obj["results"][s]["mean_cosine_nonzero"]) for s in TEST_SEQS]
            entry.update({"status": "OK", "EPE_1": epe[0], "EPE_2": epe[1], "EPE_3": epe[2],
                          "EPE_macro": sum(epe) / 3, "Outlier_macro_pct": sum(out) / 3,
                          "Cosine_macro": sum(cos) / 3,
                          "samples_total": sum(int(obj["results"][s]["eval_samples"]) for s in TEST_SEQS)})
        if training and training.is_file():
            try:
                hist = json.loads(training.read_text())
                if hist:
                    entry["train_final_epoch"] = hist[-1].get("epoch")
                    entry["train_final_loss"] = hist[-1].get("train_loss")
                    entry["train_online_EPE"] = hist[-1].get("train_event_valid_epe")
            except (KeyError, TypeError, ValueError):
                pass
        rows.append(entry)
    return rows

def output_report(root: Path, rows: list[dict]) -> None:
    raw = next((r for r in rows if r["method"] == "Raw SEA-RAFT (pretrained)" and r["status"] == "OK"), None)
    old = next((r for r in rows if r["method"] == "Full FT LR=2e-5 BS8" and r["status"] == "OK"), None)
    continued = next((r for r in rows if r["method"] == "Continued (old base)" and r["status"] == "OK"), None)
    refs = {"raw": raw, "oldbase": old, "continued": continued}
    for r in rows:
        ref = refs.get(r.get("reference"))
        if r["status"] == "OK" and ref is not None:
            r["relative_epe_change_pct"] = 100 * (r["EPE_macro"] / ref["EPE_macro"] - 1)
        else:
            r["relative_epe_change_pct"] = None
    print("\n" + "=" * 117)
    print("MVSEC v6 | complete 7775 test samples | EPE lower is better; cosine higher is better")
    print("=" * 117)
    print(f"{'METHOD':34} {'F1 EPE':>9} {'F2 EPE':>9} {'F3 EPE':>9} {'MACRO':>9} {'OUT%':>9} {'COSINE':>9} {'Δref%':>9}")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['method']:34} -- NOT EVALUATED --")
            continue
        dr = r["relative_epe_change_pct"]
        drs = f"{dr:+.2f}" if dr is not None else "---"
        print(f"{r['method']:34} {r['EPE_1']:9.4f} {r['EPE_2']:9.4f} {r['EPE_3']:9.4f} "
              f"{r['EPE_macro']:9.4f} {r['Outlier_macro_pct']:9.3f} {r['Cosine_macro']:9.3f} {drs:>9}")
    print("\nΔref%: Raw+GEP/JEPA and LR2e-6 compared with raw pretrained; ")
    print("  Continued compared with LR2e-5 full FT; old-base GEP/JEPA compared with Continued.")
    print("  Different groups/trainability regimes should not be treated as a single controlled ranking.")
    print("  Training EPE (if present) is an online training diagnostic, NOT held-out test EPE.")
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / "searaft_v8_all_comparison.csv"
    json_path = result_dir / "searaft_v8_all_comparison.json"
    keys = ("group", "method", "status", "EPE_1", "EPE_2", "EPE_3", "EPE_macro",
            "Outlier_macro_pct", "Cosine_macro", "relative_epe_change_pct", "samples_total",
            "train_final_epoch", "train_final_loss", "train_online_EPE", "checkpoint", "metrics")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-missing", action="store_true", help="Evaluate only runs lacking COMPLETE three-sequence full-test metrics")
    ap.add_argument("--gpu", default=None, help="Physical GPU ID e.g. 6, overriding CUDA_VISIBLE_DEVICES")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.project_root.resolve()
    if args.eval_missing:
        evaluate_missing(root, args.gpu, args.workers)
    output_report(root, read_results(root))

if __name__ == "__main__":
    main()
