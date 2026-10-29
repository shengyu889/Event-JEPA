import csv
import json
import re
from pathlib import Path

MODES = ["frozen", "ft10", "full", "stage2-only"]
SEQS = ["indoor_flying1", "indoor_flying2", "indoor_flying3"]

def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def get_sequence(text):
    m = re.search(r"indoor[_-]?flying[_-]?([123])", str(text).lower())
    return f"indoor_flying{m.group(1)}" if m else None

def get_mode(text):
    s = str(text).lower()
    for mode in MODES:
        if f"searaft_{mode}_" in s or f"searaft/{mode}_" in s:
            return mode
    return None

def get_metric(d, names):
    names = {norm(n) for n in names}
    for k, v in d.items():
        if norm(k) in names and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), str(k)
    return None, None

def walk(obj, seq_hint=None):
    if isinstance(obj, list):
        for v in obj:
            yield from walk(v, seq_hint)
        return

    if not isinstance(obj, dict):
        return

    seq = seq_hint

    for k in ("sequence", "dataset", "split", "name"):
        if k in obj and isinstance(obj[k], str):
            seq = get_sequence(obj[k]) or seq

    epe, _ = get_metric(
        obj, ["EPE", "mean_epe", "avg_epe", "endpoint_error"]
    )

    if epe is not None and seq in SEQS:
        outlier, out_key = get_metric(
            obj,
            ["outlier", "outlier_pct", "outlier_percent",
             "outlier_percentage", "outliers"]
        )
        pe1, _ = get_metric(obj, ["1PE", "1PE_pct"])
        pe2, _ = get_metric(obj, ["2PE", "2PE_pct"])
        pe3, _ = get_metric(obj, ["3PE", "3PE_pct"])

        yield {
            "sequence": seq,
            "EPE": epe,
            "Outlier_raw": outlier,
            "Outlier_key": out_key,
            "1PE_raw": pe1,
            "2PE_raw": pe2,
            "3PE_raw": pe3,
        }

    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            yield from walk(v, get_sequence(k) or seq)

found = {}

for root in [Path("runs"), Path("results")]:
    if not root.exists():
        continue

    for path in root.rglob("*.json"):
        if "searaft" not in str(path).lower():
            continue

        mode = get_mode(path)
        if mode is None:
            continue

        try:
            obj = json.loads(path.read_text())
        except (ValueError, OSError):
            continue

        seq_hint = get_sequence(path)

        for row in walk(obj, seq_hint):
            key = (mode, row["sequence"])

            # Prefer the most recently written report
            mtime = path.stat().st_mtime
            if key not in found or mtime > found[key][0]:
                found[key] = (mtime, row, str(path))

rows = []

for mode in MODES:
    for seq in SEQS:
        key = (mode, seq)

        if key in found:
            _, r, source = found[key]
            rows.append({
                "Protocol": mode,
                "Sequence": seq,
                "EPE": r["EPE"],
                "Outlier_raw": r["Outlier_raw"],
                "Outlier_key": r["Outlier_key"],
                "1PE_raw": r["1PE_raw"],
                "2PE_raw": r["2PE_raw"],
                "3PE_raw": r["3PE_raw"],
                "Source": source,
            })
        else:
            rows.append({
                "Protocol": mode,
                "Sequence": seq,
                "EPE": "",
                "Outlier_raw": "",
                "Outlier_key": "",
                "1PE_raw": "",
                "2PE_raw": "",
                "3PE_raw": "",
                "Source": "MISSING",
            })

out = Path("results/searaft_abcd_summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 94)
print("EVENT-JEPA × SEA-RAFT — MVSEC FLOW RESULTS")
print("=" * 94)
print(f"{'Protocol':<15} {'Sequence':<20} {'EPE':>12} {'Outlier(raw)':>16}")
print("-" * 94)

for r in rows:
    epe = f"{r['EPE']:.5f}" if r["EPE"] != "" else "MISSING"
    outlier = (
        f"{r['Outlier_raw']:.5f}"
        if r["Outlier_raw"] is not None and r["Outlier_raw"] != ""
        else "MISSING"
    )
    print(
        f"{r['Protocol']:<15} "
        f"{r['Sequence']:<20} "
        f"{epe:>12} "
        f"{outlier:>16}"
    )

n = len(found)
print("-" * 94)
print(f"Matched: {n}/12")
print(f"CSV saved: {out}")

if n < 12:
    print("WARNING: Some evaluation reports were not found.")
    print("Check evaluation logs, checkpoint paths, and JSON schema.")
else:
    print("All 12 protocol/sequence combinations found.")
    print("Verify Outlier units and evaluation masks before publication.")
