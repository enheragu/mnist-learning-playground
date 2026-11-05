"""Rows of the two example reporting tables of the discussion of the article, from the data deposit (export of
src/08_export_zenodo.py).

Usage: python3 discussion_tables.py <export_dir>
Groups: the YOLOv8 runs of the model comparison section, KAIST day (VTHS, HSVT) and LLVIP without equalization (VT,
VTHS). Every run of yolo_runs.csv is a valid run. In these groups the evaluation holds the person class only (checked:
all-class and person mAP are identical), so P and R are P_person and R_person.
Std. dev. is the sample one (ddof=1); P90 is the 90th percentile of the normal fitted with mean and sample std.
Prints the LaTeX rows of both tables, values in %.
"""
import os
import sys

import pandas as pd
from scipy.stats import norm

export_dir = sys.argv[1]
runs = pd.read_csv(os.path.join(export_dir, "yolo_runs.csv"))
groups = pd.read_csv(os.path.join(export_dir, "yolo_groups.csv")).set_index("group")

ROWS = [  # block, label in the table, group
    (0, "VTHS day (KAIST)", "kaist_day_vths_v2"),
    (0, "HSVT day (KAIST)", "kaist_day_hsvt"),
    (1, "VT (LLVIP)", "LLVIP_VT_no_equalization"),
    (1, "VTHS (LLVIP)", "LLVIP_VTHS_no_equalization"),
]
METRICS = [("P_person", "P"), ("R_person", "R"), ("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")]


def p90(v):
    return norm.ppf(0.9, loc=v.mean(), scale=v.std(ddof=1))


for metric in ("mAP50", "mAP50_95"):
    if (runs[metric] - runs[f"{metric}_person"]).abs().max() > 1e-12:
        raise SystemExit(f"{metric}: all-class and person values differ")

detail, compact = [], []
for block, label, group in ROWS:
    d = runs[runs["group"] == group]
    if len(d) != groups.loc[group, "n_runs"] or d["seed"].duplicated().any():
        raise SystemExit(f"{group}: number of runs or seeds do not match the deposit")
    v = 100 * d["mAP50"]
    detail.append((block, f"    {label} & {v.max():.3f} & {len(v)} & {v.mean():.3f} & {v.std(ddof=1):.3f} & "
                          f"{p90(v):.3f} \\\\"))
    compact.append((block, f"    {label} & " + " & ".join(f"{p90(100 * d[m]):.3f}" for m, _ in METRICS) + r" \\"))

for title, rows in (("Best, N, mean, std, P90 of mAP50", detail),
                    ("P90 of " + ", ".join(name for _, name in METRICS), compact)):
    print(f"% {title}")
    for i, (block, row) in enumerate(rows):
        if i and block != rows[i - 1][0]:
            print("    \\hline\n    \\hline")
        print(row)
    print()
