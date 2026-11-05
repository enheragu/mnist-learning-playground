#!/usr/bin/env python3
# encoding: utf-8
"""
Compares the JSON files produced by probe_ops.py and shows, operator by
operator, which configurations/machines match bit for bit and which do not.

Usage:
    python compare_probes.py                      # all probe_*.json files
    python compare_probes.py --ref 4090-defaults  # marks one as the reference
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "probe_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference tag (default: the first one)")
    ap.add_argument("--field", default="forward", choices=["forward", "grad_input"],
                    help="which hash to compare")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RES, "probe_*.json")))
    if not files:
        raise SystemExit(f"No probe_*.json files in {RES}")

    data = {}
    for f in files:
        d = json.load(open(f))
        data[d["tag"]] = d

    tags = list(data)
    ref = args.ref or tags[0]
    if ref not in data:
        raise SystemExit(f"Reference '{ref}' not found. Available: {tags}")
    others = [t for t in tags if t != ref]

    print(f"Reference: {ref}   (compared field: {args.field})")
    e = data[ref]["environment"]
    print(f"  {e['gpu_name']} {e['compute_capability']} {e['sm_count']}SMs "
          f"torch {e['torch']} cudnn {e['cudnn']}\n")

    ops = [k for k in data[ref]["ops"] if "error" not in data[ref]["ops"][k]]
    w = max(len(o) for o in ops) + 1
    header = f"{'operator':{w}}" + "".join(f"{t[:18]:>20}" for t in others)
    print(header)
    print("-" * len(header))

    diffs = {t: [] for t in others}
    for op in ops:
        row = f"{op:{w}}"
        for t in others:
            other_op = data[t]["ops"].get(op, {})
            if "error" in other_op or args.field not in other_op:
                row += f"{'n/a':>20}"
                continue
            same = other_op[args.field] == data[ref]["ops"][op][args.field]
            row += f"{'= same' if same else '! DIFFERENT':>20}"
            if not same:
                diffs[t].append(op)
        print(row)

    print()
    for t in others:
        ee = data[t]["environment"]
        flag_delta = {k: v for k, v in ee["flags"].items()
                      if data[ref]["environment"]["flags"].get(k) != v}
        n = len(diffs[t])
        status = "100% bit-exact" if n == 0 else f"{n}/{len(ops)} operators diverge"
        print(f"{t}: {status}")
        if flag_delta:
            print(f"   flags that change vs {ref}: {flag_delta}")
        if diffs[t]:
            print(f"   diverging: {', '.join(diffs[t])}")


if __name__ == "__main__":
    main()
