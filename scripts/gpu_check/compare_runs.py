#!/usr/bin/env python3
# encoding: utf-8
"""
Compares the results of several machines, seed by seed.

Initial version, aimed at calibration. It answers what needs to be known
before launching production:

  1. Do the INITIAL weights match across machines? If not, the initialization
     on CPU is not doing its job and everything else is moot: different
     starting points would be compared, not different arithmetic.
  2. Do the FINAL weights match? This is the study's `bit_exact`, reduced to a
     string comparison.
  3. How much does each condition really cost? It replaces the projections.

It only compares over the INTERSECTION of seeds present on all machines:
comparing different sets would give meaningless numbers.

Usage:
    python compare_runs.py <dir1> <dir2> [...]
    python compare_runs.py --glob 'output_data_gpucheck_cal-*'
"""

import argparse
import glob as globmod
import os
import statistics as st
import sys
from collections import defaultdict

import yaml


def load_dir(d):
    """Merges all the shards of a results directory."""
    merged = {}
    for f in sorted(globmod.glob(os.path.join(d, "gpupool_training_metrics_p*.yaml"))):
        merged.update(yaml.safe_load(open(f)) or {})
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--glob", default=None)
    args = ap.parse_args()

    dirs = args.dirs
    if args.glob:
        dirs = sorted(globmod.glob(args.glob))
    if len(dirs) < 1:
        sys.exit("Give at least one results directory")

    runs = {}
    for d in dirs:
        data = load_dir(d)
        if not data:
            print(f"  (empty: {d})")
            continue
        tag = next(iter(data.values())).get("tag", os.path.basename(d))
        runs[tag] = data

    if not runs:
        sys.exit("No results")

    print("=" * 78)
    print("MACHINES")
    print("=" * 78)
    for tag, data in runs.items():
        r = next(iter(data.values()))
        print(f"  {tag:22} {r.get('gpu','?'):28} {r.get('capability','?'):8} "
              f"{r.get('sm_count','?')} SMs  torch {r.get('torch','?')}  n={len(data)}")
        cks = {v.get("pool_checksum") for v in data.values()}
        if len(cks) > 1 or (cks and next(iter(cks)) is None):
            print(f"     WARNING: inconsistent pool checksums: {cks}")

    # Pool consistency across machines: if it differs, they are not comparable.
    all_cks = {v.get("pool_checksum") for data in runs.values() for v in data.values()}
    if len(all_cks) > 1:
        print(f"\n  ERROR: the machines use DIFFERENT pools {all_cks}. They are not comparable.")
        sys.exit(1)

    common = set.intersection(*(set(d) for d in runs.values()))
    print(f"\n  trial intersection: {len(common)}")

    # --------------------------------------------------------------- times
    print("\n" + "=" * 78)
    print("MEASURED COST PER CONDITION  (s/trial and epochs)")
    print("=" * 78)
    tags = list(runs)
    per_model = defaultdict(lambda: defaultdict(list))
    epochs = defaultdict(list)
    for key in common:
        m = runs[tags[0]][key]["model"]
        epochs[m].append(runs[tags[0]][key]["total_epochs"])
        for t in tags:
            per_model[m][t].append(runs[t][key]["train_duration"])

    hdr = f"  {'condition':24} {'epochs':>7}" + "".join(f"{t[:16]:>18}" for t in tags)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in sorted(per_model):
        ep = st.mean(epochs[m])
        row = f"  {m:24} {ep:7.1f}"
        for t in tags:
            row += f"{st.mean(per_model[m][t]):18.1f}"
        print(row)

    # ------------------------------------------------------ initial weights
    print("\n" + "=" * 78)
    print("INITIAL WEIGHTS  (must match: init on CPU, hardware independent)")
    print("=" * 78)
    init_bad = []
    for key in sorted(common):
        hs = {t: runs[t][key]["init_weights_sha256"] for t in tags}
        if len(set(hs.values())) > 1:
            init_bad.append((key, hs))
    if init_bad:
        print(f"  FAILS in {len(init_bad)}/{len(common)}:")
        for key, hs in init_bad[:6]:
            print(f"    {key}")
            for t, h in hs.items():
                print(f"      {t:22} {h}")
        print("\n  -> The initialization is NOT hardware independent.")
        print("     The study would compare different starting points. It has to")
        print("     be fixed before launching production.")
    else:
        print(f"  OK: identical in {len(common)}/{len(common)} trials, on all machines.")

    # -------------------------------------------------------- final weights
    print("\n" + "=" * 78)
    print("FINAL WEIGHTS  (bit_exact across machines)")
    print("=" * 78)
    by_model = defaultdict(lambda: {"n": 0, "equal": 0})
    for key in common:
        m = runs[tags[0]][key]["model"]
        hs = {runs[t][key]["final_weights_sha256"] for t in tags}
        by_model[m]["n"] += 1
        if len(hs) == 1:
            by_model[m]["equal"] += 1
    print(f"  {'condition':24} {'bit_exact':>12}")
    print("  " + "-" * 38)
    for m in sorted(by_model):
        v = by_model[m]
        pct = 100.0 * v["equal"] / v["n"] if v["n"] else 0
        print(f"  {m:24} {v['equal']:4}/{v['n']:<4} {pct:5.0f}%")

    # By pairs, which is what is actually reported.
    if len(tags) > 1:
        print(f"\n  {'pair':46} {'bit_exact':>12}")
        print("  " + "-" * 60)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                eq = sum(1 for k in common
                         if runs[a][k]["final_weights_sha256"] == runs[b][k]["final_weights_sha256"])
                print(f"  {a[:22]:23} vs {b[:22]:21} {eq:4}/{len(common):<4} "
                      f"{100.0*eq/len(common):5.0f}%")

    # --------------------------------------------------------- accuracy
    print("\n" + "=" * 78)
    print("ACCURACY  (mean over the intersection)")
    print("=" * 78)
    print(f"  {'condition':24}" + "".join(f"{t[:16]:>18}" for t in tags))
    print("  " + "-" * (24 + 18 * len(tags)))
    accs = defaultdict(lambda: defaultdict(list))
    for key in common:
        m = runs[tags[0]][key]["model"]
        for t in tags:
            accs[m][t].append(runs[t][key]["accuracy"])
    for m in sorted(accs):
        row = f"  {m:24}"
        for t in tags:
            row += f"{st.mean(accs[m][t]):18.4f}"
        print(row)


if __name__ == "__main__":
    main()
