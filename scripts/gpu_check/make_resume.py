#!/usr/bin/env python3
# encoding: utf-8
"""
Prepares RESUME files for a cell, with a given shard scheme.

Why it is needed
----------------
`run_study.py` already knows how to resume: at start-up it reads its shard file
and skips the trials that are already in it (`done = set(load_shard(shard_path))`).
On the lab machines that is enough, because the file persists between runs.

On Kaggle it is not: each session starts with an empty `/kaggle/working`, so
the runner believes nothing has been done and would repeat everything. With the
expensive conditions that does not fit in the quota (the full pool is ~33 h of
wall-clock time against the ~20 h that were actually left).

The solution is to seed the output directory with what has already been done,
BEFORE launching. This script generates those files with the shard scheme that
the session will use, which does not have to be the same one each trial was
generated with: the shard a trial belongs to is decided by `i % N` over the
seed index, and the index is recovered by looking up the seed in
`seed_pools.yaml`.

Usage:
    make_resume.py --dirs output_data_gpucheck_kaggle-T4-* --nshards 12 --out resume/
"""

import argparse
import collections
import glob
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
POOL_FILE = os.path.join(REPO, "seed_pools.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="results directories of the SAME cell")
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool_doc = yaml.safe_load(open(POOL_FILE))
    pools = pool_doc["pools"]
    # seed -> index, per model. This is what makes it possible to re-shard
    # results generated with another scheme.
    index_of = {m: {s: i for i, s in enumerate(seeds)} for m, seeds in pools.items()}

    data = {}
    for pat in args.dirs:
        for d in sorted(glob.glob(pat)):
            for f in sorted(glob.glob(os.path.join(d, "gpupool_training_metrics_p*.yaml"))):
                data.update(yaml.safe_load(open(f)) or {})
    if not data:
        sys.exit("ABORT: no trial found in --dirs")

    # Consistency: mixing different cells here would be catastrophic, because
    # the runner would treat trials from OTHER hardware as done and would not
    # repeat them.
    caps = {v.get("capability") for v in data.values()}
    cks = {v.get("pool_checksum") for v in data.values()}
    if len(caps) > 1:
        sys.exit(f"ABORT: the directories mix architectures {caps}. "
                 "Resuming is only valid within a single cell.")
    if len(cks) > 1:
        sys.exit(f"ABORT: different pools {cks}")

    os.makedirs(args.out, exist_ok=True)
    per_shard = collections.defaultdict(dict)
    no_index = 0
    for key, rec in data.items():
        m = rec["model"]
        i = index_of.get(m, {}).get(rec["seed"])
        if i is None:
            no_index += 1
            continue
        per_shard[i % args.nshards][key] = rec

    for k in range(args.nshards):
        path = os.path.join(args.out, f"gpupool_training_metrics_p{k}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(per_shard.get(k, {}), f, sort_keys=False)

    print(f"  cell       : {caps.pop()}  pool {cks.pop()}")
    print(f"  trials     : {len(data)}" + (f"  ({no_index} without index, skipped)" if no_index else ""))
    print(f"  scheme     : k/{args.nshards}")
    print(f"  written to : {args.out}")
    for k in range(args.nshards):
        n = len(per_shard.get(k, {}))
        c = collections.Counter(v["model"] for v in per_shard.get(k, {}).values())
        print(f"    p{k:<3} {n:4} trials   {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
