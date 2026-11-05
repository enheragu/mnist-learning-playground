#!/usr/bin/env python3
# encoding: utf-8
"""
PRODUCTION kernel of the cross-GPU variance study.

Trains the part of the pool assigned to this session, using BOTH T4s
provided by the session (`GPU T4 x2`) instead of just one.

SPLIT
-----
The `sm_75` cell is covered over several sessions. The shards of each
session are set by the PHASES list, which is updated between sessions with
what is left pending. Always look at PHASES, not at this header.

Within a session, the 4 shards are split between the 2 GPUs (2 and 2) via
CUDA_VISIBLE_DEVICES. 4 processes are run, one per Kaggle vCPU.

DEADLINE
--------
Kaggle kills the session at 12 h. Each process receives --deadline-min with a
margin to stop cleanly: the results are written trial by trial atomically, and
whatever is left undone is a PREFIX of the pool (because of the seed-outside
order), so it is still comparable with the rest of the machines.

CONFIGURATION -- READ BEFORE CHANGING ANYTHING
---------------------------------------------
On Kaggle there is NO way to inject environment variables into a kernel,
neither through the API nor from the editor. The `os.environ.get(...)` calls
below ALWAYS fall back to the default value. That is: **the default value IS
the configuration**, and changing it requires editing this file and uploading
it again. The env vars only help if this script is run outside Kaggle.

The only thing still read from the environment is EEHA_TAG (with a default).
The rest of the configuration lives in the PHASES list below: each phase is a
batch of `run_study.py` with its own conditions, shards, limit and deadline,
and the phases run in order. This serves to chain, for example, a short check
before the long batch without spending two sessions.

SPLIT BY CONDITION, NOT BY SHARD
--------------------------------
On T4 the cost is very unbalanced (s/trial measured in production):

    CNN_14L      3088 x100 = 308,800 s     DNN_6L    584 x100 = 58,360 s
    CNN_14L_B10  4585 x 50 = 229,250 s     CNN_3L    345 x100 = 34,470 s
                                           Hidden    162 x200 = 32,340 s
                                           Simple    111 x200 = 22,240 s

(The first two are means over the trials already done on T4. The first
estimates, 2780 and 4005, came out 11-14% short and that is why the sessions
lasted longer than planned.)

Two conditions take 76% of the time. Splitting by shards, each session spends
three quarters of it on those two and barely progresses on the others: session
1 (shards 0-3 of 8, all conditions) stopped at 21 trials per condition.

Splitting by condition the numbers are much better:

    4 cheap,     FULL pool (600 trials)  -> 10.2 h  = 1 session
    2 expensive, full pool (150 trials)  -> 33.2 h  = 3 sessions

Restricting conditions does NOT change which seeds are used: the index `i` is
global and `seed = pools[m][i]`, so the trials are still the same ones that the
other machines run and are directly comparable. Verified with --dry-run.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

TORCH_VERSION = "2.9.1"
CUDA_TAG = "cu128"
WORK = "/kaggle/working"
REPO = f"{WORK}/repo"

TAG = os.environ.get("EEHA_TAG", "kaggle-T4-B")

# --- COMPLETION of the sm_75 cell: only the two expensive conditions are left -
# Coverage of sm_75 before this batch: 715/750. The 4 cheap conditions at 100%,
# CNN_14L 77/100 and CNN_14L_B10 38/50. 35 trials left, all in shards 4-7.
#
# The two expensive conditions are split into sessions of 4 shards over k/12,
# with seeded resume (see section 4-bis). With the measured times (3088 s per
# CNN_14L trial, 4585 s per CNN_14L_B10 trial), the 35 trials of shards 4-7
# take about 9 h, within the 12 h limit of a session.
#
# CHANGE `shards` at each launch. The k/12 scheme must match the --nshards
# that was passed to make_resume.py when generating repo/resume/.
PHASES = [
    dict(name="completion", tag=TAG,
         conditions=["CNN_14L", "CNN_14L_B10"],
         shards=["4", "5", "6", "7"], nshards=12, limit=None, deadline=630),
]

# How many trials repo/resume/ MUST contain. If it does not match, the seeding
# is from another batch and the kernel would repeat work already done (or,
# worse, would treat as done trials that are not). It aborts instead of relying
# on someone reading the log.
# None = no check.
EXPECTED_RESUME = 715


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


# ------------------------------------------------------------- 1. code
section("1. Code and dataset")

if os.path.exists(REPO):
    shutil.rmtree(REPO)

# Kaggle mounts at a variable depth and also extracts the zip flattening it, so
# the root is identified by its CONTENT, not by name or path.
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isdir(os.path.join(root, "src")) and "requirements.lock" in files:
        src = root
        break
if src is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        print("   ", root, files[:5])
    sys.exit("ERROR: cannot find the code under /kaggle/input")

shutil.copytree(src, REPO)
print(f"  {src} -> {REPO}")

# ------------------------------------------------------------- 2. fingerprints
section("2. Fingerprints")

TFP = ("{ find src/models -type f -name '*.py' 2>/dev/null; "
       "ls src/utils/set_seed.py requirements.lock "
       "scripts/gpu_check/run_study.py scripts/gpu_check/preloaded_loader.py 2>/dev/null; } "
       "| LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
DFP = "cd data/MNIST/raw && ls *-ubyte | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16"

def fp(cmd):
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()

exp = {}
if os.path.exists(f"{REPO}/FINGERPRINTS.txt"):
    for line in open(f"{REPO}/FINGERPRINTS.txt"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            exp[k] = v

got_t, got_d = fp(TFP), fp(DFP)
print(f"  training      : {got_t}  (expected {exp.get('training_fingerprint','?')})")
print(f"  dataset       : {got_d}  (expected {exp.get('dataset_fingerprint','?')})")
if exp.get("training_fingerprint") and got_t != exp["training_fingerprint"]:
    sys.exit("ERROR: the code does not match the study reference")
if exp.get("dataset_fingerprint") and got_d != exp["dataset_fingerprint"]:
    sys.exit("ERROR: the dataset does not match the study reference")
print("  -> they match")

# --------------------------------------------------------------- 3. stack
section(f"3. Stack torch {TORCH_VERSION}+{CUDA_TAG}")
t0 = time.time()
r = subprocess.run(f"{sys.executable} -m pip install -q torch=={TORCH_VERSION} torchvision "
                   f"--index-url https://download.pytorch.org/whl/{CUDA_TAG}",
                   shell=True, text=True)
print(f"  rc={r.returncode}  {time.time()-t0:.0f} s")
if r.returncode != 0:
    sys.exit("ERROR: the stack installation failed")

# --------------------------------------------------------------- 4. GPUs
section("4. Assigned GPUs")
info = subprocess.run([sys.executable, "-c", r'''
import json, sys, torch, torch.nn as nn
if not torch.cuda.is_available(): sys.exit("no CUDA")
out = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); c = torch.cuda.get_device_capability(i)
    out.append({"idx": i, "name": p.name, "sm": f"sm_{c[0]}{c[1]}", "n_sm": p.multi_processor_count})
try:
    x = torch.randn(256, 256, device="cuda"); float((x @ x).sum())
    nn.Conv2d(8, 16, 3).cuda()(torch.randn(2, 8, 16, 16, device="cuda"))
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"GPU cannot run: {type(e).__name__}: {e}")
print("@@" + json.dumps(out))
'''], capture_output=True, text=True)
line = [l for l in info.stdout.splitlines() if l.startswith("@@")]
if not line:
    print(info.stdout, info.stderr)
    sys.exit("ERROR: the assigned GPU cannot be used for this stack (P100?). Relaunch requesting T4.")
gpus = json.loads(line[0][2:])
for g in gpus:
    print(f"  device{g['idx']}: {g['name']}  {g['sm']}  {g['n_sm']} SMs")
n_gpu = len(gpus)

# ------------------------------------------------- 4-bis. resume seeding
# `run_study.py` skips the trials it already finds in its shard file. In the
# lab that file persists; on Kaggle each session starts with an empty
# /kaggle/working, so without this everything already done would be repeated
# (the full expensive pool is ~33 h of wall-clock time, against the ~20 h that
# were actually left: it would not fit in the quota).
#
# The files in repo/resume/ are generated by make_resume.py WITH THE SAME SHARD
# SCHEME that the session uses. If it does not match, the seeding is useless:
# each process only reads its own p<k>.yaml.
OUT_DIR = f"{WORK}/output_data_gpucheck_{TAG}"
resume_dir = os.path.join(REPO, "resume")
section("4-bis. Resume")

if not os.path.isdir(resume_dir):
    if EXPECTED_RESUME:
        sys.exit(f"ERROR: {EXPECTED_RESUME} seeded trials were expected and there is no "
                 f"repo/resume/ in the dataset. Regenerate it with make_resume.py and "
                 f"upload the dataset again, or set EXPECTED_RESUME=None.")
    print("  (no repo/resume/: everything the plan asks for will be trained)")
else:
    import yaml as _yaml
    os.makedirs(OUT_DIR, exist_ok=True)
    resume_files = sorted(glob.glob(os.path.join(resume_dir, "gpupool_training_metrics_p*.yaml")))
    total_seeded = 0
    for f in resume_files:
        shutil.copy(f, OUT_DIR)
        total_seeded += len(_yaml.safe_load(open(f)) or {})
    print(f"  seeded {total_seeded} trials in {OUT_DIR}  ({len(resume_files)} files)")

    # 1) The number of files has to be the N of the shard scheme. If
    #    make_resume.py was run with a different --nshards, each process would
    #    read a p<k>.yaml that does not belong to it and the seeding would be
    #    garbage.
    expected_n = {f["nshards"] for f in PHASES}
    if len(expected_n) == 1:
        n = expected_n.pop()
        if len(resume_files) != n:
            sys.exit(f"ERROR: the seeding has {len(resume_files)} files but the phases "
                     f"use k/{n}. It was generated with another --nshards; regenerate it.")

    # 2) The total has to match what was assumed done when planning.
    if EXPECTED_RESUME is not None and total_seeded != EXPECTED_RESUME:
        sys.exit(f"ERROR: seeding of {total_seeded} trials, expected "
                 f"{EXPECTED_RESUME}. The dataset carries a different version than "
                 f"planned: it would repeat work or treat as done trials that are "
                 f"not. Re-run make_resume.py and upload the dataset.")
    print("  -> seeding verified; the processes will skip what is already done")

# ------------------------------------------------------------ 5. launch
GPU_CHECK = os.path.join(REPO, "scripts", "gpu_check")
start = time.time()


def launch(name, tag, conditions, shards, nshards, limit, deadline):
    """Launches a phase: one process per shard, spread across the GPUs."""
    section(f"5. {name}  [{tag}]  shards {' '.join(shards)} of {nshards}")
    print(f"  {len(shards)} processes on {n_gpu} GPU(s), deadline {deadline:.0f} min")
    print(f"  conditions: {' '.join(conditions) if conditions else '(all)'}"
          + (f"   limit: {limit} trials/process" if limit else ""))

    procs = []
    for n, k in enumerate(shards):
        dev = n % n_gpu                  # alternating split across the GPUs
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str(dev),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        cmd = [sys.executable, "run_study.py", "--tag", tag, "--shard", f"{k}/{nshards}",
               "--out-base", WORK, "--deadline-min", str(deadline)]
        if conditions:
            cmd += ["--conditions"] + conditions
        if limit:
            cmd += ["--limit", str(limit)]
        logf = open(f"{WORK}/run_{name}_shard{k}.log", "w")
        p = subprocess.Popen(cmd, cwd=GPU_CHECK, env=env,
                             stdout=logf, stderr=subprocess.STDOUT)
        procs.append((k, dev, p, logf))
        print(f"  shard {k}/{nshards} -> GPU {dev}  (pid {p.pid})")

    # Periodic report: without this the Kaggle log stays silent for hours.
    while any(p.poll() is None for _, _, p, _ in procs):
        time.sleep(300)
        alive = sum(1 for _, _, p, _ in procs if p.poll() is None)
        print(f"  [{(time.time()-start)/60:6.1f} min] {name}: {alive} processes alive",
              flush=True)

    for k, dev, p, logf in procs:
        logf.close()
        print(f"  shard {k}: rc={p.returncode}")


for phase in PHASES:
    launch(**phase)

# ---------------------------------------------------------- 6. summary
section("6. Results")
out_dir = f"{WORK}/output_data_gpucheck_{TAG}"
total = 0
for f in sorted(glob.glob(f"{out_dir}/*.yaml")):
    import yaml
    n = len(yaml.safe_load(open(f)) or {})
    total += n
    print(f"  {os.path.basename(f)}: {n}")
print(f"\n  TOTAL: {total} trials in {out_dir}")

with open(f"{WORK}/run_meta.json", "w") as f:
    json.dump({"tag": TAG,
               "phases": [{k: v for k, v in f.items() if k != "tag"} for f in PHASES],
               "gpus": gpus,
               "training_fingerprint": got_t, "dataset_fingerprint": got_d,
               "total_trials": total, "minutes": round((time.time()-start)/60, 1)}, f, indent=2)
