#!/usr/bin/env python3
# encoding: utf-8
"""
STACK experiment: torch 2.7.1+cu118 on the SAME seeds of the study.

WHAT IT DISENTANGLES
--------------------
The main study compared architectures with the stack fixed (2.9.1+cu128). That
leaves two things unseparated:

  - architecture and DRIVER are confounded (5080 570.133.07, 4090 580.173.02),
    and the driver cannot be pinned without touching the target machines;
  - we do not know whether breaking bit-exactness is due to the hardware or to
    any change in the compute path.

`2.7.1+cu118` is the only build that covers `sm_60` (P100) and `sm_75` (T4) at
the same time, so running it on both completes this table:

                     2.9.1+cu128      2.7.1+cu118
        T4 (sm_75)   750 done         <- this batch
        P100 (sm_60) impossible       <- this batch

  T4 2.9.1 vs T4 2.7.1    -> VERSION effect, hardware fixed
  T4 2.7.1 vs P100 2.7.1  -> ARCHITECTURE effect, stack fixed

Validated beforehand with an 8-trial probe: the initialization on CPU survives
the version change (8/8 identical hashes), so the control of the study still
holds and the seeds are directly comparable.

AUTOMATIC LABEL
---------------
TAG is derived from the detected GPU, not from a constant. This way the SAME
file works for the T4 batch and for the P100 one without editing it between
launches, which is what allows queueing the second one automatically when the
first one finishes.

LAUNCH
------
  T4   -> it has to be opened in the editor and `GPU T4 x2` chosen (the API
          does not allow choosing the accelerator; it always assigns P100).
  P100 -> it is launched just with `kernels push`, precisely because it lands
          on P100.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

TORCH_VERSION = "2.7.1"
CUDA_TAG = "cu118"
WORK = "/kaggle/working"
REPO = f"{WORK}/repo"

# FULL POOL of the four cheap conditions: 600 trials (200 Simple, 200 Hidden,
# 100 CNN_3L, 100 DNN_6L). This way the comparison against the 750 of
# `2.9.1+cu128` is over ALL the seeds of those conditions, not a subset, as in
# the main study.
#
# About 10.2 h for the 8 shards, run as two sessions of 5.1 h: shards 0-3 in
# one and shards 4-7 in the next (change SHARDS between launches).
SHARDS = ["0", "1", "2", "3"]
NSHARDS = 8

# COMPLETION of CNN_14L: up to 100 seeds, matching the n of the cu128 reference.
#
# Before this batch:  T4/cu118 = 80,  P100/cu118 = 96.  The pool has 100.
# With seeded resume, `limit=3` means up to 3 NEW trials per shard; a shard that
# is already complete simply finds nothing pending and finishes right away.
#
# TWO PHASES WITH DIFFERENT SHARDS: here one session covers the 8 shards, and
# Kaggle provides 4 vCPUs, so they are run 4 and 4 in consecutive phases
# instead of 8 competing processes.
#
# Cost: ~2.7 h per phase on T4; much less on P100, which is only missing 4.
PHASES = [
    dict(name="cnn14l-A", conditions=["CNN_14L"],
         shards=["0", "1", "2", "3"], limit=3, deadline=200),
    dict(name="cnn14l-B", conditions=["CNN_14L"],
         shards=["4", "5", "6", "7"], limit=3, deadline=200),
]

# GPU expected in THIS push.  "T4" | "P100" | None (any)
#
# It is needed because cu118 breaks an assumption we had been relying on
# without noticing: until now the run triggered by `kernels push` landed on
# P100 and aborted by itself in the first minute, because cu128 does not
# support sm_60. With cu118 it DOES start, so a push would amount to launching
# a 4 h P100 batch unintentionally -- which is exactly what happened in the
# first push of this kernel.
#
# With EXPECTED_GPU="T4" that harmless behaviour is restored: the push aborts
# in seconds and the batch only runs when someone launches it by hand with the
# right accelerator. For the P100 batch it is set to "P100" (or None) and then
# the push IS the launch.
EXPECTED_GPU = "P100"


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------- 1. code
section("1. Code and dataset")
if os.path.exists(REPO):
    shutil.rmtree(REPO)
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isdir(os.path.join(root, "src")) and "requirements.lock" in files:
        src = root
        break
if src is None:
    sys.exit("ERROR: cannot find the code under /kaggle/input")
shutil.copytree(src, REPO)
print(f"  {src} -> {REPO}")

# --------------------------------------------------------------- 2. fingerprints
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
    sys.exit("ERROR: the dataset does not match the reference")
if exp.get("training_fingerprint") and exp.get("dataset_fingerprint"):
    print("  -> they match (the stack changes ON PURPOSE; it is recorded in the `torch` field)")
else:
    print("  -> not checked: FINGERPRINTS.txt is absent or incomplete")

# ---------------------------------------------------------------- 3. stack
section(f"3. Stack torch {TORCH_VERSION}+{CUDA_TAG}")
t0 = time.time()
r = subprocess.run(f"{sys.executable} -m pip install -q torch=={TORCH_VERSION} torchvision "
                   f"--index-url https://download.pytorch.org/whl/{CUDA_TAG}",
                   shell=True, text=True)
print(f"  rc={r.returncode}  {time.time()-t0:.0f} s")
if r.returncode != 0:
    sys.exit("ERROR: the stack installation failed")

# ----------------------------------------------------------------- 4. GPU
section("4. Assigned GPU")
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
print("@@" + json.dumps({"gpus": out, "arch": torch.cuda.get_arch_list(),
                         "torch": torch.__version__}))
'''], capture_output=True, text=True)
line = [l for l in info.stdout.splitlines() if l.startswith("@@")]
if not line:
    print(info.stdout, info.stderr)
    sys.exit("ERROR: the assigned GPU cannot run this stack")
env_info = json.loads(line[0][2:])
for g in env_info["gpus"]:
    print(f"  device{g['idx']}: {g['name']}  {g['sm']}  {g['n_sm']} SMs")
print(f"  torch {env_info['torch']}")
n_gpu = len(env_info["gpus"])

# TAG derived from the GPU: the same file works for T4 and for P100.
# Without this it would have to be edited between launches, and it could not be queued.
gpu_name = env_info["gpus"][0]["name"]
short = "T4" if "T4" in gpu_name else ("P100" if "P100" in gpu_name else
                                       gpu_name.replace(" ", "-"))
TAG = f"kaggle-{short}-{CUDA_TAG}"
print(f"  -> automatic TAG: {TAG}")

if EXPECTED_GPU and short != EXPECTED_GPU:
    sys.exit(f"ABORT: this version expects {EXPECTED_GPU} and got {short}.\n"
             f"  If this is the automatic run of the push, this is correct: it stops\n"
             f"  in seconds instead of spending hours of quota on the wrong GPU.\n"
             f"  To really launch it, open the editor and choose GPU {EXPECTED_GPU}.")

# ------------------------------------------------- 4-bis. resume seeding
# `run_study.py` skips what it already finds in its shard file, but on Kaggle
# /kaggle/working starts empty. Without seeding, extending CNN_14L from 32 to 80
# would repeat the 32 already done (3.4 h wasted).
#
# The seeding goes in repo/resume/<TAG>/, with one subdirectory PER CELL: the
# same kernel runs on T4 and on P100, and seeding one with the trials of the other
# would treat results from OTHER hardware as done without repeating them. It is
# chosen by TAG, which in turn comes from the detected GPU.
OUT_DIR = f"{WORK}/output_data_gpucheck_{TAG}"
resume_dir = os.path.join(REPO, "resume", TAG)
if os.path.isdir(resume_dir):
    section("4-bis. Resume")
    import yaml as _yaml
    os.makedirs(OUT_DIR, exist_ok=True)
    resume_files = sorted(glob.glob(os.path.join(resume_dir, "gpupool_training_metrics_p*.yaml")))
    total = 0
    for f in resume_files:
        shutil.copy(f, OUT_DIR)
        total += len(_yaml.safe_load(open(f)) or {})
    print(f"  seeded {total} trials of {TAG}  ({len(resume_files)} files)")
    if len(resume_files) != NSHARDS:
        sys.exit(f"ERROR: {len(resume_files)} files but NSHARDS={NSHARDS}; "
                 f"the seeding was generated with another --nshards")
    print("  -> the processes will skip what is already done")
else:
    print(f"\n  (no repo/resume/{TAG}: everything the plan asks for will be trained)")

# ------------------------------------------------------------- 5. train
GPU_CHECK = os.path.join(REPO, "scripts", "gpu_check")
start = time.time()


def launch(name, conditions, limit, deadline, shards=None):
    """One phase: one process per shard, spread across the available GPUs.

    `shards` lets each phase cover a different group. It is needed when one
    session covers the 8 shards: Kaggle provides 4 vCPUs, so they are run 4
    and 4 in two consecutive phases instead of 8 processes at once.
    """
    shards = shards or SHARDS
    section(f"5. {name}  [{TAG}]  shards {' '.join(shards)}/{NSHARDS}")
    print(f"  conditions: {' '.join(conditions)}"
          + (f"   limit {limit} trials/process" if limit else "   (full pool)"))
    print(f"  deadline {deadline} min")
    procs = []
    for n, k in enumerate(shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(n % n_gpu),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        cmd = [sys.executable, "run_study.py", "--tag", TAG, "--shard", f"{k}/{NSHARDS}",
               "--out-base", WORK, "--deadline-min", str(deadline)]
        if limit:
            cmd += ["--limit", str(limit)]
        cmd += ["--conditions"] + conditions
        logf = open(f"{WORK}/stack_{name}_shard{k}.log", "w")
        procs.append((k, subprocess.Popen(cmd, cwd=GPU_CHECK, env=env,
                                          stdout=logf, stderr=subprocess.STDOUT), logf))
        print(f"  shard {k}/{NSHARDS} -> GPU {n % n_gpu}")
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(300)
        print(f"  [{(time.time()-start)/60:6.1f} min] {name}: "
              f"{sum(1 for _, p, _ in procs if p.poll() is None)} processes alive", flush=True)
    for k, p, logf in procs:
        logf.close()
        print(f"  shard {k}: rc={p.returncode}")


for f_ in PHASES:
    launch(**f_)

# ------------------------------------------------------------ 6. results
section("6. Results")
import yaml
out_dir = f"{WORK}/output_data_gpucheck_{TAG}"
m = {}
for f in sorted(glob.glob(f"{out_dir}/gpupool_training_metrics_p*.yaml")):
    m.update(yaml.safe_load(open(f)) or {})
import collections
c = collections.Counter(v["model"] for v in m.values())
print(f"  {len(m)} trials in {out_dir}")
for k in sorted(c):
    print(f"    {k:24} {c[k]}")

with open(f"{WORK}/run_meta.json", "w") as f:
    json.dump({"tag": TAG, "torch": env_info["torch"], "arch": env_info["arch"],
               "gpus": env_info["gpus"], "phases": PHASES, "shards": SHARDS, "nshards": NSHARDS, "trials": len(m),
               "training_fingerprint": got_t, "dataset_fingerprint": got_d,
               "minutes": round((time.time() - start) / 60, 1)}, f, indent=2)
print(f"\n  minutes: {(time.time()-start)/60:.1f}")
