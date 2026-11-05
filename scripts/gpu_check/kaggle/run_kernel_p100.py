#!/usr/bin/env python3
# encoding: utf-8
"""
PROBE of P100 + cu118 stack. Short and cheap: it does not train a set, it verifies.

WHY A PROBE AND NOT A BATCH
---------------------------
Before spending quota on P100 three things have to be answered, and the third
one could bring down the whole design:

  1. Does torch 2.7.1+cu118 start on a P100 (`sm_60`)?  The main stack
     (2.9.1+cu128) does NOT support it, which is why another one is needed.
  2. How much does a trial cost there? There is no measurement yet, and the
     P100 is a single GPU on Kaggle (versus the two T4s), so the contention is
     different.
  3. **Does the initialization on CPU survive the torch version change?**
     The whole control of the study rests on the INITIAL weights matching
     across cells. If torch 2.7.1 generates different ones, the comparison
     across versions needs a different design -- and it is better to find out
     with 8 trials, not after a whole session.

It is launched THROUGH THE API on purpose: the push always lands on P100, which
is exactly what matters here. It does not require anyone to touch the editor.

The fingerprints are still verified: `requirements.lock` does not change (the
kernel installs from TORCH_VERSION, not from the lock), so the code and the
dataset are the same as in the rest of the study. The version actually
installed is recorded in the `torch` field of each trial, which is the source
of truth.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

TORCH_VERSION = "2.7.1"
CUDA_TAG = "cu118"          # the only one that covers sm_60 (P100) and sm_75 (T4) at the same time
WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
TAG = os.environ.get("EEHA_TAG", "kaggle-P100-cu118")

# Only the two cheapest models and 1 seed index per process: the probe
# measures and verifies, it does not accumulate statistics.
PHASE = dict(conditions=["SimplePerceptron", "HiddenLayerPerceptron"],
            shards=["0", "1", "2", "3"], nshards=12, limit=2, deadline=30)


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
fp = lambda c: subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()
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
print("  -> they match (the stack changes on purpose; it is recorded in the `torch` field)")

# ---------------------------------------------------------------- 3. stack
section(f"3. Stack torch {TORCH_VERSION}+{CUDA_TAG}")
t0 = time.time()
r = subprocess.run(f"{sys.executable} -m pip install -q torch=={TORCH_VERSION} torchvision "
                   f"--index-url https://download.pytorch.org/whl/{CUDA_TAG}",
                   shell=True, text=True)
print(f"  rc={r.returncode}  {time.time()-t0:.0f} s")
if r.returncode != 0:
    sys.exit("ERROR: the cu118 stack installation failed")

# ----------------------------------------------------------------- 4. GPU
section("4. Assigned GPU")
info = subprocess.run([sys.executable, "-c", r'''
import json, sys, torch, torch.nn as nn
if not torch.cuda.is_available(): sys.exit("no CUDA")
arch = torch.cuda.get_arch_list()
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
print("@@" + json.dumps({"gpus": out, "arch": arch, "torch": torch.__version__}))
'''], capture_output=True, text=True)
line = [l for l in info.stdout.splitlines() if l.startswith("@@")]
if not line:
    print(info.stdout, info.stderr)
    sys.exit("ERROR: the assigned GPU cannot run this stack")
env_info = json.loads(line[0][2:])
for g in env_info["gpus"]:
    print(f"  device{g['idx']}: {g['name']}  {g['sm']}  {g['n_sm']} SMs")
print(f"  torch {env_info['torch']}   arch list: {env_info['arch']}")
n_gpu = len(env_info["gpus"])
subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader",
               shell=True, text=True)

# ------------------------------------------------------------- 5. train
section(f"5. Probe  [{TAG}]  {PHASE['conditions']}  shards {PHASE['shards']}/{PHASE['nshards']}")
GPU_CHECK = os.path.join(REPO, "scripts", "gpu_check")
procs = []
start = time.time()
for n, k in enumerate(PHASE["shards"]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(n % n_gpu),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd = [sys.executable, "run_study.py", "--tag", TAG, "--shard", f"{k}/{PHASE['nshards']}",
           "--out-base", WORK, "--deadline-min", str(PHASE["deadline"]),
           "--limit", str(PHASE["limit"]), "--conditions"] + PHASE["conditions"]
    logf = open(f"{WORK}/probe_shard{k}.log", "w")
    procs.append((k, subprocess.Popen(cmd, cwd=GPU_CHECK, env=env,
                                      stdout=logf, stderr=subprocess.STDOUT), logf))
    print(f"  shard {k} -> GPU {n % n_gpu}")
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(120)
    print(f"  [{(time.time()-start)/60:5.1f} min] "
          f"{sum(1 for _,p,_ in procs if p.poll() is None)} alive", flush=True)
for k, p, logf in procs:
    logf.close()
    print(f"  shard {k}: rc={p.returncode}")

# ------------------------------------------------ 6. THE important question
section("6. Does the initialization on CPU survive the version change?")
import yaml
out_dir = f"{WORK}/output_data_gpucheck_{TAG}"
m = {}
for f in sorted(glob.glob(f"{out_dir}/gpupool_training_metrics_p*.yaml")):
    m.update(yaml.safe_load(open(f)) or {})
print(f"  {len(m)} trials\n")
print(f"  {'trial':46}{'init_weights_sha256':>20}{'s':>8}")
for k in sorted(m):
    v = m[k]
    print(f"  {k:46}{v['init_weights_sha256'][:16]:>20}{v['train_duration']:>8.0f}")
print("\n  Compare these inits with those of the other cells (they all give the SAME one).")
print("  If they match -> the control survives and the version experiment is valid.")
print("  If they do NOT match -> torch changes the CPU init and a redesign is needed.")

with open(f"{WORK}/run_meta.json", "w") as f:
    json.dump({"tag": TAG, "torch": env_info["torch"], "arch": env_info["arch"],
               "gpus": env_info["gpus"], "phase": PHASE, "trials": len(m),
               "training_fingerprint": got_t, "dataset_fingerprint": got_d,
               "minutes": round((time.time()-start)/60, 1)}, f, indent=2)
