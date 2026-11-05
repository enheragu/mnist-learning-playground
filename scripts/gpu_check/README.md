# Cross-GPU replication study

## Purpose

These scripts re-train six MNIST conditions with a common, deterministic seed
pool on several GPU architectures and software stacks, to check which of the
reported statistics transfer across machines. Every training is identified by
its model and seed, so the results of different machines can be compared seed
by seed.

## Contents

| file | role |
|---|---|
| `make_seed_pool.py` | deterministic seed pool; can be extended without changing the seeds already drawn |
| `run_study.py` | training runner with shards, deadline, resume and atomic writes |
| `preloaded_loader.py` | GPU-resident dataset that replicates the RNG consumption of the original `DataLoader` |
| `analyze_cells.py` | full analysis across cells |
| `compare_runs.py` | quick comparison between output directories |
| `make_resume.py` | rebuilds resume files for a different shard scheme |
| `determinism_test.py` | repeated runs of the same seed on the same machine |
| `init_on_cpu_test.py` | initial weights built on GPU vs on CPU |
| `probe_ops.py`, `compare_probes.py` | per-operator probe with CPU-seeded inputs, and its comparison |
| `validate_preload.py` | checks that the preloaded loader gives the same weights as the original loader |
| `kaggle/*.py` | Kaggle kernels (see below) |

`seed_pools.yaml` and `requirements.lock` are at the repository root, where the
scripts read them.

## Design

### Cells

A cell is one GPU architecture with one software stack.

| GPU | compute capability | SMs | stack | trainings |
|---|---|---|---|---|
| RTX 4090 | `sm_89` | 128 | torch 2.9.1 + cu128 | 750 |
| 2 x RTX 5080 | `sm_120` | 84 | torch 2.9.1 + cu128 | 750 |
| Tesla T4 (Kaggle) | `sm_75` | 40 | torch 2.9.1 + cu128 | 750 |
| RTX 4090 | `sm_89` | 128 | torch 2.7.1 + cu118 | 700 |
| Tesla T4 (Kaggle) | `sm_75` | 40 | torch 2.7.1 + cu118 | 700 |
| Tesla P100 (Kaggle) | `sm_60` | 56 | torch 2.7.1 + cu118 | 700 |
| 2 x RTX 5080 | `sm_120` | 84 | torch 2.7.1 + cu128 | 700 |

This covers four NVIDIA generations (Pascal, Turing, Ada, Blackwell), two torch
versions and three combinations of CUDA libraries, 5,050 trainings in total.

Fixed across cells: the seed pool, the code (training fingerprint, see below),
the dataset (fingerprint `474311efcf1fd213`), the hyperparameters and the
deterministic arm. What varies is the GPU architecture and, on the second axis,
the stack.

### Seed pool

`make_seed_pool.py` builds one seed list per condition from
`MASTER_SEED = 20260729`. Each condition has its own generator,
`random.Random(sub)`, where `sub` is taken from the SHA-256 of
`"<MASTER_SEED>:<model name>"`, so the seeds of a condition do not depend on
its position in the list. Seeds are drawn in `[0, 2**32 - 1]` without repeats within a condition.
The pool checksum is `1042813f472a716f`, and every training record stores it.

| condition | seeds |
|---|---|
| `SimplePerceptron` | 200 |
| `HiddenLayerPerceptron` | 200 |
| `CNN_3L` | 100 |
| `DNN_6L` | 100 |
| `CNN_14L` | 100 |
| `CNN_14L_B10` | 50 |

The full pool is 750 trainings. The order of the list sets the execution order:
seed index in the outer loop, model in the inner loop, so a session that stops
early leaves the same prefix of the pool on every machine.

### Deterministic arm

`set_clean_arm()` in `run_study.py` applies, before any training:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set only if it is not already defined);
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`;
- `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`;
- `torch.set_float32_matmul_precision("highest")`;
- `torch.use_deterministic_algorithms(True)`.

TF32 is disabled so that the arm is comparable with Pascal and Turing GPUs.

### CPU initialization control

The models move their layers to the device in the constructor, so a model built
on the GPU draws its initial weights from the CUDA RNG, which distributes the
numbers according to the launch configuration and therefore to the number of
SMs. The same seed would then give different initial weights on each
architecture, and the study would compare different starting points.

`run_study.py` builds each model with `torch.cuda.is_available` temporarily
returning `False`, so the layers are created on CPU and the initialization uses
the CPU RNG. The model is then moved to the GPU, and the hash of its weights is
checked to be unchanged by the move. That hash is stored as
`init_weights_sha256`, so the initial weights can be compared across cells.
`init_on_cpu_test.py` compares both construction paths on one GPU.

### Other fixed elements

- Hyperparameters of the original study: input size 784, 10 classes, learning
  rate 0.001, patience 10, at most 500 epochs; batch size as defined by each
  model.
- MNIST with `ToTensor()` and `Normalize((0.5,), (0.5,))`, read from `data/`
  without download.
- Data served by `preloaded_loader.py`, which keeps the dataset on the GPU and
  draws from the global RNG exactly as `DataLoader(shuffle=True)` and
  `DataLoader(shuffle=False)` do; `validate_preload.py` checks that the final
  weights are identical with both loaders.
- `run_study.py` refuses to write inside `output_data/`, which holds the
  results of the original study.

## How to run

Commands are given from the repository root. Install the pinned stack as
described in the header of `requirements.lock`:

```bash
uv pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.lock --index-url https://download.pytorch.org/whl/cu128 \
               --extra-index-url https://pypi.org/simple
```

On the lab machines, the reference and alternative stacks were installed with
deployment scripts that are not included here.

### 1. Seed pool

```bash
python scripts/gpu_check/make_seed_pool.py --check   # verify seed_pools.yaml
python scripts/gpu_check/make_seed_pool.py           # rewrite seed_pools.yaml
```

### 2. Training

```bash
python scripts/gpu_check/run_study.py --tag rtx4090-cu128 --dry-run   # show the plan
python scripts/gpu_check/run_study.py --tag rtx4090-cu128
```

- `--tag` names the run; results go to `output_data_gpucheck_<tag>/` under
  `--out-base` (default: the repository root). Use one tag per machine; two
  machines that belong to the same cell (for example `rtx5080-a-cu128` and
  `rtx5080-b-cu128`) are merged at analysis time.
- `--shard k/N` runs the seed indices `i` with `i % N == k`. To split a run into
  N processes, launch `--shard 0/N` to `--shard (N-1)/N` with the same tag; each
  process writes its own file:

  ```bash
  python scripts/gpu_check/run_study.py --tag rtx5080-a-cu128 --shard 0/2
  python scripts/gpu_check/run_study.py --tag rtx5080-a-cu128 --shard 1/2
  ```

- `--deadline-min` stops starting new trials once the given wall-clock minutes
  have passed; `--conditions` restricts the models; `--limit` caps the number
  of trials.
- Running the same command again resumes: trials already present in the shard
  file are skipped.
- The script aborts if the GPU cannot run a small matrix product and
  convolution with the installed stack.

### 3. Analysis

```bash
python scripts/gpu_check/analyze_cells.py \
    --cells 'sm_89=output_data_gpucheck_rtx4090-cu128/*.yaml' \
            'sm_120=output_data_gpucheck_rtx5080-a-cu128/*.yaml,output_data_gpucheck_rtx5080-b-cu128/*.yaml' \
            'sm_75=output_data_gpucheck_kaggle-T4-A/*.yaml,output_data_gpucheck_kaggle-T4-B/*.yaml' \
    --ref sm_89
```

Each `--cells` entry is `name=pattern[,pattern...]`; all files that match are
merged by record key into one cell. Quote the patterns so the script expands
them. The same call compares stacks on one architecture, for example
`'t4-cu128=output_data_gpucheck_kaggle-T4-A/*.yaml,output_data_gpucheck_kaggle-T4-B/*.yaml'`
and `'t4-cu118=output_data_gpucheck_kaggle-T4-cu118/*.yaml'`.

The script stops if the cells use different pool checksums. For each condition
it reports: bit-exactness of final weights per pair of cells; shape of the
accuracy distribution (mean, median, std, skewness, kurtosis); Mann-Whitney,
Levene and KS tests against the reference cell; shift per percentile relative
to the reference; bootstrap CI of the p90 per cell; transferability, as the
probability that the range between cells is reached by resampling seeds from
the pooled data; seed-by-seed ranking agreement (Spearman and % of inversions);
and a seed vs machine variance decomposition. `--ref` selects the reference
cell (default: the first one) and `--boot` the number of bootstrap replicates
(default 40000, as in the article). The transferability and variance sections pair trials by key
using the keys of the reference cell, so every cell must contain all of them.

`compare_runs.py` gives a shorter comparison over the trials common to several
output directories (initial-weight agreement, final-weight bit-exactness,
seconds per trial, mean accuracy):

```bash
python scripts/gpu_check/compare_runs.py output_data_gpucheck_rtx4090-cu128 output_data_gpucheck_rtx5080-a-cu128
python scripts/gpu_check/compare_runs.py --glob 'output_data_gpucheck_*-cu128'
```

### 4. Checks

```bash
# Same seed twice on the same GPU (3 epochs by default), default flags and clean arm
python scripts/gpu_check/determinism_test.py --tag rtx4090-defaults
python scripts/gpu_check/determinism_test.py --tag rtx4090-clean --no-tf32 --strict-determinism

# Initial-weight hashes with the model built on GPU and on CPU
python scripts/gpu_check/init_on_cpu_test.py --tag rtx4090

# Operator probe on each machine/configuration, then comparison
python scripts/gpu_check/probe_ops.py --tag rtx4090-defaults
python scripts/gpu_check/probe_ops.py --tag rtx4090-notf32 --no-tf32
python scripts/gpu_check/compare_probes.py --ref rtx4090-notf32 --field grad_input

# Preloaded loader vs original DataLoader (exit code 1 if the weights differ)
python scripts/gpu_check/validate_preload.py --seeds 5 --epochs 2
```

- `determinism_test.py`, `init_on_cpu_test.py` and `validate_preload.py` use the
  first five conditions (without `CNN_14L_B10`); the first two use seed
  `20260729`, `validate_preload.py` uses seeds `1000, 1001, ...`.
- `probe_ops.py` runs 14 operators in isolation (linear layers, convolutions,
  BatchNorm, max pooling, ReLU and the cross-entropy loss) with inputs and
  weights generated on CPU from a fixed seed, and stores SHA-256 hashes of the
  forward output, the input gradient and the parameter gradients. Other flags:
  `--cudnn-off`, `--benchmark`.
- These scripts write JSON files to `scripts/gpu_check/probe_results/`
  (`determinism_<tag>.json`, `initcpu_<tag>.json`, `probe_<tag>.json`);
  `compare_probes.py` reads all `probe_*.json` files in that folder. Copy the
  files of other machines there before comparing.

`make_resume.py` prepares resume files for a session that uses a different
shard scheme (used for Kaggle, where each session starts with an empty working
directory). It refuses to mix trials of different architectures or pools; it
does not check the torch version, so list only directories of one cell:

```bash
python scripts/gpu_check/make_resume.py --dirs output_data_gpucheck_kaggle-T4-A output_data_gpucheck_kaggle-T4-B --nshards 12 --out resume/
```

## Kaggle kernels

The kernels in `kaggle/` read the code and the MNIST files from a private Kaggle
dataset built from this repository. They locate its root under `/kaggle/input`
by its content (`src/` and `requirements.lock`), copy it to
`/kaggle/working/repo` and install their torch stack with `pip` from the
PyTorch index. All of them except `run_kernel_mechanism.py` also compute the
training and dataset fingerprints, abort if they differ from the values in
`FINGERPRINTS.txt` when the dataset contains that file (keys
`training_fingerprint` and `dataset_fingerprint`), and check that the assigned GPU
can run the installed stack. The scripts that build the dataset and launch the
kernels, and the lab deployment scripts, are not included.

Kaggle kernels do not receive environment variables, so the run configuration
(phases, conditions, shards, limits, deadlines) is set in constants at the top
of each file. The `EEHA_TAG` variable read by `checks_kernel.py`,
`run_kernel_prod.py` and `run_kernel_p100.py` only takes effect outside Kaggle.

| kernel | stack | what it does |
|---|---|---|
| `checks_kernel.py` | 2.9.1 + cu128 | runs `probe_ops.py` (default and `--no-tf32` arms), `init_on_cpu_test.py` and `determinism_test.py` (clean arm, 3 epochs); copies the JSON files to `/kaggle/working/results/` |
| `run_kernel_prod.py` | 2.9.1 + cu128 | production for the `sm_75` cell: runs `run_study.py` for each phase in `PHASES`, one process per shard, alternating between the two GPUs of a `GPU T4 x2` session, with a deadline below the 12 h session limit; seeds previous results from `repo/resume/` (built with `make_resume.py`) and aborts if their number or shard scheme do not match the expected ones |
| `run_kernel_stack.py` | 2.7.1 + cu118 | same seeds on the alternative stack; the tag is derived from the detected GPU (`kaggle-T4-cu118` or `kaggle-P100-cu118`), the run aborts if the GPU differs from `EXPECTED_GPU`, and previous results are seeded from `repo/resume/<tag>/` |
| `run_kernel_p100.py` | 2.7.1 + cu118 | short probe on a P100: a few `SimplePerceptron` and `HiddenLayerPerceptron` trials, then prints their `init_weights_sha256` to compare with the other cells |
| `run_kernel_mechanism.py` | both | on one T4, installs 2.9.1 + cu128 and then 2.7.1 + cu118 in the same session and runs `probe_ops.py --no-tf32` under each; the two JSON files are collected in `/kaggle/working/mechanism/` and can be compared with `compare_probes.py` |

Training kernels write to `/kaggle/working/output_data_gpucheck_<tag>/`, one log
per shard in `/kaggle/working/`, and a `run_meta.json` with the tag, GPUs,
fingerprints and phases.

## Output format

`run_study.py` writes one YAML file per shard,
`output_data_gpucheck_<tag>/gpupool_training_metrics_p<k>.yaml`, where `k` is
the shard index. The file is rewritten atomically after each trial. It holds one
record per trial, keyed `<model>__<seed>`, with these fields:

| field | content |
|---|---|
| `seed` | seed of the trial |
| `seed_index` | position of the seed in the condition's pool |
| `model` | condition name |
| `accuracy` | test accuracy of the best checkpoint, reloaded at the end of training |
| `test_loss_final` | last test loss (`null` if none) |
| `best_epoch` | best epoch, as returned by the training routine |
| `total_epochs` | `best_epoch` plus the patience |
| `train_duration` | training time in seconds |
| `accuracy_plot` | test accuracy per epoch |
| `train_accuracy_plot` | train accuracy per epoch |
| `train_loss_plot` | train loss per epoch |
| `test_loss_plot` | test loss per epoch |
| `init_weights_sha256` | hash of the initial weights |
| `final_weights_sha256` | hash of the weights at the end of training (best checkpoint) |
| `timestamp_utc` | time the record was written, UTC, ISO 8601 |
| `hostname` | host name of the machine (`socket.gethostname()`) |
| `tag` | run tag |
| `shard` | shard as `k/N` |
| `gpu` | GPU name |
| `capability` | compute capability, as `sm_XY` |
| `sm_count` | number of SMs |
| `torch` | torch version |
| `pool_checksum` | checksum of the seed pool |

Weight hashes are the first 16 hex digits of a SHA-256 over the state dict, with
keys in sorted order and values as float32 bytes. During training the model
checkpoints go to `output_data_gpucheck_<tag>/_ckpt/` and are deleted after each
trial; only the hashes are kept.

## Training fingerprint

The training fingerprint covers `src/models`, `src/utils/set_seed.py`,
`requirements.lock`, `scripts/gpu_check/run_study.py` and
`scripts/gpu_check/preloaded_loader.py`. From the repository root:

```bash
{ find src/models -type f -name '*.py'; ls src/utils/set_seed.py requirements.lock scripts/gpu_check/run_study.py scripts/gpu_check/preloaded_loader.py; } | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16
```

The fingerprint of the files used in the study was `b789b9bf7d14346e`. The
copies published here differ from those only in comments, docstrings and
messages, so their fingerprint is `f454e24cce0d2325`. The dataset fingerprint,
over the `*-ubyte` files, is computed with:

```bash
cd data/MNIST/raw && ls *-ubyte | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16
```
