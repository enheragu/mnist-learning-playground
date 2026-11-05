"""Sequential decision trees of the ablation studies of the article (LLVIP and MNIST), from the data deposit.

The simulation is the one of the MLV Tools viewer (https://github.com/enheragu/mlv-tools): the Python source embedded in
assets/tool-core/mlva/pyodide-analysis.js is extracted and executed as is, so the numbers are those the viewer gives
for the same data. Each trial draws one run per condition (Monte Carlo: fitted normal; Bootstrap: resampled run) and
follows the sequential ablation: fix the start factor, choose each remaining factor in order, then re-choose the start
factor. LLVIP uses mAP50 of the valid runs; MNIST uses the accuracy (in %, as in the viewer) of the iterations that are
complete for the nine conditions. The article uses 40000 trials and seed 0.

Usage:
  python3 ablation_trees.py <pyodide-analysis.js> <export_dir> llvip <trials> <seed> <out.json>
  python3 ablation_trees.py <pyodide-analysis.js> <export_dir> mnist <trials> <seed> <out.json>
"""
import json
import re
import sys

import numpy as np

from ablation_data import llvip_ablation, mnist_ablation

js_path, export_dir, dataset, trials, seed, out_path = sys.argv[1:7]
trials, seed = int(trials), int(seed)

source = open(js_path).read()
python_source = re.search(r"var ANALYSIS_PY = `\n(.*?)\n`;", source, re.S).group(1)
viewer = {}
exec(python_source, viewer)

if dataset == "llvip":
    table, metric = llvip_ablation(export_dir), "mAP50"
    factor_order = ["fusion", "rgb_eq", "th_eq"]
elif dataset == "mnist":
    table, metric = mnist_ablation(export_dir), "accuracy"
    table["accuracy"] = (100 * table["accuracy"]).round(2)
    factor_order = ["batch_size", "learning_rate"]
else:
    raise SystemExit(f"unknown dataset: {dataset}")

metric_data, model_factors = {}, {}
for levels, d in table.groupby(factor_order):
    name = "_".join(f"{f}-{v}" for f, v in zip(factor_order, levels))
    metric_data[name] = d[metric].to_numpy(dtype=float)
    model_factors[name] = {f: str(v) for f, v in zip(factor_order, levels)}
condition_meta = {"factor_order": factor_order, "model_factors": model_factors}
models = list(metric_data)

out = {"dataset": dataset, "metric": metric, "trials": trials, "seed": seed,
       "n_runs": {m: int(len(v)) for m, v in metric_data.items()}, "model_factors": model_factors, "trees": {}}
for method in ("mc", "bs"):
    np.random.seed(seed)
    out["trees"][method] = viewer["_simulate_decision_paths"](metric_data, models, condition_meta, trials, method)
json.dump(out, open(out_path, "w"), indent=1)

print(f"{dataset} {metric}: {sum(out['n_runs'].values())} runs, {trials} trials per method, seed {seed}")
print("global best:", out["trees"]["mc"][0]["global_best_model"])
for mc, bs in zip(out["trees"]["mc"], out["trees"]["bs"]):
    print(f"  start {mc['start']:22s} reach best: MC {100 * mc['p_reach_best']:6.2f} %  Bootstrap {100 * bs['p_reach_best']:6.2f} %")
