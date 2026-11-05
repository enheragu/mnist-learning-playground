# Numbers of the article from the data deposit

These scripts recompute the tables and the numbers of the article that are not produced by the training and analysis
pipeline of `src/`. They read only the public data deposit (the export of `src/08_export_zenodo.py`, the same files
that are archived in Zenodo), so they can be run without the training outputs. Each one takes the folder of the
deposit as its first argument (`zenodo_export/` when it is built locally) and prints the values, most of them as the
LaTeX rows of the corresponding table.

| Article | Script |
|---|---|
| Methodology and appendix: training time per run | `training_times.py` |
| Replication and comparison: probability of a switched order (MNIST and LLVIP tables) | `switched_probability.py` |
| Ablation tests: sequential decision trees (figures) and the probability of reaching the optimum (MNIST table and LLVIP appendix table) | `ablation_trees.py`, then `ablation_trees_tikz.py` |
| Ablation tests: percentile correlation between conditions and its medians, ICC(3,1) | `mnist_ablation_stats.py` |
| Ablation tests: repeated-measures ANOVA of MNIST; appendix: Mauchly's test and Greenhouse-Geisser epsilons | `mnist_ablation_anova.R` (R), `mnist_ablation_stats.py` (same values in Python) |
| Ablation tests: three-way ANOVA of the LLVIP equalization ablation | `llvip_anova.R` (R), `llvip_anova.py` (same values in Python) |
| Replication across machines: summary table, transfer table of the appendix | `cross_machine.py` |
| Discussion: example reporting tables | `discussion_tables.py` |

`ablation_data.py` builds the two ablation tables from the deposit for the other scripts: the valid LLVIP runs with
the factors fusion, visible (RGB) equalization and thermal equalization, and the MNIST ablation trials of the
iterations that are complete for the nine conditions.

## Running them

```bash
python3 scripts/article/switched_probability.py zenodo_export
python3 scripts/article/cross_machine.py zenodo_export 40000
Rscript scripts/article/mnist_ablation_anova.R zenodo_export
```

- **Python:** the packages of `requirements` (NumPy, pandas, SciPy and statsmodels are the ones used here).
- **R:** the statistical analyses of the article were run in R 4.5.2. The R scripts need `dplyr`, `tidyr`, `tibble`,
  `rstatix` and `psych`.
- **Decision trees:** `ablation_trees.py` runs the simulation of the MLV Tools viewer
  ([enheragu/mlv-tools](https://github.com/enheragu/mlv-tools)), taking the path of
  `assets/tool-core/mlva/pyodide-analysis.js` as its first argument, so that the numbers are the ones the viewer gives
  for the same data. The article uses 40,000 simulated paths per method and seed 0:

  ```bash
  python3 scripts/article/ablation_trees.py ../mlv-tools/assets/tool-core/mlva/pyodide-analysis.js zenodo_export llvip 40000 0 trees_llvip.json
  python3 scripts/article/ablation_trees.py ../mlv-tools/assets/tool-core/mlva/pyodide-analysis.js zenodo_export mnist 40000 0 trees_mnist.json
  python3 scripts/article/ablation_trees_tikz.py trees_llvip.json trees_mnist.json <out_dir>
  ```

  `ablation_trees_tikz.py` writes one TikZ picture per tree; the node styles build on the `archblock` style and the
  colour names of the article sources.

The figures and the remaining tables of the article (distributions, normality tests, estimation errors, sampling
probabilities, best configuration per seed) come from the analysis scripts of `src/`, described in the main README,
which read the same deposit when the training outputs are not available. The table of model characteristics comes
from `src/check_model_info.py`, which reads the definitions of the models.
