# Numbers of the article from the data deposit

These scripts recompute the tables and the numbers of the article that are not produced by the training and analysis
pipeline of `src/`. They read only the public data deposit (the export of `src/08_export_zenodo.py`, the same files
that are archived in Zenodo), so they can be run without the training outputs. Unless noted under *Running them*, each
one takes the folder of the deposit as its first argument (`zenodo_export/` when it is built locally) and prints the
values, most of them as the LaTeX rows of the corresponding table.

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
| Numbers quoted in the text that no table prints | `text_numbers.py` |
| Methodology: sample digits (figure, needs MNIST instead of the deposit) | `mnist_samples.py` |

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
  for the same data. The numbers of the article were checked with commit `550ed7e` of that repository. The article uses
  40,000 simulated paths per method and seed 0:

  ```bash
  git clone https://github.com/enheragu/mlv-tools.git ../mlv-tools
  python3 scripts/article/ablation_trees.py ../mlv-tools/assets/tool-core/mlva/pyodide-analysis.js zenodo_export llvip 40000 0 trees_llvip.json
  python3 scripts/article/ablation_trees.py ../mlv-tools/assets/tool-core/mlva/pyodide-analysis.js zenodo_export mnist 40000 0 trees_mnist.json
  python3 scripts/article/ablation_trees_tikz.py trees_llvip.json trees_mnist.json <out_dir>
  ```

  `ablation_trees_tikz.py` takes the two JSON files instead of the deposit and writes one TikZ picture per tree; the
  node styles build on the `archblock` style and the colour names of the article sources.
- **Sample digits:** `mnist_samples.py <out_dir> [data_dir]` takes the output folder as its first argument.

## Figures and the remaining tables

They come from the analysis scripts of `src/`, described in the main README, which read the same deposit when the
training outputs are not available. Those scripts select their analyses with boolean flags at the top of the file, so
set to `True` the ones in the table. The folder is the one under `analysis_results/` where the file appears.

| Article | Script | Flag | Folder |
|---|---|---|---|
| Accuracy distributions, single models and comparisons (`plot_accuracy_*`, `plot_best_epoch_SimplePerceptron`) | `03_plot_distribution.py` | `PLOT_DISTRIBUTIONS` | `distributions/` |
| CNN_14L with batch sizes 10 and 50, compared with the review | `04_check_distances.py` | | `distances/` |
| Ablation distributions (`plot_accuracy_ablation*`) | `03_plot_distribution.py` | `PLOT_DISTRIBUTIONS` | `distributions/` |
| Example distributions (`example_CNN_14L_*`) | `03_plot_distribution.py` | `PLOT_EXAMPLE_DISTRIBUTIONS` | `distributions/` |
| Normality test table | `03_plot_distribution.py` | `COMPUTE_ANALYSIS_METRICS` | `distributions/tables/` |
| Amplitude against parameters (`amplitude_relation_*`) | `03_plot_distribution.py` | `PLOT_AMPLITUDE_RELATION` | `distributions/` |
| Survival, MNIST (`survival_combined`) | `03_plot_distribution.py` | `COMPUTE_SURVIVAL_FUNCTION` | `distributions/` |
| Sampling error (`sampling_error_*`), MNIST | `02_compute_analysis.py` | `enable_compute_sampling_error_graphs` | `analysis/` |
| Probability of reaching a percentile (`sampling_probability_*`), MNIST | `02_compute_analysis.py` | `enable_compute_percentile_probability_graphs` | `analysis/` |
| Estimation error tables | `02_compute_analysis.py` | `enable_compute_estimation_error` | `analysis/tables/` |
| Best configuration per seed, scatter plots, correlations | `05_check_ablation_test.py` | `percentile_analysis` | `ablation_anova/` |
| Interaction plots (`interaction_plot_*`) | `05_check_ablation_test.py` | `plot_interaction` | `ablation_anova/` |
| YOLOv8 distributions (`kaist_day_hsvt_mAP50`, `plot_best_epoch_kaist_day_hsvt`, `KAIST_day_*`, `LLVIP_fusion_*`, `LLVIP_equalization_*`) | `check_yolo_results.py` | `enable_yolo_plot_distributions`, `enable_yolo_ablation_tests` | `yolo_analysis/distributions/` |
| YOLOv8 survival (`survival_mAP50*`), from the final evaluation of each run | `check_yolo_results.py` | `enable_yolo_survival_function` | `yolo_analysis/distributions/` |
| YOLOv8 sampling curves (`sampling_*_yolo_*`) | `check_yolo_results.py` | `enable_yolo_sampling_plots` | `yolo_analysis/` |

The decision table of the ablation comes only from `ablation_trees.py`, as the viewer does. `05_check_ablation_test.py`
has a different simulation that draws again in its second step, off by default (`compute_decision_error_redrawn`) and
with its own table name, so its values do not match the article.

The table of model characteristics comes from `src/check_model_info.py` (parameters and memory) and from counting the
convolutional and linear layers in the definition of each model.

## Reproducing the figures exactly

Pixel-identical figures need matplotlib 3.10, because later versions change the layout of the legends. The sphericity
correction printed by `05_check_ablation_test.py` needs pingouin 0.7 or later, which `requirements` asks for.
