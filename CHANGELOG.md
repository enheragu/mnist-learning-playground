# Changelog

All notable changes to this project are documented in this file. The format follows the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/): an `[Unreleased]` section plus
versioned, dated entries in reverse-chronological order (SemVer), with changes listed as
flat bullets prefixed by type (`Bugfix:`, `Feature:`, …) rather than grouped into
Added/Changed/Fixed sections. Each release links to its git tag.

Note: the version of the code is the one in [CITATION.cff](CITATION.cff).

Versioning note: minor releases mark visible feature jumps; patch releases cover bugfixes and
maintenance updates, including docs polish that does not change behavior. Versions prior to 1.0.0
were reconstructed from the git history, with each tag on the last commit of its version.

## [1.0.1](https://github.com/enheragu/mnist-learning-playground/releases/tag/v1.0.1) - 2026-11-06

First release archived in Zenodo, so that the articles that use this code can cite a fixed version of it.

- Bugfix: the YOLOv8 survival figures use the final evaluation of each run, like the tables, instead of the maximum over the epochs.
- Bugfix: ties are strict everywhere, so no result depends on the order of a list. A tie is not a switched order, conditions tied for the best of a seed share it, and the results of the review are counted strictly (48 and 56).
- Bugfix: the normality table also gives the p-value without the ties of the rounded accuracy, which alone rejected normality in three models, and no longer labels the p-value as a percentage.
- Bugfix: the overfitting analysis uses the final value of each run and only plots the mean while half of the runs are alive; the YOLOv8 one only uses valid runs.
- Bugfix: YOLOv8 runs are collected in a fixed order, so the seeded simulations are also reproducible from the training logs.
- Bugfix: parallel training processes no longer exceed the target number of trials, and the training scripts report an error when an iteration fails.
- Bugfix: a utility that builds a model no longer turns off the reading of the deposit.
- Bugfix: loss surfaces apply the filter normalization they promised, with seeded directions.
- Bugfix: cross-GPU scripts. The stack comparison works when the cells have different conditions, the check that refuses to mix architectures no longer depends on the order of its arguments, and the mechanism kernel reads the real results of the probes.
- Bugfix: data deposit. Two exports of the same data are byte-identical, the compressed files hold no timestamp and are also checked, and its README describes the final test loss correctly and points to the scripts that reproduce the article.
- Analysis: the decision simulation of the ablation analysis is off by default and has its own table name, because the table of the article comes from the simulation of the viewer.
- Figures: fit lines no longer depend on the scale of the data, curves beyond the palette change line style, interaction plots honor their order, and empty plots are no longer written.
- Logging: LaTeX tables bold only their header, percentile tables are labelled by percentile, and the mixed models report AIC and BIC without a stray fixed effect.
- Feature: scripts for the numbers quoted in the text of the article and for its sample digits, and a map from each figure to its script and flag.
- Training: the overfitting runs mark themselves out of the general analysis, and the tmux launcher starts the main study by default.
- Cleanup: NumPy and SciPy among the dependencies, and a minimum version of pingouin, which changes the sphericity correction.
- Docs: shorter README that opens with what the study shows and how to regenerate the article. The architectures move to their own README next to the models, the diagrams warn that they do not show the activations applied in `forward`, and the known details of the frozen training code are documented.
- Docs: the cross-GPU README adds how to get MNIST and delete the checkpoints, and says that the kernels are frozen to their last session.

## [1.0.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v1.0.0) - 2026-11-05

- Feature: data deposit. Builds the public per-trial deposit of the article (MNIST trials, the valid YOLOv8 runs and the cross-GPU study) from a whitelist of fields and checks it, so the results are published without local paths, host names or timestamps.
- Feature: analysis from the deposit. Without the training outputs, the analysis scripts read the deposit, so the figures and tables of the article that come from results can be regenerated from the Zenodo files alone.
- Feature: numbers of the article. Scripts that recompute from the deposit the tables that the analysis pipeline does not produce (switched-order probabilities, decision trees and ANOVA of the ablations, in Python and in R, cross-GPU tables, reporting examples, training times), with a map from each table to its script.
- Feature: cross-GPU replication study. Retrains MNIST conditions with a shared seed pool on several GPUs and software stacks and compares which statistics transfer between machines.
- Feature: YOLO run validity. Only finished runs with a final evaluation, a real seed and one run per seed enter the analysis, and an inventory records these checks for every run; groups without valid runs are no longer plotted.
- Bugfix: Monte Carlo and Bootstrap results changed slightly from run to run, because no seed was fixed and the YOLO groups were collected in the order their workers finished. Every simulation now starts from a fixed seed and the groups keep a fixed order, so figures and tables are reproducible.
- Bugfix: the inter-model distance analysis failed on a clean clone.
- Bugfix: parallel runs could lose trials, because the metrics file was read and written under two separate locks.
- Bugfix: the accuracy plot of each model was only generated for the last model.
- Bugfix: the amplitude tables labelled as 99.9% an interval that covers 99.8%. The values do not change.
- Figures: drawn at the size they are printed in the articles (column width, 8 pt text, legends that do not cover the data), with the vocabulary of the articles in the legends (fusion and equalization names, batch size and learning rate) and the palette colors as defined. The distribution plots only do so when asked, because the CPC18 repository draws them at its own size.
- Figures: distribution plots can show the observed values as a rug and take a fixed legend position, for the companion CPC18 repository.
- Cleanup: comments, docstrings, identifiers and messages in English, with no change in behavior (same parameter counts, initial weights and seed pool).
- Cleanup: PyYAML among the dependencies, the launcher finds the virtual environment inside or next to the repository, and outputs, the deposit and local credentials are ignored by git.
- Docs: README titled like the citation metadata; it cites the article, describes the data deposit and how to regenerate the article from it, and points to the CPC18 repository, which reuses the utilities as a submodule.
- License: MIT license and citation metadata.

## [0.10.1](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.10.1) - 2026-05-23
- Docs: the README links MLV Tools, the interactive web application that browses the results of this repository.

## [0.10.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.10.0) - 2026-05-22
- Feature: overfitting analysis. Learning curves of the overfit runs and of the YOLO runs, with a scored overfit or underfit label. A marker file keeps a results folder out of the general analysis, so the overfit runs do not enter it.
- Feature: sampling curves. Sampling error of the mean and the std, and probability of reaching a percentile, against the number of runs, computed analytically, by Monte Carlo and by Bootstrap.
- Feature: survival functions of the results.
- Feature: switched-order probability of the best of N runs (N from 1 to 5), not only of a single run.
- Feature: YOLO analysis with the same tools (sampling curves, switched-order probability, survival functions), more groups and incremental parsing of the runs.
- Analysis: lower simulation counts and new defaults. Switched-order probability uses 40000 Monte Carlo and 40000 Bootstrap draws (500000 and 100000 before), the default percentile of the better-result sample size is 90 (96 before), the estimation error is absolute (relative before) and the sampling error uses the sample standard deviation.
- Bugfix: conflict markers left by the merge of 0.9.0 stopped every analysis script from importing.
- Bugfix: the 99.9% column of the training-time and best-epoch amplitude tables used a normal fit although a gamma fit was requested.
- Figures: legends of the distribution plots move to a free corner or below the plot, survival plots can keep the legend outside, and the ablation plots show the individual trials with jitter.
- Docs: README rewritten with the models, the scripts and the outputs.

## [0.9.0] - 2026-02-10 <!-- developed in parallel to 0.7.0 and 0.8.0 and never separately tagged: its commit lacks them and the merge has conflict markers; ships within v0.10.0 -->
- Feature: overfitting experiment. CNN_14L and DNN_6L train on 2% of the dataset with an aggressive learning rate, to check whether low variance is a sign of overfitting.
- Training: every trial also stores its per-epoch train accuracy, train loss and test loss.
- Training: the ablation grid goes from 298 to 310 seeds.
- Bugfix: the lock of the metrics files uses Unix `flock`, because the previous one was not atomic.
- Figures: the gamma fit is skipped for data with non-positive values instead of failing.

## [0.8.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.8.0) - 2025-10-30
- Feature: estimation error. Error of the Monte Carlo and Bootstrap estimates of the mean and the std against the number of samples.
- Feature: probability of reaching the best result of the ablation grid when the hyperparameters are chosen from a limited number of runs.
- Figures: stored as PDF.
- Cleanup: the switched-order probability moves to a shared module, with no change in behavior.

## [0.7.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.7.0) - 2025-10-14
- Feature: YOLO results. Variance tables and distribution plots of mAP50 and mAP50-95 over the runs trained with the companion repository yolo_test_utils.
- Feature: ablation analysis with repeated-measures ANOVA, pairwise Spearman correlation between conditions, scatter plots and LaTeX tables. The ablation conditions also enter the distribution and distance analyses.
- Analysis: the sample size needed to beat a percentile is simulated by Monte Carlo instead of computed in closed form, and Bootstrap uses 100000 draws (50000 before).
- Bugfix: the training-time and best-epoch amplitude tables were multiplied by 100.
- Bugfix: the distance analysis failed on import, and the distribution script stopped after its example plots.

## [0.6.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.6.0) - 2025-07-03
- Feature: ablation test. CNN_14L trains on a grid of three batch sizes and three learning rates, repeating the same seeds in every condition, with a mode to resume incomplete seeds.
- Training: the ablation runs do not shuffle the data and stop at 100 epochs, unlike the main study (shuffled, 500 epochs), so their seed only changes the initial weights.
- Training: the seed helper also seeds Python's `random`.
- Feature: ablation analysis with ANOVA, interaction plots and Kendall's W.
- Figures: example distributions of interesting cases, generated by biased resampling of the trials.
- Cleanup: scripts numbered in pipeline order, shared code moved to `src/utils`, and a launcher for the tmux sessions.

## [0.5.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.5.0) - 2025-04-15
- Feature: BatchNormMaxoutNetInNet model.
- Training: weight initialization moves to the base model and the biases of the CNNs start at zero instead of at the random PyTorch default, so a seed gives different CNN weights than in earlier versions.
- Training: every trial stores its seed, and precision, recall and F1 are no longer computed.
- Training: the script tops up each model to a target of stored trials (400), and a failure in one model no longer stops the others.
- Feature: comparison with published results separated into runs with and without data augmentation.
- Figures: per-model figures, and figures are closed when they are only stored. Model diagrams support the layers of the new model.
- Tools: script to monitor all the tmux sessions at once.

## [0.4.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.4.0) - 2024-12-26
- Feature: batch-size study of the 14-layer CNN (batch sizes 10, 25, 50 and 80).
- Feature: model summary script that also generates a Mermaid diagram of each model.
- Feature: gamma fit for the training time and the number of epochs, and amplitude tables.
- Feature: comparison with the error rates published for other models on MNIST.
- Training: every trial stores its per-epoch test accuracy.
- Cleanup: unified model names (CNN_2L becomes CNN_3L and CNN_13L becomes CNN_14L), with no change in the layers.
- Logging: progress bars with tqdm, and tables written as text and LaTeX in organized analysis folders.
- Bugfix: the scripts created folders they did not need.

## [0.3.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.3.0) - 2024-12-21
- Feature: switched-order probability. Monte Carlo and Bootstrap estimates of the probability that a run reverses the order of two or more models.
- Feature: number of runs needed to beat a percentile of the distribution.
- Feature: sampling error against the number of runs, as a plot and a table.
- Feature: normality check of each distribution (Shapiro-Wilk and kurtosis).
- Figures: plots grouped by model family, and a plot of accuracy amplitude against trainable parameters, not called by default.
- Cleanup: the output path of a model is optional; without it, nothing is stored.

## [0.2.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.2.0) - 2024-11-15
- Feature: multi-model comparison. A common trainer with seven models (SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_2L, CNN_4L, CNN_5L and CNN_13L), and a distribution plot that overlays them with their mean training time.
- Feature: file lock so that several processes can write to the same metrics file.
- Bugfix: parallel runs of a model shared one best-model file and could reload each other's weights.
- Docs: README with the description of the repository.

## [0.1.0](https://github.com/enheragu/mnist-learning-playground/releases/tag/v0.1.0) - 2024-11-13
- Feature: first version of the study. A perceptron trained on MNIST over repeated runs with different seeds, with early stopping and best-model selection on the test accuracy, metrics stored in YAML and a plot of the accuracy distribution.


# Useful notes :)

> To commit a new release: update version in `CITATION.cff` and add the new section in the changelog with the tag URL (tag name is known in advance). Commit, tag, push.
> ```sh
> git add <files>
> git commit -m "Release vN.M.P"
> git tag -a vN.M.P -m "vN.M.P"
> git push origin main
> git push --tags
> ```
> Zenodo archives GitHub releases, not tags, so publish a release only for the versions to archive.
