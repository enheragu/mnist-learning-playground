# Repeated-measures ANOVA, sphericity test and intraclass correlation of the MNIST ablation of the article
# (batch size x learning rate), from the data deposit (export of src/08_export_zenodo.py). Analysis by A. Rujano;
# reads the deposit instead of a chosen file.
#
# Usage: Rscript mnist_ablation_anova.R <export_dir>
# Uses the iterations trained in all nine conditions; each iteration has its own seed. The ANOVA reports the
# generalized eta squared and the Greenhouse-Geisser correction for the three terms. The ICC of the article is
# ICC(3,1) from the ANOVA mean squares (lmer = FALSE); with the default lmer = TRUE, psych::ICC estimates the
# variance components with lme4, the seed variance is estimated at its boundary (zero) and ICC3 is reported as 0.
# mnist_ablation_stats.py gives the same values in Python.

options(width = 200)
library(dplyr)
library(tidyr)
library(tibble)
library(rstatix)
library(psych)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) > 0) args[1] else "zenodo_export"

##############################
#                            #
#            DATA            #
#                            #
##############################

trials <- read.csv(file.path(export_dir, "mnist_trials.csv"))

data <- trials %>%
  filter(experiment == "ablation") %>%
  group_by(iteration) %>%
  filter(n() == 9) %>%
  ungroup() %>%
  mutate(
    # variables are categorical
    batch_size = factor(batch_size),
    learning_rate = factor(learning_rate),
    seed = factor(seed),
    iteration = factor(iteration),
    # combination of batch_size and learning_rate
    combination = paste(batch_size, learning_rate, sep = '_'))

cat("complete iterations:", nlevels(droplevels(data$iteration)), " runs:", nrow(data), "\n")

#############################################
#                                           #
#    REPEATED-MEASURES ANOVA AND SPHERICITY  #
#                                           #
#############################################

anova <- anova_test(data,
                    dv = accuracy,
                    wid = seed,
                    within = c(batch_size, learning_rate),
                    type = 3,
                    detailed = TRUE)

print(get_anova_table(anova,
                      # Greenhouse-Geisser correction
                      correction = 'GG'))
print(anova$`Mauchly's Test for Sphericity`)
print(anova$`Sphericity Corrections`)

#############################################
#                                           #
#    INTRACLASS CORRELATION COEFFICIENT     #
#                                           #
#############################################

# matrix for ICC (seeds x combinations)
matrix_icc <- data %>%
  select(iteration, combination, accuracy) %>%
  pivot_wider(names_from = combination, values_from = accuracy) %>%
  column_to_rownames('iteration')

ICC(matrix_icc, lmer = FALSE)
