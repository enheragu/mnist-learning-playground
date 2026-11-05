# Three-way ANOVA of the LLVIP equalization ablation of the article (fusion x RGB eq. x thermal eq.), from the data
# deposit (export of src/08_export_zenodo.py). Analysis by A. Rujano; reads the deposit instead of a chosen file.
#
# Usage: Rscript llvip_anova.R <export_dir>
# Between-subjects ANOVA with type III sums of squares on the valid runs, partial eta squared as effect size, for
# recall, precision, mAP50 and mAP50-95 of the person class (in these groups the evaluation holds only that class).
# llvip_anova.py gives the same table in Python.

options(width = 200)
library(dplyr)
library(rstatix)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) > 0) args[1] else "zenodo_export"

##############################
#                            #
#            DATA            #
#                            #
##############################

runs <- read.csv(file.path(export_dir, "yolo_runs.csv"))

data <- runs %>%
  filter(startsWith(group, "LLVIP_")) %>%
  mutate(
    equalization = sub("^LLVIP_[^_]+_", "", group),
    # variables are categorical
    fusion = factor(sub("^LLVIP_([^_]+)_.*$", "\\1", group)),
    rgb_eq = factor(ifelse(equalization %in% c("rgb_equalization", "rgb_th_equalization"), "clahe", "none")),
    th_eq = factor(ifelse(equalization %in% c("th_equalization", "rgb_th_equalization"), "clahe", "none")),
    index = factor(run_id),
    R = R_person,
    P = P_person,
    mAP50.95 = mAP50_95)

print(count(data, fusion, rgb_eq, th_eq))

#################################################
#                                               #
#                THREE-WAY ANOVA                #
#                                               #
#################################################

for (outcome in c("R", "P", "mAP50", "mAP50.95")) {
  anova_outcome <- anova_test(data,
                              dv = !!sym(outcome),
                              wid = index,
                              between = c(fusion, rgb_eq, th_eq),
                              type = 3,
                              detailed = TRUE,
                              effect.size = 'pes')

  cat("\n==", outcome, "\n")
  print(get_anova_table(anova_outcome))
}
