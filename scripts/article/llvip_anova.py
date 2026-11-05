"""Three-way ANOVA of the LLVIP equalization ablation of the article (fusion x RGB eq. x thermal eq.), from the data
deposit (export of src/08_export_zenodo.py).

Usage: python3 llvip_anova.py <export_dir>
Between-subjects ANOVA with type III sums of squares and sum-to-zero contrasts on the valid runs, for recall,
precision, mAP50 and mAP50-95 of the person class; effect size is the partial eta squared. Same model as
llvip_anova.R (rstatix::anova_test, type = 3). Prints the LaTeX rows of the table of the article.
"""
import sys
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from ablation_data import llvip_ablation

warnings.filterwarnings("ignore")

TERMS = [  # term in the model, name in the table
    ("C(fusion, Sum)", "fusion"),
    ("C(rgb_eq, Sum)", "visible equalization (RGB)"),
    ("C(th_eq, Sum)", "thermal equalization (TH)"),
    ("C(fusion, Sum):C(rgb_eq, Sum)", "$fusion \\times RGB$"),
    ("C(fusion, Sum):C(th_eq, Sum)", "$fusion \\times TH$"),
    ("C(rgb_eq, Sum):C(th_eq, Sum)", "$RGB \\times TH$"),
    ("C(fusion, Sum):C(rgb_eq, Sum):C(th_eq, Sum)", "$fusion \\times RGB \\times TH$"),
]
METRICS = [("R", "Recall"), ("P", "Precision"), ("mAP50", "mAP\\textsubscript{50}"),
           ("mAP50-95", "mAP\\textsubscript{50-95}")]


def anova(table):
    rows = []
    d = table.rename(columns={"mAP50-95": "mAP50_95"})
    for metric, _ in METRICS:
        fit = smf.ols(f"{metric.replace('-', '_')} ~ C(fusion, Sum) * C(rgb_eq, Sum) * C(th_eq, Sum)", data=d).fit()
        t = anova_lm(fit, typ=3)
        ss_res = t.loc["Residual", "sum_sq"]
        for term, name in TERMS:
            ss = t.loc[term, "sum_sq"]
            rows.append({"metric": metric, "term": name, "F": t.loc[term, "F"], "p": t.loc[term, "PR(>F)"],
                         "eta2p": ss / (ss + ss_res), "df_res": int(t.loc["Residual", "df"])})
    return pd.DataFrame(rows)


def p_text(p):
    if p < 0.001:
        mantissa, exponent = f"{p:.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{p:.3f}".lstrip("0")


def interpretation(p):
    return "Highly significant" if p < 0.001 else "Significant" if p < 0.05 else \
        "Marginally significant" if p < 0.1 else "Non-significant"


if __name__ == "__main__":
    table = llvip_ablation(sys.argv[1])
    result = anova(table)
    print(f"% {len(table)} valid runs, residual df {result.df_res.iloc[0]}; "
          + ", ".join(f"{k}: {v}" for k, v in table.groupby(['fusion', 'rgb_eq', 'th_eq']).size().items()))
    for metric, title in METRICS:
        print(f"\\textbf{{{title}}} &  &  &  \\\\")
        for _, x in result[result.metric == metric].iterrows():
            eta = "< .001" if x.eta2p < 0.0005 else f"{x.eta2p:.3f}".lstrip("0")
            print(f"{x.term} & {p_text(x.p)} & {eta} & {interpretation(x.p)} \\\\")
        print()
