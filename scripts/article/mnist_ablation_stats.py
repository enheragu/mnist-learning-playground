"""Statistics of the MNIST ablation of the article (batch size x learning rate), from the data deposit (export of
src/08_export_zenodo.py).

Usage: python3 mnist_ablation_stats.py <export_dir>
Uses the iterations trained in all nine conditions (one seed per iteration). Prints:
- the Spearman correlation between conditions of the percentile of each seed within its condition, and its median
  by batch size and by learning rate (excluding the correlation of a condition with itself);
- the intraclass correlation ICC(3,1) from the ANOVA mean squares, with its F-based 95% CI (psych::ICC with
  lmer = FALSE gives the same value);
- the repeated-measures ANOVA with the Greenhouse-Geisser correction for the three terms and the generalized eta
  squared, Mauchly's sphericity test (with the second-order term of R's mauchly.test) and the GG epsilons.
mnist_ablation_anova.R computes the same ANOVA, sphericity test and ICC in R.
"""
import sys

import numpy as np
from scipy.stats import chi2, f as fdist

from ablation_data import mnist_ablation

ab = mnist_ablation(sys.argv[1])
ab["cond"] = ab.batch_size.astype(str) + "_" + ab.learning_rate.astype(str)
print(f"% {ab.iteration.nunique()} complete iterations, {len(ab)} runs; conditions (in order): "
      + ", ".join(f"{i}: {c}" for i, c in enumerate(sorted(ab.cond.unique(), key=lambda c: tuple(map(float, c.split("_")))))))

# Percentile correlation between conditions
ab["pct"] = ab.groupby("cond").accuracy.rank(pct=True, ascending=False)
cor = ab.pivot_table(index="seed", columns="cond", values="pct").corr(method="spearman")
print("\n% Percentile correlation by seed between conditions")
for i in range(9):
    cells = [""] * i + ["1"] + [f"{cor.iloc[i, j]:.4f}" for j in range(i + 1, 9)]
    print(f"{i} & {' & '.join(cells)} \\\\")
off_diagonal = cor.where(~np.eye(9, dtype=bool))
for key, label in ((lambda c: c.split("_")[0], "batch size"), (lambda c: c.split("_")[1], "learning rate")):
    print(f"\n% Median of the percentile correlation by {label}")
    for g, row in off_diagonal.groupby(off_diagonal.index.map(key)).median().iterrows():
        print(f"{g} & {' & '.join(f'{v:.4f}' for v in row)} \\\\")

# ICC(3,1), consistency between conditions
Y = ab.pivot_table(index="iteration", columns=["batch_size", "learning_rate"], values="accuracy").to_numpy()
n, k = Y.shape
ss_rows = k * ((Y.mean(axis=1) - Y.mean()) ** 2).sum()
ss_err = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0) + Y.mean()) ** 2).sum()
msb, msw = ss_rows / (n - 1), ss_err / ((n - 1) * (k - 1))
icc, F = (msb - msw) / (msb + (k - 1) * msw), msb / msw
lo, hi = F / fdist.ppf(0.975, n - 1, (n - 1) * (k - 1)), F * fdist.ppf(0.975, (n - 1) * (k - 1), n - 1)
ci = [(x - 1) / (x + k - 1) for x in (lo, hi)]
print(f"\nICC(3,1) = ${icc:.3f}$ (95\\% CI from ${ci[0]:.3f}$ to ${ci[1]:.3f}$); "
      f"F({n - 1}, {(n - 1) * (k - 1)}) = {F:.3f}, p = {fdist.sf(F, n - 1, (n - 1) * (k - 1)):.3f}")

# Repeated-measures ANOVA with orthonormal contrasts; columns of Y are batch size (slow) x learning rate (fast)
c3 = np.array([[1, -1, 0], [1, 1, -2]]) / np.sqrt([[2], [6]])
j3 = np.ones((1, 3)) / np.sqrt(3)
terms = {"batch size (BS)": np.kron(c3, j3), "learning rate (LR)": np.kron(j3, c3), "$BS \\times LR$": np.kron(c3, c3)}
names = {"batch size (BS)": "batch size", "learning rate (LR)": "learning rate",
         "$BS \\times LR$": "batch size $\\times$ learning rate"}
zs = {t: Y @ C.T for t, C in terms.items()}
ss_d = {t: ((Z - Z.mean(axis=0)) ** 2).sum() for t, Z in zs.items()}
anova_rows, mauchly_rows, epsilons = [], [], []
for t, Z in zs.items():
    d = Z.shape[1]
    ss_n = n * (Z.mean(axis=0) ** 2).sum()
    F = (ss_n / d) / (ss_d[t] / (d * (n - 1)))
    S = np.cov(Z, rowvar=False)
    eps = np.trace(S) ** 2 / (d * np.trace(S @ S))
    W = np.linalg.det(S) / (np.trace(S) / d) ** d
    rho = 1 - (2 * d * d + d + 2) / (6 * d * (n - 1))
    statistic = -(n - 1) * rho * np.log(W)
    # second-order term as in R's mauchly.test, which uses the number of conditions k in 3 * k
    w2 = (d + 2) * (d - 1) * (d - 2) * (2 * d ** 3 + 6 * d ** 2 + 3 * k + 2) / (288 * ((n - 1) * d * rho) ** 2)
    df_m = d * (d + 1) / 2 - 1
    p_m = chi2.sf(statistic, df_m) + w2 * (chi2.sf(statistic, df_m + 4) - chi2.sf(statistic, df_m))
    ges = ss_n / (ss_n + ss_rows + sum(ss_d.values()))
    p = fdist.sf(F, eps * d, eps * d * (n - 1))
    mantissa, exponent = f"{p:.3e}".split("e")
    meaning = "Highly significant" if p < 0.001 else "Significant" if p < 0.05 else "Non-significant"
    anova_rows.append(f"{t} & ${mantissa} \\times 10^{{{int(exponent)}}}$ & {f'{ges:.3f}'.lstrip('0')} & {meaning} \\\\"
                      f"  % F = {F:.3f}, GG df {eps * d:.2f}, {eps * d * (n - 1):.2f}")
    pm = f"{p_m:.6f}" if p_m > 0.001 else "{}\\times10^{{{}}}".format(*f"{p_m:.4e}".split("e")).replace("{-0", "{-")
    mauchly_rows.append(f"{names[t]} & ${W:.4f}$ & ${statistic:.4f}$ & ${df_m:.0f}$ & ${pm}$ \\\\")
    epsilons.append(eps)
print("\n% Repeated-measures ANOVA: p with the Greenhouse-Geisser correction, generalized eta squared")
print("\n".join(anova_rows))
print("\n% Mauchly's sphericity test")
print("\n".join(mauchly_rows))
print(f"\nGreenhouse-Geisser epsilon: {epsilons[0]:.3f} for the batch size, {epsilons[1]:.3f} for the learning rate "
      f"and {epsilons[2]:.3f} for the interaction")
